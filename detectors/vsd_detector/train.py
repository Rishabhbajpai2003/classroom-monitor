from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.vsd_detector.common import resolve_torch_device
from detectors.vsd_detector.dataset import AVAActiveSpeakerDataset, collate_vsd_batch
from detectors.vsd_detector.official_vtp import DEFAULT_PUBLIC_CNN_CKPT, DEFAULT_PUBLIC_LIP_CKPT, build_official_vsd_model


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the paper-aligned VSD head on top of the official Oxford VTP encoder")
    parser.add_argument("--train-csv", required=True, help="AVA ActiveSpeaker train CSV")
    parser.add_argument("--val-csv", required=True, help="AVA ActiveSpeaker val CSV")
    parser.add_argument("--videos-root", required=True, help="Root directory that contains the AVA videos")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "weights" / "vtp" / "vsd"), help="Directory for checkpoints and metrics")
    parser.add_argument("--lip-checkpoint", default=str(DEFAULT_PUBLIC_LIP_CKPT), help="Official lip-reading checkpoint used to initialize the VSD encoder")
    parser.add_argument("--cnn-checkpoint", default=str(DEFAULT_PUBLIC_CNN_CKPT), help="Official VTP visual backbone checkpoint")
    parser.add_argument("--resume", default=None, help="Resume from an existing VSD checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Training device")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate. The paper fine-tunes the VSD model at 1e-6")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--clip-len", type=int, default=25, help="Frames per training clip")
    parser.add_argument("--clip-stride", type=int, default=12, help="Sliding window stride when generating clips")
    parser.add_argument("--frame-size", type=int, default=160, help="Pre-crop face frame size before the 96x96 crop")
    parser.add_argument("--img-size", type=int, default=96, help="Model input size")
    parser.add_argument("--max-rotation", type=float, default=10.0, help="Random rotation for train augmentation")
    parser.add_argument("--crop-jitter", type=int, default=3, help="Random crop offset in pixels")
    parser.add_argument("--min-track-frames", type=int, default=8, help="Minimum number of frames for a usable face track segment")
    parser.add_argument("--freeze-face-encoder", action="store_true", help="Freeze the CNN+VTP face encoder during VSD head training")
    parser.add_argument("--pos-weight", type=float, default=1.0, help="Positive-class weight for BCEWithLogitsLoss")
    parser.add_argument("--limit-train", type=int, default=-1, help="Optional train sample cap for debugging")
    parser.add_argument("--limit-val", type=int, default=-1, help="Optional val sample cap for debugging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit_dataset(dataset: AVAActiveSpeakerDataset, limit: int) -> AVAActiveSpeakerDataset:
    if limit > 0:
        dataset.samples = dataset.samples[:limit]
    return dataset


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
    loss = loss * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return loss.sum() / denom


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, pos_weight: Optional[torch.Tensor]) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_frames = 0
    all_scores: List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            video = batch["video"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            src_mask = mask.unsqueeze(1)

            logits = model(video, src_mask)
            loss = masked_bce_loss(logits, labels, mask, pos_weight=pos_weight)
            total_loss += float(loss.item()) * int(mask.sum().item())
            total_frames += int(mask.sum().item())

            probs = torch.sigmoid(logits)
            active = mask.bool()
            all_scores.extend(probs[active].detach().cpu().tolist())
            all_labels.extend(labels[active].detach().cpu().tolist())

    metrics = {
        "loss": total_loss / max(1, total_frames),
        "ap": 0.0,
    }
    if all_labels and len(set(int(v >= 0.5) for v in all_labels)) > 1:
        metrics["ap"] = float(average_precision_score(all_labels, all_scores))
    return metrics


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Adam,
    epoch: int,
    best_ap: float,
    args: argparse.Namespace,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "vis_optimizer": None,
        "global_epoch": epoch,
        "best_ap": best_ap,
        "args": vars(args),
    }
    torch.save(payload, path)


def load_resume_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: Adam, device: torch.device) -> tuple[int, float]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["state_dict"], strict=True)
    if payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    start_epoch = int(payload.get("global_epoch", 0)) + 1
    best_ap = float(payload.get("best_ap", 0.0))
    return start_epoch, best_ap


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Train CSV not found: {args.train_csv}")
    if not os.path.exists(args.val_csv):
        raise FileNotFoundError(f"Val CSV not found: {args.val_csv}")
    if not os.path.exists(args.videos_root):
        raise FileNotFoundError(f"Videos root not found: {args.videos_root}")
    if not os.path.exists(args.lip_checkpoint):
        raise FileNotFoundError(f"Lip checkpoint not found: {args.lip_checkpoint}")
    if not os.path.exists(args.cnn_checkpoint):
        raise FileNotFoundError(f"CNN checkpoint not found: {args.cnn_checkpoint}")
    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

    device = resolve_torch_device(args.device)
    model = build_official_vsd_model(
        device=device,
        checkpoint_path=args.lip_checkpoint,
        cnn_checkpoint_path=args.cnn_checkpoint,
    )

    if args.freeze_face_encoder and hasattr(model, "face_encoder"):
        for param in model.face_encoder.parameters():
            param.requires_grad = False

    optimizer = Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_ap = 0.0
    if args.resume:
        start_epoch, best_ap = load_resume_checkpoint(args.resume, model, optimizer, device)

    train_dataset = maybe_limit_dataset(
        AVAActiveSpeakerDataset(
            annotation_csv=args.train_csv,
            videos_root=args.videos_root,
            clip_len=args.clip_len,
            clip_stride=args.clip_stride,
            frame_size=args.frame_size,
            img_size=args.img_size,
            train=True,
            max_rotation=args.max_rotation,
            crop_jitter=args.crop_jitter,
            min_track_frames=args.min_track_frames,
        ),
        args.limit_train,
    )
    val_dataset = maybe_limit_dataset(
        AVAActiveSpeakerDataset(
            annotation_csv=args.val_csv,
            videos_root=args.videos_root,
            clip_len=args.clip_len,
            clip_stride=args.clip_stride,
            frame_size=args.frame_size,
            img_size=args.img_size,
            train=False,
            max_rotation=0.0,
            crop_jitter=0,
            min_track_frames=args.min_track_frames,
        ),
        args.limit_val,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_vsd_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_vsd_batch,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pos_weight = torch.tensor([args.pos_weight], device=device) if not math.isclose(args.pos_weight, 1.0) else None
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    history: List[Dict[str, float]] = []

    print(
        f"Training VSD on {len(train_dataset)} train clips and {len(val_dataset)} val clips "
        f"with lr={args.lr} on {device}.",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_frames = 0

        progress = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            video = batch["video"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            src_mask = mask.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(video, src_mask)
                loss = masked_bce_loss(logits, labels, mask, pos_weight=pos_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            frames_in_batch = int(mask.sum().item())
            running_loss += float(loss.item()) * frames_in_batch
            running_frames += frames_in_batch
            progress.set_postfix(loss=f"{running_loss / max(1, running_frames):.5f}")

        train_loss = running_loss / max(1, running_frames)
        val_metrics = evaluate(model, val_loader, device, pos_weight)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(val_metrics["loss"]),
            "val_ap": float(val_metrics["ap"]),
        }
        history.append(epoch_metrics)

        if val_metrics["ap"] >= best_ap:
            best_ap = float(val_metrics["ap"])
            save_checkpoint(output_dir / "best.pth", model, optimizer, epoch, best_ap, args)
        save_checkpoint(output_dir / "last.pth", model, optimizer, epoch, best_ap, args)

        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump({"best_ap": best_ap, "history": history}, handle, indent=2)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_metrics['loss']:.5f} val_ap={val_metrics['ap']:.5f}",
            flush=True,
        )

    print(f"Done. Best VSD checkpoint: {output_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
