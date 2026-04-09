from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_VTP_ROOT = PROJECT_ROOT / "third_party" / "vtp_official"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "weights" / "vtp" / "tokenizers" / "bert-large-uncased"
DEFAULT_PUBLIC_CNN_CKPT = PROJECT_ROOT / "weights" / "vtp" / "public_train_data" / "feature_extractor.pth"
DEFAULT_PUBLIC_LIP_CKPT = PROJECT_ROOT / "weights" / "vtp" / "public_train_data" / "ft_lrs2.pth"

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
START_SYMBOL = 100
END_SYMBOL = 101


def _ensure_official_imports() -> None:
    official_root = str(OFFICIAL_VTP_ROOT)
    if official_root not in sys.path:
        sys.path.insert(0, official_root)


def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def _load_official_modules():
    _ensure_official_imports()
    if "dataloader" not in sys.modules:
        shim = types.ModuleType("dataloader")
        shim.subsequent_mask = subsequent_mask
        sys.modules["dataloader"] = shim

    from models import builders  # type: ignore
    from search import beam_search  # type: ignore
    from utils import load as official_load  # type: ignore

    return builders, beam_search, official_load


def build_official_vsd_model(
    device: torch.device | str = "cpu",
    checkpoint_path: Optional[str | Path] = None,
    cnn_checkpoint_path: Optional[str | Path] = DEFAULT_PUBLIC_CNN_CKPT,
) -> torch.nn.Module:
    builders, _, official_load = _load_official_modules()
    device = torch.device(device)
    model = builders["silencer_vtp24x24"](
        512,
        N=6,
        d_model=512,
        h=8,
        dropout=0.1,
        backbone=True,
    ).to(device)

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Official VTP checkpoint not found: {ckpt}")
        cnn_ckpt = None
        if cnn_checkpoint_path is not None:
            cnn_ckpt = Path(cnn_checkpoint_path)
            if not cnn_ckpt.exists():
                raise FileNotFoundError(f"Official VTP CNN checkpoint not found: {cnn_ckpt}")
            cnn_ckpt = str(cnn_ckpt)
        model, *_ = official_load(
            model,
            str(ckpt),
            face_encoder_ckpt=cnn_ckpt,
            device=str(device),
            strict=False,
        )

    return model


def prepare_vtp_video(
    frames_bgr: Iterable[np.ndarray],
    frame_size: int = 160,
    img_size: int = 96,
) -> torch.Tensor:
    return prepare_vtp_batch([list(frames_bgr)], frame_size=frame_size, img_size=img_size)


def prepare_vtp_batch(
    clips_bgr: Iterable[Iterable[np.ndarray]],
    frame_size: int = 160,
    img_size: int = 96,
) -> torch.Tensor:
    videos: List[np.ndarray] = []
    crop_offset = (frame_size - img_size) // 2
    for frames_bgr in clips_bgr:
        prepared: List[np.ndarray] = []
        for frame_bgr in frames_bgr:
            resized = cv2.resize(frame_bgr, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
            cropped = resized[crop_offset:crop_offset + img_size, crop_offset:crop_offset + img_size]
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            prepared.append(np.transpose(rgb, (2, 0, 1)))
        if not prepared:
            raise ValueError("At least one frame is required for VTP inference")
        videos.append(np.stack(prepared, axis=0))
    if not videos:
        raise ValueError("At least one clip is required for VTP inference")
    batch = np.stack(videos, axis=0)  # [B, T, C, H, W]
    return torch.from_numpy(batch).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]


class OfficialVTPTokenizer:
    def __init__(self, tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH) -> None:
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Official VTP tokenizer path not found: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN,
            use_fast=True,
        )

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    @property
    def bos_token_id(self) -> int:
        return START_SYMBOL

    @property
    def eos_token_id(self) -> int:
        return END_SYMBOL

    def decode(self, token_ids: Iterable[int]) -> str:
        cleaned: List[int] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.eos_token_id:
                break
            if token_id == self.bos_token_id or token_id < 0:
                continue
            cleaned.append(token_id + 1)

        if not cleaned:
            return ""

        text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(cleaned))
        return (
            text.replace(f"{BOS_TOKEN} ", "")
            .replace(f" {EOS_TOKEN}", "")
            .replace("[cls] ", "")
            .replace(" [sep]", "")
            .strip()
            .lower()
        )


class OfficialVTPLipReader:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_PUBLIC_LIP_CKPT,
        cnn_checkpoint_path: str | Path = DEFAULT_PUBLIC_CNN_CKPT,
        tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
        device: torch.device | str = "cpu",
        beam_size: int = 30,
        beam_len_alpha: float = 1.0,
        max_decode_len: int = 35,
        use_flip: bool = True,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        cnn_checkpoint_path = Path(cnn_checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Official VTP lip checkpoint not found: {checkpoint_path}")
        if not cnn_checkpoint_path.exists():
            raise FileNotFoundError(f"Official VTP CNN checkpoint not found: {cnn_checkpoint_path}")

        self.device = torch.device(device)
        self.beam_size = beam_size
        self.beam_len_alpha = beam_len_alpha
        self.max_decode_len = max_decode_len
        self.use_flip = use_flip
        self.tokenizer = OfficialVTPTokenizer(tokenizer_path)

        builders, beam_search, official_load = _load_official_modules()
        self._beam_search = beam_search
        self.model = builders["vtp24x24"](
            self.tokenizer.vocab_size + 1,
            512,
            N=6,
            d_model=512,
            h=8,
            dropout=0.1,
        ).to(self.device).eval()
        self.model, *_ = official_load(
            self.model,
            str(checkpoint_path),
            face_encoder_ckpt=str(cnn_checkpoint_path),
            device=str(self.device),
        )

    @torch.no_grad()
    def encode_frames(self, frames_bgr: Iterable[np.ndarray]) -> torch.Tensor:
        src = prepare_vtp_video(frames_bgr).to(self.device)
        src_mask = torch.ones((1, 1, src.size(2)), device=self.device)
        encoder_output, _ = self.model.encode(src, src_mask)
        return encoder_output

    @torch.no_grad()
    def encode_batch(self, clips_bgr: Iterable[Iterable[np.ndarray]]) -> torch.Tensor:
        src = prepare_vtp_batch(clips_bgr).to(self.device)
        src_mask = torch.ones((src.size(0), 1, src.size(2)), device=self.device)
        encoder_output, _ = self.model.encode(src, src_mask)
        return encoder_output

    def _forward_pass(self, src: torch.Tensor, src_mask: torch.Tensor):
        encoder_output, src_mask = self.model.encode(src, src_mask)
        return self._beam_search(
            decoder=self.model,
            bos_index=START_SYMBOL,
            eos_index=END_SYMBOL,
            max_output_length=self.max_decode_len,
            pad_index=0,
            encoder_output=encoder_output,
            src_mask=src_mask,
            size=self.beam_size,
            alpha=self.beam_len_alpha,
            n_best=self.beam_size,
        )

    @torch.no_grad()
    def predict_text(self, frames_bgr: Iterable[np.ndarray]) -> str:
        src = prepare_vtp_video(frames_bgr).to(self.device)
        src_mask = torch.ones((1, 1, src.size(2)), device=self.device)
        beam_outs, beam_scores = self._forward_pass(src, src_mask)

        if self.use_flip:
            flipped = torch.flip(src, dims=[4])
            beam_outs_f, beam_scores_f = self._forward_pass(flipped, src_mask)
            beam_outs = beam_outs[0] + beam_outs_f[0]
            beam_scores = np.array(beam_scores[0] + beam_scores_f[0], dtype=np.float32)
        else:
            beam_outs = beam_outs[0]
            beam_scores = np.array(beam_scores[0], dtype=np.float32)

        if len(beam_outs) == 0:
            return ""
        best_idx = int(np.argmax(beam_scores))
        return self.tokenizer.decode(beam_outs[best_idx].cpu().numpy().tolist())


class OfficialVTPVSD:
    def __init__(
        self,
        checkpoint_path: str | Path,
        cnn_checkpoint_path: Optional[str | Path] = DEFAULT_PUBLIC_CNN_CKPT,
        device: torch.device | str = "cpu",
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Official VTP VSD checkpoint not found: {checkpoint_path}")

        self.device = torch.device(device)
        builders, _, official_load = _load_official_modules()
        self.model = builders["silencer_vtp24x24"](
            512,
            N=6,
            d_model=512,
            h=8,
            dropout=0.1,
            backbone=True,
        ).to(self.device).eval()

        cnn_ckpt = None
        if cnn_checkpoint_path is not None:
            cnn_ckpt = Path(cnn_checkpoint_path)
            if not cnn_ckpt.exists():
                raise FileNotFoundError(f"Official VTP CNN checkpoint not found: {cnn_ckpt}")
            cnn_ckpt = str(cnn_ckpt)

        self.model, *_ = official_load(
            self.model,
            str(checkpoint_path),
            face_encoder_ckpt=cnn_ckpt,
            device=str(self.device),
            strict=False,
        )

    @torch.no_grad()
    def predict_proba(self, frames_bgr: Iterable[np.ndarray]) -> torch.Tensor:
        src = prepare_vtp_video(frames_bgr).to(self.device)
        src_mask = torch.ones((1, 1, src.size(2)), device=self.device)
        logits = self.model(src, src_mask)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_proba_batch(self, clips_bgr: Iterable[Iterable[np.ndarray]]) -> torch.Tensor:
        src = prepare_vtp_batch(clips_bgr).to(self.device)
        src_mask = torch.ones((src.size(0), 1, src.size(2)), device=self.device)
        logits = self.model(src, src_mask)
        return torch.sigmoid(logits)


class OfficialVTPEncoderMotionVSD:
    """
    Fallback VSD proxy when no trained VSD FC head is available.

    It uses the official lip-reading encoder and converts frame-to-frame encoder
    motion into a smoothed probability-like score. This is not the paper's
    trained VSD head, but it preserves the official backbone and stays usable
    without unpublished VSD weights.
    """

    def __init__(
        self,
        lip_checkpoint_path: str | Path = DEFAULT_PUBLIC_LIP_CKPT,
        cnn_checkpoint_path: str | Path = DEFAULT_PUBLIC_CNN_CKPT,
        tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
        device: torch.device | str = "cpu",
    ) -> None:
        self.reader = OfficialVTPLipReader(
            checkpoint_path=lip_checkpoint_path,
            cnn_checkpoint_path=cnn_checkpoint_path,
            tokenizer_path=tokenizer_path,
            device=device,
            use_flip=False,
        )
        self.device = self.reader.device

    @torch.no_grad()
    def predict_proba(self, frames_bgr: Iterable[np.ndarray]) -> torch.Tensor:
        features = self.reader.encode_frames(frames_bgr)  # [1, T, D]
        if features.size(1) <= 1:
            return torch.full((1, features.size(1)), 0.5, device=features.device)

        delta = torch.norm(features[:, 1:] - features[:, :-1], dim=-1)
        delta = torch.cat([delta[:, :1], delta], dim=1)
        mean = delta.mean(dim=1, keepdim=True)
        std = delta.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        z = (delta - mean) / std
        probs = torch.sigmoid(1.5 * z - 0.25)
        return probs

    @torch.no_grad()
    def predict_proba_batch(self, clips_bgr: Iterable[Iterable[np.ndarray]]) -> torch.Tensor:
        features = self.reader.encode_batch(clips_bgr)  # [B, T, D]
        if features.size(1) <= 1:
            return torch.full((features.size(0), features.size(1)), 0.5, device=features.device)

        delta = torch.norm(features[:, 1:] - features[:, :-1], dim=-1)
        delta = torch.cat([delta[:, :1], delta], dim=1)
        mean = delta.mean(dim=1, keepdim=True)
        std = delta.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        z = (delta - mean) / std
        probs = torch.sigmoid(1.5 * z - 0.25)
        return probs
