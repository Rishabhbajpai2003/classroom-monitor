from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.face_detector.run import FaceIdentityDB, FaceTracker, InsightFaceBackend, color_from_id
from detectors.vsd_detector.common import (
    TrackClipBuffer,
    crop_face_context,
    draw_lip_box,
    get_gap_fill_tracks,
    resolve_torch_device,
    temporal_subsample_frames,
)
from detectors.vsd_detector.official_vtp import (
    DEFAULT_PUBLIC_CNN_CKPT,
    DEFAULT_PUBLIC_LIP_CKPT,
    DEFAULT_TOKENIZER_PATH,
    OfficialVTPEncoderMotionVSD,
    OfficialVTPLipReader,
    OfficialVTPVSD,
)


@dataclass
class TrackSpeechState:
    speech_prob: float = 0.0
    speaking: bool = False
    last_seen_frame_idx: int = -1
    segment_frames: List[np.ndarray] = field(default_factory=list)
    segment_frame_indices: List[int] = field(default_factory=list)
    segment_probs: List[float] = field(default_factory=list)
    last_transcript: str = ""

    def reset_segment(self) -> None:
        self.segment_frames.clear()
        self.segment_frame_indices.clear()
        self.segment_probs.clear()
        self.speaking = False

def temporal_subsample(frames: List[np.ndarray], target_length: int) -> List[np.ndarray]:
    return temporal_subsample_frames(frames, target_length)


def decode_segment(
    reader: OfficialVTPLipReader,
    frames: List[np.ndarray],
    decode_clip_len: int,
) -> str:
    sampled = temporal_subsample(frames, decode_clip_len)
    return reader.predict_text(sampled)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lip reading detector using face tracks and a VTP-style visual speech backbone")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to annotated output video")
    parser.add_argument("--lip-checkpoint", default=str(DEFAULT_PUBLIC_LIP_CKPT), help="Path to the official or compatible lip-reading checkpoint")
    parser.add_argument("--cnn-checkpoint", default=str(DEFAULT_PUBLIC_CNN_CKPT), help="Path to the official VTP visual backbone checkpoint")
    parser.add_argument("--vsd-checkpoint", default=None, help="Optional trained VSD checkpoint used to gate speech segments")
    parser.add_argument("--csv", default=None, help="Optional CSV path for decoded utterances")
    parser.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER_PATH), help="Path to the official tokenizer assets")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Torch device for VSD and lip reading")
    parser.add_argument("--face-size", type=int, default=160, help="Tracked face crop size before official VTP preprocessing")
    parser.add_argument("--process-fps", type=float, default=5.0, help="Run the face detector/tracker at this many FPS and fill intermediate frames from the last tracked face box")
    parser.add_argument("--vsd-clip-len", type=int, default=25, help="Face frames used by VSD windows")
    parser.add_argument("--decode-clip-len", type=int, default=50, help="Max frames fed to the lip reader per utterance")
    parser.add_argument("--speech-thresh", type=float, default=0.5, help="Speech probability threshold")
    parser.add_argument("--min-speech-frames", type=int, default=12, help="Minimum speaking frames before decoding")
    parser.add_argument("--segment-idle-frames", type=int, default=6, help="How many missing frames end a speaking segment")
    parser.add_argument("--max-segment-frames", type=int, default=75, help="Force decode once a segment reaches this many frames")
    parser.add_argument("--beam-size", type=int, default=30, help="Official VTP beam width")
    parser.add_argument("--beam-len-alpha", type=float, default=1.0, help="Official VTP beam length penalty alpha")
    parser.add_argument("--max-decode-tokens", type=int, default=35, help="Maximum decoded sub-word tokens")
    parser.add_argument("--disable-flip", action="store_true", help="Disable official test-time horizontal flip augmentation during lip reading")
    parser.add_argument("--det-size", type=int, default=960, help="InsightFace detection size")
    parser.add_argument("--det-thresh", type=float, default=0.28, help="InsightFace detector score threshold")
    parser.add_argument("--tile-grid", type=int, default=1, help="Optional tiled detection grid for small/far faces")
    parser.add_argument("--tile-overlap", type=float, default=0.20, help="Tile overlap ratio used when --tile-grid > 1")
    parser.add_argument("--ctx", type=int, default=0, help="InsightFace ctx id. Use -1 for CPU")
    parser.add_argument("--min-face", type=int, default=20, help="Minimum face size")
    parser.add_argument("--sim-thresh", type=float, default=0.45, help="Similarity threshold for face tracking")
    parser.add_argument("--ttl", type=int, default=120, help="Frames to keep active face tracks alive")
    parser.add_argument("--archive-ttl", type=int, default=1800, help="Frames to keep archived face identities")
    parser.add_argument("--reid-sim-thresh", type=float, default=0.55, help="Similarity threshold for reviving archived identities")
    parser.add_argument("--high-det-score", type=float, default=0.55, help="High-confidence detection threshold for the first association pass")
    parser.add_argument("--identity-db", default=str(PROJECT_ROOT / "detectors" / "face_detector" / "identity_db.json"), help="Persistent identity database shared with the face detector")
    parser.add_argument("--identity-db-save-every", type=int, default=150, help="Save the identity database every N input frames")
    parser.add_argument("--max-frames", type=int, default=-1, help="Optional frame cap for debugging")
    parser.add_argument("--display", action="store_true", help="Show a live preview window")
    return parser


def draw_overlay(frame, track, state: TrackSpeechState, threshold: float) -> None:
    x1, y1, x2, y2 = track.bbox.astype(int)
    color = color_from_id(track.track_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_lip_box(frame, track.bbox, color)

    status = "talking" if state.speech_prob >= threshold else "idle"
    identity_label = f"Student {track.track_id}" if track.track_id > 0 else "Student ?"
    label = f"{identity_label} | {status} {state.speech_prob:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_top = max(0, y1 - th - 8)
    y_bottom = max(th + 8, y1)
    cv2.rectangle(frame, (x1, y_top), (x1 + tw + 8, y_bottom), color, -1)
    cv2.putText(frame, label, (x1 + 4, y_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    if state.last_transcript:
        transcript = state.last_transcript[:72]
        cv2.putText(frame, transcript, (x1, min(frame.shape[0] - 10, y2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def main() -> None:
    args = build_argparser().parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not os.path.exists(args.lip_checkpoint):
        raise FileNotFoundError(f"Lip-reading checkpoint not found: {args.lip_checkpoint}")
    if not os.path.exists(args.cnn_checkpoint):
        raise FileNotFoundError(f"CNN checkpoint not found: {args.cnn_checkpoint}")
    if args.vsd_checkpoint and not os.path.exists(args.vsd_checkpoint):
        raise FileNotFoundError(f"VSD checkpoint not found: {args.vsd_checkpoint}")

    device = resolve_torch_device(args.device)
    lip_model = OfficialVTPLipReader(
        checkpoint_path=args.lip_checkpoint,
        cnn_checkpoint_path=args.cnn_checkpoint,
        tokenizer_path=args.tokenizer_path,
        device=device,
        beam_size=args.beam_size,
        beam_len_alpha=args.beam_len_alpha,
        max_decode_len=args.max_decode_tokens,
        use_flip=not args.disable_flip,
    )

    vsd_model: Optional[OfficialVTPVSD] = None
    if args.vsd_checkpoint:
        vsd_model = OfficialVTPVSD(
            checkpoint_path=args.vsd_checkpoint,
            cnn_checkpoint_path=args.cnn_checkpoint,
            device=device,
        )
    else:
        vsd_model = OfficialVTPEncoderMotionVSD(
            lip_checkpoint_path=args.lip_checkpoint,
            cnn_checkpoint_path=args.cnn_checkpoint,
            tokenizer_path=args.tokenizer_path,
            device=device,
        )
        print("No VSD checkpoint provided. Using encoder-motion VSD proxy from the official lip model.", flush=True)

    backend = InsightFaceBackend(
        det_size=args.det_size,
        ctx_id=args.ctx,
        min_face=args.min_face,
        det_thresh=args.det_thresh,
        tile_grid=args.tile_grid,
        tile_overlap=args.tile_overlap,
    )
    tracker = FaceTracker(
        sim_thresh=args.sim_thresh,
        ttl=args.ttl,
        archive_ttl=args.archive_ttl,
        reid_sim_thresh=args.reid_sim_thresh,
        high_det_score=args.high_det_score,
    )
    identity_db = FaceIdentityDB(args.identity_db)
    next_track_id, stored_identities = identity_db.load()
    loaded_identity_count = tracker.load_identity_memory(stored_identities, next_track_id=next_track_id)
    clip_buffer = TrackClipBuffer(
        clip_len=max(args.vsd_clip_len, args.decode_clip_len),
        crop_size=args.face_size,
        max_idle_frames=max(args.archive_ttl, args.segment_idle_frames),
    )

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    process_fps = fps if args.process_fps <= 0 else min(float(args.process_fps), fps)
    process_period_frames = max(1.0, fps / max(1e-6, process_fps))
    next_process_frame = 0.0
    gap_fill_frames = max(1, int(round(process_period_frames)))

    if loaded_identity_count > 0:
        print(f"Loaded {loaded_identity_count} persistent identities from: {args.identity_db}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {args.output}")

    csv_file = None
    csv_writer = None
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["track_id", "start_frame_idx", "end_frame_idx", "mean_speech_prob", "transcript"])

    states: Dict[int, TrackSpeechState] = {}
    frame_idx = 0
    processed_face_frames = 0
    saved_identity_count = loaded_identity_count

    def finalize_track_segment(track_id: int, state: TrackSpeechState) -> None:
        if len(state.segment_frames) < args.min_speech_frames:
            state.reset_segment()
            return

        transcript = decode_segment(
            lip_model,
            state.segment_frames,
            decode_clip_len=args.decode_clip_len,
        )
        state.last_transcript = transcript

        if csv_writer is not None and transcript:
            csv_writer.writerow([
                track_id,
                state.segment_frame_indices[0],
                state.segment_frame_indices[-1],
                f"{float(np.mean(state.segment_probs)):.4f}",
                transcript,
            ])

        state.reset_segment()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            should_process_face = frame_idx + 1e-6 >= next_process_frame
            if should_process_face:
                next_process_frame += process_period_frames
                detections = backend.infer(frame)
                visible_tracks = tracker.step(detections, frame_idx)
                processed_face_frames += 1
            else:
                visible_tracks = get_gap_fill_tracks(tracker.tracks, frame_idx, gap_fill_frames)
            active_ids = []

            for track in visible_tracks:
                active_ids.append(track.track_id)
                state = states.setdefault(track.track_id, TrackSpeechState())
                state.last_seen_frame_idx = frame_idx

                clip_buffer.push(track.track_id, frame_idx, frame, track.bbox)
                face_crop = crop_face_context(frame, track.bbox, crop_size=args.face_size)

                if vsd_model is not None and clip_buffer.ready(track.track_id, min_frames=max(8, args.vsd_clip_len // 2)):
                    clip_frames = temporal_subsample_frames(clip_buffer.get_frames(track.track_id), args.vsd_clip_len)
                    speech_probs = vsd_model.predict_proba(clip_frames)
                    latest = float(speech_probs[0, -1].item())
                    state.speech_prob = latest if state.speech_prob == 0.0 else 0.7 * state.speech_prob + 0.3 * latest
                elif vsd_model is None:
                    state.speech_prob = 1.0

                currently_speaking = state.speech_prob >= args.speech_thresh
                if currently_speaking:
                    state.speaking = True
                    state.segment_frames.append(face_crop)
                    state.segment_frame_indices.append(frame_idx)
                    state.segment_probs.append(state.speech_prob)
                    if len(state.segment_frames) >= args.max_segment_frames:
                        finalize_track_segment(track.track_id, state)
                elif state.speaking:
                    finalize_track_segment(track.track_id, state)

                draw_overlay(frame, track, state, args.speech_thresh)

            for track_id, state in list(states.items()):
                if track_id in active_ids:
                    continue
                if state.speaking and frame_idx - state.last_seen_frame_idx >= args.segment_idle_frames:
                    finalize_track_segment(track_id, state)
                if frame_idx - state.last_seen_frame_idx > args.archive_ttl:
                    del states[track_id]

            clip_buffer.prune(frame_idx, active_ids)

            if args.identity_db_save_every > 0 and frame_idx > 0 and frame_idx % args.identity_db_save_every == 0:
                saved_identity_count = identity_db.save(tracker)

            cv2.putText(
                frame,
                f"Frame: {frame_idx} | Visible faces: {len(visible_tracks)} | Face FPS: {process_fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)

            if args.display:
                cv2.imshow("Lip Reading Detector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(
                    f"Processed input frame {frame_idx} | sampled face frames {processed_face_frames} at {process_fps:.2f} FPS",
                    flush=True,
                )
    finally:
        for track_id, state in list(states.items()):
            if state.segment_frames:
                finalize_track_segment(track_id, state)

        saved_identity_count = identity_db.save(tracker)
        cap.release()
        writer.release()
        if csv_file is not None:
            csv_file.close()
        if args.display:
            cv2.destroyAllWindows()

    print(f"Done. Lip-reading video saved to: {args.output}")
    print(f"Persistent identity DB saved to: {args.identity_db} ({saved_identity_count} identities)")
    if args.csv:
        print(f"Lip-reading CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
