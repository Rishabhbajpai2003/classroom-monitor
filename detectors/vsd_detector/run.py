from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.student_backbone import format_student_display_name, format_student_global_id
from detectors.vsd_detector.common import (
    TrackClipBuffer,
    draw_lip_box,
    get_gap_fill_tracks,
    resolve_torch_device,
    temporal_subsample_frames,
)
from detectors.vsd_detector.official_vtp import (
    DEFAULT_PUBLIC_CNN_CKPT,
    DEFAULT_PUBLIC_LIP_CKPT,
    OfficialVTPEncoderMotionVSD,
    OfficialVTPVSD,
)

def _default_color_from_id(track_id: int) -> tuple[int, int, int]:
    base = abs(int(track_id)) * 73
    return (50 + base % 180, 120 + (base * 3) % 100, 90 + (base * 7) % 140)


COLOR_FROM_ID = _default_color_from_id


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual Speech Detection using the official Oxford VTP silencer head on face tracks")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to annotated output video")
    parser.add_argument("--checkpoint", default=None, help="Optional trained VSD checkpoint. If omitted, a lip-encoder motion proxy is used")
    parser.add_argument("--lip-checkpoint", default=str(DEFAULT_PUBLIC_LIP_CKPT), help="Official lip-reading checkpoint used to bootstrap the VSD proxy when no VSD checkpoint is available")
    parser.add_argument("--cnn-checkpoint", default=str(DEFAULT_PUBLIC_CNN_CKPT), help="Path to the official VTP visual backbone checkpoint")
    parser.add_argument("--csv", default=None, help="Optional CSV path for per-frame speech probabilities")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Torch device for the VSD model")
    parser.add_argument("--face-size", type=int, default=160, help="Tracked face crop size before official VTP preprocessing")
    parser.add_argument("--clip-len", type=int, default=25, help="Number of tracked face frames per VSD inference window")
    parser.add_argument("--process-fps", type=float, default=5.0, help="Run the face detector/tracker at this many FPS and fill intermediate frames from the last tracked face box")
    parser.add_argument("--infer-every", type=int, default=2, help="Run the VSD model every N frames per visible track")
    parser.add_argument("--speech-thresh", type=float, default=0.5, help="Probability threshold for classifying a frame as speech")
    parser.add_argument("--det-size", type=int, default=1280, help="InsightFace detection size")
    parser.add_argument("--det-thresh", type=float, default=0.28, help="InsightFace detector score threshold")
    parser.add_argument("--tile-grid", type=int, default=2, help="Optional tiled detection grid for small/far faces")
    parser.add_argument("--tile-overlap", type=float, default=0.20, help="Tile overlap ratio used when --tile-grid > 1")
    parser.add_argument("--ctx", type=int, default=0, help="InsightFace ctx id. Use -1 for CPU")
    parser.add_argument("--min-face", type=int, default=12, help="Minimum face size in pixels")
    parser.add_argument("--sim-thresh", type=float, default=0.45, help="Similarity threshold for face identity tracking")
    parser.add_argument("--ttl", type=int, default=120, help="Frames to keep an active face track alive")
    parser.add_argument("--archive-ttl", type=int, default=1800, help="Frames to keep archived face identities for re-id")
    parser.add_argument("--reid-sim-thresh", type=float, default=0.55, help="Similarity threshold for reviving archived identities")
    parser.add_argument("--high-det-score", type=float, default=0.55, help="High-confidence detection threshold for the first association pass")
    parser.add_argument("--identity-db", default=str(PROJECT_ROOT / "detectors" / "face_detector" / "identity_db.json"), help="Persistent identity database shared with the face detector")
    parser.add_argument("--identity-db-save-every", type=int, default=150, help="Save the identity database every N input frames")
    parser.add_argument("--max-frames", type=int, default=-1, help="Optional frame cap for debugging")
    parser.add_argument("--display", action="store_true", help="Show a live preview window")
    return parser


def draw_overlay(frame, track, speech_prob: Optional[float], threshold: float) -> None:
    x1, y1, x2, y2 = track.bbox.astype(int)
    color = COLOR_FROM_ID(track.track_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_lip_box(frame, track.bbox, color)
    metadata = getattr(track, "metadata", None)
    global_id = format_student_global_id(track.track_id, metadata)
    display_name = format_student_display_name(track.track_id, metadata)

    if speech_prob is None:
        label = f"{display_name} | warming"
    else:
        status = "talking" if speech_prob >= threshold else "silent"
        label = f"{display_name} | {status} {speech_prob:.2f}"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_top = max(0, y1 - th - 8)
    y_bottom = max(th + 8, y1)
    cv2.rectangle(frame, (x1, y_top), (x1 + tw + 8, y_bottom), color, -1)
    cv2.putText(frame, label, (x1 + 4, y_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def main() -> None:
    args = build_argparser().parse_args()
    global COLOR_FROM_ID
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not os.path.exists(args.cnn_checkpoint):
        raise FileNotFoundError(f"CNN checkpoint not found: {args.cnn_checkpoint}")
    if args.checkpoint and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.checkpoint and not os.path.exists(args.lip_checkpoint):
        raise FileNotFoundError(f"Lip-reading checkpoint not found: {args.lip_checkpoint}")

    from detectors.face_detector.run import FaceIdentityDB, FaceTracker, InsightFaceBackend, color_from_id

    COLOR_FROM_ID = color_from_id

    device = resolve_torch_device(args.device)
    using_proxy = args.checkpoint is None
    if using_proxy:
        model = OfficialVTPEncoderMotionVSD(
            lip_checkpoint_path=args.lip_checkpoint,
            cnn_checkpoint_path=args.cnn_checkpoint,
            device=device,
        )
        print("No VSD checkpoint provided. Using encoder-motion VSD proxy from the official lip model.", flush=True)
    else:
        model = OfficialVTPVSD(
            checkpoint_path=args.checkpoint,
            cnn_checkpoint_path=args.cnn_checkpoint,
            device=device,
        )

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
        allow_new_persistent_identities=False,
    )
    identity_db = FaceIdentityDB(args.identity_db)
    next_track_id, stored_identities = identity_db.load()
    loaded_identity_count = tracker.load_identity_memory(stored_identities, next_track_id=next_track_id)
    clip_buffer = TrackClipBuffer(clip_len=args.clip_len, crop_size=args.face_size, max_idle_frames=args.archive_ttl)

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
        csv_writer.writerow(
            ["frame_idx", "global_id", "display_name", "track_id", "speech_prob", "is_speaking"]
        )

    track_scores: Dict[int, float] = {}
    frame_idx = 0
    processed_face_frames = 0
    saved_identity_count = loaded_identity_count

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

            batch_candidates = []
            for track in visible_tracks:
                active_ids.append(track.track_id)
                clip_buffer.push(track.track_id, frame_idx, frame, track.bbox)

                previous = track_scores.get(track.track_id)
                if frame_idx % max(1, args.infer_every) == 0:
                    min_frames = args.clip_len
                    if clip_buffer.ready(track.track_id, min_frames=min_frames):
                        clip_frames = temporal_subsample_frames(clip_buffer.get_frames(track.track_id), args.clip_len)
                        batch_candidates.append((track.track_id, previous, clip_frames))

            if batch_candidates:
                batch_probs = model.predict_proba_batch([item[2] for item in batch_candidates])
                for batch_idx, (track_id, previous, _) in enumerate(batch_candidates):
                    latest = float(batch_probs[batch_idx, -1].item())
                    if previous is None:
                        track_scores[track_id] = latest
                    else:
                        track_scores[track_id] = 0.7 * previous + 0.3 * latest

            for track in visible_tracks:
                score = track_scores.get(track.track_id)
                draw_overlay(frame, track, score, args.speech_thresh)

                if csv_writer is not None and score is not None:
                    metadata = getattr(track, "metadata", None)
                    csv_writer.writerow(
                        [
                            frame_idx,
                            format_student_global_id(track.track_id, metadata),
                            format_student_display_name(track.track_id, metadata),
                            track.track_id,
                            f"{score:.5f}",
                            int(score >= args.speech_thresh),
                        ]
                    )

            clip_buffer.prune(frame_idx, active_ids)

            if args.identity_db_save_every > 0 and frame_idx > 0 and frame_idx % args.identity_db_save_every == 0:
                saved_identity_count = identity_db.save(tracker)

            cv2.putText(
                frame,
                f"Frame: {frame_idx} | Visible faces: {len(visible_tracks)} | Face FPS: {process_fps:.1f} | VSD: {'proxy' if using_proxy else 'trained'}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)

            if args.display:
                cv2.imshow("Visual Speech Detector", frame)
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
        saved_identity_count = identity_db.save(tracker)
        cap.release()
        writer.release()
        if csv_file is not None:
            csv_file.close()
        if args.display:
            cv2.destroyAllWindows()

    print(f"Done. VSD video saved to: {args.output}")
    print(f"Persistent identity DB saved to: {args.identity_db} ({saved_identity_count} identities)")
    if args.csv:
        print(f"VSD CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
