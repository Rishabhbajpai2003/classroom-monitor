from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.identity_enrichment import (  # noqa: E402
    classify_harvest_candidate,
    compact_embedding_bank,
    current_utc_iso,
    infer_profile_bucket,
    infer_size_bucket,
    summarize_embedding_bank,
)
from detectors.face_detector.run import (  # noqa: E402
    DEFAULT_IDENTITY_DB_PATH,
    FaceIdentityDB,
    FaceTracker,
    InsightFaceBackend,
    Track,
    TrackObservation,
    ensure_parent_dir,
)
from scripts.rebuild_identity_db_from_student_details import (  # noqa: E402
    write_manifests,
    write_registry_json,
)


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass
class PhaseSpec:
    name: str
    inputs: list[str]


@dataclass
class TrackSample:
    raw_track_id: int
    frame_idx: int
    timestamp_seconds: float
    bbox: list[float]
    quality: float
    score: float
    embedding: np.ndarray
    profile_bucket: str
    size_bucket: str
    face_size: float
    detector_used: str
    embedder_used: str
    lighting_bucket: str
    sharpness: float
    brightness: float
    shadow_severity: float
    occlusion_ratio: float


@dataclass
class VideoDecision:
    phase: str
    video_name: str
    video_path: str
    raw_track_id: int
    resolved_track_id: int
    global_id: str
    name: str
    roll_number: str
    student_key: str
    decision: str
    best_similarity: float
    best_quality: float
    good_crops: int
    support_seconds: float
    first_frame_idx: int
    last_frame_idx: int
    representative_frame_idx: int
    representative_bbox: list[float]
    auto_added_samples: int
    net_new_embeddings: int
    duplicate_rejections: int
    crop_path: str = ""
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality-first multi-video enrollment to enrich the named student face identity bank.",
    )
    parser.add_argument(
        "--phase",
        action="append",
        default=[],
        help=(
            "Ordered phase specification in the form phase_name=glob_or_path. "
            "Repeat to add more phases. If omitted, defaults to classroom then midsem."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "identity_harvest"),
        help="Folder where harvest manifests, reports, and review crops are written.",
    )
    parser.add_argument(
        "--review-dir",
        default="",
        help="Optional override for where review crops are exported. Defaults to <output-dir>/review_crops.",
    )
    parser.add_argument(
        "--identity-db",
        default=str(PROJECT_ROOT / DEFAULT_IDENTITY_DB_PATH),
        help="Main named identity DB to enrich in place.",
    )
    parser.add_argument(
        "--registry",
        default=str(PROJECT_ROOT / "outputs" / "student_registry.json"),
        help="Compatibility student registry JSON to keep in sync.",
    )
    parser.add_argument(
        "--manifest-json",
        default=str(PROJECT_ROOT / "outputs" / "student_identity_manifest.json"),
        help="Student manifest JSON to keep in sync.",
    )
    parser.add_argument(
        "--manifest-csv",
        default=str(PROJECT_ROOT / "outputs" / "student_identity_manifest.csv"),
        help="Student manifest CSV to keep in sync.",
    )
    parser.add_argument("--ctx", type=int, default=0, help="InsightFace GPU id. Use -1 for CPU.")
    parser.add_argument("--process-fps", type=float, default=8.0, help="Sample FPS during harvest.")
    parser.add_argument("--det-size", type=int, default=1600, help="Detection size for offline harvest.")
    parser.add_argument("--det-thresh", type=float, default=0.18, help="Detector threshold for offline harvest.")
    parser.add_argument("--tile-grid", type=int, default=3, help="Tiled detection grid for offline harvest.")
    parser.add_argument("--tile-overlap", type=float, default=0.22, help="Tile overlap ratio for harvest.")
    parser.add_argument("--min-face", type=int, default=8, help="Minimum face size in pixels.")
    parser.add_argument("--primary-detector", default="scrfd", help="Primary detector label used in metadata.")
    parser.add_argument("--backup-detector", default="retinaface", help="Backup detector label used in metadata.")
    parser.add_argument("--disable-backup-detector", action="store_true", help="Disable the backup detector pass.")
    parser.add_argument("--adaface-weights", default="", help="Optional AdaFace ONNX weights.")
    parser.add_argument("--sim-thresh", type=float, default=0.45, help="Active track association threshold.")
    parser.add_argument("--ttl", type=int, default=140, help="Missing-track TTL in frames.")
    parser.add_argument("--archive-ttl", type=int, default=2400, help="Archived-track TTL in frames.")
    parser.add_argument("--reid-sim-thresh", type=float, default=0.52, help="Re-identification threshold.")
    parser.add_argument("--new-id-confirm-hits", type=int, default=5, help="Provisional hit count before new IDs are minted.")
    parser.add_argument("--new-id-confirm-quality", type=float, default=0.42, help="Minimum quality before a brand new unnamed ID can be minted.")
    parser.add_argument("--provisional-match-margin", type=float, default=0.04, help="Required margin before provisional merge.")
    parser.add_argument("--high-det-score", type=float, default=0.60, help="High-confidence score for first association pass.")
    parser.add_argument("--auto-add-similarity", type=float, default=0.68, help="Similarity threshold for automatic bank enrichment.")
    parser.add_argument("--review-similarity", type=float, default=0.60, help="Similarity threshold for review queue export.")
    parser.add_argument("--min-track-good-crops", type=int, default=3, help="Minimum good crops for automatic add.")
    parser.add_argument("--min-stable-seconds", type=float, default=1.5, help="Minimum stable visibility duration for automatic add.")
    parser.add_argument("--min-sample-quality", type=float, default=0.42, help="Minimum per-sample quality to be considered usable.")
    parser.add_argument("--min-face-size", type=float, default=28.0, help="Minimum per-sample face size to be considered usable.")
    parser.add_argument("--duplicate-sim-thresh", type=float, default=0.92, help="Similarity threshold used to collapse near-duplicate embeddings.")
    parser.add_argument("--max-bank", type=int, default=48, help="Maximum embeddings stored per named student.")
    parser.add_argument("--max-samples-per-track", type=int, default=24, help="Cap harvested samples merged from one resolved track segment.")
    parser.add_argument("--backup", action="store_true", help="Backup the current identity DB and manifests into the output folder before enrichment.")
    parser.add_argument("--max-videos-per-phase", type=int, default=-1, help="Optional debugging cap per phase.")
    parser.add_argument("--max-frames", type=int, default=-1, help="Optional per-video frame cap for debugging.")
    return parser.parse_args()


def parse_phase_specs(raw_specs: Sequence[str]) -> list[PhaseSpec]:
    if not raw_specs:
        return [
            PhaseSpec(name="classroom", inputs=[str(PROJECT_ROOT / "media" / "classroom*")]),
            PhaseSpec(name="midsem", inputs=[str(PROJECT_ROOT / "media" / "midsem*")]),
        ]
    phases: list[PhaseSpec] = []
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --phase value: {spec}. Expected phase_name=glob_or_path")
        name, value = spec.split("=", 1)
        inputs = [item.strip() for item in value.split(";") if item.strip()]
        if not name.strip() or not inputs:
            raise ValueError(f"Invalid --phase value: {spec}. Expected phase_name=glob_or_path")
        phases.append(PhaseSpec(name=name.strip(), inputs=inputs))
    return phases


def expand_phase_videos(phase: PhaseSpec, max_videos: int = -1) -> list[Path]:
    discovered: list[Path] = []
    for input_spec in phase.inputs:
        candidate = Path(input_spec)
        if candidate.exists() and candidate.is_file():
            discovered.append(candidate.resolve())
            continue
        if candidate.exists() and candidate.is_dir():
            for path in sorted(candidate.iterdir(), key=lambda item: item.name.lower()):
                if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                    discovered.append(path.resolve())
            continue
        for path_text in sorted(glob.glob(input_spec), key=lambda item: item.lower()):
            path = Path(path_text)
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                discovered.append(path.resolve())
    unique: list[Path] = []
    seen = set()
    for path in discovered:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if max_videos > 0:
        return unique[:max_videos]
    return unique


def global_id_for_track(track_id: int) -> str:
    return f"STU_{int(track_id):03d}"


def copy_named_tracks(identities: Dict[int, Track]) -> Dict[int, Track]:
    copied: Dict[int, Track] = {}
    for track_id, track in identities.items():
        copied[int(track_id)] = copy.deepcopy(track)
    return copied


def backup_identity_files(paths: Iterable[Path], output_dir: Path) -> list[Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = output_dir / "backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_paths: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        target = backup_dir / path.name
        shutil.copy2(path, target)
        backup_paths.append(target)
    return backup_paths


def usable_sample(observation: TrackSample, min_quality: float, min_face_size: float) -> bool:
    return float(observation.quality) >= float(min_quality) and float(observation.face_size) >= float(min_face_size)


def bbox_to_crop(frame: np.ndarray, bbox: Sequence[float], pad_ratio: float = 0.10) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * pad_ratio
    pad_y = height * pad_ratio
    nx1 = max(0, int(round(x1 - pad_x)))
    ny1 = max(0, int(round(y1 - pad_y)))
    nx2 = min(frame.shape[1], int(round(x2 + pad_x)))
    ny2 = min(frame.shape[0], int(round(y2 + pad_y)))
    if nx2 <= nx1 or ny2 <= ny1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return frame[ny1:ny2, nx1:nx2].copy()


def build_sample_from_observation(observation: TrackObservation, process_fps: float) -> TrackSample:
    bbox = np.asarray(observation.bbox, dtype=np.float32)
    face_size = float(max(0.0, min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))))
    quality_profile = dict(observation.quality_profile or {})
    return TrackSample(
        raw_track_id=int(observation.raw_track_id),
        frame_idx=int(observation.frame_idx),
        timestamp_seconds=float(observation.frame_idx) / max(1e-6, float(process_fps)),
        bbox=bbox.tolist(),
        quality=float(observation.quality),
        score=float(observation.score),
        embedding=np.asarray(observation.embedding, dtype=np.float32).copy(),
        profile_bucket=infer_profile_bucket(bbox, observation.landmarks),
        size_bucket=infer_size_bucket(face_size),
        face_size=face_size,
        detector_used=str(observation.detector_used or "scrfd"),
        embedder_used=str(observation.embedder_used or "arcface"),
        lighting_bucket=str(quality_profile.get("lighting_bucket", "balanced_light")),
        sharpness=float(quality_profile.get("sharpness", 0.0) or 0.0),
        brightness=float(quality_profile.get("brightness", 0.0) or 0.0),
        shadow_severity=float(quality_profile.get("shadow_severity", 0.0) or 0.0),
        occlusion_ratio=float(quality_profile.get("occlusion_ratio", 0.0) or 0.0),
    )


def merge_samples_into_track(
    track: Track,
    samples: Sequence[TrackSample],
    *,
    phase: str,
    video_path: Path,
    max_bank: int,
    duplicate_sim_thresh: float,
) -> tuple[int, int]:
    before_count = len(track.embeddings)
    sorted_samples = sorted(samples, key=lambda item: (item.timestamp_seconds, item.quality), reverse=True)
    for sample in sorted_samples:
        track.update_embedding_bank(
            sample.embedding,
            sample_quality=float(sample.quality),
            sample_metadata={
                "source_kind": "harvest",
                "source_video": video_path.name,
                "source_video_path": video_path.as_posix(),
                "source_frame_idx": int(sample.frame_idx),
                "source_timestamp_seconds": float(sample.timestamp_seconds),
                "harvest_phase": phase,
                "quality": float(sample.quality),
                "score": float(sample.score),
                "face_size": float(sample.face_size),
                "profile_bucket": sample.profile_bucket,
                "size_bucket": sample.size_bucket,
                "lighting_bucket": sample.lighting_bucket,
                "sharpness": float(sample.sharpness),
                "brightness": float(sample.brightness),
                "shadow_severity": float(sample.shadow_severity),
                "occlusion_ratio": float(sample.occlusion_ratio),
                "detector_used": sample.detector_used,
                "embedder_used": sample.embedder_used,
                "source_token": f"{phase}:{video_path.name}:{int(sample.raw_track_id)}:{int(sample.frame_idx)}:{sample.embedder_used}",
                "added_at": current_utc_iso(),
            },
            max_bank=max_bank,
            duplicate_sim_thresh=duplicate_sim_thresh,
        )
    after_count = len(track.embeddings)
    net_new = max(0, after_count - before_count)
    duplicate_rejections = max(0, len(sorted_samples) - net_new)

    metadata = dict(track.metadata or {})
    harvest_videos = list(metadata.get("harvest_source_videos", []) or [])
    if video_path.name not in harvest_videos:
        harvest_videos.append(video_path.name)
    harvest_phases = list(metadata.get("harvest_phases", []) or [])
    if phase not in harvest_phases:
        harvest_phases.append(phase)
    metadata["harvest_source_videos"] = harvest_videos
    metadata["harvest_phases"] = harvest_phases
    metadata["harvested_sample_count"] = int(metadata.get("harvested_sample_count", 0)) + len(sorted_samples)
    metadata["last_enriched_at"] = current_utc_iso()
    metadata["last_source_video"] = video_path.name
    track.metadata = metadata
    track.avg_embedding, track.best_embedding, track.best_embedding_quality, track.recent_embedding = summarize_embedding_bank(
        track.embeddings,
        track.embedding_qualities,
        track.embedding_metadata,
    )
    return net_new, duplicate_rejections


def build_phase_summary_rows(decisions: Sequence[VideoDecision]) -> list[dict[str, object]]:
    grouped: Dict[tuple[str, str], dict[str, object]] = {}
    for item in decisions:
        key = (item.phase, item.global_id)
        row = grouped.setdefault(
            key,
            {
                "phase": item.phase,
                "global_id": item.global_id,
                "name": item.name,
                "roll_number": item.roll_number,
                "student_key": item.student_key,
                "segments_seen": 0,
                "auto_added_samples": 0,
                "net_new_embeddings": 0,
                "duplicate_rejections": 0,
                "review_segments": 0,
                "reject_segments": 0,
            },
        )
        row["segments_seen"] += 1
        row["auto_added_samples"] += int(item.auto_added_samples)
        row["net_new_embeddings"] += int(item.net_new_embeddings)
        row["duplicate_rejections"] += int(item.duplicate_rejections)
        if item.decision == "review":
            row["review_segments"] += 1
        if item.decision == "reject":
            row["reject_segments"] += 1
    return list(grouped.values())


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_parent_dir(str(path))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: object) -> None:
    ensure_parent_dir(str(path))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def export_review_crops(
    video_path: Path,
    review_requests: dict[int, list[VideoDecision]],
    review_dir: Path,
) -> None:
    if not review_requests:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video for review crop export: {video_path}")
    try:
        current_frame_idx = 0
        sorted_requests = sorted(review_requests.items(), key=lambda item: item[0])
        request_index = 0
        while request_index < len(sorted_requests):
            target_frame_idx, decisions = sorted_requests[request_index]
            ok, frame = cap.read()
            if not ok:
                break
            if current_frame_idx < target_frame_idx:
                current_frame_idx += 1
                continue
            if current_frame_idx > target_frame_idx:
                request_index += 1
                continue
            for decision in decisions:
                crop = bbox_to_crop(frame, decision.representative_bbox)
                if crop.size == 0:
                    continue
                student_dir = review_dir / decision.student_key
                student_dir.mkdir(parents=True, exist_ok=True)
                file_name = (
                    f"{decision.phase}_{Path(decision.video_name).stem}_track{decision.raw_track_id}_"
                    f"frame{decision.representative_frame_idx}.jpg"
                )
                crop_path = student_dir / file_name
                cv2.imwrite(str(crop_path), crop)
                decision.crop_path = crop_path.as_posix()
            request_index += 1
            current_frame_idx += 1
    finally:
        cap.release()


def build_growth_rows(tracks: Sequence[Track]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for track in tracks:
        metadata = dict(track.metadata or {})
        rows.append(
            {
                "track_id": int(track.track_id),
                "global_id": global_id_for_track(track.track_id),
                "name": metadata.get("name", ""),
                "roll_number": metadata.get("roll_number", ""),
                "student_key": metadata.get("student_key", ""),
                "embedding_count": len(track.embeddings),
                "harvested_sample_count": int(metadata.get("harvested_sample_count", 0)),
                "harvest_source_videos": ";".join(metadata.get("harvest_source_videos", []) or []),
                "harvest_phases": ";".join(metadata.get("harvest_phases", []) or []),
                "last_enriched_at": metadata.get("last_enriched_at", ""),
            }
        )
    return rows


def reconcile_bank(tracks: Dict[int, Track], max_bank: int, duplicate_sim_thresh: float) -> None:
    for track in tracks.values():
        track.embeddings, track.embedding_qualities, track.embedding_metadata = compact_embedding_bank(
            track.embeddings,
            track.embedding_qualities,
            track.embedding_metadata,
            max_bank=max_bank,
            duplicate_sim_thresh=duplicate_sim_thresh,
        )
        track.avg_embedding, track.best_embedding, track.best_embedding_quality, track.recent_embedding = summarize_embedding_bank(
            track.embeddings,
            track.embedding_qualities,
            track.embedding_metadata,
        )


def process_video(
    phase_name: str,
    video_path: Path,
    *,
    master_tracks: Dict[int, Track],
    backend: InsightFaceBackend,
    args: argparse.Namespace,
) -> list[VideoDecision]:
    tracker = FaceTracker(
        sim_thresh=args.sim_thresh,
        ttl=args.ttl,
        archive_ttl=args.archive_ttl,
        reid_sim_thresh=args.reid_sim_thresh,
        new_id_confirm_hits=args.new_id_confirm_hits,
        new_id_confirm_quality=args.new_id_confirm_quality,
        provisional_match_margin=args.provisional_match_margin,
        high_det_score=args.high_det_score,
    )
    loaded_tracks = copy_named_tracks(master_tracks)
    tracker.load_identity_memory(loaded_tracks, next_track_id=max(master_tracks.keys(), default=0) + 1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for harvest: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    process_fps = min(float(args.process_fps), fps) if args.process_fps > 0 else fps
    process_period_frames = max(1.0, fps / max(1e-6, process_fps))
    next_process_frame = 0.0

    samples_by_raw_track: Dict[int, list[TrackSample]] = defaultdict(list)
    frame_idx = 0
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
            if frame_idx + 1e-6 < next_process_frame:
                frame_idx += 1
                continue
            next_process_frame += process_period_frames
            detections = backend.infer(frame)
            tracker.step(detections, frame_idx)
            for observation in tracker.last_step_observations:
                sample = build_sample_from_observation(observation, process_fps)
                samples_by_raw_track[sample.raw_track_id].append(sample)
            processed += 1
            if processed > 0 and processed % 100 == 0:
                print(f"[{phase_name}] {video_path.name}: processed {processed} sampled frames", flush=True)
            frame_idx += 1
    finally:
        cap.release()

    decisions: list[VideoDecision] = []
    review_requests: dict[int, list[VideoDecision]] = defaultdict(list)
    for raw_track_id, samples in sorted(samples_by_raw_track.items(), key=lambda item: item[0]):
        if not samples:
            continue
        resolved_track_id = tracker.resolve_track_id(raw_track_id)
        if resolved_track_id <= 0:
            continue
        master_track = master_tracks.get(resolved_track_id)
        if master_track is None:
            continue
        metadata = dict(master_track.metadata or {})
        student_name = str(metadata.get("name", "") or "").strip()
        student_key = str(metadata.get("student_key", "") or "").strip()
        if not student_name and not student_key:
            continue

        usable_samples = [sample for sample in samples if usable_sample(sample, args.min_sample_quality, args.min_face_size)]
        if not usable_samples:
            continue

        scored_samples: list[tuple[float, TrackSample]] = []
        for sample in usable_samples:
            similarity = tracker._embedding_match_score(sample.embedding, master_track)
            scored_samples.append((float(similarity), sample))
        scored_samples.sort(key=lambda item: (item[0], item[1].quality, item[1].timestamp_seconds), reverse=True)
        best_similarity, representative_sample = scored_samples[0]
        stable_seconds = max(0.0, usable_samples[-1].timestamp_seconds - usable_samples[0].timestamp_seconds)
        decision = classify_harvest_candidate(
            best_similarity,
            good_crops=len(usable_samples),
            stable_seconds=stable_seconds,
            quality=float(representative_sample.quality),
            auto_add_similarity=args.auto_add_similarity,
            review_similarity=args.review_similarity,
            min_good_crops=args.min_track_good_crops,
            min_stable_seconds=args.min_stable_seconds,
            min_quality=args.min_sample_quality,
        )
        selected_samples = [item[1] for item in scored_samples[: max(1, int(args.max_samples_per_track))]]
        net_new = 0
        duplicate_rejections = 0
        auto_added_samples = 0
        reason = ""
        if decision == "auto_add":
            auto_added_samples = len(selected_samples)
            net_new, duplicate_rejections = merge_samples_into_track(
                master_track,
                selected_samples,
                phase=phase_name,
                video_path=video_path,
                max_bank=args.max_bank,
                duplicate_sim_thresh=args.duplicate_sim_thresh,
            )
            reason = "strong_match"
        elif decision == "review":
            reason = "borderline_match"
        else:
            reason = "similarity_below_threshold"

        video_decision = VideoDecision(
            phase=phase_name,
            video_name=video_path.name,
            video_path=video_path.as_posix(),
            raw_track_id=int(raw_track_id),
            resolved_track_id=int(resolved_track_id),
            global_id=global_id_for_track(resolved_track_id),
            name=student_name or student_key,
            roll_number=str(metadata.get("roll_number", "") or ""),
            student_key=student_key or student_name,
            decision=decision,
            best_similarity=float(best_similarity),
            best_quality=float(representative_sample.quality),
            good_crops=len(usable_samples),
            support_seconds=float(stable_seconds),
            first_frame_idx=min(sample.frame_idx for sample in samples),
            last_frame_idx=max(sample.frame_idx for sample in samples),
            representative_frame_idx=int(representative_sample.frame_idx),
            representative_bbox=list(representative_sample.bbox),
            auto_added_samples=int(auto_added_samples),
            net_new_embeddings=int(net_new),
            duplicate_rejections=int(duplicate_rejections),
            reason=reason,
        )
        decisions.append(video_decision)
        if decision == "review":
            review_requests[int(video_decision.representative_frame_idx)].append(video_decision)

    if review_requests:
        export_review_crops(video_path, review_requests, Path(args.review_dir) if args.review_dir else Path(args.output_dir) / "review_crops")
    return decisions


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    review_dir = Path(args.review_dir).resolve() if args.review_dir else output_dir / "review_crops"
    identity_db_path = Path(args.identity_db).resolve()
    registry_path = Path(args.registry).resolve()
    manifest_json = Path(args.manifest_json).resolve()
    manifest_csv = Path(args.manifest_csv).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    phase_specs = parse_phase_specs(args.phase)
    expanded_phases: list[tuple[str, list[Path]]] = []
    for phase in phase_specs:
        videos = expand_phase_videos(phase, max_videos=args.max_videos_per_phase)
        if not videos:
            print(f"Warning: no videos matched for phase '{phase.name}'", flush=True)
            continue
        expanded_phases.append((phase.name, videos))
    if not expanded_phases:
        raise RuntimeError("No videos were discovered for any harvest phase.")

    if args.backup:
        backup_paths = backup_identity_files(
            [identity_db_path, registry_path, manifest_json, manifest_csv],
            output_dir,
        )
        if backup_paths:
            print("Backed up identity files:", flush=True)
            for path in backup_paths:
                print(f"  - {path}", flush=True)

    identity_db = FaceIdentityDB(str(identity_db_path))
    next_track_id, stored_identities = identity_db.load()
    if not stored_identities:
        raise RuntimeError(f"No named identities found in: {identity_db_path}")
    master_tracks = copy_named_tracks(stored_identities)

    backend = InsightFaceBackend(
        det_size=args.det_size,
        ctx_id=args.ctx,
        min_face=args.min_face,
        det_thresh=args.det_thresh,
        tile_grid=args.tile_grid,
        tile_overlap=args.tile_overlap,
        primary_detector_name=args.primary_detector,
        backup_detector_name=args.backup_detector,
        enable_backup_detector=not bool(args.disable_backup_detector),
        adaface_weights=args.adaface_weights or None,
    )

    all_decisions: list[VideoDecision] = []
    for phase_name, videos in expanded_phases:
        print(f"Starting phase '{phase_name}' with {len(videos)} videos", flush=True)
        for video_path in videos:
            print(f"Harvesting {video_path.name}", flush=True)
            video_decisions = process_video(
                phase_name,
                video_path,
                master_tracks=master_tracks,
                backend=backend,
                args=args,
            )
            all_decisions.extend(video_decisions)
            reconcile_bank(master_tracks, max_bank=args.max_bank, duplicate_sim_thresh=args.duplicate_sim_thresh)
            tracker = FaceTracker(min_confirm_hits=2)
            tracker.archived_tracks = copy_named_tracks(master_tracks)
            tracker.next_track_id = max(next_track_id, max(master_tracks.keys(), default=0) + 1)
            identity_db.save(tracker)
            write_registry_json(registry_path, list(sorted(master_tracks.values(), key=lambda track: int(track.track_id))))
            write_manifests(manifest_json, manifest_csv, list(sorted(master_tracks.values(), key=lambda track: int(track.track_id))))

    reconcile_bank(master_tracks, max_bank=args.max_bank, duplicate_sim_thresh=args.duplicate_sim_thresh)
    final_tracker = FaceTracker(min_confirm_hits=2)
    final_tracker.archived_tracks = copy_named_tracks(master_tracks)
    final_tracker.next_track_id = max(next_track_id, max(master_tracks.keys(), default=0) + 1)
    saved_count = identity_db.save(final_tracker)
    named_tracks = list(sorted(master_tracks.values(), key=lambda track: int(track.track_id)))
    write_registry_json(registry_path, named_tracks)
    write_manifests(manifest_json, manifest_csv, named_tracks)

    decision_rows = [
        {
            "phase": item.phase,
            "video_name": item.video_name,
            "video_path": item.video_path,
            "raw_track_id": item.raw_track_id,
            "resolved_track_id": item.resolved_track_id,
            "global_id": item.global_id,
            "name": item.name,
            "roll_number": item.roll_number,
            "student_key": item.student_key,
            "decision": item.decision,
            "best_similarity": f"{item.best_similarity:.4f}",
            "best_quality": f"{item.best_quality:.4f}",
            "good_crops": item.good_crops,
            "support_seconds": f"{item.support_seconds:.2f}",
            "first_frame_idx": item.first_frame_idx,
            "last_frame_idx": item.last_frame_idx,
            "representative_frame_idx": item.representative_frame_idx,
            "auto_added_samples": item.auto_added_samples,
            "net_new_embeddings": item.net_new_embeddings,
            "duplicate_rejections": item.duplicate_rejections,
            "crop_path": item.crop_path,
            "reason": item.reason,
        }
        for item in all_decisions
    ]
    review_rows = [row for row in decision_rows if row["decision"] == "review"]
    phase_summary_rows = build_phase_summary_rows(all_decisions)
    growth_rows = build_growth_rows(named_tracks)

    write_json(output_dir / "harvest_manifest.json", decision_rows)
    write_csv(
        output_dir / "harvest_manifest.csv",
        decision_rows,
        [
            "phase",
            "video_name",
            "video_path",
            "raw_track_id",
            "resolved_track_id",
            "global_id",
            "name",
            "roll_number",
            "student_key",
            "decision",
            "best_similarity",
            "best_quality",
            "good_crops",
            "support_seconds",
            "first_frame_idx",
            "last_frame_idx",
            "representative_frame_idx",
            "auto_added_samples",
            "net_new_embeddings",
            "duplicate_rejections",
            "crop_path",
            "reason",
        ],
    )
    write_csv(
        output_dir / "review_queue.csv",
        review_rows,
        [
            "phase",
            "video_name",
            "video_path",
            "raw_track_id",
            "resolved_track_id",
            "global_id",
            "name",
            "roll_number",
            "student_key",
            "best_similarity",
            "best_quality",
            "good_crops",
            "support_seconds",
            "representative_frame_idx",
            "crop_path",
            "reason",
        ],
    )
    write_csv(
        output_dir / "harvest_phase_summary.csv",
        phase_summary_rows,
        [
            "phase",
            "global_id",
            "name",
            "roll_number",
            "student_key",
            "segments_seen",
            "auto_added_samples",
            "net_new_embeddings",
            "duplicate_rejections",
            "review_segments",
            "reject_segments",
        ],
    )
    write_csv(
        output_dir / "identity_growth_report.csv",
        growth_rows,
        [
            "track_id",
            "global_id",
            "name",
            "roll_number",
            "student_key",
            "embedding_count",
            "harvested_sample_count",
            "harvest_source_videos",
            "harvest_phases",
            "last_enriched_at",
        ],
    )

    print(f"Saved enriched identity DB with {saved_count} named students to: {identity_db_path}", flush=True)
    print(f"Harvest manifest: {output_dir / 'harvest_manifest.csv'}", flush=True)
    print(f"Review queue: {output_dir / 'review_queue.csv'}", flush=True)
    print(f"Growth report: {output_dir / 'identity_growth_report.csv'}", flush=True)


if __name__ == "__main__":
    main()
