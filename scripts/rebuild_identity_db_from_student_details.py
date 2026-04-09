from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from PIL import Image, ImageOps
except Exception as exc:  # pragma: no cover - dependency is expected in runtime
    raise RuntimeError("Pillow is required to rebuild the identity DB.") from exc

try:
    from pillow_heif import register_heif_opener
except Exception:
    register_heif_opener = None

from detectors.face_detector.run import (  # noqa: E402
    DEFAULT_IDENTITY_DB_PATH,
    FaceIdentityDB,
    FaceTracker,
    InsightFaceBackend,
    Track,
    ensure_parent_dir,
)


SUPPORTED_IMAGE_EXTENSIONS = {".heic", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class StudentFolder:
    folder: Path
    student_key: str
    name: str
    roll_number: str
    image_paths: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the persistent face identity DB from student-details folders.",
    )
    parser.add_argument(
        "--student-details",
        default=str(PROJECT_ROOT / "student-details"),
        help="Folder containing name-rollno subfolders with Photo-* and optional ID-* images.",
    )
    parser.add_argument(
        "--identity-db",
        default=str(PROJECT_ROOT / DEFAULT_IDENTITY_DB_PATH),
        help="Persistent identity DB path to overwrite.",
    )
    parser.add_argument(
        "--registry",
        default=str(PROJECT_ROOT / "outputs" / "student_registry.json"),
        help="Compatibility registry JSON to overwrite.",
    )
    parser.add_argument(
        "--manifest-json",
        default=str(PROJECT_ROOT / "outputs" / "student_identity_manifest.json"),
        help="JSON manifest of the rebuilt IDs.",
    )
    parser.add_argument(
        "--manifest-csv",
        default=str(PROJECT_ROOT / "outputs" / "student_identity_manifest.csv"),
        help="CSV manifest of the rebuilt IDs.",
    )
    parser.add_argument(
        "--backup-dir",
        default=str(PROJECT_ROOT / "outputs" / "identity_seed_backups"),
        help="Where existing identity files should be backed up before overwrite.",
    )
    parser.add_argument("--ctx", type=int, default=0, help="InsightFace GPU id. Use -1 for CPU.")
    parser.add_argument("--det-size", type=int, default=1280, help="Face detector size.")
    parser.add_argument("--det-thresh", type=float, default=0.20, help="Face detector threshold.")
    parser.add_argument("--min-face", type=int, default=8, help="Minimum face size in pixels.")
    parser.add_argument("--tile-grid", type=int, default=2, help="Tiled HEIC/photo detection grid.")
    parser.add_argument("--tile-overlap", type=float, default=0.20, help="Tile overlap ratio.")
    parser.add_argument(
        "--max-images-per-student",
        type=int,
        default=12,
        help="Cap the number of images read per student folder.",
    )
    return parser.parse_args()


def split_student_key(student_key: str) -> tuple[str, str]:
    cleaned = str(student_key).strip()
    if "-" not in cleaned:
        return cleaned, ""
    name, roll_number = cleaned.rsplit("-", 1)
    return name.strip(), roll_number.strip()


def iter_student_folders(base_dir: Path, max_images_per_student: int) -> list[StudentFolder]:
    students: list[StudentFolder] = []
    for folder in sorted(base_dir.iterdir(), key=lambda item: item.name.lower()):
        if not folder.is_dir():
            continue
        if folder.name.startswith("."):
            continue
        image_paths = [
            path
            for path in sorted(folder.iterdir(), key=lambda item: item.name.lower())
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if not image_paths:
            continue

        photo_paths = [path for path in image_paths if path.name.lower().startswith("photo-")]
        id_paths = [path for path in image_paths if path.name.lower().startswith("id-")]
        other_paths = [path for path in image_paths if path not in photo_paths and path not in id_paths]
        ordered_paths = (photo_paths + id_paths + other_paths)[: max(1, int(max_images_per_student))]

        name, roll_number = split_student_key(folder.name)
        students.append(
            StudentFolder(
                folder=folder,
                student_key=folder.name,
                name=name,
                roll_number=roll_number,
                image_paths=ordered_paths,
            )
        )
    return students


def read_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is not None:
        return image

    if path.suffix.lower() == ".heic" and register_heif_opener is None:
        raise RuntimeError(
            "HEIC decoding requires pillow-heif. Install it with: pip install pillow-heif"
        )

    if register_heif_opener is not None:
        register_heif_opener()

    with Image.open(path) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        rgb = np.asarray(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def resized_variants(image_bgr: np.ndarray) -> list[np.ndarray]:
    variants = [image_bgr]
    max_dim = max(image_bgr.shape[:2])
    for limit in (1800, 1400, 1100):
        if max_dim <= limit:
            continue
        scale = float(limit) / float(max_dim)
        resized = cv2.resize(
            image_bgr,
            (max(1, int(round(image_bgr.shape[1] * scale))), max(1, int(round(image_bgr.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        variants.append(resized)
    return variants


def pick_best_detection(backend: InsightFaceBackend, image_bgr: np.ndarray):
    best_det = None
    best_score = -1.0
    for candidate in resized_variants(image_bgr):
        detections = backend.infer(candidate)
        for det in detections:
            bbox = np.asarray(det.bbox, dtype=np.float32)
            width = max(0.0, float(bbox[2] - bbox[0]))
            height = max(0.0, float(bbox[3] - bbox[1]))
            area_bonus = min(width * height, 30000.0) / 30000.0
            score = float(det.quality) + 0.10 * float(det.score) + 0.05 * area_bonus
            if score > best_score:
                best_score = score
                best_det = det
    return best_det


def create_seed_track(track_id: int, student: StudentFolder, backend: InsightFaceBackend) -> tuple[Optional[Track], list[str]]:
    track = Track(
        track_id=track_id,
        bbox=np.zeros(4, dtype=np.float32),
        last_frame_idx=0,
        first_frame_idx=0,
        hits=0,
        misses=0,
        best_score=0.0,
        metadata={
            "name": student.name,
            "roll_number": student.roll_number,
            "student_key": student.student_key,
            "source_folder": student.folder.as_posix(),
            "source_images": [path.name for path in student.image_paths],
        },
    )
    used_sources: list[str] = []
    sample_index = 0
    for image_path in student.image_paths:
        image_bgr = read_image_bgr(image_path)
        det = pick_best_detection(backend, image_bgr)
        if det is None:
            continue
        track.best_score = max(track.best_score, float(det.score))
        track.update_embedding_bank(det.embedding, sample_quality=float(det.quality))
        if det.appearance is not None:
            track.update_appearance_bank(det.appearance, sample_quality=float(det.quality))
        sample_index += 1
        used_sources.append(image_path.name)

    if track.avg_embedding is None:
        return None, []

    track.hits = max(2, sample_index * 2)
    track.first_frame_idx = 0
    track.last_frame_idx = max(0, sample_index - 1)
    track.persistent_identity = True
    track.metadata["seed_sample_count"] = sample_index
    track.metadata["seed_sources"] = used_sources
    return track, used_sources


def backup_existing_files(paths: list[Path], backup_dir: Path) -> list[Path]:
    backup_paths: list[Path] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            continue
        backup_path = backup_dir / f"{path.name}.{timestamp}.bak"
        shutil.copy2(path, backup_path)
        backup_paths.append(backup_path)
    return backup_paths


def write_registry_json(registry_path: Path, tracks: list[Track]) -> None:
    ensure_parent_dir(str(registry_path))
    payload: dict[str, dict[str, object]] = {}
    seeded_at = datetime.now(timezone.utc).isoformat()
    for track in tracks:
        global_id = f"STU_{int(track.track_id):03d}"
        metadata = dict(track.metadata or {})
        payload[global_id] = {
            "global_id": global_id,
            "track_id": int(track.track_id),
            "name": metadata.get("name"),
            "roll_number": metadata.get("roll_number"),
            "student_key": metadata.get("student_key"),
            "embedding": np.asarray(track.avg_embedding, dtype=np.float32).tolist() if track.avg_embedding is not None else None,
            "embeddings": [np.asarray(item, dtype=np.float32).tolist() for item in track.embeddings],
            "embedding_qualities": [float(value) for value in track.embedding_qualities],
            "first_seen": seeded_at,
            "last_seen": seeded_at,
            "seed_sources": metadata.get("seed_sources", []),
        }
    with registry_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_manifests(manifest_json: Path, manifest_csv: Path, tracks: list[Track]) -> None:
    ensure_parent_dir(str(manifest_json))
    ensure_parent_dir(str(manifest_csv))
    rows: list[dict[str, object]] = []
    for track in tracks:
        metadata = dict(track.metadata or {})
        row = {
            "track_id": int(track.track_id),
            "global_id": f"STU_{int(track.track_id):03d}",
            "name": metadata.get("name", ""),
            "roll_number": metadata.get("roll_number", ""),
            "student_key": metadata.get("student_key", ""),
            "seed_sample_count": int(metadata.get("seed_sample_count", 0)),
            "seed_sources": list(metadata.get("seed_sources", [])),
            "source_folder": metadata.get("source_folder", ""),
        }
        rows.append(row)

    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "track_id",
                "global_id",
                "name",
                "roll_number",
                "student_key",
                "seed_sample_count",
                "seed_sources",
                "source_folder",
            ],
        )
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["seed_sources"] = ";".join(serializable["seed_sources"])
            writer.writerow(serializable)


def save_identity_db(identity_db_path: Path, tracks: list[Track]) -> int:
    identity_db = FaceIdentityDB(str(identity_db_path))
    tracker = FaceTracker(min_confirm_hits=2)
    tracker.archived_tracks = {int(track.track_id): track for track in tracks}
    tracker.next_track_id = max((int(track.track_id) for track in tracks), default=0) + 1
    return identity_db.save(tracker)


def main() -> None:
    args = parse_args()
    student_details_dir = Path(args.student_details).resolve()
    identity_db_path = Path(args.identity_db).resolve()
    registry_path = Path(args.registry).resolve()
    manifest_json = Path(args.manifest_json).resolve()
    manifest_csv = Path(args.manifest_csv).resolve()
    backup_dir = Path(args.backup_dir).resolve()

    if not student_details_dir.exists():
        raise FileNotFoundError(f"student-details folder not found: {student_details_dir}")

    students = iter_student_folders(student_details_dir, args.max_images_per_student)
    if not students:
        raise RuntimeError(f"No student folders with supported images found in: {student_details_dir}")

    backend = InsightFaceBackend(
        det_size=args.det_size,
        ctx_id=args.ctx,
        min_face=args.min_face,
        det_thresh=args.det_thresh,
        tile_grid=args.tile_grid,
        tile_overlap=args.tile_overlap,
    )

    backup_paths = backup_existing_files(
        [identity_db_path, registry_path, manifest_json, manifest_csv],
        backup_dir,
    )

    seeded_tracks: list[Track] = []
    skipped: list[tuple[str, str]] = []
    next_track_id = 1
    for student in students:
        try:
            track, used_sources = create_seed_track(next_track_id, student, backend)
        except Exception as exc:
            skipped.append((student.student_key, f"error: {exc}"))
            continue
        if track is None:
            skipped.append((student.student_key, "no-face-detected"))
            continue
        if not used_sources:
            skipped.append((student.student_key, "no-usable-sources"))
            continue
        seeded_tracks.append(track)
        next_track_id += 1

    if not seeded_tracks:
        raise RuntimeError("No identities could be created from student-details.")

    saved_count = save_identity_db(identity_db_path, seeded_tracks)
    write_registry_json(registry_path, seeded_tracks)
    write_manifests(manifest_json, manifest_csv, seeded_tracks)

    print(f"Seeded identities: {saved_count}")
    print(f"Identity DB: {identity_db_path}")
    print(f"Registry: {registry_path}")
    print(f"Manifest JSON: {manifest_json}")
    print(f"Manifest CSV: {manifest_csv}")
    if backup_paths:
        print("Backups:")
        for path in backup_paths:
            print(f"  - {path}")
    if skipped:
        print("Skipped folders:")
        for student_key, reason in skipped:
            print(f"  - {student_key}: {reason}")


if __name__ == "__main__":
    main()
