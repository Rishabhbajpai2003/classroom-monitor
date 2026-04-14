from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.unknown_review import load_unknown_clusters, read_unknown_assignments  # noqa: E402
from detectors.face_detector.run import DEFAULT_IDENTITY_DB_PATH, FaceIdentityDB, FaceTracker  # noqa: E402
from scripts.rebuild_identity_db_from_student_details import write_manifests, write_registry_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reviewed unknown face clusters into existing roster students.")
    parser.add_argument(
        "--identity-db",
        default=str(PROJECT_ROOT / DEFAULT_IDENTITY_DB_PATH),
        help="Path to the persistent identity DB to update.",
    )
    parser.add_argument(
        "--unknown-json",
        required=True,
        help="Path to unknown_clusters.json exported from a run.",
    )
    parser.add_argument(
        "--assignments",
        required=True,
        help="Path to unknown_assignments.csv filled in by the reviewer.",
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
    parser.add_argument(
        "--max-bank",
        type=int,
        default=48,
        help="Maximum embedding bank size per student.",
    )
    parser.add_argument(
        "--duplicate-sim-thresh",
        type=float,
        default=0.92,
        help="Duplicate similarity threshold for merged embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity_db_path = Path(args.identity_db).resolve()
    unknown_json_path = Path(args.unknown_json).resolve()
    assignments_path = Path(args.assignments).resolve()
    registry_path = Path(args.registry).resolve()
    manifest_json_path = Path(args.manifest_json).resolve()
    manifest_csv_path = Path(args.manifest_csv).resolve()

    identity_db = FaceIdentityDB(str(identity_db_path))
    next_track_id, identities = identity_db.load()
    if not identities:
        raise RuntimeError(f"No identities found in {identity_db_path}")

    clusters = {str(row.get("cluster_id", "")).strip(): row for row in load_unknown_clusters(unknown_json_path)}
    assignments = [row for row in read_unknown_assignments(assignments_path) if str(row.get("status", "")).strip().lower() not in {"", "pending", "skip"}]
    if not assignments:
        print("No completed assignments found. Nothing to merge.")
        return

    by_global_id = {f"STU_{int(track_id):03d}": track for track_id, track in identities.items()}
    by_student_key = {
        str((track.metadata or {}).get("student_key", "") or "").strip(): track
        for track in identities.values()
        if str((track.metadata or {}).get("student_key", "") or "").strip()
    }

    merged_count = 0
    skipped_count = 0
    for assignment in assignments:
        cluster_id = str(assignment.get("cluster_id", "") or "").strip()
        target_global_id = str(assignment.get("assign_to_global_id", "") or "").strip()
        target_student_key = str(assignment.get("assign_to_student_key", "") or "").strip()
        cluster = clusters.get(cluster_id)
        if cluster is None:
            skipped_count += 1
            continue
        target_track = by_global_id.get(target_global_id) if target_global_id else None
        if target_track is None and target_student_key:
            target_track = by_student_key.get(target_student_key)
        if target_track is None:
            skipped_count += 1
            continue

        existing_tokens = {
            str((item or {}).get("source_token", "") or "").strip()
            for item in list(target_track.embedding_metadata or [])
            if str((item or {}).get("source_token", "") or "").strip()
        }
        embeddings = list(cluster.get("embeddings", []) or [])
        qualities = list(cluster.get("embedding_qualities", []) or [])
        metadata_items = list(cluster.get("embedding_metadata", []) or [])
        for idx, embedding in enumerate(embeddings):
            source_token = f"assigned_unknown:{cluster_id}:{idx}"
            if source_token in existing_tokens:
                continue
            quality = float(qualities[idx]) if idx < len(qualities) else 0.5
            metadata = dict(metadata_items[idx]) if idx < len(metadata_items) and isinstance(metadata_items[idx], dict) else {}
            metadata["source_kind"] = "assigned_unknown"
            metadata["source_cluster_id"] = cluster_id
            metadata["source_token"] = source_token
            metadata["review_assignment_file"] = assignments_path.as_posix()
            target_track.update_embedding_bank(
                np.asarray(embedding, dtype=np.float32),
                sample_quality=quality,
                sample_metadata=metadata,
                max_bank=args.max_bank,
                duplicate_sim_thresh=args.duplicate_sim_thresh,
            )
            merged_count += 1

    tracker = FaceTracker(min_confirm_hits=2)
    tracker.archived_tracks = {int(track_id): track for track_id, track in identities.items()}
    tracker.next_track_id = max(int(next_track_id), max(identities.keys(), default=0) + 1)
    saved = identity_db.save(tracker)
    named_tracks = list(sorted(identities.values(), key=lambda track: int(track.track_id)))
    write_registry_json(registry_path, named_tracks)
    write_manifests(manifest_json_path, manifest_csv_path, named_tracks)

    resolution_payload = {
        "identity_db_path": identity_db_path.as_posix(),
        "registry": registry_path.as_posix(),
        "manifest_json": manifest_json_path.as_posix(),
        "manifest_csv": manifest_csv_path.as_posix(),
        "unknown_json": unknown_json_path.as_posix(),
        "assignments": assignments_path.as_posix(),
        "merged_embeddings": merged_count,
        "skipped_assignments": skipped_count,
        "saved_identities": saved,
    }
    resolution_path = assignments_path.with_name("unknown_assignment_merge_result.json")
    with resolution_path.open("w", encoding="utf-8") as handle:
        json.dump(resolution_payload, handle, indent=2)

    print(f"Merged embeddings: {merged_count}")
    print(f"Skipped assignments: {skipped_count}")
    print(f"Saved identities: {saved}")
    print(f"Result: {resolution_path}")


if __name__ == "__main__":
    main()
