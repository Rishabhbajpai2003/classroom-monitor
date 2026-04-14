from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def cluster_support_score(track: Any) -> float:
    hits = float(getattr(track, "hits", 0) or 0)
    best_score = float(getattr(track, "best_score", 0.0) or 0.0)
    best_embedding_quality = float(getattr(track, "best_embedding_quality", 0.0) or 0.0)
    embedding_count = float(len(getattr(track, "embeddings", []) or []))
    appearance_quality = float(getattr(track, "best_appearance_quality", 0.0) or 0.0)
    return hits + (3.0 * best_score) + (4.0 * max(best_embedding_quality, appearance_quality)) + (0.35 * embedding_count)


def build_unknown_cluster_record(track_id: int, track: Any) -> dict[str, object]:
    metadata = dict(getattr(track, "metadata", {}) or {})
    embedding_metadata = [dict(item) for item in list(getattr(track, "embedding_metadata", []) or [])]
    best_meta = embedding_metadata[0] if embedding_metadata else {}
    representative_frame_idx = int(best_meta.get("frame_idx", best_meta.get("source_frame_idx", 0)) or 0)
    representative_timestamp = _safe_float(best_meta.get("source_timestamp_seconds", best_meta.get("timestamp_seconds", 0.0)))
    attribute_signature = str(getattr(track, "last_attribute_signature", "") or "frontal")
    bank_family_used = str(getattr(track, "last_bank_family_used", "") or "base")
    attempted_banks = []
    for item in embedding_metadata:
        bank_family = str((item or {}).get("bank_family", "") or "").strip()
        if bank_family and bank_family not in attempted_banks:
            attempted_banks.append(bank_family)
    return {
        "cluster_id": f"UNK_{abs(int(track_id)):04d}",
        "track_id": int(track_id),
        "display_name": str(metadata.get("name", "") or "Unknown"),
        "support_score": round(cluster_support_score(track), 4),
        "hits": int(getattr(track, "hits", 0) or 0),
        "best_score": round(float(getattr(track, "best_score", 0.0) or 0.0), 4),
        "best_embedding_quality": round(float(getattr(track, "best_embedding_quality", 0.0) or 0.0), 4),
        "embedding_count": len(getattr(track, "embeddings", []) or []),
        "first_frame_idx": int(getattr(track, "first_frame_idx", 0) or 0),
        "last_frame_idx": int(getattr(track, "last_frame_idx", 0) or 0),
        "representative_frame_idx": representative_frame_idx,
        "representative_timestamp_seconds": representative_timestamp,
        "attribute_signature": attribute_signature,
        "bank_family_used": bank_family_used,
        "attempted_bank_families": attempted_banks,
        "bbox": np.asarray(getattr(track, "bbox", np.zeros(4, dtype=np.float32)), dtype=np.float32).tolist(),
        "metadata": metadata,
        "embeddings": [np.asarray(item, dtype=np.float32).tolist() for item in list(getattr(track, "embeddings", []) or [])],
        "embedding_qualities": [float(item) for item in list(getattr(track, "embedding_qualities", []) or [])],
        "embedding_metadata": embedding_metadata,
    }


def write_unknown_review_package(
    output_dir: str | Path,
    cluster_rows: Sequence[dict[str, object]],
) -> tuple[Path, Path, Path]:
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / "unknown_clusters.csv"
    json_path = base_dir / "unknown_clusters.json"
    assignments_path = base_dir / "unknown_assignments.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(list(cluster_rows), handle, indent=2)

    fieldnames = [
        "cluster_id",
        "track_id",
        "display_name",
        "support_score",
        "hits",
        "best_score",
        "best_embedding_quality",
        "embedding_count",
        "first_frame_idx",
        "last_frame_idx",
        "representative_frame_idx",
        "representative_timestamp_seconds",
        "attribute_signature",
        "bank_family_used",
        "attempted_bank_families",
        "crop_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in cluster_rows:
            payload = {key: row.get(key, "") for key in fieldnames}
            attempted = payload.get("attempted_bank_families", "")
            if isinstance(attempted, (list, tuple)):
                payload["attempted_bank_families"] = ",".join(str(item) for item in attempted)
            writer.writerow(payload)

    with assignments_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cluster_id", "assign_to_student_key", "assign_to_global_id", "status", "notes"],
        )
        writer.writeheader()
        for row in cluster_rows:
            writer.writerow(
                {
                    "cluster_id": row.get("cluster_id", ""),
                    "assign_to_student_key": "",
                    "assign_to_global_id": "",
                    "status": "pending",
                    "notes": "",
                }
            )
    return csv_path, json_path, assignments_path


def read_unknown_assignments(path: str | Path) -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_unknown_clusters(path: str | Path) -> list[dict[str, object]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    return []
