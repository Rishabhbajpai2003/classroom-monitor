from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from app.models.roster_policy import capped_reportable_ids, roster_size


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def read_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_row_count(csv_path: Path) -> int:
    return len(read_csv_rows(csv_path))


def load_identity_metadata_map(identity_db_path: Path) -> dict[str, dict[str, str]]:
    if not identity_db_path.exists():
        return {}
    with identity_db_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    identities = payload.get("identities", []) if isinstance(payload, dict) else []
    metadata_map: dict[str, dict[str, str]] = {}
    for record in identities:
        track_id = _safe_int(record.get("track_id"), 0)
        if track_id <= 0:
            continue
        metadata = record.get("metadata") or {}
        metadata_map[f"STU_{track_id:03d}"] = {
            "display_name": str(metadata.get("name", "") or "").strip() or f"STU_{track_id:03d}",
            "student_name": str(metadata.get("name", "") or "").strip(),
            "roll_number": str(metadata.get("roll_number", "") or "").strip(),
            "student_key": str(metadata.get("student_key", "") or "").strip(),
        }
    return metadata_map


def build_run_manifest(run_dir: Path, source_video: str, identity_db_path: Path | None = None) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    records: list[dict] = []
    for path in sorted(run_dir.iterdir(), key=lambda item: item.name.lower()):
        if path.name == "review_frames":
            continue
        row_count = ""
        if path.is_file() and path.suffix.lower() == ".csv":
            row_count = csv_row_count(path)
        records.append(
            {
                "path": path.name,
                "kind": "directory" if path.is_dir() else path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "row_count": row_count,
            }
        )

    manifest_json_path = run_dir / "run_manifest.json"
    manifest_csv_path = run_dir / "run_manifest.csv"
    manifest_payload = {
        "source_video": source_video,
        "run_dir": str(run_dir),
        "identity_db_path": str(identity_db_path) if identity_db_path is not None else "",
        "files": records,
    }
    with manifest_json_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    with manifest_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "kind", "size_bytes", "row_count"])
        writer.writeheader()
        writer.writerows(records)

    return manifest_json_path, manifest_csv_path


def _ensure_student_row(
    table: dict[str, dict],
    student_id: str,
    identity_map: dict[str, dict[str, str]],
) -> dict:
    row = table.get(student_id)
    if row is None:
        meta = identity_map.get(
            student_id,
            {
                "display_name": "Unknown" if str(student_id).startswith("TEMP_") else student_id,
                "student_name": "",
                "roll_number": "",
                "student_key": "",
            },
        )
        row = {
            "global_student_id": student_id,
            "display_name": meta.get("display_name", ""),
            "student_name": meta.get("student_name", ""),
            "roll_number": meta.get("roll_number", ""),
            "student_key": meta.get("student_key", ""),
            "presence_time_seconds": "",
            "present": "",
            "total_frames": "",
            "seat_id": "",
            "seat_rank": "",
            "row_rank": "",
            "weighted_avg_seat_rank": "",
            "out_of_class_seconds": "",
            "late_arrival": "",
            "early_exit": "",
            "attention_state": "",
            "attention_confidence": "",
            "attention_mode": "",
            "attention_percentage": "",
            "hand_raise_count": "",
            "hand_raise_seconds": "",
            "dominant_activity": "",
            "activity_confidence": "",
            "activity_mode": "",
            "activity_reason": "",
            "electronics_seconds": "",
            "note_taking_seconds": "",
            "speaking_frames": 0,
            "speaking_seconds": 0.0,
            "mean_speech_prob": "",
            "max_speech_prob": "",
            "speech_segments": 0,
            "class_related_segments": 0,
            "off_topic_segments": 0,
            "unknown_topic_segments": 0,
        }
        table[student_id] = row
    return row


def _merge_metadata(dest: dict, source: dict) -> None:
    for key in ("display_name", "student_name", "roll_number", "student_key"):
        current = str(dest.get(key, "") or "").strip()
        incoming = str(source.get(key, "") or "").strip()
        if not current and incoming:
            dest[key] = incoming


def _iter_summary_rows(rows: Iterable[dict], record_type: str = "summary") -> Iterable[dict]:
    for row in rows:
        if str(row.get("record_type", record_type) or record_type) == record_type:
            yield row


def build_final_student_summary(run_dir: Path, identity_db_path: Path, fps: float | None = None) -> Path:
    run_dir = Path(run_dir)
    identity_map = load_identity_metadata_map(identity_db_path)
    table: dict[str, dict] = {}

    attendance_rows = read_csv_rows(run_dir / "attendance_report.csv")
    for row in attendance_rows:
        student_id = str(row.get("Global_Student_ID", "") or "").strip()
        if not student_id:
            continue
        out = _ensure_student_row(table, student_id, identity_map)
        _merge_metadata(
            out,
            {
                "display_name": row.get("Display_Name", ""),
                "student_name": row.get("Student_Name", ""),
                "roll_number": row.get("Roll_Number", ""),
                "student_key": row.get("Student_Key", ""),
            },
        )
        out["presence_time_seconds"] = row.get("Presence_Time_Seconds", "")
        out["present"] = row.get("Present", "")
        out["total_frames"] = row.get("Total_Frames", "")
        out["seat_id"] = row.get("Seat_ID", "")
        out["seat_rank"] = row.get("Seat_Rank", "")
        out["row_rank"] = row.get("Row_Rank", "")
        out["weighted_avg_seat_rank"] = row.get("Weighted_Avg_Seat_Rank", "")
        out["out_of_class_seconds"] = row.get("Out_of_Class_Seconds", "")
        out["late_arrival"] = row.get("Late_Arrival", "")
        out["early_exit"] = row.get("Early_Exit", "")
        out["hand_raise_count"] = row.get("Hand_Raise_Count", "")
        out["hand_raise_seconds"] = row.get("Hand_Raise_Seconds", "")
        out["attention_state"] = row.get("Attention_State", "")
        out["attention_confidence"] = row.get("Attention_Confidence", "")
        out["attention_mode"] = row.get("Attention_Mode", "")
        out["attention_percentage"] = row.get("Attention_Percentage", "")

    activity_rows = read_csv_rows(run_dir / "person_activity_summary.csv")
    for row in _iter_summary_rows(activity_rows):
        student_id = str(row.get("person_id", "") or "").strip()
        if not student_id:
            continue
        out = _ensure_student_row(table, student_id, identity_map)
        _merge_metadata(out, row)
        out["dominant_activity"] = row.get("dominant_activity", "")
        out["activity_confidence"] = row.get("activity_confidence", "")
        out["activity_mode"] = row.get("activity_mode", "")
        out["activity_reason"] = row.get("activity_reason", "")
        if not out.get("seat_id"):
            out["seat_id"] = row.get("seat_id", "")
            out["seat_rank"] = row.get("seat_rank", "")
            out["row_rank"] = row.get("row_rank", "")
            out["weighted_avg_seat_rank"] = row.get("weighted_avg_seat_rank", "")

    for signal_name, column_name in (("device_use.csv", "electronics_seconds"), ("note_taking.csv", "note_taking_seconds")):
        for row in _iter_summary_rows(read_csv_rows(run_dir / signal_name)):
            student_id = str(row.get("person_id", "") or "").strip()
            if not student_id:
                continue
            out = _ensure_student_row(table, student_id, identity_map)
            _merge_metadata(out, row)
            out[column_name] = row.get("signal_seconds", "")

    speaking_rows = read_csv_rows(run_dir / "visual_speaking.csv")
    if speaking_rows:
        speaking_by_student: dict[str, list[float]] = defaultdict(list)
        speaking_frame_counts: dict[str, int] = defaultdict(int)
        for row in speaking_rows:
            student_id = str(row.get("global_id", "") or "").strip()
            if not student_id:
                continue
            prob = _safe_float(row.get("speech_prob"), 0.0)
            speaking_by_student[student_id].append(prob)
            if str(row.get("is_speaking", "0")) == "1":
                speaking_frame_counts[student_id] += 1
        for student_id, probs in speaking_by_student.items():
            out = _ensure_student_row(table, student_id, identity_map)
            speaking_frames = speaking_frame_counts.get(student_id, 0)
            out["speaking_frames"] = speaking_frames
            if fps and fps > 0:
                out["speaking_seconds"] = round(speaking_frames / fps, 3)
            out["mean_speech_prob"] = f"{sum(probs) / max(1, len(probs)):.4f}"
            out["max_speech_prob"] = f"{max(probs):.4f}"

    topic_rows = read_csv_rows(run_dir / "speech_topic_segments.csv")
    for row in topic_rows:
        student_id = str(row.get("global_id", "") or "").strip()
        if not student_id:
            continue
        out = _ensure_student_row(table, student_id, identity_map)
        _merge_metadata(out, row)
        out["speech_segments"] = int(out.get("speech_segments", 0) or 0) + 1
        topic_label = str(row.get("topic_label", "unknown") or "unknown")
        if topic_label == "class_related":
            out["class_related_segments"] = int(out.get("class_related_segments", 0) or 0) + 1
        elif topic_label == "off_topic":
            out["off_topic_segments"] = int(out.get("off_topic_segments", 0) or 0) + 1
        else:
            out["unknown_topic_segments"] = int(out.get("unknown_topic_segments", 0) or 0) + 1

    summary_path = run_dir / "final_student_summary.csv"
    roster_limit = roster_size("student-details")
    ordered_student_ids = sorted(table.keys())
    if roster_limit > 0:
        named_ids = [student_id for student_id in ordered_student_ids if not str(student_id).startswith("TEMP_")]
        unnamed_candidates = []
        for student_id in ordered_student_ids:
            if not str(student_id).startswith("TEMP_"):
                continue
            row = table.get(student_id, {})
            score = _safe_float(row.get("presence_time_seconds"), 0.0) + _safe_float(row.get("attention_confidence"), 0.0)
            unnamed_candidates.append((student_id, score))
        selected_ids = capped_reportable_ids(named_ids, unnamed_candidates, roster_limit=roster_limit)
    else:
        selected_ids = set(ordered_student_ids)
    fieldnames = [
        "global_student_id",
        "display_name",
        "student_name",
        "roll_number",
        "student_key",
        "presence_time_seconds",
        "present",
        "total_frames",
        "seat_id",
        "seat_rank",
        "row_rank",
        "weighted_avg_seat_rank",
        "out_of_class_seconds",
        "late_arrival",
        "early_exit",
        "attention_state",
        "attention_confidence",
        "attention_mode",
        "attention_percentage",
        "hand_raise_count",
        "hand_raise_seconds",
        "dominant_activity",
        "activity_confidence",
        "activity_mode",
        "activity_reason",
        "electronics_seconds",
        "note_taking_seconds",
        "speaking_frames",
        "speaking_seconds",
        "mean_speech_prob",
        "max_speech_prob",
        "speech_segments",
        "class_related_segments",
        "off_topic_segments",
        "unknown_topic_segments",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for student_id in ordered_student_ids:
            if student_id not in selected_ids:
                continue
            writer.writerow({key: table[student_id].get(key, "") for key in fieldnames})
    return summary_path
