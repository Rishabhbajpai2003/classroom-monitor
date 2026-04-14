from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


MANIFEST_FIELDS = [
    "student_folder",
    "source_image",
    "generator_type",
    "family",
    "tag",
    "combination_tag",
    "status",
    "rejection_reason",
    "output_image",
]


def load_manifest(path: str | Path) -> List[Dict[str, str]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_manifest(path: str | Path, rows: List[Dict[str, str]]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows: List[Dict[str, str]] = []
    for row in rows:
        normalized = {field: str(row.get(field, "") or "") for field in MANIFEST_FIELDS}
        normalized_rows.append(normalized)
    normalized_rows.sort(
        key=lambda item: (
            item["student_folder"].lower(),
            item["source_image"].lower(),
            item["generator_type"].lower(),
            item["family"].lower(),
            item["tag"].lower(),
            item["combination_tag"].lower(),
        )
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(normalized_rows)


def upsert_manifest_row(rows: List[Dict[str, str]], new_row: Dict[str, str]) -> List[Dict[str, str]]:
    key = (
        str(new_row.get("student_folder", "") or "").lower(),
        str(new_row.get("source_image", "") or "").lower(),
        str(new_row.get("generator_type", "") or "").lower(),
        str(new_row.get("family", "") or "").lower(),
        str(new_row.get("tag", "") or "").lower(),
        str(new_row.get("combination_tag", "") or "").lower(),
    )
    updated = False
    next_rows: List[Dict[str, str]] = []
    for row in rows:
        row_key = (
            str(row.get("student_folder", "") or "").lower(),
            str(row.get("source_image", "") or "").lower(),
            str(row.get("generator_type", "") or "").lower(),
            str(row.get("family", "") or "").lower(),
            str(row.get("tag", "") or "").lower(),
            str(row.get("combination_tag", "") or "").lower(),
        )
        if row_key == key:
            next_rows.append({field: str(new_row.get(field, "") or "") for field in MANIFEST_FIELDS})
            updated = True
        else:
            next_rows.append({field: str(row.get(field, "") or "") for field in MANIFEST_FIELDS})
    if not updated:
        next_rows.append({field: str(new_row.get(field, "") or "") for field in MANIFEST_FIELDS})
    return next_rows
