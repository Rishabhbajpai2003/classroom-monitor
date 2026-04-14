from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def iter_roster_student_dirs(student_details_root: str | Path = "student-details") -> list[Path]:
    root = Path(student_details_root)
    if not root.exists():
        return []
    return [
        path
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if path.is_dir() and not path.name.startswith(".")
    ]


def roster_size(student_details_root: str | Path = "student-details") -> int:
    return len(iter_roster_student_dirs(student_details_root))


def capped_reportable_ids(
    named_ids: Sequence[str],
    unnamed_candidates: Iterable[tuple[str, float]],
    *,
    roster_limit: int,
) -> set[str]:
    """
    Return at most `roster_limit` identities, prioritising named roster students
    first and then highest-supported temporary identities.
    """
    limit = max(0, int(roster_limit))
    ordered_named = [student_id for student_id in dict.fromkeys(str(item) for item in named_ids if item)]
    if len(ordered_named) >= limit:
        return set(ordered_named[:limit])

    selected = set(ordered_named)
    remaining = max(0, limit - len(selected))
    if remaining <= 0:
        return selected

    ranked_unknown = sorted(
        (
            (str(student_id), float(score))
            for student_id, score in unnamed_candidates
            if student_id and str(student_id).startswith("TEMP_")
        ),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    for student_id, _score in ranked_unknown[:remaining]:
        selected.add(student_id)
    return selected
