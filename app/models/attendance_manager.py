"""
Attendance Manager — Tracks per-student presence and generates attendance records.

Monitors each student's visibility across frames and determines
attendance status based on configurable thresholds (presence time
and minimum visible frames).
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class StudentRecord:
    """
    Tracks attendance-related metrics for a single student.

    Attributes:
        global_id: Persistent student identifier.
        local_track_id: DeepSORT track ID for this session.
        first_seen_frame: Frame index of first appearance.
        last_seen_frame: Frame index of last appearance.
        total_visible_frames: Count of frames where student was detected.
        first_seen_time: Time in seconds of first appearance.
        last_seen_time: Time in seconds of last appearance.
    """

    __slots__ = [
        "global_id", "local_track_id",
        "first_seen_frame", "last_seen_frame", "total_visible_frames",
        "first_seen_time", "last_seen_time",
    ]

    def __init__(self, global_id: str, local_track_id: int,
                 frame_idx: int, fps: float):
        self.global_id = global_id
        self.local_track_id = local_track_id
        self.first_seen_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.total_visible_frames = 1
        self.first_seen_time = frame_idx / fps if fps > 0 else 0.0
        self.last_seen_time = self.first_seen_time

    def update(self, frame_idx: int, fps: float):
        """Update record with new frame observation."""
        self.last_seen_frame = frame_idx
        self.total_visible_frames += 1
        self.last_seen_time = frame_idx / fps if fps > 0 else 0.0

    @property
    def presence_seconds(self) -> float:
        """Total time span from first to last appearance."""
        return self.last_seen_time - self.first_seen_time


class AttendanceManager:
    """
    Manages attendance tracking for all students in a session.

    Determines attendance by checking:
        1. presence_seconds >= presence_threshold_seconds
        2. total_visible_frames >= min_track_frames

    Attributes:
        records: Dict mapping global_id → StudentRecord.
        presence_threshold: Minimum presence time in seconds.
        min_track_frames: Minimum number of visible frames.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize AttendanceManager.

        Args:
            config: Full pipeline config dict. Uses 'attendance' section.
        """
        config = config or {}
        att_cfg = config.get("attendance", {})

        self.presence_threshold = att_cfg.get("presence_threshold_seconds", 30)
        self.min_track_frames = att_cfg.get("min_track_frames", 20)
        self.records: dict[str, StudentRecord] = {}

        logger.info(
            f"AttendanceManager initialized: "
            f"presence_threshold={self.presence_threshold}s, "
            f"min_track_frames={self.min_track_frames}"
        )

    def update(self, global_id: str, local_track_id: int,
               frame_idx: int, fps: float):
        """
        Update attendance record for a student.

        Args:
            global_id: Persistent student ID.
            local_track_id: DeepSORT track ID.
            frame_idx: Current frame index.
            fps: Video frames per second.
        """
        if global_id in self.records:
            self.records[global_id].update(frame_idx, fps)
        else:
            self.records[global_id] = StudentRecord(
                global_id=global_id,
                local_track_id=local_track_id,
                frame_idx=frame_idx,
                fps=fps,
            )

    def is_present(self, global_id: str) -> bool:
        """
        Check if a student meets attendance criteria.

        Args:
            global_id: Student global ID.

        Returns:
            True if student meets both presence time and frame thresholds.
        """
        if global_id not in self.records:
            return False

        record = self.records[global_id]
        return (
            record.presence_seconds >= self.presence_threshold
            and record.total_visible_frames >= self.min_track_frames
        )

    def get_attendance_report(self) -> list[dict]:
        """
        Generate attendance data for all tracked students.

        Returns:
            List of dicts, each containing:
                global_id, local_track_id, total_frames,
                presence_time_seconds, is_present
        """
        report = []
        for gid, record in self.records.items():
            report.append({
                "global_id": gid,
                "local_track_id": record.local_track_id,
                "total_frames": record.total_visible_frames,
                "presence_time_seconds": round(record.presence_seconds, 2),
                "is_present": self.is_present(gid),
                "first_seen_frame": record.first_seen_frame,
                "last_seen_frame": record.last_seen_frame,
            })
        return report

    def get_summary(self) -> dict:
        """
        Get a quick attendance summary.

        Returns:
            Dict with total_students, present_count, absent_count.
        """
        total = len(self.records)
        present = sum(1 for gid in self.records if self.is_present(gid))
        return {
            "total_students": total,
            "present_count": present,
            "absent_count": total - present,
        }

    def reset(self):
        """Reset all records for a new session."""
        self.records.clear()
        logger.info("Attendance records cleared.")
