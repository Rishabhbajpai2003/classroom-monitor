"""
Seat occupancy, seating timeline, and attendance-event logic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.models.seat_map import SeatDefinition, SeatProjectionResult

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


@dataclass
class SeatingInterval:
    student_id: str
    track_id: int
    seat_id: Optional[str]
    seat_rank: Optional[int]
    row_rank: Optional[int]
    state: str
    start_frame: int
    end_frame: int


@dataclass
class AttendanceEvent:
    student_id: str
    track_id: int
    event_type: str
    start_frame: int
    end_frame: int
    duration_seconds: float
    seat_id: Optional[str] = None
    from_seat: Optional[str] = None
    to_seat: Optional[str] = None
    reason: str = ""
    confidence: float = 0.0


class SeatEventEngine:
    def __init__(
        self,
        seat_map: list[SeatDefinition],
        fps: float,
        initial_confirm_seconds: float = 3.0,
        shift_confirm_seconds: float = 10.0,
        seat_stick_seconds: float = 3.0,
        out_of_class_seconds: float = 20.0,
        exit_zone_seconds: float = 8.0,
        late_arrival_minutes: float = 5.0,
        early_exit_minutes: float = 5.0,
    ):
        self.seat_map = seat_map
        self.fps = max(1e-6, float(fps))
        self.initial_confirm_frames = max(1, int(round(initial_confirm_seconds * self.fps)))
        self.shift_confirm_frames = max(1, int(round(shift_confirm_seconds * self.fps)))
        self.seat_stick_frames = max(1, int(round(seat_stick_seconds * self.fps)))
        self.out_of_class_frames = max(1, int(round(out_of_class_seconds * self.fps)))
        self.exit_zone_frames = max(1, int(round(exit_zone_seconds * self.fps)))
        self.late_arrival_frames = max(1, int(round(late_arrival_minutes * 60.0 * self.fps)))
        self.early_exit_frames = max(1, int(round(early_exit_minutes * 60.0 * self.fps)))

        self.seats_by_id = {seat.seat_id: seat for seat in seat_map}
        self.current_seat: dict[str, Optional[str]] = {}
        self.current_state: dict[str, str] = {}
        self.current_interval_start: dict[str, int] = {}
        self.candidate_seat: dict[str, Optional[str]] = {}
        self.candidate_start: dict[str, int] = {}
        self.visible_empty_start: dict[str, int] = {}
        self.last_exit_seen: dict[str, int] = {}
        self.out_start: dict[str, int] = {}
        self.track_ids: dict[str, int] = {}
        self.first_confirmed_seated_frame: dict[str, int] = {}
        self.last_confirmed_seated_frame: dict[str, int] = {}
        self.weighted_rank_sum: defaultdict[str, float] = defaultdict(float)
        self.weighted_rank_frames: defaultdict[str, int] = defaultdict(int)
        self.last_assignment: dict[str, Optional[str]] = {}
        self._intervals: list[SeatingInterval] = []
        self._events: list[AttendanceEvent] = []
        self._processed_students: set[str] = set()

    @staticmethod
    def _point_in_polygon(point_xy, polygon: list[list[float]]) -> bool:
        if len(polygon) < 3:
            return False
        pt = tuple(float(v) for v in point_xy)
        contour = np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(contour, pt, False) >= 0

    def _seat_distance_score(self, seat: SeatDefinition, seat_point: list[float], student_center: np.ndarray) -> float:
        distance = float(np.linalg.norm(np.asarray(seat_point, dtype=np.float32) - student_center))
        return 1.0 - (distance / max(1.0, seat.match_radius))

    def _frame_assignments(
        self,
        projection: SeatProjectionResult,
        students: list[dict],
    ) -> dict[str, str]:
        visible_seats = [
            seat
            for seat in self.seat_map
            if projection.seat_visibility.get(seat.seat_id) == "visible" and projection.seat_points.get(seat.seat_id) is not None
        ]
        if not visible_seats or not students:
            return {}

        score_matrix = np.full((len(students), len(visible_seats)), -1e9, dtype=np.float32)
        for row_idx, student in enumerate(students):
            student_id = str(student["global_id"])
            center = np.asarray(student["center"], dtype=np.float32)
            current_seat = self.current_seat.get(student_id)
            candidate_seat = self.candidate_seat.get(student_id)
            last_assignment = self.last_assignment.get(student_id)
            current_state = self.current_state.get(student_id, "unassigned")
            current_visible = bool(
                current_seat
                and projection.seat_visibility.get(current_seat) == "visible"
                and projection.seat_points.get(current_seat) is not None
            )
            for col_idx, seat in enumerate(visible_seats):
                seat_point = projection.seat_points[seat.seat_id]
                score = self._seat_distance_score(seat, seat_point, center)
                if score < -0.35:
                    continue
                if current_seat and seat.seat_id == current_seat:
                    score += 0.22
                    if current_state in {"seated", "unobservable"}:
                        score += 0.05
                elif candidate_seat and seat.seat_id == candidate_seat:
                    score += 0.10
                elif last_assignment and seat.seat_id == last_assignment:
                    score += 0.04
                elif current_visible and current_seat and seat.seat_id != current_seat:
                    score -= 0.08
                score_matrix[row_idx, col_idx] = score

        matches: dict[str, str] = {}
        if linear_sum_assignment is not None:
            cost = np.where(score_matrix < -1e8, 1e6, -score_matrix)
            row_ind, col_ind = linear_sum_assignment(cost)
            for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
                score = float(score_matrix[row_idx, col_idx])
                if score < -0.35:
                    continue
                matches[students[row_idx]["global_id"]] = visible_seats[col_idx].seat_id
            return matches

        candidates: list[tuple[float, int, int]] = []
        for row_idx in range(score_matrix.shape[0]):
            for col_idx in range(score_matrix.shape[1]):
                candidates.append((float(score_matrix[row_idx, col_idx]), row_idx, col_idx))
        candidates.sort(reverse=True)
        used_rows = set()
        used_cols = set()
        for score, row_idx, col_idx in candidates:
            if score < -0.35:
                break
            if row_idx in used_rows or col_idx in used_cols:
                continue
            matches[students[row_idx]["global_id"]] = visible_seats[col_idx].seat_id
            used_rows.add(row_idx)
            used_cols.add(col_idx)
        return matches

    def _close_interval(self, student_id: str, frame_idx: int):
        if student_id not in self.current_interval_start:
            return
        start_frame = self.current_interval_start[student_id]
        end_frame = max(start_frame, frame_idx - 1)
        state = self.current_state.get(student_id, "unassigned")
        seat_id = self.current_seat.get(student_id)
        seat_rank = self.seats_by_id[seat_id].seat_rank if seat_id in self.seats_by_id else None
        row_rank = self.seats_by_id[seat_id].row_rank if seat_id in self.seats_by_id else None
        self._intervals.append(
            SeatingInterval(
                student_id=student_id,
                track_id=self.track_ids.get(student_id, -1),
                seat_id=seat_id,
                seat_rank=seat_rank,
                row_rank=row_rank,
                state=state,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
        self.current_interval_start[student_id] = frame_idx

    def _set_state(self, student_id: str, state: str, frame_idx: int):
        prev_state = self.current_state.get(student_id)
        if prev_state is None:
            self.current_state[student_id] = state
            self.current_interval_start[student_id] = frame_idx
            return
        if prev_state == state:
            return
        self._close_interval(student_id, frame_idx)
        self.current_state[student_id] = state

    def _append_event(
        self,
        student_id: str,
        event_type: str,
        start_frame: int,
        end_frame: int,
        *,
        seat_id: Optional[str] = None,
        from_seat: Optional[str] = None,
        to_seat: Optional[str] = None,
        reason: str = "",
        confidence: float = 0.0,
    ):
        self._events.append(
            AttendanceEvent(
                student_id=student_id,
                track_id=self.track_ids.get(student_id, -1),
                event_type=event_type,
                start_frame=start_frame,
                end_frame=end_frame,
                duration_seconds=max(0.0, (end_frame - start_frame + 1) / self.fps),
                seat_id=seat_id,
                from_seat=from_seat,
                to_seat=to_seat,
                reason=reason,
                confidence=confidence,
            )
        )

    def update(self, frame_idx: int, projection: SeatProjectionResult, students: list[dict]) -> dict[str, Optional[str]]:
        frame_assignments = self._frame_assignments(projection, students)
        student_by_id = {student["global_id"]: student for student in students}

        for student in students:
            student_id = student["global_id"]
            self._processed_students.add(student_id)
            self.track_ids[student_id] = int(student.get("track_id", -1))
            assigned_seat = frame_assignments.get(student_id)
            current_seat = self.current_seat.get(student_id)
            current_state = self.current_state.get(student_id, "unassigned")
            self.last_assignment[student_id] = assigned_seat

            if projection.exit_polygon and self._point_in_polygon(student["center"], projection.exit_polygon):
                self.last_exit_seen[student_id] = frame_idx

            if assigned_seat is not None:
                self.visible_empty_start.pop(student_id, None)

                if current_seat is None:
                    if self.candidate_seat.get(student_id) != assigned_seat:
                        self.candidate_seat[student_id] = assigned_seat
                        self.candidate_start[student_id] = frame_idx
                    if frame_idx - self.candidate_start.get(student_id, frame_idx) + 1 >= self.initial_confirm_frames:
                        self.current_seat[student_id] = assigned_seat
                        self.first_confirmed_seated_frame.setdefault(student_id, frame_idx)
                        self.last_confirmed_seated_frame[student_id] = frame_idx
                        self._set_state(student_id, "seated", frame_idx)
                        self.candidate_seat.pop(student_id, None)
                        self.candidate_start.pop(student_id, None)
                    continue

                if assigned_seat == current_seat:
                    self.last_confirmed_seated_frame[student_id] = frame_idx
                    if current_state == "out_of_class":
                        start_frame = self.out_start.pop(student_id, frame_idx)
                        self._append_event(
                            student_id,
                            "return_to_class",
                            start_frame,
                            frame_idx,
                            seat_id=current_seat,
                            reason="seat_reoccupied",
                            confidence=0.9,
                        )
                    self._set_state(student_id, "seated", frame_idx)
                    seat = self.seats_by_id.get(current_seat)
                    if seat is not None:
                        self.weighted_rank_sum[student_id] += float(seat.seat_rank)
                        self.weighted_rank_frames[student_id] += 1
                    continue

                if self.candidate_seat.get(student_id) != assigned_seat:
                    self.candidate_seat[student_id] = assigned_seat
                    self.candidate_start[student_id] = frame_idx
                elif frame_idx - self.candidate_start.get(student_id, frame_idx) + 1 >= self.shift_confirm_frames:
                    previous_seat = current_seat
                    self.current_seat[student_id] = assigned_seat
                    self.last_confirmed_seated_frame[student_id] = frame_idx
                    self._set_state(student_id, "seated", frame_idx)
                    self._append_event(
                        student_id,
                        "seat_shift",
                        self.candidate_start.get(student_id, frame_idx),
                        frame_idx,
                        seat_id=assigned_seat,
                        from_seat=previous_seat,
                        to_seat=assigned_seat,
                        reason="stable_new_seat",
                        confidence=0.9,
                    )
                    self.candidate_seat.pop(student_id, None)
                    self.candidate_start.pop(student_id, None)
                continue

            # No seat assignment this frame.
            self._handle_missing_assignment(student_id, current_seat, frame_idx, projection)

        for student_id, current_seat in list(self.current_seat.items()):
            if student_id in student_by_id:
                continue
            self._handle_missing_assignment(student_id, current_seat, frame_idx, projection)

        return {student_id: self.current_seat.get(student_id) for student_id in student_by_id}

    def _handle_missing_assignment(
        self,
        student_id: str,
        current_seat: Optional[str],
        frame_idx: int,
        projection: SeatProjectionResult,
    ) -> None:
        if current_seat is None:
            self.candidate_seat.pop(student_id, None)
            self.candidate_start.pop(student_id, None)
            self._set_state(student_id, "unassigned", frame_idx)
            return

        seat_visibility = projection.seat_visibility.get(current_seat, "unstable_view")
        if seat_visibility != "visible" or not projection.stable:
            self.visible_empty_start.pop(student_id, None)
            self._set_state(student_id, "unobservable", frame_idx)
            return

        candidate_start = self.candidate_start.get(student_id)
        if candidate_start is not None and (frame_idx - candidate_start + 1) < self.seat_stick_frames:
            self.visible_empty_start.pop(student_id, None)
            self._set_state(student_id, "seated", frame_idx)
            return

        self._set_state(student_id, "seated", frame_idx)
        if student_id not in self.visible_empty_start:
            self.visible_empty_start[student_id] = frame_idx

        empty_duration = frame_idx - self.visible_empty_start[student_id] + 1
        exit_seen_frame = self.last_exit_seen.get(student_id)
        exit_triggered = exit_seen_frame is not None and frame_idx - exit_seen_frame + 1 >= self.exit_zone_frames
        if empty_duration >= self.out_of_class_frames or exit_triggered:
            if self.current_state.get(student_id) != "out_of_class":
                self._set_state(student_id, "out_of_class", self.visible_empty_start[student_id])
                self.out_start[student_id] = self.visible_empty_start[student_id]
                self._append_event(
                    student_id,
                    "out_of_class",
                    self.visible_empty_start[student_id],
                    frame_idx,
                    seat_id=current_seat,
                    reason="exit_zone" if exit_triggered else "seat_empty_revisit",
                    confidence=0.88 if exit_triggered else 0.80,
                )

    def get_current_seat(self, student_id: str) -> Optional[str]:
        return self.current_seat.get(student_id)

    def get_current_state(self, student_id: str) -> str:
        return self.current_state.get(student_id, "unassigned")

    def finalize(self, total_frames: int):
        for student_id in list(self._processed_students):
            if student_id in self.current_interval_start:
                self._close_interval(student_id, total_frames + 1)

            first_frame = self.first_confirmed_seated_frame.get(student_id)
            if first_frame is not None and first_frame > self.late_arrival_frames:
                seat_id = self.current_seat.get(student_id)
                self._append_event(
                    student_id,
                    "late_arrival",
                    first_frame,
                    first_frame,
                    seat_id=seat_id,
                    reason="after_grace_period",
                    confidence=1.0,
                )

            out_start = self.out_start.get(student_id)
            last_seated = self.last_confirmed_seated_frame.get(student_id)
            if out_start is not None and out_start < max(0, total_frames - self.early_exit_frames):
                if last_seated is None or last_seated < max(0, total_frames - int(round(60.0 * self.fps))):
                    seat_id = self.current_seat.get(student_id)
                    self._append_event(
                        student_id,
                        "early_exit",
                        out_start,
                        total_frames,
                        seat_id=seat_id,
                        reason="final_out_interval",
                        confidence=0.92,
                    )

    def get_timeline_rows(self) -> list[dict]:
        return [
            {
                "student_id": interval.student_id,
                "track_id": interval.track_id,
                "seat_id": interval.seat_id or "",
                "seat_rank": interval.seat_rank if interval.seat_rank is not None else "",
                "row_rank": interval.row_rank if interval.row_rank is not None else "",
                "state": interval.state,
                "start_frame": interval.start_frame,
                "end_frame": interval.end_frame,
                "duration_seconds": round((interval.end_frame - interval.start_frame + 1) / self.fps, 3),
            }
            for interval in self._intervals
        ]

    def get_event_rows(self) -> list[dict]:
        return [
            {
                "student_id": event.student_id,
                "track_id": event.track_id,
                "event_type": event.event_type,
                "seat_id": event.seat_id or "",
                "from_seat": event.from_seat or "",
                "to_seat": event.to_seat or "",
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "duration_seconds": round(event.duration_seconds, 3),
                "reason": event.reason,
                "confidence": round(event.confidence, 4),
            }
            for event in self._events
        ]

    def get_student_summary(self) -> dict[str, dict]:
        summary = {}
        for student_id in self._processed_students:
            seat_id = self.current_seat.get(student_id)
            weighted_frames = self.weighted_rank_frames.get(student_id, 0)
            weighted_avg = (
                self.weighted_rank_sum.get(student_id, 0.0) / weighted_frames
                if weighted_frames > 0
                else None
            )
            out_seconds = sum(
                event.duration_seconds
                for event in self._events
                if event.student_id == student_id and event.event_type == "out_of_class"
            )
            summary[student_id] = {
                "seat_id": seat_id,
                "seat_rank": self.seats_by_id[seat_id].seat_rank if seat_id in self.seats_by_id else None,
                "row_rank": self.seats_by_id[seat_id].row_rank if seat_id in self.seats_by_id else None,
                "weighted_avg_seat_rank": round(weighted_avg, 4) if weighted_avg is not None else None,
                "out_of_class_seconds": round(out_seconds, 3),
                "late_arrival": any(
                    event.student_id == student_id and event.event_type == "late_arrival"
                    for event in self._events
                ),
                "early_exit": any(
                    event.student_id == student_id and event.event_type == "early_exit"
                    for event in self._events
                ),
                "current_state": self.current_state.get(student_id, "unassigned"),
            }
        return summary
