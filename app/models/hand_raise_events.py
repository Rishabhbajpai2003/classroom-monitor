"""
Hand raise interval tracking.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HandRaiseEvent:
    student_id: str
    track_id: int
    seat_id: str
    start_frame: int
    end_frame: int
    duration_seconds: float
    peak_confidence: float


class HandRaiseEventTracker:
    def __init__(
        self,
        fps: float,
        min_hold_seconds: float = 0.8,
        release_seconds: float = 0.4,
        min_peak_confidence: float = 0.55,
    ):
        self.fps = max(1e-6, float(fps))
        self.min_hold_frames = max(1, int(round(min_hold_seconds * self.fps)))
        self.release_frames = max(1, int(round(release_seconds * self.fps)))
        self.min_peak_confidence = float(min_peak_confidence)
        self.state: dict[str, dict] = {}
        self.events: list[HandRaiseEvent] = []

    def update(self, student_id: str, track_id: int, frame_idx: int, raised: bool, confidence: float, seat_id: str = ""):
        state = self.state.setdefault(
            student_id,
            {
                "candidate_start": None,
                "active_start": None,
                "release_start": None,
                "peak_confidence": 0.0,
                "seat_id": seat_id or "",
                "track_id": track_id,
            },
        )
        state["track_id"] = track_id
        if seat_id:
            state["seat_id"] = seat_id

        if raised:
            state["release_start"] = None
            state["peak_confidence"] = max(float(confidence), float(state["peak_confidence"]))
            if state["active_start"] is not None:
                return
            if state["candidate_start"] is None:
                state["candidate_start"] = frame_idx
            elif frame_idx - state["candidate_start"] + 1 >= self.min_hold_frames:
                state["active_start"] = state["candidate_start"]
        else:
            state["candidate_start"] = None
            if state["active_start"] is None:
                return
            if state["release_start"] is None:
                state["release_start"] = frame_idx
            elif frame_idx - state["release_start"] + 1 >= self.release_frames:
                if float(state["peak_confidence"]) >= self.min_peak_confidence:
                    self.events.append(
                        HandRaiseEvent(
                            student_id=student_id,
                            track_id=int(state["track_id"]),
                            seat_id=str(state["seat_id"] or ""),
                            start_frame=int(state["active_start"]),
                            end_frame=max(int(state["active_start"]), frame_idx - self.release_frames),
                            duration_seconds=max(0.0, (frame_idx - self.release_frames - int(state["active_start"]) + 1) / self.fps),
                            peak_confidence=float(state["peak_confidence"]),
                        )
                    )
                state["active_start"] = None
                state["release_start"] = None
                state["peak_confidence"] = 0.0

    def finalize(self, total_frames: int):
        for student_id, state in self.state.items():
            if state.get("active_start") is None:
                continue
            if float(state["peak_confidence"]) >= self.min_peak_confidence:
                self.events.append(
                    HandRaiseEvent(
                        student_id=student_id,
                        track_id=int(state["track_id"]),
                        seat_id=str(state["seat_id"] or ""),
                        start_frame=int(state["active_start"]),
                        end_frame=max(int(state["active_start"]), int(total_frames)),
                        duration_seconds=max(0.0, (int(total_frames) - int(state["active_start"]) + 1) / self.fps),
                        peak_confidence=float(state["peak_confidence"]),
                    )
                )
            state["active_start"] = None

    def get_event_rows(self) -> list[dict]:
        return [
            {
                "student_id": event.student_id,
                "track_id": event.track_id,
                "event_type": "hand_raise",
                "seat_id": event.seat_id,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "duration_seconds": round(event.duration_seconds, 3),
                "peak_confidence": round(event.peak_confidence, 4),
            }
            for event in self.events
        ]

    def get_student_summary(self) -> dict[str, dict]:
        summary = {}
        for event in self.events:
            entry = summary.setdefault(
                event.student_id,
                {"hand_raise_count": 0, "hand_raise_seconds": 0.0},
            )
            entry["hand_raise_count"] += 1
            entry["hand_raise_seconds"] += event.duration_seconds
        for entry in summary.values():
            entry["hand_raise_seconds"] = round(entry["hand_raise_seconds"], 3)
        return summary
