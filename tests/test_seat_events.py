import unittest

from app.models.seat_events import SeatEventEngine
from app.models.seat_map import SeatDefinition, SeatProjectionResult


def _projection(visibility: str = "visible", point=(100.0, 100.0)) -> SeatProjectionResult:
    return SeatProjectionResult(
        frame_idx=0,
        stable=True,
        homography=None,
        seat_points={"R01-S01": list(point) if point is not None else None},
        seat_visibility={"R01-S01": visibility},
        exit_polygon=[],
        inlier_count=20,
    )


def _projection_two() -> SeatProjectionResult:
    return SeatProjectionResult(
        frame_idx=0,
        stable=True,
        homography=None,
        seat_points={"R01-S01": [100.0, 100.0], "R02-S01": [200.0, 200.0]},
        seat_visibility={"R01-S01": "visible", "R02-S01": "visible"},
        exit_polygon=[],
        inlier_count=20,
    )


class SeatEventTests(unittest.TestCase):
    def setUp(self):
        self.seat_map = [
            SeatDefinition(
                seat_id="R01-S01",
                row_id="R01",
                seat_index=1,
                row_rank=1,
                seat_rank=1,
                ref_point=[100.0, 100.0],
                front_distance=80.0,
                match_radius=60.0,
            ),
            SeatDefinition(
                seat_id="R02-S01",
                row_id="R02",
                seat_index=1,
                row_rank=2,
                seat_rank=2,
                ref_point=[200.0, 200.0],
                front_distance=180.0,
                match_radius=60.0,
            ),
        ]

    def test_off_frame_seat_becomes_unobservable_not_out_of_class(self):
        engine = SeatEventEngine(
            self.seat_map[:1],
            fps=1.0,
            initial_confirm_seconds=1.0,
            shift_confirm_seconds=2.0,
            out_of_class_seconds=2.0,
            exit_zone_seconds=2.0,
            late_arrival_minutes=5.0,
            early_exit_minutes=5.0,
        )
        student = {"global_id": "STU_001", "track_id": 1, "center": [100.0, 100.0]}
        engine.update(0, _projection("visible"), [student])
        engine.update(1, _projection("off_frame", point=None), [])
        engine.update(2, _projection("off_frame", point=None), [])
        engine.finalize(2)
        event_types = [row["event_type"] for row in engine.get_event_rows()]
        self.assertNotIn("out_of_class", event_types)
        self.assertEqual(engine.get_current_state("STU_001"), "unobservable")

    def test_seat_shift_is_recorded_without_creating_new_id(self):
        engine = SeatEventEngine(
            self.seat_map,
            fps=1.0,
            initial_confirm_seconds=1.0,
            shift_confirm_seconds=2.0,
            out_of_class_seconds=2.0,
            exit_zone_seconds=2.0,
            late_arrival_minutes=0.05,
            early_exit_minutes=0.05,
        )
        student = {"global_id": "STU_002", "track_id": 2, "center": [100.0, 100.0]}
        moved_student = {"global_id": "STU_002", "track_id": 2, "center": [200.0, 200.0]}
        projection_two = _projection_two()
        engine.update(0, projection_two, [student])
        engine.update(1, projection_two, [moved_student])
        engine.update(2, projection_two, [moved_student])
        engine.finalize(2)

        rows = engine.get_event_rows()
        event_types = [row["event_type"] for row in rows]
        self.assertIn("seat_shift", event_types)
        self.assertEqual(engine.get_student_summary()["STU_002"]["seat_id"], "R02-S01")

    def test_late_arrival_and_early_exit_are_recorded(self):
        late_engine = SeatEventEngine(
            self.seat_map[:1],
            fps=1.0,
            initial_confirm_seconds=1.0,
            shift_confirm_seconds=2.0,
            out_of_class_seconds=2.0,
            exit_zone_seconds=2.0,
            late_arrival_minutes=0.05,
            early_exit_minutes=0.05,
        )
        student = {"global_id": "STU_003", "track_id": 3, "center": [100.0, 100.0]}
        late_engine.update(5, _projection("visible"), [student])
        late_engine.finalize(6)
        late_events = [row["event_type"] for row in late_engine.get_event_rows()]
        self.assertIn("late_arrival", late_events)

        early_engine = SeatEventEngine(
            self.seat_map[:1],
            fps=1.0,
            initial_confirm_seconds=1.0,
            shift_confirm_seconds=2.0,
            out_of_class_seconds=2.0,
            exit_zone_seconds=2.0,
            late_arrival_minutes=5.0,
            early_exit_minutes=0.05,
        )
        early_engine.update(0, _projection("visible"), [student])
        early_engine.update(61, _projection("visible"), [])
        early_engine.update(62, _projection("visible"), [])
        early_engine.finalize(70)
        event_types = [row["event_type"] for row in early_engine.get_event_rows()]
        self.assertIn("out_of_class", event_types)
        self.assertIn("early_exit", event_types)

    def test_current_seat_is_sticky_under_short_ambiguous_motion(self):
        engine = SeatEventEngine(
            self.seat_map,
            fps=1.0,
            initial_confirm_seconds=1.0,
            shift_confirm_seconds=2.0,
            seat_stick_seconds=3.0,
            out_of_class_seconds=2.0,
            exit_zone_seconds=2.0,
            late_arrival_minutes=5.0,
            early_exit_minutes=5.0,
        )
        projection_two = _projection_two()
        student = {"global_id": "STU_004", "track_id": 4, "center": [100.0, 100.0]}
        jittered = {"global_id": "STU_004", "track_id": 4, "center": [150.0, 150.0]}
        engine.update(0, projection_two, [student])
        engine.update(1, projection_two, [jittered])
        self.assertEqual(engine.get_current_seat("STU_004"), "R01-S01")
        event_types = [row["event_type"] for row in engine.get_event_rows()]
        self.assertNotIn("seat_shift", event_types)


if __name__ == "__main__":
    unittest.main()
