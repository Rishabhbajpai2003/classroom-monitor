import unittest

from app.models.hand_raise_events import HandRaiseEventTracker


class HandRaiseEventTests(unittest.TestCase):
    def test_tracker_emits_interval_and_summary(self):
        tracker = HandRaiseEventTracker(fps=2.0, min_hold_seconds=0.5, release_seconds=0.5)
        tracker.update("STU_001", 1, 0, True, 0.6, "R01-S01")
        tracker.update("STU_001", 1, 1, True, 0.8, "R01-S01")
        tracker.update("STU_001", 1, 2, False, 0.0, "R01-S01")
        tracker.update("STU_001", 1, 3, False, 0.0, "R01-S01")

        rows = tracker.get_event_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seat_id"], "R01-S01")
        self.assertGreater(rows[0]["peak_confidence"], 0.7)

        summary = tracker.get_student_summary()
        self.assertEqual(summary["STU_001"]["hand_raise_count"], 1)
        self.assertGreater(summary["STU_001"]["hand_raise_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
