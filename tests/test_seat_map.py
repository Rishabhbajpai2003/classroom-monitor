import unittest

import cv2
import numpy as np

from app.models.seat_map import CameraMotionCompensator, RowGuide, SeatCalibration, build_seat_map


class SeatMapTests(unittest.TestCase):
    def test_build_seat_map_ranks_rows_from_front(self):
        calibration = SeatCalibration(
            reference_frame_index=0,
            frame_width=640,
            frame_height=480,
            front_edge=[[0, 20], [639, 20]],
            rows=[
                RowGuide(row_id="R01", left=[100, 120], right=[300, 120], seat_count=3),
                RowGuide(row_id="R02", left=[110, 220], right=[310, 220], seat_count=3),
            ],
        )
        seat_map = build_seat_map(calibration)
        self.assertEqual(len(seat_map), 6)
        self.assertEqual(seat_map[0].seat_id, "R01-S01")
        self.assertEqual(seat_map[0].row_rank, 1)
        self.assertEqual(seat_map[-1].row_rank, 2)

    def test_motion_compensator_projects_seats_on_identical_frame(self):
        rng = np.random.default_rng(7)
        frame = (rng.integers(0, 255, size=(320, 480, 3))).astype(np.uint8)
        cv2.putText(frame, "CLASSROOM", (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.rectangle(frame, (40, 40), (160, 120), (0, 255, 0), 2)
        calibration = SeatCalibration(
            reference_frame_index=0,
            frame_width=480,
            frame_height=320,
            front_edge=[[0, 20], [479, 20]],
            rows=[RowGuide(row_id="R01", left=[100, 160], right=[240, 160], seat_count=2)],
        )
        seat_map = build_seat_map(calibration)
        compensator = CameraMotionCompensator(frame, calibration, seat_map, min_matches=8, min_inliers=6)
        projection = compensator.project(frame.copy(), frame_idx=0)
        self.assertTrue(projection.stable)
        self.assertEqual(projection.seat_visibility["R01-S01"], "visible")
        self.assertIsNotNone(projection.seat_points["R01-S02"])


if __name__ == "__main__":
    unittest.main()
