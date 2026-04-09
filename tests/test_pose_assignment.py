import unittest

import numpy as np

from app.models.pose_analyzer import assign_pose_detections


class PoseAssignmentTests(unittest.TestCase):
    def test_pose_assignment_is_one_to_one(self):
        tracked_persons = [
            {"track_id": 1, "bbox": [100, 100, 220, 340]},
            {"track_id": 2, "bbox": [230, 105, 350, 345]},
        ]
        pose_bboxes = np.asarray(
            [
                [102, 102, 222, 338],
                [235, 108, 348, 344],
            ],
            dtype=np.float32,
        )

        matches = assign_pose_detections(tracked_persons, pose_bboxes)
        self.assertEqual(matches[1], 0)
        self.assertEqual(matches[2], 1)
        self.assertEqual(len(set(matches.values())), 2)


if __name__ == "__main__":
    unittest.main()

