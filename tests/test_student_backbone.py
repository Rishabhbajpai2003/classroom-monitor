import unittest
from types import SimpleNamespace

from app.models.student_backbone import (
    associate_face_tracks_to_body_detections,
    classify_size_mode,
)


class StudentBackboneTests(unittest.TestCase):
    def test_classify_size_mode_uses_thresholds(self):
        self.assertEqual(classify_size_mode([0, 0, 50, 200]), "full")
        self.assertEqual(classify_size_mode([0, 0, 50, 120]), "reduced")
        self.assertEqual(classify_size_mode([0, 0, 50, 70]), "limited")

    def test_face_tracks_associate_to_distinct_bodies(self):
        face_tracks = [
            SimpleNamespace(track_id=1, bbox=[100, 100, 150, 160]),
            SimpleNamespace(track_id=2, bbox=[320, 110, 368, 168]),
        ]
        body_detections = [
            {"bbox": [80, 90, 180, 340], "confidence": 0.82},
            {"bbox": [300, 92, 390, 352], "confidence": 0.80},
        ]
        association = associate_face_tracks_to_body_detections(face_tracks, body_detections)
        self.assertEqual(association[1]["bbox"], body_detections[0]["bbox"])
        self.assertEqual(association[2]["bbox"], body_detections[1]["bbox"])


if __name__ == "__main__":
    unittest.main()

