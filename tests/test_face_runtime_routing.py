import unittest

import numpy as np

from app.models.face_attributes import FaceAttributePrediction, LocalFaceAttributeClassifier
from app.models.frame_change import FrameChangeGate
from detectors.face_detector.run import runtime_family_tags


class FaceRuntimeRoutingTests(unittest.TestCase):
    def test_runtime_family_tags_prioritise_combo_pose_accessory_and_quality(self):
        quality_profile = {
            "lighting_tag": "shadow",
            "quality_tags": ["blur"],
            "scale_tag": "tiny_face",
        }
        prediction = FaceAttributePrediction(
            pose_bucket="left_profile",
            pose_confidence=0.88,
            accessory_scores={"cap": 0.81, "mask": 0.12, "sunglasses": 0.05, "scarf": 0.0},
        )
        families = runtime_family_tags(quality_profile, prediction, accessory_threshold=0.55)
        self.assertEqual(families[0], "lighting/shadow")
        self.assertIn("quality/blur", families)
        self.assertIn("scale/tiny_face", families)
        self.assertIn("pose/left_profile", families)
        self.assertIn("accessory/cap", families)
        self.assertIn("combo/cap_left_profile", families)
        self.assertEqual(families[-1], "base")

    def test_frame_change_gate_flags_big_visual_changes(self):
        gate = FrameChangeGate(diff_threshold=0.08, unstable_inlier_ratio=0.25, downsample_width=96)
        frame_a = np.zeros((120, 160, 3), dtype=np.uint8)
        frame_b = np.zeros((120, 160, 3), dtype=np.uint8)
        frame_b[:, :] = 255

        first = gate.update(frame_a)
        second = gate.update(frame_a.copy())
        third = gate.update(frame_b)

        self.assertTrue(first.changed)
        self.assertFalse(second.changed)
        self.assertTrue(third.changed)
        self.assertGreater(third.score, second.score)

    def test_heuristic_pose_classifier_detects_profile_from_landmarks(self):
        classifier = LocalFaceAttributeClassifier("", "")
        crop = np.full((112, 112, 3), 180, dtype=np.uint8)
        bbox = np.asarray([0.0, 0.0, 112.0, 112.0], dtype=np.float32)
        landmarks = np.asarray(
            [
                [26.0, 42.0],
                [76.0, 44.0],
                [63.0, 58.0],
                [34.0, 82.0],
                [74.0, 84.0],
            ],
            dtype=np.float32,
        )
        prediction = classifier.predict_with_context(crop, landmarks=landmarks, bbox=bbox)
        self.assertEqual(prediction.pose_bucket, "left_profile")
        self.assertGreater(prediction.pose_confidence, 0.55)

    def test_heuristic_accessory_classifier_detects_dark_cap_band(self):
        classifier = LocalFaceAttributeClassifier("", "")
        crop = np.full((112, 112, 3), 190, dtype=np.uint8)
        crop[:24, :, :] = 18
        prediction = classifier.predict_with_context(crop, quality_profile={"shadow_severity": 0.15})
        self.assertGreater(prediction.accessory_scores["cap"], 0.58)


if __name__ == "__main__":
    unittest.main()
