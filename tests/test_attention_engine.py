import unittest

from detectors.attention_detector.attention_engine import AttentionEngine


class AttentionEngineTests(unittest.TestCase):
    def test_limited_mode_defaults_to_unknown_without_object_evidence(self):
        engine = AttentionEngine(
            {
                "attention": {
                    "min_state_hold": 1,
                    "unknown_state_hold": 1,
                }
            }
        )
        result = engine.update(
            "STU_001",
            {
                "size_mode": "limited",
                "detection_confidence": 0.55,
                "pose_confidence": 0.0,
                "head_forward": False,
                "hand_raised": False,
                "using_phone_pose": False,
                "using_phone_object": False,
            },
        )
        self.assertEqual(result["attention_state"], "unknown")

    def test_phone_object_pushes_state_to_distracted(self):
        engine = AttentionEngine(
            {
                "attention": {
                    "min_state_hold": 1,
                    "unknown_state_hold": 1,
                }
            }
        )
        result = engine.update(
            "STU_002",
            {
                "size_mode": "full",
                "detection_confidence": 0.78,
                "pose_confidence": 0.62,
                "head_forward": False,
                "hand_raised": False,
                "using_phone_pose": True,
                "using_phone_object": True,
                "phone_object_confidence": 0.88,
            },
        )
        self.assertEqual(result["attention_state"], "distracted")
        self.assertGreater(result["attention_confidence"], 0.3)


if __name__ == "__main__":
    unittest.main()

