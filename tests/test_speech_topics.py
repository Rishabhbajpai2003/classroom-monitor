import tempfile
import unittest
from pathlib import Path

from app.models.speech_topics import SpeechTopicClassifier


class SpeechTopicTests(unittest.TestCase):
    def test_classifier_marks_class_related_and_off_topic(self):
        classifier = SpeechTopicClassifier(Path("configs") / "topic_profiles.yaml", "math")

        related = classifier.classify(
            "We should solve this equation and check the derivative before the exam",
            mean_speech_prob=0.80,
        )
        self.assertEqual(related["topic_label"], "class_related")

        off_topic = classifier.classify(
            "Did you watch the movie and the cricket match after lunch",
            mean_speech_prob=0.81,
        )
        self.assertEqual(off_topic["topic_label"], "off_topic")

    def test_classifier_uses_quality_gate(self):
        classifier = SpeechTopicClassifier(Path("configs") / "topic_profiles.yaml", "default")
        result = classifier.classify("exam", mean_speech_prob=0.40)
        self.assertEqual(result["topic_label"], "unknown")


if __name__ == "__main__":
    unittest.main()
