import unittest

import numpy as np

from app.models.identity_enrichment import (
    classify_harvest_candidate,
    compact_embedding_bank,
)
from detectors.face_detector.run import Track


class IdentityEnrichmentTests(unittest.TestCase):
    def test_compaction_prefers_latest_duplicate_within_bucket(self):
        base = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        newer = np.asarray([0.999, 0.02, 0.0], dtype=np.float32)
        embeddings, qualities, metadata = compact_embedding_bank(
            [base, newer],
            [0.72, 0.78],
            [
                {"profile_bucket": "frontal", "size_bucket": "medium_face", "added_at": "2026-04-09T10:00:00+00:00", "frame_idx": 10},
                {"profile_bucket": "frontal", "size_bucket": "medium_face", "added_at": "2026-04-09T11:00:00+00:00", "frame_idx": 20},
            ],
            max_bank=48,
            duplicate_sim_thresh=0.92,
        )
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(qualities), 1)
        self.assertEqual(metadata[0]["frame_idx"], 20)

    def test_compaction_preserves_diverse_buckets(self):
        emb_a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.asarray([0.998, 0.03, 0.0], dtype=np.float32)
        embeddings, qualities, metadata = compact_embedding_bank(
            [emb_a, emb_b],
            [0.80, 0.79],
            [
                {"profile_bucket": "frontal", "size_bucket": "small_face", "added_at": "2026-04-09T10:00:00+00:00"},
                {"profile_bucket": "left_profile", "size_bucket": "large_face", "added_at": "2026-04-09T11:00:00+00:00"},
            ],
            max_bank=48,
            duplicate_sim_thresh=0.92,
        )
        self.assertEqual(len(embeddings), 2)
        self.assertCountEqual([item["profile_bucket"] for item in metadata], ["frontal", "left_profile"])

    def test_track_embedding_bank_stays_idempotent_for_duplicates(self):
        track = Track(
            track_id=1,
            bbox=np.zeros(4, dtype=np.float32),
            last_frame_idx=0,
            first_frame_idx=0,
        )
        emb = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        track.update_embedding_bank(
            emb,
            sample_quality=0.75,
            sample_metadata={"profile_bucket": "frontal", "size_bucket": "medium_face", "added_at": "2026-04-09T10:00:00+00:00", "frame_idx": 1},
            max_bank=48,
            duplicate_sim_thresh=0.92,
        )
        track.update_embedding_bank(
            emb,
            sample_quality=0.77,
            sample_metadata={"profile_bucket": "frontal", "size_bucket": "medium_face", "added_at": "2026-04-09T11:00:00+00:00", "frame_idx": 2},
            max_bank=48,
            duplicate_sim_thresh=0.92,
        )
        self.assertEqual(len(track.embeddings), 1)
        self.assertEqual(track.embedding_metadata[0]["frame_idx"], 2)

    def test_harvest_candidate_thresholds(self):
        self.assertEqual(
            classify_harvest_candidate(0.71, good_crops=4, stable_seconds=1.8, quality=0.6),
            "auto_add",
        )
        self.assertEqual(
            classify_harvest_candidate(0.63, good_crops=1, stable_seconds=0.4, quality=0.5),
            "review",
        )
        self.assertEqual(
            classify_harvest_candidate(0.42, good_crops=5, stable_seconds=2.0, quality=0.8),
            "reject",
        )


if __name__ == "__main__":
    unittest.main()
