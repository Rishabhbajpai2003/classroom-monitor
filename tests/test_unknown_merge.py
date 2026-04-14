import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from detectors.face_detector.run import FaceIdentityDB, FaceTracker, Track
from scripts.merge_unknown_assignments import main as merge_unknown_main


class UnknownMergeTests(unittest.TestCase):
    def test_merge_unknown_assignments_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            identity_db_path = root / "identity_db.json"
            unknown_json_path = root / "unknown_clusters.json"
            assignments_path = root / "unknown_assignments.csv"

            track = Track(
                track_id=1,
                bbox=np.zeros(4, dtype=np.float32),
                last_frame_idx=0,
                first_frame_idx=0,
                hits=2,
                metadata={"name": "Alice", "student_key": "Alice-001", "roll_number": "001"},
            )
            track.update_embedding_bank(
                np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                sample_quality=0.80,
                sample_metadata={"source_kind": "seed", "source_token": "seed:1", "embedder_used": "arcface"},
                max_bank=48,
            )
            tracker = FaceTracker(min_confirm_hits=2)
            tracker.archived_tracks = {1: track}
            tracker.next_track_id = 2
            FaceIdentityDB(str(identity_db_path)).save(tracker)

            with unknown_json_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {
                            "cluster_id": "UNK_0001",
                            "embeddings": [[0.99, 0.01, 0.0]],
                            "embedding_qualities": [0.77],
                            "embedding_metadata": [{"embedder_used": "adaface", "quality": 0.77}],
                        }
                    ],
                    handle,
                    indent=2,
                )

            with assignments_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["cluster_id", "assign_to_student_key", "assign_to_global_id", "status", "notes"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "cluster_id": "UNK_0001",
                        "assign_to_student_key": "Alice-001",
                        "assign_to_global_id": "",
                        "status": "approved",
                        "notes": "",
                    }
                )

            argv = [
                "merge_unknown_assignments.py",
                "--identity-db",
                str(identity_db_path),
                "--unknown-json",
                str(unknown_json_path),
                "--assignments",
                str(assignments_path),
            ]
            with patch("sys.argv", argv):
                merge_unknown_main()
            with patch("sys.argv", argv):
                merge_unknown_main()

            _next_track_id, loaded = FaceIdentityDB(str(identity_db_path)).load()
            merged_track = loaded[1]
            assigned = [
                item
                for item in merged_track.embedding_metadata
                if str(item.get("source_kind", "") or "") == "assigned_unknown"
            ]
            self.assertEqual(len(assigned), 1)


if __name__ == "__main__":
    unittest.main()
