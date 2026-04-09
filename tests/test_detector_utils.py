import unittest

from app.models.detectors import deduplicate_box_detections, generate_overlapping_tiles


class DetectorUtilsTests(unittest.TestCase):
    def test_generate_overlapping_tiles_returns_expected_count(self):
        tiles = generate_overlapping_tiles((1080, 1920, 3), tile_grid=2, overlap=0.20)
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles[0][0], 0)
        self.assertEqual(tiles[0][1], 0)

    def test_deduplicate_box_detections_prefers_stronger_detection(self):
        detections = [
            {
                "bbox": [100, 100, 200, 260],
                "confidence": 0.82,
                "class_name": "person",
                "area": 16000,
                "source": "global",
            },
            {
                "bbox": [104, 103, 203, 259],
                "confidence": 0.71,
                "class_name": "person",
                "area": 15400,
                "source": "tile",
            },
            {
                "bbox": [600, 120, 680, 260],
                "confidence": 0.64,
                "class_name": "person",
                "area": 11200,
                "source": "tile",
            },
        ]
        deduped = deduplicate_box_detections(detections)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["bbox"], [100, 100, 200, 260])


if __name__ == "__main__":
    unittest.main()

