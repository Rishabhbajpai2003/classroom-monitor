import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.generate_student_face_augmentations import (
    DEFAULT_VARIANTS,
    LOCAL_OUTPUT_DIR,
    generate_for_image,
    purge_legacy_outputs,
    should_augment_source,
)


class FaceAugmentationTests(unittest.TestCase):
    def test_photo_only_policy_and_legacy_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            student_dir = Path(tmpdir) / "Student-001"
            student_dir.mkdir(parents=True, exist_ok=True)
            photo_path = student_dir / "Photo-Student-001.jpg"
            id_path = student_dir / "ID-Student-001.jpg"
            Image.new("RGB", (180, 240), (180, 160, 140)).save(photo_path)
            Image.new("RGB", (180, 240), (220, 220, 220)).save(id_path)

            legacy_dir = student_dir / "augmentations"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            stale_id_aug = legacy_dir / "ID-Student-001__aug-mask.jpg"
            Image.new("RGB", (180, 240), (120, 120, 120)).save(stale_id_aug)

            self.assertTrue(should_augment_source(photo_path, photo_only=True))
            self.assertFalse(should_augment_source(id_path, photo_only=True))

            removed = purge_legacy_outputs(student_dir, list(DEFAULT_VARIANTS))
            self.assertEqual(removed, 1)
            self.assertFalse(stale_id_aug.exists())

    def test_generate_for_image_produces_new_deterministic_families(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            student_dir = root / "Student-001"
            student_dir.mkdir(parents=True, exist_ok=True)
            photo_path = student_dir / "Photo-Student-001.jpg"
            Image.new("RGB", (220, 280), (176, 152, 136)).save(photo_path)
            manifest_path = root / "augmentation_manifest.csv"

            rows = generate_for_image(
                image_path=photo_path,
                variants=["shadow", "compression", "tiny_face"],
                overwrite=True,
                quality=88,
                manifest_rows=[],
                manifest_path=manifest_path,
            )
            outputs = {row["output_image"] for row in rows}
            self.assertEqual(len(rows), 3)
            self.assertTrue(any("lighting--shadow" in item for item in outputs))
            self.assertTrue(any("quality--compression" in item for item in outputs))
            self.assertTrue(any("scale--tiny_face" in item for item in outputs))
            self.assertTrue((student_dir / LOCAL_OUTPUT_DIR).exists())
            self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
