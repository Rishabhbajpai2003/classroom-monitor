from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter
try:
    from pillow_heif import register_heif_opener
except Exception:
    register_heif_opener = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.augmentation_manifest import load_manifest, save_manifest, upsert_manifest_row


if register_heif_opener is not None:
    register_heif_opener()


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
LOCAL_OUTPUT_DIR = "augmentations_local"
LEGACY_OUTPUT_DIRS = ("augmentations",)
DEFAULT_VARIANTS = (
    "shadow",
    "low_light",
    "backlight",
    "harsh_light",
    "blur",
    "low_resolution",
    "noise",
    "compression",
    "tiny_face",
    "distant_face",
    "partial_crop",
)
VARIANT_FAMILY = {
    "shadow": "lighting",
    "low_light": "lighting",
    "backlight": "lighting",
    "harsh_light": "lighting",
    "blur": "quality",
    "low_resolution": "quality",
    "noise": "quality",
    "compression": "quality",
    "tiny_face": "scale",
    "distant_face": "scale",
    "partial_crop": "scale",
}
@dataclass(frozen=True)
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return self.x1 + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y1 + self.height / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic face augmentations under each student-details folder.",
    )
    parser.add_argument("--root", default="student-details", help="Root folder containing name-rollno student folders.")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated deterministic variants to generate.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing augmentation files.")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality for generated images.")
    parser.add_argument(
        "--photo-only",
        action="store_true",
        default=True,
        help="Only augment Photo-* source images. Enabled by default.",
    )
    parser.add_argument(
        "--purge-generated",
        action="store_true",
        default=True,
        help="Delete legacy augmentation directories and stale local variants before generation.",
    )
    return parser.parse_args()


def detect_primary_face(image: Image.Image) -> FaceBox:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(max(36, gray.shape[1] // 10), max(36, gray.shape[0] // 10)),
    )
    scale = 1.0
    if len(faces) == 0:
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        faces = cascade.detectMultiScale(
            small,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(24, 24),
        )
        scale = 2.0
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        return FaceBox(
            int(round(x * scale)),
            int(round(y * scale)),
            int(round((x + w) * scale)),
            int(round((y + h) * scale)),
        )

    width, height = image.size
    face_w = int(width * 0.42)
    face_h = int(height * 0.42)
    x1 = int((width - face_w) / 2)
    y1 = int(height * 0.16)
    return FaceBox(x1, y1, x1 + face_w, y1 + face_h)


def clamp_box(box: FaceBox, image: Image.Image) -> FaceBox:
    width, height = image.size
    return FaceBox(
        max(0, min(width - 1, box.x1)),
        max(0, min(height - 1, box.y1)),
        max(1, min(width, box.x2)),
        max(1, min(height, box.y2)),
    )


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image.copy()


def _gradient_mask(size: tuple[int, int], *, reverse: bool = False, vertical: bool = False) -> Image.Image:
    width, height = size
    if vertical:
        ramp = np.tile(np.linspace(0.16, 0.82, height, dtype=np.float32)[:, None], (1, width))
        if reverse:
            ramp = np.flipud(ramp)
    else:
        ramp = np.tile(np.linspace(0.16, 0.82, width, dtype=np.float32), (height, 1))
        if reverse:
            ramp = np.fliplr(ramp)
    alpha = np.clip(255.0 * ramp, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(alpha, mode="L")


def augmentation_shadow(image: Image.Image, face: FaceBox) -> Image.Image:
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    alpha = _gradient_mask(img.size, reverse=(face.center_x / max(1.0, float(img.size[0]))) > 0.5)
    overlay.putalpha(alpha)
    shaded = Image.alpha_composite(img, overlay)
    return ImageEnhance.Brightness(shaded.convert("RGB")).enhance(0.92)


def augmentation_low_light(image: Image.Image, face: FaceBox) -> Image.Image:
    shaded = augmentation_shadow(image, face)
    shaded = ImageEnhance.Brightness(shaded).enhance(0.62)
    shaded = ImageEnhance.Contrast(shaded).enhance(0.88)
    arr = np.asarray(shaded, dtype=np.float32)
    noise = np.random.default_rng(17).normal(0.0, 10.0, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def augmentation_backlight(image: Image.Image, face: FaceBox) -> Image.Image:
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 240, 220, 0))
    alpha = _gradient_mask(img.size, reverse=False, vertical=True)
    overlay.putalpha(alpha)
    lit = Image.alpha_composite(img, overlay)
    face_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(face_overlay, "RGBA")
    draw.ellipse(
        [
            int(face.x1 - face.width * 0.10),
            int(face.y1 - face.height * 0.08),
            int(face.x2 + face.width * 0.10),
            int(face.y2 + face.height * 0.10),
        ],
        fill=(0, 0, 0, 78),
    )
    return Image.alpha_composite(lit, face_overlay).convert("RGB")


def augmentation_harsh_light(image: Image.Image, face: FaceBox) -> Image.Image:
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 240, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    highlight_box = [
        int(face.x1 - face.width * 0.18),
        int(face.y1 - face.height * 0.10),
        int(face.x2 + face.width * 0.05),
        int(face.y2 + face.height * 0.12),
    ]
    draw.ellipse(highlight_box, fill=(255, 252, 230, 116))
    lit = Image.alpha_composite(img, overlay).convert("RGB")
    lit = ImageEnhance.Contrast(lit).enhance(1.26)
    return ImageEnhance.Brightness(lit).enhance(1.08)


def augmentation_blur(image: Image.Image, _: FaceBox) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=2.2))


def augmentation_low_resolution(image: Image.Image, _: FaceBox) -> Image.Image:
    width, height = image.size
    down = image.resize((max(24, width // 4), max(24, height // 4)), Image.Resampling.BILINEAR)
    return down.resize((width, height), Image.Resampling.BILINEAR)


def augmentation_noise(image: Image.Image, _: FaceBox) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = np.random.default_rng(29).normal(0.0, 14.0, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def augmentation_compression(image: Image.Image, _: FaceBox) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=22, subsampling=2, optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _composite_scaled_scene(image: Image.Image, scale: float, *, blur_radius: float = 6.0, brightness: float = 0.84) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    bg = rgb.resize((width, height), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    bg = ImageEnhance.Brightness(bg).enhance(brightness)
    scaled = rgb.resize(
        (max(24, int(round(width * scale))), max(24, int(round(height * scale)))),
        Image.Resampling.BILINEAR,
    )
    canvas = bg.copy()
    paste_x = (width - scaled.size[0]) // 2
    paste_y = (height - scaled.size[1]) // 2
    canvas.paste(scaled, (paste_x, paste_y))
    return canvas


def augmentation_tiny_face(image: Image.Image, _: FaceBox) -> Image.Image:
    return _composite_scaled_scene(image, 0.32)


def augmentation_distant_face(image: Image.Image, _: FaceBox) -> Image.Image:
    distant = _composite_scaled_scene(image, 0.42, blur_radius=8.0, brightness=0.80)
    return distant.filter(ImageFilter.GaussianBlur(radius=1.2))


def augmentation_partial_crop(image: Image.Image, face: FaceBox) -> Image.Image:
    width, height = image.size
    shift_x = int(face.width * 0.26)
    shift_y = int(face.height * 0.12)
    cropped = image.crop(
        (
            max(0, shift_x),
            max(0, shift_y),
            max(1, width),
            max(1, height - shift_y),
        )
    )
    out = Image.new("RGB", (width, height), tuple(int(v) for v in np.asarray(image.resize((1, 1))).reshape(-1)[:3]))
    out.paste(cropped.resize((width - shift_x, height - shift_y), Image.Resampling.BILINEAR), (shift_x, shift_y))
    return out


AUGMENTATION_FUNCS = {
    "shadow": augmentation_shadow,
    "low_light": augmentation_low_light,
    "backlight": augmentation_backlight,
    "harsh_light": augmentation_harsh_light,
    "blur": augmentation_blur,
    "low_resolution": augmentation_low_resolution,
    "noise": augmentation_noise,
    "compression": augmentation_compression,
    "tiny_face": augmentation_tiny_face,
    "distant_face": augmentation_distant_face,
    "partial_crop": augmentation_partial_crop,
}


def iter_source_images(student_dir: Path) -> Iterable[Path]:
    for path in sorted(student_dir.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS and not path.name.startswith("."):
            yield path


def should_augment_source(image_path: Path, photo_only: bool) -> bool:
    if not photo_only:
        return True
    return image_path.name.lower().startswith("photo-")


def purge_legacy_outputs(student_dir: Path, active_variants: List[str]) -> int:
    removed = 0
    for legacy_dir_name in LEGACY_OUTPUT_DIRS:
        legacy_dir = student_dir / legacy_dir_name
        if legacy_dir.exists():
            for path in legacy_dir.rglob("*"):
                if path.is_file():
                    path.unlink(missing_ok=True)
                    removed += 1
            for path in sorted(legacy_dir.rglob("*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
            legacy_dir.rmdir()

    local_dir = student_dir / LOCAL_OUTPUT_DIR
    if local_dir.exists():
        for path in sorted(local_dir.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file():
                continue
            stem = path.stem
            if "__local-" not in stem:
                path.unlink(missing_ok=True)
                removed += 1
                continue
            encoded = stem.split("__local-", 1)[1]
            _, _, tag = encoded.partition("--")
            if tag not in active_variants:
                path.unlink(missing_ok=True)
                removed += 1
    return removed


def save_image(image: Image.Image, output_path: Path, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=quality, optimize=True)


def generate_for_image(
    image_path: Path,
    variants: List[str],
    overwrite: bool,
    quality: int,
    manifest_rows: List[dict[str, str]],
    manifest_path: Path,
) -> List[dict[str, str]]:
    image = to_rgb(Image.open(image_path))
    face = clamp_box(detect_primary_face(image), image)
    output_dir = image_path.parent / LOCAL_OUTPUT_DIR
    rows: List[dict[str, str]] = []

    for variant in variants:
        family = VARIANT_FAMILY[variant]
        out_name = f"{image_path.stem}__local-{family}--{variant}.jpg"
        out_path = output_dir / out_name
        status = "exists"
        if not out_path.exists() or overwrite:
            aug = AUGMENTATION_FUNCS[variant](image.copy(), face)
            save_image(aug, out_path, quality=quality)
            status = "created"
        row = {
            "student_folder": image_path.parent.name,
            "source_image": image_path.name,
            "generator_type": "local",
            "family": family,
            "tag": variant,
            "combination_tag": "",
            "status": status,
            "rejection_reason": "",
            "output_image": str(out_path.relative_to(image_path.parent.parent)),
        }
        manifest_rows = upsert_manifest_row(manifest_rows, row)
        rows.append(row)
    save_manifest(manifest_path, manifest_rows)
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Student-details root not found: {root}")

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [item for item in variants if item not in AUGMENTATION_FUNCS]
    if unknown:
        raise ValueError(f"Unknown variants requested: {', '.join(unknown)}")

    manifest_path = root / "augmentation_manifest.csv"
    manifest_rows = load_manifest(manifest_path)
    if args.purge_generated:
        allowed_local = {
            ("local", VARIANT_FAMILY[variant], variant)
            for variant in variants
        }
        manifest_rows = [
            row
            for row in manifest_rows
            if (
                str(row.get("generator_type", "") or "").strip() == "local"
                and (
                    str(row.get("generator_type", "") or "").strip(),
                    str(row.get("family", "") or "").strip(),
                    str(row.get("tag", "") or "").strip(),
                )
                in allowed_local
            )
        ]
        save_manifest(manifest_path, manifest_rows)
    student_dirs = [path for path in sorted(root.iterdir(), key=lambda item: item.name.lower()) if path.is_dir()]

    removed = 0
    created = 0
    existing = 0
    for student_dir in student_dirs:
        if args.purge_generated:
            removed += purge_legacy_outputs(student_dir, variants)
        for image_path in iter_source_images(student_dir):
            if not should_augment_source(image_path, args.photo_only):
                continue
            rows = generate_for_image(
                image_path=image_path,
                variants=variants,
                overwrite=args.overwrite,
                quality=args.quality,
                manifest_rows=manifest_rows,
                manifest_path=manifest_path,
            )
            for row in rows:
                manifest_rows = upsert_manifest_row(manifest_rows, row)
                if row["status"] == "created":
                    created += 1
                else:
                    existing += 1

    save_manifest(manifest_path, manifest_rows)
    print(f"Students processed : {len(student_dirs)}")
    print(f"Variants requested : {len(variants)}")
    print(f"Augmentations made : {created}")
    print(f"Already present    : {existing}")
    print(f"Files removed      : {removed}")
    print(f"Manifest           : {manifest_path}")


if __name__ == "__main__":
    main()
