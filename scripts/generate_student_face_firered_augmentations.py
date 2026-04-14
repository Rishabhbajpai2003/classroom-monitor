from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from gradio_client import Client, handle_file
from PIL import Image, ImageOps

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


SPACE_ID = "prithivMLmods/FireRed-Image-Edit-1.0-Fast"
QWEN_LORA_SPACE_ID = "prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast"
ANGLE_SPACE_ID = "linoyts/Qwen-Image-Edit-Angles"
OUTPUT_DIR = "augmentations_generated"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
POSE_TAGS = ("left_profile", "right_profile", "slight_tilt", "up_angle", "down_angle")
ACCESSORY_TAGS = ("sunglasses", "cap", "mask", "scarf")
COMBO_TAGS = (
    "cap_left_profile",
    "sunglasses_slight_tilt",
    "mask_down_angle",
    "scarf_right_profile",
)


@dataclass(frozen=True)
class GenerationSpec:
    family: str
    tag: str
    combination_tag: str
    prompt: str


@dataclass(frozen=True)
class StudentSource:
    student_dir: Path
    source_image: Path


THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pose/accessory FireRed augmentations for student-details photos in parallel.",
    )
    parser.add_argument("--root", default="student-details", help="student-details root folder.")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "student-details" / "augmentation_manifest.csv"),
        help="Unified augmentation manifest path.",
    )
    parser.add_argument("--space", default=SPACE_ID, help="Hugging Face Space id.")
    parser.add_argument(
        "--space-kind",
        default="auto",
        choices=("auto", "firered", "qwen_lora", "angles"),
        help="Hosted API shape to use.",
    )
    parser.add_argument("--hf-token", default="", help="Optional HF token; otherwise HF_TOKEN env var is used.")
    parser.add_argument("--max-workers", type=int, default=6, help="Parallel worker count.")
    parser.add_argument("--max-students", type=int, default=0, help="Optional cap on student folders.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per generation task.")
    parser.add_argument("--request-timeout", type=float, default=180.0, help="Per request timeout in seconds.")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Base backoff seconds between retries.")
    parser.add_argument(
        "--smart-mode",
        action="store_true",
        help="Handle hosted quota windows more gracefully and resume after waiting.",
    )
    parser.add_argument(
        "--quota-max-sleep",
        type=int,
        default=1200,
        help="Maximum seconds smart mode will sleep for a quota retry window.",
    )
    parser.add_argument(
        "--quota-wait-limit",
        type=int,
        default=4,
        help="Maximum quota-window waits per task before marking it as error.",
    )
    parser.add_argument("--guidance-scale", type=float, default=1.2, help="FireRed true_cfg_scale.")
    parser.add_argument("--steps", type=int, default=6, help="Inference steps.")
    parser.add_argument(
        "--qwen-lora-adapter",
        default="Multiple-Angles",
        help="Adapter to use when talking to the Qwen LoRAs Space.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if accepted output exists.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed. Randomized per task unless disabled.")
    parser.add_argument("--disable-randomize-seed", action="store_true", help="Use deterministic seeds.")
    parser.add_argument(
        "--families",
        default="pose,accessory,combo",
        help="Comma-separated families to generate from pose,accessory,combo.",
    )
    return parser.parse_args()


def build_specs(selected_families: List[str]) -> List[GenerationSpec]:
    specs: List[GenerationSpec] = []
    if "pose" in selected_families:
        prompts = {
            "left_profile": "Edit this into a realistic left profile portrait of the same person. Preserve facial identity, age, skin tone, hairstyle, clothing, and background. Keep exactly one person.",
            "right_profile": "Edit this into a realistic right profile portrait of the same person. Preserve facial identity, age, skin tone, hairstyle, clothing, and background. Keep exactly one person.",
            "slight_tilt": "Edit this into a realistic slight head-tilt portrait of the same person. Preserve facial identity, skin tone, hairstyle, clothing, and background. Keep exactly one person.",
            "up_angle": "Edit this into a realistic portrait of the same person looking slightly upward. Preserve facial identity, skin tone, hairstyle, clothing, and background. Keep exactly one person.",
            "down_angle": "Edit this into a realistic portrait of the same person looking slightly downward. Preserve facial identity, skin tone, hairstyle, clothing, and background. Keep exactly one person.",
        }
        specs.extend(GenerationSpec("pose", tag, "", prompts[tag]) for tag in POSE_TAGS)
    if "accessory" in selected_families:
        prompts = {
            "sunglasses": "Add realistic sunglasses to the same person. Preserve facial identity, skin tone, hairstyle, clothing, and background. Keep exactly one person.",
            "cap": "Add a realistic cap to the same person. Preserve facial identity, skin tone, hairstyle where visible, clothing, and background. Keep exactly one person.",
            "mask": "Add a realistic face mask to the same person. Preserve facial identity around the visible eye region, skin tone, hairstyle, clothing, and background. Keep exactly one person.",
            "scarf": "Add a realistic scarf around the same person. Preserve facial identity, skin tone, hairstyle, clothing, and background. Keep exactly one person.",
        }
        specs.extend(GenerationSpec("accessory", tag, "", prompts[tag]) for tag in ACCESSORY_TAGS)
    if "combo" in selected_families:
        prompts = {
            "cap_left_profile": "Edit this into a realistic left profile portrait of the same person wearing a cap. Preserve facial identity, skin tone, hairstyle where visible, clothing, and background. Keep exactly one person.",
            "sunglasses_slight_tilt": "Edit this into a realistic slight head-tilt portrait of the same person wearing sunglasses. Preserve facial identity, skin tone, hairstyle, clothing, and background. Keep exactly one person.",
            "mask_down_angle": "Edit this into a realistic portrait of the same person looking slightly downward while wearing a face mask. Preserve facial identity around the visible eye region, skin tone, hairstyle, clothing, and background. Keep exactly one person.",
            "scarf_right_profile": "Edit this into a realistic right profile portrait of the same person wearing a scarf. Preserve facial identity, skin tone, hairstyle, clothing, and background. Keep exactly one person.",
        }
        specs.extend(GenerationSpec("combo", tag, tag, prompts[tag]) for tag in COMBO_TAGS)
    return specs


def iter_student_sources(root: Path, max_students: int) -> List[StudentSource]:
    students: List[StudentSource] = []
    student_dirs = [path for path in sorted(root.iterdir(), key=lambda item: item.name.lower()) if path.is_dir()]
    if max_students > 0:
        student_dirs = student_dirs[: max_students]
    for student_dir in student_dirs:
        photos = [
            path
            for path in sorted(student_dir.iterdir(), key=lambda item: item.name.lower())
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS and path.name.lower().startswith("photo-")
        ]
        if not photos:
            continue
        best_photo = max(photos, key=photo_rank)
        students.append(StudentSource(student_dir=student_dir, source_image=best_photo))
    return students


def photo_rank(path: Path) -> tuple[int, int, str]:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
    except Exception:
        width, height = 0, 0
    return (width * height, path.stat().st_size if path.exists() else 0, path.name.lower())


def encode_image_b64(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def detect_space_kind(space_id: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lowered = space_id.lower()
    if "qwen-image-edit-angles" in lowered:
        return "angles"
    if "qwen-image-edit-2511-loras-fast" in lowered:
        return "qwen_lora"
    return "firered"


def is_quota_error(message: str) -> bool:
    lowered = str(message).lower()
    return "gpu quota" in lowered or "try again in" in lowered


def parse_retry_after_seconds(message: str) -> Optional[int]:
    match = re.search(r"try again in\s+(\d+):(\d+):(\d+)", str(message), flags=re.IGNORECASE)
    if not match:
        return None
    hours, minutes, seconds = (int(part) for part in match.groups())
    return (hours * 3600) + (minutes * 60) + seconds


def angle_params_for_tag(tag: str) -> tuple[float, float, float, bool]:
    mapping = {
        "left_profile": (-55.0, 0.0, 0.0, False),
        "right_profile": (55.0, 0.0, 0.0, False),
        "slight_tilt": (0.0, 0.0, 0.45, False),
        "up_angle": (0.0, 0.0, 1.0, False),
        "down_angle": (0.0, 0.0, -1.0, False),
    }
    return mapping[tag]


def resized_dimensions_for_space(path: Path, max_dim: int = 1024) -> tuple[int, int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
    if width > height:
        new_width = max_dim
        new_height = int(new_width * height / width)
    else:
        new_height = max_dim
        new_width = int(new_height * width / height)
    return (max(256, (new_width // 8) * 8), max(256, (new_height // 8) * 8))


def output_path_for(student_dir: Path, source_image: Path, spec: GenerationSpec) -> Path:
    out_name = f"{source_image.stem}__gen-{spec.family}--{spec.tag}.png"
    return student_dir / OUTPUT_DIR / out_name


def manifest_key(student_folder: str, source_image: str, spec: GenerationSpec) -> tuple[str, str, str, str, str, str]:
    return (
        student_folder.lower(),
        source_image.lower(),
        "hosted",
        spec.family.lower(),
        spec.tag.lower(),
        spec.combination_tag.lower(),
    )


def accepted_manifest_lookup(rows: List[Dict[str, str]]) -> dict[tuple[str, str, str, str, str, str], Dict[str, str]]:
    lookup: dict[tuple[str, str, str, str, str, str], Dict[str, str]] = {}
    for row in rows:
        if str(row.get("generator_type", "")).strip().lower() != "hosted":
            continue
        if str(row.get("status", "")).strip().lower() != "accepted":
            continue
        key = (
            str(row.get("student_folder", "")).lower(),
            str(row.get("source_image", "")).lower(),
            "hosted",
            str(row.get("family", "")).lower(),
            str(row.get("tag", "")).lower(),
            str(row.get("combination_tag", "")).lower(),
        )
        lookup[key] = row
    return lookup


def get_client(space_id: str, hf_token: str, request_timeout: float) -> Client:
    client = getattr(THREAD_LOCAL, "client", None)
    client_key = getattr(THREAD_LOCAL, "client_key", None)
    desired_key = (space_id, hf_token, round(float(request_timeout), 2))
    if client is None or client_key != desired_key:
        kwargs = {}
        if hf_token:
            kwargs["token"] = hf_token
        kwargs["verbose"] = False
        kwargs["httpx_kwargs"] = {"timeout": float(request_timeout)}
        THREAD_LOCAL.client = Client(space_id, **kwargs)
        THREAD_LOCAL.client_key = desired_key
    return THREAD_LOCAL.client


def save_result_image(result_payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result_payload, str) and result_payload:
        with Image.open(result_payload) as image:
            image.save(destination)
        return
    if isinstance(result_payload, dict):
        result_path = result_payload.get("path")
        if result_path:
            with Image.open(result_path) as image:
                image.save(destination)
            return
        result_url = result_payload.get("url")
        if result_url and result_url.startswith("data:image"):
            _, data = result_url.split(",", 1)
            destination.write_bytes(base64.b64decode(data))
            return
    raise RuntimeError("FireRed response did not include a usable image payload.")


def run_generation_task(
    student: StudentSource,
    spec: GenerationSpec,
    space_id: str,
    space_kind: str,
    hf_token: str,
    retries: int,
    request_timeout: float,
    retry_backoff: float,
    smart_mode: bool,
    quota_max_sleep: int,
    quota_wait_limit: int,
    guidance_scale: float,
    steps: int,
    base_seed: int,
    randomize_seed: bool,
    qwen_lora_adapter: str,
) -> Dict[str, str]:
    out_path = output_path_for(student.student_dir, student.source_image, spec)
    payload = f'["{encode_image_b64(student.source_image)}"]'
    last_error = ""
    max_attempts = max(1, retries)
    attempt = 1
    quota_waits = 0
    while attempt <= max_attempts:
        try:
            client = get_client(space_id, hf_token, request_timeout)
            seed = int(base_seed + attempt + random.randint(0, 10_000))
            if space_kind == "angles":
                if spec.family != "pose":
                    raise RuntimeError("Angles space only supports pose-family generations.")
                rotate_deg, move_forward, vertical_tilt, wideangle = angle_params_for_tag(spec.tag)
                width, height = resized_dimensions_for_space(student.source_image)
                image_file = handle_file(str(student.source_image))
                result_image, result_seed, _prompt = client.predict(
                    image_file,
                    rotate_deg,
                    move_forward,
                    vertical_tilt,
                    wideangle,
                    seed,
                    bool(randomize_seed),
                    float(guidance_scale),
                    int(steps),
                    width,
                    height,
                    image_file,
                    api_name="/infer_and_show_video_button",
                )
            elif space_kind == "qwen_lora":
                gallery_item = {"image": handle_file(str(student.source_image)), "caption": ""}
                result_image, result_seed = client.predict(
                    [gallery_item],
                    spec.prompt,
                    qwen_lora_adapter,
                    seed,
                    bool(randomize_seed),
                    float(guidance_scale),
                    float(steps),
                    api_name="/infer",
                )
            else:
                result_image, result_seed = client.predict(
                    payload,
                    spec.prompt,
                    seed,
                    bool(randomize_seed),
                    float(guidance_scale),
                    float(steps),
                    api_name="/infer",
                )
            save_result_image(result_image, out_path)
            return {
                "student_folder": student.student_dir.name,
                "source_image": student.source_image.name,
                "generator_type": "hosted",
                "family": spec.family,
                "tag": spec.tag,
                "combination_tag": spec.combination_tag,
                "status": "accepted",
                "rejection_reason": "",
                "output_image": str(out_path.relative_to(student.student_dir.parent)),
                "seed_used": str(int(result_seed)),
            }
        except Exception as exc:
            last_error = str(exc).strip()
            if smart_mode and is_quota_error(last_error):
                wait_seconds = parse_retry_after_seconds(last_error)
                if wait_seconds is not None and quota_waits < max(0, int(quota_wait_limit)):
                    quota_waits += 1
                    sleep_for = min(int(quota_max_sleep), max(10, wait_seconds + 5))
                    print(
                        f"[smart-mode] {student.student_dir.name}/{spec.tag}: quota window detected, "
                        f"sleeping {sleep_for}s before retry {attempt}/{max_attempts} "
                        f"(quota wait {quota_waits}/{quota_wait_limit})"
                    )
                    time.sleep(sleep_for)
                    continue
            if attempt < max_attempts:
                time.sleep(float(retry_backoff) * attempt)
            attempt += 1
    return {
        "student_folder": student.student_dir.name,
        "source_image": student.source_image.name,
        "generator_type": "hosted",
        "family": spec.family,
        "tag": spec.tag,
        "combination_tag": spec.combination_tag,
        "status": "error",
        "rejection_reason": last_error[:400],
        "output_image": "",
    }


def planned_tasks(
    students: List[StudentSource],
    specs: List[GenerationSpec],
    manifest_rows: List[Dict[str, str]],
    overwrite: bool,
) -> List[tuple[StudentSource, GenerationSpec]]:
    existing_lookup = accepted_manifest_lookup(manifest_rows)
    tasks: List[tuple[StudentSource, GenerationSpec]] = []
    for student in students:
        for spec in specs:
            key = manifest_key(student.student_dir.name, student.source_image.name, spec)
            out_path = output_path_for(student.student_dir, student.source_image, spec)
            if not overwrite and key in existing_lookup and out_path.exists():
                continue
            tasks.append((student, spec))
    return tasks


def selected_families_from_arg(raw: str) -> List[str]:
    families = [item.strip().lower() for item in raw.split(",") if item.strip()]
    allowed = {"pose", "accessory", "combo"}
    unknown = [item for item in families if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown families requested: {', '.join(unknown)}")
    return families or ["pose", "accessory", "combo"]


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    manifest_path = Path(args.manifest).resolve()
    hf_token = str(args.hf_token or os.environ.get("HF_TOKEN", "")).strip()

    if not root.exists():
        raise FileNotFoundError(f"student-details root not found: {root}")

    families = selected_families_from_arg(args.families)
    specs = build_specs(families)
    students = iter_student_sources(root, args.max_students)
    manifest_rows = load_manifest(manifest_path)
    tasks = planned_tasks(students, specs, manifest_rows, args.overwrite)
    space_kind = detect_space_kind(args.space, args.space_kind)

    print(f"Students selected : {len(students)}")
    print(f"Specs selected    : {len(specs)}")
    print(f"Tasks queued      : {len(tasks)}")
    print(f"Max workers       : {args.max_workers}")
    print(f"Space kind        : {space_kind}")
    if args.smart_mode:
        print(
            f"Smart mode        : on (quota_max_sleep={int(args.quota_max_sleep)}s, "
            f"quota_wait_limit={int(args.quota_wait_limit)})"
        )

    accepted = 0
    errors = 0
    skipped = len(students) * len(specs) - len(tasks)
    completed = 0

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        future_map = {
            executor.submit(
                run_generation_task,
                student,
                spec,
                args.space,
                space_kind,
                hf_token,
                int(args.retries),
                float(args.request_timeout),
                float(args.retry_backoff),
                bool(args.smart_mode),
                int(args.quota_max_sleep),
                int(args.quota_wait_limit),
                float(args.guidance_scale),
                int(args.steps),
                int(args.seed),
                not bool(args.disable_randomize_seed),
                str(args.qwen_lora_adapter),
            ): (student, spec)
            for student, spec in tasks
        }
        for future in as_completed(future_map):
            row = future.result()
            manifest_rows = upsert_manifest_row(manifest_rows, row)
            save_manifest(manifest_path, manifest_rows)
            completed += 1
            if row["status"] == "accepted":
                accepted += 1
            else:
                errors += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(
                    f"Progress {completed}/{len(tasks)} | accepted={accepted} | errors={errors} | skipped={skipped}"
                )

    print(f"Accepted outputs  : {accepted}")
    print(f"Errored outputs   : {errors}")
    print(f"Skipped existing  : {skipped}")
    print(f"Manifest          : {manifest_path}")


if __name__ == "__main__":
    main()
