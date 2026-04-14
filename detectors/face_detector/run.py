#!/usr/bin/env python3
"""
run command : .\.venv\Scripts\python.exe detectors\face_detector\run.py --input media\classroom.mp4 --output outputs\face_tracked.mp4 --ctx 0 --ttl 120 --archive-ttl 1800 --reid-sim-thresh 0.55


Classroom face detection + persistent student ID assignment in a single file.

Stack:
- Detection + face embeddings: InsightFace FaceAnalysis (SCRFD + ArcFace family)
- Tracking / identity persistence: custom temporal association using
  cosine similarity + IoU + center-distance + TTL-based track memory

Why this design:
- Single-file and easy to run
- Very strong face detection/recognition backbone
- Stable enough for classroom videos without depending on multiple external trackers
- Keeps the same student ID across temporary misses and pose changes

Install:
    pip install insightface onnxruntime opencv-python numpy

GPU (optional, much faster):
    pip install onnxruntime-gpu

Run:
    python classroom_face_tracker.py \
        --input classroom.mp4 \
        --output classroom_tracked.mp4 \
        --csv detections.csv \
        --det-size 960 \
        --min-face 20 \
        --sim-thresh 0.45 \
        --ttl 90

Notes:
- First run may download InsightFace models automatically.
- For far-away students, increase --det-size to 1024 or 1280.
- If too many duplicate IDs are created, lower --sim-thresh slightly.
- If different students merge into one ID, raise --sim-thresh.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

try:
    from insightface.utils import face_align
except Exception:
    face_align = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

from app.models.identity_enrichment import (
    bank_match_score,
    build_grouped_embedding_summaries,
    compact_embedding_bank,
    current_utc_iso,
    grouped_bank_match_score,
    infer_lighting_bucket,
    infer_profile_bucket,
    infer_size_bucket,
    normalize_bank_family,
    sanitize_embedding_metadata,
    summarize_embedding_bank,
)
from app.models.face_attributes import FaceAttributePrediction, LocalFaceAttributeClassifier
from app.models.frame_change import FrameChangeGate
from app.models.roster_policy import capped_reportable_ids, roster_size
from app.models.unknown_review import (
    build_unknown_cluster_record,
    cluster_support_score,
    write_unknown_review_package,
)


DEFAULT_IDENTITY_DB_PATH = os.path.join(os.path.dirname(__file__), "identity_db.json")


# -----------------------------
# Utility functions
# -----------------------------

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        return v
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def nms_boxes(boxes: List[np.ndarray], scores: List[float], iou_thresh: float = 0.45) -> List[int]:
    if not boxes:
        return []

    order = sorted(range(len(boxes)), key=lambda idx: scores[idx], reverse=True)
    keep: List[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        remaining = []
        for idx in order:
            if iou_xyxy(boxes[current], boxes[idx]) < iou_thresh:
                remaining.append(idx)
        order = remaining
    return keep


def box_center(box: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_diag(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return float(math.hypot(x2 - x1, y2 - y1))


def normalized_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    dist = math.hypot(ax - bx, ay - by)
    denom = max(1.0, max(box_diag(box_a), box_diag(box_b)))
    return float(dist / denom)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def weighted_average_embeddings(vectors: List[np.ndarray], qualities: List[float]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    if len(qualities) != len(vectors):
        qualities = [1.0] * len(vectors)

    stacked = np.vstack(vectors).astype(np.float32)
    weights = np.asarray([max(0.05, float(q)) for q in qualities], dtype=np.float32)
    weights = weights / max(1e-6, float(np.sum(weights)))
    avg = np.sum(stacked * weights[:, None], axis=0)
    return l2_normalize(avg.astype(np.float32))


def expanded_face_context_box(
    bbox: np.ndarray,
    frame_shape: Tuple[int, int, int],
    scale_x: float = 1.45,
    scale_y: float = 1.95,
    shift_y: float = 0.18,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    frame_h, frame_w = frame_shape[:2]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2) + shift_y * (y2 - y1)
    width = (x2 - x1) * scale_x
    height = (y2 - y1) * scale_y

    nx1 = max(0, int(round(cx - 0.5 * width)))
    ny1 = max(0, int(round(cy - 0.5 * height)))
    nx2 = min(frame_w, int(round(cx + 0.5 * width)))
    ny2 = min(frame_h, int(round(cy + 0.5 * height)))
    return nx1, ny1, nx2, ny2


def extract_appearance_descriptor(frame_bgr: np.ndarray, bbox: np.ndarray, crop_size: int = 96) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = expanded_face_context_box(bbox, frame_bgr.shape)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten().astype(np.float32)
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten().astype(np.float32)
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten().astype(np.float32)
    hist = np.concatenate([hist_h, hist_s, hist_v], axis=0)
    hist_sum = float(np.sum(hist))
    if hist_sum > 0:
        hist /= hist_sum

    small_gray = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    small_gray = small_gray - float(np.mean(small_gray))
    small_gray /= float(np.std(small_gray) + 1e-6)
    small_gray *= 0.25

    descriptor = np.concatenate([hist, small_gray], axis=0)
    return l2_normalize(descriptor.astype(np.float32))


def estimate_crop_sharpness(frame_bgr: np.ndarray, bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1], x2)
    y2 = min(frame_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return clamp(math.log1p(variance) / 5.0, 0.0, 1.0)


def estimate_detection_quality(
    bbox: np.ndarray,
    score: float,
    landmarks: Optional[np.ndarray],
    sharpness: float = 0.0,
) -> float:
    x1, y1, x2, y2 = bbox
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))

    size_term = clamp(min(width, height) / 90.0, 0.0, 1.0)
    pose_term = 0.55
    if landmarks is not None and len(landmarks) >= 5:
        left_eye, right_eye, _, mouth_left, mouth_right = landmarks[:5]
        eye_dist = float(np.linalg.norm(left_eye - right_eye) / width)
        mouth_dist = float(np.linalg.norm(mouth_left - mouth_right) / width)
        pose_term = 0.5 * clamp(eye_dist / 0.30, 0.0, 1.0) + 0.5 * clamp(mouth_dist / 0.38, 0.0, 1.0)

    sharpness_term = clamp(sharpness, 0.0, 1.0)
    return float(
        0.40 * clamp(score, 0.0, 1.0)
        + 0.25 * size_term
        + 0.20 * pose_term
        + 0.15 * sharpness_term
    )


def estimate_crop_brightness(frame_bgr: np.ndarray, bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1], x2)
    y2 = min(frame_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def estimate_shadow_severity(frame_bgr: np.ndarray, bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1], x2)
    y2 = min(frame_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_pixels = float(np.mean(gray < 48))
    bright_pixels = float(np.mean(gray > 175))
    return clamp(dark_pixels * 0.85 + max(0.0, dark_pixels - bright_pixels) * 0.35, 0.0, 1.0)


def estimate_noise_level(frame_bgr: np.ndarray, bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1], x2)
    y2 = min(frame_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray - blurred
    noise_std = float(np.std(residual))
    return clamp(noise_std / 24.0, 0.0, 1.0)


def estimate_blockiness(frame_bgr: np.ndarray, bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1], x2)
    y2 = min(frame_bgr.shape[0], y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return 0.0
    gray = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    vertical_edges = 0.0
    if gray.shape[1] > 8:
        vertical_a = gray[:, 8::8]
        vertical_b = gray[:, 7::8]
        cols = min(vertical_a.shape[1], vertical_b.shape[1])
        if cols > 0:
            vertical_edges = float(np.abs(vertical_a[:, :cols] - vertical_b[:, :cols]).mean())
    horizontal_edges = 0.0
    if gray.shape[0] > 8:
        horizontal_a = gray[8::8, :]
        horizontal_b = gray[7::8, :]
        rows = min(horizontal_a.shape[0], horizontal_b.shape[0])
        if rows > 0:
            horizontal_edges = float(np.abs(horizontal_a[:rows, :] - horizontal_b[:rows, :]).mean())
    return clamp((float(vertical_edges) + float(horizontal_edges)) / 48.0, 0.0, 1.0)


def estimate_backlight_score(frame_bgr: np.ndarray, bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1], x2)
    y2 = min(frame_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    face_crop = frame_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return 0.0
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    face_brightness = float(np.mean(gray_face))
    pad_x = int(round((x2 - x1) * 0.40))
    pad_y = int(round((y2 - y1) * 0.40))
    sx1 = max(0, x1 - pad_x)
    sy1 = max(0, y1 - pad_y)
    sx2 = min(frame_bgr.shape[1], x2 + pad_x)
    sy2 = min(frame_bgr.shape[0], y2 + pad_y)
    scene_crop = frame_bgr[sy1:sy2, sx1:sx2]
    if scene_crop.size == 0:
        return 0.0
    scene_gray = cv2.cvtColor(scene_crop, cv2.COLOR_BGR2GRAY)
    scene_brightness = float(np.mean(scene_gray))
    return clamp((scene_brightness - face_brightness) / 90.0, 0.0, 1.0)


def estimate_harsh_light_score(frame_bgr: np.ndarray, bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1], x2)
    y2 = min(frame_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bright_pixels = float(np.mean(gray > 200))
    dark_pixels = float(np.mean(gray < 55))
    contrast = float(np.std(gray))
    return clamp(0.55 * bright_pixels + 0.45 * dark_pixels + (contrast / 128.0), 0.0, 1.0)


def estimate_occlusion_ratio(frame_bgr: np.ndarray, bbox: np.ndarray, landmarks: Optional[np.ndarray]) -> float:
    if landmarks is None or len(landmarks) < 5:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    left_eye, right_eye, nose, mouth_left, mouth_right = [np.asarray(item, dtype=np.float32) for item in landmarks[:5]]
    eye_span = float(np.linalg.norm(left_eye - right_eye) / width)
    mouth_span = float(np.linalg.norm(mouth_left - mouth_right) / width)
    nose_y = float((nose[1] - y1) / height)
    score = 0.0
    if eye_span < 0.16:
        score += 0.35
    if mouth_span < 0.16:
        score += 0.25
    if nose_y < 0.22 or nose_y > 0.72:
        score += 0.20
    return clamp(score, 0.0, 1.0)


def build_quality_profile(
    frame_bgr: np.ndarray,
    bbox: np.ndarray,
    score: float,
    landmarks: Optional[np.ndarray],
    sharpness: float,
) -> Dict[str, object]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    face_size = float(max(0.0, min(x2 - x1, y2 - y1)))
    brightness = estimate_crop_brightness(frame_bgr, bbox)
    shadow = estimate_shadow_severity(frame_bgr, bbox)
    noise_level = estimate_noise_level(frame_bgr, bbox)
    blockiness = estimate_blockiness(frame_bgr, bbox)
    backlight = estimate_backlight_score(frame_bgr, bbox)
    harsh_light = estimate_harsh_light_score(frame_bgr, bbox)
    occlusion = estimate_occlusion_ratio(frame_bgr, bbox, landmarks)
    quality = estimate_detection_quality(bbox, score, landmarks, sharpness=sharpness)
    lighting_tag = ""
    if backlight >= 0.40:
        lighting_tag = "backlight"
    elif harsh_light >= 0.55:
        lighting_tag = "harsh_light"
    elif shadow >= 0.42:
        lighting_tag = "shadow"
    elif brightness < 78.0:
        lighting_tag = "low_light"

    quality_tags: List[str] = []
    if sharpness < 0.28:
        quality_tags.append("blur")
    if face_size < 56.0:
        quality_tags.append("low_resolution")
    if noise_level >= 0.42:
        quality_tags.append("noise")
    if blockiness >= 0.36:
        quality_tags.append("compression")

    scale_tag = ""
    if occlusion >= 0.42:
        scale_tag = "partial_crop"
    elif face_size < 28.0:
        scale_tag = "tiny_face"
    elif face_size < 58.0:
        scale_tag = "distant_face"
    return {
        "face_size": face_size,
        "sharpness": float(sharpness),
        "brightness": float(brightness),
        "shadow_severity": float(shadow),
        "noise_level": float(noise_level),
        "blockiness": float(blockiness),
        "backlight_score": float(backlight),
        "harsh_light_score": float(harsh_light),
        "occlusion_ratio": float(occlusion),
        "lighting_bucket": infer_lighting_bucket(brightness, shadow),
        "lighting_tag": lighting_tag,
        "quality_tags": quality_tags,
        "quality": float(quality),
        "size_bucket": infer_size_bucket(face_size),
        "scale_tag": scale_tag,
    }


def prefer_low_quality_embedder(quality_profile: Dict[str, object]) -> bool:
    face_size = float(quality_profile.get("face_size", 0.0) or 0.0)
    sharpness = float(quality_profile.get("sharpness", 0.0) or 0.0)
    brightness = float(quality_profile.get("brightness", 0.0) or 0.0)
    shadow = float(quality_profile.get("shadow_severity", 0.0) or 0.0)
    occlusion = float(quality_profile.get("occlusion_ratio", 0.0) or 0.0)
    quality = float(quality_profile.get("quality", 0.0) or 0.0)
    return bool(
        face_size < 48.0
        or sharpness < 0.28
        or brightness < 82.0
        or shadow >= 0.42
        or occlusion >= 0.40
        or quality < 0.52
    )


def runtime_family_tags(
    quality_profile: Dict[str, object],
    attribute_prediction: Optional[FaceAttributePrediction] = None,
    *,
    accessory_threshold: float = 0.55,
) -> List[str]:
    families: List[str] = []
    lighting_tag = str(quality_profile.get("lighting_tag", "") or "").strip()
    if lighting_tag:
        families.append(f"lighting/{lighting_tag}")
    for quality_tag in list(quality_profile.get("quality_tags", []) or []):
        if quality_tag:
            families.append(f"quality/{quality_tag}")
    scale_tag = str(quality_profile.get("scale_tag", "") or "").strip()
    if scale_tag:
        families.append(f"scale/{scale_tag}")

    if attribute_prediction is not None:
        pose_bucket = str(attribute_prediction.pose_bucket or "").strip()
        if pose_bucket and pose_bucket != "frontal":
            families.append(f"pose/{pose_bucket}")
        accessories = attribute_prediction.confident_accessories(threshold=accessory_threshold)
        for accessory in accessories:
            families.append(f"accessory/{accessory}")
        if pose_bucket and pose_bucket != "frontal" and accessories:
            for accessory in accessories:
                families.append(f"combo/{accessory}_{pose_bucket}")

    families.append("base")
    deduped: List[str] = []
    seen = set()
    for family in families:
        if family in seen:
            continue
        seen.add(family)
        deduped.append(family)
    return deduped


def prioritized_runtime_families(family_tags: Sequence[str]) -> List[str]:
    priority_by_prefix = {
        "combo/": 0,
        "pose/": 1,
        "accessory/": 2,
        "lighting/": 3,
        "quality/": 4,
        "scale/": 5,
        "base": 6,
        "global": 7,
    }

    def _priority(value: str) -> tuple[int, str]:
        text = str(value or "").strip()
        if text == "base":
            return priority_by_prefix["base"], text
        if text == "global":
            return priority_by_prefix["global"], text
        for prefix, rank in priority_by_prefix.items():
            if prefix.endswith("/") and text.startswith(prefix):
                return rank, text
        return 99, text

    ordered = []
    seen = set()
    for family in sorted((str(item or "").strip() for item in family_tags if str(item or "").strip()), key=_priority):
        if family in seen:
            continue
        seen.add(family)
        ordered.append(family)
    if "base" not in seen:
        ordered.append("base")
    return ordered


def preferred_runtime_bank_family(family_tags: Sequence[str]) -> str:
    ordered = prioritized_runtime_families(family_tags)
    return ordered[0] if ordered else "base"


def ensure_parent_dir(path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def color_from_id(track_id: int) -> Tuple[int, int, int]:
    # Deterministic pseudo-random BGR color
    rng = np.random.default_rng(abs(track_id) * 9973 + 17)
    vals = rng.integers(80, 255, size=3).tolist()
    return int(vals[0]), int(vals[1]), int(vals[2])


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Detection:
    bbox: np.ndarray               # shape (4,) -> [x1, y1, x2, y2]
    score: float
    embedding: np.ndarray          # L2-normalized embedding
    landmarks: Optional[np.ndarray] = None
    appearance: Optional[np.ndarray] = None
    quality: float = 0.0
    detector_used: str = "scrfd"
    embedder_used: str = "arcface"
    quality_profile: Dict[str, object] = field(default_factory=dict)
    attribute_prediction: Optional[FaceAttributePrediction] = None
    family_tags: List[str] = field(default_factory=list)
    attribute_signature: str = "frontal"


@dataclass
class TrackObservation:
    raw_track_id: int
    frame_idx: int
    bbox: np.ndarray
    score: float
    quality: float
    embedding: np.ndarray
    landmarks: Optional[np.ndarray] = None
    detector_used: str = "scrfd"
    embedder_used: str = "arcface"
    quality_profile: Dict[str, object] = field(default_factory=dict)
    assignment_mode: str = "full_reid"
    bank_family_used: str = "base"
    attribute_signature: str = "frontal"


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    last_frame_idx: int
    first_frame_idx: int
    hits: int = 1
    misses: int = 0
    best_score: float = 0.0
    embeddings: List[np.ndarray] = field(default_factory=list)
    embedding_qualities: List[float] = field(default_factory=list)
    embedding_metadata: List[Dict[str, object]] = field(default_factory=list)
    avg_embedding: Optional[np.ndarray] = None
    best_embedding: Optional[np.ndarray] = None
    best_embedding_quality: float = 0.0
    recent_embedding: Optional[np.ndarray] = None
    embedding_groups: Dict[str, Dict[str, object]] = field(default_factory=dict)
    appearance_embeddings: List[np.ndarray] = field(default_factory=list)
    appearance_qualities: List[float] = field(default_factory=list)
    avg_appearance: Optional[np.ndarray] = None
    best_appearance: Optional[np.ndarray] = None
    best_appearance_quality: float = 0.0
    recent_appearance: Optional[np.ndarray] = None
    persistent_identity: bool = False
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    metadata: Dict[str, object] = field(default_factory=dict)
    last_assignment_mode: str = "full_reid"
    last_bank_family_used: str = "base"
    last_attribute_signature: str = "frontal"

    def update_embedding_bank(
        self,
        new_emb: np.ndarray,
        sample_quality: float = 0.0,
        max_bank: int = 15,
        novelty_thresh: float = 0.10,
        sample_metadata: Optional[Dict[str, object]] = None,
        duplicate_sim_thresh: float = 0.92,
    ) -> None:
        """
        Keep multiple embeddings per student so different poses/lighting are remembered.
        novelty_thresh is based on cosine distance = 1 - similarity.
        """
        new_emb = l2_normalize(new_emb.astype(np.float32))
        while len(self.embedding_qualities) < len(self.embeddings):
            self.embedding_qualities.append(max(0.5, float(self.best_embedding_quality)))
        while len(self.embedding_metadata) < len(self.embeddings):
            self.embedding_metadata.append(
                sanitize_embedding_metadata(
                    quality=float(self.embedding_qualities[len(self.embedding_metadata)]) if len(self.embedding_metadata) < len(self.embedding_qualities) else float(sample_quality),
                    added_at=current_utc_iso(),
                )
            )

        effective_duplicate_thresh = max(float(duplicate_sim_thresh), 1.0 - float(novelty_thresh))
        metadata = sanitize_embedding_metadata(sample_metadata, quality=float(sample_quality), added_at=current_utc_iso())
        self.embeddings.append(new_emb)
        self.embedding_qualities.append(float(sample_quality))
        self.embedding_metadata.append(metadata)
        self.embeddings, self.embedding_qualities, self.embedding_metadata = compact_embedding_bank(
            self.embeddings,
            self.embedding_qualities,
            self.embedding_metadata,
            max_bank=max_bank,
            duplicate_sim_thresh=effective_duplicate_thresh,
        )
        self.avg_embedding, self.best_embedding, self.best_embedding_quality, self.recent_embedding = summarize_embedding_bank(
            self.embeddings,
            self.embedding_qualities,
            self.embedding_metadata,
        )
        self.embedding_groups = build_grouped_embedding_summaries(
            self.embeddings,
            self.embedding_qualities,
            self.embedding_metadata,
        )

    def update_appearance_bank(
        self,
        new_desc: Optional[np.ndarray],
        sample_quality: float = 0.0,
        max_bank: int = 12,
        novelty_thresh: float = 0.14,
    ) -> None:
        if new_desc is None:
            return

        new_desc = l2_normalize(new_desc.astype(np.float32))
        while len(self.appearance_qualities) < len(self.appearance_embeddings):
            self.appearance_qualities.append(max(0.5, float(self.best_appearance_quality)))
        if not self.appearance_embeddings:
            self.appearance_embeddings.append(new_desc)
            self.appearance_qualities.append(float(sample_quality))
        else:
            sims = [cosine_similarity(new_desc, e) for e in self.appearance_embeddings]
            best_sim = max(sims)
            cosine_dist = 1.0 - best_sim
            if cosine_dist >= novelty_thresh and len(self.appearance_embeddings) < max_bank:
                self.appearance_embeddings.append(new_desc)
                self.appearance_qualities.append(float(sample_quality))
            else:
                best_idx = int(np.argmax(sims))
                self.appearance_embeddings[best_idx] = l2_normalize(0.75 * self.appearance_embeddings[best_idx] + 0.25 * new_desc)
                while len(self.appearance_qualities) < len(self.appearance_embeddings):
                    self.appearance_qualities.append(float(sample_quality))
                self.appearance_qualities[best_idx] = max(float(sample_quality), self.appearance_qualities[best_idx])

        self.avg_appearance = weighted_average_embeddings(self.appearance_embeddings, self.appearance_qualities)
        if self.recent_appearance is None:
            self.recent_appearance = new_desc.copy()
        else:
            self.recent_appearance = l2_normalize(0.65 * self.recent_appearance + 0.35 * new_desc)
        if self.best_appearance is None or sample_quality >= self.best_appearance_quality:
            self.best_appearance = new_desc.copy()
            self.best_appearance_quality = float(sample_quality)

    def predict_bbox(self) -> np.ndarray:
        return self.bbox + self.velocity


def generate_overlapping_tiles(
    frame_shape: Tuple[int, int, int],
    tile_grid: int,
    overlap: float,
) -> List[Tuple[int, int, int, int]]:
    if tile_grid <= 1:
        return [(0, 0, frame_shape[1], frame_shape[0])]

    frame_h, frame_w = frame_shape[:2]
    overlap = clamp(overlap, 0.0, 0.45)
    step_x = frame_w / float(tile_grid)
    step_y = frame_h / float(tile_grid)
    pad_x = int(round(step_x * overlap))
    pad_y = int(round(step_y * overlap))

    tiles: List[Tuple[int, int, int, int]] = []
    for gy in range(tile_grid):
        for gx in range(tile_grid):
            x1 = max(0, int(round(gx * step_x)) - pad_x)
            y1 = max(0, int(round(gy * step_y)) - pad_y)
            x2 = min(frame_w, int(round((gx + 1) * step_x)) + pad_x)
            y2 = min(frame_h, int(round((gy + 1) * step_y)) + pad_y)
            if x2 > x1 and y2 > y1:
                tiles.append((x1, y1, x2, y2))
    return tiles


def deduplicate_detections(detections: List[Detection], iou_thresh: float = 0.45, sim_thresh: float = 0.45) -> List[Detection]:
    if not detections:
        return []

    detections = sorted(detections, key=lambda det: (det.quality, det.score), reverse=True)
    kept: List[Detection] = []
    for det in detections:
        duplicate = False
        for existing in kept:
            if iou_xyxy(det.bbox, existing.bbox) < iou_thresh:
                continue
            if cosine_similarity(det.embedding, existing.embedding) < sim_thresh:
                continue
            duplicate = True
            break
        if not duplicate:
            kept.append(det)
    return kept


# -----------------------------
# Identity tracker
# -----------------------------

class FaceTracker:
    def __init__(
        self,
        sim_thresh: float = 0.45,
        iou_weight: float = 0.25,
        sim_weight: float = 0.65,
        dist_weight: float = 0.10,
        ttl: int = 90,
        archive_ttl: int = -1,
        reid_sim_thresh: Optional[float] = None,
        min_confirm_hits: int = 2,
        appearance_weight: float = 0.18,
        min_face_sim: float = 0.22,
        merge_sim_thresh: Optional[float] = None,
        young_track_hits: int = 8,
        new_id_confirm_hits: int = 5,
        new_id_confirm_quality: float = 0.42,
        provisional_match_margin: float = 0.04,
        high_det_score: float = 0.60,
        continuity_window: int = 3,
        continuity_iou_gate: float = 0.18,
        continuity_dist_gate: float = 0.32,
        continuity_relax: float = 0.10,
        continuity_bonus: float = 0.12,
        strong_named_match_score: float = 0.72,
        allow_new_persistent_identities: bool = True,
        full_reid_interval: int = 24,
    ) -> None:
        self.sim_thresh = sim_thresh
        self.iou_weight = iou_weight
        self.sim_weight = sim_weight
        self.dist_weight = dist_weight
        self.ttl = ttl
        self.archive_ttl = archive_ttl
        self.reid_sim_thresh = reid_sim_thresh if reid_sim_thresh is not None else max(0.55, sim_thresh + 0.08)
        self.min_confirm_hits = min_confirm_hits
        self.appearance_weight = appearance_weight
        self.min_face_sim = min_face_sim
        self.merge_sim_thresh = merge_sim_thresh if merge_sim_thresh is not None else max(self.reid_sim_thresh + 0.03, 0.62)
        self.young_track_hits = young_track_hits
        self.new_id_confirm_hits = max(new_id_confirm_hits, min_confirm_hits)
        self.new_id_confirm_quality = new_id_confirm_quality
        self.provisional_match_margin = provisional_match_margin
        self.high_det_score = high_det_score
        self.continuity_window = max(0, int(continuity_window))
        self.continuity_iou_gate = continuity_iou_gate
        self.continuity_dist_gate = continuity_dist_gate
        self.continuity_relax = continuity_relax
        self.continuity_bonus = continuity_bonus
        self.strong_named_match_score = max(float(strong_named_match_score), float(self.reid_sim_thresh))
        self.allow_new_persistent_identities = bool(allow_new_persistent_identities)
        self.full_reid_interval = max(1, int(full_reid_interval))

        self.next_track_id = 1
        self.next_temp_track_id = -1
        self.tracks: Dict[int, Track] = {}
        self.archived_tracks: Dict[int, Track] = {}
        self.identity_aliases: Dict[int, int] = {}
        self.last_step_observations: List[TrackObservation] = []
        self._current_probe_family = ""
        self._current_probe_bank_families: List[str] = []
        self.global_scene_changed = True
        self.global_scene_score = 1.0
        self.global_motion_stable = False
        self.current_frame_idx = 0

    def set_frame_context(
        self,
        *,
        frame_idx: int,
        scene_changed: bool,
        scene_score: float,
        motion_stable: bool,
    ) -> None:
        self.current_frame_idx = int(frame_idx)
        self.global_scene_changed = bool(scene_changed)
        self.global_scene_score = float(scene_score)
        self.global_motion_stable = bool(motion_stable)

    def resolve_track_id(self, track_id: int) -> int:
        current = int(track_id)
        visited: List[int] = []
        while current in self.identity_aliases and current not in visited:
            visited.append(current)
            current = int(self.identity_aliases[current])
        for item in visited:
            self.identity_aliases[item] = current
        return current

    def get_track_by_any_id(self, track_id: int) -> Optional[Track]:
        resolved_track_id = self.resolve_track_id(track_id)
        track = self.tracks.get(resolved_track_id)
        if track is not None:
            return track
        return self.archived_tracks.get(resolved_track_id)

    def _record_identity_alias(self, source_track_id: int, target_track_id: int) -> None:
        source_resolved = self.resolve_track_id(source_track_id)
        target_resolved = self.resolve_track_id(target_track_id)
        if source_resolved == target_resolved:
            return
        self.identity_aliases[source_resolved] = target_resolved

    @staticmethod
    def _merge_track_metadata(target_track: Track, source_track: Track) -> None:
        source_metadata = dict(source_track.metadata or {})
        if not source_metadata:
            return
        if not target_track.metadata:
            target_track.metadata = {}
        target_metadata = target_track.metadata
        for key, value in source_metadata.items():
            if value in (None, "", [], {}):
                continue
            if key in {"seed_sources", "source_images", "harvest_source_videos", "harvest_phases"}:
                current = list(target_metadata.get(key, []) or [])
                for item in list(value):
                    if item not in current:
                        current.append(item)
                target_metadata[key] = current
                continue
            if not target_metadata.get(key):
                target_metadata[key] = value

    @staticmethod
    def _track_has_named_identity(track: Optional[Track]) -> bool:
        if track is None:
            return False
        metadata = dict(track.metadata or {})
        student_name = str(metadata.get("name", "") or "").strip()
        student_key = str(metadata.get("student_key", "") or "").strip()
        return bool(student_name or student_key)

    @staticmethod
    def _detection_sample_metadata(det: Detection, frame_idx: int) -> Dict[str, object]:
        bbox = np.asarray(det.bbox, dtype=np.float32)
        quality_profile = dict(det.quality_profile or {})
        face_size = float(
            quality_profile.get(
                "face_size",
                max(0.0, min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))),
            )
        )
        primary_bank_family = preferred_runtime_bank_family(det.family_tags or [])
        augmentation_family = ""
        augmentation_tag = ""
        combination_tag = ""
        if "/" in primary_bank_family:
            augmentation_family, augmentation_tag = primary_bank_family.split("/", 1)
            if primary_bank_family.startswith("combo/"):
                combination_tag = augmentation_tag
        return sanitize_embedding_metadata(
            {
                "frame_idx": int(frame_idx),
                "quality": float(det.quality),
                "score": float(det.score),
                "face_size": face_size,
                "profile_bucket": infer_profile_bucket(bbox, det.landmarks),
                "size_bucket": infer_size_bucket(face_size),
                "lighting_bucket": str(quality_profile.get("lighting_bucket", "balanced_light")),
                "sharpness": float(quality_profile.get("sharpness", 0.0) or 0.0),
                "brightness": float(quality_profile.get("brightness", 0.0) or 0.0),
                "shadow_severity": float(quality_profile.get("shadow_severity", 0.0) or 0.0),
                "occlusion_ratio": float(quality_profile.get("occlusion_ratio", 0.0) or 0.0),
                "detector_used": str(det.detector_used or "scrfd"),
                "embedder_used": str(det.embedder_used or "arcface"),
                "generator_type": "runtime",
                "augmentation_family": augmentation_family,
                "augmentation_tag": augmentation_tag,
                "combination_tag": combination_tag,
                "bank_family": str(primary_bank_family or "base"),
                "source_kind": "runtime",
            },
            quality=float(det.quality),
            face_size=face_size,
        )

    @staticmethod
    def _build_observation(track: Track, det: Detection, frame_idx: int) -> TrackObservation:
        return TrackObservation(
            raw_track_id=int(track.track_id),
            frame_idx=int(frame_idx),
            bbox=np.asarray(det.bbox, dtype=np.float32).copy(),
            score=float(det.score),
            quality=float(det.quality),
            embedding=l2_normalize(np.asarray(det.embedding, dtype=np.float32)),
            landmarks=np.asarray(det.landmarks, dtype=np.float32).copy() if det.landmarks is not None else None,
            detector_used=str(det.detector_used or "scrfd"),
            embedder_used=str(det.embedder_used or "arcface"),
            quality_profile=dict(det.quality_profile or {}),
            assignment_mode=str(track.last_assignment_mode or "full_reid"),
            bank_family_used=str(track.last_bank_family_used or "base"),
            attribute_signature=str(track.last_attribute_signature or det.attribute_signature or "frontal"),
        )

    def _match_by_score_matrix(
        self,
        row_ids: List[int],
        col_ids: List[int],
        score_fn,
        invalid_score: float = -1e9,
    ) -> List[Tuple[float, int, int]]:
        if not row_ids or not col_ids:
            return []

        score_matrix = np.full((len(row_ids), len(col_ids)), invalid_score, dtype=np.float32)
        for row_idx, row_id in enumerate(row_ids):
            for col_idx, col_id in enumerate(col_ids):
                score_matrix[row_idx, col_idx] = float(score_fn(row_id, col_id))

        matches: List[Tuple[float, int, int]] = []
        if linear_sum_assignment is not None:
            cost_matrix = np.where(score_matrix <= invalid_score / 2.0, 1e6, -score_matrix)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
                score = float(score_matrix[row_idx, col_idx])
                if score <= invalid_score / 2.0:
                    continue
                matches.append((score, row_ids[row_idx], col_ids[col_idx]))
        else:
            candidate_pairs: List[Tuple[float, int, int]] = []
            for row_idx, row_id in enumerate(row_ids):
                for col_idx, col_id in enumerate(col_ids):
                    candidate_pairs.append((float(score_matrix[row_idx, col_idx]), row_id, col_id))

            candidate_pairs.sort(key=lambda item: item[0], reverse=True)
            consumed_rows = set()
            consumed_cols = set()
            for score, row_id, col_id in candidate_pairs:
                if score <= invalid_score / 2.0:
                    break
                if row_id in consumed_rows or col_id in consumed_cols:
                    continue
                matches.append((score, row_id, col_id))
                consumed_rows.add(row_id)
                consumed_cols.add(col_id)

        matches.sort(key=lambda item: item[0], reverse=True)
        return matches

    def load_identity_memory(self, identities: Dict[int, Track], next_track_id: Optional[int] = None) -> int:
        loaded = 0
        for track_id, track in identities.items():
            if track.avg_embedding is None:
                continue
            track.track_id = int(track_id)
            track.persistent_identity = True
            track.misses = 0
            self.archived_tracks[track.track_id] = track
            loaded += 1

        if next_track_id is not None:
            self.next_track_id = max(self.next_track_id, int(next_track_id))
        elif identities:
            self.next_track_id = max(self.next_track_id, max(int(tid) for tid in identities.keys()) + 1)
        return loaded

    def persistent_identity_tracks(self) -> Dict[int, Track]:
        tracks: Dict[int, Track] = {}
        for source in (self.archived_tracks, self.tracks):
            for track_id, track in source.items():
                if track_id <= 0:
                    continue
                if track.avg_embedding is None or track.hits < self.min_confirm_hits:
                    continue
                metadata = dict(track.metadata or {})
                student_name = str(metadata.get("name", "") or "").strip()
                student_key = str(metadata.get("student_key", "") or "").strip()
                if not student_name and not student_key:
                    continue
                track.persistent_identity = True
                tracks[track_id] = track
        return tracks

    def _allocate_provisional_track_id(self) -> int:
        track_id = self.next_temp_track_id
        self.next_temp_track_id -= 1
        return track_id

    def _group_summary_score(self, probe_embedding: np.ndarray, group_payload: Optional[Dict[str, object]]) -> float:
        if not group_payload:
            return -1.0
        global_payload = dict(group_payload.get("global", {}) or {})
        return grouped_bank_match_score(probe_embedding, global_payload)

    def _embedding_match_details(self, det: Detection, track: Track) -> Tuple[float, str]:
        full_score = bank_match_score(
            det.embedding,
            track.avg_embedding,
            track.embeddings,
            track.embedding_qualities,
            recent_embedding=track.recent_embedding,
            best_embedding=track.best_embedding,
        )
        if full_score < 0.0:
            return full_score, "global"

        probe_family = str(det.embedder_used or "arcface")
        candidate_families = prioritized_runtime_families(det.family_tags or [])
        groups = dict(track.embedding_groups or {})
        best_score = -1.0
        best_family = "global"

        for family in candidate_families:
            family_payload = groups.get(family)
            if not family_payload:
                continue
            per_embedder = dict(family_payload.get("per_embedder", {}) or {})
            if probe_family in per_embedder:
                score = grouped_bank_match_score(det.embedding, per_embedder.get(probe_family))
                if score > best_score:
                    best_score = score
                    best_family = family
            fallback = self._group_summary_score(det.embedding, family_payload)
            if fallback > best_score:
                best_score = fallback
                best_family = family

        if "base" in groups:
            base_payload = groups["base"]
            per_embedder = dict(base_payload.get("per_embedder", {}) or {})
            if probe_family in per_embedder:
                base_score = grouped_bank_match_score(det.embedding, per_embedder.get(probe_family))
                if base_score > best_score:
                    best_score = base_score
                    best_family = "base"
            base_fallback = self._group_summary_score(det.embedding, base_payload)
            if base_fallback > best_score:
                best_score = base_fallback
                best_family = "base"

        if best_score >= 0.0:
            return max(best_score, 0.96 * full_score), best_family
        return full_score, "global"

    def _appearance_match_score(self, det_desc: Optional[np.ndarray], track: Track) -> float:
        if det_desc is None or track.avg_appearance is None:
            return 0.0

        avg_sim = cosine_similarity(det_desc, track.avg_appearance)
        sims = [avg_sim]
        if track.recent_appearance is not None:
            sims.append(0.97 * cosine_similarity(det_desc, track.recent_appearance))
        if track.appearance_embeddings:
            bank_scores = []
            for idx, desc in enumerate(track.appearance_embeddings):
                sim = cosine_similarity(det_desc, desc)
                quality = 1.0
                if idx < len(track.appearance_qualities):
                    quality = max(0.05, float(track.appearance_qualities[idx]))
                bank_scores.append(sim * (0.85 + 0.15 * quality))
            if bank_scores:
                sims.append(max(bank_scores))
        if track.best_appearance is not None:
            sims.append(cosine_similarity(det_desc, track.best_appearance))

        best_bank = max(sims)
        return float(0.55 * best_bank + 0.45 * avg_sim)

    def _short_term_continuity(self, track: Track, iou: float, dist: float) -> bool:
        if track.hits < 2:
            return False
        if track.misses > self.continuity_window:
            return False
        if iou >= self.continuity_iou_gate:
            return True
        if dist <= self.continuity_dist_gate:
            return True
        return False

    def _continuity_geometry_score(self, det: Detection, track: Track) -> float:
        pred_box = track.predict_bbox()
        iou = iou_xyxy(det.bbox, pred_box)
        dist = normalized_center_distance(det.bbox, pred_box)
        dist_term = 1.0 - clamp(dist, 0.0, 1.5) / 1.5
        pred_diag = max(1.0, box_diag(pred_box))
        det_diag = max(1.0, box_diag(det.bbox))
        scale_delta = abs(math.log(det_diag / pred_diag))
        if iou < self.continuity_iou_gate and dist > self.continuity_dist_gate:
            return -1e9
        if scale_delta > 0.40:
            return -1e9
        if track.last_attribute_signature and det.attribute_signature and track.last_attribute_signature != det.attribute_signature:
            return -1e9
        return float(0.65 * iou + 0.35 * dist_term - 0.15 * scale_delta)

    def _identity_score(self, face_sim: float, appearance_sim: float) -> float:
        appearance_weight = self.appearance_weight if appearance_sim > 0.0 else 0.0
        face_weight = 1.0 - appearance_weight
        return float(face_weight * face_sim + appearance_weight * appearance_sim)

    def _association_score(self, det: Detection, track: Track) -> float:
        pred_box = track.predict_bbox()
        face_sim, _bank_family = self._embedding_match_details(det, track)
        appearance_sim = self._appearance_match_score(det.appearance, track)
        identity_sim = self._identity_score(face_sim, appearance_sim)
        iou = iou_xyxy(det.bbox, pred_box)
        dist = normalized_center_distance(det.bbox, pred_box)
        dist_term = 1.0 - clamp(dist, 0.0, 1.5) / 1.5
        continuity_mode = self._short_term_continuity(track, iou, dist)
        min_face_gate = self.min_face_sim - (self.continuity_relax if continuity_mode else 0.0)
        min_identity_gate = self.sim_thresh - (self.continuity_relax if continuity_mode else 0.0)

        # Hard gate on face similarity to avoid merging completely different students.
        if face_sim < min_face_gate or identity_sim < min_identity_gate:
            return -1e9

        # As a track gets stale, lean more on identity embedding and less on old geometry.
        stale_ratio = clamp(track.misses / max(1.0, float(self.ttl)), 0.0, 1.0)
        iou_weight = self.iou_weight * (1.0 - stale_ratio)
        dist_weight = self.dist_weight * (1.0 - 0.5 * stale_ratio)
        sim_weight = self.sim_weight + (self.iou_weight - iou_weight) + (self.dist_weight - dist_weight)

        score = (sim_weight * identity_sim) + (iou_weight * iou) + (dist_weight * dist_term)
        if continuity_mode:
            continuity_geom = max(iou, dist_term)
            score += self.continuity_bonus * continuity_geom
        return float(score)

    def _reid_score(self, det: Detection, track: Track) -> float:
        if track.hits < self.min_confirm_hits or track.avg_embedding is None:
            return -1e9

        face_sim, _bank_family = self._embedding_match_details(det, track)
        appearance_sim = self._appearance_match_score(det.appearance, track)
        identity_sim = self._identity_score(face_sim, appearance_sim)

        if face_sim < self.min_face_sim or identity_sim < self.reid_sim_thresh:
            return -1e9

        # Small tie-break toward more reliable long-lived identities.
        confidence_bonus = min(track.hits, 20) * 0.0025
        quality_bonus = 0.01 * max(track.best_embedding_quality, track.best_appearance_quality)
        return float(identity_sim + confidence_bonus + quality_bonus)

    def _track_reid_score(self, probe_track: Track, reference_track: Track, sim_thresh: Optional[float] = None) -> float:
        if probe_track.avg_embedding is None or reference_track.avg_embedding is None:
            return -1e9

        face_sim = bank_match_score(
            probe_track.avg_embedding,
            reference_track.avg_embedding,
            reference_track.embeddings,
            reference_track.embedding_qualities,
            recent_embedding=reference_track.recent_embedding,
            best_embedding=reference_track.best_embedding,
        )
        appearance_sim = self._appearance_match_score(probe_track.avg_appearance, reference_track)
        identity_sim = self._identity_score(face_sim, appearance_sim)
        effective_thresh = self.merge_sim_thresh if sim_thresh is None else sim_thresh
        if face_sim < self.min_face_sim or identity_sim < effective_thresh:
            return -1e9

        confidence_bonus = 0.0025 * min(reference_track.hits, 20)
        return float(identity_sim + confidence_bonus)

    def _create_track(self, det: Detection, frame_idx: int) -> Track:
        track = Track(
            track_id=self._allocate_provisional_track_id(),
            bbox=det.bbox.copy(),
            last_frame_idx=frame_idx,
            first_frame_idx=frame_idx,
            hits=1,
            misses=0,
            best_score=det.score,
            embeddings=[],
            avg_embedding=None,
        )
        track.update_embedding_bank(
            det.embedding,
            sample_quality=det.quality,
            sample_metadata=self._detection_sample_metadata(det, frame_idx),
        )
        track.update_appearance_bank(det.appearance, sample_quality=det.quality)
        track.last_assignment_mode = "full_reid"
        track.last_bank_family_used = "base"
        track.last_attribute_signature = str(det.attribute_signature or "frontal")
        if track.track_id > 0 and track.hits >= self.min_confirm_hits and track.avg_embedding is not None:
            track.persistent_identity = True
        self.tracks[track.track_id] = track
        return track

    def _update_track(
        self,
        track: Track,
        det: Detection,
        frame_idx: int,
        *,
        assignment_mode: str = "full_reid",
        bank_family_used: str = "base",
    ) -> None:
        pred_box = track.predict_bbox()
        iou = iou_xyxy(det.bbox, pred_box)
        dist = normalized_center_distance(det.bbox, pred_box)
        face_sim, _resolved_bank_family = self._embedding_match_details(det, track)
        continuity_mode = self._short_term_continuity(track, iou, dist)
        bank_quality = det.quality
        if continuity_mode and face_sim < self.sim_thresh:
            bank_quality *= 0.35

        new_velocity = det.bbox - track.bbox
        track.velocity = 0.7 * track.velocity + 0.3 * new_velocity
        track.bbox = det.bbox.copy()
        track.last_frame_idx = frame_idx
        track.hits += 1
        track.misses = 0
        track.best_score = max(track.best_score, det.score)
        track.update_embedding_bank(
            det.embedding,
            sample_quality=bank_quality,
            sample_metadata=self._detection_sample_metadata(det, frame_idx),
        )
        track.update_appearance_bank(det.appearance, sample_quality=bank_quality)
        track.last_assignment_mode = str(assignment_mode or "full_reid")
        track.last_bank_family_used = str(bank_family_used or _resolved_bank_family or "base")
        track.last_attribute_signature = str(det.attribute_signature or "frontal")
        if track.track_id > 0 and track.hits >= self.min_confirm_hits and track.avg_embedding is not None:
            track.persistent_identity = True

    def _archive_track(self, track_id: int) -> None:
        track = self.tracks.pop(track_id)
        if track.hits < self.min_confirm_hits or track.avg_embedding is None:
            return
        track.misses = 0
        track.persistent_identity = track.track_id > 0
        self.archived_tracks[track_id] = track

    def _restore_archived_track(self, track: Track, det: Detection, frame_idx: int) -> None:
        self.archived_tracks.pop(track.track_id, None)
        track.velocity = np.zeros(4, dtype=np.float32)
        track.bbox = det.bbox.copy()
        track.last_frame_idx = frame_idx
        track.misses = 0
        track.hits += 1
        track.best_score = max(track.best_score, det.score)
        track.update_embedding_bank(
            det.embedding,
            sample_quality=det.quality,
            sample_metadata=self._detection_sample_metadata(det, frame_idx),
        )
        track.update_appearance_bank(det.appearance, sample_quality=det.quality)
        track.last_assignment_mode = "full_reid"
        track.last_bank_family_used = "base"
        track.last_attribute_signature = str(det.attribute_signature or "frontal")
        if track.track_id > 0 and track.hits >= self.min_confirm_hits and track.avg_embedding is not None:
            track.persistent_identity = True
        self.tracks[track.track_id] = track

    def _prune_archived_tracks(self, frame_idx: int) -> None:
        to_delete = []
        transient_archive_ttl = self.archive_ttl if self.archive_ttl > 0 else self.ttl
        for tid, track in self.archived_tracks.items():
            if track.persistent_identity:
                continue
            if frame_idx - track.last_frame_idx > transient_archive_ttl:
                to_delete.append(tid)
        for tid in to_delete:
            del self.archived_tracks[tid]

    def _merge_track_into_archived_identity(self, young_track_id: int, archived_track_id: int, frame_idx: int) -> None:
        young_track = self.tracks.pop(young_track_id)
        restored_track = self.archived_tracks.pop(archived_track_id)
        self._record_identity_alias(young_track_id, archived_track_id)
        self._merge_track_metadata(restored_track, young_track)

        for idx, emb in enumerate(young_track.embeddings):
            sample_quality = (
                float(young_track.embedding_qualities[idx])
                if idx < len(young_track.embedding_qualities)
                else float(young_track.best_embedding_quality)
            )
            sample_metadata = young_track.embedding_metadata[idx] if idx < len(young_track.embedding_metadata) else None
            restored_track.update_embedding_bank(emb, sample_quality=sample_quality, sample_metadata=sample_metadata)
        for desc in young_track.appearance_embeddings:
            restored_track.update_appearance_bank(desc, sample_quality=young_track.best_appearance_quality)

        restored_track.bbox = young_track.bbox.copy()
        restored_track.velocity = young_track.velocity.copy()
        restored_track.last_frame_idx = frame_idx
        restored_track.misses = 0
        restored_track.hits += young_track.hits
        restored_track.best_score = max(restored_track.best_score, young_track.best_score)
        restored_track.persistent_identity = True
        self.tracks[archived_track_id] = restored_track

    def _merge_track_into_active_identity(self, young_track_id: int, active_track_id: int, frame_idx: int) -> None:
        young_track = self.tracks.pop(young_track_id)
        target_track = self.tracks.get(active_track_id)
        if target_track is None:
            return
        self._record_identity_alias(young_track_id, active_track_id)
        self._merge_track_metadata(target_track, young_track)

        for idx, emb in enumerate(young_track.embeddings):
            sample_quality = (
                float(young_track.embedding_qualities[idx])
                if idx < len(young_track.embedding_qualities)
                else float(young_track.best_embedding_quality)
            )
            sample_metadata = young_track.embedding_metadata[idx] if idx < len(young_track.embedding_metadata) else None
            target_track.update_embedding_bank(emb, sample_quality=sample_quality, sample_metadata=sample_metadata)
        for desc in young_track.appearance_embeddings:
            target_track.update_appearance_bank(desc, sample_quality=young_track.best_appearance_quality)

        if young_track.last_frame_idx >= target_track.last_frame_idx:
            target_track.bbox = young_track.bbox.copy()
            target_track.velocity = young_track.velocity.copy()
            target_track.last_frame_idx = frame_idx
            target_track.misses = 0
        target_track.hits += young_track.hits
        target_track.best_score = max(target_track.best_score, young_track.best_score)
        target_track.persistent_identity = True

    def _promote_track_to_new_identity(self, provisional_track_id: int) -> int:
        track = self.tracks.pop(provisional_track_id)
        new_track_id = self.next_track_id
        self.next_track_id += 1
        track.track_id = new_track_id
        track.persistent_identity = track.avg_embedding is not None and track.hits >= self.min_confirm_hits
        self.tracks[new_track_id] = track
        return new_track_id

    def _best_identity_candidate_for_track(self, probe: Track, frame_idx: int) -> Tuple[float, Optional[int], Optional[str], float]:
        candidate_entries: List[Tuple[int, str, Track]] = []
        for track_id, track in self.archived_tracks.items():
            if track_id <= 0 or track.avg_embedding is None:
                continue
            candidate_entries.append((track_id, "archived", track))
        for track_id, track in self.tracks.items():
            if track_id <= 0 or track.avg_embedding is None:
                continue
            if track.last_frame_idx == frame_idx:
                continue
            candidate_entries.append((track_id, "active", track))

        best_score = -1e9
        second_score = -1e9
        best_track_id: Optional[int] = None
        best_source: Optional[str] = None
        sim_thresh = max(self.reid_sim_thresh - 0.03, self.min_face_sim + 0.10)

        for track_id, source, track in candidate_entries:
            score = self._track_reid_score(probe, track, sim_thresh=sim_thresh)
            if score > best_score:
                second_score = best_score
                best_score = score
                best_track_id = track_id
                best_source = source
            elif score > second_score:
                second_score = score

        margin = best_score - second_score if second_score > -1e8 else 999.0
        return best_score, best_track_id, best_source, margin

    def _resolve_provisional_tracks(self, frame_idx: int) -> None:
        provisional_ids = [tid for tid, track in self.tracks.items() if tid < 0 and track.avg_embedding is not None]
        provisional_ids.sort(key=lambda tid: self.tracks[tid].hits, reverse=True)

        for provisional_id in provisional_ids:
            probe = self.tracks.get(provisional_id)
            if probe is None or probe.avg_embedding is None:
                continue

            best_score, best_track_id, best_source, margin = self._best_identity_candidate_for_track(probe, frame_idx)
            if best_track_id is not None and best_score > -1e8:
                target_track: Optional[Track] = None
                if best_source == "archived":
                    target_track = self.archived_tracks.get(best_track_id)
                elif best_source == "active":
                    target_track = self.tracks.get(best_track_id)

                strong_named_match = (
                    self._track_has_named_identity(target_track)
                    and probe.hits >= self.min_confirm_hits
                    and best_score >= self.strong_named_match_score
                )
                if margin >= self.provisional_match_margin or strong_named_match:
                    if best_source == "archived":
                        self._merge_track_into_archived_identity(provisional_id, best_track_id, frame_idx)
                    elif best_source == "active":
                        self._merge_track_into_active_identity(provisional_id, best_track_id, frame_idx)
                    continue

            if not self.allow_new_persistent_identities:
                continue

            if probe.hits < self.new_id_confirm_hits:
                continue
            if max(probe.best_embedding_quality, probe.best_appearance_quality) < self.new_id_confirm_quality:
                continue
            if best_score > -1e8 and best_score >= (self.reid_sim_thresh - 0.02):
                continue

            self._promote_track_to_new_identity(provisional_id)

    def _merge_young_tracks_into_archived_identities(self, frame_idx: int) -> None:
        if not self.archived_tracks:
            return

        young_track_ids = [tid for tid, track in self.tracks.items() if track.hits <= self.young_track_hits and track.avg_embedding is not None]
        if not young_track_ids:
            return

        archived_ids = list(self.archived_tracks.keys())
        matches = self._match_by_score_matrix(
            young_track_ids,
            archived_ids,
            lambda young_tid, archived_tid: self._track_reid_score(self.tracks[young_tid], self.archived_tracks[archived_tid]),
        )
        for score, young_tid, archived_tid in matches:
            if score < -1e8:
                continue
            if young_tid not in self.tracks or archived_tid not in self.archived_tracks:
                continue
            self._merge_track_into_archived_identity(young_tid, archived_tid, frame_idx)

    def step(self, detections: List[Detection], frame_idx: int) -> List[Track]:
        observations: List[TrackObservation] = []
        # Age unmatched tracks.
        for track in self.tracks.values():
            if track.last_frame_idx != frame_idx:
                track.misses = max(1, int(frame_idx - track.last_frame_idx))

        self._prune_archived_tracks(frame_idx)

        det_indices = list(range(len(detections)))
        high_det_indices = [di for di in det_indices if detections[di].score >= self.high_det_score]
        low_det_indices = [di for di in det_indices if di not in high_det_indices]
        track_ids = list(self.tracks.keys())
        matched_det_indices = set()
        matched_track_ids = set()

        force_full_reid = bool(self.global_scene_changed) or (int(frame_idx) % max(1, self.full_reid_interval) == 0)
        allow_continuity = (
            not force_full_reid
            and (self.global_motion_stable or self.global_scene_score <= 0.06)
            and bool(track_ids)
            and bool(det_indices)
        )

        if allow_continuity:
            continuity_matches = self._match_by_score_matrix(
                det_indices,
                track_ids,
                lambda di, tid: self._continuity_geometry_score(detections[di], self.tracks[tid]),
            )
            for score, di, tid in continuity_matches:
                if score < -1e8:
                    continue
                if di in matched_det_indices or tid in matched_track_ids:
                    continue
                continuity_family = str(self.tracks[tid].last_bank_family_used or "base")
                self._update_track(
                    self.tracks[tid],
                    detections[di],
                    frame_idx,
                    assignment_mode="continuity",
                    bank_family_used=continuity_family,
                )
                observations.append(self._build_observation(self.tracks[tid], detections[di], frame_idx))
                matched_det_indices.add(di)
                matched_track_ids.add(tid)

        high_track_ids = [tid for tid in track_ids if tid not in matched_track_ids]
        high_matches = self._match_by_score_matrix(
            [di for di in high_det_indices if di not in matched_det_indices],
            high_track_ids,
            lambda di, tid: self._association_score(detections[di], self.tracks[tid]),
        )
        for score, di, tid in high_matches:
            if score < -1e8:
                continue
            if di in matched_det_indices or tid in matched_track_ids:
                continue
            _score, bank_family_used = self._embedding_match_details(detections[di], self.tracks[tid])
            self._update_track(
                self.tracks[tid],
                detections[di],
                frame_idx,
                assignment_mode="full_reid",
                bank_family_used=bank_family_used,
            )
            observations.append(self._build_observation(self.tracks[tid], detections[di], frame_idx))
            matched_det_indices.add(di)
            matched_track_ids.add(tid)

        unmatched_track_ids = [tid for tid in self.tracks.keys() if tid not in matched_track_ids]
        if low_det_indices and unmatched_track_ids:
            low_matches = self._match_by_score_matrix(
                low_det_indices,
                unmatched_track_ids,
                lambda di, tid: self._association_score(detections[di], self.tracks[tid]) - 0.04 * (self.high_det_score - detections[di].score),
            )
            for score, di, tid in low_matches:
                if score < -1e8:
                    continue
                if di in matched_det_indices or tid in matched_track_ids:
                    continue
                _score, bank_family_used = self._embedding_match_details(detections[di], self.tracks[tid])
                self._update_track(
                    self.tracks[tid],
                    detections[di],
                    frame_idx,
                    assignment_mode="full_reid",
                    bank_family_used=bank_family_used,
                )
                observations.append(self._build_observation(self.tracks[tid], detections[di], frame_idx))
                matched_det_indices.add(di)
                matched_track_ids.add(tid)

        # Move dead active tracks into long-term identity memory before creating new IDs.
        to_archive = []
        for tid, track in self.tracks.items():
            if frame_idx - track.last_frame_idx > self.ttl:
                to_archive.append(tid)
        for tid in to_archive:
            self._archive_track(tid)

        # Re-identify unmatched detections against archived identities using face and context cues.
        archived_ids = list(self.archived_tracks.keys())
        unmatched_det_indices = [di for di in det_indices if di not in matched_det_indices]
        reid_matches = self._match_by_score_matrix(
            unmatched_det_indices,
            archived_ids,
            lambda di, tid: self._reid_score(detections[di], self.archived_tracks[tid]),
        )
        revived_track_ids = set()
        for score, di, tid in reid_matches:
            if score < -1e8:
                continue
            if di in matched_det_indices or tid in revived_track_ids or tid not in self.archived_tracks:
                continue
            self._restore_archived_track(self.archived_tracks[tid], detections[di], frame_idx)
            restored_track = self.tracks.get(tid)
            if restored_track is not None:
                _score, bank_family_used = self._embedding_match_details(detections[di], restored_track)
                restored_track.last_assignment_mode = "full_reid"
                restored_track.last_bank_family_used = str(bank_family_used or "global")
                restored_track.last_attribute_signature = str(detections[di].attribute_signature or "frontal")
                observations.append(self._build_observation(restored_track, detections[di], frame_idx))
            matched_det_indices.add(di)
            revived_track_ids.add(tid)

        # Create provisional tracks only when neither active matching nor re-id found a candidate.
        for di in det_indices:
            if di not in matched_det_indices:
                track = self._create_track(detections[di], frame_idx)
                observations.append(self._build_observation(track, detections[di], frame_idx))
                matched_track_ids.add(track.track_id)

        # Resolve provisional tracklets against existing identities before assigning
        # a brand new permanent ID. This avoids creating a fresh student ID from a
        # single weak re-entry frame.
        self._resolve_provisional_tracks(frame_idx)

        # Return active tracks visible in this frame.
        visible_tracks = [t for t in self.tracks.values() if t.last_frame_idx == frame_idx]
        visible_tracks.sort(key=lambda t: t.track_id)
        self.last_step_observations = observations
        return visible_tracks


# -----------------------------
# Face detector / embedder
# -----------------------------

class AdaFaceEmbedder:
    def __init__(self, weights_path: str) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is required for AdaFace inference.")
        self.weights_path = str(weights_path)
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"AdaFace weights not found: {self.weights_path}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        providers = [provider for provider in providers if provider in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.weights_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, aligned_rgb_112: np.ndarray) -> np.ndarray:
        crop = np.asarray(aligned_rgb_112, dtype=np.float32)
        if crop.shape[:2] != (112, 112):
            crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        tensor = crop.transpose(2, 0, 1)[None, ...].astype(np.float32)
        tensor = (tensor / 127.5) - 1.0
        outputs = self.session.run(None, {self.input_name: tensor})
        if not outputs:
            raise RuntimeError("AdaFace session returned no outputs.")
        emb = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        return l2_normalize(emb)


class RetinaFaceRecoveryDetector:
    def __init__(self) -> None:
        self._backend = None
        try:
            from retinaface import RetinaFace  # type: ignore
        except Exception:
            RetinaFace = None
        self._backend = RetinaFace

    @property
    def available(self) -> bool:
        return self._backend is not None

    def detect(self, frame_bgr: np.ndarray) -> list[dict[str, object]]:
        if self._backend is None:
            return []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            detections = self._backend.detect_faces(rgb)
        except Exception:
            return []
        if isinstance(detections, dict):
            items = detections.values()
        elif isinstance(detections, list):
            items = detections
        else:
            items = []
        recovered: list[dict[str, object]] = []
        for item in items:
            area = item.get("facial_area") if isinstance(item, dict) else None
            if area is None or len(area) != 4:
                continue
            bbox = np.asarray(area, dtype=np.float32)
            landmarks = None
            landmarks_dict = item.get("landmarks") if isinstance(item, dict) else None
            if isinstance(landmarks_dict, dict):
                landmark_keys = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
                if all(key in landmarks_dict for key in landmark_keys):
                    landmarks = np.asarray([landmarks_dict[key] for key in landmark_keys], dtype=np.float32)
            recovered.append(
                {
                    "bbox": bbox,
                    "score": float(item.get("score", 0.0) if isinstance(item, dict) else 0.0),
                    "landmarks": landmarks,
                }
            )
        return recovered

class InsightFaceBackend:
    def __init__(
        self,
        det_size: int = 960,
        ctx_id: int = 0,
        min_face: int = 20,
        det_thresh: float = 0.35,
        tile_grid: int = 1,
        tile_overlap: float = 0.20,
        primary_detector_name: str = "scrfd",
        backup_detector_name: str = "retinaface",
        enable_backup_detector: bool = True,
        adaface_weights: Optional[str] = None,
        attribute_classifier_weights: Optional[str] = None,
        attribute_classifier_config: Optional[str] = None,
        accessory_conf_threshold: float = 0.55,
    ) -> None:
        if FaceAnalysis is None:
            raise RuntimeError(
                "insightface is required for runtime face detection. "
                "Install with: pip install insightface onnxruntime "
                "(or onnxruntime-gpu for CUDA)."
            )
        providers = ["CPUExecutionProvider"] if ctx_id < 0 else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ctx_id >= 0 and ort is not None:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "GPU was requested with --ctx >= 0, but CUDAExecutionProvider is not available. "
                    "Install onnxruntime-gpu and ensure CUDA/cuDNN requirements are satisfied."
                )
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=(det_size, det_size))
        self.min_face = min_face
        self.tile_grid = max(1, int(tile_grid))
        self.tile_overlap = clamp(tile_overlap, 0.0, 0.45)
        self.primary_detector_name = str(primary_detector_name or "scrfd").strip() or "scrfd"
        self.backup_detector_name = str(backup_detector_name or "retinaface").strip() or "retinaface"
        self.enable_backup_detector = bool(enable_backup_detector)
        self.recognition_model = self.app.models.get("recognition")
        self.adaface_weights = str(adaface_weights or "").strip()
        self.adaface = None
        if self.adaface_weights:
            try:
                self.adaface = AdaFaceEmbedder(self.adaface_weights)
            except Exception as exc:
                print(f"Warning: AdaFace disabled ({exc})", file=sys.stderr)
                self.adaface = None
        self.retinaface = RetinaFaceRecoveryDetector() if self.enable_backup_detector else None
        self.attribute_classifier = LocalFaceAttributeClassifier(attribute_classifier_weights, attribute_classifier_config)
        self.accessory_conf_threshold = float(accessory_conf_threshold)

    def _faces_to_detections(
        self,
        frame_bgr: np.ndarray,
        faces,
        offset_x: int = 0,
        offset_y: int = 0,
        detector_used: Optional[str] = None,
    ) -> List[Detection]:
        detections: List[Detection] = []
        for face in faces:
            bbox = np.asarray(face.bbox, dtype=np.float32)
            bbox[[0, 2]] += float(offset_x)
            bbox[[1, 3]] += float(offset_y)
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            if min(w, h) < self.min_face:
                continue

            score = float(getattr(face, "det_score", 1.0))
            kps = np.asarray(getattr(face, "kps", None), dtype=np.float32) if getattr(face, "kps", None) is not None else None
            if kps is not None:
                kps[:, 0] += float(offset_x)
                kps[:, 1] += float(offset_y)
            appearance = extract_appearance_descriptor(frame_bgr, bbox)
            sharpness = estimate_crop_sharpness(frame_bgr, bbox)
            quality_profile = build_quality_profile(frame_bgr, bbox, score, kps, sharpness=sharpness)
            quality = float(quality_profile["quality"])
            emb, embedder_used = self._compute_embedding(frame_bgr, face, bbox, kps, quality_profile)
            attribute_crop = self._crop_face_region(frame_bgr, bbox)
            attribute_prediction = self.attribute_classifier.predict_with_context(
                attribute_crop,
                landmarks=kps,
                bbox=bbox,
                quality_profile=quality_profile,
            )
            family_tags = runtime_family_tags(
                quality_profile,
                attribute_prediction=attribute_prediction,
                accessory_threshold=self.accessory_conf_threshold,
            )
            detections.append(
                Detection(
                    bbox=bbox,
                    score=score,
                    embedding=emb,
                    landmarks=kps,
                    appearance=appearance,
                    quality=quality,
                    detector_used=str(detector_used or self.primary_detector_name),
                    embedder_used=embedder_used,
                    quality_profile=quality_profile,
                    attribute_prediction=attribute_prediction,
                    family_tags=family_tags,
                    attribute_signature=attribute_prediction.signature(threshold=self.accessory_conf_threshold),
                )
            )
        return detections

    def _align_face(self, frame_bgr: np.ndarray, bbox: np.ndarray, landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if face_align is not None and landmarks is not None and len(landmarks) >= 5:
            try:
                aligned = face_align.norm_crop(rgb, landmark=np.asarray(landmarks[:5], dtype=np.float32), image_size=112)
                return np.asarray(aligned, dtype=np.uint8)
            except Exception:
                pass
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_bgr.shape[1], x2)
        y2 = min(frame_bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

    def _crop_face_region(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_bgr.shape[1], x2)
        y2 = min(frame_bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

    def _compute_embedding(
        self,
        frame_bgr: np.ndarray,
        face: Any,
        bbox: np.ndarray,
        landmarks: Optional[np.ndarray],
        quality_profile: Dict[str, object],
    ) -> tuple[np.ndarray, str]:
        arcface_embedding = np.asarray(face.normed_embedding, dtype=np.float32)
        arcface_embedding = l2_normalize(arcface_embedding)
        if self.adaface is None or not prefer_low_quality_embedder(quality_profile):
            return arcface_embedding, "arcface"
        aligned = self._align_face(frame_bgr, bbox, landmarks)
        if aligned is None:
            return arcface_embedding, "arcface_fallback"
        try:
            return self.adaface.embed(aligned), "adaface"
        except Exception:
            return arcface_embedding, "arcface_fallback"

    def _backup_faces(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.retinaface is None or not self.retinaface.available:
            return []
        recovered_faces = self.retinaface.detect(frame_bgr)
        detections: List[Detection] = []
        for item in recovered_faces:
            bbox = np.asarray(item.get("bbox"), dtype=np.float32)
            score = float(item.get("score", 0.0))
            landmarks = item.get("landmarks")
            quality_profile = build_quality_profile(frame_bgr, bbox, score, landmarks, sharpness=estimate_crop_sharpness(frame_bgr, bbox))
            aligned = self._align_face(frame_bgr, bbox, landmarks)
            if aligned is None or self.recognition_model is None:
                continue
            try:
                embedding = self.recognition_model.get_feat(np.asarray([aligned], dtype=np.uint8))[0]
                embedding = l2_normalize(np.asarray(embedding, dtype=np.float32))
            except Exception:
                continue
            embedder_used = "arcface"
            if self.adaface is not None and prefer_low_quality_embedder(quality_profile):
                try:
                    embedding = self.adaface.embed(aligned)
                    embedder_used = "adaface"
                except Exception:
                    embedder_used = "arcface_fallback"
            attribute_crop = self._crop_face_region(frame_bgr, bbox)
            attribute_prediction = self.attribute_classifier.predict_with_context(
                attribute_crop,
                landmarks=landmarks,
                bbox=bbox,
                quality_profile=quality_profile,
            )
            family_tags = runtime_family_tags(
                quality_profile,
                attribute_prediction=attribute_prediction,
                accessory_threshold=self.accessory_conf_threshold,
            )
            detections.append(
                Detection(
                    bbox=bbox,
                    score=score,
                    embedding=embedding,
                    landmarks=np.asarray(landmarks, dtype=np.float32) if landmarks is not None else None,
                    appearance=extract_appearance_descriptor(frame_bgr, bbox),
                    quality=float(quality_profile["quality"]),
                    detector_used=self.backup_detector_name,
                    embedder_used=embedder_used,
                    quality_profile=quality_profile,
                    attribute_prediction=attribute_prediction,
                    family_tags=family_tags,
                    attribute_signature=attribute_prediction.signature(threshold=self.accessory_conf_threshold),
                )
            )
        return detections

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        detections = self._faces_to_detections(
            frame_bgr,
            self.app.get(frame_bgr),
            detector_used=self.primary_detector_name,
        )

        if self.tile_grid > 1:
            tiles = generate_overlapping_tiles(frame_bgr.shape, self.tile_grid, self.tile_overlap)
            for x1, y1, x2, y2 in tiles:
                if x1 == 0 and y1 == 0 and x2 == frame_bgr.shape[1] and y2 == frame_bgr.shape[0]:
                    continue
                tile = frame_bgr[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                detections.extend(
                    self._faces_to_detections(
                        frame_bgr,
                        self.app.get(tile),
                        offset_x=x1,
                        offset_y=y1,
                        detector_used=self.primary_detector_name,
                    )
                )

        should_try_backup = self.enable_backup_detector and (
            not detections or any(float(det.quality) < 0.46 for det in detections)
        )
        if should_try_backup:
            detections.extend(self._backup_faces(frame_bgr))

        return deduplicate_detections(detections)


# -----------------------------
# Persistent identity database
# -----------------------------

class FaceIdentityDB:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    @staticmethod
    def _to_array(values: Optional[List[float]]) -> Optional[np.ndarray]:
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return None
        return l2_normalize(arr)

    @staticmethod
    def _to_array_bank(values: Optional[List[List[float]]]) -> List[np.ndarray]:
        if not values:
            return []
        bank = []
        for item in values:
            arr = FaceIdentityDB._to_array(item)
            if arr is not None:
                bank.append(arr)
        return bank

    @staticmethod
    def _to_list(arr: Optional[np.ndarray]) -> Optional[List[float]]:
        if arr is None:
            return None
        return np.asarray(arr, dtype=np.float32).tolist()

    @staticmethod
    def _to_list_bank(bank: List[np.ndarray]) -> List[List[float]]:
        return [np.asarray(item, dtype=np.float32).tolist() for item in bank]

    @staticmethod
    def _build_average(bank: List[np.ndarray]) -> Optional[np.ndarray]:
        if not bank:
            return None
        stacked = np.vstack(bank)
        return l2_normalize(np.mean(stacked, axis=0).astype(np.float32))

    def _group_summaries_from_record(self, payload: Optional[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
        if not isinstance(payload, dict):
            return {}
        groups: Dict[str, Dict[str, object]] = {}
        for family, family_payload in payload.items():
            if not isinstance(family_payload, dict):
                continue
            global_payload = dict(family_payload.get("global", {}) or {})
            per_embedder_payload = dict(family_payload.get("per_embedder", {}) or {})
            groups[str(family)] = {
                "count": int(family_payload.get("count", 0) or 0),
                "global": {
                    "count": int(global_payload.get("count", 0) or 0),
                    "avg_embedding": self._to_array(global_payload.get("avg_embedding")),
                    "best_embedding": self._to_array(global_payload.get("best_embedding")),
                    "recent_embedding": self._to_array(global_payload.get("recent_embedding")),
                    "best_quality": float(global_payload.get("best_quality", 0.0) or 0.0),
                },
                "per_embedder": {
                    str(embedder): {
                        "count": int(item.get("count", 0) or 0),
                        "avg_embedding": self._to_array(item.get("avg_embedding")),
                        "best_embedding": self._to_array(item.get("best_embedding")),
                        "recent_embedding": self._to_array(item.get("recent_embedding")),
                        "best_quality": float(item.get("best_quality", 0.0) or 0.0),
                    }
                    for embedder, item in per_embedder_payload.items()
                    if isinstance(item, dict)
                },
            }
        return groups

    def _group_summaries_to_record(self, payload: Dict[str, Dict[str, object]]) -> Dict[str, object]:
        record: Dict[str, object] = {}
        for family, family_payload in dict(payload or {}).items():
            global_payload = dict(family_payload.get("global", {}) or {})
            per_embedder_payload = dict(family_payload.get("per_embedder", {}) or {})
            record[str(family)] = {
                "count": int(family_payload.get("count", 0) or 0),
                "global": {
                    "count": int(global_payload.get("count", 0) or 0),
                    "avg_embedding": self._to_list(global_payload.get("avg_embedding")),
                    "best_embedding": self._to_list(global_payload.get("best_embedding")),
                    "recent_embedding": self._to_list(global_payload.get("recent_embedding")),
                    "best_quality": float(global_payload.get("best_quality", 0.0) or 0.0),
                },
                "per_embedder": {
                    str(embedder): {
                        "count": int(item.get("count", 0) or 0),
                        "avg_embedding": self._to_list(item.get("avg_embedding")),
                        "best_embedding": self._to_list(item.get("best_embedding")),
                        "recent_embedding": self._to_list(item.get("recent_embedding")),
                        "best_quality": float(item.get("best_quality", 0.0) or 0.0),
                    }
                    for embedder, item in per_embedder_payload.items()
                    if isinstance(item, dict)
                },
            }
        return record

    def _record_to_track(self, record: Dict[str, object]) -> Optional[Track]:
        track_id = int(record.get("track_id", 0))
        if track_id <= 0:
            return None

        embeddings = self._to_array_bank(record.get("embeddings"))
        appearance_embeddings = self._to_array_bank(record.get("appearance_embeddings"))
        embedding_qualities = [float(v) for v in record.get("embedding_qualities", [])]
        appearance_qualities = [float(v) for v in record.get("appearance_qualities", [])]
        embedding_metadata = record.get("embedding_metadata", [])
        if not isinstance(embedding_metadata, list):
            embedding_metadata = []
        while len(embedding_qualities) < len(embeddings):
            embedding_qualities.append(float(record.get("best_embedding_quality", 0.5)))
        while len(appearance_qualities) < len(appearance_embeddings):
            appearance_qualities.append(float(record.get("best_appearance_quality", 0.5)))
        while len(embedding_metadata) < len(embeddings):
            quality = float(embedding_qualities[len(embedding_metadata)]) if len(embedding_metadata) < len(embedding_qualities) else float(record.get("best_embedding_quality", 0.5))
            embedding_metadata.append(sanitize_embedding_metadata(quality=quality, added_at=current_utc_iso()))
        avg_embedding = self._to_array(record.get("avg_embedding"))
        avg_appearance = self._to_array(record.get("avg_appearance"))
        if avg_embedding is None:
            avg_embedding = weighted_average_embeddings(embeddings, embedding_qualities)
        if avg_appearance is None:
            avg_appearance = weighted_average_embeddings(appearance_embeddings, appearance_qualities)
        if avg_embedding is None:
            return None

        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = dict(metadata)
        for legacy_key in ("name", "roll_number", "student_key"):
            if legacy_key in record and legacy_key not in metadata:
                metadata[legacy_key] = record.get(legacy_key)

        track = Track(
            track_id=track_id,
            bbox=np.zeros(4, dtype=np.float32),
            last_frame_idx=int(record.get("last_frame_idx", 0)),
            first_frame_idx=int(record.get("first_frame_idx", 0)),
            hits=max(1, int(record.get("hits", 1))),
            misses=0,
            best_score=float(record.get("best_score", 0.0)),
            embeddings=embeddings,
            embedding_qualities=embedding_qualities,
            embedding_metadata=[sanitize_embedding_metadata(item, quality=embedding_qualities[idx] if idx < len(embedding_qualities) else 0.5) for idx, item in enumerate(embedding_metadata[: len(embeddings)])],
            avg_embedding=avg_embedding,
            best_embedding=self._to_array(record.get("best_embedding")),
            best_embedding_quality=float(record.get("best_embedding_quality", 0.0)),
            recent_embedding=self._to_array(record.get("recent_embedding")),
            embedding_groups=self._group_summaries_from_record(record.get("embedding_groups")),
            appearance_embeddings=appearance_embeddings,
            appearance_qualities=appearance_qualities,
            avg_appearance=avg_appearance,
            best_appearance=self._to_array(record.get("best_appearance")),
            best_appearance_quality=float(record.get("best_appearance_quality", 0.0)),
            recent_appearance=self._to_array(record.get("recent_appearance")),
            persistent_identity=True,
            metadata=metadata,
            last_assignment_mode=str(record.get("last_assignment_mode", "full_reid") or "full_reid"),
            last_bank_family_used=str(record.get("last_bank_family_used", "base") or "base"),
            last_attribute_signature=str(record.get("last_attribute_signature", "frontal") or "frontal"),
        )
        if track.best_embedding is None or track.recent_embedding is None:
            avg_embedding, best_embedding, best_embedding_quality, recent_embedding = summarize_embedding_bank(
                track.embeddings,
                track.embedding_qualities,
                track.embedding_metadata,
            )
            track.avg_embedding = avg_embedding
            track.best_embedding = best_embedding
            track.best_embedding_quality = max(float(track.best_embedding_quality), float(best_embedding_quality))
            track.recent_embedding = recent_embedding
        if not track.embedding_groups:
            track.embedding_groups = build_grouped_embedding_summaries(
                track.embeddings,
                track.embedding_qualities,
                track.embedding_metadata,
            )
        return track

    def _track_to_record(self, track: Track) -> Dict[str, object]:
        return {
            "track_id": int(track.track_id),
            "hits": int(track.hits),
            "first_frame_idx": int(track.first_frame_idx),
            "last_frame_idx": int(track.last_frame_idx),
            "best_score": float(track.best_score),
            "embeddings": self._to_list_bank(track.embeddings),
            "embedding_qualities": [float(v) for v in track.embedding_qualities],
            "embedding_metadata": [dict(item) for item in track.embedding_metadata],
            "avg_embedding": self._to_list(track.avg_embedding),
            "best_embedding": self._to_list(track.best_embedding),
            "best_embedding_quality": float(track.best_embedding_quality),
            "recent_embedding": self._to_list(track.recent_embedding),
            "embedding_groups": self._group_summaries_to_record(track.embedding_groups),
            "appearance_embeddings": self._to_list_bank(track.appearance_embeddings),
            "appearance_qualities": [float(v) for v in track.appearance_qualities],
            "avg_appearance": self._to_list(track.avg_appearance),
            "best_appearance": self._to_list(track.best_appearance),
            "best_appearance_quality": float(track.best_appearance_quality),
            "recent_appearance": self._to_list(track.recent_appearance),
            "metadata": dict(track.metadata or {}),
            "last_assignment_mode": str(track.last_assignment_mode or "full_reid"),
            "last_bank_family_used": str(track.last_bank_family_used or "base"),
            "last_attribute_signature": str(track.last_attribute_signature or "frontal"),
        }

    def load(self) -> Tuple[int, Dict[int, Track]]:
        if not self.db_path or not os.path.exists(self.db_path):
            return 1, {}

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"Warning: could not load identity DB at {self.db_path}: {exc}", file=sys.stderr)
            return 1, {}

        records = payload.get("identities", [])
        identities: Dict[int, Track] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            track = self._record_to_track(record)
            if track is None:
                continue
            identities[track.track_id] = track

        next_track_id = int(payload.get("next_track_id", 1))
        next_track_id = max(next_track_id, max(identities.keys(), default=0) + 1)
        return next_track_id, identities

    def save(self, tracker: FaceTracker) -> int:
        if not self.db_path:
            return 0

        identities = tracker.persistent_identity_tracks()
        payload = {
            "version": 2,
            "next_track_id": int(tracker.next_track_id),
            "identities": [self._track_to_record(identities[track_id]) for track_id in sorted(identities.keys())],
        }

        ensure_parent_dir(self.db_path)
        tmp_path = f"{self.db_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, self.db_path)
        return len(payload["identities"])


# -----------------------------
# Rendering + CSV writing
# -----------------------------

def draw_track(frame: np.ndarray, track: Track) -> None:
    draw_track_annotation(frame, track.bbox, track.track_id, track.hits, track.metadata)


def draw_track_annotation(
    frame: np.ndarray,
    bbox: np.ndarray,
    track_id: int,
    hits: int,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32).astype(int)
    color = color_from_id(track_id)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    resolved_metadata = dict(metadata or {})
    student_name = str(resolved_metadata.get("name", "") or "").strip()
    label = student_name if student_name else "Unknown"

    if not student_name and (hits < 2 or track_id <= 0):
        label += " ?"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_y1 = max(0, y1 - th - 8)
    text_y2 = max(th + 8, y1)
    cv2.rectangle(frame, (x1, text_y1), (x1 + tw + 8, text_y2), color, -1)
    cv2.putText(frame, label, (x1 + 4, text_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


class CsvLogger:
    def __init__(self, csv_path: Optional[str]) -> None:
        self.csv_path = csv_path
        self.file = None
        self.writer = None
        if csv_path:
            ensure_parent_dir(csv_path)
            self.file = open(csv_path, "w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            self.writer.writerow([
                "frame_idx",
                "student_id",
                "display_name",
                "student_name",
                "roll_number",
                "student_key",
                "x1",
                "y1",
                "x2",
                "y2",
                "hits",
                "first_frame_idx",
                "last_frame_idx",
                "assignment_mode",
                "bank_family_used",
                "attribute_signature",
            ])

    def log(self, frame_idx: int, tracks: List[Track]) -> None:
        if not self.writer:
            return
        for t in tracks:
            x1, y1, x2, y2 = t.bbox.tolist()
            self.writer.writerow([
                frame_idx,
                t.track_id,
                "",
                "",
                "",
                "",
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{x2:.2f}",
                f"{y2:.2f}",
                t.hits,
                t.first_frame_idx,
                t.last_frame_idx,
                str(getattr(t, "last_assignment_mode", "full_reid") or "full_reid"),
                str(getattr(t, "last_bank_family_used", "base") or "base"),
                str(getattr(t, "last_attribute_signature", "frontal") or "frontal"),
            ])

    def log_annotations(self, frame_idx: int, annotations: List[Dict[str, object]]) -> None:
        if not self.writer:
            return
        for item in annotations:
            bbox = np.asarray(item.get("bbox"), dtype=np.float32).tolist()
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            metadata = dict(item.get("metadata") or {})
            display_name = str(metadata.get("name", "") or "").strip() or "Unknown"
            self.writer.writerow([
                frame_idx,
                int(item.get("student_id", 0)),
                display_name,
                str(metadata.get("name", "") or "").strip(),
                str(metadata.get("roll_number", "") or "").strip(),
                str(metadata.get("student_key", "") or "").strip(),
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{x2:.2f}",
                f"{y2:.2f}",
                int(item.get("hits", 0)),
                int(item.get("first_frame_idx", frame_idx)),
                int(item.get("last_frame_idx", frame_idx)),
                str(item.get("assignment_mode", "") or "full_reid"),
                str(item.get("bank_family_used", "") or "base"),
                str(item.get("attribute_signature", "") or "frontal"),
            ])

    def close(self) -> None:
        if self.file:
            self.file.close()


def build_frame_annotation(tracker: FaceTracker, frame_idx: int, visible_tracks: List[Track]) -> Dict[str, object]:
    active_identity_count = sum(1 for track in tracker.tracks.values() if track.track_id > 0)
    observation_by_track = {
        int(observation.raw_track_id): observation
        for observation in list(tracker.last_step_observations or [])
        if int(observation.frame_idx) == int(frame_idx)
    }
    annotations: List[Dict[str, object]] = []
    for track in visible_tracks:
        observation = observation_by_track.get(int(track.track_id))
        annotations.append(
            {
                "raw_track_id": int(track.track_id),
                "bbox": np.asarray(track.bbox, dtype=np.float32).tolist(),
                "hits": int(track.hits),
                "first_frame_idx": int(track.first_frame_idx),
                "last_frame_idx": int(track.last_frame_idx),
                "assignment_mode": str(
                    (observation.assignment_mode if observation is not None else track.last_assignment_mode) or "full_reid"
                ),
                "bank_family_used": str(
                    (observation.bank_family_used if observation is not None else track.last_bank_family_used) or "base"
                ),
                "attribute_signature": str(
                    (observation.attribute_signature if observation is not None else track.last_attribute_signature) or "frontal"
                ),
            }
        )
    return {
        "frame_idx": int(frame_idx),
        "visible_count": len(visible_tracks),
        "active_identity_count": active_identity_count,
        "tracks": annotations,
    }


def resolved_frame_annotations(tracker: FaceTracker, frame_payload: Dict[str, object]) -> List[Dict[str, object]]:
    resolved_items: List[Dict[str, object]] = []
    for item in list(frame_payload.get("tracks", [])):
        raw_track_id = int(item.get("raw_track_id", 0))
        resolved_track_id = tracker.resolve_track_id(raw_track_id)
        resolved_track = tracker.get_track_by_any_id(raw_track_id)
        metadata = dict(resolved_track.metadata or {}) if resolved_track is not None else {}
        resolved_items.append(
            {
                "raw_track_id": raw_track_id,
                "student_id": resolved_track_id,
                "bbox": item.get("bbox", []),
                "hits": int(item.get("hits", 0)),
                "first_frame_idx": int(item.get("first_frame_idx", frame_payload.get("frame_idx", 0))),
                "last_frame_idx": int(item.get("last_frame_idx", frame_payload.get("frame_idx", 0))),
                "assignment_mode": str(item.get("assignment_mode", "") or "full_reid"),
                "bank_family_used": str(item.get("bank_family_used", "") or "base"),
                "attribute_signature": str(item.get("attribute_signature", "") or "frontal"),
                "metadata": metadata,
            }
        )
    return resolved_items


def build_reportable_identity_set(tracker: FaceTracker, roster_limit: int) -> set[int]:
    named_ids = sorted(tracker.persistent_identity_tracks().keys())
    unnamed_candidates: list[tuple[str, float]] = []
    for source in (tracker.tracks, tracker.archived_tracks):
        for track_id, track in source.items():
            if int(track_id) > 0:
                continue
            if FaceTracker._track_has_named_identity(track):
                continue
            if getattr(track, "hits", 0) < max(2, tracker.min_confirm_hits):
                continue
            unnamed_candidates.append((f"TEMP_{abs(int(track_id)):04d}", cluster_support_score(track)))

    selected_strings = capped_reportable_ids(
        [f"STU_{track_id:03d}" for track_id in named_ids],
        unnamed_candidates,
        roster_limit=max(0, int(roster_limit)),
    )
    selected_ids: set[int] = set()
    for item in selected_strings:
        text = str(item)
        if text.startswith("STU_"):
            try:
                selected_ids.add(int(text.split("_", 1)[1]))
            except Exception:
                continue
        elif text.startswith("TEMP_"):
            try:
                selected_ids.add(-int(text.split("_", 1)[1]))
            except Exception:
                continue
    return selected_ids


def export_unknown_cluster_package(
    tracker: FaceTracker,
    unknown_output_dir: str,
    input_video: str,
    reportable_ids: set[int],
) -> tuple[Path, Path, Path]:
    output_dir = Path(unknown_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unknown_crops_dir = output_dir / "unknown_crops"
    unknown_crops_dir.mkdir(parents=True, exist_ok=True)

    cluster_rows: list[dict[str, object]] = []
    for source in (tracker.tracks, tracker.archived_tracks):
        for track_id, track in source.items():
            if int(track_id) >= 0 or int(track_id) not in reportable_ids:
                continue
            if FaceTracker._track_has_named_identity(track):
                continue
            row = build_unknown_cluster_record(track_id, track)
            row["crop_path"] = ""
            cluster_rows.append(row)
    cluster_rows.sort(key=lambda item: (float(item.get("support_score", 0.0)), str(item.get("cluster_id", ""))), reverse=True)
    csv_path, json_path, assignments_path = write_unknown_review_package(output_dir, cluster_rows)
    if not cluster_rows:
        return csv_path, json_path, assignments_path

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        return csv_path, json_path, assignments_path
    try:
        rows_by_frame: dict[int, list[dict[str, object]]] = {}
        for row in cluster_rows:
            frame_idx = int(row.get("representative_frame_idx", 0) or 0)
            rows_by_frame.setdefault(frame_idx, []).append(row)
        current_frame_idx = 0
        for target_frame_idx in sorted(rows_by_frame.keys()):
            while current_frame_idx <= target_frame_idx:
                ok, frame = cap.read()
                if not ok:
                    return csv_path, json_path, assignments_path
                if current_frame_idx == target_frame_idx:
                    for row in rows_by_frame[target_frame_idx]:
                        bbox = np.asarray(row.get("bbox", []), dtype=np.float32)
                        if bbox.size != 4:
                            continue
                        crop = bbox_to_crop(frame, bbox)
                        if crop.size == 0:
                            continue
                        crop_path = unknown_crops_dir / f"{row['cluster_id']}_frame{target_frame_idx}.jpg"
                        cv2.imwrite(str(crop_path), crop)
                        row["crop_path"] = crop_path.as_posix()
                current_frame_idx += 1
    finally:
        cap.release()

    write_unknown_review_package(output_dir, cluster_rows)
    return csv_path, json_path, assignments_path


# -----------------------------
# Main processing loop
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classroom face detection + student ID tracking")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output annotated video")
    parser.add_argument("--csv", default=None, help="Optional CSV path to save detections")
    parser.add_argument("--det-size", type=int, default=960, help="InsightFace detection size, e.g. 640/960/1280")
    parser.add_argument("--primary-detector", default="scrfd", help="Primary detector label used in metadata.")
    parser.add_argument("--backup-detector", default="retinaface", help="Backup detector label used in metadata.")
    parser.add_argument(
        "--disable-backup-detector",
        action="store_true",
        help="Disable the harder-case recovery detector pass.",
    )
    parser.add_argument(
        "--adaface-weights",
        default="",
        help="Optional AdaFace ONNX weights for low-quality face embeddings.",
    )
    parser.add_argument(
        "--attribute-classifier-weights",
        default="",
        help="Optional local ONNX weights for runtime face pose/accessory classification.",
    )
    parser.add_argument(
        "--attribute-classifier-config",
        default="",
        help="Optional JSON config for the runtime face attribute classifier.",
    )
    parser.add_argument(
        "--accessory-conf-threshold",
        type=float,
        default=0.55,
        help="Confidence threshold for accessory-aware routing.",
    )
    parser.add_argument(
        "--process-fps",
        type=float,
        default=-1.0,
        help="Sample the input video at this many frames per second for detection/tracking. Use <= 0 to process every frame.",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.35,
        help="Detector score threshold passed into InsightFace. Lower values keep weaker detections for recovery.",
    )
    parser.add_argument(
        "--tile-grid",
        type=int,
        default=1,
        help="Optional tiled second-pass detection grid. Use 2 or 3 to improve recall on small/far faces.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.20,
        help="Tile overlap ratio used when --tile-grid > 1.",
    )
    parser.add_argument("--ctx", type=int, default=0, help="GPU device id. Use -1 for CPU")
    parser.add_argument("--min-face", type=int, default=20, help="Ignore faces smaller than this many pixels")
    parser.add_argument("--sim-thresh", type=float, default=0.45, help="Cosine similarity gate for same-student matching")
    parser.add_argument("--ttl", type=int, default=90, help="How many frames a missing student track is kept alive")
    parser.add_argument(
        "--archive-ttl",
        type=int,
        default=-1,
        help="How many frames expired identities are kept for re-identification. Use <= 0 to keep them forever.",
    )
    parser.add_argument("--reid-sim-thresh", type=float, default=None, help="Stricter cosine threshold to revive an expired identity")
    parser.add_argument(
        "--new-id-confirm-hits",
        type=int,
        default=5,
        help="How many consecutive hits a provisional track needs before it can receive a brand new permanent ID.",
    )
    parser.add_argument(
        "--new-id-confirm-quality",
        type=float,
        default=0.42,
        help="Minimum best sample quality required before a provisional track becomes a brand new permanent ID.",
    )
    parser.add_argument(
        "--provisional-match-margin",
        type=float,
        default=0.04,
        help="Required score margin between the best and second-best identity candidates before a provisional track is merged.",
    )
    parser.add_argument(
        "--high-det-score",
        type=float,
        default=0.60,
        help="High-confidence detection threshold for the first association pass. Lower-score detections are used in a second recovery pass.",
    )
    parser.add_argument(
        "--identity-db",
        default=DEFAULT_IDENTITY_DB_PATH,
        help="Path to the persistent cross-video identity database. Defaults to detectors/face_detector/identity_db.json",
    )
    parser.add_argument(
        "--identity-db-save-every",
        type=int,
        default=150,
        help="Save the identity database every N frames during processing. Use <= 0 to save only at shutdown.",
    )
    parser.add_argument(
        "--allow-new-identities",
        action="store_true",
        help="Allow provisional tracks to become brand new permanent identities. Disabled by default for roster-only runs.",
    )
    parser.add_argument(
        "--student-details-root",
        default=str(Path(PROJECT_ROOT) / "student-details"),
        help="Roster root used to cap total reportable identities.",
    )
    parser.add_argument(
        "--roster-limit",
        type=int,
        default=0,
        help="Explicit cap for total reportable identities. <= 0 derives it from student-details.",
    )
    parser.add_argument(
        "--unknown-output-dir",
        default="",
        help="Optional directory to export unknown clusters, crops, and assignments. Defaults near the output video.",
    )
    parser.add_argument(
        "--scene-change-thresh",
        type=float,
        default=0.14,
        help="Scene change threshold for continuity-vs-reid gating.",
    )
    parser.add_argument(
        "--scene-unstable-inlier-ratio",
        type=float,
        default=0.28,
        help="Minimum ORB/RANSAC inlier ratio to treat motion as stable.",
    )
    parser.add_argument(
        "--scene-downsample-width",
        type=int,
        default=160,
        help="Downsample width used for the scene change gate.",
    )
    parser.add_argument(
        "--full-reid-interval",
        type=int,
        default=24,
        help="Force a full re-identification refresh every N processed frames.",
    )
    parser.add_argument("--max-frames", type=int, default=-1, help="Optional cap for debugging")
    parser.add_argument("--display", action="store_true", help="Show a live preview window while processing")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")

    backend = InsightFaceBackend(
        det_size=args.det_size,
        ctx_id=args.ctx,
        min_face=args.min_face,
        det_thresh=args.det_thresh,
        tile_grid=args.tile_grid,
        tile_overlap=args.tile_overlap,
        primary_detector_name=args.primary_detector,
        backup_detector_name=args.backup_detector,
        enable_backup_detector=not bool(args.disable_backup_detector),
        adaface_weights=args.adaface_weights or None,
        attribute_classifier_weights=args.attribute_classifier_weights or None,
        attribute_classifier_config=args.attribute_classifier_config or None,
        accessory_conf_threshold=args.accessory_conf_threshold,
    )
    tracker = FaceTracker(
        sim_thresh=args.sim_thresh,
        ttl=args.ttl,
        archive_ttl=args.archive_ttl,
        reid_sim_thresh=args.reid_sim_thresh,
        new_id_confirm_hits=args.new_id_confirm_hits,
        new_id_confirm_quality=args.new_id_confirm_quality,
        provisional_match_margin=args.provisional_match_margin,
        high_det_score=args.high_det_score,
        allow_new_persistent_identities=bool(args.allow_new_identities),
        full_reid_interval=args.full_reid_interval,
    )
    frame_change_gate = FrameChangeGate(
        diff_threshold=args.scene_change_thresh,
        unstable_inlier_ratio=args.scene_unstable_inlier_ratio,
        downsample_width=args.scene_downsample_width,
    )
    identity_db = FaceIdentityDB(args.identity_db)
    next_track_id, stored_identities = identity_db.load()
    loaded_identity_count = tracker.load_identity_memory(stored_identities, next_track_id=next_track_id)
    csv_logger = CsvLogger(args.csv)
    effective_roster_limit = int(args.roster_limit) if int(args.roster_limit) > 0 else roster_size(args.student_details_root)
    if effective_roster_limit <= 0:
        effective_roster_limit = max(1, len(stored_identities))

    if loaded_identity_count > 0:
        print(f"Loaded {loaded_identity_count} persistent identities from: {args.identity_db}", flush=True)
    print(f"Roster cap: {effective_roster_limit} reportable identities", flush=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_fps = fps if args.process_fps <= 0 else min(float(args.process_fps), fps)
    process_period_frames = max(1.0, fps / max(1e-6, process_fps))
    next_process_frame = 0.0

    frame_idx = 0
    processed_frame_count = 0
    saved_identity_count = loaded_identity_count
    frame_history: List[Dict[str, object]] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            if frame_idx + 1e-6 < next_process_frame:
                frame_idx += 1
                continue
            next_process_frame += process_period_frames

            detections = backend.infer(frame)
            scene_state = frame_change_gate.update(frame)
            tracker.set_frame_context(
                frame_idx=frame_idx,
                scene_changed=scene_state.changed,
                scene_score=scene_state.score,
                motion_stable=scene_state.motion_stable,
            )
            visible_tracks = tracker.step(detections, frame_idx)
            frame_history.append(build_frame_annotation(tracker, frame_idx, visible_tracks))
            processed_frame_count += 1

            if args.identity_db_save_every > 0 and frame_idx > 0 and frame_idx % args.identity_db_save_every == 0:
                saved_identity_count = identity_db.save(tracker)

            if args.display:
                preview = frame.copy()
                for track in visible_tracks:
                    draw_track(preview, track)
                active_identity_count = sum(1 for track in tracker.tracks.values() if track.track_id > 0)
                cv2.putText(
                    preview,
                    f"Frame: {frame_idx} | Visible students: {len(visible_tracks)} | Active IDs: {active_identity_count}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Classroom Face Tracker", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            frame_idx += 1
            if processed_frame_count > 0 and processed_frame_count % 50 == 0:
                msg_total = total_frames if total_frames > 0 else "?"
                print(
                    f"Processed {processed_frame_count} sampled frames at {process_fps:.2f} FPS "
                    f"| input frame {frame_idx}/{msg_total}",
                    flush=True,
                )

    finally:
        saved_identity_count = identity_db.save(tracker)
        cap.release()
        if args.display:
            cv2.destroyAllWindows()

    reportable_identity_ids = build_reportable_identity_set(tracker, effective_roster_limit)
    unknown_output_dir = args.unknown_output_dir or str(Path(args.output).resolve().parent / "unknown_review")
    unknown_csv_path, unknown_json_path, unknown_assignments_path = export_unknown_cluster_package(
        tracker,
        unknown_output_dir,
        args.input,
        reportable_identity_ids,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_parent_dir(args.output)
    writer = cv2.VideoWriter(args.output, fourcc, process_fps, (width, height))
    if not writer.isOpened():
        csv_logger.close()
        raise RuntimeError(f"Could not open video writer for: {args.output}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        writer.release()
        csv_logger.close()
        raise RuntimeError(f"Could not reopen video for rendering: {args.input}")

    frame_history_by_idx = {int(item["frame_idx"]): item for item in frame_history}
    frame_idx = 0
    next_process_frame = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            if frame_idx + 1e-6 < next_process_frame:
                frame_idx += 1
                continue
            next_process_frame += process_period_frames

            frame_payload = frame_history_by_idx.get(frame_idx)
            if frame_payload is None:
                frame_idx += 1
                continue

            resolved_tracks = resolved_frame_annotations(tracker, frame_payload)
            resolved_tracks = [
                item
                for item in resolved_tracks
                if int(item.get("student_id", 0)) in reportable_identity_ids
            ]
            for item in resolved_tracks:
                draw_track_annotation(
                    frame,
                    np.asarray(item.get("bbox", []), dtype=np.float32),
                    int(item.get("student_id", 0)),
                    int(item.get("hits", 0)),
                    item.get("metadata", {}),
                )

            cv2.putText(
                frame,
                f"Frame: {frame_idx} | Visible students: {int(frame_payload.get('visible_count', 0))} | Active IDs: {int(frame_payload.get('active_identity_count', 0))}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            csv_logger.log_annotations(frame_idx, resolved_tracks)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        csv_logger.close()

    print(f"Done. Output video saved to: {args.output}")
    print(f"Persistent identity DB saved to: {args.identity_db} ({saved_identity_count} identities)")
    if args.csv:
        print(f"CSV saved to: {args.csv}")
    print(f"Unknown review CSV: {unknown_csv_path}")
    print(f"Unknown review JSON: {unknown_json_path}")
    print(f"Unknown assignment template: {unknown_assignments_path}")


if __name__ == "__main__":
    main()
