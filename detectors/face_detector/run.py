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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    print(
        "Failed to import insightface. Install with: pip install insightface onnxruntime "
        "(or onnxruntime-gpu for CUDA).",
        file=sys.stderr,
    )
    raise

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


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
    avg_embedding: Optional[np.ndarray] = None
    best_embedding: Optional[np.ndarray] = None
    best_embedding_quality: float = 0.0
    recent_embedding: Optional[np.ndarray] = None
    appearance_embeddings: List[np.ndarray] = field(default_factory=list)
    appearance_qualities: List[float] = field(default_factory=list)
    avg_appearance: Optional[np.ndarray] = None
    best_appearance: Optional[np.ndarray] = None
    best_appearance_quality: float = 0.0
    recent_appearance: Optional[np.ndarray] = None
    persistent_identity: bool = False
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    metadata: Dict[str, object] = field(default_factory=dict)

    def update_embedding_bank(
        self,
        new_emb: np.ndarray,
        sample_quality: float = 0.0,
        max_bank: int = 15,
        novelty_thresh: float = 0.10,
    ) -> None:
        """
        Keep multiple embeddings per student so different poses/lighting are remembered.
        novelty_thresh is based on cosine distance = 1 - similarity.
        """
        new_emb = l2_normalize(new_emb.astype(np.float32))
        while len(self.embedding_qualities) < len(self.embeddings):
            self.embedding_qualities.append(max(0.5, float(self.best_embedding_quality)))
        if not self.embeddings:
            self.embeddings.append(new_emb)
            self.embedding_qualities.append(float(sample_quality))
        else:
            sims = [cosine_similarity(new_emb, e) for e in self.embeddings]
            best_sim = max(sims)
            cosine_dist = 1.0 - best_sim
            if cosine_dist >= novelty_thresh and len(self.embeddings) < max_bank:
                self.embeddings.append(new_emb)
                self.embedding_qualities.append(float(sample_quality))
            else:
                # Replace the most similar entry with a smoothed version.
                best_idx = int(np.argmax(sims))
                self.embeddings[best_idx] = l2_normalize(0.7 * self.embeddings[best_idx] + 0.3 * new_emb)
                while len(self.embedding_qualities) < len(self.embeddings):
                    self.embedding_qualities.append(float(sample_quality))
                self.embedding_qualities[best_idx] = max(float(sample_quality), self.embedding_qualities[best_idx])

        self.avg_embedding = weighted_average_embeddings(self.embeddings, self.embedding_qualities)
        if self.recent_embedding is None:
            self.recent_embedding = new_emb.copy()
        else:
            self.recent_embedding = l2_normalize(0.65 * self.recent_embedding + 0.35 * new_emb)
        if self.best_embedding is None or sample_quality >= self.best_embedding_quality:
            self.best_embedding = new_emb.copy()
            self.best_embedding_quality = float(sample_quality)

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

        self.next_track_id = 1
        self.next_temp_track_id = -1
        self.tracks: Dict[int, Track] = {}
        self.archived_tracks: Dict[int, Track] = {}

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
                track.persistent_identity = True
                tracks[track_id] = track
        return tracks

    def _allocate_provisional_track_id(self) -> int:
        track_id = self.next_temp_track_id
        self.next_temp_track_id -= 1
        return track_id

    def _embedding_match_score(self, det_emb: np.ndarray, track: Track) -> float:
        if track.avg_embedding is None:
            return -1.0
        avg_sim = cosine_similarity(det_emb, track.avg_embedding)
        sims = [avg_sim]
        if track.recent_embedding is not None:
            sims.append(0.97 * cosine_similarity(det_emb, track.recent_embedding))
        if track.embeddings:
            bank_scores = []
            for idx, emb in enumerate(track.embeddings):
                sim = cosine_similarity(det_emb, emb)
                quality = 1.0
                if idx < len(track.embedding_qualities):
                    quality = max(0.05, float(track.embedding_qualities[idx]))
                bank_scores.append(sim * (0.85 + 0.15 * quality))
            if bank_scores:
                sims.append(max(bank_scores))
        if track.best_embedding is not None:
            sims.append(cosine_similarity(det_emb, track.best_embedding))

        best_bank = max(sims)
        return 0.55 * best_bank + 0.45 * avg_sim

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

    def _identity_score(self, face_sim: float, appearance_sim: float) -> float:
        appearance_weight = self.appearance_weight if appearance_sim > 0.0 else 0.0
        face_weight = 1.0 - appearance_weight
        return float(face_weight * face_sim + appearance_weight * appearance_sim)

    def _association_score(self, det: Detection, track: Track) -> float:
        pred_box = track.predict_bbox()
        face_sim = self._embedding_match_score(det.embedding, track)
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

        face_sim = self._embedding_match_score(det.embedding, track)
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

        face_sim = self._embedding_match_score(probe_track.avg_embedding, reference_track)
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
        track.update_embedding_bank(det.embedding, sample_quality=det.quality)
        track.update_appearance_bank(det.appearance, sample_quality=det.quality)
        if track.track_id > 0 and track.hits >= self.min_confirm_hits and track.avg_embedding is not None:
            track.persistent_identity = True
        self.tracks[track.track_id] = track
        return track

    def _update_track(self, track: Track, det: Detection, frame_idx: int) -> None:
        pred_box = track.predict_bbox()
        iou = iou_xyxy(det.bbox, pred_box)
        dist = normalized_center_distance(det.bbox, pred_box)
        face_sim = self._embedding_match_score(det.embedding, track)
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
        track.update_embedding_bank(det.embedding, sample_quality=bank_quality)
        track.update_appearance_bank(det.appearance, sample_quality=bank_quality)
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
        track.update_embedding_bank(det.embedding, sample_quality=det.quality)
        track.update_appearance_bank(det.appearance, sample_quality=det.quality)
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

        for emb in young_track.embeddings:
            restored_track.update_embedding_bank(emb, sample_quality=young_track.best_embedding_quality)
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

        for emb in young_track.embeddings:
            target_track.update_embedding_bank(emb, sample_quality=young_track.best_embedding_quality)
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
            if best_track_id is not None and best_score > -1e8 and margin >= self.provisional_match_margin:
                if best_source == "archived":
                    self._merge_track_into_archived_identity(provisional_id, best_track_id, frame_idx)
                elif best_source == "active":
                    self._merge_track_into_active_identity(provisional_id, best_track_id, frame_idx)
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

        high_matches = self._match_by_score_matrix(
            high_det_indices,
            track_ids,
            lambda di, tid: self._association_score(detections[di], self.tracks[tid]),
        )
        for score, di, tid in high_matches:
            if score < -1e8:
                continue
            self._update_track(self.tracks[tid], detections[di], frame_idx)
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
                self._update_track(self.tracks[tid], detections[di], frame_idx)
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
            matched_det_indices.add(di)
            revived_track_ids.add(tid)

        # Create provisional tracks only when neither active matching nor re-id found a candidate.
        for di in det_indices:
            if di not in matched_det_indices:
                track = self._create_track(detections[di], frame_idx)
                matched_track_ids.add(track.track_id)

        # Resolve provisional tracklets against existing identities before assigning
        # a brand new permanent ID. This avoids creating a fresh student ID from a
        # single weak re-entry frame.
        self._resolve_provisional_tracks(frame_idx)

        # Return active tracks visible in this frame.
        visible_tracks = [t for t in self.tracks.values() if t.last_frame_idx == frame_idx]
        visible_tracks.sort(key=lambda t: t.track_id)
        return visible_tracks


# -----------------------------
# Face detector / embedder
# -----------------------------

class InsightFaceBackend:
    def __init__(
        self,
        det_size: int = 960,
        ctx_id: int = 0,
        min_face: int = 20,
        det_thresh: float = 0.35,
        tile_grid: int = 1,
        tile_overlap: float = 0.20,
    ) -> None:
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

    def _faces_to_detections(
        self,
        frame_bgr: np.ndarray,
        faces,
        offset_x: int = 0,
        offset_y: int = 0,
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
            emb = np.asarray(face.normed_embedding, dtype=np.float32)
            emb = l2_normalize(emb)
            kps = np.asarray(getattr(face, "kps", None), dtype=np.float32) if getattr(face, "kps", None) is not None else None
            if kps is not None:
                kps[:, 0] += float(offset_x)
                kps[:, 1] += float(offset_y)
            appearance = extract_appearance_descriptor(frame_bgr, bbox)
            sharpness = estimate_crop_sharpness(frame_bgr, bbox)
            quality = estimate_detection_quality(bbox, score, kps, sharpness=sharpness)
            detections.append(
                Detection(
                    bbox=bbox,
                    score=score,
                    embedding=emb,
                    landmarks=kps,
                    appearance=appearance,
                    quality=quality,
                )
            )
        return detections

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        detections = self._faces_to_detections(frame_bgr, self.app.get(frame_bgr))

        if self.tile_grid > 1:
            tiles = generate_overlapping_tiles(frame_bgr.shape, self.tile_grid, self.tile_overlap)
            for x1, y1, x2, y2 in tiles:
                if x1 == 0 and y1 == 0 and x2 == frame_bgr.shape[1] and y2 == frame_bgr.shape[0]:
                    continue
                tile = frame_bgr[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                detections.extend(self._faces_to_detections(frame_bgr, self.app.get(tile), offset_x=x1, offset_y=y1))

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

    def _record_to_track(self, record: Dict[str, object]) -> Optional[Track]:
        track_id = int(record.get("track_id", 0))
        if track_id <= 0:
            return None

        embeddings = self._to_array_bank(record.get("embeddings"))
        appearance_embeddings = self._to_array_bank(record.get("appearance_embeddings"))
        embedding_qualities = [float(v) for v in record.get("embedding_qualities", [])]
        appearance_qualities = [float(v) for v in record.get("appearance_qualities", [])]
        while len(embedding_qualities) < len(embeddings):
            embedding_qualities.append(float(record.get("best_embedding_quality", 0.5)))
        while len(appearance_qualities) < len(appearance_embeddings):
            appearance_qualities.append(float(record.get("best_appearance_quality", 0.5)))
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
            avg_embedding=avg_embedding,
            best_embedding=self._to_array(record.get("best_embedding")),
            best_embedding_quality=float(record.get("best_embedding_quality", 0.0)),
            appearance_embeddings=appearance_embeddings,
            appearance_qualities=appearance_qualities,
            avg_appearance=avg_appearance,
            best_appearance=self._to_array(record.get("best_appearance")),
            best_appearance_quality=float(record.get("best_appearance_quality", 0.0)),
            persistent_identity=True,
            metadata=metadata,
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
            "avg_embedding": self._to_list(track.avg_embedding),
            "best_embedding": self._to_list(track.best_embedding),
            "best_embedding_quality": float(track.best_embedding_quality),
            "appearance_embeddings": self._to_list_bank(track.appearance_embeddings),
            "appearance_qualities": [float(v) for v in track.appearance_qualities],
            "avg_appearance": self._to_list(track.avg_appearance),
            "best_appearance": self._to_list(track.best_appearance),
            "best_appearance_quality": float(track.best_appearance_quality),
            "metadata": dict(track.metadata or {}),
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
            "version": 1,
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
    x1, y1, x2, y2 = track.bbox.astype(int)
    color = color_from_id(track.track_id)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"Student {track.track_id}" if track.track_id > 0 else "Student"

    if track.hits < 2 or track.track_id <= 0:
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
                "frame_idx", "student_id", "x1", "y1", "x2", "y2", "hits", "first_frame_idx", "last_frame_idx"
            ])

    def log(self, frame_idx: int, tracks: List[Track]) -> None:
        if not self.writer:
            return
        for t in tracks:
            x1, y1, x2, y2 = t.bbox.tolist()
            self.writer.writerow([
                frame_idx, t.track_id, f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                t.hits, t.first_frame_idx, t.last_frame_idx
            ])

    def close(self) -> None:
        if self.file:
            self.file.close()


# -----------------------------
# Main processing loop
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classroom face detection + student ID tracking")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output annotated video")
    parser.add_argument("--csv", default=None, help="Optional CSV path to save detections")
    parser.add_argument("--det-size", type=int, default=960, help="InsightFace detection size, e.g. 640/960/1280")
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
    )
    identity_db = FaceIdentityDB(args.identity_db)
    next_track_id, stored_identities = identity_db.load()
    loaded_identity_count = tracker.load_identity_memory(stored_identities, next_track_id=next_track_id)
    csv_logger = CsvLogger(args.csv)

    if loaded_identity_count > 0:
        print(f"Loaded {loaded_identity_count} persistent identities from: {args.identity_db}", flush=True)

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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_parent_dir(args.output)
    writer = cv2.VideoWriter(args.output, fourcc, process_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {args.output}")

    frame_idx = 0
    processed_frame_count = 0
    saved_identity_count = loaded_identity_count
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
            visible_tracks = tracker.step(detections, frame_idx)

            for track in visible_tracks:
                draw_track(frame, track)

            # Overlay simple stats.
            active_identity_count = sum(1 for track in tracker.tracks.values() if track.track_id > 0)
            cv2.putText(
                frame,
                f"Frame: {frame_idx} | Visible students: {len(visible_tracks)} | Active IDs: {active_identity_count}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            csv_logger.log(frame_idx, visible_tracks)
            processed_frame_count += 1

            if args.identity_db_save_every > 0 and frame_idx > 0 and frame_idx % args.identity_db_save_every == 0:
                saved_identity_count = identity_db.save(tracker)

            if args.display:
                cv2.imshow("Classroom Face Tracker", frame)
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
        writer.release()
        csv_logger.close()
        if args.display:
            cv2.destroyAllWindows()

    print(f"Done. Output video saved to: {args.output}")
    print(f"Persistent identity DB saved to: {args.identity_db} ({saved_identity_count} identities)")
    if args.csv:
        print(f"CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
