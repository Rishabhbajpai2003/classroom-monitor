"""
Shared recall-first student backbone built around the face tracker.

The face tracker is treated as the source of truth for student presence and
identity. Person/body detections are associated afterwards when available so
far-back students can still remain valid tracked entities even when body/pose
evidence is weak.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def _bbox_height(box: Optional[list[float] | np.ndarray]) -> float:
    if box is None:
        return 0.0
    return max(0.0, float(box[3]) - float(box[1]))


def _bbox_center(box: list[float] | np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def has_named_identity(metadata: Optional[dict[str, Any]] = None) -> bool:
    if not metadata:
        return False
    student_key = str(metadata.get("student_key", "") or "").strip()
    student_name = str(metadata.get("name", "") or "").strip()
    return bool(student_key or student_name)


def student_metadata_fields(metadata: Optional[dict[str, Any]] = None) -> dict[str, str]:
    metadata = metadata or {}
    return {
        "student_name": str(metadata.get("name", "") or "").strip(),
        "roll_number": str(metadata.get("roll_number", "") or "").strip(),
        "student_key": str(metadata.get("student_key", "") or "").strip(),
    }


def format_student_global_id(track_id: int, metadata: Optional[dict[str, Any]] = None) -> str:
    if int(track_id) > 0 and has_named_identity(metadata):
        return f"STU_{int(track_id):03d}"
    return f"TEMP_{abs(int(track_id)):04d}"


def format_student_display_name(
    track_id: int,
    metadata: Optional[dict[str, Any]] = None,
    *,
    include_roll: bool = False,
    unknown_label: str = "Unknown",
) -> str:
    fields = student_metadata_fields(metadata)
    name = fields["student_name"]
    roll_number = fields["roll_number"]
    student_key = fields["student_key"]
    if name and include_roll and roll_number:
        return f"{name}-{roll_number}"
    if name:
        return name
    if student_key:
        return student_key
    return unknown_label


def seat_anchor_point(
    body_bbox: Optional[list[float] | np.ndarray],
    face_bbox: Optional[list[float] | np.ndarray] = None,
) -> np.ndarray:
    if body_bbox is not None:
        x1, y1, x2, y2 = [float(v) for v in body_bbox]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        return np.asarray(
            [
                0.5 * (x1 + x2),
                y1 + 0.82 * height,
            ],
            dtype=np.float32,
        )

    if face_bbox is not None:
        x1, y1, x2, y2 = [float(v) for v in face_bbox]
        face_h = max(1.0, y2 - y1)
        return np.asarray(
            [
                0.5 * (x1 + x2),
                y2 + 1.85 * face_h,
            ],
            dtype=np.float32,
        )

    return np.zeros(2, dtype=np.float32)


def classify_size_mode(
    body_bbox: Optional[list[float] | np.ndarray],
    face_bbox: Optional[list[float] | np.ndarray] = None,
    full_min_height: int = 160,
    reduced_min_height: int = 96,
) -> str:
    body_height = _bbox_height(body_bbox)
    if body_height <= 0.0 and face_bbox is not None:
        body_height = _bbox_height(face_bbox) * 3.0

    if body_height >= float(full_min_height):
        return "full"
    if body_height >= float(reduced_min_height):
        return "reduced"
    return "limited"


def _association_score(face_bbox: list[float] | np.ndarray, body_bbox: list[float] | np.ndarray) -> float:
    fx1, fy1, fx2, fy2 = [float(v) for v in face_bbox]
    bx1, by1, bx2, by2 = [float(v) for v in body_bbox]
    if bx2 <= bx1 or by2 <= by1 or fx2 <= fx1 or fy2 <= fy1:
        return -1e9

    body_w = max(1.0, bx2 - bx1)
    body_h = max(1.0, by2 - by1)
    face_w = max(1.0, fx2 - fx1)
    face_h = max(1.0, fy2 - fy1)
    face_cx, face_cy = _bbox_center(face_bbox)
    body_cx, _ = _bbox_center(body_bbox)

    in_x = 1.0 if (bx1 - 0.12 * body_w) <= face_cx <= (bx2 + 0.12 * body_w) else 0.0
    in_upper = 1.0 if (by1 - 0.08 * body_h) <= face_cy <= (by1 + 0.55 * body_h) else 0.0
    if in_x == 0.0 or in_upper == 0.0:
        return -1e9

    x_align = max(0.0, 1.0 - abs(face_cx - body_cx) / max(1.0, 0.5 * body_w))
    expected_face_ratio = 0.22
    observed_ratio = face_h / body_h
    scale_score = max(0.0, 1.0 - abs(observed_ratio - expected_face_ratio) / 0.22)
    top_gap = max(0.0, fy1 - by1)
    vertical_score = max(0.0, 1.0 - top_gap / max(1.0, 0.4 * body_h))
    width_score = max(0.0, 1.0 - abs((face_w / body_w) - 0.24) / 0.24)
    return (
        0.34 * x_align
        + 0.24 * scale_score
        + 0.22 * vertical_score
        + 0.20 * width_score
    )


def associate_face_tracks_to_body_detections(
    face_tracks: list[Any],
    body_detections: list[dict],
    min_score: float = 0.20,
) -> dict[int, dict]:
    if not face_tracks or not body_detections:
        return {}

    score_matrix = np.full((len(face_tracks), len(body_detections)), -1e9, dtype=np.float32)
    for row_idx, track in enumerate(face_tracks):
        face_bbox = np.asarray(track.bbox if hasattr(track, "bbox") else track["bbox"], dtype=np.float32)
        for col_idx, det in enumerate(body_detections):
            score_matrix[row_idx, col_idx] = _association_score(face_bbox, det["bbox"])

    matches: list[tuple[int, int]] = []
    if linear_sum_assignment is not None:
        cost = np.where(score_matrix < -1e8, 1e6, -score_matrix)
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = list(zip(row_ind.tolist(), col_ind.tolist()))
    else:
        candidates: list[tuple[float, int, int]] = []
        for row_idx in range(score_matrix.shape[0]):
            for col_idx in range(score_matrix.shape[1]):
                candidates.append((float(score_matrix[row_idx, col_idx]), row_idx, col_idx))
        candidates.sort(reverse=True)
        used_rows = set()
        used_cols = set()
        for score, row_idx, col_idx in candidates:
            if score < min_score:
                break
            if row_idx in used_rows or col_idx in used_cols:
                continue
            matches.append((row_idx, col_idx))
            used_rows.add(row_idx)
            used_cols.add(col_idx)

    association: dict[int, dict] = {}
    for row_idx, col_idx in matches:
        score = float(score_matrix[row_idx, col_idx])
        if score < min_score:
            continue
        track = face_tracks[row_idx]
        track_id = int(track.track_id if hasattr(track, "track_id") else track["track_id"])
        det = dict(body_detections[col_idx])
        det["association_score"] = score
        association[track_id] = det
    return association


@dataclass
class StudentObservation:
    global_id: str
    track_id: int
    face_bbox: list[float]
    body_bbox: Optional[list[float]]
    face_confidence: float
    detection_confidence: float
    size_mode: str
    body_detection: Optional[dict] = None
    face_track: Any = None


class SharedStudentBackbone:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        sys_cfg = self.config.get("system", {})
        back_cfg = self.config.get("student_backbone", {})
        self.device = str(sys_cfg.get("device", "auto"))
        self.process_fps = float(back_cfg.get("face_process_fps", 8.0))
        self.det_size = int(back_cfg.get("face_det_size", 1600))
        self.det_thresh = float(back_cfg.get("face_det_thresh", 0.20))
        self.tile_grid = int(back_cfg.get("face_tile_grid", 3))
        self.tile_overlap = float(back_cfg.get("face_tile_overlap", 0.22))
        self.min_face = int(back_cfg.get("face_min_size", 12))
        self.identity_db_path = str(
            back_cfg.get("identity_db_path", "detectors/face_detector/identity_db.json")
        )
        self.identity_db_save_every = int(back_cfg.get("identity_db_save_every", 150))
        self.body_memory_frames = int(back_cfg.get("body_memory_frames", 15))
        self.full_min_height = int(back_cfg.get("full_min_height", 160))
        self.reduced_min_height = int(back_cfg.get("reduced_min_height", 96))
        self.next_process_frame = 0.0
        self.process_period_frames = 1.0
        self.gap_fill_frames = 1
        self._video_fps: Optional[float] = None
        self._last_body_by_track: dict[int, tuple[int, dict]] = {}

        self._backend = None
        self._tracker = None
        self._identity_db = None
        self._get_gap_fill_tracks = None

        self._init_runtime()

    def _init_runtime(self) -> None:
        try:
            from detectors.face_detector.run import FaceIdentityDB, FaceTracker, InsightFaceBackend
        except Exception as exc:
            raise RuntimeError(
                "The recall-first student backbone requires insightface and onnxruntime. "
                "Install them with: pip install insightface onnxruntime "
                "(or onnxruntime-gpu for CUDA)."
            ) from exc
        from detectors.vsd_detector.common import get_gap_fill_tracks

        ctx_id = -1
        if self.device in {"auto", "cuda"}:
            ctx_id = 0

        self._backend = InsightFaceBackend(
            det_size=self.det_size,
            ctx_id=ctx_id,
            min_face=self.min_face,
            det_thresh=self.det_thresh,
            tile_grid=self.tile_grid,
            tile_overlap=self.tile_overlap,
        )
        self._tracker = FaceTracker(
            sim_thresh=float(self.config.get("student_backbone", {}).get("sim_thresh", 0.45)),
            iou_weight=float(self.config.get("student_backbone", {}).get("iou_weight", 0.22)),
            sim_weight=float(self.config.get("student_backbone", {}).get("sim_weight", 0.70)),
            dist_weight=float(self.config.get("student_backbone", {}).get("dist_weight", 0.08)),
            ttl=int(self.config.get("student_backbone", {}).get("ttl", 120)),
            archive_ttl=int(self.config.get("student_backbone", {}).get("archive_ttl", 1800)),
            reid_sim_thresh=float(self.config.get("student_backbone", {}).get("reid_sim_thresh", 0.52)),
            min_confirm_hits=int(self.config.get("student_backbone", {}).get("min_confirm_hits", 2)),
            appearance_weight=float(self.config.get("student_backbone", {}).get("appearance_weight", 0.16)),
            min_face_sim=float(self.config.get("student_backbone", {}).get("min_face_sim", 0.20)),
            merge_sim_thresh=float(self.config.get("student_backbone", {}).get("merge_sim_thresh", 0.60)),
            young_track_hits=int(self.config.get("student_backbone", {}).get("young_track_hits", 10)),
            new_id_confirm_hits=int(self.config.get("student_backbone", {}).get("new_id_confirm_hits", 2)),
            new_id_confirm_quality=float(self.config.get("student_backbone", {}).get("new_id_confirm_quality", 0.20)),
            provisional_match_margin=float(
                self.config.get("student_backbone", {}).get("provisional_match_margin", 0.02)
            ),
            high_det_score=float(self.config.get("student_backbone", {}).get("high_det_score", 0.48)),
            continuity_window=int(self.config.get("student_backbone", {}).get("continuity_window", 4)),
            continuity_iou_gate=float(
                self.config.get("student_backbone", {}).get("continuity_iou_gate", 0.16)
            ),
            continuity_dist_gate=float(
                self.config.get("student_backbone", {}).get("continuity_dist_gate", 0.36)
            ),
            continuity_relax=float(self.config.get("student_backbone", {}).get("continuity_relax", 0.12)),
            continuity_bonus=float(self.config.get("student_backbone", {}).get("continuity_bonus", 0.14)),
            strong_named_match_score=float(
                self.config.get("student_backbone", {}).get("strong_named_match_score", 0.72)
            ),
            allow_new_persistent_identities=bool(
                self.config.get("student_backbone", {}).get("allow_new_identities", False)
            ),
        )
        self._identity_db = FaceIdentityDB(self.identity_db_path)
        next_track_id, stored_identities = self._identity_db.load()
        loaded = self._tracker.load_identity_memory(stored_identities, next_track_id=next_track_id)
        if loaded > 0:
            logger.info("Loaded %s persistent identities from %s", loaded, self.identity_db_path)
        self._get_gap_fill_tracks = get_gap_fill_tracks

    def _configure_video_cadence(self, fps: float) -> None:
        fps = max(1.0, float(fps))
        self._video_fps = fps
        effective_fps = fps if self.process_fps <= 0 else min(self.process_fps, fps)
        self.process_period_frames = max(1.0, fps / max(1e-6, effective_fps))
        self.gap_fill_frames = max(1, int(round(self.process_period_frames)))
        self.next_process_frame = 0.0

    def _should_process_face(self, frame_idx: int, fps: float) -> bool:
        if self._video_fps is None or not math.isclose(float(self._video_fps), float(fps), rel_tol=0.01):
            self._configure_video_cadence(fps)
        if frame_idx + 1e-6 >= self.next_process_frame:
            self.next_process_frame += self.process_period_frames
            return True
        return False

    def step(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        body_detections: Optional[list[dict]] = None,
    ) -> list[StudentObservation]:
        if self._should_process_face(frame_idx, fps):
            detections = self._backend.infer(frame)
            visible_tracks = self._tracker.step(detections, frame_idx)
        else:
            visible_tracks = self._get_gap_fill_tracks(self._tracker.tracks, frame_idx, self.gap_fill_frames)

        body_detections = body_detections or []
        associated = associate_face_tracks_to_body_detections(visible_tracks, body_detections)
        observations: list[StudentObservation] = []

        for track in visible_tracks:
            track_id = int(track.track_id)
            face_bbox = np.asarray(track.bbox, dtype=np.float32).tolist()
            body_detection = associated.get(track_id)
            if body_detection is not None:
                self._last_body_by_track[track_id] = (frame_idx, body_detection)
            else:
                cached = self._last_body_by_track.get(track_id)
                if cached is not None and frame_idx - int(cached[0]) <= self.body_memory_frames:
                    body_detection = cached[1]

            body_bbox = None if body_detection is None else list(body_detection["bbox"])
            size_mode = classify_size_mode(
                body_bbox,
                face_bbox=face_bbox,
                full_min_height=self.full_min_height,
                reduced_min_height=self.reduced_min_height,
            )
            face_conf = float(getattr(track, "best_score", 0.0) or 0.0)
            det_conf = face_conf
            if body_detection is not None:
                det_conf = max(det_conf, float(body_detection.get("confidence", 0.0)))

            observations.append(
                StudentObservation(
                    global_id=format_student_global_id(track_id, getattr(track, "metadata", None)),
                    track_id=track_id,
                    face_bbox=face_bbox,
                    body_bbox=body_bbox,
                    face_confidence=face_conf,
                    detection_confidence=det_conf,
                    size_mode=size_mode,
                    body_detection=body_detection,
                    face_track=track,
                )
            )

        if self.identity_db_save_every > 0 and frame_idx > 0 and frame_idx % self.identity_db_save_every == 0:
            self._identity_db.save(self._tracker)
        return observations

    def close(self) -> None:
        if self._identity_db is not None and self._tracker is not None:
            self._identity_db.save(self._tracker)
