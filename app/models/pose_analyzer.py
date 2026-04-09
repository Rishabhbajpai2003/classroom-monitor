"""
Recall-first pose analysis for classroom monitoring.

The analyzer runs one pose model pass on the full frame, then performs a
one-to-one assignment between tracked students and pose detections so a single
pose cannot leak into multiple neighboring tracks.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def assign_pose_detections(
    tracked_persons: list[dict],
    pose_bboxes: np.ndarray,
    min_score: float = 0.12,
) -> dict[int, int]:
    if not tracked_persons or pose_bboxes is None or len(pose_bboxes) == 0:
        return {}

    score_matrix = np.full((len(tracked_persons), len(pose_bboxes)), -1e9, dtype=np.float32)
    for row_idx, person in enumerate(tracked_persons):
        px1, py1, px2, py2 = [float(v) for v in person["bbox"]]
        person_h = max(1.0, py2 - py1)
        person_cx = 0.5 * (px1 + px2)
        person_cy = 0.5 * (py1 + py2)
        for col_idx, pose_bbox in enumerate(pose_bboxes):
            bx1, by1, bx2, by2 = [float(v) for v in pose_bbox[:4]]
            iou = bbox_iou(person["bbox"], pose_bbox[:4])
            pose_cx = 0.5 * (bx1 + bx2)
            pose_cy = 0.5 * (by1 + by2)
            center_dist = np.hypot(person_cx - pose_cx, person_cy - pose_cy)
            dist_score = max(0.0, 1.0 - center_dist / max(1.0, 1.1 * person_h))
            score = 0.72 * iou + 0.28 * dist_score
            if score < min_score:
                continue
            score_matrix[row_idx, col_idx] = score

    matches: dict[int, int] = {}
    if linear_sum_assignment is not None:
        cost = np.where(score_matrix < -1e8, 1e6, -score_matrix)
        row_ind, col_ind = linear_sum_assignment(cost)
        for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
            score = float(score_matrix[row_idx, col_idx])
            if score < min_score:
                continue
            matches[int(tracked_persons[row_idx]["track_id"])] = int(col_idx)
        return matches

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
        matches[int(tracked_persons[row_idx]["track_id"])] = int(col_idx)
        used_rows.add(row_idx)
        used_cols.add(col_idx)
    return matches


class PoseAnalyzer:
    def __init__(self, config: dict | None = None):
        config = config or {}
        sys_cfg = config.get("system", {})
        pose_cfg = config.get("pose", {})
        self.device = str(sys_cfg.get("device", "auto"))
        self.model_name = str(pose_cfg.get("weights", "weights/yolo11x-pose.pt"))
        self.imgsz = int(pose_cfg.get("image_size", 1152))
        self.conf = float(pose_cfg.get("confidence_threshold", 0.18))
        self.iou = float(pose_cfg.get("iou_threshold", 0.85))
        self.max_det = int(pose_cfg.get("max_det", 300))
        self._model = None
        self._keypoint_ema: dict[int, np.ndarray] = {}
        self._ema_alpha = float(pose_cfg.get("ema_alpha", 0.30))

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO

            logger.info("Loading PoseAnalyzer model: %s on %s", self.model_name, self.device)
            self._model = YOLO(self.model_name)
        return self._model

    @staticmethod
    def _default_result() -> dict:
        return {
            "hand_raised": False,
            "head_forward": False,
            "using_phone": False,
            "using_phone_pose": False,
            "pose_confidence": 0.0,
            "hand_points": [],
            "head_point": None,
            "keypoints_xy": [],
            "keypoints_conf": [],
        }

    def analyze_batch(self, frame: np.ndarray, tracked_persons: list[dict]) -> dict[int, dict]:
        if not tracked_persons:
            return {}

        try:
            result = self._load_model().predict(
                frame,
                conf=self.conf,
                imgsz=self.imgsz,
                iou=self.iou,
                max_det=self.max_det,
                device=self.device,
                verbose=False,
            )[0]
        except Exception as exc:
            logger.warning("Batch pose analysis failed: %s", exc)
            return {int(person["track_id"]): self._default_result() for person in tracked_persons}

        if result.keypoints is None or len(result.keypoints) == 0 or result.boxes is None:
            return {int(person["track_id"]): self._default_result() for person in tracked_persons}

        pose_bboxes = result.boxes.xyxy.detach().cpu().numpy()
        all_keypoints = result.keypoints.data.detach().cpu().numpy()
        assignments = assign_pose_detections(tracked_persons, pose_bboxes)

        output: dict[int, dict] = {}
        for person in tracked_persons:
            track_id = int(person["track_id"])
            pose_idx = assignments.get(track_id)
            if pose_idx is None:
                output[track_id] = self._default_result()
                continue
            output[track_id] = self._classify_pose(
                frame_shape=frame.shape,
                person_bbox=person["bbox"],
                track_id=track_id,
                kp_raw=all_keypoints[pose_idx],
            )
        return output

    def _classify_pose(
        self,
        frame_shape: tuple[int, int, int],
        person_bbox: list[float],
        track_id: int,
        kp_raw: np.ndarray,
    ) -> dict:
        h_frame, w_frame = frame_shape[:2]
        kp_norm = kp_raw.copy()
        kp_norm[:, 0] = kp_norm[:, 0] / max(1, w_frame)
        kp_norm[:, 1] = kp_norm[:, 1] / max(1, h_frame)

        if track_id in self._keypoint_ema:
            prev = self._keypoint_ema[track_id]
            kp_norm[:, :2] = self._ema_alpha * kp_norm[:, :2] + (1.0 - self._ema_alpha) * prev[:, :2]
        self._keypoint_ema[track_id] = kp_norm.copy()

        keypoints_xy = kp_raw[:, :2].astype(np.float32)
        keypoints_conf = kp_raw[:, 2].astype(np.float32)
        conf_threshold = 0.30

        nose = keypoints_xy[0]
        shoulders = keypoints_xy[[5, 6]]
        elbows = keypoints_xy[[7, 8]]
        wrists = keypoints_xy[[9, 10]]
        valid = keypoints_conf >= conf_threshold

        key_confs = [
            float(keypoints_conf[idx])
            for idx in [0, 5, 6, 7, 8, 9, 10]
            if idx < len(keypoints_conf)
        ]
        pose_confidence = float(np.mean(key_confs)) if key_confs else 0.0

        hand_points = [keypoints_xy[idx].tolist() for idx in [9, 10] if idx < len(valid) and valid[idx]]
        head_point = nose.tolist() if valid[0] else None

        body_h = max(1.0, person_bbox[3] - person_bbox[1])
        hand_raised = False
        head_y = float(nose[1]) if valid[0] else None
        raise_checks = [
            (9, 7, 5),
            (10, 8, 6),
        ]
        for wrist_idx, elbow_idx, shoulder_idx in raise_checks:
            if not (valid[wrist_idx] and valid[shoulder_idx]):
                continue
            wrist_y = float(keypoints_xy[wrist_idx][1])
            shoulder_y = float(keypoints_xy[shoulder_idx][1])
            elbow_y = float(keypoints_xy[elbow_idx][1]) if valid[elbow_idx] else shoulder_y
            above_head = head_y is not None and wrist_y < head_y - 0.06 * body_h
            above_shoulder = wrist_y < shoulder_y - 0.16 * body_h and elbow_y < shoulder_y + 0.02 * body_h
            if above_head or above_shoulder:
                hand_raised = True
                break

        head_forward = False
        if valid[0]:
            if valid[5] and valid[6]:
                shoulder_center_x = 0.5 * (shoulders[0][0] + shoulders[1][0])
                shoulder_span = max(8.0, abs(shoulders[1][0] - shoulders[0][0]))
                head_forward = abs(nose[0] - shoulder_center_x) <= 0.32 * shoulder_span
            else:
                px1, _, px2, _ = [float(v) for v in person_bbox]
                head_forward = abs(nose[0] - 0.5 * (px1 + px2)) <= 0.18 * max(1.0, px2 - px1)

        using_phone_pose = False
        if valid[0]:
            shoulder_y = None
            valid_shoulder_ys = [keypoints_xy[idx][1] for idx in [5, 6] if valid[idx]]
            if valid_shoulder_ys:
                shoulder_y = float(np.mean(valid_shoulder_ys))
            if shoulder_y is not None:
                wrists_above_torso = any(valid[idx] and keypoints_xy[idx][1] <= shoulder_y + 0.18 * max(1.0, person_bbox[3] - person_bbox[1]) for idx in [9, 10])
                wrist_near_face = any(valid[idx] and np.linalg.norm(keypoints_xy[idx] - nose) <= 0.28 * max(1.0, person_bbox[3] - person_bbox[1]) for idx in [9, 10])
                head_down = nose[1] >= shoulder_y - 0.04 * max(1.0, person_bbox[3] - person_bbox[1])
                using_phone_pose = wrists_above_torso and wrist_near_face and head_down

        if using_phone_pose:
            head_forward = False

        return {
            "hand_raised": bool(hand_raised),
            "head_forward": bool(head_forward),
            "using_phone": bool(using_phone_pose),
            "using_phone_pose": bool(using_phone_pose),
            "pose_confidence": float(pose_confidence),
            "hand_points": hand_points,
            "head_point": head_point,
            "keypoints_xy": keypoints_xy.tolist(),
            "keypoints_conf": keypoints_conf.tolist(),
        }

    def analyze(self, frame: np.ndarray, bbox, track_id: Optional[int] = None):
        persons = [{"track_id": track_id or 0, "bbox": list(bbox)}]
        results = self.analyze_batch(frame, persons)
        return results.get(track_id or 0, self._default_result())

    def close(self) -> None:
        return None
