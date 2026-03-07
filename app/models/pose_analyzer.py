"""
Pose analysis module using YOLO Pose model — batch full-frame mode.

Runs YOLO pose ONCE on the full frame to get ALL keypoints in a single
inference, then matches keypoints to tracked person bounding boxes.
This is ~40x faster than running per-person-crop inference.

Outputs per-person:
    - Hand raised (wrist above shoulder)
    - Head facing forward (nose centered)
    - Phone usage heuristic
    - Pose detection confidence
"""

import logging
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PoseAnalyzer:
    """
    Batch pose analyzer — runs YOLO pose once on the full frame and
    matches detected keypoints to tracked person bounding boxes using IoU.

    Attributes:
        model: YOLO pose model instance.
        device: Compute device.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        sys_cfg = config.get("system", {})

        device = sys_cfg.get("device", "auto")
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        model_name = "yolo11n-pose.pt"
        logger.info(f"Loading PoseAnalyzer model: {model_name} on {self.device}")
        self.model = YOLO(model_name)

        # Per-track keypoint EMA for smoothing jitter
        self._keypoint_ema: dict[int, np.ndarray] = {}
        self._ema_alpha = 0.3

    def analyze_batch(self, frame, tracked_persons: list[dict]) -> dict[int, dict]:
        """
        Analyze pose for ALL persons in a single YOLO inference.

        Args:
            frame: Full BGR frame (H x W x 3).
            tracked_persons: List of dicts with 'track_id' and 'bbox' [l,t,r,b].

        Returns:
            Dict mapping track_id -> pose result dict.
        """
        default_result = {
            "hand_raised": False,
            "head_forward": False,
            "using_phone": False,
            "pose_confidence": 0.0,
        }

        if not tracked_persons:
            return {}

        try:
            # Single YOLO pose inference on full frame
            results = self.model(
                frame,
                device=self.device,
                verbose=False,
                imgsz=640,  # Lower res for speed — pose doesn't need 1280
            )[0]

            if results.keypoints is None or len(results.keypoints) == 0:
                return {p["track_id"]: dict(default_result) for p in tracked_persons}

            # Get all detected pose bboxes and keypoints
            pose_bboxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.array([])
            all_keypoints = results.keypoints.data.cpu().numpy()  # (N, 17, 3)

            # Match each tracked person to nearest pose detection via IoU
            result_map = {}

            for person in tracked_persons:
                track_id = person["track_id"]
                person_bbox = person["bbox"]  # [l, t, r, b]

                if len(pose_bboxes) == 0:
                    result_map[track_id] = dict(default_result)
                    continue

                # Find best IoU match
                best_idx = self._best_iou_match(person_bbox, pose_bboxes)

                if best_idx is None:
                    result_map[track_id] = dict(default_result)
                    continue

                # Analyze keypoints for this person
                kp_raw = all_keypoints[best_idx]  # (17, 3)
                h_frame, w_frame = frame.shape[:2]

                # Normalize to 0-1
                kp_norm = kp_raw.copy()
                kp_norm[:, 0] = kp_norm[:, 0] / w_frame
                kp_norm[:, 1] = kp_norm[:, 1] / h_frame

                # Apply per-track EMA smoothing
                if track_id in self._keypoint_ema:
                    prev = self._keypoint_ema[track_id]
                    kp_norm[:, :2] = (
                        self._ema_alpha * kp_norm[:, :2] +
                        (1 - self._ema_alpha) * prev[:, :2]
                    )
                self._keypoint_ema[track_id] = kp_norm.copy()

                result_map[track_id] = self._classify_pose(kp_norm, person_bbox, w_frame, h_frame)

            return result_map

        except Exception as e:
            logger.warning(f"Batch pose analysis failed: {e}")
            return {p["track_id"]: dict(default_result) for p in tracked_persons}

    def _best_iou_match(self, person_bbox, pose_bboxes) -> int | None:
        """Find pose detection with highest IoU to the tracked person bbox."""
        px1, py1, px2, py2 = person_bbox

        best_iou = 0.2  # Minimum IoU threshold
        best_idx = None

        for i, pb in enumerate(pose_bboxes):
            bx1, by1, bx2, by2 = pb[:4]

            # Compute IoU
            ix1 = max(px1, bx1)
            iy1 = max(py1, by1)
            ix2 = min(px2, bx2)
            iy2 = min(py2, by2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            inter = (ix2 - ix1) * (iy2 - iy1)
            area_p = (px2 - px1) * (py2 - py1)
            area_b = (bx2 - bx1) * (by2 - by1)
            union = area_p + area_b - inter

            if union <= 0:
                continue

            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        return best_idx

    def _classify_pose(self, kp_norm: np.ndarray, person_bbox, w_frame, h_frame) -> dict:
        """Classify pose signals from normalized keypoints."""
        # Extract landmarks
        nose_x, nose_y, nose_c = kp_norm[0]
        ls_x, ls_y, ls_c = kp_norm[5]
        rs_x, rs_y, rs_c = kp_norm[6]
        lw_x, lw_y, lw_c = kp_norm[9]
        rw_x, rw_y, rw_c = kp_norm[10]

        # Pose confidence
        key_confs = [nose_c, ls_c, rs_c, lw_c, rw_c, kp_norm[7][2], kp_norm[8][2]]
        pose_confidence = float(np.mean(key_confs))

        conf_threshold = 0.3
        nose_valid = nose_c > conf_threshold
        ls_valid = ls_c > conf_threshold
        rs_valid = rs_c > conf_threshold
        lw_valid = lw_c > conf_threshold
        rw_valid = rw_c > conf_threshold

        # Convert person bbox to normalized coords for relative checks
        px1, py1, px2, py2 = person_bbox
        p_cx = ((px1 + px2) / 2) / w_frame  # Person center x, normalized

        # --- Hand raised (wrist above shoulder with margin) ---
        hand_raised = False
        if lw_valid and ls_valid and lw_y < ls_y - 0.03:
            hand_raised = True
        if rw_valid and rs_valid and rw_y < rs_y - 0.03:
            hand_raised = True

        # --- Head facing forward ---
        # Check if nose is roughly centered within the person's bbox
        head_forward = False
        if nose_valid:
            # Nose should be roughly between the shoulders (horizontally)
            if ls_valid and rs_valid:
                shoulder_cx = (ls_x + rs_x) / 2
                head_forward = abs(nose_x - shoulder_cx) < 0.08
            else:
                # Fallback: nose near person center
                head_forward = abs(nose_x - p_cx) < 0.1

        # --- Phone usage ---
        using_phone = False
        if ls_valid and rs_valid:
            mid_shoulder_y = (ls_y + rs_y) / 2.0
        else:
            mid_shoulder_y = 0.5

        if lw_valid and rw_valid and nose_valid:
            wrists_near_face = (
                lw_y < mid_shoulder_y and rw_y < mid_shoulder_y and
                lw_y > nose_y - 0.1 and rw_y > nose_y - 0.1
            )
            wrists_close = abs(lw_x - rw_x) < 0.15
            head_down = nose_y > mid_shoulder_y - 0.03

            using_phone = wrists_near_face and wrists_close and head_down

        if using_phone:
            head_forward = False

        return {
            "hand_raised": hand_raised,
            "head_forward": head_forward,
            "using_phone": using_phone,
            "pose_confidence": pose_confidence,
        }

    # Keep legacy single-person method for compatibility
    def analyze(self, frame, bbox, track_id: int = None):
        """Legacy single-person analyze (calls batch internally)."""
        persons = [{"track_id": track_id or 0, "bbox": list(bbox)}]
        results = self.analyze_batch(frame, persons)
        return results.get(track_id or 0, {
            "hand_raised": False, "head_forward": False,
            "using_phone": False, "pose_confidence": 0.0,
        })

    def close(self):
        pass