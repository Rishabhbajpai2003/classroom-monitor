from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class FrameChangeState:
    changed: bool
    score: float
    motion_stable: bool
    inlier_ratio: float


class FrameChangeGate:
    def __init__(
        self,
        *,
        diff_threshold: float = 0.14,
        unstable_inlier_ratio: float = 0.28,
        downsample_width: int = 160,
        max_features: int = 250,
    ) -> None:
        self.diff_threshold = float(diff_threshold)
        self.unstable_inlier_ratio = float(unstable_inlier_ratio)
        self.downsample_width = int(downsample_width)
        self.max_features = int(max_features)
        self.prev_gray_small: Optional[np.ndarray] = None
        self.orb = cv2.ORB_create(nfeatures=self.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _prepare_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if width <= self.downsample_width:
            return gray
        scale = float(self.downsample_width) / float(width)
        target = (self.downsample_width, max(1, int(round(height * scale))))
        return cv2.resize(gray, target, interpolation=cv2.INTER_AREA)

    def update(self, frame_bgr: np.ndarray) -> FrameChangeState:
        gray_small = self._prepare_gray(frame_bgr)
        if self.prev_gray_small is None:
            self.prev_gray_small = gray_small
            return FrameChangeState(changed=True, score=1.0, motion_stable=False, inlier_ratio=0.0)

        diff_score = float(np.mean(np.abs(gray_small.astype(np.float32) - self.prev_gray_small.astype(np.float32))) / 255.0)
        keypoints_prev, desc_prev = self.orb.detectAndCompute(self.prev_gray_small, None)
        keypoints_cur, desc_cur = self.orb.detectAndCompute(gray_small, None)
        inlier_ratio = 0.0
        motion_stable = False

        if (
            desc_prev is not None
            and desc_cur is not None
            and len(keypoints_prev) >= 12
            and len(keypoints_cur) >= 12
        ):
            matches = self.matcher.match(desc_prev, desc_cur)
            matches = sorted(matches, key=lambda item: item.distance)[:64]
            if len(matches) >= 12:
                src = np.float32([keypoints_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst = np.float32([keypoints_cur[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                _mat, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if inliers is not None and len(inliers) > 0:
                    inlier_ratio = float(np.mean(inliers.astype(np.float32)))
                    motion_stable = inlier_ratio >= self.unstable_inlier_ratio

        score = 0.65 * diff_score + 0.35 * (1.0 - inlier_ratio if inlier_ratio > 0.0 else diff_score)
        changed = bool(score >= self.diff_threshold or (inlier_ratio > 0.0 and inlier_ratio < self.unstable_inlier_ratio))
        self.prev_gray_small = gray_small
        return FrameChangeState(
            changed=changed,
            score=float(score),
            motion_stable=motion_stable,
            inlier_ratio=float(inlier_ratio),
        )
