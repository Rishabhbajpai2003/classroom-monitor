from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


DEFAULT_POSE_LABELS = [
    "frontal",
    "left_profile",
    "right_profile",
    "slight_tilt",
    "up_angle",
    "down_angle",
]
DEFAULT_ACCESSORY_LABELS = ["sunglasses", "cap", "mask", "scarf"]


@dataclass
class FaceAttributePrediction:
    pose_bucket: str = "frontal"
    pose_confidence: float = 0.0
    accessory_scores: Dict[str, float] = field(default_factory=dict)

    def confident_accessories(self, threshold: float = 0.55) -> List[str]:
        return [
            label
            for label, score in sorted(self.accessory_scores.items())
            if float(score) >= float(threshold)
        ]

    def signature(self, threshold: float = 0.55) -> str:
        accessories = self.confident_accessories(threshold=threshold)
        if accessories:
            return f"{self.pose_bucket}|{'+'.join(accessories)}"
        return self.pose_bucket


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    denom = float(np.sum(exps))
    if denom <= 0.0:
        return np.zeros_like(values)
    return exps / denom


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _crop_band(gray: np.ndarray, start_ratio: float, end_ratio: float) -> np.ndarray:
    if gray.size == 0:
        return gray
    height = gray.shape[0]
    y1 = max(0, min(height - 1, int(round(height * float(start_ratio)))))
    y2 = max(y1 + 1, min(height, int(round(height * float(end_ratio)))))
    return gray[y1:y2, :]


def _edge_density(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    edges = cv2.Canny(gray, 80, 160)
    return float(np.mean(edges > 0))


def _variance_score(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    return float(np.std(gray) / 64.0)


class LocalFaceAttributeClassifier:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.weights_path = str(weights_path or "").strip()
        self.config_path = str(config_path or "").strip()
        self.session = None
        self.input_name = ""
        self.pose_labels = list(DEFAULT_POSE_LABELS)
        self.accessory_labels = list(DEFAULT_ACCESSORY_LABELS)
        self.input_size = 224
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        self.pose_output_name = ""
        self.accessory_output_name = ""
        self.enabled = False
        self._load()

    def _load(self) -> None:
        if self.config_path and Path(self.config_path).exists():
            with Path(self.config_path).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.pose_labels = list(payload.get("pose_labels", self.pose_labels))
            self.accessory_labels = list(payload.get("accessory_labels", self.accessory_labels))
            self.input_size = int(payload.get("input_size", self.input_size))
            self.pose_output_name = str(payload.get("pose_output_name", "") or "")
            self.accessory_output_name = str(payload.get("accessory_output_name", "") or "")

        if not self.weights_path or ort is None or not Path(self.weights_path).exists():
            return

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.weights_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        if outputs:
            self.pose_output_name = self.pose_output_name or outputs[0].name
        if len(outputs) > 1:
            self.accessory_output_name = self.accessory_output_name or outputs[1].name
        self.enabled = bool(self.pose_output_name and self.accessory_output_name)

    def predict(self, crop_bgr: np.ndarray) -> FaceAttributePrediction:
        return self.predict_with_context(crop_bgr)

    def predict_with_context(
        self,
        crop_bgr: np.ndarray,
        *,
        landmarks: Optional[Sequence[Sequence[float]]] = None,
        bbox: Optional[Sequence[float]] = None,
        quality_profile: Optional[Dict[str, object]] = None,
    ) -> FaceAttributePrediction:
        if crop_bgr is None or crop_bgr.size == 0:
            return FaceAttributePrediction(
                pose_bucket="frontal",
                pose_confidence=0.0,
                accessory_scores={label: 0.0 for label in self.accessory_labels},
            )
        if not self.enabled or self.session is None:
            return self._heuristic_predict(crop_bgr, landmarks=landmarks, bbox=bbox, quality_profile=quality_profile)

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - self.mean[None, None, :]) / self.std[None, None, :]
        tensor = tensor.transpose(2, 0, 1)[None, ...]

        outputs = self.session.run(
            [self.pose_output_name, self.accessory_output_name],
            {self.input_name: tensor},
        )
        pose_probs = _softmax(np.asarray(outputs[0]).reshape(-1))
        accessory_probs = _sigmoid(np.asarray(outputs[1]).reshape(-1))
        pose_idx = int(np.argmax(pose_probs)) if pose_probs.size else 0
        pose_bucket = self.pose_labels[min(max(0, pose_idx), len(self.pose_labels) - 1)]
        accessory_scores = {}
        for idx, label in enumerate(self.accessory_labels):
            score = float(accessory_probs[idx]) if idx < accessory_probs.size else 0.0
            accessory_scores[label] = score
        return FaceAttributePrediction(
            pose_bucket=pose_bucket,
            pose_confidence=float(pose_probs[pose_idx]) if pose_probs.size else 0.0,
            accessory_scores=accessory_scores,
        )

    def _heuristic_predict(
        self,
        crop_bgr: np.ndarray,
        *,
        landmarks: Optional[Sequence[Sequence[float]]] = None,
        bbox: Optional[Sequence[float]] = None,
        quality_profile: Optional[Dict[str, object]] = None,
    ) -> FaceAttributePrediction:
        pose_bucket, pose_confidence = self._heuristic_pose_bucket(landmarks=landmarks, bbox=bbox)
        accessory_scores = self._heuristic_accessory_scores(crop_bgr, quality_profile=quality_profile)
        return FaceAttributePrediction(
            pose_bucket=pose_bucket,
            pose_confidence=pose_confidence,
            accessory_scores=accessory_scores,
        )

    def _heuristic_pose_bucket(
        self,
        *,
        landmarks: Optional[Sequence[Sequence[float]]],
        bbox: Optional[Sequence[float]],
    ) -> tuple[str, float]:
        if landmarks is None or bbox is None:
            return "frontal", 0.22
        kps = np.asarray(landmarks, dtype=np.float32)
        box = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if kps.shape[0] < 5 or box.size < 4:
            return "frontal", 0.22
        width = max(1.0, float(box[2] - box[0]))
        height = max(1.0, float(box[3] - box[1]))
        rel = np.empty_like(kps[:5], dtype=np.float32)
        rel[:, 0] = (kps[:5, 0] - float(box[0])) / width
        rel[:, 1] = (kps[:5, 1] - float(box[1])) / height
        left_eye, right_eye, nose, mouth_left, mouth_right = rel[:5]
        eye_mid = 0.5 * (left_eye + right_eye)
        mouth_mid = 0.5 * (mouth_left + mouth_right)
        eye_dx = max(1e-4, float(right_eye[0] - left_eye[0]))
        eye_slope = float((right_eye[1] - left_eye[1]) / eye_dx)
        nose_x = float(nose[0] - eye_mid[0])
        nose_y_ratio = float((nose[1] - eye_mid[1]) / max(1e-4, mouth_mid[1] - eye_mid[1]))

        if abs(eye_slope) >= 0.10:
            return "slight_tilt", _clamp(0.56 + min(0.25, abs(eye_slope)), 0.0, 1.0)
        if nose_x >= 0.045:
            return "left_profile", _clamp(0.58 + min(0.22, abs(nose_x) * 2.5), 0.0, 1.0)
        if nose_x <= -0.045:
            return "right_profile", _clamp(0.58 + min(0.22, abs(nose_x) * 2.5), 0.0, 1.0)
        if nose_y_ratio <= 0.46:
            return "up_angle", _clamp(0.52 + (0.46 - nose_y_ratio) * 0.9, 0.0, 1.0)
        if nose_y_ratio >= 0.64:
            return "down_angle", _clamp(0.52 + (nose_y_ratio - 0.64) * 0.9, 0.0, 1.0)
        return "frontal", 0.62

    def _heuristic_accessory_scores(
        self,
        crop_bgr: np.ndarray,
        *,
        quality_profile: Optional[Dict[str, object]],
    ) -> Dict[str, float]:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, _w = gray.shape[:2]
        if h <= 0:
            return {label: 0.0 for label in self.accessory_labels}

        top_band = _crop_band(gray, 0.00, 0.24)
        eye_band = _crop_band(gray, 0.22, 0.48)
        lower_band = _crop_band(gray, 0.52, 0.84)
        bottom_band = _crop_band(gray, 0.72, 1.00)

        top_dark = 1.0 - float(np.mean(top_band) / 255.0) if top_band.size else 0.0
        eye_dark = 1.0 - float(np.mean(eye_band) / 255.0) if eye_band.size else 0.0
        lower_uniform = 1.0 - _clamp(_variance_score(lower_band), 0.0, 1.0)
        bottom_color = 0.0
        if bottom_band.size:
            bottom_rgb = rgb[int(round(h * 0.72)) :, :, :]
            if bottom_rgb.size:
                bottom_color = _clamp(float(np.std(bottom_rgb.astype(np.float32), axis=(0, 1)).mean()) / 48.0, 0.0, 1.0)
        edge_penalty = _edge_density(lower_band)
        shadow_score = float((quality_profile or {}).get("shadow_severity", 0.0) or 0.0)
        backlight_score = float((quality_profile or {}).get("backlight_score", 0.0) or 0.0)

        sunglasses = _clamp(0.82 * eye_dark + 0.20 * shadow_score - 0.16 * backlight_score, 0.0, 1.0)
        cap = _clamp(0.92 * top_dark + 0.10 * shadow_score, 0.0, 1.0)
        mask = _clamp(0.72 * lower_uniform + 0.14 * (1.0 - edge_penalty), 0.0, 1.0)
        scarf = _clamp(0.48 * bottom_color + 0.24 * top_dark + 0.18 * lower_uniform, 0.0, 1.0)

        return {
            "sunglasses": sunglasses,
            "cap": cap,
            "mask": mask,
            "scarf": scarf,
        }
