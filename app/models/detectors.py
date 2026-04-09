"""
Shared detector utilities for recall-first classroom processing.

This module provides tiled YOLO detection helpers plus concrete detectors for:
    - person detection
    - open-vocabulary object detection (YOLO-World style)

The detectors are intentionally lazy-loaded so unit tests can import the helper
functions without needing the heavyweight inference stack available at import
time.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_model_path(model_name: str, default_dir: str = "weights") -> str:
    if not model_name:
        raise ValueError("model_name must be provided")
    if os.path.exists(model_name):
        return model_name
    if model_name.endswith(".pt"):
        return model_name
    return os.path.join(default_dir, f"{model_name}.pt")


def bbox_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
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


def center_distance_ratio(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)
    dist = float(np.hypot(acx - bcx, acy - bcy))
    scale = max(1.0, max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1))
    return dist / scale


def generate_overlapping_tiles(
    frame_shape: tuple[int, int, int],
    tile_grid: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    if tile_grid <= 1:
        return [(0, 0, frame_shape[1], frame_shape[0])]

    frame_h, frame_w = frame_shape[:2]
    overlap = float(np.clip(overlap, 0.0, 0.45))
    step_x = frame_w / float(tile_grid)
    step_y = frame_h / float(tile_grid)
    pad_x = int(round(step_x * overlap))
    pad_y = int(round(step_y * overlap))

    tiles: list[tuple[int, int, int, int]] = []
    for gy in range(tile_grid):
        for gx in range(tile_grid):
            x1 = max(0, int(round(gx * step_x)) - pad_x)
            y1 = max(0, int(round(gy * step_y)) - pad_y)
            x2 = min(frame_w, int(round((gx + 1) * step_x)) + pad_x)
            y2 = min(frame_h, int(round((gy + 1) * step_y)) + pad_y)
            if x2 > x1 and y2 > y1:
                tiles.append((x1, y1, x2, y2))
    return tiles


def deduplicate_box_detections(
    detections: list[dict],
    iou_thresh: float = 0.55,
    center_ratio_thresh: float = 0.25,
    class_aware: bool = True,
) -> list[dict]:
    if not detections:
        return []

    ordered = sorted(
        detections,
        key=lambda det: (
            float(det.get("confidence", 0.0)),
            float(det.get("area", 0.0)),
            1 if det.get("source") == "global" else 0,
        ),
        reverse=True,
    )
    kept: list[dict] = []
    for det in ordered:
        duplicate = False
        for existing in kept:
            if class_aware and det.get("class_name") != existing.get("class_name"):
                continue
            iou = bbox_iou(det["bbox"], existing["bbox"])
            if iou >= iou_thresh:
                duplicate = True
                break
            if iou > 0.15 and center_distance_ratio(det["bbox"], existing["bbox"]) <= center_ratio_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


@dataclass
class DetectorConfig:
    weights: str
    conf: float
    imgsz: int
    tile_grid: int = 1
    tile_overlap: float = 0.20
    device: str = "auto"
    max_det: Optional[int] = None
    iou: Optional[float] = None


class _LazyYOLODetector:
    def __init__(self, detector_config: DetectorConfig):
        self.cfg = detector_config
        self.device = resolve_device(detector_config.device)
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO

            logger.info("Loading YOLO model %s on %s", self.cfg.weights, self.device)
            self._model = YOLO(self.cfg.weights)
            self._configure_model()
        return self._model

    def _configure_model(self) -> None:
        # Subclasses may override.
        return None

    def _predict_tile(self, frame: np.ndarray):
        model = self._load_model()
        kwargs = {
            "conf": self.cfg.conf,
            "imgsz": self.cfg.imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.cfg.max_det is not None:
            kwargs["max_det"] = self.cfg.max_det
        if self.cfg.iou is not None:
            kwargs["iou"] = self.cfg.iou
        return model.predict(frame, **kwargs)[0]

    def _extract_result_detections(
        self,
        result,
        offset_x: int = 0,
        offset_y: int = 0,
        source: str = "global",
    ) -> list[dict]:
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> list[dict]:
        detections = self._extract_result_detections(self._predict_tile(frame), source="global")
        if self.cfg.tile_grid > 1:
            tiles = generate_overlapping_tiles(frame.shape, self.cfg.tile_grid, self.cfg.tile_overlap)
            for x1, y1, x2, y2 in tiles:
                if x1 == 0 and y1 == 0 and x2 == frame.shape[1] and y2 == frame.shape[0]:
                    continue
                tile = frame[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                detections.extend(
                    self._extract_result_detections(
                        self._predict_tile(tile),
                        offset_x=x1,
                        offset_y=y1,
                        source="tile",
                    )
                )
        deduped = deduplicate_box_detections(detections)
        deduped.sort(key=lambda det: float(det.get("confidence", 0.0)), reverse=True)
        return deduped


class PersonDetector(_LazyYOLODetector):
    def __init__(self, config: dict | None = None):
        config = config or {}
        det_cfg = config.get("detection", {})
        sys_cfg = config.get("system", {})
        model_name = normalize_model_path(det_cfg.get("model_size", "yolo26s"))
        super().__init__(
            DetectorConfig(
                weights=model_name,
                conf=float(det_cfg.get("confidence_threshold", 0.22)),
                imgsz=int(det_cfg.get("image_size", 1536)),
                tile_grid=int(det_cfg.get("tile_grid", 2)),
                tile_overlap=float(det_cfg.get("tile_overlap", 0.20)),
                device=str(sys_cfg.get("device", "auto")),
                max_det=int(det_cfg.get("max_det", 300)),
                iou=float(det_cfg.get("iou_threshold", 0.65)),
            )
        )

    def _extract_result_detections(
        self,
        result,
        offset_x: int = 0,
        offset_y: int = 0,
        source: str = "global",
    ) -> list[dict]:
        detections: list[dict] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        names = result.names if isinstance(result.names, dict) else {i: n for i, n in enumerate(result.names)}
        for box, conf, cls_id in zip(xyxy, confs, classes):
            class_name = str(names.get(int(cls_id), cls_id)).lower()
            if cls_id != 0 and class_name != "person":
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 += float(offset_x)
            x2 += float(offset_x)
            y1 += float(offset_y)
            y2 += float(offset_y)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": "person",
                    "area": area,
                    "source": source,
                }
            )
        return detections


class OpenVocabularyObjectDetector(_LazyYOLODetector):
    def __init__(
        self,
        target_classes: list[str],
        config: dict | None = None,
        config_section: str = "object_detection",
        default_weights: str = "weights/yolov8x-worldv2.pt",
    ):
        config = config or {}
        det_cfg = config.get(config_section, {})
        sys_cfg = config.get("system", {})
        self.target_classes = [str(name).strip().lower() for name in target_classes if str(name).strip()]
        super().__init__(
            DetectorConfig(
                weights=det_cfg.get("weights", default_weights),
                conf=float(det_cfg.get("confidence_threshold", 0.14)),
                imgsz=int(det_cfg.get("image_size", 1280)),
                tile_grid=int(det_cfg.get("tile_grid", 2)),
                tile_overlap=float(det_cfg.get("tile_overlap", 0.20)),
                device=str(sys_cfg.get("device", "auto")),
                max_det=int(det_cfg.get("max_det", 400)),
                iou=float(det_cfg.get("iou_threshold", 0.65)),
            )
        )

    def _configure_model(self) -> None:
        if self.target_classes:
            self._model.set_classes(self.target_classes)

    def _extract_result_detections(
        self,
        result,
        offset_x: int = 0,
        offset_y: int = 0,
        source: str = "global",
    ) -> list[dict]:
        detections: list[dict] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        names = result.names if isinstance(result.names, dict) else {i: n for i, n in enumerate(result.names)}
        for box, conf, cls_id in zip(xyxy, confs, classes):
            class_name = str(names.get(int(cls_id), cls_id)).lower()
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 += float(offset_x)
            x2 += float(offset_x)
            y1 += float(offset_y)
            y2 += float(offset_y)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "area": area,
                    "source": source,
                }
            )
        return detections

