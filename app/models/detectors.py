"""
Detection modules for classroom monitoring pipeline.

Includes:
    - PersonDetector: YOLO26-based person detection with auto device selection.
    - BehaviorDetector, HandRaiseDetector, PoseDetector: retained from original.
"""

import logging
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> str:
    """
    Select the best available compute device.

    Args:
        preference: One of 'auto', 'cuda', 'mps', 'cpu'.

    Returns:
        Device string compatible with ultralytics and PyTorch.
    """
    if preference != "auto":
        return preference

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Auto-selected device: {device}")
    return device


class PersonDetector:
    """
    YOLO26-based person detector with confidence filtering and auto device selection.

    Attributes:
        model: Loaded YOLO model instance.
        device: Compute device string.
        conf_threshold: Minimum confidence to accept a detection.
        img_size: Input image size for inference.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize PersonDetector.

        Args:
            config: Full pipeline config dict. Uses 'detection' and 'system' sections.
        """
        config = config or {}
        det_cfg = config.get("detection", {})
        sys_cfg = config.get("system", {})

        model_name = det_cfg.get("model_size", "yolo26n")
        weight_path = f"weights/{model_name}.pt"
        self.conf_threshold = det_cfg.get("confidence_threshold", 0.5)
        self.img_size = det_cfg.get("image_size", 640)
        self.device = get_device(sys_cfg.get("device", "auto"))

        logger.info(f"Loading PersonDetector model: {weight_path} on {self.device}")
        self.model = YOLO(weight_path)

    def detect(self, frame):
        """
        Detect persons in a frame.

        Args:
            frame: BGR numpy array (H, W, 3).

        Returns:
            List of dicts, each with keys:
                'bbox': [x1, y1, x2, y2]
                'confidence': float
        """
        results = self.model(
            frame,
            device=self.device,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            verbose=False,
        )[0]

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                # Class 0 = person in COCO
                if cls_id != 0:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                })

        return detections


class BehaviorDetector:
    """YOLO-based behavior detector (retained from original pipeline)."""

    def __init__(self, weight_path: str):
        self.model = YOLO(weight_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        return results


class HandRaiseDetector:
    """YOLO-based hand-raise detector (retained from original pipeline)."""

    def __init__(self, weight_path: str):
        self.model = YOLO(weight_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        return results


class PoseDetector:
    """YOLO-based pose detector (retained from original pipeline)."""

    def __init__(self, weight_path: str = "yolo11n-pose.pt"):
        self.model = YOLO(weight_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        return results