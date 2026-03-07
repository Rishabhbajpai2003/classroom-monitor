"""
Face embedding extraction module (MediaPipe-free).

Uses a lightweight CNN (MobileNetV3-Small pretrained on ImageNet) to extract
normalized embedding vectors from person crops. These embeddings are used
for identity matching and persistence across sessions.

No face detection step — we use the full person crop (or upper portion)
directly, which acts as a holistic appearance descriptor for ReID.
"""

import logging

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """
    Extracts normalized appearance embeddings using MobileNetV3-Small (pretrained).

    The classifier head is removed, producing a 576-dim feature vector
    that is L2-normalized for cosine similarity comparison.

    Attributes:
        model: MobileNetV3 feature extractor (eval mode, no grad).
        device: Torch device.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the embedder.

        Args:
            device: Torch device string ('cpu', 'cuda', 'mps').
        """
        try:
            from torchvision import models, transforms

            self.device = torch.device(device if device != "auto" else "cpu")

            # Load pretrained MobileNetV3-Small
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            base_model = models.mobilenet_v3_small(weights=weights)

            # Remove classifier — keep only feature extractor
            self.model = nn.Sequential(
                base_model.features,
                base_model.avgpool,
                nn.Flatten(),
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            logger.info(f"FaceEmbedder initialized on {self.device}")
            self._available = True

        except Exception as e:
            logger.warning(
                f"FaceEmbedder initialization failed: {e}. "
                f"Identity persistence will be disabled."
            )
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @torch.no_grad()
    def extract_embedding(self, frame, bbox) -> np.ndarray | None:
        """
        Extract a normalized embedding vector from a person crop.

        Args:
            frame: Full BGR frame (numpy array).
            bbox: Bounding box as [x1, y1, x2, y2].

        Returns:
            1-D numpy array (576-dim, L2-normalized), or None on failure.
        """
        if not self._available:
            return None

        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]

            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Transform and run through model
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            embedding = self.model(tensor).squeeze(0).cpu().numpy()

            # L2-normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")
            return None


class FaceDetector:
    """
    Person appearance embedding extractor for identity persistence.

    Uses the upper portion of a person bounding box (where the face/head is)
    to extract a MobileNetV3-based embedding. No separate face detection
    needed — YOLO26 handles person detection.

    Attributes:
        embedder: FaceEmbedder for identity matching.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize FaceDetector (embedding extractor).

        Args:
            config: Full pipeline config dict. Uses 'system' section for device.
        """
        config = config or {}
        sys_cfg = config.get("system", {})

        # Determine device
        device = sys_cfg.get("device", "cpu")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.embedder = FaceEmbedder(device=device)

    def extract_embedding(self, frame, bbox) -> np.ndarray | None:
        """
        Extract appearance embedding from a person bounding box.

        Uses the upper 40% of the person crop (head/shoulders region)
        for a more discriminative embedding.

        Args:
            frame: Full BGR frame.
            bbox: Person bounding box [x1, y1, x2, y2].

        Returns:
            Normalized 576-dim embedding vector, or None.
        """
        if not self.embedder.is_available:
            return None

        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Use upper 40% of person crop (head + shoulders) for more
        # discriminative identity embedding
        upper_y2 = y1 + int((y2 - y1) * 0.4)
        upper_y2 = max(upper_y2, y1 + 10)  # ensure min height

        return self.embedder.extract_embedding(frame, [x1, y1, x2, upper_y2])

    def close(self):
        """No resources to release (no MediaPipe)."""
        pass