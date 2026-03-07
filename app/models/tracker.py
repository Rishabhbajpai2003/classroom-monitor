"""
DeepSORT tracker with ReID embedder support for robust person tracking.

Configurable via the pipeline config dict with parameters for max_age,
cosine distance thresholds, and ReID embedder selection.
"""

import logging
from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)


class PersonTracker:
    """
    Wraps DeepSORT with configurable parameters and built-in ReID embedder.

    The ReID embedder (e.g., MobileNet) extracts appearance features from
    each detection crop, enabling re-identification across occlusions and
    reducing ID switches.

    Attributes:
        tracker: DeepSort instance.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize PersonTracker.

        Args:
            config: Full pipeline config dict. Uses 'tracking' and 'system' sections.
        """
        config = config or {}
        trk_cfg = config.get("tracking", {})
        sys_cfg = config.get("system", {})

        max_age = trk_cfg.get("max_age", 30)
        n_init = trk_cfg.get("min_hits", 3)
        max_cosine_distance = trk_cfg.get("max_cosine_distance", 0.4)
        nn_budget = trk_cfg.get("nn_budget", 100)
        embedder = trk_cfg.get("embedder", "mobilenet")

        # Determine if embedder should use GPU
        device_pref = sys_cfg.get("device", "auto")
        embedder_gpu = trk_cfg.get("embedder_gpu", True)
        if device_pref == "cpu":
            embedder_gpu = False

        logger.info(
            f"Initializing DeepSORT tracker: max_age={max_age}, n_init={n_init}, "
            f"max_cosine_distance={max_cosine_distance}, embedder={embedder}, "
            f"embedder_gpu={embedder_gpu}"
        )

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder=embedder,
            embedder_gpu=embedder_gpu,
        )

    def update(self, detections, frame):
        """
        Update tracks with new detections.

        Args:
            detections: List of tuples ([x, y, w, h], confidence, class_name).
            frame: Current BGR frame (numpy array).

        Returns:
            List of Track objects from DeepSORT.
        """
        return self.tracker.update_tracks(detections, frame=frame)

    def reset(self):
        """Reset tracker state (e.g., between videos)."""
        self.tracker.delete_all_tracks()
        logger.info("Tracker state reset.")