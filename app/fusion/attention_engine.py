"""
Confidence-Weighted Attention Engine with Hysteresis.

Computes attention scores per student using detection and pose confidences
as weights, with EMA (Exponential Moving Average) smoothing and hysteresis
to prevent rapid state flickering.

Key anti-flicker mechanisms:
    1. EMA smoothing (alpha=0.15) instead of simple rolling average
    2. Hysteresis thresholds (different thresholds for entering vs leaving a state)
    3. Minimum state hold (a student must stay in a state for N frames before switching)
"""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class AttentionEngine:
    """
    Confidence-weighted attention scoring with EMA smoothing and hysteresis.

    Anti-flicker design:
        - EMA smoothing (α=0.15) gives more weight to recent frames while
          dampening single-frame noise.
        - Hysteresis: A student classified as "attentive" stays attentive until
          their score drops below a LOWER threshold (not the same threshold).
        - Minimum hold: State must persist for at least N consecutive frames
          before the counters are incremented, preventing brief flickers from
          polluting the data.

    Attributes:
        ema_alpha: EMA smoothing factor (higher = more responsive, lower = smoother).
        weights: Dict of behavior weights.
        ema_scores: Per-student EMA-smoothed score.
        current_state: Per-student current attention state (with hysteresis).
        state_hold_counter: Frames the student has been in the candidate state.
        counters: Per-student frame counters.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize AttentionEngine.

        Args:
            config: Full pipeline config dict. Uses 'attention' section.
        """
        config = config or {}
        att_cfg = config.get("attention", {})

        self.confidence_weighted = att_cfg.get("confidence_weighted", True)

        # EMA smoothing factor — lower = smoother, higher = more responsive
        # 0.08 is very smooth — takes ~12 frames to converge
        self.ema_alpha = att_cfg.get("ema_alpha", 0.08)

        # Hysteresis thresholds: different thresholds for entering vs leaving
        # A student becomes "attentive" when score > enter_attentive (0.2)
        # A student stays "attentive" until score < exit_attentive (0.0)
        # A student becomes "distracted" when score < enter_distracted (-0.2)
        # A student stays "distracted" until score > exit_distracted (0.0)
        self.enter_attentive = att_cfg.get("enter_attentive", 0.2)
        self.exit_attentive = att_cfg.get("exit_attentive", 0.0)
        self.enter_distracted = att_cfg.get("enter_distracted", -0.2)
        self.exit_distracted = att_cfg.get("exit_distracted", 0.0)

        # Minimum consecutive frames in a state before counters are updated
        # 15 frames ≈ ~1 second of consistent state needed to transition
        self.min_state_hold = att_cfg.get("min_state_hold", 15)

        default_weights = {
            "looking_forward": 1.0,
            "hand_raised": 1.0,
            "distracted": -1.0,
            "using_phone": -2.0,
        }
        self.weights = att_cfg.get("weights", default_weights)

        # Per-student EMA-smoothed attention score
        self.ema_scores: dict[str, float] = {}

        # Per-student current confirmed state (after hysteresis)
        self.current_state: dict[str, str] = {}

        # Per-student: how many consecutive frames the candidate state has held
        self.state_hold_counter: dict[str, int] = defaultdict(int)

        # Per-student: the candidate state being tested
        self.candidate_state: dict[str, str] = {}

        # Per-student frame counters
        self.counters: dict[str, dict] = defaultdict(lambda: {
            "attentive_frames": 0,
            "distracted_frames": 0,
            "handraise_frames": 0,
            "using_phone_frames": 0,
            "total_frames": 0,
            "cumulative_weighted_score": 0.0,
        })

    def update(self, person_id: str, signals: dict) -> str:
        """
        Update attention score for a person based on behavior signals.

        Uses EMA smoothing + hysteresis for stable state classification.

        Args:
            person_id: Global student ID.
            signals: Dict with keys:
                'hand_raised': bool
                'head_forward': bool
                'using_phone': bool
                'pose_confidence': float (0-1)
                'detection_confidence': float (0-1)

        Returns:
            Attention status string: 'attentive', 'distracted', or 'neutral'.
        """
        hand_raised = signals.get("hand_raised", False)
        head_forward = signals.get("head_forward", False)
        using_phone = signals.get("using_phone", False)
        pose_conf = signals.get("pose_confidence", 0.5)
        det_conf = signals.get("detection_confidence", 0.5)

        counters = self.counters[person_id]
        counters["total_frames"] += 1

        # --- Compute raw score for this frame ---
        score = 0.0

        if self.confidence_weighted:
            if hand_raised:
                score += self.weights["hand_raised"] * det_conf
            if head_forward and not using_phone:
                score += self.weights["looking_forward"] * pose_conf
            if using_phone:
                score += self.weights["using_phone"] * det_conf
            if not head_forward and not hand_raised and not using_phone:
                score += self.weights["distracted"] * pose_conf
        else:
            if hand_raised:
                score += 2.0
            if head_forward:
                score += 1.0
            if using_phone:
                score -= 2.0
            if not head_forward and not hand_raised:
                score -= 1.0

        counters["cumulative_weighted_score"] += score

        # --- EMA smoothing ---
        if person_id not in self.ema_scores:
            self.ema_scores[person_id] = score
        else:
            self.ema_scores[person_id] = (
                self.ema_alpha * score +
                (1 - self.ema_alpha) * self.ema_scores[person_id]
            )

        ema_score = self.ema_scores[person_id]

        # --- Hysteresis state classification ---
        # Default initial state is "attentive" (student in class = likely paying attention)
        prev_state = self.current_state.get(person_id, "attentive")

        # Determine the raw candidate state from score.
        # KEY DESIGN: No "neutral" oscillation — when score is in the dead zone,
        # we KEEP the previous state. This eliminates attentive↔neutral flicker.
        if prev_state == "attentive":
            # Already attentive — only leave if score drops significantly
            if ema_score < self.enter_distracted:
                raw_state = "distracted"
            else:
                # Stay attentive (even in neutral zone)
                raw_state = "attentive"
        elif prev_state == "distracted":
            # Already distracted — only leave if score rises significantly
            if ema_score > self.enter_attentive:
                raw_state = "attentive"
            else:
                # Stay distracted (even in neutral zone)
                raw_state = "distracted"
        else:
            # First frame or truly neutral — classify based on score
            if ema_score > self.enter_attentive:
                raw_state = "attentive"
            elif ema_score < self.enter_distracted:
                raw_state = "distracted"
            else:
                # Default to attentive (student is in class)
                raw_state = "attentive"

        # --- Minimum hold: require N consecutive frames in the candidate state ---
        prev_candidate = self.candidate_state.get(person_id, prev_state)

        if raw_state == prev_candidate:
            self.state_hold_counter[person_id] += 1
        else:
            # New candidate state — reset counter
            self.candidate_state[person_id] = raw_state
            self.state_hold_counter[person_id] = 1

        # Only transition if held for min_state_hold frames
        if (raw_state != prev_state and
                self.state_hold_counter[person_id] >= self.min_state_hold):
            confirmed_state = raw_state
            self.current_state[person_id] = confirmed_state
        else:
            confirmed_state = prev_state
            self.current_state[person_id] = confirmed_state

        # --- Update counters based on confirmed stable state ---
        if hand_raised:
            counters["handraise_frames"] += 1
        if using_phone:
            counters["using_phone_frames"] += 1

        if confirmed_state == "attentive":
            counters["attentive_frames"] += 1
        elif confirmed_state == "distracted":
            counters["distracted_frames"] += 1

        return confirmed_state

    def get_student_metrics(self, person_id: str) -> dict:
        """
        Get comprehensive attention metrics for a student.

        Args:
            person_id: Global student ID.

        Returns:
            Dict with all counters, weighted score, and attention percentage.
        """
        counters = self.counters.get(person_id, {
            "attentive_frames": 0,
            "distracted_frames": 0,
            "handraise_frames": 0,
            "using_phone_frames": 0,
            "total_frames": 0,
            "cumulative_weighted_score": 0.0,
        })

        total = counters["total_frames"]
        attentive = counters["attentive_frames"]
        attention_pct = round((attentive / total) * 100, 2) if total > 0 else 0.0

        return {
            "attentive_frames": attentive,
            "distracted_frames": counters["distracted_frames"],
            "handraise_frames": counters["handraise_frames"],
            "using_phone_frames": counters["using_phone_frames"],
            "total_frames": total,
            "confidence_weighted_score": round(
                counters["cumulative_weighted_score"], 4
            ),
            "attention_percentage": attention_pct,
        }

    def get_all_metrics(self) -> dict[str, dict]:
        """Get metrics for all tracked students."""
        return {pid: self.get_student_metrics(pid) for pid in self.counters}

    def reset(self):
        """Reset all state."""
        self.ema_scores.clear()
        self.current_state.clear()
        self.state_hold_counter.clear()
        self.candidate_state.clear()
        self.counters.clear()
        logger.info("AttentionEngine state reset.")