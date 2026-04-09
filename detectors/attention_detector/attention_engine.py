"""
Uncertainty-aware attention engine with hysteresis.

The engine keeps presence tracking separate from fine-grained attention so
back-row or weak-evidence students can remain present while their attention
state is explicitly marked as ``unknown``.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class AttentionEngine:
    def __init__(self, config: dict | None = None):
        config = config or {}
        att_cfg = config.get("attention", {})
        self.ema_alpha = float(att_cfg.get("ema_alpha", 0.10))
        self.enter_attentive = float(att_cfg.get("enter_attentive", 0.25))
        self.exit_attentive = float(att_cfg.get("exit_attentive", 0.05))
        self.enter_distracted = float(att_cfg.get("enter_distracted", -0.35))
        self.exit_distracted = float(att_cfg.get("exit_distracted", -0.05))
        self.min_state_hold = int(att_cfg.get("min_state_hold", 8))
        self.unknown_hold = int(att_cfg.get("unknown_state_hold", 2))
        self.enable_unknown = bool(att_cfg.get("enable_unknown_state", True))
        self.mode_min_evidence = {
            "full": float(att_cfg.get("full_min_evidence", 0.28)),
            "reduced": float(att_cfg.get("reduced_min_evidence", 0.36)),
            "limited": float(att_cfg.get("limited_min_evidence", 0.60)),
        }
        self.weights = {
            "looking_forward": float(att_cfg.get("weights", {}).get("looking_forward", 1.0)),
            "hand_raised": float(att_cfg.get("weights", {}).get("hand_raised", 1.0)),
            "distracted": float(att_cfg.get("weights", {}).get("distracted", -0.7)),
            "using_phone": float(att_cfg.get("weights", {}).get("using_phone", -2.0)),
        }

        self.ema_scores: dict[str, float] = {}
        self.current_state: dict[str, str] = {}
        self.candidate_state: dict[str, str] = {}
        self.state_hold_counter: dict[str, int] = defaultdict(int)
        self.last_result: dict[str, dict] = {}
        self.counters: dict[str, dict] = defaultdict(
            lambda: {
                "attentive_frames": 0,
                "distracted_frames": 0,
                "unknown_frames": 0,
                "confident_frames": 0,
                "handraise_frames": 0,
                "using_phone_frames": 0,
                "total_frames": 0,
                "cumulative_weighted_score": 0.0,
            }
        )

    def _compute_score(self, signals: dict) -> tuple[float, float, str]:
        mode = str(signals.get("size_mode", "limited"))
        det_conf = float(signals.get("detection_confidence", 0.0))
        pose_conf = float(signals.get("pose_confidence", 0.0))
        phone_obj_conf = float(signals.get("phone_object_confidence", 0.0))
        hand_raised = bool(signals.get("hand_raised", False))
        head_forward = bool(signals.get("head_forward", False))
        using_phone_pose = bool(signals.get("using_phone_pose", False))
        using_phone_object = bool(signals.get("using_phone_object", False))
        reasons: list[str] = [f"mode={mode}"]

        score = 0.0
        evidence = max(0.20 * det_conf, 0.0)

        if using_phone_object:
            score += self.weights["using_phone"] * max(phone_obj_conf, det_conf, 0.35)
            evidence = max(evidence, phone_obj_conf, det_conf)
            reasons.append("phone-object")

        if mode == "limited":
            if using_phone_object:
                return score, evidence, ";".join(reasons)
            reasons.append("presence-only")
            return 0.0, min(evidence, 0.45), ";".join(reasons)

        if hand_raised and pose_conf >= 0.20:
            score += self.weights["hand_raised"] * pose_conf
            evidence = max(evidence, pose_conf)
            reasons.append("hand-raised")

        if head_forward and pose_conf >= 0.20 and not using_phone_object:
            head_weight = self.weights["looking_forward"]
            if mode == "reduced":
                head_weight *= 0.85
            score += head_weight * pose_conf
            evidence = max(evidence, pose_conf)
            reasons.append("head-forward")

        if mode == "full" and using_phone_pose and pose_conf >= 0.25 and not using_phone_object:
            score += 0.65 * self.weights["using_phone"] * pose_conf
            evidence = max(evidence, pose_conf)
            reasons.append("phone-pose")

        if not hand_raised and not head_forward and not using_phone_object:
            if pose_conf >= self.mode_min_evidence.get(mode, 0.30):
                score += self.weights["distracted"] * pose_conf
                evidence = max(evidence, pose_conf)
                reasons.append("weak-engagement")
            else:
                reasons.append("weak-pose")

        if mode == "reduced" and using_phone_pose and not using_phone_object:
            reasons.append("pose-phone-weak")

        return score, min(1.0, evidence), ";".join(reasons)

    def _raw_state_from_score(self, person_id: str, ema_score: float, evidence: float, mode: str) -> str:
        min_evidence = self.mode_min_evidence.get(mode, 0.30)
        prev_state = self.current_state.get(person_id, "unknown")

        if self.enable_unknown and evidence < min_evidence:
            return "unknown"

        if prev_state == "attentive":
            if ema_score < self.enter_distracted:
                return "distracted"
            if ema_score < self.exit_attentive and self.enable_unknown and evidence < min_evidence + 0.05:
                return "unknown"
            return "attentive"

        if prev_state == "distracted":
            if ema_score > self.enter_attentive:
                return "attentive"
            if ema_score > self.exit_distracted and self.enable_unknown and evidence < min_evidence + 0.05:
                return "unknown"
            return "distracted"

        if ema_score >= self.enter_attentive:
            return "attentive"
        if ema_score <= self.enter_distracted:
            return "distracted"
        return "unknown" if self.enable_unknown else "attentive"

    @staticmethod
    def _state_confidence(state: str, ema_score: float, evidence: float) -> float:
        if state == "unknown":
            return max(0.35, min(0.95, 1.0 - 0.65 * evidence))
        margin = min(1.0, abs(ema_score) / 2.0)
        return max(0.30, min(0.99, 0.35 + 0.65 * margin * max(evidence, 0.25)))

    def update(self, person_id: str, signals: dict) -> dict:
        counters = self.counters[person_id]
        counters["total_frames"] += 1

        mode = str(signals.get("size_mode", "limited"))
        score, evidence, reason = self._compute_score(signals)
        counters["cumulative_weighted_score"] += score

        previous = self.ema_scores.get(person_id)
        if previous is None:
            ema_score = score
        else:
            ema_score = self.ema_alpha * score + (1.0 - self.ema_alpha) * previous
        self.ema_scores[person_id] = ema_score

        raw_state = self._raw_state_from_score(person_id, ema_score, evidence, mode)
        prev_state = self.current_state.get(person_id, "unknown")
        prev_candidate = self.candidate_state.get(person_id, prev_state)
        if raw_state == prev_candidate:
            self.state_hold_counter[person_id] += 1
        else:
            self.candidate_state[person_id] = raw_state
            self.state_hold_counter[person_id] = 1

        hold_needed = self.unknown_hold if raw_state == "unknown" else self.min_state_hold
        if raw_state != prev_state and self.state_hold_counter[person_id] >= hold_needed:
            confirmed = raw_state
        else:
            confirmed = prev_state
        self.current_state[person_id] = confirmed

        if bool(signals.get("hand_raised", False)):
            counters["handraise_frames"] += 1
        if bool(signals.get("using_phone_object", False) or signals.get("using_phone_pose", False)):
            counters["using_phone_frames"] += 1

        if confirmed == "attentive":
            counters["attentive_frames"] += 1
            counters["confident_frames"] += 1
        elif confirmed == "distracted":
            counters["distracted_frames"] += 1
            counters["confident_frames"] += 1
        else:
            counters["unknown_frames"] += 1

        result = {
            "attention_state": confirmed,
            "attention_confidence": round(self._state_confidence(confirmed, ema_score, evidence), 4),
            "attention_mode": mode,
            "attention_reason": reason,
            "attention_score": round(float(ema_score), 4),
        }
        self.last_result[person_id] = result
        return result

    def get_student_metrics(self, person_id: str) -> dict:
        counters = self.counters.get(
            person_id,
            {
                "attentive_frames": 0,
                "distracted_frames": 0,
                "unknown_frames": 0,
                "confident_frames": 0,
                "handraise_frames": 0,
                "using_phone_frames": 0,
                "total_frames": 0,
                "cumulative_weighted_score": 0.0,
            },
        )
        confident = counters["confident_frames"]
        attentive = counters["attentive_frames"]
        attention_pct = round((attentive / confident) * 100.0, 2) if confident > 0 else 0.0
        latest = self.last_result.get(
            person_id,
            {
                "attention_state": "unknown",
                "attention_confidence": 0.0,
                "attention_mode": "limited",
                "attention_reason": "no-observations",
            },
        )
        return {
            "attentive_frames": attentive,
            "distracted_frames": counters["distracted_frames"],
            "unknown_frames": counters["unknown_frames"],
            "confident_frames": confident,
            "handraise_frames": counters["handraise_frames"],
            "using_phone_frames": counters["using_phone_frames"],
            "total_frames": counters["total_frames"],
            "confidence_weighted_score": round(counters["cumulative_weighted_score"], 4),
            "attention_percentage": attention_pct,
            **latest,
        }

    def get_all_metrics(self) -> dict[str, dict]:
        return {person_id: self.get_student_metrics(person_id) for person_id in self.counters}

    def reset(self) -> None:
        self.ema_scores.clear()
        self.current_state.clear()
        self.candidate_state.clear()
        self.state_hold_counter.clear()
        self.last_result.clear()
        self.counters.clear()
        logger.info("AttentionEngine state reset.")

