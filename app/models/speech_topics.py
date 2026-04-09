"""
Transcript topic classification for classroom-related speech detection.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml


def _normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _count_phrase_hits(text: str, phrases: list[str]) -> tuple[int, list[str]]:
    hits = []
    for phrase in phrases:
        token = _normalize_text(phrase)
        if token and token in text:
            hits.append(token)
    return len(hits), sorted(set(hits))


def load_topic_profiles(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload.get("profiles", payload)


class SpeechTopicClassifier:
    def __init__(self, config_path: str | Path, course_profile: str = "default"):
        self.config_path = str(config_path)
        self.profiles = load_topic_profiles(config_path)
        if course_profile not in self.profiles:
            raise KeyError(f"Unknown course profile: {course_profile}")
        self.course_profile = course_profile
        self.profile = self._expand_profile(self.profiles[course_profile])

    @staticmethod
    def _expand_profile(profile: dict) -> dict:
        aliases = profile.get("aliases", {})

        def expand(items: list[str]) -> list[str]:
            expanded = []
            seen = set()
            for item in items or []:
                token = _normalize_text(item)
                if token and token not in seen:
                    seen.add(token)
                    expanded.append(token)
                for alias in aliases.get(item, []) + aliases.get(token, []):
                    alias_token = _normalize_text(alias)
                    if alias_token and alias_token not in seen:
                        seen.add(alias_token)
                        expanded.append(alias_token)
            return expanded

        return {
            "required_keywords": expand(profile.get("required_keywords", [])),
            "supporting_keywords": expand(profile.get("supporting_keywords", [])),
            "off_topic_keywords": expand(profile.get("off_topic_keywords", [])),
        }

    def classify(
        self,
        transcript: str,
        *,
        mean_speech_prob: float,
        min_quality: float = 0.55,
        min_tokens: int = 2,
    ) -> dict:
        normalized = _normalize_text(transcript)
        token_count = len(normalized.split()) if normalized else 0
        if mean_speech_prob < min_quality or token_count < min_tokens:
            return {
                "topic_label": "unknown",
                "topic_score": 0.0,
                "topic_reason": "low_quality_transcript",
                "matched_required": [],
                "matched_supporting": [],
                "matched_off_topic": [],
            }

        required_count, matched_required = _count_phrase_hits(normalized, self.profile["required_keywords"])
        supporting_count, matched_supporting = _count_phrase_hits(normalized, self.profile["supporting_keywords"])
        off_count, matched_off = _count_phrase_hits(normalized, self.profile["off_topic_keywords"])

        if required_count >= 1:
            return {
                "topic_label": "class_related",
                "topic_score": min(1.0, 0.75 + 0.10 * required_count),
                "topic_reason": "required_keyword_match",
                "matched_required": matched_required,
                "matched_supporting": matched_supporting,
                "matched_off_topic": matched_off,
            }

        if supporting_count >= 2 and supporting_count > off_count:
            return {
                "topic_label": "class_related",
                "topic_score": min(1.0, 0.55 + 0.08 * supporting_count),
                "topic_reason": "supporting_keyword_combo",
                "matched_required": matched_required,
                "matched_supporting": matched_supporting,
                "matched_off_topic": matched_off,
            }

        if off_count > max(required_count, supporting_count):
            return {
                "topic_label": "off_topic",
                "topic_score": min(1.0, 0.55 + 0.10 * off_count),
                "topic_reason": "off_topic_keywords_dominate",
                "matched_required": matched_required,
                "matched_supporting": matched_supporting,
                "matched_off_topic": matched_off,
            }

        return {
            "topic_label": "unknown",
            "topic_score": 0.0,
            "topic_reason": "insufficient_topic_signal",
            "matched_required": matched_required,
            "matched_supporting": matched_supporting,
            "matched_off_topic": matched_off,
        }

