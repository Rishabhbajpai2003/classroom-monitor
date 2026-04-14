from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return arr
    return arr / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def weighted_average_embeddings(vectors: Sequence[np.ndarray], qualities: Sequence[float]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    weights = [max(0.05, float(q)) for q in qualities]
    if len(weights) != len(vectors):
        weights = [1.0] * len(vectors)
    stacked = np.vstack([l2_normalize(item) for item in vectors]).astype(np.float32)
    weight_arr = np.asarray(weights, dtype=np.float32)
    weight_arr = weight_arr / max(1e-6, float(np.sum(weight_arr)))
    avg = np.sum(stacked * weight_arr[:, None], axis=0)
    return l2_normalize(avg.astype(np.float32))


def current_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _timestamp_sort_value(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except Exception:
        pass
    try:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except Exception:
        return 0.0


def infer_size_bucket(face_size: float) -> str:
    size = float(face_size)
    if size < 72.0:
        return "small_face"
    if size < 128.0:
        return "medium_face"
    return "large_face"


def infer_lighting_bucket(mean_brightness: float, shadow_severity: float = 0.0) -> str:
    brightness = float(mean_brightness)
    shadow = float(shadow_severity)
    if brightness < 70.0 or shadow >= 0.48:
        return "low_light"
    if brightness > 180.0:
        return "bright_light"
    return "balanced_light"


def normalize_bank_family(metadata: Optional[Dict[str, object]] = None) -> str:
    cleaned = dict(metadata or {})
    combo = str(cleaned.get("combination_tag", "") or "").strip()
    if combo:
        return f"combo/{combo}"

    family = str(cleaned.get("augmentation_family", "") or "").strip()
    tag = str(cleaned.get("augmentation_tag", "") or "").strip()
    if family and tag:
        return f"{family}/{tag}"
    if family:
        return family
    return "base"


def infer_profile_bucket(bbox: Sequence[float], landmarks: Optional[np.ndarray]) -> str:
    if landmarks is None or len(landmarks) < 3:
        return "frontal"
    try:
        x1, _, x2, _ = [float(v) for v in bbox]
        width = max(1.0, x2 - x1)
        left_eye = np.asarray(landmarks[0], dtype=np.float32)
        right_eye = np.asarray(landmarks[1], dtype=np.float32)
        nose = np.asarray(landmarks[2], dtype=np.float32)
        eye_mid_x = 0.5 * float(left_eye[0] + right_eye[0])
        nose_offset = float(nose[0] - eye_mid_x) / width
    except Exception:
        return "frontal"
    if nose_offset <= -0.04:
        return "left_profile"
    if nose_offset >= 0.04:
        return "right_profile"
    return "frontal"


def sanitize_embedding_metadata(
    metadata: Optional[Dict[str, object]] = None,
    *,
    quality: float = 0.0,
    face_size: Optional[float] = None,
    profile_bucket: Optional[str] = None,
    size_bucket: Optional[str] = None,
    added_at: Optional[str] = None,
) -> Dict[str, object]:
    cleaned = dict(metadata or {})
    cleaned["quality"] = float(cleaned.get("quality", quality))
    if face_size is None:
        face_size = cleaned.get("face_size")
    if face_size is not None:
        cleaned["face_size"] = float(face_size)
    profile = str(cleaned.get("profile_bucket", "") or profile_bucket or "").strip()
    if not profile:
        profile = "frontal"
    cleaned["profile_bucket"] = profile
    size = str(cleaned.get("size_bucket", "") or size_bucket or "").strip()
    if not size:
        size = infer_size_bucket(float(cleaned.get("face_size", 0.0) or 0.0))
    cleaned["size_bucket"] = size
    cleaned["detector_used"] = str(cleaned.get("detector_used", "") or "scrfd").strip() or "scrfd"
    cleaned["embedder_used"] = str(cleaned.get("embedder_used", "") or "arcface").strip() or "arcface"
    cleaned["lighting_bucket"] = str(cleaned.get("lighting_bucket", "") or "balanced_light").strip() or "balanced_light"
    cleaned["generator_type"] = str(cleaned.get("generator_type", "") or "base").strip() or "base"
    cleaned["augmentation_family"] = str(cleaned.get("augmentation_family", "") or "").strip()
    cleaned["augmentation_tag"] = str(cleaned.get("augmentation_tag", "") or "").strip()
    cleaned["combination_tag"] = str(cleaned.get("combination_tag", "") or "").strip()
    cleaned["bank_family"] = str(cleaned.get("bank_family", "") or normalize_bank_family(cleaned)).strip() or "base"
    cleaned["added_at"] = str(cleaned.get("added_at", "") or added_at or current_utc_iso())
    return cleaned


@dataclass
class EmbeddingRecord:
    embedding: np.ndarray
    quality: float
    metadata: Dict[str, object]


def _record_sort_key(record: EmbeddingRecord) -> Tuple[float, float, float]:
    timestamp = _timestamp_sort_value(record.metadata.get("added_at"))
    face_size = float(record.metadata.get("face_size", 0.0) or 0.0)
    return (timestamp, float(record.quality), face_size)


def _bucket_key(metadata: Dict[str, object]) -> Tuple[str, str]:
    return (
        str(metadata.get("bank_family", "base") or "base"),
        f"{str(metadata.get('embedder_used', 'arcface') or 'arcface')}|{str(metadata.get('profile_bucket', 'frontal') or 'frontal')}|{str(metadata.get('size_bucket', 'medium_face') or 'medium_face')}",
    )


def compact_embedding_bank(
    embeddings: Sequence[np.ndarray],
    qualities: Sequence[float],
    metadata_list: Sequence[Dict[str, object]],
    *,
    max_bank: int = 48,
    duplicate_sim_thresh: float = 0.92,
) -> Tuple[List[np.ndarray], List[float], List[Dict[str, object]]]:
    if not embeddings:
        return [], [], []

    padded_metadata: List[Dict[str, object]] = []
    for idx, emb in enumerate(embeddings):
        quality = float(qualities[idx]) if idx < len(qualities) else 0.5
        meta = metadata_list[idx] if idx < len(metadata_list) else {}
        padded_metadata.append(sanitize_embedding_metadata(meta, quality=quality))

    records = [
        EmbeddingRecord(
            embedding=l2_normalize(np.asarray(emb, dtype=np.float32)),
            quality=float(qualities[idx]) if idx < len(qualities) else 0.5,
            metadata=padded_metadata[idx],
        )
        for idx, emb in enumerate(embeddings)
    ]
    records.sort(key=_record_sort_key, reverse=True)

    kept_by_bucket: Dict[Tuple[str, str], List[EmbeddingRecord]] = {}
    for record in records:
        bucket = _bucket_key(record.metadata)
        bucket_records = kept_by_bucket.setdefault(bucket, [])
        if any(cosine_similarity(record.embedding, existing.embedding) >= duplicate_sim_thresh for existing in bucket_records):
            continue
        bucket_records.append(record)

    deduped_records: List[EmbeddingRecord] = []
    for bucket in sorted(kept_by_bucket.keys()):
        bucket_records = kept_by_bucket[bucket]
        bucket_records.sort(key=_record_sort_key, reverse=True)
        deduped_records.extend(bucket_records)

    if len(deduped_records) > max_bank:
        protected: List[EmbeddingRecord] = []
        remainder: List[EmbeddingRecord] = []
        for bucket in sorted(kept_by_bucket.keys()):
            bucket_records = kept_by_bucket[bucket]
            if not bucket_records:
                continue
            protected.append(bucket_records[0])
            remainder.extend(bucket_records[1:])
        protected_ids = {id(item) for item in protected}
        remainder_ids = {id(item) for item in remainder}
        for item in deduped_records:
            item_id = id(item)
            if item_id in protected_ids or item_id in remainder_ids:
                continue
            remainder.append(item)
            remainder_ids.add(item_id)
        remainder.sort(key=_record_sort_key, reverse=True)
        available = max(0, max_bank - len(protected))
        deduped_records = protected[:max_bank] + remainder[:available]

    deduped_records.sort(key=_record_sort_key, reverse=True)
    return (
        [record.embedding.copy() for record in deduped_records],
        [float(record.quality) for record in deduped_records],
        [dict(record.metadata) for record in deduped_records],
    )


def summarize_embedding_bank(
    embeddings: Sequence[np.ndarray],
    qualities: Sequence[float],
    metadata_list: Sequence[Dict[str, object]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, Optional[np.ndarray]]:
    if not embeddings:
        return None, None, 0.0, None
    avg_embedding = weighted_average_embeddings(list(embeddings), list(qualities))
    best_idx = 0
    best_key = (-1.0, -1.0)
    recent_idx = 0
    recent_timestamp = -1.0
    for idx, emb in enumerate(embeddings):
        quality = float(qualities[idx]) if idx < len(qualities) else 0.5
        metadata = metadata_list[idx] if idx < len(metadata_list) else {}
        sort_key = (quality, _timestamp_sort_value(metadata.get("added_at")))
        if sort_key > best_key:
            best_key = sort_key
            best_idx = idx
        timestamp = _timestamp_sort_value(metadata.get("added_at"))
        if timestamp >= recent_timestamp:
            recent_timestamp = timestamp
            recent_idx = idx
    best_embedding = l2_normalize(np.asarray(embeddings[best_idx], dtype=np.float32))
    recent_embedding = l2_normalize(np.asarray(embeddings[recent_idx], dtype=np.float32))
    return avg_embedding, best_embedding, float(qualities[best_idx]), recent_embedding


def _summary_payload(
    embeddings: Sequence[np.ndarray],
    qualities: Sequence[float],
    metadata_list: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    avg_embedding, best_embedding, best_quality, recent_embedding = summarize_embedding_bank(
        embeddings,
        qualities,
        metadata_list,
    )
    return {
        "count": len(embeddings),
        "avg_embedding": None if avg_embedding is None else avg_embedding.copy(),
        "best_embedding": None if best_embedding is None else best_embedding.copy(),
        "recent_embedding": None if recent_embedding is None else recent_embedding.copy(),
        "best_quality": float(best_quality),
    }


def build_grouped_embedding_summaries(
    embeddings: Sequence[np.ndarray],
    qualities: Sequence[float],
    metadata_list: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, Dict[str, List[object]]] = {}
    for idx, emb in enumerate(embeddings):
        quality = float(qualities[idx]) if idx < len(qualities) else 0.5
        metadata = sanitize_embedding_metadata(
            metadata_list[idx] if idx < len(metadata_list) else {},
            quality=quality,
        )
        bank_family = str(metadata.get("bank_family", "base") or "base")
        embedder_used = str(metadata.get("embedder_used", "arcface") or "arcface")
        family_bucket = grouped.setdefault(bank_family, {})
        family_bucket.setdefault("_embeddings", []).append(l2_normalize(np.asarray(emb, dtype=np.float32)))
        family_bucket.setdefault("_qualities", []).append(quality)
        family_bucket.setdefault("_metadata", []).append(metadata)
        per_embedder = family_bucket.setdefault("_per_embedder", {})
        embedder_bucket = per_embedder.setdefault(embedder_used, {"embeddings": [], "qualities": [], "metadata": []})
        embedder_bucket["embeddings"].append(l2_normalize(np.asarray(emb, dtype=np.float32)))
        embedder_bucket["qualities"].append(quality)
        embedder_bucket["metadata"].append(metadata)

    summaries: Dict[str, Dict[str, object]] = {}
    for bank_family, bucket in grouped.items():
        family_embeddings = bucket["_embeddings"]
        family_qualities = bucket["_qualities"]
        family_metadata = bucket["_metadata"]
        per_embedder_payload: Dict[str, Dict[str, object]] = {}
        for embedder_used, embedder_bucket in bucket["_per_embedder"].items():
            per_embedder_payload[embedder_used] = _summary_payload(
                embedder_bucket["embeddings"],
                embedder_bucket["qualities"],
                embedder_bucket["metadata"],
            )
        summaries[bank_family] = {
            "count": len(family_embeddings),
            "global": _summary_payload(family_embeddings, family_qualities, family_metadata),
            "per_embedder": per_embedder_payload,
        }
    return summaries


def bank_match_score(
    probe_embedding: np.ndarray,
    avg_embedding: Optional[np.ndarray],
    bank_embeddings: Sequence[np.ndarray],
    bank_qualities: Sequence[float],
    *,
    recent_embedding: Optional[np.ndarray] = None,
    best_embedding: Optional[np.ndarray] = None,
) -> float:
    if avg_embedding is None:
        return -1.0
    probe = l2_normalize(np.asarray(probe_embedding, dtype=np.float32))
    avg_sim = cosine_similarity(probe, l2_normalize(np.asarray(avg_embedding, dtype=np.float32)))
    scores = [avg_sim]
    if recent_embedding is not None:
        scores.append(0.97 * cosine_similarity(probe, l2_normalize(np.asarray(recent_embedding, dtype=np.float32))))
    if bank_embeddings:
        bank_scores: List[float] = []
        for idx, emb in enumerate(bank_embeddings):
            quality = float(bank_qualities[idx]) if idx < len(bank_qualities) else 1.0
            bank_scores.append(cosine_similarity(probe, l2_normalize(np.asarray(emb, dtype=np.float32))) * (0.85 + 0.15 * max(0.05, quality)))
        if bank_scores:
            scores.append(max(bank_scores))
    if best_embedding is not None:
        scores.append(cosine_similarity(probe, l2_normalize(np.asarray(best_embedding, dtype=np.float32))))
    best_bank = max(scores)
    return float(0.55 * best_bank + 0.45 * avg_sim)


def grouped_bank_match_score(
    probe_embedding: np.ndarray,
    group_summary: Optional[Dict[str, object]],
) -> float:
    if not group_summary:
        return -1.0
    probe = l2_normalize(np.asarray(probe_embedding, dtype=np.float32))
    avg_embedding = group_summary.get("avg_embedding")
    best_embedding = group_summary.get("best_embedding")
    recent_embedding = group_summary.get("recent_embedding")
    best_quality = float(group_summary.get("best_quality", 0.0) or 0.0)
    if avg_embedding is None:
        return -1.0

    avg_sim = cosine_similarity(probe, l2_normalize(np.asarray(avg_embedding, dtype=np.float32)))
    scores = [avg_sim]
    if recent_embedding is not None:
        scores.append(0.97 * cosine_similarity(probe, l2_normalize(np.asarray(recent_embedding, dtype=np.float32))))
    if best_embedding is not None:
        scores.append(cosine_similarity(probe, l2_normalize(np.asarray(best_embedding, dtype=np.float32))))
    best_bank = max(scores)
    quality_bonus = 0.02 * clamp(best_quality, 0.0, 1.0)
    return float(0.55 * best_bank + 0.45 * avg_sim + quality_bonus)


def classify_harvest_candidate(
    similarity: float,
    *,
    good_crops: int,
    stable_seconds: float,
    quality: float,
    auto_add_similarity: float = 0.68,
    review_similarity: float = 0.60,
    min_good_crops: int = 3,
    min_stable_seconds: float = 1.5,
    min_quality: float = 0.40,
) -> str:
    support_ok = int(good_crops) >= int(min_good_crops) or float(stable_seconds) >= float(min_stable_seconds)
    if similarity >= auto_add_similarity and support_ok and float(quality) >= min_quality:
        return "auto_add"
    if similarity >= review_similarity:
        return "review"
    return "reject"
