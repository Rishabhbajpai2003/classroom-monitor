"""
Identity Manager — Persistent student registry with embedding-based matching.

Maintains a JSON registry of known students, each with a stored embedding vector.
When a new face/person is detected, the embedding is compared against stored
embeddings using cosine similarity. If above threshold → existing student;
otherwise → new student registered.

This enables identity persistence across multiple video sessions and days.
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class IdentityManager:
    """
    Manages persistent student identities via embedding similarity matching.

    Thread-safe registry operations via a threading.Lock.

    Attributes:
        registry_path: Path to student_registry.json.
        similarity_threshold: Cosine similarity threshold for matching.
        registry: Dict mapping global_id → student record.
        _lock: Thread lock for concurrent access safety.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize IdentityManager.

        Args:
            config: Full pipeline config dict. Uses 'identity' section.
        """
        config = config or {}
        id_cfg = config.get("identity", {})

        self.registry_path = id_cfg.get("registry_path", "data/student_registry.json")
        self.similarity_threshold = id_cfg.get("embedding_similarity_threshold", 0.75)
        self._lock = threading.Lock()

        # Track-ID to Global-ID mapping for current session
        self._track_to_global = {}

        # Load existing registry
        self.registry = self._load_registry()
        logger.info(
            f"IdentityManager initialized with {len(self.registry)} known students. "
            f"Registry: {self.registry_path}"
        )

    def _load_registry(self) -> dict:
        """Load student registry from JSON file."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} students from registry.")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load registry: {e}. Starting fresh.")
                return {}
        return {}

    def _save_registry(self):
        """Persist registry to JSON file."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        try:
            with open(self.registry_path, "w") as f:
                json.dump(self.registry, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save registry: {e}")

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def match_or_register(
        self, embedding: np.ndarray | None, timestamp: str | None = None
    ) -> str:
        """
        Match an embedding against known students, or register a new one.

        Args:
            embedding: Normalized face/ReID embedding vector.
            timestamp: ISO-format timestamp string.

        Returns:
            Global student ID (e.g., 'STU_001').
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        if embedding is None:
            # Cannot match without embedding — generate a temporary ID
            temp_id = f"STU_TEMP_{uuid.uuid4().hex[:6].upper()}"
            logger.debug(f"No embedding available, assigned temp ID: {temp_id}")
            return temp_id

        with self._lock:
            best_match_id = None
            best_similarity = -1.0

            for global_id, record in self.registry.items():
                stored_emb = np.array(record["embedding"], dtype=np.float32)
                similarity = self._cosine_similarity(embedding, stored_emb)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = global_id

            if best_match_id and best_similarity >= self.similarity_threshold:
                # Update last_seen and refresh embedding with exponential moving average
                record = self.registry[best_match_id]
                record["last_seen"] = timestamp

                # EMA update of stored embedding (alpha=0.1 for stability)
                stored_emb = np.array(record["embedding"], dtype=np.float32)
                updated_emb = 0.9 * stored_emb + 0.1 * embedding
                norm = np.linalg.norm(updated_emb)
                if norm > 0:
                    updated_emb = updated_emb / norm
                record["embedding"] = updated_emb.tolist()

                logger.debug(
                    f"Matched student {best_match_id} "
                    f"(similarity={best_similarity:.3f})"
                )
                self._save_registry()
                return best_match_id
            else:
                # Register new student
                new_id = f"STU_{len(self.registry) + 1:03d}"
                self.registry[new_id] = {
                    "global_id": new_id,
                    "embedding": embedding.tolist(),
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "name": None,
                }
                logger.info(
                    f"Registered new student: {new_id} "
                    f"(best_sim={best_similarity:.3f})"
                )
                self._save_registry()
                return new_id

    def get_global_id_for_track(
        self, track_id: int, embedding: np.ndarray | None = None,
        timestamp: str | None = None
    ) -> str:
        """
        Get or assign a global ID for a local track ID.

        Caches the mapping for the current session to avoid repeated
        embedding lookups.

        Args:
            track_id: Local DeepSORT track ID.
            embedding: Face/ReID embedding (may be None).
            timestamp: ISO-format timestamp.

        Returns:
            Global student ID string.
        """
        if track_id in self._track_to_global:
            return self._track_to_global[track_id]

        global_id = self.match_or_register(embedding, timestamp)
        self._track_to_global[track_id] = global_id
        return global_id

    def get_registry_summary(self) -> list[dict]:
        """Return a summary of all registered students."""
        summary = []
        for gid, record in self.registry.items():
            summary.append({
                "global_id": gid,
                "first_seen": record.get("first_seen"),
                "last_seen": record.get("last_seen"),
                "name": record.get("name"),
            })
        return summary

    def reset_session_cache(self):
        """Clear track-to-global mapping for a new session."""
        self._track_to_global.clear()
        logger.info("Session track-to-global cache cleared.")
