"""PubMedBERT text matcher."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

DEFAULT_PUBMEDBERT_MODEL = "NeuML/pubmedbert-base-embeddings"
LOCAL_PUBMEDBERT_CANDIDATES = (
    Path("tmp/pubmedbert-base-embeddings"),
    Path("/tmp/pubmedbert-base-embeddings"),
)


@dataclass
class PubMedBERTMatcher:
    model_name: str
    device: Optional[str] = None
    batch_size: int = 64

    def __post_init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._resolved_device = self._resolve_device(self.device)
        self._resolved_model_name = self._resolve_model_name(self.model_name)
        self._embedding_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device:
            return str(device)
        try:
            import torch
        except Exception:
            return "cpu"
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        model_name_str = str(model_name)
        candidate = Path(model_name_str)
        if candidate.exists():
            return str(candidate)
        if model_name_str == DEFAULT_PUBMEDBERT_MODEL:
            for local_candidate in LOCAL_PUBMEDBERT_CANDIDATES:
                if local_candidate.exists():
                    return str(local_candidate)
        return model_name_str

    @property
    def resolved_device(self) -> str:
        return str(self._resolved_device)

    def _get_model(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise ImportError(
                        "sentence-transformers is required for PubMedBERT matcher. "
                        "Install with: pip install sentence-transformers"
                    ) from exc
                self._model = SentenceTransformer(self._resolved_model_name, device=self._resolved_device)
            return self._model

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        normalized_texts = [str(text) for text in texts]
        missing_texts: List[str] = []
        seen_missing = set()
        with self._lock:
            for text in normalized_texts:
                if text in self._embedding_cache or text in seen_missing:
                    continue
                seen_missing.add(text)
                missing_texts.append(text)

        if missing_texts:
            model = self._get_model()
            with self._lock:
                uncached_texts = [text for text in missing_texts if text not in self._embedding_cache]
                if uncached_texts:
                    embeddings = model.encode(
                        uncached_texts,
                        batch_size=int(self.batch_size),
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    embedding_array = np.asarray(embeddings, dtype=np.float32)
                    if embedding_array.ndim != 2:
                        raise ValueError(f"Unexpected embedding shape: {embedding_array.shape}")
                    for text, embedding in zip(uncached_texts, embedding_array):
                        self._embedding_cache[text] = np.asarray(embedding, dtype=np.float32)

        with self._lock:
            resolved_embeddings = [self._embedding_cache[text] for text in normalized_texts]
        array = np.asarray(resolved_embeddings, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Unexpected embedding shape: {array.shape}")
        return array

    def match_pairs(
        self,
        prediction_texts: Sequence[str],
        ground_truth_texts: Sequence[str],
        *,
        similarity_threshold: float,
    ) -> Dict[str, Any]:
        if float(similarity_threshold) < 0.0 or float(similarity_threshold) > 1.0:
            raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")
        if not prediction_texts or not ground_truth_texts:
            return {
                "matched_pairs": [],
                "matched_prediction_indices": [],
                "matched_gt_indices": [],
                "pair_similarities": [],
                "similarity_threshold": float(similarity_threshold),
            }

        prediction_embeddings = self._encode_texts(prediction_texts)
        gt_embeddings = self._encode_texts(ground_truth_texts)
        if prediction_embeddings.shape[1] != gt_embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: pred={prediction_embeddings.shape}, gt={gt_embeddings.shape}"
            )
        similarity_matrix = np.matmul(prediction_embeddings, gt_embeddings.T)

        matched_pairs: List[Dict[str, int]] = []
        matched_prediction_indices: List[int] = []
        matched_gt_indices: List[int] = []
        pair_similarities: List[Dict[str, Any]] = []

        for prediction_index in range(similarity_matrix.shape[0]):
            row = similarity_matrix[prediction_index]
            gt_index = int(np.argmax(row))
            similarity = float(row[gt_index])
            if similarity < float(similarity_threshold):
                continue
            matched_pairs.append({"prediction_index": int(prediction_index), "gt_index": int(gt_index)})
            matched_prediction_indices.append(int(prediction_index))
            matched_gt_indices.append(int(gt_index))
            pair_similarities.append(
                {
                    "prediction_index": int(prediction_index),
                    "gt_index": int(gt_index),
                    "similarity": float(similarity),
                }
            )

        matched_prediction_indices = sorted(set(matched_prediction_indices))
        matched_gt_indices = sorted(set(matched_gt_indices))
        matched_pairs = sorted(
            matched_pairs,
            key=lambda row: (int(row["prediction_index"]), int(row["gt_index"])),
        )
        pair_similarities = sorted(
            pair_similarities,
            key=lambda row: (int(row["prediction_index"]), int(row["gt_index"])),
        )
        return {
            "matched_pairs": matched_pairs,
            "matched_prediction_indices": matched_prediction_indices,
            "matched_gt_indices": matched_gt_indices,
            "pair_similarities": pair_similarities,
            "similarity_threshold": float(similarity_threshold),
        }
