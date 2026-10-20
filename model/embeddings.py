"""A shared, explicitly configured text encoder for RAG and action matching."""

import threading

import numpy as np


class TextEncoder:
    def __init__(self, *, model_name: str, device: str, batch_size: int, local_files_only: bool):
        if not model_name.strip() or not device.strip() or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Embedding model, device, and positive batch_size are required")
        if type(local_files_only) is not bool:
            raise ValueError("local_files_only must be boolean")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self._model = None
        self._cache = {}
        self._lock = threading.Lock()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return self._encode_texts(texts)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.model_name, device=self.device, local_files_only=self.local_files_only,
                )
            embeddings = np.asarray(self._model.encode(
                missing, batch_size=self.batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ), dtype=np.float32)
            if embeddings.ndim != 2 or len(embeddings) != len(missing) or not np.isfinite(embeddings).all():
                raise ValueError("Encoder returned invalid embeddings")
            self._cache.update(zip(missing, embeddings))
        return np.asarray([self._cache[text] for text in texts], dtype=np.float32)
