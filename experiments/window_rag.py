from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def _normalize_event_lines(event_lines: Sequence[str]) -> List[str]:
    normalized = [str(line).strip() for line in event_lines if str(line).strip()]
    if normalized:
        return normalized
    return ["(No events)"]


def _normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    return array / safe_norms


def _resolve_local_hf_snapshot(model_name: str) -> str:
    candidate = str(model_name).strip()
    if not candidate:
        raise ValueError("model_name must be non-empty")
    if "/" not in candidate:
        return candidate
    if Path(candidate).exists():
        return candidate
    repo_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{candidate.replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"
    if not refs_main.exists():
        return candidate
    snapshot_id = refs_main.read_text(encoding="utf-8").strip()
    if not snapshot_id:
        return candidate
    snapshot_dir = repo_dir / "snapshots" / snapshot_id
    if snapshot_dir.exists():
        return str(snapshot_dir)
    return candidate


@dataclass(frozen=True)
class RAGWindowRecord:
    window_index: int
    start_hour: float
    end_hour: float
    event_lines: List[str]
    event_text: str


@dataclass(frozen=True)
class RAGRetrievedWindow:
    rank: int
    window_index: int
    start_hour: float
    end_hour: float
    similarity: float
    event_lines: List[str]

    def to_payload(self) -> Dict[str, float | int]:
        return {
            "rank": int(self.rank),
            "window_index": int(self.window_index),
            "start_hour": float(self.start_hour),
            "end_hour": float(self.end_hour),
            "similarity": float(self.similarity),
        }


def build_rag_window_record(
    *,
    window_index: int,
    start_hour: float,
    end_hour: float,
    event_lines: Sequence[str],
) -> RAGWindowRecord:
    normalized_event_lines = _normalize_event_lines(event_lines)
    return RAGWindowRecord(
        window_index=int(window_index),
        start_hour=float(start_hour),
        end_hour=float(end_hour),
        event_lines=normalized_event_lines,
        event_text="\n".join(normalized_event_lines),
    )


def render_rag_event_context(
    *,
    current_window: RAGWindowRecord,
    retrieved_windows: Sequence[RAGRetrievedWindow],
    retrieval_top_k: int,
) -> str:
    lines: List[str] = [
        "## Current Window Events",
        (
            f"Window {int(current_window.window_index)} "
            f"(Hour {float(current_window.start_hour):.1f}-{float(current_window.end_hour):.1f})"
        ),
    ]
    lines.extend(current_window.event_lines)
    lines.append("")
    lines.append(f"## Retrieved Historical Windows (Top {int(retrieval_top_k)})")
    if not retrieved_windows:
        lines.append("- None")
        return "\n".join(lines)

    for item in retrieved_windows:
        lines.append(
            (
                f"### Rank {int(item.rank)} | Window {int(item.window_index)} "
                f"(Hour {float(item.start_hour):.1f}-{float(item.end_hour):.1f}) | "
                f"Similarity {float(item.similarity):.4f}"
            )
        )
        lines.extend(item.event_lines)
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


class SentenceTransformerWindowEncoder:
    def __init__(
        self,
        *,
        model_name: str,
        device: Optional[str],
        batch_size: int,
    ):
        self.model_name = str(model_name)
        self.device = self._resolve_device(model_name=self.model_name, device=device)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        self._model = None

    @staticmethod
    def _resolve_device(*, model_name: str, device: Optional[str]) -> Optional[str]:
        if device is not None:
            return str(device)
        model_name_normalized = str(model_name).strip().lower()
        if "pubmedbert" in model_name_normalized:
            return "cuda"
        return None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for rag_history_events mode. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            resolved_model_name = _resolve_local_hf_snapshot(self.model_name)
            self._model = SentenceTransformer(
                resolved_model_name,
                device=self.device,
                local_files_only=True,
            )
        return self._model

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        embeddings = self.model.encode(
            [str(text) for text in texts],
            batch_size=int(self.batch_size),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return _normalize_embeddings(np.asarray(embeddings, dtype=np.float32))


class IncrementalWindowRAGRetriever:
    def __init__(
        self,
        *,
        window_records: Sequence[RAGWindowRecord],
        encoder: SentenceTransformerWindowEncoder,
        top_k: int,
    ):
        self.encoder = encoder
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")

        records_sorted = sorted(window_records, key=lambda row: int(row.window_index))
        self._records_by_window: Dict[int, RAGWindowRecord] = {}
        self._ordered_window_indices: List[int] = []
        for record in records_sorted:
            window_index = int(record.window_index)
            if window_index in self._records_by_window:
                raise ValueError(f"Duplicate window_index in RAG records: {window_index}")
            self._records_by_window[window_index] = record
            self._ordered_window_indices.append(window_index)

        self._encoded_until_position = -1
        self._window_embeddings: Dict[int, np.ndarray] = {}
        self._query_embedding_cache: Dict[str, np.ndarray] = {}

    def _target_position(self, current_window_index: int) -> int:
        target = -1
        for idx, window_index in enumerate(self._ordered_window_indices):
            if int(window_index) <= int(current_window_index):
                target = idx
            else:
                break
        return target

    def ensure_embeddings_through(self, current_window_index: int) -> None:
        target = self._target_position(current_window_index)
        if target <= self._encoded_until_position:
            return
        if target < 0:
            self._encoded_until_position = target
            return

        start = self._encoded_until_position + 1
        end = target + 1
        window_ids_to_encode = self._ordered_window_indices[start:end]
        texts_to_encode = [self._records_by_window[window_id].event_text for window_id in window_ids_to_encode]
        embeddings = self.encoder.encode_texts(texts_to_encode)
        if embeddings.shape[0] != len(window_ids_to_encode):
            raise ValueError(
                f"Embedding row count mismatch: expected {len(window_ids_to_encode)}, got {embeddings.shape[0]}"
            )
        for row_index, window_index in enumerate(window_ids_to_encode):
            self._window_embeddings[int(window_index)] = np.asarray(embeddings[row_index], dtype=np.float32)
        self._encoded_until_position = target

    def _encode_query(self, query_text: str) -> np.ndarray:
        query = str(query_text)
        cached = self._query_embedding_cache.get(query)
        if cached is not None:
            return cached
        embedding = self.encoder.encode_texts([query])
        if embedding.shape[0] != 1:
            raise ValueError(f"Expected single query embedding row, got shape={embedding.shape}")
        vector = np.asarray(embedding[0], dtype=np.float32)
        self._query_embedding_cache[query] = vector
        return vector

    def retrieve(
        self,
        *,
        current_window_index: int,
        query_text: str,
    ) -> List[RAGRetrievedWindow]:
        self.ensure_embeddings_through(current_window_index=int(current_window_index))
        query_embedding = self._encode_query(query_text)
        candidate_window_ids = [
            window_index
            for window_index in self._ordered_window_indices
            if int(window_index) < int(current_window_index) and int(window_index) in self._window_embeddings
        ]
        if not candidate_window_ids:
            return []

        scored: List[tuple[int, float]] = []
        for window_index in candidate_window_ids:
            similarity = float(np.dot(query_embedding, self._window_embeddings[int(window_index)]))
            scored.append((int(window_index), similarity))
        scored.sort(key=lambda item: (-float(item[1]), int(item[0])))
        top_scored = scored[: self.top_k]

        retrieved: List[RAGRetrievedWindow] = []
        for rank, (window_index, similarity) in enumerate(top_scored, start=1):
            record = self._records_by_window[int(window_index)]
            retrieved.append(
                RAGRetrievedWindow(
                    rank=int(rank),
                    window_index=int(record.window_index),
                    start_hour=float(record.start_hour),
                    end_hour=float(record.end_hour),
                    similarity=float(similarity),
                    event_lines=list(record.event_lines),
                )
            )
        return retrieved
