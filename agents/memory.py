"""ICU-Evo memory structures."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from agents.rendering import (
    _serialize_critical_event_strings,
    _serialize_event_strings,
)


@dataclass
class LLMCallLog:
    timestamp: str
    patient_id: str
    window_index: int
    hours_since_admission: float
    prompt: str
    response: str
    parsed_response: Optional[Dict[str, Any]]
    input_tokens: int
    output_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "patient_id": self.patient_id,
            "window_index": self.window_index,
            "hours_since_admission": self.hours_since_admission,
            "prompt": self.prompt,
            "response": self.response,
            "parsed_response": self.parsed_response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": self.metadata,
        }


@dataclass
class WorkingWindow:
    window_id: int
    start_hour: float
    end_hour: float
    events: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "events": _serialize_event_strings(self.events),
        }


@dataclass
class TrendMemoryEntry:
    window_index: int
    start_hour: float
    end_hour: float
    raw_event_count: int
    vital_trends: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_index": self.window_index,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "raw_event_count": self.raw_event_count,
            "vital_trends": deepcopy(self.vital_trends),
        }


@dataclass
class Episode:
    episode_id: int
    start_window: int
    end_window: int
    start_hour: float
    end_hour: float
    episode_summary: str
    supporting_event_ids: List[int]
    supporting_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "episode",
            "episode_id": self.episode_id,
            "start_window": self.start_window,
            "end_window": self.end_window,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "episode_summary": self.episode_summary,
            "supporting_event_ids": self.supporting_event_ids,
            "supporting_events": _serialize_event_strings(self.supporting_events),
        }


@dataclass
class CriticalEventsMemoryEntry:
    episode_id: int
    start_window: int
    end_window: int
    start_hour: float
    end_hour: float
    critical_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "episode_critical_events",
            "episode_id": self.episode_id,
            "start_window": self.start_window,
            "end_window": self.end_window,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "critical_events": _serialize_critical_event_strings(self.critical_events),
        }


@dataclass
class EvidenceEvent:
    event_id: int
    name_str: str

    def to_dict(self) -> str:
        return self.name_str


@dataclass
class Insight:
    insight_id: int
    hypothesis: str
    supporting_event_ids: List[int]
    counter_event_ids: List[int]
    created_at: float
    updated_at: float
    supporting_trend_evidence: List[str] = field(default_factory=list)
    counter_trend_evidence: List[str] = field(default_factory=list)
    supporting_evidence: List[EvidenceEvent] = field(default_factory=list)
    counter_evidence: List[EvidenceEvent] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        supporting_evidence = [item.to_dict() for item in self.supporting_evidence] + list(
            self.supporting_trend_evidence
        )
        counter_evidence = [item.to_dict() for item in self.counter_evidence] + list(self.counter_trend_evidence)
        return {
            "insight_id": self.insight_id,
            "hypothesis": self.hypothesis,
            "supporting_event_ids": self.supporting_event_ids,
            "counter_event_ids": self.counter_event_ids,
            "supporting_trend_evidence": self.supporting_trend_evidence,
            "counter_trend_evidence": self.counter_trend_evidence,
            "supporting_evidence": supporting_evidence,
            "counter_evidence": counter_evidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "score": self.score,
        }


@dataclass
class ICUEvoMemory:
    patient_metadata: Dict[str, Any] = field(default_factory=dict)
    working_memory: List[WorkingWindow] = field(default_factory=list)
    trend_memory: List[TrendMemoryEntry] = field(default_factory=list)
    critical_events_memory: List[CriticalEventsMemoryEntry] = field(default_factory=list)
    trajectory_memory: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)
    last_processed_window_index: int = -1
    last_processed_start_hour: float = 0.0
    last_processed_end_hour: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_metadata": deepcopy(self.patient_metadata),
            "working_memory": [item.to_dict() for item in self.working_memory],
            "trend_memory": [item.to_dict() for item in self.trend_memory],
            "critical_events_memory": [item.to_dict() for item in self.critical_events_memory],
            "trajectory_memory": deepcopy(self.trajectory_memory),
            "insights": [item.to_dict() for item in self.insights],
            "last_processed_window_index": self.last_processed_window_index,
            "last_processed_start_hour": self.last_processed_start_hour,
            "last_processed_end_hour": self.last_processed_end_hour,
        }
