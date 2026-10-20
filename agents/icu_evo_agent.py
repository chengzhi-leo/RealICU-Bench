"""Build observation-grounded ICU-Evo memory."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from model.llms import LLMClient
from prompts.icu_evo_prompts import format_pre_icu_compression_prompt

from agents.artifacts import compact_snapshot

from agents.memory import (
    CriticalEventsMemoryEntry,
    Episode,
    EvidenceEvent,
    ICUEvoMemory,
    Insight,
    LLMCallLog,
    TrendMemoryEntry,
    WorkingWindow,
)
from agents.validation import (
    _chat_with_schema_validation,
    _derive_raw_event_name,
    _extract_event_reference_id,
    _normalize_int_list,
    _normalize_token_count,
    _safe_text,
    _normalize_text_list,
    _split_evidence_references,
    _to_optional_float,
    _validate_pre_icu_compression_output_schema,
)
from agents.rendering import (
    _format_prior_episode_summary_text,
    _format_single_event_line,
    _trend_entry_by_window,
    _trend_entry_summary,
)
from agents.observation import (
    ObservationAgent,
    PerceptionAgent,
)
from agents.reasoning import (
    AssessmentAgent,
    InsightAgent,
)


PRE_ICU_COMPRESSION_STEP_TYPE = "icu_evo_pre_icu_history_compressor"


class ICUEvoAgent:
    def __init__(
        self, *, llm_client: LLMClient, observation_config_path: str,
        episode_block_windows: int, insight_block_windows: int,
        max_working_windows: int, max_insights: int, max_trajectory_entries: int,
        schema_max_attempts: int, pre_icu_summary_chars: int, enable_logging: bool,
    ):
        self.llm_client = llm_client
        self.schema_max_attempts = schema_max_attempts
        self.pre_icu_summary_chars = pre_icu_summary_chars
        settings = (episode_block_windows, insight_block_windows, max_working_windows,
                    max_insights, max_trajectory_entries, schema_max_attempts, pre_icu_summary_chars)
        if any(type(value) is not int or value < 1 for value in settings):
            raise ValueError("ICU-Evo window counts, limits, and attempts must be positive integers")
        self.max_working_windows = max_working_windows
        self.max_insights = max_insights
        if max_trajectory_entries is None:
            raise ValueError("max_trajectory_entries must be provided")
        self.max_trajectory_entries = max_trajectory_entries
        self.episode_block_windows = episode_block_windows
        self.insight_block_windows = insight_block_windows
        self.observation_config_path = _safe_text(observation_config_path)
        if not self.observation_config_path:
            raise ValueError("observation_config_path must be a non-empty string")

        self.enable_logging = enable_logging
        self.call_logs: List[LLMCallLog] = []

        self.perception_agent = PerceptionAgent(
            max_working_windows=self.max_working_windows,
        )
        self.observation_agent = ObservationAgent(self.observation_config_path)
        self.assessment_agent = AssessmentAgent(self.llm_client, schema_max_attempts)
        self.insight_agent = InsightAgent(self.llm_client, schema_max_attempts)

        self.total_patients = 0
        self.total_tokens_used = 0
        self.total_observation_runs = 0
        self.total_episode_calls = 0
        self.total_insight_calls = 0
        self.total_pre_icu_compression_calls = 0
        self.total_pre_icu_compression_tokens = 0
        self.total_grounding_rejections = 0
        self.total_insights_pruned = 0
        self.total_trajectory_entries_pruned = 0

        self._current_patient_id = ""
        self._event_id_to_window: Dict[int, int] = {}
        self._event_id_to_time: Dict[int, str] = {}
        self._event_id_to_raw_event: Dict[int, Dict[str, Any]] = {}
        self._event_id_to_name_str: Dict[int, str] = {}
        self._next_insight_id = 1
        self._next_episode_id = 1

    def clear_logs(self) -> None:
        self.call_logs = []

    def get_logs(self) -> List[Dict[str, Any]]:
        return [log.to_dict() for log in self.call_logs]

    def save_logs(self, path: str) -> None:
        payload = {
            "total_calls": len(self.call_logs),
            "total_tokens": self.total_tokens_used,
            "calls": self.get_logs(),
        }
        with open(path, "w") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_patients": self.total_patients,
            "total_tokens_used": self.total_tokens_used,
            "total_observation_runs": self.total_observation_runs,
            "total_episode_calls": self.total_episode_calls,
            "total_insight_calls": self.total_insight_calls,
            "total_pre_icu_compression_calls": self.total_pre_icu_compression_calls,
            "total_pre_icu_compression_tokens": self.total_pre_icu_compression_tokens,
            "total_grounding_rejections": self.total_grounding_rejections,
            "total_insights_pruned": self.total_insights_pruned,
            "total_trajectory_entries_pruned": self.total_trajectory_entries_pruned,
            "episode_block_windows": self.episode_block_windows,
            "insight_block_windows": self.insight_block_windows,
            "max_trajectory_entries": self.max_trajectory_entries,
            "observation_config_path": self.observation_config_path,
            "total_llm_calls": len(self.call_logs) if self.enable_logging else 0,
        }

    def _log_call(
        self,
        step_type: str,
        window_index: int,
        hours_since_admission: float,
        prompt: str,
        response: str,
        usage: Dict[str, Any],
        parsed_response: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        usage_dict = usage if isinstance(usage, dict) else {}
        input_tokens = _normalize_token_count(usage_dict.get("input_tokens", 0))
        output_tokens = _normalize_token_count(usage_dict.get("output_tokens", 0))
        self.total_tokens_used += input_tokens + output_tokens

        if not self.enable_logging:
            return

        log_metadata = dict(metadata or {})
        log_metadata.update(
            {
                "step_type": step_type,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
            }
        )

        self.call_logs.append(
            LLMCallLog(
                timestamp=datetime.now().isoformat(),
                patient_id=self._current_patient_id,
                window_index=window_index,
                hours_since_admission=hours_since_admission,
                prompt=prompt,
                response=response,
                parsed_response=parsed_response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata=log_metadata,
            )
        )

    @staticmethod
    def _first_window_pre_icu_history(
        windows: List[Dict[str, Any]],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if not windows:
            return None
        first_window = windows[0]
        if not isinstance(first_window, dict):
            return None
        pre_icu_history = first_window.get("pre_icu_history")
        if not isinstance(pre_icu_history, dict):
            return None
        return first_window, pre_icu_history

    def _compress_pre_icu_history_for_windows(self, windows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        first_payload = self._first_window_pre_icu_history(windows)
        if first_payload is None:
            return None

        first_window, pre_icu_history = first_payload
        source = _safe_text(pre_icu_history.get("source")).lower()
        raw_content = _safe_text(pre_icu_history.get("content"))
        if source in {"", "none", "disabled"} or not raw_content:
            return None

        if source == "llm_compressed":
            return {
                "summary": raw_content,
                "source": source,
                "items": int(_normalize_token_count(pre_icu_history.get("items"))),
                "history_hours": _to_optional_float(pre_icu_history.get("history_hours")),
                "historical_discharge_summary_items": int(
                    _normalize_token_count(pre_icu_history.get("historical_discharge_summary_items"))
                ),
                "compression": pre_icu_history.get("compression") if isinstance(pre_icu_history, dict) else None,
            }

        prompt = format_pre_icu_compression_prompt(pre_icu_history, self.pre_icu_summary_chars)
        parsed, response_raw, response_usage, parse_error, attempt_index = _chat_with_schema_validation(
            llm_client=self.llm_client,
            prompt=prompt,
            schema_validator=_validate_pre_icu_compression_output_schema,
            max_attempts=self.schema_max_attempts,
        )
        compressed_text = _safe_text(parsed.get("compressed_pre_icu_history"))
        if parse_error is not None:
            parse_error = f"{parse_error} | final_attempt={attempt_index}/{self.schema_max_attempts}"

        input_tokens = _normalize_token_count(response_usage.get("input_tokens", 0))
        output_tokens = _normalize_token_count(response_usage.get("output_tokens", 0))
        self.total_pre_icu_compression_calls += 1
        self.total_pre_icu_compression_tokens += input_tokens + output_tokens

        self._log_call(
            step_type=PRE_ICU_COMPRESSION_STEP_TYPE,
            window_index=-1,
            hours_since_admission=float(first_window.get("hours_since_admission", 0.0)),
            prompt=prompt,
            response=response_raw,
            usage=response_usage if isinstance(response_usage, dict) else {},
            parsed_response=parsed if isinstance(parsed, dict) else None,
            metadata={
                "pre_icu_history_source": first_window.get("pre_icu_history_source"),
                "pre_icu_history_items": first_window.get("pre_icu_history_items"),
                "original_source": source or "unknown",
                "original_content_chars": len(raw_content),
                "compressed_chars": len(compressed_text),
                "parse_error": parse_error,
            },
        )

        if not compressed_text:
            return None

        compression_metadata = {
            "method": "llm",
            "step_type": PRE_ICU_COMPRESSION_STEP_TYPE,
            "original_source": source or "unknown",
            "original_content_chars": len(raw_content),
            "compressed_chars": len(compressed_text),
        }

        compressed_pre_icu_history = dict(pre_icu_history)
        compressed_pre_icu_history["source"] = "llm_compressed"
        compressed_pre_icu_history["content"] = compressed_text
        compressed_pre_icu_history["compression"] = compression_metadata

        for window in windows:
            if not isinstance(window, dict):
                continue
            if not isinstance(window.get("pre_icu_history"), dict):
                continue
            window["pre_icu_history"] = dict(compressed_pre_icu_history)
            window["pre_icu_history_source"] = "llm_compressed"

        return {
            "summary": compressed_text,
            "source": "llm_compressed",
            "items": int(_normalize_token_count(pre_icu_history.get("items"))),
            "history_hours": _to_optional_float(pre_icu_history.get("history_hours")),
            "historical_discharge_summary_items": int(
                _normalize_token_count(pre_icu_history.get("historical_discharge_summary_items"))
            ),
            "compression": compression_metadata,
        }

    def _build_patient_metadata_with_pre_icu_summary(
        self,
        *,
        windows: List[Dict[str, Any]],
        patient_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        enriched = deepcopy(patient_metadata)
        compressed_pre_icu = self._compress_pre_icu_history_for_windows(windows)
        if not isinstance(compressed_pre_icu, dict):
            return enriched

        summary = _safe_text(compressed_pre_icu.get("summary"))
        if not summary:
            return enriched

        enriched["pre_icu_history_summary"] = summary
        enriched["pre_icu_history_source"] = _safe_text(compressed_pre_icu.get("source")) or "llm_compressed"
        enriched["pre_icu_history_items"] = int(_normalize_token_count(compressed_pre_icu.get("items")))
        history_hours = _to_optional_float(compressed_pre_icu.get("history_hours"))
        if history_hours is not None:
            enriched["pre_icu_history_hours"] = history_hours
        enriched["historical_discharge_summary_items"] = int(
            _normalize_token_count(compressed_pre_icu.get("historical_discharge_summary_items"))
        )

        return enriched

    def _register_event_ids(self, current_events: List[Dict[str, Any]], window_index: int) -> None:
        for event in current_events:
            event_id = _extract_event_reference_id(event)
            if event_id is None:
                continue

            self._event_id_to_window[event_id] = window_index
            raw_event = deepcopy(event)
            raw_event["event_id"] = event_id
            raw_event["event_index"] = event_id
            raw_event["icu_event_index"] = event_id

            existing = self._event_id_to_raw_event.get(event_id)
            if existing is None or len(raw_event) >= len(existing):
                self._event_id_to_raw_event[event_id] = raw_event

            event_time = _safe_text(raw_event.get("time"))
            if event_time:
                self._event_id_to_time[event_id] = event_time
            elif event_id not in self._event_id_to_time:
                self._event_id_to_time[event_id] = ""

            name_str = _format_single_event_line(raw_event)
            if name_str:
                existing_name = self._event_id_to_name_str.get(event_id, "")
                if not existing_name or len(name_str) >= len(existing_name):
                    self._event_id_to_name_str[event_id] = name_str
            elif event_id not in self._event_id_to_name_str:
                self._event_id_to_name_str[event_id] = f"[{event_id}]"

    def _resolve_event_name_str(self, event_id: int) -> str:
        resolved = _safe_text(self._event_id_to_name_str.get(event_id))
        if resolved:
            return resolved
        raw_event = self._event_id_to_raw_event.get(event_id)
        if isinstance(raw_event, dict):
            resolved = _format_single_event_line(raw_event)
            if resolved:
                self._event_id_to_name_str[event_id] = resolved
                return resolved
        return f"[{event_id}]"

    def _resolve_grounded_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        raw_event = self._event_id_to_raw_event.get(event_id)
        if not isinstance(raw_event, dict):
            return None
        return {
            "event_id": event_id,
            "timestamp": _safe_text(raw_event.get("time")) or self._event_id_to_time.get(event_id, ""),
            "event_name": _derive_raw_event_name(raw_event),
            "code": _safe_text(raw_event.get("code")),
            "code_specifics": _safe_text(raw_event.get("code_specifics")),
            "numeric_value": _to_optional_float(raw_event.get("numeric_value")),
            "text_value": _safe_text(raw_event.get("text_value")),
            "raw_event": deepcopy(raw_event),
        }

    def _resolve_supporting_events(self, event_ids: List[int]) -> List[Dict[str, Any]]:
        supporting_events: List[Dict[str, Any]] = []
        seen = set()
        for event_id in event_ids:
            if event_id in seen:
                continue
            seen.add(event_id)
            grounded = self._resolve_grounded_event(event_id)
            if grounded is None:
                self.total_grounding_rejections += 1
                continue
            supporting_events.append(grounded)
        return supporting_events

    def _validate_event_ids(self, event_ids: List[int]) -> List[int]:
        valid: List[int] = []
        seen = set()
        for event_id in event_ids:
            if event_id in seen:
                continue
            seen.add(event_id)
            if event_id not in self._event_id_to_window:
                self.total_grounding_rejections += 1
                continue
            if event_id not in self._event_id_to_raw_event:
                self.total_grounding_rejections += 1
                continue
            valid.append(event_id)
        return valid

    def _trend_entry_from_detector_output(
        self,
        detector_output: Dict[str, Any],
        window_index: int,
        start_hour: float,
        end_hour: float,
    ) -> TrendMemoryEntry:
        vital_trends = detector_output.get("vital_trends", {})
        if not isinstance(vital_trends, dict):
            vital_trends = {}
        return TrendMemoryEntry(
            window_index=window_index,
            start_hour=float(start_hour),
            end_hour=float(end_hour),
            raw_event_count=_normalize_token_count(detector_output.get("raw_event_count", 0)),
            vital_trends=deepcopy(vital_trends),
        )

    def _collect_event_ids_from_raw_window_block(self, raw_window_block: List[WorkingWindow]) -> List[int]:
        event_ids: List[int] = []
        for window in raw_window_block:
            for event in window.events:
                event_id = _extract_event_reference_id(event if isinstance(event, dict) else {})
                if event_id is None:
                    continue
                event_ids.append(event_id)
        return self._validate_event_ids(event_ids)


    def _ground_episode_critical_events(self, payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            return []

        grounded_items: List[Dict[str, Any]] = []
        seen = set()
        for item in payload:
            if not isinstance(item, dict):
                continue

            critical_event_text = _safe_text(item.get("event"))
            if not critical_event_text:
                critical_event_text = _safe_text(item.get("critical_event"))
            if not critical_event_text:
                critical_event_text = _safe_text(item.get("name"))

            reason = _safe_text(item.get("reason"))
            if not reason:
                reason = _safe_text(item.get("rationale"))

            supporting_event_ids = _normalize_int_list(
                item.get("supporting_event_ids", item.get("supporting_evidence", []))
            )
            direct_event_ids = _normalize_int_list([item.get("event_id")])
            supporting_event_ids = self._validate_event_ids(direct_event_ids + supporting_event_ids)
            supporting_events = self._resolve_supporting_events(supporting_event_ids)
            grounded_supporting_event_ids = [
                int(event.get("event_id")) for event in supporting_events if event.get("event_id") is not None
            ]

            if not critical_event_text and supporting_events:
                critical_event_text = _safe_text(supporting_events[0].get("event_name"))
            if not critical_event_text:
                continue

            dedupe_key = (critical_event_text, tuple(grounded_supporting_event_ids))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            grounded_items.append(
                {
                    "event": critical_event_text,
                    "reason": reason,
                    "supporting_event_ids": grounded_supporting_event_ids,
                    "supporting_events": supporting_events,
                }
            )

        return grounded_items

    def _build_critical_events_memory_entry(
        self,
        *,
        episode_id: int,
        raw_window_block: List[WorkingWindow],
        critical_events: List[Dict[str, Any]],
    ) -> CriticalEventsMemoryEntry:
        first_window = raw_window_block[0]
        last_window = raw_window_block[-1]
        return CriticalEventsMemoryEntry(
            episode_id=episode_id,
            start_window=first_window.window_id,
            end_window=last_window.window_id,
            start_hour=first_window.start_hour,
            end_hour=last_window.end_hour,
            critical_events=deepcopy(critical_events),
        )

    def _build_default_episode_payload(
        self,
        raw_window_block: List[WorkingWindow],
        trend_block: List[TrendMemoryEntry],
    ) -> Dict[str, Any]:
        text_segments = []
        supporting_event_ids = self._collect_event_ids_from_raw_window_block(raw_window_block)
        trend_by_window = _trend_entry_by_window(trend_block)
        for window in raw_window_block:
            trend_entry = trend_by_window.get(window.window_id)
            trend_summary = "no tracked vitals"
            if trend_entry is not None:
                trend_summary = _trend_entry_summary(trend_entry.to_dict())
            text_segments.append(
                f"Window {window.window_id} (hour {window.start_hour:.1f}-{window.end_hour:.1f}): "
                f"raw_events={len(window.events)}; {trend_summary}"
            )
        if not text_segments:
            text_segments.append("No clinically meaningful progression identified in this episode block.")
        return {
            "episode_summary": {
                "text": " ".join(text_segments),
                "supporting_event_ids": supporting_event_ids[:12],
            },
            "critical_events": [],
        }

    def _build_default_insight_payload(self, episode: Episode) -> Dict[str, Any]:
        if not episode.supporting_event_ids:
            return {"insight_updates": [], "new_insights": []}
        return {
            "insight_updates": [],
            "new_insights": [
                {
                    "hypothesis": (
                        f"Window {episode.start_window}-{episode.end_window} trajectory reveals a patient-specific pattern "
                        f"that requires continued monitoring."
                    ),
                    "supporting_event_ids": episode.supporting_event_ids[:2],
                    "counter_event_ids": [],
                }
            ],
        }

    def _ground_episode(
        self,
        payload: Dict[str, Any],
        raw_window_block: List[WorkingWindow],
        trend_block: List[TrendMemoryEntry],
    ) -> Tuple[Episode, CriticalEventsMemoryEntry]:
        source_payload = payload if isinstance(payload, dict) else {}
        summary_payload = source_payload.get("episode_summary", source_payload.get("episode", {}))
        if not isinstance(summary_payload, dict):
            summary_payload = {}

        episode_text = _safe_text(summary_payload.get("text"))
        if not episode_text:
            episode_text = _safe_text(summary_payload.get("episode_summary"))
        if not episode_text:
            episode_text = _safe_text(source_payload.get("text"))

        default_payload = self._build_default_episode_payload(raw_window_block, trend_block)
        if not episode_text:
            episode_text = _safe_text(default_payload.get("episode_summary", {}).get("text"))

        supporting_event_ids = _normalize_int_list(
            summary_payload.get(
                "supporting_event_ids",
                summary_payload.get("supporting_evidence", source_payload.get("supporting_event_ids", [])),
            )
        )
        supporting_event_ids = self._validate_event_ids(supporting_event_ids)
        if not supporting_event_ids:
            supporting_event_ids = self._validate_event_ids(
                _normalize_int_list(default_payload.get("episode_summary", {}).get("supporting_event_ids", []))
            )

        supporting_events = self._resolve_supporting_events(supporting_event_ids)
        supporting_event_ids = [
            int(item.get("event_id")) for item in supporting_events if item.get("event_id") is not None
        ]

        critical_events_payload = source_payload.get("critical_events", default_payload.get("critical_events", []))
        grounded_critical_events = self._ground_episode_critical_events(critical_events_payload)

        first_window = raw_window_block[0]
        last_window = raw_window_block[-1]
        episode_id = self._next_episode_id
        episode = Episode(
            episode_id=episode_id,
            start_window=first_window.window_id,
            end_window=last_window.window_id,
            start_hour=first_window.start_hour,
            end_hour=last_window.end_hour,
            episode_summary=episode_text,
            supporting_event_ids=supporting_event_ids,
            supporting_events=supporting_events,
        )
        critical_events_entry = self._build_critical_events_memory_entry(
            episode_id=episode_id,
            raw_window_block=raw_window_block,
            critical_events=grounded_critical_events,
        )
        self._next_episode_id += 1
        return episode, critical_events_entry

    def _build_evidence_events(self, event_ids: List[int]) -> List[EvidenceEvent]:
        evidence_events: List[EvidenceEvent] = []
        seen = set()
        for event_id in event_ids:
            if event_id in seen:
                continue
            seen.add(event_id)
            evidence_events.append(EvidenceEvent(event_id=event_id, name_str=self._resolve_event_name_str(event_id)))
        return evidence_events

    def _refresh_insight_evidence_objects(self, insight: Insight) -> None:
        insight.supporting_evidence = self._build_evidence_events(insight.supporting_event_ids)
        insight.counter_evidence = self._build_evidence_events(insight.counter_event_ids)

    def _recompute_insight_scores(self, insights: List[Insight]) -> None:
        for insight in insights:
            supporting_count = len(insight.supporting_event_ids) + len(insight.supporting_trend_evidence)
            counter_count = len(insight.counter_event_ids) + len(insight.counter_trend_evidence)
            insight.score = float(supporting_count - counter_count)

    def _apply_insight_updates(
        self,
        existing_insights: List[Insight],
        parsed_payload: Dict[str, Any],
    ) -> List[Insight]:
        now_ts = time.time()
        insights_by_id = {insight.insight_id: deepcopy(insight) for insight in existing_insights}

        updates = parsed_payload.get("insight_updates", parsed_payload.get("updated_insights", []))
        if not isinstance(updates, list):
            updates = []

        new_insights = parsed_payload.get("new_insights", [])
        if not isinstance(new_insights, list):
            new_insights = []

        for item in updates + new_insights:
            if not isinstance(item, dict):
                continue

            hypothesis = _safe_text(item.get("hypothesis"))
            supporting_evidence_ids, supporting_trend_evidence = _split_evidence_references(
                item.get("supporting_evidence", [])
            )
            counter_evidence_ids, counter_trend_evidence = _split_evidence_references(item.get("counter_evidence", []))
            supporting = self._validate_event_ids(
                _normalize_int_list(item.get("supporting_event_ids", [])) + supporting_evidence_ids
            )
            counter = self._validate_event_ids(
                _normalize_int_list(item.get("counter_event_ids", [])) + counter_evidence_ids
            )
            supporting_trends = _normalize_text_list(
                _normalize_text_list(item.get("supporting_trend_evidence", [])) + supporting_trend_evidence
            )
            counter_trends = _normalize_text_list(
                _normalize_text_list(item.get("counter_trend_evidence", [])) + counter_trend_evidence
            )

            if not hypothesis and not supporting and not counter and not supporting_trends and not counter_trends:
                continue

            raw_insight_id = item.get("insight_id", item.get("hypothesis_id"))
            insight_id: Optional[int] = None
            if raw_insight_id is not None:
                try:
                    parsed_id = int(raw_insight_id)
                    if parsed_id in insights_by_id:
                        insight_id = parsed_id
                except (TypeError, ValueError):
                    insight_id = None

            if insight_id is not None:
                insight = insights_by_id[insight_id]
                if hypothesis:
                    insight.hypothesis = hypothesis
                insight.supporting_event_ids = self._validate_event_ids(
                    _normalize_int_list(insight.supporting_event_ids + supporting)
                )
                insight.counter_event_ids = self._validate_event_ids(
                    _normalize_int_list(insight.counter_event_ids + counter)
                )
                insight.supporting_trend_evidence = _normalize_text_list(
                    insight.supporting_trend_evidence + supporting_trends
                )
                insight.counter_trend_evidence = _normalize_text_list(insight.counter_trend_evidence + counter_trends)
                self._refresh_insight_evidence_objects(insight)
                insight.updated_at = now_ts
                insights_by_id[insight_id] = insight
                continue

            if not hypothesis:
                continue

            insight_id = self._next_insight_id
            self._next_insight_id += 1
            new_insight = Insight(
                insight_id=insight_id,
                hypothesis=hypothesis,
                supporting_event_ids=supporting,
                counter_event_ids=counter,
                supporting_trend_evidence=supporting_trends,
                counter_trend_evidence=counter_trends,
                created_at=now_ts,
                updated_at=now_ts,
                score=0.0,
            )
            self._refresh_insight_evidence_objects(new_insight)
            insights_by_id[insight_id] = new_insight

        updated_insights = list(insights_by_id.values())
        for insight in updated_insights:
            self._refresh_insight_evidence_objects(insight)
        self._recompute_insight_scores(updated_insights)

        if len(updated_insights) > self.max_insights:
            updated_insights.sort(key=lambda insight: (insight.score, insight.updated_at))
            prune_count = len(updated_insights) - self.max_insights
            self.total_insights_pruned += prune_count
            updated_insights = updated_insights[prune_count:]

        updated_insights.sort(key=lambda insight: insight.insight_id)
        return updated_insights


    def _append_trend_entry(self, memory: ICUEvoMemory, entry: TrendMemoryEntry) -> None:
        memory.trend_memory.append(entry)
        memory.last_processed_window_index = entry.window_index
        memory.last_processed_start_hour = entry.start_hour
        memory.last_processed_end_hour = entry.end_hour

    def _prune_episode_level_memory(self, memory: ICUEvoMemory) -> None:
        if len(memory.trajectory_memory) <= self.max_trajectory_entries:
            return

        prune_count = len(memory.trajectory_memory) - self.max_trajectory_entries
        self.total_trajectory_entries_pruned += prune_count
        memory.trajectory_memory = memory.trajectory_memory[prune_count:]

        if prune_count <= 0:
            return
        if prune_count >= len(memory.critical_events_memory):
            memory.critical_events_memory = []
            return
        memory.critical_events_memory = memory.critical_events_memory[prune_count:]

    def create_memory_snapshots(
        self,
        windows: List[Dict[str, Any]],
        patient_metadata: Dict[str, Any],
        verbose: bool = True,
    ) -> Tuple[ICUEvoMemory, List[Dict[str, Any]]]:
        self.total_patients += 1
        self._current_patient_id = (
            f"{patient_metadata.get('subject_id', 'unknown')}_{patient_metadata.get('icu_stay_id', 'unknown')}"
        )
        self._event_id_to_window = {}
        self._event_id_to_time = {}
        self._event_id_to_raw_event = {}
        self._event_id_to_name_str = {}
        self._next_insight_id = 1
        self._next_episode_id = 1

        patient_metadata_with_pre_icu = self._build_patient_metadata_with_pre_icu_summary(
            windows=windows,
            patient_metadata=patient_metadata,
        )
        memory = ICUEvoMemory(patient_metadata=deepcopy(patient_metadata_with_pre_icu))
        snapshots = []
        episode_raw_window_block: List[WorkingWindow] = []
        latest_episode: Optional[Episode] = None
        latest_critical_events_entry: Optional[CriticalEventsMemoryEntry] = None
        latest_episode_trend_block: List[TrendMemoryEntry] = []

        if verbose:
            print(f"Processing patient with {len(windows)} windows...")
            pre_icu_summary = _safe_text(memory.patient_metadata.get("pre_icu_history_summary"))
            if pre_icu_summary:
                print(f"  Pre-ICU summary prepared ({len(pre_icu_summary)} chars)")

        for idx, window in enumerate(windows):
            current_events = window.get("current_events", [])
            if not isinstance(current_events, list):
                current_events = []
            hours = float(window.get("hours_since_admission", 0.0))
            end_hour = hours + float(window["current_window_hours"])

            if verbose:
                print(f"  Window {idx+1}/{len(windows)} (Hour {hours:.1f})...", end=" ")

            self._register_event_ids(current_events, idx)
            memory.working_memory = self.perception_agent.update_working_memory(memory.working_memory, window, idx)
            episode_raw_window_block.append(deepcopy(memory.working_memory[-1]))
            if len(episode_raw_window_block) > self.episode_block_windows:
                episode_raw_window_block = episode_raw_window_block[-self.episode_block_windows :]

            observation_output = self.observation_agent.analyze(window)
            self.total_observation_runs += 1
            trend_entry = self._trend_entry_from_detector_output(
                detector_output=observation_output,
                window_index=idx,
                start_hour=hours,
                end_hour=end_hour,
            )
            self._append_trend_entry(memory, trend_entry)

            run_episode_block = len(memory.trend_memory) >= self.episode_block_windows and (
                len(memory.trend_memory) % self.episode_block_windows == 0
            )

            if run_episode_block:
                trend_block = memory.trend_memory[-self.episode_block_windows :]
                raw_window_block = episode_raw_window_block[-self.episode_block_windows :]
                prior_episode_summary = _format_prior_episode_summary_text(memory.trajectory_memory[-1:])
                episode_parsed, episode_raw, episode_usage, episode_prompt, episode_parse_error = (
                    self.assessment_agent.analyze(
                        raw_window_block=raw_window_block,
                        trend_block=trend_block,
                        patient_metadata=memory.patient_metadata,
                        prior_episode_summary=prior_episode_summary,
                    )
                )
                self.total_episode_calls += 1

                if episode_parse_error is not None:
                    episode_parsed = self._build_default_episode_payload(raw_window_block, trend_block)

                grounded_episode, grounded_critical_events = self._ground_episode(
                    episode_parsed,
                    raw_window_block,
                    trend_block,
                )
                latest_episode = grounded_episode
                latest_critical_events_entry = grounded_critical_events
                latest_episode_trend_block = deepcopy(trend_block)
                memory.trajectory_memory.append(grounded_episode.to_dict())
                memory.critical_events_memory.append(grounded_critical_events)
                self._prune_episode_level_memory(memory)

                self._log_call(
                    step_type="assessment_agent",
                    window_index=idx,
                    hours_since_admission=hours,
                    prompt=episode_prompt,
                    response=episode_raw,
                    usage=episode_usage,
                    parsed_response=episode_parsed,
                    metadata={
                        "episode_parse_error": episode_parse_error,
                        "episode_id": grounded_episode.episode_id,
                        "episode_start_window": grounded_episode.start_window,
                        "episode_end_window": grounded_episode.end_window,
                        "episode_support_count": len(grounded_episode.supporting_event_ids),
                        "episode_critical_event_count": len(grounded_critical_events.critical_events),
                    },
                )

            run_insight_block = len(memory.trend_memory) >= self.insight_block_windows and (
                len(memory.trend_memory) % self.insight_block_windows == 0
            )

            if run_insight_block and latest_episode is not None:
                insight_parsed, insight_raw, insight_usage, insight_prompt, insight_parse_error = (
                    self.insight_agent.analyze(
                        current_insights=memory.insights,
                        episode=latest_episode,
                        critical_events_entry=latest_critical_events_entry,
                        trend_block=latest_episode_trend_block,
                        patient_metadata=memory.patient_metadata,
                    )
                )
                self.total_insight_calls += 1

                if insight_parse_error is not None:
                    insight_parsed = self._build_default_insight_payload(latest_episode)

                memory.insights = self._apply_insight_updates(memory.insights, insight_parsed)

                self._log_call(
                    step_type="insight_agent",
                    window_index=idx,
                    hours_since_admission=hours,
                    prompt=insight_prompt,
                    response=insight_raw,
                    usage=insight_usage,
                    parsed_response=insight_parsed,
                    metadata={
                        "insight_parse_error": insight_parse_error,
                        "num_insights": len(memory.insights),
                        "episode_id": latest_episode.episode_id,
                    },
                )
            else:
                self._recompute_insight_scores(memory.insights)

            snapshots.append(compact_snapshot(memory.to_dict()))

            if verbose:
                print(
                    f"trends={len(memory.trend_memory)} "
                    f"episodes={len(memory.trajectory_memory)} "
                    f"critical_events={len(memory.critical_events_memory)} "
                    f"insights={len(memory.insights)}"
                )

        return memory, snapshots
