import json
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from utils.discharge_summary_selector import select_discharge_summaries_for_icu_stays
from utils.event_format import format_event_line as format_shared_event_line
from utils.time_format import format_timestamp_minute

VITAL_CODE = "VITALS"

# Canonical vital names and matching rules derived from the MIMIC-demo VITALS code_specifics.
RELEVANT_VITAL_PATTERNS = {
    "heart rate": [r"\bheart rate\b"],
    "respiratory rate": [r"\brespiratory rate\b"],
    "oxygen saturation": [r"\bo2 saturation\b", r"\bspo2\b", r"\barterial o2 saturation\b"],
    "systolic blood pressure": [
        r"\bblood pressure systolic\b",
        r"\bart bp systolic\b",
        r"\bmanual blood pressure systolic\b",
    ],
    "diastolic blood pressure": [
        r"\bblood pressure diastolic\b",
        r"\bart bp diastolic\b",
        r"\bmanual blood pressure diastolic\b",
    ],
    "mean arterial pressure": [
        r"\bblood pressure mean\b",
        r"\bart bp mean\b",
        r"\bmap\b",
        r"\biabp mean\b",
    ],
    "temperature": [
        r"\btemperature fahrenheit\b",
        r"\btemperature celsius\b",
        r"\bblood temperature\b",
        r"^temperature$",
    ],
}

IRRELEVANT_VITAL_PATTERNS = [
    r"\balarm\b",
    r"\bsite\b",
    r"\bsource\b",
    r"\bdesat limit\b",
    r"\bskin temperature\b",
]

DEFAULT_RELATIVE_REPORT_CODES: List[str] = []


class PreICUHistoryProcessor:
    """
    Encapsulates pre-ICU history extraction/selection/formatting.

    This keeps report-first history logic separate from window slicing so we can
    later swap in an LLM-based summarizer without touching parser core logic.
    """

    def __init__(
        self,
        events_df_getter: Callable[[], Optional[pd.DataFrame]],
        clean_events_fn: Callable[[List[Dict]], List[Dict]],
    ) -> None:
        self._events_df_getter = events_df_getter
        self._clean_events_fn = clean_events_fn

    def _events_df(self) -> Optional[pd.DataFrame]:
        return self._events_df_getter()

    def extract_report_candidates(
        self,
        trajectory: Dict,
        relative_report_codes: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Extract report candidates from before current ICU admission.

        Ranking priority:
        1) NOTE_DISCHARGESUMMARY (highest priority)
        2) Relative report NOTE_* codes (optional, default none)
        Then by recency (most recent first).
        """
        events_df = self._events_df()
        if events_df is None:
            return []

        subject_id = trajectory["subject_id"]
        current_enter_time = pd.to_datetime(trajectory["enter_time"])

        report_codes = {str(code) for code in (relative_report_codes or DEFAULT_RELATIVE_REPORT_CODES) if str(code).strip()}
        report_codes.add("NOTE_DISCHARGESUMMARY")

        patient_events = events_df[events_df["subject_id"] == subject_id].copy()
        if len(patient_events) == 0:
            return []
        if "code" not in patient_events.columns or "time" not in patient_events.columns:
            return []

        patient_events["event_time"] = pd.to_datetime(patient_events["time"], errors="coerce")
        candidates = patient_events[
            (patient_events["code"].isin(report_codes))
            & pd.notna(patient_events["event_time"])
            & (patient_events["event_time"] < current_enter_time)
        ].copy()

        if len(candidates) == 0:
            return []

        candidates["priority"] = candidates["code"].apply(lambda value: 0 if value == "NOTE_DISCHARGESUMMARY" else 1)
        candidates = candidates.sort_values(["priority", "event_time"], ascending=[True, False])

        results: List[Dict] = []
        for _, event in candidates.iterrows():
            event_time = event.get("event_time")
            results.append(
                {
                    "subject_id": int(subject_id),
                    "time": event_time.isoformat() if pd.notna(event_time) else None,
                    "code": event.get("code"),
                    "code_specifics": event.get("code_specifics"),
                    "text_value": event.get("text_value"),
                    "is_discharge_summary": event.get("code") == "NOTE_DISCHARGESUMMARY",
                    "hours_before_current_icu": (
                        (current_enter_time - event_time).total_seconds() / 3600 if pd.notna(event_time) else None
                    ),
                }
            )

        return results

    @staticmethod
    def format_reports_content(reports: List[Dict]) -> str:
        """Render selected pre-ICU reports into a compact text block."""
        lines = []
        for i, report in enumerate(reports, start=1):
            report_code = str(report.get("code") or "UNKNOWN")
            report_title = "Discharge Summary" if report_code == "NOTE_DISCHARGESUMMARY" else report_code
            hours_before = report.get("hours_before_current_icu")
            relation_text = "relative to ICU admission: unknown"
            if isinstance(hours_before, (int, float)):
                signed_hours = float(hours_before)
                if signed_hours >= 0:
                    relation_text = f"{signed_hours / 24.0:.1f} days / {signed_hours:.1f} hours before ICU admission"
                else:
                    relation_text = (
                        f"{abs(signed_hours) / 24.0:.1f} days / {abs(signed_hours):.1f} hours after ICU admission"
                    )

            timestamp_text = format_timestamp_minute(report.get("time"))
            if timestamp_text == "Unknown":
                timestamp_text = "unknown"

            lines.append((f"--- Report {i}: {report_title} " f"(timestamp: {timestamp_text}; {relation_text}) ---"))

            text = str(report.get("text_value") or "").strip()
            lines.append(text if text else "(No report text)")
            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def select_reports_with_per_code_cap(report_candidates: List[Dict], per_code_cap: int) -> List[Dict]:
        """
        Select prioritized pre-ICU reports while enforcing a per-code cap.

        report_candidates must already be sorted by priority/recency. This preserves
        NOTE_DISCHARGESUMMARY-first ordering while allowing optional relative NOTE_*
        codes to be included up to the same per-code cap.
        """
        if per_code_cap <= 0:
            return []

        selected: List[Dict] = []
        counts_by_code: Dict[str, int] = {}

        for report in report_candidates:
            code = str(report.get("code") or "").strip()
            if not code:
                continue

            current_count = counts_by_code.get(code, 0)
            if current_count >= per_code_cap:
                continue

            selected.append(report)
            counts_by_code[code] = current_count + 1

        return selected

    @staticmethod
    def _format_pre_icu_event_line(event: Dict[str, Any]) -> str:
        """Format pre-ICU history events with shared event formatter."""
        return format_shared_event_line(
            event,
            time_keys=("time",),
            missing_time_text="unknown",
            missing_code_text="UNKNOWN",
            empty_as_json=False,
        )

    @staticmethod
    def _annotate_pre_icu_history_prompt_ids(
        events: List[Dict[str, Any]],
        *,
        prefix: str = "H",
        id_field: str = "prompt_event_id",
    ) -> List[Dict[str, Any]]:
        """Attach pre-ICU history prompt IDs like H0, H1, ... for text rendering."""
        annotated: List[Dict[str, Any]] = []
        for idx, event in enumerate(events):
            if isinstance(event, dict):
                enriched = dict(event)
                enriched[id_field] = f"{prefix}{idx}"
                annotated.append(enriched)
            else:
                annotated.append(event)
        return annotated

    def extract_fallback_events(
        self,
        trajectory: Dict,
        lookback_hours: float = 72.0,
    ) -> List[Dict]:
        """Extract pre-ICU events from [enter_time-lookback_hours, enter_time)."""
        events_df = self._events_df()
        if events_df is None:
            return []
        if lookback_hours <= 0:
            return []

        subject_id = trajectory["subject_id"]
        current_enter_time = pd.to_datetime(trajectory["enter_time"])
        fallback_start = current_enter_time - timedelta(hours=lookback_hours)

        patient_events = events_df[events_df["subject_id"] == subject_id].copy()
        if len(patient_events) == 0:
            return []
        if "time" not in patient_events.columns:
            return []

        patient_events["event_time"] = pd.to_datetime(patient_events["time"], errors="coerce")
        fallback_events = patient_events[
            pd.notna(patient_events["event_time"])
            & (patient_events["event_time"] >= fallback_start)
            & (patient_events["event_time"] < current_enter_time)
        ].copy()
        fallback_events = fallback_events.sort_values("event_time", ascending=True)
        return fallback_events.to_dict("records")

    @staticmethod
    def format_pre_icu_events_content(
        pre_icu_history_events: List[Dict],
        pre_icu_history_hours: float,
    ) -> str:
        """Render cleaned pre-ICU events into a compact text block."""
        if not pre_icu_history_events:
            return ""

        lines = [
            (
                "Pre-ICU history events "
                f"({len(pre_icu_history_events)} events in last {float(pre_icu_history_hours):.1f}h):"
            )
        ]
        display_events = PreICUHistoryProcessor._annotate_pre_icu_history_prompt_ids(pre_icu_history_events)
        for event in display_events:
            lines.append(PreICUHistoryProcessor._format_pre_icu_event_line(event))
        return "\n".join(lines)

    def build_history_context(
        self,
        trajectory: Dict,
        num_discharge_summaries: int,
        relative_report_codes: Optional[List[str]],
        pre_icu_history_hours: float = 72.0,
    ) -> Dict[str, Any]:
        """
        Build unified pre-ICU history content from events and optional reports.

        Returns:
            {
                "source": "pre_icu_history|none",
                "items": int,
                "content": Optional[str],
                "history_events": List[Dict],
                "event_items": int,
                "report_items": int,
                "per_code_cap": int,
            }
        """
        pre_icu_raw_events = self.extract_fallback_events(
            trajectory=trajectory,
            lookback_hours=float(pre_icu_history_hours),
        )
        pre_icu_history_events = self._clean_events_fn(pre_icu_raw_events)
        events_content = self.format_pre_icu_events_content(
            pre_icu_history_events,
            pre_icu_history_hours=float(pre_icu_history_hours),
        )

        per_code_cap = max(0, int(num_discharge_summaries))
        report_candidates = self.extract_report_candidates(
            trajectory,
            relative_report_codes=relative_report_codes,
        )
        selected_reports = self.select_reports_with_per_code_cap(
            report_candidates,
            per_code_cap=per_code_cap,
        )
        historical_discharge_summary_items = int(
            sum(1 for report in selected_reports if str(report.get("code") or "").strip() == "NOTE_DISCHARGESUMMARY")
        )
        report_content = self.format_reports_content(selected_reports) if selected_reports else ""

        sections: List[str] = []
        if events_content:
            sections.append(events_content)
        if report_content:
            if sections:
                sections.append("")
            sections.append("Pre-ICU historical reports:")
            sections.append(report_content)

        content = "\n".join(sections).strip() if sections else None
        if content:
            return {
                "source": "pre_icu_history",
                "items": len(pre_icu_history_events) + len(selected_reports),
                "event_items": len(pre_icu_history_events),
                "report_items": len(selected_reports),
                "historical_discharge_summary_items": historical_discharge_summary_items,
                "content": content,
                "history_events": pre_icu_history_events,
                "per_code_cap": per_code_cap,
                "pre_icu_history_hours": float(pre_icu_history_hours),
            }

        return {
            "source": "none",
            "items": 0,
            "event_items": 0,
            "report_items": 0,
            "historical_discharge_summary_items": 0,
            "content": None,
            "history_events": [],
            "per_code_cap": per_code_cap,
            "pre_icu_history_hours": float(pre_icu_history_hours),
        }


class MIMICDataParser:
    """
    Parser for MIMIC-demo dataset to format patient trajectories for Oracle evaluation.

    The parser loads ICU stay metadata and clinical events, then formats them into
    time-windowed trajectories suitable for retrospective evaluation.
    """

    def __init__(
        self,
        events_path: str,
        icu_stay_path: str,
        discharge_summary_max_days_after_leave: float = 7.0,
        require_discharge_summary_for_icu_stays: bool = True,
    ):
        """
        Initialize the parser with paths to MIMIC-demo data files.

        Args:
            events_path: Path to events parquet dataset directory (e.g., data/mimic-demo/events)
            icu_stay_path: Path to ICU-stay parquet dataset directory (e.g., data/mimic-demo/icu_stay)
            discharge_summary_max_days_after_leave: Selector window for post-ICU discharge summary matching.
            require_discharge_summary_for_icu_stays: Keep only ICU stays with selector-linked discharge
                summary available. ICU stays without extractable summary are skipped.
        """
        if discharge_summary_max_days_after_leave <= 0:
            raise ValueError("discharge_summary_max_days_after_leave must be > 0")

        self.events_path = events_path
        self.icu_stay_path = icu_stay_path
        self.events_df = None
        self.icu_stay_df = None

        self._relevant_vitals_cache: Dict[int, Dict[str, List[str]]] = {}
        self.discharge_summary_max_days_after_leave = float(discharge_summary_max_days_after_leave)
        self.require_discharge_summary_for_icu_stays = bool(require_discharge_summary_for_icu_stays)
        self.discharge_summary_selection_df: Optional[pd.DataFrame] = None
        self._selected_discharge_summary_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.pre_icu_history_processor = PreICUHistoryProcessor(
            events_df_getter=lambda: self.events_df,
            clean_events_fn=self._clean_events_list,
        )

    def _compute_discharge_summary_selection(self, icu_stay_df: pd.DataFrame) -> pd.DataFrame:
        """Run discharge-summary selector for a given ICU-stay table."""
        if self.events_df is None:
            return pd.DataFrame()
        if icu_stay_df is None or len(icu_stay_df) == 0:
            return pd.DataFrame()

        selection_df = select_discharge_summaries_for_icu_stays(
            events_df=self.events_df,
            icu_stay_df=icu_stay_df,
            max_days_after_leave=self.discharge_summary_max_days_after_leave,
        ).copy()
        if len(selection_df) == 0:
            return selection_df

        selection_df["subject_id"] = pd.to_numeric(selection_df["subject_id"], errors="coerce").astype("Int64")
        selection_df["icu_stay_id"] = pd.to_numeric(selection_df["icu_stay_id"], errors="coerce").astype("Int64")
        selection_df["enter_time"] = pd.to_datetime(selection_df["enter_time"], errors="coerce")
        selection_df["leave_time"] = pd.to_datetime(selection_df["leave_time"], errors="coerce")
        selection_df["selected_note_time"] = pd.to_datetime(selection_df["selected_note_time"], errors="coerce")
        selection_df["stay_hosp_stay_id"] = pd.to_numeric(selection_df["stay_hosp_stay_id"], errors="coerce").astype(
            "Int64"
        )
        selection_df["selected_note_hosp_stay_id"] = pd.to_numeric(
            selection_df["selected_note_hosp_stay_id"], errors="coerce"
        ).astype("Int64")
        return selection_df

    def _build_selected_discharge_summary_map(self) -> None:
        """Build (subject_id, icu_stay_id) -> selected discharge summary lookup map."""
        self._selected_discharge_summary_map = {}
        if self.discharge_summary_selection_df is None or len(self.discharge_summary_selection_df) == 0:
            return

        selected_rows = self.discharge_summary_selection_df[
            self.discharge_summary_selection_df["selected"] == True
        ]  # noqa: E712
        for _, row in selected_rows.iterrows():
            if pd.isna(row.get("subject_id")) or pd.isna(row.get("icu_stay_id")):
                continue
            key = (int(row["subject_id"]), int(row["icu_stay_id"]))
            note_time = row.get("selected_note_time")
            self._selected_discharge_summary_map[key] = {
                "time": note_time.isoformat() if pd.notna(note_time) else None,
                "text_value": row.get("selected_note_text_value"),
                "code_specifics": row.get("selected_note_code_specifics"),
                "hosp_stay_id": row.get("selected_note_hosp_stay_id"),
                "selection_rule": row.get("selection_rule"),
                "delta_hours_after_leave": row.get("selected_note_delta_hours_after_leave"),
            }

    def _selected_discharge_summary_for_stay(self, subject_id: int, icu_stay_id: int) -> Optional[Dict[str, Any]]:
        """Fetch selected discharge summary metadata for one ICU stay."""
        return self._selected_discharge_summary_map.get((int(subject_id), int(icu_stay_id)))

    def load_data(self):
        """Load the MIMIC-demo datasets into memory."""
        print(f"Loading events data from {self.events_path}...")
        self.events_df = pd.read_parquet(self.events_path)

        print(f"Loading ICU stay data from {self.icu_stay_path}...")
        self.icu_stay_df = pd.read_parquet(self.icu_stay_path)

        # Convert time columns to datetime
        self.icu_stay_df["enter_time"] = pd.to_datetime(self.icu_stay_df["enter_time"])
        self.icu_stay_df["leave_time"] = pd.to_datetime(self.icu_stay_df["leave_time"])
        self.icu_stay_df["birth_time"] = pd.to_datetime(self.icu_stay_df["birth_time"])

        initial_count = len(self.icu_stay_df)
        self.icu_stay_df = self.icu_stay_df[self.icu_stay_df["icu_duration_hours"] >= 4]
        filtered_count = initial_count - len(self.icu_stay_df)
        print(f"Filtered out {filtered_count} ICU stays with duration < 4h")

        # Keep ALL ICU stays for each patient (treat multiple stays independently)
        # Sort by subject_id and enter_time for consistent ordering
        self.icu_stay_df = self.icu_stay_df.sort_values(["subject_id", "enter_time"]).reset_index(drop=True)

        selection_df = self._compute_discharge_summary_selection(self.icu_stay_df)
        if self.require_discharge_summary_for_icu_stays:
            selected_pairs = {
                (int(row["subject_id"]), int(row["icu_stay_id"]))
                for _, row in selection_df.iterrows()
                if row.get("selected") == True  # noqa: E712
                and pd.notna(row.get("subject_id"))
                and pd.notna(row.get("icu_stay_id"))
            }

            before_require_filter = len(self.icu_stay_df)
            if selected_pairs:
                self.icu_stay_df = self.icu_stay_df[
                    self.icu_stay_df.apply(
                        lambda stay: (int(stay["subject_id"]), int(stay["icu_stay_id"])) in selected_pairs,
                        axis=1,
                    )
                ].reset_index(drop=True)
            else:
                self.icu_stay_df = self.icu_stay_df.iloc[0:0].copy()

            removed_without_summary = before_require_filter - len(self.icu_stay_df)
            if removed_without_summary > 0:
                print(f"Filtered out {removed_without_summary} ICU stays without extractable discharge summary")

            if len(selection_df) > 0:
                selection_df = selection_df[selection_df["selected"] == True].reset_index(drop=True)  # noqa: E712

        # Count patients with multiple ICU stays
        multiple_stays = self.icu_stay_df.groupby("subject_id").size()
        patients_with_multiple_stays = (multiple_stays > 1).sum()
        total_stays = len(self.icu_stay_df)
        unique_patients = len(multiple_stays)

        self.discharge_summary_selection_df = selection_df
        self._build_selected_discharge_summary_map()
        selected_summaries = len(self._selected_discharge_summary_map)

        print(
            f"Loaded {len(self.events_df)} events and {total_stays} ICU stays from {unique_patients} unique patients"
        )
        print(f"  - Patients with multiple ICU stays: {patients_with_multiple_stays}")
        print(f"  - ICU stays with selected discharge summary: {selected_summaries}")

    def get_patient_trajectory(
        self,
        subject_id: int,
        icu_stay_id: int,
        icu_stay: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Extract a complete patient trajectory for a specific ICU stay.

        Args:
            subject_id: Patient identifier
            icu_stay_id: ICU stay identifier
            icu_stay: Optional pre-fetched ICU stay row. Supplying this avoids
                an extra lookup in icu_stay_df.

        Returns:
            Dictionary containing patient metadata and all events for this ICU stay
        """
        if self.events_df is None or self.icu_stay_df is None:
            raise ValueError("Dataframes are empty. Call load_data() first.")

        if icu_stay is None:
            # Fallback for external callers that pass only IDs.
            icu_stay = self.icu_stay_df[
                (self.icu_stay_df["subject_id"] == subject_id) & (self.icu_stay_df["icu_stay_id"] == icu_stay_id)
            ].iloc[0]
        else:
            # Ensure consistency if caller passes an ICU row.
            if int(icu_stay["subject_id"]) != int(subject_id) or int(icu_stay["icu_stay_id"]) != int(icu_stay_id):
                raise ValueError("Provided ICU stay row does not match subject_id / icu_stay_id.")

        # Get all events for this ICU stay using event_idx bounds.
        # Contract: min_event_idx/max_event_idx in icu_stay_df are event_idx values.
        min_idx = int(icu_stay["min_event_idx"])
        max_idx = int(icu_stay["max_event_idx"])
        if min_idx > max_idx:
            raise ValueError(
                f"Invalid event_idx bounds for subject_id={subject_id}, icu_stay_id={icu_stay_id}: "
                f"min_event_idx={min_idx} > max_event_idx={max_idx}"
            )

        expected_n_events: Optional[int] = None
        raw_n_events = icu_stay.get("n_events")
        if pd.notna(raw_n_events):
            try:
                expected_n_events = int(raw_n_events)
            except (TypeError, ValueError):
                expected_n_events = None

        if "subject_id" not in self.events_df.columns:
            raise ValueError("events_df is missing required column: subject_id")
        if "event_idx" not in self.events_df.columns:
            raise ValueError("events_df is missing required column: event_idx")

        subject_events = self.events_df[self.events_df["subject_id"] == int(subject_id)].copy()
        event_idx_series = pd.to_numeric(subject_events["event_idx"], errors="coerce")
        patient_events = subject_events[(event_idx_series >= min_idx) & (event_idx_series <= max_idx)].copy()
        patient_events = patient_events.sort_values("event_idx", ascending=True, kind="mergesort")

        if expected_n_events is not None and len(patient_events) != expected_n_events:
            raise ValueError(
                "Extracted event count does not match n_events for "
                f"subject_id={subject_id}, icu_stay_id={icu_stay_id}: "
                f"expected={expected_n_events}, extracted={len(patient_events)}, "
                f"min_event_idx={min_idx}, max_event_idx={max_idx}"
            )

        # Preserve a raw source index for traceability back to the dataset.
        source_index: Optional[pd.Series] = None
        for candidate in ("source_event_index", "event_id", "event_idx", "event_index"):
            if candidate not in patient_events.columns:
                continue
            parsed_source = pd.to_numeric(patient_events[candidate], errors="coerce")
            if parsed_source.notna().any():
                source_index = parsed_source
                break
        if source_index is None:
            source_index = pd.Series(patient_events.index, index=patient_events.index, dtype="float64")
        patient_events["source_event_index"] = source_index.astype("Int64")

        if "time" in patient_events.columns:
            event_time_series = pd.to_datetime(patient_events["time"], errors="coerce")
        else:
            event_time_series = pd.Series(pd.NaT, index=patient_events.index, dtype="datetime64[ns]")

        # ICU-local event index (0-based, only for events within this ICU stay).
        patient_events["icu_event_index"] = pd.Series(pd.NA, index=patient_events.index, dtype="Int64")
        icu_time_mask = (
            event_time_series.notna()
            & (event_time_series >= pd.to_datetime(icu_stay["enter_time"]))
            & (event_time_series <= pd.to_datetime(icu_stay["leave_time"]))
        )
        if bool(icu_time_mask.any()):
            icu_order = event_time_series[icu_time_mask].sort_values(kind="mergesort").index
            patient_events.loc[icu_order, "icu_event_index"] = range(len(icu_order))

        # Keep trajectory-wide index for backward compatibility with downstream tooling.
        patient_events["event_index"] = range(len(patient_events))

        # Calculate age at admission
        age_at_admission = (icu_stay["enter_time"] - icu_stay["birth_time"]).days / 365.25

        # Extract gender from META_GENDER event
        gender = None
        gender_events = patient_events[patient_events["code"] == "META_GENDER"]
        if len(gender_events) > 0:
            gender = gender_events.iloc[0].get("code_specifics", None)

        death_time = icu_stay["death_time"] if pd.notna(icu_stay["death_time"]) else None

        trajectory = {
            "subject_id": int(subject_id),
            "icu_stay_id": int(icu_stay_id),
            "enter_time": icu_stay["enter_time"].isoformat(),
            "leave_time": icu_stay["leave_time"].isoformat(),
            "age_at_admission": float(age_at_admission),
            "gender": gender,
            "icu_duration_hours": float(icu_stay["icu_duration_hours"]),
            "survived": bool(icu_stay["survived"]),
            "death_time": death_time.isoformat() if death_time is not None else None,
            "readmission": pd.notna(icu_stay["readm_time"]),
            "readm_duration_hours": (
                float(icu_stay["readm_duration_hours"]) if pd.notna(icu_stay["readm_duration_hours"]) else None
            ),
            "events": patient_events.to_dict("records"),
        }

        return trajectory

    def _clean_event(self, event: Dict, prev_event_time=None) -> Dict:
        """
        Clean and filter event data to keep only relevant fields.

        This reduces context size by keeping only clinically relevant information:
        - time: Event timestamp (formatted as YYYY-MM-DD HH:MM:SS)
        - code: Item identifier
        - numeric_value: Numeric measurement
        - code_specifics: Label/description
        - end_time: End timestamp if available (formatted as YYYY-MM-DD HH:MM:SS)
        - text_value: Text value or unit
        - time_delta_minutes: Minutes since previous event

        Args:
            event: Raw event dictionary
            prev_event_time: Previous event timestamp for calculating delta

        Returns:
            Cleaned event dictionary with only non-null values
        """
        cleaned = {}

        # Stable ICU-local ID priority: icu_event_index > event_index > event_id/event_idx.
        stable_event_id: Optional[int] = None
        for key in ("icu_event_index", "event_index", "event_id", "event_idx"):
            raw_id = event.get(key)
            if not pd.notna(raw_id):
                continue
            try:
                stable_event_id = int(raw_id)
                break
            except (TypeError, ValueError):
                continue

        source_event_index: Optional[int] = None
        for key in ("source_event_index", "event_id", "event_idx", "event_index"):
            raw_source = event.get(key)
            if not pd.notna(raw_source):
                continue
            try:
                source_event_index = int(raw_source)
                break
            except (TypeError, ValueError):
                continue

        if source_event_index is not None:
            cleaned["source_event_index"] = source_event_index

        if stable_event_id is not None:
            cleaned["event_id"] = stable_event_id
            cleaned["event_index"] = stable_event_id
            cleaned["icu_event_index"] = stable_event_id

        if "time" in event and pd.notna(event["time"]):
            # Format time as YYYY-MM-DD HH:MM:SS to preserve second-level precision.
            try:
                time_dt = pd.to_datetime(event["time"])
                cleaned["time"] = time_dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                cleaned["time"] = str(event["time"])

        if "end" in event and pd.notna(event["end"]):
            # Format end time as YYYY-MM-DD HH:MM:SS to preserve second-level precision.
            try:
                end_dt = pd.to_datetime(event["end"])
                cleaned["end_time"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                cleaned["end_time"] = str(event["end"])

        if "code" in event and pd.notna(event["code"]):
            cleaned["code"] = str(event["code"])

        # Numeric value
        if "numeric_value" in event and pd.notna(event["numeric_value"]):
            cleaned["numeric_value"] = float(event["numeric_value"])

        if "code_specifics" in event and pd.notna(event["code_specifics"]):
            cleaned["code_specifics"] = str(event["code_specifics"])

        if "text_value" in event and pd.notna(event["text_value"]):
            cleaned["text_value"] = str(event["text_value"])

        return cleaned

    def _clean_events_list(self, events: List[Dict]) -> List[Dict]:
        """
        Clean a list of events, calculating time deltas between consecutive events.

        Args:
            events: List of raw event dictionaries

        Returns:
            List of cleaned event dictionaries
        """
        cleaned_events = []
        prev_time = None

        for event in events:
            cleaned = self._clean_event(event, prev_time)
            if cleaned:  # Only add if there's any data
                cleaned_events.append(cleaned)
                # Update prev_time for next iteration
                if "time" in cleaned:
                    try:
                        prev_time = pd.to_datetime(cleaned["time"])
                    except:
                        pass

        return cleaned_events

    def filter_events(
        self,
        events_df: pd.DataFrame,
        remove_discharge_summary: bool = True,
    ) -> pd.DataFrame:
        """
        Filter events based on specified criteria.

        This method provides a centralized place for event filtering logic.
        Note: Outcome-revealing events (META_DEATH, LEAVE_*, etc.) are handled
        by stopping window creation at their first occurrence, not by filtering.

        Args:
            events_df: DataFrame containing patient events
            remove_discharge_summary: Remove NOTE_DISCHARGESUMMARY events (default True)

        Returns:
            Filtered DataFrame with events that meet the criteria
        """
        filtered_df = events_df.copy()
        initial_count = len(filtered_df)

        # Filter: Remove NOTE_DISCHARGESUMMARY events if requested
        # These are handled separately via use_discharge_summary_for_history parameter
        if remove_discharge_summary and "code" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["code"] != "NOTE_DISCHARGESUMMARY"].copy()
            removed_count = initial_count - len(filtered_df)
            if removed_count > 0:
                print(f"  Removed {removed_count} NOTE_DISCHARGESUMMARY event(s)")

        return filtered_df

    @staticmethod
    def _normalize_vital_label(label: Optional[str]) -> str:
        """Normalize a code_specifics label for robust matching."""
        if label is None:
            return ""
        return re.sub(r"\s+", " ", str(label).strip().lower())

    def _classify_vital_label(self, label: Optional[str]) -> Optional[str]:
        """
        Map a raw vital code_specifics label to a canonical vital name.

        Returns:
            Canonical vital name, or None if label is not a relevant vital sign.
        """
        normalized = self._normalize_vital_label(label)
        if not normalized:
            return None

        for pattern in IRRELEVANT_VITAL_PATTERNS:
            if re.search(pattern, normalized):
                return None

        for canonical_name, patterns in RELEVANT_VITAL_PATTERNS.items():
            if any(re.search(pattern, normalized) for pattern in patterns):
                return canonical_name

        return None

    def discover_relevant_vital_code_specifics(self, min_occurrences: int = 20) -> Dict[str, List[str]]:
        """
        Discover relevant vital code_specifics values from the loaded events dataframe.

        Discovery rules:
        - Restrict to code == "VITALS"
        - Restrict to rows with numeric_value (measurement rows, not text-only annotations)
        - Map labels to canonical vital names using regex rules
        - Keep labels appearing at least min_occurrences times

        Args:
            min_occurrences: Minimum count threshold to keep a label (default 20)

        Returns:
            Dict mapping canonical vital name -> list of matching code_specifics labels
            ordered by descending frequency.
        """
        if self.events_df is None:
            raise ValueError("events_df is empty. Call load_data() before discovering vital codes.")

        if min_occurrences < 1:
            raise ValueError("min_occurrences must be >= 1")

        if min_occurrences in self._relevant_vitals_cache:
            cached = self._relevant_vitals_cache[min_occurrences]
            return {key: value[:] for key, value in cached.items()}

        vitals_df = self.events_df[self.events_df["code"] == VITAL_CODE]
        if len(vitals_df) == 0:
            self._relevant_vitals_cache[min_occurrences] = {}
            return {}

        if "code_specifics" not in vitals_df.columns:
            self._relevant_vitals_cache[min_occurrences] = {}
            return {}

        # Keep only rows with numeric measurement values.
        if "numeric_value" in vitals_df.columns:
            vitals_df = vitals_df[pd.notna(vitals_df["numeric_value"])]

        label_counts = vitals_df["code_specifics"].dropna().astype(str).value_counts()

        discovered: Dict[str, List[str]] = {name: [] for name in RELEVANT_VITAL_PATTERNS}
        for raw_label, count in label_counts.items():
            if count < min_occurrences:
                continue

            canonical_name = self._classify_vital_label(raw_label)
            if canonical_name is None:
                continue

            discovered[canonical_name].append(raw_label)

        discovered = {key: value for key, value in discovered.items() if value}
        self._relevant_vitals_cache[min_occurrences] = discovered
        return {key: value[:] for key, value in discovered.items()}

    def extract_vitals_snapshot(
        self,
        trajectory: Dict,
        first_n_hours_after_icu: float = 12.0,
        min_occurrences_for_codes: int = 20,
        include_empty_vitals: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Extract a patient vital-sign snapshot for the early ICU window.

        Args:
            trajectory: Patient trajectory from get_patient_trajectory()
            first_n_hours_after_icu: Time horizon from ICU enter_time to include (default 12h)
            min_occurrences_for_codes: Frequency threshold used for dataset code discovery
            include_empty_vitals: If True, include relevant vitals with empty lists

        Returns:
            Dict like:
            {
                "heart rate": [88.0, 92.0, ...],
                "systolic blood pressure": [101.0, 97.0, ...],
                ...
            }
        """
        if first_n_hours_after_icu <= 0:
            raise ValueError("first_n_hours_after_icu must be > 0")

        enter_time = pd.to_datetime(trajectory["enter_time"])
        leave_time = pd.to_datetime(trajectory["leave_time"])
        end_time = min(enter_time + timedelta(hours=first_n_hours_after_icu), leave_time)

        events = trajectory.get("events", [])
        if not events:
            return {name: [] for name in RELEVANT_VITAL_PATTERNS} if include_empty_vitals else {}

        events_df = pd.DataFrame(events)
        required_columns = {"time", "code", "code_specifics", "numeric_value"}
        if not required_columns.issubset(set(events_df.columns)):
            return {name: [] for name in RELEVANT_VITAL_PATTERNS} if include_empty_vitals else {}

        events_df["event_time"] = pd.to_datetime(events_df["time"], errors="coerce")
        events_df["numeric_value"] = pd.to_numeric(events_df["numeric_value"], errors="coerce")

        events_df = events_df[
            (events_df["code"] == VITAL_CODE)
            & (events_df["event_time"] >= enter_time)
            & (events_df["event_time"] < end_time)
            & pd.notna(events_df["numeric_value"])
            & pd.notna(events_df["code_specifics"])
        ].copy()

        if len(events_df) == 0:
            return {name: [] for name in RELEVANT_VITAL_PATTERNS} if include_empty_vitals else {}

        # Dataset-grounded whitelist of labels; if events_df is unavailable, fallback to regex-only classification.
        allowed_labels = set()
        try:
            discovered = self.discover_relevant_vital_code_specifics(min_occurrences=min_occurrences_for_codes)
            for labels in discovered.values():
                allowed_labels.update(labels)
        except ValueError:
            pass

        snapshot: Dict[str, List[float]] = {}
        if include_empty_vitals:
            for vital_name in RELEVANT_VITAL_PATTERNS:
                snapshot[vital_name] = []

        for _, row in events_df.sort_values("event_time").iterrows():
            label = str(row["code_specifics"])
            canonical_name = self._classify_vital_label(label)
            if canonical_name is None:
                continue

            # Keep labels discovered in this dataset; allow MAP alias for external trajectories.
            if allowed_labels and label not in allowed_labels and self._normalize_vital_label(label) != "map":
                continue

            snapshot.setdefault(canonical_name, []).append(float(row["numeric_value"]))

        if include_empty_vitals:
            return {name: snapshot.get(name, []) for name in RELEVANT_VITAL_PATTERNS}

        return {name: values for name, values in snapshot.items() if values}

    def _extract_pre_icu_report_candidates(
        self,
        trajectory: Dict,
        relative_report_codes: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Compatibility shim delegating to pre-ICU processor."""
        return self.pre_icu_history_processor.extract_report_candidates(
            trajectory,
            relative_report_codes=relative_report_codes,
        )

    @staticmethod
    def _format_pre_icu_reports_content(reports: List[Dict]) -> str:
        """Compatibility shim delegating to pre-ICU processor."""
        return PreICUHistoryProcessor.format_reports_content(reports)

    @staticmethod
    def _select_pre_icu_reports_with_per_code_cap(report_candidates: List[Dict], per_code_cap: int) -> List[Dict]:
        """Compatibility shim delegating to pre-ICU processor."""
        return PreICUHistoryProcessor.select_reports_with_per_code_cap(
            report_candidates=report_candidates,
            per_code_cap=per_code_cap,
        )

    def _extract_pre_icu_fallback_events(
        self,
        trajectory: Dict,
        lookback_hours: float = 72.0,
    ) -> List[Dict]:
        """Compatibility shim delegating to pre-ICU processor."""
        return self.pre_icu_history_processor.extract_fallback_events(
            trajectory,
            lookback_hours=lookback_hours,
        )

    @staticmethod
    def _build_window_payload(
        *,
        trajectory: Dict,
        current_start: pd.Timestamp,
        current_end: pd.Timestamp,
        enter_time: pd.Timestamp,
        current_window_hours: float,
        current_discharge_summary: Optional[Dict[str, Any]],
        cleaned_history: List[Dict[str, Any]],
        cleaned_current: List[Dict[str, Any]],
        cleaned_future: List[Dict[str, Any]],
        pre_icu_history_source: str,
        pre_icu_history_items: int,
        historical_discharge_summary_items: int,
        pre_icu_history_content: Optional[str],
        pre_icu_history_hours_applied: float,
    ) -> Dict[str, Any]:
        return {
            "subject_id": trajectory["subject_id"],
            "icu_stay_id": trajectory["icu_stay_id"],
            "current_window_start": current_start.isoformat(),
            "current_window_end": current_end.isoformat(),
            "hours_since_admission": (current_start - enter_time).total_seconds() / 3600,
            "current_window_hours": current_window_hours,
            "patient_metadata": {
                "age": trajectory["age_at_admission"],
                "gender": trajectory.get("gender"),
                "survived": trajectory["survived"],
                "death_time": trajectory["death_time"],
                "total_icu_duration_hours": trajectory["icu_duration_hours"],
            },
            "current_discharge_summary": (dict(current_discharge_summary) if current_discharge_summary else None),
            "history_events": cleaned_history,
            "current_events": cleaned_current,
            "future_events": cleaned_future,
            "num_history_events": len(cleaned_history),
            "num_current_events": len(cleaned_current),
            "num_future_events": len(cleaned_future),
            "pre_icu_history": {
                "source": pre_icu_history_source,
                "items": pre_icu_history_items,
                "historical_discharge_summary_items": historical_discharge_summary_items,
                "content": pre_icu_history_content,
                "history_hours": float(pre_icu_history_hours_applied),
            },
            "pre_icu_history_source": pre_icu_history_source,
            "pre_icu_history_items": pre_icu_history_items,
        }

    def _create_windows_from_events_df(
        self,
        *,
        events_df: pd.DataFrame,
        trajectory: Dict[str, Any],
        enter_time: pd.Timestamp,
        effective_leave_time: pd.Timestamp,
        current_window_hours: float,
        window_step_hours: float,
        include_pre_icu_data: bool,
        use_discharge_summary_for_history: bool,
        pre_icu_history_source: str,
        pre_icu_history_items: int,
        historical_discharge_summary_items: int,
        pre_icu_history_content: Optional[str],
        pre_icu_history_events: List[Dict],
        pre_icu_history_hours_applied: float,
        current_discharge_summary: Optional[Dict[str, Any]],
        history_context_hours: Optional[float],
        future_context_hours: Optional[float],
    ) -> List[Dict]:
        if len(events_df) == 0:
            return []

        event_time_index = pd.DatetimeIndex(pd.to_datetime(events_df["event_time"], errors="coerce"))
        events_records = events_df.to_dict("records")
        cleaned_events_records: Optional[List[Dict[str, Any]]] = None

        use_report_history_mode = bool(include_pre_icu_data and use_discharge_summary_for_history)
        clean_event_cache: Optional[List[Any]] = None
        cache_sentinel = object()
        if use_report_history_mode:
            clean_event_cache = [cache_sentinel] * len(events_records)
        else:
            # Non-report-history mode can reuse prefix slices for both history/current events.
            cleaned_events_records = self._clean_events_list(events_records)

        windows = []
        current_start = enter_time
        while current_start < effective_leave_time:
            current_end = min(current_start + timedelta(hours=current_window_hours), effective_leave_time)

            start_idx = int(event_time_index.searchsorted(current_start, side="left"))
            end_idx = int(event_time_index.searchsorted(current_end, side="left"))
            history_start_idx = 0
            if history_context_hours is not None:
                history_start_time = current_start - timedelta(hours=history_context_hours)
                if use_report_history_mode:
                    history_start_time = max(history_start_time, enter_time)
                history_start_idx = int(event_time_index.searchsorted(history_start_time, side="left"))
            elif use_report_history_mode:
                # Keep window history local to current ICU trajectory when report-history mode is enabled.
                history_start_idx = int(event_time_index.searchsorted(enter_time, side="left"))

            future_end_idx = len(events_records)
            if future_context_hours is not None:
                if future_context_hours == 0:
                    future_end_idx = end_idx
                else:
                    future_end_time = current_end + timedelta(hours=future_context_hours)
                    future_end_idx = int(event_time_index.searchsorted(future_end_time, side="right"))

            if end_idx > start_idx:
                if use_report_history_mode and clean_event_cache is not None:
                    cleaned_current: List[Dict[str, Any]] = []
                    for idx in range(start_idx, end_idx):
                        cached = clean_event_cache[idx]
                        if cached is cache_sentinel:
                            cleaned = self._clean_event(events_records[idx], prev_event_time=None)
                            clean_event_cache[idx] = cleaned if cleaned else None
                            cached = clean_event_cache[idx]
                        if isinstance(cached, dict) and cached:
                            cleaned_current.append(cached)
                else:
                    cleaned_current = cleaned_events_records[start_idx:end_idx] if cleaned_events_records else []

                if use_report_history_mode:
                    cleaned_history: List[Dict[str, Any]] = []
                    if clean_event_cache is not None:
                        for idx in range(history_start_idx, start_idx):
                            cached = clean_event_cache[idx]
                            if cached is cache_sentinel:
                                cleaned = self._clean_event(events_records[idx], prev_event_time=None)
                                clean_event_cache[idx] = cleaned if cleaned else None
                                cached = clean_event_cache[idx]
                            if isinstance(cached, dict) and cached:
                                cleaned_history.append(cached)
                else:
                    cleaned_history = (
                        cleaned_events_records[history_start_idx:start_idx] if cleaned_events_records else []
                    )

                if use_report_history_mode and clean_event_cache is not None:
                    cleaned_future: List[Dict[str, Any]] = []
                    for idx in range(end_idx, future_end_idx):
                        cached = clean_event_cache[idx]
                        if cached is cache_sentinel:
                            cleaned = self._clean_event(events_records[idx], prev_event_time=None)
                            clean_event_cache[idx] = cleaned if cleaned else None
                            cached = clean_event_cache[idx]
                        if isinstance(cached, dict) and cached:
                            cleaned_future.append(cached)
                else:
                    cleaned_future = cleaned_events_records[end_idx:future_end_idx] if cleaned_events_records else []

                windows.append(
                    self._build_window_payload(
                        trajectory=trajectory,
                        current_start=current_start,
                        current_end=current_end,
                        enter_time=enter_time,
                        current_window_hours=current_window_hours,
                        current_discharge_summary=current_discharge_summary,
                        cleaned_history=cleaned_history,
                        cleaned_current=cleaned_current,
                        cleaned_future=cleaned_future,
                        pre_icu_history_source=pre_icu_history_source,
                        pre_icu_history_items=pre_icu_history_items,
                        historical_discharge_summary_items=historical_discharge_summary_items,
                        pre_icu_history_content=pre_icu_history_content,
                        pre_icu_history_hours_applied=pre_icu_history_hours_applied,
                    )
                )

            current_start += timedelta(hours=window_step_hours)

        return windows

    def create_time_windows(
        self,
        trajectory: Dict,
        current_window_hours: float = 0.5,
        window_step_hours: float = 0.5,
        include_pre_icu_data: bool = True,
        use_first_n_hours_after_icu: Optional[float] = None,
        use_discharge_summary_for_history: bool = False,
        num_discharge_summaries: int = 2,
        relative_report_codes: Optional[List[str]] = None,
        pre_icu_history_hours: float = 72.0,
        history_context_hours: Optional[float] = None,
        future_context_hours: Optional[float] = None,
    ) -> List[Dict]:
        """
        Split a patient trajectory into time windows for Oracle or Agent evaluation.

        Each window contains:
        - History: Local events before current window start
          If history_context_hours is set, history is limited to
          [current_window_start - history_context_hours, current_window_start).
          When use_discharge_summary_for_history is True, history remains local per-window
          and does not get replaced by the shared pre-ICU history block.
        - Current: Events from current_start to (current_start + current_window_hours)
        - Future: Events after current window end within this ICU trajectory
          If future_context_hours is set, future is limited to
          [current_window_end, current_window_end + future_context_hours].
        - Pre-ICU shared context: Optional `pre_icu_history` text block (from events/reports)
          attached identically to every window for optional LLM compression/reuse.
        - Metadata: Patient info and outcome

        IMPORTANT: To prevent information leakage, window creation automatically stops at the first
        occurrence of outcome-revealing events (LEAVE_HOSPITALIZATION, NOTE_DISCHARGESUMMARY,
        META_DEATH, LEAVE_ED) within the ICU stay. This ensures predictions are made before
        the outcome is revealed.

        Args:
            trajectory: Patient trajectory from get_patient_trajectory()
            current_window_hours: Size of current observation window (default 0.5 = 30 minutes)
            window_step_hours: Step size between sliding windows (default 0.5 = 30 minutes)
            include_pre_icu_data: Whether to include pre-ICU hospital data (default True)
            use_first_n_hours_after_icu: Use only the first N hours after ICU entry (default None)
                                         When set, only creates windows within the first N hours after ICU admission
                                         Example: use_first_n_hours_after_icu=12 uses only first 12 hours
                                         When None, uses the full ICU duration (or until outcome event)
            use_discharge_summary_for_history: Use pre-ICU report content as history context
                                               to build a shared `pre_icu_history` block
                                               (default False). This shared block is identical
                                               across windows and separate from per-window
                                               `history_events`. Current ICU-stay-matched discharge
                                               summary is always exposed separately in
                                               `current_discharge_summary`.
            num_discharge_summaries: Maximum number of prioritized pre-ICU reports to
                                     include per NOTE_* code (default 2). With
                                     NOTE_DISCHARGESUMMARY only, up to 2 reports can
                                     be selected in total by default.
            relative_report_codes: Optional additional NOTE_* codes to consider as pre-ICU
                                   reports. NOTE_DISCHARGESUMMARY is always prioritized first.
            pre_icu_history_hours: Pre-ICU lookback window in hours (default 72h) used for
                                   pre-ICU history events.
            history_context_hours: Optional history horizon (hours) before current window
                                   start for window history_events. None keeps full history.
            future_context_hours: Optional future horizon (hours) after current window end
                                  for window future_events. None keeps full future.

        Example with current=0.5, step=0.5:
            Window 0: history=[ICU_start, 0h], current=[0h, 0.5h]
            Window 1: history=[ICU_start, 0.5h], current=[0.5h, 1h]
            Window 2: history=[ICU_start, 1h], current=[1h, 1.5h]
            ...

        Returns:
            List of time windows, each suitable for Oracle or Agent evaluation
        """
        history_context_hours_applied: Optional[float] = None
        if history_context_hours is not None:
            history_context_hours_applied = float(history_context_hours)
            if history_context_hours_applied < 0:
                raise ValueError(f"history_context_hours must be >= 0 or None, got {history_context_hours_applied}")

        future_context_hours_applied: Optional[float] = None
        if future_context_hours is not None:
            future_context_hours_applied = float(future_context_hours)
            if future_context_hours_applied < 0:
                raise ValueError(f"future_context_hours must be >= 0 or None, got {future_context_hours_applied}")

        enter_time = pd.to_datetime(trajectory["enter_time"])
        leave_time = pd.to_datetime(trajectory["leave_time"])

        # Determine effective leave time based on use_first_n_hours_after_icu parameter
        # If use_first_n_hours_after_icu is set, only use the first N hours after ICU entry
        # Otherwise, use the full ICU duration
        if use_first_n_hours_after_icu is not None:
            effective_leave_time = enter_time + timedelta(hours=use_first_n_hours_after_icu)
            # Make sure we don't exceed the actual leave time
            effective_leave_time = min(effective_leave_time, leave_time)
        else:
            effective_leave_time = leave_time

        # Find first occurrence of outcome-revealing events and stop there
        # This prevents information leakage by stopping before outcome is revealed.
        #
        # Additional guardrail:
        # - If the timeline is truly truncated by outcome reveal and the usable ICU time is <4h,
        #   skip this patient by returning no windows.
        outcome_codes = ["LEAVE_HOSPITALIZATION", "NOTE_DISCHARGESUMMARY", "META_DEATH", "LEAVE_ICU"]
        events_df_temp = pd.DataFrame(trajectory["events"])
        outcome_truncated = False
        first_outcome_time = None

        if len(events_df_temp) > 0 and "code" in events_df_temp.columns and "time" in events_df_temp.columns:
            # Filter for outcome-revealing events within current ICU stay
            events_df_temp["event_time"] = pd.to_datetime(events_df_temp["time"])

            # Basic outcome codes
            outcome_events = events_df_temp[
                (events_df_temp["code"].isin(outcome_codes))
                & (events_df_temp["event_time"] >= enter_time)
                & (events_df_temp["event_time"] <= leave_time)
            ]

            # Add LEAVE_ED events that reveal outcome (exclude "ADMITTED" which is just transition to ICU)
            leave_ed_outcome = events_df_temp[
                (events_df_temp["code"] == "LEAVE_ED")
                & (events_df_temp["event_time"] >= enter_time)
                & (events_df_temp["event_time"] <= leave_time)
                & (events_df_temp["code_specifics"] != "ADMITTED")  # Exclude admission transitions
            ]

            # Combine outcome events
            outcome_events = pd.concat([outcome_events, leave_ed_outcome], ignore_index=True)

            if len(outcome_events) > 0:
                # Find the earliest outcome event
                first_outcome_time = outcome_events["event_time"].min()
                # Stop windows just before the outcome event (subtract 1 second to ensure we don't include it)
                stop_time = first_outcome_time - timedelta(seconds=1)
                if stop_time < effective_leave_time:
                    effective_leave_time = stop_time
                    outcome_truncated = True
                    print(f"  Found outcome event at {first_outcome_time}, stopping windows at {stop_time}")

        if outcome_truncated:
            truncated_icu_hours = (effective_leave_time - enter_time).total_seconds() / 3600.0
            if truncated_icu_hours < 4.0:
                print(
                    "  Outcome-truncated ICU duration " f"{truncated_icu_hours:.2f}h < 4.00h, skipping patient windows"
                )
                return []

        # Build pre-ICU history block once, then attach to every window.
        pre_icu_history_source = "disabled"
        pre_icu_history_items = 0
        pre_icu_history_content = None
        pre_icu_history_events: List[Dict] = []
        historical_discharge_summary_items = 0
        pre_icu_history_hours_applied = float(pre_icu_history_hours)
        selected_discharge_summary = self._selected_discharge_summary_for_stay(
            subject_id=int(trajectory["subject_id"]),
            icu_stay_id=int(trajectory["icu_stay_id"]),
        )
        current_discharge_summary = None
        if isinstance(selected_discharge_summary, dict):
            selected_time = pd.to_datetime(selected_discharge_summary.get("time"), errors="coerce")
            hours_since_icu_admission = None
            hours_after_icu_leave = None
            if pd.notna(selected_time):
                if pd.notna(enter_time):
                    hours_since_icu_admission = float((selected_time - enter_time).total_seconds() / 3600.0)
                if pd.notna(leave_time):
                    hours_after_icu_leave = float((selected_time - leave_time).total_seconds() / 3600.0)

            current_discharge_summary = {
                "time": selected_discharge_summary.get("time"),
                "text_value": selected_discharge_summary.get("text_value"),
                "code_specifics": selected_discharge_summary.get("code_specifics"),
                "selection_rule": selected_discharge_summary.get("selection_rule"),
                "delta_hours_after_leave": selected_discharge_summary.get("delta_hours_after_leave"),
                "hours_since_icu_admission": hours_since_icu_admission,
                "hours_after_icu_leave": hours_after_icu_leave,
            }

        if include_pre_icu_data and use_discharge_summary_for_history:
            pre_icu_context = self.pre_icu_history_processor.build_history_context(
                trajectory=trajectory,
                num_discharge_summaries=num_discharge_summaries,
                relative_report_codes=relative_report_codes,
                pre_icu_history_hours=pre_icu_history_hours,
            )
            pre_icu_history_source = str(pre_icu_context.get("source") or "none")
            pre_icu_history_items = int(pre_icu_context.get("items") or 0)
            historical_discharge_summary_items = int(pre_icu_context.get("historical_discharge_summary_items") or 0)
            pre_icu_history_content = pre_icu_context.get("content")
            pre_icu_history_events = pre_icu_context.get("history_events") or []
            per_code_cap = int(pre_icu_context.get("per_code_cap") or 0)
            pre_icu_history_hours_applied = float(
                pre_icu_context.get("pre_icu_history_hours") or pre_icu_history_hours
            )
            report_items = int(pre_icu_context.get("report_items") or 0)
            event_items = int(pre_icu_context.get("event_items") or 0)

            if pre_icu_history_content:
                print(
                    "  Built unified pre-ICU history context "
                    f"({event_items} event(s) from previous {pre_icu_history_hours_applied:.1f}h, "
                    f"{report_items} report(s), max {per_code_cap} per code, "
                    f"{len(pre_icu_history_content)} characters, "
                    f"historical discharge summaries: {historical_discharge_summary_items})"
                )
            else:
                print("  No pre-ICU events or prioritized reports found before current ICU stay")

        # Convert events to DataFrame for easier manipulation
        events_df = pd.DataFrame(trajectory["events"])

        if len(events_df) == 0:
            return []

        # Ensure events have timestamps
        if "time" in events_df.columns:
            events_df["event_time"] = pd.to_datetime(events_df["time"])
        else:
            # If no timestamp, distribute events evenly across the ICU stay
            total_duration = (leave_time - enter_time).total_seconds()
            events_df["event_time"] = [
                enter_time + timedelta(seconds=total_duration * i / len(events_df)) for i in range(len(events_df))
            ]

        # Filter events based on include_pre_icu_data setting
        if include_pre_icu_data:
            # Keep all events up to leave_time (including pre-ICU hospital data)
            events_df = events_df[events_df["event_time"] <= leave_time].copy()
        else:
            # Only keep events during ICU stay (discard pre-ICU hospital history)
            events_df = events_df[
                (events_df["event_time"] >= enter_time) & (events_df["event_time"] <= leave_time)
            ].copy()

        # Keep deterministic time order for downstream ICU-local indexing.
        events_df = events_df.sort_values("event_time", ascending=True, kind="mergesort").reset_index(drop=True)
        if "event_index" not in events_df.columns:
            events_df["event_index"] = range(len(events_df))
        else:
            parsed_index = pd.to_numeric(events_df["event_index"], errors="coerce")
            if parsed_index.isna().any():
                parsed_index = pd.Series(range(len(events_df)), index=events_df.index, dtype="int64")
            events_df["event_index"] = parsed_index.astype("int64")

        # ICU-local IDs always restart at 0 for events within [enter_time, leave_time].
        events_df["icu_event_index"] = pd.Series(pd.NA, index=events_df.index, dtype="Int64")
        icu_event_mask = (events_df["event_time"] >= enter_time) & (events_df["event_time"] <= leave_time)
        if bool(icu_event_mask.any()):
            events_df.loc[icu_event_mask, "icu_event_index"] = range(int(icu_event_mask.sum()))

        return self._create_windows_from_events_df(
            events_df=events_df,
            trajectory=trajectory,
            enter_time=enter_time,
            effective_leave_time=effective_leave_time,
            current_window_hours=current_window_hours,
            window_step_hours=window_step_hours,
            include_pre_icu_data=include_pre_icu_data,
            use_discharge_summary_for_history=use_discharge_summary_for_history,
            pre_icu_history_source=pre_icu_history_source,
            pre_icu_history_items=pre_icu_history_items,
            historical_discharge_summary_items=historical_discharge_summary_items,
            pre_icu_history_content=pre_icu_history_content,
            pre_icu_history_events=pre_icu_history_events,
            pre_icu_history_hours_applied=pre_icu_history_hours_applied,
            current_discharge_summary=current_discharge_summary,
            history_context_hours=history_context_hours_applied,
            future_context_hours=future_context_hours_applied,
        )

    def extract_discharge_summary(self, trajectory: Dict, k: int = 1) -> Optional[List[Dict]]:
        """
        Extract discharge summary(ies) for the current ICU stay.

        Primary path:
        - Use selector-linked discharge summary for this ICU stay (rule1/rule2).
        Fallback path (when selector map is unavailable):
        - Use most recent pre-ICU NOTE_DISCHARGESUMMARY events.

        Args:
            trajectory: Patient trajectory from get_patient_trajectory()
            k: Number of most recent discharge summaries to extract (default 1)

        Returns:
            List of discharge summary dictionaries sorted by time (most recent first),
            or None if no discharge summaries found
        """
        if k <= 0:
            return None

        subject_id = int(trajectory["subject_id"])
        icu_stay_id = int(trajectory["icu_stay_id"])
        selected = self._selected_discharge_summary_for_stay(subject_id, icu_stay_id)
        if selected is not None:
            selected_time = pd.to_datetime(selected.get("time"), errors="coerce")
            current_enter_time = pd.to_datetime(trajectory.get("enter_time"), errors="coerce")
            hours_before_current_icu = None
            if pd.notna(selected_time) and pd.notna(current_enter_time):
                hours_before_current_icu = (current_enter_time - selected_time).total_seconds() / 3600.0

            return [
                {
                    "subject_id": subject_id,
                    "time": selected.get("time"),
                    "text_value": selected.get("text_value"),
                    "code_specifics": selected.get("code_specifics"),
                    "hours_before_current_icu": hours_before_current_icu,
                    "selection_rule": selected.get("selection_rule"),
                }
            ][:k]

        report_candidates = self.pre_icu_history_processor.extract_report_candidates(
            trajectory,
            relative_report_codes=[],
        )
        discharge_only = [item for item in report_candidates if item.get("code") == "NOTE_DISCHARGESUMMARY"]
        if not discharge_only:
            return None

        results = []
        for item in discharge_only[:k]:
            results.append(
                {
                    "subject_id": item.get("subject_id"),
                    "time": item.get("time"),
                    "text_value": item.get("text_value"),
                    "code_specifics": item.get("code_specifics"),
                    "hours_before_current_icu": item.get("hours_before_current_icu"),
                }
            )
        return results

    def _limit_icu_stays(self, max_patients: Optional[int] = None) -> pd.DataFrame:
        """Return ICU stay table after applying an optional max-patient limit."""
        if self.icu_stay_df is None:
            raise ValueError("icu_stay_df is empty. Call load_data() first.")

        if max_patients is None:
            return self.icu_stay_df
        if max_patients < 0:
            raise ValueError("max_patients must be >= 0")
        return self.icu_stay_df.head(max_patients)

    def iter_trajectories(self, max_patients: Optional[int] = None) -> Iterator[Dict]:
        """
        Stream patient trajectories one-by-one.

        Args:
            max_patients: Optional number of ICU stays to process from the top.
        """
        limited_stays = self._limit_icu_stays(max_patients=max_patients)

        for _, icu_stay in limited_stays.iterrows():
            subject_id = int(icu_stay["subject_id"])
            icu_stay_id = int(icu_stay["icu_stay_id"])
            try:
                yield self.get_patient_trajectory(
                    subject_id=subject_id,
                    icu_stay_id=icu_stay_id,
                    icu_stay=icu_stay,
                )
            except Exception as e:
                print(f"Error processing ICU stay {icu_stay_id}: {e}")
                continue

    def get_all_trajectories(self, max_patients: Optional[int] = None) -> List[Dict]:
        """
        Get all patient trajectories from the dataset.

        Args:
            max_patients: Optional number of ICU stays to process from the top.

        Returns:
            List of all patient trajectories
        """
        return list(self.iter_trajectories(max_patients=max_patients))

    @staticmethod
    def _json_default(value):
        """
        Convert pandas/datetime scalar values to JSON-serializable forms.
        """
        if value is pd.NaT or value is pd.NA:
            return None

        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()

        if isinstance(value, timedelta):
            return value.total_seconds()

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        return str(value)

    def save_trajectories(self, trajectories: List[Dict], output_path: str):
        """
        Save trajectories to a JSONL file.

        Args:
            trajectories: List of patient trajectories
            output_path: Path to output JSONL file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for trajectory in trajectories:
                f.write(json.dumps(trajectory, ensure_ascii=False, default=self._json_default) + "\n")
        print(f"Saved {len(trajectories)} trajectories to {output_path}")

    def load_trajectories(self, input_path: str) -> List[Dict]:
        """
        Load trajectories from a JSONL file.

        Args:
            input_path: Path to input JSONL file

        Returns:
            List of patient trajectories
        """
        trajectories = []
        with open(input_path, "r") as f:
            for line in f:
                trajectories.append(json.loads(line))
        print(f"Loaded {len(trajectories)} trajectories from {input_path}")
        return trajectories
