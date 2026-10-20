"""Action matching without experiment or agent dependencies."""

import json

import numpy as np

from model.embeddings import TextEncoder
from model.llms import LLMClient
from prompts.action_matcher_prompts import get_action_matcher_prompt
from utils.json_parse import parse_json_dict_or_raise


def action_text(action: dict) -> str:
    name = action["action_name"].strip()
    description = str(action.get("action_description") or action.get("reason") or action.get("rationale") or "").strip()
    if name and description and name.lower() not in description.lower():
        return f"{name}. {description}"
    return description or name


def matching_prompt(predictions: list[dict], actions: list[dict], red_flags: list[dict]) -> str:
    payloads = [[{
        "prediction_index": index,
        "action_name": action["action_name"].strip(),
        "action_description": str(action.get("action_description") or "").strip(),
    } for index, action in enumerate(predictions)]]
    payloads.extend([
        [{"gt_index": index, "action_description": item["gt_action_text"].strip()} for index, item in enumerate(group)]
        for group in (actions, red_flags)
    ])
    prompt = get_action_matcher_prompt()
    for name, payload in zip(("predicted_actions", "reference_recommended_actions", "reference_red_flag_actions"), payloads):
        prompt = prompt.replace("{" + name + "}", json.dumps(payload, ensure_ascii=False, indent=2))
    return prompt


def matched_pairs(section, num_predictions: int, num_references: int) -> list[dict]:
    if not num_references or section is None:
        return []
    if isinstance(section, dict):
        if isinstance(section.get("matched_pairs"), list):
            section = [{"pred_idx": row["prediction_index"], "gt_indices": [row["gt_index"]]}
                       for row in section["matched_pairs"]]
        elif isinstance(section.get("matches"), list):
            section = section["matches"]
        elif isinstance(section.get("matched_prediction_indices"), list):
            indices = list(dict.fromkeys(int(index) for index in section["matched_prediction_indices"]))
            if any(not 0 <= index < num_predictions for index in indices):
                raise ValueError("Action matcher prediction index out of range")
            return [{"prediction_index": index, "gt_index": -1} for index in indices]
        else:
            raise ValueError("Invalid action matcher section")
    if not isinstance(section, list):
        raise ValueError("Action matcher section must contain a list of matches")
    pairs, seen = [], set()
    for row in section:
        index, references = int(row["pred_idx"]), row["gt_indices"]
        if not 0 <= index < num_predictions:
            raise ValueError("Action matcher prediction index out of range")
        if index in seen:
            continue
        seen.add(index)
        if not isinstance(references, list):
            raise ValueError("Action matcher gt_indices must be a list")
        references = [int(reference) for reference in references]
        if any(not 0 <= reference < num_references for reference in references):
            raise ValueError("Action matcher reference index out of range")
        if references:
            pairs.append({"prediction_index": index, "gt_index": references[0]})
    return pairs


class ActionMatcher:
    def __init__(self, settings: dict):
        self.settings = settings
        self.backend = settings["backend"]
        if self.backend == "pubmedbert":
            self.encoder = TextEncoder(**settings["encoder"])
            self.threshold = settings["similarity_threshold"]
            if not isinstance(self.threshold, (int, float)) or not 0 <= self.threshold <= 1:
                raise ValueError("matcher.similarity_threshold must be in [0, 1]")
        elif self.backend == "llm":
            self.client = LLMClient(**settings["llm"])
        else:
            raise ValueError(f"Unsupported action matcher: {self.backend}")

    def match(self, predictions: list[dict], actions: list[dict], red_flags: list[dict]) -> dict:
        result = {"recommended_pairs": [], "red_flag_pairs": [], "usage": {"input_tokens": 0, "output_tokens": 0}}
        if not predictions or (not actions and not red_flags):
            return result
        predicted_texts = [action_text(action) for action in predictions]
        references = [[item["gt_action_text"] for item in group] for group in (actions, red_flags)]
        if self.backend == "pubmedbert":
            embeddings = self.encoder.encode_texts(predicted_texts)
            for key, texts in zip(("recommended_pairs", "red_flag_pairs"), references):
                if not texts:
                    continue
                similarities = embeddings @ self.encoder.encode_texts(texts).T
                for index, row in enumerate(similarities):
                    best = int(np.argmax(row))
                    if row[best] >= self.threshold:
                        result[key].append({"prediction_index": index, "gt_index": best})
            return result
        response = self.client.chat(matching_prompt(predictions, actions, red_flags))
        parsed = parse_json_dict_or_raise(response["content"])
        result["usage"] = response["usage"]
        sections = (parsed.get("recommended_action_matches"), parsed.get("red_flag_matches")) if (
            "recommended_action_matches" in parsed or "red_flag_matches" in parsed
        ) else (parsed, None)
        for key, section, texts in zip(
            ("recommended_pairs", "red_flag_pairs"),
            sections, references,
        ):
            result[key] = matched_pairs(section, len(predictions), len(texts))
        return result
