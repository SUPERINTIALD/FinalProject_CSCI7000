from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SemanticAction:
    action_type: str
    target_object: str | None = None
    target_surface: str | None = None
    target_zone: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    memory_used: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_model_output: str = ""
    valid_json: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionParser:
    """
    Parses model output into a semantic action.

    Supported styles:
    1. New structured schema:
       {
         "action_type": "pick",
         "target_object": "plate_1",
         "target_surface": null,
         "target_zone": null,
         "parameters": {},
         "reason": "...",
         "memory_used": ["..."],
         "confidence": 0.9
       }

    2. Legacy schema:
       {
         "action": "start_cleanup",
         "reason": "...",
         "memory_used": ["..."],
         "confidence": 0.9
       }
    """

    LEGACY_ACTION_MAP = {
        "wait": {"action_type": "wait"},
        "start_cleanup": {"action_type": "start_task"},
        "pick_dish_right_side": {"action_type": "pick", "target_zone": "right_side"},
        "place_clean_left_side": {"action_type": "place", "target_surface": "left_counter"},
    }

    @staticmethod
    def _extract_json_blob(text: str) -> str:
        text = text.strip()
        if not text:
            return "{}"

        if text.startswith("{") and text.endswith("}"):
            return text

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            return text[start:end + 1]

        return "{}"

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _normalize_action_string(action: str) -> str:
        return re.sub(r"[^a-z0-9_ ]+", "", action.lower()).strip().replace(" ", "_")

    @staticmethod
    def _first_matching_object(
        scene_state: dict[str, Any],
        *,
        allowed_states: set[str] | None = None,
        allowed_locations: set[str] | None = None,
    ) -> str | None:
        for obj in scene_state.get("objects", []):
            state = obj.get("state")
            location = obj.get("location")
            if allowed_states is not None and state not in allowed_states:
                continue
            if allowed_locations is not None and location not in allowed_locations:
                continue
            return obj.get("name")
        return None

    @staticmethod
    def _held_object(scene_state: dict[str, Any]) -> str | None:
        return scene_state.get("held_object")

    @classmethod
    def _infer_from_legacy_action(
        cls,
        action: str,
        scene_state: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = cls._normalize_action_string(action)

        if normalized in cls.LEGACY_ACTION_MAP:
            parsed = dict(cls.LEGACY_ACTION_MAP[normalized])
        else:
            parsed = {}

            if "wait" in normalized:
                parsed["action_type"] = "wait"
            elif "start" in normalized:
                parsed["action_type"] = "start_task"
            elif "pick" in normalized or "grab" in normalized:
                parsed["action_type"] = "pick"
            elif "place" in normalized or "put" in normalized:
                parsed["action_type"] = "place"
            elif "clean" in normalized or "wipe" in normalized:
                parsed["action_type"] = "clean_surface"
            elif "inspect" in normalized or "scan" in normalized:
                parsed["action_type"] = "inspect"
            else:
                parsed["action_type"] = "wait"

            if "right" in normalized:
                parsed["target_zone"] = "right_side"
            elif "left" in normalized:
                parsed["target_zone"] = "left_side"

        action_type = parsed.get("action_type")

        if action_type == "pick" and "target_object" not in parsed:
            zone = parsed.get("target_zone")
            if zone is not None:
                parsed["target_object"] = cls._first_matching_object(
                    scene_state,
                    allowed_states={"dirty", "used", "misplaced", "unsorted"},
                    allowed_locations={zone},
                )
            else:
                parsed["target_object"] = cls._first_matching_object(
                    scene_state,
                    allowed_states={"dirty", "used", "misplaced", "unsorted"},
                )

        if action_type == "place" and "target_object" not in parsed:
            parsed["target_object"] = cls._held_object(scene_state)

        if action_type == "place" and "target_surface" not in parsed:
            preference_surface = scene_state.get("placement_preference_surface")
            if preference_surface:
                parsed["target_surface"] = preference_surface

        return parsed

    @classmethod
    def parse(
        cls,
        raw_text: str,
        scene_state: dict[str, Any],
    ) -> SemanticAction:
        json_blob = cls._extract_json_blob(raw_text)

        try:
            parsed = json.loads(json_blob)
            valid_json = True
        except json.JSONDecodeError:
            parsed = {}
            valid_json = False

        if "action_type" in parsed:
            action_type = str(parsed.get("action_type", "wait")).strip().lower()
            action = SemanticAction(
                action_type=action_type or "wait",
                target_object=parsed.get("target_object"),
                target_surface=parsed.get("target_surface"),
                target_zone=parsed.get("target_zone"),
                parameters=parsed.get("parameters", {}) if isinstance(parsed.get("parameters", {}), dict) else {},
                reason=str(parsed.get("reason", "")).strip(),
                memory_used=parsed.get("memory_used", []) if isinstance(parsed.get("memory_used", []), list) else [],
                confidence=cls._safe_float(parsed.get("confidence", 0.0)),
                raw_model_output=raw_text,
                valid_json=valid_json,
            )
            return action

        legacy_action = str(parsed.get("action", "wait")).strip()
        inferred = cls._infer_from_legacy_action(legacy_action, scene_state)

        return SemanticAction(
            action_type=inferred.get("action_type", "wait"),
            target_object=inferred.get("target_object"),
            target_surface=inferred.get("target_surface"),
            target_zone=inferred.get("target_zone"),
            parameters={},
            reason=str(parsed.get("reason", "")).strip(),
            memory_used=parsed.get("memory_used", []) if isinstance(parsed.get("memory_used", []), list) else [],
            confidence=cls._safe_float(parsed.get("confidence", 0.0)),
            raw_model_output=raw_text,
            valid_json=valid_json,
        )