from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.vlm_client import VLMClient


VLM_SYSTEM_PROMPT = """
You are the vision perception module for a household robot.

Your job is NOT to plan robot actions.
Your job is only to inspect the image and convert it into structured scene JSON.

Return valid JSON only.
Do not include markdown fences.
Do not invent objects that are not visible.
Use simple names like plate_1, fork_1, cup_1, wrapper_1.

Allowed object states:
- clean
- dirty
- used
- misplaced
- discarded
- unsorted

Allowed task templates:
- dish_cleanup
- trash_cleanup
- counter_cleanup

Allowed user states:
- still_eating
- finished_eating
- left_area
- using_counter
- using_kitchen
- unknown

Allowed robot states:
- idle
- active
- holding_item

Return this exact schema:
{
  "task_family": "cleanup",
  "template_name": "dish_cleanup_or_trash_cleanup_or_counter_cleanup",
  "scene_family": "image_kitchen_scene",
  "robot_platform": "Franka Panda",
  "user_state": "one_allowed_user_state",
  "robot_state": "idle",
  "objects": [
    {
      "name": "object_name",
      "kind": "plate_or_cup_or_fork_or_wrapper_or_trash_or_other",
      "state": "clean_or_dirty_or_used_or_misplaced_or_discarded_or_unsorted",
      "location": "table_center_or_left_counter_or_right_counter_or_floor_or_near_counter_or_sink_or_unknown"
    }
  ],
  "surfaces": [
    {
      "name": "table_center",
      "state": "available"
    }
  ],
  "held_object": null,
  "placement_preference_surface": "sink",
  "last_action_result": "none"
}
"""


VLM_USER_PROMPT = """
Inspect this kitchen image for a robot assistant.

Identify whether the robot should think about:
1. dish cleanup after eating,
2. trash cleanup,
3. counter cleanup.

Return structured JSON only.

Important:
- If food remains and a person is still actively eating, use user_state = "still_eating".
- If the meal appears finished or the person is gone, use user_state = "finished_eating" or "left_area".
- Dirty/used dishes, utensils, cups, napkins, wrappers, or trash should be listed as objects.
- Clean irrelevant objects should be listed only if important.
- If trash or wrapper is on the floor or near counter, template_name should usually be "trash_cleanup".
- If used cup/fork/plate is left on a counter, template_name should usually be "counter_cleanup".
- If used dishes are on a dining table after eating, template_name should usually be "dish_cleanup".
"""


DEFAULT_SURFACES = {
    "dish_cleanup": [
        {"name": "table_center", "state": "available"},
        {"name": "left_counter", "state": "available"},
        {"name": "right_counter", "state": "available"},
        {"name": "sink", "state": "available"},
        {"name": "drying_rack", "state": "available"},
    ],
    "trash_cleanup": [
        {"name": "floor", "state": "available"},
        {"name": "trash_bin", "state": "available"},
        {"name": "counter", "state": "available"},
        {"name": "sink", "state": "available"},
    ],
    "counter_cleanup": [
        {"name": "left_counter", "state": "available"},
        {"name": "right_counter", "state": "available"},
        {"name": "sink", "state": "available"},
        {"name": "drying_rack", "state": "available"},
    ],
}


ACTIONABLE_STATES = {"dirty", "used", "misplaced", "discarded", "unsorted"}


def _extract_json_blob(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start:end + 1]

    return "{}"


def _safe_load_json(text: str) -> dict[str, Any]:
    blob = _extract_json_blob(text)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        data = {}
    return data if isinstance(data, dict) else {}


def _normalize_template(scene: dict[str, Any]) -> str:
    template_name = str(scene.get("template_name", "")).strip()

    if template_name in {"dish_cleanup", "trash_cleanup", "counter_cleanup"}:
        return template_name

    objects = scene.get("objects", [])
    object_kinds = " ".join(str(obj.get("kind", "")).lower() for obj in objects)
    object_locations = " ".join(str(obj.get("location", "")).lower() for obj in objects)

    if any(word in object_kinds for word in ["wrapper", "trash", "napkin", "paper"]):
        return "trash_cleanup"

    if "floor" in object_locations:
        return "trash_cleanup"

    if any(word in object_locations for word in ["counter", "sink"]):
        return "counter_cleanup"

    return "dish_cleanup"


def _normalize_object_names(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    normalized = []

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        kind = str(obj.get("kind", "object")).strip().lower().replace(" ", "_")
        kind = re.sub(r"[^a-z0-9_]+", "", kind) or "object"

        counts[kind] = counts.get(kind, 0) + 1
        name = str(obj.get("name", "")).strip()

        if not name:
            name = f"{kind}_{counts[kind]}"

        state = str(obj.get("state", "unsorted")).strip().lower()
        if state not in ACTIONABLE_STATES and state != "clean":
            state = "unsorted"

        location = str(obj.get("location", "unknown")).strip().lower().replace(" ", "_")
        location = re.sub(r"[^a-z0-9_]+", "", location) or "unknown"

        normalized.append(
            {
                "name": name,
                "kind": kind,
                "state": state,
                "location": location,
            }
        )

    return normalized


def normalize_scene_state(raw_scene: dict[str, Any]) -> dict[str, Any]:
    template_name = _normalize_template(raw_scene)
    objects = _normalize_object_names(raw_scene.get("objects", []))

    user_state = str(raw_scene.get("user_state", "unknown")).strip()
    if user_state not in {
        "still_eating",
        "finished_eating",
        "left_area",
        "using_counter",
        "using_kitchen",
        "unknown",
    }:
        user_state = "unknown"

    # Unknown should not block proactivity by default.
    if user_state == "unknown":
        user_state = "left_area"

    robot_state = str(raw_scene.get("robot_state", "idle")).strip()
    if robot_state not in {"idle", "active", "holding_item"}:
        robot_state = "idle"

    surfaces = raw_scene.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        surfaces = DEFAULT_SURFACES[template_name]

    if template_name == "trash_cleanup":
        placement_surface = "trash_bin"
    elif template_name == "counter_cleanup":
        placement_surface = "sink"
    else:
        placement_surface = "sink"

    return {
        "task_family": "cleanup",
        "template_name": template_name,
        "scene_family": "image_kitchen_scene",
        "robot_platform": "Franka Panda",
        "user_state": user_state,
        "robot_state": robot_state,
        "objects": objects,
        "surfaces": surfaces,
        "held_object": raw_scene.get("held_object", None),
        "placement_preference_surface": raw_scene.get(
            "placement_preference_surface",
            placement_surface,
        ),
        "last_action_result": raw_scene.get("last_action_result", "none"),
    }


class VLMPerception:
    def __init__(self, vlm_client: VLMClient) -> None:
        self.vlm_client = vlm_client

    def perceive_image(self, image_path: str | Path) -> dict[str, Any]:
        raw_text = self.vlm_client.generate_from_image(
            image_path=image_path,
            system_prompt=VLM_SYSTEM_PROMPT,
            user_prompt=VLM_USER_PROMPT,
            temperature=0.0,
            max_tokens=900,
        )

        raw_scene = _safe_load_json(raw_text)
        scene_state = normalize_scene_state(raw_scene)

        return {
            "scene_state": scene_state,
            "raw_vlm_output": raw_text,
        }