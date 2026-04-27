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
Important output rules:
- Never output option strings such as "dish_cleanup_or_trash_cleanup" or "cup_or_fork_or_plate".
- Choose exactly one value from the allowed list.
- If no cleanup-relevant object is visible, return "objects": [].
- Do not list surfaces as objects. The floor, table, counter, sink, and trash bin are surfaces/regions, not objects.
- Do not create an object named "floor", "table", "counter", "sink", or "trash_bin".
- If the floor is clean, do not create a trash object.
- For cups/glasses:
  - If liquid is visible, use state = "filled".
  - If it is clean and empty, use state = "clean".
  - If it has residue or appears used after drinking, use state = "used".
- A filled cup should not be cleaned up yet unless the scene clearly says the user is done with it.

- If a plate/bowl still has edible food remaining, use state = "filled" and user_state = "still_eating" unless it is clearly abandoned.
- A filled plate, filled bowl, or filled cup should not be treated as cleanup-ready.
- If the meal is mostly finished with only crumbs/residue, use state = "used" or "dirty".



Allowed object states:
- clean
- dirty
- used
- filled
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
SURFACE_WORDS = {
    "floor",
    "table",
    "table_center",
    "counter",
    "left_counter",
    "right_counter",
    "sink",
    "trash_bin",
    "drying_rack",
}

TRASH_KINDS = {"trash", "waste", "wrapper", "paper", "paper_towel", "aluminum", "foil", "napkin"}
DISH_KINDS = {"plate", "bowl", "cup", "glass", "mug", "fork", "spoon", "knife", "chopsticks", "utensil"}

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

    # Reject bad VLM option strings like "dish_cleanup_or_trash_cleanup"
    if template_name in {"dish_cleanup", "trash_cleanup", "counter_cleanup"}:
        return template_name

    objects = _normalize_object_names(scene.get("objects", []))

    has_trash = any(
        obj.get("kind") in TRASH_KINDS and obj.get("state") in ACTIONABLE_STATES
        for obj in objects
    )

    has_counter_item = any(
        obj.get("kind") in DISH_KINDS
        and obj.get("location") in {"left_counter", "right_counter", "counter", "near_counter", "sink"}
        for obj in objects
    )

    has_dish_item = any(
        obj.get("kind") in DISH_KINDS
        and obj.get("location") in {"table_center", "table", "on_plate", "plate", "center"}
        for obj in objects
    )

    if has_trash:
        return "trash_cleanup"
    if has_counter_item:
        return "counter_cleanup"
    if has_dish_item:
        return "dish_cleanup"

    # Default no-object kitchen scene can still be evaluated as cleanup/no-op.
    return "trash_cleanup"
def _canonical_kind(raw_kind: str, raw_name: str) -> str:
    kind = str(raw_kind or "").lower()
    name = str(raw_name or "").lower()
    combined = f"{name} {kind}"

    # Prefer the object name when VLM outputs bad option strings like cup_or_fork_or_trash.
    if any(word in name for word in ["glass", "cup", "mug"]):
        return "cup"
    if "plate" in name:
        return "plate"
    if "bowl" in name:
        return "bowl"
    if "fork" in name:
        return "fork"
    if "spoon" in name:
        return "spoon"
    if "knife" in name:
        return "knife"
    if "chopstick" in name:
        return "chopsticks"
    if "wrapper" in name:
        return "wrapper"
    if "paper" in name:
        return "paper"
    if "aluminum" in name or "foil" in name:
        return "aluminum"
    if "waste" in name or "trash" in name:
        return "waste"

    if any(word in combined for word in ["glass", "cup", "mug"]):
        return "cup"
    if "plate" in combined:
        return "plate"
    if "bowl" in combined:
        return "bowl"
    if "fork" in combined:
        return "fork"
    if "spoon" in combined:
        return "spoon"
    if "knife" in combined:
        return "knife"
    if "chopstick" in combined:
        return "chopsticks"
    if "wrapper" in combined:
        return "wrapper"
    if "paper" in combined:
        return "paper"
    if "aluminum" in combined or "foil" in combined:
        return "aluminum"
    if "waste" in combined or "trash" in combined:
        return "waste"

    return "object"


def _canonical_state(raw_state: str, kind: str) -> str:
    state = str(raw_state or "").lower().strip()

    # Reject bad option strings like "clean_or_discarded".
    if "_or_" in state or " or " in state:
        if "clean" in state:
            return "clean"
        return "unknown"

    if "filled" in state or "full" in state or "liquid" in state:
        return "filled"
    if "dirty" in state:
        return "dirty"
    if "used" in state or "residue" in state or "empty_after_drink" in state:
        return "used"
    if "misplaced" in state:
        return "misplaced"
    if "discarded" in state:
        return "discarded"
    if "unsorted" in state:
        return "unsorted"
    if "clean" in state or "empty" in state:
        return "clean"

    # Trash-like object with unknown state is usually actionable only if it is actually an object.
    if kind in TRASH_KINDS:
        return "misplaced"

    return "unknown"


def _canonical_location(raw_location: str) -> str:
    location = str(raw_location or "unknown").strip().lower().replace(" ", "_")
    location = re.sub(r"[^a-z0-9_]+", "", location)
    return location or "unknown"


def _normalize_object_names(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    normalized = []

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        raw_name = str(obj.get("name", "")).strip()
        raw_kind = str(obj.get("kind", "")).strip()

        name_lower = raw_name.lower().strip()
        kind = _canonical_kind(raw_kind, raw_name)

        # Drop hallucinated surfaces-as-objects, e.g. name="floor", kind="trash".
        if name_lower in SURFACE_WORDS:
            continue

        state = _canonical_state(str(obj.get("state", "")), kind)
        if state == "unknown":
            continue

        # Clean trash is contradictory. If VLM says clean trash/floor, ignore it.
        if kind in TRASH_KINDS and state == "clean":
            continue

        location = _canonical_location(str(obj.get("location", "unknown")))

        counts[kind] = counts.get(kind, 0) + 1
        name = raw_name

        if not name or not re.search(r"_\d+$", name):
            name = f"{kind}_{counts[kind]}"

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
        surfaces = []

    clean_surfaces = []
    for surface in surfaces:
        if not isinstance(surface, dict):
            continue
        name = str(surface.get("name", "")).strip()
        if not name:
            continue
        clean_surfaces.append(
            {
                "name": name,
                "state": surface.get("state", "available"),
            }
        )

    surface_names = {surface["name"] for surface in clean_surfaces}

    for default_surface in DEFAULT_SURFACES[template_name]:
        if default_surface["name"] not in surface_names:
            clean_surfaces.append(default_surface)

    surfaces = clean_surfaces

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