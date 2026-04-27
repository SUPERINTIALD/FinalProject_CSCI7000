from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from app.task_templates import get_task_template


@dataclass
class Scenario:
    scenario_id: str
    template_name: str
    scene_state: dict[str, Any]
    memory_query: str
    evaluation_hints: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ACTIONABLE_STATES = {"dirty", "used", "misplaced", "unsorted"}


def _build_object_name(kind: str, index: int) -> str:
    return f"{kind}_{index}"


def _make_objects(
    template_name: str,
    rng: random.Random,
    *,
    min_objects: int = 1,
    max_objects: int = 4,
) -> list[dict[str, Any]]:
    template = get_task_template(template_name)
    count = rng.randint(min_objects, max_objects)
    objects: list[dict[str, Any]] = []

    for i in range(count):
        spec = rng.choice(template.object_catalog)
        kind = spec["kind"]
        state = rng.choice(spec["possible_states"])
        location = rng.choice(spec["possible_locations"])

        objects.append(
            {
                "name": _build_object_name(kind, i + 1),
                "kind": kind,
                "state": state,
                "location": location,
            }
        )

    return objects


def _first_actionable_object(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    for obj in objects:
        if obj.get("state") in ACTIONABLE_STATES:
            return obj
    return None


def _set_dish_cleanup_scene(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    user_state = rng.choice(["still_eating", "finished_eating", "left_area"])
    robot_state = rng.choice(["idle", "active", "holding_item"])

    objects = _make_objects("dish_cleanup", rng)
    actionable = _first_actionable_object(objects)

    held_object = None
    if robot_state == "holding_item":
        held_object = actionable["name"] if actionable is not None else objects[0]["name"]

    scene_state = {
        "task_family": "cleanup",
        "template_name": "dish_cleanup",
        "user_state": user_state,
        "robot_state": robot_state,
        "objects": objects,
        "surfaces": [
            {"name": "left_counter", "state": "available"},
            {"name": "right_counter", "state": "available"},
            {"name": "sink", "state": "available"},
            {"name": "drying_rack", "state": "available"},
        ],
        "held_object": held_object,
        "placement_preference_surface": "left_counter",
        "last_action_result": rng.choice(["none", "cleanup_started", "picked_successfully", "place_failed"]),
    }

    should_wait = user_state == "still_eating" or (actionable is None and held_object is None)
    should_be_proactive = user_state in {"finished_eating", "left_area"} and (actionable is not None or held_object is not None)

    preferred_action_types: list[str]
    if should_wait:
        preferred_action_types = ["wait"]
    elif held_object is not None:
        preferred_action_types = ["place"]
    elif robot_state == "idle":
        preferred_action_types = ["start_task", "pick"]
    else:
        preferred_action_types = ["pick", "inspect"]

    hints = {
        "should_wait": should_wait,
        "should_be_proactive": should_be_proactive,
        "preferred_action_types": preferred_action_types,
        "memory_relevant": held_object is not None or any(obj.get("state") == "clean" for obj in objects),
        "expected_preference_surface": "left_counter",
        "actionable_object_present": actionable is not None,
        "task_family": "cleanup",
    }

    return scene_state, hints


def _set_table_reset_scene(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    user_state = rng.choice(["using_table", "finished_using_table", "left_area"])
    robot_state = rng.choice(["idle", "active", "holding_item"])

    objects = _make_objects("table_reset", rng)
    actionable = _first_actionable_object(objects)

    held_object = None
    if robot_state == "holding_item":
        held_object = actionable["name"] if actionable is not None else objects[0]["name"]

    scene_state = {
        "task_family": "cleanup",
        "template_name": "table_reset",
        "user_state": user_state,
        "robot_state": robot_state,
        "objects": objects,
        "surfaces": [
            {"name": "table_center", "state": "occupied"},
            {"name": "bin", "state": "available"},
            {"name": "left_counter", "state": "available"},
            {"name": "right_counter", "state": "available"},
        ],
        "held_object": held_object,
        "placement_preference_surface": "right_counter",
        "last_action_result": rng.choice(["none", "reset_started", "picked_successfully", "place_failed"]),
    }

    should_wait = user_state == "using_table" or (actionable is None and held_object is None)
    should_be_proactive = user_state in {"finished_using_table", "left_area"} and (actionable is not None or held_object is not None)

    if should_wait:
        preferred_action_types = ["wait"]
    elif held_object is not None:
        preferred_action_types = ["place"]
    elif robot_state == "idle":
        preferred_action_types = ["start_task", "pick"]
    else:
        preferred_action_types = ["pick", "clean_surface"]

    hints = {
        "should_wait": should_wait,
        "should_be_proactive": should_be_proactive,
        "preferred_action_types": preferred_action_types,
        "memory_relevant": held_object is not None,
        "expected_preference_surface": "right_counter",
        "actionable_object_present": actionable is not None,
        "task_family": "cleanup",
    }

    return scene_state, hints


def _set_object_sorting_scene(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    user_state = rng.choice(["idle", "watching", "left_area"])
    robot_state = rng.choice(["idle", "active", "holding_item"])

    objects = _make_objects("object_sorting", rng)
    actionable = _first_actionable_object(objects)

    held_object = None
    if robot_state == "holding_item":
        held_object = actionable["name"] if actionable is not None else objects[0]["name"]

    scene_state = {
        "task_family": "sorting",
        "template_name": "object_sorting",
        "user_state": user_state,
        "robot_state": robot_state,
        "objects": objects,
        "surfaces": [
            {"name": "bin_a", "state": "available"},
            {"name": "bin_b", "state": "available"},
            {"name": "table_center", "state": "available"},
        ],
        "held_object": held_object,
        "placement_preference_surface": None,
        "last_action_result": rng.choice(["none", "sorting_started", "picked_successfully", "place_failed"]),
    }

    should_wait = actionable is None and held_object is None
    should_be_proactive = user_state in {"idle", "left_area"} and (actionable is not None or held_object is not None)

    if should_wait:
        preferred_action_types = ["wait"]
    elif held_object is not None:
        preferred_action_types = ["place"]
    elif robot_state == "idle":
        preferred_action_types = ["start_task", "pick"]
    else:
        preferred_action_types = ["pick", "place", "inspect"]

    hints = {
        "should_wait": should_wait,
        "should_be_proactive": should_be_proactive,
        "preferred_action_types": preferred_action_types,
        "memory_relevant": held_object is not None,
        "expected_preference_surface": None,
        "actionable_object_present": actionable is not None,
        "task_family": "sorting",
    }

    return scene_state, hints

def _set_trash_cleanup_scene(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    user_state = rng.choice(["using_kitchen", "finished_eating", "left_area"])
    robot_state = rng.choice(["idle", "active", "holding_item"])

    objects = _make_objects("trash_cleanup", rng)
    actionable = _first_actionable_object(objects)

    held_object = None
    if robot_state == "holding_item":
        held_object = actionable["name"] if actionable is not None else objects[0]["name"]

    scene_state = {
        "task_family": "cleanup",
        "template_name": "trash_cleanup",
        "scene_family": "libero_style_kitchen",
        "robot_platform": "Franka Panda",
        "user_state": user_state,
        "robot_state": robot_state,
        "objects": objects,
        "surfaces": [
            {"name": "floor", "state": "available"},
            {"name": "trash_bin", "state": "available"},
            {"name": "counter", "state": "available"},
            {"name": "sink", "state": "available"},
        ],
        "held_object": held_object,
        "placement_preference_surface": "trash_bin",
        "last_action_result": rng.choice(["none", "trash_seen", "picked_successfully", "place_failed"]),
    }

    should_wait = actionable is None and held_object is None
    should_be_proactive = user_state in {"finished_eating", "left_area"} and (
        actionable is not None or held_object is not None
    )

    if should_wait:
        preferred_action_types = ["wait"]
    elif held_object is not None:
        preferred_action_types = ["place"]
    elif robot_state == "idle":
        preferred_action_types = ["start_task", "pick"]
    else:
        preferred_action_types = ["pick", "inspect"]

    hints = {
        "should_wait": should_wait,
        "should_be_proactive": should_be_proactive,
        "preferred_action_types": preferred_action_types,
        "memory_relevant": held_object is not None or actionable is not None,
        "expected_preference_surface": "trash_bin",
        "actionable_object_present": actionable is not None,
        "task_family": "cleanup",
    }

    return scene_state, hints


def _set_counter_cleanup_scene(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    user_state = rng.choice(["using_counter", "finished_eating", "left_area"])
    robot_state = rng.choice(["idle", "active", "holding_item"])

    objects = _make_objects("counter_cleanup", rng)
    actionable = _first_actionable_object(objects)

    held_object = None
    if robot_state == "holding_item":
        held_object = actionable["name"] if actionable is not None else objects[0]["name"]

    scene_state = {
        "task_family": "cleanup",
        "template_name": "counter_cleanup",
        "scene_family": "libero_style_kitchen",
        "robot_platform": "Franka Panda",
        "user_state": user_state,
        "robot_state": robot_state,
        "objects": objects,
        "surfaces": [
            {"name": "left_counter", "state": "available"},
            {"name": "right_counter", "state": "available"},
            {"name": "sink", "state": "available"},
            {"name": "drying_rack", "state": "available"},
        ],
        "held_object": held_object,
        "placement_preference_surface": "sink",
        "last_action_result": rng.choice(["none", "counter_item_seen", "picked_successfully", "place_failed"]),
    }

    should_wait = user_state == "using_counter" or (actionable is None and held_object is None)
    should_be_proactive = user_state in {"finished_eating", "left_area"} and (
        actionable is not None or held_object is not None
    )

    if should_wait:
        preferred_action_types = ["wait"]
    elif held_object is not None:
        preferred_action_types = ["place"]
    elif robot_state == "idle":
        preferred_action_types = ["start_task", "pick"]
    else:
        preferred_action_types = ["pick", "inspect"]

    hints = {
        "should_wait": should_wait,
        "should_be_proactive": should_be_proactive,
        "preferred_action_types": preferred_action_types,
        "memory_relevant": held_object is not None or actionable is not None,
        "expected_preference_surface": "sink",
        "actionable_object_present": actionable is not None,
        "task_family": "cleanup",
    }

    return scene_state, hints


SCENARIO_BUILDERS = {
    "dish_cleanup": _set_dish_cleanup_scene,
    "table_reset": _set_table_reset_scene,
    "trash_cleanup": _set_trash_cleanup_scene,
    "counter_cleanup": _set_counter_cleanup_scene,
    "object_sorting": _set_object_sorting_scene,
}


def _build_memory_query(scene_state: dict[str, Any]) -> str:
    parts = [
        scene_state.get("template_name", ""),
        scene_state.get("task_family", ""),
        scene_state.get("user_state", ""),
        scene_state.get("robot_state", ""),
    ]

    held_object = scene_state.get("held_object")
    if held_object:
        parts.append(f"holding {held_object}")

    preference_surface = scene_state.get("placement_preference_surface")
    if preference_surface:
        parts.append(f"preference {preference_surface}")

    for obj in scene_state.get("objects", []):
        parts.append(obj.get("kind", ""))
        parts.append(obj.get("state", ""))
        parts.append(obj.get("location", ""))

    return " ".join(part for part in parts if part).strip()


def generate_scenarios(
    *,
    num_scenarios: int = 20,
    template_names: list[str] | None = None,
    seed: int = 42,
) -> list[Scenario]:
    rng = random.Random(seed)
    template_names = template_names or ["dish_cleanup"]

    scenarios: list[Scenario] = []

    for idx in range(num_scenarios):
        template_name = rng.choice(template_names)
        builder = SCENARIO_BUILDERS[template_name]
        scene_state, hints = builder(rng)

        scenario = Scenario(
            scenario_id=f"{template_name}_{idx + 1:04d}",
            template_name=template_name,
            scene_state=scene_state,
            memory_query=_build_memory_query(scene_state),
            evaluation_hints=hints,
        )
        scenarios.append(scenario)

    return scenarios