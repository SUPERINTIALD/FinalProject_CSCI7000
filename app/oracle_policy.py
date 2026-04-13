from __future__ import annotations

from typing import Any

ACTIONABLE_STATES = {"dirty", "used", "misplaced", "unsorted"}
BUSY_USER_STATES = {"still_eating", "using_table"}


def _find_object(scene_state: dict[str, Any], object_name: str | None) -> dict[str, Any] | None:
    if object_name is None:
        return None
    for obj in scene_state.get("objects", []):
        if obj.get("name") == object_name:
            return obj
    return None


def _actionable_objects(scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [obj for obj in scene_state.get("objects", []) if obj.get("state") in ACTIONABLE_STATES]


def _choose_pick_target(scene_state: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    actionable = _actionable_objects(scene_state)
    if not actionable:
        return None, []

    memory_used: list[str] = []

    # Prefer right_side first if there are multiple actionable objects
    if len(actionable) >= 2:
        right_side = [obj for obj in actionable if obj.get("location") == "right_side"]
        if right_side:
            memory_used.append("When clearing dishes, start from the right side first to avoid crossing over cleaned space.")
            actionable = right_side

    # Prefer truly actionable objects; keep cups if they are the only ones but attach risk note
    target = actionable[0]
    if target.get("kind") in {"cup", "mug"}:
        memory_used.append("A mug slipped during a fast grasp. Move slower on cups and mugs.")

    return target, memory_used


def oracle_next_action(
    scene_state: dict[str, Any],
    allowed_action_types: list[str],
) -> dict[str, Any]:
    user_state = scene_state.get("user_state")
    held_object_name = scene_state.get("held_object")
    held_object = _find_object(scene_state, held_object_name)
    preference_surface = scene_state.get("placement_preference_surface", "left_counter")

    # 1. If user is busy and robot is not already holding something, wait
    if user_state in BUSY_USER_STATES and held_object is None:
        return {
            "action_type": "wait",
            "target_object": None,
            "target_surface": None,
            "target_zone": None,
            "parameters": {},
            "reason": "The user is still busy, so the robot should wait.",
            "memory_used": [],
            "confidence": 1.0,
        }

    # 2. If holding something, place it
    if held_object is not None and "place" in allowed_action_types:
        if held_object.get("state") in ACTIONABLE_STATES:
            # Dirty/used objects go to sink
            target_surface = "sink"
            memory_used = []
            reason = f"The robot is already holding {held_object_name}, so it should place it in the sink."
        else:
            # Clean objects follow user preference
            target_surface = preference_surface
            memory_used = ["User prefers clean dishes placed on the left side of the counter."]
            reason = f"The robot is already holding {held_object_name}, so it should place it on the preferred clean-dish surface."

        if held_object.get("kind") in {"cup", "mug"}:
            memory_used.append("A mug slipped during a fast grasp. Move slower on cups and mugs.")

        return {
            "action_type": "place",
            "target_object": held_object_name,
            "target_surface": target_surface,
            "target_zone": None,
            "parameters": {},
            "reason": reason,
            "memory_used": memory_used,
            "confidence": 1.0,
        }

    # 3. Otherwise pick an actionable object if allowed
    target, extra_memory = _choose_pick_target(scene_state)
    if target is not None and "pick" in allowed_action_types:
        return {
            "action_type": "pick",
            "target_object": target["name"],
            "target_surface": None,
            "target_zone": target.get("location"),
            "parameters": {},
            "reason": f"The next useful step is to pick the actionable object {target['name']}.",
            "memory_used": extra_memory,
            "confidence": 1.0,
        }

    # 4. If no actionable move exists, wait
    return {
        "action_type": "wait",
        "target_object": None,
        "target_surface": None,
        "target_zone": None,
        "parameters": {},
        "reason": "There is no better valid task-progress action available right now.",
        "memory_used": [],
        "confidence": 1.0,
    }