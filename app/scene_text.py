from __future__ import annotations

from typing import Any


def scene_to_text(scene_state: dict[str, Any]) -> str:
    template_name = scene_state.get("template_name", "unknown_task")
    robot_platform = scene_state.get("robot_platform", "Franka Panda")
    user_state = scene_state.get("user_state", "unknown")
    robot_state = scene_state.get("robot_state", "unknown")
    held_object = scene_state.get("held_object")

    object_lines = []
    for obj in scene_state.get("objects", []):
        object_lines.append(
            f"- {obj.get('name')} is a {obj.get('state')} {obj.get('kind')} at {obj.get('location')}"
        )

    surface_lines = []
    for surface in scene_state.get("surfaces", []):
        surface_lines.append(f"- {surface.get('name')} is {surface.get('state')}")

    held_text = held_object if held_object is not None else "nothing"

    return f"""
LIBERO-style kitchen scene.
Task template: {template_name}
Robot: {robot_platform}
User state: {user_state}
Robot state: {robot_state}
Robot is holding: {held_text}

Visible objects:
{chr(10).join(object_lines)}

Available surfaces/regions:
{chr(10).join(surface_lines)}
""".strip()