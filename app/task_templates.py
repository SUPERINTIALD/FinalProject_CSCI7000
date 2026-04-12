from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskTemplate:
    name: str
    description: str
    task_family: str
    user_states: tuple[str, ...]
    robot_states: tuple[str, ...]
    object_catalog: tuple[dict[str, Any], ...]
    surface_names: tuple[str, ...]
    action_types: tuple[str, ...] = (
        "wait",
        "start_task",
        "pick",
        "place",
        "clean_surface",
        "inspect",
    )
    proactive_user_states: tuple[str, ...] = ("finished_eating", "left_area")
    memory_examples: tuple[dict[str, Any], ...] = field(default_factory=tuple)


TASK_TEMPLATES: dict[str, TaskTemplate] = {
    "dish_cleanup": TaskTemplate(
        name="dish_cleanup",
        description="Clear dirty dishes after a meal and place cleaned items in preferred locations.",
        task_family="cleanup",
        user_states=("still_eating", "finished_eating", "left_area"),
        robot_states=("idle", "active", "holding_item"),
        object_catalog=(
            {"kind": "plate", "possible_states": ("dirty", "clean"), "possible_locations": ("right_side", "center", "left_side", "table_edge")},
            {"kind": "bowl", "possible_states": ("dirty", "clean"), "possible_locations": ("right_side", "center", "left_side")},
            {"kind": "cup", "possible_states": ("dirty", "clean"), "possible_locations": ("right_side", "center", "left_side")},
            {"kind": "fork", "possible_states": ("dirty", "clean"), "possible_locations": ("right_side", "center", "left_side")},
        ),
        surface_names=("left_counter", "right_counter", "sink", "drying_rack"),
        memory_examples=(
            {"kind": "preference", "text": "User prefers clean dishes on the left counter."},
            {"kind": "strategy", "text": "Start collecting dishes from the right side first."},
            {"kind": "failure", "text": "Move slower when picking up mugs or cups."},
        ),
    ),
    "table_reset": TaskTemplate(
        name="table_reset",
        description="Reset a table after use by removing used items and straightening surfaces.",
        task_family="cleanup",
        user_states=("using_table", "finished_using_table", "left_area"),
        robot_states=("idle", "active", "holding_item"),
        object_catalog=(
            {"kind": "napkin", "possible_states": ("used", "clean"), "possible_locations": ("center", "left_side", "right_side")},
            {"kind": "plate", "possible_states": ("used", "clean"), "possible_locations": ("center", "left_side", "right_side")},
            {"kind": "cup", "possible_states": ("used", "clean"), "possible_locations": ("center", "left_side", "right_side")},
        ),
        surface_names=("table_center", "bin", "left_counter", "right_counter"),
        proactive_user_states=("finished_using_table", "left_area"),
        memory_examples=(
            {"kind": "preference", "text": "User prefers cups grouped together when clearing the table."},
        ),
    ),
    "object_sorting": TaskTemplate(
        name="object_sorting",
        description="Sort mixed objects into preferred zones.",
        task_family="sorting",
        user_states=("idle", "watching", "left_area"),
        robot_states=("idle", "active", "holding_item"),
        object_catalog=(
            {"kind": "block_red", "possible_states": ("unsorted", "sorted"), "possible_locations": ("bin_a", "bin_b", "table_center")},
            {"kind": "block_blue", "possible_states": ("unsorted", "sorted"), "possible_locations": ("bin_a", "bin_b", "table_center")},
            {"kind": "block_green", "possible_states": ("unsorted", "sorted"), "possible_locations": ("bin_a", "bin_b", "table_center")},
        ),
        surface_names=("bin_a", "bin_b", "table_center"),
        proactive_user_states=("idle", "left_area"),
        memory_examples=(
            {"kind": "preference", "text": "User prefers red objects in bin_a and blue objects in bin_b."},
        ),
    ),
}


def get_task_template(name: str) -> TaskTemplate:
    if name not in TASK_TEMPLATES:
        raise KeyError(f"Unknown task template: {name}")
    return TASK_TEMPLATES[name]


def list_task_templates() -> list[str]:
    return list(TASK_TEMPLATES.keys())