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
    "trash_cleanup": TaskTemplate(
        name="trash_cleanup",
        description="Pick up trash or used disposable items from the kitchen floor and place them in the trash bin.",
        task_family="cleanup",
        user_states=("using_kitchen", "finished_eating", "left_area"),
        robot_states=("idle", "active", "holding_item"),
        object_catalog=(
            {
                "kind": "wrapper",
                "possible_states": ("misplaced", "discarded"),
                "possible_locations": ("floor", "near_counter", "under_table"),
            },
            {
                "kind": "paper_towel",
                "possible_states": ("used", "clean"),
                "possible_locations": ("floor", "near_sink", "counter"),
            },
            {
                "kind": "napkin",
                "possible_states": ("used", "clean"),
                "possible_locations": ("floor", "table_edge", "counter"),
            },
            {
                "kind": "spoon",
                "possible_states": ("clean",),
                "possible_locations": ("counter", "table_edge"),
            },
        ),
        surface_names=("floor", "trash_bin", "counter", "sink"),
        proactive_user_states=("finished_eating", "left_area"),
        memory_examples=(
            {"kind": "strategy", "text": "Trash on the floor should be picked up and placed in the trash bin."},
            {"kind": "preference", "text": "User prefers the kitchen floor to stay clear of trash."},
        ),
    ),

    "counter_cleanup": TaskTemplate(
        name="counter_cleanup",
        description="Clear leftover cups, utensils, or dishes from the kitchen counter after the user leaves.",
        task_family="cleanup",
        user_states=("using_counter", "finished_eating", "left_area"),
        robot_states=("idle", "active", "holding_item"),
        object_catalog=(
            {
                "kind": "cup",
                "possible_states": ("used", "clean"),
                "possible_locations": ("left_counter", "right_counter", "table_edge"),
            },
            {
                "kind": "fork",
                "possible_states": ("used", "clean"),
                "possible_locations": ("left_counter", "right_counter", "table_edge"),
            },
            {
                "kind": "plate",
                "possible_states": ("used", "clean"),
                "possible_locations": ("left_counter", "right_counter", "table_edge"),
            },
            {
                "kind": "sponge",
                "possible_states": ("clean",),
                "possible_locations": ("sink", "right_counter"),
            },
        ),
        surface_names=("left_counter", "right_counter", "sink", "drying_rack"),
        proactive_user_states=("finished_eating", "left_area"),
        memory_examples=(
            {"kind": "preference", "text": "User prefers used cups and utensils placed in the sink."},
            {"kind": "failure", "text": "Cups can slip if moved too quickly."},
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