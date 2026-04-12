from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.action_parser import SemanticAction
from app.task_templates import get_task_template


ACTIONABLE_STATES = {"dirty", "used", "misplaced", "unsorted"}


@dataclass
class EvaluationResult:
    scenario_id: str
    template_name: str
    action_type: str
    valid_action_type: bool
    valid_target: bool
    proactive_score: float
    restraint_score: float
    progress_score: float
    memory_alignment_score: float
    total_score: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _actionable_objects(scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        obj
        for obj in scene_state.get("objects", [])
        if obj.get("state") in ACTIONABLE_STATES
    ]


def get_valid_action_types(scene_state: dict[str, Any], template_name: str) -> list[str]:
    template = get_task_template(template_name)
    user_state = scene_state.get("user_state")
    robot_state = scene_state.get("robot_state")
    held_object = scene_state.get("held_object")
    actionable = _actionable_objects(scene_state)

    if user_state in {"still_eating", "using_table"}:
        return ["wait", "inspect"]

    if held_object:
        return ["place", "wait", "inspect"]

    if actionable and robot_state == "idle":
        return [a for a in template.action_types if a in {"start_task", "pick", "inspect", "wait"}]

    if actionable and robot_state == "active":
        return [a for a in template.action_types if a in {"pick", "place", "clean_surface", "inspect", "wait"}]

    return ["wait", "inspect"]


def _valid_target(scene_state: dict[str, Any], action: SemanticAction) -> tuple[bool, list[str]]:
    notes: list[str] = []

    if action.action_type in {"wait", "start_task", "inspect"}:
        return True, notes

    object_names = {obj.get("name") for obj in scene_state.get("objects", [])}
    surface_names = {surface.get("name") for surface in scene_state.get("surfaces", [])}
    held_object = scene_state.get("held_object")

    if action.action_type == "pick":
        if not action.target_object:
            notes.append("pick action missing target_object")
            return False, notes
        if action.target_object not in object_names:
            notes.append(f"pick target_object '{action.target_object}' not present in scene")
            return False, notes
        return True, notes

    if action.action_type == "place":
        if not held_object:
            notes.append("place action proposed but robot is not holding an object")
            return False, notes
        if action.target_object and action.target_object != held_object:
            notes.append("place target_object does not match held_object")
            return False, notes
        if action.target_surface is None:
            notes.append("place action missing target_surface")
            return False, notes
        if action.target_surface not in surface_names:
            notes.append(f"place target_surface '{action.target_surface}' not present in scene")
            return False, notes
        return True, notes

    if action.action_type == "clean_surface":
        if action.target_surface is None:
            notes.append("clean_surface missing target_surface")
            return False, notes
        if action.target_surface not in surface_names:
            notes.append(f"clean_surface target_surface '{action.target_surface}' not present in scene")
            return False, notes
        return True, notes

    return True, notes


def evaluate_action(
    *,
    scenario_id: str,
    template_name: str,
    scene_state: dict[str, Any],
    action: SemanticAction,
    evaluation_hints: dict[str, Any],
) -> EvaluationResult:
    notes: list[str] = []

    valid_types = get_valid_action_types(scene_state, template_name)
    valid_action_type = action.action_type in valid_types
    if not valid_action_type:
        notes.append(f"invalid action_type '{action.action_type}' for current state; valid={valid_types}")

    valid_target, target_notes = _valid_target(scene_state, action)
    notes.extend(target_notes)

    should_wait = bool(evaluation_hints.get("should_wait", False))
    should_be_proactive = bool(evaluation_hints.get("should_be_proactive", False))
    preferred_action_types = set(evaluation_hints.get("preferred_action_types", []))
    memory_relevant = bool(evaluation_hints.get("memory_relevant", False))
    expected_surface = evaluation_hints.get("expected_preference_surface")

    # Proactiveness
    if should_be_proactive:
        proactive_score = 1.0 if action.action_type != "wait" else 0.0
        if proactive_score == 0.0:
            notes.append("robot should have acted proactively but chose wait")
    else:
        proactive_score = 1.0 if action.action_type == "wait" else 0.5

    # Restraint
    if should_wait:
        restraint_score = 1.0 if action.action_type == "wait" else 0.0
        if restraint_score == 0.0:
            notes.append("robot should have waited but chose to act")
    else:
        restraint_score = 1.0 if action.action_type != "wait" else 0.5

    # Progress
    if action.action_type in preferred_action_types:
        progress_score = 1.0
    elif action.action_type == "wait" and should_wait:
        progress_score = 1.0
    elif action.action_type == "inspect" and not should_wait:
        progress_score = 0.6
        notes.append("inspect is acceptable but lower progress than a direct task action")
    else:
        progress_score = 0.0
        notes.append("action does not move task in the preferred direction")

    # Memory alignment
    if not memory_relevant:
        memory_alignment_score = 1.0
    elif action.action_type == "place" and expected_surface is not None:
        memory_alignment_score = 1.0 if action.target_surface == expected_surface else 0.0
        if memory_alignment_score == 0.0:
            notes.append("placement did not align with stored preference surface")
    else:
        memory_alignment_score = 0.5 if action.memory_used else 0.0
        if memory_alignment_score == 0.0:
            notes.append("memory was relevant but not reflected in action")

    weights = {
        "validity": 0.30,
        "proactive": 0.20,
        "restraint": 0.20,
        "progress": 0.20,
        "memory": 0.10,
    }

    validity_score = 1.0 if (valid_action_type and valid_target) else 0.0

    total_score = (
        validity_score * weights["validity"]
        + proactive_score * weights["proactive"]
        + restraint_score * weights["restraint"]
        + progress_score * weights["progress"]
        + memory_alignment_score * weights["memory"]
    )

    return EvaluationResult(
        scenario_id=scenario_id,
        template_name=template_name,
        action_type=action.action_type,
        valid_action_type=valid_action_type,
        valid_target=valid_target,
        proactive_score=proactive_score,
        restraint_score=restraint_score,
        progress_score=progress_score,
        memory_alignment_score=memory_alignment_score,
        total_score=round(total_score, 4),
        notes=notes,
    )


def summarize_evaluations(results: list[EvaluationResult]) -> dict[str, Any]:
    if not results:
        return {
            "num_scenarios": 0,
            "avg_total_score": 0.0,
            "valid_action_rate": 0.0,
            "avg_proactive_score": 0.0,
            "avg_restraint_score": 0.0,
            "avg_progress_score": 0.0,
            "avg_memory_alignment_score": 0.0,
        }

    n = len(results)
    return {
        "num_scenarios": n,
        "avg_total_score": round(sum(r.total_score for r in results) / n, 4),
        "valid_action_rate": round(sum(1 for r in results if r.valid_action_type and r.valid_target) / n, 4),
        "avg_proactive_score": round(sum(r.proactive_score for r in results) / n, 4),
        "avg_restraint_score": round(sum(r.restraint_score for r in results) / n, 4),
        "avg_progress_score": round(sum(r.progress_score for r in results) / n, 4),
        "avg_memory_alignment_score": round(sum(r.memory_alignment_score for r in results) / n, 4),
    }