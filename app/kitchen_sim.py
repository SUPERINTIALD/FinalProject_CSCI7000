from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from app.action_parser import SemanticAction


ACTIONABLE_STATES = {"dirty", "used", "misplaced", "unsorted"}


@dataclass
class SimStepResult:
    success: bool
    message: str
    updated_scene_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KitchenSim:
    """
    Lightweight symbolic kitchen simulator.

    This does not simulate joints or physics.
    It simulates whether a Franka Panda-style skill call is valid:
    - pick object
    - place object
    - start task
    - inspect
    - wait
    """

    def __init__(self, scene_state: dict[str, Any]) -> None:
        self.scene_state = deepcopy(scene_state)

    def get_scene_state(self) -> dict[str, Any]:
        return deepcopy(self.scene_state)

    def _find_object(self, name: str | None) -> dict[str, Any] | None:
        if name is None:
            return None
        for obj in self.scene_state.get("objects", []):
            if obj.get("name") == name:
                return obj
        return None

    def _surface_names(self) -> set[str]:
        return {s.get("name") for s in self.scene_state.get("surfaces", [])}

    def step(self, action: SemanticAction) -> SimStepResult:
        action_type = action.action_type

        if action_type == "wait":
            self.scene_state["last_action_result"] = "waited"
            return SimStepResult(True, "Robot waited.", self.get_scene_state())

        if action_type == "inspect":
            self.scene_state["last_action_result"] = "inspected"
            return SimStepResult(True, "Robot inspected the scene.", self.get_scene_state())

        if action_type == "start_task":
            self.scene_state["robot_state"] = "active"
            self.scene_state["last_action_result"] = "task_started"
            return SimStepResult(True, "Robot started the task.", self.get_scene_state())

        if action_type == "pick":
            return self._pick(action)

        if action_type == "place":
            return self._place(action)

        if action_type == "clean_surface":
            self.scene_state["last_action_result"] = "surface_cleaned"
            return SimStepResult(True, "Robot cleaned the target surface.", self.get_scene_state())

        self.scene_state["last_action_result"] = "unknown_action_failed"
        return SimStepResult(False, f"Unknown action type: {action_type}", self.get_scene_state())

    def _pick(self, action: SemanticAction) -> SimStepResult:
        if self.scene_state.get("held_object") is not None:
            self.scene_state["last_action_result"] = "pick_failed_already_holding"
            return SimStepResult(False, "Cannot pick because robot is already holding an object.", self.get_scene_state())

        obj = self._find_object(action.target_object)
        if obj is None:
            self.scene_state["last_action_result"] = "pick_failed_missing_object"
            return SimStepResult(False, f"Target object not found: {action.target_object}", self.get_scene_state())

        if obj.get("state") not in ACTIONABLE_STATES:
            self.scene_state["last_action_result"] = "pick_failed_non_actionable"
            return SimStepResult(False, f"Object is not actionable: {obj.get('state')}", self.get_scene_state())

        self.scene_state["held_object"] = obj["name"]
        self.scene_state["robot_state"] = "holding_item"
        obj["location"] = "gripper"
        self.scene_state["last_action_result"] = "picked_successfully"

        return SimStepResult(True, f"Picked {obj['name']}.", self.get_scene_state())

    def _place(self, action: SemanticAction) -> SimStepResult:
        held = self.scene_state.get("held_object")
        if held is None:
            self.scene_state["last_action_result"] = "place_failed_not_holding"
            return SimStepResult(False, "Cannot place because robot is not holding an object.", self.get_scene_state())

        if action.target_surface not in self._surface_names():
            self.scene_state["last_action_result"] = "place_failed_bad_surface"
            return SimStepResult(False, f"Unknown target surface: {action.target_surface}", self.get_scene_state())

        obj = self._find_object(held)
        if obj is None:
            self.scene_state["last_action_result"] = "place_failed_missing_held_object"
            return SimStepResult(False, "Held object is missing from scene objects.", self.get_scene_state())

        obj["location"] = action.target_surface

        if action.target_surface == "trash_bin":
            obj["state"] = "discarded"
        elif action.target_surface in {"sink", "drying_rack"} and obj.get("state") in ACTIONABLE_STATES:
            obj["state"] = "placed"
        elif obj.get("state") in ACTIONABLE_STATES:
            obj["state"] = "placed"

        self.scene_state["held_object"] = None
        self.scene_state["robot_state"] = "active"
        self.scene_state["last_action_result"] = "placed_successfully"

        return SimStepResult(True, f"Placed {held} on/in {action.target_surface}.", self.get_scene_state())