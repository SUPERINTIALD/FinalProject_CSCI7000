# from __future__ import annotations

# import json
# from typing import Any

# from app.llm_client import LMStudioClient
# from app.memory import JsonMemory
# from app.prompts import SYSTEM_PROMPT


# class Planner:
#     def __init__(self, llm: LMStudioClient, memory: JsonMemory) -> None:
#         self.llm = llm
#         self.memory = memory

#     def _build_user_prompt(
#         self,
#         scene_state: dict[str, Any],
#         allowed_actions: list[str],
#         relevant_memory: list[dict[str, Any]],
#     ) -> str:
#         return f"""
# SCENE STATE:
# {json.dumps(scene_state, indent=2)}

# ALLOWED ACTIONS:
# {json.dumps(allowed_actions, indent=2)}

# RELEVANT MEMORY:
# {json.dumps(relevant_memory, indent=2)}
# """.strip()

#     def _safe_parse(self, raw_text: str, allowed_actions: list[str]) -> dict[str, Any]:
#         try:
#             parsed = json.loads(raw_text)
#         except json.JSONDecodeError as exc:
#             return {
#                 "action": "wait",
#                 "reason": f"Failed to parse model JSON: {exc}",
#                 "memory_used": [],
#                 "confidence": 0.0,
#                 "raw_model_output": raw_text,
#             }

#         action = parsed.get("action", "wait")
#         if action not in allowed_actions:
#             return {
#                 "action": "wait",
#                 "reason": f"Model selected invalid action: {action}",
#                 "memory_used": parsed.get("memory_used", []),
#                 "confidence": 0.0,
#                 "raw_model_output": raw_text,
#             }

#         reason = str(parsed.get("reason", "")).strip()
#         memory_used = parsed.get("memory_used", [])
#         confidence = parsed.get("confidence", 0.0)

#         try:
#             confidence = float(confidence)
#         except (ValueError, TypeError):
#             confidence = 0.0

#         confidence = max(0.0, min(1.0, confidence))

#         return {
#             "action": action,
#             "reason": reason,
#             "memory_used": memory_used if isinstance(memory_used, list) else [],
#             "confidence": confidence,
#             "raw_model_output": raw_text,
#         }

#     def choose_action(
#         self,
#         scene_state: dict[str, Any],
#         allowed_actions: list[str],
#         memory_query: str,
#         use_memory: bool = True,
#     ) -> dict[str, Any]:
#         # relevant_memory = self.memory.search(memory_query, limit=5) if use_memory else []
#         relevant_memory = (
#             self.memory.search(
#                 memory_query,
#                 limit=5,
#                 allowed_kinds=["preference", "strategy", "failure"],
#             )
#             if use_memory
#             else []
#         )

#         user_prompt = self._build_user_prompt(
#             scene_state=scene_state,
#             allowed_actions=allowed_actions,
#             relevant_memory=relevant_memory,
#         )

#         raw = self.llm.generate(
#             system_prompt=SYSTEM_PROMPT,
#             user_prompt=user_prompt,
#             temperature=0.1,
#         )

#         result = self._safe_parse(raw, allowed_actions)
#         result["used_memory"] = use_memory
#         result["retrieved_memory"] = relevant_memory
#         return result


from __future__ import annotations

import json
from typing import Any

from app.llm_client import LMStudioClient
from app.memory import JsonMemory
from app.prompts import SYSTEM_PROMPT


class Planner:
    def __init__(self, llm: LMStudioClient, memory: JsonMemory) -> None:
        self.llm = llm
        self.memory = memory

    def _build_user_prompt(
        self,
        scene_state: dict[str, Any],
        allowed_actions: list[str],
        relevant_memory: list[dict[str, Any]],
    ) -> str:
        return f"""
SCENE STATE:
{json.dumps(scene_state, indent=2)}

ALLOWED ACTION TYPES:
{json.dumps(allowed_actions, indent=2)}

RELEVANT MEMORY:
{json.dumps(relevant_memory, indent=2)}
""".strip()

    def _default_response(self, reason: str, raw_text: str = "") -> dict[str, Any]:
        return {
            "action_type": "wait",
            "target_object": None,
            "target_surface": None,
            "target_zone": None,
            "parameters": {},
            "reason": reason,
            "memory_used": [],
            "confidence": 0.0,
            "raw_model_output": raw_text,
        }

    def _safe_float(self, value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        return max(0.0, min(1.0, value))

    def _get_scene_object_names(self, scene_state: dict[str, Any]) -> set[str]:
        return {
            obj.get("name")
            for obj in scene_state.get("objects", [])
            if obj.get("name") is not None
        }

    def _get_scene_surface_names(self, scene_state: dict[str, Any]) -> set[str]:
        return {
            surface.get("name")
            for surface in scene_state.get("surfaces", [])
            if surface.get("name") is not None
        }

    def _search_memory(
        self,
        memory_query: str,
        scene_state: dict[str, Any],
        limit: int = 5,
        use_memory: bool = True,
    ) -> list[dict[str, Any]]:
        if not use_memory:
            return []

        task_name = scene_state.get("template_name")

        try:
            task_memory = self.memory.search(
                memory_query,
                limit=limit,
                allowed_kinds=["preference", "strategy", "failure"],
                metadata_filters={"task": task_name},
            )
            return task_memory
        except TypeError:
            return self.memory.search(memory_query, limit=limit)

    def _safe_parse(
        self,
        raw_text: str,
        allowed_actions: list[str],
        scene_state: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            return self._default_response(
                reason=f"Failed to parse model JSON: {exc}",
                raw_text=raw_text,
            )

        action_type = str(parsed.get("action_type", "wait")).strip()
        if action_type not in allowed_actions:
            return self._default_response(
                reason=f"Model selected invalid action_type: {action_type}",
                raw_text=raw_text,
            )

        target_object = parsed.get("target_object")
        target_surface = parsed.get("target_surface")
        target_zone = parsed.get("target_zone")
        parameters = parsed.get("parameters", {})
        reason = str(parsed.get("reason", "")).strip()
        memory_used = parsed.get("memory_used", [])
        confidence = self._safe_float(parsed.get("confidence", 0.0))

        if target_object == "":
            target_object = None
        if target_surface == "":
            target_surface = None
        if target_zone == "":
            target_zone = None

        if target_object is not None and not isinstance(target_object, str):
            target_object = None
        if target_surface is not None and not isinstance(target_surface, str):
            target_surface = None
        if target_zone is not None and not isinstance(target_zone, str):
            target_zone = None

        if not isinstance(parameters, dict):
            parameters = {}

        if not isinstance(memory_used, list):
            memory_used = []

        scene_object_names = self._get_scene_object_names(scene_state)
        scene_surface_names = self._get_scene_surface_names(scene_state)

        if target_object is not None and target_object not in scene_object_names:
            return self._default_response(
                reason=f"Model selected unknown target_object: {target_object}",
                raw_text=raw_text,
            )

        if target_surface is not None and target_surface not in scene_surface_names:
            return self._default_response(
                reason=f"Model selected unknown target_surface: {target_surface}",
                raw_text=raw_text,
            )

        return {
            "action_type": action_type,
            "target_object": target_object,
            "target_surface": target_surface,
            "target_zone": target_zone,
            "parameters": parameters,
            "reason": reason,
            "memory_used": memory_used,
            "confidence": confidence,
            "raw_model_output": raw_text,
        }

    def choose_action(
        self,
        scene_state: dict[str, Any],
        allowed_actions: list[str],
        memory_query: str,
        use_memory: bool = True,
    ) -> dict[str, Any]:
        # relevant_memory = self._search_memory(
        #     memory_query=memory_query,
        #     limit=5,
        #     use_memory=use_memory,
        # )
        relevant_memory = self._search_memory(
            memory_query=memory_query,
            scene_state=scene_state,
            limit=5,
            use_memory=use_memory,
        )

        user_prompt = self._build_user_prompt(
            scene_state=scene_state,
            allowed_actions=allowed_actions,
            relevant_memory=relevant_memory,
        )

        raw = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        result = self._safe_parse(
            raw_text=raw,
            allowed_actions=allowed_actions,
            scene_state=scene_state,
        )
        result["used_memory"] = use_memory
        result["retrieved_memory"] = relevant_memory
        return result