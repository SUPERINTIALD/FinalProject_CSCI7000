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

ALLOWED ACTIONS:
{json.dumps(allowed_actions, indent=2)}

RELEVANT MEMORY:
{json.dumps(relevant_memory, indent=2)}
""".strip()

    def _safe_parse(self, raw_text: str, allowed_actions: list[str]) -> dict[str, Any]:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            return {
                "action": "wait",
                "reason": f"Failed to parse model JSON: {exc}",
                "memory_used": [],
                "confidence": 0.0,
                "raw_model_output": raw_text,
            }

        action = parsed.get("action", "wait")
        if action not in allowed_actions:
            return {
                "action": "wait",
                "reason": f"Model selected invalid action: {action}",
                "memory_used": parsed.get("memory_used", []),
                "confidence": 0.0,
                "raw_model_output": raw_text,
            }

        reason = str(parsed.get("reason", "")).strip()
        memory_used = parsed.get("memory_used", [])
        confidence = parsed.get("confidence", 0.0)

        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        return {
            "action": action,
            "reason": reason,
            "memory_used": memory_used if isinstance(memory_used, list) else [],
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
        # relevant_memory = self.memory.search(memory_query, limit=5) if use_memory else []
        relevant_memory = (
            self.memory.search(
                memory_query,
                limit=5,
                allowed_kinds=["preference", "strategy", "failure"],
            )
            if use_memory
            else []
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

        result = self._safe_parse(raw, allowed_actions)
        result["used_memory"] = use_memory
        result["retrieved_memory"] = relevant_memory
        return result