# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any


# class JsonMemory:
#     def __init__(self, path: str) -> None:
#         self.path = Path(path)
#         self.path.parent.mkdir(parents=True, exist_ok=True)
#         if not self.path.exists():
#             self.path.write_text("[]", encoding="utf-8")

#     def _load(self) -> list[dict[str, Any]]:
#         try:
#             return json.loads(self.path.read_text(encoding="utf-8"))
#         except json.JSONDecodeError:
#             return []

#     def _save(self, items: list[dict[str, Any]]) -> None:
#         self.path.write_text(json.dumps(items, indent=2), encoding="utf-8")

#     def add(self, kind: str, text: str, metadata: dict[str, Any] | None = None) -> None:
#         items = self._load()
#         items.append(
#             {
#                 "kind": kind,
#                 "text": text,
#                 "metadata": metadata or {},
#             }
#         )
#         self._save(items)

#     def all(self) -> list[dict[str, Any]]:
#         return self._load()

#     def recent(self, limit: int = 10, kind: str | None = None) -> list[dict[str, Any]]:
#         items = self._load()
#         if kind is not None:
#             items = [item for item in items if item.get("kind") == kind]
#         return items[-limit:]

#     # def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
#     #     query_lower = query.lower().strip()
#     #     if not query_lower:
#     #         return self.recent(limit=limit)

#     #     scored: list[tuple[int, dict[str, Any]]] = []
#     #     for item in self._load():
#     #         haystack = f"{item.get('kind', '')} {item.get('text', '')} {item.get('metadata', {})}".lower()
#     #         score = sum(1 for term in query_lower.split() if term in haystack)
#     #         if score > 0:
#     #             scored.append((score, item))

#     #     scored.sort(key=lambda x: x[0], reverse=True)
#     #     return [item for _, item in scored[:limit]]
#     def search(
#         self,
#         query: str,
#         limit: int = 5,
#         allowed_kinds: list[str] | None = None,
#     ) -> list[dict[str, Any]]:
#         query_lower = query.lower().strip()
#         if not query_lower:
#             items = self.recent(limit=limit)
#             if allowed_kinds is not None:
#                 items = [item for item in items if item.get("kind") in allowed_kinds]
#             return items

#         scored: list[tuple[int, dict[str, Any]]] = []
#         for item in self._load():
#             if allowed_kinds is not None and item.get("kind") not in allowed_kinds:
#                 continue

#             haystack = f"{item.get('kind', '')} {item.get('text', '')} {item.get('metadata', {})}".lower()
#             score = sum(1 for term in query_lower.split() if term in haystack)
#             if score > 0:
#                 scored.append((score, item))

#         scored.sort(key=lambda x: x[0], reverse=True)
#         return [item for _, item in scored[:limit]]


from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ACTIONABLE_STATES = {"dirty", "used", "misplaced", "unsorted"}


class JsonMemory:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _save(self, items: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def add(self, kind: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        items = self._load()
        items.append(
            {
                "kind": kind,
                "text": text,
                "metadata": metadata or {},
            }
        )
        self._save(items)

    def all(self) -> list[dict[str, Any]]:
        return self._load()

    def recent(self, limit: int = 10, kind: str | None = None) -> list[dict[str, Any]]:
        items = self._load()
        if kind is not None:
            items = [item for item in items if item.get("kind") == kind]
        return items[-limit:]

    def search(
        self,
        query: str,
        limit: int = 5,
        allowed_kinds: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_lower = query.lower().strip()
        items = self._load()

        if allowed_kinds is not None:
            items = [item for item in items if item.get("kind") in allowed_kinds]

        if not query_lower:
            return items[-limit:]

        scored: list[tuple[int, dict[str, Any]]] = []
        for item in items:
            haystack = f"{item.get('kind', '')} {item.get('text', '')} {item.get('metadata', {})}".lower()
            score = sum(1 for term in query_lower.split() if term in haystack)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    # ---------- New: structured episode history ----------

    def add_episode_event(
        self,
        *,
        task_name: str,
        action_type: str,
        outcome: str,
        scene_state: dict[str, Any],
        target_object: str | None = None,
        target_surface: str | None = None,
        notes: str = "",
    ) -> None:
        self.add(
            kind="episode_event",
            text=f"{action_type} -> {outcome}",
            metadata={
                "task_name": task_name,
                "action_type": action_type,
                "outcome": outcome,
                "target_object": target_object,
                "target_surface": target_surface,
                "scene_state": scene_state,
                "notes": notes,
            },
        )

    def get_recent_episode_events(
        self,
        *,
        task_name: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        events = [x for x in self._load() if x.get("kind") == "episode_event"]
        if task_name is not None:
            events = [x for x in events if x.get("metadata", {}).get("task_name") == task_name]
        return events[-limit:]

    def build_history_summary(
        self,
        *,
        scene_state: dict[str, Any],
        max_recent: int = 4,
    ) -> str:
        template_name = scene_state.get("template_name")
        recent_events = self.get_recent_episode_events(task_name=template_name, limit=max_recent)

        lines: list[str] = []

        # Recent action-outcome tuples
        if recent_events:
            recent_bits = []
            for event in recent_events:
                md = event.get("metadata", {})
                action_type = md.get("action_type", "unknown_action")
                outcome = md.get("outcome", "unknown_outcome")
                target_object = md.get("target_object")
                target_surface = md.get("target_surface")

                extra = ""
                if target_object:
                    extra += f" object={target_object}"
                if target_surface:
                    extra += f" surface={target_surface}"

                recent_bits.append(f"{action_type}{extra} -> {outcome}")

            lines.append("Recent action history: " + " | ".join(recent_bits))
        else:
            lines.append("Recent action history: none")

        # Preference summary
        pref_items = self.search(
            query=template_name or "",
            limit=3,
            allowed_kinds=["preference"],
        )
        if pref_items:
            pref_summary = "; ".join(item["text"] for item in pref_items[:2])
            lines.append(f"Preference summary: {pref_summary}")
        else:
            lines.append("Preference summary: none")

        # Risk summary
        risk_items = self.search(
            query=template_name or "",
            limit=3,
            allowed_kinds=["failure"],
        )
        if risk_items:
            risk_summary = "; ".join(item["text"] for item in risk_items[:2])
            lines.append(f"Risk summary: {risk_summary}")
        else:
            lines.append("Risk summary: none")

        # Strategy summary
        strat_items = self.search(
            query=template_name or "",
            limit=3,
            allowed_kinds=["strategy"],
        )
        if strat_items:
            strat_summary = "; ".join(item["text"] for item in strat_items[:2])
            lines.append(f"Strategy summary: {strat_summary}")
        else:
            lines.append("Strategy summary: none")

        return "\n".join(lines)

    # ---------- New: selective retrieval ----------

    def retrieve_selective_memory(
        self,
        *,
        scene_state: dict[str, Any],
        allowed_actions: list[str],
        memory_query: str,
        limit_per_kind: int = 2,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        template_name = scene_state.get("template_name", "")
        objects = scene_state.get("objects", [])
        held_object = scene_state.get("held_object")
        user_state = scene_state.get("user_state")
        robot_state = scene_state.get("robot_state")

        actionable_objects = [obj for obj in objects if obj.get("state") in ACTIONABLE_STATES]
        object_kinds = {obj.get("kind") for obj in objects}

        # 1. Placement preference memory only when placing / holding an object
        if "place" in allowed_actions or held_object is not None:
            selected.extend(
                self.search(
                    query=f"{memory_query} {template_name} place placement preference",
                    limit=limit_per_kind,
                    allowed_kinds=["preference"],
                )
            )

        # 2. Failure/risk memory for fragile or risky objects
        if any(kind in {"cup", "mug"} for kind in object_kinds):
            selected.extend(
                self.search(
                    query=f"{memory_query} {template_name} cup mug slip risk",
                    limit=limit_per_kind,
                    allowed_kinds=["failure"],
                )
            )

        # 3. Strategy memory when there are multiple actionable objects
        if len(actionable_objects) >= 2:
            selected.extend(
                self.search(
                    query=f"{memory_query} {template_name} ordering strategy first",
                    limit=limit_per_kind,
                    allowed_kinds=["strategy"],
                )
            )

        # 4. If user is still busy, we usually do not need preference memory
        #    unless the robot is already holding something and deciding where to place it.
        if user_state in {"still_eating", "using_table"} and held_object is None:
            selected = [m for m in selected if m.get("kind") != "preference"]

        # 5. Deduplicate by (kind, text)
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in selected:
            key = (item.get("kind", ""), item.get("text", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(item)

        return deduped[:6]