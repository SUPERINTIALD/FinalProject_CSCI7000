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
from pathlib import Path
from typing import Any


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

    def upsert(self, kind: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        items = self._load()

        for item in items:
            if item.get("kind") == kind and item.get("text") == text and item.get("metadata") == metadata:
                return

        items.append(
            {
                "kind": kind,
                "text": text,
                "metadata": metadata,
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
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_lower = query.lower().strip()
        metadata_filters = metadata_filters or {}

        candidates = []
        for item in self._load():
            if allowed_kinds is not None and item.get("kind") not in allowed_kinds:
                continue

            metadata = item.get("metadata", {})
            matches_metadata = all(metadata.get(k) == v for k, v in metadata_filters.items())
            if not matches_metadata:
                continue

            candidates.append(item)

        if not query_lower:
            return candidates[-limit:]

        scored: list[tuple[int, dict[str, Any]]] = []
        for item in candidates:
            haystack = f"{item.get('kind', '')} {item.get('text', '')} {item.get('metadata', {})}".lower()
            score = sum(1 for term in query_lower.split() if term in haystack)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]