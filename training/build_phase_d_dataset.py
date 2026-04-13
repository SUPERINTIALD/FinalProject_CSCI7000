from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from app.evaluator import get_valid_action_types
from app.oracle_policy import oracle_next_action
from app.scenario_generator import generate_scenarios


SYSTEM_PROMPT = """
You are the high-level planner for a small local household robot.

You do NOT control motors, coordinates, or joint angles.
You only choose the next semantic action.

You are given:
- SCENE STATE
- ALLOWED ACTION TYPES
- RECENT EVENTS
- RELEVANT MEMORY

Rules:
1. Use the scene state first.
2. Use memory only when relevant for personalization, preferences, or tie-breaking.
3. Never let memory override the scene state.
4. Be proactive when help is clearly needed.
5. Choose exactly one action_type from ALLOWED ACTION TYPES.
6. Only refer to objects and surfaces that actually appear in the SCENE STATE.
7. Do not invent object names, surface names, or zones.
8. If no action should be taken yet, choose "wait".
9. If more information is needed before acting, choose "inspect".
10. Return valid JSON only.
11. Do not include markdown fences.
12. Keep the reason short.
13. Do NOT return the key "action". You MUST return "action_type".
14. target_object and target_surface must be null or exact names from the scene.
15. For "pick", choose only an actionable object such as one that is dirty, used, misplaced, or unsorted.

Return this exact schema:
{
  "action_type": "one_allowed_action_type",
  "target_object": "object_name_or_null",
  "target_surface": "surface_name_or_null",
  "target_zone": "zone_name_or_null",
  "parameters": {},
  "reason": "short explanation based on scene first, memory second",
  "memory_used": ["short memory snippet"],
  "confidence": 0.0
}
""".strip()


SEED_MEMORY = [
    {
        "kind": "preference",
        "text": "User prefers clean dishes placed on the left side of the counter.",
        "metadata": {"task": "dish_cleanup"},
    },
    {
        "kind": "strategy",
        "text": "When clearing dishes, start from the right side first to avoid crossing over cleaned space.",
        "metadata": {"task": "dish_cleanup"},
    },
    {
        "kind": "failure",
        "text": "A mug slipped during a fast grasp. Move slower on cups and mugs.",
        "metadata": {"task": "dish_cleanup"},
    },
]


def build_recent_events(scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    last_result = scene_state.get("last_action_result", "none")
    held_object = scene_state.get("held_object")
    if last_result == "place_failed":
        return [{"action_type": "place", "outcome": "failure", "target_object": held_object}]
    if last_result == "picked_successfully":
        return [{"action_type": "pick", "outcome": "success", "target_object": held_object}]
    if last_result == "cleanup_started":
        return [{"action_type": "start_task", "outcome": "success"}]
    return []


def select_relevant_memory(scene_state: dict[str, Any], allowed_action_types: list[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    objects = scene_state.get("objects", [])
    held_object_name = scene_state.get("held_object")
    held_obj = None
    for obj in objects:
        if obj.get("name") == held_object_name:
            held_obj = obj
            break

    actionable = [o for o in objects if o.get("state") in {"dirty", "used", "misplaced", "unsorted"}]

    # Preference only when placing clean objects
    if held_obj is not None and held_obj.get("state") not in {"dirty", "used", "misplaced", "unsorted"}:
        selected.extend([m for m in SEED_MEMORY if m["kind"] == "preference"])

    # Risk only when cups/mugs are involved
    if any(obj.get("kind") in {"cup", "mug"} for obj in objects):
        selected.extend([m for m in SEED_MEMORY if m["kind"] == "failure"])

    # Strategy only when multiple actionable objects exist
    if len(actionable) >= 2:
        selected.extend([m for m in SEED_MEMORY if m["kind"] == "strategy"])

    # Dedup
    seen = set()
    deduped = []
    for item in selected:
        key = (item["kind"], item["text"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def render_user_prompt(
    *,
    scene_state: dict[str, Any],
    allowed_action_types: list[str],
    recent_events: list[dict[str, Any]],
    relevant_memory: list[dict[str, Any]],
) -> str:
    return f"""
SCENE STATE:
{json.dumps(scene_state, indent=2)}

ALLOWED ACTION TYPES:
{json.dumps(allowed_action_types, indent=2)}

RECENT EVENTS:
{json.dumps(recent_events, indent=2)}

RELEVANT MEMORY:
{json.dumps(relevant_memory, indent=2)}
""".strip()


def maybe_dropout_reason(target: dict[str, Any], prob: float, rng: random.Random) -> dict[str, Any]:
    target = dict(target)
    if rng.random() < prob:
        target["reason"] = ""
        target["memory_used"] = []
    return target


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-scenarios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template", type=str, default="dish_cleanup")
    parser.add_argument("--output-dir", type=str, default="data/phase_d_sft")
    parser.add_argument("--reason-dropout-prob", type=float, default=0.3)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    scenarios = generate_scenarios(
        num_scenarios=args.num_scenarios,
        template_names=[args.template],
        seed=args.seed,
    )

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        allowed = get_valid_action_types(scenario.scene_state, scenario.template_name)
        recent_events = build_recent_events(scenario.scene_state)
        relevant_memory = select_relevant_memory(scenario.scene_state, allowed)

        target = oracle_next_action(scenario.scene_state, allowed)
        target = maybe_dropout_reason(target, args.reason_dropout_prob, rng)

        user_prompt = render_user_prompt(
            scene_state=scenario.scene_state,
            allowed_action_types=allowed,
            recent_events=recent_events,
            relevant_memory=relevant_memory,
        )

        assistant_text = json.dumps(target, ensure_ascii=False, indent=2)

        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
        )

    n = len(rows)
    train_end = int(0.9 * n)
    val_end = int(0.95 * n)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", rows[:train_end])
    write_jsonl(output_dir / "val.jsonl", rows[train_end:val_end])
    write_jsonl(output_dir / "test.jsonl", rows[val_end:])

    print(f"Wrote {train_end} train / {val_end - train_end} val / {n - val_end} test examples to {output_dir}")


if __name__ == "__main__":
    main()