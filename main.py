from __future__ import annotations

import json
import traceback
from dataclasses import dataclass

from app.config import settings
from app.llm_client import LMStudioClient
from app.memory import JsonMemory
from app.planner import Planner


@dataclass
class Scenario:
    name: str
    scene_state: dict
    allowed_actions: list[str]
    memory_query: str
    expected_action: str


def seed_memory(memory: JsonMemory) -> None:
    print("[debug] seeding memory check...", flush=True)

    if memory.all():
        print("[debug] memory already exists, skipping seed", flush=True)
        return

    memory.add(
        kind="preference",
        text="User prefers clean dishes placed on the left side of the counter.",
        metadata={"task": "dish_cleanup"},
    )
    memory.add(
        kind="strategy",
        text="When clearing dishes, start from the right side first to avoid crossing over cleaned space.",
        metadata={"task": "dish_cleanup"},
    )
    memory.add(
        kind="failure",
        text="A mug slipped during a fast grasp. Move slower on cups and mugs.",
        metadata={"task": "dish_cleanup"},
    )
    print("[debug] seeded memory", flush=True)


def build_scenarios() -> list[Scenario]:
    allowed_actions = [
        "start_cleanup",
        "pick_dish_right_side",
        "place_clean_left_side",
        "wait",
    ]

    return [
        Scenario(
            name="proactive cleanup trigger",
            scene_state={
                "task_context": "dish_cleanup",
                "user_done_eating": True,
                "dirty_dishes_visible": True,
                "clean_counter_left_available": True,
                "robot_state": "idle",
                "last_action_result": "none",
            },
            allowed_actions=allowed_actions,
            memory_query="dish cleanup user preference clean dishes left side right side start first",
            expected_action="start_cleanup",
        ),
        Scenario(
            name="user still eating",
            scene_state={
                "task_context": "dish_cleanup",
                "user_done_eating": False,
                "dirty_dishes_visible": True,
                "clean_counter_left_available": True,
                "robot_state": "idle",
                "last_action_result": "none",
            },
            allowed_actions=allowed_actions,
            memory_query="dish cleanup wait until user finished eating",
            expected_action="wait",
        ),
        Scenario(
            name="already cleaning, next pick",
            scene_state={
                "task_context": "dish_cleanup",
                "user_done_eating": True,
                "dirty_dishes_visible": True,
                "clean_counter_left_available": True,
                "robot_state": "active",
                "last_action_result": "cleanup_started",
            },
            allowed_actions=allowed_actions,
            memory_query="dish cleanup start from right side first",
            expected_action="pick_dish_right_side",
        ),
        Scenario(
            name="nothing to clean",
            scene_state={
                "task_context": "dish_cleanup",
                "user_done_eating": True,
                "dirty_dishes_visible": False,
                "clean_counter_left_available": True,
                "robot_state": "idle",
                "last_action_result": "none",
            },
            allowed_actions=allowed_actions,
            memory_query="dish cleanup no dirty dishes visible",
            expected_action="wait",
        ),
        Scenario(
            name="object already in hand, place by preference",
            scene_state={
                "task_context": "dish_cleanup",
                "user_done_eating": True,
                "dirty_dishes_visible": True,
                "clean_counter_left_available": True,
                "robot_state": "holding_clean_dish",
                "last_action_result": "picked_dish_successfully",
            },
            allowed_actions=allowed_actions,
            memory_query="user prefers clean dishes on left side of counter",
            expected_action="place_clean_left_side",
        ),
    ]

def run_pass(
    planner: Planner,
    scenarios: list[Scenario],
    use_memory: bool,
) -> list[dict]:
    results = []

    print(f"\n=== Running scenarios | use_memory={use_memory} ===", flush=True)

    for idx, scenario in enumerate(scenarios, start=1):
        print(f"\n[{idx}] {scenario.name}", flush=True)

        allowed_actions = get_allowed_actions(scenario.scene_state)

        decision = planner.choose_action(
            scene_state=scenario.scene_state,
            allowed_actions=allowed_actions,
            memory_query=scenario.memory_query,
            use_memory=use_memory,
        )

        correct = decision["action"] == scenario.expected_action

        result = {
            "scenario": scenario.name,
            "allowed_actions": allowed_actions,
            "expected_action": scenario.expected_action,
            "predicted_action": decision["action"],
            "correct": correct,
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "memory_used": decision.get("memory_used", []),
            "retrieved_memory": decision.get("retrieved_memory", []),
            "raw_model_output": decision.get("raw_model_output", ""),
        }

        results.append(result)

        print(json.dumps(result, indent=2), flush=True)

    return results
# def run_pass(
#     planner: Planner,
#     scenarios: list[Scenario],
#     use_memory: bool,
# ) -> list[dict]:
#     results = []

#     print(f"\n=== Running scenarios | use_memory={use_memory} ===", flush=True)

#     for idx, scenario in enumerate(scenarios, start=1):
#         print(f"\n[{idx}] {scenario.name}", flush=True)

#         decision = planner.choose_action(
#             scene_state=scenario.scene_state,
#             allowed_actions=scenario.allowed_actions,
#             memory_query=scenario.memory_query,
#             use_memory=use_memory,
#         )

#         correct = decision["action"] == scenario.expected_action

#         result = {
#             "scenario": scenario.name,
#             "expected_action": scenario.expected_action,
#             "predicted_action": decision["action"],
#             "correct": correct,
#             "confidence": decision["confidence"],
#             "reason": decision["reason"],
#             "memory_used": decision.get("memory_used", []),
#             "retrieved_memory": decision.get("retrieved_memory", []),
#             "raw_model_output": decision.get("raw_model_output", ""),
#         }

#         results.append(result)

#         print(json.dumps(result, indent=2), flush=True)

#     return results


def summarize_results(results: list[dict], label: str) -> None:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total else 0.0

    print(f"\n=== Summary: {label} ===", flush=True)
    print(f"Correct: {correct}/{total}", flush=True)
    print(f"Accuracy: {accuracy:.2%}", flush=True)


def log_results(memory: JsonMemory, results: list[dict], label: str) -> None:
    for result in results:
        memory.add(
            kind="evaluation_result",
            text=(
                f"{label} | Scenario='{result['scenario']}' | "
                f"Predicted='{result['predicted_action']}' | "
                f"Expected='{result['expected_action']}' | "
                f"Correct={result['correct']}"
            ),
            metadata=result,
        )

def get_allowed_actions(scene_state: dict) -> list[str]:
    if not scene_state["user_done_eating"]:
        return ["wait"]

    if scene_state["robot_state"] == "holding_clean_dish":
        return ["place_clean_left_side", "wait"]

    if scene_state["last_action_result"] == "cleanup_started" and scene_state["dirty_dishes_visible"]:
        return ["pick_dish_right_side", "wait"]

    if scene_state["dirty_dishes_visible"] and scene_state["robot_state"] == "idle":
        return ["start_cleanup", "wait"]

    return ["wait"]
def main() -> None:
    print("[debug] main started", flush=True)

    memory = JsonMemory(settings.memory_path)
    print(f"[debug] memory path: {settings.memory_path}", flush=True)

    seed_memory(memory)

    llm = LMStudioClient()
    print("[debug] llm client created", flush=True)

    planner = Planner(llm=llm, memory=memory)
    print("[debug] planner created", flush=True)

    scenarios = build_scenarios()
    print(f"[debug] loaded {len(scenarios)} scenarios", flush=True)

    results_with_memory = run_pass(planner, scenarios, use_memory=True)
    summarize_results(results_with_memory, "with_memory")
    log_results(memory, results_with_memory, "with_memory")

    results_without_memory = run_pass(planner, scenarios, use_memory=False)
    summarize_results(results_without_memory, "without_memory")
    log_results(memory, results_without_memory, "without_memory")

    print("\n[debug] evaluation complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[debug] exception occurred", flush=True)
        traceback.print_exc()