from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict

from app.action_parser import ActionParser
from app.config import settings
from app.evaluator import evaluate_action, get_valid_action_types, summarize_evaluations
from app.kitchen_sim import KitchenSim
from app.llm_client import LMStudioClient
from app.memory import JsonMemory
from app.planner import Planner
from app.scenario_generator import generate_scenarios
from app.scene_text import scene_to_text


TASKS = ["dish_cleanup", "trash_cleanup", "counter_cleanup"]


def seed_memory(memory: JsonMemory) -> None:
   

    memory.add("preference", "User prefers clean dishes on the left counter.", {"task": "dish_cleanup"})
    memory.add("strategy", "Start collecting dishes from the right side first.", {"task": "dish_cleanup"})
    memory.add("failure", "Move slower when picking up mugs or cups.", {"task": "dish_cleanup"})

    memory.add("strategy", "Trash on the floor should be picked up and placed in the trash bin.", {"task": "trash_cleanup"})
    memory.add("preference", "User prefers the kitchen floor to stay clear of trash.", {"task": "trash_cleanup"})

    memory.add("preference", "User prefers used cups and utensils placed in the sink.", {"task": "counter_cleanup"})
    memory.add("failure", "Cups can slip if moved too quickly.", {"task": "counter_cleanup"})


def main() -> None:
    memory = JsonMemory(settings.memory_path)
    seed_memory(memory)

    llm = LMStudioClient()
    planner = Planner(llm=llm, memory=memory)
    parser = ActionParser()

    scenarios = generate_scenarios(
        num_scenarios=50,
        template_names=TASKS,
        seed=42,
    )

    eval_results = []
    decision_times = []
    execution_successes = []
    by_task = defaultdict(list)

    for scenario in scenarios:
        sim = KitchenSim(scenario.scene_state)
        current_scene = sim.get_scene_state()

        allowed_actions = get_valid_action_types(
            current_scene,
            scenario.template_name,
        )

        scene_description = scene_to_text(current_scene)
        print(f"\n=== {scenario.scenario_id} | {scenario.template_name} ===")
        print(scene_description)

        start = time.perf_counter()

        planner_output = planner.choose_action(
            scene_state=current_scene,
            allowed_actions=allowed_actions,
            memory_query=scenario.memory_query,
            use_memory=True,
        )

        decision_time = time.perf_counter() - start
        decision_times.append(decision_time)

        raw = planner_output.get("raw_model_output") or json.dumps(planner_output)
        semantic_action = parser.parse(raw_text=raw, scene_state=current_scene)

        sim_result = sim.step(semantic_action)
        execution_successes.append(1 if sim_result.success else 0)

        eval_result = evaluate_action(
            scenario_id=scenario.scenario_id,
            template_name=scenario.template_name,
            scene_state=current_scene,
            action=semantic_action,
            evaluation_hints=scenario.evaluation_hints,
        )

        eval_results.append(eval_result)
        by_task[scenario.template_name].append(eval_result)

        print("\nAction:")
        print(json.dumps(semantic_action.to_dict(), indent=2))

        print("\nSimulation step:")
        print(json.dumps(sim_result.to_dict(), indent=2))

        print("\nEvaluation:")
        print(json.dumps(eval_result.to_dict(), indent=2))

    summary = summarize_evaluations(eval_results)
    summary["avg_decision_time_sec"] = round(statistics.mean(decision_times), 4)
    summary["median_decision_time_sec"] = round(statistics.median(decision_times), 4)
    summary["execution_success_rate"] = round(sum(execution_successes) / len(execution_successes), 4)

    print("\n=== Overall Agentic Kitchen Benchmark Summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Per-task summaries ===")
    for task_name, task_results in by_task.items():
        print(f"\n[{task_name}]")
        print(json.dumps(summarize_evaluations(task_results), indent=2))


if __name__ == "__main__":
    main().__annotations__


