from __future__ import annotations

import json
import traceback

from app.config import settings
from app.llm_client import LMStudioClient
from app.memory import JsonMemory
from app.planner import Planner
from app.action_parser import ActionParser
from app.scenario_generator import generate_scenarios
from app.evaluator import evaluate_action, summarize_evaluations


def seed_memory(memory: JsonMemory) -> None:
    if memory.all():
        return

    memory.add("preference", "User prefers clean dishes on the left counter.", {"task": "dish_cleanup"})
    memory.add("strategy", "Start collecting dishes from the right side first.", {"task": "dish_cleanup"})
    memory.add("failure", "Move slower when picking up mugs or cups.", {"task": "dish_cleanup"})


def main() -> None:
    print("[debug] main started", flush=True)

    memory = JsonMemory(settings.memory_path)
    seed_memory(memory)

    llm = LMStudioClient()
    planner = Planner(llm=llm, memory=memory)

    scenarios = generate_scenarios(
        num_scenarios=50,
        template_names=["dish_cleanup"],
        seed=42,
    )

    parser = ActionParser()
    eval_results = []

    for scenario in scenarios:
        print(f"\n[scenario] {scenario.scenario_id}", flush=True)
        from app.evaluator import get_valid_action_types

        allowed_actions = get_valid_action_types(
            scenario.scene_state,
            scenario.template_name,
        )

        planner_output = planner.choose_action(
            scene_state=scenario.scene_state,
            allowed_actions=allowed_actions,
            memory_query=scenario.memory_query,
            use_memory=True,
        )
        # planner_output = planner.choose_action(
        #     scene_state=scenario.scene_state,
        #     allowed_actions=[
        #         "wait",
        #         "start_task",
        #         "pick",
        #         "place",
        #         "clean_surface",
        #         "inspect",
        #     ],
        #     memory_query=scenario.memory_query,
        #     use_memory=True,
        # )
        print("\n[planner output]")
        print(json.dumps(planner_output, indent=2))

        semantic_action = parser.parse(
            raw_text=planner_output["raw_model_output"],
            scene_state=scenario.scene_state,
        )

        print("\n[parsed semantic action]")
        print(json.dumps(semantic_action.to_dict(), indent=2))
        semantic_action = parser.parse(
            raw_text=planner_output["raw_model_output"],
            scene_state=scenario.scene_state,
        )

        result = evaluate_action(
            scenario_id=scenario.scenario_id,
            template_name=scenario.template_name,
            scene_state=scenario.scene_state,
            action=semantic_action,
            evaluation_hints=scenario.evaluation_hints,
        )

        eval_results.append(result)

        print(json.dumps({
            "scenario_id": scenario.scenario_id,
            "scene_state": scenario.scene_state,
            "semantic_action": semantic_action.to_dict(),
            "evaluation": result.to_dict(),
        }, indent=2), flush=True)

    summary = summarize_evaluations(eval_results)

    print("\n=== Evaluation Summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[debug] exception occurred", flush=True)
        traceback.print_exc()