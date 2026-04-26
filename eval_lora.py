from __future__ import annotations

import json

from app.memory import JsonMemory
from app.planner import Planner
from app.action_parser import ActionParser
from app.scenario_generator import generate_scenarios
from app.evaluator import evaluate_action, summarize_evaluations, get_valid_action_types
from app.hf_lora_client import HFLoraClient


def main() -> None:
    memory = JsonMemory("data/memory.json")

    llm = HFLoraClient(
        base_model_name="Qwen/Qwen3.5-0.8B-Base",
        # adapter_path="outputs/qwen35-08b-phase-d-lora-small",
        adapter_path="outputs/qwen35-08b-phase-d-lora",
    )

    planner = Planner(llm=llm, memory=memory)
    parser = ActionParser()

    scenarios = generate_scenarios(
        num_scenarios=50,
        template_names=["dish_cleanup"],
        seed=42,
    )

    results = []

    for scenario in scenarios:
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

        results.append(result)

    summary = summarize_evaluations(results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()