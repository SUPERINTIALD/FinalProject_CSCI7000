# from __future__ import annotations

# import json

# from smolagents import ToolCallingAgent, tool

# try:
#     from smolagents import OpenAIModel as SmolOpenAIModel
# except ImportError:
#     from smolagents import OpenAIServerModel as SmolOpenAIModel

# from app.action_parser import ActionParser
# from app.config import settings
# from app.evaluator import get_valid_action_types
# from app.kitchen_sim import KitchenSim
# from app.llm_client import LMStudioClient
# from app.memory import JsonMemory
# from app.planner import Planner
# from app.scenario_generator import generate_scenarios
# from app.scene_text import scene_to_text


# CONTEXT = {}


# def setup_context() -> None:
#     memory = JsonMemory(settings.memory_path)

#     llm = LMStudioClient()
#     planner = Planner(llm=llm, memory=memory)
#     parser = ActionParser()

#     scenario = generate_scenarios(
#         num_scenarios=1,
#         template_names=["dish_cleanup", "trash_cleanup", "counter_cleanup"],
#         seed=7,
#     )[0]

#     sim = KitchenSim(scenario.scene_state)

#     CONTEXT["scenario"] = scenario
#     CONTEXT["sim"] = sim
#     CONTEXT["planner"] = planner
#     CONTEXT["parser"] = parser
#     CONTEXT["last_action"] = None


# @tool
# def inspect_kitchen_scene() -> str:
#     """
#     Inspect the current simulated kitchen scene.

#     Returns:
#         A text description and JSON state of the current kitchen scene.
#     """
#     sim: KitchenSim = CONTEXT["sim"]
#     scene = sim.get_scene_state()

#     return json.dumps(
#         {
#             "scene_text": scene_to_text(scene),
#             "scene_state": scene,
#         },
#         indent=2,
#     )


# @tool
# def plan_proactive_robot_action() -> str:
#     """
#     Choose the next proactive robot action for the current kitchen scene.

#     Returns:
#         A JSON semantic robot action selected by the local Qwen planner.
#     """
#     scenario = CONTEXT["scenario"]
#     sim: KitchenSim = CONTEXT["sim"]
#     planner: Planner = CONTEXT["planner"]

#     scene = sim.get_scene_state()
#     allowed_actions = get_valid_action_types(scene, scenario.template_name)

#     action = planner.choose_action(
#         scene_state=scene,
#         allowed_actions=allowed_actions,
#         memory_query=scenario.memory_query,
#         use_memory=True,
#     )

#     CONTEXT["last_action"] = action
#     return json.dumps(action, indent=2)


# @tool
# def execute_last_robot_action() -> str:
#     """
#     Execute the last planned robot action in the symbolic Franka Panda kitchen simulator.

#     Returns:
#         A JSON result containing whether the simulated robot skill succeeded.
#     """
#     sim: KitchenSim = CONTEXT["sim"]
#     parser: ActionParser = CONTEXT["parser"]
#     action = CONTEXT.get("last_action")

#     if action is None:
#         return json.dumps({"success": False, "message": "No planned action exists yet."}, indent=2)

#     raw = action.get("raw_model_output") or json.dumps(action)
#     semantic_action = parser.parse(raw_text=raw, scene_state=sim.get_scene_state())
#     result = sim.step(semantic_action)

#     return json.dumps(
#         {
#             "semantic_action": semantic_action.to_dict(),
#             "sim_result": result.to_dict(),
#         },
#         indent=2,
#     )


# def main() -> None:
#     setup_context()

#     model = SmolOpenAIModel(
#         model_id=settings.lmstudio_model,
#         api_base=settings.lmstudio_base_url,
#         api_key=settings.lmstudio_api_key,
#     )

#     # agent = ToolCallingAgent(
#     #     model=model,
#     #     tools=[
#     #         inspect_kitchen_scene,
#     #         plan_proactive_robot_action,
#     #         execute_last_robot_action,
#     #     ],
#     # )
#     agent = ToolCallingAgent(
#         model=model,
#         tools=[
#             inspect_kitchen_scene,
#             plan_proactive_robot_action,
#             execute_last_robot_action,
#         ],
#         max_steps=3,
#     )

#     result = agent.run(
#         """
#         You are controlling a simulated Franka Panda kitchen robot.
#         Do exactly one inspect-plan-execute cycle:
#         1. Inspect the scene.
#         2. Plan one proactive robot action.
#         3. Execute that one action.
#         Then stop and summarize the result.
#         """
#     )

#     print("\n=== SmolAgents kitchen demo result ===")
#     print(result)


# if __name__ == "__main__":
#     main()


# from __future__ import annotations

# import json

# from smolagents import ToolCallingAgent, tool

# try:
#     from smolagents import OpenAIModel as SmolOpenAIModel
# except ImportError:
#     from smolagents import OpenAIServerModel as SmolOpenAIModel

# from app.action_parser import ActionParser
# from app.config import settings
# from app.evaluator import get_valid_action_types
# from app.kitchen_sim import KitchenSim
# from app.llm_client import LMStudioClient
# from app.memory import JsonMemory
# from app.planner import Planner
# from app.scenario_generator import generate_scenarios
# from app.scene_text import scene_to_text


# CONTEXT = {}


# def setup_context() -> None:
#     memory = JsonMemory(settings.memory_path)

#     llm = LMStudioClient()
#     planner = Planner(llm=llm, memory=memory)
#     parser = ActionParser()

#     scenario = generate_scenarios(
#         num_scenarios=1,
#         template_names=["dish_cleanup", "trash_cleanup", "counter_cleanup"],
#         seed=7,
#     )[0]

#     sim = KitchenSim(scenario.scene_state)

#     CONTEXT["scenario"] = scenario
#     CONTEXT["sim"] = sim
#     CONTEXT["planner"] = planner
#     CONTEXT["parser"] = parser
#     CONTEXT["last_action"] = None


# @tool
# def inspect_kitchen_scene() -> str:
#     """
#     Inspect the current simulated kitchen scene.

#     Returns:
#         A text description and JSON state of the current kitchen scene.
#     """
#     sim: KitchenSim = CONTEXT["sim"]
#     scene = sim.get_scene_state()

#     return json.dumps(
#         {
#             "scene_text": scene_to_text(scene),
#             "scene_state": scene,
#         },
#         indent=2,
#     )


# @tool
# def plan_proactive_robot_action() -> str:
#     """
#     Choose the next proactive robot action for the current kitchen scene.

#     Returns:
#         A JSON semantic robot action selected by the local Qwen planner.
#     """
#     scenario = CONTEXT["scenario"]
#     sim: KitchenSim = CONTEXT["sim"]
#     planner: Planner = CONTEXT["planner"]

#     scene = sim.get_scene_state()
#     allowed_actions = get_valid_action_types(scene, scenario.template_name)

#     action = planner.choose_action(
#         scene_state=scene,
#         allowed_actions=allowed_actions,
#         memory_query=scenario.memory_query,
#         use_memory=True,
#     )

#     CONTEXT["last_action"] = action
#     return json.dumps(action, indent=2)


# @tool
# def execute_last_robot_action() -> str:
#     """
#     Execute the last planned robot action in the symbolic Franka Panda kitchen simulator.

#     Returns:
#         A JSON result containing whether the simulated robot skill succeeded.
#     """
#     sim: KitchenSim = CONTEXT["sim"]
#     parser: ActionParser = CONTEXT["parser"]
#     action = CONTEXT.get("last_action")

#     if action is None:
#         return json.dumps({"success": False, "message": "No planned action exists yet."}, indent=2)

#     raw = action.get("raw_model_output") or json.dumps(action)
#     semantic_action = parser.parse(raw_text=raw, scene_state=sim.get_scene_state())
#     result = sim.step(semantic_action)

#     return json.dumps(
#         {
#             "semantic_action": semantic_action.to_dict(),
#             "sim_result": result.to_dict(),
#         },
#         indent=2,
#     )


# def main() -> None:
#     setup_context()

#     model = SmolOpenAIModel(
#         model_id=settings.lmstudio_model,
#         api_base=settings.lmstudio_base_url,
#         api_key=settings.lmstudio_api_key,
#     )

#     agent = ToolCallingAgent(
#         model=model,
#         tools=[
#             inspect_kitchen_scene,
#             plan_proactive_robot_action,
#             execute_last_robot_action,
#         ],
#     )

#     result = agent.run(
#         """
#         You are controlling a simulated Franka Panda kitchen robot.
#         Inspect the kitchen scene, decide whether the robot should proactively act,
#         then execute the planned action if appropriate.
#         Return a short final summary of what happened.
#         """
#     )

#     print("\n=== SmolAgents kitchen demo result ===")
#     print(result)


# if __name__ == "__main__":
#     main()



from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from smolagents import ToolCallingAgent, tool

try:
    from smolagents import OpenAIModel as SmolOpenAIModel
except ImportError:
    from smolagents import OpenAIServerModel as SmolOpenAIModel

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
CONTEXT: dict[str, Any] = {}


def seed_memory(memory: JsonMemory) -> None:
    try:
        memory.upsert("preference", "User prefers clean dishes on the left counter.", {"task": "dish_cleanup"})
        memory.upsert("strategy", "Start collecting dishes from the right side first.", {"task": "dish_cleanup"})
        memory.upsert("failure", "Move slower when picking up mugs or cups.", {"task": "dish_cleanup"})

        memory.upsert(
            "strategy",
            "Trash on the floor or near the counter should be picked up and placed in the trash bin.",
            {"task": "trash_cleanup"},
        )
        memory.upsert(
            "preference",
            "User prefers the kitchen floor and counter area to stay clear of trash.",
            {"task": "trash_cleanup"},
        )

        memory.upsert("preference", "User prefers used cups and utensils placed in the sink.", {"task": "counter_cleanup"})
        memory.upsert("failure", "Cups can slip if moved too quickly.", {"task": "counter_cleanup"})
    except AttributeError:
        # Backward compatible with older JsonMemory without upsert()
        if memory.all():
            return
        memory.add("preference", "User prefers clean dishes on the left counter.", {"task": "dish_cleanup"})
        memory.add("strategy", "Start collecting dishes from the right side first.", {"task": "dish_cleanup"})
        memory.add("failure", "Move slower when picking up mugs or cups.", {"task": "dish_cleanup"})
        memory.add("strategy", "Trash should be picked up and placed in the trash bin.", {"task": "trash_cleanup"})
        memory.add("preference", "User prefers used cups and utensils placed in the sink.", {"task": "counter_cleanup"})


def setup_case_context(scenario, planner: Planner, parser: ActionParser) -> None:
    sim = KitchenSim(scenario.scene_state)

    CONTEXT.clear()
    CONTEXT["scenario"] = scenario
    CONTEXT["sim"] = sim
    CONTEXT["planner"] = planner
    CONTEXT["parser"] = parser
    CONTEXT["last_action"] = None
    CONTEXT["semantic_action"] = None
    CONTEXT["sim_result"] = None
    CONTEXT["allowed_actions"] = []
    CONTEXT["decision_time_sec"] = 0.0
    CONTEXT["execution_time_sec"] = 0.0


@tool
def inspect_kitchen_scene() -> str:
    """
    Inspect the current simulated kitchen scene.

    Returns:
        A text description and JSON state of the current kitchen scene.
    """
    sim: KitchenSim = CONTEXT["sim"]
    scene = sim.get_scene_state()

    return json.dumps(
        {
            "scene_text": scene_to_text(scene),
            "scene_state": scene,
        },
        indent=2,
    )


@tool
def plan_proactive_robot_action() -> str:
    """
    Choose the next proactive robot action for the current kitchen scene.

    Returns:
        A JSON semantic robot action selected by the local Qwen planner.
    """
    scenario = CONTEXT["scenario"]
    sim: KitchenSim = CONTEXT["sim"]
    planner: Planner = CONTEXT["planner"]

    scene = sim.get_scene_state()
    allowed_actions = get_valid_action_types(scene, scenario.template_name)

    start = time.perf_counter()
    action = planner.choose_action(
        scene_state=scene,
        allowed_actions=allowed_actions,
        memory_query=scenario.memory_query,
        use_memory=True,
    )
    CONTEXT["decision_time_sec"] = time.perf_counter() - start

    CONTEXT["last_action"] = action
    CONTEXT["allowed_actions"] = allowed_actions

    return json.dumps(
        {
            "allowed_actions": allowed_actions,
            "planner_output": action,
        },
        indent=2,
    )


@tool
def execute_last_robot_action() -> str:
    """
    Execute the last planned robot action in the symbolic Franka Panda kitchen simulator.

    Returns:
        A JSON result containing whether the simulated robot skill succeeded.
    """
    sim: KitchenSim = CONTEXT["sim"]
    parser: ActionParser = CONTEXT["parser"]
    action = CONTEXT.get("last_action")

    if action is None:
        return json.dumps({"success": False, "message": "No planned action exists yet."}, indent=2)

    # Use safe planner output, not raw model output, because Planner may have corrected invalid raw output.
    safe_action_for_parse = {
        "action_type": action.get("action_type"),
        "target_object": action.get("target_object"),
        "target_surface": action.get("target_surface"),
        "target_zone": action.get("target_zone"),
        "parameters": action.get("parameters", {}),
        "reason": action.get("reason", ""),
        "memory_used": action.get("memory_used", []),
        "confidence": action.get("confidence", 0.0),
    }

    semantic_action = parser.parse(
        raw_text=json.dumps(safe_action_for_parse),
        scene_state=sim.get_scene_state(),
    )

    start = time.perf_counter()
    result = sim.step(semantic_action)
    CONTEXT["execution_time_sec"] = time.perf_counter() - start

    CONTEXT["semantic_action"] = semantic_action
    CONTEXT["sim_result"] = result

    return json.dumps(
        {
            "semantic_action": semantic_action.to_dict(),
            "sim_result": result.to_dict(),
        },
        indent=2,
    )


def make_agent() -> ToolCallingAgent:
    model = SmolOpenAIModel(
        model_id=settings.lmstudio_model,
        api_base=settings.lmstudio_base_url,
        api_key=settings.lmstudio_api_key,
    )

    return ToolCallingAgent(
        model=model,
        tools=[
            inspect_kitchen_scene,
            plan_proactive_robot_action,
            execute_last_robot_action,
        ],
        max_steps=4,
    )


def run_agent_once(agent: ToolCallingAgent) -> tuple[str, str | None, float]:
    prompt = """
You are controlling a simulated Franka Panda kitchen robot.

Perform exactly one observe-plan-execute cycle by calling tools in this order:
1. inspect_kitchen_scene
2. plan_proactive_robot_action
3. execute_last_robot_action

After the third tool call, stop and summarize the result.
Do not call any tool more than once.
Do not loop.
"""
    start = time.perf_counter()
    try:
        result = agent.run(prompt)
        return str(result), None, time.perf_counter() - start
    except Exception as exc:
        return "", str(exc), time.perf_counter() - start


def recover_missing_steps() -> None:
    """
    If the agent stops early, run missing tool steps directly so the benchmark can still score the case.
    """
    if CONTEXT.get("last_action") is None:
        plan_proactive_robot_action.forward()

    if CONTEXT.get("semantic_action") is None:
        execute_last_robot_action.forward()


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--num-scenarios", type=int, default=50)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--results-dir", default=None)
    cli.add_argument("--print-each", action="store_true")
    args = cli.parse_args()

    memory = JsonMemory(settings.memory_path)
    seed_memory(memory)

    llm = LMStudioClient()
    planner = Planner(llm=llm, memory=memory)
    parser = ActionParser()

    scenarios = generate_scenarios(
        num_scenarios=args.num_scenarios,
        template_names=TASKS,
        seed=args.seed,
    )

    agent = make_agent()

    eval_results = []
    execution_successes = []
    decision_times = []
    agent_times = []
    by_task = defaultdict(list)
    agent_errors = []

    results_dir = Path(args.results_dir or "results/smolagents_kitchen_demo_benchmark")
    results_dir.mkdir(parents=True, exist_ok=True)

    for i, scenario in enumerate(scenarios, start=1):
        setup_case_context(scenario, planner, parser)

        agent_result, agent_error, agent_time = run_agent_once(agent)

        try:
            recover_missing_steps()
        except Exception as exc:
            if agent_error is None:
                agent_error = f"Recovery failed: {exc}"

        semantic_action = CONTEXT.get("semantic_action")
        sim_result = CONTEXT.get("sim_result")

        if semantic_action is None:
            # Make a safe fallback action so evaluator can still run.
            semantic_action = parser.parse(
                raw_text=json.dumps(
                    {
                        "action_type": "inspect",
                        "target_object": None,
                        "target_surface": None,
                        "target_zone": None,
                        "parameters": {},
                        "reason": "Fallback: agent failed to produce action.",
                        "memory_used": [],
                        "confidence": 0.0,
                    }
                ),
                scene_state=scenario.scene_state,
            )

        eval_result = evaluate_action(
            scenario_id=scenario.scenario_id,
            template_name=scenario.template_name,
            scene_state=scenario.scene_state,
            action=semantic_action,
            evaluation_hints=scenario.evaluation_hints,
        )

        eval_results.append(eval_result)
        by_task[scenario.template_name].append(eval_result)

        success = bool(sim_result.success) if sim_result is not None else False
        execution_successes.append(1 if success else 0)
        decision_times.append(CONTEXT.get("decision_time_sec", 0.0))
        agent_times.append(agent_time)

        if agent_error is not None:
            agent_errors.append({"scenario_id": scenario.scenario_id, "error": agent_error})

        detail = {
            "scenario_id": scenario.scenario_id,
            "template_name": scenario.template_name,
            "scene_state": scenario.scene_state,
            "allowed_actions": CONTEXT.get("allowed_actions", []),
            "planner_output": CONTEXT.get("last_action"),
            "semantic_action": semantic_action.to_dict(),
            "sim_result": sim_result.to_dict() if sim_result is not None else None,
            "evaluation": eval_result.to_dict(),
            "agent_result": agent_result,
            "agent_error": agent_error,
            "decision_time_sec": CONTEXT.get("decision_time_sec", 0.0),
            "agent_time_sec": agent_time,
        }

        detail_path = results_dir / "details" / f"{scenario.scenario_id}.json"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")

        if args.print_each:
            print(f"\n[{i}/{len(scenarios)}] {scenario.scenario_id} | {scenario.template_name}")
            print(json.dumps(detail, indent=2))

    summary = summarize_evaluations(eval_results)
    summary["avg_decision_time_sec"] = round(statistics.mean(decision_times), 4)
    summary["median_decision_time_sec"] = round(statistics.median(decision_times), 4)
    summary["avg_agent_time_sec"] = round(statistics.mean(agent_times), 4)
    summary["execution_success_rate"] = round(sum(execution_successes) / len(execution_successes), 4)
    summary["agent_error_rate"] = round(len(agent_errors) / len(scenarios), 4)

    per_task = {}
    for task_name, task_results in by_task.items():
        per_task[task_name] = summarize_evaluations(task_results)

    output = {
        "overall": summary,
        "per_task": per_task,
        "agent_errors": agent_errors,
    }

    (results_dir / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n=== SmolAgents Kitchen Benchmark Summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Per-task summaries ===")
    print(json.dumps(per_task, indent=2))

    print("\nSaved:")
    print(results_dir / "summary.json")
    print(results_dir / "details")


if __name__ == "__main__":
    main()

    # python .\smolagents_kitchen_demo.py --num-scenarios 50 --seed 42