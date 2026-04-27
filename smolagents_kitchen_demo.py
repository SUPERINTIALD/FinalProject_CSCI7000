from __future__ import annotations

import json

from smolagents import ToolCallingAgent, tool

try:
    from smolagents import OpenAIModel as SmolOpenAIModel
except ImportError:
    from smolagents import OpenAIServerModel as SmolOpenAIModel

from app.action_parser import ActionParser
from app.config import settings
from app.evaluator import get_valid_action_types
from app.kitchen_sim import KitchenSim
from app.llm_client import LMStudioClient
from app.memory import JsonMemory
from app.planner import Planner
from app.scenario_generator import generate_scenarios
from app.scene_text import scene_to_text


CONTEXT = {}


def setup_context() -> None:
    memory = JsonMemory(settings.memory_path)

    llm = LMStudioClient()
    planner = Planner(llm=llm, memory=memory)
    parser = ActionParser()

    scenario = generate_scenarios(
        num_scenarios=1,
        template_names=["dish_cleanup", "trash_cleanup", "counter_cleanup"],
        seed=7,
    )[0]

    sim = KitchenSim(scenario.scene_state)

    CONTEXT["scenario"] = scenario
    CONTEXT["sim"] = sim
    CONTEXT["planner"] = planner
    CONTEXT["parser"] = parser
    CONTEXT["last_action"] = None


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

    action = planner.choose_action(
        scene_state=scene,
        allowed_actions=allowed_actions,
        memory_query=scenario.memory_query,
        use_memory=True,
    )

    CONTEXT["last_action"] = action
    return json.dumps(action, indent=2)


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

    raw = action.get("raw_model_output") or json.dumps(action)
    semantic_action = parser.parse(raw_text=raw, scene_state=sim.get_scene_state())
    result = sim.step(semantic_action)

    return json.dumps(
        {
            "semantic_action": semantic_action.to_dict(),
            "sim_result": result.to_dict(),
        },
        indent=2,
    )


def main() -> None:
    setup_context()

    model = SmolOpenAIModel(
        model_id=settings.lmstudio_model,
        api_base=settings.lmstudio_base_url,
        api_key=settings.lmstudio_api_key,
    )

    # agent = ToolCallingAgent(
    #     model=model,
    #     tools=[
    #         inspect_kitchen_scene,
    #         plan_proactive_robot_action,
    #         execute_last_robot_action,
    #     ],
    # )
    agent = ToolCallingAgent(
        model=model,
        tools=[
            inspect_kitchen_scene,
            plan_proactive_robot_action,
            execute_last_robot_action,
        ],
        max_steps=3,
    )

    result = agent.run(
        """
        You are controlling a simulated Franka Panda kitchen robot.
        Do exactly one inspect-plan-execute cycle:
        1. Inspect the scene.
        2. Plan one proactive robot action.
        3. Execute that one action.
        Then stop and summarize the result.
        """
    )

    print("\n=== SmolAgents kitchen demo result ===")
    print(result)


if __name__ == "__main__":
    main()