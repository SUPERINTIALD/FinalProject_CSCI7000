from __future__ import annotations

import argparse
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
from app.vlm_client import VLMClient
from app.vlm_perception import VLMPerception


CONTEXT = {}


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
        pass


def build_memory_query(scene_state: dict) -> str:
    parts = [
        scene_state.get("template_name", ""),
        scene_state.get("task_family", ""),
        scene_state.get("user_state", ""),
        scene_state.get("robot_state", ""),
    ]

    for obj in scene_state.get("objects", []):
        parts.append(obj.get("kind", ""))
        parts.append(obj.get("state", ""))
        parts.append(obj.get("location", ""))

    return " ".join(part for part in parts if part).strip()


def setup_context(image_path: str) -> None:
    memory = JsonMemory(settings.memory_path)
    seed_memory(memory)

    vlm_client = VLMClient()
    perception = VLMPerception(vlm_client)

    planner_llm = LMStudioClient()
    planner = Planner(llm=planner_llm, memory=memory)

    parser = ActionParser()

    CONTEXT["image_path"] = image_path
    CONTEXT["memory"] = memory
    CONTEXT["perception"] = perception
    CONTEXT["planner"] = planner
    CONTEXT["parser"] = parser
    CONTEXT["scene_state"] = None
    CONTEXT["sim"] = None
    CONTEXT["last_action"] = None


@tool
def inspect_image_scene() -> str:
    """
    Inspect the kitchen image using the VLM perception model.

    Returns:
        JSON containing raw VLM output and normalized scene_state.
    """
    perception: VLMPerception = CONTEXT["perception"]
    image_path = CONTEXT["image_path"]

    result = perception.perceive_image(image_path)
    scene_state = result["scene_state"]

    CONTEXT["scene_state"] = scene_state
    CONTEXT["sim"] = KitchenSim(scene_state)

    return json.dumps(
        {
            "image_path": image_path,
            "raw_vlm_output": result["raw_vlm_output"],
            "scene_state": scene_state,
        },
        indent=2,
    )


@tool
def plan_proactive_robot_action_from_image_scene() -> str:
    """
    Use the local Qwen planner to choose a proactive semantic robot action
    from the VLM-generated scene_state.

    Returns:
        JSON semantic robot action.
    """
    scene_state = CONTEXT.get("scene_state")
    if scene_state is None:
        return json.dumps({"error": "No scene_state exists yet. Call inspect_image_scene first."}, indent=2)

    planner: Planner = CONTEXT["planner"]

    allowed_actions = get_valid_action_types(
        scene_state,
        scene_state.get("template_name", "dish_cleanup"),
    )

    action = planner.choose_action(
        scene_state=scene_state,
        allowed_actions=allowed_actions,
        memory_query=build_memory_query(scene_state),
        use_memory=True,
    )

    CONTEXT["last_action"] = action

    return json.dumps(
        {
            "allowed_actions": allowed_actions,
            "planner_output": action,
        },
        indent=2,
    )


@tool
def execute_last_robot_action_symbolically() -> str:
    """
    Execute the last planned semantic action in the symbolic Franka Panda kitchen simulator.

    Returns:
        JSON containing semantic action and simulation step result.
    """
    sim: KitchenSim | None = CONTEXT.get("sim")
    parser: ActionParser = CONTEXT["parser"]
    action = CONTEXT.get("last_action")

    if sim is None:
        return json.dumps({"success": False, "message": "No simulator exists yet."}, indent=2)

    if action is None:
        return json.dumps({"success": False, "message": "No planned action exists yet."}, indent=2)

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

    result = sim.step(semantic_action)

    return json.dumps(
        {
            "semantic_action": semantic_action.to_dict(),
            "sim_result": result.to_dict(),
        },
        indent=2,
    )


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--image", required=True)
    args = cli.parse_args()

    setup_context(args.image)

    model = SmolOpenAIModel(
        model_id=settings.lmstudio_model,
        api_base=settings.lmstudio_base_url,
        api_key=settings.lmstudio_api_key,
    )

    agent = ToolCallingAgent(
        model=model,
        tools=[
            inspect_image_scene,
            plan_proactive_robot_action_from_image_scene,
            execute_last_robot_action_symbolically,
        ],
        max_steps=4,
    )

    result = agent.run(
        """
        You are controlling a simulated Franka Panda kitchen robot.
        Do exactly one image-observe, plan, and execute cycle:

        1. Inspect the image scene.
        2. Plan one proactive robot action.
        3. Execute that one action symbolically.
        4. Stop and summarize the result.

        Do not loop.
        """
    )

    print("\n=== SmolAgents VLM image kitchen demo result ===")
    print(result)


if __name__ == "__main__":
    main()