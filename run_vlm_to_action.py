from __future__ import annotations

import argparse
import json

from app.action_parser import ActionParser
from app.config import settings
from app.evaluator import get_valid_action_types
from app.llm_client import LMStudioClient
from app.memory import JsonMemory
from app.planner import Planner
from app.vlm_client import VLMClient
from app.vlm_perception import VLMPerception


def seed_memory(memory: JsonMemory) -> None:
    try:
        memory.upsert("preference", "User prefers clean dishes on the left counter.", {"task": "dish_cleanup"})
        memory.upsert("strategy", "Start collecting dishes from the right side first.", {"task": "dish_cleanup"})
        memory.upsert("failure", "Move slower when picking up mugs or cups.", {"task": "dish_cleanup"})

        memory.upsert("strategy", "Trash on the floor or near the counter should be picked up and placed in the trash bin.", {"task": "trash_cleanup"})
        memory.upsert("preference", "User prefers the kitchen floor and counter area to stay clear of trash.", {"task": "trash_cleanup"})

        memory.upsert("preference", "User prefers used cups and utensils placed in the sink.", {"task": "counter_cleanup"})
        memory.upsert("failure", "Cups can slip if moved too quickly.", {"task": "counter_cleanup"})
    except AttributeError:
        if memory.all():
            return
        memory.add("preference", "User prefers clean dishes on the left counter.", {"task": "dish_cleanup"})
        memory.add("strategy", "Start collecting dishes from the right side first.", {"task": "dish_cleanup"})
        memory.add("failure", "Move slower when picking up mugs or cups.", {"task": "dish_cleanup"})


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to kitchen image or simulation screenshot.")
    parser.add_argument("--use-memory", action="store_true")
    args = parser.parse_args()

    memory = JsonMemory(settings.memory_path)
    seed_memory(memory)

    # Separate VLM perception model
    vlm_client = VLMClient()
    perception = VLMPerception(vlm_client)

    # Existing Qwen 0.8B planner model
    llm_client = LMStudioClient()
    planner = Planner(llm=llm_client, memory=memory)
    parser_obj = ActionParser()

    perception_result = perception.perceive_image(args.image)
    scene_state = perception_result["scene_state"]

    allowed_actions = get_valid_action_types(
        scene_state,
        scene_state["template_name"],
    )

    planner_output = planner.choose_action(
        scene_state=scene_state,
        allowed_actions=allowed_actions,
        memory_query=build_memory_query(scene_state),
        use_memory=args.use_memory,
    )

    raw = planner_output.get("raw_model_output", json.dumps(planner_output))
    semantic_action = parser_obj.parse(
        raw_text=raw,
        scene_state=scene_state,
    )

    result = {
        "image": args.image,
        "vlm_model": settings.vlm_model,
        "planner_model": settings.lmstudio_model,
        "raw_vlm_output": perception_result["raw_vlm_output"],
        "scene_state": scene_state,
        "allowed_actions": allowed_actions,
        "planner_output": planner_output,
        "semantic_action": semantic_action.to_dict(),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()