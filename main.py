
from __future__ import annotations

import json
import traceback

from app.config import settings
from app.llm_client import LMStudioClient
from app.memory import JsonMemory
from app.planner import Planner


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


def main() -> None:
    print("[debug] main started", flush=True)

    memory = JsonMemory(settings.memory_path)
    print(f"[debug] memory path: {settings.memory_path}", flush=True)

    seed_memory(memory)

    llm = LMStudioClient()
    print("[debug] llm client created", flush=True)

    planner = Planner(llm=llm, memory=memory)
    print("[debug] planner created", flush=True)

    scene_state = {
        "task_context": "dish_cleanup",
        "user_done_eating": True,
        "dirty_dishes_visible": True,
        "clean_counter_left_available": True,
        "robot_state": "idle",
        "last_action_result": "none",
    }

    allowed_actions = [
        "start_cleanup",
        "pick_dish_right_side",
        "place_clean_left_side",
        "wait",
    ]

    print("[debug] calling planner...", flush=True)

    decision = planner.choose_action(
        scene_state=scene_state,
        allowed_actions=allowed_actions,
        memory_query="dish cleanup user preference clean dishes left side right side start first",
    )

    print("[debug] planner returned", flush=True)
    print(json.dumps(decision, indent=2), flush=True)

    memory.add(
        kind="episode_decision",
        text=f"Scene led to action '{decision['action']}' because: {decision['reason']}",
        metadata={"scene": scene_state, "confidence": decision["confidence"]},
    )

    print("[debug] done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[debug] exception occurred", flush=True)
        traceback.print_exc()