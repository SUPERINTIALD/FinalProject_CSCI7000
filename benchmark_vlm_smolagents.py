from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from smolagents import ToolCallingAgent, tool

try:
    from smolagents import OpenAIModel as SmolOpenAIModel
except ImportError:
    from smolagents import OpenAIServerModel as SmolOpenAIModel

from app.action_parser import ActionParser
from app.config import settings
from app.evaluator import evaluate_action, get_valid_action_types
from app.kitchen_sim import KitchenSim
from app.llm_client import LMStudioClient
from app.memory import JsonMemory
from app.planner import Planner
from app.vlm_client import VLMClient
from app.vlm_perception import ACTIONABLE_STATES, VLMPerception


DEFAULT_VLM_MODELS = [
    "qwen2.5-vl-3b-instruct",
    # "qwen3.5-2b",
    # "qwen3.5-4b",
    # "glm-4.1v-9b",
]


CONTEXT: dict[str, Any] = {}


@dataclass
class ImageBenchmarkCase:
    image_path: str
    scenario: str
    expected_template: str
    expected_action: str
    expected_has_object: bool
    notes: str = ""


@dataclass
class BenchmarkRow:
    model: str
    image_path: str
    scenario: str

    expected_template: str
    predicted_template: str

    expected_action: str
    predicted_action: str

    expected_has_object: bool
    predicted_has_object: bool

    template_correct: bool
    object_presence_correct: bool
    action_correct: bool
    unsafe_false_positive: bool
    valid_action_bool: bool

    vlm_latency_sec: float
    planner_latency_sec: float
    execution_latency_sec: float
    agent_latency_sec: float
    total_latency_sec: float

    score: float

    raw_vlm_output: str
    scene_state: dict[str, Any]
    allowed_actions: list[str]
    planner_output: dict[str, Any]
    semantic_action: dict[str, Any]
    sim_result: dict[str, Any]
    planner_eval: dict[str, Any]
    agent_result: str
    agent_error: str | None


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
        if memory.all():
            return
        memory.add("preference", "User prefers clean dishes on the left counter.", {"task": "dish_cleanup"})


def load_cases(path: str | Path) -> list[ImageBenchmarkCase]:
    cases: list[ImageBenchmarkCase] = []

    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(
                ImageBenchmarkCase(
                    image_path=row["image_path"],
                    scenario=row.get("scenario", ""),
                    expected_template=row["expected_template"],
                    expected_action=row["expected_action"],
                    expected_has_object=row["expected_has_object"].strip().lower() in {"true", "1", "yes", "y"},
                    notes=row.get("notes", ""),
                )
            )

    return cases


def build_memory_query(scene_state: dict[str, Any]) -> str:
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


def has_actionable_object(scene_state: dict[str, Any]) -> bool:
    for obj in scene_state.get("objects", []):
        if obj.get("state") in ACTIONABLE_STATES:
            return True
    return False


def action_matches(predicted_action: str, expected_action_spec: str) -> bool:
    allowed = {item.strip() for item in expected_action_spec.split("|") if item.strip()}
    return predicted_action in allowed


def is_active_action(action_type: str) -> bool:
    return action_type in {"pick", "place", "start_task", "clean_surface"}


def score_case(
    *,
    template_correct: bool,
    object_presence_correct: bool,
    action_correct: bool,
    unsafe_false_positive: bool,
) -> float:
    score = 0.0
    score += 0.30 if template_correct else 0.0
    score += 0.30 if object_presence_correct else 0.0
    score += 0.40 if action_correct else 0.0

    if unsafe_false_positive:
        score -= 0.50

    return max(0.0, round(score, 4))


def build_evaluation_hints(case: ImageBenchmarkCase, scene_state: dict[str, Any]) -> dict[str, Any]:
    expected_actions = {item.strip() for item in case.expected_action.split("|") if item.strip()}

    should_wait = bool(expected_actions & {"wait", "inspect"}) and not bool(
        expected_actions & {"pick", "place", "start_task", "clean_surface"}
    )

    should_be_proactive = bool(expected_actions & {"pick", "place", "start_task", "clean_surface"})

    template_name = case.expected_template

    if template_name == "trash_cleanup":
        expected_surface = "trash_bin"
    elif template_name in {"dish_cleanup", "counter_cleanup"}:
        expected_surface = "sink"
    else:
        expected_surface = scene_state.get("placement_preference_surface", "sink")

    return {
        "should_wait": should_wait,
        "should_be_proactive": should_be_proactive,
        "preferred_action_types": sorted(expected_actions),
        "memory_relevant": case.expected_has_object,
        "expected_preference_surface": expected_surface,
        "actionable_object_present": case.expected_has_object,
        "task_family": "cleanup",
    }


def _get_metric(item: dict[str, Any], names: list[str], default: float = 0.0) -> float:
    for name in names:
        if name in item:
            value = item[name]
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
    return default


def setup_case_context(
    *,
    case: ImageBenchmarkCase,
    vlm_model_name: str,
    perception: VLMPerception,
    planner: Planner,
    parser_obj: ActionParser,
    use_memory: bool,
) -> None:
    CONTEXT.clear()

    CONTEXT["case"] = case
    CONTEXT["vlm_model_name"] = vlm_model_name
    CONTEXT["perception"] = perception
    CONTEXT["planner"] = planner
    CONTEXT["parser"] = parser_obj
    CONTEXT["use_memory"] = use_memory

    CONTEXT["raw_vlm_output"] = ""
    CONTEXT["scene_state"] = None
    CONTEXT["allowed_actions"] = []
    CONTEXT["planner_output"] = {}
    CONTEXT["semantic_action"] = {}
    CONTEXT["sim"] = None
    CONTEXT["sim_result"] = {}

    CONTEXT["vlm_latency_sec"] = 0.0
    CONTEXT["planner_latency_sec"] = 0.0
    CONTEXT["execution_latency_sec"] = 0.0


@tool
def inspect_image_scene() -> str:
    """
    Inspect the current kitchen image using the VLM perception model.

    Returns:
        JSON containing the raw VLM output and normalized scene_state.
    """
    case: ImageBenchmarkCase = CONTEXT["case"]
    perception: VLMPerception = CONTEXT["perception"]

    start = time.perf_counter()
    result = perception.perceive_image(case.image_path)
    CONTEXT["vlm_latency_sec"] = time.perf_counter() - start

    scene_state = result["scene_state"]

    CONTEXT["raw_vlm_output"] = result["raw_vlm_output"]
    CONTEXT["scene_state"] = scene_state
    CONTEXT["sim"] = KitchenSim(scene_state)

    return json.dumps(
        {
            "image_path": case.image_path,
            "raw_vlm_output": result["raw_vlm_output"],
            "scene_state": scene_state,
        },
        indent=2,
    )


@tool
def plan_proactive_robot_action() -> str:
    """
    Plan the next proactive semantic robot action from the VLM-generated scene_state.

    Returns:
        JSON containing allowed actions and the local Qwen planner output.
    """
    scene_state = CONTEXT.get("scene_state")
    if scene_state is None:
        return json.dumps({"error": "No scene_state available. Call inspect_image_scene first."}, indent=2)

    planner: Planner = CONTEXT["planner"]
    use_memory: bool = CONTEXT["use_memory"]

    allowed_actions = get_valid_action_types(
        scene_state,
        scene_state.get("template_name", "dish_cleanup"),
    )

    start = time.perf_counter()
    planner_output = planner.choose_action(
        scene_state=scene_state,
        allowed_actions=allowed_actions,
        memory_query=build_memory_query(scene_state),
        use_memory=use_memory,
    )
    CONTEXT["planner_latency_sec"] = time.perf_counter() - start

    CONTEXT["allowed_actions"] = allowed_actions
    CONTEXT["planner_output"] = planner_output

    return json.dumps(
        {
            "allowed_actions": allowed_actions,
            "planner_output": planner_output,
        },
        indent=2,
    )


@tool
def execute_last_robot_action_symbolically() -> str:
    """
    Execute the last planned semantic action in the symbolic Franka Panda kitchen simulator.

    Returns:
        JSON containing the parsed semantic action and symbolic simulation result.
    """
    scene_state = CONTEXT.get("scene_state")
    sim: KitchenSim | None = CONTEXT.get("sim")
    planner_output: dict[str, Any] = CONTEXT.get("planner_output", {})
    parser_obj: ActionParser = CONTEXT["parser"]

    if scene_state is None or sim is None:
        return json.dumps({"success": False, "message": "No scene/sim available."}, indent=2)

    if not planner_output:
        return json.dumps({"success": False, "message": "No planner action available."}, indent=2)

    safe_action_for_parse = {
        "action_type": planner_output.get("action_type"),
        "target_object": planner_output.get("target_object"),
        "target_surface": planner_output.get("target_surface"),
        "target_zone": planner_output.get("target_zone"),
        "parameters": planner_output.get("parameters", {}),
        "reason": planner_output.get("reason", ""),
        "memory_used": planner_output.get("memory_used", []),
        "confidence": planner_output.get("confidence", 0.0),
    }

    semantic_action = parser_obj.parse(
        raw_text=json.dumps(safe_action_for_parse),
        scene_state=scene_state,
    )

    start = time.perf_counter()
    sim_result = sim.step(semantic_action)
    CONTEXT["execution_latency_sec"] = time.perf_counter() - start

    CONTEXT["semantic_action"] = semantic_action.to_dict()
    CONTEXT["sim_result"] = sim_result.to_dict()

    return json.dumps(
        {
            "semantic_action": semantic_action.to_dict(),
            "sim_result": sim_result.to_dict(),
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
            inspect_image_scene,
            plan_proactive_robot_action,
            execute_last_robot_action_symbolically,
        ],
        max_steps=4,
    )


def run_agent_for_case(agent: ToolCallingAgent) -> tuple[str, str | None, float]:
    prompt = """
You are controlling a simulated Franka Panda kitchen robot.

You must perform exactly one observe-plan-execute cycle by calling tools in this order:
1. inspect_image_scene
2. plan_proactive_robot_action
3. execute_last_robot_action_symbolically

After the third tool call, stop and give a concise final summary.

Do not call any tool more than once.
Do not skip a tool.
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
    Safety fallback: if SmolAgents stopped early, run missing tool steps directly.
    This keeps the benchmark from crashing while still recording agent_error.
    """
    if CONTEXT.get("scene_state") is None:
        inspect_image_scene.forward()

    if not CONTEXT.get("planner_output"):
        plan_proactive_robot_action.forward()

    if not CONTEXT.get("semantic_action"):
        execute_last_robot_action_symbolically.forward()


def build_row_from_context(
    *,
    model_name: str,
    case: ImageBenchmarkCase,
    agent_result: str,
    agent_error: str | None,
    agent_latency_sec: float,
    total_latency_sec: float,
) -> BenchmarkRow:
    scene_state = CONTEXT.get("scene_state") or {}
    allowed_actions = CONTEXT.get("allowed_actions") or []
    planner_output = CONTEXT.get("planner_output") or {}
    semantic_action = CONTEXT.get("semantic_action") or {}
    sim_result = CONTEXT.get("sim_result") or {}

    predicted_template = scene_state.get("template_name", "ERROR")
    predicted_action = semantic_action.get("action_type", "ERROR")
    predicted_has_object = has_actionable_object(scene_state)

    template_correct = predicted_template == case.expected_template
    object_presence_correct = predicted_has_object == case.expected_has_object
    action_correct = action_matches(predicted_action, case.expected_action)

    unsafe_false_positive = not case.expected_has_object and is_active_action(predicted_action)
    valid_action_bool = predicted_action in allowed_actions if allowed_actions else False

    planner_eval = {}
    if semantic_action and predicted_action != "ERROR":
        evaluation_hints = build_evaluation_hints(case, scene_state)

        try:
            semantic_action_obj = ActionParser().parse(
                raw_text=json.dumps(semantic_action),
                scene_state=scene_state,
            )
            planner_eval_result = evaluate_action(
                scenario_id=Path(case.image_path).stem,
                template_name=scene_state.get("template_name", case.expected_template),
                scene_state=scene_state,
                action=semantic_action_obj,
                evaluation_hints=evaluation_hints,
            )
            planner_eval = planner_eval_result.to_dict()
        except Exception as exc:
            planner_eval = {"error": str(exc)}

    return BenchmarkRow(
        model=model_name,
        image_path=case.image_path,
        scenario=case.scenario,
        expected_template=case.expected_template,
        predicted_template=predicted_template,
        expected_action=case.expected_action,
        predicted_action=predicted_action,
        expected_has_object=case.expected_has_object,
        predicted_has_object=predicted_has_object,
        template_correct=template_correct,
        object_presence_correct=object_presence_correct,
        action_correct=action_correct,
        unsafe_false_positive=unsafe_false_positive,
        valid_action_bool=valid_action_bool,
        vlm_latency_sec=round(CONTEXT.get("vlm_latency_sec", 0.0), 4),
        planner_latency_sec=round(CONTEXT.get("planner_latency_sec", 0.0), 4),
        execution_latency_sec=round(CONTEXT.get("execution_latency_sec", 0.0), 4),
        agent_latency_sec=round(agent_latency_sec, 4),
        total_latency_sec=round(total_latency_sec, 4),
        score=score_case(
            template_correct=template_correct,
            object_presence_correct=object_presence_correct,
            action_correct=action_correct,
            unsafe_false_positive=unsafe_false_positive,
        ),
        raw_vlm_output=CONTEXT.get("raw_vlm_output", ""),
        scene_state=scene_state,
        allowed_actions=allowed_actions,
        planner_output=planner_output,
        semantic_action=semantic_action,
        sim_result=sim_result,
        planner_eval=planner_eval,
        agent_result=agent_result,
        agent_error=agent_error,
    )


def benchmark_one_model(
    *,
    model_name: str,
    cases: list[ImageBenchmarkCase],
    planner: Planner,
    parser_obj: ActionParser,
    use_memory: bool,
    results_dir: Path,
) -> list[BenchmarkRow]:
    print(f"\n=== SmolAgents benchmark for VLM model: {model_name} ===")

    vlm_client = VLMClient(model=model_name)
    perception = VLMPerception(vlm_client)

    agent = make_agent()
    rows: list[BenchmarkRow] = []

    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case.image_path}")

        setup_case_context(
            case=case,
            vlm_model_name=model_name,
            perception=perception,
            planner=planner,
            parser_obj=parser_obj,
            use_memory=use_memory,
        )

        start_total = time.perf_counter()
        agent_result, agent_error, agent_latency_sec = run_agent_for_case(agent)

        try:
            recover_missing_steps()
        except Exception as exc:
            if agent_error is None:
                agent_error = f"Recovery failed: {exc}"

        total_latency_sec = time.perf_counter() - start_total

        row = build_row_from_context(
            model_name=model_name,
            case=case,
            agent_result=agent_result,
            agent_error=agent_error,
            agent_latency_sec=agent_latency_sec,
            total_latency_sec=total_latency_sec,
        )

        rows.append(row)

        print(
            f"    action={row.predicted_action} "
            f"template={row.predicted_template} "
            f"score={row.score} "
            f"unsafe={row.unsafe_false_positive} "
            f"valid={row.valid_action_bool} "
            f"time={row.total_latency_sec}s"
        )

        detail_path = results_dir / "details" / model_name.replace("/", "_") / f"{Path(case.image_path).stem}.json"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(json.dumps(asdict(row), indent=2), encoding="utf-8")

    return rows


def summarize(rows: list[BenchmarkRow]) -> dict[str, Any]:
    if not rows:
        return {}

    perception_action_summary = {
        "num_cases": len(rows),
        "avg_score": round(statistics.mean(row.score for row in rows), 4),
        "template_accuracy": round(sum(row.template_correct for row in rows) / len(rows), 4),
        "object_presence_accuracy": round(sum(row.object_presence_correct for row in rows) / len(rows), 4),
        "action_accuracy": round(sum(row.action_correct for row in rows) / len(rows), 4),
        "unsafe_false_positive_rate": round(sum(row.unsafe_false_positive for row in rows) / len(rows), 4),
        "valid_action_rate": round(sum(row.valid_action_bool for row in rows) / len(rows), 4),
        "agent_error_rate": round(sum(row.agent_error is not None for row in rows) / len(rows), 4),
        "avg_vlm_latency_sec": round(statistics.mean(row.vlm_latency_sec for row in rows), 4),
        "avg_planner_latency_sec": round(statistics.mean(row.planner_latency_sec for row in rows), 4),
        "avg_execution_latency_sec": round(statistics.mean(row.execution_latency_sec for row in rows), 4),
        "avg_agent_latency_sec": round(statistics.mean(row.agent_latency_sec for row in rows), 4),
        "avg_total_latency_sec": round(statistics.mean(row.total_latency_sec for row in rows), 4),
    }

    eval_results = [row.planner_eval for row in rows if row.planner_eval and "error" not in row.planner_eval]

    if eval_results:
        planner_evaluator_summary = {
            "num_scenarios": len(eval_results),
            "avg_total_score": round(
                statistics.mean(_get_metric(item, ["total_score", "avg_total_score", "score"], default=0.0) for item in eval_results),
                4,
            ),
            "valid_action_rate": round(sum(row.valid_action_bool for row in rows) / len(rows), 4),
            "avg_proactive_score": round(
                statistics.mean(_get_metric(item, ["proactive_score", "proactivity_score"], default=0.0) for item in eval_results),
                4,
            ),
            "avg_restraint_score": round(
                statistics.mean(_get_metric(item, ["restraint_score"], default=0.0) for item in eval_results),
                4,
            ),
            "avg_progress_score": round(
                statistics.mean(_get_metric(item, ["progress_score"], default=0.0) for item in eval_results),
                4,
            ),
            "avg_memory_alignment_score": round(
                statistics.mean(_get_metric(item, ["memory_alignment_score", "memory_score"], default=0.0) for item in eval_results),
                4,
            ),
        }
    else:
        planner_evaluator_summary = {}

    return {
        "perception_action_summary": perception_action_summary,
        "planner_evaluator_summary": planner_evaluator_summary,
    }


def write_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "image_path",
        "scenario",
        "expected_template",
        "predicted_template",
        "expected_action",
        "predicted_action",
        "expected_has_object",
        "predicted_has_object",
        "template_correct",
        "object_presence_correct",
        "action_correct",
        "unsafe_false_positive",
        "valid_action_bool",
        "vlm_latency_sec",
        "planner_latency_sec",
        "execution_latency_sec",
        "agent_latency_sec",
        "total_latency_sec",
        "score",
        "agent_error",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            row_dict = asdict(row)
            writer.writerow({k: row_dict[k] for k in fieldnames})


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--annotations", default="data/vlm_annotations.csv")
    cli.add_argument("--models", nargs="*", default=DEFAULT_VLM_MODELS)
    cli.add_argument("--use-memory", action="store_true")
    cli.add_argument("--results-dir", default=None)
    cli.add_argument("--planner-model-note", default="qwen3.5-0.8b")
    args = cli.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir or f"results/smolagents_vlm_benchmark_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.annotations)

    memory = JsonMemory(settings.memory_path)
    seed_memory(memory)

    planner_llm = LMStudioClient()
    planner = Planner(llm=planner_llm, memory=memory)
    parser_obj = ActionParser()

    all_rows: list[BenchmarkRow] = []
    summary_by_model: dict[str, Any] = {}

    print("\n=== SmolAgents VLM-to-Action Benchmark ===")
    print("Planner model stays fixed:", args.planner_model_note)
    print("Testing VLM models:", args.models)
    print("Cases:", len(cases))
    print("Results dir:", results_dir)

    for model_name in args.models:
        rows = benchmark_one_model(
            model_name=model_name,
            cases=cases,
            planner=planner,
            parser_obj=parser_obj,
            use_memory=args.use_memory,
            results_dir=results_dir,
        )

        all_rows.extend(rows)
        summary_by_model[model_name] = summarize(rows)

    write_csv(results_dir / "all_results.csv", all_rows)
    (results_dir / "summary.json").write_text(json.dumps(summary_by_model, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary_by_model, indent=2))

    print("\nSaved:")
    print(results_dir / "all_results.csv")
    print(results_dir / "summary.json")
    print(results_dir / "details")


if __name__ == "__main__":
    main()