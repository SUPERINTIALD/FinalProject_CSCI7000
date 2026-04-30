from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.action_parser import ActionParser
from app.config import settings
# from app.evaluator import get_valid_action_types

from app.evaluator import evaluate_action, get_valid_action_types, summarize_evaluations
from app.llm_client import LMStudioClient
from app.memory import JsonMemory
from app.planner import Planner
from app.vlm_client import VLMClient
from app.vlm_perception import ACTIONABLE_STATES, VLMPerception


DEFAULT_VLM_MODELS = [
    "qwen2.5-vl-3b-instruct",
    "qwen3.5-2b",
    # Add more only when they are loaded in LM Studio with image support:
    # "qwen3.5-4b",
    # "glm-4.1v-9b",
]


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
    vlm_latency_sec: float
    planner_latency_sec: float
    total_latency_sec: float
    score: float
    raw_vlm_output: str
    scene_state: dict[str, Any]
    planner_output: dict[str, Any]
    semantic_action: dict[str, Any]
    planner_eval: dict[str, Any]


def apply_safety_gate(
    scene_state: dict[str, Any],
    semantic_action_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Conservative guardrail for proactive household behavior.
    Prevents picking when the VLM indicates the object may still be in use.
    """
    action_type = semantic_action_dict.get("action_type")
    target_object = semantic_action_dict.get("target_object")

    if action_type not in {"pick", "start_task"}:
        return semantic_action_dict

    objects = scene_state.get("objects", [])

    target = None
    for obj in objects:
        if obj.get("name") == target_object:
            target = obj
            break

    # If there is no clear target, do not act.
    if target_object and target is None:
        semantic_action_dict["action_type"] = "inspect"
        semantic_action_dict["target_object"] = None
        semantic_action_dict["target_surface"] = None
        semantic_action_dict["reason"] = "Safety gate: target object was not clearly grounded in the scene."
        return semantic_action_dict

    # Filled cups / plates / bowls should not be picked.
    for obj in objects:
        kind = str(obj.get("kind", "")).lower()
        state = str(obj.get("state", "")).lower()

        if kind in {"cup", "glass", "mug", "plate", "bowl"} and state == "filled":
            semantic_action_dict["action_type"] = "wait"
            semantic_action_dict["target_object"] = None
            semantic_action_dict["target_surface"] = None
            semantic_action_dict["reason"] = "Safety gate: food or drink appears unfinished, so the robot should wait."
            return semantic_action_dict

    # If user/scene state says still eating, do not pick.
    if scene_state.get("user_state") in {"still_eating", "using_counter", "using_kitchen"}:
        semantic_action_dict["action_type"] = "wait"
        semantic_action_dict["target_object"] = None
        semantic_action_dict["target_surface"] = None
        semantic_action_dict["reason"] = "Safety gate: user may still be using the item."
        return semantic_action_dict
    print("[safety gate input]", semantic_action_dict)
    print("[safety gate scene]", scene_state)
    return semantic_action_dict

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
    expected_actions = {
        item.strip()
        for item in case.expected_action.split("|")
        if item.strip()
    }

    should_wait = bool(expected_actions & {"wait", "inspect"}) and not bool(
        expected_actions & {"pick", "place", "start_task", "clean_surface"}
    )

    should_be_proactive = bool(
        expected_actions & {"pick", "place", "start_task", "clean_surface"}
    )

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

def benchmark_one_model(
    *,
    model_name: str,
    cases: list[ImageBenchmarkCase],
    planner: Planner,
    parser_obj: ActionParser,
    use_memory: bool,
    results_dir: Path,
) -> list[BenchmarkRow]:
    print(f"\n=== Benchmarking VLM model: {model_name} ===")

    vlm_client = VLMClient(model=model_name)
    perception = VLMPerception(vlm_client)

    rows: list[BenchmarkRow] = []

    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case.image_path}")

        start_total = time.perf_counter()

        try:
            start_vlm = time.perf_counter()
            perception_result = perception.perceive_image(case.image_path)
            vlm_latency = time.perf_counter() - start_vlm

            scene_state = perception_result["scene_state"]
            allowed_actions = get_valid_action_types(scene_state, scene_state["template_name"])

            start_planner = time.perf_counter()
            planner_output = planner.choose_action(
                scene_state=scene_state,
                allowed_actions=allowed_actions,
                memory_query=build_memory_query(scene_state),
                use_memory=use_memory,
            )
            planner_latency = time.perf_counter() - start_planner

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

            safe_semantic_action_dict = apply_safety_gate(
                scene_state=scene_state,
                semantic_action_dict=semantic_action.to_dict(),
            )

            semantic_action = parser_obj.parse(
                raw_text=json.dumps(safe_semantic_action_dict),
                scene_state=scene_state,
            )
            evaluation_hints = build_evaluation_hints(case, scene_state)

            planner_eval_result = evaluate_action(
                scenario_id=Path(case.image_path).stem,
                template_name=scene_state.get("template_name", case.expected_template),
                scene_state=scene_state,
                action=semantic_action,
                evaluation_hints=evaluation_hints,
            )
            if idx == 1:
                print("Evaluator keys:", planner_eval_result.to_dict().keys())  
            predicted_template = scene_state.get("template_name", "")
            predicted_action = semantic_action.action_type
            predicted_has_object = has_actionable_object(scene_state)

            template_correct = predicted_template == case.expected_template
            object_presence_correct = predicted_has_object == case.expected_has_object
            action_correct = action_matches(predicted_action, case.expected_action)

            unsafe_false_positive = (
                not case.expected_has_object
                and is_active_action(predicted_action)
            )

            total_latency = time.perf_counter() - start_total

            row = BenchmarkRow(
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
                vlm_latency_sec=round(vlm_latency, 4),
                planner_latency_sec=round(planner_latency, 4),
                total_latency_sec=round(total_latency, 4),
                score=score_case(
                    template_correct=template_correct,
                    object_presence_correct=object_presence_correct,
                    action_correct=action_correct,
                    unsafe_false_positive=unsafe_false_positive,
                ),
                raw_vlm_output=perception_result["raw_vlm_output"],
                scene_state=scene_state,
                planner_output=planner_output,
                semantic_action=semantic_action.to_dict(),

                planner_eval=planner_eval_result.to_dict(),
            )

        except Exception as exc:
            total_latency = time.perf_counter() - start_total
            row = BenchmarkRow(
                model=model_name,
                image_path=case.image_path,
                scenario=case.scenario,
                expected_template=case.expected_template,
                predicted_template="ERROR",
                expected_action=case.expected_action,
                predicted_action="ERROR",
                expected_has_object=case.expected_has_object,
                predicted_has_object=False,
                template_correct=False,
                object_presence_correct=False,
                action_correct=False,
                unsafe_false_positive=False,
                vlm_latency_sec=0.0,
                planner_latency_sec=0.0,
                total_latency_sec=round(total_latency, 4),
                score=0.0,
                raw_vlm_output=str(exc),
                scene_state={},
                planner_output={},
                semantic_action={},
                planner_eval={},
            )

        rows.append(row)

        print(
            f"    action={row.predicted_action} "
            f"template={row.predicted_template} "
            f"score={row.score} "
            f"unsafe={row.unsafe_false_positive} "
            f"time={row.total_latency_sec}s"
        )

        detail_path = results_dir / "details" / model_name.replace("/", "_") / f"{Path(case.image_path).stem}.json"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(json.dumps(asdict(row), indent=2), encoding="utf-8")

    return rows

def _get_metric(item: dict[str, Any], names: list[str], default: float = 0.0) -> float:
    """
    Safely get a metric from evaluator output even if the evaluator uses
    slightly different key names.
    """
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


def summarize(rows: list[BenchmarkRow]) -> dict[str, Any]:
    if not rows:
        return {}

    perception_summary = {
        "num_cases": len(rows),
        "avg_score": round(statistics.mean(row.score for row in rows), 4),
        "template_accuracy": round(sum(row.template_correct for row in rows) / len(rows), 4),
        "object_presence_accuracy": round(sum(row.object_presence_correct for row in rows) / len(rows), 4),
        "action_accuracy": round(sum(row.action_correct for row in rows) / len(rows), 4),
        "unsafe_false_positive_rate": round(sum(row.unsafe_false_positive for row in rows) / len(rows), 4),
        "avg_vlm_latency_sec": round(statistics.mean(row.vlm_latency_sec for row in rows), 4),
        "avg_planner_latency_sec": round(statistics.mean(row.planner_latency_sec for row in rows), 4),
        "avg_total_latency_sec": round(statistics.mean(row.total_latency_sec for row in rows), 4),
    }

    eval_results = [row.planner_eval for row in rows if row.planner_eval]

    if eval_results:
        valid_values = [
            _get_metric(
                item,
                [
                    "valid_action",
                    "is_valid_action",
                    "valid_action_score",
                    "valid_score",
                    "valid_json",
                ],
                default=1.0,
            )
            for item in eval_results
        ]

        planner_summary = {
            "num_scenarios": len(eval_results),
            "avg_total_score": round(
                statistics.mean(
                    _get_metric(item, ["total_score", "avg_total_score", "score"], default=0.0)
                    for item in eval_results
                ),
                4,
            ),
            "valid_action_rate": round(statistics.mean(valid_values), 4),
            "avg_proactive_score": round(
                statistics.mean(
                    _get_metric(item, ["proactive_score", "proactivity_score"], default=0.0)
                    for item in eval_results
                ),
                4,
            ),
            "avg_restraint_score": round(
                statistics.mean(
                    _get_metric(item, ["restraint_score"], default=0.0)
                    for item in eval_results
                ),
                4,
            ),
            "avg_progress_score": round(
                statistics.mean(
                    _get_metric(item, ["progress_score"], default=0.0)
                    for item in eval_results
                ),
                4,
            ),
            "avg_memory_alignment_score": round(
                statistics.mean(
                    _get_metric(item, ["memory_alignment_score", "memory_score"], default=0.0)
                    for item in eval_results
                ),
                4,
            ),
        }
    else:
        planner_summary = {}

    return {
        "perception_action_summary": perception_summary,
        "planner_evaluator_summary": planner_summary,
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
        "vlm_latency_sec",
        "planner_latency_sec",
        "total_latency_sec",
        "score",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_dict = asdict(row)
            writer.writerow({k: row_dict[k] for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/vlm_annotations.csv")
    parser.add_argument("--models", nargs="*", default=DEFAULT_VLM_MODELS)
    parser.add_argument("--use-memory", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--planner-model-note", default="qwen3.5-0.8b")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir or f"results/vlm_model_benchmark_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.annotations)

    memory = JsonMemory(settings.memory_path)
    seed_memory(memory)

    planner_llm = LMStudioClient()
    planner = Planner(llm=planner_llm, memory=memory)
    parser_obj = ActionParser()

    all_rows: list[BenchmarkRow] = []
    summary_by_model: dict[str, Any] = {}

    print("\nPlanner model stays fixed:", args.planner_model_note)
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