from __future__ import annotations

import textwrap


import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Embedded benchmark data
# -----------------------------
PLANNER_PHASES = {
    "V1": {
        "avg_total_score": 0.515,
        "valid_action_rate": 0.40,
        "avg_proactive_score": 0.75,
        "avg_restraint_score": 0.50,
        "avg_progress_score": 0.40,
        "avg_memory_alignment_score": 0.65,
    },
    "V2": {
        "avg_total_score": 0.515,
        "valid_action_rate": 0.40,
        "avg_proactive_score": 0.75,
        "avg_restraint_score": 0.50,
        "avg_progress_score": 0.40,
        "avg_memory_alignment_score": 0.65,
    },
    "V3_with_memory": {
        "avg_total_score": 0.857,
        "valid_action_rate": 0.98,
        "avg_proactive_score": 0.80,
        "avg_restraint_score": 0.90,
        "avg_progress_score": 0.80,
        "avg_memory_alignment_score": 0.63,
    },
    "V3_without_memory": {
        "avg_total_score": 0.876,
        "valid_action_rate": 0.98,
        "avg_proactive_score": 0.90,
        "avg_restraint_score": 0.95,
        "avg_progress_score": 0.88,
        "avg_memory_alignment_score": 0.36,
    },
    "V4_selective_memory": {
        "avg_total_score": 0.77,
        "valid_action_rate": 1.00,
        "avg_proactive_score": 0.62,
        "avg_restraint_score": 0.81,
        "avg_progress_score": 0.62,
        "avg_memory_alignment_score": 0.60,
    },
    "LoRA_small": {
        "avg_total_score": 0.662,
        "valid_action_rate": 1.00,
        "avg_proactive_score": 0.40,
        "avg_restraint_score": 0.70,
        "avg_progress_score": 0.40,
        "avg_memory_alignment_score": 0.62,
    },
    "LoRA_full": {
        "avg_total_score": 0.804,
        "valid_action_rate": 0.84,
        "avg_proactive_score": 0.82,
        "avg_restraint_score": 0.91,
        "avg_progress_score": 0.72,
        "avg_memory_alignment_score": 0.62,
    },
    "PhaseE_agentic_kitchen": {
        "avg_total_score": 0.80,
        "valid_action_rate": 0.94,
        "avg_proactive_score": 0.80,
        "avg_restraint_score": 0.88,
        "avg_progress_score": 0.72,
        "avg_memory_alignment_score": 0.38,
    },
    "PhaseE_smolagents_kitchen": {
        "avg_total_score": 0.828,
        "valid_action_rate": 1.00,
        "avg_proactive_score": 0.76,
        "avg_restraint_score": 0.86,
        "avg_progress_score": 0.74,
        "avg_memory_alignment_score": 0.56,
    },
}

PHASE_E_TASKS = {
    "Agentic Kitchen": {
        "counter_cleanup": {
            "avg_total_score": 0.875,
            "valid_action_rate": 1.0,
            "avg_proactive_score": 0.9,
            "avg_restraint_score": 0.95,
            "avg_progress_score": 0.9,
            "avg_memory_alignment_score": 0.25,
        },
        "dish_cleanup": {
            "avg_total_score": 0.8765,
            "valid_action_rate": 1.0,
            "avg_proactive_score": 0.8235,
            "avg_restraint_score": 0.9118,
            "avg_progress_score": 0.8235,
            "avg_memory_alignment_score": 0.6471,
        },
        "trash_cleanup": {
            "avg_total_score": 0.5846,
            "valid_action_rate": 0.7692,
            "avg_proactive_score": 0.6154,
            "avg_restraint_score": 0.7308,
            "avg_progress_score": 0.3077,
            "avg_memory_alignment_score": 0.2308,
        },
    },
    "SmolAgents Kitchen": {
        "counter_cleanup": {
            "avg_total_score": 0.8825,
            "valid_action_rate": 1.0,
            "avg_proactive_score": 0.85,
            "avg_restraint_score": 0.925,
            "avg_progress_score": 0.85,
            "avg_memory_alignment_score": 0.575,
        },
        "dish_cleanup": {
            "avg_total_score": 0.9088,
            "valid_action_rate": 1.0,
            "avg_proactive_score": 0.8824,
            "avg_restraint_score": 0.9412,
            "avg_progress_score": 0.8824,
            "avg_memory_alignment_score": 0.6765,
        },
        "trash_cleanup": {
            "avg_total_score": 0.6385,
            "valid_action_rate": 1.0,
            "avg_proactive_score": 0.4615,
            "avg_restraint_score": 0.6538,
            "avg_progress_score": 0.3846,
            "avg_memory_alignment_score": 0.3846,
        },
    },
}

LATEST_DIRECT_VLM = {
    "qwen2.5-vl-3b-instruct": {
        "perception_action_summary": {
            "avg_score": 0.7128,
            "template_accuracy": 0.6923,
            "object_presence_accuracy": 0.7949,
            "action_accuracy": 0.6667,
            "unsafe_false_positive_rate": 0.0513,
            "avg_vlm_latency_sec": 10.0772,
            "avg_planner_latency_sec": 3.3028,
            "avg_total_latency_sec": 13.3803,
        },
        "planner_evaluator_summary": {
            "avg_total_score": 0.759,
            "valid_action_rate": 1.0,
            "avg_proactive_score": 0.6667,
            "avg_restraint_score": 0.7564,
            "avg_progress_score": 0.6667,
            "avg_memory_alignment_score": 0.641,
        },
    },
    "qwen3.5-2b": {
        "perception_action_summary": {
            "avg_score": 0.7333,
            "template_accuracy": 0.6667,
            "object_presence_accuracy": 0.7692,
            "action_accuracy": 0.7949,
            "unsafe_false_positive_rate": 0.1026,
            "avg_vlm_latency_sec": 6.7662,
            "avg_planner_latency_sec": 3.3177,
            "avg_total_latency_sec": 10.0841,
        },
        "planner_evaluator_summary": {
            "avg_total_score": 0.8077,
            "valid_action_rate": 1.0,
            "avg_proactive_score": 0.8333,
            "avg_restraint_score": 0.8205,
            "avg_progress_score": 0.7949,
            "avg_memory_alignment_score": 0.641,
        },
    },
    "qwen3.5-4b": {
        "perception_action_summary": {
            "avg_score": 0.2513,
            "template_accuracy": 0.1795,
            "object_presence_accuracy": 0.2821,
            "action_accuracy": 0.2821,
            "unsafe_false_positive_rate": 0.0,
            "avg_vlm_latency_sec": 53.2404,
            "avg_planner_latency_sec": 3.1842,
            "avg_total_latency_sec": 56.4248,
        },
        "planner_evaluator_summary": {
            "avg_total_score": 0.8221,
            "valid_action_rate": 1.0,
            "avg_proactive_score": 0.8590,
            "avg_restraint_score": 0.7179,
            "avg_progress_score": 0.7128,
            "avg_memory_alignment_score": 0.641,
        },
    },
}

LATEST_SMOL_VLM = {
    "qwen2.5-vl-3b-instruct": {
        "perception_action_summary": {
            "avg_score": 0.7026,
            "template_accuracy": 0.6923,
            "object_presence_accuracy": 0.7949,
            "action_accuracy": 0.6410,
            "unsafe_false_positive_rate": 0.0513,
            "valid_action_rate": 0.9744,
            "agent_error_rate": 0.0256,
            "avg_vlm_latency_sec": 9.7976,
            "avg_planner_latency_sec": 0.5263,
            "avg_execution_latency_sec": 0.0,
            "avg_agent_latency_sec": 15.3496,
            "avg_total_latency_sec": 15.3887,
        },
        "planner_evaluator_summary": {
            "avg_total_score": 0.7618,
            "valid_action_rate": 0.9744,
            "avg_proactive_score": 0.6579,
            "avg_restraint_score": 0.7500,
            "avg_progress_score": 0.6579,
            "avg_memory_alignment_score": 0.6447,
        },
    },
    "qwen3.5-2b": {
        "perception_action_summary": {
            "avg_score": 0.7333,
            "template_accuracy": 0.6667,
            "object_presence_accuracy": 0.7692,
            "action_accuracy": 0.7949,
            "unsafe_false_positive_rate": 0.1026,
            "valid_action_rate": 1.0,
            "agent_error_rate": 0.0,
            "avg_vlm_latency_sec": 6.3599,
            "avg_planner_latency_sec": 0.5257,
            "avg_execution_latency_sec": 0.0,
            "avg_agent_latency_sec": 11.7101,
            "avg_total_latency_sec": 11.7101,
        },
        "planner_evaluator_summary": {
            "avg_total_score": 0.7923,
            "valid_action_rate": 1.0,
            "avg_proactive_score": 0.8333,
            "avg_restraint_score": 0.8205,
            "avg_progress_score": 0.7949,
            "avg_memory_alignment_score": 0.6410,
        },
    },
    "qwen3.5-4b": {
        "perception_action_summary": {
            "avg_score": 0.2513,
            "template_accuracy": 0.1795,
            "object_presence_accuracy": 0.2821,
            "action_accuracy": 0.2821,
            "unsafe_false_positive_rate": 0.0,
            "valid_action_rate": 1.0,
            "agent_error_rate": 0.0,
            "avg_vlm_latency_sec": 53.5753,
            "avg_planner_latency_sec": 0.4998,
            "avg_execution_latency_sec": 0.0,
            "avg_agent_latency_sec": 112.5930,
            "avg_total_latency_sec": 112.5932,
        },
        "planner_evaluator_summary": {
            "avg_total_score": 0.8297,
            "valid_action_rate": 1.0,
            "avg_proactive_score": 0.8718,
            "avg_restraint_score": 0.7436,
            "avg_progress_score": 0.7128,
            "avg_memory_alignment_score": 0.6410,
        },
    },
}


def save_json() -> None:
    data = {
        "planner_phases": PLANNER_PHASES,
        "phase_e_tasks": PHASE_E_TASKS,
        "latest_direct_vlm": LATEST_DIRECT_VLM,
        "latest_smol_vlm": LATEST_SMOL_VLM,
    }
    with (OUTPUT_DIR / "benchmark_data_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _finish_plot(title: str, xlabel: str, ylabel: str, filename: str, rotation: int = 0) -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotation:
        plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()


def chart_planner_progression() -> None:
    labels = list(PLANNER_PHASES.keys())
    values = [PLANNER_PHASES[k]["avg_total_score"] for k in labels]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    _finish_plot(
        "Planner / Agent Benchmark Progression (Avg Total Score)",
        "Version / Phase",
        "Average Total Score",
        "01_planner_progression_avg_total_score.png",
        rotation=35,
    )


def chart_planner_metrics_for_best_versions() -> None:
    chosen = ["V3_without_memory", "V3_with_memory", "V4_selective_memory", "LoRA_full", "PhaseE_smolagents_kitchen"]
    metrics = [
        "valid_action_rate",
        "avg_proactive_score",
        "avg_restraint_score",
        "avg_progress_score",
        "avg_memory_alignment_score",
    ]
    pretty = {
        "valid_action_rate": "Valid",
        "avg_proactive_score": "Proactive",
        "avg_restraint_score": "Restraint",
        "avg_progress_score": "Progress",
        "avg_memory_alignment_score": "Memory",
    }

    # one plot, multiple lines
    plt.figure(figsize=(10, 6))
    for version in chosen:
        y = [PLANNER_PHASES[version][m] for m in metrics]
        plt.plot([pretty[m] for m in metrics], y, marker="o", label=version)
    plt.legend()
    _finish_plot(
        "Key Metric Profiles for Strong Planner / Agent Versions",
        "Metric",
        "Score",
        "02_key_metric_profiles.png",
    )


def chart_phase_e_task_comparison() -> None:
    tasks = ["counter_cleanup", "dish_cleanup", "trash_cleanup"]
    values = [PHASE_E_TASKS["SmolAgents Kitchen"][t]["avg_total_score"] for t in tasks]
    plt.figure(figsize=(8, 5))
    plt.bar(tasks, values)
    _finish_plot(
        "SmolAgents Kitchen: Avg Total Score by Task",
        "Task",
        "Average Total Score",
        "03_smolagents_task_avg_total_score.png",
        rotation=20,
    )


def chart_phase_e_memory_alignment() -> None:
    tasks = ["counter_cleanup", "dish_cleanup", "trash_cleanup"]
    symbolic = [PHASE_E_TASKS["Agentic Kitchen"][t]["avg_memory_alignment_score"] for t in tasks]
    smol = [PHASE_E_TASKS["SmolAgents Kitchen"][t]["avg_memory_alignment_score"] for t in tasks]

    x = range(len(tasks))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], symbolic, width=width, label="Agentic Kitchen")
    plt.bar([i + width / 2 for i in x], smol, width=width, label="SmolAgents Kitchen")
    plt.xticks(list(x), tasks, rotation=20, ha="right")
    plt.legend()
    _finish_plot(
        "Memory Alignment by Task: Symbolic vs SmolAgents Kitchen",
        "Task",
        "Memory Alignment Score",
        "04_memory_alignment_task_comparison.png",
    )


def chart_latest_direct_vlm_scores() -> None:
    models = list(LATEST_DIRECT_VLM.keys())
    scores = [LATEST_DIRECT_VLM[m]["perception_action_summary"]["avg_score"] for m in models]
    plt.figure(figsize=(9, 5))
    plt.bar(models, scores)
    _finish_plot(
        "Latest Direct VLM Benchmark: Avg Score",
        "Model",
        "Average Score",
        "05_latest_direct_vlm_avg_score.png",
        rotation=20,
    )


def chart_latest_direct_vlm_latency() -> None:
    models = list(LATEST_DIRECT_VLM.keys())
    lat = [LATEST_DIRECT_VLM[m]["perception_action_summary"]["avg_total_latency_sec"] for m in models]
    plt.figure(figsize=(9, 5))
    plt.bar(models, lat)
    _finish_plot(
        "Latest Direct VLM Benchmark: Avg Total Latency",
        "Model",
        "Seconds",
        "06_latest_direct_vlm_latency.png",
        rotation=20,
    )


def chart_latest_smolagents_vlm_scores() -> None:
    models = list(LATEST_SMOL_VLM.keys())
    scores = [LATEST_SMOL_VLM[m]["perception_action_summary"]["avg_score"] for m in models]
    plt.figure(figsize=(9, 5))
    plt.bar(models, scores)
    _finish_plot(
        "Latest SmolAgents VLM Benchmark: Avg Score",
        "Model",
        "Average Score",
        "07_latest_smolagents_vlm_avg_score.png",
        rotation=20,
    )


def chart_latest_smolagents_vlm_tradeoff() -> None:
    models = list(LATEST_SMOL_VLM.keys())
    x = [LATEST_SMOL_VLM[m]["perception_action_summary"]["unsafe_false_positive_rate"] for m in models]
    y = [LATEST_SMOL_VLM[m]["perception_action_summary"]["action_accuracy"] for m in models]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    for i, m in enumerate(models):
        plt.annotate(m, (x[i], y[i]))
    _finish_plot(
        "SmolAgents VLM Tradeoff: Unsafe FP vs Action Accuracy",
        "Unsafe False Positive Rate",
        "Action Accuracy",
        "08_smolagents_tradeoff_accuracy_vs_unsafe_fp.png",
    )


def chart_latest_smolagents_vlm_agent_latency() -> None:
    models = list(LATEST_SMOL_VLM.keys())
    lat = [LATEST_SMOL_VLM[m]["perception_action_summary"]["avg_agent_latency_sec"] for m in models]
    plt.figure(figsize=(9, 5))
    plt.bar(models, lat)
    _finish_plot(
        "Latest SmolAgents VLM Benchmark: Avg Agent Loop Time",
        "Model",
        "Seconds",
        "09_smolagents_agent_loop_time.png",
        rotation=20,
    )


def chart_latest_smolagents_planner_scores() -> None:
    models = list(LATEST_SMOL_VLM.keys())
    vals = [LATEST_SMOL_VLM[m]["planner_evaluator_summary"]["avg_total_score"] for m in models]
    plt.figure(figsize=(9, 5))
    plt.bar(models, vals)
    _finish_plot(
        "Latest SmolAgents VLM Benchmark: Planner Avg Total Score",
        "Model",
        "Planner Avg Total Score",
        "10_smolagents_planner_avg_total_score.png",
        rotation=20,
    )


def write_recommendations() -> None:
    text = """
    Suggested presentation visualizations:
    1. Planner / agent progression over versions (V1 -> V7) using avg_total_score.
    2. Key metric profile comparison for V3_without_memory, V3_with_memory, V4, LoRA_full, and PhaseE_smolagents_kitchen.
    3. Per-task comparison for counter_cleanup, dish_cleanup, and trash_cleanup.
    4. Direct VLM comparison using avg_score, action_accuracy, unsafe false positives, and total latency.
    5. SmolAgents VLM comparison using avg_score, planner avg_total_score, agent latency, and unsafe FP vs action accuracy tradeoff.

    Recommended for slides:
    - Definitely include:
      * 01_planner_progression_avg_total_score.png
      * 03_smolagents_task_avg_total_score.png
      * 05_latest_direct_vlm_avg_score.png
      * 08_smolagents_tradeoff_accuracy_vs_unsafe_fp.png
      * 09_smolagents_agent_loop_time.png
    - Optional backup slide:
      * 04_memory_alignment_task_comparison.png
      * 10_smolagents_planner_avg_total_score.png
    """
    (OUTPUT_DIR / "presentation_visualization_notes.txt").write_text(
        "\n".join(line.strip() for line in textwrap.dedent(text).splitlines() if line.strip()),
        encoding="utf-8",
    )


def main() -> None:
    save_json()
    chart_planner_progression()
    chart_planner_metrics_for_best_versions()
    chart_phase_e_task_comparison()
    chart_phase_e_memory_alignment()
    chart_latest_direct_vlm_scores()
    chart_latest_direct_vlm_latency()
    chart_latest_smolagents_vlm_scores()
    chart_latest_smolagents_vlm_tradeoff()
    chart_latest_smolagents_vlm_agent_latency()
    chart_latest_smolagents_planner_scores()
    write_recommendations()
    print(f"Wrote plots and notes to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
