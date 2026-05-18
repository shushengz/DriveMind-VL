"""Compare normal/text-only/wrong-image/blank-image external metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_metric(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if "overall" in obj and isinstance(obj["overall"], dict):
        return obj["overall"].get("metrics", {})
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DriveMind visual-ablation metric JSON files.")
    parser.add_argument("--normal", required=True)
    parser.add_argument("--text_only", required=True)
    parser.add_argument("--wrong_image", required=True)
    parser.add_argument("--blank_image", required=True)
    parser.add_argument("--output", default="outputs/eval_results/intelli_visual_ablation_summary.json")
    args = parser.parse_args()

    paths = {
        "normal": Path(args.normal),
        "text_only": Path(args.text_only),
        "wrong_image": Path(args.wrong_image),
        "blank_image": Path(args.blank_image),
    }
    summary = {name: load_metric(path) for name, path in paths.items()}
    normal_f1 = float(summary["normal"].get("external_answer_f1", 0.0))
    control_max_f1 = max(
        float(summary["text_only"].get("external_answer_f1", 0.0)),
        float(summary["wrong_image"].get("external_answer_f1", 0.0)),
        float(summary["blank_image"].get("external_answer_f1", 0.0)),
    )
    control_refusal_rate = round(
        (
            float(summary["text_only"].get("external_refusal_rate", 0.0))
            + float(summary["wrong_image"].get("external_refusal_rate", 0.0))
            + float(summary["blank_image"].get("external_refusal_rate", 0.0))
        )
        / 3,
        4,
    )
    summary["_visual_dependency"] = {
        "normal_external_answer_f1": round(normal_f1, 4),
        "control_max_external_answer_f1": round(control_max_f1, 4),
        "visual_dependency_gap": round(normal_f1 - control_max_f1, 4),
        "normal_refusal_rate": round(float(summary["normal"].get("external_refusal_rate", 0.0)), 4),
        "text_only_refusal_rate": round(float(summary["text_only"].get("external_refusal_rate", 0.0)), 4),
        "wrong_image_refusal_rate": round(float(summary["wrong_image"].get("external_refusal_rate", 0.0)), 4),
        "blank_image_refusal_rate": round(float(summary["blank_image"].get("external_refusal_rate", 0.0)), 4),
        "control_refusal_rate": control_refusal_rate,
    }
    for name, metrics in summary.items():
        if isinstance(metrics, dict) and "external_answer_f1" in metrics:
            metrics["external_answer_f1_delta_vs_normal"] = round(float(metrics.get("external_answer_f1", 0.0)) - normal_f1, 4)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("+-------------+--------------+---------------------+------------+------------+")
    print("| setting     | json_validity | external_answer_f1  | refusal    | avg_reward |")
    print("+-------------+--------------+---------------------+------------+------------+")
    for name, metrics in summary.items():
        if name.startswith("_"):
            continue
        print(
            f"| {name:<11} | {float(metrics.get('json_validity', 0.0)):<12.4f} | "
            f"{float(metrics.get('external_answer_f1', 0.0)):<19.4f} | "
            f"{float(metrics.get('external_refusal_rate', 0.0)):<10.4f} | "
            f"{float(metrics.get('avg_reward', 0.0)):<10.4f} |"
        )
    print("+-------------+--------------+---------------------+------------+------------+")
    print(
        "visual_dependency_gap="
        f"{summary['_visual_dependency']['visual_dependency_gap']:.4f} "
        "(normal_f1 - max(text_only, wrong_image, blank_image))"
    )
    print(
        "control_refusal_rate="
        f"{summary['_visual_dependency']['control_refusal_rate']:.4f}; "
        "normal_refusal_rate="
        f"{summary['_visual_dependency']['normal_refusal_rate']:.4f}"
    )
    print(f"wrote ablation summary to {args.output}")


if __name__ == "__main__":
    main()
