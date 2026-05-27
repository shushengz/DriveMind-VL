"""Break down Stage 13 DriveLM OOD behavior by question capability."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.build_drivelm_ood_eval_pool import classify_question
from src.eval.build_drivelm_ood_results import behavior
from src.eval.metrics_visual_control import mean
from src.eval.rescore_answer_only import align_settings, load_model_settings, score_prediction, summarize_aligned

CAPABILITIES = ["camera_specific", "object_token", "spatial_relation", "counting", "yes_no", "action_reasoning", "other"]
MODELS = ["base_qwen25vl_3b", "sft_v3_r3_lingo_smoke"]
COLUMNS = [
    "model_name", "capability", "num_samples", "normal_f1", "case_gap",
    "text_only_f1", "wrong_image_f1", "blank_image_f1",
    "control_high_f1_rate", "wrong_image_high_f1_rate", "normal_refusal_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DriveLM OOD capability breakdown.")
    parser.add_argument("--prediction_root", default="outputs/predictions_drivelm_ood")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--output_csv", default="outputs/final_report/drivelm_ood_capability_breakdown.csv")
    parser.add_argument("--output_md", default="outputs/final_report/drivelm_ood_capability_breakdown.md")
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    model_groups: dict[str, list[dict[str, dict[str, Any]]]] = {}
    for model in [name.strip() for name in args.models.split(",") if name.strip()]:
        aligned = align_settings(load_model_settings(Path(args.prediction_root), "drivelm", model, "strict_visual"))
        model_groups[model] = aligned
        buckets: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
        for group in aligned:
            capability = classify_question(str(group["normal"].get("question", "")))
            buckets[capability].append(group)
        for capability in CAPABILITIES:
            groups = buckets.get(capability, [])
            if not groups:
                continue
            summary = summarize_aligned(groups, model, "drivelm", "strict_visual", "answer_only", with_ci=False)
            controls = behavior(groups, "answer_only")
            wrong_rows = [score_prediction(group["wrong_image"], "answer_only") for group in groups]
            all_rows.append({
                "model_name": model,
                "capability": capability,
                "num_samples": len(groups),
                "normal_f1": summary["normal_f1"],
                "case_gap": summary["case_gap"],
                "text_only_f1": summary["text_only_f1"],
                "wrong_image_f1": summary["wrong_image_f1"],
                "blank_image_f1": summary["blank_image_f1"],
                "control_high_f1_rate": controls["control_high_f1_rate_0_20"],
                "wrong_image_high_f1_rate": mean([float(row["f1"] >= 0.20) for row in wrong_rows]),
                "normal_refusal_rate": summary["normal_refusal_rate"],
            })

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    r3 = [row for row in all_rows if row["model_name"] == "sft_v3_r3_lingo_smoke"]
    base = {row["capability"]: row for row in all_rows if row["model_name"] == "base_qwen25vl_3b"}
    supported = [row for row in r3 if row["num_samples"] >= 5]
    hardest = min(supported or r3, key=lambda row: row["normal_f1"]) if r3 else None
    rare_hardest = min(r3, key=lambda row: row["normal_f1"]) if r3 else None
    strongest_prior = max(r3, key=lambda row: row["control_high_f1_rate"]) if r3 else None
    wrong_confound = max(r3, key=lambda row: row["wrong_image_high_f1_rate"]) if r3 else None
    gains = [
        row["capability"] for row in r3
        if row["capability"] in base and row["normal_f1"] > base[row["capability"]]["normal_f1"]
    ]
    lines = [
        "# DriveLM OOD Capability Breakdown",
        "",
        "本表仅分析 DriveLM OOD 100-case strict visual-control 推理结果，不构造训练数据。",
        "",
        "| model | capability | n | normal_f1 | case_gap | text_only_f1 | wrong_image_f1 | blank_image_f1 | control_high_f1 | wrong_high_f1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['model_name']} | {row['capability']} | {row['num_samples']} | "
            f"{row['normal_f1']:.4f} | {row['case_gap']:.4f} | {row['text_only_f1']:.4f} | "
            f"{row['wrong_image_f1']:.4f} | {row['blank_image_f1']:.4f} | "
            f"{row['control_high_f1_rate']:.4f} | {row['wrong_image_high_f1_rate']:.4f} |"
        )
    lines += ["", "## 诊断回答", ""]
    if hardest:
        lines.append(f"1. 在样本量至少 5 的能力类型中，r3 normal F1 最低的是 `{hardest['capability']}`（n={hardest['num_samples']}, normal_f1={hardest['normal_f1']:.4f}）。")
        if rare_hardest and rare_hardest["capability"] != hardest["capability"]:
            lines.append(f"   极小样本桶 `{rare_hardest['capability']}` 的 normal_f1={rare_hardest['normal_f1']:.4f}（n={rare_hardest['num_samples']}），仅作观察，不作为主要难点判断。")
        lines.append(f"2. r3 control high-F1 最强的语言先验类型是 `{strongest_prior['capability']}`（rate={strongest_prior['control_high_f1_rate']:.4f}）。")
        lines.append(f"3. r3 wrong-image confound 最强的类型是 `{wrong_confound['capability']}`（rate={wrong_confound['wrong_image_high_f1_rate']:.4f}）；据此判断是否集中于 camera/object 类型。")
        lines.append(f"4. r3 normal F1 高于 Base 的能力类型：{', '.join(gains) if gains else '无'}。")
    lines.append("5. 该池包含 camera label 与多视角样本；若 camera/object 类型持续弱，应记录为跨数据集视角对齐 limitation。")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "hardest_r3_supported": hardest, "rare_hardest_r3": rare_hardest, "strongest_prior_r3": strongest_prior, "wrong_confound_r3": wrong_confound, "r3_gain_capabilities": gains}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
