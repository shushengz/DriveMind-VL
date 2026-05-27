"""Score existing held-out predictions with the proposed GRPO-lite reward only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.rescore_answer_only import align_settings, load_model_settings
from src.rl.reward_vc_grpo_lite import RewardWeights, compute_case_reward

MODELS = [
    ("sft_v3_r3_lingo_smoke", "SFT-v3-r3"),
    ("dpo_v8_lingo_smoke", "DPO-v8"),
    ("dpo_v8_1_lingo_smoke_step25", "DPO-v8.1 step-25"),
    ("dpo_v8_2_lingo_smoke_step25", "DPO-v8.2 step-25"),
]
FIELDS = ["model_name", "id", "normal_reward", "control_penalty", "blank_penalty", "text_penalty", "wrong_penalty", "normal_refusal_penalty", "length_penalty", "total_reward", "normal_f1", "text_only_f1", "wrong_image_f1", "blank_image_f1", "max_control_f1", "normal_refusal", "reward_tags"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug GRPO-lite reward over saved held-out predictions.")
    parser.add_argument("--prediction_root", default="outputs/predictions_heldout")
    parser.add_argument("--output_debug", default="outputs/final_report/grpo_lite_reward_debug.csv")
    parser.add_argument("--output_summary", default="outputs/final_report/grpo_lite_reward_summary.csv")
    parser.add_argument("--output_diagnosis", default="outputs/final_report/grpo_lite_reward_diagnosis.md")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    weights = RewardWeights()
    rows, summaries = [], []
    for internal, display in MODELS:
        aligned = align_settings(load_model_settings(Path(args.prediction_root), "lingoqa", internal, "strict_visual"))
        rewards = []
        for group in aligned:
            reward = compute_case_reward(group, weights)
            rewards.append(reward)
            rows.append({**reward, "model_name": display, "reward_tags": ",".join(reward["reward_tags"])})
        tags = Counter(tag for reward in rewards for tag in reward["reward_tags"])
        summaries.append({
            "model_name": display,
            "num_samples": len(rewards),
            "mean_total_reward": mean(item["total_reward"] for item in rewards),
            "mean_normal_reward": mean(item["normal_reward"] for item in rewards),
            "mean_control_penalty": mean(item["control_penalty"] for item in rewards),
            "mean_blank_penalty": mean(item["blank_penalty"] for item in rewards),
            "mean_text_penalty": mean(item["text_penalty"] for item in rewards),
            "mean_wrong_penalty": mean(item["wrong_penalty"] for item in rewards),
            "normal_refusal_count": sum(item["normal_refusal"] for item in rewards),
            "blank_high_f1_cases": tags["blank_high_f1"],
            "control_high_f1_cases": tags["control_high_f1"],
        })
    summaries.sort(key=lambda item: item["mean_total_reward"], reverse=True)
    best = summaries[0]["model_name"]
    r3_best = best == "SFT-v3-r3"
    buckets = {
        "normal correct + control low F1": [row["total_reward"] for row in rows if row["normal_f1"] >= 0.20 and row["max_control_f1"] < 0.20],
        "normal correct + blank high F1": [row["total_reward"] for row in rows if row["normal_f1"] >= 0.20 and row["blank_image_f1"] >= 0.20],
        "normal wrong + control high F1": [row["total_reward"] for row in rows if row["normal_f1"] < 0.20 and row["max_control_f1"] >= 0.20],
        "normal refusal": [row["total_reward"] for row in rows if row["normal_refusal"]],
    }
    sanity = {name: {"count": len(values), "mean_reward": mean(values) if values else None} for name, values in buckets.items()}
    if args.dry_run:
        print(json.dumps({"best_by_reward": best, "r3_best": r3_best, "summaries": summaries, "sanity": sanity}, ensure_ascii=False, indent=2))
        return
    Path(args.output_debug).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_debug).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in FIELDS} for row in rows])
    summary_fields = list(summaries[0])
    with Path(args.output_summary).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# GRPO-lite Offline Reward Harness Diagnosis", "",
        "This is offline scoring over existing held-out predictions only. No model output was generated and no training was run.", "",
        "## Model Summary", "", "| model | mean reward | normal reward | control penalty | blank penalty | text penalty | wrong penalty |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(f"| {item['model_name']} | {item['mean_total_reward']:.4f} | {item['mean_normal_reward']:.4f} | {item['mean_control_penalty']:.4f} | {item['mean_blank_penalty']:.4f} | {item['mean_text_penalty']:.4f} | {item['mean_wrong_penalty']:.4f} |")
    lines += ["", "## Sanity Checks", ""]
    for name, item in sanity.items():
        value = "n/a" if item["mean_reward"] is None else f"{item['mean_reward']:.4f}"
        lines.append(f"- {name}: n={item['count']}, mean_reward={value}")
    lines += [
        "", "## Conclusions", "",
        f"1. Reward 是否能解释 r3 是当前 best：{'是' if r3_best else '否，需要调整 lambda 后才能考虑训练'}；当前 reward 排名第一为 `{best}`。",
        "2. Reward 会对 blank/control high-F1 施加明确惩罚；DPO-v8.2 的 control overlap 不会因 hallucination 略低而被忽略。",
        "3. Normal refusal 使用高权重独立惩罚，因此 harness 不鼓励通过全拒答投机；仍需在未来 sampling 输出上复查。",
        "4. 如果排名或案例顺序与人工诊断冲突，应优先调整 `lambda_control_f1`、`lambda_blank` 与 `lambda_refusal`，而不是启动 GRPO。",
        "5. 当前不进入 GRPO-lite 训练；先扩充独立 held-out 并运行 DriveLM OOD 评测。",
    ]
    Path(args.output_diagnosis).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"best_by_reward": best, "r3_best": r3_best, "summaries": summaries, "sanity": sanity}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
