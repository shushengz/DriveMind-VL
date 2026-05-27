"""Offline dataset/model and failure-level debug for reward v2.1."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.reward_vc_grpo_lite_v2_1 import RewardWeightsV21, compute_reward_v2_1

COMPONENTS = ["normal_reward", "control_high_f1_penalty", "blank_high_f1_penalty", "text_direct_penalty",
              "wrong_image_penalty", "camera_penalty", "spatial_penalty", "object_penalty",
              "normal_refusal_penalty", "length_penalty", "valid_control_caution_reward"]


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["message"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key, "")) for key in keys)].append(row)
    out = []
    for values, items in buckets.items():
        row: dict[str, Any] = dict(zip(keys, values))
        row.update({"num_samples": len(items), "mean_total_reward": mean(item["total_reward"] for item in items),
                    **{f"mean_{key}": mean(item[key] for item in items) for key in COMPONENTS},
                    "mean_case_gap": mean(item["case_gap"] for item in items),
                    "camera_trigger_rate": mean(float(item["camera_grounding_triggered"]) for item in items),
                    "object_trigger_rate": mean(float(item["object_grounding_triggered"]) for item in items),
                    "spatial_trigger_rate": mean(float(item["spatial_grounding_triggered"]) for item in items)})
        out.append(row)
    return sorted(out, key=lambda row: (row.get("dataset", ""), -row["mean_total_reward"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug reward v2.1 over saved evaluation records.")
    parser.add_argument("--records", default="outputs/final_report/reward_v2_eval_records.jsonl")
    parser.add_argument("--debug", default="outputs/final_report/grpo_lite_reward_v2_1_debug.csv")
    parser.add_argument("--summary", default="outputs/final_report/grpo_lite_reward_v2_1_summary.csv")
    parser.add_argument("--by_dataset", default="outputs/final_report/grpo_lite_reward_v2_1_by_dataset.csv")
    parser.add_argument("--by_failure", default="outputs/final_report/grpo_lite_reward_v2_1_by_failure_type.csv")
    parser.add_argument("--diagnosis", default="outputs/final_report/grpo_lite_reward_v2_1_diagnosis.md")
    args = parser.parse_args()
    records = load(Path(args.records))
    rewards = []
    for record in records:
        result = compute_reward_v2_1(record, RewardWeightsV21())
        rewards.append({**result, "failure_tags": "|".join(result["failure_tags"]),
                        "reward_tags": "|".join(result["reward_tags"]),
                        "question": record.get("question", ""), "gold": record.get("gold", "")})
    summary = aggregate(rewards, ("dataset", "model_name"))
    expanded = [{**row, "failure_type": tag} for row in rewards for tag in (row["failure_tags"].split("|") if row["failure_tags"] else ["untagged"])]
    failure = aggregate(expanded, ("dataset", "model_name", "failure_type"))
    write(Path(args.debug), rewards); write(Path(args.summary), summary); write(Path(args.by_dataset), summary); write(Path(args.by_failure), failure)
    lingo = [row for row in summary if row["dataset"] == "lingoqa"]
    drive = [row for row in summary if row["dataset"] == "drivelm"]
    drive_r3 = next(row for row in drive if row["model_name"] == "SFT-v3-r3")
    detail = {
        tag: next((row for row in failure if row["dataset"] == "drivelm" and row["model_name"] == "SFT-v3-r3" and row["failure_type"] == tag), None)
        for tag in ("spatial_relation_failure", "object_token_failure", "camera_specific_failure", "blank_prior_answer")
    }
    lines = ["# GRPO-lite Reward Harness v2.1 Debug", "", "仅对已有预测离线打分，不训练、不推理。", "",
             "## LingoQA Ranking", "", "| rank | model | reward | normal | control | blank | wrong | caution bonus |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for index, row in enumerate(lingo, 1):
        lines.append(f"| {index} | {row['model_name']} | {row['mean_total_reward']:.4f} | {row['mean_normal_reward']:.4f} | {row['mean_control_high_f1_penalty']:.4f} | {row['mean_blank_high_f1_penalty']:.4f} | {row['mean_wrong_image_penalty']:.4f} | {row['mean_valid_control_caution_reward']:.4f} |")
    lines += ["", "## DriveLM Ranking", "", "| rank | model | reward | camera | spatial | object | blank |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for index, row in enumerate(drive, 1):
        lines.append(f"| {index} | {row['model_name']} | {row['mean_total_reward']:.4f} | {row['mean_camera_penalty']:.4f} | {row['mean_spatial_penalty']:.4f} | {row['mean_object_penalty']:.4f} | {row['mean_blank_high_f1_penalty']:.4f} |")
    lines += ["", "## DriveLM r3 Failure Rewards", "", "| failure | n | mean reward | camera | spatial | object |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for tag, row in detail.items():
        if row:
            lines.append(f"| {tag} | {row['num_samples']} | {row['mean_total_reward']:.4f} | {row['mean_camera_penalty']:.4f} | {row['mean_spatial_penalty']:.4f} | {row['mean_object_penalty']:.4f} |")
    lines += ["", "## 诊断", "", f"- LingoQA ranking：`{' > '.join(row['model_name'] for row in lingo)}`。",
              f"- DriveLM ranking：`{' > '.join(row['model_name'] for row in drive)}`。",
              "- v2.1 的结构化 penalty 仅在 control 失败联合出现时触发，因此比 v2 的标签宽口径更适合作为后续审计基础。",
              "- readiness 仍必须由 sensitivity 与 pairwise 检查共同决定。"]
    Path(args.diagnosis).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "lingoqa_ranking": [row["model_name"] for row in lingo], "drivelm_ranking": [row["model_name"] for row in drive],
                      "drive_r3_structural_penalties": {key: (row["mean_total_reward"] if row else None) for key, row in detail.items()},
                      "drive_r3_mean_reward": drive_r3["mean_total_reward"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
