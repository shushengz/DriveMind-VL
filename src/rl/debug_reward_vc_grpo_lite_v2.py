"""Score saved LingoQA/DriveLM outputs with reward harness v2."""
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

from src.rl.reward_vc_grpo_lite_v2 import RewardWeightsV2, compute_reward_v2

COMPONENTS = [
    "normal_reward", "control_high_f1_penalty", "blank_high_f1_penalty",
    "text_direct_penalty", "wrong_image_penalty", "camera_penalty",
    "spatial_penalty", "object_penalty", "normal_refusal_penalty", "length_penalty",
]


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["message"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    result = []
    for group_key, items in groups.items():
        row: dict[str, Any] = dict(zip(keys, group_key))
        row.update({
            "num_samples": len(items),
            "mean_total_reward": mean(item["total_reward"] for item in items),
            **{f"mean_{component}": mean(item[component] for item in items) for component in COMPONENTS},
            "mean_control_penalty": mean(item["control_high_f1_penalty"] for item in items),
            "mean_blank_penalty": mean(item["blank_high_f1_penalty"] for item in items),
            "mean_text_penalty": mean(item["text_direct_penalty"] for item in items),
            "mean_wrong_penalty": mean(item["wrong_image_penalty"] for item in items),
            "mean_camera_penalty": mean(item["camera_penalty"] for item in items),
            "mean_spatial_penalty": mean(item["spatial_penalty"] for item in items),
            "mean_object_penalty": mean(item["object_penalty"] for item in items),
            "mean_refusal_penalty": mean(item["normal_refusal_penalty"] for item in items),
            "mean_case_gap": mean(item["case_gap"] for item in items),
            "control_high_cases": sum("control_high_f1" in item["reward_tags"] for item in items),
            "blank_high_cases": sum("blank_high_f1" in item["reward_tags"] for item in items),
            "normal_refusal_cases": sum("normal_refusal" in item["reward_tags"] for item in items),
        })
        result.append(row)
    return sorted(result, key=lambda row: (row.get("dataset", ""), -row["mean_total_reward"]))


def hypothetical_sanity() -> dict[str, float]:
    def group(normal: str, control: str) -> dict[str, Any]:
        return {
            setting: {"id": "hyp", "setting": setting, "prediction": normal if setting == "normal" else control, "gold": "stop"}
            for setting in ("normal", "text_only", "wrong_image", "blank_image")
        }
    return {
        "normal_correct_control_caution": compute_reward_v2(group("stop", "insufficient evidence"))["total_reward"],
        "normal_refusal_control_caution": compute_reward_v2(group("insufficient evidence", "insufficient evidence"))["total_reward"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline debug for GRPO-lite reward v2.")
    parser.add_argument("--records", default="outputs/final_report/reward_v2_eval_records.jsonl")
    parser.add_argument("--debug", default="outputs/final_report/grpo_lite_reward_v2_debug.csv")
    parser.add_argument("--summary", default="outputs/final_report/grpo_lite_reward_v2_summary.csv")
    parser.add_argument("--by_dataset", default="outputs/final_report/grpo_lite_reward_v2_by_dataset.csv")
    parser.add_argument("--by_failure", default="outputs/final_report/grpo_lite_reward_v2_by_failure_type.csv")
    parser.add_argument("--diagnosis", default="outputs/final_report/grpo_lite_reward_v2_diagnosis.md")
    args = parser.parse_args()
    records = read_records(Path(args.records))
    scored = []
    for record in records:
        reward = compute_reward_v2(record, RewardWeightsV2())
        scored.append({
            **reward,
            "failure_tags": "|".join(reward["failure_tags"]),
            "reward_tags": "|".join(reward["reward_tags"]),
            "question": record.get("question", ""),
            "gold": record.get("gold", ""),
        })
    summary = aggregate(scored, ("dataset", "model_name"))
    by_dataset = aggregate(scored, ("dataset", "model_name"))
    expanded = []
    for row in scored:
        tags = row["failure_tags"].split("|") if row["failure_tags"] else ["untagged"]
        for tag in tags:
            expanded.append({**row, "failure_type": tag})
    by_failure = aggregate(expanded, ("dataset", "model_name", "failure_type"))
    write_csv(Path(args.debug), scored)
    write_csv(Path(args.summary), summary)
    write_csv(Path(args.by_dataset), by_dataset)
    write_csv(Path(args.by_failure), by_failure)

    lingo = [row for row in summary if row["dataset"] == "lingoqa"]
    drive = [row for row in summary if row["dataset"] == "drivelm"]
    sanity = hypothetical_sanity()
    drive_r3 = next(row for row in drive if row["model_name"] == "SFT-v3-r3")
    spatial = next((row for row in by_failure if row["dataset"] == "drivelm" and row["model_name"] == "SFT-v3-r3" and row["failure_type"] == "spatial_relation_failure"), None)
    non_spatial = [row for row in scored if row["dataset"] == "drivelm" and row["model_name"] == "SFT-v3-r3" and "spatial_relation_failure" not in row["failure_tags"]]
    non_spatial_reward = mean(row["total_reward"] for row in non_spatial) if non_spatial else None
    lines = [
        "# GRPO-lite Reward Harness v2 Offline Debug", "",
        "本结果仅对已有 predictions 离线打分，不生成模型输出，不构造训练数据。",
        "", "## LingoQA Ranking", "", "| rank | model | mean total reward | normal reward | control penalty | blank penalty | wrong penalty |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(lingo, 1):
        lines.append(f"| {index} | {row['model_name']} | {row['mean_total_reward']:.4f} | {row['mean_normal_reward']:.4f} | {row['mean_control_high_f1_penalty']:.4f} | {row['mean_blank_high_f1_penalty']:.4f} | {row['mean_wrong_image_penalty']:.4f} |")
    lines += ["", "## DriveLM Ranking", "", "| rank | model | mean total reward | normal reward | camera penalty | spatial penalty | object penalty |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for index, row in enumerate(drive, 1):
        lines.append(f"| {index} | {row['model_name']} | {row['mean_total_reward']:.4f} | {row['mean_normal_reward']:.4f} | {row['mean_camera_penalty']:.4f} | {row['mean_spatial_penalty']:.4f} | {row['mean_object_penalty']:.4f} |")
    lines += [
        "", "## Sanity Checks", "",
        f"- normal correct + control caution reward: {sanity['normal_correct_control_caution']:.4f}",
        f"- normal refusal + control caution reward: {sanity['normal_refusal_control_caution']:.4f}",
        f"- DriveLM r3 camera/spatial/object mean penalties: {drive_r3['mean_camera_penalty']:.4f} / {drive_r3['mean_spatial_penalty']:.4f} / {drive_r3['mean_object_penalty']:.4f}",
    ]
    if spatial:
        lines.append(f"- DriveLM r3 spatial failure mean reward: {spatial['mean_total_reward']:.4f}; non-spatial mean reward: {non_spatial_reward:.4f}.")
    lines += [
        "", "## 初步诊断", "",
        f"1. LingoQA reward 排名：{' > '.join(row['model_name'] for row in lingo)}。",
        f"2. DriveLM reward 排名：{' > '.join(row['model_name'] for row in drive)}；r3 不应仅因 normal F1 增长而被判为 OOD 成功。",
        "3. `normal_refusal` 由独立高权重惩罚保护；hypothetical refusal reward 必须低于正常回答。",
        "4. 最终是否 ready for GRPO-lite 仍需结合 sensitivity 与 case study 判断。",
    ]
    Path(args.diagnosis).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "lingoqa_ranking": [row["model_name"] for row in lingo], "drivelm_ranking": [row["model_name"] for row in drive], "sanity": sanity, "drive_r3_spatial_reward": spatial["mean_total_reward"] if spatial else None, "drive_r3_non_spatial_reward": non_spatial_reward}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
