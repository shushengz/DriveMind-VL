"""Run Reward v2.2 on existing evaluation records only."""
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

from src.rl.reward_vc_grpo_lite_v2_2 import RewardWeightsV22, compute_reward_v2_2

COMPONENTS = [
    "normal_reward", "control_high_f1_penalty", "blank_high_f1_penalty", "text_direct_penalty",
    "wrong_image_penalty", "camera_penalty", "spatial_penalty", "object_penalty",
    "object_token_invariant_penalty", "normal_object_category_mismatch_penalty",
    "invalid_generic_answer_penalty", "control_same_as_normal_penalty",
    "normal_refusal_penalty", "length_penalty", "valid_control_caution_reward",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    output = []
    for values, items in groups.items():
        result: dict[str, Any] = dict(zip(keys, values))
        result.update({"num_samples": len(items), "mean_total_reward": mean(item["total_reward"] for item in items)})
        result.update({f"mean_{key}": mean(float(item[key]) for item in items) for key in COMPONENTS})
        result["mean_case_gap"] = mean(float(item["case_gap"]) for item in items)
        for trigger in ("object_token_invariant_triggered", "normal_object_category_mismatch_triggered", "invalid_generic_answer_triggered", "control_same_as_normal_triggered"):
            result[f"{trigger}_rate"] = mean(float(bool(item[trigger])) for item in items)
        output.append(result)
    return sorted(output, key=lambda row: (row.get("dataset", ""), -row["mean_total_reward"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="outputs/final_report/reward_v2_eval_records.jsonl")
    parser.add_argument("--debug", default="outputs/final_report/grpo_lite_reward_v2_2_debug.csv")
    parser.add_argument("--summary", default="outputs/final_report/grpo_lite_reward_v2_2_summary.csv")
    parser.add_argument("--by_dataset", default="outputs/final_report/grpo_lite_reward_v2_2_by_dataset.csv")
    parser.add_argument("--by_failure", default="outputs/final_report/grpo_lite_reward_v2_2_by_failure_type.csv")
    parser.add_argument("--diagnosis", default="outputs/final_report/grpo_lite_reward_v2_2_diagnosis.md")
    args = parser.parse_args()
    records = load_jsonl(Path(args.records))
    rewards = []
    for record in records:
        result = compute_reward_v2_2(record, RewardWeightsV22())
        rewards.append({
            **result,
            "failure_tags": "|".join(result["failure_tags"]),
            "reward_tags": "|".join(result["reward_tags"]),
            "gold_object_categories": "|".join(result["gold_object_categories"]),
            "answer_object_categories": "|".join(result["answer_object_categories"]),
            "invalid_generic_answer_setting": "|".join(result["invalid_generic_answer_setting"]),
            "invalid_generic_pattern": json.dumps(result["invalid_generic_pattern"], ensure_ascii=False),
            "same_as_normal_settings": "|".join(result["same_as_normal_settings"]),
            "same_as_normal_similarity": json.dumps(result["same_as_normal_similarity"], ensure_ascii=False),
            "weights": json.dumps(result["weights"], sort_keys=True),
            "question": record.get("question", ""),
            "gold": record.get("gold", ""),
        })
    summary = aggregate(rewards, ("dataset", "model_name"))
    expanded = [
        {**row, "failure_type": tag}
        for row in rewards
        for tag in (row["failure_tags"].split("|") if row["failure_tags"] else ["untagged"])
    ]
    failure = aggregate(expanded, ("dataset", "model_name", "failure_type"))
    write_csv(Path(args.debug), rewards)
    write_csv(Path(args.summary), summary)
    write_csv(Path(args.by_dataset), summary)
    write_csv(Path(args.by_failure), failure)
    lingo = [row for row in summary if row["dataset"] == "lingoqa"]
    drive = [row for row in summary if row["dataset"] == "drivelm"]
    lines = ["# GRPO-lite Reward v2.2 Debug", "", "仅对已有 prediction records 进行离线评分；未训练或启动推理。", "",
             "## Model Ranking", "", "| dataset | rank | model | mean reward | invariant | category mismatch | invalid generic | same-as-normal |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset_rows in (lingo, drive):
        for index, row in enumerate(dataset_rows, 1):
            lines.append(f"| {row['dataset']} | {index} | {row['model_name']} | {row['mean_total_reward']:.4f} | {row['mean_object_token_invariant_penalty']:.4f} | {row['mean_normal_object_category_mismatch_penalty']:.4f} | {row['mean_invalid_generic_answer_penalty']:.4f} | {row['mean_control_same_as_normal_penalty']:.4f} |")
    Path(args.diagnosis).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "lingoqa_ranking": [r["model_name"] for r in lingo], "drivelm_ranking": [r["model_name"] for r in drive]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
