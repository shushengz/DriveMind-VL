"""Collect case-level disagreements revealed by reward v2 and sensitivity analysis."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.reward_vc_grpo_lite_v2 import RewardWeightsV2, compute_reward_v2, with_overrides


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row(record: dict[str, Any], reward: dict[str, Any], dtype: str, reason: str, fix: str) -> dict[str, Any]:
    s = record["settings"]
    components = {key: reward[key] for key in (
        "normal_reward", "control_high_f1_penalty", "blank_high_f1_penalty",
        "text_direct_penalty", "wrong_image_penalty", "camera_penalty",
        "spatial_penalty", "object_penalty", "normal_refusal_penalty", "length_penalty",
    )}
    return {
        "id": record["id"], "dataset": record["dataset"], "model_name": record["model_name"],
        "question": record.get("question", ""), "gold": record.get("gold", ""),
        "normal_answer": s["normal"].get("prediction", ""), "text_only_answer": s["text_only"].get("prediction", ""),
        "wrong_image_answer": s["wrong_image"].get("prediction", ""), "blank_image_answer": s["blank_image"].get("prediction", ""),
        "normal_f1": reward["normal_f1"], "text_only_f1": reward["text_only_f1"],
        "wrong_image_f1": reward["wrong_image_f1"], "blank_image_f1": reward["blank_image_f1"],
        "case_gap": reward["case_gap"], "failure_tags": "|".join(record.get("failure_tags", [])),
        "reward_components": json.dumps(components, ensure_ascii=False), "total_reward": reward["total_reward"],
        "disagreement_type": dtype, "suspected_reason": reason, "suggested_fix": fix,
        "chinese_comment": "该案例用于人工确认 reward 是否正确反映视觉依赖，而不是仅复现自动分数。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect reward-v2 disagreement cases.")
    parser.add_argument("--records", default="outputs/final_report/reward_v2_eval_records.jsonl")
    parser.add_argument("--output_csv", default="outputs/final_report/reward_v2_disagreement_cases.csv")
    parser.add_argument("--output_md", default="outputs/final_report/reward_v2_disagreement_cases.md")
    args = parser.parse_args()
    records = load_records(Path(args.records))
    default = RewardWeightsV2()
    stronger_blank = with_overrides(default, w_blank=1.1)
    scored = [(record, compute_reward_v2(record, default)) for record in records]
    indexed = {(record["dataset"], record["model_name"], record["id"]): (record, reward) for record, reward in scored}
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    def add(record: dict[str, Any], reward: dict[str, Any], dtype: str, reason: str, fix: str) -> None:
        selected.append(row(record, reward, dtype, reason, fix))
        counts[dtype] += 1

    lingo_ids = {record["id"] for record, _ in scored if record["dataset"] == "lingoqa" and record["model_name"] == "SFT-v3-r3"}
    r3_v81_deltas = []
    for sample_id in lingo_ids:
        r3_record, r3_reward = indexed[("lingoqa", "SFT-v3-r3", sample_id)]
        _, v81_reward = indexed[("lingoqa", "DPO-v8.1 step-25", sample_id)]
        delta = v81_reward["total_reward"] - r3_reward["total_reward"]
        if delta > 0:
            r3_v81_deltas.append((delta, r3_record, r3_reward))
    for delta, record, reward in sorted(r3_v81_deltas, reverse=True, key=lambda x: x[0])[:15]:
        add(record, reward, "r3_lower_than_dpo_v8_1", f"同 ID 下 v8.1 reward 比 r3 高 {delta:.4f}，常由 r3 blank/control penalty 较高贡献。", "复核 blank 饱和惩罚与 case-gap 权重，不按单模型排名硬调。")

    blank_deltas = []
    for sample_id in lingo_ids:
        r3_record, _ = indexed[("lingoqa", "SFT-v3-r3", sample_id)]
        v82_record, _ = indexed[("lingoqa", "DPO-v8.2 step-25", sample_id)]
        r3 = compute_reward_v2(r3_record, stronger_blank)
        v82 = compute_reward_v2(v82_record, stronger_blank)
        if v82["total_reward"] > r3["total_reward"] and (r3["blank_image_f1"] >= 0.20 or v82["blank_image_f1"] >= 0.20):
            blank_deltas.append((v82["total_reward"] - r3["total_reward"], r3_record, r3))
    for delta, record, reward in sorted(blank_deltas, reverse=True, key=lambda x: x[0])[:15]:
        add(record, reward, "dpo_v8_2_over_r3_under_stronger_blank", f"增强 blank 权重后 v8.2 比 r3 高 {delta:.4f}。", "将 blank penalty 改为分段饱和并重新跑 sensitivity。")

    totals = [reward["total_reward"] for _, reward in scored]
    midpoint = median(totals)
    for record, reward in sorted(scored, key=lambda pair: pair[1]["total_reward"], reverse=True):
        if reward["total_reward"] >= midpoint and reward["case_gap"] < -0.10 and "control_high_f1" in reward["reward_tags"]:
            add(record, reward, "reward_high_but_case_gap_bad", "总体 reward 相对不低但 case_gap 仍显著为负。", "检查 control dominance 与结构化惩罚是否不足。")
            if counts["reward_high_but_case_gap_bad"] >= 10:
                break
    for record, reward in scored:
        if reward["normal_f1"] >= 0.30 and reward["case_gap"] > 0 and reward["total_reward"] < midpoint:
            add(record, reward, "reward_low_but_metrics_ok", "normal 与 case_gap 都较好，但综合 reward 偏低。", "人工确认是否存在过罚组件。")
            if counts["reward_low_but_metrics_ok"] >= 10:
                break
    for record, reward in scored:
        tags = set(record.get("failure_tags", []))
        if record["dataset"] == "drivelm" and "object_token_failure" in tags and reward["object_penalty"] <= 0.25:
            add(record, reward, "object_failure_under_penalized", "object-token failure 的 object penalty 偏低。", "改为 object token 与 control confound 联合触发。")
        if record["dataset"] == "drivelm" and "camera_specific_failure" in tags and reward["wrong_image_f1"] >= 0.20 and reward["camera_penalty"] <= 0.25:
            add(record, reward, "camera_failure_under_penalized", "wrong-image 高重合但 camera penalty 不足。", "改为 camera clue + wrong-image confound 联合触发。")
        if counts["object_failure_under_penalized"] >= 10 and counts["camera_failure_under_penalized"] >= 10:
            break
    for record, reward in scored:
        if 0.18 <= reward["blank_image_f1"] <= 0.32 and "blank_high_f1" in reward["reward_tags"]:
            add(record, reward, "blank_penalty_over_sensitive", "blank F1 落在阈值附近，线性惩罚易影响排名。", "使用有限档位的饱和 blank penalty。")
            if counts["blank_penalty_over_sensitive"] >= 10:
                break
    # Actual normal refusal records are retained if they appear; current saved predictions may contain none.
    for record, reward in scored:
        if "normal_refusal" in reward["reward_tags"]:
            add(record, reward, "normal_refusal_sanity", "实际 normal refusal 需要确认惩罚足够。", "保留独立 normal refusal penalty。")
    fields = list(selected[0]) if selected else ["id"]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(selected)
    lines = [
        "# Reward v2 Disagreement Cases", "",
        f"共收集 {len(selected)} 条 case-type 记录；同一实际案例可因不同 disagreement 原因重复出现。",
        "", "| disagreement type | count |", "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in counts.most_common())
    lines += ["", "## 主要问题", "",
        "- LingoQA 排名冲突主要需要复核 r3 与 DPO-v8.1 的 blank/control 贡献差异。",
        "- `stronger_blank_penalty` 翻转提示 v2 对阈值附近 blank F1 过敏，应使用饱和惩罚而非继续加权。",
        "- DriveLM camera/object 项必须由真实 control confound 联合触发，避免仅因存在标签而宽口径惩罚。",
        "- 所有样本均需人工复核；自动标签是定位工具，不等同于人工结论。",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"disagreement_rows": len(selected), "counts": dict(counts), "r3_v81_positive_delta_cases": len(r3_v81_deltas), "strong_blank_flip_cases": len(blank_deltas)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
