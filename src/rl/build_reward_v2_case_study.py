"""Build human-readable case studies for reward harness v2."""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.reward_vc_grpo_lite_v2 import compute_reward_v2


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_row(record: dict[str, Any], reward: dict[str, Any], category: str, comment: str) -> dict[str, Any]:
    settings = record["settings"]
    max_control = max(reward["text_only_f1"], reward["wrong_image_f1"], reward["blank_image_f1"])
    components = {
        key: round(float(reward[key]), 4)
        for key in (
            "normal_reward", "control_high_f1_penalty", "blank_high_f1_penalty",
            "text_direct_penalty", "wrong_image_penalty", "camera_penalty",
            "spatial_penalty", "object_penalty", "normal_refusal_penalty", "length_penalty",
        )
    }
    return {
        "case_category": category,
        "id": record["id"],
        "dataset": record["dataset"],
        "model_name": record["model_name"],
        "question": record.get("question", ""),
        "gold": record.get("gold", ""),
        "normal_answer": settings["normal"].get("prediction", ""),
        "text_only_answer": settings["text_only"].get("prediction", ""),
        "wrong_image_answer": settings["wrong_image"].get("prediction", ""),
        "blank_image_answer": settings["blank_image"].get("prediction", ""),
        "normal_f1": reward["normal_f1"],
        "control_f1_max": max_control,
        "case_gap": reward["case_gap"],
        "failure_tags": "|".join(record.get("failure_tags", [])),
        "reward_components": json.dumps(components, ensure_ascii=False),
        "total_reward": reward["total_reward"],
        "why_reward_high_or_low": "|".join(reward["reward_tags"]),
        "chinese_comment": comment,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reward v2 case study gallery.")
    parser.add_argument("--records", default="outputs/final_report/reward_v2_eval_records.jsonl")
    parser.add_argument("--output_csv", default="outputs/final_report/grpo_lite_reward_v2_case_study.csv")
    parser.add_argument("--output_html", default="outputs/final_report/grpo_lite_reward_v2_case_study.html")
    args = parser.parse_args()
    records = load_jsonl(Path(args.records))
    evaluated = [(record, compute_reward_v2(record)) for record in records]
    rows: list[dict[str, Any]] = []

    def add(category: str, candidates: list[tuple[dict[str, Any], dict[str, Any]]], reverse: bool, comment: str, limit: int = 3) -> None:
        candidates = sorted(candidates, key=lambda pair: pair[1]["total_reward"], reverse=reverse)[:limit]
        rows.extend(as_row(record, reward, category, comment) for record, reward in candidates)

    add("high_reward_good", [pair for pair in evaluated if pair[1]["normal_f1"] >= 0.20 and pair[1]["case_gap"] > 0], True, "normal 优于控制输入且 reward 较高，是 reward 希望鼓励的视觉依赖行为。")
    add("low_reward_bad", evaluated, False, "多项惩罚叠加导致低 reward，适合作为 reward 反例审计。")
    add("normal_correct_blank_high", [pair for pair in evaluated if pair[1]["normal_f1"] >= 0.20 and pair[1]["blank_image_f1"] >= 0.20], False, "正常回答看似正确，但空白图高重合，应由 blank penalty 拉低奖励。")
    add("drivelm_spatial_failure", [pair for pair in evaluated if "spatial_relation_failure" in pair[0].get("failure_tags", [])], False, "DriveLM 空间关系失败，reward 叠加 spatial 与控制输入惩罚。")
    add("drivelm_object_token_failure", [pair for pair in evaluated if "object_token_failure" in pair[0].get("failure_tags", [])], False, "对象引用 grounding 失败，object penalty 不能被 normal 分数掩盖。")
    add("reward_disagreement_good_metric_low_reward", [pair for pair in evaluated if pair[1]["normal_f1"] >= 0.30 and pair[1]["total_reward"] < 0], False, "normal 指标不错但控制条件也高重合，reward 正确地拒绝只看 normal F1。")
    add("reward_disagreement_high_reward_bad_gap", [pair for pair in evaluated if pair[1]["total_reward"] > 0 and pair[1]["case_gap"] <= 0], True, "reward 较高但 case gap 不佳，需要人工关注潜在漏惩罚。")

    hypothetical = {
        "id": "hypothetical_normal_refusal",
        "dataset": "sanity",
        "model_name": "hypothetical",
        "question": "Should the vehicle stop?",
        "gold": "stop",
        "failure_tags": ["normal_refusal"],
        "settings": {
            setting: {"id": "hypothetical_normal_refusal", "setting": setting, "prediction": "insufficient evidence", "gold": "stop"}
            for setting in ("normal", "text_only", "wrong_image", "blank_image")
        },
    }
    rows.append(as_row(hypothetical, compute_reward_v2(hypothetical), "normal_refusal_penalty_sanity", "这是人工构造的 sanity case，不是 prediction；用于证明 normal 全拒答会被显式压低 reward。"))
    fields = list(rows[0])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    chunks = [
        "<!doctype html><meta charset='utf-8'><title>Reward v2 Case Study</title>",
        "<style>body{font:14px Arial,sans-serif;color:#202124;margin:28px}table{border-collapse:collapse;width:100%;margin:16px 0 30px}"
        "th,td{border:1px solid #ddd;padding:7px;vertical-align:top;white-space:pre-wrap;max-width:330px}th{background:#f4f6f8}"
        ".note{padding:12px;background:#eef4ff;border-left:4px solid #386ee8}</style>",
        "<h1>GRPO-lite Reward Harness v2 Case Study</h1>",
        "<p class='note'>离线审计展示；normal_refusal_penalty_sanity 为人工构造测试，不是模型 prediction。</p>",
    ]
    for category in dict.fromkeys(row["case_category"] for row in rows):
        group = [row for row in rows if row["case_category"] == category]
        chunks.append(f"<h2>{html.escape(category)}</h2><table><tr>")
        chunks.extend(f"<th>{html.escape(field)}</th>" for field in fields)
        chunks.append("</tr>")
        for row in group:
            chunks.append("<tr>")
            chunks.extend(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields)
            chunks.append("</tr>")
        chunks.append("</table>")
    Path(args.output_html).write_text("".join(chunks), encoding="utf-8")
    print(json.dumps({"cases": len(rows), "categories": {category: sum(row["case_category"] == category for row in rows) for category in dict.fromkeys(row["case_category"] for row in rows)}, "csv": args.output_csv, "html": args.output_html}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
