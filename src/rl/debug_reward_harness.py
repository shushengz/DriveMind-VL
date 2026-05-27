"""Run the offline reward harness on existing predictions or case scores."""

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

from src.eval.metrics_visual_control import CONTROL_SETTINGS, SETTINGS, is_control_hallucination, is_refusal, token_f1
from src.rl.reward_driving_vqa import compute_reward, reward_hacking_warnings


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rows_from_case_scores(path: Path) -> list[dict[str, Any]]:
    rows = []
    for case in read_csv(path):
        for setting in SETTINGS:
            rows.append({"id": case.get("id"), "setting": setting, "dataset": case.get("dataset", ""), "gold": case.get("gold", ""), "prediction": case.get(f"{setting}_prediction", "")})
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader(); writer.writerow(row)


def summarize(rows: list[dict[str, Any]], rewards: list[dict[str, Any]]) -> dict[str, Any]:
    by_setting = defaultdict(list)
    for reward in rewards:
        by_setting[str(reward.get("setting"))].append(reward)
    def avg(setting: str, key: str) -> float:
        vals = [float(r.get(key, 0.0)) for r in by_setting.get(setting, [])]
        return sum(vals) / len(vals) if vals else 0.0
    normal_rows = [row for row in rows if row.get("setting") == "normal"]
    controls = [row for row in rows if row.get("setting") in CONTROL_SETTINGS]
    warnings = reward_hacking_warnings(rewards)
    normal_f1 = sum(token_f1(r.get("prediction", ""), r.get("gold", "")) for r in normal_rows) / len(normal_rows) if normal_rows else 0.0
    control_hall = sum(1 for r in controls if is_control_hallucination(r.get("prediction", ""), str(r.get("setting")))) / len(controls) if controls else 0.0
    over_refusal = sum(1 for r in normal_rows if is_refusal(r.get("prediction", ""))) / len(normal_rows) if normal_rows else 0.0
    return {"normal_avg_reward": avg("normal", "reward"), "text_only_avg_reward": avg("text_only", "reward"), "wrong_image_avg_reward": avg("wrong_image", "reward"), "blank_image_avg_reward": avg("blank_image", "reward"), "normal_f1": normal_f1, "visual_reward_gap": avg("normal", "reward") - max(avg(k, "reward") for k in CONTROL_SETTINGS), "hallucination_rate": control_hall, "over_refusal_rate": over_refusal, "reward_hacking_warning_count": len(warnings), "format_error_rate": sum(1 for r in rewards if r.get("format_error")) / len(rewards) if rewards else 0.0, "average_answer_length": sum(float(r.get("answer_length", 0.0)) for r in rewards) / len(rewards) if rewards else 0.0, "warnings": "; ".join(warnings)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline reward harness, CPU-only.")
    parser.add_argument("--raw_predictions", default="")
    parser.add_argument("--case_scores", default="")
    parser.add_argument("--output_dir", default="outputs/rl_debug")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw_predictions:
        rows = read_jsonl(Path(args.raw_predictions))
    elif args.case_scores:
        rows = rows_from_case_scores(Path(args.case_scores))
    else:
        raise ValueError("provide --raw_predictions or --case_scores")
    if args.dry_run:
        rows = rows[:32]
    grouped = defaultdict(dict)
    for row in rows:
        grouped[str(row.get("id"))][str(row.get("setting"))] = row
    rewards = [{**row, **compute_reward(row, grouped.get(str(row.get("id"))))} for row in rows]
    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "reward_harness_offline.jsonl", rewards)
    summary = summarize(rows, rewards)
    write_csv(out_dir / "reward_harness_summary.csv", summary)
    print(json.dumps({"output_dir": out_dir.as_posix(), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
