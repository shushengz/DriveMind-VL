"""Evaluate reward v2.1 on transparent good-vs-bad offline pairs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.reward_vc_grpo_lite_v2_1 import compute_pairwise_preference_reward, compute_reward_v2_1


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def max_control(record: dict[str, Any]) -> float:
    scores = record["answer_only_f1"]
    return max(float(scores[key]) for key in ("text_only", "wrong_image", "blank_image"))


def add_pair(pairs: list[dict[str, Any]], kind: str, a: dict[str, Any], b: dict[str, Any], expected: str = "a", note: str = "", synthetic: bool = False) -> None:
    scored = compute_pairwise_preference_reward(a, b)
    pairs.append({
        "pair_type": kind, "id_a": a["id"], "model_a": a["model_name"], "dataset_a": a["dataset"],
        "id_b": b["id"], "model_b": b["model_name"], "dataset_b": b["dataset"],
        "expected_preferred": expected, "actual_preferred": scored["preferred"],
        "correct": scored["preferred"] == expected, "reward_a": scored["reward_a"],
        "reward_b": scored["reward_b"], "margin": scored["margin"],
        "case_gap_a": a.get("case_gap", ""), "case_gap_b": b.get("case_gap", ""),
        "synthetic_sanity_pair": synthetic, "note": note,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise sanity check for reward v2.1.")
    parser.add_argument("--records", default="outputs/final_report/reward_v2_eval_records.jsonl")
    parser.add_argument("--output_csv", default="outputs/final_report/grpo_lite_reward_v2_1_pairwise.csv")
    parser.add_argument("--output_md", default="outputs/final_report/grpo_lite_reward_v2_1_pairwise_summary.md")
    args = parser.parse_args()
    records = load(Path(args.records))
    pairs: list[dict[str, Any]] = []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[(record["dataset"], record["model_name"])].append(record)

    for key, items in buckets.items():
        good = sorted([r for r in items if float(r["answer_only_f1"]["normal"]) >= 0.20 and float(r["case_gap"]) > 0.05 and max_control(r) < 0.30], key=lambda r: float(r["case_gap"]), reverse=True)
        bad = sorted([r for r in items if float(r["case_gap"]) < -0.10 and max_control(r) >= 0.20], key=lambda r: float(r["case_gap"]))
        for a, b in zip(good[:6], bad[:6]):
            add_pair(pairs, "good_vs_bad_same_dataset", a, b, note=f"same source group {key}")

    r3 = {r["id"]: r for r in buckets[("lingoqa", "SFT-v3-r3")]}
    for dpo_name in ("DPO-v8", "DPO-v8.2 step-25"):
        dpo = {r["id"]: r for r in buckets[("lingoqa", dpo_name)]}
        candidates = [(r3[i], dpo[i]) for i in r3 if i in dpo and float(r3[i]["case_gap"]) > float(dpo[i]["case_gap"]) + 0.10 and float(r3[i]["answer_only_f1"]["normal"]) >= float(dpo[i]["answer_only_f1"]["normal"]) - 0.10]
        for a, b in candidates[:10]:
            add_pair(pairs, "r3_good_vs_dpo_bad", a, b, note=f"same ID comparison against {dpo_name}")

    correct = [r for r in records if float(r["answer_only_f1"]["normal"]) >= 0.30 and r["dataset"] == "lingoqa"][:20]
    for good in correct:
        bad = deepcopy(good)
        bad["id"] = f"{good['id']}_synthetic_normal_refusal"
        bad["model_name"] = "hypothetical_normal_refusal"
        bad["settings"]["normal"]["prediction"] = "insufficient evidence"
        add_pair(pairs, "normal_correct_vs_normal_refusal", good, bad, note="bad side is a labeled refusal sanity mutation", synthetic=True)

    drive_r3 = buckets[("drivelm", "SFT-v3-r3")]
    spatial_bad = [r for r in drive_r3 if "spatial_relation_failure" in r.get("failure_tags", []) and max_control(r) >= 0.20]
    spatial_good = [r for r in drive_r3 if "spatial_relation_failure" not in r.get("failure_tags", []) and float(r["case_gap"]) > 0]
    for a, b in zip(spatial_good[:15], spatial_bad[:15]):
        add_pair(pairs, "drive_spatial_good_vs_spatial_bad", a, b, note="good is non-spatial/positive-gap; bad is tagged spatial high-control")

    for key, items in buckets.items():
        low = [r for r in items if float(r["answer_only_f1"]["blank_image"]) < 0.20]
        high = [r for r in items if float(r["answer_only_f1"]["blank_image"]) >= 0.20]
        matches = []
        for good in low:
            options = sorted(high, key=lambda bad: abs(float(good["answer_only_f1"]["normal"]) - float(bad["answer_only_f1"]["normal"])))
            if options and abs(float(good["answer_only_f1"]["normal"]) - float(options[0]["answer_only_f1"]["normal"])) <= 0.20:
                matches.append((good, options[0]))
        for a, b in matches[:8]:
            add_pair(pairs, "blank_low_vs_blank_high", a, b, note=f"normal-F1-matched within {key}")

    fields = list(pairs[0]) if pairs else ["pair_type"]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(pairs)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["pair_type"]].append(pair)
    metrics = {kind: {"count": len(items), "accuracy": mean(float(item["correct"]) for item in items), "average_margin": mean(float(item["margin"]) for item in items), "failures": sum(not item["correct"] for item in items)} for kind, items in grouped.items()}
    overall = mean(float(item["correct"]) for item in pairs) if pairs else 0.0
    lines = ["# GRPO-lite Reward v2.1 Pairwise Sanity Check", "", "构造规则透明记录于 CSV；`normal_correct_vs_normal_refusal` 的 bad side 是明确标注的人工 sanity mutation，其余来自已有 predictions。",
             "", "| pair type | pairs | accuracy | avg margin | failures |", "| --- | ---: | ---: | ---: | ---: |"]
    for kind, item in metrics.items():
        lines.append(f"| {kind} | {item['count']} | {item['accuracy']:.4f} | {item['average_margin']:.4f} | {item['failures']} |")
    lines += ["", f"- Overall pairwise accuracy：{overall:.4f}。",
              f"- Gate: overall>=0.70 {'PASS' if overall >= 0.70 else 'FAIL'}。",
              f"- Gate: blank_low_vs_blank_high>=0.75 {'PASS' if metrics.get('blank_low_vs_blank_high', {}).get('accuracy', 0) >= 0.75 else 'FAIL'}。",
              f"- Gate: normal_correct_vs_normal_refusal>=0.90 {'PASS' if metrics.get('normal_correct_vs_normal_refusal', {}).get('accuracy', 0) >= 0.90 else 'FAIL'}。",
              f"- Gate: drive_spatial_good_vs_spatial_bad>=0.65 {'PASS' if metrics.get('drive_spatial_good_vs_spatial_bad', {}).get('accuracy', 0) >= 0.65 else 'FAIL'}。"]
    failed = [pair for pair in pairs if not pair["correct"]][:10]
    if failed:
        lines += ["", "## Reward 选择错误的示例", "", "| pair type | id_a | id_b | margin | note |", "| --- | --- | --- | ---: | --- |"]
        lines.extend(f"| {p['pair_type']} | {p['id_a']} | {p['id_b']} | {p['margin']:.4f} | {p['note']} |" for p in failed)
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "overall_pairwise_accuracy": overall, "by_pair_type": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
