"""Pairwise reward sanity checks including manual-review patch failure modes."""
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

from src.rl.reward_vc_grpo_lite_v2_2 import compute_pairwise_preference_reward, compute_reward_v2_2


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(record: dict[str, Any]) -> dict[str, Any]:
    return compute_reward_v2_2(record)


def max_control(record: dict[str, Any]) -> float:
    return max(float(record["answer_only_f1"][setting]) for setting in ("text_only", "wrong_image", "blank_image"))


def add(pairs: list[dict[str, Any]], kind: str, good: dict[str, Any], bad: dict[str, Any], note: str, synthetic: bool = False) -> None:
    result = compute_pairwise_preference_reward(good, bad)
    pairs.append({
        "pair_type": kind, "id_good": good["id"], "model_good": good["model_name"],
        "id_bad": bad["id"], "model_bad": bad["model_name"], "dataset": good["dataset"],
        "actual_preferred": result["preferred"], "correct": result["preferred"] == "a",
        "reward_good": result["reward_a"], "reward_bad": result["reward_b"], "margin": result["margin"],
        "synthetic_sanity_pair": synthetic, "note": note,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="outputs/final_report/reward_v2_eval_records.jsonl")
    parser.add_argument("--review", default="outputs/final_report/reward_v2_manual_review_sheet_reviewed.csv")
    parser.add_argument("--csv", default="outputs/final_report/grpo_lite_reward_v2_2_pairwise.csv")
    parser.add_argument("--md", default="outputs/final_report/grpo_lite_reward_v2_2_pairwise_summary.md")
    args = parser.parse_args()
    records = load_records(Path(args.records))
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[(record["dataset"], record["model_name"])].append(record)
    pairs: list[dict[str, Any]] = []
    for key, items in buckets.items():
        good = [r for r in items if float(r["answer_only_f1"]["normal"]) >= .20 and float(r["case_gap"]) > .05 and max_control(r) < .30]
        bad = [r for r in items if float(r["case_gap"]) < -.10 and max_control(r) >= .20]
        for a, b in zip(sorted(good, key=lambda r: float(r["case_gap"]), reverse=True)[:6], sorted(bad, key=lambda r: float(r["case_gap"]))[:6]):
            add(pairs, "good_vs_bad_same_dataset", a, b, f"actual records in {key}")
    r3 = {r["id"]: r for r in buckets[("lingoqa", "SFT-v3-r3")]}
    for name in ("DPO-v8", "DPO-v8.2 step-25"):
        dpo = {r["id"]: r for r in buckets[("lingoqa", name)]}
        candidates = [(r3[i], dpo[i]) for i in r3 if i in dpo and float(r3[i]["case_gap"]) > float(dpo[i]["case_gap"]) + .10 and float(r3[i]["answer_only_f1"]["normal"]) >= float(dpo[i]["answer_only_f1"]["normal"]) - .10]
        for a, b in candidates[:10]:
            add(pairs, "r3_good_vs_dpo_bad", a, b, f"same ID against {name}")
    for good in [r for r in records if r["dataset"] == "lingoqa" and float(r["answer_only_f1"]["normal"]) >= .30][:20]:
        bad = deepcopy(good); bad["id"] += "_synthetic_refusal"; bad["model_name"] = "hypothetical_normal_refusal"; bad["settings"]["normal"]["prediction"] = "insufficient evidence"
        add(pairs, "normal_correct_vs_normal_refusal", good, bad, "explicit refusal sanity mutation", True)
    drive = buckets[("drivelm", "SFT-v3-r3")]
    spatial_bad = [r for r in drive if "spatial_relation_failure" in r.get("failure_tags", []) and max_control(r) >= .20]
    spatial_good = [r for r in drive if "spatial_relation_failure" not in r.get("failure_tags", []) and float(r["case_gap"]) > 0]
    for a, b in zip(spatial_good[:15], spatial_bad[:15]):
        add(pairs, "drive_spatial_good_vs_spatial_bad", a, b, "actual DriveLM r3 records")
    for key, items in buckets.items():
        low = [r for r in items if float(r["answer_only_f1"]["blank_image"]) < .20]
        high = [r for r in items if float(r["answer_only_f1"]["blank_image"]) >= .20]
        matches = []
        for good in low[:]:
            choices = sorted(high, key=lambda bad: abs(float(good["answer_only_f1"]["normal"]) - float(bad["answer_only_f1"]["normal"])))
            if choices and abs(float(good["answer_only_f1"]["normal"]) - float(choices[0]["answer_only_f1"]["normal"])) <= .20:
                matches.append((good, choices[0]))
        for good, bad in matches[:8]:
            add(pairs, "blank_low_vs_blank_high", good, bad, f"normal-F1 matched in {key}")
    drive_scored = [(r, score(r)) for r in drive]
    invariant_bad = [r for r, s in drive_scored if s["object_token_invariant_triggered"]]
    grounded_good = [r for r, s in drive_scored if "object_token_failure" in r.get("failure_tags", []) and not s["object_token_invariant_triggered"] and float(r["case_gap"]) >= 0]
    if not grounded_good:
        grounded_good = [r for r, s in drive_scored if not s["object_token_invariant_triggered"] and float(r["case_gap"]) > 0]
    for good, bad in zip(grounded_good[:15], invariant_bad[:15]):
        add(pairs, "object_invariant_bad_vs_object_grounded_good", good, bad, "actual DriveLM records; good has lower invariant risk")
    all_scored = [(r, score(r)) for r in records]
    invalid_bad = [r for r, s in all_scored if s["invalid_generic_answer_triggered"]]
    valid_good = [r for r, s in all_scored if not s["invalid_generic_answer_triggered"] and float(r["case_gap"]) > 0 and float(r["answer_only_f1"]["normal"]) >= .20]
    for good, bad in zip(valid_good[:15], invalid_bad[:15]):
        add(pairs, "invalid_generic_bad_vs_valid_answer_good", good, bad, "actual records")
    same_bad = [r for r, s in all_scored if s["control_same_as_normal_triggered"]]
    caution_good = [r for r, s in all_scored if not s["control_same_as_normal_triggered"] and float(s["valid_control_caution_reward"]) > 0]
    for good, bad in zip(caution_good[:15], same_bad[:15]):
        add(pairs, "control_same_as_normal_bad_vs_control_caution_good", good, bad, "actual records")
    fields = list(pairs[0])
    with Path(args.csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(pairs)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[pair["pair_type"]].append(pair)
    metrics = {kind: {"count": len(items), "accuracy": mean(float(item["correct"]) for item in items), "average_margin": mean(float(item["margin"]) for item in items)} for kind, items in groups.items()}
    overall = mean(float(pair["correct"]) for pair in pairs) if pairs else 0.0
    lines = ["# Reward v2.2 Pairwise Sanity", "", "| pair type | pairs | accuracy | average margin |", "| --- | ---: | ---: | ---: |"]
    for kind, metric in metrics.items():
        lines.append(f"| {kind} | {metric['count']} | {metric['accuracy']:.4f} | {metric['average_margin']:.4f} |")
    lines += ["", f"- Overall pairwise accuracy: {overall:.4f}.",
              "- `normal_correct_vs_normal_refusal` 的 bad 侧为明确标记的 synthetic sanity mutation；其余 pair 来自已有 predictions。", ""]
    Path(args.md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "overall_pairwise_accuracy": overall, "by_pair_type": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
