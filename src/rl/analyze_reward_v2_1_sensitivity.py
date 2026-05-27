"""Sensitivity analysis for calibrated reward harness v2.1."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.reward_vc_grpo_lite_v2_1 import RewardWeightsV21, compute_reward_v2_1, with_overrides

BASE = RewardWeightsV21()
CONFIGS = {
    "default": BASE,
    "stronger_control_penalty": with_overrides(BASE, w_control=1.0, w_wrong=0.9),
    "weaker_blank_penalty": with_overrides(BASE, w_blank=0.50),
    "stronger_blank_penalty": with_overrides(BASE, w_blank=0.80),
    "stronger_camera_object_penalty": with_overrides(BASE, w_camera=0.90, w_object=1.0),
    "stronger_spatial_penalty": with_overrides(BASE, w_spatial=1.0),
    "stronger_normal_reward": with_overrides(BASE, w_normal=1.20),
    "stronger_refusal_penalty": with_overrides(BASE, w_refusal=1.20),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v2.1 lambda sensitivity.")
    parser.add_argument("--records", default="outputs/final_report/reward_v2_eval_records.jsonl")
    parser.add_argument("--output_csv", default="outputs/final_report/grpo_lite_reward_v2_1_sensitivity.csv")
    parser.add_argument("--output_md", default="outputs/final_report/grpo_lite_reward_v2_1_sensitivity.md")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    counts = {"drive": 0, "v8": 0, "v81": 0, "v82": 0, "refusal": 0}
    for name, weights in CONFIGS.items():
        buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
        failure: dict[str, list[float]] = defaultdict(list)
        for record in records:
            result = compute_reward_v2_1(record, weights)
            buckets[(record["dataset"], record["model_name"])].append(result["total_reward"])
            if record["dataset"] == "drivelm" and record["model_name"] == "SFT-v3-r3":
                for tag in record.get("failure_tags", []):
                    if tag in {"spatial_relation_failure", "object_token_failure", "camera_specific_failure"}:
                        failure[tag].append(result["total_reward"])
        lingo = sorted([(m, mean(v)) for (d, m), v in buckets.items() if d == "lingoqa"], key=lambda x: x[1], reverse=True)
        drive = sorted([(m, mean(v)) for (d, m), v in buckets.items() if d == "drivelm"], key=lambda x: x[1], reverse=True)
        ld, dd = dict(lingo), dict(drive)
        checks = {"drive": dd["Base Qwen2.5-VL-3B"] > dd["SFT-v3-r3"],
                  "v8": ld["SFT-v3-r3"] >= ld["DPO-v8"], "v81": ld["SFT-v3-r3"] >= ld["DPO-v8.1 step-25"],
                  "v82": ld["SFT-v3-r3"] >= ld["DPO-v8.2 step-25"]}
        hypothetical = {"id": "h", "dataset": "lingoqa", "model_name": "h", "settings": {
            setting: {"id": "h", "setting": setting, "prediction": "insufficient evidence", "gold": "stop"} for setting in ("normal", "text_only", "wrong_image", "blank_image")}}
        checks["refusal"] = compute_reward_v2_1(hypothetical, weights)["total_reward"] < 0
        for key, ok in checks.items():
            counts[key] += int(ok)
        rows.append({"weight_config": name, "weights": json.dumps(asdict_safe(weights), sort_keys=True),
                     "lingoqa_ranking": " > ".join(m for m, _ in lingo), "drivelm_ranking": " > ".join(m for m, _ in drive),
                     "r3_ge_dpo_v8": checks["v8"], "r3_ge_dpo_v8_1": checks["v81"], "r3_ge_dpo_v8_2": checks["v82"],
                     "drivelm_base_gt_r3": checks["drive"], "normal_refusal_low": checks["refusal"],
                     "spatial_failure_reward": mean(failure["spatial_relation_failure"]) if failure["spatial_relation_failure"] else "",
                     "object_failure_reward": mean(failure["object_token_failure"]) if failure["object_token_failure"] else "",
                     "camera_failure_reward": mean(failure["camera_specific_failure"]) if failure["camera_specific_failure"] else ""})
    fields = list(rows[0])
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    stable = counts["drive"] >= 7 and counts["v8"] >= 7 and counts["v82"] >= 7 and counts["refusal"] == len(CONFIGS)
    lines = ["# Reward v2.1 Sensitivity Analysis", "", "| config | LingoQA ranking | DriveLM ranking | r3>=v8 | r3>=v8.1 | r3>=v8.2 | Base>r3 | spatial reward | object reward | camera reward |",
             "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['weight_config']} | {row['lingoqa_ranking']} | {row['drivelm_ranking']} | {row['r3_ge_dpo_v8']} | {row['r3_ge_dpo_v8_1']} | {row['r3_ge_dpo_v8_2']} | {row['drivelm_base_gt_r3']} | {float(row['spatial_failure_reward']):.4f} | {float(row['object_failure_reward']):.4f} | {float(row['camera_failure_reward']):.4f} |")
    lines += ["", "## 稳定性判断", "", f"- DriveLM Base > r3：{counts['drive']}/8。", f"- LingoQA r3 >= DPO-v8：{counts['v8']}/8。",
              f"- LingoQA r3 >= DPO-v8.2：{counts['v82']}/8。", f"- LingoQA r3 >= DPO-v8.1：{counts['v81']}/8（该项不作为硬门槛，但必须解释）。",
              f"- Normal refusal 始终低 reward：{counts['refusal']}/8。", f"- 基本稳定：{'是' if stable else '否'}。"]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"configs": len(CONFIGS), "counts": counts, "stable": stable}, ensure_ascii=False, indent=2))


def asdict_safe(weights: RewardWeightsV21) -> dict[str, float]:
    return {field: getattr(weights, field) for field in weights.__dataclass_fields__}


if __name__ == "__main__":
    main()
