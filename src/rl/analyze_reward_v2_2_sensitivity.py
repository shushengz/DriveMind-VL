"""Sensitivity analysis for the human-reviewed Reward v2.2 patch."""
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

from src.rl.reward_vc_grpo_lite_v2_2 import RewardWeightsV22, compute_reward_v2_2, with_overrides

BASE = RewardWeightsV22()
CONFIGS = {
    "default": BASE,
    "stronger_control_penalty": with_overrides(BASE, w_control=1.0, w_wrong=0.9),
    "weaker_blank_penalty": with_overrides(BASE, w_blank=0.50),
    "stronger_blank_penalty": with_overrides(BASE, w_blank=0.80),
    "stronger_camera_object_penalty": with_overrides(BASE, w_camera=0.90, w_object=1.0),
    "stronger_spatial_penalty": with_overrides(BASE, w_spatial=1.0),
    "stronger_normal_reward": with_overrides(BASE, w_normal=1.20),
    "stronger_refusal_penalty": with_overrides(BASE, w_refusal=1.20),
    "stronger_object_invariant_penalty": with_overrides(BASE, w_obj_inv=1.0),
    "stronger_invalid_generic_penalty": with_overrides(BASE, w_invalid=1.1),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="outputs/final_report/reward_v2_eval_records.jsonl")
    parser.add_argument("--csv", default="outputs/final_report/grpo_lite_reward_v2_2_sensitivity.csv")
    parser.add_argument("--md", default="outputs/final_report/grpo_lite_reward_v2_2_sensitivity.md")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    passes = defaultdict(int)
    for config, weights in CONFIGS.items():
        buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
        failures: dict[str, list[float]] = defaultdict(list)
        for record in records:
            result = compute_reward_v2_2(record, weights)
            buckets[(record["dataset"], record["model_name"])].append(result["total_reward"])
            if record["dataset"] == "drivelm" and record["model_name"] == "SFT-v3-r3":
                for tag in record.get("failure_tags", []):
                    if tag in {"spatial_relation_failure", "object_token_failure", "camera_specific_failure"}:
                        failures[tag].append(result["total_reward"])
        lingo = sorted(((model, mean(values)) for (dataset, model), values in buckets.items() if dataset == "lingoqa"), key=lambda x: x[1], reverse=True)
        drive = sorted(((model, mean(values)) for (dataset, model), values in buckets.items() if dataset == "drivelm"), key=lambda x: x[1], reverse=True)
        ld, dd = dict(lingo), dict(drive)
        checks = {
            "drive": dd["Base Qwen2.5-VL-3B"] > dd["SFT-v3-r3"],
            "v8": ld["SFT-v3-r3"] >= ld["DPO-v8"],
            "v81": ld["SFT-v3-r3"] >= ld["DPO-v8.1 step-25"],
            "v82": ld["SFT-v3-r3"] >= ld["DPO-v8.2 step-25"],
        }
        hypothetical = {"id": "h", "dataset": "lingoqa", "model_name": "h", "settings": {
            setting: {"id": "h", "setting": setting, "prediction": "insufficient evidence", "gold": "stop"} for setting in ("normal", "text_only", "wrong_image", "blank_image")
        }}
        checks["refusal"] = compute_reward_v2_2(hypothetical, weights)["total_reward"] < 0
        for key, passed in checks.items():
            passes[key] += int(passed)
        rows.append({
            "weight_config": config,
            "weights": json.dumps(weights.__dict__, sort_keys=True),
            "lingoqa_ranking": " > ".join(model for model, _ in lingo),
            "drivelm_ranking": " > ".join(model for model, _ in drive),
            "r3_ge_dpo_v8": checks["v8"], "r3_ge_dpo_v8_1": checks["v81"], "r3_ge_dpo_v8_2": checks["v82"],
            "drivelm_base_gt_r3": checks["drive"], "normal_refusal_low": checks["refusal"],
            "spatial_failure_reward": mean(failures["spatial_relation_failure"]) if failures["spatial_relation_failure"] else "",
            "object_failure_reward": mean(failures["object_token_failure"]) if failures["object_token_failure"] else "",
            "camera_failure_reward": mean(failures["camera_specific_failure"]) if failures["camera_specific_failure"] else "",
        })
    with Path(args.csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    stable = passes["drive"] >= 8 and passes["v8"] >= 8 and passes["v82"] >= 8 and passes["refusal"] == len(CONFIGS)
    lines = ["# Reward v2.2 Sensitivity", "", "| config | LingoQA ranking | DriveLM ranking | r3>=v8 | r3>=v8.1 | r3>=v8.2 | Base>r3 | object failure reward |",
             "| --- | --- | --- | --- | --- | --- | --- | ---: |"]
    for row in rows:
        lines.append(f"| {row['weight_config']} | {row['lingoqa_ranking']} | {row['drivelm_ranking']} | {row['r3_ge_dpo_v8']} | {row['r3_ge_dpo_v8_1']} | {row['r3_ge_dpo_v8_2']} | {row['drivelm_base_gt_r3']} | {float(row['object_failure_reward']):.4f} |")
    lines += ["", "## 稳定性判断", "",
              f"- DriveLM Base > r3：{passes['drive']}/{len(CONFIGS)}。",
              f"- LingoQA r3 >= DPO-v8：{passes['v8']}/{len(CONFIGS)}。",
              f"- LingoQA r3 >= DPO-v8.2：{passes['v82']}/{len(CONFIGS)}。",
              f"- LingoQA r3 >= DPO-v8.1：{passes['v81']}/{len(CONFIGS)}。",
              f"- Normal refusal 始终为低 reward：{passes['refusal']}/{len(CONFIGS)}。",
              f"- 基本稳定：{'是' if stable else '否'}。", ""]
    Path(args.md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"configs": len(CONFIGS), "passes": dict(passes), "stable": stable}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
