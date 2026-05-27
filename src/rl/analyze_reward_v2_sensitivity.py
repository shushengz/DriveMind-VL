"""Check whether reward v2 conclusions survive reasonable lambda changes."""
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

from src.rl.reward_vc_grpo_lite_v2 import RewardWeightsV2, compute_reward_v2, with_overrides

BASE = RewardWeightsV2()
CONFIGS = {
    "default": BASE,
    "stronger_control_penalty": with_overrides(BASE, w_control=1.0, w_wrong=0.9),
    "stronger_blank_penalty": with_overrides(BASE, w_blank=1.1),
    "stronger_spatial_camera_object_penalty": with_overrides(BASE, w_camera=0.75, w_spatial=1.0, w_object=0.9),
    "stronger_normal_reward": with_overrides(BASE, w_normal=1.25),
    "stronger_refusal_penalty": with_overrides(BASE, w_refusal=1.2),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reward v2 lambda sensitivity analysis.")
    parser.add_argument("--records", default="outputs/final_report/reward_v2_eval_records.jsonl")
    parser.add_argument("--output_csv", default="outputs/final_report/grpo_lite_reward_v2_sensitivity.csv")
    parser.add_argument("--output_md", default="outputs/final_report/grpo_lite_reward_v2_sensitivity.md")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    stable_lingo = 0
    stable_drive = 0
    for config_name, weights in CONFIGS.items():
        scored = [compute_reward_v2(record, weights) for record in records]
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        spatial = []
        for record, reward in zip(records, scored):
            grouped[(record["dataset"], record["model_name"])].append(reward["total_reward"])
            if record["dataset"] == "drivelm" and record["model_name"] == "SFT-v3-r3" and "spatial_relation_failure" in record["failure_tags"]:
                spatial.append(reward["total_reward"])
        lingo = sorted([(model, mean(values)) for (dataset, model), values in grouped.items() if dataset == "lingoqa"], key=lambda pair: pair[1], reverse=True)
        drive = sorted([(model, mean(values)) for (dataset, model), values in grouped.items() if dataset == "drivelm"], key=lambda pair: pair[1], reverse=True)
        r3_l = dict(lingo)["SFT-v3-r3"]
        v82 = dict(lingo)["DPO-v8.2 step-25"]
        base_d = dict(drive)["Base Qwen2.5-VL-3B"]
        r3_d = dict(drive)["SFT-v3-r3"]
        lingo_ok = r3_l >= v82
        drive_ok = base_d >= r3_d
        stable_lingo += int(lingo_ok)
        stable_drive += int(drive_ok)
        warning = "" if lingo_ok and drive_ok else "ranking conflicts with final/OOD diagnosis"
        rows.append({
            "weight_config": config_name,
            "weights": json.dumps(weights.__dict__, ensure_ascii=False, sort_keys=True),
            "lingoqa_ranking": " > ".join(model for model, _ in lingo),
            "drivelm_ranking": " > ".join(model for model, _ in drive),
            "lingoqa_r3_reward": r3_l,
            "lingoqa_dpo_v8_2_reward": v82,
            "lingoqa_r3_above_dpo_v8_2": lingo_ok,
            "drivelm_base_reward": base_d,
            "drivelm_r3_reward": r3_d,
            "drivelm_base_above_r3": drive_ok,
            "drivelm_r3_spatial_failure_avg_reward": mean(spatial) if spatial else "",
            "warning": warning,
        })
    fields = list(rows[0])
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    stable = stable_lingo == len(CONFIGS) and stable_drive == len(CONFIGS)
    lines = [
        "# GRPO-lite Reward v2 Lambda Sensitivity", "",
        "| config | LingoQA ranking | DriveLM ranking | r3 >= DPO-v8.2 | DriveLM Base >= r3 | spatial failure reward | warning |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['weight_config']} | {row['lingoqa_ranking']} | {row['drivelm_ranking']} | {row['lingoqa_r3_above_dpo_v8_2']} | {row['drivelm_base_above_r3']} | {float(row['drivelm_r3_spatial_failure_avg_reward']):.4f} | {row['warning']} |")
    lines += [
        "", "## 判断", "",
        f"1. 在 {len(CONFIGS)} 组权重下，r3 >= DPO-v8.2 成立 {stable_lingo}/{len(CONFIGS)} 次；DriveLM Base >= r3 成立 {stable_drive}/{len(CONFIGS)} 次。",
        f"2. 排名稳定性：{'基本稳定' if stable else '存在敏感性，不能据此直接训练'}。",
        "3. 若排名仅在显著增强某个单项后翻转，应警惕 reward 对特定模型/数据集过拟合。",
        "4. 后续如进入 smoke，应以 default 作为起点并保留 failure-specific offline gate；当前是否可进入由 readiness 报告最终决定。",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"configs": len(CONFIGS), "lingoqa_consistent_count": stable_lingo, "drivelm_consistent_count": stable_drive, "stable": stable}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
