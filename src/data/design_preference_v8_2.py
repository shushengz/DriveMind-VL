"""Generate an offline design proposal for Preference-v8.2 from Stage 8 errors."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

R3 = "sft_v3_r3_lingo_smoke"
STEP25 = "dpo_v8_1_lingo_smoke_step25"
STEP50 = "dpo_v8_1_lingo_smoke_step50"
PAIR_RATIOS = {
    "normal_anchor_gold_vs_model_wrong": 0.35,
    "normal_gold_vs_refusal": 0.05,
    "control_direct_answer_vs_caution": 0.20,
    "blank_high_f1_vs_caution": 0.15,
    "text_only_gold_overlap_vs_caution": 0.10,
    "wrong_image_gold_overlap_vs_caution": 0.10,
    "spatial_gold_vs_spatial_wrong": 0.05,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def indexed(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["model_name"], row["setting"]): row for row in rows}


def f(row: dict[str, str] | None, field: str) -> float:
    return float((row or {}).get(field, 0) or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Design Preference-v8.2 without constructing training pairs.")
    parser.add_argument("--behavior_by_setting", default="outputs/final_report/stage8_5_control_behavior_by_setting.csv")
    parser.add_argument("--regression", default="outputs/final_report/stage8_5_case_gap_regression.csv")
    parser.add_argument("--fix_cases", default="outputs/final_report/stage8_5_dpo_fix_cases.csv")
    parser.add_argument("--preference_stats", default="data/train/preference_v8_1/preference_v8_1_stats.json")
    parser.add_argument("--preference_audit", default="outputs/data_audit/preference_v8_1_audit.json")
    parser.add_argument("--output_md", default="outputs/final_report/preference_v8_2_design.md")
    parser.add_argument("--output_json", default="outputs/final_report/preference_v8_2_design.json")
    parser.add_argument("--config_output", default="configs/dpo_v8_2_lingo_smoke.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Accepted for CPU-only orchestration; use isolated output paths for previews.")
    args = parser.parse_args()
    settings = indexed(read_csv(Path(args.behavior_by_setting)))
    regressions = read_csv(Path(args.regression))
    fixes = read_csv(Path(args.fix_cases))
    v81_stats = json.loads(Path(args.preference_stats).read_text(encoding="utf-8"))
    v81_audit = json.loads(Path(args.preference_audit).read_text(encoding="utf-8"))
    setting_regressions = Counter(row["main_regression_setting"] for row in regressions)
    reasons = Counter(row["regression_reason"] for row in regressions)
    f1_deltas = {
        setting: f(settings.get((STEP50, setting)), "avg_f1") - f(settings.get((R3, setting)), "avg_f1")
        for setting in ("text_only", "wrong_image", "blank_image")
    }
    primary = max(f1_deltas, key=f1_deltas.get)
    design: dict[str, Any] = {
        "source_stage": "stage8_5_offline_error_attribution",
        "formal_pairs_generated": False,
        "initialize_from": "checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/",
        "reference_adapter": "checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/",
        "primary_regression_setting": primary,
        "step50_control_f1_delta_vs_r3": f1_deltas,
        "top50_regression_setting_counts": dict(setting_regressions),
        "top50_regression_reason_counts": dict(reasons),
        "fix_case_count": len(fixes),
        "v8_1_ratios": {
            "normal": v81_stats.get("normal_pair_ratio"),
            "control": v81_stats.get("control_pair_ratio"),
            "spatial": v81_stats.get("spatial_pair_ratio"),
        },
        "v8_1_train_ready": v81_audit.get("train_ready"),
        "recommended_pair_ratios": PAIR_RATIOS,
        "recommended_groups": {"normal_related": 0.40, "control_related": 0.55, "spatial": 0.05},
        "principles": [
            "Mine direct-answer and high-F1 control overlap, not hallucination triggers alone.",
            "Increase normal anchors from 26 percent to 40 percent.",
            "Reduce control share from 67 percent to 55 percent.",
            "Start from r3 with a frozen r3 reference, not from DPO-v8.1.",
        ],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Preference-v8.2 设计草案",
        "",
        "本文件仅定义下一轮数据设计，不生成正式 preference pairs，也不触发训练或推理。",
        "",
        "## Stage 8.5 证据",
        "",
        f"- step-50 相对 r3 的主要 control F1 上升来源：`{primary}`。",
        f"- F1 delta (text-only / wrong-image / blank-image): {f1_deltas['text_only']:.4f} / {f1_deltas['wrong_image']:.4f} / {f1_deltas['blank_image']:.4f}。",
        f"- Top-50 gap regression settings: {dict(setting_regressions)}。",
        f"- Top-50 regression reasons: {dict(reasons)}。",
        "",
        "## Pair 比例建议",
        "",
        "| Pair Type | Ratio |",
        "| --- | ---: |",
    ]
    lines += [f"| `{name}` | {ratio:.0%} |" for name, ratio in PAIR_RATIOS.items()]
    lines += [
        "",
        "- Normal-related: 40%",
        "- Control-related: 55%",
        "- Spatial: 5%",
        "",
        "## 设计决策",
        "",
        "- 将训练重点从仅压制 hallucination 触发词，转为直接约束 control direct answer 与 high-F1 gold overlap。",
        "- normal anchor 从 v8.1 的 26% 提高到 40%，避免继续损害正常视觉问答能力。",
        "- control pair 从 v8.1 的 67% 降至 55%，并单独加强 blank/text/wrong-image overlap 样本。",
        "- v8.2 仍从 r3 policy 与 frozen r3 reference 开始；不从 DPO-v8.1 checkpoint 继续。",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = """model_path: /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
init_adapter: checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/
reference_adapter: checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/
preference_file: data/train/preference_v8_2/preference_v8_2_pairs.jsonl
output_dir: checkpoints/qwen25vl_lora_dpo_v8_2_lingo_smoke/
log_dir: outputs/train_logs/dpo_v8_2_lingo_smoke/

method: dpo
reference_free: false
beta: 0.05
learning_rate: 5.0e-8
max_steps: 25
save_steps: 25
eval_steps: 25
batch_size: 1
gradient_accumulation_steps: 8
bf16: true
qlora: true
gradient_checkpointing: true
max_new_tokens: 64
max_pixels: 200704
"""
    config_path = Path(args.config_output)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config, encoding="utf-8")
    print(json.dumps({"primary_regression_setting": primary, "recommended_pair_ratios": PAIR_RATIOS, "config_output": config_path.as_posix()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
