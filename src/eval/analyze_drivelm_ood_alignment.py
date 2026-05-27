"""Validate Stage 13 DriveLM OOD prediction alignment before attribution."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.drivelm_ood_attribution_utils import BASE, R3, CONTROL_SETTINGS, PREDICTION_ROOT, groups
from src.eval.rescore_answer_only import SETTINGS, score_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Stage 13 DriveLM OOD prediction alignment.")
    parser.add_argument("--prediction_root", default=PREDICTION_ROOT.as_posix())
    parser.add_argument("--output_json", default="outputs/final_report/drivelm_ood_alignment_check.json")
    parser.add_argument("--output_md", default="outputs/final_report/drivelm_ood_alignment_check.md")
    args = parser.parse_args()
    root = Path(args.prediction_root)
    model_groups = {model: groups(model, root) for model in (BASE, R3)}
    anomalies: dict[str, list[str]] = {
        "model_id_mismatch": [], "missing_image_labels_key": [], "text_only_has_images": [],
        "wrong_image_equals_normal": [], "blank_image_not_placeholder": [], "answer_parse_failure": [],
    }
    id_sets = {model: set(rows) for model, rows in model_groups.items()}
    anomalies["model_id_mismatch"] = sorted(id_sets[BASE] ^ id_sets[R3])
    counts: dict[str, dict[str, int]] = {}
    parse_success: dict[str, dict[str, float]] = {}
    for model, cases in model_groups.items():
        counts[model] = {setting: len(cases) for setting in SETTINGS}
        parsed = {setting: 0 for setting in SETTINGS}
        for sample_id, group in cases.items():
            for setting in SETTINGS:
                if "image_labels" not in group[setting]:
                    anomalies["missing_image_labels_key"].append(f"{model}:{setting}:{sample_id}")
                if score_prediction(group[setting], "answer_only")["parse_success"]:
                    parsed[setting] += 1
                else:
                    anomalies["answer_parse_failure"].append(f"{model}:{setting}:{sample_id}")
            if group["text_only"].get("image_paths"):
                anomalies["text_only_has_images"].append(f"{model}:{sample_id}")
            if group["wrong_image"].get("image_paths") == group["normal"].get("image_paths"):
                anomalies["wrong_image_equals_normal"].append(f"{model}:{sample_id}")
            blank_paths = [str(path).lower() for path in group["blank_image"].get("image_paths", [])]
            if not blank_paths or not all(any(term in path for term in ("blank", "placeholder", "empty")) for path in blank_paths):
                anomalies["blank_image_not_placeholder"].append(f"{model}:{sample_id}")
        parse_success[model] = {setting: parsed[setting] / len(cases) if cases else 0.0 for setting in SETTINGS}
    required_failures = {key: values for key, values in anomalies.items() if key != "answer_parse_failure" and values}
    warnings = []
    if anomalies["answer_parse_failure"]:
        warnings.append("Some raw outputs are not strict JSON; answer-only parser fallback was used and cases remain traceable.")
    passed = not required_failures and all(len(cases) == 100 for cases in model_groups.values())
    result = {
        "passed": passed, "prediction_root": root.as_posix(), "counts": counts,
        "base_r3_id_aligned": not anomalies["model_id_mismatch"],
        "parse_success_rate": parse_success,
        "anomalies": {key: {"count": len(values), "example_ids": values[:10]} for key, values in anomalies.items()},
        "warnings": warnings,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DriveLM OOD Prediction Alignment Check", "",
        f"- 分析前置检查：{'通过' if passed else '未通过'}。",
        f"- Base 与 r3 ID 完全对齐：{'是' if result['base_r3_id_aligned'] else '否'}。",
        f"- Base/r3 样本数：{len(model_groups[BASE])}/{len(model_groups[R3])}。",
        "", "## 异常检查", "",
        "| check | count |", "| --- | ---: |",
    ]
    lines.extend(f"| {key} | {len(values)} |" for key, values in anomalies.items())
    lines += ["", "## Parser Success Rate", "", "| model | normal | text_only | wrong_image | blank_image |", "| --- | ---: | ---: | ---: | ---: |"]
    for model, rates in parse_success.items():
        lines.append(f"| {model} | {rates['normal']:.4f} | {rates['text_only']:.4f} | {rates['wrong_image']:.4f} | {rates['blank_image']:.4f} |")
    if warnings:
        lines += ["", "## Warnings", *[f"- {warning}" for warning in warnings]]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
