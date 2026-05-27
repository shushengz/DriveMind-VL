"""Build Qwen3-VL strict visual-control result tables after future GPU evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.build_drivelm_ood_results import COLUMNS, behavior
from src.eval.rescore_answer_only import align_settings, load_model_settings, summarize_aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_root", default="outputs/predictions_qwen3_vl")
    parser.add_argument("--models", default="qwen3_vl_4b_base")
    parser.add_argument("--datasets", default="lingoqa,drivelm")
    parser.add_argument("--output", default="outputs/final_report/qwen3_vl_base_eval_results.csv")
    parser.add_argument("--diagnosis", default="outputs/final_report/qwen3_vl_base_eval_diagnosis.md")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    warnings = []
    for dataset in [value.strip() for value in args.datasets.split(",") if value.strip()]:
        for model in [value.strip() for value in args.models.split(",") if value.strip()]:
            target = Path(args.prediction_root) / dataset / model / "strict_visual"
            if not target.exists():
                warnings.append(f"predictions absent: {target}; run future GPU base eval first")
                continue
            aligned = align_settings(load_model_settings(Path(args.prediction_root), dataset, model, "strict_visual"))
            for mode in ("raw_full", "answer_only"):
                result = summarize_aligned(aligned, model, dataset, "strict_visual", mode, with_ci=False)
                result.update(behavior(aligned, mode))
                result["eval_split"] = f"{dataset}_qwen3_base"
                rows.append(result)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})
    lines = ["# Qwen3-VL Base Eval Diagnosis", "", "该构建器只汇总后续 GPU eval 的 prediction；Stage 15 不执行推理。"]
    lines += ["", "## Warnings", *[f"- {warning}" for warning in warnings]] if warnings else ["", f"- 汇总结果行数：{len(rows)}。主结论应只看 `answer_only`。"]
    Path(args.diagnosis).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "warnings": warnings, "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
