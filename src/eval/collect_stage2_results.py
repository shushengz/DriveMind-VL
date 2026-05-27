"""Collect Stage 2 strict visual-control summary CSVs into one main table."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import Any
COLUMNS = ["model_name", "dataset", "mode", "num_samples", "normal_f1", "text_only_f1", "wrong_image_f1", "blank_image_f1", "setting_gap", "case_gap", "positive_gap_rate", "control_hallucination_rate", "normal_refusal_rate", "avg_answer_length", "normal_f1_ci_low", "normal_f1_ci_high", "case_gap_ci_low", "case_gap_ci_high"]
def read_one(path: Path) -> dict[str, Any] | None:
    if not path.exists(): return None
    with path.open("r", encoding="utf-8", newline="") as f: rows=list(csv.DictReader(f))
    if not rows: return None
    r=rows[0]
    return {"model_name":r.get("model_name",""),"dataset":r.get("dataset",""),"mode":r.get("mode","strict_visual"),"num_samples":r.get("case_count",r.get("count","")),"normal_f1":r.get("normal_f1",""),"text_only_f1":r.get("text_only_f1",""),"wrong_image_f1":r.get("wrong_image_f1",""),"blank_image_f1":r.get("blank_image_f1",""),"setting_gap":r.get("setting_gap",""),"case_gap":r.get("case_gap",""),"positive_gap_rate":r.get("positive_gap_rate",""),"control_hallucination_rate":r.get("control_hallucination_rate",""),"normal_refusal_rate":r.get("normal_refusal_rate",""),"avg_answer_length":r.get("average_answer_length",""),"normal_f1_ci_low":r.get("normal_f1_ci_low",""),"normal_f1_ci_high":r.get("normal_f1_ci_high",""),"case_gap_ci_low":r.get("case_gap_ci_low",""),"case_gap_ci_high":r.get("case_gap_ci_high","")}
def _split_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--eval_results_dir",default="outputs/eval_results")
    p.add_argument("--output",default="outputs/final_report/stage2_strict_eval_main_results.csv")
    p.add_argument("--include_models", default="", help="Comma-separated model names to include.")
    p.add_argument("--exclude_models", default="", help="Comma-separated model names to exclude.")
    p.add_argument("--min_num_samples", type=int, default=0, help="Drop rows with fewer case samples.")
    args=p.parse_args()
    include_models = _split_csv(args.include_models)
    exclude_models = _split_csv(args.exclude_models)
    rows=[]
    for path in sorted(Path(args.eval_results_dir).glob("*_strict_visual_summary.csv")):
        row=read_one(path)
        if not row:
            continue
        model_name = str(row.get("model_name", ""))
        try:
            num_samples = int(float(str(row.get("num_samples", 0) or 0)))
        except ValueError:
            num_samples = 0
        if include_models and model_name not in include_models:
            continue
        if model_name in exclude_models:
            continue
        if args.min_num_samples and num_samples < args.min_num_samples:
            continue
        rows.append(row)
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
if __name__ == "__main__": main()
