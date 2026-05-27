"""Build Stage 4.5 held-out main table from saved raw predictions only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.build_stage4_main_results import build_common_id_rows
from src.eval.rescore_answer_only import MAIN_COLUMNS, write_csv

MODELS = ["base_qwen25vl_3b", "sft_v2", "dpo_v7", "sft_v3_r2_lingo_smoke", "sft_v3_r3_lingo_smoke"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Stage 4.5 held-out raw_full and answer_only metrics.")
    parser.add_argument("--prediction_root", default="outputs/predictions_heldout")
    parser.add_argument("--dataset", default="lingoqa")
    parser.add_argument("--mode", default="strict_visual")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--output", default="outputs/final_report/stage4_5_heldout_main_results.csv")
    parser.add_argument("--warnings_output", default="outputs/final_report/stage4_5_heldout_main_results_warnings.json")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    rows, warnings = build_common_id_rows(Path(args.prediction_root), args.dataset, args.mode, models, args.dry_run)
    write_csv(Path(args.output), rows, MAIN_COLUMNS)
    warning_path = Path(args.warnings_output)
    warning_path.parent.mkdir(parents=True, exist_ok=True)
    warning_path.write_text(json.dumps({"warnings": warnings}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "rows": len(rows), "warnings": warnings}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
