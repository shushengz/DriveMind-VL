"""Best-effort Qwen-VL-Series-Finetune exports for later manual format check."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.drivemind_vl_schema import DriveMindRecord, answer_only_json, excluded_eval_ids, extract_answer, leakage_ids, load_sft_records, make_user_content, normalize_id, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_input", default="data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl")
    parser.add_argument("--dpo_input", default="data/train/preference_v8_2/preference_v8_2_pairs.jsonl")
    parser.add_argument("--sft_output", default="data/qwen3_vl/qwen_vl_series/sft_lingoqa_r3_train.jsonl")
    parser.add_argument("--dpo_output", default="data/qwen3_vl/qwen_vl_series/dpo_preference_v8_2.jsonl")
    parser.add_argument("--audit_json", default="outputs/data_audit/qwen_vl_series_export_audit.json")
    parser.add_argument("--audit_md", default="outputs/data_audit/qwen_vl_series_export_audit.md")
    args = parser.parse_args()
    sft_records = load_sft_records(args.sft_input)
    sft = [{"messages": [{"role": "user", "content": make_user_content(record)}, {"role": "assistant", "content": answer_only_json(record.answer)}], "images": record.image_paths, "id": record.id} for record in sft_records]
    source_pairs = read_jsonl(args.dpo_input)
    dpo_records = []
    dpo = []
    for row in source_pairs:
        rec = DriveMindRecord(str(row["id"]), str(row.get("dataset", "lingoqa")), str(row.get("setting", "normal")), str(row.get("prompt", "")), "", list(row.get("image_paths") or []), list(row.get("image_labels") or []), {"source_id": normalize_id(row.get("source_id") or row["id"])})
        dpo_records.append(rec)
        dpo.append({"messages": [{"role": "user", "content": make_user_content(rec)}], "chosen": answer_only_json(extract_answer(row.get("chosen"))), "rejected": answer_only_json(extract_answer(row.get("rejected"))), "images": rec.image_paths, "id": rec.id})
    write_jsonl(args.sft_output, sft); write_jsonl(args.dpo_output, dpo)
    leaks = leakage_ids(sft_records + dpo_records, excluded_eval_ids())
    audit = {"sft_samples": len(sft), "dpo_pairs": len(dpo), "heldout_leakage_count": len(leaks), "needs_manual_format_check": True, "train_ready": False, "note": "The 2U1 repository documents Qwen3-VL and alignment workflows, but this best-effort export must be checked against the selected commit before launch."}
    Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_json).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.audit_md).write_text("# Qwen-VL-Series-Finetune Export Audit\n\n" + "\n".join(f"- {key}: {value}" for key, value in audit.items()) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
