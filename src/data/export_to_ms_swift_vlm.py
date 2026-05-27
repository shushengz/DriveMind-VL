"""Export DriveMind-VL SFT records to ms-swift/Qwen3-VL messages format."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.drivemind_vl_schema import answer_only_json, excluded_eval_ids, is_answer_only_json, leakage_ids, load_sft_records, make_ms_swift_user_text, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl")
    parser.add_argument("--output", default="data/qwen3_vl/ms_swift/sft_lingoqa_r3_train.jsonl")
    parser.add_argument("--preview", default="data/qwen3_vl/ms_swift/sft_lingoqa_r3_preview.jsonl")
    parser.add_argument("--audit_json", default="outputs/data_audit/ms_swift_sft_export_audit.json")
    parser.add_argument("--audit_md", default="outputs/data_audit/ms_swift_sft_export_audit.md")
    args = parser.parse_args()
    records = load_sft_records(args.input)
    rows = [{
        "messages": [{"role": "user", "content": make_ms_swift_user_text(record)}, {"role": "assistant", "content": answer_only_json(record.answer)}],
        "images": record.image_paths if record.setting != "text_only" else [],
        "id": record.id, "dataset": record.dataset, "metadata": record.metadata,
    } for record in records]
    write_jsonl(args.output, rows)
    write_jsonl(args.preview, rows[:8])
    leaks = leakage_ids(records, excluded_eval_ids())
    missing = sum(not Path(path).exists() for record in records for path in record.image_paths if record.setting != "text_only")
    metadata_leaks = sum(any(key in record.answer.lower() for key in ("source_id", "metadata", "sample_type")) for record in records)
    audit = {
        "format": "ms-swift standard messages + images VLM SFT",
        "total_samples": len(records),
        "image_sample_count": sum(bool(record.image_paths) and record.setting != "text_only" for record in records),
        "text_only_count": sum(record.setting == "text_only" for record in records),
        "multi_image_count": sum(len(record.image_paths) > 1 and record.setting != "text_only" for record in records),
        "missing_image_path_count": missing,
        "heldout_leakage_count": len(leaks),
        "heldout_leakage_ids": leaks,
        "assistant_answer_json_valid_rate": sum(is_answer_only_json(row["messages"][-1]["content"]) for row in rows) / len(rows) if rows else 0,
        "metadata_leak_count": metadata_leaks,
        "train_ready": bool(records) and not leaks and missing == 0 and metadata_leaks == 0,
        "note": "Follows ms-swift standard messages/images keys; assistant content is answer-only JSON with no reason and metadata stays outside assistant output.",
    }
    Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_json).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.audit_md).write_text("# ms-swift SFT Export Audit\n\n" + "\n".join(f"- {key}: {value}" for key, value in audit.items()) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
