"""Export Preference-v8.2 to an answer-only ms-swift DPO draft dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.drivemind_vl_schema import DriveMindRecord, answer_only_json, excluded_eval_ids, extract_answer, is_answer_only_json, leakage_ids, make_ms_swift_user_text, normalize_id, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train/preference_v8_2/preference_v8_2_pairs.jsonl")
    parser.add_argument("--output", default="data/qwen3_vl/ms_swift/dpo_preference_v8_2.jsonl")
    parser.add_argument("--preview", default="data/qwen3_vl/ms_swift/dpo_preference_v8_2_preview.jsonl")
    parser.add_argument("--audit_json", default="outputs/data_audit/ms_swift_dpo_export_audit.json")
    parser.add_argument("--audit_md", default="outputs/data_audit/ms_swift_dpo_export_audit.md")
    args = parser.parse_args()
    source = read_jsonl(args.input)
    records = []
    rows = []
    for row in source:
        setting = str(row.get("setting", "normal"))
        record = DriveMindRecord(str(row.get("id", "")), str(row.get("dataset", "lingoqa")), setting, str(row.get("prompt", "")), "", list(row.get("image_paths") or []), list(row.get("image_labels") or []), {"source_id": normalize_id(row.get("source_id") or row.get("id")), "split": "train", "is_control": setting != "normal", "pair_type": row.get("pair_type", "")})
        records.append(record)
        chosen = extract_answer(row.get("chosen"))
        rejected = extract_answer(row.get("rejected"))
        rows.append({"messages": [{"role": "user", "content": make_ms_swift_user_text(record)}, {"role": "assistant", "content": answer_only_json(chosen)}], "rejected_response": answer_only_json(rejected), "images": record.image_paths if record.setting != "text_only" else [], "id": record.id, "pair_type": row.get("pair_type", ""), "metadata": record.metadata})
    write_jsonl(args.output, rows); write_jsonl(args.preview, rows[:8])
    leaks = leakage_ids(records, excluded_eval_ids())
    chosen_values = [item["messages"][-1]["content"] for item in rows]
    rejected_values = [item["rejected_response"] for item in rows]
    invalid = sum(not is_answer_only_json(chosen) or not is_answer_only_json(rejected) for chosen, rejected in zip(chosen_values, rejected_values))
    # Natural answers may contain the word "reason"; check structured keys.
    reason_leaks = sum("reason" in json.loads(chosen) or "reason" in json.loads(rejected) for chosen, rejected in zip(chosen_values, rejected_values) if is_answer_only_json(chosen) and is_answer_only_json(rejected))
    metadata_leaks = sum(any(term in chosen.lower() or term in rejected.lower() for term in ("source_id", "metadata", "candidate_type")) for chosen, rejected in zip(chosen_values, rejected_values))
    actual_rate = sum(bool((row.get("metadata") or {}).get("rejected_from_actual_prediction")) for row in source) / len(source) if source else 0
    audit = {"format": "ms-swift DPO messages + rejected_response + images", "total_pairs": len(rows), "heldout_leakage_count": len(leaks), "chosen_rejected_answer_json_valid_rate": (len(rows) - invalid) / len(rows) if rows else 0, "invalid_answer_count": invalid, "reason_field_count": reason_leaks, "metadata_leak_count": metadata_leaks, "rejected_from_actual_prediction_rate": actual_rate, "train_ready": bool(rows) and not leaks and invalid == 0 and reason_leaks == 0 and metadata_leaks == 0 and actual_rate >= 0.80, "note": "Official ms-swift standard represents chosen in assistant messages and rejected in rejected_response; both responses retain answer-only JSON output contract."}
    Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_json).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.audit_md).write_text("# ms-swift DPO Export Audit\n\n" + "\n".join(f"- {key}: {value}" for key, value in audit.items()) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
