"""Export a leakage-safe GRPO-lite prompt draft; it is not a training launch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.drivemind_vl_schema import excluded_eval_ids, leakage_ids, load_sft_records, make_ms_swift_user_text, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl")
    parser.add_argument("--output", default="data/qwen3_vl/ms_swift/grpo_lite_prompts.jsonl")
    parser.add_argument("--preview", default="data/qwen3_vl/ms_swift/grpo_lite_prompts_preview.jsonl")
    parser.add_argument("--audit_json", default="outputs/data_audit/ms_swift_grpo_export_audit.json")
    parser.add_argument("--audit_md", default="outputs/data_audit/ms_swift_grpo_export_audit.md")
    args = parser.parse_args()
    records = load_sft_records(args.input)
    leaks = leakage_ids(records, excluded_eval_ids())
    rows = [{
        "messages": [{"role": "user", "content": make_ms_swift_user_text(record)}],
        "images": record.image_paths if record.setting != "text_only" else [],
        "id": record.id, "dataset": record.dataset, "solution": record.answer,
        "metadata": {**record.metadata, "normal_gold": record.answer, "control_setting": record.setting, "reward_fields": {"setting": record.setting, "requires_visual_dependency_reward": True}},
    } for record in records if record.metadata["source_id"] not in set(leaks)]
    write_jsonl(args.output, rows); write_jsonl(args.preview, rows[:8])
    audit = {
        "format": "ms-swift GRPO-lite prompt draft",
        "source": args.input,
        "total_prompts": len(rows),
        "normal_prompt_count": sum(row["metadata"]["control_setting"] == "normal" for row in rows),
        "control_prompt_count": sum(row["metadata"]["control_setting"] != "normal" for row in rows),
        "heldout_ood_leakage_count": len(leaks),
        "uses_reward_debug_eval_records": False,
        "contains_assistant_training_answer": False,
        "solution_is_reward_reference_only": True,
        "format_ready": bool(rows) and not leaks,
        "train_ready": False,
        "launch_allowed": False,
        "warning": "Reward v2.2 is not ready; this prompt export must not be used to launch GRPO.",
    }
    Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_json).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.audit_md).write_text("# ms-swift GRPO-lite Prompt Draft Audit\n\n" + "\n".join(f"- {key}: {value}" for key, value in audit.items()) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
