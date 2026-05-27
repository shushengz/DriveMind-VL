"""Consolidate Stage 15 format and leakage audits without launching training."""
from __future__ import annotations

import json
from pathlib import Path


def read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    audits = {
        "ms_swift_sft": read("outputs/data_audit/ms_swift_sft_export_audit.json"),
        "ms_swift_dpo": read("outputs/data_audit/ms_swift_dpo_export_audit.json"),
        "ms_swift_grpo": read("outputs/data_audit/ms_swift_grpo_export_audit.json"),
        "qwen_vl_series": read("outputs/data_audit/qwen_vl_series_export_audit.json"),
    }
    leakage = {
        name: int(audit.get("heldout_leakage_count", audit.get("heldout_ood_leakage_count", -1)))
        for name, audit in audits.items()
    }
    checks = {
        "all_leakage_zero": all(value == 0 for value in leakage.values()),
        "ms_swift_sft_format_ready": bool(audits["ms_swift_sft"].get("train_ready")),
        "ms_swift_dpo_format_ready": bool(audits["ms_swift_dpo"].get("train_ready")),
        "ms_swift_grpo_format_ready_but_launch_blocked": bool(audits["ms_swift_grpo"].get("format_ready")) and not audits["ms_swift_grpo"].get("launch_allowed", True),
        "qwen_vl_series_requires_manual_format_check": bool(audits["qwen_vl_series"].get("needs_manual_format_check")),
        "grpo_not_launchable_until_reward_ready": not audits["ms_swift_grpo"].get("train_ready", True),
    }
    report = {
        "stage": "Stage 15 Qwen3-VL stack preparation",
        "cpu_only": True,
        "gpu_used": False,
        "training_or_inference_executed": False,
        "audits": audits,
        "heldout_leakage": leakage,
        "checks": checks,
        "data_format_audit_passed": all(checks.values()),
        "warning": "Passing format audit does not authorize SFT, DPO, or GRPO. Reward v2.2 remains not ready for GRPO.",
    }
    Path("outputs/data_audit/qwen3_vl_stage15_export_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Qwen3-VL Stage 15 Export Summary", "", "| Check | Pass |", "| --- | --- |"]
    lines.extend(f"| {key} | {'yes' if passed else 'no'} |" for key, passed in checks.items())
    lines += ["", f"- Held-out/OOD leakage by export: `{json.dumps(leakage, ensure_ascii=False)}`.",
              "- ms-swift SFT/DPO 数据格式已准备；GRPO prompt 仅格式可用，因 Reward v2.2 not ready 而禁止启动。",
              "- Qwen-VL-Series-Finetune 输出是参考导出，运行前必须按固定 commit 人工核查字段。", ""]
    Path("outputs/data_audit/qwen3_vl_stage15_export_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
