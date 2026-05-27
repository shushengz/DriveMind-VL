"""Check Stage 11 deliverables are present and non-empty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = [
    "outputs/final_report/final_main_results_answer_only.csv",
    "outputs/final_report/final_ablation_results.csv",
    "outputs/final_report/final_checkpoint_selection.md",
    "outputs/final_report/final_case_gallery.html",
    "outputs/final_report/README_results.md",
    "outputs/final_report/resume_and_interview.md",
    "outputs/final_report/grpo_lite_future_plan.md",
    "README_CN.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage 11 final deliverables.")
    parser.add_argument("--output_json", default="outputs/final_report/final_deliverables_check.json")
    parser.add_argument("--output_md", default="outputs/final_report/final_deliverables_check.md")
    args = parser.parse_args()
    checks = []
    for item in REQUIRED:
        path = Path(item)
        checks.append({"path": item, "exists": path.exists(), "nonempty": path.exists() and path.stat().st_size > 0, "size": path.stat().st_size if path.exists() else 0})
    ok = all(item["exists"] and item["nonempty"] for item in checks)
    report = {"ok": ok, "cpu_only_stage": True, "required": checks}
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Final Deliverables Check", "", f"- Overall: {'PASS' if ok else 'FAIL'}", "- Stage mode: CPU-only report consolidation; no training or inference.", "", "| File | Present | Non-empty | Bytes |", "| --- | --- | --- | ---: |"]
    for item in checks:
        lines.append(f"| `{item['path']}` | {'yes' if item['exists'] else 'no'} | {'yes' if item['nonempty'] else 'no'} | {item['size']} |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": ok, "checked": len(checks)}, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
