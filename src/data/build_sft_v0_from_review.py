"""Build LingoQA SFT-v0 split files from reviewed candidate CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {"keep": 25, "fix": 7, "drop": 9}
VALID_ACTIONS = set(EXPECTED_COUNTS)

ROLE_BY_CANDIDATE_TYPE = {
    "positive_grounding": "positive_visual_grounding",
    "hard_negative_spatial_reasoning": "spatial_reasoning_correction",
    "wrong_image_confound": "anti_hallucination_counterfactual",
    "language_prior_confound": "language_prior_rejection",
    "all_settings_failed": "hard_case_needs_caution",
    "mixed_review": "manual_review_mixed",
}

WRONG_IMAGE_KEEP_WARNING = (
    "wrong_image_confound_keep_use_as_anti_hallucination_or_visual_evidence_sample"
)

FIX_FIELDNAMES = [
    "id",
    "candidate_type",
    "capability",
    "question",
    "gold_answer",
    "normal_prediction",
    "human_note",
    "suggested_fix",
    "image",
    "image_paths",
    "priority_score",
    "visual_dependency_gap",
    "normal_f1",
    "control_max_f1",
]

DROP_FIELDNAMES = [
    "id",
    "candidate_type",
    "capability",
    "question",
    "gold_answer",
    "normal_prediction",
    "human_note",
    "drop_reason",
    "priority_score",
    "visual_dependency_gap",
    "normal_f1",
    "control_max_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SFT-v0 keep/fix/drop assets from reviewed LingoQA candidates."
    )
    parser.add_argument(
        "--review_csv",
        type=Path,
        default=Path("outputs/cases/lingoqa_sft_candidate_review_agent_checked.csv"),
    )
    parser.add_argument(
        "--candidate_jsonl",
        type=Path,
        default=Path("data/processed/lingoqa_sft_candidates_needs_review.jsonl"),
    )
    parser.add_argument(
        "--keep_output",
        type=Path,
        default=Path("data/processed/lingoqa_sft_v0_keep_25.jsonl"),
    )
    parser.add_argument(
        "--fix_output",
        type=Path,
        default=Path("outputs/cases/lingoqa_sft_v0_need_fix_7.csv"),
    )
    parser.add_argument(
        "--drop_output",
        type=Path,
        default=Path("outputs/cases/lingoqa_sft_v0_drop_9.csv"),
    )
    parser.add_argument(
        "--report_output",
        type=Path,
        default=Path("docs/lingoqa_sft_v0_build_report.md"),
    )
    parser.add_argument(
        "--note_output",
        type=Path,
        default=Path("Note/2026-05-17_LingoQA_SFT_v0数据分流.md"),
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return rows


def assert_inputs_exist(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))


def assert_outputs_are_new(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output file(s): " + ", ".join(existing)
        )


def nested_get(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = row
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def extract_suggested_fix(human_note: str) -> str:
    patterns = [
        r"Rewrite as:\s*(.+)$",
        r"rewrite as:\s*(.+)$",
        r"应改写为[：:]\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, human_note, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("\"'“”")
    return ""


def csv_image_paths(candidate: dict[str, Any]) -> str:
    image_paths = nested_get(candidate, ["meta", "external", "image_paths"], [])
    if not image_paths:
        return ""
    return json.dumps(image_paths, ensure_ascii=False)


def candidate_role(candidate_type: str) -> str:
    return ROLE_BY_CANDIDATE_TYPE.get(candidate_type, "manual_review_mixed")


def build_keep_row(review_row: dict[str, str], candidate: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(candidate)
    candidate_type = review_row["candidate_type"]

    meta = output.setdefault("meta", {})
    meta["curation_status"] = "reviewed_keep"
    meta["human_action"] = "keep"
    meta["human_note"] = review_row["human_note"]
    meta["sft_v0_role"] = candidate_role(candidate_type)
    if candidate_type == "wrong_image_confound":
        meta["training_warning"] = WRONG_IMAGE_KEEP_WARNING

    review = output.setdefault("review", {})
    review["human_action"] = "keep"
    review["human_note"] = review_row["human_note"]
    review["candidate_type"] = candidate_type
    return output


def build_fix_row(review_row: dict[str, str], candidate: dict[str, Any]) -> dict[str, str]:
    return {
        "id": review_row["id"],
        "candidate_type": review_row["candidate_type"],
        "capability": review_row.get("capability", ""),
        "question": review_row.get("question", ""),
        "gold_answer": review_row.get("gold_answer", ""),
        "normal_prediction": review_row.get("normal_prediction", ""),
        "human_note": review_row.get("human_note", ""),
        "suggested_fix": extract_suggested_fix(review_row.get("human_note", "")),
        "image": str(candidate.get("image") or ""),
        "image_paths": csv_image_paths(candidate),
        "priority_score": review_row.get("priority_score", ""),
        "visual_dependency_gap": review_row.get("visual_dependency_gap", ""),
        "normal_f1": review_row.get("normal_f1", ""),
        "control_max_f1": review_row.get("control_max_f1", ""),
    }


def build_drop_row(review_row: dict[str, str]) -> dict[str, str]:
    return {
        "id": review_row["id"],
        "candidate_type": review_row["candidate_type"],
        "capability": review_row.get("capability", ""),
        "question": review_row.get("question", ""),
        "gold_answer": review_row.get("gold_answer", ""),
        "normal_prediction": review_row.get("normal_prediction", ""),
        "human_note": review_row.get("human_note", ""),
        "drop_reason": review_row.get("human_note", ""),
        "priority_score": review_row.get("priority_score", ""),
        "visual_dependency_gap": review_row.get("visual_dependency_gap", ""),
        "normal_f1": review_row.get("normal_f1", ""),
        "control_max_f1": review_row.get("control_max_f1", ""),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_list(items: list[str]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- `{item}`" for item in items)


def markdown_count_table(title: str, rows: list[tuple[str, dict[str, int]]]) -> str:
    lines = [f"## {title}", "", "| key | keep | fix | drop | total |", "|---|---:|---:|---:|---:|"]
    for key, counts in rows:
        keep = counts.get("keep", 0)
        fix = counts.get("fix", 0)
        drop = counts.get("drop", 0)
        lines.append(f"| {key} | {keep} | {fix} | {drop} | {keep + fix + drop} |")
    return "\n".join(lines)


def build_report(
    args: argparse.Namespace,
    total_counts: Counter[str],
    by_type: dict[str, Counter[str]],
    role_counts: Counter[str],
    wrong_image_keep_count: int,
    unmatched_review_ids: list[str],
    invalid_action_ids: list[str],
    empty_note_ids: list[str],
) -> str:
    by_type_rows = sorted((key, dict(value)) for key, value in by_type.items())
    role_lines = ["| sft_v0_role | count |", "|---|---:|"]
    for role, count in sorted(role_counts.items()):
        role_lines.append(f"| {role} | {count} |")

    return "\n".join(
        [
            "# LingoQA SFT-v0 Build Report",
            "",
            "## Input Files",
            "",
            f"- Review CSV: `{args.review_csv}`",
            f"- Candidate JSONL: `{args.candidate_jsonl}`",
            "",
            "## Output Files",
            "",
            f"- Keep JSONL: `{args.keep_output}`",
            f"- Need-fix CSV: `{args.fix_output}`",
            f"- Drop CSV: `{args.drop_output}`",
            f"- Report: `{args.report_output}`",
            f"- Chinese note: `{args.note_output}`",
            "",
            "## Total Counts",
            "",
            f"- keep: {total_counts.get('keep', 0)}",
            f"- fix: {total_counts.get('fix', 0)}",
            f"- drop: {total_counts.get('drop', 0)}",
            "",
            markdown_count_table("Distribution By Candidate Type", by_type_rows),
            "",
            "## Keep Role Distribution",
            "",
            *role_lines,
            "",
            "## wrong_image_confound Keep",
            "",
            f"- count: {wrong_image_keep_count}",
            f"- training_warning: `{WRONG_IMAGE_KEEP_WARNING}`",
            "",
            "## Quality Check Lists",
            "",
            "### unmatched_review_ids",
            "",
            markdown_list(unmatched_review_ids),
            "",
            "### invalid_action_ids",
            "",
            markdown_list(invalid_action_ids),
            "",
            "### empty_note_ids",
            "",
            markdown_list(empty_note_ids),
            "",
            "## Next Steps",
            "",
            "- Use only the reviewed keep JSONL for the first small SFT-v0 candidate pool.",
            "- Manually rewrite the fix rows before promoting them into any training JSONL.",
            "- Keep wrong_image_confound rows tagged as anti-hallucination or visual-evidence samples, not ordinary positive samples.",
            "- Re-run this builder after additional review batches are completed, using new output filenames.",
            "",
        ]
    )


def build_note(
    total_counts: Counter[str],
    by_type: dict[str, Counter[str]],
    wrong_image_keep_count: int,
) -> str:
    by_type_lines = []
    for candidate_type, counts in sorted(by_type.items()):
        by_type_lines.append(
            f"- {candidate_type}: keep {counts.get('keep', 0)}, "
            f"fix {counts.get('fix', 0)}, drop {counts.get('drop', 0)}"
        )

    return "\n".join(
        [
            "# LingoQA SFT-v0 数据分流",
            "",
            "## 本轮做了什么",
            "",
            "本轮基于已经复核过的 LingoQA 候选样本 CSV，按 `human_action` 精确分流为 keep、fix、drop 三类资产。keep 样本写入可进入 SFT-v0 候选池的 JSONL；fix 样本保留为人工改写清单；drop 样本保留为明确丢弃清单。",
            "",
            "## 为什么不能直接训练全部候选",
            "",
            "候选样本中包含语言先验可猜中、wrong image 也能答对、图像证据不足、参考答案过短或答案歧义等情况。直接训练全部候选会把错误视觉归因、弱 grounding 和含糊答案带入模型，因此必须先做保守分流。",
            "",
            "## keep / fix / drop 的含义",
            "",
            "- keep: 图像证据支持参考答案，可以作为第一版小规模 SFT-v0 候选样本。",
            "- fix: 样本方向可能可用，但答案或 reason 需要人工改写后才能进入训练。",
            "- drop: 图像证据不足、参考答案不可靠、问题歧义较大或缺少视觉 grounding 价值，暂不进入训练。",
            "",
            "## wrong_image_confound 的特殊处理",
            "",
            "wrong_image_confound 不能当普通正样本使用。即使被标记为 keep，也需要作为抗幻觉或视觉证据样本处理，训练重点是要求模型依据当前图像作答，而不是依赖语言先验或错误图像。所有这类 keep 样本都带有 `training_warning` 标记。",
            "",
            "## 当前数量",
            "",
            f"- keep: {total_counts.get('keep', 0)}",
            f"- fix: {total_counts.get('fix', 0)}",
            f"- drop: {total_counts.get('drop', 0)}",
            f"- wrong_image_confound + keep: {wrong_image_keep_count}",
            "",
            "## 按类型分布",
            "",
            *by_type_lines,
            "",
            "## fix 样本下一步",
            "",
            "fix 清单中的样本需要人工根据 `human_note` 和 `suggested_fix` 改写训练答案，尤其要补齐交通灯状态、行人/车辆位置、是否需要让行、是否可通行等关键视觉条件。改写完成后应重新复核，再生成新的 keep JSONL。",
            "",
            "## 后续训练前还需要什么",
            "",
            "- 确认 keep JSONL 中图片路径在训练环境可访问。",
            "- 抽样检查 wrong_image_confound keep 的 `training_warning` 和训练角色是否被训练脚本正确读取。",
            "- 对 fix 样本完成答案改写和二次复核。",
            "- 在启动训练前固定数据版本、记录输入文件哈希或至少记录文件路径与生成时间。",
            "",
        ]
    )


def validate_review_rows(
    review_rows: list[dict[str, str]], candidate_by_id: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str], list[str], Counter[str], dict[str, Counter[str]]]:
    unmatched_review_ids: list[str] = []
    invalid_action_ids: list[str] = []
    empty_note_ids: list[str] = []
    total_counts: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = defaultdict(Counter)

    for row in review_rows:
        row_id = row.get("id", "")
        action = (row.get("human_action") or "").strip()
        candidate_type = row.get("candidate_type", "")

        if action not in VALID_ACTIONS:
            invalid_action_ids.append(row_id)
            continue
        if not (row.get("human_note") or "").strip():
            empty_note_ids.append(row_id)
        if row_id not in candidate_by_id:
            unmatched_review_ids.append(row_id)

        total_counts[action] += 1
        by_type[candidate_type][action] += 1

    return (
        unmatched_review_ids,
        invalid_action_ids,
        empty_note_ids,
        total_counts,
        by_type,
    )


def main() -> int:
    args = parse_args()
    outputs = [
        args.keep_output,
        args.fix_output,
        args.drop_output,
        args.report_output,
        args.note_output,
    ]

    try:
        assert_inputs_exist([args.review_csv, args.candidate_jsonl])
        assert_outputs_are_new(outputs)

        review_rows = load_csv(args.review_csv)
        candidate_rows = load_jsonl(args.candidate_jsonl)
        candidate_by_id = {str(row.get("id")): row for row in candidate_rows}

        (
            unmatched_review_ids,
            invalid_action_ids,
            empty_note_ids,
            total_counts,
            by_type,
        ) = validate_review_rows(review_rows, candidate_by_id)

        count_errors = [
            f"{action}: expected {expected}, got {total_counts.get(action, 0)}"
            for action, expected in EXPECTED_COUNTS.items()
            if total_counts.get(action, 0) != expected
        ]

        validation_errors = []
        if invalid_action_ids:
            validation_errors.append("Invalid human_action ids: " + ", ".join(invalid_action_ids))
        if empty_note_ids:
            validation_errors.append("Rows with empty human_note: " + ", ".join(empty_note_ids))
        if unmatched_review_ids:
            validation_errors.append(
                "Review ids missing from candidate JSONL: " + ", ".join(unmatched_review_ids)
            )
        if count_errors:
            validation_errors.append("Count mismatch: " + "; ".join(count_errors))
        if validation_errors:
            raise ValueError("\n".join(validation_errors))

        keep_rows: list[dict[str, Any]] = []
        fix_rows: list[dict[str, str]] = []
        drop_rows: list[dict[str, str]] = []

        for row in review_rows:
            row["human_action"] = row["human_action"].strip()
            row["human_note"] = row["human_note"].strip()
            candidate = candidate_by_id[row["id"]]
            if row["human_action"] == "keep":
                keep_rows.append(build_keep_row(row, candidate))
            elif row["human_action"] == "fix":
                fix_rows.append(build_fix_row(row, candidate))
            elif row["human_action"] == "drop":
                drop_rows.append(build_drop_row(row))

        missing_image_ids = [
            row["id"]
            for row in keep_rows
            if not row.get("image") and not nested_get(row, ["meta", "external", "image_paths"], [])
        ]
        if missing_image_ids:
            raise ValueError(
                "Keep rows missing both image and meta.external.image_paths: "
                + ", ".join(missing_image_ids)
            )

        wrong_image_keep_rows = [
            row
            for row in keep_rows
            if nested_get(row, ["review", "candidate_type"]) == "wrong_image_confound"
        ]
        missing_warning_ids = [
            row["id"]
            for row in wrong_image_keep_rows
            if nested_get(row, ["meta", "training_warning"]) != WRONG_IMAGE_KEEP_WARNING
        ]
        if missing_warning_ids:
            raise ValueError(
                "wrong_image_confound keep rows missing training_warning: "
                + ", ".join(missing_warning_ids)
            )

        role_counts = Counter(str(nested_get(row, ["meta", "sft_v0_role"], "")) for row in keep_rows)
        report = build_report(
            args=args,
            total_counts=total_counts,
            by_type=by_type,
            role_counts=role_counts,
            wrong_image_keep_count=len(wrong_image_keep_rows),
            unmatched_review_ids=unmatched_review_ids,
            invalid_action_ids=invalid_action_ids,
            empty_note_ids=empty_note_ids,
        )
        note = build_note(
            total_counts=total_counts,
            by_type=by_type,
            wrong_image_keep_count=len(wrong_image_keep_rows),
        )

        write_jsonl(args.keep_output, keep_rows)
        write_csv(args.fix_output, fix_rows, FIX_FIELDNAMES)
        write_csv(args.drop_output, drop_rows, DROP_FIELDNAMES)
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(report, encoding="utf-8")
        args.note_output.parent.mkdir(parents=True, exist_ok=True)
        args.note_output.write_text(note, encoding="utf-8")

        print("SFT-v0 split build succeeded.")
        print(f"keep: {len(keep_rows)}")
        print(f"fix: {len(fix_rows)}")
        print(f"drop: {len(drop_rows)}")
        print(f"wrong_image_confound_keep: {len(wrong_image_keep_rows)}")
        print("unmatched_review_ids: 0")
        print("invalid_action_ids: 0")
        print("empty_note_ids: 0")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
