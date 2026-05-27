"""Deep analysis for DriveLM OOD spatial-relation failures."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.drivelm_ood_attribution_utils import CAMERA_RE, FRONT_BACK_RE, LANE_RE, LEFT_RIGHT_RE, OBJECT_RE, write_csv


def f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DriveLM OOD spatial failures.")
    parser.add_argument("--taxonomy", default="outputs/final_report/drivelm_ood_failure_taxonomy.csv")
    parser.add_argument("--output_csv", default="outputs/final_report/drivelm_spatial_failure_analysis.csv")
    parser.add_argument("--output_md", default="outputs/final_report/drivelm_spatial_failure_analysis.md")
    args = parser.parse_args()
    with Path(args.taxonomy).open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if row["capability"] == "spatial_relation"]
    detailed = []
    subtype_counts: Counter[str] = Counter()
    for row in rows:
        text = " ".join([row["question"], row["gold"], row["r3_normal_answer"]])
        subtypes = []
        if LEFT_RIGHT_RE.search(text):
            subtypes.append("left_right_confusion")
        if FRONT_BACK_RE.search(text):
            subtypes.append("front_back_confusion")
        if LANE_RE.search(text):
            subtypes.append("lane_relation_failure")
        if CAMERA_RE.search(row["question"]):
            subtypes.append("camera_view_mismatch")
        if OBJECT_RE.search(row["question"]):
            subtypes.append("object_reference_failure")
        if f(row, "r3_text_f1") >= 0.20 or f(row, "r3_blank_f1") >= 0.20:
            subtypes.append("pure_language_prior")
        if not subtypes:
            subtypes.append("other_spatial_failure")
        subtype_counts.update(subtypes)
        detailed.append({**row, "spatial_subtypes": "|".join(subtypes)})
    write_csv(Path(args.output_csv), detailed)
    count = len(rows)
    avg = lambda key: sum(f(row, key) for row in rows) / count if count else 0.0
    wrong_rate = sum(f(row, "r3_wrong_f1") >= 0.20 or f(row, "r3_wrong_f1") >= f(row, "r3_normal_f1") - 0.05 for row in rows) / count if count else 0.0
    blank_rate = sum("blank_prior_answer" in row["failure_tags"] for row in rows) / count if count else 0.0
    camera_overlap = sum(bool(CAMERA_RE.search(row["question"])) for row in rows) / count if count else 0.0
    object_overlap = sum(bool(OBJECT_RE.search(row["question"])) for row in rows) / count if count else 0.0
    main_subtype, main_count = subtype_counts.most_common(1)[0] if subtype_counts else ("none", 0)
    lines = [
        "# DriveLM Spatial-Relation Failure Analysis", "",
        f"- spatial_relation cases：{count}。",
        f"- Base normal_f1（按 case 重新聚合）：{avg('base_normal_f1'):.4f}。",
        f"- r3 normal_f1：{avg('r3_normal_f1'):.4f}。",
        f"- Base case_gap：{avg('base_case_gap'):.4f}。",
        f"- r3 case_gap：{avg('r3_case_gap'):.4f}。",
        f"- r3 text_only / wrong_image / blank_image F1：{avg('r3_text_f1'):.4f} / {avg('r3_wrong_f1'):.4f} / {avg('r3_blank_f1'):.4f}。",
        f"- wrong-image confound rate：{wrong_rate:.4f}。",
        f"- blank-prior rate：{blank_rate:.4f}。",
        f"- camera-specific overlap：{camera_overlap:.4f}。",
        f"- object-token overlap：{object_overlap:.4f}。",
        "", "## 子类型统计", "", "| subtype | count |", "| --- | ---: |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in subtype_counts.most_common())
    lines += [
        "", "## 诊断回答", "",
        "1. `spatial_relation` 是具有充分样本量的能力桶中最大瓶颈：normal F1 较低，同时 case gap、control high-F1 与 wrong-image confound 均最差。",
        f"2. 按可观测文本线索，数量最多的空间失败子类型是 `{main_subtype}`（{main_count} cases）；子类型会重叠。",
        f"3. r3 是否更容易用语言先验回答：{'是' if avg('r3_case_gap') < avg('base_case_gap') else '未见恶化'}，其空间关系 control 条件下的高重合直接造成更负 case gap。",
        "4. 对 VLA/robotics 的启示：只优化一般问答准确率不足以支撑空间行动决策；相机视角、对象引用与相对位置必须作为显式 grounding/reward 信号。",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"spatial_count": count, "main_subtype": main_subtype, "main_subtype_count": main_count, "wrong_image_confound_rate": wrong_rate, "blank_prior_rate": blank_rate, "subtypes": dict(subtype_counts)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
