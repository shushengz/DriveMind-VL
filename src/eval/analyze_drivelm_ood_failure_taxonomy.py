"""Assign per-case failure tags to Stage 13 DriveLM OOD predictions."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.drivelm_ood_attribution_utils import PREDICTION_ROOT, load_cases, taxonomy_row, write_csv


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DriveLM OOD failure taxonomy.")
    parser.add_argument("--prediction_root", default=PREDICTION_ROOT.as_posix())
    parser.add_argument("--output_csv", default="outputs/final_report/drivelm_ood_failure_taxonomy.csv")
    parser.add_argument("--output_md", default="outputs/final_report/drivelm_ood_failure_taxonomy.md")
    parser.add_argument("--output_json", default="outputs/final_report/drivelm_ood_failure_taxonomy.json")
    args = parser.parse_args()
    rows = [taxonomy_row(case) for case in load_cases(Path(args.prediction_root))]
    write_csv(Path(args.output_csv), rows)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for tag in row["failure_tags"].split("|"):
            buckets[tag].append(row)
    stats = []
    for tag, members in buckets.items():
        stats.append({
            "failure_type": tag,
            "count": len(members),
            "avg_normal_f1": mean([row["r3_normal_f1"] for row in members]),
            "avg_case_gap": mean([row["r3_case_gap"] for row in members]),
            "control_high_f1_rate": mean([
                float(max(row["r3_text_f1"], row["r3_wrong_f1"], row["r3_blank_f1"]) >= 0.20)
                for row in members
            ]),
            "avg_delta_normal_f1": mean([row["delta_normal_f1"] for row in members]),
            "avg_delta_case_gap": mean([row["delta_case_gap"] for row in members]),
        })
    stats.sort(key=lambda row: (-row["count"], row["failure_type"]))
    wrong_rows = [row for row in rows if "wrong_image_confound" in row["failure_tags"]]
    wrong_object = [row for row in wrong_rows if "object_token_failure" in row["failure_tags"]]
    wrong_spatial = [row for row in wrong_rows if "spatial_relation_failure" in row["failure_tags"]]
    wrong_camera = [row for row in wrong_rows if "camera_specific_failure" in row["failure_tags"]]
    summary = {
        "num_cases": len(rows),
        "tag_statistics": stats,
        "most_frequent_failure_type": stats[0]["failure_type"] if stats else "",
        "most_frequent_failure_count": stats[0]["count"] if stats else 0,
        "primary_type_counts": {
            tag: sum(row["primary_failure_type"] == tag for row in rows)
            for tag in sorted({row["primary_failure_type"] for row in rows})
        },
        "cross_tag_analysis": {
            "wrong_image_confound_count": len(wrong_rows),
            "wrong_with_camera_labeled_failure": len(wrong_camera),
            "wrong_with_object_token_failure": len(wrong_object),
            "wrong_with_spatial_relation_failure": len(wrong_spatial),
            "camera_tag_caveat": "DriveLM inputs are camera-labelled throughout; camera_specific_failure is broad and should be interpreted together with object/spatial overlap.",
        },
    }
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DriveLM OOD Failure Taxonomy", "",
        "Failure tags 可重叠：同一案例可能同时暴露空间关系、错图混淆和空白图先验。这比强制归入单类更适合解释 OOD 失败。",
        "", "| failure tag | count | avg normal_f1 | avg case_gap | control high-F1 rate | delta normal_f1 | delta case_gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stats:
        lines.append(
            f"| {row['failure_type']} | {row['count']} | {row['avg_normal_f1']:.4f} | "
            f"{row['avg_case_gap']:.4f} | {row['control_high_f1_rate']:.4f} | "
            f"{row['avg_delta_normal_f1']:.4f} | {row['avg_delta_case_gap']:.4f} |"
        )
    lines += [
        "", "## 主要结论", "",
        f"- 数量最多的 failure tag：`{summary['most_frequent_failure_type']}`（{summary['most_frequent_failure_count']} / {len(rows)} cases）。",
        "- `normal_answer_style_transfer` 表示 r3 正常答案变好、但 visual dependency 没有随之改善的直接证据。",
        "- `blank_prior_answer`、`text_only_prior_answer` 与 `wrong_image_confound` 直接服务于后续 reward harness v2 设计。",
        f"- Wrong-image confound 共 {len(wrong_rows)} 条，其中与 object-token failure 重叠 {len(wrong_object)} 条、与 spatial failure 重叠 {len(wrong_spatial)} 条。",
        "- 注意：DriveLM 输入普遍带有 camera labels，因而 `camera_specific_failure` 是宽口径标签；它说明错图/视角依赖问题广泛存在，但不能单凭数量定位为某一个 camera 子类型。",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
