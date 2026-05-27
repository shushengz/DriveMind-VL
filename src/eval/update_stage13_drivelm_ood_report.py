"""Append the verified Stage 13 OOD section to project reports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BEGIN = "<!-- STAGE13_DRIVELM_OOD_BEGIN -->"
END = "<!-- STAGE13_DRIVELM_OOD_END -->"
BASE = "base_qwen25vl_3b"
R3 = "sft_v3_r3_lingo_smoke"


def value(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def replace_section(path: Path, section: str) -> None:
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"\n{BEGIN}\n{section.rstrip()}\n{END}\n"
    if BEGIN in content and END in content:
        prefix = content.split(BEGIN, 1)[0].rstrip()
        suffix = content.split(END, 1)[1].lstrip("\n")
        content = prefix + block + suffix
    else:
        content = content.rstrip() + "\n" + block
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Stage 13 project reports.")
    parser.add_argument("--results", default="outputs/final_report/drivelm_ood_results_100_answer_only.csv")
    parser.add_argument("--diagnosis", default="outputs/final_report/drivelm_ood_diagnosis.json")
    args = parser.parse_args()
    with Path(args.results).open("r", encoding="utf-8", newline="") as handle:
        rows = {row["model_name"]: row for row in csv.DictReader(handle)}
    report = json.loads(Path(args.diagnosis).read_text(encoding="utf-8"))
    base, r3 = rows[BASE], rows[R3]
    recommend_300 = report["recommend_eval_300"]
    hardest = report.get("hardest_capability_r3", {}).get("capability", "n/a")
    statement = report["generalization_statement"]

    section = f"""## DriveLM OOD Strict Visual-Control Evaluation

### 目的与边界

DriveLM 与主评测 LingoQA 不同，包含多相机 camera labels、object token 与更复杂的空间/行动推理问题。本轮只在 100 个结构校验通过的 OOD cases 上推理 Base 与最终 `SFT-v3-r3`，不训练模型、不构造训练数据，也不替换 LingoQA held-out 上选出的最终 checkpoint。

### Answer-only 结果

| 模型 | normal_f1 | case_gap | control_high_f1 | hallucination | normal_refusal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen2.5-VL-3B | {value(base, 'normal_f1'):.4f} | {value(base, 'case_gap'):.4f} | {value(base, 'control_high_f1_rate_0_20'):.4f} | {value(base, 'control_hallucination_rate'):.4f} | {value(base, 'normal_refusal_rate'):.4f} |
| SFT-v3-r3 | {value(r3, 'normal_f1'):.4f} | {value(r3, 'case_gap'):.4f} | {value(r3, 'control_high_f1_rate_0_20'):.4f} | {value(r3, 'control_hallucination_rate'):.4f} | {value(r3, 'normal_refusal_rate'):.4f} |

### 结论

- 跨数据集泛化判定：{statement}。
- r3 最难的能力类型为 `{hardest}`；完整拆解见 `outputs/final_report/drivelm_ood_capability_breakdown.md`。
- DriveLM OOD 必须写入 limitation：它衡量跨域视觉依赖，不可替代 LingoQA 主结论。
- 是否建议扩大到 300 cases：{'是，在本轮不自动执行，需单独授权 GPU 推理。' if recommend_300 else '否，先分析 100-case 暴露的失败模式。'}
- 当前最终 checkpoint 仍为 `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/`；当前不继续训练，也不进入 GRPO-lite。
"""
    replace_section(Path("outputs/final_report/README_results.md"), section)
    replace_section(Path("README_CN.md"), section)
    roadmap_section = f"""## Stage 13 Update：DriveLM OOD Strict Visual-Control

- 已完成 Base 与 SFT-v3-r3 的 DriveLM OOD 100-case GPU 推理评测；GPU 仅用于推理。
- Answer-only 诊断结论：{statement}。
- r3 OOD 指标：normal_f1={value(r3, 'normal_f1'):.4f}, case_gap={value(r3, 'case_gap'):.4f}, control_high_f1={value(r3, 'control_high_f1_rate_0_20'):.4f}。
- 最难能力类型：`{hardest}`。
- 后续是否运行 300-case OOD：{'可在单独授权后运行' if recommend_300 else '当前不建议立即扩大'}。
- DriveLM OOD 作为 limitation/future work 记录；不改变 r3 作为 LingoQA 主模型的结论，不触发 GRPO-lite。
"""
    replace_section(Path("outputs/final_report/depth_extension_roadmap.md"), roadmap_section)
    print(json.dumps({"updated": ["outputs/final_report/README_results.md", "README_CN.md", "outputs/final_report/depth_extension_roadmap.md"], "recommend_eval_300": recommend_300}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
