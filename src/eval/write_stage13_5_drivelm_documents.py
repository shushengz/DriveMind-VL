"""Generate Stage 13.5 limitation/reward writeups and update project reports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BEGIN = "<!-- STAGE13_5_FAILURE_ATTRIBUTION_BEGIN -->"
END = "<!-- STAGE13_5_FAILURE_ATTRIBUTION_END -->"
BASE = "base_qwen25vl_3b"
R3 = "sft_v3_r3_lingo_smoke"


def read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def n(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def replace_section(path: Path, section: str) -> None:
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"\n{BEGIN}\n{section.rstrip()}\n{END}\n"
    if BEGIN in content and END in content:
        content = content.split(BEGIN, 1)[0].rstrip() + block + content.split(END, 1)[1].lstrip("\n")
    else:
        content = content.rstrip() + "\n" + block
    path.write_text(content, encoding="utf-8")


def load_facts() -> dict:
    results = {row["model_name"]: row for row in read_csv("outputs/final_report/drivelm_ood_results_100_answer_only.csv")}
    taxonomy = json.loads(Path("outputs/final_report/drivelm_ood_failure_taxonomy.json").read_text(encoding="utf-8"))
    diagnosis = json.loads(Path("outputs/final_report/drivelm_ood_diagnosis.json").read_text(encoding="utf-8"))
    spatial_text = Path("outputs/final_report/drivelm_spatial_failure_analysis.md").read_text(encoding="utf-8")
    fix_report = Path("outputs/final_report/drivelm_base_vs_r3_fix_break_report.md").read_text(encoding="utf-8")
    control = read_csv("outputs/final_report/drivelm_control_behavior_summary.csv")
    control_by_model = {row["model_name"]: row for row in control}
    fix_count = sum(1 for _ in read_csv("outputs/final_report/drivelm_base_vs_r3_fix_cases.csv"))
    break_count = sum(1 for _ in read_csv("outputs/final_report/drivelm_base_vs_r3_break_cases.csv"))
    spatial = diagnosis["hardest_capability_r3"]
    return {
        "base": results[BASE], "r3": results[R3], "taxonomy": taxonomy, "diagnosis": diagnosis,
        "spatial": spatial, "spatial_text": spatial_text, "fix_report": fix_report,
        "base_control": control_by_model[BASE], "r3_control": control_by_model[R3],
        "fix_count": fix_count, "break_count": break_count,
    }


def write_limitation(facts: dict) -> None:
    base, r3, spatial = facts["base"], facts["r3"], facts["spatial"]
    content = f"""# DriveLM OOD Limitation Writeup

## 中文版 Limitation

在 DriveLM OOD 100-case strict visual-control 评测中，LingoQA 上选出的 SFT-v3-r3 将 normal F1 从 Base 的 {n(base, 'normal_f1'):.4f} 提升至 {n(r3, 'normal_f1'):.4f}，但 case gap 从 {n(base, 'case_gap'):.4f} 恶化为 {n(r3, 'case_gap'):.4f}，control high-F1 rate 从 {n(base, 'control_high_f1_rate_0_20'):.4f} 上升至 {n(r3, 'control_high_f1_rate_0_20'):.4f}，blank high-F1 rate 从 {n(base, 'blank_high_f1_rate_0_20'):.4f} 上升至 {n(r3, 'blank_high_f1_rate_0_20'):.4f}。这说明 r3 仅表现出有限的正常回答风格迁移，并未获得可靠的跨域 visual dependency。最突出瓶颈为 spatial relation：n={spatial['num_samples']}，r3 case_gap={float(spatial['case_gap']):.4f}，control_high_f1={float(spatial['control_high_f1_rate']):.4f}，同时与多相机视角、object-token/camera alignment 问题相互耦合。

## English Limitation

On the 100-case DriveLM OOD strict visual-control evaluation, SFT-v3-r3 improves normal-answer F1 over the base model ({n(base, 'normal_f1'):.4f} to {n(r3, 'normal_f1'):.4f}), but worsens case gap ({n(base, 'case_gap'):.4f} to {n(r3, 'case_gap'):.4f}) and raises both control high-F1 ({n(base, 'control_high_f1_rate_0_20'):.4f} to {n(r3, 'control_high_f1_rate_0_20'):.4f}) and blank-image high-F1 ({n(base, 'blank_high_f1_rate_0_20'):.4f} to {n(r3, 'blank_high_f1_rate_0_20'):.4f}). Thus, the LingoQA-calibrated model transfers some answer style but not robust cross-domain visual dependency. Spatial-relation reasoning, multi-camera view alignment, and object-token grounding remain the primary OOD bottlenecks.

## 简历 / 面试可用说法

我没有把 OOD normal F1 上升包装成泛化成功：通过四路 visual-control 分析发现，模型在 DriveLM 上虽然更会回答，但空白图和错图下也更容易输出高重合答案，因此把跨域 visual grounding 明确写为 limitation，并据此提出 camera/object-aware reward 方向。

## README 可用说法

DriveLM OOD 表明 SFT-v3-r3 具有有限的 normal QA 迁移，但未形成可靠的跨域视觉依赖；空间关系、多相机与对象引用对齐仍是关键限制。本结果仅用于诊断，不参与训练数据构造，也不改变 LingoQA held-out 上的 checkpoint 选择。

## 论文式表述

Although SFT-v3-r3 yields a modest gain in OOD normal-answer accuracy, controlled counterfactual evaluation reveals degraded visual dependency: its responses remain overly compatible with blank and mismatched visual inputs. This gap is most pronounced for spatial-relation cases, suggesting that answer calibration alone does not transfer robust camera- and object-grounded reasoning across datasets.
"""
    Path("outputs/final_report/drivelm_ood_limitation_writeup.md").write_text(content, encoding="utf-8")


def write_reward_implications(facts: dict) -> None:
    spatial = facts["spatial"]
    content = f"""# DriveLM OOD to GRPO Reward Implications

## 1. OOD 暴露的目标错位

DriveLM 证明了只奖励 normal answer 或只惩罚显式 hallucination 不够：r3 的 normal F1 提升，但控制输入下高重合变多。尤其在 `spatial_relation` 中，case_gap={float(spatial['case_gap']):.4f}、control_high_f1={float(spatial['control_high_f1_rate']):.4f}，说明 reward 必须显式关注视觉依赖而非答题风格。

## 2. Reward Harness v2 建议

```text
R = R_normal_answer
  - lambda_control * P_control_high_f1
  - lambda_blank * P_blank_high_f1
  - lambda_wrong * P_wrong_image_confound
  - lambda_text * P_text_direct_answer
  - lambda_camera * P_camera_mismatch
  - lambda_spatial * P_spatial_relation_error
  - lambda_object * P_object_token_mismatch
  - lambda_refusal * P_normal_refusal
```

## 3. 各惩罚项应覆盖的行为

- `P_control_high_f1`：任何 text-only / wrong-image / blank-image 与 gold 高重合，特别是控制输入下直接给出短答案。
- `P_blank_high_f1`：空白图仍输出 yes/no、count、action 或可命中 gold 的简短先验答案。
- `P_wrong_image_confound`：错图回答与 normal 输出相似，或错图 F1 接近/超过 normal F1。
- `P_text_direct_answer`：无图时不表达不确定，而直接复现数据集常见答案。
- `P_camera_mismatch`：问题指定 camera/object view，但生成依据不匹配视角；可结合 camera label 与扰动视角对照。
- `P_spatial_relation_error`：left/right、front/back、lane/relative position 的答案在正确视觉上错误，或在错误/空白视觉上仍高分。
- `P_object_token_mismatch`：`<c...,CAM_...>` 目标引用与回答中的对象/相机对应不上。
- `P_normal_refusal`：正常图像下过度拒答，防止 reward hacking。

## 4. 为什么不能只靠 hallucination reward

显式 hallucination 只捕捉某些拒答/断言形式；DriveLM 的关键问题是答案与 gold 高重合却不依赖正确视觉。case gap、control high-F1、wrong-image/blank-image 对照才能识别这种伪 grounding。

## 5. 是否需要 camera-aware / object-aware reward

需要。DriveLM 的 spatial、camera 与 object-token 问题具有结构化信息，reward v2 应先在离线 case 上验证 camera mismatch 和 object reference 判据的精度，再考虑 RL。

## 6. 当前决策

先实现并审计 reward harness v2，不直接启动 GRPO-lite；当前也不利用 DriveLM OOD 样本构造训练数据。只有在独立主评测与 OOD 诊断都显示 reward 与人工判断一致后，才讨论最小规模 GRPO smoke。
"""
    Path("outputs/final_report/drivelm_to_grpo_reward_implications.md").write_text(content, encoding="utf-8")


def update_reports(facts: dict) -> None:
    base, r3 = facts["base"], facts["r3"]
    taxonomy = facts["taxonomy"]
    spatial = facts["spatial"]
    most = taxonomy["most_frequent_failure_type"]
    count = taxonomy["most_frequent_failure_count"]
    cross = taxonomy.get("cross_tag_analysis", {})
    section = f"""## DriveLM OOD Failure Attribution

Stage 13.5 在不运行模型、不训练、不开启 GPU 的条件下，对已有 DriveLM OOD predictions 做离线归因。r3 的 normal F1 较 Base 提升（{n(base, 'normal_f1'):.4f} -> {n(r3, 'normal_f1'):.4f}），但 case gap 恶化（{n(base, 'case_gap'):.4f} -> {n(r3, 'case_gap'):.4f}），control high-F1 与 blank high-F1 分别上升至 {n(r3, 'control_high_f1_rate_0_20'):.4f} 与 {n(r3, 'blank_high_f1_rate_0_20'):.4f}。因此，r3 体现的是有限 normal-answer style transfer，而非可靠跨域 visual grounding。

- 数量最多的 failure tag：`{most}`（{count}/100，标签可重叠）。
- 样本量充分的主要瓶颈：`spatial_relation`（n={spatial['num_samples']}, case_gap={float(spatial['case_gap']):.4f}, control_high_f1={float(spatial['control_high_f1_rate']):.4f}）。
- Wrong-image confound 共 {cross.get('wrong_image_confound_count', 'n/a')} 条，其中与 object-token / spatial failure 重叠 {cross.get('wrong_with_object_token_failure', 'n/a')} / {cross.get('wrong_with_spatial_relation_failure', 'n/a')} 条。由于 DriveLM 样本普遍携带 camera labels，camera-specific tag 属于宽口径诊断，不应脱离这些交叉统计过度解释。
- Base -> r3 fix/break：{facts['fix_count']} / {facts['break_count']} cases。
- 多相机、camera/object alignment 与空间关系应成为下一版 reward harness 的显式约束。
- DriveLM OOD 被保留为 limitation 与方法设计证据，不用于训练数据，不改变 `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/` 作为当前主模型。
- 当前不建议自动运行 DriveLM 300、不继续训练、不进入 GRPO-lite；下一步应先完成 camera-aware/object-aware reward harness v2 的离线审计。

详细材料：`outputs/final_report/drivelm_ood_limitation_writeup.md` 与 `outputs/final_report/drivelm_to_grpo_reward_implications.md`。
"""
    for path in (Path("outputs/final_report/README_results.md"), Path("README_CN.md"), Path("outputs/final_report/depth_extension_roadmap.md")):
        replace_section(path, section)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Stage 13.5 DriveLM OOD documents.")
    parser.add_argument("--writeup", action="store_true")
    parser.add_argument("--reward_implications", action="store_true")
    parser.add_argument("--update_report", action="store_true")
    args = parser.parse_args()
    facts = load_facts()
    if args.writeup:
        write_limitation(facts)
    if args.reward_implications:
        write_reward_implications(facts)
    if args.update_report:
        update_reports(facts)
    print(json.dumps({"limitation_writeup": args.writeup, "reward_implications": args.reward_implications, "reports_updated": args.update_report, "fix_cases": facts["fix_count"], "break_cases": facts["break_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
