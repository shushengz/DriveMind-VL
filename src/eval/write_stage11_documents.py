"""Write final reports and career-facing documents from Stage 11 tables only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

R3 = "SFT-v3-r3"
CHECKPOINT = "checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/"
BEGIN = "<!-- STAGE11_FINAL_CONSOLIDATION_BEGIN -->"
END = "<!-- STAGE11_FINAL_CONSOLIDATION_END -->"
DISPLAY_FIELDS = [
    ("normal_f1", "Normal F1"),
    ("case_gap", "Case Gap"),
    ("blank_high_f1_rate_0_20", "Blank High-F1"),
    ("control_high_f1_rate_0_20", "Control High-F1"),
    ("control_hallucination_rate", "Hallucination"),
    ("normal_refusal_rate", "Normal Refusal"),
]


def load_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8", newline="")))


def table(rows: list[dict[str, str]]) -> list[str]:
    lines = ["| Model | Method | Normal F1 | Case Gap | Blank High-F1 | Control High-F1 | Hallucination | Normal Refusal |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['model_name']} | {row['method_type']} | {float(row['normal_f1']):.4f} | {float(row['case_gap']):.4f} | {float(row['blank_high_f1_rate_0_20']):.4f} | {float(row['control_high_f1_rate_0_20']):.4f} | {float(row['control_hallucination_rate']):.4f} | {float(row['normal_refusal_rate']):.4f} |")
    return lines


def readme_results(rows: list[dict[str, str]]) -> str:
    return "\n".join([
        "# DriveMind-VL Final Results",
        "",
        "## 1. 项目问题定义",
        "",
        "驾驶视觉语言模型在普通 normal F1 下可能被高估：模型可以利用问题模板、驾驶常识或答案先验，在没有可靠视觉证据时仍输出看似正确的答案。DriveMind-VL 的核心工作不是只提高问答分数，而是诊断并校准模型是否真正依赖图像。",
        "",
        "## 2. Strict Visual-Control Protocol",
        "",
        "在同一 held-out 问题上构造四路输入：`normal` 使用正确视觉信息；`text-only` 移除图像；`wrong-image` 提供不匹配图像；`blank-image` 提供空白图像。四路样本按相同 ID 对齐。",
        "",
        "- `Normal F1`：正确视觉输入下的 answer-only 问答质量。",
        "- `Setting Gap`：总体 normal F1 与最强控制 setting F1 的差。",
        "- `Case Gap`：逐样本 normal F1 减去其最强控制 F1 后的均值。",
        "- `Control Hallucination`：控制条件下仍输出明确视觉断言的比例。",
        "- `Control Direct Answer`：无可靠证据时仍直接作答的比例。",
        "- `Control High-F1 Rate`：控制条件下答案与 gold 的 F1 至少为 0.20 的比例。",
        "- `Normal Refusal`：正常图像下被过度校准为拒答的比例。",
        "",
        "## 3. Evaluation Hygiene",
        "",
        "项目执行了 train/eval leakage audit、answer-only rescore 与独立 held-out strict evaluation。最终表格只从 `outputs/predictions_heldout` 中同一组 100 个 held-out IDs 的原始预测重算；Stage 4 的 pre-heldout/leaked 表仅保留为审计轨迹，不作为最终结论。",
        "",
        "## 4. Method Evolution",
        "",
        "- `Base / SFT-v2 / DPO-v7`：建立基础视觉问答与偏好校准参照。",
        "- `SFT-v3-r2`：增强控制校准的 SFT 尝试，揭示正常能力与保守输出之间的权衡。",
        "- `SFT-v3-r3`：采用 answer-only calibration 与更稳健 normal replay，成为最终主模型。",
        "- `DPO-v8`：rule-based preference，可轻微减少显式 hallucination，但 case gap 退化。",
        "- `DPO-v8.1`：model-mined rejected preference，暴露 blank-image high-F1 prior answer 回归。",
        "- `DPO-v8.2`：case-gap-aware preference，针对上述失败模式进行 smoke 验证，但未稳定解决问题。",
        "",
        "## 5. Main Results",
        "",
        *table(rows),
        "",
        f"当前推荐 checkpoint：`{CHECKPOINT}`。",
        "",
        "## 6. Key Findings",
        "",
        "1. `SFT-v3-r3` 是当前最佳主模型，在 normal 能力、case gap 与零 normal refusal 之间最稳定。",
        "2. answer-only output 与 answer-only rescore 对公平比较很关键，避免 reason/格式干扰分数。",
        "3. held-out evaluation 避免将被训练或调参间接使用的样本写进最终结论。",
        "4. DPO preference tuning 可以轻微降低部分显式 control hallucination。",
        "5. DPO-v8/v8.1/v8.2 均没有稳定改善 case gap，因此只作为消融和机制发现。",
        "6. blank-image high-F1 prior answer 是最关键的未解决失败模式。",
        "7. control hallucination 并非充分指标，必须同时报告 high-F1 overlap 与 direct-answer 行为。",
        "",
        "## 7. Case Study",
        "",
        "详见 [final_case_gallery.html](final_case_gallery.html)，其中按 failure type 汇集了 normal、text-only、wrong-image 与 blank-image 的 raw outputs 对比。",
        "",
        "## 8. Limitations",
        "",
        "1. 当前最终 held-out 集仅包含 100 cases，统计置信度仍有限。",
        "2. DriveLM OOD 尚未形成可作为最终主结论的统一评测。",
        "3. DPO 未能稳定解决 case gap 或 blank-image prior 答案。",
        "4. GRPO-lite 尚未训练，本报告不包含其性能声明。",
        "5. 当前方法仍可能受数据集语言先验和问题模板分布影响。",
        "",
        "## 9. Future Work",
        "",
        "1. 将 held-out 扩展至 300/500 cases 并重做置信区间分析。",
        "2. 建立 DriveLM OOD strict visual-control evaluation。",
        "3. 设计可审计 reward harness，将 case gap 与过度拒答纳入统一目标。",
        "4. 在满足进入条件后进行最小 GRPO-lite smoke，而非直接长训。",
        "5. 增强 object-aware / grounding-aware 的细粒度视觉依赖评估。",
        "",
    ]) + "\n"


def readme_cn_section(rows: list[dict[str, str]]) -> str:
    return "\n".join([
        BEGIN,
        "## 项目总览与最终结论（Stage 11）",
        "",
        "### 1. 项目定位与研究问题",
        "",
        "DriveMind-VL 是面向驾驶 VLM 的视觉依赖诊断与后训练校准系统。项目关注的问题是：一个在普通问答上表现较好的模型，是否真的根据图像作答，还是在缺失或错误视觉输入下依赖语言先验猜测。",
        "",
        "### 2. 方法总览",
        "",
        "方法链路包括 strict visual-control protocol、train/eval leakage audit、answer-only rescore、SFT calibration、三代 DPO preference ablation，以及基于 case gap 和 high-F1 control 行为的失败归因。最终主模型来自 SFT 分支，DPO 分支保留为消融证据。",
        "",
        "### 3. Strict Visual-Control Protocol",
        "",
        "所有最终结论都在共同 held-out 100 IDs 上评测，并对齐 `normal`、`text-only`、`wrong-image`、`blank-image` 四路输入。主指标包括 Normal F1、Case Gap、Control Hallucination、Control Direct Answer、Control High-F1、Blank High-F1 与 Normal Refusal。",
        "",
        "### 4. 后训练流程与数据评测",
        "",
        "SFT-v3-r3 通过 answer-only calibration 与稳健 normal replay 获得最佳综合表现。随后 DPO-v8（rule-based）、DPO-v8.1（model-mined）与 DPO-v8.2（case-gap-aware）均以 r3 为初始化或参考进行受控 smoke；最终只使用 held-out strict evaluation 与 answer-only 评分作结论。",
        "",
        "### 5. 最终结果",
        "",
        *table(rows),
        "",
        f"最终 checkpoint：`{CHECKPOINT}`。",
        "",
        "### 6. 消融与 Case Gallery",
        "",
        "- DPO 可降低部分显式 hallucination，但未稳定改善 case gap。",
        "- blank-image high-F1 prior answer 与 control direct-answer overlap 是后续最需要处理的失败模式。",
        "- 详细消融见 `outputs/final_report/final_ablation_results.md`；案例见 `outputs/final_report/final_case_gallery.html`。",
        "",
        "### 7. 已知问题与后续计划",
        "",
        "当前 held-out 规模为 100，DriveLM OOD 尚未作为最终结论；DPO 没有成为最终主模型，GRPO-lite 也尚未训练。当前不建议继续盲目追加 DPO steps，也不建议直接进入 GRPO-lite。下一阶段应优先扩大评测并设计 reward harness。",
        "",
        "### 8. 一键复现最终报告",
        "",
        "```bash",
        "bash scripts/run_stage11_final_report.sh --all --run",
        "```",
        "",
        "该命令只读取已存在的结果与 predictions，生成最终表格、报告、gallery 和职业材料；不训练、不推理、不使用 GPU。",
        END,
    ]) + "\n"


def resume_document() -> str:
    return """# 简历与面试材料

## 简历项目标题

**DriveMind-VL：面向驾驶 VLM 的视觉依赖诊断与后训练校准系统**

## 简历 Bullet 中文版

- 设计驾驶 VLM 的 strict visual-control 评测协议，对齐 normal、text-only、wrong-image、blank-image 四路 held-out 输入，以 Case Gap 与 Control High-F1 识别“脱离图像仍可作答”的语言先验捷径。
- 建立 train/eval leakage audit 与 answer-only rescore 流程，在共同 held-out 100 cases 上完成 Base、SFT 与三代 DPO preference tuning 的可复现实验对比。
- 迭代 SFT 校准方案并选定 `SFT-v3-r3` 为当前最佳模型，held-out `normal_f1=0.3278`、`case_gap=-0.0011` 且 `normal_refusal=0`。
- 构建 rule-based、model-mined 与 case-gap-aware DPO 消融，发现 DPO 可轻微降低显式幻觉但无法稳定改善 case-level visual dependency。
- 通过 case attribution 定位 blank-image high-F1 prior answer 失败模式，为后续 reward harness 与受控 GRPO-lite 方案提供设计依据。

## Resume Bullets (English)

- Designed a strict visual-control evaluation protocol for driving VLMs with aligned normal, text-only, wrong-image and blank-image held-out inputs, exposing language-prior shortcuts via Case Gap and Control High-F1 metrics.
- Built leakage-audited, answer-only evaluation pipelines and compared Base, SFT, and three generations of DPO preference tuning on a shared held-out set of 100 cases.
- Selected `SFT-v3-r3` as the most reliable checkpoint, reaching held-out `normal_f1=0.3278`, `case_gap=-0.0011`, and zero normal refusals.
- Conducted rule-based, model-mined and case-gap-aware DPO ablations, showing that lower explicit hallucination does not reliably translate into stronger visual dependency.
- Identified blank-image high-F1 prior answers as a key failure mode and translated the diagnosis into a reward-harness roadmap for future policy optimization.

## 面试讲法

### 1. 项目背景

我关注驾驶场景 VLM 的可靠性：模型即使 normal F1 较高，也可能没有真正使用图像，而是用语言或驾驶常识先验猜答案。

### 2. 为什么普通 F1 不够

普通 F1 只测正确输入下能否命中答案，不测答案是否来自视觉证据。因此我设计同题四路控制输入，观察图像被移除、置错或置空后模型是否仍“答对”。

### 3. 四路 visual-control 怎么做

对同一个 held-out ID 生成 normal、text-only、wrong-image、blank-image 四组预测，并用 answer-only 统一评分。Case Gap 衡量逐样本 normal 相对最强控制答案的优势，high-F1 指标捕获无图仍与 gold 高重合的先验答案。

### 4. 为什么 r2 失败

r2 偏向更强控制校准，容易把保守行为做强，却没有在正常问答能力与视觉依赖之间取得最稳平衡。

### 5. 为什么 r3 成功

r3 调整了 normal replay 与 answer-only calibration，保持正常回答能力并避免 normal 拒答；在 held-out 上其 case gap 最稳定，因此被选为主模型。

### 6. 为什么 DPO 没有最终成功

DPO 确实轻微降低了部分明确 hallucination，但控制条件下的高 F1 先验回答没有稳定下降，case gap 反而常变差。这说明优化的 preference target 与最终视觉依赖目标仍不完全一致。

### 7. 学到了什么

可靠性问题不能只看一个漂亮指标；需要无泄漏评测、正确评分口径，以及能解释失败模式的控制实验。负结果也能转化为明确的下一轮目标。

### 8. 继续做 GRPO-lite 会怎么做

不会直接启动训练，而是先构建 reward harness，将 normal 正确性、control high-F1、blank prior、wrong-image confound、过度拒答和长度惩罚统一编码，并在更大 held-out 集上先验证 reward 是否与目标一致。

## 面试问答

1. **为什么要做 text-only / wrong-image / blank-image？**
   它们分别测试语言先验、错误视觉干扰和无视觉证据条件，使“答对但没看图”的行为可测。

2. **Case Gap 是什么？**
   对每个样本，用 normal F1 减去三个控制条件中最高的 F1，再求均值；越高越说明答案依赖正确图像。

3. **为什么 answer-only eval 重要？**
   模型可能输出 JSON 或解释文本；把 reason 一起评分会混入格式和冗余表述的影响，歪曲答案质量。

4. **train/eval leakage 怎么发现的？**
   对训练构造来源与评测 IDs 做集合审计，发现早期结果不能作为独立最终证据，因此转为独立 held-out 100 IDs。

5. **为什么 r3 比 r2 好？**
   r3 更好地平衡 normal replay 和控制校准，在保住 normal F1 的同时维持零正常拒答及更稳定 case gap。

6. **为什么 DPO 没有成为最终模型？**
   DPO 的 hallucination 改善较小，且 high-F1 control 与 case gap 未稳定改善；可靠性不能以单一指标换取。

7. **什么是 blank-image high-F1 prior answer？**
   输入空白图像时答案仍与 gold 高重合，说明模型可能根据问题先验猜对，而非依赖视觉证据。

8. **你如何避免过度拒答？**
   保留 normal anchor/replay，并把 normal refusal 作为硬评测指标；正常图像上的拒答上升即判定校准失败。

9. **为什么现在不直接上 GRPO？**
   DPO 已暴露 objective mismatch；若 reward 没设计好，RL 可能只学会拒答或投机压低某个指标。

10. **这个项目和自动驾驶 / VLA / Embodied AI 有什么关系？**
    它研究的是视觉证据是否真正驱动决策语言，这正是驾驶决策、VLA 控制与 embodied agent 安全泛化的共同基础。
"""


def grpo_plan() -> str:
    return """# GRPO-lite Future Plan（不训练）

## 1. 为什么现在还不训练 GRPO

Stage 10 已证明：在严格视觉控制评测下，偏好优化能够轻微降低显式 hallucination，却不能稳定提升 case gap。没有验证好的 reward harness 时直接进入 RL，会把目标错配放大，或诱发过度拒答。

## 2. DPO 暴露出的 Objective Mismatch

DPO preference 主要鼓励谨慎答案，但最终目标同时要求 normal 问答正确、错误视觉证据下不猜测、且不过度拒答。单纯压低 hallucination 不足以压低 control high-F1 overlap，也不能保证 case-level visual dependency。

## 3. Reward 应对齐的指标

- normal 输入的 answer-only 正确性；
- text-only / wrong-image / blank-image 下的高 F1 重合；
- 控制条件下的 direct answer 与具体视觉断言；
- normal 输入下的过度拒答；
- 冗长或模板化规避答案。

## 4. 建议 Reward

```text
R = R_normal_answer
  - lambda1 * R_control_high_f1
  - lambda2 * R_blank_high_f1
  - lambda3 * R_text_direct_answer
  - lambda4 * R_wrong_image_confound
  - lambda5 * R_over_refusal
  - lambda6 * R_length_penalty
```

其中 `R_normal_answer` 必须承担主锚点作用，防止策略通过全部拒答获得表面上的控制安全。

## 5. Reward Hacking 风险

- 模型对所有输入统一输出“信息不足”，压低 control 指标却损害 normal QA；
- 模型输出短而模糊的安全话术规避 hallucination detector；
- 奖励过度依赖词法匹配，导致对未覆盖表达方式失真；
- 在小 held-out 集上过拟合评测模式。

## 6. 最小 GRPO-lite Smoke 方案

1. 先将 held-out 扩展到至少 300 cases，并冻结评测 ID。
2. 在无训练条件下验证 reward 与现有 case gallery 排序是否一致。
3. 固定从 `SFT-v3-r3` 起点启动极小步数 smoke，禁止自动加 steps。
4. 同时报 normal_f1、case_gap、control/blank high-F1、hallucination、normal refusal 与答案长度。
5. 任一 normal 能力或拒答门禁失败即停止，保留 r3。

## 7. 进入 GRPO-lite 的条件

- reward harness 通过人工案例核对，能区分 visual gain、prior answer 和 over-refusal；
- held-out 规模扩大并保留独立 OOD 验证集；
- DPO 分析结论已转化为明确 reward 项和安全阈值；
- 明确授权小规模 GPU smoke，且不将探索结果直接当作最终主模型。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Stage 11 final narrative documents.")
    parser.add_argument("--results", default="outputs/final_report/final_main_results_answer_only.csv")
    parser.add_argument("--readme_cn", default="README_CN.md")
    parser.add_argument("--output_dir", default="outputs/final_report")
    parser.add_argument("--readme_results", action="store_true")
    parser.add_argument("--update_readme_cn", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--grpo_plan", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.all:
        args.readme_results = args.update_readme_cn = args.resume = args.grpo_plan = True
    rows = load_rows(Path(args.results))
    if not any(row["model_name"] == R3 for row in rows):
        raise ValueError("final results do not contain SFT-v3-r3")
    actions = []
    if args.dry_run:
        print(json.dumps({"would_write": ["README_results.md", "README_CN.md", "resume_and_interview.md", "grpo_lite_future_plan.md"]}, ensure_ascii=False, indent=2))
        return
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.readme_results:
        (output / "README_results.md").write_text(readme_results(rows), encoding="utf-8")
        actions.append("README_results.md")
    if args.update_readme_cn:
        readme_path = Path(args.readme_cn)
        current = readme_path.read_text(encoding="utf-8")
        section = readme_cn_section(rows)
        if BEGIN in current and END in current:
            start = current.index(BEGIN)
            stop = current.index(END, start) + len(END)
            current = current[:start].rstrip() + "\n\n" + section + current[stop:].lstrip()
        else:
            current = current.rstrip() + "\n\n" + section
        readme_path.write_text(current, encoding="utf-8")
        actions.append("README_CN.md")
    if args.resume:
        (output / "resume_and_interview.md").write_text(resume_document(), encoding="utf-8")
        actions.append("resume_and_interview.md")
    if args.grpo_plan:
        (output / "grpo_lite_future_plan.md").write_text(grpo_plan(), encoding="utf-8")
        actions.append("grpo_lite_future_plan.md")
    print(json.dumps({"written": actions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
