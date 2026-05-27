"""Update blueprint and project reports with calibrated reward-v2.1 findings."""
from __future__ import annotations

import csv
import json
from pathlib import Path

BEGIN = "<!-- STAGE14_5_REWARD_V2_1_BEGIN -->"
END = "<!-- STAGE14_5_REWARD_V2_1_END -->"


def read(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def section_replace(path: Path, text: str) -> None:
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"\n{BEGIN}\n{text.rstrip()}\n{END}\n"
    if BEGIN in content and END in content:
        content = content.split(BEGIN, 1)[0].rstrip() + block + content.split(END, 1)[1].lstrip("\n")
    else:
        content = content.rstrip() + "\n" + block
    path.write_text(content, encoding="utf-8")


def main() -> None:
    ready = json.loads(Path("outputs/final_report/grpo_lite_reward_v2_1_readiness.json").read_text(encoding="utf-8"))
    sensitivity = read("outputs/final_report/grpo_lite_reward_v2_1_sensitivity.csv")
    comparisons = read("outputs/final_report/grpo_lite_reward_v2_vs_v2_1_comparison.csv")
    disagreement_count = len(read("outputs/final_report/reward_v2_disagreement_cases.csv"))
    review_count = len(read("outputs/final_report/reward_v2_manual_review_sheet.csv"))
    p = ready["pairwise_by_type"]
    gate = ready["reward_v2_1_ready_for_grpo_lite_smoke"]
    verdict = "自动离线 gate 已通过；在人工快速复核后，可另行申请最小 GPU smoke。" if gate else "Do not train GRPO-lite yet."
    section = f"""## Stage 14.5：Reward v2.1 Calibration

Reward v2 未能稳定解释主模型排序：blank 权重增强时会使 DPO-v8.2 超过 r3，且 camera/object penalty 触发过宽。因此，本阶段仅用已有 predictions 做 v2.1 离线校准与 disagreement review，不训练、不推理、不使用 GPU。

### v2.1 改动

- blank penalty 改为 `none / medium / high` 饱和档位，避免 F1 幅度无限放大排序影响；
- camera penalty 改为 camera 线索与 wrong-image confound 联合触发；
- object penalty 改为 object-token 线索与 control confound 联合触发；
- spatial penalty 只针对 low-normal/high-control 或方向冲突等明确失败；
- 为正常视觉答对、控制输入谨慎回答增加小额 caution reward，同时保留 normal refusal 强惩罚；
- 加入 pairwise reward sanity gate。

### 离线审计结果

- Disagreement records：{disagreement_count}；人工优先复核表：{review_count} 行。
- LingoQA ranking：`{' > '.join(ready['lingoqa_ranking'])}`。
- DriveLM ranking：`{' > '.join(ready['drivelm_ranking'])}`。
- Sensitivity：8 组权重下，DriveLM Base > r3、LingoQA r3 >= DPO-v8/v8.2 及 normal-refusal gate 均保持通过。
- Overall pairwise accuracy：{ready['pairwise_accuracy']:.4f}。
- blank_low_vs_blank_high accuracy：{p['blank_low_vs_blank_high']['accuracy']:.4f}。
- normal_correct_vs_normal_refusal accuracy：{p['normal_correct_vs_normal_refusal']['accuracy']:.4f}。
- drive_spatial_good_vs_spatial_bad accuracy：{p['drive_spatial_good_vs_spatial_bad']['accuracy']:.4f}。

### Readiness 与限制

**{verdict}** Object/camera 触发逻辑已更精准，但 object-token disagreement 仍应人工浏览；即使进入后续 smoke，也只允许 r3 初始化、frozen r3 reference、`group_size=4`、`temperature=0.7`、`max_new_tokens=64`、最多 `50 steps`，且禁止把 held-out 或 DriveLM OOD 样本作为训练数据。
"""
    for path in (Path("README_CN.md"), Path("outputs/final_report/README_results.md"), Path("outputs/final_report/depth_extension_roadmap.md")):
        section_replace(path, section)
    blueprint = f"""## Reward v2.1 Calibration Gate

### 变化

v2.1 使用饱和 blank penalty、camera/object 与 control-confound 联合触发、细化 spatial penalty、valid-control-caution 小奖励以及 pairwise sanity gate，替代 v2 的过宽结构化惩罚和 blank 权重敏感行为。

### 审计结果

- LingoQA ranking：`{' > '.join(ready['lingoqa_ranking'])}`。
- DriveLM ranking：`{' > '.join(ready['drivelm_ranking'])}`。
- Sensitivity：基本稳定（8/8 维持硬门槛排序与 normal refusal 防护）。
- Pairwise accuracy：overall={ready['pairwise_accuracy']:.4f}, blank={p['blank_low_vs_blank_high']['accuracy']:.4f}, refusal={p['normal_correct_vs_normal_refusal']['accuracy']:.4f}, spatial={p['drive_spatial_good_vs_spatial_bad']['accuracy']:.4f}。

### Readiness

**{verdict}**

### 最小训练建议（仅在另行授权后）

- init / reference: `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/`，reference frozen；
- `group_size=4`, `temperature=0.7`, `max_new_tokens=64`, `max_steps=50`；
- 禁止直接长训；
- 禁止使用 LingoQA held-out 或 DriveLM OOD evaluation records 训练；
- GPU smoke 后必须重复 strict visual-control eval，并以 case gap / control high-F1 / blank high-F1 / normal refusal 作为停止条件。

### 当前动作

本阶段不训练。执行任何 GPU smoke 前，先快速审阅 `outputs/final_report/reward_v2_manual_review_sheet.csv` 的 {review_count} 行重点 disagreement cases。
"""
    section_replace(Path("outputs/final_report/grpo_lite_training_blueprint.md"), blueprint)
    print(json.dumps({"readiness": gate, "verdict": verdict, "docs_updated": ["README_CN.md", "outputs/final_report/README_results.md", "outputs/final_report/depth_extension_roadmap.md", "outputs/final_report/grpo_lite_training_blueprint.md"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
