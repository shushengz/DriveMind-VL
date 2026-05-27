"""Write Stage 12 blueprint, roadmap, and README consolidation from offline outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BEGIN = "<!-- STAGE12_DEPTH_EXTENSION_BEGIN -->"
END = "<!-- STAGE12_DEPTH_EXTENSION_END -->"


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_reward_summary(path: str) -> list[dict[str, str]]:
    return list(csv.DictReader(Path(path).open(encoding="utf-8", newline="")))


def blueprint() -> str:
    return """# GRPO-lite Minimal Training Blueprint（仅计划，不训练）

## 1. 为什么需要 GRPO-lite

DPO-v8/v8.1/v8.2 表明，偏好微调可以稍微降低显式 hallucination，但无法稳定提高 case-level visual dependency。若希望同时优化正常回答质量和控制条件可靠性，需要能够直接组合这些指标的 reward-based 方法。

## 2. DPO 的 Objective Mismatch

DPO 倾向于把 chosen caution 拉高，但没有直接保证 `normal_f1`、`case_gap`、`blank_high_f1`、`control_high_f1` 与 `normal_refusal` 的联合最优。GRPO-lite 仅在 reward harness 通过离线验证后才值得尝试。

## 3. Reward 组成

```text
R = R_normal_answer
  - lambda_control_f1 * P_control_high_f1
  - lambda_blank * P_blank_high_f1
  - lambda_text * P_text_direct_answer
  - lambda_wrong * P_wrong_image_confound
  - lambda_refusal * P_normal_refusal
  - lambda_length * P_length
```

实现参考：`src/rl/reward_vc_grpo_lite.py`。

## 4. Reward Hacking 风险

- 对 normal 和 control 一律拒答以规避控制惩罚；
- 输出极短、安全但无信息的套话；
- 通过冗长复述提高词法重合；
- 适配小评测集题型先验而非提升视觉 grounding。

## 5. 训练数据来源与隔离原则

- 训练仅使用已明确归入 train/mining pool 的样本及其严格视觉控制变体；
- 禁止使用 `outputs/final_report/lingoqa_heldout_ids_*.json` 及 DriveLM OOD IDs 作为训练输入；
- held-out 与 OOD 只用于停止条件和独立诊断。

## 6. 最小 Smoke 配置草案

- init adapter: `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/`
- reference: frozen `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/`
- `group_size = 4`
- `temperature = 0.7`
- `max_new_tokens = 64`
- steps: `50` 或 `100` smoke，禁止自动追加

## 7. 成功条件

- `normal_f1 >= r3 normal_f1 - 0.02`
- `case_gap >= r3 case_gap`
- `control_high_f1_rate` 下降
- `blank_high_f1_rate` 下降
- `normal_refusal_rate <= 0.02`

## 8. 失败条件

- normal F1 下跌超过 0.02；
- case gap 恶化；
- control/blank high-F1 上升；
- normal refusal 超过 0.02；
- reward/debug 案例显示拒答或短套话投机。

## 9. 需要 GPU 的部分

仅未来授权后的 sampling、GRPO-lite smoke 训练和 held-out/OOD 推理评测需要 GPU。本 Stage 12 不执行这些任务。

## 10. 下一步命令草案（不执行）

```bash
# 在 reward harness、LingoQA larger held-out 与 DriveLM OOD 评估验证后再实现并运行：
# bash scripts/run_grpo_lite_smoke.sh --run --steps 50 --group_size 4 --temperature 0.7
```
"""


def roadmap(lingo: dict, drive: dict, rewards: list[dict[str, str]]) -> str:
    reward_leader = rewards[0]["model_name"] if rewards else "n/a"
    return f"""# Depth Extension Roadmap

## 1. 当前项目深度评估

DriveMind-VL 已从单一分数优化扩展为可审计的视觉依赖研究链路：包含严格对照评测、泄漏控制、SFT 校准、DPO 消融、失败归因与 reward harness 设计。当前可信主模型仍是 `SFT-v3-r3`，其结论建立在独立 held-out 100 上。

## 2. 已完成能力

- strict visual-control protocol；
- train/eval leakage audit 与 answer-only rescore；
- held-out strict evaluation；
- SFT-v3-r3 calibration；
- DPO-v8/v8.1/v8.2 ablation；
- case-gap 与 blank prior failure attribution；
- GRPO-lite offline reward harness design/debug。

## 3. 仍需补齐

- LingoQA 300/500 独立 held-out：当前 leakage-free eligible IDs 为 `{lingo['eligible_heldout_ids']}`，现有资产不足以形成真实 300/500；
- DriveLM OOD GPU inference：结构合格的 OOD pool 为 `{drive['eligible_ood_ids']}` 条；
- reward harness 在未来新输出上的防投机验证；
- 可选的 GRPO-lite smoke training；
- 方法图、可视化案例页与开源整理。

## 4. 推荐执行顺序

| 顺序 | 任务 | 是否需要 GPU | 预期耗时 | 主要风险 |
| --- | --- | --- | --- | --- |
| A | 当前 CPU-only 池构造与 reward debug | 否 | 已完成 | 数据不足会限制统计结论 |
| B | 获取未见 LingoQA 样本后运行 300/500 larger held-out eval | 是，仅推理 | 0.5-1 天 | 目前仅 {lingo['eligible_heldout_ids']} 条可用 |
| C | DriveLM OOD strict eval（优先 Base 与 r3） | 是，仅推理 | 0.5-1 天 | 多相机对齐与早期分支数据暴露限制 |
| D | 根据 B/C 结果再次 CPU reward harness debug | 否 | 0.5 天 | lambda 与真实目标不一致 |
| E | 可选 GRPO-lite 50/100-step smoke | 是，训练+评测 | 1 天 | reward hacking / normal 能力损伤 |

## 5. 当前主要结论

- LingoQA larger held-out 不能凭现有数据直接宣称有 300/500；必须补充未参与训练与挖掘的新样本。
- DriveLM OOD 池足够大，可作为下一次最有价值的跨数据集泛化诊断，但不应替代 LingoQA 主结论。
- 离线 reward 当前排名第一模型为 `{reward_leader}`；只有 reward 与人工案例判断一致后才考虑 RL。
- 当前仍不建议进入 GRPO-lite 训练；最优下一步是获得更大干净 held-out，并运行 DriveLM OOD 推理评测。
"""


def readme_section(lingo: dict, drive: dict, rewards: list[dict[str, str]]) -> str:
    leader = rewards[0]["model_name"] if rewards else "n/a"
    return f"""\
{BEGIN}
## Stage 12：Depth Extension Preparation

Stage 11 已将当前实验收束为可靠的 held-out 结论，但研究深度仍受两个边界限制：LingoQA 独立 held-out 仅 100 条，且 DriveLM 尚未作为 OOD strict visual-control 诊断运行。因此 Stage 12 不继续盲目训练，而是在 CPU 上准备扩大评测与未来 reward-based 优化的可审计基础设施。

### CPU-only 准备内容

- **LingoQA larger held-out**：统一 ID normalize 并排除 SFT、DPO preference 与 mining 来源。当前候选经排除后仅有 `{lingo['eligible_heldout_ids']}` 条 leakage-free IDs；`300/500` 输出文件已生成但仅包含真实可用数量，不能作为完成的大规模评测集宣称。
- **DriveLM OOD**：已构造 `{drive['eligible_ood_ids']}` 条结构验证通过的四路 strict visual-control 候选，可用于最终 r3 的跨数据集诊断。DriveLM 不要求一定提升，重点检查多相机和 object-token 泛化。
- **GRPO-lite reward harness**：已实现离线奖励分解与已有 prediction debug，当前 reward 排名第一为 `{leader}`。奖励显式惩罚 blank/control high-F1、text direct answer、wrong-image confound 与 normal refusal。

### 下一步需要 GPU 的任务

1. 在获得新的干净 LingoQA 样本后运行 larger held-out inference；
2. 运行 DriveLM OOD inference（建议先 Base 与 SFT-v3-r3）；
3. 只有在 reward harness 经更大评测验证后，才考虑 GRPO-lite smoke 训练。

本阶段不训练 GRPO 的原因是：DPO 已显示 objective mismatch，而现有 LingoQA 干净 held-out 尚不足 300 条。先完善独立评测与 OOD 证据，比继续调参更能提高项目可信度。

复现 CPU-only 准备流程：

```bash
bash scripts/run_stage12_depth_extension_cpu.sh --all --run --seed 42
```

该命令不加载模型、不运行推理、不使用 GPU。
{END}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Stage 12 documentation from offline summaries.")
    parser.add_argument("--lingoqa_summary", default="outputs/final_report/lingoqa_large_heldout_pool_summary.json")
    parser.add_argument("--drivelm_summary", default="outputs/final_report/drivelm_ood_pool_summary.json")
    parser.add_argument("--reward_summary", default="outputs/final_report/grpo_lite_reward_summary.csv")
    parser.add_argument("--readme", default="README_CN.md")
    args = parser.parse_args()
    lingo = load_json(args.lingoqa_summary)
    drive = load_json(args.drivelm_summary)
    rewards = load_reward_summary(args.reward_summary)
    output = Path("outputs/final_report")
    (output / "grpo_lite_training_blueprint.md").write_text(blueprint(), encoding="utf-8")
    (output / "depth_extension_roadmap.md").write_text(roadmap(lingo, drive, rewards), encoding="utf-8")
    readme = Path(args.readme)
    current = readme.read_text(encoding="utf-8")
    section = readme_section(lingo, drive, rewards)
    if BEGIN in current and END in current:
        start = current.index(BEGIN)
        stop = current.index(END, start) + len(END)
        current = current[:start].rstrip() + "\n\n" + section + current[stop:].lstrip()
    else:
        current = current.rstrip() + "\n\n" + section
    readme.write_text(current, encoding="utf-8")
    print(json.dumps({"blueprint": True, "roadmap": True, "readme_updated": True, "reward_leader": rewards[0]["model_name"] if rewards else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
