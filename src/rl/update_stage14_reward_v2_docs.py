"""Update blueprint and reports with audited Stage 14 reward-v2 findings."""
from __future__ import annotations

import csv
import json
from pathlib import Path

BEGIN = "<!-- STAGE14_REWARD_V2_BEGIN -->"
END = "<!-- STAGE14_REWARD_V2_END -->"


def read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def replace(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"\n{BEGIN}\n{section.rstrip()}\n{END}\n"
    if BEGIN in text and END in text:
        text = text.split(BEGIN, 1)[0].rstrip() + block + text.split(END, 1)[1].lstrip("\n")
    else:
        text = text.rstrip() + "\n" + block
    path.write_text(text, encoding="utf-8")


def main() -> None:
    readiness = json.loads(Path("outputs/final_report/grpo_lite_reward_v2_readiness.json").read_text(encoding="utf-8"))
    summary = read_rows("outputs/final_report/grpo_lite_reward_v2_summary.csv")
    sensitivity = read_rows("outputs/final_report/grpo_lite_reward_v2_sensitivity.csv")
    model = {(row["dataset"], row["model_name"]): row for row in summary}
    lingo_rank = " > ".join(readiness["lingoqa_ranking"])
    drive_rank = " > ".join(readiness["drivelm_ranking"])
    stable_count = sum(row["lingoqa_r3_above_dpo_v8_2"] == "True" and row["drivelm_base_above_r3"] == "True" for row in sensitivity)
    drive_r3 = model[("drivelm", "SFT-v3-r3")]
    ready = readiness["ready_for_grpo_lite"]
    verdict = "可在另行授权后考虑最小 GPU smoke" if ready else "Do not train GRPO-lite yet."
    section = f"""## GRPO-lite Reward Harness v2

Stage 14 在 CPU-only 条件下对已有 LingoQA 与 DriveLM predictions 构建了可审计 reward harness v2，不启动推理或训练。v1 仅覆盖一般 control/blank/wrong/refusal 行为；v2 新增了 DriveLM 暴露出的 `camera_mismatch`、`spatial_relation_error` 与 `object_token_mismatch` 结构化惩罚。

### Reward 组成与默认权重

```text
R = 1.0 * R_normal_answer
  - 0.8 * P_control_high_f1
  - 0.8 * P_blank_high_f1
  - 0.6 * P_text_direct_answer
  - 0.8 * P_wrong_image_confound
  - 0.5 * P_camera_mismatch
  - 0.7 * P_spatial_relation_error
  - 0.6 * P_object_token_mismatch
  - 0.8 * P_normal_refusal
  - 0.1 * P_length
```

### 离线结果

- LingoQA reward 排名：`{lingo_rank}`。
- DriveLM reward 排名：`{drive_rank}`；r3 的 camera / spatial / object 平均惩罚为 {f(drive_r3, 'mean_camera_penalty'):.4f} / {f(drive_r3, 'mean_spatial_penalty'):.4f} / {f(drive_r3, 'mean_object_penalty'):.4f}。
- Sensitivity：6 组可解释权重扰动中，所需 LingoQA/DriveLM 排序同时保持 {stable_count}/6 组，说明当前 reward 仍存在排名敏感性。
- Normal refusal sanity gate 已加入，正常图像全拒答不会得到高 reward。

### 当前决策

**{verdict}** Reward v2 已能惩罚 DriveLM OOD 的结构化失败与 blank/control overlap，但尚不能稳定、完整解释 LingoQA 最终模型选择；在进入 GRPO-lite 前，必须先继续校准 reward 并人工审阅 disagreement cases。禁止使用 held-out 或 DriveLM OOD evaluation cases 作为训练样本。
"""
    for path in (Path("README_CN.md"), Path("outputs/final_report/README_results.md"), Path("outputs/final_report/depth_extension_roadmap.md")):
        replace(path, section)
    blueprint = f"""## Reward v2 Gate

### Reward v2 组成

v2 在 normal/control/blank/text/wrong/refusal/length 的基础上，加入 camera mismatch、spatial relation error 与 object-token mismatch 惩罚；默认 lambda 为 `1.0 / 0.8 / 0.8 / 0.6 / 0.8 / 0.5 / 0.7 / 0.6 / 0.8 / 0.1`（依次对应 normal、control、blank、text、wrong、camera、spatial、object、refusal、length）。

### 离线验证结果

- LingoQA ranking: `{lingo_rank}`。
- DriveLM ranking: `{drive_rank}`。
- Sensitivity gate: {stable_count}/6 权重配置同时维持所需主评测/OOD 排序。

### 是否 Ready

**{verdict}**

当前失败保护：

1. 不允许直接长训；
2. 不允许使用 LingoQA held-out 或 DriveLM OOD eval cases 训练；
3. reward 排名与人工 case review 不一致时停止；
4. normal refusal、blank prior 与 camera/spatial/object penalty 必须持续保留；
5. 仅在 gate 通过并得到单独授权后，才可考虑 r3 初始化、frozen r3 reference、`group_size=4`、`temperature=0.7`、`max_new_tokens=64`、最多 `50 steps` 的最小 smoke。
"""
    replace(Path("outputs/final_report/grpo_lite_training_blueprint.md"), blueprint)
    print(json.dumps({"ready_for_grpo_lite": ready, "verdict": verdict, "sensitivity_consistent": stable_count, "documents_updated": ["README_CN.md", "outputs/final_report/README_results.md", "outputs/final_report/depth_extension_roadmap.md", "outputs/final_report/grpo_lite_training_blueprint.md"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
