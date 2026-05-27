"""Append/update the Stage 14.6 summary in project-facing documents."""
from __future__ import annotations

import json
from pathlib import Path

BEGIN = "<!-- STAGE14_6_REWARD_V2_2_BEGIN -->"
END = "<!-- STAGE14_6_REWARD_V2_2_END -->"


def replace_block(path: Path, text: str) -> None:
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"\n{BEGIN}\n{text.rstrip()}\n{END}\n"
    if BEGIN in content and END in content:
        content = content.split(BEGIN, 1)[0].rstrip() + block + content.split(END, 1)[1].lstrip("\n")
    else:
        content = content.rstrip() + "\n" + block
    path.write_text(content, encoding="utf-8")


def main() -> None:
    ready = json.loads(Path("outputs/final_report/grpo_lite_reward_v2_2_readiness.json").read_text(encoding="utf-8"))
    p = ready["pairwise_by_type"]
    recommendation = ready["recommendation"]
    section = f"""## Stage 14.6：Reward v2.2 Patch and Manual Review Integration

Stage 14.5 的自动门槛虽已通过，但人工复核指出了四类 v2.1 漏洞：object-token 在四路输入下保持近似答案、normal 对象类别错配、`terminate`/`task completed` 等无效模板，以及 control 答案与 normal 几乎不变。因此本阶段仅对现有预测做 CPU-only reward 修补与审计，没有训练或推理。

### v2.2 新增 Penalties

- `object_token_invariant_answer_penalty`：object-token 问题在至少三路输入下输出高度相似且非谨慎回答时触发。
- `normal_object_category_mismatch_penalty`：normal 下 gold 与答案出现明确交通对象类别冲突时触发，并允许 `vehicle` 与具体车辆类别的兼容关系。
- `invalid_generic_answer_penalty`：显式惩罚异常模板输出，如 `terminate`、`no further actions needed` 或 `placeholder`。
- `control_same_as_normal_penalty`：control 输入变化后答案仍与 normal 高相似且非 caution/refusal 时触发。

### 离线结论

- 人工标签分布：`{json.dumps(ready['human_label_counts'], ensure_ascii=False)}`。
- LingoQA ranking：`{' > '.join(ready['lingoqa_ranking'])}`。
- DriveLM ranking：`{' > '.join(ready['drivelm_ranking'])}`。
- Overall pairwise accuracy：{ready['overall_pairwise_accuracy']:.4f}。
- Object-invariant pair accuracy：{p.get('object_invariant_bad_vs_object_grounded_good', {}).get('accuracy', 0):.4f}。
- Control-same-as-normal pair accuracy：{p.get('control_same_as_normal_bad_vs_control_caution_good', {}).get('accuracy', 0):.4f}。
- 人工漏罚样本改善：{ready['manual_under_penalized_improved']}/{ready['manual_under_penalized_total']}。

### GPU 边界

**{recommendation}** 即使离线 gate 通过，也只允许在明确授权后进行一次从 frozen-r3 reference 出发的 50-step GRPO-lite smoke；禁止长训，禁止将 held-out 或 DriveLM OOD evaluation records 用作训练数据。
"""
    for path in (Path("README_CN.md"), Path("outputs/final_report/README_results.md"), Path("outputs/final_report/depth_extension_roadmap.md")):
        replace_block(path, section)
    blueprint = section + """
### Minimal GRPO-lite Configuration If Authorized

- `init_adapter`: `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/`
- `reference_adapter`: `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/` (frozen)
- `group_size=4`, `temperature=0.7`, `max_new_tokens=64`, `max_steps=50`
- No held-out or DriveLM OOD records in training; rerun strict visual-control evaluation after the smoke.
"""
    replace_block(Path("outputs/final_report/grpo_lite_training_blueprint.md"), blueprint)
    print(json.dumps({"readiness": ready["reward_v2_2_ready_for_grpo_lite_smoke"], "docs_updated": ["README_CN.md", "outputs/final_report/README_results.md", "outputs/final_report/depth_extension_roadmap.md", "outputs/final_report/grpo_lite_training_blueprint.md"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
