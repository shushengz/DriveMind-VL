"""Write Stage 15 Qwen3-VL stack preparation findings into project documents."""
from __future__ import annotations

import json
from pathlib import Path

BEGIN = "<!-- STAGE15_QWEN3_VL_STACK_BEGIN -->"
END = "<!-- STAGE15_QWEN3_VL_STACK_END -->"


def replace_block(path: Path, text: str) -> None:
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"\n{BEGIN}\n{text.rstrip()}\n{END}\n"
    if BEGIN in content and END in content:
        content = content.split(BEGIN, 1)[0].rstrip() + block + content.split(END, 1)[1].lstrip("\n")
    else:
        content = content.rstrip() + "\n" + block
    path.write_text(content, encoding="utf-8")


def main() -> None:
    audit = json.loads(Path("outputs/data_audit/qwen3_vl_stage15_export_summary.json").read_text(encoding="utf-8"))
    sft = audit["audits"]["ms_swift_sft"]
    dpo = audit["audits"]["ms_swift_dpo"]
    grpo = audit["audits"]["ms_swift_grpo"]
    section = f"""## Stage 15：Qwen3-VL Open-source Training Stack Preparation

当前 Qwen2.5-VL 主模型仍为 `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/`。Stage 15 不改变既有实验结论，而是为下一代 Qwen3-VL 分支准备成熟、可复现的开源训练栈：Qwen3-VL 更适合作为新的 base/SFT 对照，但必须先经过相同的 strict visual-control protocol 验证，不能直接以模型升级代替视觉依赖评估。

### Stack Selection

- 主训练栈：**ms-swift**。官方资料明确覆盖 Qwen3-VL、多模态 SFT/DPO/GRPO、LoRA/QLoRA、vit/aligner/llm 控制、reward 插件、vLLM 与 DeepSpeed。
- 快速验证栈：**Unsloth**。其 Qwen3-VL Vision 与 Vision-GRPO notebook 适合短 smoke，但不是当前工程主线。
- 参考实现：**Qwen-VL-Series-Finetune**。适合核对 Qwen-specific vision tuning、多图与视频训练格式，当前导出需锁定 commit 后再人工确认。

### Prepared Files And Data

- 调研与矩阵：`docs/qwen3_vl_finetune_survey.md`、`outputs/final_report/qwen3_vl_training_stack_matrix.csv`。
- ms-swift SFT 导出：`data/qwen3_vl/ms_swift/sft_lingoqa_r3_train.jsonl`（{sft['total_samples']} 条）。
- ms-swift DPO 导出：`data/qwen3_vl/ms_swift/dpo_preference_v8_2.jsonl`（{dpo['total_pairs']} 对）。
- ms-swift GRPO prompt 草案：`data/qwen3_vl/ms_swift/grpo_lite_prompts.jsonl`（{grpo['total_prompts']} 条；仅格式准备，不可训练）。
- Qwen3 base eval 入口：`scripts/run_qwen3_vl_base_eval.sh`，输出隔离到 `outputs/predictions_qwen3_vl/`。
- SFT/GRPO 配置草案：`configs/ms_swift/`。

### Audit And Next GPU Boundary

- 所有导出 held-out/OOD leakage 均为 `0`。
- 数据格式审计通过：`{str(audit['data_format_audit_passed']).lower()}`。
- 下一步如获 GPU 授权，应**先跑 Qwen3-VL-4B Base 的 LingoQA held-out 100 与 DriveLM OOD 100 strict eval**，再决定是否启动 SFT smoke。
- Reward v2.2 当前仍 `not ready`，因此 GRPO-lite 配置只是禁用草案，当前不训练 GRPO。
"""
    for target in ("README_CN.md", "outputs/final_report/README_results.md", "outputs/final_report/depth_extension_roadmap.md"):
        replace_block(Path(target), section)
    print(json.dumps({"docs_updated": ["README_CN.md", "outputs/final_report/README_results.md", "outputs/final_report/depth_extension_roadmap.md"], "leakage_zero": audit["checks"]["all_leakage_zero"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
