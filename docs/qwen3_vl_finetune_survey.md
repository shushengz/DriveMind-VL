# Qwen3-VL Open-source Fine-tuning Stack Survey

## Scope

本调研服务于 DriveMind-VL 的下一阶段迁移准备：在不训练、不推理、不下载模型的前提下，判断 Qwen3-VL 的 base evaluation、SFT smoke 与未来 GRPO-lite reward-smoke 应采用哪套开源栈。当前 Qwen2.5-VL 主结论保持不变；Qwen3-VL 是后续实验分支，而不是已验证替代模型。

调研时间：2026-05-26。判断优先依据官方仓库或官方文档；没有明确证据的能力统一写为 `unknown` 或 `partial`。

## Source Ledger

| Source | Evidence used | Limitation |
| --- | --- | --- |
| [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | 官方模型、Transformers/AutoProcessor 多图输入与 vLLM 部署方式 | README 重点是推理/部署，不等价于完整后训练栈 |
| [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | README 明确列出 Qwen3-VL、多模态训练、LoRA/QLoRA、DPO、GRPO、插件奖励、vLLM 与 DeepSpeed | 仍需在固定版本上 dry-run 校验 Qwen3-VL 参数名 |
| [ms-swift custom dataset](https://github.com/modelscope/ms-swift/blob/main/docs/source/Customization/Custom-dataset.md) | 标准格式键为 `messages`、`images`、`rejected_response`；PPO/GRPO 输入格式说明 | custom reward 与本项目四路 rollout 仍需后续 glue code |
| [2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) | README 明确 Qwen3-VL、SFT/DPO/GRPO、video DPO/GRPO、multi-image、vision LoRA/冻结提示 | 非项目主线；导出格式需锁定 commit 后人工复核 |
| [Unsloth notebooks](https://github.com/unslothai/notebooks) | 提供 Qwen3-VL Vision 与 Qwen3-VL Vision-GRPO notebook | Notebook-first，完整 DPO/多卡/生产集成信息不如 ms-swift 清晰 |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | README 列出 Qwen3-VL、multimodal SFT、DPO/PPO、LoRA/QLoRA、vLLM、视频任务 | Qwen3-VL 专属 GRPO/custom reward 流程未在本次证据中确认 |
| [Hugging Face TRL](https://huggingface.co/docs/trl/) | 通用 DPO/GRPO Trainer 与 reward function 组件 | Qwen3-VL 多模态端到端训练兼容性需另行验证 |

## Capability Survey

| Framework | Qwen3-VL | SFT | DPO | GRPO | Custom reward | LoRA / QLoRA | Vision tuning | Multi-image / video | vLLM rollout | DeepSpeed | DriveMind-VL fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ms-swift | yes | yes | yes | yes | yes | yes / yes | yes, vit/aligner/llm controllable | yes / yes | yes | yes | **主线最佳**：训练、RLHF、奖励插件和多模态均在同一框架 |
| Qwen-VL-Series-Finetune | yes | yes | yes | yes | partial | yes / partial | yes; README warns against quantized vision tuning | yes / yes | unknown | partial | **参考实现**：适合核对 Qwen-specific vision tuning |
| Unsloth | yes | yes | unknown | yes | partial | yes / yes | partial | partial / unknown | partial | unknown | **快速 smoke 备选**：notebook 验证快，工程闭环较弱 |
| LLaMA-Factory | yes | yes | yes | unknown | unknown | yes / yes | partial | yes / yes | inference yes; rollout unknown | yes | **SFT/DPO 备选**，GRPO-lite 尚不优先 |
| TRL direct | partial | partial | partial | partial | yes | PEFT-dependent | unknown | partial / unknown | partial | Accelerate-dependent | **算法参考**，需要自行补 VLM 数据与 rollout glue |
| Qwen3-VL official repo | yes | unknown | unknown | unknown | unknown | unknown | unknown | yes / yes | deployment yes | unknown | **模型/推理参考**，不是现成训练栈 |

## Data Format Decision

主线采用 ms-swift 官方标准数据字段：

- SFT：`messages` 中包含 user 与 answer-only assistant，媒体路径通过顶层 `images` 存放；user 文本使用 `<image>` 占位符。
- DPO：chosen 写入 assistant message；rejected 写入 `rejected_response`；不导出 reason。
- GRPO-lite draft：仅提供 user `messages`、`images` 与 reward 参考用 `solution`；当前 Reward v2.2 未 ready，因此仅作为不可执行草案。

Qwen-VL-Series-Finetune 导出仅作为 best-effort 参考格式，并在 audit 中强制标记 `needs_manual_format_check=true`。

## Recommendation

| Purpose | Recommendation | Reason |
| --- | --- | --- |
| `best_for_sft` | ms-swift | Qwen3-VL 与多模态 LoRA/QLoRA/vision freezing 有官方一体化支持 |
| `best_for_dpo` | ms-swift | 已支持多模态 DPO，且标准 preference 数据字段清楚 |
| `best_for_grpo` | ms-swift | 官方明确支持 GRPO、reward 插件与同步/异步 vLLM |
| `best_for_fast_smoke` | Unsloth | Qwen3-VL Vision/GRPO notebook 适合低成本验证 |
| `best_for_project_integration` | ms-swift | 最容易承接 DriveMind-VL 的 strict eval、数据审计和 reward harness |
| `reference_implementation` | Qwen-VL-Series-Finetune | Qwen 系专门栈，适合对照 vision tuning 与多图格式 |

## GPU Memory And Risk Notes

本阶段没有加载模型，也没有执行显存测试，因此不写虚假的显存数字。Qwen3-VL-4B LoRA + frozen vision tower 应优先作为 GPU smoke 入口；8B 或解冻 vision tower 必须在后续 GPU readiness 中实际测量。对于 vision tower 训练，不建议量化 vision tower；该限制也与 Qwen-VL-Series-Finetune README 的警告一致。

## Stage 15 Decision

1. 以 ms-swift 作为主迁移栈，先准备 Qwen3-VL-4B base strict visual-control eval。
2. Base eval 比 SFT 更优先，因为需要先建立 Qwen3-VL 与当前 Qwen2.5-VL/r3 的无训练基准。
3. Base eval 完整后，再授权一次 Qwen3-VL-4B SFT LoRA smoke，默认 `freeze_vit=true`。
4. Reward v2.2 当前 `not ready`，GRPO-lite 配置只能存档，不能执行。
