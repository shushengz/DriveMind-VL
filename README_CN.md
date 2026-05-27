# DriveMind-VL 中文说明

## CPU-only 阶段说明

当前阶段不会训练模型，不会启动 Qwen2.5-VL 推理，不会下载大模型，也不会给出最终性能结论。本阶段只完成 CPU 可完成的工程改造，为后续 GPU 训练和评测准备干净、可复现的数据与指标基础。

先做 CPU-only 工作，是为了在昂贵的 GPU 实验前固定 visual-control 协议、LingoQA / DriveLM 输入格式、SFT / Preference 数据构造、离线 reward 口径和 case gallery。这样后续实验能更清楚地区分真实视觉依赖、语言先验、错误图像干扰和过度拒答。

### strict visual-control protocol

新增 `src/data/visual_control_formatter.py`，统一构造四路输入：

- `normal`：真实图像 + 问题。
- `text_only`：只给问题，不给图像。
- `wrong_image`：错误图像 + 原问题。
- `blank_image`：空白图像 + 原问题。

`strict_visual` 模式不注入 perception JSON，不在 control 设置中泄漏原始视觉标注，只保留 question、gold、image_paths、image_labels、prompt 和轻量 metadata。`perception_augmented` 模式可以保留 perception，但样本会显式标记 `mode=perception_augmented`，不能和 strict 样本混用。

LingoQA 多帧输入使用 `[Frame 0]`、`[Frame 1]` 等标签。DriveLM 多相机输入使用 `[CAM_FRONT]`、`[CAM_FRONT_LEFT]`、`[CAM_FRONT_RIGHT]`、`[CAM_BACK]`、`[CAM_BACK_LEFT]`、`[CAM_BACK_RIGHT]` 标签，并根据问题中的 camera 名称或 object token 优先选择相机；没有相机信息时默认 `CAM_FRONT`。

### SFT-v3 数据构造逻辑

新增 `src/data/build_sft_v3.py`，构造 normal_visual_qa、blank_refusal、wrong_image_caution、text_only_caution 和 spatial_hard_negative。assistant 输出统一为 `{"answer": "...", "reason": "..."}`，不包含 references、category、subcategory、debug、内部标签或数据集路径。

### Preference-v8 数据构造逻辑

新增 `src/data/build_preference_v8.py`，构造 normal_gold > normal_bad_prediction、normal_gold > normal_refusal、control_abstention > control_hallucination、wrong_image_detected > wrong_image_confident_wrong_answer 和 spatial_gold > spatial_wrong_relation 等 pairs，并记录权重与统计。

### Offline Reward Harness

新增 `src/rl/reward_driving_vqa.py` 和 `src/rl/debug_reward_harness.py`。当前只读取已有 raw predictions 或 case scores 做离线 reward 计算，不调用模型，不训练 GRPO。reward 包含 answer F1、visual gap、control calibration、JSON format、normal over-refusal 和 length penalty。

### Case Gallery

新增 `src/eval/build_case_gallery.py`，从 case_scores、raw predictions 和 visual-control 样本生成 `outputs/final_report/case_gallery.csv` 与 `case_gallery.html`。HTML 是简单表格，图片存在时显示图片，不存在时显示路径文本。

### CPU-only 一键命令

默认 dry-run：

```bash
bash scripts/run_cpu_stage.sh
```

完整生成文件需要显式传入：

```bash
bash scripts/run_cpu_stage.sh --run
```

已有 raw predictions 或 case scores 时：

```bash
RAW_PREDICTIONS=outputs/raw_predictions/visual_control.jsonl \
CASE_SCORES=outputs/cases/lingoqa_model_strict_visual_case_scores.csv \
bash scripts/run_cpu_stage.sh
```

### 后续 GPU 阶段计划

后续 GPU 阶段再进行小规模 strict visual-control eval、SFT-v3、DPO-v8、VC-GRPO-lite 和 ablation study。当前阶段只构建基础设施，任何最终性能结论都必须等后续真实推理和训练完成后再报告。


## GPU Stage 2：Strict Visual-Control 小规模验证

### 本阶段目的

Stage 2 的目标是用 Qwen2.5-VL-3B-Instruct 做小规模、可复现的 strict visual-control 验证：先确认输入构造干净，再比较 Base、旧 SFT-v2、旧 DPO-v7 和 SFT-v3 smoke checkpoint。这个阶段不做大规模训练，也不直接进入 GRPO。

### 为什么不直接上 GRPO

GRPO 依赖稳定的 reward 和稳定的 init checkpoint。如果 strict eval 还不能区分 normal 与 text-only / wrong-image / blank-image，或者 SFT-v3 已经出现过度拒答，那么直接上 GRPO 很容易优化出 reward hacking：control 全拒答、normal 也拒答，或者靠长答案刷 F1。

### Base strict eval

默认 dry-run，不加载模型：

```bash
bash scripts/run_gpu_stage2_smoke.sh --dataset both --eval_samples 50 --skip_train
```

真实 GPU smoke 需要显式传入 `--run`：

```bash
bash scripts/run_gpu_stage2_smoke.sh --run --dataset both --eval_samples 50 --skip_train --bf16
```

也可以只跑 eval 脚本：

```bash
python src/eval/run_visual_control_eval.py \
  --dataset lingoqa \
  --eval_samples 50 \
  --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --model_name base_qwen25vl_3b \
  --bf16
```

所有推理都会保存 raw predictions，并自动生成 summary 和 case scores。

### SFT-v3 smoke checkpoint

默认 dry-run：

```bash
python src/train/train_qwen25vl_sft_v3_lora.py --dry_run \
  --train_file data/train/sft_v3/lingoqa_sft_v3.jsonl
```

真实 100 step smoke：

```bash
python src/train/train_qwen25vl_sft_v3_lora.py \
  --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --train_file data/train/sft_v3/lingoqa_sft_v3.jsonl \
  --output_dir checkpoints/qwen25vl_lora_sft_v3_lingo_smoke \
  --log_dir outputs/train_logs/sft_v3_lingo_smoke \
  --train_samples 1000 \
  --max_steps 100 \
  --bf16
```

训练会保存 `train_config.yaml`、`dataset_stats.json`、`loss_log.jsonl`、`command.txt` 和 LoRA adapter checkpoint。

### 比较 Base / SFT-v2 / DPO-v7 / SFT-v3

Stage 2 总控会查找旧 checkpoint，存在则回归评测，不存在则写入 `outputs/gpu_stage/missing_checkpoints.json` 并继续。汇总主表输出到：

```text
outputs/final_report/stage2_strict_eval_main_results.csv
```

诊断输出到：

```text
outputs/final_report/stage2_diagnosis.md
outputs/final_report/stage2_diagnosis.json
```

### 如何解读 Setting Gap 和 Case Gap

`Setting Gap = Normal F1 - max(Text-only F1, Wrong-image F1, Blank-image F1)`。它反映整体设置层面的视觉依赖。

`Case Gap` 是逐 case 计算 normal 与最强 control 的差值后求平均。它比 Setting Gap 更能发现单样本层面的 wrong-image confound 和 text prior bias。

理想情况是 Normal F1 上升，同时 Case Gap 不下降，control hallucination rate 下降，normal refusal rate 不明显升高。

### 如何判断过度拒答

如果 SFT-v3 的 `normal_refusal_rate` 明显高于 Base，尤其同时 Normal F1 下降，就说明模型可能把 control calibration 学成了 normal 下也拒答。此时不应该进入 GRPO。

### 如何判断语言先验增强

如果 SFT-v3 的 `text_only_f1`、`wrong_image_f1`、`blank_image_f1` 同时明显上升，而 `case_gap` 没有提升，说明模型可能只是更会猜数据集答案，而不是更依赖视觉证据。

### 进入 DPO-v8 的条件

- SFT-v3 normal_f1 不低于 Base；
- SFT-v3 normal_refusal_rate 不明显升高；
- SFT-v3 case_gap 至少不恶化；
- raw predictions 和 case_scores 已保存完整。

### 进入 GRPO-lite 的条件

- reward harness 能正确区分 normal / control；
- SFT-v3 没有严重过度拒答；
- control hallucination 问题仍然存在；
- DPO-v8 或 SFT-v3 已经提供稳定 init checkpoint。

## Stage 3：SFT-v3-r2 数据比例修复

Stage 2 发现 SFT-v3 smoke 的 `normal_visual_qa` 比例只有约 23%，control/refusal 样本过多，导致模型更谨慎但 Normal F1 下降。Stage 3 的目标是验证“更强 normal anchor + 适量 control calibration”能否同时保住 normal 图像问答能力并改善 strict visual-control。

### 数据比例

SFT-v3-r2 使用新的构造和审计脚本：

```bash
python src/data/build_sft_v3_r2.py --output_dir data/train/sft_v3_r2
python src/data/audit_sft_v3_r2.py --input data/train/sft_v3_r2/lingoqa_sft_v3_r2.jsonl
```

目标比例：

- `normal_visual_qa`：55% 到 60%
- `blank_refusal`：8% 到 10%
- `wrong_image_caution`：8% 到 10%
- `text_only_caution`：6% 到 8%
- `spatial_hard_negative`：10% 到 15%

normal anchor 使用 gold answer，不允许写成保守拒答；control/refusal 样本只用于校准 text-only、blank-image 和 wrong-image 条件，不能压过 normal QA 能力。

### 一键运行

默认 dry-run，不训练、不推理：

```bash
bash scripts/run_stage3_sft_v3_r2.sh --all --dataset lingoqa --max_eval_samples 100 --max_steps 200 --bf16 --qlora
```

真实运行必须显式加 `--run`：

```bash
bash scripts/run_stage3_sft_v3_r2.sh --run --all --dataset lingoqa --max_train_samples 1500 --max_eval_samples 100 --max_steps 200 --bf16 --qlora --rerun_baselines
```

也可以分阶段运行：

```bash
bash scripts/run_stage3_sft_v3_r2.sh --run --build_data --audit_only
bash scripts/run_stage3_sft_v3_r2.sh --run --train --max_train_samples 1500 --max_steps 200 --bf16 --qlora
bash scripts/run_stage3_sft_v3_r2.sh --run --eval --diagnose --max_eval_samples 100 --bf16 --rerun_baselines
```

### 结果解读

主要结果表：

```text
outputs/final_report/stage3_sft_v3_r2_main_results.csv
```

诊断报告：

```text
outputs/final_report/stage3_sft_v3_r2_diagnosis.md
```

Case gallery：

```text
outputs/final_report/stage3_sft_v3_r2_case_gallery.html
```

进入 DPO-v8 的条件：

- SFT-v3-r2 `normal_f1` 不低于 Base；
- `case_gap` 不恶化；
- `normal_refusal_rate` 不明显上升；
- raw predictions 与 case scores 保存完整；
- 不出现明显 language-prior strengthening。

进入 GRPO-lite 的条件：

- 先完成 DPO-v8 或得到稳定 SFT init；
- reward harness 能稳定区分 normal/control；
- 仍存在明显 control hallucination；
- 没有严重过度拒答。

当前 Stage 3 的定位不是最终模型，而是验证数据比例修复是否能恢复 SFT-v3 smoke 中下降的 normal ability。若 r2 `normal_f1` 仍低于 Base，则不进入 DPO-v8/GRPO-lite。

## Stage 4 CPU-only：SFT-v3-r3 数据与评测准备

Stage 3 的 SFT-v3-r2 暴露出两个评测和训练层面的风险：一是 `{"answer": "...", "reason": "..."}` 会让 raw full-prediction F1 被 JSON 字段和 reason 稀释；二是 reason 中的 `image`、`visual`、`visible`、`scene` 等词会污染 control hallucination 检测。Stage 4 因此先做 CPU-only 准备，不训练、不推理、不加载 Qwen2.5-VL。

SFT-v3-r3 改为 answer-only 输出，只保留：

```json
{"answer": "..."}
```

r3 计划从 SFT-v2 adapter 继续做轻量校准，而不是从 Stage 3 的 r2 继续训练。原因是 SFT-v2 的 normal QA 能力更稳，r3 只负责小比例 control calibration，避免再次把模型教得过度谨慎。

### r3 数据比例

- `normal_replay`：约 65% 到 70%
- `spatial_normal_qa`：约 10%
- `blank_calibration`：约 4% 到 5%
- `wrong_image_calibration`：约 4% 到 5%
- `text_only_calibration`：约 3% 到 5%
- 剩余用 normal replay 填充，不额外增加 control 样本

control calibration 降到 10% 到 15%，用于降低 blank/text-only/wrong-image 下的幻觉，但不压过 normal QA anchor。control answer 禁止出现 `image`、`visual`、`visible`、`scene`、`frame`、`picture`、`图像`、`画面`、`图中`、`视觉` 等词，避免污染 hallucination 检测。

### Answer-only Eval

Stage 4 新增 answer-only parser 和 rescore 工具。主结论以后优先看 `score_mode=answer_only`，raw full 仅作为格式和输出行为参考。

```bash
python src/eval/rescore_answer_only.py --dataset lingoqa --allow_missing
python src/eval/build_stage4_main_results.py --dataset lingoqa
```

### CPU-only 能做什么

- 构造 `data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl`
- 审计 r3 数据比例、answer-only 格式、forbidden terms 和 metadata 泄漏
- 对已有 Base / SFT-v2 / DPO-v7 / SFT-v3-r2 predictions 做 answer-only 重评估
- 生成 Stage 4 主表、诊断占位和 case gallery 占位
- 跑单元测试

### CPU-only 不能做什么

- 不训练 SFT/DPO/GRPO
- 不启动模型推理
- 不加载 Qwen2.5-VL
- 不使用 GPU
- 不得据此宣称 r3 模型性能结论

### 一键命令

```bash
bash scripts/run_stage4_sft_v3_r3_cpu.sh --all --run
```

默认是 dry-run；只有显式 `--run` 才写正式 `data/train/sft_v3_r3/` 和 `outputs/final_report/`。

### 后续 GPU 阶段

1. 从 SFT-v2 adapter 继续训练 r3；
2. 跑 LingoQA 100/200 strict visual-control eval；
3. 生成 `outputs/final_report/stage4_sft_v3_r3_main_results.csv`；
4. 若 r3 answer-only normal_f1 不低于 Base、接近 SFT-v2、case_gap 不恶化且 hallucination 下降，再考虑 DPO-v8；
5. 只有 DPO-v8 后仍有明显 control hallucination，且 reward harness 通过，才考虑 GRPO-lite。

本阶段不会产生 r3 模型性能结论，只为后续 GPU 训练与评估准备干净数据和 answer-only 评测基础。


## Stage 4 GPU：SFT-v3-r3 Lightweight Calibration

本阶段目标是在 SFT-v2 adapter 的 normal QA 能力基础上，继续做一个很小规模的 SFT-v3-r3 answer-only calibration smoke checkpoint。它不是最终训练，也不会自动进入 DPO-v8 或 GRPO-lite。

为什么从 SFT-v2 adapter 继续训练：Stage 3 说明从 Base 重新做控制校准容易损伤 normal QA。r3 的核心假设是保留 SFT-v2 已学到的 LingoQA 能力，只用少量 answer-only r3 样本校准 blank / text-only / wrong-image 下的行为。

为什么只训练 100 steps：当前只验证数据比例和输出格式是否能保住 normal ability，同时降低 control hallucination。若 100 steps 已经导致 normal F1 下降，应先诊断数据和学习率，而不是盲目加长训练。

answer-only 是主指标：r2 暴露出 raw JSON 和 reason 会稀释 F1，并污染 hallucination 检测。因此 Stage 4 主结论只看 answer-only 指标；raw_full 仅作为格式和冗余输出风险参考。

GPU smoke 命令：

```bash
bash scripts/run_stage4_sft_v3_r3_gpu.sh \
  --all \
  --run \
  --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --init_adapter checkpoints/qwen25vl_lora_sft_v2/ \
  --max_steps 100 \
  --eval_samples 100 \
  --qlora \
  --bf16
```

输出重点文件：

- `outputs/gpu_stage/stage4_r3_gpu_ready.json`
- `outputs/train_logs/sft_v3_r3_lingo_smoke/train_summary.json`
- `outputs/predictions/lingoqa/sft_v3_r3_lingo_smoke/strict_visual/`
- `outputs/final_report/stage4_sft_v3_r3_main_results.csv`
- `outputs/final_report/stage4_sft_v3_r3_diagnosis.md`
- `outputs/final_report/stage4_sft_v3_r3_case_gallery.html`

进入 DPO-v8 的条件：r3 answer-only normal F1 不低于 Base，且与 SFT-v2 的差距不超过 0.03；normal refusal rate 不明显高于 Base；control hallucination rate 低于 SFT-v2 或 r2；case gap 不低于 Base；raw predictions 和 case scores 保存完整。

当前不直接进入 GRPO-lite。只有完成 DPO-v8 后，如果仍然存在明显 control hallucination，并且 reward harness 能稳定区分 normal / control，才考虑 GRPO-lite。

常见失败情况：`init_adapter` 不存在时停止训练；OOM 时优先降低 `max_pixels`、batch size 或 max sequence length；normal F1 下降时不继续加步数；hallucination 没降或 case gap 恶化时不进入 DPO-v8。

## Stage 5 CPU-only：DPO-v8 Preference Data Preparation

Stage 4.5 的 held-out strict eval 显示，SFT-v3-r3 在无训练重叠的样本上获得了最强的 normal answer-only F1，但 control hallucination rate 仍然偏高。Stage 4 的原始结果存在 100% train/eval overlap，因此不得直接据此进入新的训练决策。本阶段只准备用于抑制 control hallucination 的 DPO-v8 preference 数据，不训练、不推理、不使用 GPU，也不进入 GRPO-lite。

### 为什么准备 DPO-v8，而不是 GRPO

r3 已经提供了较好的 normal QA 初始化点，当前问题集中在 text-only、wrong-image 和 blank-image 条件下仍会产生不可靠断言。DPO-v8 可以用显式 chosen/rejected 对先约束这一行为，并用足量 normal anchor 防止 normal F1 下滑。GRPO-lite 只有在 DPO-v8 smoke 后仍有明显 hallucination、reward harness 已验证可靠且 normal 能力稳定时才值得讨论。

### Held-out 隔离

`outputs/final_report/stage4_5_heldout_ids_100.json` 只作为排除名单使用。DPO-v8 pairs 不得来自 `outputs/predictions_heldout/`、held-out case gallery 或 held-out IDs 对应的 visual-control 样本。`src/data/exclude_heldout_ids.py` 会在构造 pairs 前过滤候选池，并输出 `outputs/data_audit/dpo_v8_heldout_exclusion.json`；只要过滤后仍有 leakage，就必须停止。

### Pair 设计

- `normal_anchor_gold_vs_bad`：约 42%，gold 优于错误或过度保守答案，保护 normal QA。
- `normal_gold_vs_refusal`：约 13%，防止 normal 输入下过度拒答。
- `control_abstain_vs_hallucination`：约 22%，让缺失或无效证据时优先给出中性校准答案。
- `wrong_image_caution_vs_confident_answer`：约 15%，重点压制 wrong-image confound。
- `spatial_gold_vs_spatial_wrong`：约 8%，保护左右、前后、车道和交通参与者相关能力。

所有 chosen/rejected 输出均为 answer-only JSON：

```json
{"answer": "..."}
```

control chosen 禁止包含 `image`、`visual`、`visible`、`scene`、`frame`、`picture`、`图像`、`画面`、`图中`、`视觉` 等词，避免再次污染 hallucination 指标。若没有合规的 r3 train-pool predictions，本阶段使用 rule-based pairs，并在 stats 中标记 `uses_model_predictions: false`。

### 数据审计

`src/data/audit_preference_v8.py` 检查 pair 类型比例、answer-only 格式、held-out leakage、forbidden terms、normal refusal、metadata 泄漏与重复率。进入 DPO-v8 GPU smoke 至少需要：

- `train_ready = true` 且 `heldout_leakage_count = 0`；
- normal pair ratio 不低于 0.50，control pair ratio 不高于 0.45；
- control chosen forbidden visual terms rate 低于 2%；
- normal chosen refusal rate 低于 2%；
- r3 checkpoint 可用；
- DPO config 的 `reference_adapter` 为冻结的 r3 adapter。

### CPU-only 命令

```bash
bash scripts/run_stage5_dpo_v8_cpu.sh --all --run --max_pairs 500 --seed 42
```

脚本默认仅 dry-run；显式 `--run` 也只会生成数据、审计报告并验证 `configs/dpo_v8_lingo_smoke.yaml`，绝不会启动训练或推理。

### 风险与后续

DPO-v8 GPU smoke 需要重点观察 normal F1 是否下降、normal refusal 是否上升、control hallucination 是否真正下降，以及 policy/reference 是否都正确从 r3 初始化。当前阶段不会得出 DPO-v8 的性能结论，也不建议进入 GRPO-lite。

## Stage 6：DPO-v8 GPU Smoke

Stage 4.5 的无泄漏 held-out 结果表明，SFT-v3-r3 的 normal QA 最强，但在 text-only、wrong-image 与 blank-image 条件下仍有较高 hallucination。Stage 6 因此只运行一次小规模 DPO-v8 smoke，用来检验 Stage 5 的 answer-only Preference-v8 能否降低 control hallucination，同时尽量保住 r3 的 normal 能力。

### 为什么不是 GRPO-lite

当前已有明确的偏好约束问题，可以先用 DPO 做更可控的校准。GRPO-lite 需要稳定的初始化 checkpoint 与经过验证的 reward harness；在 DPO-v8 还没有完成 held-out 复核前直接进入 GRPO 会把风险放大。

### Policy 与 Reference

policy 和 reference 都从 `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/` 初始化。policy 可训练，reference 必须冻结，并且 `reference_free: false`。这样 DPO 学到的是相对于 r3 的轻量偏好修正，而不是用 base reference 改写已经获得的 normal QA 能力。

### Smoke 规模与 Held-out 评测

训练仅运行 50 steps。若 loss 出现 NaN、显存不足、reward margin 异常，或 held-out normal F1 明显下降，应立即停止而不是盲目增加训练步数。所有结论必须使用 `outputs/final_report/stage4_5_heldout_ids_100.json` 对应的无泄漏 held-out 100 cases，不允许回到 Stage 4 的重叠评测集合。

### 运行方式

```bash
bash scripts/run_stage6_dpo_v8_gpu_smoke.sh \
  --all \
  --run \
  --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --max_steps 50 \
  --eval_ids outputs/final_report/stage4_5_heldout_ids_100.json \
  --bf16 \
  --qlora
```

脚本默认 dry-run；只有显式 `--run` 才启动 smoke 训练与 held-out 推理。总控流程不会启动 GRPO。

### 结果解读

主表位于 `outputs/final_report/stage6_dpo_v8_heldout_main_results.csv`，主结论只查看 `score_mode=answer_only`。DPO-v8 的成功条件为：

- `normal_f1 >= r3 normal_f1 - 0.03`；
- `control_hallucination_rate < r3 control_hallucination_rate`；
- `case_gap >= r3 case_gap`；
- `normal_refusal_rate <= 0.02`；
- answer length 未异常变长。

常见失败模式包括 normal F1 下跌、过度拒答、hallucination 没有下降、case gap 恶化，以及 reference 配置错误。若 hallucination 没有改善，应回退 r3 并构造来自真实 train-pool prediction 的 model-mined preference_v8.1 hard negatives；若 smoke 达标，也应先扩大验证与人工检查案例，再决定是否继续 DPO。当前不直接进入 GRPO-lite。

## Stage 7：Model-mined Preference-v8.1

Stage 6 说明 rule-based Preference-v8 的作用有限：DPO-v8 保住并略微提高了 r3 的 normal F1，wrong-image hallucination 有小幅下降，但 text-only 与 blank-image 没有得到充分修正，整体 case gap 反而恶化。因此本阶段不继续堆 DPO steps，而是先把 preference 数据从规则构造升级为来自 r3 真实错误的 hard preferences。

### Mining 与 Leakage Guard

Stage 7 先从 `data/processed/visual_control/lingoqa_strict_*.jsonl` 构建 non-heldout mining pool。`outputs/final_report/stage4_5_heldout_ids_100.json` 只用作排除名单，held-out predictions、held-out case gallery 和 held-out 样本均不得成为 pair 来源。pool、hard-negative 和 Preference-v8.1 审计都会检查 held-out leakage；只要发现重叠，后续 DPO-v8.1 必须停止。

### Hard Negative 类型

- `text_prior_bias`：text-only 条件下仍给出非拒答且具有高 overlap 或自信答案。
- `blank_prior_answer`：blank-image 条件下仍直接作答，针对 Stage 6 中 blank 控制组的问题。
- `wrong_image_confound`：wrong-image 输出与 normal 竞争或高度相似。
- `control_over_gold_overlap`：任一 control 得分达到或超过 normal，直接针对 case gap。
- `normal_wrong_anchor`：r3 normal 下真实错误，用于保护正常问答能力。
- `spatial_relation_error`：空间关系问题下的真实错误，用于保护 grounding。

所有判断使用 answer-only 解析后的 r3 raw predictions；Preference-v8.1 的 rejected 必须来自 r3 的实际回答，而不是规则生成答案。

### Preference-v8.1 设计

v8.1 更强调 control hard negatives，同时保留 normal anchor：control pairs 目标约 60%，normal anchors 至少 25%，spatial pairs 约 5% 到 10%。chosen/rejected 均使用 `{"answer": "..."}`，不输出 `reason`，control chosen 继续禁止会污染 visual hallucination 检测的视觉词汇。

DPO-v8.1 的配置草案仍以 `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/` 同时作为 policy init 和 frozen reference，而不是从 DPO-v8 初始化。原因是 DPO-v8 的 case gap 已经恶化，r3 是目前更稳的起点。

### 运行方式

CPU-only 准备 mining pool 与配置草案：

```bash
bash scripts/run_stage7_mine_preference_v8_1.sh \
  --build_pool \
  --make_config \
  --run \
  --max_mining_ids 300
```

后续在明确允许 GPU inference 后运行 r3 mining 与数据构造：

```bash
bash scripts/run_stage7_mine_preference_v8_1.sh \
  --all \
  --run \
  --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --max_mining_ids 300 \
  --bf16
```

这个流程只允许 r3 推理与离线数据构造，不会启动 DPO、SFT 或 GRPO 训练。进入 DPO-v8.1 GPU smoke 前，必须满足 Preference-v8.1 审计 `train_ready=true`、held-out leakage 为 0、rejected 来自 r3 actual predictions、normal/control 比例合格。当前仍不进入 GRPO-lite。

## Stage 8：DPO-v8.1 Model-mined GPU Smoke

Stage 8 使用 Stage 7B 构建并审计通过的 model-mined Preference-v8.1，检验来自 r3 真实失败回答的 preference 是否比 rule-based v8 更有效。`rejected` 来源于 r3 在 non-heldout mining pool 的实际输出，使训练目标直接覆盖 `text_prior_bias`、`blank_prior_answer` 与 `wrong_image_confound`。

Policy 从 `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/` 初始化，reference 也是冻结的 r3 adapter，且 `reference_free: false`。不从 DPO-v8 继续训练，因为 rule-based DPO-v8 的 held-out case gap 已经较 r3 恶化，r3 仍是更稳的校准起点。

本轮只训练 50 steps，并同时保存、评估 step-25 和 step-50。model-mined pairs 的 control 占比较高，过长训练可能在降低幻觉的同时损伤 normal QA；双 checkpoint 评测用于及早发现 over-calibration。评测继续严格使用 `outputs/final_report/stage4_5_heldout_ids_100.json`，不使用泄漏的 Stage 4 集合。

### 运行方式

```bash
bash scripts/run_stage8_dpo_v8_1_gpu_smoke.sh \
  --all \
  --run \
  --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --max_steps 50 \
  --eval_ids outputs/final_report/stage4_5_heldout_ids_100.json \
  --bf16 \
  --qlora
```

脚本默认 dry-run；显式 `--run` 后也只允许 DPO-v8.1 smoke 与 held-out inference，不会启动 GRPO。训练前会阻止未通过审计的数据、held-out leakage、错误的 reference 设置以及缺失的 r3 adapter。

### 成功与失败标准

主表为 `outputs/final_report/stage8_dpo_v8_1_heldout_main_results.csv`，主要查看 `score_mode=answer_only`。成功要求 best checkpoint 满足：

- `normal_f1 >= r3 normal_f1 - 0.03`；
- `control_hallucination_rate < r3 control_hallucination_rate`；
- `case_gap >= r3 case_gap`；
- `normal_refusal_rate <= 0.02`；
- answer length 未异常增长。

若 step-25 优于 step-50，则选择 step-25 并停止增加训练步数；若 normal F1 明显下降、hallucination 未下降、case gap 恶化或 normal refusal 上升，则回到 r3 与 preference 设计继续修正。即使 v8.1 smoke 成功，也应先扩大 held-out 复核，不直接进入 GRPO-lite。

### 输入来源：Stage 7B 执行边界

Stage 7B 中唯一使用 GPU 的步骤，是用 r3 adapter 在 non-heldout mining pool 上生成四路 strict visual-control raw predictions。hard negative mining、Preference-v8.1 构造、审计与 DPO 配置校验均为离线文件处理，不加载模型。

r3 mining prediction 的目的不是重新评测泛化能力，而是在训练池内找出模型真实产生的 `text_prior_bias`、`blank_prior_answer`、`wrong_image_confound` 与 `control_over_gold_overlap`。这些实际失败回答将作为 rejected；相比规则生成的错误答案，它们更贴近后续 DPO 需要修正的决策边界。因此 v8.1 必须标记 `uses_model_predictions=true`，并要求 rejected-from-actual-prediction 比例超过 90%。

在运行 prediction 前仍须检查 CUDA、base model 与 r3 adapter、本地 mining pool、held-out exclusion，以及评测脚本对 `--eval_ids`、`--prediction_root`、`--adapter_path` 和 strict mode 的支持。预测输出只写入 `outputs/predictions_mining/`，不覆盖 `outputs/predictions_heldout/`。

只有 v8.1 审计通过、held-out leakage 为 0、answer-only 格式合格、normal anchor 比例合格且 actual prediction 来源可验证时，才可以在后续阶段考虑 DPO-v8.1 GPU smoke。当前阶段仍然不训练 DPO，也不进入 GRPO-lite。

## Stage 8.5：DPO-v8.1 Error Attribution

Stage 8 的 model-mined DPO-v8.1 能轻微降低显式 control hallucination，却没有改善 case gap。两者不是同一个问题：hallucination 规则关注带视觉断言的内容，而 case gap 还会被 `"Yes"`、`"No"`、计数或动作这类没有视觉触发词、但和 gold 发生重合的 control direct answers 拉低。

因此 Stage 8.5 完全离线读取 held-out raw predictions，增加以下 control behavior metrics：

- `direct_answer_rate` 与 `short_prior_answer_rate`：定位短直接答案与语言先验；
- `caution_rate`：检查 DPO 是否减少了谨慎回答；
- `action_answer_rate` 与 `count_answer_rate`：定位驾驶动作和计数先验；
- `high_f1_rate_0_20` 与 `high_f1_rate_0_30`：直接追踪 control 与 gold 的词面重合。

同时生成 `stage8_5_case_gap_regression.csv` 定位 r3 到 step-25/step-50 的 gap 恶化样本，并通过 fix/break 清单区分“修掉 control hallucination”与“损伤 normal 或提高 control gold overlap”的两类效应。

### Preference-v8.2 设计原则

Preference-v8.2 不再只根据 hallucination 触发词构造 pair，而会加入 `control_direct_answer_vs_caution`、`blank_high_f1_vs_caution`、`text_only_gold_overlap_vs_caution` 与 `wrong_image_gold_overlap_vs_caution`。建议比例为 normal-related `40%`、control-related `55%`、spatial `5%`；相较 v8.1，提高 normal anchor 并降低 control 总比重，防止 case gap 继续恶化。v8.2 仍应从 r3 与 frozen r3 reference 出发，不从 DPO-v8.1 续训。

### 离线运行方式

```bash
bash scripts/run_stage8_5_dpo_error_attribution.sh --all --run
```

该命令只执行 CSV/Markdown/JSON 归因与 v8.2 配置草案生成，不加载模型、不运行推理、不训练，也不使用 GPU。当前仍不建议进入 GRPO-lite；下一次需要 GPU 的环节，应仅在正式构造并审计 Preference-v8.2 后，另行授权小规模 DPO-v8.2 smoke。

## Stage 9：Preference-v8.2 CPU-only Construction

Stage 8.5 说明 DPO-v8.1 虽然降低了少量显式 hallucination，却没有提升 case gap：step-50 的主要回归来自 `blank_image` 下与 gold 高重合的短先验答案。因而 v8.2 不再只按视觉断言触发词选 pair，而是显式处理 control direct answer 与 high-F1 overlap。

### 数据设计

Preference-v8.2 的目标比例为：

- `normal_anchor_gold_vs_model_wrong`: 35%；
- `normal_gold_vs_refusal`: 5%；
- `control_direct_answer_vs_caution`: 20%；
- `blank_high_f1_vs_caution`: 15%；
- `text_only_gold_overlap_vs_caution`: 10%；
- `wrong_image_gold_overlap_vs_caution`: 10%；
- `spatial_gold_vs_spatial_wrong`: 5%。

相较 v8.1，normal-related pair 从 26% 提高到 40%，control-related pair 从 67% 降到 55%。这样做是为了在压制 blank/prior/direct control 答案时，仍保住 r3 的 normal QA 能力。所有 `chosen` / `rejected` 均为 answer-only JSON，不含 `reason`。

### Leakage Boundary

`outputs/final_report/stage8_5_*.csv` 来源于 held-out 评测，只作为误差归因依据，绝不生成训练 pair。正式 candidate 与 rejected 答案只能来自已过滤 held-out 的 r3 mining pool 与 normal anchor 数据；审计会再次检查 `heldout_leakage_count = 0`。

### 运行方式

```bash
bash scripts/run_stage9_preference_v8_2_cpu.sh \
  --all \
  --run \
  --max_pairs 500 \
  --seed 42
```

该流程仅收集 candidate、构造与审计 Preference-v8.2、生成 DPO-v8.2 配置草案及诊断报告。即使指定 `--run`，也不会加载 Qwen2.5-VL、不会推理、不会训练、不会使用 GPU。

### 进入下一阶段的条件

只有当 `outputs/data_audit/preference_v8_2_audit.json` 中 `train_ready=true`、held-out leakage 为 0、answer-only 输出合法、normal-related 不低于 35%、control-related 不高于 60%、blank/direct hard pairs 数量达标，才可以另行授权 DPO-v8.2 GPU smoke。DPO-v8.2 应继续从 r3 初始化并使用 frozen r3 reference，默认仅评估 25 steps。当前仍不进入 GRPO-lite。

## Stage 10：DPO-v8.2 Case-gap-aware GPU Smoke

Stage 8.5 表明，DPO-v8.1 虽然压低了部分显式 control hallucination，但 `blank_image` 下与 gold 高重合的先验答案增加，导致 `case_gap` 退化。因此 Preference-v8.2 从仅抑制 hallucination，转为直接优化 case gap：加入 blank/text-only/wrong-image 的 high-F1 overlap 与 control direct-answer 负例，同时将 normal anchor 提升到 40%，保护正常视觉问答。

本轮重点观察 `blank_high_f1_rate_0_20` 与 `control_high_f1_rate_0_20`。这些指标比只看显式 hallucination 更能捕获“没有有效图像仍猜中常见答案”的失败模式。

训练严格限定为 25 steps：这是验证新 preference 方向的 GPU smoke，而非追求收敛的长训。policy 与 frozen reference 均从 `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/` 初始化，不从 DPO-v8.1 继续，以便将变化归因于 v8.2 数据本身。

### 运行方式

```bash
bash scripts/run_stage10_dpo_v8_2_gpu_smoke.sh \
  --all \
  --run \
  --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --eval_ids outputs/final_report/stage4_5_heldout_ids_100.json \
  --bf16 \
  --qlora
```

总控脚本默认 `dry_run`；只有显式传入 `--run` 才会训练和评测。它首先执行 readiness gate，检查 CUDA、r3 adapter、Preference-v8.2 审计、held-out IDs 与固定 DPO 配置；任一关键项失败都会停止。脚本不会覆盖已有 held-out predictions，不包含 GRPO-lite 路径，也不接受自动增加 steps 的参数。

### 如何解读结果

主表位于 `outputs/final_report/stage10_dpo_v8_2_heldout_main_results.csv`，结论只使用 `answer_only` 行。关键成功条件为：

- `normal_f1 >= r3 normal_f1 - 0.02`；
- `case_gap >= r3 case_gap`；
- `blank_high_f1_rate_0_20` 与 `control_high_f1_rate_0_20` 均不高于 r3；
- `normal_refusal_rate <= 0.02`，且答案长度没有异常变长。

若 `normal_f1` 明显下降、`case_gap` 恶化、blank/control high-F1 上升或 normal refusal 上升，本轮视为失败并保留 r3 为主模型。即便出现部分改进，也不自动延长 DPO 训练；需先根据诊断与 case gallery 决定是否构造 v8.3。当前阶段不自动进入 GRPO-lite，因为 v8.2 首先需要证明偏好数据修正足以稳定改善视觉依赖。

### 实际结果（2026-05-26）

DPO-v8.2 已严格完成 25 steps，`final_loss=0.682665`、`final_reward_margin=0.021126`，无 NaN/OOM。held-out answer-only 结果为：`normal_f1=0.3211`、`case_gap=-0.0127`、`blank_high_f1_rate_0_20=0.2900`、`control_high_f1_rate_0_20=0.3367`、`control_hallucination_rate=0.0967`、`normal_refusal_rate=0.0000`。

相较 r3，v8.2 的 normal F1 在容差内且 hallucination 略低，但 case gap 恶化、总体 control high-F1 上升，blank high-F1 仅持平而没有下降。因此 Stage 10 判定失败，当前最佳 checkpoint 仍为 `sft_v3_r3_lingo_smoke`；不追加 DPO steps，不进入 GRPO-lite，后续如继续应先重构 v8.3 的 hard-pair 分布与诊断目标。

<!-- STAGE11_FINAL_CONSOLIDATION_BEGIN -->
## 项目总览与最终结论（Stage 11）

### 1. 项目定位与研究问题

DriveMind-VL 是面向驾驶 VLM 的视觉依赖诊断与后训练校准系统。项目关注的问题是：一个在普通问答上表现较好的模型，是否真的根据图像作答，还是在缺失或错误视觉输入下依赖语言先验猜测。

### 2. 方法总览

方法链路包括 strict visual-control protocol、train/eval leakage audit、answer-only rescore、SFT calibration、三代 DPO preference ablation，以及基于 case gap 和 high-F1 control 行为的失败归因。最终主模型来自 SFT 分支，DPO 分支保留为消融证据。

### 3. Strict Visual-Control Protocol

所有最终结论都在共同 held-out 100 IDs 上评测，并对齐 `normal`、`text-only`、`wrong-image`、`blank-image` 四路输入。主指标包括 Normal F1、Case Gap、Control Hallucination、Control Direct Answer、Control High-F1、Blank High-F1 与 Normal Refusal。

### 4. 后训练流程与数据评测

SFT-v3-r3 通过 answer-only calibration 与稳健 normal replay 获得最佳综合表现。随后 DPO-v8（rule-based）、DPO-v8.1（model-mined）与 DPO-v8.2（case-gap-aware）均以 r3 为初始化或参考进行受控 smoke；最终只使用 held-out strict evaluation 与 answer-only 评分作结论。

### 5. 最终结果

| Model | Method | Normal F1 | Case Gap | Blank High-F1 | Control High-F1 | Hallucination | Normal Refusal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Qwen2.5-VL-3B | base | 0.2230 | -0.0548 | 0.2400 | 0.2333 | 0.0433 | 0.0000 |
| SFT-v2 | sft | 0.2762 | -0.0160 | 0.2800 | 0.2833 | 0.0567 | 0.0000 |
| DPO-v7 | dpo | 0.2727 | -0.0100 | 0.2900 | 0.2767 | 0.0533 | 0.0000 |
| SFT-v3-r2 | sft | 0.2434 | -0.0693 | 0.2500 | 0.2600 | 0.0567 | 0.0000 |
| SFT-v3-r3 | sft | 0.3278 | -0.0011 | 0.2900 | 0.3267 | 0.1000 | 0.0000 |
| DPO-v8 rule-based | dpo | 0.3308 | -0.0109 | 0.3800 | 0.3667 | 0.0900 | 0.0000 |
| DPO-v8.1 step-25 | dpo | 0.3257 | -0.0068 | 0.2900 | 0.3367 | 0.0933 | 0.0000 |
| DPO-v8.1 step-50 | dpo | 0.3168 | -0.0291 | 0.3800 | 0.3633 | 0.0867 | 0.0000 |
| DPO-v8.2 step-25 | dpo | 0.3211 | -0.0127 | 0.2900 | 0.3367 | 0.0967 | 0.0000 |

最终 checkpoint：`checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/`。

### 6. 消融与 Case Gallery

- DPO 可降低部分显式 hallucination，但未稳定改善 case gap。
- blank-image high-F1 prior answer 与 control direct-answer overlap 是后续最需要处理的失败模式。
- 详细消融见 `outputs/final_report/final_ablation_results.md`；案例见 `outputs/final_report/final_case_gallery.html`。

### 7. 已知问题与后续计划

当前 held-out 规模为 100，DriveLM OOD 尚未作为最终结论；DPO 没有成为最终主模型，GRPO-lite 也尚未训练。当前不建议继续盲目追加 DPO steps，也不建议直接进入 GRPO-lite。下一阶段应优先扩大评测并设计 reward harness。

### 8. 一键复现最终报告

```bash
bash scripts/run_stage11_final_report.sh --all --run
```

该命令只读取已存在的结果与 predictions，生成最终表格、报告、gallery 和职业材料；不训练、不推理、不使用 GPU。
<!-- STAGE11_FINAL_CONSOLIDATION_END -->

<!-- STAGE12_DEPTH_EXTENSION_BEGIN -->
## Stage 12：Depth Extension Preparation

Stage 11 已将当前实验收束为可靠的 held-out 结论，但研究深度仍受两个边界限制：LingoQA 独立 held-out 仅 100 条，且 DriveLM 尚未作为 OOD strict visual-control 诊断运行。因此 Stage 12 不继续盲目训练，而是在 CPU 上准备扩大评测与未来 reward-based 优化的可审计基础设施。

### CPU-only 准备内容

- **LingoQA larger held-out**：统一 ID normalize 并排除 SFT、DPO preference 与 mining 来源。当前候选经排除后仅有 `100` 条 leakage-free IDs；`300/500` 输出文件已生成但仅包含真实可用数量，不能作为完成的大规模评测集宣称。
- **DriveLM OOD**：已构造 `2400` 条结构验证通过的四路 strict visual-control 候选，可用于最终 r3 的跨数据集诊断。DriveLM 不要求一定提升，重点检查多相机和 object-token 泛化。
- **GRPO-lite reward harness**：已实现离线奖励分解与已有 prediction debug，当前 reward 排名第一为 `DPO-v8.1 step-25`。奖励显式惩罚 blank/control high-F1、text direct answer、wrong-image confound 与 normal refusal。

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
<!-- STAGE12_DEPTH_EXTENSION_END -->
<!-- STAGE13_DRIVELM_OOD_BEGIN -->
## DriveLM OOD Strict Visual-Control Evaluation

### 目的与边界

DriveLM 与主评测 LingoQA 不同，包含多相机 camera labels、object token 与更复杂的空间/行动推理问题。本轮只在 100 个结构校验通过的 OOD cases 上推理 Base 与最终 `SFT-v3-r3`，不训练模型、不构造训练数据，也不替换 LingoQA held-out 上选出的最终 checkpoint。

### Answer-only 结果

| 模型 | normal_f1 | case_gap | control_high_f1 | hallucination | normal_refusal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen2.5-VL-3B | 0.1566 | -0.1309 | 0.2033 | 0.0533 | 0.0000 |
| SFT-v3-r3 | 0.1888 | -0.1387 | 0.2767 | 0.0667 | 0.0000 |

### 结论

- 跨数据集泛化判定：仅 normal QA 风格迁移，视觉依赖没有泛化。
- r3 最难的能力类型为 `spatial_relation`；完整拆解见 `outputs/final_report/drivelm_ood_capability_breakdown.md`。
- DriveLM OOD 必须写入 limitation：它衡量跨域视觉依赖，不可替代 LingoQA 主结论。
- 是否建议扩大到 300 cases：否，先分析 100-case 暴露的失败模式。
- 当前最终 checkpoint 仍为 `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/`；当前不继续训练，也不进入 GRPO-lite。
<!-- STAGE13_DRIVELM_OOD_END -->
<!-- STAGE13_5_FAILURE_ATTRIBUTION_BEGIN -->
## DriveLM OOD Failure Attribution

Stage 13.5 在不运行模型、不训练、不开启 GPU 的条件下，对已有 DriveLM OOD predictions 做离线归因。r3 的 normal F1 较 Base 提升（0.1566 -> 0.1888），但 case gap 恶化（-0.1309 -> -0.1387），control high-F1 与 blank high-F1 分别上升至 0.2767 与 0.3000。因此，r3 体现的是有限 normal-answer style transfer，而非可靠跨域 visual grounding。

- 数量最多的 failure tag：`camera_specific_failure`（90/100，标签可重叠）。
- 样本量充分的主要瓶颈：`spatial_relation`（n=19, case_gap=-0.3234, control_high_f1=0.5614）。
- Wrong-image confound 共 90 条，其中与 object-token / spatial failure 重叠 63 / 22 条。由于 DriveLM 样本普遍携带 camera labels，camera-specific tag 属于宽口径诊断，不应脱离这些交叉统计过度解释。
- Base -> r3 fix/break：14 / 23 cases。
- 多相机、camera/object alignment 与空间关系应成为下一版 reward harness 的显式约束。
- DriveLM OOD 被保留为 limitation 与方法设计证据，不用于训练数据，不改变 `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/` 作为当前主模型。
- 当前不建议自动运行 DriveLM 300、不继续训练、不进入 GRPO-lite；下一步应先完成 camera-aware/object-aware reward harness v2 的离线审计。

详细材料：`outputs/final_report/drivelm_ood_limitation_writeup.md` 与 `outputs/final_report/drivelm_to_grpo_reward_implications.md`。
<!-- STAGE13_5_FAILURE_ATTRIBUTION_END -->
<!-- STAGE14_REWARD_V2_BEGIN -->
## GRPO-lite Reward Harness v2

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

- LingoQA reward 排名：`DPO-v8.1 step-25 > SFT-v3-r3 > DPO-v8.2 step-25 > DPO-v8`。
- DriveLM reward 排名：`Base Qwen2.5-VL-3B > SFT-v3-r3`；r3 的 camera / spatial / object 平均惩罚为 0.2700 / 0.1018 / 0.1968。
- Sensitivity：6 组可解释权重扰动中，所需 LingoQA/DriveLM 排序同时保持 5/6 组，说明当前 reward 仍存在排名敏感性。
- Normal refusal sanity gate 已加入，正常图像全拒答不会得到高 reward。

### 当前决策

**Do not train GRPO-lite yet.** Reward v2 已能惩罚 DriveLM OOD 的结构化失败与 blank/control overlap，但尚不能稳定、完整解释 LingoQA 最终模型选择；在进入 GRPO-lite 前，必须先继续校准 reward 并人工审阅 disagreement cases。禁止使用 held-out 或 DriveLM OOD evaluation cases 作为训练样本。
<!-- STAGE14_REWARD_V2_END -->

<!-- STAGE14_5_REWARD_V2_1_BEGIN -->
## Stage 14.5：Reward v2.1 Calibration

Reward v2 未能稳定解释主模型排序：blank 权重增强时会使 DPO-v8.2 超过 r3，且 camera/object penalty 触发过宽。因此，本阶段仅用已有 predictions 做 v2.1 离线校准与 disagreement review，不训练、不推理、不使用 GPU。

### v2.1 改动

- blank penalty 改为 `none / medium / high` 饱和档位，避免 F1 幅度无限放大排序影响；
- camera penalty 改为 camera 线索与 wrong-image confound 联合触发；
- object penalty 改为 object-token 线索与 control confound 联合触发；
- spatial penalty 只针对 low-normal/high-control 或方向冲突等明确失败；
- 为正常视觉答对、控制输入谨慎回答增加小额 caution reward，同时保留 normal refusal 强惩罚；
- 加入 pairwise reward sanity gate。

### 离线审计结果

- Disagreement records：82；人工优先复核表：50 行。
- LingoQA ranking：`SFT-v3-r3 > DPO-v8.1 step-25 > DPO-v8.2 step-25 > DPO-v8`。
- DriveLM ranking：`Base Qwen2.5-VL-3B > SFT-v3-r3`。
- Sensitivity：8 组权重下，DriveLM Base > r3、LingoQA r3 >= DPO-v8/v8.2 及 normal-refusal gate 均保持通过。
- Overall pairwise accuracy：0.9752。
- blank_low_vs_blank_high accuracy：0.9375。
- normal_correct_vs_normal_refusal accuracy：1.0000。
- drive_spatial_good_vs_spatial_bad accuracy：1.0000。

### Readiness 与限制

**自动离线 gate 已通过；在人工快速复核后，可另行申请最小 GPU smoke。** Object/camera 触发逻辑已更精准，但 object-token disagreement 仍应人工浏览；即使进入后续 smoke，也只允许 r3 初始化、frozen r3 reference、`group_size=4`、`temperature=0.7`、`max_new_tokens=64`、最多 `50 steps`，且禁止把 held-out 或 DriveLM OOD 样本作为训练数据。
<!-- STAGE14_5_REWARD_V2_1_END -->
<!-- STAGE14_6_REWARD_V2_2_BEGIN -->
## Stage 14.6：Reward v2.2 Patch and Manual Review Integration

Stage 14.5 的自动门槛虽已通过，但人工复核指出了四类 v2.1 漏洞：object-token 在四路输入下保持近似答案、normal 对象类别错配、`terminate`/`task completed` 等无效模板，以及 control 答案与 normal 几乎不变。因此本阶段仅对现有预测做 CPU-only reward 修补与审计，没有训练或推理。

### v2.2 新增 Penalties

- `object_token_invariant_answer_penalty`：object-token 问题在至少三路输入下输出高度相似且非谨慎回答时触发。
- `normal_object_category_mismatch_penalty`：normal 下 gold 与答案出现明确交通对象类别冲突时触发，并允许 `vehicle` 与具体车辆类别的兼容关系。
- `invalid_generic_answer_penalty`：显式惩罚异常模板输出，如 `terminate`、`no further actions needed` 或 `placeholder`。
- `control_same_as_normal_penalty`：control 输入变化后答案仍与 normal 高相似且非 caution/refusal 时触发。

### 离线结论

- 人工标签分布：`{"reward_correct": 15, "reward_under_penalized": 23, "metric_conflict": 11, "reward_over_penalized": 1}`。
- LingoQA ranking：`DPO-v8.1 step-25 > DPO-v8.2 step-25 > SFT-v3-r3 > DPO-v8`。
- DriveLM ranking：`Base Qwen2.5-VL-3B > SFT-v3-r3`。
- Overall pairwise accuracy：0.9277。
- Object-invariant pair accuracy：0.9333。
- Control-same-as-normal pair accuracy：0.7333。
- 人工漏罚样本改善：23/23。

### GPU 边界

**Do not train GRPO-lite yet.** 即使离线 gate 通过，也只允许在明确授权后进行一次从 frozen-r3 reference 出发的 50-step GRPO-lite smoke；禁止长训，禁止将 held-out 或 DriveLM OOD evaluation records 用作训练数据。
<!-- STAGE14_6_REWARD_V2_2_END -->
<!-- STAGE15_QWEN3_VL_STACK_BEGIN -->
## Stage 15：Qwen3-VL Open-source Training Stack Preparation

当前 Qwen2.5-VL 主模型仍为 `checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/`。Stage 15 不改变既有实验结论，而是为下一代 Qwen3-VL 分支准备成熟、可复现的开源训练栈：Qwen3-VL 更适合作为新的 base/SFT 对照，但必须先经过相同的 strict visual-control protocol 验证，不能直接以模型升级代替视觉依赖评估。

### Stack Selection

- 主训练栈：**ms-swift**。官方资料明确覆盖 Qwen3-VL、多模态 SFT/DPO/GRPO、LoRA/QLoRA、vit/aligner/llm 控制、reward 插件、vLLM 与 DeepSpeed。
- 快速验证栈：**Unsloth**。其 Qwen3-VL Vision 与 Vision-GRPO notebook 适合短 smoke，但不是当前工程主线。
- 参考实现：**Qwen-VL-Series-Finetune**。适合核对 Qwen-specific vision tuning、多图与视频训练格式，当前导出需锁定 commit 后再人工确认。

### Prepared Files And Data

- 调研与矩阵：`docs/qwen3_vl_finetune_survey.md`、`outputs/final_report/qwen3_vl_training_stack_matrix.csv`。
- ms-swift SFT 导出：`data/qwen3_vl/ms_swift/sft_lingoqa_r3_train.jsonl`（429 条）。
- ms-swift DPO 导出：`data/qwen3_vl/ms_swift/dpo_preference_v8_2.jsonl`（500 对）。
- ms-swift GRPO prompt 草案：`data/qwen3_vl/ms_swift/grpo_lite_prompts.jsonl`（429 条；仅格式准备，不可训练）。
- Qwen3 base eval 入口：`scripts/run_qwen3_vl_base_eval.sh`，输出隔离到 `outputs/predictions_qwen3_vl/`。
- SFT/GRPO 配置草案：`configs/ms_swift/`。

### Audit And Next GPU Boundary

- 所有导出 held-out/OOD leakage 均为 `0`。
- 数据格式审计通过：`true`。
- 下一步如获 GPU 授权，应**先跑 Qwen3-VL-4B Base 的 LingoQA held-out 100 与 DriveLM OOD 100 strict eval**，再决定是否启动 SFT smoke。
- Reward v2.2 当前仍 `not ready`，因此 GRPO-lite 配置只是禁用草案，当前不训练 GRPO。
<!-- STAGE15_QWEN3_VL_STACK_END -->
