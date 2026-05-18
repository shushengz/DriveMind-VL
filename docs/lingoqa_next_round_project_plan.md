# LingoQA 下一轮优化方案：从性能提升到可展示项目

日期：2026-05-18

## 当前结论

当前最可靠的正结果是 SFT-v2 visual-scale：

| model | normal | text-only | wrong | blank | setting gap | per-case gap | positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| base clean dev | 0.2763 | 0.1902 | 0.2039 | 0.2100 | 0.0663 | -0.0484 | 0.24 |
| SFT-v2 visual-scale | 0.3680 | 0.2873 | 0.2965 | 0.2564 | 0.0715 | -0.0070 | 0.22 |
| v4 guarded SimPO | 0.0668 | 0.0510 | 0.0668 | 0.0524 | 0.0000 | -0.0040 | 0.00 |

SFT-v2 把 normal strict F1 从 0.2763 提升到 0.3680，是可以写进简历的主结果。但 text/wrong/blank control 也同步升高，说明它主要增强了任务遵循和答案模式，严格视觉依赖还没有被解决。

v4 guarded SimPO 不是训练不充分，而是目标错了：数据把大量 control chosen 写成拒答，训练器又对 chosen 统一加 SFT loss，导致模型学到“遇到异常就统一拒答”的捷径。

## 不建议全量训练

现在不建议对 3B 模型做全量训练：

1. 当前瓶颈是 preference 信号和数据边界，不是模型容量。
2. 32GB 单卡全参训练 Qwen2.5-VL-3B 的显存、吞吐和 checkpoint sweep 成本都不划算。
3. 如果目标函数仍然污染，全参会把拒答捷径写进 backbone，比 LoRA 更难恢复。

下一阶段应继续 LoRA / QLoRA，小步 checkpoint sweep。只有当 LoRA 在 clean dev/test 上稳定提升 normal 且不靠拒答刷 control，才考虑 projector/top-layer partial tune，而不是直接全参。

## 下一轮目标

下一轮命名为 `preference-v5-from-predictions`：

1. 以 SFT-v2 作为当前 best adapter。
2. 先用 SFT-v2 在 clean train 的 normal/text/wrong/blank 上生成真实预测。
3. 只对 SFT-v2 已经真实犯错的 control 输出构造 preference pair。
4. control pair 的 rejected 是模型真实预测，不再是 gold answer。
5. normal pair 保留 SFT anchor，control pair 的 SFT anchor 必须为 0。
6. 每 30 step 保存 checkpoint，用 clean dev sweep 选择最稳 checkpoint。

硬门槛：

- normal dev F1 不低于 0.35；
- normal refusal rate 保持很低；
- per-case gap 不低于 -0.01，最好转正；
- control refusal rate 可上升，但不能牺牲 normal；
- 最终 test 只跑一次，不能用 test 调参。

## 已实现的代码调整

1. 新增 `src/data/build_lingoqa_preference_v5_from_predictions.py`
   - 从 SFT-v2 train predictions 挖真实 control 幻觉；
   - normal 生成 `answer_over_refusal/bad_prediction` anchor；
   - text/wrong/blank 只在模型非拒答时生成 `control_refusal_over_model_hallucination`；
   - control pair `sft_anchor_weight=0.0`，normal pair `sft_anchor_weight=1.0`。

2. 修改 `src/train_qwen25vl_dpo_lora.py`
   - 支持 per-pair `sft_anchor_weight`；
   - 默认只给 normal pair 加 chosen SFT loss；
   - 支持 `--max_steps`、`--save_steps`；
   - 输出 `train_config.json`、`dataset_summary.json`、`train_log.jsonl`；
   - 保存 `checkpoint-step-XXXXXX`，方便 dev sweep。

3. 新增拒答率评测
   - `src/eval/refusal_detection.py`
   - `run_all_eval.py` 输出 `external_refusal_rate`
   - visual-control summary 输出 normal/control refusal rate。

4. 新增执行脚本
   - `scripts/49_generate_lingoqa_sft_v2_train_predictions_for_pref.sh`
   - `scripts/50_build_lingoqa_preference_v5_from_predictions.sh`
   - `scripts/51_train_lingoqa_pref_v5_from_predictions.sh`
   - `scripts/52_eval_lingoqa_pref_v5_checkpoint_sweep.sh`

## GPU 模式执行顺序

先生成 SFT-v2 在 train split 的四路预测：

```bash
source /root/miniconda3/bin/activate drivemind-vl
cd /root/autodl-tmp/DriveMind-VL
mkdir -p outputs/logs

ADAPTER=outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v2_visual_scale \
PREFIX=lingoqa_clean_v2_train_sft_v2_visual_scale_for_pref \
bash scripts/49_generate_lingoqa_sft_v2_train_predictions_for_pref.sh \
2>&1 | tee outputs/logs/lingoqa_clean_v2_train_sft_v2_visual_scale_for_pref_eval.log
```

构建 preference-v5：

```bash
bash scripts/50_build_lingoqa_preference_v5_from_predictions.sh \
2>&1 | tee outputs/logs/lingoqa_preference_v5_from_predictions_build.log
```

训练 preference-v5：

```bash
OUTPUT_DIR=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v5_from_predictions \
INIT_ADAPTER=outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v2_visual_scale \
LR=8e-7 \
DPO_BETA=0.03 \
CHOSEN_SFT_WEIGHT=0.08 \
MAX_STEPS=120 \
SAVE_STEPS=30 \
bash scripts/51_train_lingoqa_pref_v5_from_predictions.sh \
2>&1 | tee outputs/logs/qwen25vl_3b_lingoqa_pref_v5_from_predictions_train.log
```

dev checkpoint sweep：

```bash
RUN_DIR=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v5_from_predictions \
SPLIT=dev \
PREFIX_BASE=lingoqa_clean_v2_dev_pref_v5_sweep \
bash scripts/52_eval_lingoqa_pref_v5_checkpoint_sweep.sh \
2>&1 | tee outputs/logs/lingoqa_clean_v2_dev_pref_v5_sweep.log
```

选择 checkpoint 后，最后只在 test 上跑一次。

## 简历定位

推荐定位为：

DriveMind-VL：面向车载场景的多模态视觉 grounding 评测与微调项目。基于 Qwen2.5-VL-3B 构建 segment-level clean split 和 normal/text-only/wrong-image/blank-image 四路视觉对照评测，使用 LoRA SFT 将 clean dev normal strict F1 从 27.63% 提升到 36.80%，并系统定位 preference optimization 在小规模视觉 grounding 数据上的拒答坍缩问题。
