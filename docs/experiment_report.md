# Experiment Report

This file records the first Base vs LoRA/SFT experiment for DriveMind-VL.

## Goal

Compare Qwen2.5-VL-3B-Instruct base prompting against a small LoRA/SFT run on DriveMind-Instruct seed data.

## Fixed Split

- Train: `data/processed/drivemind_train.jsonl`
- Eval: `data/processed/drivemind_eval.jsonl`
- Split seed: `42`
- Train size: `80`
- Eval size: `20`
- Task balance: 16 train / 4 eval samples for each of the 5 task types.

## Training Setup

- Model: `/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct`
- GPU: RTX 4080 SUPER, 31.47GB visible memory
- LoRA rank: `16`
- LoRA alpha: `32`
- Trainable params: `37,152,768`
- Trainable ratio: `0.9798%`
- Epochs: `1`
- Optimizer steps: `10`
- Gradient accumulation: `8`
- Precision: `bf16`
- Adapter output: `outputs/checkpoints/qwen25vl_3b_lora_smoke`
- Adapter size: about `158M`

## Commands

```bash
bash scripts/11_split_data.sh

MAX_SAMPLES=20 bash scripts/14_eval_qwen25vl_3b_base_fixed_eval.sh \
  /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct

MAX_SAMPLES=80 EPOCHS=1 bash scripts/12_train_qwen25vl_3b_lora_smoke.sh \
  /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct

MAX_SAMPLES=20 bash scripts/13_eval_qwen25vl_3b_lora.sh \
  /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  outputs/checkpoints/qwen25vl_3b_lora_smoke
```

## Metrics

| Metric | Base 3B | LoRA 3B | Delta |
|---|---:|---:|---:|
| json_validity | 1.0000 | 1.0000 | +0.0000 |
| risk_accuracy | 1.0000 | 1.0000 | +0.0000 |
| tool_accuracy | 1.0000 | 1.0000 | +0.0000 |
| tool_argument_accuracy | 0.6667 | 0.6667 | +0.0000 |
| unsafe_rejection_rate | 1.0000 | 1.0000 | +0.0000 |
| schema_completeness | 1.0000 | 1.0000 | +0.0000 |
| reason_keyword_hit | 0.5000 | 0.6500 | +0.1500 |
| avg_reward | 0.7315 | 0.7505 | +0.0190 |
| bad_cases | 0 | 0 | 0 |

## Qualitative Observation

Base 3B already follows the strict prompt well on the fixed eval set. LoRA did not improve tool argument accuracy in this small run, but it improved reasoning alignment and corrected at least one cabin fatigue case:

- Base predicted `driver_state=normal` for a fatigue cabin sample.
- LoRA predicted `driver_state=fatigued`, `passenger_state=normal`, and `suggestion=remind_driver`.

The remaining argument mismatch is mainly fine-grained slot naming, such as `door=left` vs `door=left_front`. This suggests the next bottleneck is not JSON formatting, but precise schema grounding and richer slot-value supervision.

## Initial Conclusion

The first LoRA/SFT smoke experiment supports a cautious positive conclusion: with only 80 synthetic DriveMind-Instruct samples and 10 optimizer steps, Qwen2.5-VL-3B preserves all strong base prompt-following metrics and improves reasoning-related reward. However, it does not yet improve fine-grained tool argument accuracy.

The next experiment should focus on data quality instead of model scale: expand hard cases for tool arguments and safety arguments, then rerun a slightly larger 3B LoRA experiment before trying 7B QLoRA.

