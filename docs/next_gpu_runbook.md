# Next GPU Runbook

This runbook keeps the next paid GPU session focused. Run the first command in
no-GPU mode to verify existing artifacts, then switch to GPU mode for the
prediction, preference, training, and sweep steps.

## No-GPU Check

```bash
source /root/miniconda3/bin/activate drivemind-vl
cd /root/autodl-tmp/DriveMind-VL
PYTHON_BIN=/root/miniconda3/envs/drivemind-vl/bin/python \
  bash scripts/53_compare_lingoqa_visual_control_runs.sh
```

Expected output:

- `docs/lingoqa_visual_control_run_comparison.md`
- `outputs/eval_results/lingoqa_visual_control_run_comparison.csv`
- `outputs/eval_results/lingoqa_visual_control_run_comparison.json`

Current decision from the latest no-GPU report:

- `sft_v2_visual_scale` is still the incumbent.
- `pref_v5_sweep_best` currently points to `checkpoint-step-000030`.
- `checkpoint-step-000030` is rejected as an incumbent regression: normal F1
  and setting-level visual gap are both lower than SFT-v2, even though
  per-case gap is slightly better.

## GPU Step 1: Generate SFT-v2 Train Predictions

```bash
source /root/miniconda3/bin/activate drivemind-vl
cd /root/autodl-tmp/DriveMind-VL
mkdir -p outputs/logs

ADAPTER=outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v2_visual_scale \
PREFIX=lingoqa_clean_v2_train_sft_v2_visual_scale_for_pref \
bash scripts/49_generate_lingoqa_sft_v2_train_predictions_for_pref.sh \
2>&1 | tee outputs/logs/lingoqa_clean_v2_train_sft_v2_visual_scale_for_pref_eval.log
```

## GPU Step 2: Build Preference-v5 Data

```bash
bash scripts/50_build_lingoqa_preference_v5_from_predictions.sh \
2>&1 | tee outputs/logs/lingoqa_preference_v5_from_predictions_build.log
```

## GPU Step 3: Train Preference-v5

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

## GPU Step 4: Dev Checkpoint Sweep

To keep GPU cost down, prefer selective checkpoint evaluation first. This
reuses any existing checkpoint outputs and only evaluates missing ones:

```bash
RUN_DIR=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v5_from_predictions \
SPLIT=dev \
PREFIX_BASE=lingoqa_clean_v2_dev_pref_v5_sweep \
REUSE_EXISTING=1 \
CHECKPOINTS=checkpoint-step-000060,checkpoint-step-000090,checkpoint-step-000120 \
bash scripts/52_eval_lingoqa_pref_v5_checkpoint_sweep.sh \
2>&1 | tee outputs/logs/lingoqa_clean_v2_dev_pref_v5_selective_sweep.log
```

Only run the full sweep if the selective sweep shows a promising checkpoint:

```bash
RUN_DIR=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v5_from_predictions \
SPLIT=dev \
PREFIX_BASE=lingoqa_clean_v2_dev_pref_v5_sweep \
bash scripts/52_eval_lingoqa_pref_v5_checkpoint_sweep.sh \
2>&1 | tee outputs/logs/lingoqa_clean_v2_dev_pref_v5_sweep.log
```

## GPU Step 5: Preference-v5b If v5 Still Loses to SFT-v2

If the selective sweep does not beat `sft_v2_visual_scale`, stop sweeping v5 and
train a cheaper v5b run. The intent is to preserve normal-image quality while
increasing pressure on high-F1 control hallucinations.

Build the stronger-control preference set:

```bash
OUTPUT=data/processed/lingoqa_preference_v5b_stronger_controls.jsonl \
SUMMARY=outputs/eval_results/lingoqa_preference_v5b_stronger_controls_summary.json \
TEXT_ONLY_WEIGHT=0.50 \
BLANK_IMAGE_WEIGHT=0.70 \
WRONG_IMAGE_WEIGHT=0.80 \
NORMAL_SFT_ANCHOR_WEIGHT=1.20 \
NORMAL_BAD_F1_THRESHOLD=0.20 \
MIN_CONTROL_F1=0.10 \
bash scripts/50_build_lingoqa_preference_v5_from_predictions.sh \
2>&1 | tee outputs/logs/lingoqa_preference_v5b_stronger_controls_build.log
```

Train a short v5b run:

```bash
TRAIN_FILE=data/processed/lingoqa_preference_v5b_stronger_controls.jsonl \
OUTPUT_DIR=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v5b_stronger_controls \
INIT_ADAPTER=outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v2_visual_scale \
LR=5e-7 \
DPO_BETA=0.05 \
CHOSEN_SFT_WEIGHT=0.12 \
DEFAULT_NORMAL_SFT_ANCHOR_WEIGHT=1.0 \
MAX_STEPS=60 \
SAVE_STEPS=30 \
bash scripts/51_train_lingoqa_pref_v5_from_predictions.sh \
2>&1 | tee outputs/logs/qwen25vl_3b_lingoqa_pref_v5b_stronger_controls_train.log
```

Then evaluate just the two v5b checkpoints:

```bash
RUN_DIR=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v5b_stronger_controls \
SPLIT=dev \
PREFIX_BASE=lingoqa_clean_v2_dev_pref_v5b_sweep \
CHECKPOINTS=checkpoint-step-000030,checkpoint-step-000060 \
bash scripts/52_eval_lingoqa_pref_v5_checkpoint_sweep.sh \
2>&1 | tee outputs/logs/lingoqa_clean_v2_dev_pref_v5b_sweep.log
```

## GPU Step 6: Preference-v5c From the Best v5 Checkpoint

The selective v5 sweep showed `checkpoint-step-000090` as the best practical
starting point: it improved normal dev F1 and setting-level visual dependency
without the stronger control loss seen at `checkpoint-step-000120`.

Build the v5c preference set with stronger control weights:

```bash
OUTPUT=data/processed/lingoqa_preference_v5c_step90_stronger_controls.jsonl \
SUMMARY=outputs/eval_results/lingoqa_preference_v5c_step90_stronger_controls_summary.json \
TEXT_ONLY_WEIGHT=0.50 \
BLANK_IMAGE_WEIGHT=0.80 \
WRONG_IMAGE_WEIGHT=0.90 \
NORMAL_SFT_ANCHOR_WEIGHT=1.20 \
NORMAL_BAD_F1_THRESHOLD=0.20 \
MIN_CONTROL_F1=0.10 \
bash scripts/50_build_lingoqa_preference_v5_from_predictions.sh \
2>&1 | tee outputs/logs/lingoqa_preference_v5c_step90_stronger_controls_build.log
```

Train a short v5c run from v5 `checkpoint-step-000090`:

```bash
TRAIN_FILE=data/processed/lingoqa_preference_v5c_step90_stronger_controls.jsonl \
OUTPUT_DIR=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v5c_step90_stronger_controls \
INIT_ADAPTER=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v5_from_predictions/checkpoint-step-000090 \
LR=3e-7 \
DPO_BETA=0.05 \
CHOSEN_SFT_WEIGHT=0.12 \
DEFAULT_NORMAL_SFT_ANCHOR_WEIGHT=1.0 \
MAX_STEPS=40 \
SAVE_STEPS=20 \
bash scripts/51_train_lingoqa_pref_v5_from_predictions.sh \
2>&1 | tee outputs/logs/qwen25vl_3b_lingoqa_pref_v5c_step90_stronger_controls_train.log
```

Evaluate both short-run checkpoints:

```bash
RUN_DIR=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v5c_step90_stronger_controls \
SPLIT=dev \
PREFIX_BASE=lingoqa_clean_v2_dev_pref_v5c_step90_stronger_controls_sweep \
CHECKPOINTS=checkpoint-step-000020,checkpoint-step-000040 \
bash scripts/52_eval_lingoqa_pref_v5_checkpoint_sweep.sh \
2>&1 | tee outputs/logs/lingoqa_clean_v2_dev_pref_v5c_step90_stronger_controls_sweep.log
```

Current dev result: use `checkpoint-step-000020` as the preferred checkpoint.
It has stronger normal-image quality than `checkpoint-step-000040` while keeping
the visual-control gaps positive.

| checkpoint | normal | text | wrong | blank | setting_gap | case_gap | pos_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 step90 | 0.3777 | 0.2904 | 0.2976 | 0.2780 | 0.0801 | -0.0005 | 0.24 |
| v5c step20 | 0.3827 | 0.2914 | 0.2982 | 0.2826 | 0.0845 | 0.0051 | 0.27 |
| v5c step40 | 0.3791 | 0.2866 | 0.2931 | 0.2854 | 0.0860 | 0.0054 | 0.27 |

## After Sweep

The sweep script now selects a best checkpoint automatically. It prefers
checkpoints that satisfy:

- normal dev F1 >= 0.35
- normal refusal rate stays low
- per-case visual gap >= -0.01, ideally positive
- control scores decrease without collapsing normal-image quality
- among passing candidates, higher normal-image F1 is preferred before small
  differences in visual-gap metrics

It writes:

- `docs/lingoqa_clean_v2_dev_pref_v5_sweep_best_selection.md`
- `outputs/eval_results/lingoqa_clean_v2_dev_pref_v5_sweep_best_visual_control_summary.json`
- `outputs/eval_results/lingoqa_clean_v2_dev_pref_v5_sweep_best_visual_control_case_summary.json`

Rerun the no-GPU comparison command after the sweep. If the best checkpoint is
accepted, run clean test exactly once.
