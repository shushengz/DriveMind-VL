# DriveLM Scene-Heldout Training Runbook

This stage moves the project from LingoQA-only tuning to a public benchmark aligned with autonomous-driving VQA.

## Current No-GPU Artifacts

- Train split: `data/processed/drivelm_train_scene.jsonl`
- Dev split: `data/processed/drivelm_dev_scene.jsonl`
- Split report: `docs/drivelm_scene_split_report.md`
- V7 failure report: `docs/drivelm_eval_v7_step24_failure_analysis.md`

The split is scene-heldout: train and dev scene overlap is 0.

## Why This Is The Next Step

The LingoQA preference adapters improved internal visual-control behavior, but failed to generalize to DriveLM. On the 300-sample DriveLM probe, the base model still has the best normal F1, while SFT-v2, v5c, and v7 all have negative visual dependency gaps. This points to domain shift and language-prior shortcuts rather than a lack of more DPO steps.

## GPU Step 1: Establish Scene-Heldout Base

```bash
cd /root/autodl-tmp/DriveMind-VL
source /root/miniconda3/bin/activate drivemind-vl

PREFIX=drivelm_dev_scene_base \
MAX_SAMPLES=300 \
bash scripts/62_eval_drivelm_scene_visual_controls.sh /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
2>&1 | tee outputs/logs/drivelm_dev_scene_base.log
```

## GPU Step 2: Train DriveLM Scene-SFT Adapter

Start with the cheap 1200-sample run. If normal F1 improves without worsening visual dependency, expand to 2400.

```bash
cd /root/autodl-tmp/DriveMind-VL
source /root/miniconda3/bin/activate drivemind-vl

MAX_SAMPLES=1200 \
EPOCHS=1 \
OUTPUT_DIR=outputs/checkpoints/qwen25vl_3b_drivelm_scene_sft_1200 \
bash scripts/61_train_drivelm_scene_sft.sh /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
2>&1 | tee outputs/logs/train_drivelm_scene_sft_1200.log
```

## GPU Step 3: Evaluate The Adapter

```bash
cd /root/autodl-tmp/DriveMind-VL
source /root/miniconda3/bin/activate drivemind-vl

PREFIX=drivelm_dev_scene_sft_1200 \
ADAPTER=outputs/checkpoints/qwen25vl_3b_drivelm_scene_sft_1200 \
MAX_SAMPLES=300 \
bash scripts/62_eval_drivelm_scene_visual_controls.sh /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
2>&1 | tee outputs/logs/drivelm_dev_scene_sft_1200.log
```

## Compare

```bash
cd /root/autodl-tmp/DriveMind-VL
source /root/miniconda3/bin/activate drivemind-vl

BASE_PREFIX=drivelm_dev_scene_base \
ADAPTER_PREFIX=drivelm_dev_scene_sft_1200 \
bash scripts/63_compare_drivelm_scene_runs.sh
```

## Decision Rule

A useful checkpoint should meet both conditions:

- normal F1 is higher than base or at least not clearly lower;
- visual dependency gap improves, especially wrong-image and blank-image controls dropping below normal.

If normal F1 improves but visual dependency remains negative, the next iteration should be preference data mined from DriveLM failure cases, not another LingoQA-only DPO run.
