# LingoQA SFT-v1 DPO Optimization Plan

## Motivation

The previous ordinary SFT variants improved or preserved normal-image F1, but
also raised text-only, blank-image, or wrong-image controls. This means ordinary
SFT is still teaching the model the answer style and task prior more than
strict visual dependency.

The next training objective should directly optimize the contrast required by
strict visual-control evaluation:

- correct frames: grounded visual answer is preferred over generic refusal;
- text-only / blank / wrong frames: uncertainty or refusal is preferred over
  hallucinating the original visual answer.

## Data

Input:

- `data/processed/lingoqa_sft_v1_control_aware_77.jsonl`
- `data/processed/lingoqa_sft_v1_balanced_67.jsonl` as original-answer lookup

Generated output:

- `data/processed/lingoqa_sft_v1_preference_pairs_77.jsonl`
- `outputs/eval_results/lingoqa_sft_v1_preference_pairs_77_summary.json`

Expected preference pair distribution:

- `normal_answer_over_refusal`: 57
- `control_refusal_over_hallucination`: 20
- modes: normal 57, text_only 5, blank_image 5, wrong_image 10

## Training

Use `src/train_qwen25vl_dpo_lora.py`. It is a minimal DPO LoRA trainer that
uses the base model with adapters disabled as the reference model, so no second
model copy or TRL dependency is required.

Conservative defaults:

- LoRA target: attention projections only
- rank: 8
- learning rate: `5e-6`
- beta: `0.1`
- epochs: 1
- chosen SFT auxiliary weight: `0.05`

These defaults intentionally reduce adapter drift. If normal F1 collapses,
increase `CHOSEN_SFT_WEIGHT` to `0.1`. If wrong-image remains high, try
`DPO_BETA=0.2` before increasing epochs.

## Evaluation Rule

The original 100-case visual-control set is now diagnostic only because the
training data was mined from related cases. Use it to compare directions, not
to claim clean benchmark improvement. A clean split is still required for the
final claim.

Primary diagnostic target:

- preserve normal 3-frame F1 near or above the base best strategy;
- reduce wrong-image and blank-image F1;
- move per-case conservative visual dependency gap above zero.
