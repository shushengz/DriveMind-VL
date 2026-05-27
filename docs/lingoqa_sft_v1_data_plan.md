# LingoQA SFT-v1 Data Plan

## Objective

SFT-v1 should improve strict visual-control grounding, not just normal-answer F1. The primary acceptance metric remains the per-case conservative visual dependency gap:

```text
normal_f1 - max(text_only_f1, wrong_image_f1, blank_image_f1)
```

A useful SFT-v1 adapter should increase normal-image performance while keeping text-only, wrong-image, and blank-image controls from rising in parallel.

## Target Mix

The next ordinary SFT dataset should be 50-80 high-quality examples, but the exact count is less important than role balance and visual evidence quality.

| role | target count | use in ordinary SFT | notes |
|---|---:|---|---|
| positive_visual_grounding | 30-40 | yes | Main driver of strict grounding; prioritize examples where normal clearly beats controls. |
| spatial_reasoning_correction | 20-30 | yes, after evidence check | Must state visible objects and relative positions, not only a driving conclusion. |
| anti_hallucination_counterfactual | 10-15 | capped only | Use sparingly in plain SFT; preferably convert to preference/rejection samples. |
| high_quality_fix_rewritten | 7-15 | yes after rewrite | Promote existing fix rows only after rewritten answer/reason passes second review. |
| synthetic_drivemind_safety_tool_risk | 0-10 | optional | Keep small and source-tagged; do not use LingoQA eval frames or answers. |

The mix should not reuse the SFT-v0 ratio. A 64% anti-hallucination share is too high for a grounding objective.

## Current Pool Reality Check

Current assets show the following gap against the target:

| source | count |
|---|---:|
| candidate pool | 100 |
| reviewed rows | 41 |
| SFT-v0 keep | 25 |
| SFT-v0 fix | 7 |
| SFT-v0 drop | 9 |
| candidate pool positive_grounding | 27 |
| SFT-v0 keep positive_visual_grounding | 3 |
| SFT-v0 keep anti_hallucination_counterfactual | 16 |

This means SFT-v1 cannot be built by simply retraining SFT-v0. The current pool has many unreviewed positive candidates, but even accepting all 27 would still be below the 30-40 target. Additional positive and spatial samples should be mined from a clean split.

## Data Source Policy

Use the existing 100-sample control set for diagnosis and smoke-loop planning. Do not use it to claim final held-out benchmark gains if its cases are included in training.

For credible SFT-v1 evaluation:

1. Keep the existing strict visual-control evaluation fixed as a diagnostic baseline, or explicitly label it as contaminated if any of its examples enter training.
2. Prefer mining SFT-v1 candidates from non-reporting data: a separate LingoQA subset, additional unused 500-set rows, or a new held-out split.
3. Preserve a clean evaluation set that is never used for SFT candidate construction, rewrite, or preference generation.
4. Source-tag every row with `benchmark_source`, `curation_role`, `prompt_variant`, `frame_strategy`, and whether it came from LingoQA or DriveMind synthetic data.

## Row-Level Acceptance Criteria

A row can enter ordinary SFT only if all of these are true:

- The answer is visually supported by the selected frames.
- The answer is not a pure text prior or generic driving guess.
- The target response includes the visual evidence needed for the conclusion.
- The row is not marked drop.
- If it came from a confound category, its use is explicitly capped or converted to a different objective.

For spatial/reasoning rows, the target answer should include at least one of:

- object identity plus position, such as pedestrian on the zebra crossing, cyclist in the left lane, traffic light ahead;
- ego action plus visible cause, such as stop because the light is red, proceed because the light is green and the crosswalk is clear;
- temporal cue when needed, such as later frames show the pedestrian entering or leaving the crossing.

Avoid short answers such as `No.` unless the image evidence is explicitly described in `reason`.

## Treatment by Candidate Type

| candidate_type | SFT-v1 treatment |
|---|---|
| positive_grounding | Highest priority for ordinary SFT after visual review. |
| hard_negative_spatial_reasoning | Rewrite into evidence-rich spatial correction, then ordinary SFT. |
| wrong_image_confound | Prefer contrastive preference/rejection; cap if kept in ordinary SFT. |
| language_prior_confound | Do not use as ordinary positive SFT; use only for rejection/uncertainty if reopened. |
| blank_image_confound | Do not use as ordinary positive SFT; use only for rejection/uncertainty if reopened. |
| all_settings_failed | Manual image review first; rewrite or drop. |
| mixed_review | Manual decision required. |

## Preference / Rejection Plan for Wrong-Image Confounds

`wrong_image_confound` should not be treated as normal positive SFT by default. A better structure is:

- chosen: answer grounded in the normal frames, with visual evidence;
- rejected: answer produced from wrong/blank/no-image context, or an unsupported confident answer;
- optional uncertainty target: when visual evidence is absent or inconsistent, respond that the current frames do not support the claim.

This can be trained later with DPO, ORPO, or a lightweight reward/preference objective. If the project stays with plain SFT for the next smoke run, cap anti-hallucination rows to 10-15 and keep them evidence-rich.

## Fix Row Promotion

The 7 fix rows are valuable, but not in their current form. Promotion requires:

1. Replace the answer/reason with the suggested rewrite or a stronger human rewrite.
2. Verify the rewrite against the selected first/middle/last frames.
3. Keep the original gold/reference in metadata for traceability.
4. Mark the row as `high_quality_fix_rewritten` and record reviewer notes.
5. Run the strict visual-control evaluation after training to ensure fixes do not raise controls.

## Synthetic DriveMind Rows

A small number of DriveMind-specific rows may be mixed in only if they do not contaminate LingoQA evaluation:

- safety refusal rows for unsupported visual claims;
- tool/risk rows using non-LingoQA images or synthetic structured state;
- cockpit personalization rows independent of LingoQA frames.

Keep this at 0-10 rows for SFT-v1. The next experiment should remain focused on LingoQA grounding, so synthetic rows should not dominate.

## Proposed SFT-v1 Build Sequence

1. Run `src/data/build_lingoqa_sft_v1_plan.py` to generate the planning CSV and summary.
2. Review the 24 currently unreviewed `positive_grounding` rows first.
3. Rewrite and second-review the 7 fix rows.
4. Select at most 10-15 anti-hallucination rows for ordinary SFT; move the rest to preference planning.
5. Mine additional positive/spatial examples from a clean non-reporting subset until the target mix is reachable.
6. Build `data/processed/lingoqa_sft_v1_*.jsonl` only after the above review is complete.
7. Run a dry-run encode check before training, then a short LoRA smoke training.

## Training Readiness Gate

Do not train SFT-v1 until these conditions are met:

- At least 30 positive visual-grounding rows are reviewed or newly mined.
- At least 20 spatial/reasoning correction rows are reviewed or rewritten.
- Anti-hallucination rows are capped at 10-15 or moved to preference.
- The 7 fix rows are rewritten and second-reviewed.
- The evaluation split used for reporting is not the same split used for training data construction, or the report clearly marks the contamination.

## Expected Next Training Command Template

After a final JSONL is built, use the same smoke-training entry point with a new output directory. Do not reuse the SFT-v0 adapter as the base adapter.

```bash
cd /root/autodl-tmp/DriveMind-VL
source /root/miniconda3/bin/activate drivemind-vl

python src/train_qwen25vl_lora.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --train_file data/processed/lingoqa_sft_v1_balanced_50_80.jsonl \
  --output_dir outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v1_balanced_smoke \
  --max_samples 80 \
  --epochs 1 \
  --gradient_accumulation_steps 4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --learning_rate 3e-5 \
  --use_all_images \
  --max_images 3 \
  --frame_strategy first_middle_last \
  --prompt_variant spatial \
  --max_pixels 200704 \
  --bf16 \
  --gradient_checkpointing
```

Start with `LR=3e-5` rather than increasing the SFT-v0 setup, because the previous adapter already showed control inflation and normal F1 regression.
