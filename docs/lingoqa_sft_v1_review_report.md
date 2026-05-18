# LingoQA SFT-v1 Review Report

## Scope

This pass rebuilt a conservative SFT-v1 reviewed plan from the locally available project files because the requested server-only v1 candidate plan was not present in the workspace and SSH authentication was unavailable during this run.

## Inputs Read

- `docs/lingoqa_sft_v0_build_report.md`
- `docs/lingoqa_sft_v0_adapter_eval_report.md`
- `docs/lingoqa_best_strategy_visual_control_report.md`
- `docs/lingoqa_bad_case_analysis.md`
- `docs/lingoqa_sft_candidate_curation.md`
- `outputs/cases/lingoqa_sft_candidate_review_agent_checked.csv`
- `outputs/cases/lingoqa_sft_v0_need_fix_7.csv`
- `outputs/cases/lingoqa_sft_v0_drop_9.csv`
- `data/processed/lingoqa_sft_candidates_needs_review.jsonl`
- `data/processed/lingoqa_sft_v0_keep_25.jsonl`
- `src/train_qwen25vl_lora.py`
- `src/eval/base_infer_qwen25vl.py`

Requested but missing locally:

- `docs/lingoqa_sft_v0_failure_analysis.md`
- `docs/lingoqa_sft_v1_data_plan.md`
- `outputs/cases/lingoqa_sft_v1_candidate_plan.csv`
- `outputs/eval_results/lingoqa_sft_v1_candidate_plan_summary.json`
- `src/data/build_lingoqa_sft_v1_plan.py`

## Manual Review Policy

Positive and spatial samples were promoted only when the visible frames supported the answer and the final answer/reason could be rewritten with explicit visual evidence. Missing-image rows were not promoted. Wrong-image confounds were capped and rewritten as current-image visual-evidence samples; the remainder were marked for later preference/rejection use.

## Fix-7 Decisions

| id | decision | note |
|---|---|---|
| lingoqa_eval_000031_hard_negative_spatial_reasoning | rewritten_keep | No yield vehicle visible; rewrote with zebra-crossing pedestrian caution. |
| lingoqa_eval_000032_hard_negative_spatial_reasoning | rewritten_keep | Slowing grounded in pedestrian/cyclist positions near crossing. |
| lingoqa_eval_000066_hard_negative_spatial_reasoning | rewritten_keep | Safe only with slow cautious movement after checking crossing pedestrians. |
| lingoqa_eval_000077_hard_negative_spatial_reasoning | rewritten_keep | Replaced unsupported car-distance rationale with visible 20 mph/bus-stop context. |
| lingoqa_eval_000044_wrong_image_confound | rewritten_keep | Negative traffic-light answer grounded by distinguishing crossing beacons from vehicle lights. |
| lingoqa_eval_000025_wrong_image_confound | rewritten_drop | Dropped because safe-to-proceed answer is not visually clear enough. |
| lingoqa_eval_000085_wrong_image_confound | rewritten_keep | No yield vehicle visible; green ego signal and clear ego path noted. |

## Output Distribution

- Total ordinary SFT JSONL samples: 40
- By role: `{'positive_visual_grounding': 16, 'spatial_reasoning_correction': 8, 'anti_hallucination_counterfactual': 10, 'high_quality_fix_rewritten': 6}`
- By capability: `{'object_recognition': 5, 'other': 8, 'counting': 7, 'reasoning_world_knowledge': 10, 'spatial_localization': 10}`
- By source status: `{'v0_keep_rewritten': 9, 'unreviewed_positive_manual_review': 9, 'all_settings_failed_manual_promote': 6, 'v0_keep_anti_capped_rewritten': 10, 'v0_fix_rewritten': 6}`
- Reviewed plan decisions: `{'keep_sft_v1': 40, 'drop_sft_v1': 15, 'keep_preference_later': 6, 'needs_human_review': 39}`

## Target Ratio Check

- total 50-80: False
- positive_visual_grounding 30-40: False
- spatial_reasoning_correction 20-30: False
- anti_hallucination_counterfactual 10-15: True
- high_quality_fix_rewritten 7-15: False

The target ratio is **not reached**. The set is too small and still short on positive visual grounding and spatial correction. This is preferable to padding with unreviewed or weakly grounded examples.

## Contamination Risk

Eval contamination risk is **high** because the available candidate pool is mined from the 100-case LingoQA eval/control subset. If this JSONL is trained, later evaluation should use a fresh LingoQA split or clearly report that the original 100-case control set is contaminated.

## Recommendation

Do **not** start smoke training yet. First recover the server-side v1 candidate plan, review the missing positive rows with images available, and mine additional spatial correction examples from a clean split.
