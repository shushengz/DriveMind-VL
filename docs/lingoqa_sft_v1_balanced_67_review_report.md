# LingoQA SFT-v1 Balanced-67 Review Report

## Scope

This revision starts from the conservative alpha40 reviewed set, mines clean candidates from the 500-sample LingoQA run while excluding the original 100-control ids, manually reviews contact sheets, and adds 27 selected rows. No training was run.

## Outputs

- JSONL: `data/processed/lingoqa_sft_v1_balanced_67.jsonl`
- Reviewed plan: `outputs/cases/lingoqa_sft_v1_reviewed_plan_balanced_67.csv`
- Summary: `outputs/eval_results/lingoqa_sft_v1_balanced_67_summary.json`

## Distribution

```json
{
  "total_samples": 67,
  "base_alpha40_samples": 40,
  "clean500_added_samples": 27,
  "by_sft_v1_role": {
    "positive_visual_grounding": 30,
    "spatial_reasoning_correction": 20,
    "anti_hallucination_counterfactual": 10,
    "high_quality_fix_rewritten": 7
  },
  "by_capability": {
    "object_recognition": 8,
    "other": 10,
    "counting": 8,
    "reasoning_world_knowledge": 10,
    "spatial_localization": 31
  },
  "by_source_status": {
    "v0_keep_rewritten": 9,
    "unreviewed_positive_manual_review": 9,
    "all_settings_failed_manual_promote": 6,
    "v0_keep_anti_capped_rewritten": 10,
    "v0_fix_rewritten": 6,
    "clean500_excludes_100_control_manual_review": 27
  },
  "target_ratio_status": {
    "total_50_80": true,
    "positive_visual_grounding_30_40": true,
    "spatial_reasoning_correction_20_30": true,
    "anti_hallucination_counterfactual_10_15": true,
    "high_quality_fix_rewritten_7_15": true
  },
  "clean500_added_ids": [
    "lingoqa_eval_000168",
    "lingoqa_eval_000270",
    "lingoqa_eval_000355",
    "lingoqa_eval_000410",
    "lingoqa_eval_000442",
    "lingoqa_eval_000461",
    "lingoqa_eval_000429",
    "lingoqa_eval_000109",
    "lingoqa_eval_000141",
    "lingoqa_eval_000161",
    "lingoqa_eval_000103",
    "lingoqa_eval_000387",
    "lingoqa_eval_000271",
    "lingoqa_eval_000262",
    "lingoqa_eval_000409",
    "lingoqa_eval_000107",
    "lingoqa_eval_000196",
    "lingoqa_eval_000383",
    "lingoqa_eval_000301",
    "lingoqa_eval_000451",
    "lingoqa_eval_000399",
    "lingoqa_eval_000147",
    "lingoqa_eval_000155",
    "lingoqa_eval_000352",
    "lingoqa_eval_000376",
    "lingoqa_eval_000497",
    "lingoqa_eval_000493"
  ],
  "eval_contamination_risk": {
    "lingoqa_100_control": "high for alpha40 rows; do not report on original 100 as clean holdout",
    "clean500_added": "lower for original 100-control evaluation because ids are excluded from that set, but still from LingoQA eval distribution; use fresh holdout for final claim"
  },
  "recommend_smoke_training": true,
  "recommendation_reason": "Meets 50-80 total and role-balance gates after manual contact-sheet review; use only as smoke training, then evaluate on a fresh/unseen control split."
}
```

## Training Decision

The dataset now meets the planned role-count gates for a smoke training run. The original 100-control set remains contaminated by alpha40 rows, so any post-training claim must use a fresh strict visual-control split or clearly mark the 100-control result as diagnostic only.
