# LingoQA SFT-v1 Final Visual57 Report

This final ordinary-SFT dataset removes `anti_hallucination_counterfactual` rows from the balanced-67 set because plain SFT on confound rows can strengthen text-only or blank-image priors.

## Summary

```json
{
  "total_samples": 57,
  "input_jsonl": "data/processed/lingoqa_sft_v1_balanced_67.jsonl",
  "output_jsonl": "data/processed/lingoqa_sft_v1_final_visual57.jsonl",
  "excluded_plain_sft_roles": [
    "anti_hallucination_counterfactual"
  ],
  "by_sft_v1_role": {
    "positive_visual_grounding": 30,
    "spatial_reasoning_correction": 20,
    "high_quality_fix_rewritten": 7
  },
  "by_capability": {
    "object_recognition": 6,
    "other": 5,
    "counting": 7,
    "reasoning_world_knowledge": 9,
    "spatial_localization": 30
  },
  "by_source_status": {
    "v0_keep_rewritten": 9,
    "unreviewed_positive_manual_review": 9,
    "all_settings_failed_manual_promote": 6,
    "v0_fix_rewritten": 6,
    "clean500_excludes_100_control_manual_review": 27
  },
  "target_ratio_status": {
    "total_50_80": true,
    "positive_visual_grounding_at_least_30": true,
    "spatial_reasoning_correction_at_least_20": true,
    "high_quality_fix_rewritten_at_least_7": true,
    "anti_hallucination_plain_sft_zero": true
  },
  "recommend_smoke_training": true,
  "recommendation_reason": "Use this visual-only dataset for the next ordinary SFT run; keep anti/counterfactual samples for preference or rejection training, not plain SFT.",
  "training_hparams_recommendation": {
    "epochs": 1,
    "learning_rate": 2e-05,
    "lora_rank": 8,
    "lora_alpha": 16,
    "gradient_accumulation_steps": 4,
    "processor": "slow use_fast=False default"
  },
  "eval_warning": "The original 100-control split is contaminated by some source rows. Use a fresh strict visual-control split for final reporting."
}
```

## Decision

Use `data/processed/lingoqa_sft_v1_final_visual57.jsonl` for the next smoke training. Keep the excluded anti/counterfactual rows for DPO/ORPO/RFT-lite or explicit rejection/uncertainty training.
