# LingoQA SFT-v1 Control-Aware 77 Report

```json
{
  "total_samples": 77,
  "normal_visual_samples": 57,
  "control_rejection_samples": 20,
  "by_role": {
    "positive_visual_grounding": 30,
    "spatial_reasoning_correction": 20,
    "high_quality_fix_rewritten": 7,
    "control_rejection_text_only": 5,
    "control_rejection_blank_image": 5,
    "control_rejection_wrong_image": 10
  },
  "by_train_input_mode": {
    "normal": 57,
    "text_only": 5,
    "blank_image": 5,
    "wrong_image": 10
  },
  "by_capability": {
    "object_recognition": 8,
    "other": 10,
    "counting": 10,
    "reasoning_world_knowledge": 14,
    "spatial_localization": 35
  },
  "purpose": "Ordinary SFT with explicit rejection targets for text-only, blank-image, and wrong-image controls to approximate strict visual dependency training.",
  "recommend_training": true,
  "recommended_hparams": {
    "epochs": 1,
    "learning_rate": 1e-05,
    "lora_rank": 8,
    "lora_alpha": 16,
    "gradient_accumulation_steps": 4
  },
  "risk": "May increase refusal/uncertainty if overtrained; keep LR lower than visual57 and run strict controls immediately after training."
}
```
