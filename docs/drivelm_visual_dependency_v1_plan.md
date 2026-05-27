# DriveLM Visual Dependency V1

The current DriveLM-SFT adapter improves heldout normal F1, but visual-control results show it still relies on language priors. V1 changes the next training target from answer-style SFT to visual-dependency preference learning.

## No-GPU Outputs

- Builder: `src/data/build_drivelm_visual_preference_v1.py`
- Data script: `scripts/64_build_drivelm_visual_pref_v1.sh`
- Diagnostic preference data: `data/processed/drivelm_visual_pref_v1_dev_diagnostic.jsonl`
- Diagnostic audit: `docs/drivelm_visual_pref_v1_dev_diagnostic.md`

The diagnostic file is built from heldout dev predictions and is for format validation and analysis only. Final training data should be rebuilt from train-scene predictions after GPU inference.

## Pair Recipe

- `normal_anchor_gold_over_refusal`: preserves DriveLM answer style on normal visual input.
- `normal_repair_gold_over_bad_prediction`: repairs low-F1 normal-image answers.
- `control_refusal_over_prediction`: teaches text-only, wrong-image, and blank-image inputs to refuse scene-specific answers.

The builder prioritizes these failure types:

- `blank_beats_normal`
- `text_beats_normal`
- `wrong_image_beats_normal`
- `wrong_image_invariant_low`
- `normal_low`

## GPU Follow-Up

1. Run visual-control inference on `data/processed/drivelm_train_scene.jsonl` with the SFT-1200 adapter.
2. Rebuild preference data with `OUTPUT=data/processed/drivelm_visual_pref_v1_train.jsonl`.
3. Train with `scripts/65_train_drivelm_visual_pref_v1.sh`.
4. Evaluate with `scripts/66_eval_drivelm_visual_pref_v1.sh`.
5. Compare with `scripts/67_compare_drivelm_visual_pref_v1.sh`.

## Success Criteria

- Normal F1 stays at or above `0.35`.
- Blank-image F1 drops below normal F1.
- Text-only F1 drops below normal F1.
- Case-level visual dependency gap improves from `-0.1576` to better than `-0.10`.
