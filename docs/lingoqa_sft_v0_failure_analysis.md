# LingoQA SFT-v0 Failure Analysis

## Scope

This note audits why `outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v0_smoke` did not improve strict visual grounding. It is based on the existing reports, the strict visual-control JSON summaries, the SFT-v0 split assets, and the visual-control case files. No additional training was run.

Audited files:

- `docs/lingoqa_sft_v0_adapter_eval_report.md`
- `docs/lingoqa_sft_v0_build_report.md`
- `docs/lingoqa_best_strategy_visual_control_report.md`
- `docs/lingoqa_prompt_frame_ablation_report.md`
- `docs/lingoqa_bad_case_analysis.md`
- `outputs/eval_results/lingoqa_qwen25vl_3b_100_best_spatial_3frame_visual_control_summary.json`
- `outputs/eval_results/lingoqa_qwen25vl_3b_100_best_spatial_3frame_visual_control_case_summary.json`
- `outputs/eval_results/lingoqa_qwen25vl_3b_sft_v0_adapter_100_best_spatial_3frame_visual_control_summary.json`
- `outputs/eval_results/lingoqa_qwen25vl_3b_sft_v0_adapter_100_best_spatial_3frame_visual_control_case_summary.json`

## Metric Check

The base best strategy is still `prompt_variant=spatial`, `frame_strategy=first_middle_last`, `max_images=3`.

| metric | base best | SFT-v0 adapter | change |
|---|---:|---:|---:|
| normal F1 | 0.2912 | 0.2470 | -0.0442 |
| text-only F1 | 0.1448 | 0.1634 | +0.0186 |
| wrong-image F1 | 0.2095 | 0.2106 | +0.0011 |
| blank-image F1 | 0.1349 | 0.1773 | +0.0424 |
| setting-level gap | +0.0817 | +0.0364 | -0.0453 |
| per-case conservative gap | +0.0212 | -0.0412 | -0.0624 |
| positive-gap rate | 0.3100 | 0.1500 | -0.1600 |

The setting-level gap remains positive only because normal images still beat the strongest aggregate control. The stricter per-case metric reverses sign, meaning that on an average individual example the best control is now stronger than the normal visual input.

## Data Composition Problem

SFT-v0 used only 25 `keep` samples:

| role | count | share |
|---|---:|---:|
| anti_hallucination_counterfactual | 16 | 64% |
| spatial_reasoning_correction | 6 | 24% |
| positive_visual_grounding | 3 | 12% |

This is not a balanced visual-grounding SFT set. It is dominated by counterfactual or confound-driven examples. The current candidate pool actually contains 27 `positive_grounding` candidates, but only 3 were included in SFT-v0. The skew came from selecting reviewed high-risk cases first rather than building a target-aligned training mix.

The reviewed split also shows that the highest-risk cases were not clean positive SFT material:

| action | count | avg gap | avg normal F1 | avg control-max F1 |
|---|---:|---:|---:|---:|
| keep | 25 | +0.0195 | 0.4906 | 0.4712 |
| fix | 7 | -0.2382 | 0.4119 | 0.6501 |
| drop | 9 | -0.3119 | 0.2085 | 0.5204 |

The `keep` average gap is only mildly positive, and many kept rows came from `wrong_image_confound`. Those rows are useful diagnostically, but they do not automatically become good ordinary SFT targets.

## Objective Mismatch

Strict visual-control evaluation is a relative objective:

```text
normal prediction should beat text-only, wrong-image, and blank-image controls on the same case.
```

Ordinary next-token SFT on normal-image messages optimizes only:

```text
P(gold answer | normal image, prompt)
```

It does not optimize:

```text
P(gold answer | normal image) > P(gold answer | wrong image or blank image or no image)
```

Therefore, plain SFT can improve language priors, answer templates, and short generic rationales while doing nothing to suppress the same answer under controls. That is exactly what happened: text-only and blank-image F1 increased, while normal F1 and per-case gap decreased.

## Risk of Plain SFT on `wrong_image_confound`

`wrong_image_confound` means a wrong-frame control matched or beat the normal-image output. If such a sample is used as an ordinary positive SFT example, the model only sees the normal-image input paired with the gold answer. It never sees the wrong image as an explicitly rejected alternative.

The risk is concrete:

- The model can learn the question-answer prior instead of image discrimination.
- The answer may be valid for multiple similar driving scenes, so the wrong frame is not penalized.
- Anti-hallucination intent is only stored as metadata; the training loss does not read `training_warning` unless the training script changes the prompt/target.
- A high proportion of these examples can strengthen conservative or generic answers under text-only/blank-image settings.

For SFT-v1, these rows should either be capped in ordinary SFT or converted into contrastive preference/rejection samples where the normal image is preferred and the wrong/blank/no-image response is rejected or marked uncertain.

## Positive Grounding Shortage

Only 3 positive grounding rows were included. That is too small to teach the adapter that the project rewards normal-image-specific evidence. It also leaves object recognition, counting, and spatial localization underrepresented as direct visual successes.

The missed opportunity is visible in the pool:

- Candidate pool `positive_grounding`: 27 rows.
- Reviewed and kept positive rows: 3 rows.
- Unreviewed positive rows: 24 rows.

SFT-v1 should first recover these positive candidates through visual review, then mine additional positives from a clean split. The target should be at least 30-40 positive visual-grounding examples, and this target cannot be met by the current reviewed keep set.

## Spatial and Reasoning Regression

The adapter regressed most on the capabilities that matter for driving grounding:

| capability | base normal F1 | adapter normal F1 | normal delta | base gap | adapter gap |
|---|---:|---:|---:|---:|---:|
| spatial_localization | 0.3134 | 0.1468 | -0.1666 | -0.0195 | -0.0798 |
| reasoning_world_knowledge | 0.2034 | 0.1156 | -0.0878 | -0.0553 | -0.0682 |
| object_recognition | 0.3791 | 0.2959 | -0.0832 | +0.0927 | -0.0508 |
| counting | 0.1738 | 0.2881 | +0.1143 | +0.0902 | +0.0451 |

Likely causes:

1. Only 6 kept rows targeted spatial/reasoning correction directly.
2. The spatial correction answers were not consistently rewritten into evidence-rich targets with lane, crosswalk, traffic-light, pedestrian, and relative-position details.
3. The LoRA adapter was trained on very few optimizer steps, so small data imbalance can produce style and prior shifts instead of robust visual behavior.
4. The training script optimizes the assistant JSON answer only; it does not add a contrastive loss for controls or a reward for visual dependency.
5. The base model plus prompt/frame strategy was already the best known inference setup; a tiny skewed adapter can easily disturb that behavior.

## Per-Case Gap Movement

Joining the base and adapter visual-control case files gives this movement:

| metric | count/value |
|---|---:|
| joined cases | 100 |
| average gap delta | -0.0624 |
| average normal F1 delta | -0.0443 |
| average control-max F1 delta | +0.0182 |
| cases with improved gap | 20 |
| cases with worse gap | 44 |
| base positive-gap cases | 31 |
| adapter positive-gap cases | 15 |
| lost positive-gap cases | 20 |
| gained positive-gap cases | 4 |

This confirms the failure mode: the adapter did not merely fail to improve normal F1. It also raised control strength and destroyed many previously positive visual-dependency cases.

## Conclusion

Do not continue training the current SFT-v0 adapter. It is useful as a pipeline smoke test only.

Recommended decision:

1. Keep the base Qwen2.5-VL-3B with `spatial + first_middle_last + 3 images` as the current best inference baseline.
2. Do not increase epochs on the same 25 rows.
3. Rebuild SFT-v1 around a target-aligned mix: many more positive visual-grounding rows, more rewritten spatial/reasoning rows, and a capped anti-hallucination subset.
4. Move `wrong_image_confound` primarily to a preference/rejection objective rather than ordinary SFT.
5. Rewrite the 7 fix rows before promotion.
6. Evaluate future adapters with the same strict controls, plus a clean holdout split if any training data is mined from the current evaluation pool.
