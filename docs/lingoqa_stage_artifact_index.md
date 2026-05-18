# LingoQA Stage Artifact Index

This index records the main artifacts produced during the LingoQA evaluation and curation stage.

## Evaluation Reports

| Artifact | Purpose |
|---|---|
| `docs/lingoqa_100_eval_report.md` | Initial 100-sample LingoQA dry/real evaluation summary. |
| `docs/lingoqa_500_multiframe_report.md` | 500-question text-only / single-frame / 5-frame comparison. |
| `docs/lingoqa_100_visual_control_report.md` | Strict visual controls for default 5-frame inference. |
| `docs/lingoqa_prompt_frame_ablation_report.md` | Prompt and frame-selection ablation results. |
| `docs/lingoqa_best_strategy_visual_control_report.md` | Strict controls for the best `spatial + first/middle/last 3-frame` strategy. |
| `docs/lingoqa_bad_case_analysis.md` | Bad-case taxonomy from visual-control predictions. |
| `docs/lingoqa_sft_candidate_curation.md` | Candidate SFT data curation report. |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/21_prepare_lingoqa_subset.sh` | Convert LingoQA metadata/images into DriveMind external VQA JSONL. |
| `scripts/24_build_lingoqa_500.sh` | Build the 500-question LingoQA benchmark subset. |
| `scripts/25_run_lingoqa_500_qwen_ablation.sh` | Run 500-question text/single/multi-frame ablation. |
| `scripts/26_run_lingoqa_100_visual_controls.sh` | Run default 5-frame strict visual controls. |
| `scripts/27_run_lingo_judge_smoke.sh` | Run local Lingo-Judge scoring where available. |
| `scripts/28_analyze_lingoqa_bad_cases.sh` | Build automatic bad-case taxonomy. |
| `scripts/29_run_lingoqa_prompt_frame_ablation.sh` | Run prompt/frame strategy matrix. |
| `scripts/30_run_lingoqa_best_strategy_visual_controls.sh` | Run strict controls for the best strategy. |
| `scripts/31_build_lingoqa_sft_candidates.sh` | Build reviewable SFT candidate data. |

## Data And Outputs

| Artifact | Purpose |
|---|---|
| `data/processed/drivemind_lingoqa_eval_100_control.jsonl` | 100-sample LingoQA control dataset. |
| `data/processed/drivemind_lingoqa_eval_500.jsonl` | 500-question LingoQA evaluation dataset on the server. |
| `data/processed/lingoqa_sft_candidates_needs_review.jsonl` | Candidate SFT rows requiring human review. |
| `outputs/cases/lingoqa_bad_case_taxonomy.csv` | Reviewable automatic bad-case taxonomy. |
| `outputs/cases/lingoqa_sft_candidate_review.csv` | Human-review CSV for SFT candidate approval. |
| `outputs/eval_results/lingoqa_prompt_frame_ablation_summary.json` | Prompt/frame ablation summary. |
| `outputs/eval_results/lingoqa_qwen25vl_3b_100_best_spatial_3frame_visual_control_summary.json` | Setting-level best-strategy visual-control summary. |
| `outputs/eval_results/lingoqa_qwen25vl_3b_100_best_spatial_3frame_visual_control_case_summary.json` | Per-case conservative best-strategy visual-control summary. |

## Current Best Strategy

```text
prompt_variant = spatial
frame_strategy = first_middle_last
max_images = 3
```

Best 100-sample result:

| Setting | Answer F1 |
|---|---:|
| normal 3-frame | 0.2912 |
| text-only | 0.1448 |
| wrong 3-frame | 0.2095 |
| blank 3-frame | 0.1349 |

Strict visual dependency:

```text
setting-level gap = +0.0817
per-case conservative gap = +0.0212
positive_gap_rate = 0.3100
```

## Next Decision

Do not start broad training yet. First review `outputs/cases/lingoqa_sft_candidate_review.csv`, approve or fix high-priority rows, then create a small clean SFT set for a 3B LoRA smoke experiment.
