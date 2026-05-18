# LingoQA SFT-v0 Build Report

## Input Files

- Review CSV: `outputs\cases\lingoqa_sft_candidate_review_agent_checked.csv`
- Candidate JSONL: `data\processed\lingoqa_sft_candidates_needs_review.jsonl`

## Output Files

- Keep JSONL: `data\processed\lingoqa_sft_v0_keep_25.jsonl`
- Need-fix CSV: `outputs\cases\lingoqa_sft_v0_need_fix_7.csv`
- Drop CSV: `outputs\cases\lingoqa_sft_v0_drop_9.csv`
- Report: `docs\lingoqa_sft_v0_build_report.md`
- Chinese note: `Note\2026-05-17_LingoQA_SFT_v0数据分流.md`

## Total Counts

- keep: 25
- fix: 7
- drop: 9

## Distribution By Candidate Type

| key | keep | fix | drop | total |
|---|---:|---:|---:|---:|
| hard_negative_spatial_reasoning | 6 | 4 | 1 | 11 |
| language_prior_confound | 0 | 0 | 4 | 4 |
| positive_grounding | 3 | 0 | 0 | 3 |
| wrong_image_confound | 16 | 3 | 4 | 23 |

## Keep Role Distribution

| sft_v0_role | count |
|---|---:|
| anti_hallucination_counterfactual | 16 |
| positive_visual_grounding | 3 |
| spatial_reasoning_correction | 6 |

## wrong_image_confound Keep

- count: 16
- training_warning: `wrong_image_confound_keep_use_as_anti_hallucination_or_visual_evidence_sample`

## Quality Check Lists

### unmatched_review_ids

- None

### invalid_action_ids

- None

### empty_note_ids

- None

## Next Steps

- Use only the reviewed keep JSONL for the first small SFT-v0 candidate pool.
- Manually rewrite the fix rows before promoting them into any training JSONL.
- Keep wrong_image_confound rows tagged as anti-hallucination or visual-evidence samples, not ordinary positive samples.
- Re-run this builder after additional review batches are completed, using new output filenames.
