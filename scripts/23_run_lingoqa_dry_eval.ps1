$ErrorActionPreference = "Stop"

$inputPath = if ($env:INPUT) { $env:INPUT } else { "data/processed/drivemind_lingoqa_eval_subset.jsonl" }
$predictions = if ($env:PREDICTIONS) { $env:PREDICTIONS } else { "outputs/eval_results/lingoqa_dry_predictions.jsonl" }
$metrics = if ($env:METRICS) { $env:METRICS } else { "outputs/eval_results/lingoqa_dry_metrics.json" }
$breakdown = if ($env:BREAKDOWN) { $env:BREAKDOWN } else { "outputs/eval_results/lingoqa_dry_breakdown.json" }
$badCases = if ($env:BAD_CASES) { $env:BAD_CASES } else { "outputs/cases/lingoqa_dry_bad_cases.jsonl" }

python src/eval/base_infer_dryrun.py `
  --input $inputPath `
  --output $predictions `
  --dry_run

python src/eval/run_all_eval.py `
  --predictions $predictions `
  --output $metrics `
  --bad_cases_output $badCases

python src/eval/eval_external_vqa_breakdown.py `
  --predictions $predictions `
  --output $breakdown
