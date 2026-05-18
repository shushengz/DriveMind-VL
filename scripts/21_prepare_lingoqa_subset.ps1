$ErrorActionPreference = "Stop"

$inputPath = if ($args.Count -ge 1) { $args[0] } elseif ($env:INPUT) { $env:INPUT } else { "data/external/lingoqa/evaluation.parquet" }
$targetSize = if ($env:TARGET_SIZE) { $env:TARGET_SIZE } else { "100" }
$outputPath = if ($env:OUTPUT) { $env:OUTPUT } else { "data/processed/drivemind_lingoqa_eval_subset.jsonl" }
$imageRoot = if ($env:IMAGE_ROOT) { $env:IMAGE_ROOT } else { $null }
$videoRoot = if ($env:VIDEO_ROOT) { $env:VIDEO_ROOT } else { $null }

$prepareArgs = @(
  "src/data/prepare_lingoqa_subset.py",
  "--input", $inputPath,
  "--target_size", $targetSize,
  "--output", $outputPath,
  "--manifest_output", "outputs/eval_results/lingoqa_eval_subset_manifest.json"
)
if ($imageRoot) {
  $prepareArgs += @("--image_root", $imageRoot)
}
if ($videoRoot) {
  $prepareArgs += @("--video_root", $videoRoot)
}

python @prepareArgs

python src/data/validate_dataset.py `
  --input $outputPath
