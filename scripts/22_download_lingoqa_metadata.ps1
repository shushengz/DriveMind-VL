$ErrorActionPreference = "Stop"

$outputPath = if ($env:OUTPUT) { $env:OUTPUT } else { "data/external/lingoqa/evaluation.parquet" }

if ($env:ACCEPT_LINGOQA_TERMS -ne "1") {
  Write-Host "Refusing to download until ACCEPT_LINGOQA_TERMS=1 is set."
  Write-Host "Review upstream repository and license first: https://github.com/wayveai/LingoQA"
  exit 2
}

python src/data/download_lingoqa_metadata.py `
  --output $outputPath `
  --accept_terms
