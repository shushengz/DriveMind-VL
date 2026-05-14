#!/usr/bin/env bash
set -euo pipefail
python src/data/convert_to_llamafactory.py --input data/processed/drivemind_seed.jsonl --output data/processed/drivemind_llamafactory_sft.json

