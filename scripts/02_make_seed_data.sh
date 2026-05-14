#!/usr/bin/env bash
set -euo pipefail
python src/data/build_seed_data.py --output data/processed/drivemind_seed.jsonl --image_dir data/images --num_samples 100

