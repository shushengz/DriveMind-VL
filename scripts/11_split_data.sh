#!/usr/bin/env bash
set -euo pipefail
python src/data/split_dataset.py \
  --input data/processed/drivemind_seed.jsonl \
  --train_output data/processed/drivemind_train.jsonl \
  --eval_output data/processed/drivemind_eval.jsonl \
  --eval_size 20 \
  --seed 42
python src/data/convert_to_llamafactory.py \
  --input data/processed/drivemind_train.jsonl \
  --output data/processed/drivemind_train_llamafactory.json
python src/data/convert_to_llamafactory.py \
  --input data/processed/drivemind_eval.jsonl \
  --output data/processed/drivemind_eval_llamafactory.json

