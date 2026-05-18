# IntelliCockpitBench Data Scaling Plan

## Current Blocker

The IntelliCockpitBench paper and README describe a full benchmark with 16,154 queries and 7,622 images. However, the public GitHub repository currently included in this workspace only contains a 7-sample `Evaluation/data/jsonl/english_test.jsonl` and 7 images.

This means the engineering pipeline is ready, but a 50-100 sample external eval requires access to the full JSONL and image directory from the benchmark authors or an official release location.

## What Is Now Implemented

The repository now supports scaling as soon as full metadata is available:

```bash
bash scripts/20_build_intelli_eval_subset.sh \
  /path/to/full/english_test.jsonl \
  /path/to/full/images
```

Environment knobs:

```bash
TARGET_SIZE=100
SEED=42
OUTPUT=data/processed/drivemind_intelli_eval_subset.jsonl
```

The script performs:

1. metadata and image availability audit;
2. balanced sampling by external VQA capability;
3. DriveMind-Instruct conversion;
4. dataset validation;
5. manifest generation.

## Capability Buckets

The subset builder balances samples across:

- `counting`
- `object_recognition`
- `spatial_localization`
- `weather_road_condition`
- `scene_completeness`
- `reasoning_world_knowledge`
- `other`

## Expected Artifacts

Generated files are runtime artifacts and should not be committed:

- `data/processed/drivemind_intelli_eval_subset.jsonl`
- `outputs/eval_results/intelli_dataset_audit.json`
- `outputs/eval_results/intelli_eval_subset_manifest.json`

## Data Acquisition Options

1. Check whether the benchmark authors provide a full release outside the GitHub sample repository.
2. Contact the authors with the ACL paper citation and request academic access to the full JSONL/image package.
3. If full access is delayed, build an interim mixed eval from other public driving/cockpit datasets, but clearly label it as `DriveMind external eval v0`, not full IntelliCockpitBench.

## Next Run After Data Is Available

```bash
TARGET_SIZE=100 bash scripts/20_build_intelli_eval_subset.sh \
  /path/to/full/english_test.jsonl \
  /path/to/full/images

bash scripts/18_run_intelli_visual_ablation.sh \
  /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  /path/to/IntelliCockpitBench
```

For non-standard paths, run the Python entrypoints directly.
