# LingoQA Integration Plan

## Why LingoQA

IntelliCockpitBench is still the best aligned cockpit benchmark, but the public GitHub copy available in this workspace only exposes 7 usable examples. LingoQA provides a more practical next external VQA source:

- official benchmark repository;
- evaluation split: 100 videos, 1000 QA pairs;
- expected benchmark predictions: 500 answers;
- Lingo-Judge metric for answer truthfulness.

## Scope

LingoQA is front-view driving video VQA, not an intelligent-cockpit dataset. In DriveMind-VL it should be used for:

- `external_vqa`;
- visual grounding;
- counting/object/weather/spatial reasoning;
- normal vs text-only/wrong-video/blank-video or extracted-frame controls.

It should not be used as evidence for cabin services or vehicle-control tools.

## Current Adapter

Implemented:

```bash
ACCEPT_LINGOQA_TERMS=1 bash scripts/22_download_lingoqa_metadata.sh
```

```bash
TARGET_SIZE=100 bash scripts/21_prepare_lingoqa_subset.sh \
  data/external/lingoqa/evaluation.parquet
```

After extracting official evaluation frames:

```bash
IMAGE_ROOT=data/external/lingoqa \
TARGET_SIZE=100 bash scripts/21_prepare_lingoqa_subset.sh \
  data/external/lingoqa/val.parquet
```

```bash
bash scripts/23_run_lingoqa_dry_eval.sh
```

Supported input formats:

- `.parquet`
- `.csv`
- `.json`
- `.jsonl`

The adapter:

1. reads `question_id`, `segment_id`, `question`, `answer`;
2. groups multiple references per unique question;
3. balances by external VQA capability;
4. writes DriveMind `external_vqa` JSONL;
5. records optional `video` paths based on `segment_id`;
6. records optional extracted LingoQA image paths and uses the first key frame as `image`.

The download helper is metadata-only. It does not download videos, training data, checkpoints, or any Qwen files.

## Data Size Policy

50-100 samples are enough for:

- adapter smoke test;
- first failure taxonomy;
- debugging visual-ablation scripts;
- deciding whether the benchmark is usable.

They are not enough for a final claim. The next levels should be:

- 100 samples: first meaningful engineering report;
- 100 samples with videos or extracted frames: first visual-grounding report;
- 500 official LingoQA benchmark questions: main external VQA score;
- 1k+ or full official split: stronger research-style conclusion.

## Important Limitation

LingoQA is video VQA. Metadata-only conversion can support text-only and schema tests, but real visual grounding requires local videos or extracted frames. The Qwen inference script now supports optional `video` paths when local files exist.

## Next Commands

After obtaining the official LingoQA evaluation annotation file:

```bash
ACCEPT_LINGOQA_TERMS=1 bash scripts/22_download_lingoqa_metadata.sh

TARGET_SIZE=100 bash scripts/21_prepare_lingoqa_subset.sh \
  data/external/lingoqa/evaluation.parquet

bash scripts/23_run_lingoqa_dry_eval.sh
```

If local videos are available:

```bash
VIDEO_ROOT=data/external/lingoqa/videos \
TARGET_SIZE=100 bash scripts/21_prepare_lingoqa_subset.sh \
  data/external/lingoqa/evaluation.parquet
```
