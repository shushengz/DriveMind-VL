# LingoQA 100-Sample Evaluation Report

## Setup

- Server: SeetaCloud mirror instance under `/root/autodl-tmp/DriveMind-VL`
- Model: local `/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct`
- Dataset: LingoQA official evaluation folder
- Media used: extracted key frames from `images.zip`
- Sample size: 100 DriveMind `external_vqa` records sampled from 500 unique LingoQA questions
- Inference mode: single first frame per question, no training

## Data

Official evaluation files obtained:

- `val.parquet`: 1000 rows, 500 unique questions
- `images.zip`: 500 jpg frames after extraction

Converted subset:

```json
{
  "source_rows": 1000,
  "unique_questions": 500,
  "selected": {
    "total": 100,
    "capability_counts": {
      "counting": 21,
      "object_recognition": 20,
      "other": 20,
      "reasoning_world_knowledge": 17,
      "spatial_localization": 20,
      "weather_road_condition": 2
    }
  }
}
```

## Results

| Setting | JSON Validity | Schema Completeness | External Answer F1 | Pass Rate | Avg Reward |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B + image | 1.0000 | 1.0000 | 0.2698 | 0.4700 | 0.4591 |
| Qwen2.5-VL-3B text-only | 1.0000 | 1.0000 | 0.1572 | 0.2600 | 0.4294 |

Visual gap:

- answer F1 gap: `+0.1126`
- pass-rate gap: `+0.2100`
- avg-reward gap: `+0.0297`

## Capability Breakdown

| Capability | Image F1 | Text-only F1 | Gap |
|---|---:|---:|---:|
| counting | 0.1706 | 0.0499 | +0.1207 |
| object_recognition | 0.4018 | 0.1105 | +0.2913 |
| other | 0.3137 | 0.3027 | +0.0110 |
| reasoning_world_knowledge | 0.2510 | 0.1601 | +0.0909 |
| spatial_localization | 0.2379 | 0.1842 | +0.0537 |
| weather_road_condition | 0.0286 | 0.0000 | +0.0286 |

## Interpretation

The current Qwen2.5-VL-3B baseline follows the DriveMind JSON format reliably on LingoQA. The image-vs-text-only gap shows real visual input helps, especially for object recognition and counting.

The absolute answer F1 is still low. This is expected for three reasons:

1. The current inference uses only the first key frame, while LingoQA is video VQA.
2. The metric is token-F1 against references, not the official Lingo-Judge metric.
3. The prompt is optimized for DriveMind schema control, not LingoQA answer quality.

## Next Step

Run a 500-question LingoQA evaluation using all official questions. Then add a multi-frame input path that passes all 5 extracted frames or a short video clip to Qwen2.5-VL.
