# LingoQA 500-Question Multi-Frame Report

## Setup

- Dataset: LingoQA official evaluation split
- Converted records: 500 unique questions
- Images: 500 extracted jpg frames, 5 frames per driving scenario
- Model: Qwen2.5-VL-3B-Instruct
- Hardware: RTX 4080 SUPER 32GB server
- Training: none
- Evaluation metric: DriveMind external answer token-F1 plus schema/reward metrics

## Data Distribution

```json
{
  "source_rows": 1000,
  "unique_questions": 500,
  "selected": {
    "total": 500,
    "capability_counts": {
      "counting": 60,
      "object_recognition": 42,
      "other": 108,
      "reasoning_world_knowledge": 17,
      "spatial_localization": 271,
      "weather_road_condition": 2
    }
  }
}
```

The official evaluation subset is heavily weighted toward spatial localization. Capability-level reporting is therefore more useful than the overall score alone.

## Main Results

| Setting | Images Per Sample | JSON Validity | Schema Completeness | Answer F1 | Lingo-Judge | Pass Rate | Avg Reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| text-only | 0 | 1.0000 | 1.0000 | 0.1781 | 0.4180 | 0.2460 | 0.4350 |
| single-frame | 1 | 1.0000 | 1.0000 | 0.2072 | 0.5020 | 0.3280 | 0.4405 |
| 5-frame | 5 | 1.0000 | 1.0000 | 0.2270 | 0.5580 | 0.3280 | 0.4452 |

Visual gains:

- single-frame vs text-only: `+0.0291` answer F1
- 5-frame vs text-only: `+0.0489` answer F1
- 5-frame vs single-frame: `+0.0198` answer F1

Lingo-Judge gains:

- single-frame vs text-only: `+0.0840`
- 5-frame vs text-only: `+0.1400`
- 5-frame vs single-frame: `+0.0560`

## Capability Breakdown

| Capability | Count | Text-only F1 | Single-frame F1 | 5-frame F1 | 5-frame vs Text |
|---|---:|---:|---:|---:|---:|
| counting | 60 | 0.1763 | 0.1635 | 0.1275 | -0.0488 |
| object_recognition | 42 | 0.1152 | 0.2993 | 0.3429 | +0.2277 |
| other | 108 | 0.3021 | 0.3083 | 0.3371 | +0.0350 |
| reasoning_world_knowledge | 17 | 0.1691 | 0.1978 | 0.1290 | -0.0401 |
| spatial_localization | 271 | 0.1406 | 0.1640 | 0.1946 | +0.0540 |
| weather_road_condition | 2 | 0.0000 | 0.0513 | 0.0513 | +0.0513 |

## Interpretation

The model is very stable at following the DriveMind JSON schema: all three settings reached `1.0` JSON validity and schema completeness. This validates the current inference prompt and parser path.

The visual signal is real but modest at 500-question scale. The strongest gain is object recognition, where 5-frame F1 improves from `0.1152` text-only to `0.3429`. Spatial localization also improves, from `0.1406` to `0.1946`, but remains weak despite being the dominant category.

5-frame input improves overall F1 over single-frame, but not dramatically. Counting and reasoning-world-knowledge become worse with 5 frames, which suggests the current prompt and frame packing may add visual distraction or temporal ambiguity. This is an actionable model/data problem rather than only an engineering problem.

## Current Project Conclusion

DriveMind-VL now has a reproducible external benchmark result beyond synthetic data:

1. It can run Qwen2.5-VL-3B on a 500-question public driving VQA benchmark.
2. It can compare text-only, single-frame, and multi-frame settings.
3. It shows visual grounding gains on real data, especially object recognition and spatial localization.
4. It also exposes weak points: counting, temporal reasoning, spatial precision, and answer semantic alignment.

This is enough for a credible interim project conclusion. Lingo-Judge confirms the same ordering as token-F1 and shows a larger semantic gain from multi-frame input. It is still not a final model-training claim because no model adaptation was performed and wrong-frame/blank-frame controls have only been run at 100-sample scale.

## Next Step

Next, improve multi-frame prompting and run targeted bad-case analysis for counting, spatial localization, and reasoning-world-knowledge failures.
