# IntelliCockpitBench Smoke Report

This report records the first external benchmark smoke test for DriveMind-VL.

## Scope

This is a 7-sample smoke test using the small `Evaluation/data/jsonl/english_test.jsonl` sample and images included in the public IntelliCockpitBench repository. It is not a full benchmark result.

## Setup

- Server path: `/root/autodl-tmp/DriveMind-VL`
- Model: `/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct`
- Benchmark source: IntelliCockpitBench public repository sample
- Samples: 7
- Task mapping: all 7 samples mapped to `external_vqa`
- Precision: bf16
- Max new tokens: 160

## Commands

```bash
bash scripts/17_run_intelli_qwen25vl_3b_eval.sh \
  /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  third_party/IntelliCockpitBench
```

## Metrics

| Metric | Value |
|---|---:|
| json_validity | 1.0000 |
| schema_completeness | 1.0000 |
| external_answer_f1 | 0.2070 |
| avg_reward | 0.6750 |
| error_analysis failed_cases | 4 / 7 |

The model is strong at producing valid DriveMind JSON under the new `external_vqa` schema, but weak on fine-grained visual answer agreement with IntelliCockpitBench references.

## Representative Errors

- Vehicle brand recognition: predicted `toyota`, reference says `Volkswagen Passat`.
- Quantitative scene reasoning: described a foggy mountain-road scene, but missed the reference answer `about two cars ahead`.
- Description completeness: answered only `trees` while the reference also includes shrubs and streetlights.
- Description completeness: answered only `a bus` while the reference includes bus, pedestrian, black car, trees, and buildings.

## Conclusion

This external smoke test gives a more rigorous project conclusion than the synthetic seed eval:

1. Qwen2.5-VL-3B can follow the DriveMind output schema on public real images.
2. Synthetic high scores do not transfer to exact real VQA grounding.
3. The next bottleneck is not JSON formatting; it is visual grounding, object detail, counting, fine-grained recognition, and complete scene description.
4. The next data work should add external VQA supervision and external answer scoring before another LoRA round.
