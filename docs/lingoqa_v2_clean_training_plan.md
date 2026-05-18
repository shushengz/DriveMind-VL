# LingoQA v2 Clean Training Plan

## Current Bottleneck

The earlier 25/57/77-sample experiments mostly improved answer style. They did
not reliably improve strict visual grounding because normal-image answers and
wrong-image controls rose together. The next round must stop using the old
100-case control set as the final target and move to segment-level clean splits.

## Prepared Data Pipeline

1. Split the 500 DriveMind-formatted LingoQA examples by video `segment_id`.
   The default split is 60 train segments, 20 dev segments, and 20 test
   segments, about 300/100/100 examples.
2. Build wrong-frame and blank-frame controls for each split.
3. Build SFT-v2 visual data from the train split only. Each question can use
   all reference answers as variants, and rare capabilities are repeated.
4. Build control-heavy preference pairs from the train split only. The default
   pair mix strongly favors controls:
   - text-only refusal over hallucinated answer
   - blank-image refusal over hallucinated answer
   - wrong-image refusal over hallucinated answer
   - a smaller subset of normal answer over refusal

## Training Strategy

Stage 1: SFT-v2 visual scale

- Train from the base Qwen2.5-VL-3B model.
- Use 5 frames with uniform frame selection.
- Use a higher image budget than smoke experiments.
- Use LoRA rank 32 on attention and MLP modules.

Stage 2: preference-v3 control-heavy

- Continue from the SFT-v2 adapter.
- Use reference-free DPO-style loss so no second model copy is required.
- Keep beta low and make the data control-heavy. The goal is to reduce
  wrong/blank/text-only controls without collapsing normal performance.

## Success Criteria

Evaluate on clean dev first, then clean test once. A useful final result should:

- preserve or improve normal F1 over the base model;
- keep wrong-image F1 below the normal-image score by a larger margin;
- improve per-case conservative visual dependency gap;
- avoid using the old contaminated 100-case set as the final claim.
