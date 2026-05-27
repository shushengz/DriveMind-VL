# Resume Readiness Review

## Current Status

DriveMind-VL is close to being a credible interview project, but it should be framed as an evaluation-and-alignment project rather than a pure fine-tuning project.

Strong parts:

- It has a clear autonomous-driving multimodal scenario.
- It uses a modern VLM backbone with LoRA/SFT/DPO-style preference tuning.
- It has visual-control evaluation: normal, text-only, wrong-image, blank-image.
- It has case-level error attribution instead of only aggregate scores.
- It now has a path to public benchmarks: DriveLM and NuScenes-QA.

Weak parts before this iteration:

- The main evidence came from a small LingoQA subset, which is not enough for resume-level credibility.
- Previous tuning runs improved some control behavior but sometimes hurt normal visual QA.
- The project lacked a clean public benchmark runbook and data audit step.
- The story was not yet tied tightly enough to "large model evaluation" and "multimodal alignment."

## What Makes It Resume-Worthy

The project becomes resume-worthy if the final report contains three layers:

1. Public benchmark layer:
   - DriveLM subset and/or NuScenes-QA subset.
   - Compare base Qwen2.5-VL, SFT-v2, v5c, and v7.
   - Report normal performance and visual dependency controls.

2. Diagnostic layer:
   - Capability breakdown: spatial localization, object recognition, counting, reasoning.
   - Case-level regression analysis.
   - Explicit failure modes and data fixes.

3. Alignment layer:
   - Show why v5/v6 failed.
   - Show v7 data gates: high-quality normal prediction, low-quality ablated prediction, minimum contrast gap.
   - Emphasize preserving normal visual understanding while improving visual dependency.

This directly maps to the JD:

- Multimodal feature fusion/alignment/representation: visual dependency controls and ablated visual evidence.
- Model lightweight deployment: Qwen2.5-VL-3B + LoRA adapters; later package inference with adapter switching.
- Large model evaluation: public benchmarks, visual-control protocol, capability breakdown, regression reports.

## Remaining Gaps

High priority:

- Run DriveLM or NuScenes-QA public subset on GPU.
- Add at least one public benchmark table to the README or final report.
- Train v7 only after public benchmark baseline is ready.

Medium priority:

- Improve metric quality beyond token F1. A Lingo-Judge or GPT-style semantic evaluator would make spatial and planning answers more reliable.
- Add latency/VRAM numbers for base model and LoRA adapter inference.
- Add a compact demo script that loads one adapter and runs one image sequence.

Low priority:

- More training recipes.
- Larger internal LingoQA sweeps.
- Complex model architecture changes before the evaluation story is stable.

## Go/No-Go For Resume

Ready to put on resume when:

- A public benchmark subset runs end-to-end.
- The table includes at least base model vs SFT-v2 vs one preference adapter.
- There is a written failure analysis explaining v5/v6 and the v7 fix.
- The result is honest: even if v7 is not better, the project can still be framed as robust evaluation and alignment diagnosis.

Not ready if:

- Only LingoQA internal numbers are shown.
- Only one aggregate F1 is shown.
- The project claims "improved VLM" while normal visual QA regresses.

## Recommended Resume Bullet

Built DriveMind-VL, an autonomous-driving VLM evaluation and alignment pipeline on Qwen2.5-VL-3B, integrating LingoQA plus public driving QA benchmarks and designing normal/text-only/wrong-image/blank-image controls to measure visual dependency beyond language-prior guessing.

Designed a conservative DPO data-mining recipe with capability-aware SFT anchors and contrast-based control filtering, reducing the risk of answer suppression while preserving normal visual QA; produced case-level regression reports for spatial localization, counting, object recognition, and reasoning failures.

## Interview Talking Points

- The first tuning attempts failed because control refusal pairs were noisy and over-optimized refusal behavior.
- The key metric is not just normal F1; it is normal F1 together with the visual dependency gap and per-case gap.
- Public benchmarks are necessary because a private small set cannot prove generalization.
- The most valuable engineering work is the evaluation protocol and error attribution loop, not just calling a training script.
