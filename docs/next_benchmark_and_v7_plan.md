# Next Benchmark And V7 Plan

## Goal

Make DriveMind-VL resume-worthy for roles focused on multimodal feature fusion/alignment, lightweight deployment, and large-model evaluation.

The project should no longer be presented as "I fine-tuned a VLM on a small private driving QA set." The stronger framing is:

1. Build a driving-domain VLM evaluation protocol with visual dependency controls.
2. Diagnose whether a model truly uses visual evidence instead of language priors.
3. Improve alignment with conservative preference tuning while preserving normal visual QA.
4. Prepare a lightweight LoRA deployment path after the evaluation story is credible.

## Public Benchmark Priority

Primary:

- DriveLM: ECCV 2024 Oral, CVPR 2024 Autonomous Driving Challenge Driving-with-Language track. It is built around graph VQA and covers perception, prediction, planning, behavior, and motion. Source: https://github.com/OpenDriveLab/DriveLM
- NuScenes-QA: AAAI 2024 official benchmark, 34K scenes and 460K QA pairs according to the paper/repository. Source: https://github.com/qiantianwen/NuScenes-QA

Secondary:

- BDD-X: explanation-oriented driving dataset, useful for behavior explanation and "why" answers, but not the first choice for visual-control QA. Source: https://github.com/JinkyuKimUCB/BDD-X-dataset

Use DriveLM/NuScenes-QA as the resume benchmark layer. Keep LingoQA as an internal diagnostic set because our visual-control cases and failure mining are already built around it.

## What Failed In V5/V6

The issue was not simply "not enough training." The failure pattern is data objective mismatch.

Observed results:

- SFT-v2 test: normal 0.3342, wrong-image 0.2641, setting gap 0.0566.
- v5c step20 test: normal 0.3273, wrong-image 0.2418, setting gap 0.0537.
- v6 step15 dev: normal 0.3732, wrong-image 0.2987, setting gap 0.0745.
- v5c step20 dev: normal 0.3827, wrong-image 0.2982, setting gap 0.0845.

Interpretation:

- v5c suppressed some control answers, but it also lowered normal performance on test.
- v6 used stricter wrong-image mining, but did not improve wrong-image F1 and still regressed normal F1 versus v5c on dev.
- The worst regression cluster is spatial localization. The model often changes from a detailed lane/object relationship answer to a generic or wrong maneuver answer.

Root causes:

1. Control refusal pairs were too noisy. A wrong-image or blank-image answer can overlap with the gold answer by chance, especially for generic yes/no and common driving priors.
2. The preference objective rewarded refusal under ablation but did not sufficiently protect normal visual answering.
3. Text-only controls were especially risky: they can teach the model to be less answerable rather than more visually grounded.
4. Token F1 is useful for quick diagnostics but too weak as a final driving QA metric; it can over-reward generic overlap and under-reward spatial correctness.

## V7 Design

The v7 builder is intentionally conservative.

Implemented file:

- `src/data/build_lingoqa_preference_v7_grounded.py`
- Runner: `scripts/56_build_lingoqa_preference_v7_grounded.sh`
- Training runner for next GPU session: `scripts/57_train_lingoqa_pref_v7_grounded.sh`

Mining rules:

- Only mine control pairs when the normal prediction F1 is at least 0.45.
- Only keep control predictions whose F1 is at most 0.35.
- Require normal F1 minus control F1 to be at least 0.20.
- Disable text-only controls by default.
- Down-weight control pairs for spatial localization.
- Up-weight normal SFT anchors for spatial localization, counting, and reasoning.

Current no-GPU build result:

- Total preference pairs: 358.
- Normal pairs: 300.
- Control pairs: 58.
- Control modes: blank-image 36, wrong-image 22.
- Normal SFT anchor weight sum: 584.0.
- Control pair weight sum: 13.165.

This is designed to test whether a small, careful DPO update can improve visual dependency without repeating v5/v6 normal regression.

## Added Evaluation Infrastructure

Implemented:

- `scripts/55_eval_external_visual_controls.sh`
  - Runs normal, text-only, wrong-image, blank-image evaluation for any external JSONL in DriveMind format.
- `src/data/convert_external_to_drivemind.py`
  - Handles generic VQA-like metadata for DriveLM/NuScenes-QA style fields.
  - Preserves `image_paths`, `video_path`, `scene_token`, `sample_token`, source category, and inferred capability.
- `src/eval/compare_visual_control_cases.py`
  - Compares two visual-control runs per case and classifies regressions/improvements.
- `src/eval/compare_visual_control_runs.py`
  - Compares run-level summaries across versions.

## Next No-GPU Tasks

1. Download or place legally obtained DriveLM/NuScenes-QA metadata under `data/external/<dataset>/`.
2. Convert a small validation subset to DriveMind JSONL.
3. Generate wrong-image and blank-image variants.
4. Verify ID/image path consistency and capability breakdown.
5. Prepare a benchmark table template with rows for base Qwen2.5-VL, SFT-v2, v5c, v7.

## Next GPU Tasks

Run public benchmark inference first, then train v7 only if the baseline picture is clear.

Build v7 data:

```bash
cd /root/autodl-tmp/DriveMind-VL
source /root/miniconda3/bin/activate drivemind-vl
bash scripts/56_build_lingoqa_preference_v7_grounded.sh
```

Train v7:

```bash
cd /root/autodl-tmp/DriveMind-VL
source /root/miniconda3/bin/activate drivemind-vl
bash scripts/57_train_lingoqa_pref_v7_grounded.sh
```

Evaluate v7 dev sweep:

```bash
RUN_DIR=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v7_grounded \
SPLIT=dev \
PREFIX_BASE=lingoqa_clean_v2_dev_pref_v7_grounded_sweep \
CHECKPOINTS=checkpoint-step-000012,checkpoint-step-000024 \
bash scripts/52_eval_lingoqa_pref_v5_checkpoint_sweep.sh
```

Evaluate a public benchmark:

```bash
DATA=data/processed/drivelm_eval.jsonl \
PREFIX=drivelm_dev_qwen25vl_3b_base \
MAX_SAMPLES=300 \
bash scripts/55_eval_external_visual_controls.sh /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
```

With adapter:

```bash
DATA=data/processed/drivelm_eval.jsonl \
PREFIX=drivelm_dev_pref_v7_step12 \
ADAPTER=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v7_grounded/checkpoint-step-000012 \
MAX_SAMPLES=300 \
bash scripts/55_eval_external_visual_controls.sh /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
```

## Resume Framing

Best project bullets after the next benchmark layer is added:

- Built an autonomous-driving VLM evaluation pipeline with normal/text-only/wrong-image/blank-image controls to measure true visual dependency instead of language-prior guessing.
- Integrated public driving VQA benchmarks such as DriveLM/NuScenes-QA into a unified DriveMind schema with capability-level breakdowns for spatial localization, object recognition, counting, and reasoning.
- Designed conservative DPO preference mining that only uses high-contrast visual failures and capability-aware SFT anchors, reducing the risk of answer suppression during alignment.
- Produced case-level regression reports to guide data curation and model iteration rather than relying only on aggregate F1.

## Decision Rule

Do not call a trained version better unless it improves or preserves normal F1 and improves visual dependency.

Minimum dev acceptance rule:

- normal F1 must be at least SFT-v2.
- setting gap must be at least SFT-v2.
- per-case gap must be non-negative.
- spatial localization normal regression count must not exceed the previous best version.

If v7 does not pass this rule, stop tuning and focus on public benchmark evaluation plus metric quality.
