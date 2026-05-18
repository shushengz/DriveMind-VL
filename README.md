# DriveMind-VL

DriveMind-VL is a vehicle multimodal agent MVP for intelligent cabins. The final system is planned around Qwen2.5-VL with front-view images, cabin images, vehicle state, structured perception, and user instructions for driving-risk explanation, tool calling, safety rejection, and personalized cabin service.

This repository is currently the local MVP for a 3080Ti machine. It does not train or download Qwen2.5-VL by default. All model-facing scripts support dry-run behavior so the project can be tested before moving to a 48GB/5090 server.

## Directory

```text
scripts/                 local commands
configs/                 local and server training templates
data/                    raw, image, annotation, processed, and sample data
data/external/           metadata placeholders for public benchmarks
src/data/                seed data, validation, conversion
src/perception/          lightweight perception JSON and stubs
src/agent/               mock tools, parser, planner, safety guard
src/rewards/             reward functions
src/eval/                dry-run inference and metrics
src/demo/                Gradio MVP
outputs/                 logs, checkpoints, eval results, cases
docs/                    project notes
```

## Install

```bash
bash scripts/00_check_env.sh
bash scripts/01_install_env.sh
```

## Local MVP Commands

```bash
bash scripts/02_make_seed_data.sh
python src/data/validate_dataset.py --input data/processed/drivemind_seed.jsonl
bash scripts/03_convert_data.sh
bash scripts/04_run_dry_infer.sh
bash scripts/05_run_eval.sh
bash scripts/06_run_safety_guard.sh
python src/rewards/total_reward.py --demo
bash scripts/07_run_demo.sh
```

For a lighter local MVP install that avoids full model dependencies:

```bash
bash scripts/01_install_env.sh --mvp-only
```

## Seed Data

`src/data/build_seed_data.py` creates synthetic DriveMind-Instruct JSONL records and placeholder images if `data/images` is empty. It covers:

- `risk_reasoning`
- `tool_call`
- `safety_rejection`
- `cabin_understanding`
- `personalized_service`

The synthetic seed is only an MVP smoke-test source. For stronger conclusions, the next dataset version should mix public cockpit/driving benchmarks with DriveMind-specific tool and safety cases. See `docs/benchmark_survey.md` and `docs/data_benchmark_plan.md`.

## Public Benchmark Direction

The recommended v2 data path is a hybrid benchmark:

- `IntelliCockpitBench` for intelligent-cockpit VQA and non-decision cockpit interaction.
- `NuScenes-QA`, `DriveLM`, and `DriveBench` for front-view driving VQA, risk reasoning, and robustness.
- `Drive&Act` and `DMD` for driver monitoring and in-cabin behavior.
- DriveMind synthetic/human-reviewed cases for vehicle control tools and safety refusal.

External metadata can be converted with:

```bash
python src/data/convert_external_to_drivemind.py \
  --source intelli_cockpit_bench \
  --input data/external/intelli_cockpit_bench/sample.jsonl \
  --image_root data/external/intelli_cockpit_bench/images \
  --output data/processed/drivemind_external_intelli_eval.jsonl \
  --limit 100
```

See `docs/external_eval_protocol.md` and `docs/intelli_cockpitbench_integration.md` for the next external-eval workflow.

## Data Conversion

`src/data/convert_to_llamafactory.py` converts JSONL samples into a common multimodal SFT JSON format with `messages` and `images`.

## Dry-Run Inference And Eval

`src/eval/base_infer_dryrun.py` emits deterministic dummy predictions without loading a model. `src/eval/run_all_eval.py` computes JSON validity, risk accuracy, tool accuracy, unsafe rejection rate, and average reward.

`src/eval/base_infer_qwen25vl.py` is the optional phase-1 real-inference entrypoint. It defaults to dry-run unless `--model_name_or_path` points to a local Qwen2.5-VL model directory.

For deeper reports, use grouped evaluation and error analysis:

```bash
bash scripts/15_eval_by_source.sh outputs/eval_results/base_predictions.jsonl
```

This reports metrics by `meta.benchmark_source`, by task, and writes categorized failure cases for experiment analysis.

For the small IntelliCockpitBench external smoke test on the server:

```bash
bash scripts/17_run_intelli_qwen25vl_3b_eval.sh \
  /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  third_party/IntelliCockpitBench
```

See `docs/intelli_cockpitbench_smoke_report.md` for the first 7-sample result.

To check whether the model actually uses image input, run the visual ablation:

```bash
bash scripts/18_run_intelli_visual_ablation.sh \
  /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  third_party/IntelliCockpitBench
```

See `docs/visual_grounding_ablation_report.md` for the first normal/text-only/wrong-image/blank-image comparison.

For capability-level and per-case visual dependency analysis after the ablation has run:

```bash
bash scripts/19_analyze_intelli_ablation_breakdown.sh
```

See `docs/intelli_ablation_breakdown_report.md` for the stricter per-case gap interpretation.

To build a larger balanced IntelliCockpitBench subset after obtaining the full official JSONL and image directory:

```bash
TARGET_SIZE=100 bash scripts/20_build_intelli_eval_subset.sh \
  /path/to/full/english_test.jsonl \
  /path/to/full/images
```

See `docs/intelli_data_scaling_plan.md` and `docs/intelli_data_access_request.md`.

Alternative public datasets are tracked in `docs/public_dataset_options.md`. The current recommended next adapters are LingoQA, DriveBench, Talk2CarSlim, and Drive&Act.

LingoQA metadata can be downloaded and converted without downloading videos:

```bash
ACCEPT_LINGOQA_TERMS=1 bash scripts/22_download_lingoqa_metadata.sh
```

The download helper requires explicit upstream license acknowledgement through the script flag and only writes the small evaluation annotation table to `data/external/lingoqa/evaluation.parquet`.

```bash
TARGET_SIZE=100 bash scripts/21_prepare_lingoqa_subset.sh \
  data/external/lingoqa/evaluation.parquet
```

If the official evaluation `images.zip` has been extracted under `data/external/lingoqa/images`, map real key frames into the subset:

```bash
IMAGE_ROOT=data/external/lingoqa \
TARGET_SIZE=100 bash scripts/21_prepare_lingoqa_subset.sh \
  data/external/lingoqa/val.parquet
```

Run a metadata-only dry evaluation to validate schema, metrics, and grouped reporting:

```bash
bash scripts/23_run_lingoqa_dry_eval.sh
```

LingoQA is video VQA. Real visual inference needs local videos or extracted frames; metadata-only conversion is still useful for auditing and text-only baselines.

## Safety Guard

`src/agent/safety_guard.py` blocks unsafe tool calls such as opening or unlocking doors while moving and blocks unknown tools by default.

## Demo

`src/demo/gradio_app.py` runs a Gradio skeleton with image upload, vehicle-state JSON, instruction input, perception JSON, dummy model output, safety guard result, and reward details.

## Server Training Goal

Server-stage templates are reserved under `configs/` for Qwen2.5-VL-3B LoRA, Qwen2.5-VL-7B QLoRA, and offline reward/preference optimization. `scripts/10_server_train_placeholder.sh` only prints future steps and never launches training.

## Real-Inference Smoke Test

Use `docs/model_download.md` and `docs/real_infer_smoke_test.md` before server training. The first real-model target is Qwen2.5-VL-3B on 5 samples:

```bash
bash scripts/09_smoke_qwen25vl_3b.sh /mnt/models/Qwen2.5-VL-3B-Instruct
```

## TODO

- Replace synthetic perception with YOLO/GroundingDINO and Depth Anything V2.
- Add real Qwen2.5-VL inference behind an explicit non-default flag.
- Build DriveMind-Instruct v2 from public benchmark subsets plus human-reviewed tool/safety data.
- Run LoRA/SFT on a 48GB/5090 server.
- Build DPO/ORPO/RFT-lite data from reward-scored outputs.
