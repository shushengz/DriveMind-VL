# DriveMind-VL

DriveMind-VL is a vehicle multimodal agent MVP for intelligent cabins. The final system is planned around Qwen2.5-VL with front-view images, cabin images, vehicle state, structured perception, and user instructions for driving-risk explanation, tool calling, safety rejection, and personalized cabin service.

This repository is currently the local MVP for a 3080Ti machine. It does not train or download Qwen2.5-VL by default. All model-facing scripts support dry-run behavior so the project can be tested before moving to a 48GB/5090 server.

## Directory

```text
scripts/                 local commands
configs/                 local and server training templates
data/                    raw, image, annotation, processed, and sample data
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

## Data Conversion

`src/data/convert_to_llamafactory.py` converts JSONL samples into a common multimodal SFT JSON format with `messages` and `images`.

## Dry-Run Inference And Eval

`src/eval/base_infer_dryrun.py` emits deterministic dummy predictions without loading a model. `src/eval/run_all_eval.py` computes JSON validity, risk accuracy, tool accuracy, unsafe rejection rate, and average reward.

`src/eval/base_infer_qwen25vl.py` is the optional phase-1 real-inference entrypoint. It defaults to dry-run unless `--model_name_or_path` points to a local Qwen2.5-VL model directory.

## Safety Guard

`src/agent/safety_guard.py` blocks unsafe tool calls such as opening or unlocking doors while moving and blocks unknown tools by default.

## Demo

`src/demo/gradio_app.py` runs a Gradio skeleton with image upload, vehicle-state JSON, instruction input, perception JSON, dummy model output, safety guard result, and reward details.

## Server Training Goal

Server-stage templates are reserved under `configs/` for Qwen2.5-VL-3B LoRA, Qwen2.5-VL-7B QLoRA, and offline reward/preference optimization. `scripts/10_server_train_placeholder.sh` only prints future steps and never launches training.

## TODO

- Replace synthetic perception with YOLO/GroundingDINO and Depth Anything V2.
- Add real Qwen2.5-VL inference behind an explicit non-default flag.
- Expand DriveMind-Instruct with real front-view/cabin data.
- Run LoRA/SFT on a 48GB/5090 server.
- Build DPO/ORPO/RFT-lite data from reward-scored outputs.
