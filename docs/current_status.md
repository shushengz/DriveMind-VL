# Current Status

Updated after the local MVP implementation.

## Phase 0: Local MVP

- [x] Initialize Git repository and `initial-mvp` branch
- [x] Create project directory structure
- [x] Create `requirements.txt` and lightweight `requirements-mvp.txt`
- [x] Create environment check scripts
- [x] Generate seed data
- [x] Validate seed data
- [x] Convert to multimodal SFT format
- [x] Run dry-run inference
- [x] Compute eval metrics
- [x] Save bad cases
- [x] Run reward demo
- [x] Run safety guard demo
- [x] Create Gradio demo skeleton and smoke test path
- [x] Create README and docs

## Phase 0.5: Engineering Hardening

- [x] Add richer synthetic hard cases for risk and safety
- [x] Add schema completeness, tool argument accuracy, and reason keyword metrics
- [x] Add `outputs/cases/bad_cases.jsonl` generation
- [x] Add optional Windows PowerShell environment check
- [x] Add local MVP dependency file for demo/debug use

## Phase 1: Local Lightweight Real Inference

- [x] Add optional Qwen2.5-VL inference entrypoint
- [x] Support `--model_name_or_path`
- [x] Support `--max_samples`
- [x] Support `--load_in_4bit` and `--load_in_8bit`
- [x] Keep default path dry-run and no model download
- [x] Add local/server conda environment templates
- [x] Add model file checker
- [x] Add 3B smoke-test script
- [ ] Run 5 real samples after model files are prepared
- [ ] Save real-model bad cases
