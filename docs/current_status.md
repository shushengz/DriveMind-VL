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
- [x] Run 5 and 20 real samples after model files are prepared
- [x] Save real-model bad cases

## Phase 2: 3B LoRA/SFT Smoke Experiment

- [x] Create deterministic 80/20 train/eval split
- [x] Run fixed Base 3B eval on 20 samples
- [x] Run minimal 3B LoRA/SFT smoke training on 80 samples
- [x] Load LoRA adapter and run fixed eval
- [x] Write `docs/experiment_report.md`
- [x] Write Chinese round note under `Note/`
- [x] Add external benchmark survey and DriveMind-Instruct v2 data plan
- [x] Add grouped by-source eval and error analysis scripts
- [x] Run 7-sample IntelliCockpitBench external smoke eval with Qwen2.5-VL-3B
- [x] Run visual grounding ablation on IntelliCockpitBench samples
- [x] Add IntelliCockpitBench dataset audit and balanced subset builder
- [x] Add LingoQA metadata-only download helper, subset adapter, and dry-eval script
- [ ] Expand argument hard cases to 300-500 samples
- [ ] Run second 3B LoRA with richer data
