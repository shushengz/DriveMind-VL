# Real Inference Smoke Test

This is the first step after the local MVP. It validates Qwen2.5-VL inference on 5 samples without training.

## 1. Create Environment

Local or server:

```bash
conda env create -f environment_local_infer.yaml
conda activate drivemind-vl-local
```

Server:

```bash
conda env create -f environment_server.yaml
conda activate drivemind-vl-server
```

If CUDA PyTorch is not installed correctly, install the wheel matching the server CUDA version from the official PyTorch selector.

## 2. Download 3B First

Do not download into the repository.

```bash
mkdir -p /mnt/models
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct \
  --local_dir /mnt/models/Qwen2.5-VL-3B-Instruct
```

Alternative:

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /mnt/models/Qwen2.5-VL-3B-Instruct
```

## 3. Check Model Files

```bash
bash scripts/08_check_model_files.sh /mnt/models/Qwen2.5-VL-3B-Instruct
```

## 4. Run 5-Sample Smoke Test

```bash
bash scripts/09_smoke_qwen25vl_3b.sh /mnt/models/Qwen2.5-VL-3B-Instruct
```

Outputs:

```text
outputs/eval_results/qwen25vl_3b_smoke_predictions.jsonl
outputs/eval_results/qwen25vl_3b_smoke_metrics.json
outputs/cases/qwen25vl_3b_smoke_bad_cases.jsonl
```

## 5. Decide Prompt Fixes

Inspect bad cases for:

- invalid JSON
- task mismatch
- unsafe action executed instead of rejected
- missing vehicle state in reasoning
- missing perception object in reasoning
- wrong tool arguments

Do not start LoRA/SFT until 3B base inference produces parseable outputs on the smoke test.

## 6. 7B Timing

Only download and test 7B after the 3B smoke test is stable. 7B should run on the 48GB/5090 server, preferably with 4-bit inference for initial checks.

