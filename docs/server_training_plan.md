# Server Training Plan

Target hardware: 1x vGPU-48GB or 1x RTX 5090.

## Preparation

1. Copy the repository and processed data.
2. Install CUDA-compatible PyTorch.
3. Clone LLaMA-Factory under `third_party/`.
4. Validate the converted multimodal SFT JSON.

## SFT

Start with `configs/qwen25vl_3b_lora_server.yaml`:

- batch size 1
- gradient accumulation 8
- LoRA rank 16
- bf16
- gradient checkpointing

Then test `configs/qwen25vl_7b_qlora_server.yaml`:

- 4-bit quantization
- batch size 1
- gradient accumulation 16
- LoRA rank 16
- bf16
- gradient checkpointing

## RFT-lite / DPO / ORPO

Use `configs/rft_lite_server.yaml` after SFT:

1. Sample multiple outputs per case.
2. Score with local rewards.
3. Review safety-critical pairs.
4. Train preference optimization.

