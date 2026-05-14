#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
DriveMind-VL server training placeholder.

This local MVP does not start training or download Qwen2.5-VL weights.

Planned server steps:
1. Provision 1x vGPU-48GB or 1x RTX 5090 environment.
2. Clone LLaMA-Factory under third_party/ on the server.
3. Install server dependencies and verify CUDA/bf16.
4. Convert DriveMind-Instruct data:
   bash scripts/03_convert_data.sh
5. Run Qwen2.5-VL-3B LoRA/SFT with configs/qwen25vl_3b_lora_server.yaml.
6. Run Qwen2.5-VL-7B QLoRA/SFT with configs/qwen25vl_7b_qlora_server.yaml.
7. Score offline outputs with rewards and prepare DPO/ORPO/RFT-lite data.
8. Run RFT-lite / DPO / ORPO using configs/rft_lite_server.yaml.
EOF

