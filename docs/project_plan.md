# Project Plan

## 1. Local MVP

Build the repository skeleton, synthetic DriveMind-Instruct data, validation, SFT conversion, dry-run inference, local metrics, safety guard, reward functions, and Gradio demo. This phase is designed for a 3080Ti and does not download or train large models.

## 2. Server SFT

Move the converted data to a 48GB/5090 server. Use Qwen2.5-VL-3B LoRA first, then Qwen2.5-VL-7B QLoRA if memory and throughput are acceptable.

## 3. Perception Enhancement

Connect front-view detection, cabin understanding, and depth estimation. Candidate components include YOLO/GroundingDINO for detection and Depth Anything V2 for relative depth.

## 4. RFT-lite

Generate multiple model outputs, score them with the reward stack, review safety-critical samples, and prepare DPO/ORPO/RFT-lite preference data.

## 5. Demo

Upgrade the Gradio MVP into an end-to-end demo with image inputs, structured vehicle state, perception overlays, model output, safety decisions, and tool-call traces.

## 6. Resume Packaging

After experiments, report measured data size, model size, training setup, eval metrics, safety rejection rate, and demo capabilities.

