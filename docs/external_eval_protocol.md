# External Evaluation Protocol

DriveMind-VL 的外部评测目标不是追求一个漂亮总分，而是把能力拆开：schema following、视觉 grounding、风险推理、舱内理解、工具调用、安全拒识分别看。

## 实验问题

1. Base Qwen2.5-VL-3B 在公开真实图像/真实 query 上是否还能稳定输出 DriveMind JSON？
2. LoRA 改善的是格式与字段，还是改善了真实视觉 grounding？
3. 模型是否依赖文本先验，在错图、无图、占位图下仍给出看似合理的答案？
4. synthetic tool/safety 数据是否能迁移到真实座舱 query？

## 数据分层

- `synthetic_seed`: 本项目规则生成数据，只用于闭环和工具安全策略。
- `intelli_cockpit_bench`: 智能座舱 VQA 主评测来源。
- `nuscenes_qa` / `drivelm` / `drivebench`: 车外驾驶 VQA、风险推理、鲁棒性。
- `drive_and_act` / `dmd`: 舱内驾驶员行为和状态。

所有报告必须按 `meta.benchmark_source` 分组，不能只报告混合平均分。

## 推荐运行流程

先生成或准备预测文件：

```bash
bash scripts/04_run_dry_infer.sh
```

然后运行分组评测和错误分析：

```bash
bash scripts/15_eval_by_source.sh outputs/eval_results/base_predictions.jsonl
```

外部 metadata 小样本转换示例：

```bash
python src/data/convert_external_to_drivemind.py \
  --source intelli_cockpit_bench \
  --input data/samples/external_benchmark_sample.jsonl \
  --image_root data/external/intelli_cockpit_bench/images \
  --output data/processed/drivemind_external_sample.jsonl
```

## 报告要求

每轮外部评测报告至少包含：

- 数据来源与许可说明。
- 样本数、任务分布、是否人工抽检。
- overall metrics。
- by-source metrics。
- by-task metrics。
- error category counts。
- 典型失败案例。
- 当前结论边界。

## 严谨性红线

- 不把 synthetic seed 高分描述成真实座舱能力。
- 不提交公开 benchmark 原图、视频或大数据。
- 不混淆 train/eval split。
- 不用 LoRA 训练集样本做最终结论。
- 不只看 JSON validity，必须看 grounding、task、safety 和错误类型。
