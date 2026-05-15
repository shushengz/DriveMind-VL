# 外部 Benchmark 评测闭环设计与实现

## 本轮背景

上一轮确认了一个关键问题：DriveMind-VL 如果只依赖 synthetic seed 数据，项目结论不够严谨。当前 MVP 和 3B LoRA smoke test 已经证明工程闭环能跑通，但不能证明模型具备真实智能座舱理解能力。

因此本轮目标是建立外部 benchmark 评测闭环，让后续接入 IntelliCockpitBench、NuScenes-QA、DriveLM、DriveBench、Drive&Act、DMD 时，能够按来源、任务和错误类型分析模型表现。

## 本轮完成内容

1. 新增 `src/eval/eval_by_source.py`
   - 支持按 `meta.benchmark_source` 分组。
   - 支持按 `meta.task_type` 分组。
   - 支持 `source::task` 组合分组。
   - 输出 overall、by_source、by_task、by_source_task 指标。

2. 新增 `src/eval/error_analysis.py`
   - 对失败样本做错误类型归因。
   - 覆盖 invalid JSON、task mismatch、schema missing、risk/tool/safety/cabin 等任务级错误。
   - 输出错误类别统计和失败 case JSONL。

3. 新增 `scripts/15_eval_by_source.sh`
   - 一键运行分组评测和错误分析。

4. 新增 `data/samples/external_benchmark_sample.jsonl`
   - 提供不依赖真实外部数据的小样例，用于验证转换和评测链路。

5. 新增文档
   - `docs/external_eval_protocol.md`
   - `docs/intelli_cockpitbench_integration.md`

6. 更新 README 和 `.gitignore`
   - README 增加 grouped eval 用法。
   - `.gitignore` 忽略外部图片/视频和 eval JSON 运行产物。

## 验证结果

本地 PowerShell 没有 `bash`，所以本轮使用等价 Python 命令验证。

已通过：

```bash
python -m py_compile src/eval/eval_by_source.py src/eval/error_analysis.py src/data/convert_external_to_drivemind.py
python src/eval/base_infer_dryrun.py --input data/processed/drivemind_seed.jsonl --output outputs/eval_results/base_predictions.jsonl
python src/eval/eval_by_source.py --predictions outputs/eval_results/base_predictions.jsonl
python src/eval/error_analysis.py --predictions outputs/eval_results/base_predictions.jsonl
python src/data/convert_external_to_drivemind.py --source intelli_cockpit_bench --input data/samples/external_benchmark_sample.jsonl --image_root data/external/intelli_cockpit_bench/images --output data/processed/drivemind_external_sample.jsonl --limit 2
```

dry-run seed 数据分组评测结果：

- total: 100
- source: `synthetic_seed`
- json_validity: 0.9000
- avg_reward: 0.6190
- bad cases: 25

错误分析显示主要错误类别：

- `json_parse_failed`: 10
- `tool_name_mismatch`: 10
- `tool_argument_mismatch`: 10
- `driver_state_mismatch`: 5
- `unsafe_not_refused`: 5
- `missing_safe_alternative`: 5

这些结果来自 dummy dry-run，不能解释真实模型能力，但证明错误归因链路可用。

## 当前结论

DriveMind-VL 现在不再只是一个能跑通 synthetic eval 的 MVP。项目已经具备了下一阶段严谨评测所需的基础设施：

- 数据来源可追踪；
- 指标可以按来源拆分；
- 错误可以按类型归因；
- 外部 benchmark metadata 有转换入口；
- IntelliCockpitBench 有明确接入协议。

这为后续得到更有深度的项目结论打下基础：我们可以明确区分 schema-following、视觉 grounding、真实座舱理解、工具调用和安全拒识，而不是只报告一个混合平均分。

## 下一步

下一轮建议真正接入 IntelliCockpitBench 小样本：

1. 获取 20-100 条 IntelliCockpitBench metadata 和可用图像路径。
2. 适配 source-specific 字段映射。
3. 用 Qwen2.5-VL-3B base 跑真实外部样本。
4. 用 `eval_by_source.py` 和 `error_analysis.py` 记录失败类型。
5. 根据真实失败类型决定是否做 3B LoRA v2，而不是盲目扩大训练。
