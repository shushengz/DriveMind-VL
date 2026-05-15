# IntelliCockpitBench Integration Notes

IntelliCockpitBench 是 DriveMind-VL 下一阶段最优先对齐的公开 benchmark，因为它直接面向 intelligent cockpit，而不是只面向自动驾驶规划。

## 为什么优先它

根据公开 README 和论文说明，IntelliCockpitBench 覆盖：

- front / side / rear / interior views；
- 多种 road types 和 weather conditions；
- moving 和 stationary vehicle states；
- description、recognition、world-knowledge QA、reasoning 等 query 类型；
- 中英文问题；
- LLM-as-a-judge 和多维评分。

这些维度和 DriveMind-VL 的“智能座舱多模态 Agent”目标最接近。

## DriveMind 映射方式

初版映射建议：

| IntelliCockpitBench 类型 | DriveMind task_type | 说明 |
|---|---|---|
| interior view + driver/passenger state | `cabin_understanding` | 舱内状态理解 |
| front/road risk query | `risk_reasoning` | 前视风险解释 |
| general cockpit interaction | `personalized_service` | 需要人工确认是否可映射工具 |
| recognition/description | `risk_reasoning` or `cabin_understanding` | 取决于视角和问题对象 |

## 当前脚手架

已提供通用转换入口：

```bash
python src/data/convert_external_to_drivemind.py \
  --source intelli_cockpit_bench \
  --input data/samples/external_benchmark_sample.jsonl \
  --image_root data/external/intelli_cockpit_bench/images \
  --output data/processed/drivemind_external_sample.jsonl
```

这个脚本只做轻量字段归一化，不等价于最终精细转换。拿到真实字段后，需要补充 source-specific mapping。

## 第一阶段成功标准

1. 20-100 条样本能转换为 DriveMind JSONL。
2. `validate_dataset.py` 能通过或给出明确缺口。
3. Qwen2.5-VL-3B base 能跑出预测。
4. `eval_by_source.py` 能输出 `intelli_cockpit_bench` 分组。
5. `error_analysis.py` 能列出失败类型。

## 后续要补的内容

- 明确 IntelliCockpitBench 原始字段名和 split。
- 建立 query type 到 DriveMind task 的人工映射表。
- 对外部答案加入语义评分，而不是只做 exact match。
- 增加 text-only / wrong-image / corrupted-image 对照实验，测试视觉 grounding。
