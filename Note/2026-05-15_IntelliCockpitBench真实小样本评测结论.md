# IntelliCockpitBench 真实小样本评测结论

## 本轮背景

前两轮已经完成公开 benchmark 调研、DriveMind-Instruct v2 数据路线、按来源分组评测和错误分析。本轮开始真正接入公开 benchmark 小样本，目标是验证 Qwen2.5-VL-3B 在真实车载图片和真实 VQA 问题上的表现。

本轮使用 IntelliCockpitBench 仓库自带的 `Evaluation/data/jsonl/english_test.jsonl` 和对应 7 张图片。该样本很小，只能作为 smoke test，不能当完整 benchmark 结论。

## 本轮完成内容

1. 浅克隆 IntelliCockpitBench 到本地 `third_party/IntelliCockpitBench`。
2. 确认真实字段：
   - `question_id`
   - `category`
   - `subcategory`
   - `question`
   - `reference`
   - `img_path`
   - `roadway`
   - `sub_roadway`
   - `shooting_angle`
   - `weather_conditions`
   - `answer`
3. 修正一个关键数据严谨性问题：
   - `reference` 是 gold；
   - `answer` 是示例模型输出，不能作为 gold。
4. 新增 `external_vqa` 桥接任务，避免把普通 VQA 强行塞进 `risk_reasoning`。
5. 新增 `src/data/prepare_intelli_cockpitbench_sample.py`。
6. 新增 `scripts/17_run_intelli_qwen25vl_3b_eval.sh`。
7. 在服务器上运行 Qwen2.5-VL-3B base 的真实图片推理。
8. 新增 `docs/intelli_cockpitbench_smoke_report.md`。

## 服务器运行结果

服务器路径：

```bash
/root/autodl-tmp/DriveMind-VL
```

模型路径：

```bash
/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
```

数据转换与校验：

- 样本数：7
- valid：7
- task_counts：
  - `external_vqa`: 7

Qwen2.5-VL-3B base 评测：

| 指标 | 数值 |
|---|---:|
| json_validity | 1.0000 |
| schema_completeness | 1.0000 |
| external_answer_f1 | 0.2070 |
| avg_reward | 0.6750 |
| error_analysis failed_cases | 4 / 7 |

## 关键错误

1. 车辆品牌识别错误：
   - 预测：Toyota
   - 金标：Volkswagen Passat

2. 数量/场景细节不精确：
   - 问题：前方能看到什么？
   - 预测描述了雾中山路和白车；
   - 金标强调 about two cars ahead。

3. 描述不完整：
   - 预测：trees
   - 金标：trees、shrubs、streetlights。

4. 场景描述不完整：
   - 预测：a bus
   - 金标：bus、pedestrian、black car、trees、buildings。

## 本轮结论

这个结果非常重要：Qwen2.5-VL-3B 在真实外部样本上能稳定输出 DriveMind JSON，但真实视觉 grounding 和细粒度回答明显不足。

也就是说，第一轮 synthetic seed 和 LoRA smoke test 的高分，主要证明了 schema following、prompt following 和工具链可运行；它不能证明模型已经具备真实智能座舱 VQA 能力。

这正是项目深度所在：我们现在能够把“格式能力”和“真实视觉理解能力”拆开评估，而不是用一个混合平均分掩盖问题。

## 下一步

下一轮不应该直接训练 7B。建议：

1. 扩大 IntelliCockpitBench 外部样本到 50-100 条。
2. 把外部评测指标从 token F1 升级为：
   - category-level exact/soft score；
   - object/count/location 子指标；
   - 可选 LLM judge，但要保留人工抽检。
3. 加入 text-only / wrong-image 对照，判断模型是否真的看图。
4. 再用这些失败类型构造 DriveMind-Instruct v2 的训练集。
5. 最后再跑 3B LoRA v2，比较 external_answer_f1 和错误类型是否改善。
