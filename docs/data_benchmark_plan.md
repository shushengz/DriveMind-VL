# DriveMind-Instruct v2 Data And Evaluation Plan

## 背景

第一轮实验已经证明 MVP 工程可运行，并且 Qwen2.5-VL-3B 在严格 prompt 下能稳定输出 DriveMind JSON。现在的问题是数据严谨性：如果继续只扩大 synthetic seed，模型和评测会过度贴合我们自己写的模板。

DriveMind-Instruct v2 的目标不是简单增加样本量，而是建立一个可以解释的 benchmark mix：哪些能力来自公开真实图像，哪些能力来自自建安全规则，哪些结论只能算工程 smoke test。

## v2 Schema 原则

DriveMind 的统一样本结构保持不变：

- `image`
- `vehicle_state`
- `perception`
- `instruction`
- `answer`
- `meta`

但 `meta` 需要增加来源描述：

- `source`: `synthetic_seed` / `external_benchmark` / `human_reviewed`
- `benchmark_source`: `intelli_cockpit_bench` / `nuscenes_qa` / `drivelm` / `drivebench` / `drive_and_act` / `dmd`
- `external.original_id`
- `external.split`
- `external.category`
- `review_status`: `unreviewed` / `auto_checked` / `human_checked`

## 推荐优先级

### P0: IntelliCockpitBench

理由：这是目前最贴近智能座舱 VLM 评测的公开 benchmark。它包含 front/side/rear/interior 视角、天气、道路、移动/静止状态和中英文 query。我们应优先验证 DriveMind schema 是否能表达它的 query 和 answer。

目标：

- 先接入 50-100 条 metadata 小样本。
- 不提交原图，只记录下载路径和转换脚本。
- 建立 cockpit VQA eval 子集。

### P1: DriveBench 或 DriveLM

理由：用于补足前视驾驶风险和鲁棒性评测。DriveBench 特别适合测试 corrupted/text-only 场景，能判断模型是不是只靠语言先验回答。

目标：

- 构建 risk_reasoning 外部评测子集。
- 增加视觉 grounding 指标：无图/错图/模糊图下性能下降是否合理。

### P2: Drive&Act 或 DMD

理由：用于舱内驾驶员状态，不应只靠 synthetic “我有点困”推断疲劳。Drive&Act 和 DMD 有真实舱内行为/疲劳/分心标签。

目标：

- 把动作/疲劳/分心 label 映射到 `cabin_understanding`。
- 建立 driver_state、passenger_state、suggestion 三类指标。

### P3: synthetic_tool_safety

理由：公开 benchmark 很少覆盖“行驶中开车门、解锁、开窗、调空调”等座舱工具策略。这里仍然需要自建数据，但必须规则化和人工抽检。

目标：

- 保留 rule-generated hard cases。
- 每类工具至少覆盖 allowed/blocked 两种状态。
- 引入 paraphrase，不让模型只记模板。

## 新增转换入口

已新增：

```bash
python src/data/convert_external_to_drivemind.py \
  --source intelli_cockpit_bench \
  --input data/external/intelli_cockpit_bench/sample.jsonl \
  --image_root data/external/intelli_cockpit_bench/images \
  --output data/processed/drivemind_external_intelli_eval.jsonl \
  --limit 100
```

该脚本是轻量脚手架，不负责下载数据。真实 benchmark 的字段格式确认后，需要再写 source-specific 精细映射。

## v2 Eval 指标

保留现有指标：

- `json_validity`
- `risk_accuracy`
- `tool_accuracy`
- `tool_argument_accuracy`
- `unsafe_rejection_rate`
- `schema_completeness`
- `reason_keyword_hit`
- `avg_reward`

新增指标：

- `source_breakdown`: 按 benchmark source 分组统计。
- `visual_grounding_gap`: 正常图像 vs text-only/错图/遮挡图的性能差。
- `cabin_state_accuracy`: 舱内状态分类准确率。
- `external_answer_score`: 对公开 benchmark 原始 answer 的语义相似/LLM judge 分数。
- `human_review_pass_rate`: 抽检通过率。

## 下一步最小闭环

1. 不下载大数据，先从 IntelliCockpitBench repo 或公开样例中整理 20 条 metadata。
2. 跑 `convert_external_to_drivemind.py` 生成 DriveMind 格式。
3. 修改 eval 支持按 `meta.benchmark_source` 分组。
4. 用 Qwen2.5-VL-3B base 跑 20 条外部样本，记录明显失败类型。
5. 再决定是否扩展到 200-500 条混合 eval。
