# 公开 Benchmark 调研与 DriveMind 数据 v2 方案

## 本轮背景

第一轮 DriveMind-VL 已经完成本地 MVP、Qwen2.5-VL-3B smoke test 和 3B LoRA/SFT 小实验。结果显示模型在 synthetic DriveMind-Instruct seed 上 JSON 输出、risk/tool/safety 指标都很稳定，LoRA 主要改善 reasoning keyword 和 avg_reward。

但这个结论有边界：当前 seed 数据由规则生成，图像多为占位图，不能证明模型真正具备真实智能座舱感知和风险理解能力。因此本轮重点是确认公开项目的数据与评测方式，并规划下一步更严谨的数据路线。

## 核心判断

纯自生成数据不够严谨，只适合工程闭环和工具/安全规则验证。严肃项目结论需要引入公开 benchmark 或真实图像标注。

目前可参考的数据方向如下：

1. 智能座舱 VQA：IntelliCockpitBench。
2. 自动驾驶车外 VQA/推理：NuScenes-QA、DriveLM、DriveBench。
3. 舱内驾驶员行为：Drive&Act、DMD Driver Monitoring Dataset。
4. 工具调用过程评测：nuScenes-Agent 可借鉴，但它是自动驾驶感知工具，不是座舱车控工具。

最重要的新发现是：IntelliCockpitBench 已经明确面向 intelligent cockpit，包含车内外视角、天气、道路、移动/静止状态和多类 query。它应成为 DriveMind-VL 下一阶段最优先对齐的公开 benchmark。

## 本轮完成内容

1. 新增 `docs/benchmark_survey.md`，记录公开 benchmark 对比、数据严谨性判断和 DriveMind-Instruct v2 数据构成。
2. 新增 `docs/data_benchmark_plan.md`，给出 v2 schema、数据源优先级、eval 指标和下一步最小闭环。
3. 新增 `src/data/convert_external_to_drivemind.py`，作为外部 benchmark metadata 到 DriveMind-Instruct JSONL 的轻量转换脚手架。
4. 新建 `data/external/*` 目录，用于后续放置公开 benchmark 的 metadata 或本地路径占位，不提交原图和大文件。
5. 更新 README、data schema 和实验报告，明确第一轮实验是 synthetic smoke test，下一步目标是 hybrid benchmark。

## 对当前项目结论的修正

可以说：

- DriveMind-VL 已经具备可运行的本地/服务器 MVP 闭环。
- Qwen2.5-VL-3B 能稳定遵循 DriveMind JSON schema。
- 小规模 LoRA 不破坏 base 能力，并略微改善 reasoning reward。
- 当前 safety guard 和 reward 机制可作为后续 RFT/DPO/ORPO 的基础。

暂时不能说：

- 模型已经在真实座舱场景上可靠。
- synthetic eval 的高分代表真实视觉理解能力。
- 当前 LoRA 已经显著提升真实驾驶风险判断。

## 下一步建议

下一轮不要直接扩大训练，也不要急着跑 7B。建议先做外部小样本评测闭环：

1. 优先处理 IntelliCockpitBench，整理 20-100 条可访问样本 metadata。
2. 用 `convert_external_to_drivemind.py` 转成 DriveMind schema。
3. 修改 eval，按 `meta.benchmark_source` 分组输出指标。
4. 用 3B base 跑外部样本，观察外部真实图像上 JSON 稳定性和错误类型。
5. 根据失败类型再决定是补数据、改 prompt、扩 reward，还是上 7B QLoRA。

这条路线比继续扩大 synthetic 数据更稳，也更利于后面写论文式实验报告、项目展示和简历包装。
