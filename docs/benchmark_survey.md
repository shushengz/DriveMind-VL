# DriveMind-VL Benchmark Survey

本文档记录 DriveMind-VL 下一阶段的数据与评测依据。结论先行：纯 synthetic seed 数据只适合验证工程闭环，不足以支撑严肃的模型结论。下一阶段应转为 hybrid benchmark：公开车载/自动驾驶 benchmark 做外部 grounding，DriveMind 自建数据只保留在车控工具调用、安全拒识和个性化座舱服务这些公开数据缺口明显的任务上。

## 关键结论

1. 已经存在智能座舱方向的公开 benchmark：`IntelliCockpitBench`。它比一般自动驾驶 VQA 更贴近座舱交互，适合作为 DriveMind-VL 的重点参考。
2. 车外驾驶视觉语言评测已有较成熟数据：`NuScenes-QA`、`DriveLM`、`DriveBench`。
3. 舱内驾驶员行为/状态有独立 benchmark：`Drive&Act`、`DMD Driver Monitoring Dataset`。
4. 工具调用式车载 Agent 仍没有完全统一的公认 benchmark。`nuScenes-Agent` 已经开始覆盖多摄像头与感知工具调用，但它面向自动驾驶感知工具，不是座舱车控工具。
5. 因此 DriveMind-Instruct v2 应该分层：公开 benchmark 子集用于真实图像和客观评测，自建 synthetic/rule 数据用于车控工具与安全策略，再通过人工抽检建立可信度。

## Benchmark 对比

| Benchmark | 方向 | 数据/规模 | 评测方式 | 对 DriveMind 的价值 | 局限 |
|---|---|---:|---|---|---|
| IntelliCockpitBench | 智能座舱 VQA | 7,622 images, 16,154 queries | LLM-as-judge + 多维评分 | 最贴近座舱交互，覆盖车内/车外、移动/静止、天气/道路 | 主要是 VQA，不直接覆盖车控工具执行与 safety guard |
| NuScenes-QA | 自动驾驶 VQA | 34K scenes, 460K QA | Accuracy / VQA baselines | 图像、点云、多帧、3D 标注 grounding 强 | 主要是车外场景，座舱服务弱 |
| DriveLM | Graph VQA / driving language | nuScenes + CARLA | challenge pipeline, language/GPT score | 感知、预测、规划的逻辑链强 | 不是座舱 Agent；工具调用不是重点 |
| DriveBench | 自动驾驶 VLM 可靠性 | 19,200 frames, 20,498 QA | clean/corrupt/text-only robustness | 可测试视觉 grounding 和鲁棒性，避免只靠 prompt 猜答案 | 仍偏自动驾驶外部场景 |
| Drive&Act | 舱内行为识别 | 12h video, 5 views, NIR/Depth/Color, 83 labels | action recognition | 可支持 driver/passenger state、疲劳/动作理解 | 不是语言问答/Agent 数据 |
| DMD | Driver monitoring | 多摄像头 RGB/Depth/IR，真实车+模拟器 | distraction/fatigue/gaze/head pose | 可支持疲劳、分心、视线、手部状态 | 数据许可和下载成本需要确认 |
| nuScenes-Agent | 自动驾驶工具调用 | 5,000 QA plan, sample currently visible | camera/tool process eval | 可借鉴工具调用过程评测 | 工具是感知工具，不是座舱车控工具；公开完整集状态需复核 |

## 为什么不能只用自生成数据

当前 seed 数据有两个优点：格式可控、能跑通本地闭环。但它也有明显风险：

- 图像是占位图，模型没有真正利用视觉信息。
- perception JSON 和 answer 由同一套规则生成，评测容易变成“规则复述”。
- eval 集和 train 集同分布，容易高估 LoRA 效果。
- 工具参数、风险等级、舱内状态缺少真实场景边界。

所以第一轮实验只能表述为：`Qwen2.5-VL-3B + prompt/LoRA 能在 DriveMind schema 上稳定输出 JSON，并能通过 rule-based safety/eval 闭环`。不能表述为模型已经具备真实座舱风险识别能力。

## DriveMind-Instruct v2 数据组成

建议将数据源拆成 4 个 split/source：

1. `synthetic_tool_safety`
   - 来源：现有 DriveMind seed + 后续人工扩写。
   - 用途：车控工具调用、安全拒识、schema 稳定性。
   - 评测：exact tool、argument slot、safety block/allow、JSON validity。

2. `public_cockpit_vqa`
   - 来源：优先 IntelliCockpitBench。
   - 用途：座舱/车辆交互 VQA、场景描述、识别、推理。
   - 评测：参考其多维评分，同时映射到 DriveMind schema。

3. `public_driving_vqa`
   - 来源：NuScenes-QA、DriveLM、DriveBench。
   - 用途：前视风险解释、目标/距离/场景理解、鲁棒性。
   - 评测：accuracy、schema completeness、reason grounding、robustness gap。

4. `public_driver_monitoring`
   - 来源：Drive&Act、DMD。
   - 用途：疲劳、分心、手部/头部/视线、舱内动作理解。
   - 评测：driver_state accuracy、action label mapping、safety suggestion accuracy。

## 下一轮执行计划

1. 下载或小样本抽取公开 benchmark metadata，不直接提交原图。
2. 用 `src/data/convert_external_to_drivemind.py` 先转换 20-100 条小样本，验证 schema 能否承载外部数据。
3. 建立 `data/processed/drivemind_v2_eval.jsonl`，至少包含：
   - synthetic safety/tool 50 条；
   - IntelliCockpitBench/cockpit VQA 50 条；
   - NuScenes-QA/DriveLM/DriveBench driving VQA 50 条；
   - Drive&Act/DMD cabin monitoring 50 条。
4. 更新 eval，把指标拆成：
   - schema/format；
   - task accuracy；
   - safety guard；
   - visual grounding；
   - robustness；
   - human-review pass rate。
5. 在 3B base 和 3B LoRA 上重跑固定 v2 eval，观察 synthetic 到 public data 的泛化落差。

## Sources

- IntelliCockpitBench: https://github.com/Lane315/IntelliCockpitBench
- IntelliCockpitBench paper: https://aclanthology.org/2025.findings-acl.798.pdf
- NuScenes-QA AAAI page: https://ojs.aaai.org/index.php/AAAI/article/view/28253
- DriveLM: https://github.com/OpenDriveLab/DriveLM
- DriveBench: https://github.com/worldbench/DriveBench
- Drive&Act: https://driveandact.com/
- DMD Driver Monitoring Dataset: https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset
- nuScenes-Agent: https://huggingface.co/datasets/se-bench-anon/nuScenes-Agent
