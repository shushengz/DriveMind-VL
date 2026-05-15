# IntelliCockpitBench 视觉 Grounding 消融实验

## 本轮背景

上一轮我们在 7 条 IntelliCockpitBench 真实样本上跑了 Qwen2.5-VL-3B base。结果是 JSON 和 schema 都稳定，但 `external_answer_f1=0.2070`，说明真实 VQA 细节能力弱。

这轮要进一步回答一个更严谨的问题：模型是否真的依赖图像，还是主要靠问题文本和语言先验在猜？

## 本轮完成内容

1. 新增 `src/data/make_visual_ablation.py`
   - 生成 `wrong_image` 数据集：把样本图片循环错配。
   - 生成 `blank_image` 数据集：所有样本使用空白灰图。

2. 新增 `src/eval/compare_ablation_metrics.py`
   - 汇总 normal、text-only、wrong-image、blank-image 四组指标。

3. 新增 `scripts/18_run_intelli_visual_ablation.sh`
   - 一键准备 IntelliCockpitBench 小样本；
   - 生成消融数据；
   - 跑四组 Qwen2.5-VL-3B 推理；
   - 分组评测、错误分析、汇总结果。

4. 修正 `external_vqa` 的 reward 口径
   - 之前 `avg_reward` 主要奖励 task/schema，无法反映答案是否正确；
   - 现在 `external_vqa` 的 task reward 使用 answer token-F1；
   - 后续报告仍以 `external_answer_f1` 为主指标。

## 服务器实验结果

运行命令：

```bash
bash scripts/18_run_intelli_visual_ablation.sh \
  /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  third_party/IntelliCockpitBench
```

结果：

| 设置 | json_validity | external_answer_f1 | avg_reward | failed cases |
|---|---:|---:|---:|---:|
| normal | 1.0000 | 0.2070 | 0.4371 | 4 / 7 |
| text_only | 1.0000 | 0.0801 | 0.4133 | 6 / 7 |
| wrong_image | 1.0000 | 0.1135 | 0.4162 | 5 / 7 |
| blank_image | 1.0000 | 0.1766 | 0.4423 | 5 / 7 |

## 结论

正常图像相对 text-only 和 wrong-image 有提升，说明模型确实使用了一部分视觉信号。

但提升幅度不大，而且 blank-image 的 F1 仍接近 normal。这说明 7 条样本太少，且部分问题存在语言先验；当前还不能强声称模型具备可靠视觉 grounding。

更严谨的说法应该是：

> Qwen2.5-VL-3B 在 DriveMind external_vqa schema 下格式稳定；原图相对无图/错图有一定收益，但小样本上视觉 grounding 证据不足，需要扩大 IntelliCockpitBench 样本并加入任务子类指标。

## 下一步

1. 扩大 IntelliCockpitBench 样本到 50-100 条。
2. 按任务子类拆分：
   - counting；
   - object recognition；
   - spatial localization；
   - weather / road condition；
   - scene completeness。
3. 对每个子类都跑 normal / text-only / wrong-image / blank-image。
4. 做每题级别的 answer overlap 和人工抽检。
5. 基于真实失败类型构建 DriveMind-Instruct v2 训练样本，再跑 3B LoRA v2。
