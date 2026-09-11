# Stage20 round1_sft全量验证报告

更新时间：2026-09-11 15:32:04 +0800。

状态：**RUNNING**。状态时间：2026-09-11T15:31:40+0800。 当前步骤：`validation_round1_sft_full`。 已处理6,783/19,412（34.9%）。

近期速度约1.63条/秒，当前步骤预计还需2.2小时。仅当前步骤外推，不含后续处理，也不是完成保证。

固定模型：`sprec_gram_toys_v1/round1_sft.pt`，SHA256 `d88678743b9d4650057647aa644ecb3e78dd82dcdb4138642f9a85e671e20193`。完成第1轮SFT的855次训练更新；本任务零新增训练。GPU0，FP32/TF32关闭、原生beam50，复用同精度父参考。该模型不是三轮SPRec的训练预算匹配SFT控制。12小时硬预算，无自动重试。

工程检查：PASSED；16名已有用户的候选和raw/PCRF rank须精确一致。

尚无全量终态结果，暂不判断SFT是否提点。

配置：[round1_sft_full_v1.json](../../experiment/phase20/round1_sft_full_v1.json)。证据目录：`artifacts/phase20/round1_sft_full_v1/`。

本报告由后台观察程序依据已落盘状态和summary自动更新。ANALYZED表示读取并分析现有产物；新评估复算指标不等于重新训练复现。只使用train/validation，未读取test。validation长期重复开发，所有配对区间仅为描述性用户不确定性，未校正多重比较，不能替代独立域或训练seed确认。
