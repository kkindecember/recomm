## Material Passport

- Material ID: `GRAM-PHASE4-RPCD-T0-ENGINEERING-20260727`
- Type: Experiment Result
- Verification Status: `INVALID_ENGINEERING_RUN`
- Scope: RPCD T0 首次启动、异常诊断与修复验证
- Scientific decision: `NONE`

# RPCD T0 首次工程运行报告

## 结论

首次任务不能用于判断 RPCD 是否有效。数据与评估预检通过，但 SASRec 从 epoch 1
开始 loss 为 `NaN`，所以归档 summary 中的
`STOP_RPCD_NO_TEACHER_COMPLEMENTARITY` 是无效的自动输出，已明确作废。

## 已通过的部分

- Toys：19,412 用户、11,924 商品、19,412 条 matched GRAM validation 预测；
- Beauty：22,363 用户、12,101 商品、22,363 条 matched GRAM validation 预测；
- 两域用户集合、catalog mapping 与 validation target 对齐率均为 100%；
- 未读取 test prediction，代码未索引 `sequence[-1]`；
- GPU3 资源在任务结束后恢复成功。

## 异常与根因

Toys/Beauty 每个 epoch 的 loss 均为 `NaN`，internal-calibration 指标完全不变且接近
随机水平。最小复现确认：PyTorch 1.11 下，左 padding 与 causal/key-padding mask
叠加会形成全遮蔽 attention 行，非有限值传播到有效位置。

这属于工程实现失败，不是无效假设、负结果或 teacher 不互补的证据。

## 修复与验证

- 序列改为右 padding；
- evaluation 按每个序列真实长度提取最后 hidden state；
- 任一 batch loss 或 epoch 指标非有限时立即抛错，禁止继续生成科学 decision；
- 回归测试由 4 项增至 5 项，5/5 通过；
- 独立两层 SASRec forward/backward 检查：
  `finite_hidden=True`、`finite_loss=True`。

原始无效产物保留在
`artifacts/phase4/rpcd_t0_invalid_nan_20260727_123239/`，没有删除或覆盖。

## 当前状态

`RPCD_T0_ENGINEERING_FIX_VERIFIED_AWAITING_RERUN`

尚无有效 T0 结果，不得进入 RPCD T1。
