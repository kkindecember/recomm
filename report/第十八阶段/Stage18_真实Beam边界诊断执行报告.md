# 第十八阶段：真实 Beam 边界诊断

完成时间：2026-09-08T02:46:30.933561+00:00。

执行状态：`DESCRIPTIVE_DIAGNOSTIC_COMPLETED`。S18-2 仍为 `FAILED_SCIENTIFIC_GATE`。
本轮冻结模型，只计算轨迹和梯度，optimizer step=0；没有新训练或 accuracy 晋级。

| 域/输入 | 用户 | 首次非 EOS drop | 局部 top50 仍掉出 | 跨父边界 | 旧 sibling proxy 覆盖 |
|---|---:|---:|---:|---:|---:|
| Toys/training | 100 | 71 | 70 | 64 | 0.0279 |
| Toys/evaluation | 100 | 85 | 85 | 85 | 0.0296 |
| Beauty/training | 100 | 61 | 42 | 47 | 0.1344 |

梯度测量状态：`MEASUREMENT_UNRESOLVED`。
非零 teacher-forcing/native 残差在解释前保留为测量未决；方向统计仅作探索，不能据此启动训练。
Toys evaluation 为同一用户下一 offset，未使用新的确认集。Beauty 只有 training 诊断。

完整结果与逐步轨迹：[run-0001](../../artifacts/phase18/beam_boundary_diagnostic/run-0001)。
