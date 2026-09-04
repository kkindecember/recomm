# 第十七阶段步骤报告索引

| Step | 状态 | 报告 | 下一门槛 |
|---|---|---|---|
| S17-0 | `COMPLETED` | `Stage17_S0_证据源码数据与资源审计报告.md` | 已解锁 S17-1 |
| S17-1 | `COMPLETED` | `Stage17_S1_公共迁移框架与运行合约报告.md` | 已解锁 S17-2 |
| S17-2 | `COMPLETED_WITH_TRACK_FAILURES` | `Stage17_S2_P0七方向机制探针汇总报告.md` | 已解锁 S17-3 前置修订与正式筛选准备 |
| S17-3 | `COMPLETED` | `Stage17_S3_P0独立正式筛选报告.md` | 已解锁 S17-4；A0 为 provisional D1 candidate，其余按定向诊断处理 |
| S17-4 | `COMPLETED` | `Stage17_S4_P1定向迁移筛选报告.md` | PAWA-lite 为置信区间跨 0 的 weak-positive；局部迁移未形成稳健 winner |
| S17-2R | `COMPLETED_NO_R3_CANDIDATE` | `Stage17_S2R_架构级候选重选与大改筛选报告.md` | 四个 P0 family 均未通过 R2；不运行 R3；S17-5 保持 HOLD |
| S17-FP0 | `COMPLETED` | `Stage17_FP0_来源数据与Fidelity冻结报告.md` | 来源、环境、full-data tokenizer 与 FP1/FP2 训练前合同已冻结 |
| S17-FP1 | `COMPLETED / FP1_NOT_STRONG_PASS` | `Stage17_FP1_FullLATTE_NativeParity报告.md` | Native NDCG 弱正但 CI 跨 0、Hit 下降；不进 D1，不追调 |
| S17-FP2 | `COMPLETED / FP2_NO_PROMOTION` | `Stage17_FP2_GRAM_LATTE_Full正式结果报告.md` | G2 对 G1/G0 均显著为负；关闭 LATTE 与 FP4，下一主线为 FP3 实现/资源准入 |
| S17-FP12-EXTERNAL-D0 | `COMPLETED` | `Stage17_FP12_ExternalD0评测准备报告.md` | 五臂 one-shot D0 与受控恢复审计完成；D1/D2/test/Sports 保持锁定 |

每个步骤只保留一份汇总报告；步骤内试错写入 `artifacts/phase17/attempts/`，不拆成多份报告。
