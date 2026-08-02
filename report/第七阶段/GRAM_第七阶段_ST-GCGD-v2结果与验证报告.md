# GRAM 第七阶段：ST-GCGD-v2 P0-R / P0-G 结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-02
- Verification Status: ANALYZED
- Version Label: `phase7_st_gcgd_v2_p0_rg_analysis_v1`
- Scientific Result: `FAIL_CLOSED_BEFORE_P1`
- Governance Result: `RESOURCE_LEASE_UNDERSHOOT`

## 1. 执行结论

P0-R 与 P0-G 已按 Toys、Beauty、seed=2023 执行完毕。所有图边只使用 `items[:-2]`
内 pseudo-future 之前的 prefix；validation、test、Sports 均未读取，parent checkpoint SHA 前后
一致，运行结束后 CodeLlama 已恢复到物理 GPU0。

科学资格门的结论是 **停止进入 P1**：Toys 的完整 ST arm 通过预冻结资格规则，Beauty 未通过，
因此两域整体 fail closed。结果同时显示一个值得单独保留的机制信号：有向 transition-only arm
在两域的 Recall/NDCG 均高于 static arm；失败主要发生在 relation mixer，不能据此否定 `R_ii`
本身。

本轮 P0-G 还有独立的资源合规失败。P0-R 测量的是“冻结 GRAM + 图训练”联合生命周期，P0-G
实际只加载图模型，导致按联合预算计算的 sidecar 过小，总占用没有补足 30,720 MiB。科学结果可作
开发诊断，但本轮不能标记为资源合规成功，也不得自动重跑。

## 2. P0-R 工程与显存结果

| 数据域 | peak allocated | peak reserved | 冻结 workload budget | sidecar | 合计 |
|---|---:|---:|---:|---:|---:|
| Toys | 6,870.9 MiB | 22,370 MiB | 22,528 MiB | 8,192 MiB | 30,720 MiB |
| Beauty | 6,878.4 MiB | 26,176 MiB | 26,368 MiB | 4,352 MiB | 30,720 MiB |

P0-R 使用冻结 GRAM train-only beam 各生成 64 条 hard-negative 记录，并在 GRAM 保持驻留时完成
完整 ST graph 的 forward/backward。两域 checkpoint SHA 前后完全一致，P0-R 本身成功完成，
CodeLlama 正常恢复。

## 3. P0-G train-only 资格结果

### 3.1 Toys（calibration n=3,878）

| arm | Recall@10 | NDCG@10 | Recall@50 | MRR | target margin |
|---|---:|---:|---:|---:|---:|
| static | 0.000516 | 0.000339 | 0.003352 | 0.000901 | -0.090009 |
| R_ui | 0.000516 | 0.000249 | 0.003868 | 0.000761 | -0.091334 |
| R_ii | 0.004126 | 0.002138 | 0.011088 | 0.002291 | -0.167802 |
| full ST | 0.002837 | 0.001264 | 0.006704 | 0.001396 | -0.085285 |

完整 ST 的 margin 高于 static，且 Recall@10/NDCG@10 未同时下降，Toys 资格门通过。注意绝对
命中率仍很低，且 transition-only 的排序指标高于 full ST，mixer 没有保留全部转移收益。

### 3.2 Beauty（calibration n=4,458）

| arm | Recall@10 | NDCG@10 | Recall@50 | MRR | target margin |
|---|---:|---:|---:|---:|---:|
| static | 0.000449 | 0.000140 | 0.002692 | 0.000579 | -0.089503 |
| R_ui | 0.000449 | 0.000146 | 0.002467 | 0.000584 | -0.090862 |
| R_ii | 0.002916 | 0.001344 | 0.008524 | 0.001511 | -0.167314 |
| full ST | 0.002692 | 0.001019 | 0.007178 | 0.001184 | -0.096352 |

虽然 full ST 的 rank-based 指标高于 static，但 target margin 从 -0.089503 降至 -0.096352，未通过
预冻结的“双条件”资格规则。Beauty 因此 fail closed；根据计划，两域不能进入新 P1 cohort。

## 4. 机制诊断

- `R_ii` 在 Toys、Beauty 都产生明显更高的 Recall@10、NDCG@10、Recall@50 和 MRR，支持“显式
  有向转移比静态 user-item 图更贴近 next-item 排序”的方向。
- `R_ui` 相对 static 没有稳定改善，recency weighting 单独不足以解决目标错配。
- full ST 在两域都低于 transition-only，说明当前 mixer 把较弱的 `R_ui` 信号重新混入，并损失了
  transition 排序优势；Beauty 的 margin 也进一步触发资格失败。
- hard-negative cache 每域各有 64 条，其中实际进入 fit split 的仅 53 条；相对 15,534/17,905 条
  fit 记录覆盖很低。训练目标形式满足“随机 + 冻结 GRAM hard negative”的组合，但强 hard-negative
  覆盖不足，后续设计必须先扩大 cache 或改成确定性的离线 candidate bank。

## 5. 资源与完整性审计

P0-G 遥测每域仅采到 3 个 5 秒样本，且 telemetry 子进程继承了空的 shell 变量快照，dataset 列为空。
根据 runner 固定的 Toys→Beauty 顺序，观测峰值分别为约 10,460 MiB 与 6,620 MiB，较 30,720 MiB
短缺约 20,260 MiB 与 24,100 MiB。根因是 P0-R/P0-G 测量路径不一致，不是 workload 超额。

完整性状态：train-only 切分通过；pseudo-future 未写入图边；test/Sports/validation 均为 false；
checkpoint SHA 通过；无 NaN/Infinity；无自动重试；CodeLlama 恢复通过。资源租约与 telemetry
domain 标记失败，因此治理总状态为 `RESOURCE_LEASE_UNDERSHOOT`。

## 6. 决定与下一步

1. 不实现或启动 P1，不选择新 development cohort。
2. 保留 `R_ii` 作为下一版唯一积极机制；重新设计 mixer，使其能 fail-closed 回退到 transition-only，
   而不是默认混合较弱的 `R_ui`。
3. 在任何新运行前，单独注册 graph-only 显存 pilot，并修复 telemetry 通过共享文件读取 dataset；
   sidecar 必须按实际 graph-only 峰值补足 30,720 MiB。
4. 扩大冻结 GRAM train-only hard-negative bank 覆盖后，再定义新的 P0-G 配置。旧结果和 SHA 必须保留，
   不得把修复后的运行当作本次自动重试。

## 7. 关键产物 SHA-256

- Toys P0-G summary: `078fc757f2a3173908f3b6460548a005ce637ce7fa8c8b8583480c63119a4385`
- Beauty P0-G summary: `88575240e8b151cce2475f2dffbbe66bd3ca87a2b3badd3d13fd49ba7f0e6179`
- P0-G telemetry: `5dcf40ac4e1f2fe3ad4ea13021b1b1bbca08e3ac568640b8e52a33546f5aa55a`
