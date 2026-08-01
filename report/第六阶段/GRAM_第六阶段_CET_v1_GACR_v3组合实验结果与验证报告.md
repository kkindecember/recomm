# GRAM 第六阶段：CET-v1 × GACR-v3 组合实验结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-01
- Verification Status: ANALYZED
- Version Label: `phase6_cet_v1_x_gacr_v3_partial_validation_v1`
- Source Experiment: `GRAM_PHASE6_CET_GACR_V1`
- Result Scope: Toys 完整；Beauty 不完整

## 1. 执行结论

本次运行**没有完成预注册的双域实验**。工作负载在预注册的 `21,600` 秒硬超时处
以 exit code `124` 终止；当时 Toys 四组评估及三个 residual seed 已全部写出，Beauty
仅完成 GRAM arm 的 `192/1024`，未生成 Beauty 逐用户结果，也未生成总
`summary.json`。

- 启动：2026-07-31 23:54:24+08:00；GPU telemetry 截止约
  2026-08-01 05:54:26+08:00；
- 终态：`failed_to_restore_resource`，其中科学工作负载 exit=`124`，并伴随当时的
  CodeLlama 恢复探针失败；
- 当前资源复核：CodeLlama tmux 已运行，但状态为 `waiting_for_model`；原实验状态文件
  是历史终态，不代表组合科学计算成功；
- Sports/test 均未读取；没有自动重试。

因此，本报告不能给出 Beauty、双域宏平均或完整四组验证结论，也不能把本次运行标记为
`VERIFIED`。不过，完整的 Toys 结果已经足以判定预注册的“保留组合”门不可能通过。

## 2. 完整性审计

| 检查项 | 结果 | 解释 |
|---|---|---|
| 实现 SHA256 | PASS | `4bb97a52...16f6313`，与预注册值一致 |
| Toys cohort | PASS | 1024 个唯一用户；用户 SHA256=`848c387a...c4bf63`，与预注册值一致 |
| 四组配对 | PASS（Toys） | 三个 seed 的 GRAM/CET 基础记录完全一致，四组均为同一用户/target |
| Toys 逐用户文件 | PASS | 3/3 文件，每份 1024 行、22 列 |
| Beauty 逐用户文件 | FAIL | 0/3；日志只到 GRAM `192/1024` |
| 总 summary | FAIL | `summary.json` 不存在 |
| 双域决策门 | 不可完整计算 | Beauty 与宏平均缺失 |
| checkpoint/optimizer 终态证明 | 不完整 | 程序在写 summary 前超时，未生成统一 lineage/integrity 区块 |
| Sports/test 封存 | PASS（日志与状态） | `sports_read=false`、`test_read=false` |

预注册元数据还有一个轻微时间顺序问题：配置文件和计划文件的文件修改时间均早于启动，
实现 SHA 也匹配，但配置内 `registered_at=23:55:00` 比 runner 的
`started_at=23:54:24` 晚 36 秒。更符合现有证据的解释是内嵌时间戳填写不准确，而非
结果后改配置；后续实验仍应保证“冻结时间 < 启动时间”在元数据中严格成立。

## 3. Toys 完整局部结果

### 3.1 三 seed 平均

| 方法 | mean NDCG@10 | 相对 GRAM | 相对最强单组件 |
|---|---:|---:|---:|
| GRAM | 0.070871 | — | — |
| CET-v1 | 0.067475 | -4.792% | -6.074% vs GACR-v3 |
| GACR-v3 | **0.071839** | **+1.366%** | — |
| CET-v1+GACR-v3 | 0.069886 | -1.390% | **-2.718% vs GACR-v3** |

组合相对 CET-v1 为 `+3.573%`，说明 GACR residual 能补回 CET-v1 的一部分损失；但
CET-v1 本身在这个 fresh Toys cohort 上低于 GRAM，组合最终仍低于 GRAM，更低于冻结
GACR-v3。组合不是两个正向组件的叠加，而是 GACR 对一个较弱 backbone 的部分修复。

### 3.2 逐 seed 比较

| seed | CET vs GRAM NDCG | GACR vs GRAM NDCG | 组合 vs GRAM NDCG | 组合 vs GACR NDCG | 组合 vs GACR 95% CI | 组合 Recall@10 vs GRAM | broad harm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 | -4.792% | +1.185% | -0.805% | -1.967% | [-7.503%, +3.632%] | +0.488pp | 0.781% |
| 2024 | -4.792% | +1.983% | -1.449% | -3.365% | [-8.529%, +1.971%] | +0.293pp | 0.781% |
| 2025 | -4.792% | +0.929% | -1.914% | -2.817% | [-8.367%, +2.697%] | +0.293pp | 0.879% |

- 组合在 3/3 seeds 的 NDCG@10 都低于 GACR-v3；
- 组合相对 GRAM 的 Recall@10 和 broad-harm 安全门在 Toys 3/3 cells 均通过；
- 这不是安全性失败，而是主指标互补性失败：组合提高 Recall，却把 top-10 的相关位置
  排得不如 GACR-v3；
- combo-vs-GACR 的三个 CI 都跨 0，统计证据仍属不确定；但预注册决策使用严格点估计门，
  不以 CI 为否决条件。

## 4. 预注册决策

预注册要求组合同时满足：双域宏平均超过两个单组件、Toys 和 Beauty 各自三 seed mean
都超过两个单组件，以及所有安全 cells 通过。

Toys 已出现：

`mean NDCG@10(combo)=0.069886 < 0.071839=mean NDCG@10(GACR-v3)`。

所以即使未知的 Beauty 结果非常好，“每个域严格超过两个单组件”仍然不可能成立。
据此可执行不依赖 Beauty 的单向停止判定：

> **`REJECT_COMBINATION_RETURN_TO_GACR_V3`**

这不是完整复现原程序预期生成的 `RETURN_TO_STRONGER_SINGLE_METHOD` 双域 summary，
而是基于已完整域触发的逻辑性 futility 判定。不得报告 Beauty 结果或双域宏平均。

## 5. Statistical Interpretation

Overall Confidence：**CAUTION**。

Toys 的方向性很一致：GACR-v3 3/3 高于组合，组合 3/3 低于 GRAM；然而 combo-vs-GACR
的用户级 bootstrap CI 均跨 0。应表述为“组合未通过预注册保留门”，不应表述为“已经
证明 CET 与 GACR 在总体上不兼容”。Beauty 缺失也限制了跨域机制解释。

当前项目已经进行多轮方向、cohort 和配置探索，CI 未作全项目多重比较校正。预注册门
降低了本轮结果后选择风险，但不能消除整个研究过程的 look-elsewhere 风险。

## 6. Fallacy Scan

覆盖：**11/11 checked**

| 谬误 | 严重度 | 检查结果 |
|---|---|---|
| Simpson's paradox | CAUTION | Beauty 缺失，禁止用 Toys 方向代替双域或宏平均方向 |
| Ecological fallacy | NOTE | 报告 changed-user/broad-harm，不由均值断言所有用户受损 |
| Berkson's paradox | CAUTION | Toys/Beauty 均为持续开发域，不是最终确认域 |
| Collider bias | NOTE | cohort 由 salted ID 选择，未按模型结果或 target 选择 |
| Base-rate neglect | NOTE | 已报告 Recall、changed coverage 与 broad-harm 基率 |
| Regression to mean | NOTE | fresh cohort 未按历史极端结果抽取 |
| Survivorship bias | CAUTION | Toys 完整，但 Beauty 被硬超时截断，禁止只用完成域声称完整实验 |
| Look-elsewhere effect | CAUTION | 长期多方向探索；本轮未作全项目多重比较修正 |
| Garden of forking paths | NOTE | 四组、seed 与决策门已冻结；失败后未调参救援 |
| Correlation != causation | NOTE | 仅解释离线排序干预，不外推线上用户行为因果 |
| Reverse causality | NOTE | 排序特征来自 target 发生前信息，未见未来 target 注入 |

## 7. 复现状态

- Method：未重跑；对保存的 Toys CSV 使用原实现中的汇总与 10,000 次用户级配对
  bootstrap 函数重新计算；
- Verdict：`CANNOT_VERIFY_FULL_EXPERIMENT`；
- Toys 局部汇总：`ANALYZED`，不能升级为完整实验 `VERIFIED`；
- 原始 CSV SHA256：seed 2023=`19355301...a13`、2024=`51021cf5...164`、
  2025=`8b51041c...0e9`。

## 8. 下一步建议

### 8.1 当前方法决策

1. 冻结并回到 **GACR-v3 单方法**；不对 CET+GACR 做结果后调参，也不读取
   Sports/test。
2. 不建议为了“是否保留组合”而原样重跑整个双域实验，因为完整 Toys 已使保留门
   数学上不可达。
3. 如果论文或审计需要完整四组跨域表，再单独预注册一次 **Beauty-only completion**：
   保持 checkpoint、residual、cohort、特征和评估完全不变，使用新 experiment ID 和
   新输出目录，不覆盖本次失败产物；其定位是补全描述性证据，不得翻转组合停止决定。

### 8.2 下一项增长实验（研究者修正后）

组合失败否定的是 CET backbone 与冻结 GACR residual 的互补性，不是否定 GACR 方向。
GACR-v3 在两个 fresh cohort 的 Toys overall NDCG 均为正，Beauty 证据更强，因此应先
围绕已有增长继续迭代，而不是切换到 S0。

下一主实验为 **GACR-v4 target-free learned residual-application gate**：冻结 v3
residual，只增加用户级收益门，针对 Toys tail/Recall@50 的局部负 cell；同时比较
`GRAM`、冻结 `GACR-v3`、`GACR-v4`。详细预注册设计见
`plan/GRAM_第六阶段_GACR-v4目标无关收益门控实验计划.md`。

### 8.3 工程修复

后续 GPU 验证应按域拆分并支持可验证的 resume/merge。当前程序对每个域依次重算 GRAM
与 CET 两套 1024 用户候选，Toys 已耗去约 5 小时 14 分钟，原 6 小时门不可能稳定覆盖
双域。工程修复只能改变调度、缓存和输出原子性，不能改变科学配置；每域完成后应立即
写独立 summary/lineage，最终再做只读合并。

## 9. 产物

- `artifacts/phase6/configs/cet_gacr_v1_preregistered.json`
- `artifacts/phase6/cet_gacr_v1/status.json`
- `artifacts/phase6/cet_gacr_v1/run.log`
- `artifacts/phase6/cet_gacr_v1/gpu_telemetry.csv`
- `artifacts/phase6/cet_gacr_v1/Toys/four_arm_seed{2023,2024,2025}_per_user.csv`
- `experiment/phase6/cet_gacr_v1.py`
- `experiment/phase6/run_phase6_cet_gacr_v1.sh`
