# Stage 17 S3：P0 独立正式筛选报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Origin Mode：`run closeout + canonical evidence validation`
- Step：`S17-3`
- 日期：2026-08-30
- 科学终态：`COMPLETED`
- `scientific_completed=true`
- Canonical 科学组合：`artifacts/phase17/s3_exploration/run-0001`
- 机器汇总：`artifacts/phase17/s3_exploration/run-0001/summary.json`
- 冻结结果清单：`artifacts/phase17/manifests/s17_s3_one_epoch_portfolio_seed2023.run-0001.canonical_results.json`
- Verification Status：`VERIFIED_FROM_CANONICAL_RESULTS`
- 数据边界：Toys official validation，`exploration-only`；`test_read=false`，`sports_read=false`
- 结论边界：单 seed、单额外 epoch 的方向筛选；不是独立 fold、多 seed 或论文级确认

## 1. 结论摘要

S17-3 已按冻结预算完成 `GRAM-Continue + 七个 P0 lite + E0 两个必要 control` 共 10 个 arm。10/10 arm 均为 exit code 0，完成一个额外训练 epoch 与 validation，存在 checkpoint、无 traceback、无 forbidden test/Sports 读取，并低于 30 GiB 单 job 规划线。Canonical 科学运行于北京时间 2026-08-29 19:20:31 开始，2026-08-30 11:52:17 完成；双 GPU 动态交接后的科学 wall span 为 `16.53 h`，累计 `27.08 GPU-hours`，最大 process-local CUDA reserved memory 为 `25,322 MiB`。

主要公平对照是同一 epoch-30 parent、fresh matched optimizer、同一额外 epoch 的 `GRAM-Continue`，其 Hit@10=`0.119411`、NDCG@10=`0.075836`。筛选结果为：

1. **A0 BEAR full-vocabulary proxy 是唯一在 Hit@10 与 NDCG@10 上都高于 matched continuation 的 treatment**：Hit@10 `+0.000155`，NDCG@10 `+0.000165`。幅度很小、没有 paired uncertainty，且它不是 legal-Trie BEAR，因此只定为 `WEAK_POSITIVE / PROVISIONAL_D1_CANDIDATE`，不能称为 winner 或 BEAR 复现。
2. **B1 Latte 与 GRAM-Continue 的 NDCG@10 基本持平**（`-0.000023`），但 Hit@10 下降 `-0.001339`，生成 duplicate-path rate 为 `0.259966`。当前 root/path 版本不直接进入 D1，转入 S17-4 的 item scoring / path-ranking 定向诊断。
3. **A1 与 C0 的机制均真实激活，但 accuracy 没有转化**：A1 NDCG@10 `-0.000086`；C0 `-0.000588`。二者保留为 `MECHANISM_ONLY`，只解锁一次预注册的 P1 control/桥接检查，不作为 standalone winner。
4. **B0、D0、E0 semantic 当前 standalone 均负向**。E0 semantic 还同时弱于 full-history 与 random same-size controls，说明本轮没有证据支持“语义选择优于同规模随机选择”；当前实现从 D1 候选中移除。
5. `e0_random_control` 的轻微正值不是方法证据，不能登记成科学方向或进入组合。

本步没有形成已确认 `WINNER`。A0 只是下一独立 fold 的优先候选；其他 track 的 P1 静态映射与定向 smoke 仍按 S17-4 计划执行，不因本轮负结果取消。

## 2. 实验合同与可追溯性

| 项目 | 冻结设置/证据 |
|---|---|
| 数据 | Toys official validation，明确标注 `exploration-only` |
| parent | `GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt` |
| parent SHA256 | `b0d76ea4da9a40b1be43c55c1c7e20cdca4e1eff0194acbbe1b3cc15f471fd82` |
| 历史 parent source commit | `7ac4d9272a57beed9df35c27ea34221f6e4a8fb1` |
| 当前仓库 base commit | `c12271be3d1090646e1c5e36e17f3f276f0ac821` |
| 精确执行源码 | run snapshot manifest SHA256 `688379edad07f1fa1248d8bf1533de3aa8fca3e18ab527eb0887078824386741`；优先于 dirty worktree/base commit 表述 |
| budget | `experiment/phase17/config/s17_s3_formal_budget.json`；SHA256 `5da2cbbc352b9d87945b947f36f676366d595c472ed435129613995825ffd612` |
| frozen portfolio config | SHA256 `e6456f65d99b1457595cf93a8aa37bdfeff35c70e5eb3b1e54fa4c33a2e4f140` |
| data manifest hash | `421a41a9c32f9670dcf0bbf28029a6860227c76d3f8452ad54e42d1b5dc5777c` |
| seed / budget | seed 2023；每 arm 1 个额外 epoch；fresh matched optimizer |
| backbone / decoding | T5-small；native lexical ID；beam 50 |
| GPU | RTX A6000 physical GPU 0/1；每 arm 单卡 |
| validation/test | 自动 last-checkpoint validation；official test 与 Sports 均未读取 |

共同 canonical 命令合同为：

```text
CUDA_VISIBLE_DEVICES=<physical_gpu> \
python ../src/main_generative_gram.py \
  --datasets Toys --seed 2023 --train 1 \
  --rec_epochs 1 --rec_batch_size 16 --gradient_accumulation_steps 8 \
  --rec_lr 1e-3 --beam_size 50 --save_predictions 0 \
  --s17_modules <frozen_module_id> \
  --rec_model_path GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt
```

每个 arm 的完整参数数组冻结在其 `config.json`；运行时没有修改命令。B1 活跃科学进程通过 zero-restart handoff 被接管，交接记录 SHA256 为 `122a0617af4433daea4ac3cd910da1abc45aa277cf7fb5756d1992055745702c`。

## 3. 主结果

以下 `Δ` 均相对 matched `GRAM-Continue`，不是相对历史零额外步 parent。小数差值只作 discovery screen，不表示统计显著性。

| Arm | Track | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | ΔHit@10 | ΔNDCG@10 | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `gram_continue` | CONTROL | 0.089738 | 0.119411 | 0.066284 | 0.075836 | — | — | matched control |
| `b1_latte` | B1 | 0.091696 | 0.118071 | 0.067261 | 0.075813 | -0.001339 | -0.000023 | `HOLD_FOR_PATH_DIAGNOSTIC` |
| `b0_mvi` | B0 | 0.086441 | 0.115290 | 0.063904 | 0.073201 | -0.004121 | -0.002635 | `REJECT_CURRENT_MAPPING` |
| `c0_biflow` | C0 | 0.089584 | 0.118277 | 0.065939 | 0.075249 | -0.001133 | -0.000588 | `MECHANISM_ONLY` |
| `a0_bear_proxy` | A0 | 0.090202 | 0.119565 | 0.066540 | 0.076001 | +0.000155 | +0.000165 | `PROVISIONAL_D1_CANDIDATE` |
| `a1_prefixcurr` | A1 | 0.089481 | 0.119050 | 0.066210 | 0.075750 | -0.000361 | -0.000086 | `MECHANISM_ONLY` |
| `d0_ted` | D0 | 0.088863 | 0.117453 | 0.065912 | 0.075112 | -0.001958 | -0.000724 | `REJECT_CURRENT_MAPPING` |
| `e0_semantic` | E0 | 0.089120 | 0.118432 | 0.065734 | 0.075243 | -0.000979 | -0.000593 | `REJECT_SEMANTIC_SELECTOR` |
| `e0_full_control` | E0 control | 0.089532 | 0.118947 | 0.065952 | 0.075447 | -0.000464 | -0.000389 | control only |
| `e0_random_control` | E0 control | 0.090047 | 0.119823 | 0.066321 | 0.075941 | +0.000412 | +0.000105 | control only；不得升级 |

相对历史 epoch-30 零额外步 NDCG@10=`0.076275`，包括 `GRAM-Continue` 在内的所有 1-epoch arm 均未超过该数值。该历史数值只能作上下文：主因果对照仍是 matched continuation。

## 4. 机制指标与归因

| Track | Canonical 机制指标 | 解释边界 |
|---|---|---|
| A0 | target prefix top-B survival=`1.0`；mean target rank=`3.3402`；mean survival weight=`1.9986` | full-vocabulary proxy 已激活；不能写成 legal-Trie BEAR |
| A1 | active depth=`7/7`；active token fraction=`1.0`；mean per-depth token accuracy=`0.7232` | curriculum 接口有效；单 epoch accuracy 未转化 |
| B0 | 2 paths/item；unique item coverage=`1.0`；duplicate-path rate=`0.169129` | 多路径链路可用，但排序显著负向 |
| B1 | 2 paths/item；unique item coverage=`1.0`；duplicate-path rate=`0.259966` | NDCG 近似持平，路径重复提示 item scoring 瓶颈 |
| C0 | g→s delta norm=`0.1151`；s→g delta norm=`0.0686`；gates=`0.0992/0.1116`；alignment=`0.1626` | 双向总线确实交换信息；未带来 accuracy 收益 |
| D0 | transition teacher coverage=`1.0`；teacher gate=`0.2463`；recent/long gates=`0.5107/0.5206` | teacher 与 gate 活跃；当前蒸馏映射负向 |
| E0 semantic | selected ratio=`0.5628`；filtered ratio=`0.4372`；gate=`0.1675` | 非退化 selector 成立，但弱于两个必要 control |

E0 的归因最清楚：semantic 与 random control 的 selected ratio 相同，但 semantic NDCG@10 比 random 低 `0.000698`；semantic 也比 full-history control 低 `0.000204`。因此不能把“过滤了约 43.7% 历史”本身当成有效机制证据。

## 5. 分组结果与统计边界

Canonical 工件只保存 overall validation metrics，且命令冻结为 `save_predictions=0`；没有 user-level prediction、paired bootstrap、head/mid/tail、history-length、frequency 或 memorization/generalization 分组结果。因此本报告明确登记：

- paired uncertainty：`NOT_AVAILABLE_IN_S17_3_ARTIFACTS`；
- subgroup results：`NOT_AVAILABLE_IN_S17_3_ARTIFACTS`；
- 多比较校正：未执行；本步是 discovery portfolio；
- 小于约 `1e-3` 的差值不得解释为稳定收益；
- 在 S17-5 独立 D1 准入前，runner 必须保存可审计的 user-level prediction 或等价 paired statistics，并补齐计划规定分组。

缺失分组不通过猜测或重读 test 补齐。它是本轮筛选的证据限制，不改变 10 个 canonical arm 已运行完成的事实。

## 6. 尝试台账

S17-3 共登记 15 个 attempt：5 个前置修订 gate + 10 个 canonical one-epoch arms。全部保留在 `artifacts/phase17/attempts/S17-3.attempts.jsonl`，本步不另建零散报告。

| Attempt ID | Track | 类型 | 配置差异/目的 | 终态 | 计入主结果 |
|---|---|---|---|---|---|
| `s3pf_a0_proxy_001` | A0 | revision gate smoke | 修复无 labels generation 旁路 | COMPLETED | 否 |
| `s3pf_a1_prefixcurr_001` | A1 | revision gate smoke | 修复无 labels generation 旁路 | COMPLETED | 否 |
| `s3pf_e0_semantic_001` | E0 | revision gate smoke | 自适应语义 selector 非退化检查 | COMPLETED | 否 |
| `s3pf_e0_full_001` | E0 | revision gate smoke | full-history control | COMPLETED | 否 |
| `s3pf_e0_random_001` | E0 | revision gate smoke | random same-size control | COMPLETED | 否 |
| `s3_gram_continue_001` | CONTROL | one-epoch exploration | matched continuation | COMPLETED | 是 |
| `s3_b1_latte_001` | B1 | one-epoch exploration | Latte root/path lite | COMPLETED | 是 |
| `s3_b0_mvi_001` | B0 | one-epoch exploration | native-token multi-view | COMPLETED | 是 |
| `s3_c0_biflow_001` | C0 | one-epoch exploration | two-bus gated exchange | COMPLETED | 是 |
| `s3_a0_bear_proxy_001` | A0 | one-epoch exploration | full-vocabulary survival proxy | COMPLETED | 是 |
| `s3_a1_prefixcurr_001` | A1 | one-epoch exploration | progressive identifier depth | COMPLETED | 是 |
| `s3_d0_ted_001` | D0 | one-epoch exploration | transition teacher distillation lite | COMPLETED | 是 |
| `s3_e0_semantic_001` | E0 | one-epoch exploration | adaptive semantic subset | COMPLETED | 是 |
| `s3_e0_full_control_001` | E0 | one-epoch exploration | extra full-history branch | COMPLETED | 是，control |
| `s3_e0_random_control_001` | E0 | one-epoch exploration | random same-size subset | COMPLETED | 是，control |

没有科学 arm crash、自动重试或失败 attempt。调度从单卡队列切换为 GPU 0/1 动态队列时，活跃 B1 arm 原进程接管、未重启；后续每个 arm 都有独立 config/result/log/checkpoint。

## 7. 资源与运行完整性

| Arm | Physical GPU | Wall h | Peak reserved MiB |
|---|---:|---:|---:|
| gram_continue | 1 | 2.419 | 25,214 |
| b1_latte | 1 | 2.734 | 25,268 |
| b0_mvi | 0 | 3.166 | 25,214 |
| c0_biflow | 1 | 2.735 | 25,216 |
| a0_bear_proxy | 0 | 2.744 | 25,020 |
| a1_prefixcurr | 1 | 2.744 | 25,228 |
| d0_ted | 0 | 2.244 | 25,322 |
| e0_semantic | 1 | 2.977 | 25,216 |
| e0_full_control | 0 | 2.395 | 25,216 |
| e0_random_control | 1 | 2.920 | 25,216 |

累计 GPU-hours=`27.08`，双 GPU 科学 wall span=`16.53 h`。预算原估算为 `16.67 GPU-hours`，实际更高，主要因为各 arm 的 full validation/beam generation 占比较大；后续 GPU 申请应采用本步实测，而不是继续沿用旧估计。

完整性结论：

- 10 个 result contract 全部通过；
- canonical summary SHA256=`4aba234293a273c53a650ba39bea32c874d2f10b6796f2a78909b29c87ab7a08`；
- canonical 文件树由冻结结果清单逐文件记录 size 与 SHA256；
- exact source/config 由 immutable run snapshot 固定；
- official test、Sports、runtime-cycle 指标均未进入本报告或方法选择；
- 当前仓库存在其他 Phase 16/17 dirty worktree，不能用当前 `git diff` 代替本次 run snapshot。

## 8. 决策与下一门槛

| Track | S17-3 决策 | 下一动作 |
|---|---|---|
| A0 | `WEAK_POSITIVE / PROVISIONAL_D1_CANDIDATE` | 保持 proxy 命名边界；S17-4 完成后，若配置冻结，进入 D1 独立 fold；不得用本轮数值宣称 winner |
| A1 | `MECHANISM_ONLY` | 只允许一次 prefix-aware/PAWA/TreeCL 定向 control；无正信号则停止当前 curriculum |
| B0 | `REJECT_CURRENT_MAPPING` | 不进 D1；S17-4 可检查 item-level scoring 是否直接对应“覆盖有、排序差”的失败表型 |
| B1 | `HOLD_FOR_PATH_DIAGNOSTIC` | 不以当前 root/path 版进 D1；优先 item scoring/set-head 对照，不能无限换 root |
| C0 | `MECHANISM_ONLY` | S17-4 做 concat/one-way/blocked control；只有 accuracy 转正才考虑 full BiFlow |
| D0 | `REJECT_CURRENT_MAPPING` | 不进 D1；graph/DCRec P1 独立 smoke 仍按计划完成，不用 TED gate 活跃替代 accuracy |
| E0 | `REJECT_SEMANTIC_SELECTOR` | semantic、full、random 三者归因已完成；不继续调 selector 追数 |

S17-3 终态为 `COMPLETED`，解锁下一必经门槛 **S17-4：P1 定向轻量迁移**。S17-4 以静态卡片和短 smoke 为主；若升级到正式 D0 screen，按每 arm 单卡、至少 30 GiB 安全容量估算。任何预计超过 4 GPU-hours或需要多卡的新增实验，仍须按计划单独申报，不由本次收尾自动启动。

S17-5 并未因本报告自动全面解锁：只有 A0 进入 provisional candidate pool；A1/C0 等机制方向必须先完成 S17-4 的预注册 control，配置冻结后才能决定是否进入 D1。

## 9. 终态

`S17-3 COMPLETED`。科学任务、attempt ledger、canonical summary、逐 arm 结果、唯一步骤报告和结果冻结清单齐备。当前最强但仍很弱的方向是 A0 full-vocabulary proxy；没有论文级确认、没有已确认跨 fold winner，也没有读取 official test 或 Sports。
