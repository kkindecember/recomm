# Stage16 S16-2 Faithful SpecGR→GRAM 合约与 Admission 报告

> 日期：2026-08-29
> 当前状态：`COMPLETED_SAUX_PASS_SPLUS_PASS_SPLUS_CTRL_MATCHED_EXECUTION_PASS`
> 已通过：`PASS_S16_2_SPECGR_IMPLEMENTATION_CONTRACT_SMALL_SMOKE`、`PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION`、`PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION`、`PASS_S16_2_SPLUS_CTRL_MATCHED_FORMAL_EXECUTION`
> 尚未通过：无；S16-2 contract/admission 已收口。standalone efficacy 属于 S16-4，不在本报告晋升。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-23
- Verification Status: VERIFIED（实现、固定源码、one-step smoke、S-AUX formal training/admission、S-PLUS formal execution、matched CTRL formal execution与跨 attempt artifact contract）
- Version Label: stage16_s2_specgr_faithful_completed_v1

## 1. 当前结论

S16-2 已完成固定版本官方 UniSRec/RecBole 运行链、SpecGR→GRAM clean-room adapters、初期 40 项合约/正式执行/拆分配对测试，以及不超过 10 分钟的 train-only one-step smoke；纳入后续 recovery 与 S16-3 回归后，Stage16 最终全量 `118/118` CPU tests PASS。上一步 S16-1 的 S-AUX 资源代理已被正式源码执行取代：本次实际执行的类来自固定 SpecGR commit 的 `models/draft/UniSRec/model.py`，TransformerEncoder 来自 RecBole v1.2.0 的 `recbole/model/layers.py`。

随后 S-AUX formal a2 完成 train-only internal-dev 选模与固定规模 admission，晋升 `PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION`。S-PLUS/CTRL 的独立 FP32 objective-complete 资源标定 PASS；释放 holder 后的加速 batch sweep 确定 `embedding/generation/accumulation=16/4/64`，峰值 reserved 17,466 MiB。GPU5 a3 完成 S-PLUS 全部 12,535 optimizer steps，GPU7 a4 完成 matched S-PLUS-CTRL 全部 12,535 optimizer steps；one-shot guard 在重复 CTRL 启动后只终止 a3 runner，并由原 terminal trap 恢复 holder。独立 CPU finalizer a2 完成两臂 full-checkpoint SHA、科学配置、起始 checkpoint、预算、finite/admission、防泄漏与 artifact contract 配对，最终晋升 `PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION` 与 `PASS_S16_2_SPLUS_CTRL_MATCHED_FORMAL_EXECUTION`。所有 admission/internal-dev 数值均不是 source validation/test efficacy 结论；S16-2 已完成，后续 efficacy 只在 S16-4 打开。

## 2. 官方运行链

| 组件 | 固定来源 | 状态 |
|---|---|---|
| SpecGR UniSRec | commit `f0ded8884b1df97b5f0599d4ec300bb20b5d1eff` | clean，直接执行官方类 |
| RecBole TransformerEncoder | v1.2.0 / commit `362d31f00af801d7d99bc635c902d1df1405e79d` | clean，直接执行官方层 |
| UniSRec model | 3,555,680 parameters | official forward/backward finite |

RecBole 顶层 initializer 会导入与模型层无关的日志和实验依赖。Stage16 bootstrap 直接加载同一固定仓库中的官方 enum 与 `model/layers.py`，没有复制或修改第三方源码；运行时还检查 Python class 的 source file 必须指回固定官方文件。

S-AUX 唯一内容接口改动是将官方 Amazon sentence embedding 的 768 维输入改成项目已冻结 BGE 的 1024 维输入；`1024→300` 只替换 adaptor 输入宽度。8 experts、2-layer/2-head Transformer、hidden 300、inner 256、temperature 0.07、CE、Adam 及训练参数保持官方配置。该改动属于 S16-0 已冻结的 F1 interface adaptation。

## 3. 实现与测试

新增实现覆盖：

- pinned official UniSRec/RecBole loader；
- S-AUX content-only cold candidate 约束；
- constrained draft without replacement；
- variable lexical path target-aware score；
- strict `score > -1.8` acceptance；
- verifier-prefix guided re-draft、adaptive exit、unique fallback；
- GRAM encoder self-drafter、normalized item index；
- S-PLUS `6×contrastive + 1×generative` pretrain objective；
- S-PLUS fine-tune ranking/generative objective；
- S-PLUS-CTRL 全字段预算一致性审计。

26/26 Stage16 tests 通过，其中包含 2 项 formal train-only transition/isolation 合约；fixed-width、variable lexical、官方 objective 和 source-file identity 均通过。

## 4. One-step train-only smoke

精确命令：

`bash experiment/phase16/run_stage16_s2_specgr_contract_smoke.sh`

运行状态：`COMPLETED`，exit code 0，无自动重试。自动选择物理 GPU 5（RTX A6000），admission free 11,552 MiB、utilization 59%。总 GPU 工作耗时 23.74 秒，最高进程内 allocated 峰值 3,712.55 MiB，低于 8,192 MiB small-experiment Gate。

| Arm | 执行 | 结果 | 峰值显存 |
|---|---|---|---:|
| S-AUX | official UniSRec + official RecBole；batch 64；1 step | loss 8.7018，finite | 419.04 MiB |
| S-PLUS | real frozen GRAM；batch 2；1 joint step | CL 0.3205，Gen 1.0167，Joint 2.9394，均 finite | 3,712.55 MiB |
| S-PLUS-CTRL | 同一 GRAM checkpoint；batch 2；1 generative step | loss 1.1125，finite | 2,081.37 MiB |

S-AUX admission 使用 32 个 train-derived pseudo-cold events；5,963 个 real-cold 全部只通过 content candidate universe 进入，interaction label leak 为 0。S-PLUS 与 CTRL 均只读 S16-1 train manifest；冻结 GRAM checkpoint 的运行前后 SHA 完全一致。

这些 loss 只用于 finite/gradient contract，不能作方法效果比较，尤其不能比较 S-PLUS 2.9394 与 CTRL 1.1125 的大小。

## 5. 正式预算冻结

Toys S-PLUS 使用 27,659 条 train transitions。按官方 Video Games 配置：

- pretrain：100 epochs，embedding/generation effective batch 1024/256，AdamW，lr `1e-3`，weight decay `0.05`，warmup 10,000，10,900 optimizer steps；
- fine-tune：15 epochs，effective batch 256，AdamW，lr `1e-4`，weight decay `0.05`，warmup 10,000，1,635 optimizer steps；
- 单卡 F1 batching 使用 embedding/generation microbatch 4/1、gradient accumulation 256，保持 effective batch；
- S-PLUS-CTRL 使用相同 checkpoint、manifest、epoch、batch、optimizer、step、GPU 数和 timeout，但只使用 GRAM generative objective，不使用 contrastive/ranking/projection/self-drafting objective。

第一个正式大实验 S-AUX 使用单 GPU；经第 9 节官方 batch-2048 单步资源标定后，每卡最低空闲显存由 S16-1 代理估算的 24,576 MiB 修订为 9,216 MiB。原 wall-time 预算为 18–48 小时、48 小时 hard timeout、8 GiB 磁盘；a2 实际因 internal-dev early stopping 在 135.19 秒结束。资源修订只适用于新 attempt，没有回写或覆盖失败的 a1。

## 6. Gate 状态

| Gate | 当前状态 | 原因 |
|---|---|---|
| `PASS_S16_2_SPECGR_IMPLEMENTATION_CONTRACT_SMALL_SMOKE` | PASS | 官方源码链、26 tests、三路 finite smoke 与资源 Gate 均通过 |
| `PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION` | PASS | a2 完成 50 epochs/700 steps、train-only internal-dev early stopping、7,435-event fixed admission、零 cold-label leak 与完整 artifact contract |
| `PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION` | PASS | GPU5 a3 S-PLUS 与 GPU7 a4 matched CTRL 各完成 12,535 optimizer steps；跨 attempt a2 配对合约、完整 checkpoint SHA 与 artifact contract 全部通过 |
| `PASS_S16_2_SPLUS_CTRL_MATCHED_FORMAL_EXECUTION` | PASS | seed、数据、起点、100+15 epochs、effective/physical batch、optimizer、scheduler、steps 与 timeout 一致；仅 physical GPU/artifact root 不同 |

## 7. 工件

- `artifacts/phase16/s2_specgr_contract_smoke/summary.json`
- `artifacts/phase16/s2_specgr_contract_smoke/smoke_summary.json`
- `artifacts/phase16/s2_specgr_contract_smoke/source_manifest.json`
- `artifacts/phase16/s2_specgr_contract_smoke/resource_summary.json`
- `artifacts/phase16/s2_specgr_contract_smoke/code_sha256.json`
- `artifacts/phase16/s2_specgr_contract_smoke/status.json`
- `artifacts/phase16/s2_saux_batch2048_sweep/gpu2_a1/summary.json`
- `artifacts/phase16/s2_saux_batch2048_sweep/gpu2_a1/status.json`
- `artifacts/phase16/s2_saux_formal/toys_seed1502_a2/status.json`
- `artifacts/phase16/s2_saux_formal/toys_seed1502_a2/summary.json`
- `artifacts/phase16/s2_saux_formal/toys_seed1502_a2/artifact_contract.json`
- `artifacts/phase16/s2_splus_objective_resource_sweep/gpu5_a1/status.json`
- `artifacts/phase16/s2_splus_objective_resource_sweep/gpu5_a1/code_sha256.json`
- `artifacts/phase16/s2_splus_objective_resource_sweep/gpu5_a2_fp32/status.json`
- `artifacts/phase16/s2_splus_objective_resource_sweep/gpu5_a2_fp32/summary.json`
- `artifacts/phase16/s2_splus_objective_resource_sweep/gpu5_a2_fp32/code_sha256.json`
- `artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a3_accel_fp32/arms/S-PLUS/summary.json`
- `artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu7_a4_ctrl_split_fp32/summary.json`
- `artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a3_gpu7_a4_duplicate_guard/summary.json`
- `artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a3_gpu7_a4_split_pair/status.json`
- `artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a3_gpu7_a4_split_pair_a2/summary.json`
- `artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a3_gpu7_a4_split_pair_a2/artifact_contract.json`

本文件是 S16-2 唯一报告，当前为 `COMPLETED`；所有失败、阻塞、中断、恢复与最终 PASS attempt 均保留在本文件和各自独立 artifact root，不另建第二份 S16-2 report。

## 8. Formal S-AUX attempt 1：GPU admission failure

用户指定物理 GPU 2。启动前只读检查一度显示空闲 24,990 MiB；正式 runner 于 2026-08-23 21:54:01（Asia/Shanghai）再次检查时，空闲显存已降为 23,906 MiB，低于冻结门槛 24,576 MiB。

- exact command：`bash experiment/phase16/run_stage16_s2_saux_formal.sh 2`；
- attempt：`s16_s2_saux_toys_a1`；
- final status：`GPU_ADMISSION_FAILED`，exit code `9`；
- workload PID：0，optimizer progress：0/4,200；
- formal training、internal-dev evaluation、checkpoint 均未启动或生成；
- 没有终止或修改 GPU 2 上既有进程；
- `test_read=false`，`automatic_retry=false`。

该结果只表示 admission 瞬时资源不足，不是算法、实现或科学 Gate failure。失败 attempt 保留在 `artifacts/phase16/s2_saux_formal/toys_seed1502_a1/status.json`；任何后续运行必须使用新的 attempt 目录，不能覆盖 a1。

## 9. 官方 batch-2048 单步资源标定

用户确认在物理 GPU 2 上执行一次独立资源标定，以核验 24,576 MiB 是否只是 S16-1 资源代理产生的过度保守门槛。标定在运行前冻结如下规则：使用完整 4,799 项训练目录、formal batch `2048`、一个 Adam forward/backward/step；以进程 `torch.cuda.max_memory_reserved` 为基准，安全余量取 `max(4096 MiB, 50% × peak_reserved)`，总值向上取整至 1,024 MiB，最低建议值为 8,192 MiB。只有 loss finite、规模字段完全匹配、peak reserved 不超过 8,192 MiB 且不读 validation/test 时才可下调。

- exact command：`bash experiment/phase16/run_stage16_s2_saux_batch2048_sweep.sh 2`；
- attempt：`s16_s2_saux_batch2048_gpu2_a1`，exit code `0`，无自动重试；
- official runtime：固定 SpecGR UniSRec + RecBole v1.2.0；
- admission free：23,626 MiB；batch：2,048；train catalog：4,799；train transitions：27,659；
- loss：`8.6059427261`，finite；optimizer step：1.845 秒；总运行：14.24 秒；
- peak allocated：3,440.69 MiB；peak reserved：4,314 MiB；
- safety margin：4,096 MiB；向上取整后的建议最低空闲显存：`9,216 MiB`；
- `test_read=false`，`validation_used=false`，未停止、修改 GPU 2 上既有进程；
- verdict：`PASS_S16_2_SAUX_BATCH2048_MEMORY_SWEEP`，`recalibration_eligible=true`。

因此，24,576 MiB 不是官方 UniSRec 在当前 Toys/formal batch 上的硬需求，而是缺少真实大 batch 测量时的保守代理值。9,216 MiB 包含相对 4,314 MiB reserved 峰值的 4,096 MiB 额外余量；该 sweep 仅是资源证据，不是 efficacy metric，也没有单独使 `PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION` 晋升。随后 formal a2 使用独立配置/runner 和新目录完成，见第 11 节。

## 10. Formal S-AUX attempt 2：GPU 2 启动前资源等待（历史记录）

用户随后确认继续使用 GPU 2 与重标后的 9,216 MiB admission。已生成独立 a2 配置和 runner，并通过 shell syntax、Python compile、2/2 formal train-only data-contract tests；a1 没有被覆盖。预计命令为 `bash experiment/phase16/run_stage16_s2_saux_formal_a2.sh 2`，后台 session 为 `phase16_s2_saux_a2_gpu2`。

实际启动前的两次只读 GPU 2 快照分别为：

- 第一次：free 9,035 MiB、utilization 76%，比门槛少 181 MiB；
- 30 秒后：free 3,927 MiB、utilization 55%；
- GPU 2 当时存在三个既有外部进程，分别约占 19,370、5,548、19,694 MiB；未终止或修改任何进程。

第二次 free 已低于 4,314 MiB 的实测 peak reserved，本轮按 GPU 安全规则在加载模型前停止。tmux session、runner 和 workload 均未启动，workload PID 为 0；这不是 a2 算法或 scientific Gate failure，也没有自动重试。当时 a2 工件目录尚未创建；此记录随后被第 11 节用户授权的 GPU 5 正式启动取代。

## 11. Formal S-AUX attempt 2：GPU 5 正式完成并恢复 holder

用户明确授权临时释放其 GPU 5 项目 holder，并要求 formal 结束后使用同一方法恢复约 20 GiB 占位。只读核验确认 holder session 为 `gram_ablation_scan_gpu5`，worker PID `2083287`，原始参数 `reserve_mib=18263`，实际稳定占用约 20,292 MiB；控制器为 `tools/gram_ablation_scan.sh`。用户提供的 Stage15 Beauty B2 status 与 GPU 5 进程均只读保留，没有停止或修改 Stage15 workload 及其他用户进程。

a2 runner 已升级为原子资源控制器：精确验证 holder PID/命令行/state/session 后才正常释放；`EXIT` trap 覆盖 completed、failed、timeout 和 TERM/INT/HUP 路径，使用原 session、state root、`reserve_mib=18263` 恢复，并要求新 PID 的实际占用至少 19,000 MiB。恢复失败会将 final status 改为 `HOLDER_RESTORE_FAILED`，不会静默忽略。

- background command：`tmux new-session -d -s phase16_s2_saux_a2_gpu5 'cd /mnt/18T/jiangtangyunzhi/projects/recomm && bash experiment/phase16/run_stage16_s2_saux_formal_a2.sh 5'`；
- started at：2026-08-23 23:22:51（Asia/Shanghai）；
- holder released at：23:22:53；initial PID `2083287`；reserve `18,263 MiB`；
- post-release admission free：31,849 MiB；minimum：9,216 MiB；
- runner/workload PID：`445804` / `446318`；physical GPU 5 → visible `cuda:0`；
- 终态：`COMPLETED`，exit code 0；50/300 epochs 后按 patience 40 early stop，共 700 optimizer steps；runtime 135.19 秒；
- best epoch 10；internal-dev selection NDCG@10 `0.01582898`；final fixed admission 的 Hit@50/NDCG@10/MRR 为 `0.10880968/0.01582898/0.01413622`，7,435 events、11,924 candidates，全部 finite；
- train transitions 27,659；pseudo-cold/real-cold items 1,162/5,963；cold interaction label leaks 0；content embedding SHA 前后完全一致；
- process peak allocated/reserved：3,481.60/4,320 MiB；artifact contract：`PASS_SAUX_FORMAL_ARTIFACT_CONTRACT`；
- `test_read=false`，`automatic_retry=false`；48 h hard timeout；
- holder restored at：23:25:40；新 PID `464054`；同一 `reserve_mib=18263`；即时验证 20,276 MiB，随后稳定为 20,292 MiB；formal tmux session 已结束、`gram_ablation_scan_gpu5` session 正常存在。

最终 verdict 为 `PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION`，holder terminal contract 同时完成。未读取 validation/test，未自动重试，也未停止 Stage15 Beauty B2 或其他用户进程。上述 internal-dev admission 数值不作 standalone efficacy promotion；S16-2 整体仍须等待 S-PLUS/CTRL formal Gate。

## 12. S-PLUS/CTRL objective-complete 资源 sweep attempt 1：bf16 non-finite

S-AUX Gate 通过后继续进行 S-PLUS/CTRL formal 资源复核。该 sweep 预注册完整 Toys 预算：27,659 transitions；pretrain 100 epochs、embedding/generation physical microbatch `4/1`、accumulation 256、2,765,900 physical microsteps、10,900 optimizer steps；finetune 15 epochs、microbatch 1、accumulation 256、414,885 physical microsteps、1,635 optimizer steps。测量计划覆盖 S-PLUS/CTRL 的 pretrain/finetune 四条路径、AdamW state 和完整 4,799-item frozen index，不生成 efficacy metric。

- exact command：`bash experiment/phase16/run_stage16_s2_splus_objective_resource_sweep.sh 5`；
- attempt：`s16_s2_splus_resource_gpu5_a1`；
- GPU 5 admission free：11,560 MiB；holder PID `464054` 全程保留，`holder_released=false`；
- 26/26 Stage16 tests PASS；
- final status：`FAILED`，exit code 1；无自动重试；
- failure point：第一个 S-PLUS pretrain objective-complete microstep；`FloatingPointError: Non-finite S-PLUS pretrain sweep loss`；
- 未进入 CTRL、finetune 或 full-item-index 阶段；未生成 `summary.json`；
- `test_read=false`，原 GRAM checkpoint 未被原地写入。

当前根因标记为 `BF16_RUNTIME_COMPATIBILITY_SUSPECTED_NOT_PROVEN`。官方 SpecGR runner 使用 `precision='bf16-mixed'`，本地冻结 `gram-repro` 环境为 PyTorch `1.11.0+cu113` / cuDNN 8200，缺少当前版 `torch.cuda.is_bf16_supported()` 接口；同一 GRAM joint objective 在此前 FP32 one-step smoke 中 finite。首批四个 contrastive targets 互异，因此没有重复正例造成退化的直接证据。以上只支持“旧软件栈下 bf16 路径高度可疑”，尚未用新 attempt 区分 embedding loss 与 generation loss 的具体非有限来源。

按 no-auto-retry 规则，a1 保留在 `artifacts/phase16/s2_splus_objective_resource_sweep/gpu5_a1/`，并补充了包含当时代码哈希的 post-failure traceability manifest。用户随后明确确认独立 FP32 a2；该 attempt 的结果见第 13 节。

## 13. S-PLUS/CTRL objective-complete 资源 sweep attempt 2：FP32 PASS

用户确认只将执行精度从 `bf16-mixed` 改为 `fp32`，算法 objective、数据、batch、accumulation、optimizer、step、matched-control 预算与 GPU 均不变。a2 使用新配置、runner 和输出目录，不覆盖 a1；现有 GPU 5 holder 全程保留。

- exact command：`bash experiment/phase16/run_stage16_s2_splus_objective_resource_sweep_a2_fp32.sh 5`；
- attempt：`s16_s2_splus_resource_gpu5_a2_fp32`；2026-08-24 00:46:55–00:47:31；exit code `0`；
- admission free：11,552 MiB；minimum free：10,240 MiB；600 秒 hard timeout；无自动重试；
- 26/26 Stage16 tests PASS；四条 objective-complete 路径及各自 AdamW step 全部 finite；完整 frozen train item index 覆盖 4,799 items；
- S-PLUS pretrain：median microstep 0.16756 秒，optimizer step 0.02223 秒，peak allocated/reserved 4,731.55/4,978 MiB；
- CTRL pretrain：0.05710/0.04618 秒，peak reserved 1,732 MiB；
- S-PLUS finetune：0.20361/0.02341 秒，full index build 2.0623 秒，peak reserved 2,482 MiB；
- CTRL finetune：0.10125/0.02507 秒，peak reserved 1,734 MiB；
- 全局 peak reserved：4,978 MiB；按 `ceil_to_1024(peak + 4096 MiB)` 得到建议最低空闲显存 `9,216 MiB`；
- 冻结 GRAM checkpoint SHA 前后均为 `d71fcf5...3048550`；`test_read=false`、`validation_used=false`、`network_used=false`、未开始 formal training、未产出 efficacy metric；
- holder PID 始终为 `464054`，原 `reserve_mib=18263`，终态实测占用 20,292 MiB，`holder_released=false`；
- verdict：`PASS_S16_2_SPLUS_OBJECTIVE_RESOURCE_SWEEP`。

由实测 median physical microstep 与 AdamW step 外推，双臂核心预算为 `207.97 GPU·h`：S-PLUS pretrain/fine-tune 128.81/23.48，CTRL pretrain/fine-tune 44.01/11.68 GPU·h。包含 dataloading、internal-dev evaluation、checkpoint 与 scheduler overhead 的保守预算为 `259.97–415.95 GPU·h`，单卡约 10.83–17.33 天。显存建议为 1 GPU、至少 9,216 MiB free；工程上继续保留当前 10,240 MiB admission 更稳妥。

磁盘只作工程估算：冻结起始 checkpoint 为 242,132,665 bytes（约 231 MiB）；按每臂至少一个含 model+Adam state 的 resumable state 与 final/best checkpoint 估算，双臂约 3 GiB，建议预留 8 GiB。正式 runner 的 checkpoint cadence 与恢复策略尚须在启动前冻结，因此该磁盘数不是实测 formal artifact size。

FP32 是旧 PyTorch 1.11/CUDA 11.3 软件栈的披露性执行适配；它解决 a1 的非有限问题，但不能证明 bf16 的具体失败组件，也不能将资源 sweep 晋升为 scientific Gate。本节形成时的下一动作是由用户明确授权约 260–416 GPU·h 的正式训练；随后授权与启动记录见第 14 节。

## 14. S-PLUS/CTRL formal GPU5 attempt 1：preprocessing compatibility failure

用户于 2026-08-24 明确确认继续。结合此前连续 GPU5 指令，本次确认按“物理 GPU5 单卡顺序执行、FP32、保留现有 holder、接受 259.97–415.95 GPU·h”执行；启动前已向用户逐项披露该解释、精确命令、输出、成功判据和每臂 14 天 hard timeout。

- exact foreground command：`bash experiment/phase16/run_stage16_s2_splus_ctrl_formal_toys_gpu5_a1_fp32.sh 5`；
- background session：`phase16_s2_splus_ctrl_formal_gpu5_a1`；started at 2026-08-24 01:15:47（Asia/Shanghai）；
- attempt：`s16_s2_splus_ctrl_formal_toys_gpu5_a1_fp32`；输出 `artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a1_fp32/`；
- 顺序：S-PLUS pretrain 100 epochs → finetune 15 epochs → S-PLUS-CTRL matched pretrain 100 epochs → finetune 15 epochs；总计 25,070 optimizer steps；
- scheduler 忠实边界：cosine；官方 raw warmup argument 10,000，经单卡 accumulation 256 映射为 39 optimizer warmup steps；官方整除 scheduler totals 为 pretrain/fine-tune 10,800/1,620，实际 epoch 尾批刷新 totals 为 10,900/1,635；两组值均冻结并记录；
- checkpoint：每 2 个 pretrain epoch、每 1 个 finetune epoch 原子覆盖 last state，同时保存 stage final；包含 model/drafter/optimizer/scheduler/RNG/epoch/step，禁止自动 resume；
- admission：3,108 internal-dev generation events；S-PLUS 另覆盖 7,435 pseudo-cold events 与完整 11,924-item content-only index；均不是 efficacy promotion；
- 启动 admission free 11,560 MiB；29/29 Stage16 tests PASS；formal runner/workload PID `1135434/1136093`；
- 初始状态：`RUNNING` / S-PLUS preprocessing，process alive；尚未产生 optimizer step；
- holder PID `464054`、`reserve_mib=18263`、实测约 20,292 MiB，`holder_released=false`；
- 每臂相同 hard timeout 1,209,600 秒（14 天），每 60 秒 heartbeat/telemetry；除 hard timeout 外不自动 kill，不自动重试；
- `test_read=false`、`validation_used=false`。

终态于 2026-08-24 01:17:51 写入：`FAILED`，exit code `1`，progress `0/25,070`，无自动重试。失败发生在三类 CPU token cache 完成之后、加载 GRAM 之前的 CUDA telemetry 初始化：冻结环境 PyTorch `1.11.0+cu113` 的 `torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))` 抛出 `RuntimeError: Invalid device argument`。该旧 API 在本环境需要传整数 device index；resource sweep 中调用发生在模型已加载/上下文已初始化后，因而没有暴露这一时序兼容问题。

这是已证实的运行时 API compatibility failure，不是数据、算法 loss、显存或 scientific Gate failure。S-PLUS/CTRL 模型均未加载，forward/backward/optimizer/checkpoint 均未开始；CTRL 未启动，未生成 paired summary。`test_read=false`、`validation_used=false`，GPU5 holder PID `464054` 仍稳定约 20,292 MiB 且从未释放。a1 工件保留，不覆盖。用户确认独立 a2 后，GPU5 preflight 进一步证明“只传整数 index”仍失败，而 `torch.cuda.init()` 后再以整数 index reset 成功；因此真实最小修复冻结为 context-init-then-reset，其余代码/预算不变。

## 15. S-PLUS/CTRL formal GPU5 attempt 2：context-init patch 后由用户授权中断

用户明确确认独立 a2。a2 不复用或覆盖 a1 输出；a1 `status.json` / `run.log` SHA 分别冻结为 `466557b...09a` / `a0a2d3...02d8`。a1/a2 除 attempt provenance、输出/命令和唯一 compatibility patch 外的科学配置逐字段相同。

- 唯一代码语义改动：在 peak-memory reset 前调用 `torch.cuda.init()`，随后以整数 visible device index 调用 PyTorch 1.11 allocator API；算法 objective、数据、batch、scheduler、optimizer、step 和 admission 均不变；
- GPU5 preflight：integer-index-only 仍复现 `Invalid device argument`；context-init-then-reset PASS；31/31 Stage16 tests PASS，其中包含调用顺序 mock contract；
- exact foreground command：`bash experiment/phase16/run_stage16_s2_splus_ctrl_formal_toys_gpu5_a2_fp32.sh 5`；
- tmux：`phase16_s2_splus_ctrl_formal_gpu5_a2`；started at 2026-08-24 02:07:42（Asia/Shanghai）；
- output：`artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a2_fp32/`；runner/workload PID `1412452/1413349`；
- admission free：11,560 MiB；holder PID `464054`、`reserve_mib=18263`，`holder_released=false`；
- 已越过 a1 failure point；当前 S-PLUS pretrain，stage optimizer step `1/10,900`，paired `1/25,070`，process alive；
- 每臂相同 14 天 hard timeout，60 秒 heartbeat/telemetry；不自动 retry/resume，只有 hard timeout 自动终止；
- `test_read=false`、`validation_used=false`，formal Gate 仍为 PENDING。

用户随后要求完全释放 GPU5 holder 并以更大物理 microbatch 加速。经明确确认后，runner 于 2026-08-24 02:21:05 正常收到中断信号并写入 `INTERRUPTED`，exit code 143；最终保留 progress `15/25,070`，runner/workload 与 tmux 会话均退出，a2 目录、状态和日志未删除或覆盖。该终止是用户授权的工程切换，不是算法或科学失败，也没有自动重试。

## 16. GPU5 holder 全释放加速 batch sweep

用户明确要求 formal 运行期间不保留 20G holder，只在实验结束后用同一方法恢复。a2 停止后核验 holder PID `464054`、`reserve_mib=18263`、session/state/controller，再只停止该 holder；GPU5 空闲显存由约 16.4 GiB 增至 36,705 MiB，未停止其他进程。

预注册候选按吞吐优先顺序为 `64/16/16`、`32/8/32`、`16/4/64`、`8/2/128`，每组均保持 effective embedding/generation batch `1024/256`，执行真实 FP32 S-PLUS joint forward/backward/AdamW step，不读取 validation/test，不产生 efficacy metric。峰值 eligibility ceiling 为 28,672 MiB。

- sweep a1 的 `64/16/16` 在 PyTorch reserved 约 30.04 GiB 时 OOM；旧 PyTorch 1.11 不存在 `torch.cuda.OutOfMemoryError` 类，导致 wrapper 将已观察 OOM 记为 compatibility error；终态 holder 正常恢复为 PID `1514419`；
- recovery a2 明确禁止重跑 `64/16/16`，仅从下一个预注册候选继续；`32/8/32` 在 reserved 约 31.12 GiB 时 OOM；
- `16/4/64` PASS：三次 joint loss/gradient 均 finite，暖态 microstep median `0.485317 s`，optimizer step `0.152621 s`，peak allocated/reserved `16,886.89/17,466 MiB`，GRAM checkpoint SHA 前后不变；
- recovery exact command：`bash experiment/phase16/run_stage16_s2_splus_accelerated_batch_sweep_a2_recovery.sh 5`；终态 `COMPLETED`、exit code 0；holder 恢复为 PID `1533786`、同一 `reserve_mib=18263`。

最终 verdict 为 `PASS_S16_2_SPLUS_ACCELERATED_BATCH_SWEEP`。这只支持 F1 batching adaptation，不是 a2 的逐位续跑：物理 batch 变化会改变 dropout 分组；数据、epochs、effective batch、objective、optimizer、scheduler 和 optimizer-step 总数保持不变。

## 17. S-PLUS/CTRL formal GPU5 attempt 3：加速 batch、holder 全释放（历史启动记录）

a3 使用独立 overlay/resolved config、runner 和输出目录，不覆盖 a1/a2。加速入口将 27,659 个 epoch 样本切为 6,915 个 generation microsteps；每 64 个 microsteps 刷新一次，仍为每 epoch 109 optimizer steps。尾窗为 3 microsteps/11 samples，generation/ranking loss 按样本数加权；embedding microbatch 通过确定性循环保持 16 条满 batch，并按实际 microstep 数归一。4 项新增 batching/tail tests 与全部 35 项 Stage16 tests PASS，9 个 code-freeze SHA 一致。

- exact command：`bash experiment/phase16/run_stage16_s2_splus_ctrl_formal_toys_gpu5_a3_accel_fp32.sh 5`；
- tmux：`phase16_s2_splus_ctrl_formal_gpu5_a3_accel`；started at 2026-08-24 02:36:07（Asia/Shanghai）；
- output：`artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a3_accel_fp32/`；runner/workload PID `1573426/1574495`；
- holder initial PID `1533786` 已由 runner 核验并完全释放；post-release admission free `33,995 MiB`，`holder_released=true`；
- 当前 arm：S-PLUS pretrain；physical embedding/generation batch `16/4`、accumulation 64；effective batch `1024/256`；截至 02:40:00 已完成 `3/25,070` paired optimizer steps；首步后的两个完整窗口平均约 `36.83 s/optimizer step`；
- runner 的 EXIT terminal controller 覆盖 completed、failed、timeout、TERM/INT/HUP 和 preflight terminal paths；恢复目标固定 session/state/controller 与 `reserve_mib=18263`，恢复失败会写 `HOLDER_RESTORE_FAILED`；
- 每臂 14 天 hard timeout；`test_read=false`、`validation_used=false`、`automatic_retry=false`。

本节只记录当时已核验的启动、资源释放和批处理合约；本节形成时 formal Gate 为 PENDING。最终 arm、guard 与 paired artifact contract 终态见第 19 节。

## 18. GPU7 S-PLUS-CTRL 并行拆分（历史启动记录）

用户于 2026-08-28 先明确同意准备 GPU7 并行 `S-PLUS-CTRL` 拆分方案，随后又明确确认“在 GPU7 启动 a4”。启动前实时闸门确认 GPU7 free 48,568 MiB、利用率 0%，a3 仍为 `S-PLUS/running` 且原串行 CTRL 未创建 checkpoint/summary；GPU5 a3、holder 及其 artifacts 均未修改、暂停或终止。

拆分的科学边界固定为：GPU5 a3 继续产生 `S-PLUS` arm；GPU7 a4 从 a3 已冻结的 `resolved_config.json` 派生，只产生 `S-PLUS-CTRL` arm。seed、GRAM checkpoint、train/internal-dev manifests、100+15 epochs、effective batch 1024/256、physical batch 16/4、accumulation 64、AdamW、cosine scheduler、optimizer steps 与每臂 14 天 timeout 必须 exact match。唯一允许差异是同型号 RTX A6000 的 physical GPU `5→7` 和隔离 artifact root；这属于 F1 execution-layout adaptation，不改变算法 objective。

a4 启动配置：

- exact foreground command：`bash experiment/phase16/run_stage16_s2_splus_ctrl_formal_toys_gpu7_a4_split_fp32.sh 7`；
- executed tmux command：`tmux new-session -d -s phase16_s2_splus_ctrl_formal_gpu7_a4_split 'cd /mnt/18T/jiangtangyunzhi/projects/recomm && bash experiment/phase16/run_stage16_s2_splus_ctrl_formal_toys_gpu7_a4_split_fp32.sh 7'`；
- output：`artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu7_a4_ctrl_split_fp32/`；
- status：`artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu7_a4_ctrl_split_fp32/status.json`；
- admission：GPU7 至少 28,672 MiB free；expected peak reserved 17,466 MiB；8 GiB disk；14 天 hard timeout；
- 运行期不操作任何 holder，不自动 retry/resume，不读取 validation/test；
- 启动 hard guard：若 a3 的 serial `S-PLUS-CTRL` 已创建 checkpoint 或 summary，a4 以 `PARENT_CTRL_ALREADY_STARTED` 阻塞，不制造重复 formal workload。

跨 attempt 配对命令冻结为 `bash experiment/phase16/run_stage16_s2_splus_ctrl_split_pair_finalize.sh`。它只在 a3 `S-PLUS` arm summary 与 a4 `S-PLUS-CTRL` arm summary 均 PASS 后执行，将配对结果写入独立目录 `artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a3_gpu7_a4_split_pair/`。finalizer 逐字段比较 scientific config、pretrain/finetune budget、起始 checkpoint、step count、finite/internal-dev admission、pseudo-cold full-catalog admission、peak memory ceiling、sealed-data flags 与恢复 checkpoint；两侧 source artifacts 全程只读。

启动结果：Stage16 全量 `40/40` tests PASS，GPU7 resolved config 与 GPU5 a3 scientific core exact match，split preflight 返回 `PASS_S16_2_SPLUS_CTRL_SPLIT_PREFLIGHT`。a4 于 `2026-08-28T10:33:33+08:00` 启动，tmux `phase16_s2_splus_ctrl_formal_gpu7_a4_split`、runner/workload PID `1708057/1708947`；admission free 48,568 MiB。10:35 复核时两个 PID 均存活，`status=running`、`current_arm=S-PLUS-CTRL`、progress `0/12,535`，处于 CPU preprocessing，尚未创建 checkpoint/summary；`test_read=false`、`validation_used=false`、`automatic_retry=false`。GPU5 a3 同时仍为 S-PLUS，已推进至至少 epoch 94 / 10,247 paired optimizer steps。该启动快照当时为 `RUNNING`、formal Gate 为 PENDING；最终终态见第 19 节。

10:37:35 首步复核时 a4 已从 preprocessing 进入 pretrain，`progress.json` 达到 14/12,535 CTRL optimizer steps；runner/workload PID 均存活。同期 GPU5 a3 达到至少 10,250/25,070 paired optimizer steps，仍为 S-PLUS。`status.json` 的 4-step 心跳快照早于 progress 文件，不构成进度倒退。

原 a3 runner 仍保留其串行 CTRL 逻辑，本次不会自动终止它。若 a3 的 S-PLUS arm 先完成并准备进入重复 CTRL，必须先核验 S-PLUS arm checkpoint/summary 完整性，再取得用户明确授权后才能中断该重复 arm；a4 与跨 attempt finalizer 不承担自动 kill。

### 18.1 GPU5 重复 CTRL one-shot guard（历史启动记录）

用户询问能否将重复 CTRL 去重写成自动脚本后，新增 `splus_ctrl_duplicate_guard.py`、冻结配置、8 项定向测试与 exact runner。守卫为 fail-closed one-shot：只有以下条件同时成立才向 a3 runner PID 发送一次 SIGTERM，而不直接 signal workload：

- a3 `arms/S-PLUS/summary.json` 为 completed PASS，12,535 steps、internal-dev/pseudo-cold admission、checkpoint parity、sealed-data 与 4 个恢复 checkpoint 全部通过；
- a3 status 已明确切换到 `S-PLUS-CTRL`，冻结 runner PID `1573426`、start ticks `947916462`、exact cmdline 均匹配；未来 CTRL workload 必须是该 runner 的直接子进程且命令行精确指向 a3 overlay 与 `--arm S-PLUS-CTRL`；
- GPU7 a4 要么仍为 heartbeat/progress 新鲜的健康 CTRL，且 runner/workload PID、start ticks、cmdline 全匹配，要么已经形成完整 CTRL PASS artifact；
- a4 失败/timeout/blocked、心跳或进度陈旧、summary/checkpoint 不完整、配置/SHA/PID 身份漂移时，一律不 signal，保留 GPU5 原生 CTRL 后备；
- SIGTERM 后由 a3 原 terminal trap 终止 child 并以 `reserve_mib=18263` 恢复 `gram_ablation_scan_gpu5`；guard 只有在 a3 `INTERRUPTED`、runner 退出、holder state/PID/cmdline/tmux 全部恢复后才 PASS，否则标记 manual attention。

纯 CPU 定向测试 `8/8`、Stage16 全量 `66/66` PASS；真实无 armed `check` 返回 `WAIT / a3 S-PLUS is still running`，`signal_sent=false`。exact foreground command 冻结为 `bash experiment/phase16/run_stage16_s2_splus_ctrl_duplicate_guard_gpu5_a3_gpu7_a4.sh`；background command 冻结为 `tmux new-session -d -s phase16_s2_splus_ctrl_duplicate_guard_gpu5_a3_gpu7_a4 'cd /mnt/18T/jiangtangyunzhi/projects/recomm && bash experiment/phase16/run_stage16_s2_splus_ctrl_duplicate_guard_gpu5_a3_gpu7_a4.sh'`。

用户随后明确确认启动自动守卫。最终闸门确认 a3/a4 分别仍为健康的 S-PLUS/S-PLUS-CTRL、目标 guard session/summary/armed.lock 均不存在、所有冻结 SHA 与主机 `/proc` 身份匹配后，background command 于 `2026-08-28T11:04:58+08:00` 执行。tmux `phase16_s2_splus_ctrl_duplicate_guard_gpu5_a3_gpu7_a4`、guard PID `1807506`、`armed.lock` 均已建立，启动 8/8 tests PASS；跨多个 20 秒轮询周期状态持续为 `WAIT / a3 S-PLUS is still running`、`signal_sent=false`。11:26 复核时 a3/a4 已分别推进至至少 10,331/25,070 paired steps 与 527/12,535 CTRL steps，训练未受守卫影响。该时点状态为 `ARMED_RUNNING`；最终 guard PASS 见第 19 节。

## 19. S-PLUS/CTRL 最终配对与 S16-2 收口

### 19.1 两臂正式终态

GPU5 a3 的 S-PLUS arm 完成全部 100-epoch pretrain 与 15-epoch finetune：

- optimizer steps：10,900 + 1,635 = `12,535/12,535`；
- physical microsteps：691,500 + 103,725；
- internal-dev generation events：3,108，mean loss `3.0295813084`，全部 finite；
- fixed pseudo-cold admission：7,435 events、11,924 candidates，全部 finite，cold interaction label leak 为 0；
- peak CUDA reserved：`17,506 MiB`；起始/终态 GRAM checkpoint SHA 均为 `d71fcf5...3048550`；
- verdict：`PASS_S16_2_S_PLUS_FORMAL_EXECUTION`；`test_read=false`、`validation_used=false`。

a3 随后进入原 runner 的重复 CTRL 路径。one-shot guard 在 S-PLUS summary/checkpoints、a3/a4 进程身份与 GPU7 CTRL 健康性全部 fail-closed 通过后，只向 a3 runner 发送一次 SIGTERM；a3 原 EXIT trap 终止其 child 并恢复同一 `reserve_mib=18263` holder。a3 顶层 status 因受控去重保持 `INTERRUPTED`，但已经完成且不可变的 S-PLUS arm summary 保留为配对权威来源。guard 终态为 `PASS_S16_2_DUPLICATE_CTRL_GUARD`，GPU7 a4 未被修改。

GPU7 a4 的 S-PLUS-CTRL arm 于 2026-08-29 15:19:06+08:00 正常完成：

- optimizer steps：10,900 + 1,635 = `12,535/12,535`；
- physical microsteps：691,500 + 103,725；
- internal-dev generation events：3,108，mean loss `2.9443118572`，全部 finite；
- 按冻结 matched-control 合约不生成 pseudo-cold efficacy admission；
- peak CUDA reserved：`4,536 MiB`；起始/终态 GRAM checkpoint SHA 均为 `d71fcf5...3048550`；
- verdict：`PASS_S16_2_S_PLUS_CTRL_FORMAL_EXECUTION`；exit code 0；`test_read=false`、`validation_used=false`。

### 19.2 CPU-only 配对 recovery

第一次 exact command `bash experiment/phase16/run_stage16_s2_splus_ctrl_split_pair_finalize.sh` 于 2026-08-29 18:47:15+08:00 启动。5/5 tests PASS 后，finalizer 对 8 个共约 3.7 GiB 的 source checkpoints 做完整 SHA-256；存储读取在 600 秒原 hard timeout 内未结束，于 18:57:15 由 `timeout` 终止，exit code 3。runner 的通用状态码为 `SOURCE_OR_PAIR_CONTRACT_FAILED`，但精确 600 秒边界、已写出的前置 manifests、缺失 recovery/summary/artifact contract 与空日志 traceback 共同定位为 CPU hash timeout，不是 source/scientific contract failure。该 blocked attempt 原样保留，未修改 source artifacts，也未自动重跑。

用户确认后建立独立 a2，只做如下工程恢复：

- 新 attempt/output root，不覆盖 blocked attempt；
- full checkpoint SHA-256 保持，CPU pair-finalizer timeout `600→1800` 秒；
- 两臂 source attempts、科学配置、数据、checkpoint、Gate 与 finalizer Python 实现全部不变；
- timeout 与 source-contract failure 分开写机器状态；定向 `8/8`、Stage16 全量 `118/118` CPU tests PASS。

a2 exact command：

`bash experiment/phase16/run_stage16_s2_splus_ctrl_split_pair_finalize_a2.sh`

a2 于 2026-08-29 19:03:47–19:05:41+08:00 完成，exit code 0。finalizer 逐项确认：

- S-PLUS 与 CTRL scientific core 完全相同，仅同型号 RTX A6000 的 physical GPU 5/7 与隔离 artifact root 不同；
- 两臂 dataset manifest、100+15 epochs、effective batch 1024/256、physical batch 16/4、accumulation 64、AdamW、learning rate、weight decay、scheduler、warmup、optimizer steps、GPU count 与 timeout 完全匹配；
- 同一起始 GRAM checkpoint，base checkpoint 均未修改；所有 stage loss/admission finite；
- S-PLUS 完整 pseudo-cold full-catalog admission、CTRL 禁止 efficacy admission、零 cold-label leak 与 sealed-data flags 均通过；
- 8 个 source recovery checkpoints 存在且完成 SHA-256；source artifacts 未修改；
- artifact contract：`PASS_SPLUS_CTRL_SPLIT_PAIR_ARTIFACT_CONTRACT`。

最终 paired verdict 为 `PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION`，matched-control execution verdict 为 `PASS_S16_2_SPLUS_CTRL_MATCHED_FORMAL_EXECUTION`。连同既有 `PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION`，S16-2 的 implementation、contract 与 admission 完整收口。本步骤没有读取 source validation/test，也没有产生或宣称 standalone efficacy；S-PLUS 与 S-AUX 是否改善 cold recommendation 只能由后续 S16-4 frozen validation 判定。
