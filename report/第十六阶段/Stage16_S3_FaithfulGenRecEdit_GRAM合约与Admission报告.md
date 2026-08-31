# Stage16 S3：Faithful 终态与 GenRecEdit-inspired G-RIDGE→GRAM 合约/Admission 报告

> 状态：`COMPLETED / S16_3F_RESOURCE_A3_TIMEOUT_PRESERVED / S16_3F_TERMINAL_LINEAR_SYSTEM_BLOCKED / S16_3B_B1_FAILED_PRESERVED / S16_3B_RECOVERY_C1_STRUCTURAL_BLOCKED_PASS / S16_3R_RESOURCE_R1_GPU5_FP32_CAST_BLOCKED_PRESERVED / S16_3R_RESOURCE_R2_FP64_SOLVE_PASS / S16_3R_FORMAL_F1_CODE_IDENTITY_FAILED_PRESERVED / S16_3R_FORMAL_F2_ISOLATED_PATH_FAILED_PRESERVED / S16_3R_F2_REPEAT_INTERRUPTED_FOR_FORMAL_PRIORITY / S16_3R_FORMAL_F3_PASS / S16_3R_F3_REPEAT_USER_INTERRUPTED`
>
> 当前 Gate：faithful 历史 Gate `NOT_PASSED / PROVEN_STRUCTURAL_RANK_BLOCKED_FAITHFUL_NO_RIDGE`；新方法资源 Gate `PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP` 与正式 Gate `PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION` 均已通过
>
> 科学边界：未打开 validation/test；资源 sweep 与 formal admission 均不得作为 efficacy 结论。

## 1. 冻结目标

历史 faithful `G-FULL` 与新 `G-RIDGE` 在 Toys 上共同固定为 5,963 个 real-cold edit targets、59,630 个 train-only contexts、302,400 个 prefix-next-token requests，以及按 lexical position 统计的全量 train-only raw second moment。Stage15 的 4 requests/position、256-row covariance 与全局 `0.3` z-success Gate 不得替代本方法。`G-RIDGE` 只替换不可逆的 no-ridge solve，不能改名或冒充 faithful reproduction。

## 2. Faithful 语义

- cached-z probe：GRAM legal-trie children 内比较 argmax，同时用 full-vocabulary softmax 检查 `p(target)>0.3`；pinned 官方实现没有 cache population path，因此 formal primary cache 保持空，cache hit 只作隔离诊断。
- new-z optimizer：full-vocabulary argmax、无 `0.3` Gate；Adam、cosine scheduler、active/satisfied lifecycle、absolute norm cap 与 step 10/20/21–29 检查保持官方语义。
- z loss：`softmax → clamp_min(1e-12) → -log`，不以数值上近似但极端区间不同的 fused cross-entropy 替代。
- covariance：FP64 累积 raw `E[x x^T]`，finalize 后按官方路径 round 为 FP32，再进入 double closed-form solve；primary 无 ridge。
- variable path routing：positions 0–5 映射 layers `0,1,2,3,0,1`；映射同一参数的 position delta 先加和，所有 live position 再引用同一个 aggregate tensor。

## 3. 已实现 contract

- full-target request/data builder：严格使用 S16-1 interaction-train 与 retained-warm occurrence；分片、稳定 SHA、原子 checkpoint、显式 resume 校验；不复用已被 S16 pseudo-cold 污染的 Stage15 contexts。
- faithful core：request batching、cache probe、new-z optimizer、terminal diagnostic delta、covariance、key extraction、valid-z filtering、closed-form delta、additive aggregation、One-One trigger、base parameter parity 与 admission schema。
- resource sweep：三候选 `4/8/16` 共享同一组 6×16=96 条、每 position 16 个不同 cold item 的固定请求；选择只看前十个 outcome-independent objective step 的完整耗时与 peak，不看 z success/probability。
- 独立 lifecycle probe：使用不参与候选选择和资源外推的固定失败行机械执行到 step 29；official early success 允许候选提前结束，不构成候选失格。
- diagnostics：valid z 用 satisfaction-time delta 重新 probe；failed z 用 terminal optimizer delta 重新 probe；保存 full-vocabulary probability/rank 与 legal rank。
- trigger/parity：真实 GRAM 上覆盖 positions 0–5、共享 layers 0/4 和 1/5、complete/EOS/padding/dead-prefix inactive rows、edited-output change、hook 后 restored-output exact parity 与 base parameter byte parity。
- 资源外推组件：full context/request build、z objective、全量 final-z re-probe 与 post-z filter/rank diagnostics、按 position coverage 分别外推的 covariance、formal covariance convergence diagnostics 与全请求 key extraction；按 position valid-z 率外推的 `KᵀK`/RHS matrix products；以及每 position 固定一次的 covariance transfer/FP64/λC/system assembly 与 factorization/solve/diagnostics、aggregation/trigger、7,435-event item-disjoint admission、512-event warm-preservation。key extraction 冻结为与 z 相同的 selected microbatch，且仅抽取 position 最终使用层；这只删除官方 key bank 中从未进入 solve 的层，不改变进入 delta 的 key 或数值语义。generation 计时覆盖逐事件 tokenization/context transfer，且 finalizer 从 raw measurement 与冻结计数机械复算每一项，缺任一实测组件不得生成 PASS 资源结论。
- provenance：runner 在 CPU preflight 前冻结 S16-3 config 与全部直接/传递代码 SHA；CPU 测试与 GPU worker 必须复用并重新核对同一 identity。S16-1 preflight config 也冻结 SHA，并逐文件核对其 resolved inputs 与 S16-3 声明一致。finalizer 要求执行前后身份完全一致，否则 fail closed。

## 4. 当前验证

- Stage16 CPU 回归：a3 启动时 `79/79 PASS`；a4 执行身份内新增资源分层/GPU4/tmux contract 后 `80/80 PASS`；a4 终态后为非权威 `solve_status` 标签增加回归至 `81/81 PASS`；S16-3B 新增 8 项 rank/上界/父 artifact/GPU4 contract 后为 `89/89 PASS`；recovery c1 新增 6 项 eligibility/ineligible/prefix-diagnostic/CPU-only/immutable-input contract 后为 `95/95 PASS`；G-RIDGE 初始 method/ridge/evidence/runner contract 为 `104/104 PASS`；GPU5 隔离入口、FP64 solve regression 与 r2 immutable-parent contract 加入后为 `107/107 PASS`；formal parent/request/progress/ridge/status 与 full-compute stability isolation contract 加入后当前 `115/115 PASS`。另行执行 S16-1→S16-3 八文件 resolution 检查，计数 `5963/59630/302400`、`maximum_history_items=20`，PASS。
- resource attempt a1：`IDENTITY_FREEZE_FAILED`。直接脚本入口未把 repo root 放入 Python import path，报 `ModuleNotFoundError: experiment`；失败发生在 GPU admission 前，未加载模型、未使用 GPU，artifact 原样保留且未自动重试。
- resource attempt a2：Python module 入口修复有效，identity freeze、`77/77` 当时 CPU preflight、三候选 batch sweep、covariance/trigger/generation probes 均完成；在 position-0 official no-ridge solve 处以 singular system 受控失败。根因是 `wo` key width=2048，而旧资源配置每位置 covariance 仅 8–12 rows、keys 16，`rank(C)+rank(K^T K)` 上界仅 24–28，数学上不可能满秩；未使用 ridge/pinv/fallback，未晋升 Gate。
- resource attempt a3：用户确认后于 2026-08-28 14:44:01+08:00 执行 exact command `bash experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a3.sh`。runner 在 admission 瞬间选择物理 GPU4→logical CUDA0，启动空闲显存 19,735 MiB；GPU0/5/7 未使用，未修改任何已有进程。execution identity 冻结、`79/79` CPU preflight 与完整 train-only dataset build 通过；manifest 固定 `5963/59630/302400`，47/47 shards 完整，validation/test/internal-dev occurrence 打开计数均为 0。
- a3 三个固定 z-microbatch candidate 全部运行完并选择 batch 16；随后完成 covariance positions 0–4 各 4,096 rows，实测分别为 `40.410/45.803/44.216/45.771/44.354s`。worker 在 360 秒硬预算内未完成 position 5，于 14:50:09+08:00 以 exit 124 终止；terminal 为 `TIMEOUT / RESOURCE_BLOCKED_BOUNDED_TIMEOUT`，`process_alive=false`。未进入 covariance convergence、trigger/generation、position z/key/rank/solve 或 formal projection，因此无 raw/final summary、无 Gate 晋升，也不能对 position-5 no-ridge 可逆性下结论。
- a3 partial checkpoint 原子保留 5/6 covariance position 及每 position 计时；status、progress、execution identity、request dataset、log 和 telemetry 均保留。GPU telemetry 总 used memory 从首个样本 28,835 MiB 升至最高 39,507 MiB，观测差值 10,672 MiB；该差值包含同卡全局状态，不冒充 PyTorch peak-reserved，但足以说明下一次资源边界不应继续按 8,192 MiB 小实验假设。a3 未自动 resume/retry。
- resource attempt a4 按用户指定在物理 GPU4 单卡执行。它使用全新 attempt/output/tmux session，不读写 a3 root；seed/data/96 candidate requests/三个 z batch/covariance rows/position requests/z 超参/no-ridge solve/Gate 完全不变。候选可用性 cap 仍为 8,192 MiB，整次 resource-attempt expected peak/cap 为 12,288 MiB、最低空闲 18,432 MiB、worker hard timeout 900s；三者从 config→inner→runner→worker raw→finalizer 机械核对。后台 tmux `phase16_s3_gfull_resource_a4_gpu4`，exact command `bash experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a4_gpu4.sh`，artifact `artifacts/phase16/s3_genrecedit/resource_sweep/toys_seed1502_a4_gpu4/`。
- 用户看到精确命令并最终确认后，a4 于 2026-08-28 15:13:22+08:00 启动；GPU4 启动空闲 19,735 MiB、readmission 空闲 19,469 MiB、物理 4→logical 0、runner/workload PID `2690300/2691233`。execution identity 内 `80/80` CPU tests PASS，15:20:18 终止，worker 实测 400.769 秒，未触发 900 秒 timeout；terminal 为 `BLOCKED / RESOURCE_BLOCKED_FAITHFUL_LINEAR_SYSTEM / exit 10`，`process_alive=false`，test/validation 封存、automatic retry=false。
- a4 三候选均完成：microbatch `4/8/16` 的 throughput 分别为 `75.490/134.630/178.062 objective-step/s`，peak reserved 分别为 `738/1218/2260 MiB`，均得到 `54 valid / 42 failed`，按冻结的 outcome-independent 规则选择 16。整次 worker 的 PyTorch peak allocated/reserved 为 `6895.492/8668 MiB`，低于 12,288 MiB attempt cap；全局 telemetry used-memory 增量为 10,820 MiB，仅作同卡观测，不替代 PyTorch peak。
- a4 完成全部 covariance resource：positions 0–4 各 4,096 rows、position 5 为全量 2,036 rows，总计时 242.259 秒；逐位置为 `42.484/44.525/44.032/45.921/42.869/22.428s`，另完成 2.088 秒 convergence diagnostic。trigger contract、generation resource 与 base-checkpoint byte parity 均 PASS。

| position | requests | valid / failed z | rank(C) | rank(KᵀK) | rank(system) | faithful no-ridge solve |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 16 | 10 / 6 | 71 | 9 | 71 | FAIL |
| 1 | 16 | 9 / 7 | 1058 | 9 | 1058 | FAIL |
| 2 | 16 | 10 / 6 | 1812 | 10 | 1813 | FAIL |
| 3 | 16 | 13 / 3 | 1982 | 13 | 1982 | FAIL |
| 4 | 16 | 11 / 5 | 2043 | 11 | 2043 | FAIL |
| 5 | 64 | 3 / 61 | 1741 | 3 | 1741 | FAIL |

- 六个固定 2,048 维 system 的 rank 均小于 2,048；所有位置都按预注册 faithful 路径受控失败，未使用 ridge、pinv、jitter、resample 或其他 fallback。因而没有任何 completed delta、aggregate parameter tensor 或 formal projection，失败 contract 正好是 `solve_aggregate_trigger_exercised_if_valid`、`faithful_solve_completed_for_every_valid_position` 与 `formal_projection_objective_complete`。这证明本次冻结 resource subset 无法建立可逆的官方 no-ridge system；不外推为“全量 covariance/formal full-key 系统一定奇异”。
- raw artifact 的非权威 `solve_status` 字段误写为 `NO_VALID_Z_IN_PREREGISTERED_RESOURCE_SUBSET`，但同一 artifact 明确记录 56 个 valid z，且每位置 diagnostics、三个 failed contract checks、顶层 verdict 与 `status.json` 均正确指向线性系统阻断。终态后只修复未来 attempt 的标签生成并增加回归测试，a4 execution identity 与原始 artifact 未改写；status/raw SHA256 分别为 `5487ad7022175a503d3d46f8e65dd57048bfe2eb2c54ed6c9bca9d8e2aec4f5e` / `85859b5e0dc47eabfe10bbf7d76d490104c69ed70fe1ff83fae4fbe73aa66d7d`。
- resource runner 硬界限按 attempt 分开冻结：a3 worker 为 `360s + 10s`、保守组件上界 545s；a4 worker 为 `900s + 10s`，且在界内正常返回结构化 BLOCKED。不能用 a3 的小实验时限描述 a4。
- formal G-FULL：未启动，且未获 admission；S16-3 Gate 未通过，因此不解锁 S16-4 的 G-FULL arm。
- validation/test：未读取、未使用。
- automatic retry/resume：关闭；formal 只允许新 attempt 或显式、身份校验通过的 safe-boundary resume。

## 5. 终态裁决

1. a4 已正常结束，终态为 `RESOURCE_BLOCKED_FAITHFUL_LINEAR_SYSTEM`；不是 timeout、OOM 或 GPU admission failure，不自动换卡、resume 或重试。
2. `PASS_S16_3_GFULL_FAITHFUL_CONTRACT_ADMISSION` 未通过；formal G-FULL 与 S16-4 的 G-FULL arm 保持锁定，不能用当前 resource subset 生成的 projection 冒充 formal admission。
3. 用户已授权直接开发独立 S16-3B train-only rank diagnostic；它不能覆盖本次 faithful primary 终态。任何正则化 solve 仍不属于 S16-3B，也不得命名为 faithful G-FULL。
4. faithful 子轨 `S16-3F` 以该 BLOCKED verdict 收口；用户决定后的 S16-3 可执行主轨改为独立 `S16-3R / G-RIDGE`，不覆盖历史 artifact；validation/test 始终未读取。
5. S16-3B recovery c1 已从 immutable b1 证据正式裁决 `PROVEN_STRUCTURAL_RANK_BLOCKED`；这关闭当前冻结 GRAM representation 上的 faithful no-ridge G-FULL 路径，但不把 S16-3 Gate 改成 PASS，也不产生 efficacy 结论。

## 6. S16-3B：full-universe rank-sufficiency diagnostic

### 6.1 可证伪问题与单向结论

S16-3B 不优化 z、不做 valid-z filtering、不形成 RHS、不求 weight delta，也不运行 admission/validation。它对每个 lexical position 计算冻结的 full train-only covariance `C_full`，并以全部 train-only request keys 形成 `K_allᵀK_all`，检查 `1000·C_full + K_allᵀK_all` 的数值秩。总 request 计数严格为 positions `59630/59630/59630/59630/59630/4250 = 302400`；covariance rows 为 `27659/27659/27659/27659/27659/2036`。

faithful system 的 `K_valid` 只能是 `K_all` 的行子集。对通过冻结 tolerance 数值 PSD 检查的 covariance/Gram 项，若最有利的 all-request superset system 仍不满秩，则其剩余 nullspace 不可能被任一 valid-z subset 消除，严格裁决 `PROVEN_STRUCTURAL_RANK_BLOCKED`。任一位置已足以形成该单向反证；若六个 superset systems 均满秩，只能裁决 `ALL_REQUEST_UPPER_BOUND_FULL_RANK_VALID_Z_DIAGNOSTIC_REQUIRED`，不能据此晋升 S16-3 Gate；后续必须另立 S16-3C 测量实际 valid-z key coverage。

### 6.2 冻结实现与 contract

- 复用 A4 的 47 个 train-only request shards，但逐文件重新核对 SHA/行数；parent raw/status/manifest/checkpoint 四个 SHA 在执行前后必须不变。
- 每位置使用 seed-1502 对 request semantic identity 冻结一个 outcome-independent SHA256 顺序；检查点为 `16/64/256/1024/4096/16384/full`，超过该位置总数的检查点自动并入 full。最终 full Gram 与顺序无关。
- covariance 延续 faithful `FP64 raw accumulation → FP32 finalize → FP64 system/rank`；key Gram 以 FP64 streaming `addmm` 累积，不保留 302,400×2,048 全 key bank。
- rank tolerance 与 A4 完全相同：`max(matrix_shape) × float64_eps × max_abs_eigenvalue`；同时 hard-fail 显著负 eigenvalue，记录 covariance/key/system rank、nullity、condition 与 progressive curve。
- 明确冻结 `z_optimization=false`、`valid_z_filter=false`、`solve=false`、`ridge/pinv/jitter/resample=false`、`faithful_gate_promotion=false`、`scientific_efficacy_metric=false`、`validation/test=false`。
- 已实现 worker、机械 finalizer、execution identity、atomic position checkpoint、45 秒 heartbeat、30 秒 GPU telemetry、三小时 hard timeout、无自动 retry/resume；Stage16 全量 CPU 回归 `89/89 PASS`，父 A4 `47/47` shards 与 `5963/59630/302400` 只读 preflight PASS。

### 6.3 GPU4 资源与启动闸门

- attempt：`s16_s3b_gfull_rank_sufficiency_b1_gpu4`；独立输出 `artifacts/phase16/s3_genrecedit/rank_sufficiency/toys_seed1502_b1_gpu4/`，不覆盖 A4。
- 固定物理 GPU4→logical CUDA0；minimum free `18,432 MiB`、expected/cap `12,288 MiB`、CPU RAM reservation `16 GiB`、新增 disk reservation `4 GiB`、hard timeout `10,800s`。不自动换 GPU，不释放或修改任何现有进程。
- 依据 A4 的 full covariance projection `1506.876s` 与 full request key extraction projection `3576.160s`，加全位置 spectral curves、shard verification、排序与安全余量后，预计 wall `90–130 分钟`，单卡约 `1.5–2.2 GPU·h`；超过 3 小时受控 TIMEOUT 并保留 completed-position checkpoint。
- exact command：`bash experiment/phase16/run_stage16_s3b_gfull_rank_sufficiency_b1_gpu4.sh`。用户明确确认后，b1 于 2026-08-28 16:01:35+08:00 在 tmux `phase16_s3b_gfull_rank_b1_gpu4` 启动；GPU4 admission free `25,525 MiB`，runner/timeout-workload/实际 Python PID 为 `2843623/2844400/2844403`，execution identity 与 `89/89` CPU tests PASS。test/validation 封存、automatic retry/resume=false。

### 6.4 b1 终态、失败根因与可用证据

- b1 于 2026-08-28 17:38:58+08:00 完成全部 `302400/302400` train-only request keys 和六个 position；raw worker elapsed `5830.412s`，PyTorch peak allocated/reserved `4938.328/8648 MiB`，未发生 timeout、OOM、GPU admission failure 或未完成计算。进程已退出，terminal 为 `FAILED / exit 3`；raw contract 失败，因此 finalizer 未生成 `summary.json`，且没有自动 retry/resume。
- 唯一失败的 raw contract check 是 `positive_semidefinite_evidence`，其他 checks 全部 PASS。position 5 的 FP32-finalized covariance 最小特征值为 `-2.8636433355862193e-08`，冻结 tolerance 为 `6.166051108776179e-10`，有 2 个 significant negative eigenvalues；position 5 的 16/64-key prefix system 也各有 2 个 significant negatives。该数值路径不能被事后宣称为 PSD，所以原 b1 artifact 必须保持 FAILED。
- position 0–3 covariance 的 significant negative eigenvalue 均为 0，final all-key systems 也均通过 numerical-PSD 检查；其 full system rank 分别为 `74/1216/1938/2033 < 2048`。其中任意一个位置已经足以按 6.1 的单向逻辑证明 structural rank blockage，position 5 的 PSD 歧义不影响该反证。raw worker 的科学分类也为 `PROVEN_STRUCTURAL_RANK_BLOCKED`，但这不覆盖 artifact-contract FAILED，更不晋升 S16-3 Gate。

| position | rank(C_full) | rank(K_allᵀK_all) | rank(full system) | nullity | PSD proof eligibility |
|---:|---:|---:|---:|---:|---|
| 0 | 74 | 74 | 74 | 1974 | eligible |
| 1 | 1191 | 1195 | 1216 | 832 | eligible |
| 2 | 1907 | 1929 | 1938 | 110 | eligible |
| 3 | 2019 | 2023 | 2033 | 15 | eligible |
| 4 | 2048 | 2047 | 2048 | 0 | eligible, not rank-blocked |
| 5 | 1741 | 1763 | 1830 | 218 | ineligible: covariance numerical-PSD check failed |

- immutable evidence：execution identity SHA256 `3946cfddeea2f20137e6f0340a4f2bbbd989fb2cf14eb11e9aabc52aa7447c50`；terminal status/raw/checkpoint SHA256 分别为 `f8fda8eb745e89fe86c3bf2deecf270f5ef35e65cdee58777f5ca42d366e81fc`、`c68dda9bd88dcca21b4bbc24551f8ddb2ca0a90ed2f079156766228265b4dc08`、`30ccd5cd65fc1a5d431dd0976908cf24c05ec0d9b5b69697f4e1ea024b18b72f`。父 A4 artifacts/base checkpoint 前后 SHA 保持不变。
- b1 原始 FAILED、缺失 `summary.json` 与上述 immutable SHA 均保持不变；正式裁决由下述独立 CPU-only recovery c1 承载，不改写或重跑 b1。

### 6.5 CPU-only recovery c1 正式裁决

- 用户确认 exact command `bash experiment/phase16/run_stage16_s3b_rank_sufficiency_recovery_c1_cpu.sh` 后，c1 于 2026-08-28 18:07:35+08:00 前台执行并在一秒内 exit 0；runner 内 6/6 recovery tests PASS，终态后 Stage16 全量 `95/95 PASS`。`gpu_count=0`、`gpu_used=false`、`CUDA_VISIBLE_DEVICES=""`，没有访问或占用 GPU。
- c1 只读取冻结的 b1 config/raw/status/rank checkpoint/execution identity，执行前后五个 SHA 完全一致；机械确认 source b1 仍为 `FAILED / exit 3`、`302400/302400`、六位置完成、无 `summary.json`，且唯一失败 raw check 仍是 `positive_semidefinite_evidence`。c1 没有重算 covariance、keys、eigenvalues 或 ranks，也没有 z、solve、ridge、pinv、jitter、resample、retry/resume。
- proof eligibility 固定为：完整 covariance、final all-request key Gram 与 final full system 三者在原 b1 tolerance 下均无 significant negative eigenvalue；intermediate prefix system 只作诊断。positions `0–4` eligible，position `5` 因 covariance 有 2 个 significant negatives 而明确 ineligible、未被重标为 PSD。
- eligible positions 中 `0–3` 的 final system rank 为 `74/1216/1938/2033 < 2048`；任意一个位置已足以证明其 faithful valid-z key subset 不能消除 nullspace。正式 verdict 为 `PASS_S16_3B_RECOVERY_ADJUDICATION_COMPLETE`，diagnostic classification 为 `PROVEN_STRUCTURAL_RANK_BLOCKED`。
- c1 status/adjudication/execution-identity SHA256 分别为 `045f9bd6468cf8e82e6bd13cf993b20ac09d26d3c1796b0b7cc92fb01b708ddf`、`b900a0d4386ffaf80303155c07dd8b50c24f0efcc1dcee6a2e69d546f86ba362`、`c3cc596840b0ff63208a35137c51e1ef54a1c1cddf5a30b79060e51d9daa977a`；全部 15 项 final contract checks PASS。S16-3 faithful Gate 保持 `NOT_PASSED_UNCHANGED`，S16-4 G-FULL 仍锁定；任何 ridge、pseudoinverse、representation change 或 modified solve 必须另立方法名与 Gate。

## 7. S16-3R：GenRecEdit-inspired G-RIDGE→GRAM

### 7.1 方法边界与冻结差异

- 用户决定将 S16-3 的后续可执行方向改为 `GenRecEdit-inspired → GRAM`。新 arm 命名为 `G-RIDGE`，`faithful_reproduction=false`；faithful A4、S16-3B b1 与 recovery c1 的状态、SHA 和科学裁决全部原样保留。
- `G-RIDGE` 保持 A4 的 seed、数据、5,963 edit targets、59,630 contexts、302,400 requests、z optimizer/lifecycle、cache/probability 语义、FP64 raw covariance→FP32 finalize、key extraction、valid-z filtering、position→layer routing、delta aggregation、One-One trigger、base parity 与 validation/test sealing。唯一方法变化是把 `A=1000·C+KᵀK` 的 faithful no-ridge solve 替换为 train-only、逐位置、FP64 的 condition-targeted spectral ridge solve。
- 固定 `target_condition_number=1,000,000`、`ridge_safety_margin=1e-6`，并令 `mu=(1+safety_margin)·max(max_abs_eigenvalue/target_condition, (max_eigenvalue-target_condition·min_eigenvalue)/(target_condition-1))`，随后解 `(A+mu·I)Δ=RHS`。该规则只读取当前 train-only system 的谱尺度，不读取 validation/test，不按 z outcome 或 efficacy 调参。
- 禁止 pseudoinverse、额外 jitter fallback、outcome resampling 与 automatic retry。必须保存 ridge 前后的最小/最大特征值、rank/nullity、condition、ridge 绝对值/相对谱尺度、Cholesky 与 solve residual；原 system 是否秩亏或含显著负特征值仍照实报告，不能被正则化后的可逆性重标。

### 7.2 独立 Gate、实现与静态验证

- faithful Gate `PASS_S16_3_GFULL_FAITHFUL_CONTRACT_ADMISSION` 永久不由 G-RIDGE 继承或晋升。新资源 Gate 为 `PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP`，新正式 Gate 为 `PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION`；只有后者通过才能解锁 S16-4 的 G-RIDGE arm。
- 独立实现位于 `experiment/phase16/protocol/genrecedit_inspired.py`；通用 worker/finalizer 以 method contract 分支执行，faithful 默认路径保持 no-ridge。新配置、outer/inner runner 与 artifact root 均与 A4 隔离。
- 6 项谱正则/求解单测、3 项配置/证据/runner contract 加入后，Stage16 全量 CPU 回归 `104/104 PASS`；两个执行入口 bash syntax、JSON 配置、worker/finalizer 27 项 execution-identity path exact equality、15 项冻结输入 SHA 与新输出不存在性检查均 PASS。历史 A4、b1、c1 artifacts 未改写。

### 7.3 resource r1 GPU5 终态

- 用户指定并确认物理 GPU5；exact command `bash experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu5.sh` 于 2026-08-29 10:54:47+08:00 启动。admission/readmission free 为 `28,998/28,731 MiB`，minimum `18,432 MiB`、expected/cap `12,288 MiB`、hard timeout `900s`；现有 GPU5 外部进程未停止或修改。
- r1 完成三档 z batch、六位置 covariance 与六位置 ridge solve，elapsed `320.838s`，peak allocated/reserved `6,895.492/8,668 MiB`。六位置正则系统均 rank `2048`、Cholesky info `0`、condition 约 `999,999`，solve/aggregation/trigger 均实际执行；validation/test 封存。
- 唯一失败 contract 为 `inspired_ridge_solve_completed_for_every_valid_position`：position 0 relative residual `3.920733e-6` 超过冻结 `1e-6`，其余五位置均通过。terminal 为 `BLOCKED / RESOURCE_BLOCKED_INSPIRED_RIDGE_LINEAR_SYSTEM / exit 10`，无自动 retry；raw/status/identity SHA256 分别为 `efc021e1a2a456481672246f40562fae5e47e55c6e42509e2ca37a6ad2c6ca2c`、`4f80140dc63d94616558d3317648388f16fea2aeeaeb76493fdfdddb179d8d8b`、`a08b3018d1f078fce7fa322d02451f6ebe915f71e02e400d7879a1d002f9002b`，永久保留。
- 根因定位为实现把预注册 FP64 solve 的返回值在 residual diagnostic 和 aggregation 前按 FP32 parameter template 降精度。固定 condition=`1e6` 的合成隔离复现中，同一 solve 的 FP64 residual 为 `1.185633e-11`，提前 FP32 cast 后为 `1.189193e-3`；因此这是与冻结“solve in FP64”不一致的工程错误，不是改 Gate 或调方法参数的理由。
- r1 仍提供非权威 formal resource projection：单卡 `17.58–28.13 GPU·h`、minimum free `13,312 MiB`、expected peak reserved `8,668 MiB`、CPU RAM peak proxy `6,067.66 MiB`、disk reservation `32,768 MiB`、hard timeout `604,800s`。这些投影须由通过后的 r2 finalizer 机械确认才能冻结为 formal admission 资源合约。

### 7.4 resource r2 FP64-solve 工程恢复终态

- 最小修复只删除 G-RIDGE solver 的提前 FP32 cast；ridge 规则、condition target、安全 margin、residual threshold、seed、请求、covariance、数据、Gate 与 test/validation 边界全部不变。FP64 delta 继续聚合，现有 One-One generation hook 在模型应用点才执行 dtype/device cast。
- 独立 config 锁定 r1 raw/status/identity SHA；attempt/output/tmux 与 r1 隔离。用户确认 exact command `bash experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r2_gpu5_fp64solve.sh` 后，r2 于 2026-08-29 11:11:04+08:00 在 GPU5 启动，admission free `28,998 MiB`，现有 GPU5 进程未修改。
- r2 于 11:16:46+08:00 正常结束，终态 `COMPLETED / PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP / exit 0`；worker elapsed `317.274s`，peak allocated/reserved `6,895.492/8,668 MiB`，validation/test 未读且 `automatic_retry=false`。
- 六位置正则化 system 均 full-rank、regularized Cholesky info `0`，FP64 solve relative residual 分别为 `3.457258e-14 / 2.747894e-14 / 2.784601e-14 / 2.132900e-14 / 3.564543e-15 / 5.519000e-14`，均低于冻结 `1e-6`。raw/status/identity SHA256 分别为 `f353a457474ab92961b4a0083e5a7f61af2a00b6ebaae2e5e356dcfa87df9439`、`1d662e30d54b6bc891c0bf1d215fbb568f0017678df1eb91e99f8401a0cd9b04`、`b32f3206d85fc9513af9b410d96aa4d04fcb34cec0573b65eeb4204660be41df`。
- r2 finalizer 冻结 formal resource contract：单卡 `16.859–26.974 GPU·h`、minimum free `13,312 MiB`、expected peak reserved `8,668 MiB`、CPU RAM peak proxy `6,049.613 MiB`、disk reservation `32,768 MiB`、hard timeout `604,800s`。该 PASS 只放行 formal，不替代 formal Gate。

### 7.5 formal GPU5 f1 准备与启动闸门

- formal config/worker/finalizer/outer+inner runner 已独立实现；只读复用 r2 的 47-shard request dataset 与 manifest SHA，不覆盖 resource 或历史 faithful artifacts。全量工作量固定为 302,400 z requests、六个 full covariance/ridge systems、7,435 个 train-derived item-disjoint admission events 与 512 个 deterministic train-only warm-preservation pairs。
- held pseudo-cold event 文件只在六位置 delta、aggregate checkpoint 与 state-freeze manifest 完成后打开；不得用于 z/ridge/state selection。formal status 明确写出 authoritative experiment 是否完成，且在正式终态前后均区分后续 stability queue；validation/test 保持 sealed，所有 admission 指标仅作 non-promotional contract evidence。
- 当前 Stage16 `115/115` CPU tests、bash syntax、Python compile、JSON、23-file execution identity、17 个冻结直接输入、97 个 request artifact 文件与 formal output 不存在性均 PASS；formal config SHA256 为 `1cd2cd10de4a7d5812255510dbc94f9f6cf95bf4bbe02c79935d7e172f19c654`。
- 待用户确认的 exact command：`bash experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f1.sh`。固定物理 GPU5、minimum free `13,312 MiB`、expected peak reserved `8,668 MiB`、单卡投影 `16.859–26.974 GPU·h`（约 16.9–27.0 小时）、disk reservation `32,768 MiB`、hard timeout 7 天；不停止或修改 GPU5 既有进程，不自动切卡、retry 或 resume。

### 7.6 formal f1/f2 终态、f3 启动与非权威 repeat 队列

- f1 于 2026-08-30 05:05:50+08:00 终态 `FAILED / exit 1`。六个 position delta、aggregate checkpoint 与 state-freeze manifest 已生成，进度到 7,435-event item-disjoint admission 后的 warm preservation `512/512`；但最终 `config_and_code_identity_unchanged` 检查发现 `GRAM/src/model/gram.py` 在运行期由冻结 SHA `275f10...` 变为 Stage17 版本 `fdd628...`。因此未写 `formal_admission_summary.json`、`summary.json` 或 `authoritative_completion.json`，formal Gate 未晋升。heartbeat 固定临时文件名的并发异常是次要工程问题，已改成 PID+thread 唯一临时名；不据此改写 f1。
- 用户明确授权独立 f2 与“任意 f2 终态后仍继续非权威重复”。f2 保持 seed/data/method/workload/Gate/resource 完全相同，新增 f1 lineage SHA，并在 `.runtime/phase16_s3r_gridge_f2_runtime` 使用 git-HEAD GRAM 源码；主工作树 Stage17 变化不可见。主工作树与快照 Stage16 回归均 `121/121 PASS`。
- f2 于 2026-08-30 14:35:30+08:00 启动，admission free `28,998 MiB`；14:36:04 在请求 materialization 前的 S16-1 input resolution 因隔离快照的 allowlisted artifact 父路径尚未映射而 `FAILED / exit 1`。失败发生在模型加载和 GPU 计算前；f2 status/identity 保留，不自动覆盖、resume 或晋升。
- 后置 repeat queue 按授权启动，formal parent 明确为 `FAILED`，所有 cycle 固定 `affects_scientific_results=false / promotion_eligible=false / automatic_retry=false`，不会打开 validation/test，也不生成 S16-3 formal PASS。cycle 1–3 因同一 snapshot config/input 路径边界在 GPU 前失败并保留；仅在 f2 终态后对 repeat-only runtime 增加 allowlisted `artifacts`/`GRAM/rec_datasets` parent-link 映射与 snapshot-local cycle config，未修改 f2 artifact。
- cycle 4 于 14:39:17+08:00 进入真实 GPU5 full reexecution；14:39:47 快照为 covariance `0/6`，项目 Python PID `4187045` 占用 `10,634 MiB`，GPU5 总占用/空闲 `30,213/18,358 MiB`、utilization `83%`。队列只将 f2 启动前已有 GPU5 PID 视为 baseline；检测到任何新增非本 cycle PID 时只终止自己的 repeat cycle 并等待，不向其他进程发信号。repeat 失败后使用新 cycle id/目录继续，这是用户预授权的计划性独立重复，不是对 formal artifact 的自动 retry。
- f1/f2 时点裁决保持历史有效：resource Gate 为 PASS，但当时 formal Gate 尚未通过；其失败 artifact 不被后续 f3 覆盖。最终 formal 裁决由下述独立 f3 terminal artifact 承载。
- 用户于 2026-08-30 确认继续独立 formal f3。切换前核验 f2 repeat tmux pane/runner/cycle/实际 Python 为项目 PID `4173017/4187041/4187045`；只向 runner `4173017` 发送 TERM，cycle 4 与 queue 均以 `INTERRUPTED / exit 143 / process_alive=false` 封存。该操作只释放项目 repeat 的 `10,634 MiB`，GPU5 原有外部 PID `1648062` 未停止或修改；切换后 GPU5 free 恢复为 `28,998 MiB`。
- f3 从 f2 配置机械派生，seed、domain、method、tokenizer、302,400-request workload、admission、resource parent 与所有数值 Gate 不变；只改变 attempt/output/exact command/isolated-runtime/repeat root，并冻结 f2 status SHA `c200f702…641c` 与 identity SHA `e99239ca…eee`。allowlisted `artifacts`/`GRAM/rec_datasets` parent-link 映射在 `.runtime/phase16_s3r_gridge_f3_runtime` 创建时即纳入 immutable code identity，不再对已启动快照热修。f3 config SHA 为 `672739e9719173ffd78430338189ecb4e9108e932c8550b2ee4a44e9c0b6b2ca`，主仓与快照 Stage16 回归均 `123/123 PASS`。
- exact command `bash experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f3.sh` 于 15:00:28+08:00 在 tmux `phase16_s3r_gridge_formal_gpu5_f3` 启动，admission free `28,998 MiB`。15:11:12+08:00 status 为 `RUNNING / full_covariance_positions 2/6`，runner/workload/实际 Python PID `69806/70821/70827`，项目显存 `10,634 MiB`、GPU5 总占用/空闲 `30,213/18,358 MiB`、utilization `100%`；identity capture 与 123 项 CPU preflight PASS，validation/test=false、automatic_retry=false。
- f3 formal 运行期间没有 repeat 与其并发。f3 于 2026-08-31 07:08:55+08:00 正常结束，workload/runner exit 均为 `0`，终态 `COMPLETED / PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION / process_alive=false`。全部 `302400` requests、六位置 FP64 G-RIDGE solve、`7435` 个 item-disjoint admission events 与 `512` 个 warm-preservation pairs 均完成；elapsed `58061.718s`，最大 peak allocated/reserved `6910.633/8880 MiB`。formal status、formal summary、compact summary 与 checkpoint manifest SHA256 分别为 `bfd304c5…5920`、`5d2cfe96…5e49`、`92997f6a…a319`、`cc804aeb…329`。所有 admission 结果明确 `scientific_efficacy_metric_produced=false`，validation/test 未读、automatic retry=false；因此 S16-4 的 G-RIDGE arm 已解锁，但没有提前形成 efficacy 结论。
- f3 terminal 后自动建立的 `phase16_s3r_gridge_repeat_gpu5_f3` 只具非晋升重复语义。它检测到 formal baseline 之外的新 GPU5 PID `464283` 后保持 `WAITING_FOR_PRIORITY_GPU5`，`current_cycle=1 / cycle_pid=0`，从未启动 repeat compute、从未占用 GPU。用户于 2026-08-31 明确要求停止；核验 tmux、runner PID `776933` 与 cycle PID 0 后，只向该 runner 发送 SIGTERM。队列于 10:02:25+08:00 由自身 signal/EXIT trap 原子写成 `FAILED / INTERRUPTED / exit 143 / process_alive=false` 并退出 tmux；这里的 `FAILED` 只表示用户终止非权威队列，不影响 formal f3。GPU5 既有 PID `1648062/464283` 未收到信号、显存占用未变化；repeat terminal status SHA256 为 `fec306a8…4d04`。

## 8. S16-3 最终裁决与下一唯一步骤

1. faithful no-ridge `G-FULL` 因可证明的结构性秩亏保持 `NOT_PASSED`，不进入 S16-4 efficacy；该负结果与 G-RIDGE 的新方法 PASS 必须同时披露。
2. `G-RIDGE` 明确为 `faithful_reproduction=false`，resource 与 formal contract/admission Gates 均已通过；f1/f2 工程失败和全部 immutable artifacts 继续保留。
3. S16-3 到此收口为 `COMPLETED`。repeat 队列已按用户要求停止，不是阶段完成条件，也不得作为 formal 或 efficacy 证据。
4. 下一唯一步骤为 S16-4 Toys standalone frozen validation：在隔离 runtime 中冻结 `F0/R2/S-AUX/S-PLUS-CTRL/S-PLUS/G-RIDGE` 的代码、输入、state 与 evaluator SHA，运行 seed 1502 validation；保持 test sealed，并对每个方法使用其正确 control 形成 cold H@50、cold NDCG@10、warm NDCG@10、cost 与 paired-bootstrap Gate。
