# GRAM 第十三阶段：T1-4 多兴趣 Resolver 续跑交接报告

> **交接结论（2026-08-19）**：T1-4 的配置、评测程序、单测和后台 runner 已实现；21 项相关单测通过。第一次 GPU smoke 成功跑通 CUDA、数据与 M=1/2 前向，但 M=1 因批量形状和 CUDA 算子口径变化出现 `7/128` 个 top-50 差异，**不能视为控制复现通过**。该问题已在代码中修正，但修正后尚未重新执行 GPU smoke，正式 M=1/2/4 筛查也尚未启动。续跑时必须先完成一次 256 用户 smoke，并确认 M=1 mismatch 为 0；通过后才可后台启动正式筛查。

## Material Passport

- Origin Skill: `experiment-agent`
- Origin Mode: `validate / run handoff`
- Origin Date: 2026-08-19
- Verification Status: `ANALYZED`（修正后的 GPU 精确复现尚待验证）
- Version Label: `phase13_t1_4_handoff_v1`
- Dataset: `Toys_cold50 validation`（development-only）
- test_read: **false**
- Beauty_read: **false**
- Sports_read: **false**
- Formal experiment status: **NOT_STARTED**
- Automatic retry: **false**
- Automatic next stage: **false**

---

## 1. 当前研究位置

本轮承接以下已完成的 Tier-1 结果：

1. **T1-1 checkpoint trajectory**：epoch 12 精确复现 P0；继续训练未改善 cold Recall@50，反而下降；
2. **T1-2 static warm-only hard negative**：HN=0 精确复现 P0，HN=8/16/32 全部下降，正式 verdict 为 `FAIL_STOP_STATIC_HARD_NEGATIVE`；
3. 因 T1-1 与 T1-2 已否定“仅增加训练轮数”和“静态 warm hard negative”两条便宜路径，跳过高度相似的温度/容量小调参，进入最后一个便宜的结构候选 **T1-4 多兴趣用户表示**；
4. T1-4 若仍失败，则 Tier-1 resolver 轻量探索应结束。路线 B 属于高成本重训，**不得自动进入，必须另行确认资源与启动权限**。

T1-2 的正式冻结 projector：

```text
artifacts/phase13/explore/tier1_resolver_toys_static_warm_hard_negative/resolver_hn_000.pt
```

该 checkpoint 是 HN=0、epoch 12 的 P0 对照，完整 Toys validation 上已经验证：

- exact top-50 mismatch vs P0：`0`
- cold Recall@50 events：`498`
- cold Recall@50：`0.114037`

---

## 2. T1-4 冻结实验协议

### 2.1 唯一实验变量

冻结以下所有部分：

- item embedding catalog；
- HN=0 epoch-12 P0 projector；
- GRAM validation predictions；
- P0 baseline predictions；
- portfolio@2 后处理；
- validation 用户与 target；
- retrieve K、评测指标和统计检验。

只改变**用户验证历史的表示方式**：

| Arm | 用户表示 | 候选打分 |
|---|---|---|
| M=1 | 原 P0 recency-weighted mean | 单向量 cosine |
| M=2 | 确定性语义聚类得到 2 个兴趣向量 | 两向量 cosine 最大值 |
| M=4 | 确定性语义聚类得到最多 4 个兴趣向量 | 多向量 cosine 最大值 |

M=2/4 的聚类规则冻结为：

1. 历史最多取 20 个 item；
2. validation target（序列倒数第二项）和 held-out test item（最后一项）均不进入历史；
3. 首个中心取最近 item；后续中心使用确定性 farthest-point 初始化；
4. 固定 5 次 spherical Lloyd iteration；
5. cluster centroid 使用与 P0 相同的 `0.85` recency decay；
6. 历史长度小于 M 时，有效兴趣数为 `min(M, history_length)`；
7. 聚类不读取 target，不训练任何参数。

### 2.2 主指标与 Gate

- 主指标：resolver cold Recall@50；
- 两个主比较：M=2 vs M=1、M=4 vs M=1；
- paired bootstrap：10,000 次；
- family-wise alpha：0.05；
- Bonferroni 调整后每个主比较使用 **97.5% CI**；
- 信号 Gate：调整后 CI 下界大于 0，且 warm Recall@50 相对下降不超过 5%；
- Toys 30% 目标：cold Recall@50 至少 `0.148`，同时通过 warm guard；
- winner：优先 cold R@50，其次 eligible cold R@3，最后选择更小的 M。

可能 verdict：

- `PASS_T1_4_TOYS_GATE_REQUIRES_REPLICATION`
- `PASS_T1_4_SIGNAL_REQUIRES_REPLICATION`
- `FAIL_STOP_MULTI_INTEREST_TIER1`
- `CONTROL_REPRODUCTION_FAILED_STOP`
- `SMOKE_COMPLETED`

任何 PASS 都只是 Toys development signal，不能直接声称 confirmatory efficacy。

---

## 3. 已实现文件

| 文件 | 用途 | 当前 SHA-256 |
|---|---|---|
| `experiment/phase13/configs/tier1_resolver_toys_multi_interest.json` | 冻结协议和 Gate | `b2ee4d810393cb799e9989117a84cf419ed88ed90163b742ceb019b3e2411bb5` |
| `experiment/phase13/protocol/tier1_resolver_multi_interest.py` | 聚类、冻结前向、评测、bootstrap、verdict | `a57548cdfd3f791d4358bbc1c78fc69f9f11ede0b0824de06e0ab431d29ca32f` |
| `experiment/phase13/tests/test_tier1_resolver_multi_interest.py` | T1-4 单测 | `9505271359d9c613d0b55e36a1ab3fdb15129d58ebb916675846dc6a2da5e93c` |
| `experiment/phase13/run_tier1_resolver_toys_multi_interest.sh` | GPU admission、tmux 后台、status、timeout、telemetry | `2e6dab74d8f032bbed05a487e978fea10a8f81a5cd39627dd8cf4d5df790fe96` |

注意：当前 worktree 原本已有大量用户/其他 AI 的未提交文件和修改。续跑时不要清理、reset 或覆盖这些内容。

---

## 4. 已完成验证

### 4.1 静态检查与单测

最新检查结果：

```text
21 passed in 0.87s
```

覆盖：

- runner Bash 语法；
- Python 编译；
- 原 route resolver 单测；
- checkpoint trajectory 单测；
- static hard-negative 单测；
- T1-4 interest count 约束；
- M=1 recency pooling 的 bitwise 等价；
- M=2 聚类确定性与语义分离；
- 短历史下的有效兴趣数；
- validation/test 位置排除；
- max-over-interest 打分语义。

### 4.2 第一次 GPU smoke（修正前，仅作故障证据）

- GPU：物理 GPU2；
- 环境：`gram-repro`，真实 CUDA；
- 用户数：128；
- Arms：M=1、M=2；
- 临时输出：`/tmp/phase13_t1_multi_interest_smoke.pVsQ2F`；
- 程序 exit code：0；
- summary runtime：28.13 秒；
- verdict：`SMOKE_COMPLETED`。

| Arm | n | cold R@50 events | cold R@50 | warm R@50 | mismatch vs P0 |
|---|---:|---:|---:|---:|---:|
| M=1 | 128 | 6 | 0.080000 | 0.113208 | **7** |
| M=2 | 128 | 7 | 0.093333 | 0.094340 | 128（预期会变化） |

M=2 的小样本数值**不能作效果结论**：样本仅 128，且控制口径当时未通过。

---

## 5. 第一次 smoke 的异常、诊断与已做修正

### 5.1 异常

M=1 本应逐用户精确复现 P0，但出现 `7/128` 个 top-50 mismatch。虽然程序正常退出，按确定性复现标准，这次 smoke 的控制验证仍视为**未通过**。

### 5.2 根因判断

旧的冻结 P0 评测使用：

```python
model(base) @ catalog.T
```

且完整评测批量大小为 256。第一次 T1-4 smoke 改成了 128 用户，并统一走三维 einsum：

```python
einsum("bmd,nd->bmn", ...)
```

对非常接近的 top-50 边界分数，CUDA kernel 与 GEMM shape 的变化可改变并列/近并列 item 的最后排序，因此产生少量 mismatch。现有证据支持这是**数值评测口径差异**，不是 checkpoint 错误；但修正后的精确复现仍需第二次 GPU smoke 才能确认。

### 5.3 已完成修正

1. `evaluation_batch_size` 从 128 恢复为原 P0 的 256；
2. M=1 强制使用原始二维 matmul，不再走三维 einsum；
3. smoke 样本数从 128 改为 256，确保构成一个完整的原口径 batch；
4. 修正后重新执行静态检查，21 项单测通过；
5. 遵守“不自动 retry”，修正后**没有再次启动 GPU smoke**。

---

## 6. 当前精确状态

| 项目 | 状态 |
|---|---|
| T1-4 协议 | 已冻结 |
| 配置/实现/单测/runner | 已完成 |
| CPU 静态验证 | 已通过 |
| 修正前 CUDA smoke | 跑通，但 M=1 精确控制未通过 |
| 修正后 256 用户 CUDA smoke | **待运行** |
| 正式 M=1/2/4 筛查 | **未启动** |
| 正式 artifact 目录 | 当前不存在 |
| Beauty / Sports / test | 未读取 |
| 路线 B | 未启动、未授权自动启动 |

正式输出目录预定为：

```text
artifacts/phase13/explore/tier1_resolver_toys_multi_interest_screen
```

用户通过以下文件观察状态：

```text
artifacts/phase13/explore/tier1_resolver_toys_multi_interest_screen/status.json
```

---

## 7. 续跑顺序（必须按顺序）

### Step 1：重新检查实时 GPU 资源

GPU 状态持续变化，**不要沿用本报告中的旧快照**。只读检查：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
```

资源规则：

- GPU0 与 GPU5 为用户保护卡，**绝对不能使用**；
- 不能终止、迁移或修改任何其他人的 GPU 进程；
- T1-4 runner 要求候选卡 compute mode 为 `Default`、启动时至少 4096 MiB free；
- 预计本实验增量显存上界约 3072 MiB；
- 小实验可自行选择当时空闲的非 0/5 GPU；
- 如果没有满足条件的卡，停止并告诉用户大约需要一张有 **至少 4 GiB 空闲显存**的卡，由用户指定资源。

### Step 2：执行一次修正后的 256 用户 CUDA smoke

这是修正后的首次验证，不得省略。可使用当时选中的非 0/5 GPU；以下 `<GPU>` 必须替换为实际物理卡号：

```bash
smoke_dir=$(mktemp -d /tmp/phase13_t1_multi_interest_smoke_fix.XXXXXX)
CUDA_VISIBLE_DEVICES=<GPU> timeout --signal=TERM --kill-after=10 600 \
  /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
  experiment/phase13/protocol/tier1_resolver_multi_interest.py \
  --frozen-config experiment/phase13/configs/tier1_resolver_toys_multi_interest.json \
  --output-dir "$smoke_dir" \
  --device cuda:0 \
  --smoke
```

smoke 的硬性通过条件：

1. exit code 为 0；
2. `summary.json` 存在且 `verdict == "SMOKE_COMPLETED"`；
3. M=1 `n_users == 256`；
4. M=1 `exact_top50_mismatch_vs_p0 == 0`；
5. 日志无 Traceback、OOM、NaN。

如果任一条件失败：

- 立即停止；
- 保留临时目录和日志；
- 向用户报告；
- **不得自动修改后重跑，不得启动正式实验**。

### Step 3：smoke 通过后，后台启动正式筛查

正式实验必须通过 runner 在 tmux 后台运行：

```bash
bash experiment/phase13/run_tier1_resolver_toys_multi_interest.sh start <GPU>
```

runner 会自动完成：

- Bash/Python 静态检查；
- 21 项相关单测；
- 输入文件存在性检查；
- 禁止 GPU0/5；
- free-memory 与 compute-mode admission；
- 真实 CUDA allocation admission；
- tmux 后台运行；
- 30 分钟硬超时；
- 每 10 秒 GPU telemetry；
- 原子更新 `status.json`；
- 禁止覆盖已有正式 artifact；
- 失败后不自动 retry；
- 完成后不自动进入下一阶段。

启动后只需确认一次状态确实进入 `arm_evaluation` 且有 `workload_pid`：

```bash
bash experiment/phase13/run_tier1_resolver_toys_multi_interest.sh status
```

确认后台任务已正常开始后，不需要实时监看；用户会自行观察 artifact 下的 `status.json`。

### Step 4：正式结果完成后的最低审计

正式完成后至少核对：

1. `status.json` 为 `completed`；
2. `summary.json`、`config.json`、`arm_m_001.json`、`arm_m_002.json`、`arm_m_004.json` 均存在；
3. M=1 exact top-50 mismatch 为 0；
4. M=1 cold R@50 events 为 498；
5. `control_reproduction_passed == true`；
6. 两个比较均有 10,000 次 paired bootstrap；
7. 主比较 confidence 为 0.975；
8. 日志无 Traceback、OOM、NaN；
9. `test_read/beauty_read/sports_read` 均为 false；
10. `automatic_next_stage == false`。

---

## 8. 结果解释边界

正式结果完成后：

- 若 verdict 为 `FAIL_STOP_MULTI_INTEREST_TIER1`：T1-4 失败，Tier-1 轻量 resolver 探索结束；不要继续调 M、聚类轮数、decay 或做第 5 个便宜变体；
- 若 verdict 为 `PASS_T1_4_SIGNAL_REQUIRES_REPLICATION`：只有开发集信号，先汇报，不自动跑 Beauty；
- 若 verdict 为 `PASS_T1_4_TOYS_GATE_REQUIRES_REPLICATION`：达到 Toys 门槛，但仍需用户决定是否申请 Beauty replication；
- 若 verdict 为 `CONTROL_REPRODUCTION_FAILED_STOP`：实验无效，保留 artifact 并停止；
- 无论结果如何，都不能自动读取 test，也不能自动启动路线 B。

路线 B 预计需要明显更多显存与时间，需单独向用户申请资源；此前计划中的资源口径约为一张 **30 GiB 级可用显存** GPU，实际启动前必须重新确认。

---

## 9. 给续跑 AI 的最短交接提示

```text
阅读 report/第十三阶段/GRAM_第十三阶段_T1-4_多兴趣Resolver_续跑交接报告.md。
不要使用 GPU0/5，不要动别人的进程，不要自动 retry，不要自动进入路线 B。
当前正式实验未启动。先实时检查 GPU；选择 compute_mode=Default 且至少 4 GiB free 的非 0/5 卡。
先运行修正后的 256-user CUDA smoke；只有 M=1 exact_top50_mismatch_vs_p0==0 才能继续。
smoke 通过后，用 run_tier1_resolver_toys_multi_interest.sh 在 tmux 后台启动正式 M=1/2/4 筛查。
启动后确认一次 status 进入 arm_evaluation 即停止监看，用户自行观察 artifacts 下 status.json。
```
