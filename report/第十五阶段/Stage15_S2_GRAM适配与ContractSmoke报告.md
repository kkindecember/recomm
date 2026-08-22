# Stage15 S2：GRAM 适配与 Contract Smoke 报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-22
- Verification Status: VERIFIED（本地 artifact、状态文件、hash、日志与测试已核对）
- Version Label: stage15_s2_contract_smoke_v1

## 结论

S15-2 contract Gate 结论为 `PASS_S15_2_CONTRACT`。Toys 的安全 train+validation 投影、B0 逐位 parity、B2/B3 train-only 输入、冻结 GRAM verifier/probe、B2 auxiliary drafter state 与 B3 edit state 均形成可审计闭环。该结论只证明机制、接口、安全与状态合约成立，不构成 efficacy 结论。

## Gate 证据

| 子 Gate | Verdict | 关键证据 |
|---|---|---|
| S15-2P 双域安全投影 | PASS | Toys 8,789 行，Beauty 10,655 行；机械删除每行原末项；后续运行 denylist 原 `user_sequence.txt` |
| Toys B0 projection parity | `PASS_B0_PROJECTION_PARITY` | 16/16 target-independent users 的 beam-50 逐位一致；0 mismatch；projected target parity=true |
| B2/B3 输入合约 | `PASS_B2_B3_INPUT_CONTRACT` | 5,963/5,963 cold catalog coverage；59,630 pseudo-context；302,400 position-wise requests；train-only；0 cold item seen in train |
| verifier + layer probe | `PASS_B2_VERIFIER_GPU_HOOK_AND_B3_TRAIN_ONLY_PROBE` | 512/512 score finite；direct parity 最大误差 `4.29153e-6 < 2e-5`；selected layers=`[5,5,5,5,5,4]`；GRAM hash 不变 |
| B2 drafter state | `PASS_B2_TRAIN_ONLY_DRAFTER_STATE_SMOKE` | 2 epoch loss finite；state 改变；16×top-50 均为唯一已知 item；GRAM 未加载进 optimizer；peak CUDA 229.47 MiB |
| B3 edit state | `PASS_B3_TRAIN_ONLY_EDIT_STATE_SMOKE` | delta finite/nonzero；6/6 positions trigger 生效；EOS/padding inactive；未编辑 prompt exact parity；GRAM hash 与 checkpoint SHA 不变；peak CUDA 2,649.24 MiB |

## B3 GPU7 attempt-2 恢复核验

- Exact command: `bash experiment/phase15/run_stage15_s2_toys_b3_edit_state_smoke_attempt2.sh start 7`
- Started: `2026-08-22T15:09:13+08:00`
- Completed: `2026-08-22T15:10:14+08:00`
- Worker rc: `0`
- Runtime: `55.72 s`
- Artifact bytes: `196,733,573`
- Covariance: 256 条 SHA 固定 train-only transitions；positions 0–4 各 256 activations，position 5 为 32 条最长路径覆盖
- Edit smoke: 每 position 4 个互异 request，共 24 条；成功 z requests by position=`[2,2,1,1,1,1]`
- Frozen base: model hash `80d089...cc6f` 前后一致；checkpoint SHA256 `d71fc...550` 前后一致
- Leakage guards: validation 未用于 covariance/request/layer selection；未打开原 `user_sequence.txt`、`similar_item_sasrec.txt`、test predictions 或 test metrics

## 历史尝试与保留策略

- B0 GPU3 attempt-1/2 均只在显存 admission 阻塞；GPU4 attempt-3 PASS。
- verifier/probe attempt-1 为 chronological-key 工程错误；attempt-2 因 TF32 numerical parity FAIL；attempt-3 缺 CuBLAS deterministic 前置环境；attempt-4 PASS。全部失败产物保留。
- B2 drafter GPU2 attempt-1 只在显存 admission 阻塞；GPU1 attempt-2 PASS。
- B3 GPU1 attempt-1 只在显存 admission 阻塞；GPU7 attempt-2 PASS。未自动 retry，未覆盖旧 artifact。

## 限制与下一 Gate

Smoke 只验证有限状态构建与接口语义，不评估 cold gain、warm cost 或统计显著性。下一步是 S15-3A：使用 Stage14 item-disjoint pseudo-cold 数据执行固定 512-event admission，仅检查完整 beam/candidate 路径、finite score/loss/update、对象变化、资源与 artifact schema；admission 不用于 efficacy 选择。
