# GRAM 第十四阶段 Stage 14-0B：Toys Smoke 报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-20
- Verification Status: VERIFIED_SMOKE
- Version Label: phase14_stage14_0b_toys_smoke_v1

## 结论

`PASS_STAGE14_0B_TOYS_SMOKE`。冻结 GRAM checkpoint 的 validation-only 诊断管线已端到端跑通：2 users、beam=50、`rc=0`，frozen prediction parity mismatch = 0，未读取 test prediction。正式规模暴露 beam filler 后，已用 score-aware observer 重跑并保持上述 parity。

- Phase14 回归测试：20/20 通过。
- 运行时：12.41 秒；peak CUDA allocated：2,340.99 MiB。
- 2-user 样本均在 depth 3（normalized depth 0.6）首次跌出 beam；其中 cold 样本 depth-3 legal rank=139。该结果仅作管线 sanity check，样本量不足以支持路线判断。
- 工程结论：当合法 continuation 少于 beam width 时，Transformers 会保留累计分数为 `-inf/-1e9` 的 filler rows，它们可能形成任意越界前缀。约束回调必须完全匹配冻结 GRAM trie 的空返回语义；prefix survival 由约束后的 score-aware observer 统计，仅接受非 sentinel 的有限累计分数，不能仅凭 callback 出现与否判 live。

## 产出与下一步

128-user score-aware medium smoke 进一步通过：`rc=0`、frozen beam parity mismatch=0、运行 35.70 秒、peak CUDA allocated=8,085.10 MiB；覆盖 cold 75 / warm 53 users。cold H@50=0，首次跌出 beam 的中位 raw depth=2（normalized=0.4），但该便利样本仅用于验证诊断器，不作正式路线 Gate。

- 状态：`artifacts/phase14/diagnostics/oracle_prefix_probe_toys_smoke_gpu0_retry4/status.json`
- 汇总：`artifacts/phase14/diagnostics/oracle_prefix_probe_toys_smoke_gpu0_retry4/summary.json`
- 明细：`artifacts/phase14/diagnostics/oracle_prefix_probe_toys_smoke_gpu0_retry4/per_user_validation.jsonl`
- Medium 状态/汇总：`artifacts/phase14/diagnostics/oracle_prefix_probe_toys_medium_gpu0_score_aware/{status.json,summary.json}`

下一步是 Toys/Beauty 全 validation 的 Stage 14-0B 正式双域诊断；在双域汇总前不触发 R2PD 路线 Gate，也不启动模型训练。
