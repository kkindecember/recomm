# GRAM 第七阶段：ST-GCGD-v2.1 深层时序图修复计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-08-02
- Verification Status: PREREGISTERED_P0_R2
- Parent Result: `ST_GCGD_V2_FAIL_CLOSED_BEFORE_P1`
- Development Domains: Toys、Beauty
- Sports/Test/P1: 封存

## 1. 修复假设

v2 的有向 `R_ii` 在两域 rank metrics 均优于 static，但浅层 mixer 稀释了 transition 信号。
v2.1 不做小幅 alpha 搜索，而升级为 256 维、三层 relation-specific propagation，并增加
GRU session encoder。完整方法以 transition 为主干，`R_ui` 只能通过上限 20%、初始近零的 residual
gate 进入；若 full 不能守住 transition-only 的 Recall@10、NDCG@10、Recall@50，则自动选择
transition-only 作为 train-only P0 输出。

目标分离统一使用 catalog 内 z-separation，避免不同 arm 的原始 logit 尺度不可比。模型仍通过
static、R_ui、R_ii、full 四臂保持结构可辨识。

## 2. P0-R2

- 冻结 GRAM checkpoint，只使用 train-only pseudo-future；每域构造 1,024 head + 1,024 tail
  hard-negative bank。
- GRAM bank 完成后释放 GRAM CUDA context、清空 cache、重置 PyTorch peak counter，再执行完整
  graph-only forward/backward，测得 P0-G2 实际 workload budget。
- telemetry 通过共享 `current_dataset.txt` 读取域名，不依赖后台 shell 的变量快照。
- P0-R2 无 sidecar，只测 workload；退出后恢复 CodeLlama。

## 3. P0-G2 门

- seed=2023；新的 train-only split salt；不读取 validation/test/Sports/P1 cohort。
- hard-negative bank、实现、测试、P0-R2 summary 与 checkpoint 均 SHA 锁定。
- 每域 workload budget + sidecar 必须等于 30,720 MiB。
- full 安全门先比较 transition-only；不安全则回退 transition-only。
- 最终 selected arm 必须相对 static 提高 mean target z-separation，且 Recall@10/NDCG@10 不得同时下降。
- 任一域失败则不进入 P1；不得自动重试。
