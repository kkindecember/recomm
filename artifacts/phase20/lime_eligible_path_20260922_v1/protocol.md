# LIME 过滤与路径融合后台筛查

用户授权：2026-09-22 明确要求自主后台启动，通过 status 文件观察；允许分卡和共享 GPU。本轮不等待 Beauty 干净基础训练结束，使用已冻结旧模型完成开发筛查。

## 固定设计

- Toys、Beauty 各 512 名评价用户；按用户 ID 的固定 SHA256 顺序选取，排除前三轮解码的所有用户和门控拟合用户。复用最早 3 个用户做工程检查，不参与评价。
- 使用既有 GRAM、专家分数和历史校准，alpha=0.3、专家分数尺度=20、beam=50、长度惩罚=1。无新增训练或选参，rho=0、0.5、1 完整报告。
- 过滤范围严格沿用原最终排序的输入 history，通常最后 20 项；不扩展到全生命周期历史。
- 原始模型和历史开发数据被多次使用，因此只能作为机制筛查，不能冒充干净测试结论。

## 七个真实解码对照

| 名称 | 内容 |
| --- | --- |
| GRAM | 原模型，生成后过滤 |
| fixed | 当前固定融合，生成后过滤 |
| GRAM_filter | GRAM 提前过滤 |
| hard_filter | 固定融合提前过滤；专家只在剩余立即子分支内重新归一化 |
| eligible_fixed | 提前过滤，且专家在所有层级只聚合合格商品质量；仍固定融合 |
| path_rho05 | 与 eligible_fixed 相同过滤，路径后验更新权重，rho=0.5 |
| path_rho10 | 与 eligible_fixed 相同过滤，路径后验更新权重，rho=1 |

另外保留 fixed 的同规则后重排，四个过滤融合方案在 GRAM_filter 候选上的同规则后重排，原始/过滤 GRAM 候选上的直接 LIME 重排，以及 LIME 全目录结果。每用户只做一次编码，复用 cross-attention cache；后重排由真实路径上的原生分数重算，不额外生成。

## 机制与分数约定

在当前合格子分支上，p 为 GRAM 条件概率、q 为专家条件概率、m 为 GRAM 原始合法词元总质量。输出保留 `m * ((1-w)*p + w*q)`。每选一个词元，计算 `r=w*q/((1-w)*p+w*q)`，下一步权重为 `(1-rho)*alpha+rho*r`。

rho=0 精确回到当前固定融合；rho=1 的完整路径目标为 `product(m) * ((1-alpha)*product(p)+alpha*product(q))`，再遵循原 HF EOS/变长评分。不能忽略 m 或长度惩罚而宣称完整目录分布等价。

“该模型更支持已生成路径”不等于“该模型更可靠”。记录权重变化和饱和，避免把自我强化解释成正确性保证。两档 rho 只是先验指定的可证伪筛查，不依据本轮评价挑出赢家后再把同批评价当确认结果。

## 检查和结果

运行前 9 项 CPU 测试；每域先做 3 用户工程检查，包含原始和过滤条件下 alpha=0 的精确退化、全部路径分数重算、共享候选与同规则后重排分数一致、过滤后 50 个候选全合格。每域工程检查 27 次、评价 3584 次真实解码，总 3611 次。

输出 NDCG/Hit@5/10/20/50、成对差异与描述性 bootstrap 区间、命中新增/丢失、有效候选数量、生成步数、墙钟耗时和权重诊断。最后由单独脚本检查输入 SHA256、用户排除、原始打分排序、目录路径和所有指标。

重点比较：hard_filter 对 fixed、eligible_fixed 对 hard_filter、path_rho05/path_rho10 对 eligible_fixed。两个域同时 NDCG@10 相对增益至少 0.5% 且净 Hit@10 不降，才标记为值得干净复验。此阈值仅用于算力分配，不是显著性、等效性或投稿保证；完整保留负结果。

## 后台资源与状态

Toys 固定 GPU 1（UUID `GPU-2afcc1df-7df8-1f3f-8970-a252a06c6160`），Beauty 固定 GPU 5（UUID `GPU-07374397-b213-09c1-2988-718acb6ed231`）。每进程显存上限 11 GiB，启动前等待至少 12 GiB 空闲。两域并行，不占用当前 Beauty 基础训练的 GPU 6/7。

主状态：`artifacts/phase20/lime_eligible_path_20260922_v1/status.json`；按 15 秒聚合域状态。各域在每个实验臂开始和每用户完成时更新自己的 `status.json`。`evaluation_users_written/total` 为进度，`projected_remaining_hours` 为近期完整用户耗时估计，受共享 GPU 影响会波动。

完成后自动核验并生成 `summary.json`、`verification.json` 和报告。异常写 FAILED/INCOMPLETE/AUDIT_FAILED/SUPERVISOR_FAILED 及错误；后台不静默重试、不按结果追加实验、不设置短墙钟超时。由后台 supervisor 检查子进程存活；交互助手只做启动检查，不持续监看。

启动命令：`bash experiment/phase20/run_lime_eligible_path.sh supervise`，宿主 tmux 持久运行。数据准备单独执行 `prepare`，禁止覆盖已有目录或重复启动。
