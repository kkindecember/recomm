# GRAM 第十八阶段：S18-2 机制探针执行补遗 v0.1

## Material Passport

- Parent Plan：`GRAM_第十八阶段_PCPS-GRAM词法锚定协同前缀生存与低风险验证计划v0.2.md`
- Created：2026-09-07
- Authorization：研究者明确要求“阅读我的实验计划，我应该是18-1已经过了，帮我开始18-2”。本次授权包括实现、合同验证和单卡 bounded probe。
- Predecessor：`artifacts/phase18/s1r_disjoint_confirmation/run-0001/summary.json` 的 `S18_1R_ACTIONABILITY_REPAIRED_PASS`。
- Status：`PREREGISTERED_BEFORE_S18_2_TREATMENT_OR_EFFECT_READ`
- Config：`experiment/phase18/config/s18_s2_mechanism_probe.json`。
- Scope：只执行 S18-2；不自动进入 S18-3，不打开 I1/I2、D1/D2、official validation/test 或 Sports。

## 1. 本补遗冻结的未决实现细节

主计划要求精确 normalized、margin、截断与 empty fallback 在代码合同中冻结，但现有 S18-1
配置只定义了诊断，尚未包含 treatment loss。本补遗在任何 S18-2 treatment 结果之前补足这些定义。

- Toys/Beauty 分别使用原 S18-1 cohort 固定排序前 1,000 名；前 100 名用于梯度校准与 overfit。
  不使用 S18-1R 未见确认 cohort，不按 target 是否命中、first-drop 或 CF 效果筛用户。
- 复用两域 I0 epoch-10 parent 与 item-head，校验原始 SHA；每臂与每子阶段均从同一 parent 独立开始。
- 每名用户训练一个 I0 **可见 prefix 内最后一个 transition**：visible[:-1] → visible[-1]。
  这不是 I0 留出 target。I0 target 仅用于 probe evaluation，独立保存在 evaluation artifact。
  不利用未来 offset 构造训练样本；不重新拟合 teacher、frequency 或 catalog。
- 100-user overfit 固定 5 epochs，四臂均必须降低这 100 个训练样本的平均 CE，未降低只检查实现合同，
  不增加 epoch。1k probe 固定 1 epoch，重新从 parent 开始，不沿用 overfit checkpoint。
- 单用户 microbatch，16 用户梯度累积；不足一组按实际用户数归一化。AdamW lr=1e-3、weight decay=0.01、
  eps=1e-6、warmup=0.05、clip norm=1.0。四臂参数量、用户顺序、步数相同，每用户重设同一随机种子。

## 2. 精确 loss、负例和对照

- `m_prefix=m_path=0`，`a(u,d)=1`。前缀项按每用户非空合法 sibling 节点求均值；空节点不产生项，
  用户无分支时 prefix=0；不按 identifier 长度、深度或效果再加权。
- 每个 target prefix 节点按主计划构建 K=8 item negatives，再对 next-token child 去重；非法 child
  和 target child 永不进入 denominator。M0 用冻结 PCRF 排序，A0 用 parent native 排序，
  A0 不足 K 时按 item 字典序补齐，不访问 CF 或 frequency。
- 完整路径项固定最多 8 个 parent beam200 实际生成的错误 item（native parent 顺序最高的前 8 个）。
  此 bounded 截断在首次 treatment 前冻结，A0/M0/S0 使用**同一批完整路径**，只改变 CF 权重和 prefix mining。
  `S_parent` 和 `S_student` 均为完整未掩码词表 teacher-forced log-probability 总和除以含 EOS 的 path 长度，eta=1。
- `w_model=softmax(S_parent)`；M0/S0 的 CF path 分数使用全 catalog 的
  `z(z(cf)-0.5*z(log1p(freq)))`；`w_cf=1+reliability*sigmoid(cf_pc_negative-cf_pc_target)`，最终权重和为 1。
  reliability 来自当前用户 parent 原生 beam50 top10 的 train-only tail mass。
- `L_PCPS=0.5*L_prefix+0.5*L_path`，按用户等权。使用稳定 logsumexp/logaddexp，无数值裁剪；
  无错误 beam path 时 path=0，保留其余损失。负例数量不触发增大 K 或替换参数。
- S0 使用 1,000 名同域用户的哈希排序循环置换，固定且无 fixed point。只替换 user→CF score vector，
  保留当前用户的 parent beam、target、frequency、reliability、训练 batch 与参数量。
  置换用户的输入也仅是该用户的训练 transition history，不读 I0 留出目标。
- A0、M0、S0 的架构和实际完整路径打分数目相同；C1/M1 仅从 C0/M0 beam50 确定性重排。

## 3. Alpha、工程与机制门槛

在任何 treatment 优化和 I0 效果评估之前，以 eval mode（关闭 dropout）分别累积两域前 100 个
训练样本的 CE 与 PCPS **平均梯度向量**，计算全可训练参数的 L2 norm ratio。
实际约束为 `alpha*||mean grad PCPS||/||mean grad CE||` 在 `[0.10,0.30]`，
同一 alpha 必须同时满足两域。仅允许 `{0.1,0.3}`；若两个都符合，选择 macro ratio 更接近 0.20 的值，
再以较小 alpha 破平局；若都不符合，记录 `GRADIENT_CALIBRATION_FAILED` 并停止。
不存在按 accuracy/NDCG 调整归一化、alpha、margin 或训练轮数的分支。

工程 smoke 使用每域最长可见历史样本测显存；确认 native GRAM input identity、alpha=0 loss/logits
误差 <=1e-6，以及同一训练更新后的 beam50 exact match；确认 prefix/path 梯度对 target 为负、
对合法 negative 为正，反向传播 finite，真实 beam50/200 路径合法。

机制 survival 定义为每用户**非 EOS target token prefixes**在 traced active beam50 中存活的比例，
再按用户平均；EOS 的完整 item 命中另报 Hit@50，避免两个指标机械相同。
path margin 使用每个 I0 evaluation 用户冻结 parent top8 wrong paths 的 parent-softmax 加权均值，
所有 arm 使用同一 anchor；不随 treatment candidate 变化而更换负例。

两域分别执行主计划原数值门槛。对未明确的 A0 “相同或更好”，采取保守解释：M0 必须在 survival
和 Hit@50 上均严格超过 A0，否则 `CF_GUIDANCE_NOT_ADDITIVE`。若 Toys probe 已失败即停止，
不继续消费 Beauty probe 预算；报告明确未完成域，不把局部结果写为双域成功。

## 4. 运行与审计

- 单卡串行；GPU1/GPU4 排除，不中断任何现有进程，不启动占位轮（本步骤属于 bounded 单卡实验）。
- 超过 10 分钟的阶段均在具名 tmux 中执行，由 supervisor 写 status、进程存活与 24 小时 hard timeout。
  agent 只作短启动握手，研究者通过 status 查看长任务，不进行实时盯跑。
- 先做独立 engineering smoke，根据实测 peak + 1 GiB 检查启动余量，并报告基于 smoke 的时长估计。
- canonical scientific attempt 仅 `run-0001`；config/input/source SHA、source snapshot、attempt ledger、
  子阶段日志、逐用户结果及报告各自落盘。任何科学 Gate 失败立即停止，不自动重试、不启动 S18-3。

```bash
bash experiment/phase18/run_stage18_s2_mechanism_probe.sh prepare
bash experiment/phase18/run_stage18_s2_mechanism_probe.sh launch --kind smoke --run-name smoke-0001 --gpu 7
bash experiment/phase18/run_stage18_s2_mechanism_probe.sh launch --kind formal --run-name run-0001 --smoke-run smoke-0001 --gpu 7
```
