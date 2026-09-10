# 第十九阶段：GRAM 提点实验总计划

## Material Passport

- Date: 2026-09-10
- Origin Skill: academic-research-suite / experiment-agent
- Mode: implementation + run
- Objective: 当前只以实际提点为目标，允许迁移已有方法，不评估创新性。
- Scope: Toys 与 Beauty 正常场景，seed 2023，训练集与验证集；不读取测试集。

## 阶段目标与工作规则

本阶段寻找相对原 GRAM 能提高最终指标的方法。2026-09-10 用户明确：RaSeRec→GRAM 与 RaSeRec→GRAM+PCRF 都以原 GRAM 为主对照；原 GRAM+PCRF 仅作辅助对照，不作为补充数据集或继续实验的门槛。允许直接迁移已有论文及作者代码，现阶段不以创新性作为筛选条件。当前候选是 RaSeRec→GRAM；后续方法、日程调整和重要决定都在本总计划中更新。

遵循用户确认的实验规则：

- 预计超过10分钟的实验放入 tmux 等后台环境执行；确认启动成功后结束当前监看，不让助手持续等待或轮询。
- 运行程序自己更新 artifacts 下的状态文件。用户询问进度时按需读取一次，不为了观察实验持续消耗对话 token。
- 本阶段维护一个总 plan 和一个总 report。重要结论更新到[实验进展与结论总报告](../../report/第十九阶段/GRAM_第十九阶段_实验进展与结论总报告.md)，不按每次检查拆分文档。
- 仅凭候选覆盖、训练 loss 下降或预训练分支变好，不宣称 GRAM 已经提点；最终以验证集上实际生成结果及 PCRF 组合结果判断。

本计划由原第十八阶段的 RaSeRec 单项计划归并迁入。已经运行的脚本、配置和输出仍使用原路径，避免中断任务；第十九阶段提供统一入口，不复制训练或重复启动。

## 第一版做什么

直接采用 RaSeRec 作者实现中的两层64维序列编码器、`us_x` 同目标对比预训练，以及检索增强的双通道注意力。作者代码固定在提交 `44d2d2a4f52ddd97a5bd4153dab6ee0e89d5ceca`，保留 MIT 许可证。核心神经层和 DuoRec 类从作者源码逐字提取，使用语法树一致性检查核对。

完成预训练后，冻结序列编码器，使用训练集的“历史→后续商品”构造案例库，训练原方法的注意力模块。之后将原用户表示、RaSeRec增强表示及20组案例的历史/后续商品表示投影成42个额外记忆token，交给原 GRAM 解码器读取。保留 GRAM 原始词汇ID、文本输入、FiD记忆和父模型权重。

GRAM 这部分是必要的接口改造，不是原 RaSeRec 的独立复现。它允许生成器直接读取案例，同时保留完整 RaSeRec 输出。

## 与作者代码的明确差异

1. 使用当前工程固定的数据划分和 item ID 映射；记忆库始终只含训练集。
2. 用精确余弦 top20 替代 Faiss IVF 的近似搜索，不需要安装新的检索框架。
3. 排除查询用户的所有案例，较作者允许同用户早期案例的规则更严格。
4. 用验证 NDCG@10 选模型，匹配当前提点目标；序列长度20匹配 GRAM。
5. 保留作者示例的 alpha=0.5、beta=1.0、dropout=0.5。beta=1.0 下原方法的第二通道不参与最终混合；第一版不额外搜索它。作者在两个通道中共享使用 seq_tar FFN 的实现也保留。
6. 在 GRAM 阶段训练记忆投影与 RaSeRec 注意力，再联合微调原 GRAM；额外加入权重0.1的原商品预测损失以保留推荐表示。

## 执行安排

**2026-09-10 核查快照：** Toys 已完成序列预训练、案例库、注意力训练及真实 GPU 接口检查，进入 GRAM 第11/11轮。用户已授权补充 Beauty 同配置实验并使用 GPU1 的实际可用显存。实时状态以 artifacts 中的文件为准。

| 阶段 | 数据与训练安排 | 输出 |
|---|---|---|
| 序列预训练 | 完整109361条训练前缀，batch1024，最多100轮；至少10轮后允许验证连续10轮无改善停止 | pretrain/best.pt |
| 案例库 | 全部训练前缀；训练/验证查询仅使用历史 | bank/ |
| 原 RaSeRec 注意力训练 | 冻结预训练编码器，完整数据，最多100轮，验证耐心10轮 | reader/best.pt |
| GPU 接口检查 | 冻结/联合反传与真实beam50生成，失败记录原因并停止 | smoke.json |
| GRAM训练 | 完整数据，1轮冻结父模型 + 10轮联合训练，microbatch4、有效batch128；固定日程，不由小队列提前停止 | gram/latest.pt |
| 验证 | 每轮固定2000用户观察；分别按单独模型及PCRF组合选择检查点，最后全19412人验证 | gram/*full.metrics.json |

首轮使用 GPU 7（UUID写入配置），启动要求至少18500 MiB空闲，进程PyTorch显存上限设为卡总容量的34%。总运行硬时限48小时，不自动延长，不自动重试；不改变现有DIFF任务。

## 时长与执行顺序

启动时间为 **2026-09-09 18:34:57（北京时间）**，48小时硬截止时间为 **2026-09-11 18:34:57**。48小时是预算上限，不代表一定要运行到该时间，也不保证能够在上限内完成所有步骤。

9月10日09:50，依据本轮 Toys 实测吞吐修订为剩余约6–8小时，预计当日16–18点完成。最近2000人验证约15–16分钟，完整19412人验证每次约2.5小时，训练后须执行两个选中检查点的完整验证。

Beauty 暂按总计24–32小时排期，包含完整预训练、案例库、注意力训练、11轮 GRAM 及两次22363人验证；这是参考 Toys 实测和 Beauty 规模的估算，等待本轮实际完整轮次后再修订。训练硬上限48小时，资源等待不计入训练预算。共享 GPU 负载会影响完成时间。

## 如何判断是否值得继续

最终分别比较：

- RaSeRec→GRAM 与原 GRAM 的 NDCG@10、Hit@10；
- RaSeRec→GRAM+PCRF 与原 GRAM 的相同指标；原 GRAM+PCRF 保留为辅助对照。

PCRF 固定各域已有 item-head、lambda=1、beta=0.5、gamma=1；Toys 的 q1=5，Beauty 的 q1=6，均沿用各域历史固定参考。每域已在64个原GRAM验证beam上核对，PCRF排名与既有结果完全一致；所有验证用户的目标训练频次也与既有记录一致。

## Beauty 补充实验（2026-09-10 授权）

- 配置：[s19_raserec_gram_beauty.json](../../experiment/phase19/config/s19_raserec_gram_beauty.json)。所有模型、预训练、reader、GRAM训练超参数及48小时预算与Toys一致。
- 数据：131413条训练前缀、22363名验证用户、12101个商品；使用Beauty原GRAM epoch25父模型及Beauty已有PCRF item-head。
- 资源：GPU1。用户明确以实际可运行显存为准，替代18.5 GiB保守门槛及旧30 GiB占位规则；参考Toys进程实测10882 MiB，准入设为12000 MiB，PyTorch分配上限为整卡28%。这仅改变资源限制，不改变模型、样本、microbatch4、有效batch128、1冻结+10联合轮次或beam50。
- 后台会话：`s19_raserec_gram_beauty_v1`；启动命令：`bash experiment/phase19/run_stage19_raserec_beauty.sh --mode run`。
- [Beauty状态](../../artifacts/phase19/beauty_status.json)由监督进程每30秒原子更新，包含工作阶段、进度、PID、心跳、GPU、日志位置和终态。训练原始状态仍由原runner写入输出目录的`status.json`。
- 状态测试5项通过；[Beauty真实数据预检查](../../artifacts/phase19/raserec_gram_beauty_preflight_v1/summary.json)通过；真实GRAM GPU检查在前置训练后自动执行，通过后才进入GRAM训练。
- 最早一次显存准入拒绝发生于训练启动前，未创建模型或优化器；记录保留于输出目录`attempts/`。不自动重试已经失败的训练。

当前目标是找到能涨点的组合。首轮不要求证明案例上下文的独立贡献，也不并行启动额外消融；若出现增益，再决定重复验证和后续改进。单次验证提升仍只代表该次运行，不能提前宣称稳定提升。

## 执行与追踪

**统一状态总览：** [artifacts/phase19/status.json](../../artifacts/phase19/status.json)，包含Toys与Beauty，由Beauty后台监督进程在运行期间每30秒更新；单独入口为[Toys状态](../../artifacts/phase19/toys_status.json)和[Beauty状态](../../artifacts/phase19/beauty_status.json)。总览带更新时间，监督进程结束后使用各域状态文件检查最新记录。原Toys训练状态文件不被覆盖。

**统一实验入口：** [artifacts/phase19/raserec_gram_toys_v1](../../artifacts/phase19/raserec_gram_toys_v1)。

后台会话：`s18_raserec_gram_toys_v1`；启动PID：`3680679`。PID仅作启动记录，不作为永久进程存活证明。

配置：`experiment/phase18/config/s18_raserec_gram_toys.json`

启动：`bash experiment/phase18/run_stage18_raserec_gram.sh --stage all`

输出：`artifacts/phase18/raserec_gram/toys/v1/`

进度：`status.json`、`events.jsonl`；配置、数据哈希、导入源码快照存入 `manifest.json` 和 `sources/`。

基础验证：7项测试通过；真实数据预检查存于 `artifacts/phase18/raserec_gram/toys/preflight_v1/summary.json`。

前期CPU检索检查已归档于 `artifacts/phase19/memory_probe_v1/`；2026-09-09状态与估时依据冻结于 `artifacts/phase19/review_20260909.json`。

来源：[作者仓库](https://github.com/HITsz-TMG/RaSeRec)、[论文](https://arxiv.org/abs/2412.18378)。
