# 第十九阶段：GRAM 提点实验总计划

## 2026-09-10 最终收尾决定（覆盖下文历史执行安排）

**本阶段已关闭。** Toys完成11轮和两份19,412人全量验证；单独模型NDCG@10相对原GRAM+0.029%，预选epoch8组合+2.917%，相对原GRAM+PCRF为−0.275%。用户本次决定停止剩余Beauty：已保存5/11轮，第6轮中断，20:41确认进程退出；没有Beauty最终全量结果。规范状态为`USER_STOPPED`，取消当前版剩余训练、全量验证与自动扩展，不重启或追加种子。

主对照仍为原GRAM；本次是用户调整资源安排，不是事后更换主比较或把不完整Beauty视为完整负结果。下文关于继续Beauty、未来完成时间和完成后扩展的安排均为历史。详见[最终总报告](../../report/第十九阶段/GRAM_第十九阶段_实验进展与结论总报告.md)及[停止证据](../../artifacts/phase19/closeout_20260910/evidence.json)。

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

**2026-09-10 16:00 核查快照：** Toys 于15:52:42完成全部11轮GRAM及两个检查点各19412人的完整验证，总耗时约21小时18分。按既定选模规则，单独模型使用epoch4，PCRF组合使用epoch8。单独模型相对原GRAM基本持平；组合NDCG@10提高2.92%、Hit@10提高4.18%。Beauty已通过真实GPU接口检查，正在GPU1进行GRAM第3/11轮。建议保留已授权的Beauty完整日程；实时状态以artifacts中的文件为准。

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

Toys实际于9月10日15:52:42完成。最后两个19412人完整验证分别约2小时24分和2小时19分。

Beauty第2轮联合训练加2000人验证实测约2小时15分，其中验证约21分钟。按此速度，9月10日16:00起预计还需约26–30小时，可能于9月11日18–22点完成；这替代先前总计24–32小时的粗估。估算包括余下训练和两次22363人完整验证，后者按当前子集吞吐推算每次约3.9小时。训练硬上限仍为48小时，实际截止9月12日10:28:01；资源等待不计入训练预算。共享GPU负载会影响完成时间。

## 如何判断是否值得继续

最终分别比较：

- RaSeRec→GRAM 与原 GRAM 的 NDCG@10、Hit@10；
- RaSeRec→GRAM+PCRF 与原 GRAM 的相同指标；原 GRAM+PCRF 保留为辅助对照。

PCRF 固定各域已有 item-head、lambda=1、beta=0.5、gamma=1；Toys 的 q1=5，Beauty 的 q1=6，均沿用各域历史固定参考。每域已在64个原GRAM验证beam上核对，PCRF排名与既有结果完全一致；所有验证用户的目标训练频次也与既有记录一致。

## Toys最终结果后的安排（2026-09-10）

Toys完整验证的主结果如下；详细核查及限制持续记录于唯一总报告，逐用户复算与来源哈希见[9月10日核查快照](../../artifacts/phase19/review_20260910.json)。两份完整预测的原始指标和PCRF指标已在CPU重算，与保存结果一致。

| 方法 | Hit@10 | NDCG@10 | 相对原GRAM的NDCG@10 |
|---|---:|---:|---:|
| 原GRAM | 0.119410674 | 0.076274514 | — |
| RaSeRec→GRAM，epoch4 | 0.118483412 | 0.076296849 | +0.029% |
| RaSeRec→GRAM+PCRF，epoch8 | 0.124407583 | 0.078499149 | +2.917% |
| 原GRAM+PCRF，辅助对照 | 0.125334844 | 0.078715522 | +3.200% |

1. **继续Beauty当前完整日程。** 组合已在Toys全量验证上超过指定主对照原GRAM，单独模型则尚无明确提点；不把原GRAM+PCRF改成继续Beauty的门槛。Beauty第2轮子集暂低于原GRAM，但目前只有1轮联合训练结果，不据此提前截断既定11轮。
2. **冻结本轮选模解释。** Toys的epoch4加PCRF全量NDCG@10为0.078816546，高于epoch8组合，但它不是按组合子集指标选中的检查点，只作补充观察，不事后替换epoch8主结果。单独模型在2000人选模子集的+2.82%未保持到全量验证，后续不能仅凭该子集决定成功。
3. **Beauty完成后再决定扩大实验。** 先报告两个既定检查点在全部22363人上的NDCG@10、Hit@10及逐用户得失。如果组合在第二域也超过原GRAM，再优先补充两个固定种子（总计3个），冻结超参数与选模规则，汇总均值和波动；这属于下一轮建议，本次不启动新训练。
4. **区分组合效果与新增模块收益。** Toys的epoch8组合略低于原GRAM+PCRF，RaSeRec单独基本持平，尚无其额外收益的证据。如果Beauty同样只呈现已有PCRF的收益，下一步优先评估原GRAM+PCRF的成本收益，避免继续扩大同版RaSeRec调参；若Beauty两种候选最终都不超过原GRAM，则不扩展此版跨域复验，转向下一候选。辅助对照用于解释与资源取舍，不改变本轮主比较。

以上均为单次验证运行的判断，尚不是多种子稳定结论；继续不读取测试集。

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
