# 第十九阶段：GRAM 实验进展与结论总报告

## Material Passport

- 更新日期：2026-09-10，Beauty后台启动及Toys中期状态快照。
- Origin Skill: academic-research-suite / experiment-agent。
- 状态：`RUNNING / INTERIM`，尚无 RaSeRec→GRAM 最终提点结果。
- 当前目标：相对原GRAM提高实际推荐指标；单独模型与PCRF组合都以原GRAM为主对照。允许迁移已有方法，暂不评估创新性。
- 文档约定：本阶段重要结论持续更新本报告；整体安排维护在[唯一总计划](../../plan/第十九阶段/GRAM_第十九阶段_提点实验总计划.md)，不按每个小检查新建报告。

## 1. 当前可以确认什么

1. **检索其他用户的历史案例能够提供推荐线索，但简单案例检索尚未优于较强的转移统计对照。** 因此，它值得作为一个可训练模块尝试，不能仅凭检索覆盖宣称已经提点。
2. **RaSeRec 核心模块和 GRAM 接口已经实现，基础检查通过。** 原作者神经层与预训练类保持一致；工程适配、案例隔离和 PCRF 复用已完成检查。Toys真实大模型GPU接口检查已经通过；Beauty在前置训练后自动执行同样检查。
3. **Toys已进入GRAM最后一轮，子集有相对原GRAM的提升，完整验证尚待完成。** 用户已明确主对照为原GRAM，并授权同步补充Beauty；不以超过原GRAM+PCRF作为准入条件。
4. **实验已在后台运行，状态文件由程序自行更新。** 本次仅因用户询问而读取状态，不持续监看，不等待长实验结束。

## 2. CPU 检索检查：线索存在，不能等同于提点

每域固定2000名验证用户。案例库只使用训练集“历史→后续商品”，排除查询用户全部案例，查询输入不含当前真实目标。按带历史位置权重的TF-IDF余弦相似度找案例，每名用户至多取20个不同的后续商品。未使用测试集，未针对本队列调参。

| 线索来源 | Toys找到正确商品人数 | Beauty找到正确商品人数 | Toys正确商品在原GRAM前50之外的人数 | Beauty对应人数 |
|---|---:|---:|---:|---:|
| 随机训练案例 | 9 | 4 | 8 | 1 |
| 仅最近商品的后续转移统计 | 184 | 173 | 25 | 12 |
| 历史加权的已有SASRec商品邻居 | 209 | 200 | 48 | 28 |
| 相似历史案例 | 248 | 213 | 46 | 27 |
| 全部历史商品的后续转移统计 | 249 | 215 | 39 | 23 |

最后一行是初次检查后追加的更强对照，使用原固定用户且没有搜索参数。相似案例与该对照的总覆盖几乎相同；找到原GRAM候选外正确商品的数量也没有超过已有商品邻居对照。因此，不能选择弱对照来宣称案例检索已经更强。

这些数字是“提供的20个线索商品是否含正确商品”，不是最终Hit@10或NDCG@10。每个查询的上限一致，但部分方法填不满：仅最近商品转移平均约11.10/12.07个，全部历史转移约19.56/19.73个，案例检索接近20个。这一点也限制了直接比较。

当前据此作出的决定是：尝试让可训练模块利用案例，观察最终推荐表现。按照用户当前目标，首轮不要求先证明案例上下文的独立贡献，也不安排一系列额外消融。

证据：[首轮统计](../../artifacts/phase19/memory_probe_v1/summary.json)、[全部历史转移对照](../../artifacts/phase19/memory_probe_v1/additional_all_history_transition_control.json)。用户级结果、固定规则与首轮脚本保存在同一目录。

## 3. 实现与检查结论

采用[RaSeRec作者仓库](https://github.com/HITsz-TMG/RaSeRec)提交 `44d2d2a4f52ddd97a5bd4153dab6ee0e89d5ceca`。保留作者两层64维序列编码器、`us_x`同目标对比预训练、案例检索和双通道注意力。核心神经层及DuoRec类从原代码逐字提取，保留MIT许可证。

必要适配包括：使用工程现有划分；精确余弦检索替代Faiss近似检索；排除同一用户案例；用验证NDCG@10选模；将用户表示、增强表示及20组案例投影成42个额外记忆token，接入原GRAM解码器。原GRAM的词汇ID、文本、FiD记忆和父模型结构保留。因此，这是有明确接口改造的迁移实验，不能写成原作者独立模型的逐项复现。

首版保留作者示例 `alpha=0.5、beta=1.0`：第二通道虽在代码中计算，但不参与最终加权混合；作者两个通道共享使用seq_tar FFN的行为也保留。它们是当前实现事实，不据此声称额外模块贡献。

| 检查 | 已确认结果 |
|---|---|
| 基础行为测试 | 7项通过，覆盖作者核心类一致性、同用户案例排除、目标与历史输入分离、单样本处理、冻结/联合梯度、权重重载及小模型生成接口 |
| 真实数据划分 | Toys训练109361条、验证19412人 |
| 训练频次口径 | 所有验证用户的目标训练频次与既有PCRF参考一致 |
| PCRF复用口径 | 固定64个原GRAM验证beam，重算PCRF排名与原记录完全一致 |
| 真实GRAM大模型GPU接口 | Toys已通过冻结/联合梯度与beam50检查；Beauty将在前置训练后自动检查 |

证据：[真实数据预检查](../../artifacts/phase18/raserec_gram/toys/preflight_v1/summary.json)、[行为测试](../../experiment/phase18/tests/test_raserec_gram.py)、[配置与源码快照](../../artifacts/phase19/raserec_gram_toys_v1/manifest.json)。

## 4. 当前训练进展与指标含义

Toys启动于 **2026-09-09 18:34:57（北京时间）**，GPU7，tmux会话 `s18_raserec_gram_toys_v1`，PID `3680679`。9月10日检查时已进入GRAM第11/11轮；随后执行两个选中检查点的完整验证。

序列预训练完成100轮，最佳完整验证NDCG@10为0.058434；reader训练于第41轮结束，最佳为0.061998。真实GRAM GPU接口检查已经通过。这些前置结果不作为最终生成模型的提点结论。

截至GRAM第10轮结束，同一固定2000人子集上：

| 方法（按各自验证NDCG@10选模） | NDCG@10 | 相对原GRAM |
|---|---:|---:|
| 原GRAM | 0.071596210 | — |
| RaSeRec→GRAM，epoch4 | 0.073617012 | +2.82% |
| RaSeRec→GRAM+PCRF，epoch8 | 0.074927959 | +4.65% |
| 原GRAM+PCRF，辅助对照 | 0.075440606 | +5.37% |

这是选模子集的中期结果，不能替代完整验证，也不是独立测试结果。用户确认两个候选都以原GRAM为主对照；组合相对原GRAM+PCRF的差值仅作辅助说明，不阻止Beauty复现。

证据：[epoch4](../../artifacts/phase19/raserec_gram_toys_v1/gram/epoch_04_trend.metrics.json)、[epoch8](../../artifacts/phase19/raserec_gram_toys_v1/gram/epoch_08_trend.metrics.json)、[GPU接口检查](../../artifacts/phase19/raserec_gram_toys_v1/smoke.json)。

最终比较的固定参考为：

| Toys完整验证参考 | Hit@10 | NDCG@10 |
|---|---:|---:|
| 原GRAM | 0.119411 | 0.076275 |
| 原GRAM+PCRF | 0.125335 | 0.078716 |
| RaSeRec→GRAM | 待完成 | 待完成 |
| RaSeRec→GRAM+PCRF | 待完成 | 待完成 |

PCRF使用已有item-head及固定 `lambda=1、beta=0.5、gamma=1、q1=5`，不直接相加两种方法的历史收益。最终报告更新时记录新增正确结果和丢失结果的净变化；单次运行改善仍不等同于稳定改善。

证据：[本次冻结快照](../../artifacts/phase19/review_20260909.json)、[原PCRF完整验证参考](../../artifacts/phase9/p9s_multiseed/Toys/seed2023/validation/summary.json)。

## 5. 预计还要多久

Toys在9月10日09:50的估计为剩余6–8小时，预计当日16–18点完成。依据本轮最近2000人验证约15–16分钟，每次19412人完整验证约2.5小时；最后一轮训练后还有两次完整验证。硬截止仍为9月11日18:34:57。

Beauty暂按整体24–32小时排期，依据Toys实测及Beauty更大的数据规模，预计9月11日完成；尚不是Beauty实测ETA。训练硬上限48小时，实际起止和deadline见Beauty status。共享GPU负载可能使耗时变化。

## 6. Beauty同配置补充与后台状态

2026-09-10用户授权补充Beauty，并明确GPU1按实际可用显存运行，无需18.5 GiB保守余量或30 GiB占位。10:28:01在GPU1完成显存准入并启动工作进程，准入时空闲15563 MiB；工作PID `3121432`。后台会话为 `s19_raserec_gram_beauty_v1`，后续阶段和存活情况以status为准。 启动核查确认已完成首轮预训练及验证、进入后续轮次；该阶段nvidia-smi实测占用3148 MiB，未发生OOM。监督进程心跳新鲜，实际执行配置与训练manifest一致。见[startup_verification.json](../../artifacts/phase19/raserec_gram_beauty_v1/startup_verification.json)。

训练配置沿用Toys的所有模型和科学超参数，换为Beauty数据、原epoch25父模型及Beauty PCRF；其固定q1为6。完整训练131413条，验证22363人，seed2023，GRAM固定1冻结+10联合轮次，microbatch4，有效batch128，beam50。资源设置为最低空闲12000 MiB、PyTorch分配上限整卡28%，约13.4 GiB；CUDA上下文等额外开销不计入该PyTorch上限。

检查结果：父模型哈希匹配，所有验证用户的训练频次与历史参考一致；64个原GRAM验证beam的PCRF重算排名完全一致；后台状态测试5项通过。Beauty真实GRAM GPU检查将在预训练、案例库和reader之后执行，尚未宣称通过。

首个18500 MiB准入被共享显存变化拒绝，之后的16000 MiB资源等待按用户要求取消，两者均未启动训练、未创建模型或优化器；原始记录保留在输出目录`attempts/`。实际训练使用本次12000 MiB准入配置。

观察入口：

- [Beauty状态](../../artifacts/phase19/beauty_status.json)：每30秒心跳，记录阶段、轮次、处理数、PID、GPU、日志、退出码与失败原因。
- [双域状态总览](../../artifacts/phase19/status.json)：由Beauty监督进程在运行期间更新，同时显示Toys原始状态；原Toys状态文件保持独立。
- [Toys状态](../../artifacts/phase19/toys_status.json)。
- [Beauty训练日志](../../artifacts/phase19/raserec_gram_beauty_v1/run.log)、[预检查](../../artifacts/phase19/raserec_gram_beauty_preflight_v1/summary.json)、[实际执行配置](../../artifacts/phase19/raserec_gram_beauty_v1/execution_config.json)。

## 7. 后续重要结果如何归档

继续使用这一份总报告，按进展更新以下结论：检索分支训练完成及选模结果、案例注意力训练效果、真实GRAM接口检查、完整GRAM验证，以及PCRF组合的净变化。如果发生中断或未跑完整日程，明确写出实际完成范围和原因，不补写缺失结果。

实时观察只需要查看[artifacts/phase19/status.json](../../artifacts/phase19/status.json)。该总览由Beauty后台监督进程维护，带更新时间；监督进程结束后查看各域独立状态文件。本报告是最近一次人工核查记录，不假装自动实时更新。超过10分钟的实验保持后台执行，助手不会为等待结果持续轮询。
