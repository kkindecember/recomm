# Toys完整候选列表内部蒸馏实验计划

**2026-09-16后续已收敛：** 原三臂后补Q0诊断及两臂32步L0学习率筛查，L0未通过事先约定的继续门槛，当前停止内部KL扩展，不再追加完整L1/Beauty。见[L0终态报告](../../report/第二十阶段/Stage20_Toys_L0短程学习率筛查_20260916.md)及[下一项排序对照计划](L0止损后排序路线与最小对照计划_20260916.md)。下方原协议、恢复记录和完成证据保持原样。

**2026-09-15完成更新：** recovery_v3于20:19:27完成9/9任务及三臂全量验证；真教师未建立匹配迁移收益。下方原协议不追改，终态解释见[最终分析](../../report/第二十阶段/Stage20_Toys完整列表内部蒸馏最终分析_20260915.md)，后续顺序见[学习诊断与单因素计划](Toys蒸馏终态后的学习诊断与单因素实验计划_20260915.md)。第10节末的“完整训练与验证尚未完成”仅指当时启动核验快照。

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent；Origin Mode：run / validate。
- 2026-09-13用户“继续吧”授权执行；先CPU正确性检查，再真实GPU工程profile，超过10分钟的阶段全部由后台队列执行。
- 初版预登记配置：`experiment/phase20/gacr_list_distill_toys_v1.json`；该计划与科学配置在新验证指标产生前固定。2026-09-14实际执行使用资源修订版`gacr_list_distill_toys_gpu1_v2.json`，见第7节。
- 动态报告：[Toys完整列表内部蒸馏](../../report/第二十阶段/Stage20_Toys完整列表内部蒸馏进展与结果.md)。

## 1. 要解决的问题

B1+PCRF在Toys/Beauty的三个残差初始化均提升N10/H10，但仍是固定C1候选上的排序方法，H50及低频覆盖未修复。本轮优先检验：完整商品列表教师能否让生成器自身学到排序能力，并在真实重新生成的候选上保留或超过当前强基线。

首先仅做Toys，降低首次完整列表骨干训练的成本。采用冻结B1 seed2023作为教师，不挑后来的最好初始化。B1验证N10=0.08235733；原PCRF、C1、C1+PCRF均保留为对照。并不预设这次迁移会成功，也不以方法名称或新增损失证明创新。

## 2. 数据、教师与更新范围

复用`gacr_fusion_train_v1/Toys`的合法训练前缀、C1候选50、CF分数和全部union特征。fit共21,839个前缀/3,882个用户；其中12,399个真值在候选50内，9,440个在外。源缓存另256个calibration前缀不参加梯度训练，也不以其结果选超参数。缓存构建可以计算其教师和父分数以维持逐行对应。

教师先按原实现，在每个前缀完整union形状上运行B1残差网络，再保留原C1对应的前50项分数；根据g排序后的Top10计算PCRF低频可靠性，使用原固定λ=1、β=0.5、γ=1、q1=5组合。最终教师概率为完整50项PCRF分数标准化后以温度1 softmax，并映射回原C1商品ID顺序。

学生从完整C1 backbone初始化，不装载catalog head参与训练。只更新解码器中未与shared绑定的参数；历史编码器、共享词嵌入与LM head冻结。模型保持eval以禁用dropout，但解码器开启梯度。完整参数名单、数量与冻结参数hash保存至每个任务的`parameter_contract.json`，结束时检查冻结参数与梯度。

## 3. 完整列表损失及三臂控制

商品分数为完整标签token log-probability求和除以有效标签数。标签不含BOS、含EOS，padding为−100。所有50项统一标准化后softmax；不使用局部兄弟节点的概率差代替完整商品竞争。

记学生、冻结C1父模型、冻结B1+PCRF教师分布为pθ、p0、q，T=1。父p0用同一teacher-forcing接口重算并缓存，避免将native beam与teacher forcing的细小数值差异写进保持损失。

| 臂 | 完整目标 |
|---|---|
| CE_anchor | CE + KL(p0 || pθ) |
| true_teacher | CE + KL(p0 || pθ) + 0.1 × (21839/12399) × I(目标在候选50内) × KL(q || pθ) |
| shuffled_teacher | 同上，但q按每个sample_key固定打乱商品对应关系 |

CE对所有fit前缀使用真实商品完整序列，候选外目标不强制塞入50项列表；父保持对所有fit前缀生效。教师损失按全fit/covered数量归一，使其批均值对应covered样本的均值。打乱保持每行概率集合及熵，用于检验教师的商品对应关系；该对照不能单独隔离CF贡献。

为限制显存，历史编码一次且冻结，候选解码每块4条。先无梯度得到50个分数，求完整50项损失对分数的导数，再逐块重算并传递该导数。该操作保持完整列表目标，不能将每块4项单独softmax。CPU测试以直接完整50项图验证损失与参数梯度一致，并覆盖候选外、CE臂和末批归一。

## 4. 固定训练预算

三臂完全相同C1初始权重、fit顺序、有效batch、优化器及保持项：

- 一轮完整fit，micro-history batch=1、effective batch=128；末批79个样本按79归一；每臂171步，三臂共513步。
- AdamW，lr=1e-6、weight_decay=0.01、eps=1e-8；9步warmup后线性衰减至0，clip=1。保存固定最后一步，无验证选checkpoint。
- 模型seed2023，前缀顺序seed20260913，教师打乱seed20260913；FP32模型/FP64列表概率与损失，禁用TF32，CPU4线程。
- 运行曲线按优化步保存；每50步保存进度checkpoint，最后保存解码器参数及优化器、scheduler、RNG状态和源C1/config hash。

冻结历史编码器和很小的学习率用于限制继续训练扰动，是本轮实现选择。无法据此断言旧PCPS失效只由某一个因素造成。训练仍只覆盖既有fit用户，未彻底解决旧实验的训练分布问题。

## 5. 工程profile、后台与时限

GPU3 UUID `GPU-7952ad6e-2db7-fb2e-581b-34be56287640`，启动前观察约34GB空闲；准入30,000 MiB、allocator上限60%，不终止其他进程。选8个训练历史长度分位前缀（含最长）与8个验证历史长度分位用户，测父列表评分、三臂前后向和真实beam吞吐。验证C1候选ID完全一致，native分数容差1e-6；记录teacher-forced与native最大差异，接受上限0.003但明确非精确同值。执行一次丢弃的工程更新验证参数更新/保存/恢复，正式训练重新加载C1起点。

profile上限30分钟、父缓存8小时、每臂训练12小时、每臂完整验证12小时、汇总30分钟；每次资源等待上限1小时。这些是防失控上限，实际预计工期必须由profile给出；profile估计超出任一阶段上限即停止，并保留测量，不静默修改预算。正式阶段设置闹钟及队列外部超时；异常停止后续，无自动重试、无自动Beauty扩域或超参数网格。

后台顺序：profile → 父分数/教师缓存 → CE、真教师、打乱教师三臂训练 → 三臂完整验证 → 匹配统计，共9项任务。只有工程正确性/资源约束决定是否停止，不依中途准确率筛掉任何指定臂。

`status.json`每30秒更新，含当前任务、子进程、stage进度/ETA、profile外推整轮剩余时间、正式/工程优化步和失败原因。报告同步更新；停止命令只对核对了PID、命令及启动ticks的本队列生效。

```bash
bash experiment/phase20/run_gacr_list_distill.sh start
cat artifacts/phase20/gacr_list_distill_toys_v1/status.json
bash experiment/phase20/run_gacr_list_distill.sh stop
```

## 6. 完整评估与后续判断

各指定臂使用固定最终checkpoint，为19,412名Toys验证用户重新生成合法beam50；报告raw及接原固定PCRF的结果，不用缓存列表上的学生分数冒充真实生成指标。与原GRAM、原PCRF、冻结C1、C1+PCRF及冻结B1+PCRF比较；三臂数据和设置相同，另做真教师−CE、真教师−打乱的匹配配对分析。

八指标：NDCG/Hit@5/10/20/50；五分组：全量、目标频次≤5、>5、C1训练用户外/内。N10/H10用3000次用户配对bootstrap，seed20260912；记录Hit@5/10/20/50 gained/lost及候选集合改变用户数、平均交集。单次训练/重复验证及多重比较未校正限制必须保留。

先问真教师是否超过两个匹配控制、是否保住父C1，然后问raw是否逼近当前B1+PCRF，接PCRF是否超过当前强基线；同时核查H50和低频覆盖及耗时。只改变候选不等于改进。若主效果没有超过控制，不直接开Beauty大规模矩阵；先依据CE退化、教师相对打乱、候选覆盖损失定位问题。若真教师有明确同向增量，再决定复制初始化和Beauty。

旧P9-2B只在NLL门槛失败，不能称其全量排序已被证伪；旧PCPS已有部分完整序列损失，本轮差异在完整实际50项列表分布、跨父前缀竞争、父保持与参数范围。阶段18与20父模型不同，不能混用绝对指标。既有解释见[双域路线报告第8节](../../report/第二十阶段/Stage20_GACR_PCRF双域结果与提点路线_20260912.md)。


## 7. 2026-09-14执行时资源修订

实际启动时GPU3已只有23–24GB空闲，初版队列仅进入WAITING_FOR_RESOURCE。已按本轮授权主动停止该等待队列，保留`gacr_list_distill_toys_v1/status.json`的STOPPED记录；未执行profile或任何优化。GPU1当时空闲约41GB，因此更换UUID为`GPU-2afcc1df-7df8-1f3f-8970-a252a06c6160`，实际配置为`experiment/phase20/gacr_list_distill_toys_gpu1_v2.json`，输出为`artifacts/phase20/gacr_list_distill_toys_gpu1_v2/`。

数据、教师、模型代码、三臂损失、步数、精度及所有时限保持不变；修订版新增启动脚本和旧配置hash以保留审计链。原配置及代码不覆写。新后台会话`s20_gacr_list_distill_toys_gpu1_v2`，观察与管理：

```bash
cat artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json
bash experiment/phase20/run_gacr_list_distill_gpu1.sh status
bash experiment/phase20/run_gacr_list_distill_gpu1.sh stop
```


## 8. 首次真实GPU检查结果（2026-09-14 23:26）

GPU1 profile在85.6秒内通过：8名验证用户的原C1候选ID完全一致，native分数通过1e-6容差；8个训练前缀teacher-forced与native最大绝对分数差为9.18e-6。三臂梯度有限、重放一致，冻结参数hash不变；仅一次丢弃的工程优化，正式优化0步。峰值已分配6,877 MiB，allocator保留19,096 MiB。

当前共享GPU速度外推父缓存1.74小时，CE/真教师/打乱教师训练分别4.99/4.46/4.49小时，每臂验证2.47小时，总计约23.1小时另加汇总少量时间；全部低于预登记单阶段上限。该值含样本初始化/共享负载波动，非完成保证。队列已进入cache阶段，会自动继续已登记任务。

证据：[profile](../../artifacts/phase20/gacr_list_distill_toys_gpu1_v2/profile/profile.json)、[启动后独立核对](../../artifacts/phase20/gacr_list_distill_toys_gpu1_v2/launch_verification.json)。77个配置源文件hash、32个输入hash、64份实际加载源码快照核对通过；不代表正式训练效果已验证。

## 9. 2026-09-15用户要求分卡运行：调度修订

用户明确要求“我想分卡运行”。新增独立分卡调度器，继续使用当前输出目录和GPU1正在计算的共享缓存。原模型、损失、训练步数、数据顺序、冻结范围、原配置和已登记源码保持原hash；只修订任务并发、GPU绑定和观察方式。

交接先核对原队列与缓存的PID、启动ticks、UID、命令和父子关系，确认三臂均未开始。缓存子进程在独立session内，交接时短暂停住它以排除完成竞态，复核原队列仍处于缓存阶段后，仅退役旧调度进程并接管原队列锁，随即恢复同一个缓存PID；不重算已完成前缀。新调度器接管30秒心跳及原缓存时限。缓存进程退出且状态、manifest和缓存文件hash均通过后，才允许三臂训练。原状态快照、进程身份及缓存复用依据保留在输出目录的`parallel_v2/`下。

首次工程交接尝试未通过旧调度器暂停检查，保护逻辑已恢复原队列，未启动训练、未重算缓存；失败receipt、traceback和当时调度源码保留在`parallel/`。修订版通过完整临时父子进程及队列锁交接测试后再次执行。该工程交接重试不构成新增模型训练或结果筛选。

三臂优先分配CE_anchor→GPU4、true_teacher→GPU1、shuffled_teacher→GPU6。每臂占用不同的RTX A6000；首次分配时优先卡不足原准入30,000 MiB，则从1/4/6/0/3/5/2/7中选择另一张符合准入且未绑定的同型号卡。实际UUID、启动时显存、CUDA_VISIBLE_DEVICES、工作配置hash和PID记入各任务launch记录。每臂训练后在绑定卡上验证，不等待其他臂训练；全部验证成功后仅执行一次匹配汇总。

每卡allocator上限60%、每阶段原时限和资源等待上限1小时保持不变。新任务启动前复核资源；同一输出拒绝覆盖或重试。技术失败、超时或用户停止时仅停止核对身份后的本轮子进程并保留产物。分卡不能保证共享服务器吞吐不变，也不把完成更快作为效果筛选条件。

七项新增调度测试通过：缓存屏障及三臂独立依赖/汇总条件；GPU互斥与显存准入/回退/绑定保持；科学配置不变；拒绝晚于训练起点的交接；真实临时父子进程的暂停/退役隔离及PID复用保护；交接前失败保留旧队列；完整交接流程中原锁转移、同一缓存PID恢复和心跳继续推进。原数值检查与GPU profile继续复用。

在三卡及时可用且各卡吞吐接近原profile的假设下，缓存后关键路径约为最长单臂训练4.99小时＋验证2.47小时＋汇总，约7.5小时；与原三臂串行约23.1小时的工期定义不同。资源等待和负载变化会延长实际完成时间，实时ETA不作为完成承诺。

```bash
bash experiment/phase20/run_gacr_list_distill_parallel.sh start
bash experiment/phase20/run_gacr_list_distill_parallel.sh status
bash experiment/phase20/run_gacr_list_distill_parallel.sh stop
```

原统一[status](../../artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json)及[动态报告](../../report/第二十阶段/Stage20_Toys完整列表内部蒸馏进展与结果.md)继续使用；分卡状态另镜像于`parallel_v2/status.json`。旧GPU1脚本不用于管理接管后的队列。该修订仅授权当前Toys三臂分卡，不新增Beauty、hinge控制或超参数实验。

## 10. 2026-09-15显存准入失败后的恢复

旧分卡队列于00:57:41终止。GPU4选卡时只有30,001 MiB空闲，真教师进程初始化CUDA后未达到30,000 MiB门槛，裸assert失败；调度器随即停止CE臂。共享缓存于00:53:44完成，CE仅记录1步、无checkpoint，真教师0步、打乱教师未启动，全部验证和汇总未执行。该次属于工程中断，不能判断蒸馏效果；旧报告的7.04小时为失效的残留估算。

用户了解原因后要求处理，恢复使用独立配置`experiment/phase20/gacr_list_distill_toys_recovery_v3.json`和输出`artifacts/phase20/gacr_list_distill_toys_recovery_v3/`。旧status、日志、源文件及配置保持冻结；失败时的报告另存新目录`previous_failure_report.md`。新旧目录之间通过`recovery_receipt.json`与旧目录新增的`recovery_pointer.json`关联。

恢复复用已经通过的profile和22,095行共享缓存，复制后再次校验SHA256及cohort，不重算模型缓存。三臂从原C1起点重新训练，各171步，旧CE的一步单独记为丢弃。数据、训练顺序、损失、参数冻结范围、初始化、学习率、精度、每阶段计算时限和固定末步checkpoint规则全部不变；直接调用原`gacr_list_distill.Work`的训练、验证与汇总方法。新增代码只处理启动、资源等待、进程管理和状态展示。每臂训练及验证使用同一工作配置文件，保持checkpoint配置hash兼容。

选卡仍为不同A6000、优先GPU4/1/6并保留同型号回退。调度器每30秒观察，连续两次空闲显存至少31,024 MiB才启动，即原门槛外加1,024 MiB启动余量。子进程初始化CUDA后再次检查原30,000 MiB门槛；不足时在同一进程等待，每10秒记录实际空闲值、门槛与剩余等待时间，不创建optimizer或开始训练。排队等待与进程内等待合计最多原定1小时；通过准入后才开始原计算时限。allocator上限仍60%，资源准入没有实际预留整卡显存，共享负载仍可能变化。

真实计算失败、超时和用户停止仍按原规则结束队列；资源暂时不足属于等待，未增加失败后的自动重试。清理结束后`active_children`清空，失败/停止时ETA置null并公开具体原因。报告入口不变，实时status改为新恢复目录；旧目录状态保持FAILED，不再冒充实时入口。

启动前21项检查通过：7项原数值检查、7项原分卡检查、7项新增回归检查。新增检查覆盖初始化后低显存等待并恢复、累计等待超时、拒绝仅1 MiB余量及连续检查中断、科学配置/原hash不变、真实临时子进程的三卡依赖与一次汇总、计算失败后清理进程并清除ETA、拒绝覆盖旧尝试。另通过Python编译、shell语法和diff空白检查。检查不产生模型训练或新验证指标。

```bash
bash experiment/phase20/run_gacr_list_distill_recovery.sh start
bash experiment/phase20/run_gacr_list_distill_recovery.sh status
bash experiment/phase20/run_gacr_list_distill_recovery.sh stop
```

从三臂实际启动起，原profile估计最长训练加验证约7.5小时；不含后续等资源和共享负载变化，以新status进度为准。仍只完成既定Toys三臂，不扩展Beauty或额外超参数。

**实际启动核验：** 后台控制器于10:38:45启动，三臂实际绑定CE→GPU1、真教师→GPU6、打乱教师→GPU5。10:45:48独立核验时分别完成2/3/2步，均已记录非零学习率更新，loss与梯度有限，冻结参数合同与原CE起点一致；每臂68份实际加载源码及其快照、原77项冻结源码hash核对通过。旧FAILED状态字节hash保持不变。见[启动核验](../../artifacts/phase20/gacr_list_distill_toys_recovery_v3/launch_verification.json)。这只确认早期训练在推进，完整训练与验证尚未完成。
