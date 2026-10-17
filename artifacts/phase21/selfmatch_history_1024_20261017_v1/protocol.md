# SASRec+BGE：文本自匹配与历史惩罚的独立对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / experiment design
- Origin Date: 2026-10-17
- Verification Status: protocol frozen before new generation; results pending
- Version Label: selfmatch_history_1024_20261017_v1

用户授权先以 SASRec、BGE 为专家探索文本自匹配与历史惩罚，当前五专家4096长实验保持原样继续。当前不是论文全套实验，不训练GRAM、专家或融合器，不读取test，不提交git。

## 输入与固定条件

每域选择既有wide4096中索引[1024,2048)的连续1024名用户：与旧专家筛选的前1024不重叠，但仍属于反复使用的开发验证集，不称为独立确认集。选择不读取命中或效果。两域、所有组用同一域内用户名单及目标；保留原来的五个hash评估块，另分前后512报告。

复用相同训练完成的GRAM、SASRec和BGE缓存，α=0.3，双专家β=(0.5,0.5)，专家分数原始minmax尺度、softmax系数20，beam=50，length_penalty=1。历史商品在专家、树生成、最终度量中均可输出；没有硬过滤或生成后删除。

原生GRAM、未干预的SASRec+BGE、SASRec单专家和BGE单专家结果从仍在运行的4096任务中提取已完整写出的指定用户记录，保存独立快照。记录原路径、索引、原始行hash及提取内容hash；不对仍追加的源预测文件冻结整文件hash。每个新分片重新生成三个用户，精确核对原生和未干预双专家预测，再正式执行。

## 四组主实验

| 组 | 文本自匹配 | 历史惩罚 |
|---|---|---|
| pair_raw | 保留 | λ=0 |
| pair_loo | 去除 | λ=0 |
| pair_raw_penalty | 保留 | λ=0.05 |
| pair_loo_penalty | 去除 | λ=0.05 |

去自匹配：对历史候选i，计算文本历史向量时移除i的全部出现，其他可见历史、时间权重exp(-0.1×距今位置)不变；非历史候选分数逐位复制；保留原完整目录min/max归一化基准，裁到[0,1]，空残余历史支持为0。按残余向量直接计算，不额外裁到“旧历史分数”以下；在精确算术中正自贡献本身会提高自身余弦，旧上界通常冗余。与既有实现和LIME派生评分关系如实记录，不声称原创。

历史惩罚作用于完整历史，不仅GRAM可见的最近20：

`s'(u,i)=0.5*s_SASRec(u,i)+0.5*s_BGE(u,i)-0.05*I[i∈full_history(u)]`

随后softmax(20×s')、聚合树条件分支概率、按α混合GRAM。λ预先固定，不按此轮结果调参；系数20下专家侧历史/非历史概率比乘exp(-1)，不等于最终生成概率比也乘同一数值。不是旧学习gate预测的λ，也不更新模型参数。

## 无专家内容的历史抑制对照

另跑uniform与uniform_penalty：用全目录恒定分数0代替专家分数，分别不减/减相同0.05，随后使用完全相同的softmax、树和α引导。两者的差表示在相同外部引导接口下通用历史抑制的收益。必须同时报告原生GRAM与uniform，因为恒定叶子先验本身也可能改变GRAM；不能把uniform_penalty直接标成“原生GRAM只加惩罚”，也不能把两组差异当成可直接相减的因果贡献。

这保留了无专家/有专家、有惩罚/无惩罚的匹配接口控制。若以后要声称最终模型的专家独特贡献，仍需对最终专家、最终融合方式重新做匹配消融。

## 输出、分析与完成判据

主看每域1024的NDCG@10，同时报告HR/NDCG@5/10/20/50、前后512、五评估块、完整目录纯专家排序、生成前10/50历史占比、专家历史概率质量。报告去自匹配在有/无惩罚下的条件增量、惩罚在两种文本下的条件增量及二者交互作用；报告uniform控制的变化及与SASRec单专家的探索性对照。

逐用户配对bootstrap2000次，给绝对N10差与描述性95%区间；所有预定比较保留，区间不作多重比较校正，不含训练种子和探索选择不确定性。不因某域/某512先改善就提前停或选赢家。

保存协议、配置、源码快照、输入/输出manifest、完整预测、排名和运行检查。每域2个互斥512用户分片，共4任务，最多2张卡同时运行；优先避开原4096任务仍在使用的GPU。沿用用户16GiB总租约、12GiB预算+4GiB侧车、18GiB空闲准入、模型进程11GiB上限。每任务硬超时12小时、等待资源24小时。仅显存准入退让可重排；科学失败不自动重试。

命令：`/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -u -m experiment.phase21.selfmatch_history_ablation prepare`，随后以同一解释器在tmux运行`-m experiment.phase21.selfmatch_history_ablation supervise`。状态路径：`artifacts/phase21/selfmatch_history_1024_20261017_v1/status.json`。

超过10分钟的生成后台执行，完成启动核验后交回用户，不连续轮询等待最终结果。全部2048名用户、每用户5个新生成组及全部复用组齐全、smoke/runtime通过、GRAM参数摘要不变且租约释放后才标COMPLETED；完成后不自动启动下一轮。
