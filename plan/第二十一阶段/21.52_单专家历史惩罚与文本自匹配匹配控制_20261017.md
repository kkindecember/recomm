# 补齐单专家的历史惩罚与文本自匹配控制

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-17
- Verification Status: frozen before supplemental generation
- Version Label: single_switch_controls_1024_20261017_v1

用户明确要求隔离历史惩罚和文本自匹配。原21.50包含未修改的单专家，但缺少对应开关设置的单专家；本轮独立补齐，不改动正在运行的21.49/21.50协议、源码或预测。

同两域各1024用户、相同GRAM/SASRec/BGE、α=.3、单专家权重1、beam50、softmax系数20；无历史过滤、不删除任何输出、不重训、不读test。历史惩罚固定λ=.05，完整历史内商品的专家分数减.05，再转树条件概率。文本去自匹配完全复用21.50已冻结的BGE残余向量评分。

新增生成4组：SASRec+惩罚，BGE去自匹配，BGE原评分+惩罚，BGE去自匹配+惩罚。未干预SASRec/BGE以及原生GRAM复用21.50冻结参照。SASRec无文本自匹配开关，不为等价组重复生成。

最终每一行都比较双专家与两个匹配单专家：

| 文本/惩罚 | 双专家 | 行为单专家 | 内容单专家 |
|---|---|---|---|
| 原评分/无惩罚 | pair_raw | sas_only | bge_only |
| 去自匹配/无惩罚 | pair_loo | sas_only | bge_loo |
| 原评分/有惩罚 | pair_raw_penalty | sas_penalty | bge_raw_penalty |
| 去自匹配/有惩罚 | pair_loo_penalty | sas_penalty | bge_loo_penalty |

完整预测和专家全目录排名分开保存。各域1024为主，另列两批512和五hash块，配对bootstrap2000次、描述性区间不校正多重比较。比较的是固定α/β默认设置，不声称所有方法各自已最优校准。仅改善惩罚或去自匹配，不等于证明第二位专家必要。

两域各2分片，每分片512用户，每人4次新增生成，共8192次；每个分片先重新生成3名用户的原生、SASRec单专家、BGE单专家，核对缓存、α=0和目标占位不变，结束检查GRAM摘要。每任务沿用16GiB租约、12GiB工作预算+4GiB侧车、18GiB空闲准入、模型11GiB上限。最多2个新任务并行，且原21.49、21.50及本补充的活动GPU任务总数不超过4，避开原任务在用GPU。

硬超时12小时，资源等待24小时；仅资源准入退让可重排，科学失败不自动重试。后台记录PID、阶段、计数、ETA和租约释放。完成后若21.50已齐全，自动汇总上述匹配比较；否则明确标匹配报告等待原实验，不缩小用户范围，不把部分结果当成全部完成。困难候选文本训练可独立在CPU进行。
