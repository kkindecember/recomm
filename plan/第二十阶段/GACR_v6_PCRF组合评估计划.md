# 第20阶段 GACR-v6 与 PCRF 组合评估计划

日期：2026-09-11。授权：用户要求停止S-DPO、评估已保存round1_sft，并补充GACR-v6与PCRF组合评估及相应计划/报告。本计划在组合推理产生结果前固定，不根据结果扫描融合参数。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Version Label: stage20_gacr_v6_pcrf_toys_v1
- 当前阶段：冻结模型评估；无新增训练。

## 1. 问题与范围

在Toys原官方19,412人validation上，先用冻结C1+GACR-v6选出50个商品，再执行冻结PCRF，能否超过原GRAM+PCRF的NDCG@10及Hit@10？同时区分C1继续训练、GACR候选/排序及PCRF附加重排的差异。

只执行Toys，残差训练seed2023；不自动扩展Beauty、不重新训练C1/residual/item-head。此前两域1024人缓存对比用于提出问题，不当作此次全量组合结果。

## 2. 冻结输入

- C1：`artifacts/phase4/gcdh_p0/Toys/C1/model.pt`，SHA256 `1307ab9d3aa5e56af97fad7276d63cb276260efd3d314b199e350a611c798af6`。
- GACR-v6：`artifacts/phase6/gacr_v6/Toys/residual_seed2023.pt`，SHA256 `c05f812ee4204602b58119e0411f08e4aa06c99d8ab277e2861ad48d2d9ce68c`。
- PCRF：`artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt`，SHA256 `18a30dd5782632286f93c4693892ec347a0514baa197376867333cc68b656b4d`。
- 原GRAM：阶段2 epoch30的未修改父权重；复用SPRec已完成的FP32、TF32关闭全量参考，比较前核对prepared数据、父权重、解码和PCRF协议。
- 数据：`artifacts/phase18/diff_gram/prepared_v2_categories/{train,validation,catalog}`。只读取prepared train/validation及商品映射；不加载原始user_sequence文件，不构造test。
- 完整路径与SHA、执行参数：[机器配置](../../experiment/phase20/gacr_v6_pcrf_toys_v1.json)。

C1经过“词法CE+catalog Balanced Softmax”继续训练，不等同于未经修改的原GRAM。C1目录头按原词法文件的插入顺序排列，与当前排序后的整数商品ID不同；按原商品ID显式映射并检查集合、词法编码和权重形状。

## 3. 候选和组合公式

复用原有 `experiment/phase4/gacr_s0.py::build_candidate_record` 和 `BoundedResidualRanker`，保留全部六维特征、seen-history masking、原并集顺序及稳定tie-break。生成调用只额外捕获sequence score，不更换候选构造实现。

1. C1原生词法beam50生成候选；目录头另取50个商品，去掉已见历史商品后构造去重并集。
2. GACR分数为 `g_i = 1 / C1生成rank_i + residual_i`，仅目录头出现的商品base为0；residual bound=0.2，scale=1.0。按原稳定规则选GACR top50。
3. 在这50个商品内套用原PCRF公式，但把GACR分数作为base signal：

   `score_i = z(g_i) + (1 - tail_mass) * z(z(CF_i) - 0.5 * z(log(1 + frequency_i)))`

   所有z均在本用户50个候选上计算，使用原实现的标准差下限。`tail_mass`是GACR重排前top10中训练频次≤5的比例。lambda=1、beta=0.5、gamma=1、q1=5，均固定。

PCRF是候选集合内的排列，组合Hit@50必须等于GACR Hit@50。组合的base已从生成log score改为GACR分数，这是明确的组合适配，不能宣称原PCRF在新分数分布上的效果有保证。

不使用validation目标构造候选、特征、分数或选择融合规则。目标仅用于计算rank和最终指标。

## 4. 同用户比较的六条结果

| 名称 | 定义 |
|---|---|
| original_gram | 未修改父模型，同精度参考 |
| original_pcrf | 父模型候选+冻结PCRF，主强参照 |
| c1 | C1生成beam50 |
| c1_pcrf | C1生成beam50+原生成score接口的冻结PCRF |
| gacr_v6 | C1生成/目录候选并集+冻结GACR，取top50 |
| gacr_v6_pcrf | 上述GACR top50+本计划固定PCRF组合，主实验 |

主比较：组合相对original_pcrf的NDCG@10绝对变化；同时报告Hit@10。另列组合对原GRAM、C1+PCRF、GACR的变化，不把已有PCRF收益归给GACR。

所有模型报告Hit/NDCG@5/10/20/50。完整validation外，另列C1训练用户内/外、目标频次≤5/>5分组。C1训练样本是train前缀，同一用户的validation目标仍未训练；用户内外分组用于显示泛化差异，不能把全量结果冒充用户完全隔离的历史队列。

3000次配对用户bootstrap、seed20260910；区间仅描述validation差异，不作多重搜索后的独立确认，不补跑到显著。

## 5. 工程检查、资源和执行

工程检查在独立目录进行：

- 验证所有冻结输入SHA，严格加载C1和residual，按原目录顺序核对商品映射。
- 对至少4名用户（含最长历史）比较原GACR构造与额外捕获score的路径：候选并集、base、features、target rank一致；特征容差1e-6。
- PCRF在16名父模型缓存用户上的rank必须精确一致。
- 参数全冻结，零optimizer steps；记录峰值显存；运行前后输入SHA不变。
- 正式评估保存完整候选、原分数、并集六维特征和逐用户rank，并检查组合候选覆盖不变。

原先拟排在GPU0的SFT之后。2026-09-11 14:37资源修订：SFT近期仅约0.64用户/秒，仍需约8小时；GACR的GPU5工程检查已经通过，故正式任务改为GPU5并行，避免纯资源排队。该修订发生在组合正式结果产生前，仅改变设备和排队方式；C1、residual、PCRF、用户、精度、解码及融合公式不变。

工程检查耗时36.56秒，4名用户包含长度20的最长历史，候选/特征与原实现一致；父参考PCRF排名16/16精确一致。峰值allocated6901.59MiB、reserved12372MiB，据此将正式准入提高为至少16GiB空闲，allocator上限仍40%。不使用holder或旧30GiB占位流程。

smoke硬预算30分钟；正式Toys硬预算24小时；等待资源最多18小时。正式失败不自动重跑、不延长预算。SFT结果不是GACR科学依赖，配置`wait_for_sft=false`，两者独立执行。

预计主要成本是一遍C1 beam50及额外目录特征编码。共享GPU吞吐波动大；先记录当前阶段实际速度，不将父模型约2–3小时的历史耗时当作完成保证。

执行入口：

```bash
python -m experiment.phase20.evaluate_gacr_pcrf --config experiment/phase20/gacr_v6_pcrf_toys_v1.json --mode smoke
python -m experiment.phase20.followup_queue
```

正式产物：`artifacts/phase20/gacr_v6_pcrf_toys_v1/`；独立smoke：`artifacts/phase20/gacr_v6_pcrf_toys_smoke_v1/`；排队状态：`artifacts/phase20/followup_queue/status.json`。

**2026-09-11工程执行修订：** 上述首次正式尝试在前缀阶段约0.12用户/秒，按此速度不能在24小时内完成。已显式停止并保留该目录，未产生最终组合指标。第二次尝试将原六维特征的逐候选算子批量计算，并一次性将完整分数向量拷到CPU排序，减少CUDA同步；模型、候选来源、特征数学定义、残差、融合公式、用户和预算均不变。独立检查对前16名用户（含最长历史）逐项比较原实现与批量实现的候选、base、特征和最终GACR top50：base须相同、特征容差1e-6、最终商品顺序须完全相同，且父参考PCRF的16人rank仍须精确一致。通过后才启动新目录 `artifacts/phase20/gacr_v6_pcrf_toys_fast_v2/`；检查目录 `artifacts/phase20/gacr_v6_pcrf_toys_fast_smoke_v2/`。这是吞吐问题触发的显式工程修复，不按准确率选择实现，也不静默覆盖原尝试。

**14:59资源事件与新检查：** GPU5检查启动后，其他工作负载新增约31GiB占用，检查在仅255MiB物理空闲时OOM。失败状态、日志和源码快照保留在`fast_smoke_v2`，不是数值一致性判定失败。现按实际空闲约25GiB的GPU6建立独立`gacr_v6_pcrf_toys_fast_smoke_v3`；正式`fast_v2`也改用GPU6，仍需本次全部检查通过。没有终止其他用户任务，没有修改模型参数或测试集边界，也没有自动失败重试循环。

**15:02检查通过：** GPU6上16/16用户的候选、base和最终GACR top50顺序一致，特征绝对误差≤1e-6；父模型PCRF rank 16/16精确一致。两项CPU特征测试和报告生命周期测试通过。独立检查总耗时77.64秒，原实现/批量版平均每用户1.910/1.807秒（约5.4%差异）。因此不能将先前GPU5的全部减速归因于逐候选计算，也不能称批量版带来了数倍加速；共享资源是不可忽略的因素。按本次16人计时外推全量约10小时，仅作初始资源估计，后续以正式吞吐更新。原smoke和新smoke不同硬件负载下的速度不直接作算法性能对比。

## 6. 结束后的解释

组合N10相对原PCRF为正，只记为探索正点估计；Hit@10下降或分组存在损失时一并报告。若未超过原PCRF，就结束此固定组合，不当场扫描参数。正结果也不自动启动Beauty或多seed；后续验证另记。

报告写入[组合评估报告](../../report/第二十阶段/Stage20_GACR_v6_PCRF组合评估报告.md)，摘要同步[第20阶段总报告](../../report/第二十阶段/GRAM_第二十阶段_实验总报告.md)。后台报告观察程序按真实状态和summary自动更新，失败或超时保留失败信息，不能写成科学执行完成。
