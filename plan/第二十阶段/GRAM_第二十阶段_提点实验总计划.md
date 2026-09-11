# GRAM 第20阶段：以准确率增益为目标的实验总计划

日期：2026-09-10。授权来源：用户要求全面分析既有项目、寻找可以提点的方法并启动第20阶段；允许模型大改及迁移已有方法，暂不以创新性筛选。

**2026-09-11用户授权的后续执行：** 已安全停止S-DPO hard4；SPRec原队列继续。新增已保存round1_sft的19,412人Toys全量验证（GPU0，零新增训练），以及[GACR-v6/PCRF固定组合评估](GACR_v6_PCRF组合评估计划.md)（当前GPU6，Toys全量，零新增训练、不自动Beauty；此前GPU5尝试和资源修订在专项计划保留）。[第20阶段实验总报告](../../report/第二十阶段/GRAM_第二十阶段_实验总报告.md)及各专项报告由后台观察程序随结果更新。原下文启动状态、候选顺位和当时时间预算保留为历史记录。

第18、19阶段保持已经收尾的状态。复盘及候选排序见[全项目报告](../../report/第二十阶段/全项目实验复盘与第20阶段选型报告.md)。本计划先正式启动A1，随后按用户要求增加A2并行实验；其余候选仍是后续路线。

**启动状态：2026-09-10 21:57，GPU7后台队列已启动。** GPU工程检查和5项单元测试通过，当前先补同精度父模型推理参考，再自动进入正式训练。详见[启动记录](../../report/第二十阶段/GRAM_第二十阶段_实验进展报告.md)。

**并行补充：** 用户随后希望等待期间也运行其他实验，现将A2提前为独立S-DPO四负例分支，见[并行计划](S-DPO并行实验计划.md)。原SPRec配置不变；新增统一观察入口`artifacts/phase20/overview/status.json`。

## 1. 目标与比较对象

主问题：在正常推荐、原GRAM数据协议下，能否提高NDCG@10和Hit@10？

- 主比较：新模型相对**原GRAM**，直接复用Beauty epoch25、Toys epoch30父模型权重。开跑前发现新FP32推理与历史缓存存在候选/分数差异，因此每域先补一次同精度全量validation推理，作为配对主参考；历史逐用户参考同时保留。
- 同时报告：新模型+冻结PCRF相对原GRAM，以及相对原GRAM+同一PCRF的增量。
- 成本、尾部和深截断作为重要取舍，不再设“必须极轻量”“所有尾部指标严格不降”“必须创新”的准入要求。
- 仅用train训练，validation选择与探索；不构造或重新评估test。以往重复使用validation的限制明确写进报告。

优先NDCG@10选checkpoint，同时完整记录Hit@5/10/20/50和NDCG@5/10/20/50。一个指标上升、另一个下降时照实报告，不笼统称全面改善。

## 2. A1：SPRec自博弈偏好流程迁移到GRAM

来源：[SPRec作者论文](https://hexiangnan.github.io/papers/www25-SPRec.pdf)、[代码](https://github.com/RegionCh/SPRec)。本地独立实现和差异见[来源清单](../../experiment/phase20/source_manifest.json)。这不是原作者Llama模型/原数据的完整复现。

每轮执行：

1. 由当前轮起点模型对**全部训练前缀**作合法词法路径的祖先采样，得到一个预测商品作为负例。
2. 所有训练前缀做一轮正样本SFT，保持完整encoder/decoder可训练。
3. 冻结SFT结束的模型作为本轮DPO reference；当前模型作一轮DPO。
4. 在固定3000人validation选择队列上记录SFT和DPO的真实beam50结果。下一轮从本轮DPO末状态继续，不从选择队列最优checkpoint跳转。

固定三轮，即三次全量负例生成、三轮SFT和三轮DPO。负例等于当前训练正例时，该行不进入DPO；仍参加SFT。缓存用训练行索引区分同一用户的多个前缀，禁止用validation目标生成训练偏好。

DPO响应分数为完整商品词法响应的token log概率之和，含EOS，不含decoder start和padding：

\[
L=-\log\sigma\{\beta[(\log\pi_\theta(y^+|x)-\log\pi_\theta(y^-|x))
 -(\log\pi_{ref}(y^+|x)-\log\pi_{ref}(y^-|x))]\},\quad\beta=0.1.
\]

响应策略概率取原模型完整词表softmax，推理仍用原生词法trie/beam50。负例来自经过合法trie约束的采样分布，这个分布不等于未经约束的原始语言模型分布；这是可明确审计的迁移选择，不声称严格复现原论文全部理论前提。

原作者发布代码使用贪心标题生成；本轮采用论文“预测分布采样”思路，设`do_sample=True, top_k=0, top_p=1, temperature=1`。保留作者代码中“先SFT，再将SFT权重作为DPO参考”的顺序，DPO默认beta=0.1。原作者数千样本、LoRA、标题grounding和其选模逻辑不照搬。

隐式反馈中的其他商品不必然是用户不喜欢的商品；这里是下一商品排序训练的负例近似。出现正例碰撞直接过滤，不把碰撞强制当负例。

## 3. 固定预算与实施参数

| 项目 | 设置 |
|---|---|
| 首域 | Toys |
| 第二域 | Toys完成后，若原模型增量或PCRF组合新增增量的全量validation N10点估计为正，自动按同一科学配置执行Beauty；否则停止此配置队列 |
| Toys训练/验证/商品数 | 109,361 / 19,412 / 11,924 |
| Beauty训练/验证/商品数 | 131,413 / 22,363 / 12,101 |
| 父模型 | 原GRAM完整checkpoint，不加载DIFF、RaSeRec、HI-GRAM或阶段17模块 |
| 数值参照 | 同一父权重先跑一次FP32、TF32关闭的全量validation；新模型与此参考比较，并另列历史缓存指标 |
| 训练seed | 2023；本轮单seed探索，不代表多seed确认 |
| 训练量 | 3轮；每轮全量负例生成 + 1 epoch SFT + 1 epoch DPO |
| 学习率 | SFT、DPO均1e-5；各子阶段重新建立AdamW，weight decay0.01，线性衰减，最多20步warmup |
| batch | microbatch4，有效batch128；负例生成microbatch8 |
| 精度 | FP32、TF32关闭；DPO两侧dropout关闭，SFT使用原训练模式 |
| 推理 | 原生GRAM生成，beam50，length penalty1.0，validation microbatch1；不启用阶段18cross-cache补丁 |
| 选择队列 | seed20260910随机选3000个validation用户，不依赖其标签/历史指标；每轮保持相同 |
| checkpoint选择 | 三个DPO轮末checkpoint中分别按raw N10、组合N10选择；SFT为描述性轨迹，不冒充独立匹配控制 |
| 全量评估 | 对上述最多两个不同选中checkpoint各跑一次全量validation；同时计算raw和冻结PCRF |
| GPU预算 | 单卡；每域正式任务最多72小时，达到边界保存状态并停止，不自动延长 |
| 重试 | 工程检查可修复后另建明确attempt；正式任务失败不自动重跑，不静默覆盖 |

三轮是本次明确训练预算，不是预先认定已经收敛。若loss仍下降或曲线不稳定，终态报告必须记明；不能把预算结束写成该方法已充分收敛失败。

Toys全量正点估计触发Beauty，只是资源排序规则，不是显著性结论。若组合只超过原GRAM、未超过原PCRF，且模型单独未超过原GRAM，则不因已有PCRF收益重复投入第二域。

## 4. 开跑前检查与记录

- 验证prepared文件和父模型SHA256，固定源代码快照及配置。
- 单元检查：reference一致时loss=log2；偏好梯度方向；EOS计分/padding忽略；参考无梯度；极端数值有限。
- GPU检查：最长历史batch反向传播，encoder、decoder、shared embedding与位置嵌入均有有限非零梯度；冻结reference无梯度。
- PCRF适配器在历史原候选/分数上检查前64条逐用户rank精确一致；另对前16条检查新推理与历史候选差异及TF32干预。新FP32候选不要求与旧数值路径逐位相同；正式训练之前补齐同精度父模型参考。
- 另用8条训练数据跑通真实“生成负例→SFT→DPO”执行路径；全部smoke状态和checkpoint放在独立目录，正式任务重载未修改父模型。
- 原始预测、负例、逐用户rank、配置、源码快照及阶段checkpoint均保留。
- 每100 optimizer steps及子阶段结束保存检查点；保存optimizer、scheduler和各RNG状态。当前不做自动恢复，后续恢复需显式匹配阶段、负例缓存和数据顺序。

## 5. 资源与后台执行

使用配置指定的GPU UUID；启动选择为GPU7。遵循用户第19阶段明确表达的“按实际可用显存运行”，不沿用30GiB holder和18.5GiB固定准入规则。初始显存准入值是该新工作负载的工程估计，将根据最长历史GPU检查记录调整；不占用其他任务，也不调用旧CodeLlama占位恢复流程。

检查后固定：最长历史microbatch4的DPO峰值allocated约3.15GiB，reserved约3.58GiB；正式准入为12GiB空闲，额外覆盖原生beam50和分配器变化，allocator上限40%。这是可用量准入和上限，不是实际预留/占用。

后台入口：

```bash
bash experiment/phase20/run_stage20.sh start
bash experiment/phase20/run_stage20.sh status
```

会话：`s20_sprec_gram_v1`。总状态：`artifacts/phase20/status.json`。Toys、Beauty各有独立`status.json`和`run.log`。预计超过10分钟的正式任务后台运行，确认启动成功后结束当前监看。

中止入口：`bash experiment/phase20/run_stage20.sh stop`。仅向本阶段监督进程发信号，由它转发到当前子进程；不根据旧PID或GPU编号停止其他进程。

## 6. 完成后的判断顺序

1. 确认正式训练轮次、样本量、选模和预测完整性，再看指标。
2. 同时列raw/组合相对原GRAM、组合相对原PCRF；逐用户配对CI标记为重复使用validation上的描述性区间。
3. Toys有正信号则按已固定配置跨域；不临时修改Beauty的学习率、beta、数据划分来制造双域增益。
4. 结合两域结果与已并行启动的S-DPO结果，决定是否值得追加独立训练seed/匹配SFT对照，或者进入完整排序奖励训练。当前不会自动进行额外参数搜索。
5. 若失败，优先检查正例碰撞率、DPO margin、正例NLL与SFT/DPO轨迹：区分负例质量不足、偏好训练破坏已学能力、预算不足及真实无收益。不得仅用“效果一般”作为报告。

备选的HSTU等骨干大改仍保留，不受轻量要求限制。GACR-v6/CET旧正信号可在新目标下单列复核；历史gate裁决保持原样。

## 7. 启动前数值检查修订

第二次工程检查中，16名用户的新父模型raw rank与历史一致，但PCRF有一名用户从历史41变成40；16组候选排序均非完全一致，其中3组有一个候选替换。尚未运行任何正式训练，已固定增加同精度父模型推理作为主比较参考。该修订由数值一致性问题触发，不由新方法性能触发。历史结果和已失败检查目录保持原样。
