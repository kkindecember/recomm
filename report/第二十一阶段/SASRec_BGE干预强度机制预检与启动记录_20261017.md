# SASRec+BGE：α机制预检与启动记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent，内联执行
- Origin Mode: run / training-side descriptive checks
- Origin Date: 2026-10-17
- Verification Status: STARTED；预检完成，真实生成效果待后台任务完成
- Version Label: alpha_mechanism_20261017_v1

## 判断

三条方向均可实现，但原表述不足以支持收益承诺。首先检验位置相关强度，KL锚作为防漂移对照，历史证据臂作为增量检验；三者不预先合并为一个“肯定有效”的方法。

- KL(p_GRAM||p_mix)是保持GRAM覆盖的软锚；通常称forward/inclusive KL，而mode-seeking通常指KL(p_mix||p_GRAM)。向GRAM低概率商品分配更多质量仍会抽走其他商品质量，因此仍有间接惩罚；平均KL也不保证每个用户或每个已命中目标受保护。候选池路径softmax的KL不是逐前缀原生token KL。
- m是合法概率质量而非合法分支熵。0.3(1-m)始终不超过.3，会大幅减弱整体干预；保留m的混合本身不会补回非法词元占去的概率质量。平均强度对照与质量守恒检查是必要的。
- BGE有精确的有符号历史分解；正支持的top1/top3占比和加权位置只是描述。SASRec不按线性历史贡献解释。证据聚合到前缀，给兄弟词元同一个α，以保持总质量。
- 21.18记载21.11–13已经尝试过集中度和最近性选择器，也做过BGE分差分解。因此不声称“首次使用历史证据”；本轮只检验实际贡献占比控制前缀α的增量。

理论参见[Minka (2005)](https://tminka.github.io/papers/message-passing/minka-divergence.pdf)。

## 训练侧机制预检（目标索引已更正）

更正：teacher.npz的targets已经是0基索引。此前独立预检误再减1，旧m分层和路径NLL数值撤回。正式训练/生成直接使用pool.npz中正确的商品ID，不读取这两份独立预检输出，未受影响。原记录已归档至corrections_before_target_index_fix/。

固定等权SASRec+BGE，训练侧Toys2773、Beauty3310；真实目标路径分叉前缀为9792和9690。已逐条核对teacher目标=sample target−1=pool首列−1；专家路径概率及NLL与正式训练审计一致。

| 量 | Toys | Beauty |
|---|---:|---:|
| 原生目标路径NLL | 1.606290 | 1.233442 |
| 固定α=.3目标路径NLL | 1.414539 | 1.119334 |
| α=.3(1-m)目标路径NLL | 1.555625 | 1.195667 |
| α=.3(1-m)在目标路径分叉处的平均强度 | 0.008379 | 0.018442 |
| m∈[.5,.9)时专家给目标更高条件概率的比例 | 51.51%（664位置） | 47.26%（2186位置） |
| m≥.999时专家给目标更高条件概率的比例 | 33.62%（1258位置） | 23.51%（1055位置） |
| m<.5时专家给目标更高条件概率的比例 | 60.00%（10位置） | 57.58%（33位置） |

以上比例不是命中率或专家正确率；同一用户多个前缀相关，深度、分支数与m混杂。极低m仅有10/33个位置，不能据此确认单调关系。目标路径NLL与局部概率诊断均不能替代真实生成指标。

## 代码与运行检查

小树上常数α对原实现、零α退化、前缀合法总质量守恒、变长路径混合、有限差分梯度、KL零点、1/33/36参数头均通过。真实BGE缓存两域各首/中/末3用户重建，最大归一化误差约3.3e-6。完整GPU证据缓存会继续逐用户核验。

预检发现当前torch对where条件表达式中Python标量零的float32类型兼容问题，已在训练开始前改为zeros_like；补充float32/float64损失和反传检查通过。原源码/manifest及修订说明保存于sources_before_dtype_fix/、manifest.initial.json、pre_fit_dtype_amendment.json。科学公式和输入未改变，缓存未重启。

原21.49历史目录不覆盖。它的缓存阶段因进度日志重复users_total退出，未留下竞争路径缓存；本次补建缺失的冻结GRAM竞争路径前向，复用并逐条核对已有目标路径缓存，不更新GRAM或专家。

## 后台范围

独立tmux：s21_alpha_mechanism_20261017_v1；启动时Toys使用GPU7，Beauty使用GPU4，各16GiB总租约。两域各512个既有开发用户、十个预先固定α对照；KL权重.001/.01/.1。β始终(.5,.5)，全程保留历史商品、不加历史惩罚、不排除文本自匹配、不读test。

当前仅完成启动检查，真实模型的动态α生成smoke在缓存与拟合之后执行。所有十组生成完成才汇总，不自动扩样、不自动改方法、不重启其他任务。

- 状态：artifacts/phase21/alpha_mechanism_20261017_v1/status.json
- 冻结协议：plan/第二十一阶段/21.50_SASRec_BGE干预强度的锚定位置与历史证据_20261017.md
- 完成后汇总：artifacts/phase21/alpha_mechanism_20261017_v1/summary.json
- 完成后报告：report/第二十一阶段/SASRec_BGE干预强度的锚定位置与历史证据_20261017.md

统计解释：按域保留全部预定用户和十个对照，配对区间未校正多重比较，不包含种子/方法选择不确定性；开发集重复使用，不能当作独立确认。
