# 九维融合器给alpha补GRAM状态

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-16
- Verification Status: PLANNED
- Version Label: alpha_gram_features_v1

用户授权在保留9维的基础上，为alpha补充GRAM状态。只进行本次双域匹配对照；超过10分钟的实验后台运行，启动核验后交还用户，通过artifacts/status.json观察进度。

## 问题与结构

原9维仅包含历史与专家统计。虽然alpha受包含GRAM的路径NLL训练，但推断时无法区分相同9维下不同的GRAM预测状态。本轮检验额外观察GRAM能否改善干预选择。

- base9：h=ReLU(W*x9+b)，beta=softmax(B*h+c)，alpha=sigmoid(a*h+d)，419参数。
- alpha12：保留相同9维共享干路与beta分支，只在alpha的logit增加v*z3。z3依次为GRAM首次合法分支归一化熵、行为专家与GRAM的JS/log2、内容专家与GRAM的JS/log2。新增3个权重，合计422参数；v初始化为0，两组初始输出严格一致。
- 这是alpha上的简单加性接入，新增三项没有独立非线性隐藏层；不声称覆盖所有上下文交互。beta在固定参数下不依赖z3，但联合训练可通过共享干路间接改变最终beta。
- beta与alpha依旧每次推荐计算一次，专家条件分支概率继续在所有生成前缀参与；不是逐步重算alpha。

## 数据与匹配训练

固定TiSASRec+BGE、冻结GRAM；Toys训练2773、Beauty训练3310，两域各512开发用户，fit/dev用户互斥，不访问test。

前9项直接复制已核验的reduced9缓存；三项额外特征复用decision_features的首层分布缓存，重新计算熵及JS并逐项对照旧7维列[1,5,6]。GRAM/专家分布位于同一合法首token空间；Toys30、Beauty108分支。源缓存使用decoder_input=[0]且已做placeholder目标交换不变性检查，不使用真实目标路径计算特征。

两组复用同一9维CE父模型（全ranker_fit及内部fit-only两份，原CE选轮12）；重新联合训练路径NLL，seed0，AdamW .002/wd.001，batch32，20轮预算，内部用户hash80/20，候选轮0/4/8/12/20；选轮后从相同全量CE父模型重新拟合。标准化只用对应fit部分，前9维均值方差需精确保持；新增3维独立按同规则标准化。全量/内部CE父模型所见用户与规范一致。

base9必须重现上轮reduced9的beta、alpha、全目录分数和最终排名，否则停止解释新结果。原10维结果保留为历史参照，不重训第三组。

## 核验与评价

启动前检查：base9前向/优化更新复现；新增权重为0时与base9一致；单独改变任一额外输入只影响alpha，不影响固定参数下的beta；独立NumPy前向；熵/JS边界及分支置换不变性。

两组fresh beam50生成，同规则native50后重排；沿用score_scale20、原合法概率质量、长度惩罚1、历史过滤最近20。每域native/alpha0一致性检查；全512用户base9对历史商品集合、分数及最终排名核验。结束后独立排序计算H/NDCG@5/10/20/50，配对描述性区间和命中变化，同时报告alpha/行为权重统计与新增系数。

主比较alpha12对base9的NDCG@10，同时报告对GRAM、原10维、同规则后重排。各512用户、单seed、重复开发池，区间未经多重比较校正；不证明因果纠错能力或最终泛化。不预设一定改善，不自动搜索更多特征或扩大网络。

## 资源与交付

两域最多并行2张GPU，自动选择显存充足且较空闲的卡；沿用用户授权16GiB/域预算（启动空闲18GiB、预计工作12GiB、进程分配上限11GiB）。仅资源准入失败可重新排队；科学失败不自动重试。每域6小时硬上限、资源等待24小时，状态每10秒更新，完成自动汇总及释放租约。

输出：artifacts/phase21/alpha_gram_features_20261016_v1。
报告：report/第二十一阶段/九维融合器补GRAM状态匹配对照_20261016.md。
