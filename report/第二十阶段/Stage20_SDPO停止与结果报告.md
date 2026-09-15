# Stage20 S-DPO停止与结果报告

## Material Passport

- 日期：2026-09-12；Origin Skill：academic-research-suite / experiment-agent。
- Origin Mode：validate；Verification Status：ANALYZED。
- Version Label：sdpo_stopped_final_v1。
- 科学状态：当前hard4适配配置中止，未完成三epoch及正式全量评估；不代表完整原论文复现失败。

## 1. 停止状态与设置

已于2026-09-11 14:08:28按用户要求安全停止，状态STOPPED_BY_USER；不自动重启。完成2个epoch，每轮109,361条训练前缀、855 updates；第3轮完成407 updates、52,096条前缀，总更新2,117。最近checkpoint连同optimizer与随机状态保留，本次复核其SHA与停止记录一致。

起点与reference为同一原GRAM，reference固定；全参数训练、纯S-DPO、每正例4个原父beam8高排名非target商品，beta=0.1、lr=1e-5、有效batch128、microbatch2、FP32且关闭TF32/dropout。该负例及全目录生成设置是对原论文的适配，不能把方法间差异只归因于负例个数。

## 2. 已完成的3000人选择validation

这3000人与SPRec选择队列相同；下表不是19,412人全量，不能直接与全量主表比较绝对数值。

| 方法 | N5 | N10 | N20 | N50 | H5 | H10 | H20 | H50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 同精度原GRAM | 0.064771 | 0.072783 | 0.081481 | 0.091334 | 0.084333 | 0.109333 | 0.144000 | 0.193333 |
| 原GRAM+PCRF | 0.062393 | 0.072542 | 0.081871 | 0.090084 | 0.083667 | 0.115333 | 0.152333 | 0.193333 |
| epoch1 raw | 0.033637 | 0.039089 | 0.043443 | 0.049972 | 0.043000 | 0.060000 | 0.077000 | 0.110000 |
| epoch1+PCRF | 0.044350 | 0.050069 | 0.054302 | 0.057981 | 0.057333 | 0.075000 | 0.091667 | 0.110000 |
| epoch2 raw | 0.008102 | 0.009910 | 0.012464 | 0.017240 | 0.012000 | 0.017667 | 0.027667 | 0.052000 |
| epoch2+PCRF | 0.018933 | 0.021482 | 0.023522 | 0.025656 | 0.025000 | 0.033000 | 0.041000 | 0.052000 |

epoch1/2 raw N10相对同用户父模型为-46.29%/-86.38%；epoch2加PCRF后对同用户原PCRF仍为-70.39%。两个选择指标最佳均为epoch1，但其全量推理没有执行。不存在可合法填入的最终全量summary。

中期逐用户审计：epoch2相对父模型Top10新增16、丢失291，净少275；Top50新增58、丢失482，净少424。N10增量-0.062872，4000次用户配对bootstrap描述性95%区间[-0.070627,-0.055319]。本次重算两轮各8项排名指标均与记录一致。

## 3. 退化证据与解释范围

两个epoch平均loss从1.543598降到1.447516，margin从0.089722升到0.255996，正例token NLL从0.990526升到1.298278。第3轮停止前最后进度事件NLL约1.409831。训练均值不是固定样本、固定权重时刻的探针，不能将它当精确遗忘量。

已有中期输出审计中，同一3000人Top1不同商品数从父模型2,003降到epoch1的1,022、epoch2的386；最集中Top1商品覆盖656人（21.9%）。平均响应长度只由6.116变到6.202，现有证据不支持单靠长度变化解释如此大的退化。

负例bank共109,361行、437,444个负例，target碰撞为0、每行4个负例互异、缓存reference有限且哈希匹配。11.08%的负例槽位属于当前前缀已见商品，这不等同于已证实11.08%的错误偏好标签。没有用未来validation/test目标筛训练负例。

综上，当前配置存在明确的训练目标改善与真实beam指标退化并存；尚未确定负例策略、优化步长、beta、reference或其他因素中的单一根因。S-DPO与SPRec在多方面不同，不能据此下“多负例造成崩溃”的因果结论。

## 4. 收尾决定

保留epoch1、epoch2、last_progress及完整bank，不恢复同版训练、不补第3轮或自动Beauty。以后若因新证据重开偏好训练，应先冻结一个有区分力的控制并说明与PCPS等历史路线的区别。当前优先分析GACR组合的正信号。

本配置完整科学评估未完成；主动中止与失败工程attempt分开记录。用户bootstrap仅描述重复开发validation，不包含训练随机性；没有新test读取。

## 5. 可追溯证据

- [原计划](../../plan/第二十阶段/S-DPO并行实验计划.md)、[固定配置](../../experiment/phase20/sdpo_toys_v1.json)。
- [停止记录](../../artifacts/phase20/sdpo_gram_toys_v1/stop_receipt.json)、[来源与运行manifest](../../artifacts/phase20/sdpo_gram_toys_v1/manifest.json)。
- [epoch1指标](../../artifacts/phase20/sdpo_gram_toys_v1/epoch1_sdpo.metrics.json)、[epoch2指标](../../artifacts/phase20/sdpo_gram_toys_v1/epoch2_sdpo.metrics.json)。
- [停止前中期审计](第20阶段_20260911中期审计与继续停止建议.md)、[中期机器证据](../../artifacts/phase20/review_20260911/audit.json)。
- [本次全阶段审计](../../artifacts/phase20/review_20260912/audit.json)、[终态复盘](第20阶段_终态复盘与下一步决策_20260912.md)。
