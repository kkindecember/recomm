# 第20阶段实验报告索引

**2026-09-15提交归档：** [内部蒸馏提交快照](Stage20_Toys完整列表内部蒸馏提交快照_20260915.md)、[训练教师诊断](蒸馏运行期间的教师诊断与方法对照_20260915.md)、[排序迁移文献借鉴与下一版建议](排序迁移文献借鉴与下一版方法建议_20260915.md)。实时动态报告及status仅保留在实验机器，未随Git提交；快照中的进度不代表最终评估结论。

**2026-09-15显存问题恢复：** 凌晨真教师初始化因显存门槛失败，队列已停；没有新验证结果。恢复复用已完成缓存，增加启动余量、连续资源检查和初始化后等待，三臂原科学协议不变。当前看[恢复status](../../artifacts/phase20/gacr_list_distill_toys_recovery_v3/status.json)、[动态报告](Stage20_Toys完整列表内部蒸馏进展与结果.md)及[恢复计划第10节](../../plan/第二十阶段/Toys完整候选列表内部蒸馏实验计划.md)。旧FAILED目录保留；下方“缓存持续计算”等文字是凌晨停止前的历史记录。

**2026-09-15已改分卡：** 当前Toys三臂的共享缓存持续计算，缓存完成后每臂使用不同GPU独立训练和验证；优先CE→GPU4、真教师→GPU1、打乱教师→GPU6，实际绑定受原显存准入限制。科学协议保持冻结，暂未产生新的训练结论。见[动态报告](Stage20_Toys完整列表内部蒸馏进展与结果.md)、[统一状态](../../artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json)和[接管验证](../../artifacts/phase20/gacr_list_distill_toys_gpu1_v2/parallel_v2/launch_verification.json)。

**2026-09-14内部迁移实验已启动：** [Toys完整列表蒸馏动态报告](Stage20_Toys完整列表内部蒸馏进展与结果.md)、[GPU1后台status](../../artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json)、[预登记方案与资源修订](../../plan/第二十阶段/Toys完整候选列表内部蒸馏实验计划.md)。CE＋父保持/真教师/打乱教师三臂将更新C1解码器，完整重新生成候选并比较八指标及覆盖。

**2026-09-13 15:05 B1多初始化验证已完成：** 11分57秒、6/6任务成功，两域N10/H10均3/3个残差初始化超过自身B0。B1平均N10对原PCRF+4.65%/+4.09%、H10+4.65%/+3.68%；全量八指标对B0七升一平，保留Toys2025低频H10净少2人的例外。见[最终报告与初始化对比图](Stage20_B1缓存复用多初始化验证进展与结果.md)、[完成status](../../artifacts/phase20/gacr_b1_multiseed_v1/status.json)及[下一步第11节](../../plan/第二十阶段/GACR_PCRF双域提点后的方法推进计划.md)。

**2026-09-13 05:04 B1/B2已完成：** 约7小时59分、4/4任务成功。B1相对原PCRF的Toys/Beauty N10提升4.68%/4.02%、H10提升4.69%/3.58%；B1/B2相对B0八指标七升一平，但B2未显示相对B1的额外优势。先读[完成报告：全部指标、分组、训练轨迹及下一步](Stage20_固定候选最终融合训练进展与结果.md)、[计划第9节](../../plan/第二十阶段/GACR_PCRF双域提点后的方法推进计划.md)、[完成status](../../artifacts/phase20/gacr_fusion_train_v1/status.json)。以下结论保留各批次时间点。

**2026-09-12 16:39 A1/A2完成：** Beauty固定控制与两域三个残差seed全量评估结束，耗时34.9分钟。完整组合平均N10相对原PCRF：Toys +3.13%、Beauty +1.78%；固定候选方案 +3.26%/+1.78%。两种方案两域N10/H10均3/3个seed提高；八项指标均值七升一降，H50仍下降。已独立复算1680项指标与130组配对结果。先读[最终报告：全部指标、区间和下一步](Stage20_Beauty固定控制与双域残差多seed进展报告.md)、[计划第7节](../../plan/第二十阶段/GACR_PCRF双域提点后的方法推进计划.md)。下一步选择固定候选做匹配融合训练，B/C尚未启动。

**本轮开跑前的seed2023结论：** GACR-v6+PCRF的Toys、Beauty全量评估均完成，组合相对原PCRF的N10分别+2.81%/+2.00%，Top10净增78/57；Top50净少50/20。Toys固定候选方案N10=0.081016（对原PCRF+2.97%），Beauty同定义控制在当时尚未执行，现已补齐。历史分析见[双域结果与提点路线](Stage20_GACR_PCRF双域结果与提点路线_20260912.md)。

上一批终态核查保留于[终态复盘与下一步决策](第20阶段_终态复盘与下一步决策_20260912.md)，后续冻结协议见[验证与方法决策计划](../../plan/第二十阶段/第20阶段收尾后的验证与方法决策计划.md)；当前结论以上述双域报告为准。

## 用户授权继续后的两项工作

| 工作 | 状态 | 报告与证据 |
|---|---|---|
| Toys分数/残差/候选三个固定控制 | 01:33完成；共九臂、19,412人；v1工程失败保留、v2精确核对通过 | [结果及分析](Stage20_Toys_GACR分数与候选固定对照报告.md)、[summary](../../artifacts/phase20/gacr_score_controls_toys_v2/summary.json) |
| Beauty冻结GACR-v6+PCRF | 10:54完成；22,363人六臂，队列COMPLETED/退出码0；独立复核240项指标通过 | [结果与完整分组](Stage20_后续固定对照与Beauty迁移进展报告.md)、[双域分析](Stage20_GACR_PCRF双域结果与提点路线_20260912.md)、[summary](../../artifacts/phase20/gacr_v6_pcrf_beauty_v1/summary.json) |
| Beauty固定控制与两域残差三seed | 16:39完成；4/4任务、6/6控制或seed阶段；1680项指标独立复算通过 | [最终报告](Stage20_Beauty固定控制与双域残差多seed进展报告.md)、[summary](../../artifacts/phase20/gacr_controls_multiseed_v1/summary.json) |
| 两域B1/B2匹配继续训练 | 09-13 05:04完成；120步残差更新、骨干未更新；B1/B2均新增提点，融合目标额外优势未获支持 | [最终分析报告](Stage20_固定候选最终融合训练进展与结果.md)、[summary](../../artifacts/phase20/gacr_fusion_train_v1/summary.json) |
| B1缓存复用多初始化 | 09-13 15:05完成；4项新训练、两域各3/3 N10/H10超过自身B0 | [最终报告](Stage20_B1缓存复用多初始化验证进展与结果.md)、[summary](../../artifacts/phase20/gacr_b1_multiseed_v1/summary.json) |

## 正式实验：6/6已有完整报告或停止报告

| 实验 | 状态 | 报告 | 原始证据 |
|---|---|---|---|
| SPRec Toys | 完成；Beauty未触发 | [专项报告](Stage20_SPRec运行与最终结果报告.md) | [summary](../../artifacts/phase20/sprec_gram_toys_v1/summary.json) |
| round1_sft全量 | 完成 | [专项报告](Stage20_round1_SFT全量验证报告.md) | [summary](../../artifacts/phase20/round1_sft_full_v1/summary.json) |
| GACR-v6+PCRF Toys | 完成；单域单残差seed | [专项报告](Stage20_GACR_v6_PCRF组合评估报告.md) | [summary](../../artifacts/phase20/gacr_v6_pcrf_toys_fast_v2/summary.json) |
| S-DPO hard4 | 用户要求中止；无全量结果 | [停止报告](Stage20_SDPO停止与结果报告.md) | [stop receipt](../../artifacts/phase20/sdpo_gram_toys_v1/stop_receipt.json) |
| PCRF七臂组件消融 | 完成 | [组件及复盘报告](第20阶段_研究收敛复盘与PCRF组件对照报告.md) | [summary](../../artifacts/phase20/pcrf_component_audit_20260911_v2/summary.json) |
| PP个人流行度探针 | Toys完成；Beauty未触发 | [EAGER/PPAC记录](EAGER_PPAC实验进展报告.md) | [summary](../../artifacts/phase20/personal_popularity_probe_v1/Toys/summary.json) |

完整EAGER/PPAC尚未训练；Beauty PP仅完成smoke，不能计为正式完成实验。SPRec六个子阶段均记录，只有所选DPO和另行授权的round1_sft做全量；没有漏报其余轮次的全量结果，因为原计划未执行这些评估。

## 工程attempt与观察器

| 记录 | 最终情况 | 对应报告 |
|---|---|---|
| SPRec smoke v1/v2/v3 → v4 | 梯度对象检查、父参考数值差异、EOS采样问题分别保留失败；v4通过 | [启动记录](GRAM_第二十阶段_实验进展报告.md)、[SPRec终态](Stage20_SPRec运行与最终结果报告.md) |
| S-DPO smoke | 通过，正式运行后主动停止 | [停止报告](Stage20_SDPO停止与结果报告.md) |
| SFT parity16 | 通过，正式全量完成 | [SFT](Stage20_round1_SFT全量验证报告.md) |
| GACR原正式v1 / fast_smoke_v2 / fast_smoke_v3 / fast_v2 | 慢速中止 / 共享显存OOM / 检查通过 / 全量完成 | [GACR](Stage20_GACR_v6_PCRF组合评估报告.md) |
| PCRF audit v1 → v2 | 历史tail_mass元数据检查失败 → 重算元数据后正式完成 | [组件报告](第20阶段_研究收敛复盘与PCRF组件对照报告.md) |
| PP Toys/Beauty smoke | 两域64人检查通过；正式仅Toys | [PP记录](EAGER_PPAC实验进展报告.md) |
| SPRec supervisor / followup_queue / report_watch | 均COMPLETED；旧overview含S-DPO停止而显示FINISHED_WITH_ERRORS | [总报告](GRAM_第二十阶段_实验总报告.md) |

smoke、数值审计、历史缓存对齐和观察器不计为新的推荐方法胜负。

## 历史分析与审计证据

- [9月10日全项目选型复盘](全项目实验复盘与第20阶段选型报告.md)、[SPRec/S-DPO依据](SPRec与S-DPO提点依据复核.md)。
- [9月11日停止前中期审计](第20阶段_20260911中期审计与继续停止建议.md)、[PCRF贡献边界](PCRF有效性与论文贡献边界复核.md)。
- [原自动总报告及终态补充](GRAM_第二十阶段_实验总报告.md)。
- [9月12日逐用户审计JSON](../../artifacts/phase20/review_20260912/audit.json)、[复算脚本](../../artifacts/phase20/review_20260912/audit_completed.py)、[主结果CSV](../../artifacts/phase20/review_20260912/full_validation_metrics.csv)。

旧文档保留原日期、协议和结论时间点，顶部已补最新说明。原Beauty六臂和后续A1/A2队列均已结束；本次完成结果分析与缓存复算，没有新训练或test消费。A1/A2已完成的残差seed评估列于顶部，B/C尚未启动。
