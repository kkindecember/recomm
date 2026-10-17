# 1024人筛选收尾与统一学习强度对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-10-17
- Verification Status: PLANNED
- Version Label: screen1024_final_controls_v1

用户已明确先用完成的1024人筛选、保留全部11组，不恢复4096扩样；最新要求“先继续整完这个”。此次完成同用户结果综合分析，并将此前已训练的统一学习alpha控制补到同一1024人，辅助判断用户级alpha是否超出全局校准收益。

不增加用户、专家或网络，不重训、不改损失或epoch选择。复用`joint_fusion_20261015_v2/{domain}/qwen4_calibrate.pt`：冻结同一CE beta及共享隐藏层，仅全局scalar alpha曾按目标路径NLL训练，原预算20轮、内部heldout选择epoch；Toys选20、Beauty选8。与当前用户alpha/NLL训练数据、父CE状态、归一化与NLL一致；选择epoch因参数化和内部结果而异，并非现在查看1024后重选。原控制与用户控制均为相同初值.3、相同训练上限。精确核对冻结层、fit用户、均值方差、旧alpha导出和当前beta。

实际生成直接复用当前1024的完整专家分数及用户顺序，唯一区别是alpha使用既有全局标量；Toys约.600622、Beauty约.502508。只补此一组，每域1024次，beam50，保持合法质量、不排除历史商品。新的`qwen4_global_alpha_1024_20261017_v1/status.json`后台记录进度，两域尽量分卡并行，预计约6～12分钟。16GiB总租约、18GiB准入、11GiB分配上限，科学失败不自动重试。

综合分析同时保留原11组与3种新Qwen alpha方案、统一alpha组，按每域首512、后512及合计1024报告全部NDCG/HR；所有方法按用户名单与原生GRAM排名逐项对齐。保存每组相对GRAM、双专家相对其单专家、行为与文本替换、Qwen alpha相对固定、旧joint、SAS-only和统一alpha的配对差和2000次bootstrap描述性区间。多重比较数量按实际pair列表×3分组×2域完整记录，不作未校正确认性显著性宣称。

额外保存实际GRAM top50未覆盖而引导覆盖、丢失的top50与top10命中、共同命中的升降位；N10增量分解为补回、丢失、原有命中重排。单专家独有命中和alpha分组仅为事后描述，不输入融合器、不用于标签选用户或宣布潜在收益可实现。不能将这些现象一概称为某层beam剪枝错误，也不能以覆盖互补代替融合增量消融。

原11组联合beta/alpha与新Qwen冻结beta方案训练不同，跨两者比较属于完整pipeline对照，不是仅文本编码器效应。ItemCF在这11组是单独或替代行为模型，没有证明它作为第三专家冗余。旧三专家有不同融合器与历史惩罚，不作为匹配候选混入。

结果保存`screen1024_review_20261017_v1/summary.json`与`report/第二十一阶段/1024用户专家与干预强度综合筛选结论_20261017.md`，统计检查11/11。完成后仅给明确候选优先级及缺口，不自动扩4096、删原11组记录或启动下一轮方法训练。所有既有科学文件、权重与完整预测保持原样。
