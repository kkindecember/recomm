# GRAM 第七阶段：ST-GCGD-v2 时序多关系图条件解码实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-08-02
- Verification Status: DESIGN_V1_NOT_PREREGISTERED
- Version Label: `phase7_st_gcgd_v2_temporal_multirelational_design_v1`
- Parent Result: `REJECT_GCGD_V1_KEEP_GACR_V3`
- Development Domains: Toys、Beauty
- Sports/Test: 封存
- Device: 物理 GPU0；单卡；30 GiB 总显存租约
- Execution: `execution_enabled=false`

## 1. 为什么不是继续调 GCGD-v1

GCGD-v1 已证明 token-level 图注入能真实改变解码，但它对 B/C 的 512/512 用户都施加影响，
两域却没有任何新增 top-10 命中，NDCG@10 相对 GACR-v3 分别下降约 5.3%–16.0%。静态
LightGCN 只编码“用户和商品是否交互”，不编码近期兴趣、交互次序或 item-to-item 转移；而任务是
next-item generation，这种目标错配比图层数或 alpha 大小更可能是失败主因。

因此 v2 不在当前 cohort 上搜索 v1 超参，而做一个可辨识的大改动：把静态 user-item 二部图替换为
**时序多关系图**，同时把“图是否可信”的 gate 改成“图相对冻结 GRAM 是否带来排序优势”的
fail-closed gate。GACR-v3 继续作为 incumbent，目标只要求先出现可信的正向增量，不要求第一版
一次达到预设的完美百分比。

## 2. 文献依据与创新边界

- LightGCN 的核心是静态 user-item 图上的线性邻域传播，适合协同过滤，却没有显式表达交互顺序；
  v1 的实现与该范式一致：<https://arxiv.org/abs/2002.02126>。
- SR-GNN 将序列构造成有向 session graph，专门捕获 item transition；这支持“下一项预测应显式建模
  转移边”，但不意味着本计划不能使用该思想：<https://doi.org/10.1609/AAAI.V33I01.3301346>。
- TiSASRec 表明时间间隔能影响下一项预测，支持 recency/time-bin 作为独立关系，而非把全部历史边
  等权处理：<https://doi.org/10.1145/3336191.3371786>。
- AGRec 已使用图推理增强自回归推荐解码，所以“GNN+生成模型”本身不是本项目的新颖性声明：
  <https://aclanthology.org/2025.findings-acl.369/>。

本项目可主张的差异组合是：面向 GRAM lexical hierarchical identifier 的 prefix mass 投影、
长期 user-item 与短期有向 item-transition 的 relation-specific 融合，以及以冻结 GRAM 排序优势为
训练标签的 fail-closed 解码控制。最终创新性仍需结合消融和效果证据，不以组件从未出现过为前提。

## 3. 研究问题与假设

主问题：**与静态 LightGCN 图相比，时序多关系图能否产生更贴近 next-item 的候选信号，并在
保守优势门控下相对 GACR-v3 获得可继续改进的正向结果？**

- H1（图质量）：train-only pseudo-future 上，时序图相对静态图提高 target score separation，且
  至少不出现两域一致的 Recall/NDCG 下降；
- H2（候选机制）：v2 相对 GRAM 或 v1-B 增加 target-in-beam@50 / new-hit@10，而不是只扰动排序；
- H3（主要效果）：至少一个 v2 arm 在 Toys、Beauty 的 NDCG@10 相对 GACR-v3 方向一致为正；
- H4（门控有效）：优势门控 E 相对固定融合 D 减少 changed coverage 与 broad harm，并保留正收益；
- H5（安全）：overall、head、tail 与 Recall@50 不出现无法由候选增益解释的系统性退化。

H1–H5 是可检验假设，不设 `+3%`、`+5%` 等早期硬门。任何版本都可基于结果继续改进，但必须
更换 development cohort、登记改动并保留完整的负结果。

## 4. ST-GCGD-v2 方法定义

### 4.1 严格的 train-only 时间切分

- 每位用户仍封存最后两个官方 holdout position；图只使用 `items[:-2]` 及其原有时间/次序信息；
- 在 train-only 用户内再构造 pseudo-future：prefix 用于建图和条件输入，下一条交互只用于图训练/
  calibration label，绝不进入图边或 gate feature；
- validation target、test、Sports、P1 cohort target 均不得用于训练、归一化、阈值或 early stopping。

### 4.2 两类关系与 relation-specific propagation

关系一 `R_ui`：user-item 长期偏好边。

- 边权由交互次数与相对位置/recency bucket 决定；具体函数在 train-only P0 后冻结；
- 不把验证时刻之后的信息写入权重。

关系二 `R_ii`：有向 item-transition 边。

- 从同一用户相邻 train interactions `(i_t -> i_{t+1})` 构造；保留方向与转移次数；
- 分开计算 outgoing/incoming message，避免无向化抹除 next-item 方向；
- self-loop、重复 item、序列首尾和稀疏 item 采用确定性规则并单测。

两个 relation encoder 分别输出 `q_ui(u,i)` 与 `q_tr(u,i)`，再由只读历史可见特征的 relation mixer
得到 `q_ST(u,i)`。v2 不简单增加 LightGCN 层数，也不复用 v1 的全局固定 alpha。

### 4.3 难负样本与训练目标

- 正样本是 train-only pseudo-future；随机负样本与冻结 GRAM train-only beam 中的 hard negatives
  混合，避免仅学会区分容易的 catalog 负样本；
- 目标为 relation-wise pairwise logistic loss + top-k listwise loss；权重只能在 train-only P0 冻结；
- 记录正负 score margin、Recall/NDCG、head/tail、transition-covered/uncovered、边密度和冷启动覆盖；
- 如果图在 train-only pseudo-future 上连静态 v1 都不能稳定改善，则停止进入解码实验，先修图模型。

最后一条是机制资格检查，不是对 validation 效果设置苛刻百分比门。

### 4.4 Prefix 投影与优势门控

保留 v1 已通过的 Trie 合法 prefix mass 投影，但修正控制方式：

`L_v2(t|u,p) = L_GRAM(t|u,p) + beta(u,p) * norm(G_ST(t|u,p))`

- `beta(u,p)` 有界且默认为 0；无 transition、低 margin、高熵、图/GRAM 强冲突时必须回退 GRAM；
- gate 监督标签改为：在 train-only pseudo-future 上，注入图信号是否比 `beta=0` 改善目标 rank/
  NDCG，而不是仅预测“图 top-1 是否等于 target”；
- calibration 目标优先控制 false-positive intervention；保存 reliability diagram、ECE、precision、
  intervention coverage 与 conditional gain；
- 不允许读取真实 validation target 作在线 oracle gate，也不允许在 P1 后扫描 beta。

## 5. 对照与可辨识性

同一 cohort、checkpoint、beam、Trie、tie-break、评测代码下比较：

- A：原始 GRAM；
- V3：冻结 GACR-v3；
- B：冻结 GCGD-v1 静态图固定融合，用于定位结构改动；
- D：ST-GCGD-v2 时序多关系图 + 固定的 train-only calibrated fusion；
- E：D + advantage gate，作为 v2 主方法。

`D-B` 回答时序多关系图是否优于静态图，`E-D` 回答优势门控是否减少错误干预，`E-V3` 是主要
开发比较。一次实验不再同时改 backbone、GRAM checkpoint 或 lexical ID。

## 6. 分阶段执行

### P0-R：工程与资源修复

1. 修正 GACR-v3 `target_in_candidate_beam50` 命名/计算，区分 union candidate rank 与真实 beam@50；
2. 修复 identity 空诊断集合引起的 `mean of empty slice`；
3. 增加逐 arm 行数、sample-key 唯一性、metric re-aggregation 与 finite 自动审计；
4. 用完整 CUDA 生命周期做 train-only memory pilot，而不是只测短 smoke；采集 workload-only 峰值后，
   冻结每域 workload budget 与 sidecar，使二者合计恰为 30,720 MiB；
5. P0-R 不读取新 development cohort，不作效果选择。

根据 v1 遥测反推的旧架构参考下限约为 Toys 20,982 MiB、Beauty 17,354 MiB，但 v2 架构不同，
这些数字只能用于发现旧预算错误，不能直接注册 v2。初始保守候选为 Toys workload 22,528 MiB /
sidecar 8,192 MiB，Beauty workload 19,456 MiB / sidecar 11,264 MiB；必须由 v2 全路径实测确认或上调。

### P0-G：train-only 图资格实验

- 比较 static LightGCN、仅 `R_ui`、仅 `R_ii`、完整 ST graph；只用 train-only pseudo-future；
- 固定 seed=2023，输出两域和 head/tail/transition coverage，不读取 P1 development target；
- 允许依据机制结果冻结唯一 v2 图配置；所有候选、决定和 SHA 写入 preregistration lineage；
- 若完整 ST graph 无优势，先写失败报告和新设计，不用 validation cohort 拯救配置。

### P1：新 cohort 单 seed 开发

- seed=2023；Toys/Beauty 各 512 位用户；新 salt 暂定
  `phase7-st-gcgd-v2-development-v1`，并排除第七阶段 v1 P1 与此前全部 validation salts；
- arms=A/V3/B/D/E；不读取 test/Sports；不在该 cohort 上扫描模型深度、loss、beta 或阈值；
- 指标：Recall/NDCG@5/10、Recall@50、MRR、真实 target-in-beam@50、new-hit@10 outside A beam、
  changed、broad harm、head/tail、transition-covered/uncovered、gate intervention precision；
- 用户级 paired bootstrap 10,000 次，同时报告绝对 pp、相对百分比和 95% CI；
- 只要 E 或 D 相对 V3 出现跨域方向一致的正向信号，或候选/new-hit 机制明显改善且安全退化可修，
  就可以继续 v2.1/P2；不要求一次达到固定提升百分比。

### P2：冻结后三 seed fresh validation

- 仅在 P1 完成且研究者明确批准后设计；seeds=2023/2024/2025，使用与 P1 完全互斥的新 cohort；
- P1 后只保留一个主配置；主要比较 E vs V3，机制比较 E vs D、D vs B；
- P2 前另写冻结 config、效应解释门与 test 解封条件；P2 仍不自动读取正式 test。

## 7. 第六阶段实验治理规则（全部继承）

- 所有 GPU workload 固定物理 GPU0，`CUDA_VISIBLE_DEVICES=0`；禁止自动切换 GPU；
- 所有实验均在具名后台 tmux 中运行，runner 必须提供 `{start|status|worker}`；status 展示 tmux、
  `status.json`、日志 tail、当前数据域、PID、GPU0 telemetry 与 CodeLlama 状态；
- 实验前 GPU0 必须由 CodeLlama 占位；正式 workload 前停止 CodeLlama并执行 admission gate；
  workload 成功、失败、timeout、SIGTERM/HUP 后均先清理 sidecar，再在 GPU0 恢复 CodeLlama；
- 当实验 workload 峰值低于 30,720 MiB，无论实际使用多少，都必须用 sidecar 将总占用补到
  **30,720 MiB**；workload 本身显存计入总租约，不能让“30GB sidecar + workload”叠加超额；
- telemetry 若观测总占用持续超出 30,720 MiB + 测量容差，立即标记
  `RESOURCE_LEASE_OVERSHOOT`，终止该工程运行并重新测量，不得把它写成合规成功；
- hard timeout、no automatic retry；失败后禁止自动换配置、换 seed、换 cohort 或重启；
- parent checkpoint 只读且运行前后校验 SHA；Sports/test 默认封存。

## 8. 统计与完整性规则

- A/V3/B/D/E 必须使用完全相同的用户和逐用户 paired 评估；任何 arm 缺行均 fail closed；
- primary comparison、指标与 bootstrap seed 在启动前冻结；post-hoc 分析必须显式标注；
- P1 是开发证据，不进行“显著即成功”的二元化，也不把同一 cohort 反复调参后的 CI 当确认性证据；
- 两域、head/tail、覆盖分组全部报告，禁止只挑最好域或最好指标；
- 绝对值、绝对 pp、相对百分比、CI 同时保存；baseline 很小时不得只报相对百分比；
- 完整性硬门：lineage、无泄漏、finite、identity、matched cohort、checkpoint SHA、30 GiB 租约、
  CodeLlama 恢复、后台/status 接口、test/Sports 封存全部通过。

## 9. 预期实现与产物

- implementation：`experiment/phase7/st_gcgd_v2.py`
- tests：`experiment/phase7/test_st_gcgd_v2.py`
- P0-R memory runner：`experiment/phase7/run_phase7_st_gcgd_v2_memory_pilot.sh`
- P0-G runner：`experiment/phase7/run_phase7_st_gcgd_v2_graph_p0.sh`
- P1 runner：`experiment/phase7/run_phase7_st_gcgd_v2_p1.sh`
- configs：`artifacts/phase7/configs/st_gcgd_v2_*_preregistered.json`
- outputs：`artifacts/phase7/st_gcgd_v2/`
- report：`report/第七阶段/GRAM_第七阶段_ST-GCGD-v2结果与验证报告.md`

当前只完成研究设计，尚未实现或冻结。`execution_enabled=false`；不得直接启动。下一动作应为
P0-R 的统计字段、warning 与全路径显存测量修复，然后实现 P0-G，不占用新的 P1 cohort。
