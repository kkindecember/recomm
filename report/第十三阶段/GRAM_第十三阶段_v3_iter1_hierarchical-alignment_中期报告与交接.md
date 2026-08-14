# GRAM 第十三阶段 v3 iter1 中期报告 & 会话交接文档

**撰写日期**: 2026-08-14
**状态**: ⚠️ **Stage 1 快筛已全部跑完(flat 于 2026-08-14 收尾)**;`uniform` 配置仍只有单种子(见 §7.2)
**读者**: 接手本工作的下一个会话/协作者
**结论概要**: v3 的三种 alignment 变体在 MLP 快筛上**均未超过 v1,也均未超过纯 trunk 对照组**;**尚未启动任何 GRAM 训练**,也**不建议在当前证据下启动**

---

## 0. 给接手者的 30 秒摘要

1. v2(LLM prior)已 **abandoned**,结论经完整复核确认,不要重启
2. v3(hierarchical alignment)我实现了并做了 MLP 快筛:**三种变体全部没超过 v1,也全部没超过纯 trunk 对照组**。五个配置在两域的排序完全一致:**alignment 生效越充分,结果越差**(§6.4)
3. **没有跑 GRAM,一次都没有**。GPU 只用了几分钟级别的 MLP 训练
4. **Stage 1 快筛已全部跑完**(flat 12/12)。唯一未补齐的是 `uniform` 配置只有单种子(§7.2,约 6 分钟)
5. Stage 1 已跑完(flat 12/12)。我的建议:v3 判 abandoned,然后**先做零成本的"分析 v1 为何成功"**,用它的结论决定下一个组件是 v4-retriever 还是 v5(见 §9)。**不要**简单理解成"跳过 v4" —— v4 有两种形态,只有其中一种该跳过(§9.2)
6. 资源约束务必遵守:**GPU0 和 GPU5 上有用户的 holder,绝对不要动**;跑小任务用 GPU4/GPU1/GPU6

---

## 1. 实验目的

按 CANARD 探索计划 §2 的 v3 条目,在 **v1**(不是 v2,v2 已 abandoned)基础上加 hierarchical contrastive alignment loss:

```
v1 : L = L_CE
v3 : L = L_CE + Σ_l λ_l · L_align_l
```

**核心假设(来自 v2 的失败教训)**:语义信号的可靠性随层深衰减,因此 alignment 应**按层加权**(浅层高、深层低),而不是均匀施加。

依据是 v2_iter2 实测的 LLM-vs-GRAM 一致率:

| 层 | Beauty | Toys |
|---|---|---|
| L1 | 44.5% | 60.4% |
| L2 | 22.7% | 27.5% |
| L3 | 10.1% | 16.5% |
| L4+ | 3.5-5.8% | 6.4-8.2% |

完整假设文档见 `artifacts/phase13/explore/v3_hierarchical_align/hypothesis.md`。

---

## 2. 配置

| 项 | 值 |
|---|---|
| 数据集 | Beauty_cold50 / Toys_cold50 (η=50%) |
| MLP epochs | 200, lr 1e-3, batch 512, Adam |
| 新增架构 | trunk: Linear(384→512) + ReLU + LayerNorm;per-level heads;per-level projection(128d) |
| InfoNCE τ | 0.07 |
| 快筛种子 | 12345, 777, 2024 |
| 快筛 GPU | GPU4(空闲卡,Default 模式) |

**⚠️ 重要 confound**:v1/v2 是直接在冻结文本 embedding 上挂独立 linear head,**无共享隐层**。alignment 必须作用在共享表示上,所以我加了 trunk。这意味着"v3 vs v1"混了两个变量(trunk + alignment)。因此**必须跑架构对照组**(`--align-weights 0,0,...`)来分离二者 —— 已跑,见下。

---

## 3. 命令与产物路径

```bash
# 新增代码
experiment/phase13/protocol/semantic_bridge_v3.py        # v3 训练器(新)
experiment/phase13/run_v3_stage1_screen.sh               # A+D 快筛驱动(新)
experiment/phase13/run_v3_flat_screen.sh                 # flat alignment 快筛(新)

# 快筛产物
artifacts/phase13/explore/v3_screen/stage1/<dom>_<tag>_s<seed>/
artifacts/phase13/explore/v3_screen/stage1_driver.log    # A+D 驱动日志(已 ALL_DONE)
artifacts/phase13/explore/v3_screen/flat_driver.log      # flat 驱动日志(⚠️ 可能仍在跑)

# 单次训练示例
CUDA_VISIBLE_DEVICES=4 python experiment/phase13/protocol/semantic_bridge_v3.py train \
  --embeddings artifacts/phase13/embeddings/Toys_sbert.pt \
  --id-file GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
  --cold-items GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt \
  --output-dir <out> --align-weights 1.0,0.45,0.27,0.14,0.11 \
  --epochs 200 --device cuda:0 --seed 12345
```

`--pk-sampler` 开启同父分组采样;`--flat-align W` 开启 flat 模式(与 `--align-weights` 互斥)。

---

## 4. 核心数字(MLP val_avg_acc,warm 验证集)

**这不是推荐指标,是 MLP 预测 GRAM id 的准确率。见 §6 的重要注意事项。**

### Toys(v1 基线 = 0.4060)

| 配置 | n seeds | mean | sd | vs v1 | vs ctrl |
|---|---|---|---|---|---|
| ctrl(仅 trunk,无 alignment) | 3 | 0.4041 | 0.0034 | -0.5% | — |
| weighted(按层加权) | 3 | 0.4036 | 0.0047 | -0.6% | -0.14% |
| flat0.5 | 3 | 0.4019 | 0.0028 | -1.0% | -0.55% |
| flat1.0 | 3 | 0.3994 | 0.0032 | -1.6% | -1.16% |
| **weightedpk**(加权+同父采样) | 3 | 0.3991 | 0.0021 | **-1.7%** | **-1.25%** |
| uniform(均匀加权) | 1 ⚠️ | 0.3973 | — | -2.1% | -1.68% |

### Beauty(v1 基线 = 0.2630)

| 配置 | n seeds | mean | sd | vs v1 | vs ctrl |
|---|---|---|---|---|---|
| ctrl | 3 | 0.2620 | 0.0105 | -0.4% | — |
| flat0.5 | 3 | 0.2602 | 0.0103 | -1.1% | -0.69% |
| flat1.0 | 3 | 0.2588 | 0.0097 | -1.6% | -1.20% |
| weighted | 3 | 0.2577 | 0.0108 | -2.0% | -1.63% |
| weightedpk | 3 | 0.2572 | 0.0107 | -2.2% | -1.84% |
| uniform(单种子) | 1 ⚠️ | 0.2566 | — | -2.4% | -2.06% |

⚠️ = seed 数不足,不要据此下强结论(uniform 补种子方法见 §7.2)。

**全部 3-seed 配置的排序在两域完全一致**:ctrl > weighted > flat0.5 > flat1.0 > weightedpk。**没有任何 alignment 变体超过纯 trunk 对照组,也没有任何配置超过 v1。**

### 关键:种子噪声量级

| 域 | 典型种子噪声(sd/mean) | 观察到的最大效应 |
|---|---|---|
| Toys | **±1.0%** | -1.7% |
| Beauty | **±4.1%** | -2.2% |

**Beauty 单次结果在 0.2455~0.2696 之间跳动**。这意味着 Beauty 上任何 ±4% 以内的差异都无法与噪声区分。

---

## 5. 已被推翻的中间结论(避免接手者重犯)

我在只有**单个种子**时曾向用户汇报过两条结论,**加了种子后全部作废**:

1. ❌ 曾报"Beauty 上 trunk 带来 +1.3% 收益" → 三种子均值是 **-0.4%**,那个 +1.3% 只是恰好抽到好种子
2. ❌ 曾报"Toys 上按层加权带来 +0.5% 收益" → 三种子均值是 **-0.6%**

**教训:这个快筛的种子方差很大(Beauty ±4.1%),任何单种子结果都不可信。至少 3 个种子再解读。**

---

## 6. 目前站得住的结论

### 6.1 uniform 最差,双域一致 → "深层不该均匀对齐"成立

均匀加权在两域都是最差配置(-2.1% / -2.4%),明显劣于按层加权。这独立验证了从 v2 学到的洞察。**这条是有论文价值的 negative result 素材**(虽然 uniform 只有单种子,但幅度超出 Toys 噪声范围)。

### 6.2 修好深层采样后反而更差 → 核心假设被证伪

`weightedpk` 是唯一"真正让深层 alignment 生效"的配置。此前深层几乎空转:

| 层 | 随机 batch anchor 覆盖率 | PK 采样后 |
|---|---|---|
| L1 | 99.9% | (改为专用 batch) |
| L2 | 68.0% | 31.9% |
| L3 | 11.5% | 16.6% |
| L4 | **1.8%** | **5.7%** |
| L5 | **1.4%** | **4.6%** |

修好之后:Toys -1.7%、Beauty -2.2%,**两域一致变差**,且 `weightedpk` 的 **sd 是所有配置里最小的(Toys 0.0021)**,三个种子全部低于对照组。这不像噪声。

**解读**:深层 alignment 一旦真正生效,就是在损害模型。与 v2 的结论殊途同归 —— GRAM 深层簇由 SASRec 协同共现决定,不是语义决定的,硬拉即错。

### 6.3 flat alignment 与 GRAM 的 id 设计天然不兼容

smoke test 发现:Toys 训练集 5365 个商品分成 **5224 个不同的完整 id 路径,只有 102 个路径含 ≥2 个成员**。flat alignment 需要"同类正例对",但 id 空间几乎是一物一码,能配对的样本仅 **0.5%**,loss 基本空转。

**这不是实现 bug,是方案与 id 空间稀疏性的结构性冲突。**计划把 flat 列为降级选项时未预料到这一点。

**3-seed 实测结果**(已跑完):flat0.5 = -1.0%/-1.1%、flat1.0 = -1.6%/-1.6%(Toys/Beauty,vs v1)。我原本预期"与对照组打平"(因为 loss 几乎不起作用),**实测是小幅变差,且强度越大越差**(flat1.0 < flat0.5,双域一致)。

解释:0.5% 的样本确实提供了极少量梯度,但方向是错的 —— 与 §6.2 的结论一致,不管用什么方式把语义相似性拉进 GRAM 的 id 空间,都是有害的。强度加倍则伤害加倍,这个剂量-反应关系反而佐证了"伤害真实存在,不是噪声"。

### 6.4 汇总:排序在两域完全一致

五个 3-seed 配置的排序,Toys 与 Beauty **完全相同**:

```
ctrl > weighted > flat0.5 > flat1.0 > weightedpk
```

规律很清楚:**alignment 生效得越充分(剂量越大、深层覆盖越好),结果越差**。ctrl 完全不做 alignment 反而最好。这个单调关系跨两个独立数据集复现,是本次快筛最强的信号。

---

## 7. ⚠️ 未完成的工作(接手第一件事)

### 7.1 flat alignment 已跑完 —— 无需再等

`flat_driver.log` 已出现 `[flat] ALL_DONE`,12/12 全部完成,结果已并入 §4 与 §6.3。若要复查:

```bash
tail -3 /mnt/18T/jiangtangyunzhi/projects/recomm/artifacts/phase13/explore/v3_screen/flat_driver.log
tmux ls | grep v3_   # 应为空(会话已结束)
```

重新汇总所有快筛结果(会自动带上后续新增的 seed):

```bash
cd /mnt/18T/jiangtangyunzhi/projects/recomm/artifacts/phase13/explore/v3_screen/stage1
/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -c "
import json,glob,os,statistics as st,re
V1={'toys':0.4060,'beauty':0.2630}
rows={}
for d in sorted(glob.glob('*_s*')):
    p=f'{d}/training_history.json'
    if not os.path.exists(p): continue
    m=re.match(r'(toys|beauty)_(.+)_s(\d+)\$',d)
    if not m: continue
    dom,tag,_=m.groups()
    h=json.load(open(p))['history']
    rows.setdefault((dom,tag),[]).append(max(x['val_avg_acc'] for x in h))
for dom in ['toys','beauty']:
    print(f'--- {dom} (v1={V1[dom]:.4f}) ---')
    ctrl=st.mean(rows[(dom,'ctrl')])
    for (d,tag),v in sorted(rows.items()):
        if d!=dom: continue
        m=st.mean(v); sd=st.stdev(v) if len(v)>1 else 0
        print(f'  {tag:14} n={len(v)} mean={m:.4f} sd={sd:.4f}  vs_v1={100*(m-V1[dom])/V1[dom]:+.1f}%  vs_ctrl={100*(m-ctrl)/ctrl:+.2f}%')
"
```

### 7.2 `uniform` 配置只有单种子

`uniform`(均匀加权)是 §6.1"深层不该均匀对齐"这条论点的唯一证据,但只跑了 seed=12345。若要把它写进论文,**建议补到 3 种子**:

```bash
# 参照 experiment/phase13/run_v3_stage1_screen.sh,权重传全 1
--align-weights 1,1,1,1,1        # Toys(5 层)
--align-weights 1,1,1,1,1,1,1    # Beauty(7 层)
```

约 6 分钟(6 次训练)。注意用空闲卡,不要碰 GPU0/GPU5。

---

## 8. Gate 结论

**当前判定:Stage 1 快筛未通过,不启动 GRAM。**

`hypothesis.md` §4 自定的闸门:「如果 val_avg_acc 低于 v1,不要上 GRAM」。当前所有配置(除 seed 不足的 flat)均低于 v1 → 闸门触发。

这道闸门的设立理由:v2 的两次失败在这一步都能提前看出(0.3930 / 0.3846 均低于 v1 的 0.4060),但当时没设闸,白烧了两次约 25 小时的 GRAM 训练。

### ⚠️ 这道闸门的局限(接手者必须知道)

**val_acc 不是推荐指标,二者不等价**。v1 的 val_acc 也只有 0.2630/0.4060 这个量级,但它的 cold ndcg@10 相对 v0 涨了 **+133%/+186%**。

所以:
- 这道闸门的**负向筛选力有依据**(v2 两次失败都被它提前捕捉)
- 但它的**正向预测力从未被验证**
- 严格说,"val_acc 不涨"**不能 100% 断定 "cold 指标不涨"**

如果用户愿意承担约 40 小时 GPU 的成本去排除这个不确定性,直接上 GRAM 也是合理选择。**这是判断权衡,不是技术事实。**

---

## 9. 下一步动作

用户指示:**"先做选项 1,选项 1 不行再做 2+3"**。选项 1 已实现并在跑(见 §7.1)。

以下是对"选项 2+3"的**修正版**方案。撰写本报告的初版时我写的是"跳过 v4 直接做 v5",**那个建议不准确**,理由见 §9.2。

### 9.1 先做选项 3(分析 v1 为何成功)—— 零成本,且能决定后续方向

**这是我建议的第一步,优先于任何新组件。** 纯分析,不训练、不调 API、不占 GPU 档期。

**观察**:v1 只用最朴素的 CE 就拿到 cold ndcg@10 **+133%/+186%**,而所有"加料"都让它变差 —— v2 的 KL(-40~-48%)、v3 的三种 alignment(快筛 -0.4~-2.2%)。这个反差本身需要解释。

**待验证假设**:cold item 需要的可能不是"**更准的 id**",而是"**更容易被 beam search 捞出来的 id**"。v1 的 MLP 因为只优化 CE,倾向把 cold item 预测到**高频/大簇**里(交叉熵天然偏好高先验类别);而 v2/v3 的额外约束把它推向"语义上更对但更冷僻"的簇,反而更难被检索到。

**可执行的切入点**(全部是 CPU 分析):
1. 比较 v0/v1/v2_iter2 三者 cold id 落点的**簇频率分布** —— 统计每个 cold item 被分到的簇里有多少 warm item。若 v1 的落点显著偏向大簇,假设成立。
2. 交叉 cold item 的**命中与否 × 落点簇大小**,看命中是否集中在大簇。
3. 检查 v1 vs v2 cold id 的碰撞率差异(已知:Beauty 7.4%→7.6%、Toys 12.6%→14.0%,v2 略差)与命中率的关系。

**为什么这一步值得优先**:它能直接决定后面该做 v4-retriever 还是 v5(见 §9.3),而不是靠猜。

### 9.2 关于 v4:不应简单跳过,要区分两种形态

计划 §2 的 v4 是"multi-perspective + self-reflection",**其定位是替代 v2 的 KL target**(计划原文:"融合后的 refined prediction 作为 L_llm_prior 的 target")。

| 形态 | 是否受"语义↔协同空间错位"影响 | 评估 |
|---|---|---|
| **v4 原计划**(LLM 输出 → KL target) | **是** —— 与 v2 撞同一面墙 | 建议跳过。v2 已证明问题不是"LLM 不够准"(它比随机好 175-867 倍),而是"再准也对不上 GRAM 的协同聚类编号"。把单次调用升级成 6 次只是把方向错误的信号做得更精致,成本约 $45 |
| **v4-retriever 变体**(LLM 只做 few-shot retriever,完全不做 loss) | **否** —— 绕开了那面墙 | **值得做**。这就是 v2 诊断报告里的"方向 C"。LLM 只负责判断"这个 cold item 像哪些 warm item",然后**借用**那些 item 的 id,不需要 LLM 的输出与 GRAM 编号体系对齐 |

**⚠️ 本报告初版曾笼统建议"跳过 v4",那是判断偏窄** —— 只考虑了原计划形态,漏掉了 v4-retriever。接手者不应据此认为 v4 已被否决。

### 9.3 v4-retriever 与 v5 是同一思路的两半

v5(uncertainty-aware dual-path)的低置信度分支要走"**文本相似度检索 warm item**";而 v4-retriever 正好能提供更强的检索信号。两者**不必按计划顺序分两步做**,可以合并考虑。

**建议按 §9.1 的结论分流**:

- 若分析显示 v1 的收益来自"**猜对了 id**" → **v4-retriever 更有戏**(借用相似 warm item 的 id,本质是把"猜"换成"抄",精度更高)
- 若分析显示收益来自"**猜到了易被检索的 id**" → **v5 更对路**(关键在"如何让 cold item 进入候选集",而非 id 准不准)

### 9.4 v3 本身的收尾

若 flat 确认打平(§6.3 预期),则 v3 的三种变体(按层加权 / PK 修好采样 / flat)全部试过,按计划 §1.4 的三次上限可干净判 **v3 组件 abandoned**,并保留 §6.1/§6.2 作为消融素材。

**但注意 §8 的闸门局限** —— 若用户愿意承担约 40h GPU 成本排除"快筛正向预测力未验证"这一不确定性,用 `ctrl`(纯 trunk)或 `weighted` 配置上一次 GRAM 也是合理选择。

### 9.5 Plan Z 不触发

计划 §3 的触发条件是"v0 通过但 v1-v2 就挂"。v1 双域**强 PASS**,说明"text signal 能救 cold"的核心假设成立;倒掉的是 v2/v3 两个具体组件,不是方向。

---

## 10. 资源使用与 GPU 保护确认

### 🔒 用户的资源分配规则(硬约束,务必遵守)

用户明确指示:
- **GPU0**:用户的 ablation-scan holder(PID 1386094,占 32528 MiB)。原计划 Beauty 实验用此卡 —— 需先停 holder,跑实验时保证**持续占用 ≥25G**(GRAM 本身 peak 约 15.7G,不足部分用 lease sidecar 补),跑完继续 hold 住。**因未启动 GRAM,该卡未被触碰。**
- **GPU5**:用户的 holder(PID 1445225,占 17528 MiB)。**明确要求不要释放、不要动。** 全程未触碰。
- **GPU6**:Toys 实验预定卡,不涉及释放与占位,跑完即可。
- **小任务**:用空闲卡(GPU4/GPU1/GPU6),不要占用 GPU0/GPU5。
- **超过 10 分钟的任务必须放后台**(tmux),通过 status/日志观察,不要前台 sleep 等待(烧 token)。

### 本阶段实际用量

| 项 | 值 |
|---|---|
| GRAM 训练 | **0 次** |
| MLP 快筛 | 约 30 次 × 1-3 分钟,全部在 GPU4 |
| GPU0 / GPU5 | **未触碰,holder 均存活** |
| API | 无(v3 不需要 LLM) |

**注意 GPU3 不可用**:虽显示约 30G 空闲,但为 `Exclusive_Process` 模式,已被 manxin 独占。

---

## 11. 同期完成的其它工作(非 v3)

### 11.1 v2 复核(已完结)

补齐 v2_iter2 期间因 DeepSeek 余额耗尽而失败的 4813 次 API 调用(0 失败),重训 MLP 复核:完整覆盖下 Beauty **0.2505** / Toys **0.3889**,**均仍低于 v1**,且 Beauty 补齐后反而更差。**误判假设排除,v2 维持 abandoned。**
详见 `artifacts/phase13/explore/v2_verify/CONCLUSION.md` 与主报告 §5.5。

### 11.2 三个代码缺陷已修

1. `generate_llm_priors_v2iter2.py`:API 失败不再伪装成 `<unk>`+confidence 1.0,改记 `status:"failed"`/`confidence=0`;新增 `--max-fail-rate`(默认 2%)超阈值非零退出;consumer 侧区分"失败"与"真 OOV"
2. `run_phase13_explore.sh`:protector 恢复改为按实时 `memory.free` 减 CUDA context 开销(约 2.2G)自适应,空间不足则拒绝启动而非必然 OOM
3. 同上:`holding_post_training` 新增 `HOLD_TIMEOUT_HOURS`(默认 6h)超时自动释放(此前曾空占 GPU0 达 32 小时)
4. 附带修复:`read_item_set` 返回 set 导致 prompt 每次不同、LLM cache 永不命中 → 改为 `sorted()`

### 11.3 存储迁移(共享 /home 满)

`/home` 17T 用满(仅剩 100M),Claude transcript 写入失败(ENOSPC)。**非用户造成**:用户占 63G,`abdkhan` 占 2.5T。

已处置:`~/.claude`(795 文件/66M)复制到 `/mnt/18T/jiangtangyunzhi/.claude`,文件数一致;`CLAUDE_CONFIG_DIR` 写入 `~/.bashrc` **和** `~/.profile`(因 `.bashrc` 对非交互式 shell 直接 return);备份 `~/.bashrc.bak.20260814`、`~/.profile.bak.20260814`。

**⚠️ 旧目录 `~/.claude` 尚未删除**(保留回退能力)。新路径需重启 Claude Code 才生效。确认新位置正常后可清理旧目录。

**遗留风险**:conda 环境在 `/home/jiangtangyunzhi/miniconda3`,`/home` 满仍可能导致 pip 装包/临时文件写入失败,建议找管理员或让大占用者清理。

---

## 12. 关联文档

- 假设: `artifacts/phase13/explore/v3_hierarchical_align/hypothesis.md`
- v2 终局报告: `report/第十三阶段/GRAM_第十三阶段_v2_iter2_vocab-constrained-LLM-prior_双域gate-FAIL报告.md`
- v2 复核: `artifacts/phase13/explore/v2_verify/CONCLUSION.md`
- 计划: `plan/第十三阶段/GRAM_第十三阶段_CANARD探索计划v0.1.md`(§9 进度表尚未加 v3 行,待 flat 跑完后补)
- Memory: `/mnt/18T/jiangtangyunzhi/.claude/projects/-mnt-18T-jiangtangyunzhi/memory/`
