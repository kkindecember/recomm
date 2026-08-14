# GRAM 第十三阶段 v2_iter2 实验报告
## Vocab-Constrained LLM Prior — 双域 Gate FAIL(v2 组件建议 abandoned)

**实验版本**: v2_beauty_iter2 / v2_toys_iter2
**数据集**: Beauty_cold50 / Toys_cold50 (η=50%)
**实验日期**: 2026-08-12 ~ 2026-08-14
**Gate 判定**: ❌ **FAIL(双域一致)** — cold ndcg@10 相对 v1:Beauty **-43.6%**、Toys **-39.8%**(阈值 +3%)
**核心结论**: iter2 的两项修复**技术上都生效了**,但 cold 依然系统性回退 40%。结合 iter1 的 -48%,**LLM prior 作为 KL 正则项这条路已被两次独立证伪**。
**⚠️ 本次实验存在一个未预期的执行缺陷**:DeepSeek API 余额耗尽,导致 warm 侧 KL 监督覆盖率大幅下降(Beauty 仅 47.5% 的 warm item 完全没有 KL 信号)。该缺陷**不改变 FAIL 结论** —— 已于 2026-08-14 补齐全部失败调用并重训 MLP 复核,**完整覆盖下 val_acc 依然低于 v1(Beauty 0.2505 / Toys 0.3889,v1 为 0.2630 / 0.4060),且 Beauty 补齐后反而更差**。详见 §5.5。

---

## 一、实验目的

按 CANARD 探索计划 Section 2 的 `v2_iter2` 条目,修复 iter1 诊断出的两个设计缺陷,验证"正确使用"LLM prior 能否带来 cold-start 增量:

1. **Prompt 改造(方向 D)**:给 LLM 提供每层合法 vocab 列表,目标把 OOV rate 从 61% 降到 <10%
2. **代码修复(方向 A)**:OOV 时返回 `None` + KL loss 里 mask 掉 OOV 层,替代 iter1 的 uniform 退化
3. **λ_llm 降低**:0.5 → 0.2

---

## 二、配置

| 配置项 | Beauty | Toys |
|--------|--------|------|
| 数据集 | Beauty_cold50 | Toys_cold50 |
| η (cold ratio) | 50% | 50% |
| GRAM seed | 2023 | 2023 |
| MLP seed | 12345 | 12345 |
| GRAM rec_epochs | 30 | 30 |
| MLP epochs | 200 | 200 |
| λ_llm | **0.2**(iter1: 0.5) | **0.2** |
| Hierarchical id 层数 | 7 (c128) | 5 (c32) |
| Per-level vocab | [108, 3823, 4894, 5062, 5026, 5008, 4776] | [30, 670, 4568, 5533, 5777] |
| LLM | DeepSeek-chat, 5-shot, temperature=0.7, max_tokens=200 | 同 |
| Hierarchical ID 文件 | `..._split_v2iter2_mlpcold_llmprior.txt` | 同构 |
| Backbone / beam | t5-small / 50 | t5-small / 50 |

**配置一致性已核对**:v0 / v1 / v2_iter2 在 seed、rec_epochs、rec_lr、rec_batch_size、batch_size、beam_size、max_his、item_prompt、backbone、gradient_accumulation_steps、length_penalty 上**完全一致**。唯一差异 `top_k_similar_item`(Beauty=10, Toys=5)是跨数据集差异,在各数据集内部 v0→v2 保持恒定,不影响版本间对比。

### 2.1 v1 → v2_iter2 的 loss 变化

```
v1        : L = L_CE(MLP, GRAM_ground_truth)
v2_iter1  : L = L_CE + 0.5 · KL(MLP ‖ LLM_prior)      # OOV → uniform(灾难)
v2_iter2  : L = L_CE + 0.2 · Σ(mask_l · KL_l)/Σ(mask_l)  # OOV → mask 掉
```

---

## 三、命令与产物路径

```bash
# Prep(LLM 调用 + MLP 训练 + cold id 赋值)
bash experiment/phase13/prep_v2iter2.sh beauty
bash experiment/phase13/prep_v2iter2.sh toys

# GRAM 训练(完整 protocol)
PROTECTOR_TOOL=codellama      bash experiment/phase13/run_phase13_explore.sh start v2_beauty_iter2 6
PROTECTOR_TOOL=ablation_scan  bash experiment/phase13/run_phase13_explore.sh start v2_toys_iter2   0

# Beauty test 侧手工补测(见 §5.1 缺陷 A)
#   ckpt: gram_logs/Beauty_cold50/3_20260812_2305/id_0_rec_30/model_rec_phase_1_epoch_30.pt
#   产物: v2_beauty_iter2/eval_from_ckpt/

# 补算 cold/warm 拆分(本次会话补做,见 §5.1 缺陷 B)
python experiment/phase13/protocol/eval_cold_warm.py \
  --dataset-dir GRAM/rec_datasets/Beauty_cold50 \
  --predictions-tsv artifacts/phase13/explore/v2_beauty_iter2/eval_from_ckpt/predictions/20260813_143016_Beauty_cold50_sequential_pred_test.tsv \
  --output-json artifacts/phase13/explore/v2_beauty_iter2/metrics_cold_warm.json \
  --version-tag v2_beauty_iter2 --split-name test
```

**产物目录**:
- `artifacts/phase13/explore/v2_beauty_iter2/`(含 `eval_from_ckpt/` 子目录)
- `artifacts/phase13/explore/v2_toys_iter2/`
- LLM cache: `artifacts/phase13/llm_cache.db`(674 MB)

---

## 四、核心数字

### 4.1 Cold subset(主 gate 指标)

**Beauty**(cold n=5234, warm n=5421, total=10655)

| 指标 | v0 | v1 | **v2_iter2** | v2 vs v1 | v2 vs v0 |
|------|-----|-----|--------------|----------|----------|
| **cold ndcg@10** | 0.001794 | 0.004179 | **0.002357** | **-43.6%** | +31.4% |
| cold hit@10 | 0.003057 | 0.008024 | **0.005159** | **-35.7%** | +68.8% |
| cold ndcg@50 | 0.003120 | 0.006642 | **0.004053** | **-39.0%** | +29.9% |
| cold hit@50 | 0.009553 | 0.019106 | **0.013183** | **-31.0%** | +38.0% |

**Toys**(cold n=4442, warm n=4347, total=8789)

| 指标 | v0 | v1 | **v2_iter2** | v2 vs v1 | v2 vs v0 |
|------|-----|-----|--------------|----------|----------|
| **cold ndcg@10** | 0.003048 | 0.008720 | **0.005254** | **-39.8%** | +72.4% |
| cold hit@10 | 0.006078 | 0.013507 | **0.007654** | **-43.3%** | +25.9% |
| cold ndcg@50 | 0.004318 | 0.011494 | **0.006261** | **-45.5%** | +45.0% |
| cold hit@50 | 0.011706 | 0.026114 | **0.012157** | **-53.4%** | +3.8% |

### 4.2 Warm subset(检查是否有代价转移)

| 指标 | 数据集 | v1 | v2_iter2 | Δ |
|------|--------|-----|----------|---|
| warm ndcg@10 | Beauty | 0.070782 | 0.070317 | **-0.7%** |
| warm hit@10 | Beauty | 0.119166 | 0.117875 | -1.1% |
| warm ndcg@10 | Toys | 0.053597 | 0.058289 | **+8.8%** |
| warm hit@10 | Toys | 0.087647 | 0.093858 | +7.1% |

### 4.3 Overall

| 指标 | 数据集 | v1 | v2_iter2 | Δ |
|------|--------|-----|----------|---|
| overall ndcg@10 | Beauty | 0.038065 | 0.036934 | -3.0% |
| overall ndcg@10 | Toys | 0.030916 | 0.031485 | +1.8% |

### 4.4 与 iter1 对比(Toys,唯一有三点数据的域)

| 版本 | cold ndcg@10 | vs v1 | λ_llm | OOV 处理 |
|------|--------------|-------|-------|----------|
| v1 | 0.008720 | — | — | — |
| v2_iter1 | 0.004530 | -48.0% | 0.5 | uniform(缺陷) |
| **v2_iter2** | **0.005254** | **-39.8%** | 0.2 | mask(已修复) |

**iter2 相对 iter1 回收了约 8 个百分点(-48% → -39.8%),但离 gate(+3%)差 43 个百分点。修复方向正确、幅度远不够。**

---

## 五、Gate 结论与实验完整性核查

### Gate 判定:❌ FAIL(双域一致)

| Gate 条件(计划 §2 v2_iter2) | 实测 | 判定 |
|---|---|---|
| ✅ 通过:cold NDCG@10 相对 v1 ≥ +3% | Beauty -43.6% / Toys -39.8% | 不满足 |
| ⚠️ 边缘:0~+3% 且 warm 不退化 | 不满足 | 不满足 |
| ❌ 失败:再次退化 → 跳到 v3,标记 v2 abandoned | **命中** | **FAIL** |

按计划条文,本次命中"❌ 失败"分支,处置为 **直接跳到 v3,v2 组件标记 abandoned**。

### 5.1 执行缺陷(必须记录)

**缺陷 A — Beauty 训练进程在 test 推理阶段被 OOM 打断**
`run.log` 有 2 处 `CUDA out of memory`(GPU0 上,分别在 13.22 GiB / 12.18 GiB already allocated 时申请 84 MiB / 336 MiB 失败),postflight 报 `no *_test.tsv found — training did not reach test inference`。训练本身 30 epoch 完整跑完(5 次 validation 全部落盘,epoch 5/10/15/20/25/30 checkpoint 齐全),仅 test 推理未执行。

**处置**:从 `model_rec_phase_1_epoch_30.pt` 手工补跑 test,产物在 `eval_from_ckpt/`。已核对补测 config 与训练 config 在 `hierarchical_id_type`、`item_id_type`、`seed`、`beam_size`、`max_his`、`top_k_similar_item`、`item_prompt`、`test_by_valid`、`valid_by_test`、`eval_batch_size`、`length_penalty`、`user_id_without_target_item`、`debug_test_100` 上**逐项一致**,预测行数 10655 = 全量用户数。**补测结果有效**。

**缺陷 B — Beauty 的 cold/warm 拆分此前缺失**
因缺陷 A 打断了 postflight,`metrics_cold_warm.json` 一直没生成。本次会话已用 `eval_cold_warm.py` 补算(rows: total=10655 warm=5421 cold=5234 missing=0)。

**缺陷 C — DeepSeek API 余额耗尽,warm 侧 KL 监督严重缺失(本次最严重问题)**
`prep.log` 中 Beauty 有 **2871** 次、Toys 有 **1942** 次 `DeepSeek API error 402: Insufficient Balance`。而 `generate_llm_priors_v2iter2.py:176-179` 的异常分支把失败样本写成**每层 `<unk>`、且 `confidence` 硬编码为 1.0**:

```python
except Exception as e:
    logger.error(f"API call failed for {target_id}: {e}")
    response_text = " | ".join(["<unk>"] * num_levels)   # ← 伪装成正常回答
```

后果:所有 12101 / 11924 条 prior 记录的 confidence 都是 1.0,**从 prior 文件本身完全看不出哪些是失败的**。失败样本全部落在 warm 段(cold 段先跑,0% 污染):

| | Beauty warm | Toys warm |
|---|---|---|
| warm item 总数 | 6049 | 5961 |
| 全 `<unk>`(API 失败) | **2871 (47.5%)** | **1942 (32.6%)** |
| 完全没有任何 KL 信号的 item | 2876 (47.5%) | 1942 (32.6%) |
| 逐层 KL 有效覆盖率 | L1=45.6% … L7=38.7% | L1=64.3% … L5=48.5% |

因为 `<unk>` 不在 vocab 里,OOV mask 机制**正确地把它们全部 mask 掉了**——所以这不会像 iter1 那样注入错误监督信号,但等价于把 λ_llm 在近半数 warm 样本上悄悄降到 0。**本次 Beauty 实际是一个"λ=0.2 但只作用于 52% warm 样本"的实验,不是干净的 λ=0.2 消融。**

### 5.2 两项修复是否真的生效(已验证:是)

**修复 1(OOV mask)已正确实现且生效**:`semantic_bridge_v2.py:85-104` 对 OOV 层 `mask=0`,`:241-249` 用 `loss_kl_sum / mask_sum.clamp(min=1.0)` 归一化,避免 loss 随 OOV 率缩放。iter1 的 uniform 退化路径已不存在。

**修复 2(vocab-constrained prompt)显著生效**。注意 prep.log 打印的 OOV 率(Beauty L2 48.1%)把 API 失败的 `<unk>` 也算进去了,是被污染的数字。**剔除 `<unk>` 后、只统计 LLM 真实回答的 OOV 率**:

| 层 | Beauty(iter1 基线 ~50-61%) | Toys |
|---|---|---|
| L1 | **13.6%** | **4.5%** |
| L2 | 32.0% | 13.0% |
| L3 | 32.1% | 23.1% |
| L4 | 30.2% | 26.1% |
| L5 | 27.0% | 28.0% |
| L6 | 26.0% | — |
| L7 | 27.8% | — |

L1 达成了 <10%~14% 的目标,深层仍有 26-32% 未达 <10% 目标,但相对 iter1 的 50-61% 是**实质性改善**。

### 5.3 为什么缺陷 C 不改变 FAIL 结论

四条独立理由:

1. **双域一致**。Toys 的 API 失败率(32.6%)明显低于 Beauty(47.5%),KL 覆盖率高出约 12-16 个百分点,但 cold 回退幅度几乎一样(-39.8% vs -43.6%)。若"KL 信号不足"是主因,应看到覆盖率更高的 Toys 明显更好——没有观察到。
2. **回退方向与"监督不足"预期相反**。被 mask 的样本退化为纯 L_CE,即 v1 的目标函数。如果 LLM prior 有正向价值,覆盖率降低应让 v2 **趋近 v1**(即回退变小),而不是稳定停在 -40%。实测在 λ 从 0.5 降到 0.2、且有效样本再砍半(等效 λ 进一步降低)之后,回退仅从 -48% 收窄到 -40%,**说明伤害不是来自 λ 的大小,而是来自 KL 目标本身的方向性错误**。
3. **MLP 内部指标直接显示 KL 项在损害 bridge**。这是最直接的证据:

| 版本 | best val_avg_acc | λ_llm | per-level(L1→) |
|---|---|---|---|
| v1_beauty | **0.2630** @ep121 | — | 0.831, 0.427, 0.202, 0.113, 0.098, 0.096, 0.075 |
| v2_beauty_iter2 | 0.2531 @ep166 | 0.2 | 0.785, 0.399, 0.199, 0.106, 0.108, 0.098, 0.078 |
| v1_toys | **0.4060** @ep166 | — | 0.871, 0.680, 0.255, 0.133, 0.092 |
| v2_toys(iter1) | 0.3930 @ep180 | 0.5 | 0.861, 0.616, 0.257, 0.144, 0.087 |
| v2_toys_iter2 | 0.3846 @ep196 | 0.2 | 0.854, 0.604, 0.258, 0.136, 0.070 |

加了 KL 项后 val_acc **单调下降**(Toys: 0.4060 → 0.3930 → 0.3846),且**降幅集中在 L1/L2**(Toys L2: 0.680 → 0.604),正是 LLM 与 GRAM 分歧最大的浅层。注意 λ 从 0.5 降到 0.2 时 val_acc **继续下降而非回升**,这与"λ 调小就能修好"的假设矛盾。

4. **LLM 与 GRAM 的 id 空间存在根本性错位**。在 warm item 上比较 LLM 在词表内的预测与 GRAM 真值:

| 层 | Beauty 一致率 | vs random | Toys 一致率 | vs random |
|---|---|---|---|---|
| L1 | 44.5% | 48x | 60.4% | 18x |
| L2 | 22.7% | 867x | 27.5% | 184x |
| L3 | 10.1% | 496x | 16.5% | 753x |
| L4 | 5.8% | 292x | 8.2% | 455x |
| L5 | 5.4% | 273x | 6.4% | 370x |
| L6 | 3.5% | 175x | — | — |
| L7 | 4.2% | 200x | — | — |

**这张表是本次实验最重要的产出**。LLM 远强于随机(175-867 倍),证明**它确实有真实语义能力**——这复现了 iter1 诊断报告的核心观察。但绝对一致率在 L3 以下跌到 3.5-16%,意味着 **KL 项在 85-96% 的深层样本上把 MLP 往错误的 cluster 拉**。GRAM 的 hierarchical id 来自 SASRec 协同过滤空间的聚类,LLM 推的是语义/类目空间——两者在浅层(粗类目)部分重合,在深层(细粒度 cluster)几乎无关。**把 LLM 预测当作 KL hard target 是方向性错误,不是权重调参问题。**

### 5.4 其它已核对项(无异常)

- **checkpoint 选择口径一致**:v0/v1/v2 全部使用 epoch 30(`automatic_last_checkpoint_test`),无 cherry-pick。Beauty 手工补测用的也是 epoch 30。
- **NaN**:两域 run.log 均 0 处。
- **cold item 覆盖**:Beauty 6052/6052、Toys 5963/5963 全部由 MLP 赋值,`fallback to source id = 0`。
- **cold id 碰撞率**:Beauty v1 7.4% → v2 7.6%;Toys v1 12.6% → v2 14.0%。轻微恶化,但不足以解释 -40%。Beauty 最大桶从 11 涨到 29 值得留意。
- **cold id 变动幅度**:Beauty 90.9%、Toys 84.3% 的 cold item 相对 v1 换了 id — 确认 v2 确实是一次实质性的重新赋值,不是噪声扰动。
- **Toys 的 warm 提升(+8.8%)**:因 cold/warm 共享同一个 GRAM 模型与 Trie,cold id 分布变化会改变 beam search 竞争格局,warm 的小幅波动属预期噪声,不构成"v2 对 warm 有益"的证据(Beauty 上是 -0.7%,方向相反)。

### 5.5 完整 KL 覆盖下的复核(2026-08-14 补做,决定性证据)

为排除"缺陷 C 导致误判 v2"的可能,补齐了全部失败的 API 调用后重训 MLP。

**做法**:`repair_failed_priors.py` 只重跑失败记录(Beauty 2871、Toys 1942),成功记录原样保留,cold 侧未动;**重跑 0 失败**。合并后 12101 / 11924 条 prior 中已无任何 `<unk>`/null。用完全相同的超参(λ=0.2, 200 epoch, lr 1e-3, bs 512, seed 12345)重训。v2_iter2 原始 artifacts 未被修改,新产物在 `artifacts/phase13/explore/v2_verify/`。

**结果**:

| 配置 | Beauty | Toys |
|---|---|---|
| **v1(无 KL)** | **0.2630** | **0.4060** |
| v2 iter1(λ=0.5, OOV→uniform) | — | 0.3930 |
| v2_iter2(λ=0.2, 覆盖缺失) | 0.2531 | 0.3846 |
| **v2 完整覆盖(λ=0.2, 0 失败)** | **0.2505** | **0.3889** |

**两域在完整覆盖下依然显著低于 v1**(Beauty -4.8%、Toys -4.2%)。关键点:

1. **Beauty 补齐后反而更差**(0.2531 → 0.2505)。补上 47.5% 的 KL 信号让结果**下降**,与"KL 有益但样本不够"完全相反 —— 直接证伪误判假设。
2. **Toys 补齐后小幅回升**(0.3846 → 0.3889)但仍低于 v1。即便把 +0.0043 全算作 KL 正贡献,也填不平与 v1 的 -0.0171 差距。
3. **per-level 印证机制解释**:两域 L1/L2(语义可靠层)加 KL 后均下降(Beauty L1 0.831→0.786、Toys L2 0.680→0.606);Toys L3/L4 反而略升(0.255→0.268、0.133→0.146)。浅层损失大于深层收益。
4. **OOV 真实值获确认**:修复后 loader 报告 Beauty L1=13.5%、L2=31.7%,与 §5.2 手工估算(13.6% / 32.0%)一致 —— vocab-constrained prompt 确实生效,失败的是 KL 目标本身。

**影响**:§5.3 第 3 条论证现在有干净数据支撑,不再依赖"API 失败等价于降低 λ"的间接推理。同时产出了一组可直接用于论文 ablation 的完整覆盖 λ=0.2 数据。

**成本**:4813 次 API 调用(约 29M input tokens,估算 $8-9)+ 2 次 MLP 训练(各 3-5 分钟,未占训练档期)。未重跑 GRAM。

详见 `artifacts/phase13/explore/v2_verify/CONCLUSION.md`。

---

## 六、资源使用

| 项 | Beauty | Toys |
|---|---|---|
| GPU | GPU6(CodeLlama 保护) | GPU0(ablation-scan 保护) |
| 训练起止 | 08-12 23:05 → 08-13 14:17 | 08-13 01:14 → 08-14 03:00 |
| 训练墙钟 | ~15.2 h | 89129 s ≈ 24.8 h |
| test 推理 | 36553 s ≈ 10.2 h(手工补测,3.36 s/样本) | 3615 s ≈ 1.0 h(0.39 s/样本) |
| peak_allocated | 7222 MiB(test)/ 15754 MiB 量级(训练) | 15754 MiB(训练) |
| peak_reserved | 30312 MiB(test) | 17564 MiB(训练) |
| GPU 显存峰值(telemetry,含他人进程) | 48558 MiB | 48499 MiB |
| API | DeepSeek-chat,2871 次调用失败(余额耗尽) | 同,1942 次失败 |
| LLM cache | `artifacts/phase13/llm_cache.db` 674 MB | 共用 |

**Beauty test 推理耗时异常**:3.36 s/样本 vs Toys 的 0.39 s/样本(8.6 倍)。原因是补测时 GPU6 上有其他用户进程(99% util、84°C)与之争抢。不影响指标正确性,仅影响墙钟。

### GPU 保护恢复确认(计划 §1.5 必查项)

✅ 已确认,但**过程中出现过保护失效窗口**:

- **Beauty / GPU6**:训练结束后 lease 已 released。test 补测结束(08-14 00:39)时 watcher 尝试抓 12000 MiB,`watcher.log` 报 `worker did not report ready within 15s`(实为 CUDA init 慢的误报,holder 实际起来了)。本次会话已按用户要求停掉 GPU6 holder,GPU6 现由他人进程占用 30914 MiB。
- **Toys / GPU0**:训练结束后 runner 进入 `holding_post_training` 并**空转了 32 小时**(lease PID 2353555 存活 1-08:17)。且 `watcher_hold_gpu0.log` 显示 ablation_scan protector 启动时 **CUDA OOM**(申请 22.04 GiB 时仅 21.78 GiB 可用),watchdog 亦报 `GPU0 free: 783 MiB, too little free memory, cannot hold` — **该时段 GPU0 的 30G 保护实际是失败的**,lease sidecar 只抓到 8000 MiB(配置本应 25000)。
- **本次会话已处置**:GPU0 lease(PID 2353555)与 worker(2353409)已 SIGTERM 干净退出;GPU0 重启 holder `RESERVE_MIB=30500`(实占 32528 MiB,PID 1386094);GPU5 新起 holder `RESERVE_MIB=15500`(实占 17528 MiB,PID 1445225)。

**教训**:`holding_post_training` 的无限 `while true; do sleep 60; done` 在无人值守时会长时间空占卡;且 protector 恢复用固定 `RESERVE_MIB` 在碎片化的卡上必 OOM,应按实时 free 值自适应。

---

## 七、下一步动作

**判定:v2 组件 abandoned,进入 v3。**

依据:计划 §2 的 v2_iter2 gate "❌ 失败:再次退化 → 直接跳到 v3(标记 v2 组件 abandoned)"。已消耗 iter1 + iter2 两次;计划 §1.4 允许 3 次,但**不建议再做 iter3**,理由是 §5.3 的第 3、4 条已经把失败定位到"KL hard target 方向性错误"这一机制层面,而 iter3 的三个候选选项都无法绕开它:

- 「λ_llm → 0.1」:λ 从 0.5→0.2 时 val_acc 继续下降,再降只是趋近 v1,天花板就是 v1 本身
- 「换 GPT-4o mini」:失败点不是 LLM 能力(已证明 175-867x 优于随机),而是 LLM 语义空间与 GRAM 协同聚类空间的错位
- 「方向 C:LLM 只做 few-shot retriever,不做 loss」:这已经不是 v2(KL 正则),本质是新组件,更适合放到 v3 之后或 Plan Z-C 里评估

**建议顺序**:

1. **先补齐 v2 的收尾动作**(计划 §6.6,当前缺失):
   - 写 `metrics_summary.json`(v1/v2 目录下均无,全阶段都没生成过)
   - 写 `v2_*_iter2/decision.md`
   - 更新计划 §9 进度表(v2_beauty/v2_toys iter_2 两行仍是"🔄 待启动")
2. **修 3 个代码缺陷**(否则 v3 会踩同样的坑):
   - `generate_llm_priors_v2iter2.py:176-179`:API 失败必须记 `confidence=0` / `status:"failed"`,不能伪装成 `<unk>` + confidence 1.0;并在结尾打印失败率,失败率超阈值直接非零退出
   - **充值 DeepSeek 或确认 v3 不需要 API**(v3 是纯 loss 改动,应该不需要——若如此则此项不阻塞)
   - `run_phase13_explore.sh`:protector 恢复用实时 free 值而非固定 `RESERVE_MIB`;`holding_post_training` 加超时自动释放
3. **启动 v3(Hierarchical Contrastive Alignment Loss)**,基线回到 **v1**(不是 v2_iter2)。v3 的 gate 应改为"相对 v1 提升 ≥3%"——因为 v2 已 abandoned,计划原文的"相对 v2 提升"失去意义。
4. **v3 的一个设计提示**:§5.3 第 4 条的一致率表明,**浅层(L1/L2)的语义信号是可靠的(44-60% 一致率),深层不可靠**。v3 的 hierarchical alignment 如果按层加权(浅层权重高、深层权重低甚至为 0),可能正好避开 v2 踩的坑。这条观察值得写进 v3 的 `hypothesis.md`。

**Plan Z 判定**:**暂不启动**。计划 §3 的触发条件是"v0 通过但 v1-v2 就挂"。当前 v1 双域都是强 PASS(Beauty +133%、Toys +186% cold ndcg@10),说明"text signal 能救 cold"的核心假设**成立**;挂掉的只是 v2 这一个组件。方向本身健康。

---

## 八、论文价值(negative result 素材)

本次 + iter1 构成一个**完整且干净的 negative result 链条**,消融价值高:

1. **iter1**:朴素加 LLM prior → -48%,根因是 OOV→uniform 的实现缺陷
2. **iter2**:修好 OOV(mask)+ 降 λ + vocab-constrained prompt → 仍 -40%
3. **机制解释**:LLM 语义空间与 GRAM 的 SASRec 协同聚类空间在浅层部分对齐(L1 44-60%)、深层几乎无关(L3+ 仅 3.5-16%),因此 KL hard target 在多数样本上提供错误梯度
4. **双域复现**:Beauty 与 Toys 独立复现同一结论

可写成 discussion 里的一个明确论点:**"LLM 的语义先验不能直接作为协同过滤导出的 hierarchical id 的分布目标"**,并用一致率随层数衰减的曲线作为证据图。这比单纯报"没提升"强得多。

---

## 附录:关键路径

```
artifacts/phase13/explore/v2_beauty_iter2/
├── metrics_cold_warm.json                    # 本次会话补算
├── metrics_cold_warm_val_epoch25.json        # validation(非 gate 依据)
├── run.log                                   # 含 2 处 OOM
├── prep.log                                  # 含 2871 次 API 402
├── llm_priors_{cold,warm,all}.jsonl
├── mlp/{best.pt,training_history.json,vocab.json}
├── gram_logs/Beauty_cold50/3_20260812_2305/id_0_rec_30/model_rec_phase_1_epoch_30.pt
└── eval_from_ckpt/                           # 手工补测 test
    ├── test.log
    └── predictions/20260813_143016_..._pred_test.tsv

artifacts/phase13/explore/v2_toys_iter2/
├── metrics_cold_warm.json                    # runner postflight 正常生成
├── run.log / prep.log
├── llm_priors_{cold,warm,all}.jsonl
├── mlp/{best.pt,training_history.json,vocab.json}
└── predictions/20260814_020013_..._pred_test.tsv
```

**报告撰写日期**: 2026-08-14
