# v2_verify — 完整 KL 覆盖下的复核结论

**日期**: 2026-08-14
**目的**: v2_iter2 期间 DeepSeek 余额耗尽,warm 侧 KL 监督缺失(Beauty 47.5% / Toys 32.6%)。本次补齐全部失败调用后重训 MLP,检验 "v2 FAIL 是否为 API 缺陷导致的误判"。

## 结论:❌ 不是误判。v2 的 FAIL 结论成立,维持 abandoned。

## 做法

1. `repair_failed_priors.py` 只重跑失败记录(Beauty 2871、Toys 1942),成功记录原样保留;cold 侧未动(本就 0% 污染)
2. 重跑结果:**Beauty 2871 全部成功、Toys 1942 全部成功,0 失败**
3. 合并成 `{beauty,toys}_all_repaired.jsonl`(12101 / 11924 条,0 个 `<unk>`/null)
4. 用完全相同的超参重训 MLP(λ=0.2, 200 epoch, lr 1e-3, bs 512, seed 12345)

**v2_iter2 原始 artifacts 未被修改**(report 引用了它们),新产物全部在 `v2_verify/` 下。

## 结果:MLP best val_avg_acc

| 配置 | Beauty | Toys |
|---|---|---|
| **v1(无 KL)** | **0.2630** | **0.4060** |
| v2 iter1(λ=0.5, OOV→uniform) | — | 0.3930 |
| v2_iter2(λ=0.2, 覆盖缺失) | 0.2531 | 0.3846 |
| **v2 完整覆盖(λ=0.2, 0 失败)** | **0.2505** | **0.3889** |

**两个域在完整覆盖下依然显著低于 v1**:Beauty -4.8%、Toys -4.2%(相对 v1)。

## 关键观察

1. **Beauty 补齐后反而更差**(0.2531 → 0.2505)。补上的 47.5% KL 信号让结果**下降**,与"KL 有益但样本不够"完全相反 —— 直接证伪了误判假设。
2. **Toys 补齐后小幅回升**(0.3846 → 0.3889)但**仍低于 v1 的 0.4060**。即便把这 +0.0043 全部算作 KL 的正贡献,也填不平与 v1 的 -0.0171 差距。
3. **per-level 分解印证机制解释**:两域的 L1/L2(语义可靠层)在加 KL 后均下降(Beauty L1 0.831→0.786、Toys L2 0.680→0.606);Toys 的 L3/L4 反而略升(0.255→0.268、0.133→0.146),即深层的小幅改善被浅层的损失抵消有余。
4. **OOV 率的真实值得到确认**:修复后 loader 报告 Beauty L1=13.5%、L2=31.7%,与之前手工剔除 `<unk>` 的估算(13.6% / 32.0%)一致,证明 vocab-constrained prompt 确实生效,失败的是 KL 目标本身。

## 对 v2 判定的影响

无。原报告 §5.3 的四条论证中,第 3 条(MLP val_acc 随 KL 单调下降)现在有了**干净数据支撑**:不再需要"API 失败等价于降低 λ"的间接推理,完整覆盖下的直接测量得到同样结论。

**副产物**:现在有了一组干净的 λ=0.2 完整覆盖数据,论文 ablation 里可以直接引用,不必标注"覆盖率受损"。

## 成本

- API:4813 次调用(Beauty 2871 + Toys 1942),约 29M input tokens,估算 **$8-9**
- GPU:2 次 MLP 训练各约 3-5 分钟(GPU6,与他人进程共享),**未占用训练档期**
- 未重跑 GRAM

## 产物

```
artifacts/phase13/explore/v2_verify/
├── repair_{beauty,toys}.log
├── {beauty,toys}_warm_repaired.jsonl
├── {beauty,toys}_all_repaired.jsonl      # cold(原) + warm(修复)
├── mlp_{beauty,toys}_full.log
└── mlp_{beauty,toys}_full/{best.pt,training_history.json,vocab.json}
```
