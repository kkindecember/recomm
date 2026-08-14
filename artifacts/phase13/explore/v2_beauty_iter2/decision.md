# decision — v2_beauty_iter2

**Gate 结论**: ❌ **FAIL**

- cold ndcg@10 = 0.002357,相对 v1(0.004179)**-43.6%**,阈值 +3%
- warm ndcg@10 -0.7%(无代价转移)
- 相对 v0 仍 +31.4%,即保留了 v0→v1 收益的约一半

**下一步动作**: v2 组件标记 **abandoned**,跳到 v3(Hierarchical Contrastive Alignment Loss),基线回到 **v1**。不做 iter3 —— 失败已定位到机制层(LLM 语义空间与 GRAM 协同聚类空间在深层错位,L3+ 一致率仅 3.5-10%),iter3 的三个候选选项都绕不开。

**关联 report**: `report/第十三阶段/GRAM_第十三阶段_v2_iter2_vocab-constrained-LLM-prior_双域gate-FAIL报告.md`

**本次执行缺陷(影响结果解释)**:
1. DeepSeek API 余额耗尽 → 2871 次调用失败(47.5% warm item 无 KL 信号);失败样本被写成 `<unk>` + confidence 1.0,从 prior 文件看不出来
2. 训练在 test 推理阶段 OOM → 从 epoch-30 ckpt 手工补测(config 已逐项核对一致,结果有效)
3. cold/warm 拆分于 2026-08-14 补算

**日期**: 2026-08-14
