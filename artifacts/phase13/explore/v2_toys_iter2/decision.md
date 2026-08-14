# decision — v2_toys_iter2

**Gate 结论**: ❌ **FAIL**

- cold ndcg@10 = 0.005254,相对 v1(0.008720)**-39.8%**,阈值 +3%
- 相对 iter1(0.004530,-48.0%)回收约 8 个百分点 —— 修复方向正确,幅度远不够
- warm ndcg@10 +8.8%(cold id 重分布改变 beam search 竞争格局,属噪声;Beauty 上为 -0.7%,方向相反)
- 相对 v0 仍 +72.4%

**下一步动作**: v2 组件标记 **abandoned**,跳到 v3(Hierarchical Contrastive Alignment Loss),基线回到 **v1**。不做 iter3。

**关联 report**: `report/第十三阶段/GRAM_第十三阶段_v2_iter2_vocab-constrained-LLM-prior_双域gate-FAIL报告.md`

**本次执行缺陷**:
1. DeepSeek API 余额耗尽 → 1942 次调用失败(32.6% warm item 无 KL 信号)
2. 训练结束后 runner 在 `holding_post_training` 空转 32 小时;且该时段 GPU0 的 30G 保护实际失败(protector 启动 CUDA OOM,lease 仅抓到 8000/25000 MiB)

**日期**: 2026-08-14
