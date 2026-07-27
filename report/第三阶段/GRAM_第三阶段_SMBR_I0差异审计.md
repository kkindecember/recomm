# GRAM 第三阶段：SMBR I0-N 借鉴边界与差异审计

## 固定结论

决策为 **`TRANSFER_INNOVATION_ALLOWED_I0_D_DESIGN_UNLOCKED`**。

已有工作做过某个组件，不是停止条件。SMBR 可以明确借鉴：

1. adaptive/fixed-budget context compression；
2. decision-aware benefit 或 negative-transfer scoring；
3. learning-to-defer 的 abstention/rejector。

可保留的原创增量是把这些组件重构成一个 GRAM 特定的学习问题：

> 用 training-only 成对反事实标签，学习何时恢复被 collaborative prefix 挤出的
> metadata；推理时只看 target-free 特征，不确定时严格退回 current layout。

## 已有工作覆盖了什么

- [RECOMP](https://arxiv.org/abs/2310.04408) 已覆盖 task-oriented compression 和
  selective augmentation。
- [ACC-RAG](https://aclanthology.org/2025.findings-emnlp.1307/) 与
  [SARA](https://aclanthology.org/2026.acl-long.661/) 已覆盖自适应/固定预算 context
  compression。
- [Decision-Aware Memory Cards](https://arxiv.org/abs/2606.08151) 已覆盖
  decision-uplift、necessity 和 negative-transfer-aware context selection。
- [Learning to Defer](https://proceedings.mlr.press/v119/mozannar20b.html) 已覆盖
  classifier/rejector 的一般形式。
- [AdaptRec](https://arxiv.org/abs/2504.08786) 已覆盖推荐中的自适应相似用户选择。

这些都可以引用、复现或改造，但不能改名后声称首次提出。

## “加一点修改”需要达到什么程度

小改动可以发论文，但必须是**有因果动机、可独立验证的小改动**，而不是表面变化。
SMBR 至少要新增四件事：

1. **新问题定义**：从一般压缩改为 collaborative evidence 与 displaced metadata 的
   固定预算净收益决策；
2. **新监督构造**：只从 training prefixes 产生 paired counterfactual benefit label，
   严格隔离 fit/calibration/audit 用户；
3. **新安全约束**：低置信度分支必须与 current serialization 完全相同，而非另一种
   压缩结果；
4. **新机制证据**：证明收益来自恢复有用 metadata 且控制 broad harm，不是更多 token、
   位置变化或删掉 collaborative evidence。

## 必做 baseline

- borrowed generic selector/compressor；
- fixed recovery；
- 单一 lost-token threshold；
- matched random activation；
- oracle upper bound（只作诊断，不能作部署结果）；
- current GRAM identity baseline。

如果 SMBR 只比 current 好，却没有胜过这些简单或借鉴基线，新增部分就站不住。

## 下一步边界

I0-N 现已允许进入 **I0-D 预注册设计**，但尚不允许直接调参或读 test。I0-D 先回答
training-only benefit 是否可学习；若不可学习，就停止 SMBR，而不是继续堆模块。

逐项矩阵见 `artifacts/phase3/smbr_i0/novelty_matrix.csv`，结构化结论见
`artifacts/phase3/smbr_i0/claim_evidence.json`。
