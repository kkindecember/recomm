# GRAM 第三阶段：CGI E0-N 差异审计报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-24
- Verification Status: ANALYZED
- Version Label: `cgi_e0_novelty_audit_v1`
- Design Status: PREREGISTERED E0-N EXECUTED BEFORE E0-D

## 结论

固定决策为 **`NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_NARROWING`**，只解锁 plan
第 18.3–18.4 节的冻结 checkpoint E0-D。

保留下来的研究问题非常具体：GRAM 中与 coarse lexical history 对齐的正确 item text
passages，是否仍会降低正确 lexical target path 的得分；这种负贡献是否集中在旧
passage，并与 tail miss@50 状态相关。

## 已被覆盖的宽泛贡献

- [CFT](https://arxiv.org/abs/2410.22809) 已用反事实推理强调行为序列对 LLM 推荐
  输出的作用，因此不能声称首次做 behavior counterfactual。
- [MHL](https://arxiv.org/abs/2509.23649) 已用 entropy-guided history masking 和
  reconstruction 改善生成推荐，因此不能声称首次 mask history。
- [RAGONITE](https://arxiv.org/abs/2412.10571) 已用删除 evidence 后的输出变化做
  counterfactual attribution。
- [RFiD](https://aclanthology.org/2023.findings-acl.155/) 与
  [MGFiD](https://aclanthology.org/2024.findings-naacl.142/) 已研究 FiD 的 spurious
  evidence、multi-granularity guidance 和 passage pruning。
- [MGR-LF++](https://arxiv.org/abs/2503.23333) 已研究生成推荐对 modality 与 late
  fusion 选择的敏感性。
- 最接近的是 [LWGR](https://arxiv.org/abs/2605.18771)：它明确指出不受控的 LLM
  world-knowledge fusion 会与行为信号冲突，并用 Lagrangian constraint 选择性融合。
  因此 CGI 不能声称首次发现 semantic knowledge 会伤害推荐或首次 selective fusion。

## 三项必要 gate

| Gate | 结果 | 剩余差异 |
|---|---|---|
| mechanism specificity | PASS | 尚未发现 frozen GRAM full/coarse-only 的正确 target-path 配对审计 |
| structured attribution | PASS | 尚未发现累计 fine、oldest/newest 与 tail miss/hit 的联合分解 |
| intervention room | PASS WITH NARROWING | training-only passage-contribution supervision 的 target-free GRAM gate 尚未发现，但通用 FiD pruning 与有益知识选择已存在 |

未来若存活，贡献必须是“推荐特异结构化机制证据 + target-free passage gate”，不能把
counterfactual deletion、masking、adaptive fusion 或 pruning 单独写成创新。

## 执行边界

E0-N 在加载 checkpoint 或计算 E0-D 分数前完成。未读取 test、未训练、未使用 GPU。
下一步只能原样运行四个预注册条件；任一双数据集必要门槛失败即停止 CGI。
