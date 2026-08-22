# Stage15 S15-0 源码与协议冻结报告

> 日期：2026-08-21
> 阶段：S15-0 Source、artifact 与协议冻结
> 总体结论：`PASS_S15_0_CONTINUE_WITH_GENRECEDIT_NATIVE_PREFLIGHT`
> GPU：未使用
> 模型权重：未下载
> 项目 test：未读取

---

## 1. 阶段问题与结论

本阶段回答的不是“哪个方法效果更好”，而是：SpecGR 与 GenRecEdit 的官方源码、artifact、依赖和评测入口是否足以进入可审计的原生复现，以及哪些内容不能直接搬到 GRAM。

结论如下：

| 对象 | 冻结 commit | S15-0 状态 | 裁决 |
|---|---|---|---|
| SpecGR | `f0ded8884b1df97b5f0599d4ec300bb20b5d1eff` | `SOURCE_READY_ARTIFACT_BLOCKED` | 源码存在，但 README 声称提供的 Video Games 数据、semantic IDs、模型和 item embedding 均不在 HEAD 或现有 Git 历史中；不能启动官方 inference |
| GenRecEdit | `e6878d9c7c6e57479e840ccb8c045b11a2bd69b5` | `READY_OFFICIAL_NATIVE_RUN` | 算法源码与官方 LFS 指针存在；只需按最小域选择性物化 artifact，并构建缺失 processed cache；无需把全部模型拉下来 |
| ColdGenRec | 未获得 | `SOURCE_OR_LICENSE_BLOCKED` | 只作协议参照；论文的 4open.science 入口当前转到需要授权的 API，猜测的 GitHub 镜像不存在；不阻塞两个目标基线 |

因此，S15-1A SpecGR 在不训练新模型、不使用非官方镜像的约束下正式记为 artifact blocked；下一步进入 S15-1B GenRecEdit 的**选择性 artifact 与环境 preflight**，尚不启动 GPU。

---

## 2. 冻结输入与产物

官方源码仅保存在被 Git 忽略的 `.runtime/phase15_sources/`，未修改第三方 worktree，也不提交其内容。

机器可读冻结产物：

- `artifacts/phase15/s0_source_freeze/source_manifest.json`
- `artifacts/phase15/s0_source_freeze/artifact_inventory.json`
- `artifacts/phase15/s0_source_freeze/dependency_matrix.json`
- `artifacts/phase15/s0_source_freeze/compatibility_matrix.json`

生成入口：

- `experiment/phase15/configs/stage15_s0_sources.json`
- `experiment/phase15/protocol/source_compatibility_audit.py`

审计脚本是标准库、network-free 工具：只读本地 clone、Git 元数据与 LFS pointer，不执行 fetch、checkout、install、LFS pull，也不 import 第三方项目。

---

## 3. SpecGR 审计

### 3.1 源码与 artifact

- remote：`https://github.com/Jamesding000/SpecGR.git`
- branch：`main`
- commit 时间：`2026-01-15T17:20:08+08:00`
- worktree：clean
- submodule：无
- LFS pointer：0
- license：HEAD 中未发现 LICENSE/COPYING/NOTICE

官方 quick-start 对 Video Games 至少需要：

```text
dataset/Video_Games/*.csv
semantic_ids/Video_Games.semantic_id
results/SpecGR/Video_Games_best_ft.pt
results/SpecGR/Video_Games_best_emb_ft.pt
```

以上四类输入均缺失。Git 历史中也没有这些路径的提交记录。当前不能通过“再拉一次模型”解决，因为官方仓库没有可选择物化的权重指针或 release tag。

### 3.2 原生入口风险

1. `quick_start.sh` 默认 `--eval_mode test` 且请求 `[0,1,2,3]` 四张 GPU。
2. `SpecGR/run.py` 虽解析 `args.eval_mode`，最终调用却固定为 `eval_mode='test'`。
3. checkpoint 使用 `map_location='cuda:0'`，不能直接遵守本项目的空闲卡选择规则。
4. 训练入口会顺序执行 validation 与 test；不能连接 GRAM 的封存 test。

S15-1A 若未来获得官方 artifact，只允许先写 validation-only、device-safe 的路径 wrapper，并把该修改标为入口修复，不能改算法。

### 3.3 GRAM port 边界

SpecGR 原生 semantic ID 通过 `num_layers + 1` 固定宽度 reshape；GRAM 是 variable-length hierarchical lexical path。以下内容必须在 S15-2 重新定义，而不能直接复用：

- drafter item ID 到 GRAM path catalog 的严格映射；
- guided re-drafting 的 prefix/EOS 语义；
- verifier acceptance 与 score normalization；
- duplicate/collision/unknown item hard-fail；
- 单卡 candidate/model-forward budget。

---

## 4. GenRecEdit 审计

### 4.1 LFS artifact 体量

- remote：`https://github.com/Starrylay/GenRecEdit.git`
- branch：`main`
- commit 时间：`2026-04-30T03:32:07+08:00`
- worktree：clean
- submodule：无
- license：HEAD 中未发现 LICENSE/COPYING/NOTICE
- 全仓库未物化 LFS：`542,824,815` bytes，约 `517.68 MiB`

Video Games 最小 native edit/eval 所需的已知 LFS 对象：

| artifact | bytes | MiB |
|---|---:|---:|
| TIGER checkpoint | 38,364,788 | 36.59 |
| covariance requests | 41,245,906 | 39.34 |
| augmented-10 edit requests | 32,698,016 | 31.18 |
| 合计 | 112,308,710 | 107.11 |

因此不应下载全部三个域、全部 `3/5/10` edit request。S15-1B 只考虑 Video Games 上述三个对象。

processed cache `data/cache/AmazonReviews2023/Video_Games/processed/*` 不在 Git/LFS 中。它至少涉及原始 Amazon Reviews 2023 数据、`sentence-t5-base` sentence embedding 与 semantic ID cache；在确定下载量、构建时间和显存前不能启动。

### 4.2 原生入口风险

1. `Scripts/rec_train.sh` 调用仓库中不存在的 `main.py`；`rec_main.py` 是可见的对应入口。只能把修正记录为 entry bug fix。
2. `model_bundle.py` 直接 `model.eval().cuda()`，需由 wrapper 固定所选 GPU。
3. 官方 cold split 从 native test 派生；只能用于 `native_official` sanity，不能与 GRAM validation 数字混表。
4. compact evaluator 使用 `top_position=3` 的 SID prefix match，不等价于项目 strict item-level evaluator。
5. `pos2layer=[0,1,2,3]`、`range(256)` 和固定 SID positions 是原生 TIGER 假设。

### 4.3 GRAM port 边界

S15-2 必须显式解决：

- variable-length `position_map` 与未映射深度；
- EOS/padding 是否编辑；
- 256-token codebook probe 如何替换为每层 lexical vocabulary；
- deltaW 的 layer/shape 对齐；
- trigger、未编辑 prompt parity 和 warm preservation；
- 最终 SID prefix metric 到 strict catalog item metric 的切换。

在上述 contract 通过前，任何简化 edit 都不得命名为 GenRecEdit-GRAM。

---

## 5. 依赖与许可裁决

现有 `gram-repro` 为 Python 3.9.25、Torch 1.11.0+cu113、Transformers 4.26.0；缺少两个官方仓库要求的 Lightning、datasets、sentence-transformers、pandas 和 faiss 栈。

主要冲突：

| 栈 | SpecGR | GenRecEdit | gram-repro |
|---|---|---|---|
| torch | 2.2.0 | 2.6.0+cu124 | 1.11.0+cu113 |
| transformers | 4.38.1 | 4.57.0 | 4.26.0 |
| numpy | 1.26.4 | 1.26.4 | 1.23.1 |
| scipy | 1.9.3 | 1.14.1 | 1.13.1 |
| sentence-transformers | 2.4.0 | 5.2.0 | missing |

裁决：创建 `phase15-specgr-native` 与 `phase15-genrecedit-native` 两个隔离环境；禁止升级或污染 `gram-repro`。S15-2 adapter 回到 `gram-repro`，只依赖本地 contract，不 import 官方环境。

两个目标仓库均没有 license 文件。可以保留 commit-pinned 本地副本做研究审计，但不把第三方实现代码复制进本项目或再分发；GRAM adapter 应依据论文机制和本地接口 clean-room 实现。

---

## 6. Protocol freeze

`official_native` 与 `gram_port` 的边界冻结如下：

| 项 | official_native | gram_port |
|---|---|---|
| backbone | 官方 TIGER/RQ-VAE | Phase14 冻结 GRAM v0 |
| split | 官方 repository split，单列 | Toys_cold50 validation；Beauty 条件式 |
| evaluator | 官方 evaluator，标 native | strict item-level，collision hard-fail |
| test | 可读取官方仓库自身协议数据，但不得混成 GRAM 结果 | GRAM test 全程封存，`test_read=false` |
| 目的 | 验证代码闭环 | 比较 intervention location |

官方论文数字、原生运行数字与 GRAM port 数字不得直接作同协议优劣结论。

---

## 7. Gate 与下一步

S15-0 完成定义已满足：两个目标方法均得到 source/artifact/compatibility 状态，机器可读证据与单一阶段报告齐全。

下一步只做 S15-1B GenRecEdit preflight：

1. 计算选择性 LFS、Amazon 数据与 HuggingFace 模型的完整下载/磁盘需求；
2. 创建隔离环境方案并做 CPU import/entry check；
3. 冻结 Video Games 最小 native 命令、预计时长、GPU 显存和后台 `status.json` 合约；
4. 再向用户报告是否需要下载、需要多少空间和显存；未确认前不启动 GPU。

SpecGR 保持 blocked，除非官方渠道提供与当前 commit 可核对的 checkpoint/data；不使用来源不明镜像，也不把从零重训冒充 checkpoint 复现。

---

## 8. 试错摘要

- SpecGR 与 GenRecEdit 使用 `GIT_LFS_SKIP_SMUDGE=1` 成功拉取；没有误下载权重。
- 计划中猜测的 ColdGenRec GitHub 地址返回 repository not found；论文给出的 4open.science 入口重定向后返回 HTTP 401。终止探测并记录 blocked，没有循环重试。
- 查看 JSON 时发现系统未安装 `jq`；改用 Python 标准库校验，没有安装额外工具。

这些均属于 S15-0 工程审计过程，按约定合并在本报告中，不单独创建 retry report。
