# Phase 13: Cold-Start Item for Hierarchical-ID Generative Recommendation

**创建日期**:2026-08-07
**状态**:planning
**目标会议**:RecSys 2026(CCF-B,主),CIKM 2026 / WSDM 2027(备选)
**预计总工期**:约 4 个月(6-8 周 exploratory + 8-10 周 publication)

---

## 快速导航

| 文档 | 定位 | 何时读 |
|---|---|---|
| **`./GRAM_第十三阶段_CANARD探索计划v0.1.md`** | **验证 CANARD 方向是否 work,失败就迭代改进**。MVP-style,单数据集单 seed,快速 iteration。含 Plan Z fallback。 | **现在读**(HI-GRAM 收尾后启动) |
| **`./GRAM_第十三阶段_CANARD主线设计v0.1.md`** | 完稿级 plan,假设 exploratory 已验证。全矩阵实验 + ablation + 论文。 | Exploratory 通过后回来读 |

**规范**:
- Plan 文档统一放 `plan/第N阶段/`(和之前 12 个 phase 一致)
- Report 每次 iteration 后写在 `report/第N阶段/`
- 代码在 `experiment/phase13/`,产物在 `artifacts/phase13/`

---

## 两 plan 的关系

```
┌──────────────────────────────────────────────────────────┐
│  Phase 12 (HI-GRAM) 收尾                                  │
│      │                                                    │
│      ▼                                                    │
│  ┌────────────────────────────────────────────────────┐   │
│  │  PLAN_EXPLORATORY.md (6-8 周)                       │   │
│  │  v0 → v1 → v2 → v3 → v4 → v5                        │   │
│  │  每步单数据集单 seed,gate 验证边际贡献             │   │
│  │  失败 3 次 → 该组件砍掉                              │   │
│  └────────────────────────────────────────────────────┘   │
│      │                                                    │
│      ▼                                                    │
│   累积 cold NDCG 提升?                                    │
│      │                                                    │
│      ├─ ≥ 20% → PLAN_PUBLICATION.md (8-10 周)              │
│      │           完整 CANARD 全矩阵                        │
│      │                                                    │
│      ├─ 15-20% → PLAN_PUBLICATION.md (简化版)              │
│      │           砍掉未通过组件,写更短的论文              │
│      │                                                    │
│      ├─ 5-15% → RecSys LBR / short paper (4-6 周)         │
│      │                                                    │
│      └─ v1-v2 挂 → Plan Z fallback                        │
│                    (换方向,不是换 setting)                │
└──────────────────────────────────────────────────────────┘
```

---

## 为什么分两份

**经验教训**:Phase 12 (HI-GRAM) 只写了完稿级 plan,没准备 exploratory 阶段。HI-GRAM 的 NaN bug + 效果不达预期,一挂就 4 周没法快速切换。

**Explore first**:
- 先用最低成本验证核心假设("text-based signal 能救 GRAM cold-start")
- 每步只加一个组件,失败原因隔离,好 debug
- 累积的 v0-v5 序列天然形成 ablation
- 早期止损:v1-v2 挂了就换 fallback,不硬 push

**Publication second**:
- 只有验证过的方向才做全矩阵实验(3 datasets × 3 cold ratios × seeds × ablation)
- 避免"完稿级 plan 里的组件在小规模都没验证,大规模跑挂"

---

## 阅读顺序

**第一次读**:
1. 本文档(约 3 分钟)
2. `./GRAM_第十三阶段_CANARD探索计划v0.1.md`(约 15 分钟)
3. `./GRAM_第十三阶段_CANARD主线设计v0.1.md`(可略读 Section 0/1/6/9/11,约 10 分钟)

**续接时**:
1. 本文档 Section "当前状态"(下方)
2. 最新 report `ls -lt ../../report/第十三阶段/*.md | head -3`
3. 探索计划里 Section 9 进度追踪表
4. `artifacts/phase13/explore/<current_v>/iter_*/status.json` 看当前进度

---

## 当前状态

| 阶段 | 状态 | 备注 |
|---|---|---|
| Phase 12 HI-GRAM | 运行中 | beauty_v1 (GPU6) + toys_v1_light (GPU0),等 8h 后 val 结果 |
| Phase 13 Exploratory | 待启动 | HI-GRAM 收尾后正式启动 |
| Phase 13 Publication | 待启动 | Exploratory 通过后启动 |

**HI-GRAM 收尾决策点(2026-08-08 凌晨或上午)**:
- 若 Beauty epoch 15 val 追平或超过 baseline → HI-GRAM 继续跑完,phase13 等待
- 若 Beauty epoch 15 val 明显退化 → HI-GRAM 停止,立即启动 Phase 13 Exploratory

---

## 目录结构

```
plan/第十三阶段/                            # ← plan 文档 + 本 README
├── README.md                              # 本文档
├── GRAM_第十三阶段_CANARD探索计划v0.1.md
└── GRAM_第十三阶段_CANARD主线设计v0.1.md

report/第十三阶段/                          # ← 每次 iteration 后必写 report(硬规则)
├── GRAM_第十三阶段_v0_vanilla-baseline_cold-setting验证报告.md
├── GRAM_第十三阶段_v1_iter1_MinimumSemanticBridge结果报告.md
├── GRAM_第十三阶段_v1_iter2_bge-encoder换用结果报告.md
└── ...

experiment/phase13/                        # ← 代码和运行时脚本
├── run_phase13_explore.sh                 # 完整 protocol runner(CodeLlama + 30G lease)
├── protocol/                              # 数据 split、评测
├── explore/                               # 探索阶段代码(v0-v5)
├── method/                                # 完稿阶段代码(v5 通过后填充)
├── baselines/                             # 完稿阶段 baseline
└── tests/                                 # 单元测试(preflight 必过)

artifacts/phase13/                         # ← 实验产物
├── explore/
│   ├── v0_vanilla_baseline/
│   │   ├── run.log
│   │   ├── status.json
│   │   ├── metrics_summary.json
│   │   └── gpu_lease.json
│   ├── v1_minimum_bridge/
│   │   ├── iter_1/
│   │   ├── iter_2/
│   │   └── final/
│   └── ...
└── (完稿阶段目录 exploratory 通过后创建)
```

**四个位置各司其职,不要混**:plan / report / 代码 / 产物。

---

## 关键提醒

### 探索模式(当前)
- **允许方向微调,允许砍组件**
- 每一版失败都是 negative result,是消融素材
- 别把探索阶段的 pilot 当作最终数字发论文

### Publication 模式(未来)
- **进入后不再方向微调,只做实验矩阵和写作**
- 探索通过的 vN 决定 method 章节的组件
- 严格 seed / dataset / ablation coverage

### 实验协议(硬规则,不可绕过)
- **探索和完稿阶段都必须用完整 protocol**(继承 phase12 `run_phase12_hi_gram.sh`)
  - 前置占位者(CodeLlama 或 ablation-scan holder,见下)前后让位(实验前 stop,实验后 start)
  - 30G GPU lease sidecar
  - Runner 全程监督 + status.json
  - Exit trap 保证占位者恢复
- **原因**:服务器资源紧张,一旦掉 lease 24-48h 排不回,直接卡死进度
- **不用简化 protocol**(那是给完全独立、无资源竞争的开发环境用的)

### GPU 占位者(两条路径,任选)

服务器资源紧张,任何主训练 GPU 都必须在**空档期**由一个占位者持续占 ~30G,防止被其他人抢走。目前有两种占位工具,选哪个由**该 GPU 是否常驻有 CodeLlama 使用需求**决定:

| 工具 | 位置 | 真实用途 | 副产品 | 何时用 |
|---|---|---|---|---|
| **CodeLlama 保留** | `tools/run_codellama.sh` | 常驻 CodeLlama 服务 + 顺便占位 | 真的能被 codellama tool 调用 | GPU6 上默认,因为项目里其他任务需要 CodeLlama |
| **ablation-scan holder** | `tools/gram_ablation_scan.sh` | 纯占位,伪装成 hyperparameter scan(每 10s 一次 matmul 让 util > 0) | 无真实计算 | 别的卡上占位,不需要 CodeLlama 服务时 |

两者都提供相同的三命令接口:`start <gpu>` / `status` / `stop`。runner 的 preflight/postflight 必须适配当前 GPU 用的是哪种占位者(status.json 里记下 `resource_reservation` 字段,例如 `codellama_expected_on_gpu6` 或 `ablation_scan_expected_on_gpu0`)。

**并行两卡实验的典型分配**:
- GPU6:主线实验(CodeLlama 让位 → 训练 → CodeLlama 恢复)
- GPU0:副线实验(ablation-scan holder 让位 → 训练 → holder 恢复)

**注意**:ablation-scan holder 是**伪装占位**,不产生任何 artifact。工具的 worker/sh 顶部注释明写了这点,避免半年后自己误认为是真实验。真实验来了必须先 `stop` 再启训练。

### Report 强制规则
- **每次 iteration 完成后必须写 report** 到 `report/第十三阶段/`
- 命名:`GRAM_第十三阶段_v<N>_iter<M>_<描述>报告.md`
- 参考 phase9 / phase11 的 report 格式
- 未写 report 视为该 iteration 无效,不能进下一步

### GPU 保护应急
- 每次 iteration 后必查 `nvidia-smi -i <保护卡>`,确认占位者已重新占位(CodeLlama on GPU6,或 ablation-scan on 副线卡)
- 若失效:立即用对应工具重启(`tools/run_codellama.sh start <gpu>` 或 `tools/gram_ablation_scan.sh start <gpu>`)
- 若排不到 30G,先协调再启动新训练

### 与 memory 的关系
- 本文档 + 探索计划 + 主线设计是**权威 plan**
- Memory `project_current_run.md` 只记录**当前进行到哪一步**,不重复 plan 内容
- Memory `feedback_experiment_protocol.md` 继承(CodeLlama + 30G lease + no auto retry 规则)
- Memory `feedback_experiment_mode.md` 继承(当前是探索模式)
- Memory `user_constraints.md` 继承(服务器资源紧张)

---

## 关联

- `../第十二阶段/` — HI-GRAM plan(格式参考)
- `../../experiment/phase12/run_phase12_hi_gram.sh` — **完整 protocol runner 模板**(必须参照)
- `../../artifacts/phase{1..12}/` — P0-P12 diagnostic 素材(见主线设计 Section 4.3)
- `../../report/第九阶段/` `../../report/第十一阶段/` — report 命名和格式参考
- Memory `MEMORY.md` — 用户约束、实验协议、当前状态
