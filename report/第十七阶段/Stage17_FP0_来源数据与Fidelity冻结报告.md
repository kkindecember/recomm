# Stage17 FP0 来源、数据与 Fidelity 冻结报告

## Material Passport

- Step：`S17-FP0`
- Attempt：`attempt_001`
- Status：`PASS_S17_FP0_SOURCE_DATA_FIDELITY_FREEZE`
- Generated：2026-08-31T03:17:49.824186+00:00
- Scope：正常场景 GRAM；未读取 D1/D2、official test 或 Sports；未启动 GPU 任务

## 1. 冻结结论

LATTE 固定到 commit `05e4e6d983225bcb7172f148a076890e80c524d1`，许可证为 MIT。SETRec 固定到 commit
`2ed9a75ad1ad3784c61bba3c68cbedbe3cfce2d7`；仓库没有标准 LICENSE 文件，因此后续实现强制 clean-room，
不复制其源码。

现有 S17-2R LATTE/SETRec-style 结果继续作为方向选择证据，但两个实现均明确标记为
`not Full`，不能直接复用为 FP1/FP3 的正式效果结果。

## 2. 配置审计中的关键差异

- LATTE 基础 YAML 的 `eval_interval=1`，官方 quick-start wrapper 会覆盖为 3；
  S17 native matched protocol 采用 config-primary 的每 epoch 评估，并记录该差异。
- Native-PSID 强制使用与 LATTE 相同的 `rqkmeans`，避免把 PSID 默认 OPQ 差异混入因果比较。
- SETRec 官方 Toys T5 脚本为 30 epochs、全局 batch 512、4-GPU torchrun、FP16、
  seed 42、history 50，并在 validation 上搜索 beta。
- 论文要求 user-history sparse visibility；固定 public T5 commit 仅让同一 item
  的 token 共享 position id，没有显式屏蔽它们的 encoder visibility。因此后续拆成
  repo-parity 和 paper-faithful 两个 arm，不再混称一个 `Native-Full`。
- S17 clean-room protocol 使用 seed 2023、history 20 和 train-prefix internal dev；
  保留有效 batch 512、30 epochs、五个连续 token、独立 query 与 full-catalog grounding。

## 3. 数据冻结

- Toys D0 full users：12833
- rolling train examples：56421
- internal-dev users：1283
- item catalog：11924
- external D0 target：未 materialize；只允许 family checkpoint 冻结后读取一次

Native LATTE adapter 的追加审计已通过：完整 train prefix catalog 为 11182；严格移除
internal-dev 用户末位后，RQ/PCA 参数拟合 mask 为 11138。模型仍可为 11924 个冻结 catalog
item 做 target-independent transform 与 identifier assignment，但 external target identity 未被读取。

## 4. Gate 与下一步

FP0 来源、配置、许可和数据边界通过。Fidelity matrix 中的实现组件仍为
`IMPLEMENTATION_PENDING`。full-data adapter、LATTE PSID/forest/aggregation foundation，
以及 SETRec continuous AE、paper sparse mask、repo position parity、independent query、
full-catalog grounding foundation 已通过 CPU 合约；下一门是 tokenizer 与模型集成。

追加的基础设施任务均不产生效果结论。`attempt_001` 的后台 worker 因仓库根未加入
`PYTHONPATH` 在导入期失败，现已封存失败证据；修复后的 `attempt_002` 成功进入 worker，
但 `uv` 从 GitHub 下载 CPython 3.12.12 时连续三次发生 TLS EOF，原生环境终止，固定
revision `fc5d4628481afbbaaacd7af6bb07cf9d3865f781` 的 SentenceT5 cache 与 tokenizer
bounded profile 随依赖失败退出。三项真实终态分别通过
`artifacts/phase17/status/s17_fp0_native_env_setup.status.json`、
`artifacts/phase17/status/s17_fp0_sentence_t5_cache.status.json` 和
`artifacts/phase17/status/s17_fp0_tokenizer_bounded_profile.status.json` 汇报。

服务器已发现并校验可复用的精确版本 uv-managed Python 3.12.12；研究者明确授权后，
`attempt_003` 显式复用该解释器且原生环境 gate 已通过。随后 SentenceT5 的 Hugging Face
Python 下载在 revision API 处因 TLS EOF 失败，未下载任何模型文件，tokenizer profile
随依赖退出。研究者再次明确授权的 `attempt_004` 只恢复 T5 与 profile：T5 使用固定
revision 的 curl 逐文件传输、最多 5 次传输层重连、断点续传，但 tmux 未继承代理，5 次
均在首文件以 TLS code 35 失败且 0 字节落盘。研究者继续明确授权的 `attempt_005` 只为
T5 worker 显式注入经独立 tmux 验证的本机无凭据代理；启动核验时前 9 个文件均首试成功
并产生非零字节，随后继续后台下载，完成后执行离线加载与逐文件 SHA256 inventory；
tokenizer profile 同步等待。启动核验时两项均为 `gpu_ids=[]`；没有启动效果实验，没有
读取受保护数据，也没有使用 GPU/GPU1。后续 profile 仍只允许自动选择真正空闲的非
GPU1 卡，无安全卡即阻塞。
