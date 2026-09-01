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
T5 worker 显式注入经独立 tmux 验证的本机无凭据代理。固定 revision 的 13 个文件共
223,815,215 bytes 已完成缓存、逐文件 inventory 与离线加载，终态为
`PASS_S17_FP0_SENTENCE_T5_CACHE_READY`；同步 tokenizer profile 因没有满足准入条件的
非 GPU1 空卡而 `BLOCKED_WAITING_IDLE_NON_GPU1_GPU`，没有执行模型计算。

研究者随后明确允许 tokenizer `attempt_006` 直接共享 GPU1 剩余显存，但要求原重复轮
不停止、不暂停、不接管。准入时原 PID `2602227` 仍在，GPU1 空闲约 30.8 GiB，超过预注册
的 14,336 MiB 门槛。profile 随后在 `torch.cuda.set_device(0)` 处立即失败：官方 LATTE
`pyproject.toml` 没有固定 PyTorch，2026-08-31 的冻结解析得到 `torch 2.13.0+cu130`，而
服务器驱动只支持 CUDA 12.6。该失败发生在 SentenceT5 加载和 encode 之前，不是 OOM，
也不是科学结果；结束核验显示 PID `2602227` 仍存在，状态记录
`gpu1_repeat_preserved=true`。没有启动 full-data tokenizer 或效果实验，也没有读取 D1、
D2、official test 或 Sports。下一步必须先用新的冻结环境显式固定 CUDA 兼容 PyTorch，
增加 GPU 初始化 smoke；不得把已完成的 T5 下载或 attempt_006 profile 自动重跑。

研究者确认 CUDA 兼容修复后，官方 CUDA 12.6 索引中的 Python 3.12
`torch 2.7.1+cu126` wheel 已固定到 SHA256
`63bce0590bc540fc16139e2be0177847585182b8c5e68d7f9213789d1d96c978`。
环境 `attempt_001` 只创建了空 venv，随后因 runner 使用了 status schema 未允许的
execution-state 名称而退出；torch 安装、CUDA smoke 和 profile 均未开始，GPU1 未被触碰。
该状态已修正并封存为 `S17_FP0_CUDA_COMPAT_ENV_RUNNER_SCHEMA_FAILED`。修复后的环境
`attempt_002` 与 tokenizer `attempt_007` 均已通过 175 项 Stage17 tests 并冻结快照；经
研究者再次确认后，前者从官方 PyTorch 索引下载 CUDA 12.6 依赖，但约 19 分钟后因
`nvidia-nccl-cu12` 触发 uv 30 秒网络超时而失败，后者只等待依赖并随之退出，二者均未
使用 GPU。测速显示官方代理约 0.15 MB/s，阿里云 PyTorch 镜像直连约 8.2 MB/s；研究者
确认切换后，环境 `attempt_003` 使用阿里云 `pytorch-wheels/cu126` 直连、300 秒 timeout
与同一官方 torch SHA256，完整安装、114 包依赖检查和 CPU 离线导入均已成功。但 runner
错误地让极小 CUDA smoke 复用 tokenizer profile 的 14,336 MiB 准入门槛，因此环境被写为
`BLOCKED_GPU1_SHARED_HEADROOM_INSUFFICIENT`；tokenizer `attempt_008` 又把依赖 `BLOCKED`
错误映射为 `FAILED`。两者均未执行 SentenceT5 profile，这不是环境安装失败或科学失败。

研究者确认恢复后，环境 `attempt_004` 只复用现有 6 GiB 环境，不联网、不下载、不安装。
175 项 Stage17 tests、依赖检查、CPU 离线导入和 GPU1 CUDA smoke 全部通过；smoke 的
`peak_reserved_mib=2.0`，确认 `torch 2.7.1+cu126`、CUDA 12.6 和 NVIDIA RTX A6000 可用。
smoke 开始前 GPU1 已有 PID `2790130/3862550`，结束核验二者均保留，状态为
`PASS_S17_FP0_CUDA_COMPAT_ENV_READY`。tokenizer `attempt_009` 在独立后台会话等待 GPU1，
未启动 512-item profile；研究者明确要求改用 GPU0 后，该等待 worker 被精确停止并封存为
`S17_FP0_TOKENIZER_PROFILE_STOPPED_FOR_GPU0_SWITCH`，没有 summary、GPU 计算或科学结果。

新的不可变 GPU0 `attempt_010` 通过 8 项定向测试和全部 176 项 Stage17 tests。启动前 GPU0
空闲 28,859 MiB，超过 14,336 MiB 门槛；worker 未停止或修改 GPU0 现有进程。profile 在
物理 GPU0 正常完成：512 条输入得到 `[512,768]` float32 finite embeddings，模型加载
1.450 秒、编码 5.798 秒、吞吐 88.30 items/s，峰值 allocated/reserved 显存分别为
932.95/984 MiB；开始前记录的 GPU0 PID 在结束核验时均存在。encoding-only 线性外推 Toys
全目录约 135.0 秒，不包含 PCA/RQ、冲突消解或启动开销。终态为
`PASS_S17_FP0_TOKENIZER_BOUNDED_PROFILE`。当前仍没有 full-data tokenizer、效果实验或
受保护数据读取；下一门是正式 full-data tokenizer/FP1+FP2 多卡资源申请。
