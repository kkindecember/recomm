# GRAM 第一阶段环境配置记录

日期：2026-07-17

## 结论

官方 GRAM 代码已放在 `/home/jiangtangyunzhi/projects/recomm/GRAM`，独立 Conda 环境 `gram-repro` 已按官方核心版本建立。Python 包依赖一致性检查及 GRAM 训练入口静态导入均已通过。

当前尚不能进入 GPU smoke test：主机的 NVIDIA 驱动未加载，`nvidia-smi` 无法通信。按实验计划，本次没有选择 GPU、加载模型权重或启动任何 GPU 任务。

## 实际版本

| 组件 | 版本 |
|---|---|
| GRAM commit | `7ac4d9272a57beed9df35c27ea34221f6e4a8fb1`（浅克隆） |
| Python | 3.9.25 |
| PyTorch | 1.11.0+cu113 |
| torchvision | 0.12.0+cu113 |
| torchaudio | 0.11.0+cu113 |
| PyTorch CUDA build | 11.3 |
| Transformers | 4.26.0 |
| NumPy | 1.23.1 |

官方 `requirements.txt` 不包含可执行的 PyTorch 依赖行，因此没有覆盖锁定版本。未固定的传递依赖按 2026-07-17 可解析版本安装，完整结果见 `pip_freeze.txt` 和 `conda_environment.yml`。

## 配置与修改记录

| 文件或状态 | 修改原因 | 修改内容 | 可能影响实验数值 |
|---|---|---|---|
| 外层 `.gitignore` | 防止模型缓存进入版本控制 | 忽略 `.cache/` | 否 |
| Conda 环境变量 | 默认 Hugging Face 缓存目录不可写 | 将 `HF_HOME` 与 `TRANSFORMERS_CACHE` 固定到当前项目 `.cache/huggingface/` | 否 |
| `gram-repro` 环境（2026-07-18） | 首次 smoke test 通过主机 SOCKS 代理访问 Hugging Face 时，`requests` 报错缺少 SOCKS 支持 | 增加 `PySocks==1.7.1`，并同步依赖快照 | 否；仅提供 HTTP 客户端的代理传输支持 |
| GRAM 官方仓库 | 提供复现代码 | 仅浅克隆，官方仓库工作区无修改 | 否 |

`code_changes.patch` 为空，表示 GRAM 官方仓库目前没有代码修改。

## 验证

执行了：

```bash
conda run -n gram-repro python -m pip check
cd /home/jiangtangyunzhi/projects/recomm/GRAM/src
conda run -n gram-repro python -c "import torch, transformers, main_generative_gram"
```

结果：`pip check` 无损坏依赖，核心模块导入成功。没有下载 `t5-small` 或额外的生成模型，因为这属于后续联网/GPU smoke test 准备，并会继续消耗有限磁盘空间。

## 已知阻塞与风险

1. `nvidia-smi` 报告无法与 NVIDIA 驱动通信，所以尚未验证 `torch.cuda.is_available()`、GPU 型号和显存。
2. 环境建立后工作区所在文件系统仅剩约 9.8 GiB。正式训练前需要预留模型缓存、checkpoint、预测文件和日志空间。
3. 用户级 pip 缓存已有较大占用，但没有擅自清理，因为其中可能包含其他项目使用的文件。

## 使用方式

```bash
conda activate gram-repro
cd /home/jiangtangyunzhi/projects/recomm/GRAM
```

精确环境快照位于 `artifacts/phase1_beauty/environment/conda_environment.yml`。由于快照包含本机绝对缓存路径，在另一台机器重建时应更新 `HF_HOME` 与 `TRANSFORMERS_CACHE`。

更便于跨机器重建的方式是：

```bash
conda create -n gram-repro python=3.9 -y
conda run -n gram-repro python -m pip install -r /home/jiangtangyunzhi/projects/recomm/artifacts/phase1_beauty/environment/requirements_lock.txt
```

`requirements_lock.txt` 已包含 PyTorch CUDA 11.3 官方 wheel 源。
