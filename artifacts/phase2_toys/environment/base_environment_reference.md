# Phase 2 Toys 环境复用记录

日期：2026-07-20

第二阶段复用第一阶段 `gram-repro` Conda 环境，不重建、不升级依赖。

| 组件 | 版本 |
|---|---|
| Python | 3.9.25 |
| PyTorch | 1.11.0+cu113 |
| Transformers | 4.26.0 |
| NumPy | 1.23.1 |
| GRAM commit | `7ac4d9272a57beed9df35c27ea34221f6e4a8fb1` |

阶段 A 复查结果：

- `python -m pip check`：`No broken requirements found`。
- 当前环境的有效解释器为 `/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9`；管理员调整文件后，原 `bin/python` 链接不存在，第二阶段 runner 已固定使用前者。
- `t5-small`：项目 Hugging Face 缓存离线加载成功。
- `nandakishormpai/t5-small-machine-articles-tag-generation`：项目缓存离线加载成功。
- 缓存占用约 929 MiB，位于 `/home/jiangtangyunzhi/projects/recomm/.cache/huggingface`。
- 本次检查未初始化 CUDA 训练、未选择 GPU、未释放 CodeLlama。

完整环境快照继续以第一阶段产物为准：

```text
artifacts/phase1_beauty/environment/conda_environment.yml
artifacts/phase1_beauty/environment/pip_freeze.txt
artifacts/phase1_beauty/environment/requirements_lock.txt
```

阶段 A 没有安装或删除任何 Python 包。
