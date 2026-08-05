# GRAM 第八阶段：TIPA-P0 首次启动失败记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-03
- Verification Status: `FAILED_CLOSED_BEFORE_ADAPTER_TRAINING`
- Version Label: `phase8_tipa_p0_first_start_failure_v1`

## 结果

TIPA-P0 首次启动完成 CPU 测试 5/5、teacher/GRAM lineage 锁定和 Toys 256/256 fit samples
的 prefix 构造，随后因可用的非 null、多 child prefix records 少于预注册下限 192 而
退出。adapter 未训练，Beauty 未启动，external development、Sports 和 test 均未读取。

原因是 prefix 深度在所有 path depths 中随机选择，许多深层 prefix 只剩一个合法 child，
无法构成 child-level alignment 训练样本。这是启动前未覆盖到的采样完整性错误，不是
TIPA 效果失败或 CUDA 错误。

## 产物 SHA-256

- run log: `be78681f25f33ec156a3c5bea9b3a1dbad4a931f5ecefbc1c530e515a49f4af6`
- status: `af03f80622fe7235645012d43c58abbf8f00626fd5dbfef5034eb1436f49dcf4`
- telemetry: `f3c6f736e5c54176ef5428d621a196311d89fc37327e504a973fc029b4780e24`
- GPU lease: `339f15d7f43f9931710e9b0e1624c8e97c7d1aba113153cd34d0495c66744ca0`

本次不自动重试。后续仅允许计划第 11 节所述的 branching-prefix recovery。
