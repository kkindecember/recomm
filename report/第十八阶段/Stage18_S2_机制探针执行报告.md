# Stage18 S18-2 机制探针执行报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Status：`COMPLETED`
- Decision：`FAILED_SCIENTIFIC_GATE`
- Config SHA256：`e05d127c4b3eeb71e79b3a8391c2440c68dd891c816ddd5efe6e9d6595959d41`
- Scope：Toys/Beauty I0，四臂 matched continuation；I1/I2 与所有外部保护数据保持封存。


## Toys

Gate：`FAILED_SCIENTIFIC_GATE`

| Arm | Hit@50 | NDCG@10 | Prefix survival | Path margin |
|---|---:|---:|---:|---:|
| C0_CONT | 0.128000 | 0.041045 | 0.405533 | -1.982600 |
| A0_LEGAL_GENERIC | 0.131000 | 0.043918 | 0.403933 | -1.949770 |
| M0_PCPS | 0.133000 | 0.042276 | 0.405533 | -1.942366 |
| S0_SHUFFLED_CF | 0.128000 | 0.042834 | 0.404633 | -1.941599 |

本步骤仅用于工程和机制淘汰，不构成 accuracy success；不自动重试或启动 S18-3。
