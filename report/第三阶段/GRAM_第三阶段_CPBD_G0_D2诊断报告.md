# GRAM 第三阶段：CPBD G0-D2 frozen outcome diagnosis

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-24
- Verification Status: ANALYZED
- Version Label: `cpbd_g0_d2_v1`

## 固定结论

- 决策：**`STOP_CPBD_NO_NET_VALUE`**
- 边界：锁定 validation checkpoint/cohort；固定 128-token budget 与 CF identity；未训练、未读 test。
- 资源：GPU3 frozen scoring 完成后 CodeLlama 已恢复。

## 双数据集结果

| Dataset | Integrity | Net value | Recovered value | No broad harm |
|---|---:|---:|---:|---:|
| Toys | True | False | False | False |
| Beauty | True | False | False | False |

## 锁定主统计

### Toys

- tail-miss net: mean=0.066255, 95% CI=[-0.026730, 0.153510], P(>0)=0.546875
- tail-miss recovered-all: mean=0.015748, 95% CI=[0.004449, 0.027150], P(>0)=0.566406
- tail-miss recovered-slice8: mean=0.002663, 95% CI=[-0.003280, 0.008579]
- tail-hit net mean=-0.855011

### Beauty

- tail-miss net: mean=-0.070700, 95% CI=[-0.270813, 0.089167], P(>0)=0.585938
- tail-miss recovered-all: mean=0.021275, 95% CI=[0.003927, 0.039174], P(>0)=0.578125
- tail-miss recovered-slice8: mean=0.004733, 95% CI=[0.000035, 0.009786]
- tail-hit net mean=-2.180000

## 解释边界

只有四类 gate 在双数据集同时通过才解锁 G1。matched visible slice、
residual layout 与 failure association 均为 secondary descriptive，不能挽救主 gate。

Toys tail-miss 的净均值虽为正，但 CI 跨 0 且 positive rate 低于 0.55；Beauty
tail-miss 净均值为负，因此第一道串行 gate 已失败。两数据集 recovered-all 在
tail miss 上均为小幅正值且 CI 为正，说明新恢复内容并非完全无用；但它不足以抵消
重排造成的 layout/CF visibility 代价。tail-hit 的巨大负值进一步表明固定
metadata-first 不是安全修复。

因此 CPBD 只保留“结构性 displacement 存在”的诊断结论，不能进入 G1 allocator。
不得在同一 validation 上改为只挑 tail miss、扫描 confidence threshold、字段顺序、
top-k、context length 或 mask size。
