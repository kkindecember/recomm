# GRAM 第三阶段 CGI E0 诊断报告

- 决策：**`STOP_CGI_NO_INTERFERENCE`**
- 数据边界：validation history/target 与冻结 validation beam-50；未读 test，未训练。
- 分数：gold lexical target token 的 mean log-prob；EOS/pad 排除。

## 双数据集 gate

| Dataset | Integrity | Cumulative | Old | Temporal | Failure association |
|---|---:|---:|---:|---:|---:|
| Toys | True | False | False | False | True |
| Beauty | True | False | False | False | True |

## 锁定主统计

### Toys

- tail_miss `G_all`: mean=-0.324558, 95% CI=[-0.438845, -0.214892], P(>0)=0.304688
- tail_miss `G_old`: mean=-0.032111, 95% CI=[-0.093794, 0.009510]
- tail_miss `G_old-G_new`: mean=-0.003521, 95% CI=[-0.088043, 0.073831]
- tail miss-hit `G_all`: mean=1.265395, 95% CI=[1.010699, 1.524562]

### Beauty

- tail_miss `G_all`: mean=-0.154611, 95% CI=[-0.334265, 0.008408], P(>0)=0.414062
- tail_miss `G_old`: mean=-0.007259, 95% CI=[-0.019643, 0.004976]
- tail_miss `G_old-G_new`: mean=-0.007755, 95% CI=[-0.121288, 0.139072]
- tail miss-hit `G_all`: mean=2.637946, 95% CI=[2.177965, 3.115435]

