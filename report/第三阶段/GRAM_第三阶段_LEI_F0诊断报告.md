# GRAM 第三阶段 LEI F0-D 诊断报告

- 决策：**`STOP_LEI_NO_RAW_ECHO`**
- 数据边界：冻结 validation checkpoint/cohort；未读 test，未训练，未生成 beam。
- 干预边界：输入 token/position/passage 不变，只修改预注册 span 的 attention mask。
- 执行完整性：4/4 CPU 单元测试通过；两数据集 cohort 与 CGI E0 完全一致，full
  重算误差为 0，role localization 与 matched-control eligibility 均为 100%。

第一次资源启动在 checkpoint 加载前因显存检查早于 CUDA context 完全释放而以
exit 4 结束，没有产生科学分数。唯一修复是补上 plan 已规定的最多 120 秒显存轮询；
未修改输入、span、control、seed、endpoint 或 gate，随后从头执行成功并恢复
CodeLlama 资源。

## 双数据集 gate

| Dataset | Integrity | Raw link harm | Role specificity | Metadata benefit | Failure association |
|---|---:|---:|---:|---:|---:|
| Toys | True | False | False | True | True |
| Beauty | True | False | True | True | True |

## 锁定主统计

### Toys

- tail-miss raw `R_link`: mean=0.005607, 95% CI=[-0.008848, 0.020022], P(>0)=0.531250
- tail-miss adjusted `A_link`: mean=0.007718, 95% CI=[-0.008556, 0.023076]
- tail-miss metadata `M_meta`: mean=0.413219, 95% CI=[0.342700, 0.486318]
- miss-hit adjusted association: mean=0.031164, 95% CI=[0.008914, 0.053482]
- secondary tail-miss `R_cf`: mean=0.086362; `R_all`: mean=0.088661

### Beauty

- tail-miss raw `R_link`: mean=0.020948, 95% CI=[0.002046, 0.046830], P(>0)=0.500000
- tail-miss adjusted `A_link`: mean=0.025195, 95% CI=[0.006720, 0.050103]
- tail-miss metadata `M_meta`: mean=0.110914, 95% CI=[0.003189, 0.216563]
- miss-hit adjusted association: mean=0.034683, 95% CI=[0.013275, 0.060939]
- secondary tail-miss `R_cf`: mean=-0.044840; `R_all`: mean=-0.043697

## 解释边界

只有五项 gate 在 Toys 与 Beauty 全部通过才允许进入 F1。CF-ID 结果为次要描述，
不能挽救主 link-span gate；自然重复、语义相似或 matched control 差异本身不等于 echo。

Toys 的 raw mean、CI 与 positive rate 均失败；Beauty 的 raw mean 与 CI 达标，但
positive rate 为 0.50，低于预注册 0.55。故两数据集均在串行第一项机制 gate 停止。
metadata benefit 与 failure association 虽均通过，不能越过 raw gate。F1 及后续不
解锁，不得改 cohort/span/阈值、扫描 CF 子集或用 Beauty 单数据集继续。
