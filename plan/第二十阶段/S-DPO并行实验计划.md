# 第20阶段S-DPO并行实验计划

日期：2026-09-10。用户在SPRec运行期间询问是否可以增加并行实验，本次将已列入候选的S-DPO提前执行。原SPRec进程、参数和自动Beauty规则保持其已启动协议。

**终止更新：2026-09-11 14:08:28，已按用户明确指令安全停止。** 已完成两个epoch及第三epoch的407次优化器更新，最后checkpoint与优化器/RNG状态已保存；未完成本计划的三epoch全量评估，当前配置不自动重启。结论及后续实验见[第20阶段实验总报告](../../report/第二十阶段/GRAM_第二十阶段_实验总报告.md)。以下保留原预定协议。

## 方法及本轮问题

SPRec分支采用单个采样负例、三轮SFT/DPO交替。本分支测试S-DPO多负例目标：一个正确商品同时与四个高排名错误商品比较，冻结原父模型作为reference，连续训练三个epoch。

来源：[S-DPO论文v3](https://arxiv.org/html/2406.09215v3)、[作者代码](https://github.com/chenyuxin1999/S-DPO)。已核对作者`trainer/softmax_dpo_trainer.py`的`dpo_loss`和响应log概率求和；来源文件以SHA256固定在[来源清单](../../experiment/phase20/sdpo_source_manifest.json)。

采用论文Eq.(11)的稳定等价形式，令$r_j=\beta(\log\pi_\theta(y_j|x)-\log\pi_{ref}(y_j|x))$：

\[
L=\log\left(1+\sum_{j=1}^{4}\exp(r_j^- -r^+)\right),\quad\beta=0.1.
\]

原作是在给定候选列表中选择标题，本轮迁移为GRAM全目录词法生成；原作随机负例改为原父模型beam8中前四个非target商品。该负例来源是明确的适配改动，不能称作者完整复现，也不能把与SPRec的差异全部归因于负例数量。

## 固定执行设置

| 项目 | 设置 |
|---|---|
| 数据集 | 仅Toys；本次不自动扩展Beauty |
| 数据量 | 全部109,361条训练前缀；正式validation19,412人 |
| 起点和reference | 同一原GRAM epoch30父模型；reference全程固定 |
| 负例 | 原父模型beam8，过滤当前target后取前4个不同商品；只生成训练负例 |
| reference计算 | 生成负例时缓存完整响应log概率；绑定train index、target、负例及父模型哈希，三个epoch复用 |
| 训练 | 全部60,517,376参数；3epoch；纯S-DPO，无额外SFT交替 |
| 优化 | AdamW，lr1e-5，weight decay0.01，5%warmup和全日程线性衰减，clip1.0 |
| batch | microbatch2，有效batch128；挖负例microbatch4 |
| 数值 | FP32，TF32关闭；关闭dropout，但保留参数梯度 |
| 验证 | 原生beam50；每epoch固定同一3000人选择队列；raw/组合各选最佳，最多两个checkpoint做全量validation |
| 对照 | 复用SPRec任务先生成的未修改父模型同精度推理参考；核对数据、权重、精度和解码参数，不重复跑原模型推理 |
| 资源 | GPU0；至少12GiB空闲，allocator上限整卡40%；不用holder |
| 预算 | 48小时硬上限；无自动调参、重试或续期 |

若共享父参考尚未产生，可先挖负例并训练，第一次validation前等待；依赖明确失败时记录本任务失败，不改用不匹配的历史参考。

正常推荐准确率的主比较仍为原GRAM，组合辅助比较原GRAM+PCRF。单seed、重复使用的validation仅作探索，不把选择后的配对CI当独立确认。原始rank保存后可继续分析尾部和命中变化。

## 开跑检查

四项单元测试通过：K=1时退化为DPO；初始reference相同得到log5；负例排列不改变loss且困难负例梯度更大；极端值有限、target过滤正确。

GPU0最长历史20、microbatch2检查通过：初始loss1.6094379425、margin0；encoder、decoder、共享词嵌入和位置嵌入均有非零有限梯度；优化器实际改变权重。该段峰值allocated约2,368.44MiB、reserved2,798MiB，不等于全程峰值。另以8条训练数据执行真实负例缓存和两个optimizer更新，产物与正式训练隔离。

## 状态与时间

- 本分支：`artifacts/phase20/sdpo_gram_toys_v1/status.json`。
- 两条分支总览：`artifacts/phase20/overview/status.json`，30秒刷新。
- 原SPRec队列：`artifacts/phase20/status.json`，原入口仍可用。
- 本分支已于2026-09-10 22:43启动，首条负例生成事件为22:44:18；会话`s20_sdpo_gram_toys_v1`，启动进程PID2408042，后续以状态文件为准。
- 按当前吞吐波动预留1–2天，替代早先10–20小时的乐观估计；未完成正式epoch，尚不能提供可靠的精确结束时刻。48小时是硬预算，届时可能因预算结束而未正常完成全部实验。
- `COMPLETED`表示执行完成，不代表方法提点；准确率与结论读各自`summary.json`。

启动命令：`bash experiment/phase20/run_sdpo.sh start`；状态：`bash experiment/phase20/run_sdpo.sh status`；停止：`bash experiment/phase20/run_sdpo.sh stop`。
