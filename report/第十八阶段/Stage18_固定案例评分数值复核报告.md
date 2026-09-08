# 第十八阶段：固定案例评分数值复核

状态：`NUMERICAL_COMPARISON_COMPLETED`。
完成时间：2026-09-08T03:27:38.015417+00:00。

本轮最多 9 个工程案例；固定权重、原生 frontier 和输入，无训练步或新确认样本。

完成 9 个案例；旧 native trace 精确复现 9 个，
旧 teacher-forcing 值精确复现 9 个。
开启 TF32 时，两种评分最大累计差为 0.008981704712；
关闭 TF32 后为 1.335144043e-05。
关闭后差异缩小的案例数：9/9。

| 案例 | 原设置累计分差 | 关闭 TF32 累计分差 |
|---|---:|---:|
| Toys/training/AUBF1WVKUKBKC | 0.00164890289 | 4.76837158e-06 |
| Toys/training/A3VEFQ25OSXMOQ | 0.00427055359 | 1.33514404e-05 |
| Toys/training/AKSAJKVR26RX2 | 0.00153112411 | 1.00135803e-05 |
| Toys/evaluation/A1GCFTFXELCHRP | 0.000461578369 | 1.14440918e-05 |
| Toys/evaluation/A10L132FI5QHV6 | 0.00336647034 | 1.33514404e-05 |
| Toys/evaluation/A14ENWEKTHCBXR | 0.00118780136 | 9.05990601e-06 |
| Beauty/training/A19KLWTEW9KRJ8 | 0.00387573242 | 3.81469727e-06 |
| Beauty/training/A3I947II1AZ810 | 0.00898170471 | 6.67572021e-06 |
| Beauty/training/AF7WCPEASVKA0 | 0.00397586823 | 1.43051147e-06 |

这些配对结果用于定位数值计算路径的影响，不代表 CF 的科学增量或梯度稳定性已被验证。
原梯度状态 MEASUREMENT_UNRESOLVED 与 S18-2 FAILED_SCIENTIFIC_GATE 保持不变。
逐条件 logits/log-softmax/累计分数差、encoder 差、缓存差与权重 SHA 均保存在 cases.jsonl。

[完整产物](../../artifacts/phase18/score_numerics_audit/run-0001)。

## 完成后的解释与停点

本轮于北京时间 2026-09-08 11:27:38 完成，worker 用时约 100 秒；supervisor 正常退出，exit_code=0，无超时。
9 项相关 CPU 测试通过，9 个案例原 native trace 与历史 teacher-forcing 值均精确复现，
50 路固定 frontier 的增量缓存重放与 native 捕获的 logits/log-probability 均完全一致，所有权重 SHA 前后相等。

**本轮定位到的主要来源是 TF32 下不同计算批次与缓存方式的数值差异。**
这是对固定 9 个工程案例的配对干预结论，不能外推成所有 GRAM 运行均无实现错误。
关闭 TF32 后，原两种评分的最大累计差从 0.0089817047 降至 0.0000133514，两个最大值之比约 672.7；
9/9 个案例的差异均缩小。原计算可以精确重放，不支持“随机损坏了 trace”这一解释。

进一步的分离比较显示：

- 只改变 encoder 单输入/重复输入编码，在 TF32 开启时最大累计分差为 0.00343895，关闭后为 0.00000667572。
- 固定 encoder 和 50 路前缀，只改变增量缓存/完整前缀计算，开启时最大累计分差为 0.00296974，关闭后为 0.0000324249。
- Beauty 两个 depth=1 案例中，同批次 cache 开关的分差为 0，此时还没有历史缓存；
  其中最大残差案例的 decoder 50 路/2 路比较已有 0.00914192 分差，关闭 TF32 后为 0.00000667572。
  因而不能把全部残差归因于缓存逻辑。
- 顺序 float32 累加与 float32 reduction 的最大差只有 0.00000190735，不能单独解释原先 1e-3 到 1e-2 的差异。

以上因素存在相互作用，差异也可能抵消；这些最大值不能相加后声称是完整误差分解。
关闭 TF32 后全矩阵仍有约 1e-5 的差异，并未得到逐位完全一致的普遍结论。
PyTorch 1.11 官方说明记录了该版本 allow_tf32 的默认行为及其对 float32 运算精度的影响，
与本机记录和干预结果一致。[PyTorch 1.11 CUDA/TF32 说明](https://raw.githubusercontent.com/pytorch/pytorch/v1.11.0/docs/source/notes/cuda.rst)
官方数值精度文档也说明，浮点运算的分组和批次计算方式可能导致结果不逐位相同；
该一般原则不替代本轮的配对实验。[PyTorch 数值精度说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)

### 现在可以与不可以做的判断

可以把“原生/可微评分差异的主要数值来源尚不清楚”收敛为“在这 9 个案例中，TF32 与计算形状/缓存方式的影响已定位”。
此处无需靠修改旧 epsilon 或 Gate 消除非零残差。

不能因此把 100 用户平均梯度、85 个下一 offset margin 的结论自动标为可靠；本轮没有 backward。
原记录的 `MEASUREMENT_UNRESOLVED` 是梯度测量的历史状态，本轮新增数值解释不覆盖该记录。
也没有证据说明关闭 TF32 就能修复 18-2 的科学失败或带来准确率改善。

**下一步应是一次有边界的梯度稳定性复核**：原 Toys 前 100 用户的训练梯度与原固定 85 个 evaluation 事件，
保持四目标/alpha=0.1/负例缓存/权重/前缀边界不变，比较 TF32 开启与关闭时的方向、配对差和向量差异。
需要在 GPU 结果前另行冻结完整条件；不新增用户、不训练、不重选边界。
历史未保存梯度向量，仅有 SHA/范数，故如需比较向量必须重算两种精度，不能假称已有向量可直接复用。

若真实 CF 对 generic 仍无稳定增量，应停止扩大当前 CF 设计；再决定是否研究 generic 跨父目标，
并在任何新训练前先设定 frozen parent 与 matched CE 的 continuation 对照。
本轮已完成的授权范围是 9 案例评分复核，尚未启动上述梯度稳定性复核。
