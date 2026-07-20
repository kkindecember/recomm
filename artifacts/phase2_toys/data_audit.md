# GRAM Toys 数据审计

审计日期：2026-07-20  
代码版本：`7ac4d9272a57beed9df35c27ea34221f6e4a8fb1`

## 结论

官方 Toys 预处理数据完整、非空且内部映射一致，数据规模与论文 Table 10 完全一致。本次审计只读文件，没有修改或重新生成数据。

| 项目 | 论文 | 本地 | 结果 |
|---|---:|---:|---|
| 用户数 | 19,412 | 19,412 | 一致 |
| 商品数 | 11,924 | 11,924 | 一致 |
| 交互数 | 167,597 | 167,597 | 一致 |
| 密度 | 0.0724% | 0.0724059% | 一致 |

## 输入文件

| 文件 | 大小 | 数据行 | 唯一首列 ID | 空值 | 格式错误 | 重复首列 ID |
|---|---:|---:|---:|---:|---:|---:|
| `user_sequence.txt` | 2,129,679 bytes | 19,412 | 19,412 | 0 | 0 | 0 |
| `item_plain_text.txt` | 7,595,045 bytes | 11,924 | 11,924 | 0 | 0 | 0 |
| `item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt` | 643,301 bytes | 11,924 | 11,924 | 0 | 0 | 0 |
| `similar_item_sasrec.txt`（不含表头） | 2,754,562 bytes | 11,924 | 11,924 | 0 | 0 | 0 |

## 用户序列与划分

- 用户序列长度最小 5、最大 550，少于 3 个交互的用户为 0。
- 序列中包含 11,924 个唯一商品，覆盖全部商品集。
- `MultiTaskDatasetGRAM` 以 `items[:-2]` 构造训练历史，以 `items[-2]` 作为 validation 目标。
- `TestDatasetGRAM` 以 `items[-1]` 作为 test 目标，以 `items[:-1]` 作为 test 历史。
- 19,412 个用户均能产生且只产生一个 validation 目标和一个 test 目标。

## 映射与相似商品完整性

- 用户序列商品缺失文本映射：0。
- 用户序列商品缺失层次 ID 映射：0。
- 商品文本与层次 ID 的集合差：双向均为 0。
- 重复层次 ID 值：0。
- SASRec anchor 缺失/额外：0 / 0。
- 每个 anchor 均有 20 个相似商品，官方 Toys 脚本使用前 5 个。
- 相似商品引用中缺失商品：0；含重复邻居的 anchor：0；含自身的 anchor：0。

## Full-ranking 证据

Toys test dataset 将全部 11,924 个层次文本 ID 放入 `all_items`。`SingleRunnerGRAM` 将全部候选编码为 Trie，并在 `generate()` 中使用 `prefix_allowed_tokens_fn` 进行 constrained beam search。默认 beam size 为 50，未构造负采样候选集。

## 审计方法

逐行按官方“首个空格分隔 ID 与内容”格式解析四个文件；统计唯一键、空值、映射集合差、序列长度、交互数、层次 ID 值唯一性和 SASRec 邻居。另以源码静态检查确认划分、Trie 候选和生成约束。
