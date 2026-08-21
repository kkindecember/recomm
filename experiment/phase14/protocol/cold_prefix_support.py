"""Phase-14 / Stage 14-0B 前置只读诊断：cold item 的 lexical path 在多深处脱离 warm 支持。

动机
----
arXiv 2607.21101 在 TIGER/RQ-VAE 上报告冷物品的瓶颈是「末层 fine-grained path
completion」——粗层码本共享，模型大致落对区域，死在最后一两位。

GRAM 的 identifier 不是 RQ-VAE，而是 item text embedding 上的层级 k-means，
分裂到叶子近乎唯一。因此不能照搬那个结论，必须在本仓库的 identifier 上实测。

本脚本只读 catalog 侧文件，不读任何 user interaction、不读 validation/test
target、不加载模型，因此不受数据防火墙约束（summary 里 test_read 恒为 false）。

它回答一个纯结构性的问题：

    对每个 cold item，它的 lexical path 的最深前缀 z[:k]，
    有多深仍然被「至少一个 warm item」共享？

k 小 ⇒ 冷路径很早就离开了训练分布支持的子树，decoder 在中层就无处可去；
k 接近 L ⇒ 冷路径几乎完全在 warm 子树内，只差最后的 identification token。

⚠️ 这个量**不是** learned NLL 断崖的上界（v0.2 曾如此声称，v0.3 已撤销）。

神经 decoder 可以组合出从未作为完整前缀出现过的路径：只要每一层的 token 在该层
被 warm item 见过，trie 上就存在可走的路。实测这个差距很大——Toys 在 depth≤2
的累计断裂率，按完整前缀口径是 67.8%，按逐层 token 口径只有 32.3%（Beauty 为
82.5% vs 47.5%）。反过来，decoder 也可能在结构上有 warm overlap 的 prefix 处
提前失败。

因此本脚本只提供**结构先验**，用于形成假设、解释 14-0B 的结果，不能单独否定
末层 head、search failure 或 R2PD 任何一条路线。真实的失败位置必须由
oracle_prefix_probe.py 实测 token NLL / target rank / beam survival 得出。

用法
----
    python experiment/phase14/protocol/cold_prefix_support.py \
        --dataset-dir GRAM/rec_datasets/Toys_cold50 \
        --out artifacts/phase14/diagnostics/cold_prefix_support_toys.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from typing import Dict, List


# identifier 文件里每行形如：  B0000A1Z5K |▁animals|stuffed|▁se|▁cat|hat
# item id 与 path 之间是 " |"，path 内部以 "|" 分隔。
ITEM_PATH_SEP = " |"
TOKEN_SEP = "|"


def load_paths(path_file: str) -> Dict[str, List[str]]:
    """读 baseline identifier 文件，返回 item_id -> lexical token 列表。"""
    paths: Dict[str, List[str]] = {}
    with open(path_file, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line:
                continue
            if ITEM_PATH_SEP not in line:
                raise ValueError(f"{path_file}:{lineno} 无法解析：缺少 '{ITEM_PATH_SEP}' 分隔符")
            item_id, encoded = line.split(ITEM_PATH_SEP, 1)
            tokens = encoded.split(TOKEN_SEP)
            if item_id in paths:
                raise ValueError(f"{path_file}:{lineno} item {item_id} 重复出现")
            paths[item_id] = tokens
    return paths


def load_item_set(list_file: str) -> set:
    with open(list_file, encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def resolve_baseline_path_file(dataset_dir: str) -> str:
    """定位 baseline（未经任何 cold 重映射的）identifier 文件。

    cold50 目录里同时存在大量 phase-13 产出的 *cold*.txt 变体，必须排除，
    只保留原始的 ..._split.txt。
    """
    candidates = [
        p
        for p in glob.glob(os.path.join(dataset_dir, "item_generative_indexing_*.txt"))
        if os.path.basename(p).endswith("_split.txt")
    ]
    if len(candidates) != 1:
        raise SystemExit(
            f"在 {dataset_dir} 下期望恰好 1 个 baseline identifier 文件，实际找到 {len(candidates)}：{candidates}"
        )
    return candidates[0]


def deepest_supported_prefix(tokens: List[str], warm_prefixes: Dict[int, set]) -> int:
    """返回最深的 k，使得 tokens[:k] 被至少一个 warm item 共享。

    一旦某层不被支持即停止——前缀支持天然是单调的（若 z[:k] 无 warm item 共享，
    则 z[:k+1] 也不可能有），提前 break 既正确又省时。
    """
    depth = 0
    for k in range(1, len(tokens) + 1):
        if tuple(tokens[:k]) in warm_prefixes.get(k, ()):
            depth = k
        else:
            break
    return depth


def deepest_token_supported(tokens: List[str], warm_tokens: Dict[int, set]) -> int:
    """返回最深的 k，使得 tokens[:k] 的**每一层 token 在该层**都被 warm 见过。

    这是比 deepest_supported_prefix 宽松得多的口径，也是更贴近 decoder 实际
    能力的那一个：trie 上只要每层存在该 token，路径就可走通，无需该完整前缀
    组合曾经出现过。

    两个口径的差值正是「可组合但未被观测」的空间。v0.2 曾误以为精确前缀口径
    是 learned 失败位置的上界，正是忽略了这一块。
    """
    depth = 0
    for k, token in enumerate(tokens, 1):
        if token in warm_tokens.get(k, ()):
            depth = k
        else:
            break
    return depth


def analyse(dataset_dir: str) -> dict:
    meta_dir = os.path.join(dataset_dir, "cold_split_meta")
    path_file = resolve_baseline_path_file(dataset_dir)

    paths = load_paths(path_file)
    cold_items = load_item_set(os.path.join(meta_dir, "cold_items.txt"))
    warm_items = load_item_set(os.path.join(meta_dir, "warm_items.txt"))

    # 完整性检查：cold/warm 应当划分整个 catalog，且互不相交。
    overlap = cold_items & warm_items
    if overlap:
        raise SystemExit(f"cold 与 warm 集合相交，共 {len(overlap)} 个 item，划分已损坏")

    # 重复路径审计：baseline identifier 必须逐 item 唯一，否则字符串级 evaluator
    # 会把同路径的不同 item 误判为命中（phase-13 的 v1 alias 问题就出在这里）。
    path_counts = collections.Counter(tuple(t) for t in paths.values())
    duplicate_paths = sum(1 for count in path_counts.values() if count > 1)

    # 建 warm 索引：
    #   warm_prefixes[k] = 深度 k 上出现过的完整前缀（严格口径）
    #   warm_tokens[k]   = 深度 k 上出现过的 token（宽松口径，刻画可组合空间）
    warm_prefixes: Dict[int, set] = collections.defaultdict(set)
    warm_tokens: Dict[int, set] = collections.defaultdict(set)
    for item in warm_items:
        tokens = paths.get(item)
        if tokens is None:
            continue
        for k in range(1, len(tokens) + 1):
            warm_prefixes[k].add(tuple(tokens[:k]))
            warm_tokens[k].add(tokens[k - 1])

    depth_hist: collections.Counter = collections.Counter()
    token_depth_hist: collections.Counter = collections.Counter()
    all_but_last_supported = 0
    total = 0
    for item in cold_items:
        tokens = paths.get(item)
        if tokens is None:
            continue
        total += 1
        depth_hist[deepest_supported_prefix(tokens, warm_prefixes)] += 1
        token_depth_hist[deepest_token_supported(tokens, warm_tokens)] += 1
        if tuple(tokens[:-1]) in warm_prefixes.get(len(tokens) - 1, ()):
            all_but_last_supported += 1

    if total == 0:
        raise SystemExit("没有一个 cold item 出现在 identifier 文件中，输入路径可能不对")

    length_hist = collections.Counter(len(t) for t in paths.values())
    # 「除末位外前缀唯一」的比例，刻画 k-means 分裂到叶子的锐利程度：
    # 越接近 1，说明叶子越唯一，末层越像纯 identification token。
    penultimate = collections.Counter(tuple(t[:-1]) for t in paths.values())
    unique_penultimate = sum(1 for t in paths.values() if penultimate[tuple(t[:-1])] == 1)

    cumulative = {}
    running = 0
    for depth in sorted(depth_hist):
        running += depth_hist[depth]
        cumulative[str(depth)] = round(100.0 * running / total, 2)

    token_cumulative = {}
    running = 0
    for depth in sorted(token_depth_hist):
        running += token_depth_hist[depth]
        token_cumulative[str(depth)] = round(100.0 * running / total, 2)

    return {
        "dataset_dir": dataset_dir,
        "identifier_file": os.path.basename(path_file),
        "test_read": False,
        "n_items_total": len(paths),
        "n_cold": len(cold_items),
        "n_warm": len(warm_items),
        "n_cold_with_path": total,
        "path_length_hist": {str(k): v for k, v in sorted(length_hist.items())},
        "duplicate_paths": duplicate_paths,
        "frac_unique_penultimate_prefix": round(unique_penultimate / len(paths), 4),
        "cold_depth_hist": {str(k): v for k, v in sorted(depth_hist.items())},
        "cold_depth_pct": {
            str(k): round(100.0 * v / total, 2) for k, v in sorted(depth_hist.items())
        },
        "cold_depth_cumulative_pct": cumulative,
        "cold_token_depth_hist": {
            str(k): v for k, v in sorted(token_depth_hist.items())
        },
        "cold_token_depth_cumulative_pct": token_cumulative,
        "cold_all_but_last_prefix_supported": all_but_last_supported,
        "cold_all_but_last_prefix_supported_pct": round(
            100.0 * all_but_last_supported / total, 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="如 GRAM/rec_datasets/Toys_cold50")
    parser.add_argument("--out", help="可选：把 summary 写到这个 JSON 路径")
    args = parser.parse_args()

    summary = analyse(args.dataset_dir)

    print(f"== {summary['dataset_dir']} ==")
    print(f"identifier      : {summary['identifier_file']}")
    print(f"items           : {summary['n_items_total']}  (cold {summary['n_cold']} / warm {summary['n_warm']})")
    print(f"path length     : {summary['path_length_hist']}")
    print(f"duplicate paths : {summary['duplicate_paths']}")
    print(f"除末位外前缀唯一 : {summary['frac_unique_penultimate_prefix']:.1%}")
    print("cold item 最深 warm-supported 深度（两种口径的累计百分比）：")
    print("  depth |  完整前缀(严格) |  逐层token(宽松)")
    all_depths = sorted(
        {int(d) for d in summary["cold_depth_cumulative_pct"]}
        | {int(d) for d in summary["cold_token_depth_cumulative_pct"]}
    )
    for depth in all_depths:
        strict = summary["cold_depth_cumulative_pct"].get(str(depth))
        loose = summary["cold_token_depth_cumulative_pct"].get(str(depth))
        strict_s = f"{strict:6.1f}%" if strict is not None else "     —"
        loose_s = f"{loose:6.1f}%" if loose is not None else "     —"
        print(f"  {depth:5d} | {strict_s}         | {loose_s}")
    print(
        "  ↑ 两列的差 = 「可组合但从未被完整观测」的空间；"
        "严格口径不是 learned 失败位置的上界"
    )
    print(
        f"「除末位外全路径」被 warm 支持: "
        f"{summary['cold_all_but_last_prefix_supported']}/{summary['n_cold_with_path']} "
        f"= {summary['cold_all_but_last_prefix_supported_pct']}%"
    )

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
