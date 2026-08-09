"""Phase 13 cold-split preprocessor for GRAM datasets.

Reads a source dataset (e.g. Beauty) from GRAM/rec_datasets/<name>/, samples
50% (configurable) of items as "cold" using frequency-stratified sampling,
and writes a new dataset directory <name>_cold<eta*100>/ that GRAM can load
without any code changes.

Design (see plan/第十三阶段/GRAM_第十三阶段_CANARD探索计划v0.1.md v0):
  - Cold sampling: split items into K log-frequency buckets; sample eta from
    each bucket to guarantee cold set spans the full frequency spectrum
    (not all rare, not all popular).
  - Per-user filter: from the ORIGINAL user_sequence.txt (leave-one-out
    semantics: last=test, second-to-last=val, rest=train), remove cold items
    from the TRAIN PREFIX only. Keep the val/test targets as-is even if cold
    (they become the "cold subset" for evaluation).
  - Drop users whose train prefix has < MIN_WARM_HISTORY warm items (default
    4), because GRAM needs enough history to train.
  - item_plain_text.txt: keep ALL items (warm + cold). GRAM's tokenizer must
    still be able to encode cold items when they appear as val/test targets.
  - Vanilla GRAM never sees cold items as training targets (only in eval),
    so it should be unable to recall them (Gate v0: cold Recall@10 ≤ 0.5%).

Post-hoc eval (see eval_cold_warm.py) partitions the predictions tsv by
whether the target item is in cold_items.txt.

CLI:
    python cold_split.py \\
        --source-dir /path/to/GRAM/rec_datasets/Beauty \\
        --output-dir /path/to/GRAM/rec_datasets/Beauty_cold50 \\
        --eta 0.5 \\
        --buckets 10 \\
        --seed 12345 \\
        --min-warm-history 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", required=True, help="Source GRAM dataset dir")
    p.add_argument("--output-dir", required=True, help="Destination dir (created)")
    p.add_argument("--eta", type=float, default=0.5, help="Cold fraction (0-1)")
    p.add_argument("--buckets", type=int, default=10, help="log-freq bucket count")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--min-warm-history", type=int, default=4,
                   help="Drop users with fewer warm items in train prefix")
    p.add_argument("--force", action="store_true",
                   help="Overwrite output dir if it exists")
    return p.parse_args()


def read_user_sequences(path: Path) -> list[tuple[str, list[str]]]:
    """Each line: user_id item1 item2 ... itemN (space-separated)."""
    users = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            uid, items = parts[0], parts[1:]
            users.append((uid, items))
    return users


def read_item_text_lines(path: Path) -> list[tuple[str, str]]:
    """Each line: item_id <space> text..."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            first_space = line.find(" ")
            if first_space < 0:
                rows.append((line, ""))
            else:
                rows.append((line[:first_space], line[first_space + 1:]))
    return rows


def frequency_stratified_cold_sample(
    item_freq: dict[str, int],
    eta: float,
    n_buckets: int,
    rng: random.Random,
) -> set[str]:
    """Split items into log-frequency buckets; sample eta from each bucket.

    Items with the same frequency stay in the same bucket. Bucket boundaries
    are chosen on log(freq) to give roughly equal counts per bucket.
    """
    items = list(item_freq.keys())
    log_freqs = [math.log(item_freq[it] + 1.0) for it in items]

    if not log_freqs:
        return set()

    lo, hi = min(log_freqs), max(log_freqs)
    # avoid degenerate case where all items have same freq
    if hi - lo < 1e-9:
        n_cold = int(round(eta * len(items)))
        cold = set(rng.sample(items, n_cold))
        return cold

    edges = [lo + (hi - lo) * (i / n_buckets) for i in range(n_buckets + 1)]
    buckets: list[list[str]] = [[] for _ in range(n_buckets)]
    for it, lf in zip(items, log_freqs):
        b = min(int((lf - lo) / (hi - lo) * n_buckets), n_buckets - 1)
        buckets[b].append(it)

    cold: set[str] = set()
    for b in buckets:
        if not b:
            continue
        k = int(round(eta * len(b)))
        cold.update(rng.sample(b, k))
    return cold


def build_frequency_map(users: list[tuple[str, list[str]]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for _uid, seq in users:
        counter.update(seq)
    return dict(counter)


def filter_users(
    users: list[tuple[str, list[str]]],
    cold_items: set[str],
    min_warm_history: int,
) -> tuple[
    list[tuple[str, list[str]]],
    list[dict],
    dict,
]:
    """For each user, remove cold items from train prefix; keep val/test as-is.

    Returns:
      new_users:       list of (uid, new_seq) for users kept after filtering
      dropped:         list of dropped user records (for reporting)
      stats:           aggregate stats
    """
    new_users: list[tuple[str, list[str]]] = []
    dropped: list[dict] = []
    n_users_orig = len(users)
    n_users_no_val_test = 0
    n_users_too_short = 0
    n_cold_targets_val = 0
    n_cold_targets_test = 0
    n_cold_in_train_prefix = 0

    for uid, seq in users:
        if len(seq) < 3:
            # Not enough for train/val/test split
            n_users_no_val_test += 1
            dropped.append({"user": uid, "reason": "seq_len<3", "orig_len": len(seq)})
            continue

        train_prefix = seq[:-2]
        val_item = seq[-2]
        test_item = seq[-1]

        n_cold_in_prefix_this_user = sum(1 for x in train_prefix if x in cold_items)
        n_cold_in_train_prefix += n_cold_in_prefix_this_user

        warm_prefix = [x for x in train_prefix if x not in cold_items]

        if len(warm_prefix) < min_warm_history:
            n_users_too_short += 1
            dropped.append({
                "user": uid,
                "reason": "warm_prefix_too_short",
                "orig_len": len(seq),
                "warm_prefix_len": len(warm_prefix),
            })
            continue

        if val_item in cold_items:
            n_cold_targets_val += 1
        if test_item in cold_items:
            n_cold_targets_test += 1

        new_seq = warm_prefix + [val_item, test_item]
        new_users.append((uid, new_seq))

    stats = {
        "n_users_original": n_users_orig,
        "n_users_kept": len(new_users),
        "n_users_dropped": len(dropped),
        "n_users_dropped_seq_lt_3": n_users_no_val_test,
        "n_users_dropped_warm_prefix_too_short": n_users_too_short,
        "n_users_with_cold_val_target": n_cold_targets_val,
        "n_users_with_cold_test_target": n_cold_targets_test,
        "n_cold_items_removed_from_train_prefixes_total": n_cold_in_train_prefix,
    }
    return new_users, dropped, stats


def write_user_sequences(path: Path, users: list[tuple[str, list[str]]]):
    with open(path, "w") as f:
        for uid, seq in users:
            f.write(uid + " " + " ".join(seq) + "\n")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    src = Path(args.source_dir).resolve()
    dst = Path(args.output_dir).resolve()

    if not src.is_dir():
        print(f"ERROR: source dir not found: {src}", file=sys.stderr)
        sys.exit(2)
    if dst.exists():
        if not args.force:
            print(f"ERROR: output dir exists: {dst}. Pass --force to overwrite.",
                  file=sys.stderr)
            sys.exit(2)
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    print(f"[cold_split] source={src}", flush=True)
    print(f"[cold_split] output={dst}", flush=True)
    print(f"[cold_split] eta={args.eta} buckets={args.buckets} seed={args.seed}",
          flush=True)

    src_useq = src / "user_sequence.txt"
    src_items = src / "item_plain_text.txt"
    if not src_useq.exists() or not src_items.exists():
        print(f"ERROR: source dir missing user_sequence.txt or item_plain_text.txt",
              file=sys.stderr)
        sys.exit(2)

    users = read_user_sequences(src_useq)
    item_freq = build_frequency_map(users)
    all_items_in_seq = set(item_freq.keys())

    item_text_rows = read_item_text_lines(src_items)
    all_items_in_text = {row[0] for row in item_text_rows}

    # Items that appear in user_sequence but not in item_plain_text (or vice versa)
    orphan_seq = all_items_in_seq - all_items_in_text
    orphan_text = all_items_in_text - all_items_in_seq
    if orphan_seq:
        print(f"[cold_split] WARN: {len(orphan_seq)} items in sequence but not "
              f"in item_plain_text (kept as-is)", flush=True)
    if orphan_text:
        print(f"[cold_split] {len(orphan_text)} items in item_plain_text but not "
              f"referenced by any user (ignored for cold sampling)", flush=True)

    rng = random.Random(args.seed)
    cold_items = frequency_stratified_cold_sample(
        item_freq=item_freq,
        eta=args.eta,
        n_buckets=args.buckets,
        rng=rng,
    )
    warm_items = all_items_in_seq - cold_items

    new_users, dropped, stats = filter_users(
        users=users,
        cold_items=cold_items,
        min_warm_history=args.min_warm_history,
    )

    # Write outputs
    dst_useq = dst / "user_sequence.txt"
    dst_items = dst / "item_plain_text.txt"
    write_user_sequences(dst_useq, new_users)
    shutil.copyfile(src_items, dst_items)

    # Copy any other files GRAM might expect. These are static reference
    # files or pre-computed hierarchical id tables; safe to carry over since
    # our Beauty_cold50 retains the full item_plain_text.txt (all items still
    # get valid hierarchical ids, they just never appear as training targets).
    passthrough_globs = [
        "similar_item_sasrec.txt",
        # Pre-computed hierarchical id files — required by GRAM's gram_indexing
        # (it does NOT regenerate them; missing file raises FileNotFoundError).
        "item_generative_indexing_hierarchy_*.txt",
    ]
    import glob as _glob
    for pattern in passthrough_globs:
        for src_file in _glob.glob(str(src / pattern)):
            src_p = Path(src_file)
            shutil.copyfile(src_p, dst / src_p.name)

    # Metadata for reproducibility + eval
    meta_dir = dst / "cold_split_meta"
    meta_dir.mkdir()
    with open(meta_dir / "cold_items.txt", "w") as f:
        for it in sorted(cold_items):
            f.write(it + "\n")
    with open(meta_dir / "warm_items.txt", "w") as f:
        for it in sorted(warm_items):
            f.write(it + "\n")
    with open(meta_dir / "dropped_users.jsonl", "w") as f:
        for rec in dropped:
            f.write(json.dumps(rec) + "\n")

    config = {
        "source_dir": str(src),
        "output_dir": str(dst),
        "eta": args.eta,
        "buckets": args.buckets,
        "seed": args.seed,
        "min_warm_history": args.min_warm_history,
        "n_items_total": len(all_items_in_seq),
        "n_items_cold": len(cold_items),
        "n_items_warm": len(warm_items),
        "cold_fraction_actual": len(cold_items) / max(len(all_items_in_seq), 1),
        "source_user_sequence_sha256": sha256_of_file(src_useq),
        "source_item_plain_text_sha256": sha256_of_file(src_items),
        "output_user_sequence_sha256": sha256_of_file(dst_useq),
        "stats": stats,
    }
    with open(meta_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[cold_split] items: total={config['n_items_total']} "
          f"cold={config['n_items_cold']} warm={config['n_items_warm']} "
          f"(actual eta={config['cold_fraction_actual']:.3f})", flush=True)
    print(f"[cold_split] users: kept={stats['n_users_kept']}/"
          f"{stats['n_users_original']} "
          f"dropped_short_seq={stats['n_users_dropped_seq_lt_3']} "
          f"dropped_too_few_warm={stats['n_users_dropped_warm_prefix_too_short']}",
          flush=True)
    print(f"[cold_split] cold-target users: val={stats['n_users_with_cold_val_target']} "
          f"test={stats['n_users_with_cold_test_target']}", flush=True)
    print(f"[cold_split] wrote metadata to {meta_dir}", flush=True)
    print(f"[cold_split] done.", flush=True)


if __name__ == "__main__":
    main()
