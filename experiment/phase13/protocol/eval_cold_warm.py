"""Phase 13 post-hoc cold/warm eval splitter.

GRAM's `--save_predictions` writes a tsv where each row is one test user with:
  col 0: user_id
  col 1..12: per-sequence metrics (hit@{1,3,5,10,20,50}, ndcg@{1,3,5,10,20,50})
  col 13: gold item text (tokenized hierarchical id)
  col 14: '||'-separated top-K predicted item texts
  col 15: '||'-separated top-K scores

We don't need to re-parse gold text — the test target is the last item in each
user's sequence, and we already know the split. This script:
  1. Loads user_sequence.txt from the cold-split dataset dir (test = last item)
  2. Loads cold_items.txt from cold_split_meta/
  3. Reads the predictions tsv, joins on user_id, partitions rows by
     is_cold(target), averages per-sequence metrics.
  4. Emits metrics_summary.json + a short human-readable console report.

CLI:
    python eval_cold_warm.py \\
        --dataset-dir GRAM/rec_datasets/Beauty_cold50 \\
        --predictions-tsv artifacts/phase13/explore/v0_vanilla_baseline/predictions/xxx_test.tsv \\
        --output-json artifacts/phase13/explore/v0_vanilla_baseline/metrics_cold_warm.json \\
        --version-tag v0
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


# Order must match GRAM's default self.metrics list (see runner code).
METRIC_NAMES = [
    "hit@1", "hit@3", "hit@5", "hit@10", "hit@20", "hit@50",
    "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True,
                   help="Cold-split dataset dir (must contain user_sequence.txt "
                        "and cold_split_meta/cold_items.txt)")
    p.add_argument("--predictions-tsv", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--version-tag", default="v0",
                   help="Version tag written into the summary (e.g. v0, v1_iter1)")
    p.add_argument("--split-name", default="test",
                   choices=["test", "validation"],
                   help="Whether the target is the last item (test) or "
                        "second-to-last item (validation)")
    return p.parse_args()


def load_user_target_map(useq_path: Path, split: str) -> dict[str, str]:
    """Map user_id -> target item_id for the given split."""
    idx = -1 if split == "test" else -2
    out: dict[str, str] = {}
    with open(useq_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            uid = parts[0]
            items = parts[1:]
            if len(items) < 2:
                continue
            out[uid] = items[idx]
    return out


def load_cold_items(cold_items_path: Path) -> set[str]:
    with open(cold_items_path) as f:
        return {line.strip() for line in f if line.strip()}


def parse_predictions_tsv(path: Path) -> list[tuple[str, list[float]]]:
    """Return list of (user_id, [metric values]).

    Stops when a trailing summary line (starting with 'hit@' or 'ndcg@') appears.
    """
    rows: list[tuple[str, list[float]]] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # Skip trailing "hit@10: 0.09..." summary lines
            if line.startswith(("hit@", "ndcg@")):
                continue
            # Skip header row
            if line.startswith("idx\t"):
                continue
            fields = line.split("\t")
            if len(fields) < 1 + len(METRIC_NAMES):
                continue
            uid = fields[0]
            try:
                metrics = [float(x) for x in fields[1: 1 + len(METRIC_NAMES)]]
            except ValueError:
                continue
            rows.append((uid, metrics))
    return rows


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarise(rows: list[tuple[str, list[float]]]) -> dict:
    if not rows:
        return {"n": 0, **{m: None for m in METRIC_NAMES}}
    out = {"n": len(rows)}
    for i, mname in enumerate(METRIC_NAMES):
        out[mname] = mean([r[1][i] for r in rows])
    return out


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    useq_path = dataset_dir / "user_sequence.txt"
    cold_items_path = dataset_dir / "cold_split_meta" / "cold_items.txt"
    preds_path = Path(args.predictions_tsv).resolve()
    output_path = Path(args.output_json).resolve()

    for p, label in [(useq_path, "user_sequence.txt"),
                     (cold_items_path, "cold_split_meta/cold_items.txt"),
                     (preds_path, "predictions tsv")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(2)

    user_target = load_user_target_map(useq_path, args.split_name)
    cold_items = load_cold_items(cold_items_path)
    rows = parse_predictions_tsv(preds_path)

    warm_rows: list[tuple[str, list[float]]] = []
    cold_rows: list[tuple[str, list[float]]] = []
    missing_users = 0

    for uid, metrics in rows:
        target = user_target.get(uid)
        if target is None:
            missing_users += 1
            continue
        if target in cold_items:
            cold_rows.append((uid, metrics))
        else:
            warm_rows.append((uid, metrics))

    summary = {
        "version_tag": args.version_tag,
        "split": args.split_name,
        "dataset_dir": str(dataset_dir),
        "predictions_tsv": str(preds_path),
        "n_pred_rows_total": len(rows),
        "n_pred_rows_missing_user_map": missing_users,
        "n_users_in_dataset": len(user_target),
        "n_cold_items": len(cold_items),
        "overall": summarise(rows),
        "warm": summarise(warm_rows),
        "cold": summarise(cold_rows),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    def fmt(k: str, group: dict) -> str:
        v = group.get(k)
        return f"{v:.6f}" if isinstance(v, float) else "n/a"

    print(f"[eval_cold_warm] tag={args.version_tag} split={args.split_name}")
    print(f"[eval_cold_warm] rows: total={len(rows)} warm={len(warm_rows)} "
          f"cold={len(cold_rows)} missing={missing_users}")
    print(f"[eval_cold_warm] overall  hit@10={fmt('hit@10', summary['overall'])} "
          f"ndcg@10={fmt('ndcg@10', summary['overall'])}")
    print(f"[eval_cold_warm] warm     hit@10={fmt('hit@10', summary['warm'])} "
          f"ndcg@10={fmt('ndcg@10', summary['warm'])}")
    print(f"[eval_cold_warm] cold     hit@10={fmt('hit@10', summary['cold'])} "
          f"ndcg@10={fmt('ndcg@10', summary['cold'])}")
    print(f"[eval_cold_warm] wrote {output_path}")


if __name__ == "__main__":
    main()
