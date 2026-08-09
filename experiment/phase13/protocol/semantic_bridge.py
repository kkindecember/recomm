"""Semantic bridge MLP: text_embedding -> hierarchical id per level.

Plan v1 (see plan/第十三阶段/GRAM_第十三阶段_CANARD探索计划v0.1.md Section 2):
  "1 层 MLP 映射 text embedding → hierarchical id(每层独立 softmax)
   Cross-entropy loss 训练 warm items 的 (text → id) 映射"

Design:
  - Per-level linear head: text_dim -> level_vocab_size (7 heads)
  - Total params ≈ text_dim * sum(level_vocab_sizes) ≈ 22M with all-MiniLM
    (384) and Beauty (108+3823+4894+5062+5026+5008+4776=28697)
  - Loss = sum over 7 levels of cross-entropy (deeper levels are much larger
    output spaces, but plan says "each level independent"; consider a
    per-level weight in iteration if the deep-level loss dominates)

Training targets: warm items only. Cold items are never seen during training
so the model learns "text -> hierarchical id" purely from warm supervision.
Inference on cold items reuses learned mapping.

CLI:
    python semantic_bridge.py train \\
        --embeddings artifacts/phase13/embeddings/beauty_sbert.pt \\
        --id-file GRAM/rec_datasets/Beauty/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt \\
        --cold-items GRAM/rec_datasets/Beauty_cold50/cold_split_meta/cold_items.txt \\
        --output-dir artifacts/phase13/explore/v1_beauty/mlp \\
        --epochs 200 --lr 1e-3 --batch-size 512 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hierarchical_id_utils import (
    HierIdVocab, build_vocab_from_id_file, read_id_file, read_item_set,
)


def parse_args():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    tr = sp.add_parser("train")
    tr.add_argument("--embeddings", required=True, help="output of precompute_item_embeddings.py")
    tr.add_argument("--id-file", required=True, help="GRAM item_generative_indexing_*.txt (from source dataset)")
    tr.add_argument("--cold-items", required=True, help="cold_split_meta/cold_items.txt from cold-split dataset")
    tr.add_argument("--output-dir", required=True)
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--batch-size", type=int, default=512)
    tr.add_argument("--device", default="cuda:0")
    tr.add_argument("--seed", type=int, default=12345)
    tr.add_argument("--val-fraction", type=float, default=0.1,
                    help="Held-out warm items for per-level top-1 accuracy tracking")

    return p.parse_args()


def build_model(text_dim: int, level_sizes: list[int]):
    import torch.nn as nn
    class SemanticBridgeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = nn.ModuleList([nn.Linear(text_dim, s) for s in level_sizes])

        def forward(self, x):
            return [h(x) for h in self.heads]  # list of (B, level_size)

    return SemanticBridgeMLP()


def train_cmd(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)

    embeddings_path = Path(args.embeddings).resolve()
    id_file_path = Path(args.id_file).resolve()
    cold_items_path = Path(args.cold_items).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bridge] embeddings={embeddings_path}")
    print(f"[bridge] id_file={id_file_path}")
    print(f"[bridge] cold_items={cold_items_path}")

    payload = torch.load(embeddings_path, map_location="cpu")
    ids = payload["item_ids"]
    embs = payload["embeddings"]  # (N, D)
    text_dim = embs.shape[1]
    id_to_row = {iid: i for i, iid in enumerate(ids)}

    id_map = read_id_file(id_file_path)  # dict[item_id -> [7 tokens]]
    cold_items = read_item_set(cold_items_path)

    vocab = build_vocab_from_id_file(id_file_path)
    level_sizes = vocab.level_sizes
    n_levels = vocab.n_levels
    print(f"[bridge] text_dim={text_dim} n_levels={n_levels} level_sizes={level_sizes}")

    # Assemble warm training set: (embedding, 7-int target)
    warm_x_rows = []
    warm_y = []
    n_missing_emb = 0
    n_missing_id = 0
    for item_id in id_map.keys():
        if item_id in cold_items:
            continue
        if item_id not in id_to_row:
            n_missing_emb += 1
            continue
        try:
            target = vocab.encode(id_map[item_id])
        except (KeyError, AssertionError):
            n_missing_id += 1
            continue
        warm_x_rows.append(id_to_row[item_id])
        warm_y.append(target)

    if not warm_x_rows:
        print("ERROR: no warm items with both embedding and id", file=sys.stderr)
        sys.exit(2)

    x = embs[warm_x_rows]  # (N_warm, D)
    y = torch.tensor(warm_y, dtype=torch.long)  # (N_warm, n_levels)
    print(f"[bridge] warm items with embedding+id: {len(x)} "
          f"(missing_emb={n_missing_emb}, missing_id={n_missing_id})")

    # Simple deterministic train/val split
    n = len(x)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = max(1, int(args.val_fraction * n))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    print(f"[bridge] train={len(x_tr)} val={len(x_val)}")

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    model = build_model(text_dim, level_sizes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bridge] model params: {n_params/1e6:.2f}M on {device}")

    x_tr_dev, y_tr_dev = x_tr.to(device), y_tr.to(device)
    x_val_dev, y_val_dev = x_val.to(device), y_val.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    history = []
    best_val_avg = -1.0
    best_epoch = -1
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm_e = torch.randperm(len(x_tr_dev), device=device)
        losses = []
        for i in range(0, len(x_tr_dev), args.batch_size):
            batch = perm_e[i: i + args.batch_size]
            xb, yb = x_tr_dev[batch], y_tr_dev[batch]
            logits_list = model(xb)  # list of (B, level_size)
            loss = sum(ce(logits_list[l], yb[:, l]) for l in range(n_levels))
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        # Validation: top-1 accuracy per level
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_dev)
            val_acc_per_level = []
            for l in range(n_levels):
                pred = val_logits[l].argmax(dim=1)
                acc = (pred == y_val_dev[:, l]).float().mean().item()
                val_acc_per_level.append(acc)

        avg_acc = sum(val_acc_per_level) / n_levels
        train_loss = sum(losses) / len(losses)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc_per_level": val_acc_per_level,
            "val_avg_acc": avg_acc,
        })

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"[bridge] epoch {epoch}: train_loss={train_loss:.4f} "
                  f"val_avg_acc={avg_acc:.4f} "
                  f"per_level=[{', '.join(f'{a:.3f}' for a in val_acc_per_level)}]",
                  flush=True)

        if avg_acc > best_val_avg:
            best_val_avg = avg_acc
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "text_dim": text_dim,
                "level_sizes": level_sizes,
                "epoch": epoch,
                "val_avg_acc": avg_acc,
                "val_acc_per_level": val_acc_per_level,
                "text_source_sha256": payload.get("text_source_sha256"),
                "encoder_model": payload.get("model_name"),
            }, output_dir / "best.pt")

    total_time = time.time() - t0
    print(f"[bridge] done. best_epoch={best_epoch} best_val_avg={best_val_avg:.4f} "
          f"wall={total_time:.0f}s", flush=True)

    vocab.save(output_dir / "vocab.json")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump({
            "history": history,
            "best_epoch": best_epoch,
            "best_val_avg_acc": best_val_avg,
            "n_train": len(x_tr),
            "n_val": len(x_val),
            "n_params": n_params,
            "text_dim": text_dim,
            "level_sizes": level_sizes,
            "args": vars(args),
        }, f, indent=2)


def main():
    args = parse_args()
    if args.cmd == "train":
        train_cmd(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
