#!/usr/bin/env python3
"""Two-layer residual semantic bridge for the Phase-13 MiniLM screen."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hierarchical_id_utils import build_vocab_from_id_file, read_id_file, read_item_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--embeddings", required=True)
    train.add_argument("--id-file", required=True)
    train.add_argument("--cold-items", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--hidden-dim", type=int, default=768)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--seed", type=int, default=12345)
    train.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args()


def build_model_residual(
    text_dim: int,
    level_sizes: list[int],
    hidden_dim: int,
    dropout: float = 0.0,
):
    import torch.nn as nn

    class ResidualSemanticBridge(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(text_dim, hidden_dim)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, text_dim)
            self.norm = nn.LayerNorm(text_dim)
            self.heads = nn.ModuleList([nn.Linear(text_dim, size) for size in level_sizes])

        def forward(self, x):
            residual = self.fc2(self.dropout(self.activation(self.fc1(x))))
            hidden = self.norm(x + residual)
            return [head(hidden) for head in self.heads]

    return ResidualSemanticBridge()


def train_cmd(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)

    embeddings_path = Path(args.embeddings).resolve()
    id_file_path = Path(args.id_file).resolve()
    cold_items_path = Path(args.cold_items).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(embeddings_path, map_location="cpu")
    item_ids = payload["item_ids"]
    embeddings = payload["embeddings"]
    text_dim = int(embeddings.shape[1])
    id_to_row = {item_id: row for row, item_id in enumerate(item_ids)}
    id_map = read_id_file(id_file_path)
    cold_items = read_item_set(cold_items_path)
    vocab = build_vocab_from_id_file(id_file_path)
    level_sizes = vocab.level_sizes
    n_levels = vocab.n_levels

    warm_rows: list[int] = []
    warm_targets: list[list[int]] = []
    for item_id, tokens in id_map.items():
        if item_id in cold_items or item_id not in id_to_row:
            continue
        warm_rows.append(id_to_row[item_id])
        warm_targets.append(vocab.encode(tokens))
    if not warm_rows:
        raise RuntimeError("No warm items have both embeddings and hierarchical IDs")

    x = embeddings[warm_rows]
    y = torch.tensor(warm_targets, dtype=torch.long)
    n_items = len(x)
    permutation = torch.randperm(
        n_items, generator=torch.Generator().manual_seed(args.seed)
    )
    n_val = max(1, int(args.val_fraction * n_items))
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model = build_model_residual(
        text_dim, level_sizes, args.hidden_dim, args.dropout
    ).to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    cross_entropy = nn.CrossEntropyLoss()

    print(f"[residual] embeddings={embeddings_path}")
    print(
        f"[residual] text_dim={text_dim} hidden_dim={args.hidden_dim} "
        f"dropout={args.dropout} weight_decay={args.weight_decay}"
    )
    print(f"[residual] levels={level_sizes} train={len(x_train)} val={len(x_val)}")
    print(f"[residual] params={n_params} device={device}")

    history: list[dict] = []
    best_val = -1.0
    best_hscore = -1.0
    best_prefix: list[float] = []
    best_epoch = -1
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_permutation = torch.randperm(len(x_train), device=device)
        losses: list[float] = []
        for offset in range(0, len(x_train), args.batch_size):
            batch = epoch_permutation[offset : offset + args.batch_size]
            logits = model(x_train[batch])
            loss = sum(
                cross_entropy(logits[level], y_train[batch, level])
                for level in range(n_levels)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_predictions = torch.stack(
                [logits.argmax(dim=1) for logits in val_logits], dim=1
            )
            val_matches = val_predictions == y_val
            per_level = [
                float(
                    (val_logits[level].argmax(dim=1) == y_val[:, level])
                    .float()
                    .mean()
                    .item()
                )
                for level in range(n_levels)
            ]
            prefix_accuracy = [
                float(val_matches[:, :length].all(dim=1).float().mean().item())
                for length in range(1, n_levels + 1)
            ]
        val_average = sum(per_level) / n_levels
        hscore = (
            0.5 * prefix_accuracy[1]
            + 0.3 * prefix_accuracy[2]
            + 0.2 * prefix_accuracy[-1]
        )
        train_loss = sum(losses) / len(losses)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc_per_level": per_level,
                "val_avg_acc": val_average,
                "val_prefix_accuracy": prefix_accuracy,
                "val_hscore": hscore,
            }
        )

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"[residual] epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_avg_acc={val_average:.4f} hscore={hscore:.4f} "
                f"prefix2={prefix_accuracy[1]:.4f} "
                f"prefix3={prefix_accuracy[2]:.4f} exact={prefix_accuracy[-1]:.4f} "
                f"per_level=[{', '.join(f'{value:.3f}' for value in per_level)}]",
                flush=True,
            )

        if hscore > best_hscore or (
            hscore == best_hscore and val_average > best_val
        ):
            best_val = val_average
            best_hscore = hscore
            best_prefix = prefix_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": "residual_v1",
                    "text_dim": text_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "weight_decay": args.weight_decay,
                    "level_sizes": level_sizes,
                    "epoch": epoch,
                    "val_avg_acc": val_average,
                    "val_acc_per_level": per_level,
                    "val_prefix_accuracy": prefix_accuracy,
                    "val_hscore": hscore,
                    "selection_metric": "0.5*prefix@2+0.3*prefix@3+0.2*exact",
                    "text_source_sha256": payload.get("text_source_sha256"),
                    "encoder_model": payload.get("model_name"),
                },
                output_dir / "best.pt",
            )

    wall_seconds = time.time() - start_time
    vocab.save(output_dir / "vocab.json")
    (output_dir / "training_history.json").write_text(
        json.dumps(
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_val_avg_acc": best_val,
                "best_val_hscore": best_hscore,
                "best_val_prefix_accuracy": best_prefix,
                "n_train": len(x_train),
                "n_val": len(x_val),
                "n_params": n_params,
                "text_dim": text_dim,
                "hidden_dim": args.hidden_dim,
                "level_sizes": level_sizes,
                "args": vars(args),
                "wall_seconds": wall_seconds,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[residual] done best_epoch={best_epoch} best_hscore={best_hscore:.6f} "
        f"best_val={best_val:.6f} "
        f"wall={wall_seconds:.1f}s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        train_cmd(args)


if __name__ == "__main__":
    main()
