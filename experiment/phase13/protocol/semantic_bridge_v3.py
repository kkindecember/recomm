"""Semantic bridge MLP v3: text embedding -> hierarchical id, with per-level
hierarchical contrastive alignment.

Phase 13 CANARD v3. Builds on **v1** (plain CE), NOT v2 — the v2 LLM-prior KL
component was abandoned after failing twice on both domains.

  v1 : L = L_CE
  v3 : L = L_CE + Σ_l λ_l · L_align_l

L_align_l is a supervised InfoNCE over a per-level projection of a shared trunk:
  positives     items sharing the level-l token
  hard negatives items sharing the level-(l-1) token but differing at level l

Motivation (from the v2 post-mortem): semantic signal agrees with GRAM's
SASRec-derived clusters at shallow levels (L1 44-60%) and barely at all deeper
(L3+ 3.5-16%). So λ_l is expected to be useful only for small l — hence per-level
weights rather than one global λ.

ARCHITECTURE NOTE: alignment needs a shared representation to act on, so v3 adds
a trunk (v1/v2 were independent linear heads on the frozen text embedding, with
no shared hidden layer). That is a confound versus v1, so always run the
`--align-weights 0,0,...` control to separate "trunk helped" from
"alignment helped".

CLI:
    python semantic_bridge_v3.py train \\
        --embeddings artifacts/phase13/embeddings/Beauty_sbert.pt \\
        --id-file GRAM/rec_datasets/Beauty/item_generative_indexing_...split.txt \\
        --cold-items GRAM/rec_datasets/Beauty_cold50/cold_split_meta/cold_items.txt \\
        --output-dir artifacts/phase13/explore/v3_beauty/mlp \\
        --align-weights 1.0,0.5,0,0,0,0,0 --tau 0.07
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hierarchical_id_utils import build_vocab_from_id_file, read_id_file, read_item_set


def parse_args():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    tr = sp.add_parser("train")
    tr.add_argument("--embeddings", required=True)
    tr.add_argument("--id-file", required=True)
    tr.add_argument("--cold-items", required=True)
    tr.add_argument("--output-dir", required=True)
    tr.add_argument("--align-weights", required=True,
                    help="Comma-separated per-level λ_l, one per level. "
                         "All zeros = architecture control (no alignment).")
    tr.add_argument("--align-scale", type=float, default=1.0,
                    help="Global multiplier applied on top of --align-weights")
    tr.add_argument("--tau", type=float, default=0.07, help="InfoNCE temperature")
    tr.add_argument("--hidden-dim", type=int, default=512)
    tr.add_argument("--proj-dim", type=int, default=128)
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--batch-size", type=int, default=512)
    tr.add_argument("--pk-sampler", action="store_true",
                    help="Build a dedicated alignment batch per level by sampling whole "
                         "same-parent groups (P groups x K items). Without this, deep "
                         "levels almost never get a valid triplet in a random batch "
                         "(measured: L4/L5 ~1.5% anchor coverage at batch=512).")
    tr.add_argument("--pk-groups", type=int, default=64, help="P: parent groups per align batch")
    tr.add_argument("--pk-per-group", type=int, default=8, help="K: items sampled per group")
    tr.add_argument("--flat-align", type=float, default=0.0,
                    help="Flat alignment (plan v3 iteration option 3): a single InfoNCE "
                         "on the trunk over the full leaf id, ignoring hierarchy. "
                         "Mutually exclusive with nonzero --align-weights.")
    tr.add_argument("--device", default="cuda:0")
    tr.add_argument("--seed", type=int, default=12345)
    tr.add_argument("--val-fraction", type=float, default=0.1)

    return p.parse_args()


def build_model_v3(text_dim: int, level_sizes: list, hidden_dim: int, proj_dim: int):
    import torch.nn as nn

    class SemanticBridgeV3(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )
            self.heads = nn.ModuleList([nn.Linear(hidden_dim, s) for s in level_sizes])
            self.projs = nn.ModuleList([nn.Linear(hidden_dim, proj_dim) for _ in level_sizes])

        def forward(self, x):
            h = self.trunk(x)
            return [head(h) for head in self.heads], h

        def project(self, h, lv):
            import torch.nn.functional as F
            return F.normalize(self.projs[lv](h), dim=-1)

    return SemanticBridgeV3()


def hierarchical_infonce(z, labels_l, labels_parent, tau):
    """Supervised InfoNCE restricted to same-parent hard negatives.

    z              (B, D) L2-normalised projections
    labels_l       (B,)   level-l cluster id
    labels_parent  (B,)   level-(l-1) cluster id; all-zeros for l=0 (no parent,
                          so every other item is an admissible negative)

    Anchors with no in-batch positive, or no same-parent negative, are skipped —
    they carry no contrastive signal. Returns (loss, n_valid_anchors).
    """
    import torch

    B = z.shape[0]
    if B < 2:
        return z.sum() * 0.0, 0

    sim = (z @ z.t()) / tau
    eye = torch.eye(B, dtype=torch.bool, device=z.device)

    same_l = labels_l.unsqueeze(0) == labels_l.unsqueeze(1)
    same_parent = labels_parent.unsqueeze(0) == labels_parent.unsqueeze(1)

    pos_mask = same_l & ~eye
    # Hard negatives: share the parent cluster but split at this level.
    neg_mask = same_parent & ~same_l

    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not valid.any():
        return z.sum() * 0.0, 0

    # Softmax denominator spans positives + hard negatives only.
    cand_mask = pos_mask | neg_mask
    neg_inf = torch.finfo(sim.dtype).min
    logits = sim.masked_fill(~cand_mask, neg_inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    per_anchor = -(log_prob * pos_mask).sum(dim=1) / pos_count

    loss = per_anchor[valid].mean()
    return loss, int(valid.sum().item())


def train_cmd(args):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)

    embeddings_path = Path(args.embeddings).resolve()
    id_file_path = Path(args.id_file).resolve()
    cold_items_path = Path(args.cold_items).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(embeddings_path, map_location="cpu")
    ids = payload["item_ids"]
    embs = payload["embeddings"]
    text_dim = embs.shape[1]
    id_to_row = {iid: i for i, iid in enumerate(ids)}

    id_map = read_id_file(id_file_path)
    cold_items = read_item_set(cold_items_path)
    warm_items = [iid for iid in id_map if iid not in cold_items]

    n_levels = len(next(iter(id_map.values())))
    vocab = build_vocab_from_id_file(id_file_path)
    level_sizes = vocab.level_sizes

    align_w = [float(x) * args.align_scale for x in args.align_weights.split(",")]
    if len(align_w) != n_levels:
        raise SystemExit(
            f"--align-weights has {len(align_w)} entries but the id file has {n_levels} levels")

    print(f"[bridge_v3] embeddings={embeddings_path}")
    print(f"[bridge_v3] id_file={id_file_path}")
    print(f"[bridge_v3] text_dim={text_dim}, n_levels={n_levels}, level_sizes={level_sizes}")
    print(f"[bridge_v3] hidden_dim={args.hidden_dim}, proj_dim={args.proj_dim}, tau={args.tau}")
    print(f"[bridge_v3] align_weights={align_w}")
    print(f"[bridge_v3] flat_align={args.flat_align}")
    if args.flat_align and any(w != 0 for w in align_w):
        raise SystemExit("--flat-align and nonzero --align-weights are mutually exclusive")
    if args.flat_align:
        print("[bridge_v3] MODE: flat alignment (single InfoNCE on full leaf id)")
    elif all(w == 0 for w in align_w):
        print("[bridge_v3] MODE: architecture control (no alignment, CE only)")

    warm_with_emb = [iid for iid in warm_items if iid in id_to_row]
    print(f"[bridge_v3] Warm items with embeddings: {len(warm_with_emb)}")

    # Same split procedure and seed as v1/v2 so val sets are comparable.
    import random
    random.seed(args.seed)
    random.shuffle(warm_with_emb)
    n_val = int(len(warm_with_emb) * args.val_fraction)
    val_items = warm_with_emb[:n_val]
    train_items = warm_with_emb[n_val:]

    def build_tensors(item_list):
        X = torch.stack([embs[id_to_row[iid]] for iid in item_list])
        Y = [[] for _ in range(n_levels)]
        for iid in item_list:
            tokens = id_map[iid]
            for lv in range(n_levels):
                Y[lv].append(vocab.per_level_token_to_idx[lv][tokens[lv]])
        return X, [torch.tensor(Y[lv], dtype=torch.long) for lv in range(n_levels)]

    X_train, Y_train = build_tensors(train_items)
    X_val, Y_val = build_tensors(val_items)
    print(f"[bridge_v3] train={len(train_items)}, val={len(val_items)}")

    # Flat alignment labels: distinct id for each full token path.
    flat_ids = torch.zeros(len(train_items), dtype=torch.long)
    if args.flat_align:
        path_to_id = {}
        for row, iid in enumerate(train_items):
            key = tuple(id_map[iid])
            flat_ids[row] = path_to_id.setdefault(key, len(path_to_id))
        sizes = torch.bincount(flat_ids)
        print(f"[bridge_v3] flat labels: {len(path_to_id)} distinct paths over "
              f"{len(train_items)} items; {int((sizes >= 2).sum())} paths have >=2 members "
              f"(only those can form a positive pair)")

    model = build_model_v3(text_dim, level_sizes, args.hidden_dim, args.proj_dim).to(args.device)
    print(f"[bridge_v3] Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    active_levels = [lv for lv in range(n_levels) if align_w[lv] != 0.0]

    # PK sampler index: for each level, group train rows by (parent token, own token)
    # so we can draw whole same-parent groups that are guaranteed to contain both a
    # positive and a hard negative.
    pk_index = {}
    if args.pk_sampler and active_levels:
        from collections import defaultdict
        for lv in active_levels:
            parents = defaultdict(lambda: defaultdict(list))
            for row in range(len(train_items)):
                own = int(Y_train[lv][row])
                par = int(Y_train[lv - 1][row]) if lv > 0 else 0
                parents[par][own].append(row)
            # Keep only parents that split into >=2 children (else no hard negative)
            usable = {p: ch for p, ch in parents.items() if len(ch) >= 2}
            pk_index[lv] = usable
            print(f"[bridge_v3] PK index L{lv+1}: {len(usable)} usable parent groups "
                  f"(of {len(parents)} total)")

    def sample_pk_batch(lv):
        """Draw P parent groups x up to K items, all from one level's index."""
        usable = pk_index.get(lv) or {}
        if not usable:
            return None
        pkeys = list(usable.keys())
        chosen = random.sample(pkeys, min(args.pk_groups, len(pkeys)))
        rows = []
        for p in chosen:
            children = usable[p]
            # take >=2 distinct children so the group carries a hard negative
            ckeys = list(children.keys())
            random.shuffle(ckeys)
            budget = args.pk_per_group
            for c in ckeys:
                if budget <= 0:
                    break
                pool = children[c]
                take = min(len(pool), max(1, budget // 2))
                rows.extend(random.sample(pool, take))
                budget -= take
        return torch.tensor(rows, dtype=torch.long) if len(rows) >= 4 else None

    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        perm = torch.randperm(len(train_items))
        n_batches = (len(train_items) + args.batch_size - 1) // args.batch_size
        ep_ce = 0.0
        ep_align = 0.0
        anchors_per_level = [0] * n_levels

        for i in range(n_batches):
            bidx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            x_b = X_train[bidx].to(args.device)
            y_b = [Y_train[lv][bidx].to(args.device) for lv in range(n_levels)]

            logits_list, h = model(x_b)
            loss_ce = sum(F.cross_entropy(logits_list[lv], y_b[lv])
                          for lv in range(n_levels)) / n_levels

            loss_align = torch.zeros((), device=args.device)
            for lv in active_levels:
                if args.pk_sampler:
                    # Dedicated same-parent batch: the CE batch stays as-is, the
                    # alignment term gets its own forward pass on grouped rows.
                    pk_rows = sample_pk_batch(lv)
                    if pk_rows is None:
                        continue
                    x_pk = X_train[pk_rows].to(args.device)
                    _, h_pk = model(x_pk)
                    z = model.project(h_pk, lv)
                    y_lv = Y_train[lv][pk_rows].to(args.device)
                    parent = (Y_train[lv - 1][pk_rows].to(args.device) if lv > 0
                              else torch.zeros_like(y_lv))
                else:
                    z = model.project(h, lv)
                    y_lv = y_b[lv]
                    parent = y_b[lv - 1] if lv > 0 else torch.zeros_like(y_b[lv])
                l_lv, n_valid = hierarchical_infonce(z, y_lv, parent, args.tau)
                loss_align = loss_align + align_w[lv] * l_lv
                anchors_per_level[lv] += n_valid

            if args.flat_align:
                # No hierarchy: positives share the full leaf id, negatives are
                # everything else in the batch. Uses level 0's projection head.
                z = model.project(h, 0)
                flat_lbl = flat_ids[bidx].to(args.device)
                no_parent = torch.zeros_like(flat_lbl)
                l_flat, n_valid = hierarchical_infonce(z, flat_lbl, no_parent, args.tau)
                loss_align = loss_align + args.flat_align * l_flat
                anchors_per_level[0] += n_valid

            loss = loss_ce + loss_align

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_ce += loss_ce.item()
            ep_align += float(loss_align)

        ep_ce /= n_batches
        ep_align /= n_batches

        model.eval()
        with torch.no_grad():
            logits_v, _ = model(X_val.to(args.device))
            y_v = [Y_val[lv].to(args.device) for lv in range(n_levels)]
            acc_per_level = [(logits_v[lv].argmax(dim=-1) == y_v[lv]).float().mean().item()
                             for lv in range(n_levels)]
            avg_acc = sum(acc_per_level) / n_levels

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss_ce": ep_ce,
            "train_loss_align": ep_align,
            "val_avg_acc": avg_acc,
            "val_acc_per_level": acc_per_level,
            # How many anchors actually had both a positive and a hard negative.
            # Near-zero at deep levels means the alignment term is vacuous there.
            "align_anchors_per_level": anchors_per_level,
            "time_s": elapsed,
        })
        print(f"[epoch {epoch:3d}] loss_ce={ep_ce:.4f} loss_align={ep_align:.4f} "
              f"val_acc={avg_acc:.4f} time={elapsed:.1f}s")

    best = max(history, key=lambda x: x["val_avg_acc"])
    print(f"[bridge_v3] Best val_acc={best['val_avg_acc']:.4f} at epoch {best['epoch']}")
    print(f"[bridge_v3] per-level@best: " +
          ", ".join(f"L{i+1}={v:.3f}" for i, v in enumerate(best["val_acc_per_level"])))

    n_train_anchor_slots = len(train_items)
    print("[bridge_v3] alignment anchor coverage (last epoch, per level):")
    for lv in range(n_levels):
        c = history[-1]["align_anchors_per_level"][lv]
        tag = "" if align_w[lv] else "  (inactive)"
        print(f"[bridge_v3]   L{lv+1}: {c}/{n_train_anchor_slots} "
              f"({c/max(n_train_anchor_slots,1)*100:.1f}%){tag}")

    torch.save({
        "state_dict": model.state_dict(),
        "arch": "v3",
        "text_dim": text_dim,
        "level_sizes": level_sizes,
        "hidden_dim": args.hidden_dim,
        "proj_dim": args.proj_dim,
        "epoch": best["epoch"],
        "val_avg_acc": best["val_avg_acc"],
        "val_acc_per_level": best["val_acc_per_level"],
        "align_weights": align_w,
        "tau": args.tau,
        "encoder_model": str(Path(args.embeddings).stem),
    }, output_dir / "best.pt")

    with open(output_dir / "vocab.json", "w") as f:
        json.dump({
            "n_levels": n_levels,
            "level_sizes": level_sizes,
            "text_dim": text_dim,
            "per_level_idx_to_token": vocab.per_level_idx_to_token,
        }, f, indent=2, ensure_ascii=False)

    with open(output_dir / "training_history.json", "w") as f:
        json.dump({"history": history}, f, indent=2)

    print("[bridge_v3] Done")


def main():
    args = parse_args()
    if args.cmd == "train":
        train_cmd(args)


if __name__ == "__main__":
    main()
