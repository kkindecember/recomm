"""Semantic bridge MLP v2: text_embedding -> hierarchical id with LLM prior regularization.

Plan v2 (see plan/第十三阶段/GRAM_第十三阶段_CANARD探索计划v0.1.md Section 2):
  "在 v1 基础上加一个 LLM stage:
   - DeepSeek V4 API 单次 first-pass(不做 reflection、不做 multi-perspective)
   - Prompt:cold item text + 5-shot warm examples → LLM 输出 predicted hierarchical id + confidence
   - 训练时加一个 loss:L_llm_prior = KL(MLP output ∥ LLM prediction distribution)
   - 总 loss:L_sup + λ_llm · L_llm_prior(λ_llm 从 0.5 开始调)"

Design:
  - Same MLP architecture as v1 (per-level linear heads)
  - Additional training loss: KL divergence between MLP predictions and LLM prior distributions
  - LLM prior: For each warm item, use generate_llm_priors.py output as soft targets
  - λ_llm: balances supervised CE loss vs LLM regularization

CLI:
    python semantic_bridge_v2.py train \\
        --embeddings artifacts/phase13/embeddings/Toys_sbert.pt \\
        --id-file GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \\
        --cold-items GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt \\
        --llm-priors artifacts/phase13/explore/v2_toys/llm_priors.jsonl \\
        --output-dir artifacts/phase13/explore/v2_toys/mlp \\
        --lambda-llm 0.5 \\
        --epochs 200 --lr 1e-3 --batch-size 512 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

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
    tr.add_argument("--id-file", required=True, help="GRAM item_generative_indexing_*.txt")
    tr.add_argument("--cold-items", required=True, help="cold_split_meta/cold_items.txt")
    tr.add_argument("--llm-priors", required=True, help="output of generate_llm_priors.py (JSONL)")
    tr.add_argument("--output-dir", required=True)
    tr.add_argument("--lambda-llm", type=float, default=0.5, help="Weight for LLM prior KL loss")
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--batch-size", type=int, default=512)
    tr.add_argument("--device", default="cuda:0")
    tr.add_argument("--seed", type=int, default=12345)
    tr.add_argument("--val-fraction", type=float, default=0.1)

    return p.parse_args()


def load_llm_priors(jsonl_path: str, vocab: HierIdVocab):
    """
    Load LLM prior predictions, per-level one-hot with OOV mask.

    Returns:
        priors: Dict[item_id] -> List[distribution per level]  (one-hot when in-vocab)
        masks:  Dict[item_id] -> List[int per level]  (1 if in-vocab, 0 if OOV — skip KL)

    v2_iter2 fix: OOV tokens no longer collapse to uniform distribution (which was
    catastrophic — uniform is maximum-entropy target that erases MLP's discriminative
    power). Instead, mark OOV layers with mask=0 so the training loop can skip KL
    on those levels.
    """
    priors = {}
    masks = {}
    total, oov_per_level = 0, [0] * vocab.n_levels
    n_failed = 0
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            item_id = record["item_id"]
            predicted_tokens = record["predicted_tokens"]

            # API-failed records carry predicted_tokens=null. Skip them entirely so they
            # are never confused with genuine OOV in the diagnostics below.
            if record.get("status") == "failed" or predicted_tokens is None:
                n_failed += 1
                continue

            level_dists, level_mask = [], []
            for lv, token in enumerate(predicted_tokens):
                vocab_size = len(vocab.per_level_token_to_idx[lv])
                dist = [0.0] * vocab_size
                if token in vocab.per_level_token_to_idx[lv]:
                    dist[vocab.per_level_token_to_idx[lv][token]] = 1.0
                    level_mask.append(1)
                else:
                    level_mask.append(0)
                    oov_per_level[lv] += 1
                level_dists.append(dist)

            priors[item_id] = level_dists
            masks[item_id] = level_mask
            total += 1

    print(f"[bridge_v2] Loaded {total} usable LLM priors ({n_failed} API-failed records skipped)")
    if n_failed:
        pct = n_failed / (total + n_failed) * 100
        print(f"[bridge_v2] WARNING: {n_failed} ({pct:.1f}%) priors are API failures — "
              f"those items contribute NO KL supervision")
    for lv, cnt in enumerate(oov_per_level):
        denom = max(total, 1)
        print(f"[bridge_v2]   L{lv+1} OOV: {cnt}/{denom} ({cnt/denom*100:.1f}%) — will be masked out")
    return priors, masks


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
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)

    embeddings_path = Path(args.embeddings).resolve()
    id_file_path = Path(args.id_file).resolve()
    cold_items_path = Path(args.cold_items).resolve()
    llm_priors_path = Path(args.llm_priors).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bridge_v2] embeddings={embeddings_path}")
    print(f"[bridge_v2] id_file={id_file_path}")
    print(f"[bridge_v2] cold_items={cold_items_path}")
    print(f"[bridge_v2] llm_priors={llm_priors_path}")
    print(f"[bridge_v2] lambda_llm={args.lambda_llm}")

    payload = torch.load(embeddings_path, map_location="cpu")
    ids = payload["item_ids"]
    embs = payload["embeddings"]  # (N, D)
    text_dim = embs.shape[1]
    id_to_row = {iid: i for i, iid in enumerate(ids)}

    id_map = read_id_file(id_file_path)
    cold_items = read_item_set(cold_items_path)
    warm_items = [iid for iid in id_map if iid not in cold_items]

    n_levels = len(next(iter(id_map.values())))
    vocab = build_vocab_from_id_file(id_file_path)
    level_sizes = vocab.level_sizes

    print(f"[bridge_v2] text_dim={text_dim}, n_levels={n_levels}, level_sizes={level_sizes}")

    # Load LLM priors with OOV masks (v2_iter2 fix)
    llm_priors, llm_masks = load_llm_priors(str(llm_priors_path), vocab)

    # Filter warm items to those with embeddings
    # LLM priors may not cover all warm items — items without priors use uniform distribution
    warm_with_emb = [
        iid for iid in warm_items
        if iid in id_to_row
    ]
    print(f"[bridge_v2] Warm items with embeddings: {len(warm_with_emb)}")
    print(f"[bridge_v2] Warm items with LLM priors: {sum(1 for iid in warm_with_emb if iid in llm_priors)}")

    # Split train/val
    import random
    random.seed(args.seed)
    random.shuffle(warm_with_emb)
    n_val = int(len(warm_with_emb) * args.val_fraction)
    val_items = warm_with_emb[:n_val]
    train_items = warm_with_emb[n_val:]

    def build_tensors(item_list):
        X = torch.stack([embs[id_to_row[iid]] for iid in item_list])
        Y_list = [[] for _ in range(n_levels)]
        P_list = [[] for _ in range(n_levels)]  # one-hot LLM priors
        M_list = [[] for _ in range(n_levels)]  # per-level mask (1=use KL, 0=skip)

        for iid in item_list:
            tokens = id_map[iid]
            if iid in llm_priors:
                llm_dists = llm_priors[iid]
                llm_mask = llm_masks[iid]
            else:
                # Item without LLM prior: mask=0 for all levels (skip KL entirely)
                llm_dists = [[0.0] * level_sizes[lv] for lv in range(n_levels)]
                llm_mask = [0] * n_levels
            for lv in range(n_levels):
                Y_list[lv].append(vocab.per_level_token_to_idx[lv][tokens[lv]])
                P_list[lv].append(llm_dists[lv])
                M_list[lv].append(llm_mask[lv])

        Y = [torch.tensor(Y_list[lv], dtype=torch.long) for lv in range(n_levels)]
        P = [torch.tensor(P_list[lv], dtype=torch.float32) for lv in range(n_levels)]
        M = [torch.tensor(M_list[lv], dtype=torch.float32) for lv in range(n_levels)]
        return X, Y, P, M

    X_train, Y_train, P_train, M_train = build_tensors(train_items)
    X_val, Y_val, P_val, M_val = build_tensors(val_items)

    print(f"[bridge_v2] train={len(train_items)}, val={len(val_items)}")

    model = build_model(text_dim, level_sizes).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[bridge_v2] Model params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        perm = torch.randperm(len(train_items))
        n_batches = (len(train_items) + args.batch_size - 1) // args.batch_size
        epoch_loss_ce = 0.0
        epoch_loss_kl = 0.0

        for i in range(n_batches):
            batch_idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            x_b = X_train[batch_idx].to(args.device)
            y_b = [Y_train[lv][batch_idx].to(args.device) for lv in range(n_levels)]
            p_b = [P_train[lv][batch_idx].to(args.device) for lv in range(n_levels)]
            m_b = [M_train[lv][batch_idx].to(args.device) for lv in range(n_levels)]

            logits_list = model(x_b)

            # Supervised CE loss (always applied — this is the ground truth signal)
            loss_ce = sum(F.cross_entropy(logits_list[lv], y_b[lv]) for lv in range(n_levels)) / n_levels

            # Masked KL loss (v2_iter2 fix): only compute KL on levels where LLM
            # prediction is in-vocab. OOV levels get mask=0 → contribute 0 to KL.
            # Normalize by the number of active (mask=1) elements to avoid the loss
            # scaling with OOV rate.
            loss_kl_sum = torch.zeros((), device=args.device)
            mask_sum = torch.zeros((), device=args.device)
            for lv in range(n_levels):
                log_p_mlp = F.log_softmax(logits_list[lv], dim=-1)
                # KL per-example (no batchmean reduction so we can mask)
                kl_per_example = F.kl_div(log_p_mlp, p_b[lv], reduction='none').sum(dim=-1)
                loss_kl_sum = loss_kl_sum + (kl_per_example * m_b[lv]).sum()
                mask_sum = mask_sum + m_b[lv].sum()

            loss_kl = loss_kl_sum / mask_sum.clamp(min=1.0)

            loss = loss_ce + args.lambda_llm * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_ce += loss_ce.item()
            epoch_loss_kl += loss_kl.item() if hasattr(loss_kl, 'item') else float(loss_kl)

        epoch_loss_ce /= n_batches
        epoch_loss_kl /= n_batches

        # Validation accuracy
        model.eval()
        with torch.no_grad():
            x_v = X_val.to(args.device)
            y_v = [Y_val[lv].to(args.device) for lv in range(n_levels)]
            logits_v = model(x_v)
            acc_per_level = [
                (logits_v[lv].argmax(dim=-1) == y_v[lv]).float().mean().item()
                for lv in range(n_levels)
            ]
            avg_acc = sum(acc_per_level) / n_levels

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss_ce": epoch_loss_ce,
            "train_loss_kl": epoch_loss_kl,
            "val_avg_acc": avg_acc,
            "val_acc_per_level": acc_per_level,
            "time_s": elapsed
        })

        print(f"[epoch {epoch:3d}] loss_ce={epoch_loss_ce:.4f} loss_kl={epoch_loss_kl:.4f} val_acc={avg_acc:.4f} time={elapsed:.1f}s")

    # Save best model (by val_avg_acc)
    best_epoch = max(history, key=lambda x: x["val_avg_acc"])
    print(f"[bridge_v2] Best val_acc={best_epoch['val_avg_acc']:.4f} at epoch {best_epoch['epoch']}")

    best_model_path = output_dir / "best.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "text_dim": text_dim,
        "level_sizes": level_sizes,
        "epoch": best_epoch["epoch"],
        "val_avg_acc": best_epoch["val_avg_acc"],
        "val_acc_per_level": best_epoch["val_acc_per_level"],
        "lambda_llm": args.lambda_llm,
        "encoder_model": str(Path(args.embeddings).stem),
    }, best_model_path)
    print(f"[bridge_v2] Saved model to {best_model_path}")

    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        data = {
            "n_levels": n_levels,
            "level_sizes": level_sizes,
            "text_dim": text_dim,
            "per_level_idx_to_token": vocab.per_level_idx_to_token
        }
        json.dump(data, f, indent=2, ensure_ascii=False)

    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("[bridge_v2] Done")


def main():
    args = parse_args()
    if args.cmd == "train":
        train_cmd(args)


if __name__ == "__main__":
    main()
