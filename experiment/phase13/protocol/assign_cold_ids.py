"""Given a trained semantic bridge MLP + item embeddings, assign hierarchical
ids to cold items and write a merged id file consumable by GRAM.

Output format matches GRAM's item_generative_indexing_*.txt (see
hierarchical_id_utils.parse_id_line / format_id_line).

Strategy:
  - Warm items keep their original hierarchical id (from source id file)
  - Cold items get MLP-predicted id (argmax per level)
  - Line order preserved from the source id file so GRAM's downstream code
    that assumes a particular ordering stays happy

Naming convention for output:
  <source_id_file_dir>/item_generative_indexing_<hierarchical_id_type>_v1_mlpcold.txt

The runner then passes `--hierarchical_id_type <original>_v1_mlpcold` to GRAM.

CLI:
    python assign_cold_ids.py \\
        --embeddings artifacts/phase13/embeddings/beauty_sbert.pt \\
        --mlp artifacts/phase13/explore/v1_beauty/mlp/best.pt \\
        --source-id-file GRAM/rec_datasets/Beauty/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt \\
        --cold-items GRAM/rec_datasets/Beauty_cold50/cold_split_meta/cold_items.txt \\
        --output-id-file GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split_v1_mlpcold.txt \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hierarchical_id_utils import (
    HierIdVocab, format_id_line, infer_n_levels, parse_id_line, read_id_file,
    read_item_set,
)
from semantic_bridge import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--mlp", required=True, help="best.pt from semantic_bridge train")
    p.add_argument("--source-id-file", required=True,
                   help="Original GRAM item_generative_indexing_*.txt (full items)")
    p.add_argument("--cold-items", required=True,
                   help="cold_items.txt (targets for MLP prediction)")
    p.add_argument("--output-id-file", required=True)
    p.add_argument("--vocab-json", default=None,
                   help="Optional vocab.json emitted by semantic_bridge; "
                        "if not given, rebuild from source-id-file")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--report", default=None,
                   help="Optional JSON report of stats + example predictions")
    return p.parse_args()


def main():
    args = parse_args()
    import torch

    src_id = Path(args.source_id_file).resolve()
    output_id = Path(args.output_id_file).resolve()
    output_id.parent.mkdir(parents=True, exist_ok=True)

    # Load vocab (must match what MLP was trained on)
    if args.vocab_json:
        vocab = HierIdVocab.load(Path(args.vocab_json))
    else:
        from hierarchical_id_utils import build_vocab_from_id_file
        vocab = build_vocab_from_id_file(src_id)

    ckpt = torch.load(args.mlp, map_location="cpu")
    text_dim = ckpt["text_dim"]
    level_sizes = ckpt["level_sizes"]
    if level_sizes != vocab.level_sizes:
        print(f"WARN: MLP level_sizes {level_sizes} != vocab {vocab.level_sizes}. "
              f"Vocab mismatch → predictions may be invalid.", file=sys.stderr)

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    # v3 checkpoints carry a trunk + per-level projections; v1/v2 are bare heads.
    if ckpt.get("arch") == "v3":
        from semantic_bridge_v3 import build_model_v3
        model = build_model_v3(text_dim, level_sizes,
                               ckpt["hidden_dim"], ckpt["proj_dim"])
    else:
        model = build_model(text_dim, level_sizes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    embeddings_payload = torch.load(args.embeddings, map_location="cpu")
    ids = embeddings_payload["item_ids"]
    embs = embeddings_payload["embeddings"]
    id_to_row = {iid: i for i, iid in enumerate(ids)}

    cold_items = read_item_set(Path(args.cold_items))
    source_id_map = read_id_file(src_id)

    # Compute MLP predictions for cold items with embeddings
    cold_with_emb = [iid for iid in cold_items if iid in id_to_row]
    if not cold_with_emb:
        print("ERROR: no cold items have embeddings", file=sys.stderr)
        sys.exit(2)

    rows = torch.tensor([id_to_row[iid] for iid in cold_with_emb])
    x = embs[rows].to(device)
    with torch.no_grad():
        out = model(x)
        logits_list = out[0] if isinstance(out, tuple) else out  # v3 returns (logits, h)
        preds = torch.stack([lg.argmax(dim=1) for lg in logits_list], dim=1)
    preds = preds.cpu().tolist()

    predicted_id_map: dict[str, list[str]] = {}
    for i, iid in enumerate(cold_with_emb):
        predicted_id_map[iid] = vocab.decode(preds[i])

    # Merge: warm → source id (keep full tokens incl. collision suffix),
    # cold → MLP prediction, missing cold → source id (fallback so GRAM
    # doesn't crash on missing ids).
    # NOTE: parse_id_line is called without n_levels so warm rows with
    # collision-disambiguation suffix (e.g. 5 semantic tokens + '|0') keep
    # all their tokens. Passing n_levels=5 here silently truncated 862/11924
    # collision items in the Toys_cold50 baseline, artificially merging warm
    # items into shared ids and inflating warm hit rate (fixed 2026-08-10).
    merged: dict[str, list[str]] = {}
    n_warm = 0
    n_cold_predicted = 0
    n_cold_fallback = 0
    n_warm_with_collision = 0
    with open(src_id) as f:
        order: list[str] = []
        for line in f:
            if not line.strip():
                continue
            item_id, tokens = parse_id_line(line)
            order.append(item_id)
            if item_id in cold_items:
                if item_id in predicted_id_map:
                    merged[item_id] = predicted_id_map[item_id]
                    n_cold_predicted += 1
                else:
                    merged[item_id] = tokens
                    n_cold_fallback += 1
            else:
                merged[item_id] = tokens
                n_warm += 1
                if len(tokens) > vocab.n_levels:
                    n_warm_with_collision += 1

    with open(output_id, "w") as f:
        for iid in order:
            f.write(format_id_line(iid, merged[iid]) + "\n")

    print(f"[assign] warm items (source id kept): {n_warm}")
    print(f"[assign]   of which have collision suffix (>{vocab.n_levels} tokens): {n_warm_with_collision}")
    print(f"[assign] cold items (MLP predicted): {n_cold_predicted}")
    print(f"[assign] cold items (fallback to source id, no embedding): {n_cold_fallback}")
    print(f"[assign] wrote {output_id}")

    if args.report:
        # Sample 10 cold predictions for eyeballing
        samples = []
        for iid in cold_with_emb[:10]:
            samples.append({
                "item_id": iid,
                "source_id": source_id_map[iid],
                "predicted_id": predicted_id_map[iid],
                "match_positions": [
                    i for i in range(len(source_id_map[iid]))
                    if source_id_map[iid][i] == predicted_id_map[iid][i]
                ],
            })
        report = {
            "n_warm": n_warm,
            "n_cold_predicted": n_cold_predicted,
            "n_cold_fallback": n_cold_fallback,
            "output_id_file": str(output_id),
            "mlp_checkpoint": str(args.mlp),
            "mlp_val_avg_acc": ckpt.get("val_avg_acc"),
            "mlp_val_acc_per_level": ckpt.get("val_acc_per_level"),
            "encoder_model": ckpt.get("encoder_model"),
            "sample_predictions": samples,
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[assign] wrote report to {args.report}")


if __name__ == "__main__":
    main()
