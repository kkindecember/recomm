"""Precompute item text embeddings using an HF encoder (mean-pooled).

Writes a torch .pt with:
  {
    "item_ids": list[str],                       # order-preserving
    "embeddings": torch.FloatTensor (N, D),      # mean-pooled last hidden
    "model_name": str,
    "device_used": str,
    "max_seq_len": int,
    "text_source_sha256": str,                   # sha256 of item_plain_text.txt
  }

Uses transformers directly (no sentence-transformers dep). Mean pooling over
attention-mask-weighted last_hidden_state, matching sentence-BERT convention.

CLI:
    python precompute_item_embeddings.py \\
        --item-text /path/to/item_plain_text.txt \\
        --output artifacts/phase13/embeddings/beauty_sbert.pt \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --device cuda:0 \\
        --batch-size 32 \\
        --max-seq-len 256

Runs offline (TRANSFORMERS_OFFLINE=1 respected) if the model is in HF cache;
otherwise downloads. Time on CPU is ~5-30 min for 12k items depending on
model size; GPU is minutes.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--item-text", required=True,
                   help="Path to item_plain_text.txt")
    p.add_argument("--output", required=True,
                   help="Output .pt path")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="HF model id (default: MiniLM, small & fast)")
    p.add_argument("--device", default="cuda:0",
                   help="cuda:N or cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=256,
                   help="Truncate item text to this many tokens")
    p.add_argument("--fp16", action="store_true",
                   help="Encode in fp16 (GPU only)")
    return p.parse_args()


def mean_pool(last_hidden, attention_mask):
    """(B, L, D), (B, L) -> (B, D). Sentence-BERT-style attention-weighted mean."""
    import torch
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def read_items(path: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            i = line.find(" ")
            if i < 0:
                ids.append(line)
                texts.append("")
            else:
                ids.append(line[:i])
                texts.append(line[i + 1:])
    return ids, texts


def main():
    args = parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModel

    item_text_path = Path(args.item_text).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not item_text_path.exists():
        print(f"ERROR: item_text not found: {item_text_path}", file=sys.stderr)
        sys.exit(2)

    print(f"[embed] item_text={item_text_path}")
    print(f"[embed] model={args.model}")
    print(f"[embed] device={args.device}")

    ids, texts = read_items(item_text_path)
    print(f"[embed] {len(ids)} items to encode")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval().to(device)
    if args.fp16 and device.type == "cuda":
        model = model.half()

    all_embs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i: i + args.batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc.attention_mask)
            all_embs.append(pooled.float().cpu())
            if (i // args.batch_size) % 20 == 0:
                elapsed = time.time() - t0
                done = i + len(batch_texts)
                rate = done / max(elapsed, 1e-6)
                eta = (len(texts) - done) / max(rate, 1e-6)
                print(f"[embed] {done}/{len(texts)} "
                      f"({rate:.1f} items/s, ETA {eta:.0f}s)", flush=True)

    embeddings = torch.cat(all_embs, dim=0)
    assert embeddings.shape[0] == len(ids)
    print(f"[embed] final shape: {tuple(embeddings.shape)}")

    payload = {
        "item_ids": ids,
        "embeddings": embeddings,
        "model_name": args.model,
        "device_used": str(device),
        "max_seq_len": args.max_seq_len,
        "text_source_sha256": sha256_of_file(item_text_path),
    }
    torch.save(payload, output_path)
    print(f"[embed] wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
