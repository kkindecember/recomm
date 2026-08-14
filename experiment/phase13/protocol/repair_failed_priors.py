"""One-off: re-run the LLM prior calls that failed with 402 during v2_iter2 prep.

v2_iter2 ran out of DeepSeek balance mid-prep. Failed calls were written as
per-level "<unk>" with confidence 1.0, so 47.5% (Beauty) / 32.6% (Toys) of warm
items silently contributed no KL supervision. This repairs only those records and
leaves the successful ones untouched, so we can retrain the MLP under full
coverage and check whether the KL term still hurts val_acc.

Writes *_repaired.jsonl alongside the originals — the v2_iter2 artifacts stay
intact because the report cites them.
"""
import argparse
import json
import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hierarchical_id_utils as hid_utils
from deepseek_client import DeepSeekClient
from generate_llm_priors_v2iter2 import (
    build_few_shot_pool, build_per_level_vocab, build_prompt_v2iter2,
    load_item_text, parse_llm_response,
)
from llm_cache import LLMCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_failed(record):
    """True for both the old (<unk>) and new (status=failed) failure encodings."""
    if record.get("status") == "failed":
        return True
    pt = record.get("predicted_tokens")
    if pt is None:
        return True
    return bool(pt) and all(t == "<unk>" for t in pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--warm-items", required=True)
    ap.add_argument("--item-text", required=True)
    ap.add_argument("--source-id-file", required=True)
    ap.add_argument("--cache-db", default="artifacts/phase13/llm_cache.db")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--top-n-per-level", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=4)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.priors_jsonl)]
    failed_idx = [i for i, r in enumerate(records) if is_failed(r)]
    logger.info(f"{args.priors_jsonl}: {len(records)} records, {len(failed_idx)} failed "
                f"({len(failed_idx)/len(records)*100:.1f}%)")
    if not failed_idx:
        logger.info("Nothing to repair.")
        return 0

    random.seed(args.seed)
    warm_items = sorted(hid_utils.read_item_set(args.warm_items))
    item_text = load_item_text(args.item_text)
    id_map = hid_utils.read_id_file(args.source_id_file)
    num_levels = len(next(iter(id_map.values())))
    per_level_vocab = build_per_level_vocab(id_map, args.top_n_per_level)
    few_shot_pool = build_few_shot_pool(warm_items, id_map, item_text, pool_size=200)

    # Sample few-shot sets up front: random is not thread-safe, and doing it here
    # keeps the whole repair reproducible from --seed.
    prompts = {}
    for i in failed_idx:
        iid = records[i]["item_id"]
        text = records[i].get("text") or item_text.get(iid, "")
        shots = random.sample(few_shot_pool, min(args.num_shots, len(few_shot_pool)))
        prompts[i] = (iid, text, build_prompt_v2iter2(
            iid, text, shots, per_level_vocab, args.num_shots, num_levels))

    cache = LLMCache(args.cache_db)
    client = DeepSeekClient()
    counts = {"ok": 0, "cached": 0, "failed": 0}
    lock = threading.Lock()

    def work(i):
        iid, text, prompt = prompts[i]
        cached = cache.get(args.model, prompt)
        if cached:
            tokens, conf = parse_llm_response(cached["text"], num_levels)
            with lock:
                counts["cached"] += 1
            return i, {"item_id": iid, "text": text, "predicted_tokens": tokens,
                       "confidence": conf, "cached": True, "status": "ok"}

        last = None
        for attempt in range(args.max_retries):
            try:
                resp = client.chat_completion(
                    model=args.model, messages=[{"role": "user", "content": prompt}],
                    temperature=0.7, max_tokens=200)
                txt = resp["choices"][0]["message"]["content"]
                cache.put(args.model, prompt, {"text": txt,
                                               "usage": resp.get("usage", {}),
                                               "model": args.model})
                tokens, conf = parse_llm_response(txt, num_levels)
                with lock:
                    counts["ok"] += 1
                    done = counts["ok"] + counts["cached"] + counts["failed"]
                    if done % 200 == 0:
                        logger.info(f"  {done}/{len(failed_idx)} {counts}")
                return i, {"item_id": iid, "text": text, "predicted_tokens": tokens,
                           "confidence": conf, "cached": False, "status": "ok"}
            except Exception as e:
                last = e
                time.sleep(2 ** attempt)
        logger.error(f"still failing after {args.max_retries} retries: {iid}: {last}")
        with lock:
            counts["failed"] += 1
        return i, {"item_id": iid, "text": text, "predicted_tokens": None,
                   "confidence": 0.0, "cached": False, "status": "failed",
                   "error": str(last)[:200]}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in pool.map(work, failed_idx):
            records[i] = rec

    with open(args.output_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    still_failed = sum(1 for r in records if is_failed(r))
    logger.info(f"Repair done: {counts}")
    logger.info(f"Wrote {args.output_jsonl}: {len(records)} records, "
                f"{still_failed} still failed ({still_failed/len(records)*100:.1f}%)")
    return 1 if still_failed else 0


if __name__ == "__main__":
    sys.exit(main())
