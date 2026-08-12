"""
Generate LLM prior predictions for hierarchical IDs (Phase 13 v2_iter2).

Fixed from iter1:
  - Prompt now includes per-level vocabulary constraint
  - LLM is instructed to select tokens ONLY from the provided vocab per level
  - Reduces OOV rate from ~61% to <10%

Uses DeepSeek API with 5-shot warm examples + explicit vocab constraint.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from llm_cache import LLMCache
from deepseek_client import DeepSeekClient
import hierarchical_id_utils as hid_utils


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_item_text(item_text_file: str) -> Dict[str, str]:
    item_text = {}
    with open(item_text_file) as f:
        for line in f:
            parts = line.rstrip('\n').split(' ', 1)
            if len(parts) == 2:
                item_text[parts[0]] = parts[1]
    logger.info(f"Loaded {len(item_text)} item texts")
    return item_text


def build_per_level_vocab(id_map: Dict[str, List[str]], top_n_per_level: int = 500) -> List[List[str]]:
    """
    Build per-level vocabulary from ground truth IDs.

    For levels with vocab_size > top_n, return top-N most frequent tokens.
    For small levels, return all tokens.
    """
    from collections import Counter
    n_levels = len(next(iter(id_map.values())))
    per_level_counter = [Counter() for _ in range(n_levels)]
    for tokens in id_map.values():
        for l, t in enumerate(tokens[:n_levels]):
            per_level_counter[l][t] += 1

    per_level_vocab = []
    for l, ctr in enumerate(per_level_counter):
        vocab_size = len(ctr)
        if vocab_size <= top_n_per_level:
            vocab_list = sorted(ctr.keys())
        else:
            vocab_list = [tok for tok, _ in ctr.most_common(top_n_per_level)]
        per_level_vocab.append(vocab_list)
        logger.info(f"Level {l+1}: total vocab={vocab_size}, using top {len(vocab_list)}")
    return per_level_vocab


def build_few_shot_pool(warm_items, id_map, item_text, pool_size=100):
    candidates = [
        (iid, item_text.get(iid, ""), id_map[iid])
        for iid in warm_items
        if iid in id_map and iid in item_text
    ]
    random.shuffle(candidates)
    return candidates[:pool_size]


def build_prompt_v2iter2(
    target_id: str,
    target_text: str,
    few_shot_examples: List[Tuple[str, str, List[str]]],
    per_level_vocab: List[List[str]],
    num_shots: int = 5,
    num_levels: int = 5,
) -> str:
    """Build prompt with explicit per-level vocabulary constraint."""

    prompt_parts = [
        "You are an expert at assigning hierarchical tokens to items based on their descriptions.",
        f"Each item gets {num_levels} tokens representing a hierarchical path from coarse to fine categories.",
        "The tokens are SentencePiece vocabulary tokens (may contain '▁' prefix or fragments).",
        "",
        "IMPORTANT CONSTRAINT: For each level, you MUST select a token from the provided vocabulary for that level.",
        "Do NOT invent tokens. Do NOT use tokens outside the given lists.",
        "",
    ]

    # Emit per-level vocab
    for l in range(num_levels):
        vocab_str = ", ".join(per_level_vocab[l])
        prompt_parts.append(f"Level {l+1} valid tokens: {vocab_str}")
        prompt_parts.append("")

    prompt_parts.append("Here are some examples of items and their hierarchical tokens:")
    prompt_parts.append("")

    for i, (ex_id, ex_text, ex_tokens) in enumerate(few_shot_examples[:num_shots], 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Text: {ex_text}")
        prompt_parts.append(f"Tokens: {' | '.join(ex_tokens)}")
        prompt_parts.append("")

    prompt_parts.extend([
        "Now predict the hierarchical tokens for this new item.",
        "REMEMBER: Each token MUST be from the Level N valid tokens list above.",
        f"Text: {target_text}",
        "",
        f"Output exactly {num_levels} tokens separated by ' | ' (no extra text, no reasoning):",
        "Tokens:"
    ])

    return "\n".join(prompt_parts)


def parse_llm_response(response_text: str, num_levels: int) -> Tuple[List[str], float]:
    if "Tokens:" in response_text:
        response_text = response_text.split("Tokens:")[-1]
    response_text = response_text.strip()
    tokens = [t.strip() for t in response_text.split('|')]
    if len(tokens) < num_levels:
        tokens += ["<unk>"] * (num_levels - len(tokens))
    tokens = tokens[:num_levels]
    return tokens, 1.0


def generate_llm_priors(
    target_items, item_text, few_shot_pool, per_level_vocab, num_levels,
    cache, client, model, num_shots=5, output_jsonl=None,
):
    results = {}
    num_cached = 0
    num_api_calls = 0
    outfile = open(output_jsonl, 'w') if output_jsonl else None

    for idx, target_id in enumerate(target_items, 1):
        target_text = item_text.get(target_id, "")
        if not target_text:
            logger.warning(f"No text for {target_id}, skipping")
            continue

        few_shot_examples = random.sample(few_shot_pool, min(num_shots, len(few_shot_pool)))
        prompt = build_prompt_v2iter2(
            target_id, target_text, few_shot_examples, per_level_vocab, num_shots, num_levels,
        )

        cached_response = cache.get(model, prompt)
        if cached_response:
            response_text = cached_response["text"]
            num_cached += 1
            is_cached = True
        else:
            messages = [{"role": "user", "content": prompt}]
            try:
                api_response = client.chat_completion(
                    model=model, messages=messages, temperature=0.7, max_tokens=200,
                )
                response_text = api_response["choices"][0]["message"]["content"]
                cache.put(model, prompt, {
                    "text": response_text,
                    "usage": api_response.get("usage", {}),
                    "model": model,
                })
                num_api_calls += 1
                is_cached = False
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"API call failed for {target_id}: {e}")
                response_text = " | ".join(["<unk>"] * num_levels)
                is_cached = False

        predicted_tokens, confidence = parse_llm_response(response_text, num_levels)
        results[target_id] = {
            "predicted_tokens": predicted_tokens,
            "confidence": confidence,
            "cached": is_cached,
        }

        if outfile:
            outfile.write(json.dumps({
                "item_id": target_id, "text": target_text,
                "predicted_tokens": predicted_tokens,
                "confidence": confidence, "cached": is_cached,
            }, ensure_ascii=False) + '\n')

        if idx % 100 == 0:
            logger.info(f"Processed {idx}/{len(target_items)} (cached={num_cached}, api={num_api_calls})")

    if outfile:
        outfile.close()

    logger.info(f"Done: {len(results)} items, {num_cached} cached, {num_api_calls} API calls")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-items", required=True, help="Path to items to predict (cold or warm)")
    parser.add_argument("--warm-items", required=True, help="Path to warm_items.txt (used for few-shot pool)")
    parser.add_argument("--item-text", required=True)
    parser.add_argument("--source-id-file", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--cache-db", default="artifacts/phase13/llm_cache.db")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--top-n-per-level", type=int, default=500,
                        help="Max vocab tokens per level to show in prompt (avoid oversize prompt)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    target_items = list(hid_utils.read_item_set(args.target_items))
    warm_items = list(hid_utils.read_item_set(args.warm_items))
    item_text = load_item_text(args.item_text)
    id_map = hid_utils.read_id_file(args.source_id_file)

    num_levels = len(next(iter(id_map.values())))
    logger.info(f"Detected {num_levels} hierarchical levels")

    per_level_vocab = build_per_level_vocab(id_map, args.top_n_per_level)

    few_shot_pool = build_few_shot_pool(warm_items, id_map, item_text, pool_size=200)
    logger.info(f"Built few-shot pool: {len(few_shot_pool)} examples")

    cache = LLMCache(args.cache_db)
    logger.info(f"Cache stats: {cache.stats()}")
    client = DeepSeekClient()

    generate_llm_priors(
        target_items, item_text, few_shot_pool, per_level_vocab, num_levels,
        cache, client, args.model, args.num_shots, args.output_jsonl,
    )

    logger.info(f"Output written to {args.output_jsonl}")
    logger.info(f"Final cache stats: {cache.stats()}")


if __name__ == "__main__":
    main()
