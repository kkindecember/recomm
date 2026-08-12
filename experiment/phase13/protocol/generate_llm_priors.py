"""
Generate LLM prior predictions for cold items' hierarchical IDs (Phase 13 v2).

Uses DeepSeek API with 5-shot warm examples to predict hierarchical tokens for cold items.
Caches all API responses to avoid redundant calls.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))
from llm_cache import LLMCache
from deepseek_client import DeepSeekClient
import hierarchical_id_utils as hid_utils


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_item_text(item_text_file: str) -> Dict[str, str]:
    """Load item ID -> text mapping."""
    item_text = {}
    with open(item_text_file) as f:
        for line in f:
            parts = line.rstrip('\n').split(' ', 1)
            if len(parts) == 2:
                item_id, text = parts
                item_text[item_id] = text
    logger.info(f"Loaded {len(item_text)} item texts")
    return item_text


def build_few_shot_pool(
    warm_items: List[str],
    id_map: Dict[str, List[str]],
    item_text: Dict[str, str],
    pool_size: int = 100
) -> List[Tuple[str, str, List[str]]]:
    """
    Sample warm items as few-shot examples.

    Returns:
        List of (item_id, text, hierarchical_tokens)
    """
    candidates = [
        (item_id, item_text.get(item_id, ""), id_map[item_id])
        for item_id in warm_items
        if item_id in id_map and item_id in item_text
    ]
    random.shuffle(candidates)
    return candidates[:pool_size]


def build_prompt(
    cold_item_id: str,
    cold_text: str,
    few_shot_examples: List[Tuple[str, str, List[str]]],
    num_shots: int = 5,
    num_levels: int = 5
) -> str:
    """Build prompt for LLM to predict hierarchical ID tokens."""

    prompt_parts = [
        "You are an expert at assigning hierarchical tokens to items based on their descriptions.",
        f"Each item gets {num_levels} tokens representing a hierarchical path from coarse to fine categories.",
        "The tokens are SentencePiece vocabulary tokens (may contain underscores like '▁game' or abbreviations).",
        "",
        "Here are some examples of items and their hierarchical tokens:",
        ""
    ]

    # Add few-shot examples
    for i, (ex_id, ex_text, ex_tokens) in enumerate(few_shot_examples[:num_shots], 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Text: {ex_text}")
        prompt_parts.append(f"Tokens: {' | '.join(ex_tokens)}")
        prompt_parts.append("")

    # Add target item
    prompt_parts.extend([
        "Now predict the hierarchical tokens for this new item:",
        f"Text: {cold_text}",
        "",
        f"Output exactly {num_levels} tokens separated by ' | ' (no extra text):",
        "Tokens:"
    ])

    return "\n".join(prompt_parts)


def parse_llm_response(response_text: str, num_levels: int) -> Tuple[List[str], float]:
    """
    Parse LLM response to extract tokens and confidence.

    Returns:
        (tokens, confidence) where confidence is dummy 1.0 (single first-pass doesn't provide confidence)
    """
    # Extract tokens after "Tokens:" if present
    if "Tokens:" in response_text:
        response_text = response_text.split("Tokens:")[-1]

    # Clean and split
    response_text = response_text.strip()
    tokens = [t.strip() for t in response_text.split('|')]

    # Pad or truncate to num_levels
    if len(tokens) < num_levels:
        tokens += ["<unk>"] * (num_levels - len(tokens))
    tokens = tokens[:num_levels]

    # Confidence: single first-pass has no confidence, use 1.0
    confidence = 1.0

    return tokens, confidence


def generate_llm_priors(
    cold_items: List[str],
    item_text: Dict[str, str],
    few_shot_pool: List[Tuple[str, str, List[str]]],
    num_levels: int,
    cache: LLMCache,
    client: DeepSeekClient,
    model: str,
    num_shots: int = 5,
    output_jsonl: str = None
) -> Dict[str, Dict]:
    """
    Generate LLM prior predictions for all cold items.

    Returns:
        Dict[item_id] -> {"predicted_tokens": [...], "confidence": float, "cached": bool}
    """
    results = {}
    num_cached = 0
    num_api_calls = 0

    if output_jsonl:
        outfile = open(output_jsonl, 'w')
    else:
        outfile = None

    for idx, cold_id in enumerate(cold_items, 1):
        cold_text = item_text.get(cold_id, "")
        if not cold_text:
            logger.warning(f"No text for cold item {cold_id}, skipping")
            continue

        # Sample few-shot examples
        few_shot_examples = random.sample(few_shot_pool, min(num_shots, len(few_shot_pool)))

        # Build prompt
        prompt = build_prompt(cold_id, cold_text, few_shot_examples, num_shots, num_levels)

        # Check cache
        cached_response = cache.get(model, prompt)
        if cached_response:
            response_text = cached_response["text"]
            num_cached += 1
            is_cached = True
        else:
            # Call API
            messages = [{"role": "user", "content": prompt}]
            try:
                api_response = client.chat_completion(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=200
                )
                response_text = api_response["choices"][0]["message"]["content"]

                # Cache response
                cache.put(model, prompt, {
                    "text": response_text,
                    "usage": api_response.get("usage", {}),
                    "model": model
                })
                num_api_calls += 1
                is_cached = False

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"API call failed for {cold_id}: {e}")
                response_text = " | ".join(["<unk>"] * num_levels)
                is_cached = False

        # Parse response
        predicted_tokens, confidence = parse_llm_response(response_text, num_levels)

        results[cold_id] = {
            "predicted_tokens": predicted_tokens,
            "confidence": confidence,
            "cached": is_cached
        }

        # Write to JSONL
        if outfile:
            record = {
                "item_id": cold_id,
                "text": cold_text,
                "predicted_tokens": predicted_tokens,
                "confidence": confidence,
                "cached": is_cached
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

        if idx % 100 == 0:
            logger.info(f"Processed {idx}/{len(cold_items)} (cached={num_cached}, api_calls={num_api_calls})")

    if outfile:
        outfile.close()

    logger.info(f"LLM prior generation complete: {len(results)} items, {num_cached} cached, {num_api_calls} API calls")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate LLM priors for cold items (v2)")
    parser.add_argument("--cold-items", required=True, help="Path to cold_items.txt")
    parser.add_argument("--warm-items", required=True, help="Path to warm_items.txt")
    parser.add_argument("--item-text", required=True, help="Path to item_plain_text.txt")
    parser.add_argument("--source-id-file", required=True, help="Path to source hierarchical ID file")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL file with LLM predictions")
    parser.add_argument("--cache-db", default="artifacts/phase13/llm_cache.db", help="SQLite cache database")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model name")
    parser.add_argument("--num-shots", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for few-shot sampling")

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)

    # Load data
    cold_items = hid_utils.read_item_set(args.cold_items)
    warm_items = hid_utils.read_item_set(args.warm_items)
    item_text = load_item_text(args.item_text)
    id_map = hid_utils.read_id_file(args.source_id_file)

    # Detect num_levels
    num_levels = len(next(iter(id_map.values())))
    logger.info(f"Detected {num_levels} hierarchical levels")

    # Build few-shot pool
    few_shot_pool = build_few_shot_pool(list(warm_items), id_map, item_text, pool_size=200)
    logger.info(f"Built few-shot pool: {len(few_shot_pool)} examples")

    # Initialize cache and client
    cache = LLMCache(args.cache_db)
    logger.info(f"Cache stats: {cache.stats()}")

    client = DeepSeekClient()  # Will read DEEPSEEK_API_KEY from env

    # Generate priors
    results = generate_llm_priors(
        list(cold_items),
        item_text,
        few_shot_pool,
        num_levels,
        cache,
        client,
        args.model,
        args.num_shots,
        args.output_jsonl
    )

    logger.info(f"Output written to {args.output_jsonl}")
    logger.info(f"Final cache stats: {cache.stats()}")


if __name__ == "__main__":
    main()
