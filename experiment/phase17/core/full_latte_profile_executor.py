"""One-step train + primary-beam resource workloads for FP1/FP2 arms.

These workloads measure capacity only.  They never compare predictions with a
target, emit rankings, select a checkpoint, or open the external D0 position.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable


GRAM_ARMS = {
    "G0_GRAM_B0_FRESH",
    "G1_GRAM_PSID_FULL",
    "G2_GRAM_LATTE_FULL",
}
NATIVE_ARMS = {"N0_NATIVE_PSID", "N1_NATIVE_LATTE"}


def _cuda_memory(torch) -> dict[str, float]:
    mib = 1024**2
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / mib,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / mib,
        "end_allocated_mib": torch.cuda.memory_allocated() / mib,
        "end_reserved_mib": torch.cuda.memory_reserved() / mib,
    }


def _move_tensor_batch(batch: dict[str, Any], device) -> dict[str, Any]:
    return {
        key: (value.to(device) if hasattr(value, "to") else value)
        for key, value in batch.items()
    }


def run_gram_resource_profile(
    root: Path,
    arm_id: str,
    *,
    train_microbatch: int,
    eval_batch_size: int,
    include_primary_generation: bool = True,
    heartbeat: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if arm_id not in GRAM_ARMS:
        raise ValueError(f"not a GRAM profile arm: {arm_id}")
    if train_microbatch <= 0 or eval_batch_size <= 0:
        raise ValueError("profile batch sizes must be positive")
    import torch

    from .full_latte_gram_backend import (
        PrefixTree,
        aggregate_generated_paths,
        build_gram_collator,
        create_fresh_gram_model,
        encoded_candidate_paths,
        load_fullport_examples,
        load_gram_catalog,
        render_gram_example,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for GRAM resource profile")
    device = torch.device("cuda:0")
    catalog = load_gram_catalog(root, arm_id)
    train, internal_dev = load_fullport_examples(root)
    longest = sorted(
        train,
        key=lambda example: (-len(example.history), example.user_id, example.target),
    )[:train_microbatch]
    eval_examples = sorted(
        internal_dev,
        key=lambda example: (-len(example.history), example.user_id, example.target),
    )[:eval_batch_size]
    rng = random.Random(2023)
    tokenizer, collator = build_gram_collator(root, arm_id)
    train_rows = [
        render_gram_example(example, arm_id=arm_id, catalog=catalog, rng=rng)
        .as_collator_row()
        for example in longest
    ]
    eval_rows = [
        render_gram_example(example, arm_id=arm_id, catalog=catalog, rng=rng)
        .as_collator_row()
        for example in eval_examples
    ]
    train_batch = collator(train_rows)
    eval_batch = collator(eval_rows)
    item_paths = encoded_candidate_paths(tokenizer, arm_id, catalog)
    flat_paths = [path for paths in item_paths.values() for path in paths]
    trie = PrefixTree(flat_paths)
    max_length = max(len(path) for path in flat_paths)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if heartbeat:
        heartbeat("constructing_fresh_gram", {"current": 1, "total": 3, "unit": "profile_phase"})
    model = create_fresh_gram_model(root, arm_id, tokenizer, seed=2023).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01
    )
    train_batch = _move_tensor_batch(train_batch, device)
    model.train()
    optimizer.zero_grad()
    outputs = model(
        input_ids=train_batch["item_text_ids"],
        attention_mask=train_batch["item_text_masks"],
        history_item_ids=train_batch["history_item_ids"],
        history_item_mask=train_batch["history_item_mask"],
        target_item_ids=train_batch["target_item_ids"],
        labels=train_batch["target_ids"],
    )
    loss = outputs.loss
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite GRAM profile training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    training_memory = _cuda_memory(torch)
    if not include_primary_generation:
        if heartbeat:
            heartbeat(
                "training_profile_complete",
                {"current": 1, "total": 1, "unit": "profile_phase"},
            )
        return {
            "backend": "project_GRAM_FiD",
            "parameter_count": parameter_count,
            "train_microbatch": train_microbatch,
            "effective_batch": 128,
            "gradient_accumulation": 128 // train_microbatch,
            "train_input_shape": list(train_batch["item_text_ids"].shape),
            "train_target_shape": list(train_batch["target_ids"].shape),
            "primary_generation_included": False,
            "training_only": True,
            "training_loss_finite": True,
            **training_memory,
        }
    if heartbeat:
        heartbeat("primary_beam_generation", {"current": 2, "total": 3, "unit": "profile_phase"})

    eval_batch = _move_tensor_batch(eval_batch, device)
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=eval_batch["item_text_ids"],
            attention_mask=eval_batch["item_text_masks"],
            history_item_ids=eval_batch["history_item_ids"],
            history_item_mask=eval_batch["history_item_mask"],
            max_length=max_length,
            prefix_allowed_tokens_fn=trie.prefix_allowed_tokens_fn(),
            num_beams=500,
            num_return_sequences=500,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=1.0,
            # Beam-500 expands the 21x128-token FiD encoder state 500 times.
            # Caching every decoder layer's cross-attention K/V therefore
            # consumes tens of GiB although the generated paths are only 4--6
            # tokens long.  Recomputing those tensors is prediction-equivalent
            # and trades a small amount of wall time for bounded memory.
            use_cache=False,
        )
    raw_sequences = generated["sequences"].detach().cpu().tolist()
    scores = generated["sequences_scores"].detach().cpu().tolist()
    expected_sequences = eval_batch_size * 500
    if len(raw_sequences) != expected_sequences:
        raise AssertionError("GRAM primary-beam output cardinality drifted")
    sequences = []
    for sequence in raw_sequences:
        try:
            eos_position = sequence.index(tokenizer.eos_token_id, 1)
            sequence = sequence[: eos_position + 1]
        except ValueError:
            pass
        sequences.append(sequence)
    valid_path_set = {
        tuple(path) for paths in item_paths.values() for path in paths
    }
    valid_paths = sum(tuple(sequence) in valid_path_set for sequence in sequences)
    if valid_paths != expected_sequences:
        raise RuntimeError("constrained GRAM generation emitted an invalid catalog path")
    # Aggregate only to exercise the registered item-level operation.  No
    # target is supplied and no effectiveness statistic is calculated.
    resolved_counts = []
    for batch_index in range(eval_batch_size):
        start = batch_index * 500
        ranked = aggregate_generated_paths(
            sequences[start : start + 500],
            scores[start : start + 500],
            item_paths=item_paths,
            method=("agg_max" if arm_id == "G2_GRAM_LATTE_FULL" else "agg_max"),
            top_k=50,
        )
        resolved_counts.append(len(ranked))
    torch.cuda.synchronize()
    if heartbeat:
        heartbeat("profile_complete", {"current": 3, "total": 3, "unit": "profile_phase"})
    return {
        "backend": "project_GRAM_FiD",
        "parameter_count": parameter_count,
        "train_microbatch": train_microbatch,
        "eval_batch_size": eval_batch_size,
        "effective_batch": 128,
        "gradient_accumulation": 128 // train_microbatch,
        "train_input_shape": list(train_batch["item_text_ids"].shape),
        "train_target_shape": list(train_batch["target_ids"].shape),
        "primary_beam": 500,
        "top_k": 50,
        "generation_kv_cache": False,
        "decoder_path_count": len(flat_paths),
        "generated_path_count": len(sequences),
        "valid_generated_path_count": valid_paths,
        "resolved_topk_counts": resolved_counts,
        "training_loss_finite": True,
        "training_peak_allocated_mib": training_memory["peak_allocated_mib"],
        "training_peak_reserved_mib": training_memory["peak_reserved_mib"],
        **_cuda_memory(torch),
    }


def run_native_resource_profile(
    root: Path,
    arm_id: str,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    heartbeat: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if arm_id not in NATIVE_ARMS:
        raise ValueError(f"not a native profile arm: {arm_id}")
    if train_batch_size <= 0 or eval_batch_size <= 0:
        raise ValueError("profile batch sizes must be positive")
    import torch

    from .full_latte_native_backend import (
        build_official_native_components,
        collate_native_eval_batch,
        collate_native_train_batch,
        create_fresh_official_native_model,
        load_native_examples,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for native resource profile")
    device = torch.device("cuda:0")
    components = build_official_native_components(
        root, arm_id, device="cuda:0", num_beams=500
    )
    train, internal_dev = load_native_examples(root)
    train_examples = train[:train_batch_size]
    eval_examples = internal_dev[:eval_batch_size]
    torch.manual_seed(2023)
    train_batch = collate_native_train_batch(components, train_examples)
    eval_batch = collate_native_eval_batch(components, eval_examples)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if heartbeat:
        heartbeat("constructing_official_native_model", {"current": 1, "total": 3, "unit": "profile_phase"})
    model = create_fresh_official_native_model(components, seed=2023).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.05)
    train_batch = _move_tensor_batch(train_batch, device)
    model.train()
    optimizer.zero_grad()
    outputs = model(train_batch)
    loss = outputs.loss
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite native profile training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if heartbeat:
        heartbeat("primary_beam_generation", {"current": 2, "total": 3, "unit": "profile_phase"})

    eval_batch = _move_tensor_batch(eval_batch, device)
    model.eval()
    with torch.no_grad():
        predictions = model.generate(eval_batch, n_return_sequences=50)
    prediction_shape = tuple(int(value) for value in predictions.shape)
    expected_width = int(components.tokenizer.n_digit)
    if (
        len(prediction_shape) != 3
        or prediction_shape[:2] != (eval_batch_size, 50)
        or not 1 <= prediction_shape[2] <= expected_width
    ):
        raise AssertionError(
            "official native prediction cardinality drifted: "
            f"{prediction_shape}, expected ({eval_batch_size}, 50, 1..{expected_width})"
        )
    # The pinned official PSID wrapper slices up to n_digit tokens from the
    # Hugging Face result.  A fresh untrained model can rank EOS early, so the
    # returned last dimension may be shorter than n_digit.  That is a legal
    # official-generation outcome for this capacity-only probe, not tokenizer
    # drift: CPU preflight separately proves every catalog SID has n_digit
    # tokens and the training labels contain SID+EOS.
    prediction_width = prediction_shape[2]
    # Exercise result materialization only; labels are deliberately not passed
    # to an evaluator, so this profile cannot produce an efficacy signal.
    nonzero_rows = int(predictions.detach().ne(0).any(dim=-1).sum().item())
    torch.cuda.synchronize()
    if heartbeat:
        heartbeat("profile_complete", {"current": 3, "total": 3, "unit": "profile_phase"})
    return {
        "backend": f"pinned_official_{components.model_class.__name__}",
        "official_model_module": components.model_class.__module__,
        "parameter_count": parameter_count,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "train_input_shape": list(train_batch["input_ids"].shape),
        "train_target_shape": list(train_batch["labels"].shape),
        "primary_beam": 500,
        "top_k": 50,
        "aggregation": ("identity" if arm_id == "N0_NATIVE_PSID" else "agg_max"),
        "prediction_shape": list(predictions.shape),
        "expected_prediction_width": expected_width,
        "early_eos_shortened_prediction_width": prediction_width < expected_width,
        "nonzero_prediction_rows": nonzero_rows,
        "training_loss_finite": True,
        **_cuda_memory(torch),
    }


def run_resource_profile(
    root: Path,
    arm_id: str,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    include_primary_generation: bool = True,
    heartbeat: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if arm_id in GRAM_ARMS:
        if 128 % train_batch_size:
            raise ValueError("GRAM microbatch must divide the effective batch 128")
        return run_gram_resource_profile(
            root,
            arm_id,
            train_microbatch=train_batch_size,
            eval_batch_size=eval_batch_size,
            include_primary_generation=include_primary_generation,
            heartbeat=heartbeat,
        )
    if arm_id in NATIVE_ARMS:
        return run_native_resource_profile(
            root,
            arm_id,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            heartbeat=heartbeat,
        )
    raise ValueError(f"unknown FP1/FP2 profile arm: {arm_id}")
