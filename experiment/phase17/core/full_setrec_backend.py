"""Clean-room Full SETRec backend for Stage17 FP3.

The implementation follows the frozen paper/repository contracts without
copying the unlicensed public SETRec source.  It keeps the four preregistered
arms separate:

* S0: ordered history positions plus sequential query visibility;
* S1R: repository-parity shared-within-item relative positions;
* S1P: paper-faithful sparse history visibility;
* S2: S1P plus GRAM FiD passage context.

Only train-prefix artifacts may be supplied to this module.  External D0
targets are intentionally absent from every API in this file.
"""

from __future__ import annotations

import inspect
import math
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .full_setrec_contracts import (
    SemanticSetAutoencoder,
    SetRecGroundingOutput,
    ground_continuous_queries,
    independent_query_mask,
    paper_sparse_history_mask,
    setrec_joint_loss,
)


SETREC_ARMS = (
    "S0_SETREC_ORDERED_CONTROL",
    "S1R_SETREC_REPO_PARITY",
    "S1P_SETREC_PAPER_FAITHFUL",
    "S2_GRAM_SETREC_PAPER_FULL",
)
N_QUERY = 5
N_CF = 1
N_SEM = 4
MAX_HISTORY_ITEMS = 20
SEMANTIC_FEATURE_SUFFIX = Path(
    "artifacts/phase17/fullport/fp0/full_data_tokenizer/attempt_001/"
    "tokenizer/sentence_embeddings.npy"
)
SEMANTIC_MANIFEST_SUFFIX = Path(
    "artifacts/phase17/fullport/fp0/full_data_tokenizer/attempt_001/"
    "tokenizer/manifest.json"
)
CF_EMBEDDING_SUFFIX = Path(
    "artifacts/phase17/fullport/fp3_setrec/tokenizer/attempt_001/"
    "sasrec_item_embeddings.pt"
)
T5_SMALL_SUFFIX = Path("artifacts/phase14/m2/pretrained/t5-small")
DEFAULT_INSTRUCTION = "Given a user's purchase history, predict the next item."
DEFAULT_RESPONSE = "The next item is represented by five information dimensions."


def _call_compute_bias(
    compute_bias: Any,
    query_length: int,
    key_length: int,
    *,
    device: torch.device | None,
    cache_position: torch.Tensor | None,
) -> torch.Tensor:
    """Call either the legacy or cache-position-aware Transformers API."""

    kwargs: dict[str, Any] = {"device": device}
    if "cache_position" in inspect.signature(compute_bias).parameters:
        kwargs["cache_position"] = cache_position
    return compute_bias(query_length, key_length, **kwargs)


def _load_trusted_torch_payload(path: Path) -> Any:
    """Load a local frozen checkpoint across old and new PyTorch releases."""

    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


@dataclass(frozen=True)
class SetRecCatalog:
    ordered_items: tuple[str, ...]
    item_to_index: dict[str, int]
    semantic_features: torch.Tensor
    cf_embeddings: torch.Tensor


@dataclass(frozen=True)
class SetRecBatch:
    history_item_ids: torch.Tensor
    history_item_mask: torch.Tensor
    target_item_indices: torch.Tensor
    gram_input_ids: torch.Tensor | None = None
    gram_attention_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class SetRecForwardOutput:
    loss: torch.Tensor | None
    generation_loss: torch.Tensor | None
    reconstruction_loss: torch.Tensor | None
    grounding: SetRecGroundingOutput
    semantic_reconstruction: torch.Tensor
    query_outputs: torch.Tensor


def _load_semantic_item_order(manifest_path: Path) -> tuple[str, ...]:
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        (
            row
            for row in manifest["artifacts"]
            if str(row["path"]).endswith("sentence_embeddings.npy")
        ),
        None,
    )
    if artifact is None:
        raise RuntimeError("SentenceT5 manifest lacks sentence_embeddings.npy")
    item_order = manifest.get("catalog_item_order")
    if not isinstance(item_order, list) or not item_order:
        # The tokenizer manifest v1 stores the catalog order in a sibling JSON.
        order_path = manifest_path.parent / "catalog_items.json"
        if order_path.is_file():
            item_order = json.loads(order_path.read_text(encoding="utf-8"))
        else:
            codes_path = manifest_path.parent / "item_semantic_codes.json"
            codes = json.loads(codes_path.read_text(encoding="utf-8"))
            item_order = list(codes)
    return tuple(str(item) for item in item_order)


def load_setrec_catalog(root: Path, *, require_cf: bool = True) -> SetRecCatalog:
    """Load frozen catalog-aligned SentenceT5 and train-only SASRec features."""

    import json

    root = root.resolve()
    manifest_path = root / SEMANTIC_MANIFEST_SUFFIX
    feature_path = root / SEMANTIC_FEATURE_SUFFIX
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item_order_path = feature_path.parent / "catalog_items.json"
    if item_order_path.is_file():
        ordered_items = tuple(
            str(item)
            for item in json.loads(item_order_path.read_text(encoding="utf-8"))
        )
    else:
        ordered_items = _load_semantic_item_order(manifest_path)
    semantic = np.load(feature_path, allow_pickle=False)
    if semantic.shape != (len(ordered_items), 768):
        raise RuntimeError(
            f"semantic feature shape drift: {semantic.shape} for {len(ordered_items)} items"
        )
    semantic_tensor = torch.from_numpy(
        np.asarray(semantic, dtype=np.float32)
    )
    cf_path = root / CF_EMBEDDING_SUFFIX
    if require_cf:
        if not cf_path.is_file():
            raise FileNotFoundError(
                "train-only SASRec tokenizer is not frozen; run FP3 tokenizer first"
            )
        payload = _load_trusted_torch_payload(cf_path)
        if tuple(payload["ordered_items"]) != ordered_items:
            raise RuntimeError("SASRec and SentenceT5 catalog order differ")
        cf_embeddings = payload["item_embeddings"].float()
        if cf_embeddings.shape != (len(ordered_items), 64):
            raise RuntimeError("SASRec item embedding shape drifted")
        if not bool(torch.isfinite(cf_embeddings).all()):
            raise RuntimeError("SASRec item embeddings contain non-finite values")
    else:
        cf_embeddings = torch.zeros((len(ordered_items), 64), dtype=torch.float32)
    return SetRecCatalog(
        ordered_items=ordered_items,
        item_to_index={item: index for index, item in enumerate(ordered_items)},
        semantic_features=semantic_tensor,
        cf_embeddings=cf_embeddings,
    )


def collate_setrec_examples(
    examples: Sequence[Any],
    *,
    item_to_index: Mapping[str, int],
    max_history_items: int = MAX_HISTORY_ITEMS,
    gram_batch: Mapping[str, torch.Tensor] | None = None,
) -> SetRecBatch:
    """Right-align histories and preserve zero solely as the padding index."""

    if not examples:
        raise ValueError("cannot collate an empty SETRec batch")
    histories = torch.zeros(
        (len(examples), max_history_items), dtype=torch.long
    )
    masks = torch.zeros_like(histories, dtype=torch.bool)
    targets = torch.empty((len(examples),), dtype=torch.long)
    for row, example in enumerate(examples):
        encoded = [
            item_to_index[item] + 1
            for item in tuple(example.history)[-max_history_items:]
        ]
        if not encoded:
            raise ValueError("SETRec examples require a non-empty history")
        histories[row, -len(encoded) :] = torch.tensor(encoded, dtype=torch.long)
        masks[row, -len(encoded) :] = True
        targets[row] = item_to_index[example.target]
    return SetRecBatch(
        history_item_ids=histories,
        history_item_mask=masks,
        target_item_indices=targets,
        gram_input_ids=(None if gram_batch is None else gram_batch["item_text_ids"]),
        gram_attention_mask=(
            None if gram_batch is None else gram_batch["item_text_masks"]
        ),
    )


def repo_grouped_position_ids(length: int, *, tokens_per_item: int = N_QUERY) -> torch.Tensor:
    if length <= 0 or tokens_per_item <= 0:
        raise ValueError("length and tokens_per_item must be positive")
    return torch.div(
        torch.arange(length, dtype=torch.long),
        tokens_per_item,
        rounding_mode="floor",
    )


def repo_grouped_relative_position_bias(
    attention: Any,
    query_length: int,
    key_length: int,
    *,
    device: torch.device,
    tokens_per_item: int = N_QUERY,
) -> torch.Tensor:
    """Compute T5 relative bias after grouping the five tokens of each item."""

    query = repo_grouped_position_ids(
        query_length, tokens_per_item=tokens_per_item
    ).to(device)[:, None]
    key = repo_grouped_position_ids(
        key_length, tokens_per_item=tokens_per_item
    ).to(device)[None, :]
    relative = key - query
    bucket = attention._relative_position_bucket(
        relative,
        bidirectional=(not attention.is_decoder),
        num_buckets=attention.relative_attention_num_buckets,
        max_distance=attention.relative_attention_max_distance,
    )
    values = attention.relative_attention_bias(bucket)
    return values.permute(2, 0, 1).unsqueeze(0)


def _install_repo_position_contract(t5: nn.Module) -> None:
    attention = t5.encoder.block[0].layer[0].SelfAttention
    if not getattr(attention, "has_relative_attention_bias", False):
        raise RuntimeError("T5 encoder first block lacks relative attention bias")
    original = attention.compute_bias
    attention.setrec_apply_repo_grouping = False

    def compute_bias(
        this: Any,
        query_length: int,
        key_length: int,
        device: torch.device | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not this.setrec_apply_repo_grouping:
            return _call_compute_bias(
                original,
                query_length,
                key_length,
                device=device,
                cache_position=cache_position,
            )
        del cache_position
        return repo_grouped_relative_position_bias(
            this,
            query_length,
            key_length,
            device=device or this.relative_attention_bias.weight.device,
        )

    attention.compute_bias = types.MethodType(compute_bias, attention)


def _install_paper_history_contract(t5: nn.Module) -> None:
    """Add the paper's prior-item-plus-self visibility to T5 relative bias."""

    attention = t5.encoder.block[0].layer[0].SelfAttention
    original = attention.compute_bias
    attention.setrec_apply_paper_sparse = False

    def compute_bias(
        this: Any,
        query_length: int,
        key_length: int,
        device: torch.device | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bias = _call_compute_bias(
            original,
            query_length,
            key_length,
            device=device,
            cache_position=cache_position,
        )
        if not this.setrec_apply_paper_sparse:
            return bias
        if query_length != key_length or query_length % N_QUERY:
            raise RuntimeError("paper SETRec history must be complete five-token items")
        visible = paper_sparse_history_mask(
            n_items=query_length // N_QUERY,
            n_tokens_per_item=N_QUERY,
            device=bias.device,
        )
        forbidden = (~visible).to(dtype=bias.dtype)
        return bias + forbidden[None, None] * torch.finfo(bias.dtype).min

    attention.compute_bias = types.MethodType(compute_bias, attention)


def _install_independent_query_contract(t5: nn.Module) -> None:
    """Make every query decoder position self-only while retaining cross-attention."""

    attention = t5.decoder.block[0].layer[0].SelfAttention
    original = attention.compute_bias

    def compute_bias(
        this: Any,
        query_length: int,
        key_length: int,
        device: torch.device | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bias = _call_compute_bias(
            original,
            query_length,
            key_length,
            device=device,
            cache_position=cache_position,
        )
        if query_length != N_QUERY or key_length != N_QUERY:
            raise RuntimeError("SETRec decoder must receive exactly five queries")
        visible = independent_query_mask(N_QUERY, device=bias.device)
        return bias + (~visible)[None, None].to(bias.dtype) * torch.finfo(
            bias.dtype
        ).min

    attention.compute_bias = types.MethodType(compute_bias, attention)


def history_visibility_mask(
    arm_id: str,
    history_item_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return encoder self-visibility and decoder cross-attention validity."""

    if arm_id not in SETREC_ARMS:
        raise ValueError(f"unknown SETRec arm: {arm_id}")
    batch, n_items = history_item_mask.shape
    token_valid = history_item_mask.repeat_interleave(N_QUERY, dim=1)
    total = n_items * N_QUERY
    if arm_id in {"S1P_SETREC_PAPER_FAITHFUL", "S2_GRAM_SETREC_PAPER_FULL"}:
        base = paper_sparse_history_mask(
            n_items=n_items,
            n_tokens_per_item=N_QUERY,
            device=history_item_mask.device,
        )
        self_mask = base[None].expand(batch, -1, -1).clone()
    else:
        self_mask = torch.ones(
            (batch, total, total),
            dtype=torch.bool,
            device=history_item_mask.device,
        )
    self_mask &= token_valid[:, None, :]
    # Padded query rows keep their diagonal visible to prevent all-masked NaNs;
    # they remain excluded from decoder cross-attention by token_valid.
    diagonal = torch.eye(total, dtype=torch.bool, device=self_mask.device)
    self_mask |= diagonal[None] & ~token_valid[:, :, None]
    return self_mask, token_valid


def query_visibility_mask(arm_id: str, *, device: torch.device) -> torch.Tensor:
    if arm_id == "S0_SETREC_ORDERED_CONTROL":
        return torch.ones((N_QUERY, N_QUERY), dtype=torch.bool, device=device).tril()
    if arm_id in SETREC_ARMS:
        return independent_query_mask(N_QUERY, device=device)
    raise ValueError(f"unknown SETRec arm: {arm_id}")


class FullSetRecModel(nn.Module):
    """Five-query continuous SETRec with optional GRAM FiD encoder context."""

    def __init__(
        self,
        *,
        arm_id: str,
        backbone: nn.Module,
        cf_embeddings: torch.Tensor,
        semantic_features: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        alpha: float = 0.7,
    ) -> None:
        super().__init__()
        if arm_id not in SETREC_ARMS:
            raise ValueError(f"unknown SETRec arm: {arm_id}")
        if cf_embeddings.ndim != 2 or cf_embeddings.shape[1] != 64:
            raise ValueError("CF embeddings must be [items,64]")
        if semantic_features.ndim != 2 or semantic_features.shape[0] != cf_embeddings.shape[0]:
            raise ValueError("semantic and CF catalogs differ")
        self.arm_id = arm_id
        self.t5 = backbone
        self.model_dim = int(backbone.config.d_model)
        padding = torch.zeros((1, 64), dtype=cf_embeddings.dtype)
        self.cf_embedding = nn.Embedding.from_pretrained(
            torch.cat((padding, cf_embeddings), dim=0), freeze=True, padding_idx=0
        )
        self.cf_projection = nn.Linear(64, self.model_dim)
        self.semantic_autoencoder = SemanticSetAutoencoder(
            semantic_dim=int(semantic_features.shape[1]),
            model_dim=self.model_dim,
            n_semantic_tokens=N_SEM,
            hidden_dims=(512, 256, 128),
            dropout=0.0,
        )
        self.query_vectors = nn.Parameter(torch.empty(N_QUERY, self.model_dim))
        nn.init.normal_(self.query_vectors, mean=0.0, std=0.02)
        self.register_buffer(
            "semantic_features", semantic_features.float(), persistent=True
        )
        self.register_buffer(
            "prompt_input_ids", prompt_input_ids.long(), persistent=True
        )
        self.alpha = float(alpha)
        if arm_id == "S1R_SETREC_REPO_PARITY":
            _install_repo_position_contract(self.t5)
        elif arm_id in {
            "S1P_SETREC_PAPER_FAITHFUL",
            "S2_GRAM_SETREC_PAPER_FULL",
        }:
            _install_paper_history_contract(self.t5)
        if arm_id != "S0_SETREC_ORDERED_CONTROL":
            _install_independent_query_contract(self.t5)

    def token_corpus(self) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_tokens, reconstruction = self.semantic_autoencoder(
            self.semantic_features
        )
        cf = self.cf_projection(self.cf_embedding.weight[1:]).unsqueeze(1)
        corpus = torch.cat((cf, semantic_tokens), dim=1).transpose(0, 1)
        return corpus, reconstruction

    def _encode_history(
        self,
        batch: SetRecBatch,
        token_corpus: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        item_rows = (batch.history_item_ids - 1).clamp_min(0)
        # [B,H,Q,D], then zero all padding positions.
        history_tokens = token_corpus.transpose(0, 1)[item_rows]
        history_tokens = history_tokens * batch.history_item_mask[:, :, None, None]
        flattened = history_tokens.flatten(1, 2)
        _self_mask, token_valid = history_visibility_mask(
            self.arm_id, batch.history_item_mask
        )
        attention = self.t5.encoder.block[0].layer[0].SelfAttention
        contract_flag = None
        if self.arm_id == "S1R_SETREC_REPO_PARITY":
            contract_flag = "setrec_apply_repo_grouping"
        elif self.arm_id in {
            "S1P_SETREC_PAPER_FAITHFUL",
            "S2_GRAM_SETREC_PAPER_FULL",
        }:
            contract_flag = "setrec_apply_paper_sparse"
        if contract_flag is not None:
            setattr(attention, contract_flag, True)
        try:
            encoded = self.t5.encoder(
                inputs_embeds=flattened,
                attention_mask=token_valid,
                return_dict=True,
            ).last_hidden_state
        finally:
            if contract_flag is not None:
                setattr(attention, contract_flag, False)
        return encoded, token_valid

    def _encode_prompt(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.prompt_input_ids.expand(batch_size, -1)
        mask = ids.ne(0)
        encoded = self.t5.encoder(
            input_ids=ids,
            attention_mask=mask,
            return_dict=True,
        ).last_hidden_state
        return encoded, mask

    def _encode_gram_fid(
        self, batch: SetRecBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.gram_input_ids is None or batch.gram_attention_mask is None:
            raise ValueError("S2 requires GRAM FiD passage tensors")
        batch_size, passages, length = batch.gram_input_ids.shape
        flat_ids = batch.gram_input_ids.reshape(batch_size * passages, length)
        flat_mask = batch.gram_attention_mask.reshape(batch_size * passages, length)
        encoded = self.t5.encoder(
            input_ids=flat_ids,
            attention_mask=flat_mask,
            return_dict=True,
        ).last_hidden_state
        return (
            encoded.reshape(batch_size, passages * length, self.model_dim),
            flat_mask.reshape(batch_size, passages * length).bool(),
        )

    def forward(self, batch: SetRecBatch, *, beta: float) -> SetRecForwardOutput:
        token_corpus, semantic_reconstruction = self.token_corpus()
        history_states, history_mask = self._encode_history(batch, token_corpus)
        prompt_states, prompt_mask = self._encode_prompt(
            batch.history_item_ids.shape[0]
        )
        state_parts = [prompt_states, history_states]
        mask_parts = [prompt_mask, history_mask]
        if self.arm_id == "S2_GRAM_SETREC_PAPER_FULL":
            gram_states, gram_mask = self._encode_gram_fid(batch)
            state_parts.append(gram_states)
            mask_parts.append(gram_mask)
        encoder_states = torch.cat(state_parts, dim=1)
        encoder_mask = torch.cat(mask_parts, dim=1)
        queries = self.query_vectors[None].expand(
            batch.history_item_ids.shape[0], -1, -1
        )
        decoded = self.t5.decoder(
            inputs_embeds=queries,
            attention_mask=torch.ones(
                queries.shape[:2], dtype=torch.bool, device=queries.device
            ),
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=encoder_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        grounding = ground_continuous_queries(
            decoded, token_corpus, beta=beta
        )
        losses = setrec_joint_loss(
            grounding.per_dimension_scores,
            batch.target_item_indices,
            semantic_features=self.semantic_features,
            semantic_reconstruction=semantic_reconstruction,
            alpha=self.alpha,
        )
        return SetRecForwardOutput(
            loss=losses.loss,
            generation_loss=losses.generation_loss,
            reconstruction_loss=losses.reconstruction_loss,
            grounding=grounding,
            semantic_reconstruction=semantic_reconstruction,
            query_outputs=decoded,
        )


def build_full_setrec_model(
    root: Path,
    arm_id: str,
    *,
    catalog: SetRecCatalog | None = None,
    seed: int = 2023,
) -> tuple[FullSetRecModel, Any]:
    """Load one fresh local T5-small and construct a frozen-contract FP3 arm."""

    from transformers import AutoTokenizer, T5ForConditionalGeneration

    root = root.resolve()
    catalog = catalog or load_setrec_catalog(root)
    torch.manual_seed(seed)
    path = root / T5_SMALL_SUFFIX
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    prompt = tokenizer(
        f"{DEFAULT_INSTRUCTION} {DEFAULT_RESPONSE}",
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"]
    backbone = T5ForConditionalGeneration.from_pretrained(
        str(path), local_files_only=True
    )
    backbone.config.use_cache = False
    return (
        FullSetRecModel(
            arm_id=arm_id,
            backbone=backbone,
            cf_embeddings=catalog.cf_embeddings,
            semantic_features=catalog.semantic_features,
            prompt_input_ids=prompt,
            alpha=0.7,
        ),
        tokenizer,
    )


def ndcg_at_10(rank: int | None) -> float:
    if rank is None or rank > 10:
        return 0.0
    return 1.0 / math.log2(rank + 1)
