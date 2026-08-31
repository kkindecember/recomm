"""Independent S17-2R architecture-native Semantic-ID model contracts.

The R1 implementation deliberately shares one compact T5 backbone and one
Semantic-ID codec across matched arms.  It implements the mechanism boundary
needed for local profiling; it is not a claim of paper-level reproduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration

from experiment.phase17.core.s2r_sid import SemanticIDCodec


VALID_ARMS = {
    "psid_control",
    "latte_full",
    "gryphon_beam_control",
    "gryphon_item",
}


@dataclass
class S2RForwardOutput:
    loss: torch.Tensor
    generation_loss: torch.Tensor
    item_loss: torch.Tensor
    item_logits: torch.Tensor


@dataclass(frozen=True)
class RankedPrediction:
    item_id: str
    final_score: float
    beam_score: float
    item_score: float | None
    path_count: int


def smoke_t5_config(codec: SemanticIDCodec, *, capacity: str = "r1") -> T5Config:
    if capacity == "tiny":
        dimensions = dict(d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=4)
    elif capacity == "r1":
        dimensions = dict(d_model=128, d_ff=512, num_layers=2, num_decoder_layers=2, num_heads=8)
    elif capacity == "r2":
        # Matches the public Latte depth/model width while retaining the small
        # Phase17 token vocabulary and independent implementation.
        dimensions = dict(d_model=128, d_ff=1024, num_layers=4, num_decoder_layers=4, num_heads=8)
    else:
        raise ValueError(f"unknown S17-2R capacity: {capacity}")
    return T5Config(
        vocab_size=codec.vocab_size,
        pad_token_id=codec.padding_token,
        decoder_start_token_id=codec.padding_token,
        eos_token_id=codec.eos_token,
        dropout_rate=0.0,
        feed_forward_proj="relu",
        **dimensions,
    )


class S2RSemanticIDModel(nn.Module):
    """Matched PSID/Latte/Gryphon R1 model with a shared item scorer.

    ``gryphon_item`` is the only arm whose training objective includes the
    catalog item-level cross entropy.  ``gryphon_beam_control`` keeps the exact
    same parameters and generated candidates but ranks by sequence likelihood.
    """

    def __init__(
        self,
        codec: SemanticIDCodec,
        *,
        arm: str,
        config: T5Config,
        item_loss_weight: float = 0.2,
    ) -> None:
        super().__init__()
        if arm not in VALID_ARMS:
            raise ValueError(f"unknown S17-2R arm: {arm}")
        if config.vocab_size != codec.vocab_size:
            raise ValueError("model and Semantic-ID codec vocabulary mismatch")
        self.codec = codec
        self.arm = arm
        self.item_loss_weight = float(item_loss_weight)
        self.t5 = T5ForConditionalGeneration(config)
        self.context_norm = nn.LayerNorm(config.d_model)
        catalog_tokens = torch.tensor(
            [codec.semantic_tokens(item) for item in codec.item_ids], dtype=torch.long
        )
        self.register_buffer("catalog_tokens", catalog_tokens, persistent=True)

    @property
    def latte(self) -> bool:
        return self.arm == "latte_full"

    @property
    def uses_item_loss(self) -> bool:
        return self.arm == "gryphon_item"

    def _pool_context(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.context_norm(pooled)

    def _score_from_hidden(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        context = self._pool_context(hidden, attention_mask)
        item_embeddings = self.t5.shared(self.catalog_tokens).mean(dim=1)
        return context @ item_embeddings.transpose(0, 1) / math.sqrt(context.shape[-1])

    def score_catalog(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        return self._score_from_hidden(hidden, attention_mask)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        target_item_index: torch.Tensor,
    ) -> S2RForwardOutput:
        output = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        item_logits = self._score_from_hidden(
            output.encoder_last_hidden_state, attention_mask
        )
        item_loss = nn.functional.cross_entropy(item_logits, target_item_index)
        weight = self.item_loss_weight if self.uses_item_loss else 0.0
        loss = output.loss + weight * item_loss
        return S2RForwardOutput(
            loss=loss,
            generation_loss=output.loss,
            item_loss=item_loss,
            item_logits=item_logits,
        )

    def _prefix_allowed_tokens(self, _batch_id: int, generated: torch.Tensor) -> list[int]:
        return list(
            self.codec.allowed_generation_tokens(
                generated.detach().cpu().tolist(), latte=self.latte
            )
        )

    def generate_ranked(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_beams: int,
        top_k: int,
        latte_aggregation: str = "max",
    ) -> list[list[RankedPrediction]]:
        if latte_aggregation not in {"max", "logsumexp"}:
            raise ValueError("Latte aggregation must be max or logsumexp")
        requested = min(int(num_beams), len(self.codec.item_ids))
        generated = self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.codec.n_digit + (2 if self.latte else 1),
            num_beams=requested,
            num_return_sequences=requested,
            do_sample=False,
            early_stopping=True,
            prefix_allowed_tokens_fn=self._prefix_allowed_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        sequences = generated.sequences.detach().cpu()
        sequence_scores = generated.sequences_scores.detach().cpu().tolist()
        batch_size = input_ids.shape[0]
        item_logits = None
        if self.arm == "gryphon_item":
            item_logits = self.score_catalog(input_ids, attention_mask).detach().cpu()

        all_rankings: list[list[RankedPrediction]] = []
        for batch_index in range(batch_size):
            paths: dict[str, list[float]] = {}
            start = batch_index * requested
            for row_index in range(start, start + requested):
                values = sequences[row_index].tolist()
                if values and values[0] == self.codec.padding_token:
                    values = values[1:]
                if values and values[-1] == self.codec.eos_token:
                    values = values[:-1]
                semantic = values[1:] if self.latte else values
                item = self.codec.decode_semantic_tokens(semantic)
                if item is not None:
                    paths.setdefault(item, []).append(float(sequence_scores[row_index]))

            ranked: list[RankedPrediction] = []
            for item, scores in paths.items():
                beam_score = max(scores)
                if self.latte and latte_aggregation == "logsumexp":
                    aggregate = float(torch.logsumexp(torch.tensor(scores), dim=0).item())
                else:
                    aggregate = beam_score
                scorer_value = None
                final_score = aggregate
                if item_logits is not None:
                    scorer_value = float(
                        item_logits[batch_index, self.codec.item_to_index[item]].item()
                    )
                    final_score = scorer_value
                ranked.append(
                    RankedPrediction(
                        item_id=item,
                        final_score=final_score,
                        beam_score=beam_score,
                        item_score=scorer_value,
                        path_count=len(scores),
                    )
                )
            ranked.sort(key=lambda row: (-row.final_score, row.item_id))
            all_rankings.append(ranked[: int(top_k)])
        return all_rankings


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def item_scorer_gradient_norm(model: S2RSemanticIDModel) -> float:
    squared = 0.0
    for parameter in model.context_norm.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(squared)
