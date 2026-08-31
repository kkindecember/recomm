"""Independent scaled R1 contracts for DiffGRM- and SETRec-style decoders.

No third-party source is copied here.  The goal is to validate mechanism,
gradient, legality and resource boundaries before any faithful-scale attempt.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
from torch import nn
from transformers import T5Config, T5EncoderModel

from experiment.phase17.core.s2r_sid import SemanticIDCodec


PARALLEL_ARMS = {
    "diffgrm_ar_control",
    "diffgrm_masked",
    "setrec_ar_control",
    "setrec_full",
}


@dataclass
class ParallelForwardOutput:
    loss: torch.Tensor
    token_loss: torch.Tensor
    active_token_fraction: float


@dataclass(frozen=True)
class ParallelPrediction:
    item_id: str
    score: float


def parallel_smoke_config(codec: SemanticIDCodec, *, capacity: str = "r1") -> T5Config:
    if capacity == "tiny":
        dimensions = dict(d_model=32, d_ff=64, num_layers=1, num_heads=4)
        decoder_layers = 1
    elif capacity == "r1":
        dimensions = dict(d_model=128, d_ff=512, num_layers=2, num_heads=8)
        decoder_layers = 2
    elif capacity == "r2":
        # A profile-first screen scale.  It is intentionally below the public
        # DiffGRM hidden-1024 setup and must not be called faithful-scale.
        dimensions = dict(d_model=256, d_ff=1024, num_layers=4, num_heads=8)
        decoder_layers = 4
    else:
        raise ValueError(f"unknown S17-2R parallel capacity: {capacity}")
    config = T5Config(
        vocab_size=codec.vocab_size,
        pad_token_id=codec.padding_token,
        eos_token_id=codec.eos_token,
        decoder_start_token_id=codec.padding_token,
        dropout_rate=0.0,
        feed_forward_proj="relu",
        **dimensions,
    )
    config.s2r_parallel_decoder_layers = decoder_layers
    return config


class S2RParallelIDModel(nn.Module):
    """One matched parameterization with AR, masked, or set-native objectives."""

    def __init__(self, codec: SemanticIDCodec, *, arm: str, config: T5Config) -> None:
        super().__init__()
        if arm not in PARALLEL_ARMS:
            raise ValueError(f"unknown parallel arm: {arm}")
        self.codec = codec
        self.arm = arm
        self.encoder = T5EncoderModel(config)
        self.token_embedding = nn.Embedding(codec.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(codec.n_digit, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer, num_layers=config.s2r_parallel_decoder_layers
        )
        self.context_projection = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, codec.vocab_size)
        catalog_tokens = torch.tensor(
            [codec.semantic_tokens(item) for item in codec.item_ids], dtype=torch.long
        )
        self.register_buffer("catalog_tokens", catalog_tokens, persistent=True)

    @property
    def is_ar(self) -> bool:
        return self.arm.endswith("ar_control")

    @property
    def is_diffusion(self) -> bool:
        return self.arm == "diffgrm_masked"

    @property
    def is_set(self) -> bool:
        return self.arm == "setrec_full"

    def encode_context(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.context_projection(pooled)

    def decode_from_context(
        self, context: torch.Tensor, decoder_tokens: torch.Tensor, *, causal: bool
    ) -> torch.Tensor:
        positions = torch.arange(
            self.codec.n_digit, device=decoder_tokens.device, dtype=torch.long
        )
        hidden = (
            self.token_embedding(decoder_tokens)
            + self.position_embedding(positions)[None, :, :]
            + context[:, None, :]
        )
        mask = None
        if causal:
            mask = torch.triu(
                torch.full(
                    (self.codec.n_digit, self.codec.n_digit),
                    float("-inf"),
                    device=decoder_tokens.device,
                ),
                diagonal=1,
            )
        return self.output(self.decoder(hidden, mask=mask))

    def _set_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        assignments = []
        for permutation in itertools.permutations(range(self.codec.n_digit)):
            permuted = targets[:, permutation]
            selected = torch.gather(log_probs, 2, permuted.unsqueeze(-1)).squeeze(-1)
            assignments.append(-selected.mean(dim=1))
        return torch.stack(assignments, dim=1).min(dim=1).values.mean()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        target_item_index: torch.Tensor | None = None,
    ) -> ParallelForwardOutput:
        del target_item_index
        targets = labels[:, : self.codec.n_digit]
        context = self.encode_context(input_ids, attention_mask)
        if self.is_ar:
            decoder_tokens = torch.full_like(targets, self.codec.mask_token)
            decoder_tokens[:, 1:] = targets[:, :-1]
            logits = self.decode_from_context(context, decoder_tokens, causal=True)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
            active_fraction = 1.0
        elif self.is_diffusion:
            mask = torch.rand(targets.shape, device=targets.device) < 0.5
            empty = ~mask.any(dim=1)
            if empty.any():
                mask[empty, 0] = True
            decoder_tokens = targets.masked_fill(mask, self.codec.mask_token)
            logits = self.decode_from_context(context, decoder_tokens, causal=False)
            loss = nn.functional.cross_entropy(logits[mask], targets[mask])
            active_fraction = float(mask.float().mean().detach().cpu().item())
        else:
            decoder_tokens = torch.full_like(targets, self.codec.mask_token)
            logits = self.decode_from_context(context, decoder_tokens, causal=False)
            loss = self._set_loss(logits, targets)
            active_fraction = 1.0
        return ParallelForwardOutput(
            loss=loss, token_loss=loss, active_token_fraction=active_fraction
        )

    def _score_catalog_ordered(self, logits: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        batch_size = logits.shape[0]
        catalog = self.catalog_tokens.transpose(0, 1)[None, :, :].expand(
            batch_size, -1, -1
        )
        return torch.gather(log_probs, 2, catalog).sum(dim=1)

    def _score_catalog_set(self, logits: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        scores = []
        for permutation in itertools.permutations(range(self.codec.n_digit)):
            tokens = self.catalog_tokens[:, permutation].transpose(0, 1)
            expanded = tokens[None, :, :].expand(logits.shape[0], -1, -1)
            scores.append(torch.gather(log_probs, 2, expanded).sum(dim=1))
        return torch.stack(scores, dim=0).max(dim=0).values

    def _ar_beam_single(
        self, context: torch.Tensor, *, num_beams: int
    ) -> list[ParallelPrediction]:
        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        for position in range(self.codec.n_digit):
            decoder_tokens = torch.full(
                (len(beams), self.codec.n_digit),
                self.codec.mask_token,
                device=context.device,
                dtype=torch.long,
            )
            for index, (prefix, _) in enumerate(beams):
                if prefix:
                    decoder_tokens[index, 1 : len(prefix) + 1] = torch.tensor(
                        prefix, device=context.device
                    )
            logits = self.decode_from_context(
                context.expand(len(beams), -1), decoder_tokens, causal=True
            )[:, position]
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            expanded_beams: list[tuple[tuple[int, ...], float]] = []
            for index, (prefix, score) in enumerate(beams):
                allowed = self.codec.legal_next.get(prefix, ())
                for token in allowed:
                    expanded_beams.append(
                        (prefix + (token,), score + float(log_probs[index, token].item()))
                    )
            expanded_beams.sort(key=lambda row: (-row[1], row[0]))
            beams = expanded_beams[:num_beams]
        result = []
        for tokens, score in beams:
            item = self.codec.decode_semantic_tokens(tokens)
            if item is not None:
                result.append(ParallelPrediction(item, score))
        return result

    def generate_ranked(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_beams: int,
        top_k: int,
    ) -> list[list[ParallelPrediction]]:
        context = self.encode_context(input_ids, attention_mask)
        if self.is_ar:
            return [
                self._ar_beam_single(context[index : index + 1], num_beams=num_beams)[
                    :top_k
                ]
                for index in range(context.shape[0])
            ]

        tokens = torch.full(
            (context.shape[0], self.codec.n_digit),
            self.codec.mask_token,
            device=context.device,
            dtype=torch.long,
        )
        if self.is_diffusion:
            for _ in range(self.codec.n_digit):
                logits = self.decode_from_context(context, tokens, causal=False)
                probabilities = nn.functional.softmax(logits, dim=-1)
                confidence, predicted = probabilities.max(dim=-1)
                confidence = confidence.masked_fill(tokens != self.codec.mask_token, -1.0)
                positions = confidence.max(dim=1).indices
                rows = torch.arange(tokens.shape[0], device=tokens.device)
                tokens[rows, positions] = predicted[rows, positions]
            logits = self.decode_from_context(context, tokens, causal=False)
            catalog_scores = self._score_catalog_ordered(logits)
        else:
            logits = self.decode_from_context(context, tokens, causal=False)
            catalog_scores = self._score_catalog_set(logits)

        values, indices = torch.topk(
            catalog_scores, k=min(top_k, catalog_scores.shape[1]), dim=1
        )
        result = []
        for row_values, row_indices in zip(values, indices):
            result.append(
                [
                    ParallelPrediction(
                        self.codec.index_to_item[int(index.item())], float(value.item())
                    )
                    for value, index in zip(row_values, row_indices)
                ]
            )
        return result

    def mechanism_diagnostics(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        targets = labels[:, : self.codec.n_digit]
        context = self.encode_context(input_ids, attention_mask)
        if self.is_ar:
            decoder_tokens = torch.full_like(targets, self.codec.mask_token)
            decoder_tokens[:, 1:] = targets[:, :-1]
            logits = self.decode_from_context(context, decoder_tokens, causal=True)
        else:
            decoder_tokens = torch.full_like(targets, self.codec.mask_token)
            logits = self.decode_from_context(context, decoder_tokens, causal=False)
        probabilities = nn.functional.softmax(logits, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        predictions = probabilities.argmax(dim=-1)
        result = {
            "mean_token_entropy": float(entropy.mean().item()),
            "mean_digit_accuracy": float((predictions == targets).float().mean().item()),
        }
        if self.is_set:
            recovered = []
            for predicted, target in zip(predictions, targets):
                recovered.append(
                    float(sorted(predicted.tolist()) == sorted(target.tolist()))
                )
            result["set_token_recovery"] = sum(recovered) / len(recovered)
        return result


def parallel_gradient_norm(model: S2RParallelIDModel) -> float:
    squared = 0.0
    for parameter in model.decoder.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(squared)
