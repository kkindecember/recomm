"""
Model design code refers to FID code:https://github.com/facebookresearch/FiD/blob/main/src/model.py.
########################
"""

import torch
from torch import nn
from torch.nn import functional as F
from .gram_t5 import T5ForConditionalGeneration_GRAM
from .gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions


class GRAM(T5ForConditionalGeneration_GRAM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.max_item_num = config.max_item_num

        self.use_position_embedding = config.use_position_embedding

        if self.use_position_embedding:
            pos_emb_size = self.max_item_num + 1  # one for coarse-grained user prompt
            self.position_embedding = nn.Embedding(pos_emb_size, self.config.d_model)
            self.init_position_embedding()
        else:
            self.position_embedding = None

        self.wrap_encoder()

    def init_position_embedding(self):
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward_(self, **kwargs):
        if "input_ids" in kwargs:
            kwargs["input_ids"] = kwargs["input_ids"].reshape(
                kwargs["input_ids"].size(0), -1
            )
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = kwargs["attention_mask"].reshape(
                kwargs["attention_mask"].size(0), -1
            )

        return super(GRAM, self).forward(**kwargs)

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        history_item_ids=None,
        history_item_mask=None,
        target_item_ids=None,
        **kwargs,
    ):
        # print(f">>> inside GRAM --- 1"); embed()
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.reshape(
                input_ids.size(0), -1
            )  # B x N x L -> B x (N * L)
        if attention_mask != None:
            attention_mask = attention_mask.reshape(
                attention_mask.size(0), -1
            )  # B x N x L -> B x (N * L)

        if getattr(self.config, "cf0_enabled", False) and input_ids is not None:
            self.encoder.set_cf0_context(history_item_ids, history_item_mask)

        # encoder_outputs -> beam_search()에서 forward()를 사용하기 때문에 추가
        outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        self.last_loss_components = None
        if (
            getattr(self.config, "cf0_enabled", False)
            and target_item_ids is not None
            and self.encoder.last_cf0_user_state is not None
        ):
            cf_logits = self.encoder.score_all_items()
            cf_loss = F.cross_entropy(cf_logits, target_item_ids)
            generation_loss = outputs[0]
            total_loss = generation_loss + self.config.cf0_loss_weight * cf_loss
            self.last_loss_components = {
                "generation": generation_loss.detach(),
                "cf0_item": cf_loss.detach(),
                "total": total_loss.detach(),
            }
            if isinstance(outputs, tuple):
                outputs = (total_loss,) + outputs[1:]
            else:
                outputs.loss = total_loss
        return outputs

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(
        self,
        input_ids,
        attention_mask,
        max_length,
        history_item_ids=None,
        history_item_mask=None,
        **kwargs,
    ):
        self.encoder.n_passages = input_ids.size(1)
        if getattr(self.config, "cf0_enabled", False):
            self.encoder.set_cf0_context(history_item_ids, history_item_mask)
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids2 = input_ids.reshape(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask2 = attention_mask.reshape(attention_mask.size(0), -1)

        last_hidden_states = self.encoder(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            return_dict=True,
        )[0]
        encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states
        )

        outputs = super().generate(
            input_ids=input_ids.reshape(input_ids.size(0), -1),
            attention_mask=attention_mask.reshape(attention_mask.size(0), -1),
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            **kwargs
        )

        if kwargs.get("output_hidden_states"):
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state[
                0:1, :, :
            ]  # only first beam
            outputs["encoder_outputs"] = encoder_outputs

        return outputs

    def score_cf0_candidates(self, candidate_item_ids):
        return self.encoder.score_candidates(candidate_item_ids)

    def get_crossattention_scores(self, cross_attentions, attention_mask, b_idx=0):
        """
        ## beam size must be 1 for get_crossattention_scores
        cross_attentions: list(#gen tokens: varies) of list(#layers:6) of (beam (1), n_heads, n-th beam (1), n_passages * text_maxlength)
        attention_mask: torch.tensor (bsz, n_passages, text_maxlength)
        """

        # Assuming that the cross_attentions are arranged as a list of [gen tokens][layers], where each element is
        # a tensor of shape (bsz, n_heads, 1, n_passages * text_maxlength)
        cross_attentions_first_token = [
            cross_attention_token[b_idx]
            for cross_attention_token in cross_attentions[0]
        ]
        cross_attentions_first_token = torch.stack(
            cross_attentions_first_token
        )  ## (n_layers, n_heads, 1, n_passages * text_maxlength)
        # Consider only first token
        bsz, n_passages, text_maxlength = attention_mask.size()
        n_layers, n_heads, _, _ = cross_attentions_first_token.size()

        scores = cross_attentions_first_token.view(
            bsz, n_layers, n_heads, n_passages, -1
        )
        scores = scores.masked_fill(~attention_mask[:, None, None], 0.0)
        token_scores = scores.sum(dim=[1, 2]).squeeze(0).tolist()
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = attention_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores / ntokens

        return token_scores, scores

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(
            encoder=self.encoder,
            config=self.config,
            use_checkpoint=use_checkpoint,
            position_embedding=self.position_embedding,
        )

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None


class EncoderWrapper(nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(
        self, encoder, config=None, use_checkpoint=False, position_embedding=None
    ):
        super().__init__()
        # print(f"> WARN: main_input_name not found in encoder, transformer version might be too old")
        self.main_input_name = encoder.main_input_name
        self.encoder = encoder
        self.config = config
        self.position_embedding = position_embedding
        self.cf0_enabled = bool(getattr(config, "cf0_enabled", False))
        self.cf0_arm = getattr(config, "cf0_arm", "A")
        self.cf0_history_item_ids = None
        self.cf0_history_item_mask = None
        self.last_cf0_user_state = None
        self.last_cf0_gate_mean = None
        if self.cf0_enabled:
            num_items = int(config.cf0_num_items)
            self.cf0_item_embedding = nn.Embedding(
                num_items + 1, config.d_model, padding_idx=0
            )
            self.cf0_position_embedding = nn.Embedding(
                config.max_item_num, config.d_model
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.cf0_num_heads,
                dim_feedforward=4 * config.d_model,
                dropout=config.cf0_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.cf0_transformer = nn.TransformerEncoder(
                layer, num_layers=config.cf0_num_layers
            )
            self.cf0_sequence_norm = nn.LayerNorm(config.d_model)
            self.cf0_gate = nn.Linear(2 * config.d_model, config.d_model)
            self.cf0_token_norm = nn.LayerNorm(config.d_model)
            self.cf0_injection_scale = float(config.cf0_injection_scale)

        # ----- Phase-12 HI-GRAM (Hierarchical Interaction) -----
        self.hi_gram_enabled = bool(getattr(config, "hi_gram_enabled", False))
        self.last_hi_gram_alpha = None
        if self.hi_gram_enabled:
            self.hi_gram_local_window = int(config.hi_gram_local_window)
            self.hi_gram_num_heads = int(config.hi_gram_num_heads)
            self.hi_gram_dropout = float(config.hi_gram_dropout)
            self.hi_gram_include_user_prompt = bool(
                getattr(config, "hi_gram_include_user_prompt", False)
            )
            # Item-position embedding, indexed by passage position (0..max_item_num).
            # +1 to cover the optional user-prompt slot when include_user_prompt=True.
            self.hi_gram_item_position = nn.Embedding(
                config.max_item_num + 1, config.d_model
            )
            nn.init.normal_(self.hi_gram_item_position.weight, std=0.02)
            local_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=self.hi_gram_num_heads,
                dim_feedforward=4 * config.d_model,
                dropout=self.hi_gram_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.hi_gram_local_attn = nn.TransformerEncoder(
                local_layer, num_layers=int(config.hi_gram_local_layers)
            )
            global_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=self.hi_gram_num_heads,
                dim_feedforward=4 * config.d_model,
                dropout=self.hi_gram_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.hi_gram_global_attn = nn.TransformerEncoder(
                global_layer, num_layers=int(config.hi_gram_global_layers)
            )
            self.hi_gram_token_norm = nn.LayerNorm(config.d_model)
            # Learnable fusion scalar alpha; raw scalar (v1). Initialized small so that
            # early in training the model behaves close to the original GRAM path.
            self.hi_gram_fusion_scale = nn.Parameter(
                torch.tensor(float(config.hi_gram_fusion_scale_init))
            )

        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def set_cf0_context(self, history_item_ids, history_item_mask=None):
        if not self.cf0_enabled:
            return
        self.cf0_history_item_ids = history_item_ids
        self.cf0_history_item_mask = history_item_mask
        self.last_cf0_user_state = None
        self.last_cf0_gate_mean = None

    @staticmethod
    def _reverse_valid_prefix(values, valid_mask):
        """Reverse only the left-aligned valid prefix of a padded sequence."""
        bsz, length = valid_mask.shape
        positions = torch.arange(length, device=values.device).expand(bsz, length)
        valid_lengths = valid_mask.long().sum(dim=1, keepdim=True)
        gather_index = torch.where(
            positions < valid_lengths, valid_lengths - 1 - positions, positions
        ).clamp_min(0)
        if values.dim() == 3:
            gather_index = gather_index.unsqueeze(-1).expand_as(values)
        return values.gather(1, gather_index)

    def _apply_cf0(self, last_hidden_states, attention_mask, bsz, passage_length):
        if self.cf0_history_item_ids is None:
            self.last_cf0_user_state = None
            return last_hidden_states

        passage_count = self.n_passages
        history_count = passage_count - 1
        history_item_ids = self.cf0_history_item_ids[:, :history_count].to(
            last_hidden_states.device
        )
        if self.cf0_history_item_mask is None:
            history_item_mask = history_item_ids.ne(0)
        else:
            history_item_mask = self.cf0_history_item_mask[:, :history_count].to(
                last_hidden_states.device
            ).bool()
        if not history_item_mask.any():
            self.last_cf0_user_state = None
            return last_hidden_states
        if int(history_item_ids.max()) > self.config.cf0_num_items:
            raise ValueError("CF0 item index exceeds configured catalog size")

        passage_states = last_hidden_states.reshape(
            bsz, passage_count, passage_length, -1
        ).clone()
        passage_masks = attention_mask.reshape(
            bsz, passage_count, passage_length
        ).bool()
        item_states = passage_states[:, 1 : history_count + 1]
        item_token_masks = passage_masks[:, 1 : history_count + 1]
        denom = item_token_masks.sum(dim=2, keepdim=True).clamp_min(1)
        semantic_items = (
            item_states * item_token_masks.unsqueeze(-1)
        ).sum(dim=2) / denom

        chronological_ids = self._reverse_valid_prefix(
            history_item_ids, history_item_mask
        )
        chronological_mask = self._reverse_valid_prefix(
            history_item_mask, history_item_mask
        )
        chronological_semantic = self._reverse_valid_prefix(
            semantic_items, history_item_mask
        )
        positions = torch.arange(history_count, device=last_hidden_states.device)
        collaborative_input = self.cf0_item_embedding(chronological_ids)
        collaborative_input = collaborative_input + self.cf0_position_embedding(
            positions
        ).unsqueeze(0)
        causal_mask = torch.ones(
            history_count,
            history_count,
            dtype=torch.bool,
            device=last_hidden_states.device,
        ).triu(1)
        collaborative_states = self.cf0_transformer(
            collaborative_input,
            mask=causal_mask,
            src_key_padding_mask=~chronological_mask,
        )
        collaborative_states = self.cf0_sequence_norm(collaborative_states)
        lengths = chronological_mask.long().sum(dim=1).clamp_min(1)
        user_state = collaborative_states[
            torch.arange(bsz, device=last_hidden_states.device), lengths - 1
        ]
        self.last_cf0_user_state = user_state

        collaborative_passage_order = self._reverse_valid_prefix(
            collaborative_states, chronological_mask
        )
        if self.cf0_arm == "C":
            gate = torch.sigmoid(
                self.cf0_gate(
                    torch.cat(
                        [chronological_semantic, collaborative_states], dim=-1
                    )
                )
            )
            gate = self._reverse_valid_prefix(gate, chronological_mask)
            injected_items = gate * collaborative_passage_order
            self.last_cf0_gate_mean = gate[history_item_mask].mean().detach()
        else:
            injected_items = (
                self.cf0_injection_scale * collaborative_passage_order
            )

        item_states = self.cf0_token_norm(
            item_states + injected_items.unsqueeze(2)
        )
        passage_states[:, 1 : history_count + 1] = item_states
        passage_states[:, 0] = self.cf0_token_norm(
            passage_states[:, 0]
            + self.cf0_injection_scale * user_state.unsqueeze(1)
        )
        return passage_states.reshape(bsz * passage_count, passage_length, -1)

    def _build_local_window_mask(self, n_items, device):
        """Symmetric local window attention mask of shape (n_items, n_items).

        True means the position is masked (not attended to). Item i attends to
        positions in [i - W + 1, i + W - 1] (bidirectional half-width W-1).
        """
        window = self.hi_gram_local_window
        positions = torch.arange(n_items, device=device)
        diff = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        # Allowed if |i - j| < window
        allowed = diff < window
        return ~allowed  # PyTorch attention: True = masked out

    def _apply_hi_gram(
        self, last_hidden_states, attention_mask, bsz, passage_length
    ):
        """Phase-12 hierarchical cross-item interaction on encoder outputs.

        Shapes:
            last_hidden_states: (B*N, L, D)
            attention_mask:     (B*N, L)   — token-level valid mask
            bsz:                B
            passage_length:     L
        Returns:
            last_hidden_states with per-item residual bias added, shape (B*N, L, D).
        """
        passage_count = self.n_passages  # N (= 1 user prompt + up to max_his items)
        d_model = last_hidden_states.size(-1)
        device = last_hidden_states.device

        # (B, N, L, D)
        per_item = last_hidden_states.reshape(
            bsz, passage_count, passage_length, d_model
        )
        # (B, N, L)
        per_item_mask = attention_mask.reshape(
            bsz, passage_count, passage_length
        ).bool()

        # Choose which passages participate in cross-item attention.
        # Default: skip passage 0 (user prompt). Only history items interact.
        if self.hi_gram_include_user_prompt:
            slice_start = 0
        else:
            slice_start = 1

        item_states = per_item[:, slice_start:]                     # (B, M, L, D)
        item_token_masks = per_item_mask[:, slice_start:]           # (B, M, L)
        n_items = item_states.size(1)                                # M

        if n_items <= 0:
            return last_hidden_states

        # Masked-mean pooling per passage → (B, M, D)
        token_valid = item_token_masks.unsqueeze(-1).float()         # (B, M, L, 1)
        denom = token_valid.sum(dim=2).clamp_min(1.0)                # (B, M, 1)
        pooled = (item_states * token_valid).sum(dim=2) / denom       # (B, M, D)

        # Per-item validity: True if this passage has any valid token
        item_valid_mask = item_token_masks.any(dim=-1)               # (B, M)

        # Short-circuit: if no sample in the batch has any valid item, no
        # cross-item attention is possible. Returning early also avoids the
        # PyTorch all-key-padding NaN pathology (softmax over all -inf).
        if not item_valid_mask.any():
            self.last_hi_gram_alpha = self.hi_gram_fusion_scale.detach().clone()
            return last_hidden_states

        # Add item-position embedding. When include_user_prompt=True, position 0 is
        # the user-prompt slot; otherwise position 0 is the first history item.
        pos_ids = torch.arange(n_items, device=device)
        pos_embed = self.hi_gram_item_position(pos_ids).unsqueeze(0)  # (1, M, D)
        pooled_pos = pooled + pos_embed                              # (B, M, D)

        # Local attention (symmetric window)
        local_mask = self._build_local_window_mask(n_items, device)  # (M, M) bool
        local_out = self.hi_gram_local_attn(
            pooled_pos,
            mask=local_mask,
            src_key_padding_mask=~item_valid_mask,
        )                                                            # (B, M, D)

        # Global attention (no positional restriction; padding masked)
        global_out = self.hi_gram_global_attn(
            local_out,
            src_key_padding_mask=~item_valid_mask,
        )                                                            # (B, M, D)

        # Residual bias: only inject the increment produced by cross-item interaction.
        # For samples/positions whose keys were entirely padded, attention may return
        # NaN even though we masked at attention time; replace those with zero bias.
        item_bias = global_out - pooled                              # (B, M, D)
        zero_bias = torch.zeros_like(item_bias)
        # Mask invalid item slots (padding passages) to zero bias — use torch.where so
        # NaN produced by softmax-over-all-mask is dropped rather than multiplied.
        item_bias = torch.where(
            item_valid_mask.unsqueeze(-1), item_bias, zero_bias
        )
        # LayerNorm the bias itself (not the sum) so that α=0 is a true no-op.
        item_bias = self.hi_gram_token_norm(item_bias)

        # Broadcast per-item bias to token level and add with learnable scaling
        item_bias_tokens = item_bias.unsqueeze(2).expand(
            -1, -1, passage_length, -1
        )                                                            # (B, M, L, D)
        # Only add bias on valid tokens (padding tokens keep original hidden)
        item_bias_tokens = item_bias_tokens * token_valid            # (B, M, L, D)

        updated_item_states = item_states + self.hi_gram_fusion_scale * item_bias_tokens

        per_item_out = per_item.clone()
        per_item_out[:, slice_start:] = updated_item_states

        # Record alpha for diagnostics (detached scalar)
        self.last_hi_gram_alpha = self.hi_gram_fusion_scale.detach().clone()

        return per_item_out.reshape(bsz * passage_count, passage_length, d_model)

    def score_all_items(self):
        if self.last_cf0_user_state is None:
            raise RuntimeError("CF0 user state is unavailable")
        scale = self.config.d_model**0.5
        return F.linear(
            self.last_cf0_user_state, self.cf0_item_embedding.weight
        ) / scale

    def score_candidates(self, candidate_item_ids):
        if self.last_cf0_user_state is None:
            raise RuntimeError("CF0 user state is unavailable")
        candidate_item_ids = candidate_item_ids.to(
            self.last_cf0_user_state.device
        )
        candidate_embeddings = self.cf0_item_embedding(candidate_item_ids)
        return torch.einsum(
            "bd,bkd->bk", self.last_cf0_user_state, candidate_embeddings
        ) / (self.config.d_model**0.5)

    def forward(
        self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # print(f">>> inside EncoderWrapper  --- 3"); embed()
        if input_ids is not None:
            # total_length = n_passages * passage_length
            bsz, total_length = input_ids.shape  # B x (N * L)
            passage_length = total_length // self.n_passages  # L
            input_ids = input_ids.reshape(
                bsz * self.n_passages, passage_length
            )  # B x (N * L) -> (B * N) x L
            attention_mask = attention_mask.reshape(
                bsz * self.n_passages, passage_length
            )  # B x (N * L) -> (B * N) x L
            outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )  # tuple ( (B * N) x L x D, )

        elif inputs_embeds is not None:
            bsz, total_length, _ = inputs_embeds.shape  # B x (N * L) x D
            passage_length = total_length // self.n_passages  # L
            inputs_embeds = inputs_embeds.reshape(
                bsz * self.n_passages, passage_length, -1
            )  # B x (N * L) x D -> (B * N) x L x D
            attention_mask = attention_mask.reshape(
                bsz * self.n_passages, passage_length
            )  # B x (N * L) -> (B * N) x L
            outputs = self.encoder(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
            )  # tuple ( (B * N) x L x D, )

        else:
            raise ValueError(
                "At least one of input_ids or inputs_embeds should be not None"
            )

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if self.position_embedding is not None:
            last_hidden_states = outputs[0]  # (B * N) x L x D
            position_ids = torch.arange(self.n_passages, device=device).expand(
                bsz, self.n_passages
            )  # N -> B x N
            position_embeddings = self.position_embedding(position_ids)  # B x N x D
            position_embeddings = position_embeddings.reshape(
                bsz * self.n_passages, 1, -1
            )  # (B * N) x 1 x D
            last_hidden_states = (
                last_hidden_states + position_embeddings
            )  # (B * N) x L x D
        else:
            last_hidden_states = outputs[0]
        if self.cf0_enabled:
            last_hidden_states = self._apply_cf0(
                last_hidden_states, attention_mask, bsz, passage_length
            )
        if self.hi_gram_enabled:
            last_hidden_states = self._apply_hi_gram(
                last_hidden_states, attention_mask, bsz, passage_length
            )
        # tuple ( (B * N) x L x D, ) -> (B x (N * L) x D, )
        outputs = (
            last_hidden_states.reshape(bsz, self.n_passages * passage_length, -1),
        ) + outputs[1:]
        return outputs  # tuple ( B x (N * L) x D, )


class CheckpointWrapper(nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [], dtype=torch.float, device=output[0].device, requires_grad=True
                )
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward, hidden_states, attention_mask, position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block
