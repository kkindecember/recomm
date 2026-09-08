"""DIFF history memories consumed by the native GRAM lexical decoder."""

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from .diff_sequence import DiffSequence


class DiffGram(nn.Module):
    def __init__(self, gram, num_items, attribute_tables, attribute_sizes, config):
        super().__init__()
        self.gram = gram
        self.diff = DiffSequence(num_items, attribute_tables, attribute_sizes, config)
        self.memory_projection = nn.Linear(config.hidden_size, gram.config.d_model)
        self.memory_norm = nn.LayerNorm(gram.config.d_model)
        self.last_loss_components = None
        self.backbone_frozen = False

    def set_backbone_frozen(self, frozen):
        self.backbone_frozen = bool(frozen)
        for parameter in self.gram.parameters():
            parameter.requires_grad_(not frozen)
        self.gram.train(self.training and not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.gram.eval()
        return self

    def encode(self, input_ids, attention_mask, history_item_ids, with_alignment=False):
        if input_ids.ndim != 3 or input_ids.shape != attention_mask.shape:
            raise ValueError("GRAM inputs must be matching B x passages x tokens")
        if history_item_ids.size(0) != input_ids.size(0):
            raise ValueError("History batch mismatch")
        passage_valid = attention_mask[:, 1:].bool().any(-1)
        if history_item_ids.shape != passage_valid.shape or not torch.equal(history_item_ids.ne(0), passage_valid):
            raise ValueError("DIFF item histories must align with native history passages")
        self.gram.encoder.n_passages = input_ids.size(1)
        self.gram.encoder.set_migration_context(None, None)
        native = self.gram.encoder(input_ids=input_ids.flatten(1), attention_mask=attention_mask.flatten(1),
                                   return_dict=True)[0]
        memory, valid, user, alignment = self.diff(history_item_ids, compute_alignment=with_alignment)
        memory = self.memory_norm(self.memory_projection(memory))
        memory = memory * valid.unsqueeze(-1).to(memory.dtype)
        return (BaseModelOutput(last_hidden_state=torch.cat([native, memory], dim=1)),
                torch.cat([attention_mask.flatten(1).bool(), valid], dim=1), user, alignment)

    def forward(self, input_ids, attention_mask, history_item_ids, labels, target_item_ids):
        encoded, mask, user, align = self.encode(input_ids, attention_mask, history_item_ids, with_alignment=True)
        result = self.gram(encoder_outputs=encoded, attention_mask=mask, labels=labels,
                           use_cache=False, return_dict=True)
        generation = result.loss
        item = self.diff.item_loss(user, target_item_ids)
        result.loss = generation + self.diff.config.auxiliary_weight * (
            item + self.diff.config.alignment_weight * align)
        self.last_loss_components = {"generation": generation.detach(), "item": item.detach(),
                                     "alignment": align.detach(), "total": result.loss.detach()}
        return result

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, history_item_ids, **kwargs):
        # Bypass GRAM.generate's encoder call: the augmented memory is already
        # constructed. Its inherited generator still calls GRAM.forward, with
        # cached encoder_outputs, at every decoding step.
        from model.gram_t5 import T5ForConditionalGeneration_GRAM

        encoded, mask, _, _ = self.encode(input_ids, attention_mask, history_item_ids)
        return T5ForConditionalGeneration_GRAM.generate(
            self.gram, input_ids=input_ids.flatten(1), attention_mask=mask,
            encoder_outputs=encoded, **kwargs)
