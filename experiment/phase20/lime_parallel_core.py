"""Cached exact collation and synchronous data parallelism over fixed update windows."""
import copy
import math

import torch
import torch.distributed as dist


class CachedCollator:
    """Memoize the original tokenizer transformations; retain original batch shapes."""
    def __init__(self, original):
        assert original.item_id_type == 'split'
        self.original = original
        self.inner = copy.copy(original)
        self.inner.encode_texts_split = self.encode_texts_split
        self.inner.encode_target_split = self.encode_target_split
        self.passages, self.targets = {}, {}

    def __call__(self, samples):
        return self.inner(samples)

    def encode_texts_split(self, texts, tokenizer):
        missing = list(dict.fromkeys(s for row in texts for s in row if s not in self.passages))
        for start in range(0, len(missing), 32):
            strings = missing[start:start+32]
            ids, masks = self.original.encode_texts_split([strings], tokenizer)
            for i, s in enumerate(strings):
                n = int(masks[0, i].sum())
                self.passages[s] = (ids[0, i, :n].clone(), masks[0, i, :n].clone())
        passages = min(max(map(len, texts)), self.original.max_item_num) + 1
        width = max(self.passages[s][0].numel() for row in texts for s in row)
        ids = torch.zeros(len(texts), passages, width, dtype=torch.long)
        masks = torch.zeros_like(ids, dtype=torch.bool)
        for i, row in enumerate(texts):
            for j, s in enumerate(row):
                a, b = self.passages[s]
                ids[i, j, :a.numel()] = a
                masks[i, j, :b.numel()] = b
        return ids, masks

    def encode_target_split(self, texts):
        missing = list(dict.fromkeys(s for s in texts if s not in self.targets))
        if missing:
            encoded = self.original.encode_target_split(missing)
            for i, s in enumerate(missing):
                n = int(encoded['attention_mask'][i].sum())
                self.targets[s] = encoded['input_ids'][i, :n].clone()
        width = max(self.targets[s].numel() for s in texts)
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        masks = torch.zeros_like(ids)
        for i, s in enumerate(texts):
            ids[i, :self.targets[s].numel()] = self.targets[s]
            masks[i, :self.targets[s].numel()] = 1
        return dict(input_ids=ids, attention_mask=masks)


def local_batches(batches, rank, world):
    """Partition original microbatches, with no duplication even at epoch end."""
    return batches[rank::world]


def sum_gradients(model, world):
    params = [p for p in model.parameters() if p.requires_grad]
    if world > 1:
        # One all-reduce per optimizer update, after local gradient accumulation.
        flat = torch._utils._flatten_dense_tensors([
            p.grad if p.grad is not None else torch.zeros_like(p) for p in params])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        for p, gradient in zip(params, torch._utils._unflatten_dense_tensors(flat, params)):
            if p.grad is None:
                p.grad = gradient.clone()
            else:
                p.grad.copy_(gradient)
    assert all(p.grad is not None for p in params)


def parallel_update(model, optimizer, scheduler, collator, dataset, batches,
                    stage, config, device, rank=0, world=1, stress=False):
    model.train()
    loss_total = torch.zeros((), device=device)
    shapes = []
    for indices in local_batches(batches, rank, world):
        samples = [dataset[i] for i in indices]
        batch = collator(samples)
        for key in ('item_text_ids', 'item_text_masks', 'target_ids'):
            batch[key] = batch[key].to(device)
        if stress:
            import torch.nn.functional as F
            b, p, w = batch['item_text_ids'].shape
            batch['item_text_ids'] = F.pad(batch['item_text_ids'], (0, 128-w, 0, 21-p))
            batch['item_text_masks'] = F.pad(batch['item_text_masks'], (0, 128-w, 0, 21-p))
            batch['target_ids'] = F.pad(batch['target_ids'], (0, 32-batch['target_ids'].shape[1]), value=-100)
        if stage == 'c1':
            targets = batch['target_item_ids'].to(device)-1
            loss, parts, logits = model.forward_loss(batch, targets, config['lambda_item'])
            del parts, logits
        else:
            loss = model(input_ids=batch['item_text_ids'], attention_mask=batch['item_text_masks'], labels=batch['target_ids']).loss
        # SUM collective means each microbatch keeps precisely the original 1/N weight.
        scaled = loss / len(batches)
        scaled.backward()
        loss_total += scaled.detach()
        shapes.append(list(batch['item_text_ids'].shape))
        del batch, loss, scaled
    sum_gradients(model, world)
    if world > 1:
        dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
    assert torch.isfinite(loss_total) and torch.isfinite(norm), 'Nonfinite distributed update'
    if stage == 'c1':
        assert float(model.catalog_head.weight.grad.norm()) > 0
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(loss=float(loss_total), gradient_norm=float(norm), rows=sum(map(len, batches)), shapes=shapes)
