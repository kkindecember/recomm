"""Prepared train/validation adapters; memory lookup never takes a query target."""
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from .diff_data import DiffDataset


def sequence_tensors(records, max_length=20):
    ids = torch.zeros(len(records), max_length, dtype=torch.long)
    lengths = torch.empty(len(records), dtype=torch.long)
    for i, row in enumerate(records):
        history = row['history'][-max_length:]
        if not history or any(item <= 0 for item in history):
            raise ValueError('Empty or invalid history')
        ids[i, :len(history)] = torch.tensor(history)
        lengths[i] = len(history)
    return ids, lengths, torch.tensor([r['target'] for r in records])


def training_frequencies(records, num_items):
    # Prepared data has one row per training next-item event. Counting every
    # history would overcount prefix repetitions; add each user's initial item once.
    first = {}
    counts = Counter(r['target'] for r in records)
    for r in records:
        if len(r['history']) == 1 and r['user_id'] not in first:
            first[r['user_id']] = r['history'][0]
    if set(first) != {r['user_id'] for r in records}:
        raise ValueError('Missing initial training prefixes')
    counts.update(first.values())
    return np.asarray([counts[i] for i in range(num_items)], dtype=np.int64)


class PositiveCases:
    def __init__(self, records):
        self.by_target = defaultdict(list)
        for i, r in enumerate(records):
            self.by_target[r['target']].append(i)
        self.targets = [r['target'] for r in records]

    def sample(self, indices, rng):
        result = []
        for i in indices:
            i = int(i)
            choices = self.by_target[self.targets[i]]
            if len(choices) == 1:
                result.append(i)  # Upstream same-target sampling falls back to self.
            else:
                choice = choices[int(rng.integers(len(choices)))]
                while choice == i:
                    choice = choices[int(rng.integers(len(choices)))]
                result.append(choice)
        return torch.tensor(result, dtype=torch.long)


class MemoryDataset(DiffDataset):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        row['memory_index'] = index
        return row


class MemoryCollator:
    def __init__(self, native, queries, indices, keys, values):
        self.native, self.queries, self.indices = native, queries, indices
        self.keys, self.values = keys, values

    def __call__(self, rows):
        batch = self.native(rows)
        indexes = torch.tensor([row['memory_index'] for row in rows])
        case_ids = self.indices[indexes]
        batch['memory_query'] = self.queries[indexes]
        batch['memory_keys'] = self.keys[case_ids]
        batch['memory_values'] = self.values[case_ids]
        return batch


def memory_batch(batch, device, training):
    result = {name: batch[name].to(device) for name in
              ('history_item_ids', 'memory_query', 'memory_keys', 'memory_values')}
    result['input_ids'] = batch['item_text_ids'].to(device)
    result['attention_mask'] = batch['item_text_masks'].to(device)
    if training:
        result['labels'] = batch['target_ids'].to(device)
        result['target_item_ids'] = batch['target_item_ids'].to(device)
    return result
