"""Deterministic cohort selection and a bounded, continuous LR transition."""

import hashlib
import math


def cohort_indices(records, dataset, seed, size):
    users = [str(row['user_id']) for row in records]
    if len(set(users)) != len(users):
        raise ValueError('Expected one validation record per user')
    if not 0 < size <= len(users):
        raise ValueError('Invalid validation subset size')
    return sorted(range(len(users)), key=lambda i: (
        hashlib.sha256(f'{seed}|{dataset}|{users[i]}'.encode()).hexdigest(), users[i]))[:size]


class BudgetSchedule:
    def __init__(self, start_lrs, peak_lrs, warmup_steps, total_steps):
        if len(start_lrs) != len(peak_lrs) or not 0 < warmup_steps < total_steps:
            raise ValueError('Invalid budget schedule')
        if any(not math.isfinite(x) or x < 0 for x in [*start_lrs, *peak_lrs]):
            raise ValueError('Invalid learning rate')
        self.start_lrs = list(start_lrs)
        self.peak_lrs = list(peak_lrs)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def at(self, completed_steps):
        if not 0 <= completed_steps <= self.total_steps:
            raise ValueError('Step outside declared budget')
        if completed_steps <= self.warmup_steps:
            fraction = completed_steps / self.warmup_steps
            return [start + (peak - start) * fraction for start, peak in zip(self.start_lrs, self.peak_lrs)]
        fraction = (completed_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return [peak * 0.5 * (1 + math.cos(math.pi * fraction)) for peak in self.peak_lrs]

    def apply(self, optimizer, completed_steps):
        for group, lr in zip(optimizer.param_groups, self.at(completed_steps)):
            group['lr'] = lr

    def state_dict(self):
        return dict(start_lrs=self.start_lrs, peak_lrs=self.peak_lrs,
                    warmup_steps=self.warmup_steps, total_steps=self.total_steps)


def optimizer_steps(state):
    steps = [int(float(row['step'])) for row in state['state'].values() if 'step' in row]
    if not steps:
        raise ValueError('Checkpoint has no optimizer updates')
    return max(steps)
