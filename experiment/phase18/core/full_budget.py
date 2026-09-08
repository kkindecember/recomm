"""Join an original long LR curve without discarding completed short-run updates."""

import math


class FullBudgetSchedule:
    def __init__(self, optimizer, start_step, total_steps, warmup_steps,
                 peak_lrs, phase_offset_steps=0, bridge_steps=1, decay='cosine'):
        self.optimizer = optimizer
        self.start_step = int(start_step)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.phase_offset_steps = int(phase_offset_steps)
        self.bridge_steps = int(bridge_steps)
        self.decay = decay
        self.peak_lrs = list(peak_lrs)
        self.start_lrs = [float(group['lr']) for group in optimizer.param_groups]
        self.last_epoch = self.start_step
        if not (0 <= phase_offset_steps <= start_step < total_steps and
                0 < warmup_steps < total_steps - phase_offset_steps and
                0 < bridge_steps <= total_steps - start_step):
            raise ValueError('Invalid complete training horizon')
        if decay not in ('linear', 'cosine') or len(self.peak_lrs) != len(self.start_lrs):
            raise ValueError('Invalid decay or optimizer groups')
        if any(not math.isfinite(x) or x <= 0 for x in self.peak_lrs + self.start_lrs):
            raise ValueError('Only a still-active short schedule can join this curve')

    def original_at(self, step):
        position = step - self.phase_offset_steps
        horizon = self.total_steps - self.phase_offset_steps
        if position < self.warmup_steps:
            factor = position / self.warmup_steps
        else:
            fraction = (position - self.warmup_steps) / (horizon - self.warmup_steps)
            factor = 1 - fraction if self.decay == 'linear' else (1 + math.cos(math.pi * fraction)) / 2
        return [lr * factor for lr in self.peak_lrs]

    def at(self, step):
        if not self.start_step <= step <= self.total_steps:
            raise ValueError('Update outside complete training horizon')
        fraction = min(1., (step - self.start_step) / self.bridge_steps)
        return [(1 - fraction) * before + fraction * target
                for before, target in zip(self.start_lrs, self.original_at(step))]

    def step(self):
        next_step = self.last_epoch + 1
        rates = self.at(next_step)
        for group, lr in zip(self.optimizer.param_groups, rates):
            group['lr'] = lr
        self.last_epoch = next_step

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return dict(kind='full_budget_with_preserved_short_prefix', start_step=self.start_step,
                    last_epoch=self.last_epoch, total_steps=self.total_steps,
                    warmup_steps=self.warmup_steps, phase_offset_steps=self.phase_offset_steps,
                    bridge_steps=self.bridge_steps, decay=self.decay,
                    peak_lrs=self.peak_lrs, start_lrs=self.start_lrs,
                    last_lrs=self.get_last_lr())


def schedule_for(backend, spec):
    offset = spec['evaluation_epoch_offset'] * backend.per_epoch
    horizon = spec['total_epochs'] * backend.per_epoch
    warmup = (int((horizon - offset) * backend.config['training']['warmup_fraction'])
              if backend.family == 'diff_gram' else backend.config['overrides']['warmup_steps'])
    return FullBudgetSchedule(backend.optimizer, backend.initial_steps, horizon,
        max(1, warmup), backend.peak_lrs, phase_offset_steps=offset,
        bridge_steps=backend.per_epoch, decay='linear' if backend.family == 'diff_gram' else 'cosine')
