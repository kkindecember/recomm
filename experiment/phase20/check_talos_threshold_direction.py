"""Train-only derivative diagnostic of a moving exact Top10 threshold; no optimizer steps."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from experiment.phase20.future_factor_screen import Factor
from experiment.phase20.future_fusion_alignment import standardize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/talos_candidate_screen_20260917'


def main():
    torch.set_num_threads(4); started = time.monotonic(); results = {}; hashes = {}
    def read(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as f: return {k: f[k] for k in f.files}
    for d in ('Toys', 'Beauty'):
        ctx = read(ROOT / f'artifacts/phase20/future_fusion_alignment_20260917/{d}.training_context.npz')
        a = read(ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{d}/training_cache.npz')
        ix = ctx['fit_indices']; x = torch.tensor(a['features'][ix]); prior = torch.tensor(ctx['prior']); cf = torch.tensor(ctx['cf_mean'])
        pos = torch.tensor(a['positions'][ix]); head = a['head'][ix]
        weights = torch.tensor(np.where(head, .5 / head.sum(), .5 / (~head).sum()))
        torch.manual_seed(20260917); model = Factor('next_only'); params = list(model.parameters())
        initial = [p.detach().clone() for p in params]
        beta0 = torch.topk(standardize(prior) + cf, 10, dim=1).values[:, -1:].detach()

        def compute(stop, backward=False, fixed=False):
            if backward: model.zero_grad(set_to_none=True)
            total = 0.
            with torch.set_grad_enabled(backward):
                for off in range(0, len(x), 256):
                    sl = slice(off, off + 256); l, _ = model(x[sl], prior[sl]); s = standardize(l) + cf[sl]
                    beta = beta0[sl] if fixed else torch.topk(s, 10, dim=1).values[:, -1:]
                    if stop: beta = beta.detach()
                    values = F.logsigmoid(s - beta); positive = values.gather(1, pos[sl, None]).squeeze(1)
                    negative = values.masked_fill(torch.arange(50)[None] == pos[sl, None], -torch.inf)
                    loss = ((torch.logsumexp(negative, 1) - positive) * weights[sl]).sum()
                    if backward: loss.backward()
                    total += float(loss.detach())
            gradient = torch.cat([p.grad.flatten() for p in params]).double() if backward else None
            return total, gradient

        value, partial = compute(True, backward=True)
        value2, full = compute(False, backward=True); assert abs(value - value2) < 1e-12
        rows = []
        for name, grad in (('stopped_threshold', partial), ('full_threshold', full)):
            direction = -grad / grad.norm()
            predicted = float(full @ direction)
            predicted_fixed = float(partial @ direction)
            for eps in (1e-3, 1e-4):
                observations = {}
                for sign in (-1, 1):
                    cursor = 0
                    with torch.no_grad():
                        for p, init in zip(params, initial):
                            size = p.numel(); p.copy_(init + sign * eps * direction[cursor:cursor + size].reshape(p.shape).to(p.dtype)); cursor += size
                    observations[sign] = dict(moving=compute(False)[0], fixed=compute(True, fixed=True)[0])
                rows.append(dict(direction=name, epsilon=eps, initial_loss=value,
                    predicted_moving_derivative=predicted, finite_difference_moving_derivative=(observations[1]['moving']-observations[-1]['moving'])/(2*eps),
                    predicted_fixed_derivative=predicted_fixed, finite_difference_fixed_derivative=(observations[1]['fixed']-observations[-1]['fixed'])/(2*eps),
                    plus_moving_loss=observations[1]['moving'], plus_fixed_loss=observations[1]['fixed']))
            with torch.no_grad():
                for p, init in zip(params, initial): p.copy_(init)
        results[d] = dict(prefixes=len(x), seed=20260917, partial_norm=float(partial.norm()), full_norm=float(full.norm()),
            gradient_cosine=float(partial @ full / (partial.norm()*full.norm())), directions=rows)
    result = dict(state='COMPLETED', analysis_timing='post_outcome_train_only_diagnostic', training_steps=0, optimizer_calls=0,
                  test_read=False, validation_read=False, domains=results, input_sha256=hashes,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), elapsed_seconds=time.monotonic()-started)
    (OUT / 'threshold_direction.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k,v in result.items() if k!='input_sha256'}, indent=2))


if __name__ == '__main__': main()
