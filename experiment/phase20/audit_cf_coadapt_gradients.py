"""Post-hoc score-gradient comparison; no model updates or new CF forwards."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiment.phase20.cf_coadapt_core import component, dense_ce
from experiment.phase20.future_factor_screen import sha
from experiment.phase20.future_label_multiseed import write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/cf_coadapt_toys_v1'


def describe(a):
    return dict(mean=float(a.mean()), minimum=float(a.min()),
                p10=float(np.quantile(a, .1)), median=float(np.median(a)), p90=float(np.quantile(a, .9)))


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    start = time.monotonic()
    with np.load(OUT / 'queries.npz', allow_pickle=False) as q:
        a = {k: q[k] for k in ('positions', 'base', 'pop_z', 'reliability', 'nfit')}
    n = int(a['nfit']); ix = np.flatnonzero((a['positions'][:n] >= 0) & (a['reliability'][:n] > 0))
    results = {}; paths = [Path(__file__), ROOT / 'experiment/phase20/cf_coadapt_core.py', OUT / 'queries.npz']
    for name in ('frozen', 'coadapt'):
        path = OUT / f'{name}.predictions.npz'; paths.append(path)
        with np.load(path, allow_pickle=False) as p: cf = p['cf']
        cosine, ratio, standalone_probability, fused_probability = [], [], [], []
        for off in range(0, len(ix), 128):
            sl = ix[off:off + 128]
            score = torch.as_tensor(cf[:, sl], dtype=torch.float64).requires_grad_()
            c = component(score, torch.as_tensor(a['pop_z'][sl]), torch.as_tensor(a['reliability'][sl]))
            pos = torch.as_tensor(a['positions'][sl]); base = torch.as_tensor(a['base'][sl])
            gradients = [torch.autograd.grad(dense_ce(value, pos).sum(), score, retain_graph=True)[0].permute(1, 0, 2).reshape(len(sl), -1)
                         for value in (c, c + base)]
            norm = [g.norm(dim=1) for g in gradients]; assert min(float(x.min()) for x in norm) > 0
            cosine.extend(((gradients[0]*gradients[1]).sum(1)/(norm[0]*norm[1])).detach().numpy())
            ratio.extend((norm[1]/norm[0]).detach().numpy())
            standalone_probability.extend(torch.softmax(c, -1)[torch.arange(len(sl)), pos].detach().numpy())
            fused_probability.extend(torch.softmax(c + base, -1)[torch.arange(len(sl)), pos].detach().numpy())
        cos = np.asarray(cosine)
        results[name] = dict(queries=len(ix), per_query_gradient_cosine=describe(cos),
            cosine_above_095=float((cos > .95).mean()), cosine_above_099=float((cos > .99).mean()),
            fused_to_candidate_gradient_norm_ratio=describe(np.asarray(ratio)),
            candidate_target_probability=describe(np.asarray(standalone_probability)),
            fused_target_probability=describe(np.asarray(fused_probability)))
    result = dict(state='ANALYZED', post_hoc=True, results=results,
        identical_full_catalog_loss_rows=int((a['positions'][:n] < 0).sum()), total_fit=n,
        covered_zero_reliability_rows=int(((a['positions'][:n]>=0)&(a['reliability'][:n]==0)).sum()),
        elapsed_seconds=time.monotonic()-start, optimizer_updates=0, new_model_forwards=0, test_read=False,
        interpretation_limit='Gradients w.r.t. the 3x50 CF score outputs, not CF/adapter parameters. Similar directions do not prove equal parameter updates or a causal reason for validation results.',
        sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    write(OUT / 'gradient_diagnostic.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
