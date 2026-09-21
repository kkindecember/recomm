"""Fixed two additional initializations; labels and deployment caches are frozen."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.future_factor_screen import Factor, old_score, z, ranks, metrics, sha

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/future_label_multiseed_20260917'
PARENT = ROOT / 'artifacts/phase20/future_factor_screen_20260917'
COMP = ROOT / 'artifacts/phase20/future_label_components_20260917'
CONTRACT = ROOT / 'plan/第二十阶段/未来软监督跨初始化与集成依赖_核查合同_20260917.md'
SEEDS = (20260917, 20260918, 20260919)
ARMS = ('next_only', 'global_local', 'soft_future')
FUSIONS = ('raw', 'current', 'mean')


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(values=a.tolist(), mean=float(a.mean()), sd=float(a.std(ddof=1)))


def compare(aa, bb):
    na = ((aa <= 10) / np.log2(aa + 1)).mean()
    nb = ((bb <= 10) / np.log2(bb + 1)).mean()
    return dict(delta_N10=float(na - nb), relative_N10=float(na / nb - 1),
        delta_H10=float((aa <= 10).mean() - (bb <= 10).mean()),
        hit_net=int((aa <= 10).sum() - (bb <= 10).sum()),
        hit_gained=int(((aa <= 10) & (bb > 10)).sum()),
        hit_lost=int(((aa > 10) & (bb <= 10)).sum()))


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False)
    start = time.monotonic(); steps = 0; hashes = {}; results = {}

    def status(state, **kw):
        write(OUT / 'status.json', dict(state=state, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            pid=os.getpid(), elapsed_seconds=time.monotonic() - start, training_steps=steps,
            planned_training_steps=360, gpu_used=False, test_read=False, automatic_next_experiment=False, **kw))

    def use(path):
        hashes[str(path.relative_to(ROOT))] = sha(path); return path

    def read(folder, name):
        with np.load(use(folder / name)) as f: return {k: f[k] for k in f.files}

    def timeout(sig, frame):
        raise TimeoutError('15 minute CPU budget')

    signal.signal(signal.SIGALRM, timeout); signal.alarm(900)
    try:
        status('RUNNING', stage='verify_frozen_inputs')
        use(Path(__file__)); use(CONTRACT); use(ROOT / 'experiment/phase20/future_factor_screen.py')
        prior_summary = json.loads(use(COMP / 'summary.json').read_text())
        prior_audit = json.loads(use(COMP / 'verification.json').read_text())
        assert prior_audit['state'] == 'VERIFIED'
        for group in (prior_summary['input_sha256'], prior_audit['output_sha256']):
            for p, h in group.items():
                use(ROOT / p); assert hashes[p] == h, p
        initialization = {}
        for seed in SEEDS:
            torch.manual_seed(seed); model = Factor('next_only')
            assert sum(p.numel() for p in model.parameters()) == 258
            initialization[str(seed)] = fingerprint(model)
        assert len(set(initialization.values())) == 3
        write(OUT / 'protocol.json', dict(contract=str(CONTRACT.relative_to(ROOT)), seeds=SEEDS,
            reused_seed=SEEDS[0], arms=ARMS, fusions=FUSIONS, steps_per_arm=30,
            new_optimizer_steps=360, threads=4, timeout_seconds=900,
            initialization_sha256=initialization, source_sha256=hashes.copy(), development_reused=True))
        for domain in ('Toys', 'Beauty'):
            status('RUNNING', dataset=domain, stage='load')
            stats = read(COMP, f'{domain}.label_statistics.npz')
            label_cache = read(COMP, f'{domain}.training_labels.npz')
            old = ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}'
            cache = read(old, 'training_cache.npz')
            ix = stats['fit_indices']; pos = cache['positions'][ix]; head = cache['head'][ix]
            np.testing.assert_array_equal(pos, stats['positions'])
            np.testing.assert_array_equal(head, stats['head']); assert cache['fit'][ix].all()
            onehot = (np.arange(50)[None] == pos[:, None]).astype(np.float64)
            targets = dict(next_only=onehot, global_local=label_cache['global_local'], soft_future=stats['actual'])
            x = torch.as_tensor(cache['features'][ix])
            g = old_score(x, torch.as_tensor(cache['base'][ix]), torch.load(use(old / 'B1.pt'), map_location='cpu'))
            prior = torch.as_tensor(z(g) - np.log(49))
            weights = torch.as_tensor(np.where(head, 1 / (2 * head.sum()), 1 / (2 * (~head).sum())))
            v = read(PARENT, f'{domain}.audit_validation_cache.npz')
            baseline = read(COMP, f'{domain}.predictions.npz')
            original = read(PARENT, f'{domain}.predictions.npz')
            for other in (baseline, original):
                np.testing.assert_array_equal(v['users'], other['users'])
                np.testing.assert_array_equal(v['candidate_ids'], other['candidate_ids'])
            in_fit = np.isin(v['users'], np.unique(stats['users']))
            np.testing.assert_array_equal(in_fit, baseline['in_fit'])
            masks = dict(all=np.ones(len(in_fit), bool), tail=v['tail_mask'], non_tail=~v['tail_mask'], fit=in_fit, nonfit=~in_fit)
            seed_results = {}
            for seed in SEEDS:
                scores = {}; curves = {}; initial_hashes = {}
                for arm in ARMS:
                    if seed == SEEDS[0]:
                        for fusion in FUSIONS:
                            name = arm + '.' + fusion
                            source = original if fusion == 'raw' and arm != 'global_local' else baseline
                            scores[name] = source['scores'][list(source['arm_names']).index(name)]
                        continue
                    torch.manual_seed(seed); model = Factor('next_only')
                    initial_hashes[arm] = fingerprint(model)
                    assert initial_hashes[arm] == initialization[str(seed)]
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
                    y = torch.as_tensor(targets[arm]); curve = []
                    for step in range(1, 31):
                        optimizer.zero_grad(set_to_none=True); total_loss = 0.
                        for offset in range(0, len(x), 256):
                            sl = slice(offset, offset + 256)
                            logits, _ = model(x[sl], prior[sl])
                            loss = (F.binary_cross_entropy_with_logits(logits, y[sl], reduction='none').mean(1) * weights[sl]).sum()
                            assert torch.isfinite(loss); loss.backward(); total_loss += float(loss.detach())
                        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                        optimizer.step(); steps += 1
                        curve.append(dict(step=step, loss=total_loss, grad_norm=float(grad)))
                        status('RUNNING', dataset=domain, seed=seed, arm=arm, stage='training', step=step)
                    curves[arm] = curve; model.eval()
                    torch.save(model.state_dict(), OUT / f'{domain}.{seed}.{arm}.pt')
                    chunks = []
                    with torch.no_grad():
                        for offset in range(0, len(v['users']), 256):
                            sl = slice(offset, offset + 256)
                            pred, _ = model(torch.as_tensor(v['features'][sl]), torch.as_tensor(v['prior'][sl]))
                            chunks.append(pred.numpy())
                    raw = z(np.concatenate(chunks))
                    for fusion in FUSIONS:
                        scores[arm + '.' + fusion] = raw if fusion == 'raw' else raw + v['cf_' + fusion]
                rr = {name: ranks(s, v['positions']) for name, s in scores.items()}
                met = {name: {group: metrics(rank[mask]) for group, mask in masks.items()} for name, rank in rr.items()}
                comps = {control: {fusion: {group: compare(rr['soft_future.' + fusion][mask], rr[control + '.' + fusion][mask])
                    for group, mask in masks.items()} for fusion in FUSIONS} for control in ('next_only', 'global_local')}
                seed_results[str(seed)] = dict(metrics=met, comparisons=comps)
                names = list(scores)
                np.savez_compressed(OUT / f'{domain}.{seed}.predictions.npz', users=v['users'], candidate_ids=v['candidate_ids'],
                    positions=v['positions'], tail_mask=v['tail_mask'], in_fit=in_fit, arm_names=np.asarray(names),
                    scores=np.stack([scores[n] for n in names]), ranks=np.stack([rr[n] for n in names]))
                write(OUT / f'{domain}.{seed}.training.json', dict(reused=seed == SEEDS[0], curves=curves,
                    initial_sha256=initial_hashes, prefixes=len(x), users=len(set(stats['users']))))
            aggregate_metrics = {name: {group: {key: describe([r['metrics'][name][group][key] for r in seed_results.values()])
                for key in ('N10', 'H10')} for group in masks} for name in scores}
            aggregate_comparisons = {control: {fusion: {group: {key: describe([r['comparisons'][control][fusion][group][key] for r in seed_results.values()])
                for key in ('delta_N10', 'relative_N10', 'delta_H10', 'hit_net')} for group in masks} for fusion in FUSIONS}
                for control in ('next_only', 'global_local')}
            interactions = {control: {group: describe([r['comparisons'][control]['mean'][group]['delta_N10'] -
                r['comparisons'][control]['current'][group]['delta_N10'] for r in seed_results.values()]) for group in masks}
                for control in ('next_only', 'global_local')}
            gates = {}
            for fusion in ('current', 'mean'):
                all_positive = all(r['comparisons'][c][fusion][group]['relative_N10'] > 0
                    for r in seed_results.values() for c in ('next_only', 'global_local') for group in ('all', 'nonfit'))
                next_contrast = aggregate_comparisons['next_only'][fusion]['all']
                local_contrast = aggregate_comparisons['global_local'][fusion]['all']
                basic = bool(all_positive and next_contrast['relative_N10']['mean'] >= .005 and next_contrast['delta_H10']['mean'] >= 0)
                residual = bool(basic and local_contrast['relative_N10']['mean'] >= .005 and local_contrast['delta_H10']['mean'] >= 0)
                gates[fusion] = dict(basic_stability=basic, residual_margin=residual, all_seed_directions_positive=bool(all_positive))
            results[domain] = dict(seeds=seed_results, aggregate_metrics=aggregate_metrics,
                aggregate_comparisons=aggregate_comparisons, fusion_interactions=interactions, gates=gates)
            write(OUT / f'{domain}.summary.json', results[domain])
        for p, h in hashes.items(): assert sha(ROOT / p) == h, p
        assert steps == 360
        decisions = {fusion: {gate: all(d['gates'][fusion][gate] for d in results.values())
            for gate in ('basic_stability', 'residual_margin')} for fusion in ('current', 'mean')}
        write(OUT / 'summary.json', dict(state='COMPLETED', domains=results, decisions=decisions,
            input_sha256=hashes, training_steps=steps, development_reused=True, test_read=False))
        status('COMPLETED', decisions=decisions)
        signal.alarm(0)
        print(json.dumps(decisions))
    except BaseException as exc:
        status('FAILED', error=repr(exc)); raise


if __name__ == '__main__': main()
