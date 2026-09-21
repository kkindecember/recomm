"""Frozen factorial screen: normalization versus frozen mean-CF in future-label training."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.future_factor_screen import Factor, old_score, z, ranks, metrics, sha
from experiment.phase20.future_label_multiseed import write, fingerprint, describe, compare

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiment/phase9'))
from train_cf0_b2_item_head import CF0B2ItemHead

OUT = ROOT / 'artifacts/phase20/future_fusion_alignment_20260917'
PREVIOUS = ROOT / 'artifacts/phase20/future_label_multiseed_20260917'
COMP = ROOT / 'artifacts/phase20/future_label_components_20260917'
FACTOR = ROOT / 'artifacts/phase20/future_factor_screen_20260917'
PAIR = ROOT / 'artifacts/phase20/pair_uncertainty_screen_20260917'
CONTRACT = ROOT / 'plan/第二十阶段/未来监督最终融合训练_归一化与协同拆分合同_20260917.md'
SEEDS = (20260917, 20260918, 20260919)
LABELS = ('next_only', 'global_local', 'soft_future')
MODES = ('legacy', 'normalized', 'fusion_mean')
DEPLOYMENTS = ('raw', 'current', 'seed2024', 'seed2025', 'mean')
CONTRASTS = ([(m + '.' + y, 'legacy.' + y) for m in MODES[1:] for y in LABELS] +
    [('fusion_mean.' + y, 'normalized.' + y) for y in LABELS] +
    [(m + '.soft_future', m + '.' + c) for m in MODES for c in LABELS[:2]])


def standardize(x):
    centered = x.double() - x.double().mean(-1, keepdim=True)
    return centered / centered.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()


def loss_logits(logits, component, mode):
    if mode == 'legacy': return logits
    result = standardize(logits) - np.log(49)
    if mode == 'fusion_mean': result = result + component
    elif mode != 'normalized': raise ValueError(mode)
    return result


def checks():
    rng = np.random.RandomState(22)
    a = torch.tensor(rng.normal(size=(4, 50)), requires_grad=True)
    np.testing.assert_allclose(standardize(a).detach(), z(a.detach().numpy()), atol=1e-12, rtol=0)
    np.testing.assert_allclose(standardize(3 * a + 7).detach(), standardize(a).detach(), atol=1e-12, rtol=0)
    constant = torch.ones(3, 50, dtype=torch.float64, requires_grad=True)
    standardize(constant).square().sum().backward(); assert torch.isfinite(constant.grad).all()
    component = torch.tensor(rng.normal(size=(4, 50)))
    target = torch.tensor(rng.uniform(size=(4, 50)))
    assert torch.autograd.gradcheck(lambda t: F.binary_cross_entropy_with_logits(loss_logits(t, component, 'fusion_mean'), target), (a,), eps=1e-6, atol=1e-5)
    x = torch.tensor(rng.normal(size=(4, 50, 6)), dtype=torch.float32)
    prior = torch.tensor(z(rng.normal(size=(4, 50))) - np.log(49))
    gradients = []
    for chunk in (4, 2):
        torch.manual_seed(SEEDS[0]); model = Factor('next_only')
        for start in range(0, 4, chunk):
            sl = slice(start, start + chunk); logits, _ = model(x[sl], prior[sl])
            np.testing.assert_allclose(logits.detach(), prior[sl], atol=1e-12, rtol=0)
            loss = F.binary_cross_entropy_with_logits(loss_logits(logits, component[sl], 'fusion_mean'), target[sl], reduction='sum') / 200
            loss.backward()
        gradients.append(np.concatenate([p.grad.numpy().ravel() for p in model.parameters()]))
    np.testing.assert_allclose(gradients[0], gradients[1], atol=1e-7, rtol=1e-5)
    return dict(numpy_parity=True, affine_invariance=True, constant_score_finite_gradient=True,
        finite_difference_gradient_check=True, initial_B1_parity=True, full_vs_chunk_gradient_parity=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check-only', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    health = checks()
    if args.check_only: print(json.dumps(health)); return
    OUT.mkdir(exist_ok=False); started = time.monotonic(); steps = 0; hashes = {}; results = {}

    def use(p):
        hashes[str(p.relative_to(ROOT))] = sha(p); return p

    def read(folder, name):
        with np.load(use(folder / name)) as f: return {k: f[k] for k in f.files}

    def status(state, **kw):
        write(OUT / 'status.json', dict(state=state, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            elapsed_seconds=time.monotonic() - started, pid=os.getpid(), training_steps=steps,
            planned_training_steps=1080, gpu_used=False, test_read=False, automatic_next_experiment=False, **kw))

    def timeout(sig, frame): raise TimeoutError('30 minute hard CPU budget')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(1800)

    def cf_score(path, history, ids, domain, label, model=None):
        if model is None:
            checkpoint = torch.load(use(path), map_location='cpu'); c = checkpoint['model_config']
            model = CF0B2ItemHead(c['num_items'], c['max_history'], c['d_model'], c['num_layers'], c['num_heads'], c['dropout'], c['temperature_initial'])
            model.load_state_dict(checkpoint['model_state_dict'], strict=True); model.eval()
            for p in model.parameters(): p.requires_grad_(False)
        out = np.empty(ids.shape, np.float64)
        with torch.no_grad():
            weights = F.normalize(model.item_embedding.weight, dim=-1); scale = model.logit_scale.exp().clamp(max=100)
            for offset in range(0, len(ids), 256):
                sl = slice(offset, offset + 256); h = torch.as_tensor(history[sl])
                user = model.encode(h, h.ne(0))
                out[sl] = (scale * (user[:, None] * weights[torch.as_tensor(ids[sl])]).sum(-1)).numpy()
                status('RUNNING', stage='frozen_CF_scoring', dataset=domain, cf=label, completed=min(offset + 256, len(ids)), total=len(ids))
        return out, model

    try:
        status('RUNNING', stage='verify_inputs')
        for p in (Path(__file__), CONTRACT, ROOT / 'experiment/phase20/future_factor_screen.py',
                  ROOT / 'experiment/phase20/future_label_multiseed.py', ROOT / 'experiment/phase9/train_cf0_b2_item_head.py',
                  ROOT / 'experiment/phase9/cf0_diagnostic_metrics.py'): use(p)
        prev = json.loads(use(PREVIOUS / 'summary.json').read_text())
        verified = json.loads(use(PREVIOUS / 'verification.json').read_text()); assert verified['state'] == 'VERIFIED'
        for group in (prev['input_sha256'], verified['output_sha256']):
            for p, h in group.items(): use(ROOT / p); assert hashes[p] == h, p
        config = json.loads(use(ROOT / 'experiment/phase20/gacr_fusion_train_v1.json').read_text())
        write(OUT / 'protocol.json', dict(contract=str(CONTRACT.relative_to(ROOT)), source_sha256=hashes.copy(), checks=health,
            modes=MODES, labels=LABELS, seeds=SEEDS, deployments=DEPLOYMENTS, contrasts=CONTRASTS,
            primary='fusion_mean.soft_future.mean', new_training_steps=1080, steps_per_arm=30, threads=4,
            max_seconds=1800, reused='legacy', reliability='frozen_B1_top10', development_reused=True))
        for domain in ('Toys', 'Beauty'):
            cfg = config['domains'][domain]; old = ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}'
            status('RUNNING', dataset=domain, stage='load_train_prefixes')
            a = read(old, 'training_cache.npz'); stats = read(COMP, f'{domain}.label_statistics.npz'); ys = read(COMP, f'{domain}.training_labels.npz')
            ix = stats['fit_indices']; positions = a['positions'][ix]; head = a['head'][ix]
            for key, values in (('positions', positions), ('head', head)): np.testing.assert_array_equal(values, stats[key])
            assert a['fit'][ix].all() and (positions >= 0).all()
            selected = set(ix.tolist()); metadata = []
            with use(old / 'train_candidates.jsonl').open() as f:
                for j, line in enumerate(f):
                    if j in selected:
                        row = json.loads(line); k = len(metadata)
                        assert row['partition'] == 'fit' and row['split'] == 'train' and row['user_id'] == stats['users'][k]
                        np.testing.assert_array_equal(row['union_ids'][:50], a['candidate_ids'][j])
                        assert row['target_position'] == positions[k]
                        metadata.append({key: row[key] for key in ('user_id', 'train_index', 'history', 'target')})
            assert len(metadata) == len(ix)
            train_by_index = {r['train_index']: r for r in metadata}; assert len(train_by_index) == len(ix)
            with use((ROOT / cfg['validation']).with_name('train.jsonl')).open() as f:
                for j, line in enumerate(f):
                    if j in train_by_index:
                        row = json.loads(line); record = train_by_index[j]
                        for key in ('user_id', 'history', 'target'): assert row[key] == record[key]
            histories = np.zeros((len(ix), 20), np.int64)
            for j, row in enumerate(metadata):
                assert 0 < len(row['history']) <= 20
                histories[j, :len(row['history'])] = row['history']
            ids = a['candidate_ids'][ix]
            x = torch.as_tensor(a['features'][ix])
            g = old_score(x, torch.as_tensor(a['base'][ix]), torch.load(use(old / 'B1.pt'), map_location='cpu'))
            prior = torch.as_tensor(z(g) - np.log(49))
            cf = [a['cf'][ix]]; cf_parity = {}
            reproduced, model = cf_score(ROOT / cfg['item_head'], histories[:128], ids[:128], domain, 'original_parity')
            cf_parity['original_training'] = float(np.abs(reproduced - cf[0][:128]).max()); assert cf_parity['original_training'] <= 1e-4
            del model
            cf_validation = read(PAIR, f'{domain}.cf_scores.npz')
            validation_rows = {}
            wanted = set(cf_validation['users'][:128])
            with use(ROOT / cfg['validation']).open() as f:
                for line in f:
                    row = json.loads(line)
                    if row['user_id'] in wanted: validation_rows[row['user_id']] = row
            vh = np.zeros((128, 20), np.int64)
            for j, user in enumerate(cf_validation['users'][:128]):
                h = validation_rows[user]['history']; vh[j, :len(h)] = h
            for cf_seed in (2024, 2025):
                path = ROOT / f'artifacts/phase9/p9s_multiseed/{domain}/seed{cf_seed}/item_head/best_item_head.pt'
                values, model = cf_score(path, histories, ids, domain, str(cf_seed)); cf.append(values)
                sampled, _ = cf_score(path, vh, cf_validation['candidate_ids'][:128], domain, str(cf_seed) + '_validation_parity', model=model)
                cf_parity[str(cf_seed) + '_validation'] = float(np.abs(sampled - cf_validation['scores'][cf_seed - 2023, :128]).max())
                assert cf_parity[str(cf_seed) + '_validation'] <= 1e-4
                del model
            cf = np.stack(cf); freq = a['frequencies'][ix]
            order = np.argsort(-g, axis=1, kind='stable')
            reliability = 1 - (np.take_along_axis(freq, order[:, :10], 1) <= cfg['q1']).mean(1)
            adjusted = z(z(cf) - .5 * z(np.log1p(freq))[None])
            cf_mean = reliability[:, None] * adjusted.mean(0)
            np.savez_compressed(OUT / f'{domain}.training_context.npz', users=stats['users'], fit_indices=ix, history=histories,
                train_indices=np.asarray([r['train_index'] for r in metadata]), targets=np.asarray([r['target'] for r in metadata]),
                candidate_ids=ids, cf_scores=cf, cf_mean=cf_mean, reliability=reliability, prior=prior.numpy(), frequencies=freq)
            write(OUT / f'{domain}.context_checks.json', dict(prefixes=len(ix), users=len(set(stats['users'])), cf_parity=cf_parity,
                original_cf_cache_reused=True, train_only_histories=True, q1=cfg['q1']))
            component = torch.as_tensor(cf_mean)
            weights = torch.as_tensor(np.where(head, 1 / (2 * head.sum()), 1 / (2 * (~head).sum())))
            targets = dict(next_only=(np.arange(50)[None] == positions[:, None]).astype(float), global_local=ys['global_local'], soft_future=stats['actual'])
            v = read(FACTOR, f'{domain}.audit_validation_cache.npz'); paired = read(PAIR, f'{domain}.predictions.npz')
            for key in ('users', 'candidate_ids', 'positions', 'tail_mask'): np.testing.assert_array_equal(v[key], paired[key])
            parts = {f: paired['scores'][list(paired['arm_names']).index(f)] - (v['prior'] + np.log(49)) for f in DEPLOYMENTS[1:]}
            for f in ('current', 'mean'): np.testing.assert_allclose(parts[f], v['cf_' + f], atol=1e-11, rtol=0)
            parts.update(current=v['cf_current'], mean=v['cf_mean'])
            in_fit = np.isin(v['users'], np.unique(stats['users']))
            masks = dict(all=np.ones(len(in_fit), bool), tail=v['tail_mask'], non_tail=~v['tail_mask'], fit=in_fit, nonfit=~in_fit)
            seeds = {}
            for seed in SEEDS:
                previous = read(PREVIOUS, f'{domain}.{seed}.predictions.npz')
                for key in ('users', 'candidate_ids', 'positions', 'tail_mask'): np.testing.assert_array_equal(v[key], previous[key])
                np.testing.assert_array_equal(in_fit, previous['in_fit'])
                raw_scores = {'legacy.' + y: previous['scores'][list(previous['arm_names']).index(y + '.raw')] for y in LABELS}
                curves = {}; initial = {}; train_initial = {}
                for mode in MODES[1:]:
                    for label in LABELS:
                        name = mode + '.' + label; torch.manual_seed(seed); model = Factor('next_only')
                        initial[name] = fingerprint(model)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
                        y = torch.as_tensor(targets[label]); curve = []
                        for step in range(1, 31):
                            optimizer.zero_grad(set_to_none=True); total = 0.
                            for offset in range(0, len(x), 256):
                                sl = slice(offset, offset + 256); logits, _ = model(x[sl], prior[sl])
                                output = loss_logits(logits, component[sl], mode)
                                loss = (F.binary_cross_entropy_with_logits(output, y[sl], reduction='none').mean(1) * weights[sl]).sum()
                                assert torch.isfinite(loss); loss.backward(); total += float(loss.detach())
                            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                            optimizer.step(); steps += 1
                            curve.append(dict(step=step, loss=total, grad_norm=float(grad)))
                            status('RUNNING', dataset=domain, stage='training', seed=seed, arm=name, step=step)
                        curves[name] = curve; train_initial[name] = curve[0]['loss']
                        model.eval(); torch.save(model.state_dict(), OUT / f'{domain}.{seed}.{name}.pt')
                        predictions = []
                        with torch.no_grad():
                            for offset in range(0, len(v['users']), 256):
                                sl = slice(offset, offset + 256)
                                logits, _ = model(torch.as_tensor(v['features'][sl]), torch.as_tensor(v['prior'][sl]))
                                predictions.append(logits.numpy())
                        raw_scores[name] = z(np.concatenate(predictions))
                assert len(set(initial.values())) == 1
                rr = {}
                for name, raw in raw_scores.items():
                    for f in DEPLOYMENTS: rr[name + '.' + f] = ranks(raw if f == 'raw' else raw + parts[f], v['positions'])
                for label in LABELS:
                    for f in ('raw', 'current', 'mean'):
                        np.testing.assert_array_equal(rr['legacy.' + label + '.' + f], previous['ranks'][list(previous['arm_names']).index(label + '.' + f)])
                met = {name: {group: metrics(rank[mask]) for group, mask in masks.items()} for name, rank in rr.items()}
                comps = {a + '__vs__' + b: {f: {group: compare(rr[a + '.' + f][mask], rr[b + '.' + f][mask]) for group, mask in masks.items()}
                    for f in DEPLOYMENTS} for a, b in CONTRASTS}
                seeds[str(seed)] = dict(metrics=met, comparisons=comps)
                np.savez_compressed(OUT / f'{domain}.{seed}.predictions.npz', users=v['users'], candidate_ids=v['candidate_ids'], positions=v['positions'],
                    tail_mask=v['tail_mask'], in_fit=in_fit, raw_names=np.asarray(list(raw_scores)), raw_scores=np.stack(list(raw_scores.values())),
                    rank_names=np.asarray(list(rr)), ranks=np.stack(list(rr.values())))
                write(OUT / f'{domain}.{seed}.training.json', dict(curves=curves, initial_sha256=initial, initial_losses=train_initial,
                    prefixes=len(x), users=len(set(stats['users']))))
            aggregate = {c: {f: {group: {key: describe([s['comparisons'][c][f][group][key] for s in seeds.values()])
                for key in ('delta_N10', 'relative_N10', 'delta_H10', 'hit_net')} for group in masks} for f in DEPLOYMENTS} for c in comps}
            metric_average = {name: {group: {key: describe([s['metrics'][name][group][key] for s in seeds.values()]) for key in ('N10', 'H10')}
                for group in masks} for name in rr}
            required = ['legacy.soft_future', 'normalized.soft_future', 'fusion_mean.next_only', 'fusion_mean.global_local']
            margin = all(aggregate['fusion_mean.soft_future__vs__' + b]['mean']['all']['relative_N10']['mean'] >= .005 for b in required)
            directions = all(s['comparisons']['fusion_mean.soft_future__vs__legacy.soft_future']['mean'][g]['relative_N10'] > 0 for s in seeds.values() for g in ('all', 'nonfit'))
            hits = all(aggregate['fusion_mean.soft_future__vs__' + b]['mean'][g]['hit_net']['mean'] >= 0 for b in ('legacy.soft_future', 'fusion_mean.next_only') for g in ('all', 'tail'))
            gate = dict(margins=bool(margin), directions=bool(directions), hit_protection=bool(hits), passed=bool(margin and directions and hits))
            results[domain] = dict(seeds=seeds, aggregate_comparisons=aggregate, aggregate_metrics=metric_average, gate=gate)
            write(OUT / f'{domain}.summary.json', results[domain])
        assert steps == 1080
        for p, h in hashes.items(): assert sha(ROOT / p) == h, p
        decision = 'SUPPORTED_FOR_FOLLOWUP' if all(v['gate']['passed'] for v in results.values()) else 'FUSION_ALIGNMENT_ADVANTAGE_NOT_ESTABLISHED'
        write(OUT / 'summary.json', dict(state='COMPLETED', domains=results, decision=decision, checks=health,
            input_sha256=hashes, training_steps=steps, test_read=False, development_reused=True))
        status('COMPLETED', decision=decision); signal.alarm(0)
        print(json.dumps({d: v['gate'] for d, v in results.items()}))
    except BaseException as exc:
        status('FAILED', error=repr(exc)); raise


if __name__ == '__main__': main()
