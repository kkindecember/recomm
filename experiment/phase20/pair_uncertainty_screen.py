"""Frozen CF ensemble and pair-margin uncertainty screen; CPU only, no fitting."""
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiment/phase9'))
from train_cf0_b2_item_head import CF0B2ItemHead

OUT = ROOT / 'artifacts/phase20/pair_uncertainty_screen_20260917'
CONTRACT = ROOT / 'plan/第二十阶段/商品对协同不确定性_固定缓存筛查合同_20260917.md'
ARMS = ('current', 'seed2024', 'seed2025', 'mean', 'point_lcb', 'pair_diag', 'pair_cov', 'matched_scalar')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def z(a):
    return (a - a.mean(-1, keepdims=True)) / np.maximum(a.std(-1, keepdims=True), 1e-6)


def ranks(scores, positions):
    inv = np.argsort(np.argsort(-scores, axis=1, kind='stable'), axis=1)
    return np.where(positions >= 0, inv[np.arange(len(inv)), np.maximum(positions, 0)] + 1, 51)


def metrics(rr):
    out = {'users': len(rr)}
    for k in (5, 10, 20, 50):
        out[f'H{k}'] = float((rr <= k).mean())
        out[f'N{k}'] = float(np.where(rr <= k, 1 / np.log2(rr + 1), 0).mean())
    return out


def pair_scores(g, c, r):
    """No targets. g: BxK; c: MxBxK; r: B. Return fixed scoring arms."""
    mu = c.mean(0)
    var = c.var(0, ddof=1)
    delta = r[:, None, None] * (mu[:, :, None] - mu[:, None, :])
    differences = c[:, :, :, None] - c[:, :, None, :]
    joint_std = r[:, None, None] * differences.std(0, ddof=1)
    diag_std = r[:, None, None] * np.sqrt(var[:, :, None] + var[:, None, :])
    def shrink(sd):
        return np.sign(delta) * np.maximum(np.abs(delta) - sd, 0)
    cov_part, diag_part = shrink(joint_std), shrink(diag_std)
    base_pair = g[:, :, None] - g[:, None, :]
    def wins(part):
        return (1 / (1 + np.exp(-np.clip(base_pair + part, -60, 60)))).sum(-1)
    total = np.abs(delta).sum((1, 2))
    retained = np.divide(np.abs(cov_part).sum((1, 2)), total, out=np.zeros_like(total), where=total > 1e-12)
    score = dict(mean=g + r[:, None] * mu,
        point_lcb=g + r[:, None] * (mu - np.sqrt(var)),
        pair_diag=wins(diag_part), pair_cov=wins(cov_part),
        matched_scalar=g + (r * retained)[:, None] * mu)
    assert np.allclose(cov_part, -cov_part.swapaxes(-1, -2), atol=1e-12, rtol=0)
    return score, dict(retained_cf_fraction=retained,
        joint_vs_diag_ratio=(joint_std.sum((1, 2)) / np.maximum(diag_std.sum((1, 2)), 1e-12)))


def mechanism_checks():
    rng = np.random.RandomState(17)
    g = rng.normal(size=(4, 6)); c = rng.normal(size=(3, 4, 6)); r = np.array([0., .4, .8, 1.])
    a, _ = pair_scores(g, c, r)
    np.testing.assert_array_equal(np.argsort(a['pair_cov'][0]), np.argsort(g[0]))
    equal = np.repeat(c[:1], 3, axis=0)
    b, _ = pair_scores(g, equal, r)
    np.testing.assert_array_equal(np.argsort(b['pair_cov'], axis=1), np.argsort(b['mean'], axis=1))
    shifted = c + np.array([-3., 0., 3.])[:, None, None]
    shifted_scores, _ = pair_scores(g, shifted, r)
    np.testing.assert_allclose(a['pair_cov'], shifted_scores['pair_cov'], atol=1e-12, rtol=0)
    perm = np.array([4, 1, 3, 5, 0, 2])
    pp, _ = pair_scores(g[:, perm], c[:, :, perm], r)
    np.testing.assert_allclose(a['pair_cov'][:, perm], pp['pair_cov'], atol=1e-12, rtol=0)
    # Shared per-model offsets inflate marginal variance but cancel in pair margins.
    shared = equal + np.array([-10., 0., 10.])[:, None, None]
    ss, _ = pair_scores(g, shared, r)
    np.testing.assert_allclose(b['pair_cov'], ss['pair_cov'], atol=1e-12, rtol=0)
    return 5


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    hashes = {}
    def use(p):
        p = Path(p); hashes[str(p.relative_to(ROOT))] = sha(p); return p
    def read(p):
        return json.loads(use(p).read_text())
    def status(state, **extra):
        record = dict(state=state, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(),
            elapsed_seconds=round(time.monotonic() - start, 2), gpu_used=False, training_steps=0,
            test_read=False, automatic_next_experiment=False, **extra)
        tmp = OUT / 'status.tmp'; tmp.write_text(json.dumps(record, indent=2) + '\n'); tmp.replace(OUT / 'status.json')
        if state == 'RUNNING':
            assert time.monotonic() - start < 900, 'CPU budget exceeded'
    try:
        use(Path(__file__)); use(CONTRACT); use(ROOT / 'experiment/phase9/train_cf0_b2_item_head.py')
        config = read(ROOT / 'experiment/phase20/gacr_fusion_train_v1.json')
        (OUT / 'protocol.json').write_text(json.dumps(dict(contract=str(CONTRACT.relative_to(ROOT)),
            source_sha256=hashes, arms=ARMS, primary='pair_cov', seeds=[2023, 2024, 2025],
            shrink_std_multiplier=1., sigmoid_temperature=1., sample_std_ddof=1,
            max_seconds=900, threads=4, new_training_steps=0, test_read=False), indent=2) + '\n')
        checks = mechanism_checks()
        status('RUNNING', stage='load')
        results = {}
        for dataset, domain in config['domains'].items():
            old = ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{dataset}'
            validation = list(map(json.loads, use(ROOT / domain['validation']).open()))
            by_user = {row['user_id']: row for row in validation}
            assert len(by_user) == len(validation)
            with np.load(use(old / 'B1.validation_scores.npz')) as f:
                users, ids, g = [f[k].copy() for k in ('users', 'candidate_ids', 'residual_scores')]
            with np.load(use(ROOT / domain['c1_cf'])) as f:
                np.testing.assert_array_equal(users, f['users']); np.testing.assert_array_equal(ids, f['candidate_ids'])
                cf0 = f['scores'].astype(np.float64)
            n = len(users)
            histories = [by_user[u]['history'] for u in users]
            targets = np.asarray([by_user[u]['target'] for u in users])
            cat = read((ROOT / domain['validation']).parent / 'catalog.json')
            assert cat['items'][0] == '' and cat['items'][1:] == sorted(cat['items'][1:])
            counts, initial = Counter(), {}
            for line in use((ROOT / domain['validation']).parent / 'train.jsonl').open():
                row = json.loads(line); counts[row['target']] += 1
                if len(row['history']) == 1: initial.setdefault(row['user_id'], row['history'][0])
            counts.update(initial.values())
            frequencies = np.asarray([counts[i] for i in range(len(cat['items']))])
            cf_models = [cf0]
            timings = {}
            def score_checkpoint(p, subset=None):
                checkpoint = torch.load(use(p), map_location='cpu'); c = checkpoint['model_config']
                assert c['num_items'] + 1 == len(cat['items'])
                model = CF0B2ItemHead(c['num_items'], c['max_history'], c['d_model'], c['num_layers'],
                    c['num_heads'], c['dropout'], c['temperature_initial'])
                model.load_state_dict(checkpoint['model_state_dict'], strict=True); model.eval()
                for parameter in model.parameters(): parameter.requires_grad_(False)
                weights = torch.nn.functional.normalize(model.item_embedding.weight, dim=-1)
                scale = model.logit_scale.exp().clamp(max=100)
                indices = np.arange(n) if subset is None else subset
                output = np.empty((len(indices), 50), np.float64)
                before = time.monotonic()
                with torch.no_grad():
                    for offset in range(0, len(indices), 256):
                        chosen = indices[offset:offset + 256]
                        seq = torch.zeros((len(chosen), 20), dtype=torch.long)
                        for j, i in enumerate(chosen): seq[j, :len(histories[i])] = torch.tensor(histories[i])
                        vectors = model.encode(seq, seq.ne(0))
                        item_vectors = weights[torch.as_tensor(ids[chosen])]
                        scores = scale * (vectors[:, None, :] * item_vectors).sum(-1)
                        output[offset:offset + len(chosen)] = scores.numpy()
                        status('RUNNING', stage='frozen_CF_scoring', dataset=dataset,
                            checkpoint=str(p.relative_to(ROOT)), completed=offset + len(chosen), total=len(indices))
                return output, time.monotonic() - before
            cf_source = ROOT / domain['item_head']
            subset = np.arange(min(512, n))
            parity, sec = score_checkpoint(cf_source, subset)
            max_error = float(np.abs(parity - cf0[subset]).max())
            assert max_error <= 1e-4, ('CF cache CPU parity', dataset, max_error)
            for seed in (2024, 2025):
                p = ROOT / f'artifacts/phase9/p9s_multiseed/{dataset}/seed{seed}/item_head/best_item_head.pt'
                scores, sec = score_checkpoint(p)
                cf_models.append(scores); timings[str(seed)] = sec
            cf = np.stack(cf_models)
            np.savez_compressed(OUT / f'{dataset}.cf_scores.npz', users=users, candidate_ids=ids, scores=cf)
            canonical = np.argsort(-g, axis=1, kind='stable')
            g = np.take_along_axis(g, canonical, 1)
            ids = np.take_along_axis(ids, canonical, 1)
            cf = np.take_along_axis(cf, canonical[None], 2)
            f = frequencies[ids]; q = domain['q1']; tail = frequencies[targets] <= q
            hit = ids == targets[:, None]; positions = np.where(hit.any(1), hit.argmax(1), -1)
            gz, pz = z(g), z(np.log1p(f)); adjusted = z(z(cf) - .5 * pz[None])
            reliability = 1 - (f[:, :10] <= q).mean(1)
            all_scores = {a: np.empty_like(g) for a in ARMS}
            for idx, arm in enumerate(('current', 'seed2024', 'seed2025')):
                all_scores[arm] = gz + reliability[:, None] * adjusted[idx]
            diag = {'retained_cf_fraction': [], 'joint_vs_diag_ratio': []}
            for offset in range(0, n, 256):
                batch, dd = pair_scores(gz[offset:offset+256], adjusted[:, offset:offset+256], reliability[offset:offset+256])
                for a, value in batch.items(): all_scores[a][offset:offset+len(value)] = value
                for key, value in dd.items(): diag[key].extend(value.tolist())
            rr = {a: ranks(s, positions) for a, s in all_scores.items()}
            with use(old / 'per_user.tsv').open() as handle:
                reference = {r['user_id']: int(r['B1']) for r in csv.DictReader(handle, delimiter='\t')}
            np.testing.assert_array_equal(rr['current'], [reference[u] for u in users])
            groups, comparison = {}, {}
            for name, mask in [('all', np.ones(n, bool)), ('tail', tail), ('non_tail', ~tail)]:
                groups[name] = {a: metrics(r[mask]) for a, r in rr.items()}
                comparison[name] = {}
                for reference_arm in ('current', 'mean', 'pair_diag', 'matched_scalar'):
                    a, b = rr['pair_cov'][mask], rr[reference_arm][mask]
                    comparison[name][reference_arm] = dict(delta_N10=groups[name]['pair_cov']['N10']-groups[name][reference_arm]['N10'],
                        relative_N10=groups[name]['pair_cov']['N10']/groups[name][reference_arm]['N10']-1,
                        hit10_gained=int(((a<=10)&(b>10)).sum()), hit10_lost=int(((a>10)&(b<=10)).sum()))
            np.savez_compressed(OUT / f'{dataset}.predictions.npz', users=users, candidate_ids=ids,
                tail_mask=tail, positions=positions, arm_names=np.asarray(ARMS),
                scores=np.stack([all_scores[a] for a in ARMS]), ranks=np.stack([rr[a] for a in ARMS]))
            results[dataset] = dict(groups=groups, pair_cov_comparisons=comparison,
                diagnostics={key: dict(mean=float(np.mean(value)), quantiles=np.quantile(value,[0,.1,.5,.9,1]).tolist()) for key,value in diag.items()},
                frozen_CF_seconds=timings, seed2023_CPU_cache_max_error=max_error, current_rank_parity=True)
            (OUT / f'{dataset}.summary.json').write_text(json.dumps(results[dataset], indent=2)+'\n')
        supported = all(all(v['delta_N10'] > 0 for v in d['pair_cov_comparisons']['all'].values()) for d in results.values())
        decision = 'PAIR_MECHANISM_SUPPORTED_FOR_FOLLOWUP' if supported else 'PAIR_MECHANISM_ADVANTAGE_NOT_ESTABLISHED'
        for p, expected in hashes.items(): assert sha(ROOT / p) == expected, p
        summary = dict(state='COMPLETED', decision=decision, domains=results, mechanism_checks=checks,
            input_sha256=hashes, gpu_used=False, training_steps=0, test_read=False,
            limitations='Three shared-data CF seeds; reused development; no calibrated probability/risk guarantee; '
                        'B1 fixed seed; ensemble cost must be compared with equal-cost mean baseline.')
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        status('COMPLETED', stage='finished', decision=decision)
        print(json.dumps({k:v['pair_cov_comparisons']['all'] for k,v in results.items()},indent=2))
    except Exception as error:
        status('FAILED', stage='stopped', error=repr(error)); raise


if __name__ == '__main__':
    main()
