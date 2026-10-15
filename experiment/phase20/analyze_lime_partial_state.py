"""Fixed-budget prefix-state fits; all predictions precede new evaluation."""
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from experiment.phase20.lime_partial_state import ROOT, SOURCE, EARLY, OUT
from experiment.phase20.lime_decode_screen import read, write, sha, stamp, arrays
from experiment.phase20.lime_clean_benefit_core import fold_ids, ranking
from experiment.phase20.lime_early_correction_core import GROUPS, fit, correction, add_prior
from experiment.phase20.catalog_scope_core import metrics, contrast, utilities

VARIANTS = ('prefix2', 'prefix3', 'state2', 'state3')
NEW = tuple(f'{v}_seed{s}' for v in VARIANTS for s in range(3))
OLD = tuple(f'{v}_seed{s}' for v in ('full', 'early') for s in range(3))
ALL = ('GRAM', 'fixed_late') + OLD + NEW
LATE_COLUMNS = (0, 1, 6, 7)


def subset(d, ix):
    return {k: v[ix] for k, v in d.items()}


def state(**kw):
    write(OUT / 'status.json', dict(updated_at=stamp(), pid=os.getpid(), **kw))


def fit_domain(domain):
    folder = OUT / domain; (folder / 'heads').mkdir()
    data = arrays(SOURCE / domain / 'data.npz'); cached = arrays(folder / 'features.npz')
    for key in ('users', 'ids', 'sample_indices'):
        np.testing.assert_array_equal(cached[key], data[key])
    tr = np.flatnonzero(data['roles'] == 'ranker_fit'); dv = np.flatnonzero(data['roles'] == 'development')
    train, dev = subset(data, tr), subset(data, dv)
    assert len(dv) == 512 and not set(train['users']) & set(dev['users'])
    folds = fold_ids(train['users'], domain)
    predictions = {}; corrections = {}; training = {}
    for label, src in [('oof', train), ('development', dev)]:
        old = arrays(EARLY / domain / f'{label}_predictions.npz')
        np.testing.assert_array_equal(old['users'], src['users'])
        predictions[label] = {name: old[name] for name in ('GRAM', 'fixed_late') + OLD}
        predictions[label].update({name: np.empty(src['mixed'].shape, dtype='float32') for name in NEW})
        corrections[label] = {name: np.empty(src['mixed'].shape, dtype='float32') for name in NEW}
    for variant in VARIANTS:
        depth = int(variant[-1]); group = 'full' if variant.startswith('prefix') else 'late'
        columns = GROUPS[group]
        x = data['x'].copy(); x[..., list(LATE_COLUMNS)] = cached[f'prefix{depth}']
        train_v = dict(train, x=x[tr]); dev_v = dict(dev, x=x[dv])
        for f in list(range(5)) + ['final']:
            tag = f'fold{f}' if f != 'final' else 'final'; label = 'oof' if f != 'final' else 'development'
            src = train_v if f != 'final' else dev_v
            fit_ix = np.flatnonzero(folds != f) if f != 'final' else np.arange(len(tr))
            ix = np.flatnonzero(folds == f) if f != 'final' else np.arange(len(dv))
            assert not set(train['users'][fit_ix]) & set(src['users'][ix])
            state(state='RUNNING', stage='fit', domain=domain, variant=variant, fold=tag)
            for seed in range(3):
                name = f'{variant}_seed{seed}'; c = fit(train_v, fit_ix, group, seed)
                c.update(variant=variant, depth=depth, fit_users=train['users'][fit_ix], used_users=train['users'][c['used']])
                path = folder / 'heads' / f'{tag}.{name}.pt'; torch.save(c, path)
                c = torch.load(path, map_location='cpu')
                delta = correction(c, src['x'][ix][..., list(columns)])
                predictions[label][name][ix] = add_prior(src['mixed'][ix], delta)
                corrections[label][name][ix] = delta
                training[f'{tag}.{name}'] = dict(fit_users=len(fit_ix), covered_users=len(c['used']),
                    optimizer_steps=30, curve=c['curve'], sha256=sha(path))
        print(json.dumps(dict(domain=domain, variant=variant, completed_heads=len(training))), flush=True)
    for label, src in [('oof', train), ('development', dev)]:
        common = dict(users=src['users'])
        if label == 'oof': common['folds'] = folds
        np.savez_compressed(folder / f'{label}_predictions.npz', **common, **predictions[label])
        np.savez_compressed(folder / f'{label}_corrections.npz', **common, **corrections[label])
    write(folder / 'training.json', training)


def evaluate_domain(domain):
    folder = OUT / domain; data = arrays(SOURCE / domain / 'data.npz')
    result = {}; saved = {}
    for label, role in [('oof', 'ranker_fit'), ('development', 'development')]:
        src = subset(data, np.flatnonzero(data['roles'] == role))
        pred = arrays(folder / f'{label}_predictions.npz')
        rr = {name: ranking(src, pred[name]) for name in ALL}
        result[label] = dict(users=len(src['users']), covered=int((rr['GRAM'] < 2**30).sum()),
            metrics={name: metrics(r) for name, r in rr.items()}, contrasts={}, movement={})
        for name in NEW:
            seed = name[-1]
            refs = ('GRAM', 'fixed_late', f'early_seed{seed}', f'full_seed{seed}')
            result[label]['contrasts'][name] = {ref: contrast(rr[name], rr[ref], repeats=1000, seed=20260922) for ref in refs}
            delta = utilities(rr[name]) - utilities(rr['fixed_late'])
            result[label]['movement'][name] = dict(improved=int((delta > 0).sum()), hurt=int((delta < 0).sum()), unchanged=int((delta == 0).sum()))
        if label == 'oof':
            folds = pred['folds']
            result['folds'] = {str(f): {name: metrics(r[folds == f]) for name, r in rr.items()} for f in range(5)}
        saved.update({f'{label}_{name}': r for name, r in rr.items()}); saved[f'{label}_users'] = src['users']
    checks = {}
    for name in NEW:
        dm, om = result['development']['metrics'], result['oof']['metrics']
        wins = sum(all(result['folds'][str(f)][name]['N10'] > result['folds'][str(f)][ref]['N10'] for ref in ('GRAM', 'fixed_late')) for f in range(5))
        dp = all(dm[name]['N10'] >= 1.01*dm[ref]['N10'] and dm[name]['H10'] >= dm[ref]['H10'] for ref in ('GRAM', 'fixed_late'))
        op = wins >= 4 and all(om[name]['N10'] > om[ref]['N10'] for ref in ('GRAM', 'fixed_late'))
        checks[name] = dict(development_pass=dp, oof_pass=op, positive_folds_vs_both=wins, passed=dp and op)
    result.update(domain=domain, checks=checks, test_read=False, late_prior_retained=True,
                  development_labels_used_for_fit=False, native_trajectory_only=True)
    np.savez_compressed(folder / 'ranks.npz', **saved); write(folder / 'analysis.json', result)
    return result


def main():
    assert read(OUT / 'status.json')['state'] == 'CACHE_COMPLETED'
    assert not (OUT / 'analysis_manifest.json').exists()
    files = [Path(__file__), ROOT / 'experiment/phase20/review_lime_partial_state.py']
    for domain in ('Toys', 'Beauty'):
        assert read(OUT / domain / 'feature_verification.json')['state'] == 'VERIFIED'
        for name, digest in read(OUT / domain / 'cache_manifest.json')['sha256'].items():
            p = OUT / domain / name; assert sha(p) == digest; files.append(p)
        files.append(OUT / domain / 'feature_verification.json')
    write(OUT / 'analysis_manifest.json', dict(created_at=stamp(), sha256={str(p.relative_to(ROOT)): sha(p) for p in files}))
    torch.set_num_threads(2); torch.set_num_interop_threads(1); started = time.monotonic()
    for domain in ('Toys', 'Beauty'): fit_domain(domain)
    # No new outcome evaluation before both domains' complete prediction files.
    state(state='RUNNING', stage='evaluate')
    result = {d: evaluate_domain(d) for d in ('Toys', 'Beauty')}
    decision = {name: all(r['checks'][name]['passed'] for r in result.values()) for name in NEW}
    for depth in (2, 3):
        decision[f'prefix{depth}_all_seeds'] = all(decision[f'prefix{depth}_seed{s}'] for s in range(3))
    write(OUT / 'summary.json', dict(state='ANALYZED', domains=result, decision=decision,
        seconds=time.monotonic()-started, small_head_fits=144, optimizer_steps=4320,
        pending_verification=True, test_read=False, upstream_training_steps=0))
    state(state='ANALYZED', pending_verification=True, decision=decision)
    print(json.dumps(dict(state='ANALYZED', seconds=time.monotonic()-started, decision=decision)), flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException as e:
        (OUT / 'analysis_failure.txt').write_text(traceback.format_exc()); state(state='FAILED', error=repr(e)); raise
