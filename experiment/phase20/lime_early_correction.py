"""Frozen three-group clean feature ablation, using only existing CPU caches."""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiment.phase20.lime_clean_benefit_core import fold_ids, ranking
from experiment.phase20.catalog_scope_core import metrics, contrast, utilities
from experiment.phase20.lime_early_correction_core import GROUPS, fit, correction, add_prior

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
OUT = ROOT / 'artifacts/phase20/lime_early_correction_20260922_v1'
PROTOCOL = ROOT / 'plan/第二十阶段/LIME提前纠错信息消融_20260922.md'
NAMES = tuple(f'{g}_seed{s}' for g in GROUPS for s in range(3))


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def arrays(p):
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def subset(d, ix):
    return {k: v[ix] for k, v in d.items()}


def status(**kwargs):
    write(OUT / 'status.json', dict(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                  pid=os.getpid(), **kwargs))


def check_inputs():
    assert read(SOURCE / 'status.json')['verified']
    for name, digest in read(SOURCE / 'completion_manifest.json')['sha256'].items():
        assert sha(ROOT / name) == digest, name
    paths = [PROTOCOL, SOURCE / 'completion_manifest.json', SOURCE / 'status.json']
    paths += list((ROOT / 'experiment/phase20').glob('*lime_early_correction*.py'))
    paths += [ROOT / f'experiment/phase20/{n}.py' for n in
              ('lime_clean_benefit_core', 'source_competition_core', 'catalog_scope_core',
               'review_lime_clean_benefit')]
    paths.append(ROOT / 'experiment/phase20/run_lime_early_correction.sh')
    for domain in ('Toys', 'Beauty'):
        folder = SOURCE / domain
        for name in ('data.npz', 'samples.json', 'experts.npy', 'lime.npy', 'frequencies.npy',
                     'gate.pt', 'cache_manifest.json', 'ranks.npz', 'analysis.json',
                     'oof_predictions.npz', 'development_predictions.npz'):
            p = folder / name
            digest = sha(p)
            old = read(folder / 'cache_manifest.json')['sha256']
            if name in old:
                assert digest == old[name], str(p)
            paths.append(p)
        paths.extend(folder.glob('heads/*.mlp_seed*.pt'))
    write(OUT / 'manifest.json', dict(created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
          sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
          groups={k: list(v) for k, v in GROUPS.items()}, head_seeds=[0, 1, 2],
          fits=108, optimizer_steps_per_fit=30, test_read=False,
          late_prior_retained=True, early_correction_has_no_prior_input=True))


def domain_predictions(domain):
    folder = OUT / domain
    (folder / 'heads').mkdir(parents=True)
    data = arrays(SOURCE / domain / 'data.npz')
    train = subset(data, np.flatnonzero(data['roles'] == 'ranker_fit'))
    dev = subset(data, np.flatnonzero(data['roles'] == 'development'))
    assert len(dev['users']) == 512 and not set(train['users']) & set(dev['users'])
    folds = fold_ids(train['users'], domain)
    scores = {}; residuals = {}
    for label, d in [('oof', train), ('development', dev)]:
        scores[label] = {name: np.empty(d['mixed'].shape, dtype=np.float32) for name in NAMES}
        scores[label].update(GRAM=d['native'], fixed_late=d['mixed'])
        residuals[label] = {name: np.empty(d['mixed'].shape, dtype=np.float32) for name in NAMES}
    replay = {}; training = {}; coverage = np.zeros(len(train['users']), dtype=int)
    oldpred = {label: arrays(SOURCE / domain / f'{label}_predictions.npz')
               for label in ('oof', 'development')}
    for f in list(range(5)) + ['final']:
        tag = f'fold{f}' if f != 'final' else 'final'
        label = 'oof' if f != 'final' else 'development'
        d = train if f != 'final' else dev
        fit_ix = np.flatnonzero(folds != f) if f != 'final' else np.arange(len(train['users']))
        ix = np.flatnonzero(folds == f) if f != 'final' else np.arange(len(dev['users']))
        assert not set(train['users'][fit_ix]) & set(d['users'][ix])
        if f != 'final':
            coverage[ix] += 1
        status(state='RUNNING', stage='fit', domain=domain, fold=tag)
        for group, columns in GROUPS.items():
            for seed in range(3):
                name = f'{group}_seed{seed}'
                saved = fit(train, fit_ix, group, seed)
                saved.update(fit_users=train['users'][fit_ix], used_users=train['users'][saved['used']])
                path = folder / 'heads' / f'{tag}.{name}.pt'
                torch.save(saved, path)
                saved = torch.load(path, map_location='cpu')
                delta = correction(saved, d['x'][ix][..., list(columns)])
                pred = add_prior(d['mixed'][ix], delta)
                residuals[label][name][ix] = delta
                scores[label][name][ix] = pred
                training[f'{tag}.{name}'] = dict(used=len(saved['used']), fit=len(fit_ix),
                    curve=saved['curve'], optimizer_steps=30, checkpoint_sha256=sha(path))
                if group == 'full':
                    old = torch.load(SOURCE / domain / 'heads' / f'{tag}.mlp_seed{seed}.pt', map_location='cpu')
                    for key in saved['state_dict']:
                        assert torch.equal(saved['state_dict'][key], old['state_dict'][key]), (domain, tag, seed, key)
                    for key in ('mean', 'std', 'used', 'fit_indices', 'curve'):
                        np.testing.assert_array_equal(saved[key], old[key])
                    np.testing.assert_array_equal(pred, oldpred[label][f'mlp_seed{seed}'][ix])
                    replay[f'{tag}.seed{seed}'] = True
        print(json.dumps(dict(domain=domain, fold=tag, heads_completed=len(training))), flush=True)
    assert (coverage == 1).all() and all(replay.values())
    for label, d in [('oof', train), ('development', dev)]:
        common = dict(users=d['users'])
        if label == 'oof':
            common['folds'] = folds
        np.savez_compressed(folder / f'{label}_predictions.npz', **common, **scores[label])
        np.savez_compressed(folder / f'{label}_corrections.npz', **common, **residuals[label])
    write(folder / 'training.json', dict(heads=training, full_exact_replay=replay))


def evaluate_domain(domain):
    folder = OUT / domain
    data = arrays(SOURCE / domain / 'data.npz')
    report = {}; saved_ranks = {}; fold_table = {}
    for label, role in [('oof', 'ranker_fit'), ('development', 'development')]:
        d = subset(data, np.flatnonzero(data['roles'] == role))
        pred = arrays(folder / f'{label}_predictions.npz')
        rr = {name: ranking(d, pred[name]) for name in ('GRAM', 'fixed_late') + NAMES}
        baseline = rr['fixed_late']
        report[label] = dict(users=len(d['users']), covered=int((rr['GRAM'] < 2**30).sum()),
            metrics={name: metrics(r) for name, r in rr.items()}, contrasts={}, movement={})
        for name in NAMES:
            refs = ['GRAM', 'fixed_late']
            if not name.startswith('full'):
                refs.append('full_' + name.split('_', 1)[1])
            report[label]['contrasts'][name] = {
                ref: contrast(rr[name], rr[ref], repeats=1000, seed=20260922) for ref in refs}
            delta = utilities(rr[name]) - utilities(baseline)
            report[label]['movement'][name] = dict(N10_improved=int((delta > 0).sum()),
                N10_hurt=int((delta < 0).sum()), N10_unchanged=int((delta == 0).sum()))
        saved_ranks.update({f'{label}_{k}': v for k, v in rr.items()})
        saved_ranks[f'{label}_users'] = d['users']
        if label == 'oof':
            folds = pred['folds']
            fold_table = {str(f): {name: metrics(r[folds == f]) for name, r in rr.items()} for f in range(5)}
    checks = {}
    for name in NAMES:
        dm, om = report['development']['metrics'], report['oof']['metrics']
        wins = sum(all(fold_table[str(f)][name]['N10'] > fold_table[str(f)][ref]['N10']
                       for ref in ('GRAM', 'fixed_late')) for f in range(5))
        dp = all(dm[name]['N10'] >= 1.01 * dm[ref]['N10'] and dm[name]['H10'] >= dm[ref]['H10']
                 for ref in ('GRAM', 'fixed_late'))
        op = wins >= 4 and all(om[name]['N10'] > om[ref]['N10'] for ref in ('GRAM', 'fixed_late'))
        checks[name] = dict(development_pass=dp, oof_pass=op, positive_folds_vs_both=wins, passed=dp and op)
    report.update(domain=domain, folds=fold_table, checks=checks,
                  test_read=False, development_labels_used_for_fit=False, late_prior_retained=True)
    np.savez_compressed(folder / 'ranks.npz', **saved_ranks)
    write(folder / 'analysis.json', report)
    return report


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'manifest.json').exists(), 'Refuse to overwrite a run'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    status(state='RUNNING', stage='input_hashes')
    check_inputs()
    for domain in ('Toys', 'Beauty'):
        domain_predictions(domain)
    # Both domains' predictions are persisted before this run evaluates any labels.
    status(state='RUNNING', stage='evaluate')
    results = {domain: evaluate_domain(domain) for domain in ('Toys', 'Beauty')}
    decision = {name: all(r['checks'][name]['passed'] for r in results.values()) for name in NAMES}
    decision['early_all_three_seeds'] = all(decision[f'early_seed{s}'] for s in range(3))
    write(OUT / 'summary.json', dict(state='ANALYZED', decision=decision, domains=results,
          seconds=time.monotonic() - started, pending_verification=True, test_read=False,
          new_generation_runs=0, upstream_training_steps=0, small_head_fits=108, optimizer_steps=3240))
    status(state='ANALYZED', decision=decision, pending_verification=True)
    print(json.dumps(dict(state='ANALYZED', decision=decision, seconds=time.monotonic() - started)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        status(state='FAILED', error=repr(error))
        raise
