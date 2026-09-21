"""Post-hoc zero-training check of all existing individual CF models, without selection."""
import json
import os
from pathlib import Path
import time

import numpy as np
from experiment.phase20.future_factor_screen import sha, ranks, metrics
from experiment.phase20.future_label_multiseed import compare, describe, write

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'artifacts/phase20/future_label_multiseed_20260917'
OUT = PARENT / 'cf_identity_audit'
PAIR = ROOT / 'artifacts/phase20/pair_uncertainty_screen_20260917'
FACTOR = ROOT / 'artifacts/phase20/future_factor_screen_20260917'
CONTRACT = ROOT / 'plan/第二十阶段/未来软监督_单协同模型身份排查_20260917.md'


def main():
    OUT.mkdir(exist_ok=False); start = time.monotonic(); hashes = {}; domains = {}; metric_count = 0

    def use(p):
        hashes[str(p.relative_to(ROOT))] = sha(p); return p

    def read(folder, name):
        with np.load(use(folder / name)) as f: return {k: f[k] for k in f.files}

    def status(state, **kwargs):
        write(OUT / 'status.json', dict(state=state, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            elapsed_seconds=time.monotonic() - start, training_steps=0, gpu_used=False, test_read=False,
            pid=os.getpid(), automatic_next_experiment=False, **kwargs))

    try:
        status('RUNNING')
        for p in (Path(__file__), CONTRACT, ROOT / 'experiment/phase20/future_label_multiseed.py', ROOT / 'experiment/phase20/future_factor_screen.py'): use(p)
        verification = json.loads(use(PARENT / 'verification.json').read_text()); assert verification['state'] == 'VERIFIED'
        for p, h in verification['output_sha256'].items(): use(ROOT / p); assert hashes[p] == h
        for domain in ('Toys', 'Beauty'):
            status('RUNNING', dataset=domain)
            pair = read(PAIR, f'{domain}.predictions.npz'); v = read(FACTOR, f'{domain}.audit_validation_cache.npz')
            for k in ('users', 'candidate_ids', 'positions', 'tail_mask'): np.testing.assert_array_equal(pair[k], v[k])
            gz = v['prior'] + np.log(49)
            parts = {cf: pair['scores'][list(pair['arm_names']).index(cf)] - gz for cf in ('current', 'seed2024', 'seed2025', 'mean')}
            np.testing.assert_allclose((parts['current'] + parts['seed2024'] + parts['seed2025']) / 3, parts['mean'], atol=2e-15, rtol=0)
            for cf in ('current', 'mean'): np.testing.assert_allclose(parts[cf], v['cf_' + cf], atol=2e-15, rtol=0)
            seeds = {}; saved_ranks = {}; max_error = 0.
            for seed in (20260917, 20260918, 20260919):
                old = read(PARENT, f'{domain}.{seed}.predictions.npz'); names = list(old['arm_names'])
                for k in ('users', 'candidate_ids', 'positions', 'tail_mask'): np.testing.assert_array_equal(old[k], v[k])
                masks = dict(all=np.ones(len(v['users']), bool), tail=v['tail_mask'], non_tail=~v['tail_mask'], fit=old['in_fit'], nonfit=~old['in_fit'])
                rr = {}; met = {}
                for arm in ('next_only', 'global_local', 'soft_future'):
                    raw = old['scores'][names.index(arm + '.raw')]
                    for cf, part in parts.items():
                        name = arm + '.' + cf; score = raw + part; rank = ranks(score, v['positions'])
                        pos = np.maximum(v['positions'], 0); gold = score[np.arange(len(pos)), pos, None]
                        direct = np.where(v['positions'] >= 0, 1 + (score > gold).sum(1) + ((score == gold) & (np.arange(50)[None] < pos[:, None])).sum(1), 51)
                        np.testing.assert_array_equal(rank, direct)
                        if cf in ('current', 'mean'):
                            np.testing.assert_array_equal(rank, old['ranks'][names.index(name)])
                            max_error = max(max_error, float(np.max(np.abs(score - old['scores'][names.index(name)]))))
                        rr[name] = rank; saved_ranks[str(seed) + '.' + name] = rank
                        met[name] = {g: metrics(rank[mask]) for g, mask in masks.items()}; metric_count += 8 * len(masks)
                contrasts = {control: {cf: {group: compare(rr['soft_future.' + cf][mask], rr[control + '.' + cf][mask]) for group, mask in masks.items()}
                    for cf in parts} for control in ('next_only', 'global_local')}
                seeds[str(seed)] = dict(metrics=met, comparisons=contrasts)
            aggregates = {control: {cf: {group: {key: describe([s['comparisons'][control][cf][group][key] for s in seeds.values()])
                for key in ('delta_N10', 'relative_N10', 'hit_net')} for group in masks} for cf in parts} for control in ('next_only', 'global_local')}
            domains[domain] = dict(seeds=seeds, aggregate_comparisons=aggregates, existing_score_max_error=max_error)
            np.savez_compressed(OUT / f'{domain}.ranks.npz', users=v['users'], positions=v['positions'], tail_mask=v['tail_mask'],
                in_fit=old['in_fit'], arm_names=np.asarray(list(saved_ranks)), ranks=np.stack(list(saved_ranks.values())))
        for p, h in hashes.items(): assert sha(ROOT / p) == h
        write(OUT / 'summary.json', dict(state='COMPLETED', domains=domains, input_sha256=hashes,
            checks=dict(existing_current_and_mean_all_ranks_identical=True, two_rank_formulas_identical=True, mean_cf_component_identity=True),
            metrics_computed=metric_count, posthoc=True, training_steps=0, test_read=False))
        status('COMPLETED', metrics_computed=metric_count)
        print(json.dumps({d: {c: {f: {k: a['all'][k] for k in ('relative_N10', 'hit_net')} for f, a in v['aggregate_comparisons'][c].items()}
            for c in ('next_only', 'global_local')} for d, v in domains.items()}, indent=2))
    except BaseException as exc:
        status('FAILED', error=repr(exc)); raise


if __name__ == '__main__': main()
