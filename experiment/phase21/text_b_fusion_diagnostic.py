"""Descriptive score/coefficient decomposition; no fitting or policy selection."""
import csv
import numpy as np

from experiment.phase20.catalog_scope_core import metrics, contrast, utilities
from experiment.phase21 import text_contribution as runner


def mix(scores, weights, penalty, full_history):
    scores = np.asarray(scores, dtype=np.float32)
    assert scores.ndim == 2 and scores.shape[0] == 3
    value = (np.asarray(weights, dtype=np.float32)[:, None] * scores).sum(0)
    history = np.zeros(scores.shape[1], dtype=bool)
    history[np.asarray(full_history) - 1] = True
    return value - np.float32(penalty) * history


def decomposition(old_scores, new_text, old_weights, new_weights, old_penalty, new_penalty, full_history):
    new_scores = np.array(old_scores, dtype=np.float32, copy=True)
    new_scores[2] = new_text
    return {"AA": mix(old_scores, old_weights, old_penalty, full_history),
            "AB": mix(new_scores, old_weights, old_penalty, full_history),
            "BA": mix(old_scores, new_weights, new_penalty, full_history),
            "BB": mix(new_scores, new_weights, new_penalty, full_history)}


def cohort_summary(rr):
    a, b = rr['text_A'] <= 10, rr['text_B'] <= 10
    cohorts = dict(all=np.ones(len(a), dtype=bool), B_text_gained=b & ~a,
                   B_text_lost=a & ~b, B_text_retained=a & b,
                   B_text_unique=b & (rr['SASRec'] > 10) & (rr['ItemCF'] > 10) & (rr['GRAM'] > 10))
    return {name: dict(users=int(mask.sum()),
                      top10={k: int(((r <= 10) & mask).sum()) for k, r in rr.items()},
                      conditional_N10={k: float(utilities(r[mask]).mean()) for k, r in rr.items()} if mask.any() else {})
            for name, mask in cohorts.items()}


def run(out, original, progress):
    read, write = runner.read, runner.write
    dest = out / 'diagnostic'; dest.mkdir(exist_ok=True)
    results = {}; rows = []
    for domain in runner.DOMAINS:
        samples = read(out / 'B' / domain / 'samples.json')
        chosen = [(j, s) for j, s in enumerate(samples) if s['role'] == 'development']
        assert len(chosen) == 512
        raw = np.load(runner.SOURCE / domain / 'experts.npy', mmap_mode='r')
        text = np.load(out / 'B' / domain / 'semantic.npy', mmap_mode='r')
        fused = {a: np.load(out / a / domain / 'fused.npy', mmap_mode='r') for a in ('A', 'B')}
        gates = {a: runner.base.arrays(out / a / domain / 'gate_prediction.npz') for a in ('A', 'B')}
        cached = {a: runner.base.arrays(original / 'cache_main' / domain / f'{a}_ranks.npz') for a in ('A', 'B')}
        native = runner.base.arrays(out / 'A' / domain / 'ranks.npz')
        users = np.asarray([s['user_id'] for _, s in chosen])
        for r in [*cached.values(), native]: np.testing.assert_array_equal(r['users'], users)
        rr = {k: [] for k in ('AA', 'AB', 'BA', 'BB', 'text_A', 'text_B', 'SASRec', 'ItemCF')}
        errors = dict(AA=0., BB=0.); ids = np.arange(1, raw.shape[-1] + 1)
        for count, (j, s) in enumerate(chosen, 1):
            old = raw[s['source_sample_index']]
            w_a, w_b = [gates[a]['weights'][j] for a in ('A', 'B')]
            p_a, p_b = [gates[a]['penalty'][j] for a in ('A', 'B')]
            values = decomposition(old, text[j], w_a, w_b, p_a, p_b, s['full_history'])
            for cell, arm in (('AA', 'A'), ('BB', 'B')):
                np.testing.assert_allclose(values[cell], fused[arm][j], atol=2e-6, rtol=1e-5)
                errors[cell] = max(errors[cell], float(np.max(np.abs(values[cell] - fused[arm][j]))))
            values.update(text_A=old[2], text_B=text[j], SASRec=old[0], ItemCF=old[1])
            ranks = {k: runner.verified_rank(ids, v, s['target'], s['history']) for k, v in values.items()}
            for k, r in ranks.items(): rr[k].append(r)
            for cell, arm in (('AA', 'A'), ('BB', 'B')):
                assert ranks[cell] == int(cached[arm]['fusion_catalog'][count - 1]), (domain, s['user_id'], cell)
            assert ranks['text_A'] == int(cached['A']['text'][count - 1])
            assert ranks['text_B'] == int(cached['B']['text'][count - 1])
            row = dict(domain=domain, user_id=s['user_id'], sample_index=j, target=s['target'], **ranks,
                       GRAM=int(native['GRAM'][count - 1]), A_guided=int(native['guided'][count - 1]))
            for arm in ('A', 'B'):
                row.update({f'{arm}_weight_{name}': float(gates[arm]['weights'][j, k])
                            for k, name in enumerate(('sasrec', 'itemcf', 'text'))})
                row[f'{arm}_penalty'] = float(gates[arm]['penalty'][j])
            rows.append(row)
            if count % 64 == 0: progress(domain=domain, completed=count, total=512)
        rr = {k: np.asarray(v) for k, v in rr.items()}
        rr.update(GRAM=native['GRAM'], A_guided=native['guided'])
        np.savez_compressed(dest / f'{domain}_ranks.npz', users=users, **rr)
        js = [j for j, _ in chosen]
        result = dict(users=512, metrics={k: metrics(r) for k, r in rr.items()},
                      contrasts={f'{a}_minus_{b}': contrast(rr[a], rr[b], repeats=2000, seed=20260923)
                                 for a, b in [('BB', 'AA'), ('AB', 'AA'), ('BB', 'AB'), ('BA', 'AA')]},
                      cohorts=cohort_summary(rr), reconstruction_max_abs_error=errors,
                      cached_ranks_exact=True,
                      mean_weights={a: gates[a]['weights'][js].mean(0).tolist() for a in ('A', 'B')},
                      mean_abs_weight_change=np.abs(gates['B']['weights'][js] - gates['A']['weights'][js]).mean(0).tolist())
        results[domain] = result
    with (dest / 'per_user.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = dict(state='COMPLETED', created_at=runner.stamp(), domains=results, test_read=False,
                  optimizer_steps=0, coefficient_crosses_are_diagnostics_not_methods=True,
                  target_defined_cohorts_are_descriptive_only=True)
    write(dest / 'summary.json', report)
    return report
