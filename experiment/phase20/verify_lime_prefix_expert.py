"""Independent cohort, item-ranking, contrast and frozen-input audit.

Never import the experiment runner or expert scoring implementation. NPZ arrays
are eagerly loaded once per domain; no repeated decompression inside row loops.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from experiment.phase20.verify_lime_mechanism import calculate, compare, digest, read, rows

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/lime_prefix_expert_20260921_v1'
OLD = ROOT / 'artifacts/phase20/lime_decode_screen_20260921_v1/full_parallel_v4'
PRIOR = ROOT / 'artifacts/phase20/lime_mechanism_screen_20260921_v1'
METHODS = ('fixed_cal', 'fixed_no_cal', 'dynamic_cal', 'dynamic_no_cal')


def main():
    frozen = read(OUT / 'manifest.json')['sha256']
    for path, expected in frozen.items():
        assert digest(ROOT / path) == expected, ('Changed input', path)
    verified = {}
    for domain in ('Toys', 'Beauty'):
        folder = OUT / domain
        samples = read(folder / 'samples.json')
        original = read(OLD / domain / 'samples.json')
        previous = read(PRIOR / domain / 'samples.json')
        assert len(samples) == 643 and len({r['user_id'] for r in samples}) == 643
        assert samples[:131] == original[:131]
        excluded = {r['user_id'] for r in original + previous}
        recovery = ROOT / f'artifacts/phase20/lime_comparison_20260920_v1/recovery_v2/{domain}'
        with np.load(recovery / 'evaluation_seed0_cal2048.npz', allow_pickle=False) as old:
            users = old['users'].copy()
        expected = sorted([str(u) for u in users if str(u) not in excluded],
            key=lambda u: hashlib.sha256(f'20260921|lime-prefix-expert-v1|{domain}|{u}'.encode()).digest())[:512]
        assert [r['user_id'] for r in samples[131:]] == expected
        with np.load(ROOT / f'artifacts/phase20/catalog_scope_20260919_v1/{domain}/queries.npz', allow_pickle=False) as q:
            assert {r['query_index'] for r in samples}.isdisjoint(q['calibration_indices'].tolist())
        with np.load(folder / 'signals.npz', allow_pickle=False) as archive:
            signals = {k: archive[k] for k in archive.files}
        fused = (signals['experts'] * signals['weights'][:, :, None]).sum(axis=1)
        fused -= signals['history_mask'] * signals['penalty'][:, None]
        np.testing.assert_array_equal(fused, signals['lime'])
        bank_path = ROOT / f'artifacts/phase20/nearest_comparison_20260918/{domain}.train_sequences.npz'
        with np.load(bank_path, allow_pickle=False) as archive:
            bank = {k: archive[k] for k in archive.files}
        user_index = {str(u): i for i, u in enumerate(bank['users'])}
        for j, sample in enumerate(samples):
            ui = user_index[sample['user_id']]
            hist = bank['items'][bank['offsets'][ui]:bank['offsets'][ui + 1]]
            np.testing.assert_array_equal(np.flatnonzero(signals['history_mask'][j]) + 1, np.unique(hist))
            assert hist[-20:].tolist() == sample['history']
        selection, summary = read(folder / 'selection.json'), read(folder / 'summary.json')
        assert selection['alpha'] == summary['alpha'] == .3
        assert selection['coefficient'] == summary['coefficient'] == 1.
        assert read(OLD / domain / 'selection.json')['selected_alpha']['lime'] == .3
        assert selection['evaluation_users_read'] == 0 and summary['new_tuning_users'] == 0
        assert digest(folder / 'selection.json') == summary['selection_sha256']
        assert summary['model_unchanged'] and not summary['test_read'] and summary['optimizer_steps'] == 0
        for path, expected in read(folder / 'loaded_sources.json').items():
            assert digest(ROOT / path) == expected, path
        evaluation = rows(folder / 'evaluation.jsonl')
        assert len(evaluation) == 512 and [r['sample_index'] for r in evaluation] == list(range(131, 643))
        checked = 0
        expected_names = {'GRAM', 'LIME', 'LIME_after_GRAM'} | set(METHODS) | {n + '_late' for n in METHODS}
        for row in evaluation:
            j = row['sample_index']; sample = samples[j]
            assert all(row[k] == sample[k] for k in ('user_id', 'target', 'history', 'split'))
            assert set(row['arms']) == expected_names
            history = set(row['history'])
            for name, arm in row['arms'].items():
                if name == 'LIME':
                    ids, scores = np.arange(1, signals['lime'].shape[1] + 1), signals['lime'][j]
                else:
                    values = row['candidate_scores'][name]
                    ids, scores = np.asarray(values['ids']), np.asarray(values['scores'])
                    assert len(ids) == 50 and len(set(ids.tolist())) == 50
                assert len(ids) == len(scores) and np.isfinite(scores).all()
                keep = np.asarray([int(i) not in history for i in ids])
                ids, scores = ids[keep], scores[keep]
                ordered = ids[np.lexsort((ids, -scores))][:50].tolist()
                rank = ordered.index(row['target']) + 1 if row['target'] in ordered else 51
                assert ordered == arm['items'] and rank == arm['rank'], (domain, j, name)
                checked += 1
            for name in METHODS:
                assert row['candidate_scores'][name + '_late']['ids'] == row['candidate_scores']['GRAM']['ids']
            assert row['arms']['LIME']['items'] == signals['lime_top50'][j].tolist()
        ranks = {n: [r['arms'][n]['rank'] for r in evaluation] for n in expected_names}
        assert set(summary['metrics']) == expected_names
        for name, rr in ranks.items():
            for key, value in calculate(rr).items():
                np.testing.assert_allclose(value, summary['metrics'][name][key], atol=1e-14, rtol=0)
        pairs = {n + '_vs_GRAM': (n, 'GRAM') for n in ranks if n != 'GRAM'}
        for n in METHODS:
            pairs[n + '_vs_late'] = (n, n + '_late')
            pairs[n + '_vs_direct_LIME_rerank'] = (n, 'LIME_after_GRAM')
        for a, b in [('dynamic_cal', 'fixed_cal'), ('dynamic_no_cal', 'fixed_no_cal'),
                     ('fixed_no_cal', 'fixed_cal'), ('dynamic_no_cal', 'dynamic_cal'),
                     ('dynamic_no_cal', 'fixed_cal')]:
            pairs[a + '_vs_' + b] = (a, b)
        assert set(pairs) == set(summary['comparisons'])
        for name, (a, b) in pairs.items():
            assert list((a, b)) == summary['comparison_pairs'][name]
            for key, value in compare(ranks[a], ranks[b]).items():
                np.testing.assert_allclose(value, summary['comparisons'][name][key], atol=1e-12, rtol=0)
        assert summary['weight_changed_users'] == sum(r['prefix_diagnostics']['max_weight_change'] > 0 for r in evaluation)
        assert summary['completed_decodes'] == 18 + 512 * 5
        verified[domain] = dict(predictions_checked=checked, evaluation_users=512,
            both_previous_cohorts_excluded=True, gate_users_excluded=True,
            frozen_alpha_verified=True, history_masks_and_fusion_rebuilt=True,
            all_metrics_and_bootstrap_recomputed=True, summary_sha256=digest(folder / 'summary.json'))
    result = dict(state='VERIFIED', verified_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        domains=verified, input_hashes_verified=len(frozen), test_read=False,
        independent_of_runner_and_policy=True)
    temporary = OUT / f'verification.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(OUT / 'verification.json')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
