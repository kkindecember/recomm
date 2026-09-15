"""Independent audit of four new B1 runs and the three-initialization aggregate."""
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from verify_fusion_completion import ROOT, read, rows, ranks, sha, value, close, z, KEYS

OUT = ROOT / 'artifacts/phase20/gacr_b1_multiseed_v1'


def main():
    cfg = read(ROOT / 'experiment/phase20/gacr_b1_multiseed_v1.json')
    status, result = read(OUT / 'status.json'), read(OUT / 'summary.json')
    assert status['state'] == result['state'] == 'COMPLETED'
    assert status['completed_tasks'] == status['total_tasks'] == 6
    assert status['completed_new_training_tasks'] == 4 and status['new_optimizer_steps'] == 120
    assert result['new_optimizer_steps'] == 120 and result['backbone_optimizer_steps'] == 0
    assert not result['test_read'] and not result['new_beam_inference']
    assert all(t['exit_code'] == 0 and t['state'] == 'COMPLETED' for t in status['tasks'])
    checked, details = Counter(), {}
    for path, expected in cfg['source_sha256'].items():
        assert sha(ROOT / path) == expected, path
        checked['source_hashes'] += 1
    for domain, dc in cfg['domains'].items():
        for path, expected in dc['input_sha256'].items():
            assert sha(ROOT / path) == expected, path
            checked['domain_input_hashes'] += 1
        manifest = read(ROOT / dc['cache_manifest'])
        assert manifest['cohort'] == dc['frozen_cohort']
        assert manifest['cache_sha256'] == sha(ROOT / dc['training_cache'])
        profile = read(OUT / domain / 'profile/profile.json')
        assert profile['state'] == 'PASSED' and profile['seed2023_score_rank_exact_parity'] and profile['new_optimizer_steps'] == 0
        validation = {r['user_id']: r for r in rows(ROOT / dc['validation'])}
        users = list(validation)
        assert len(users) == dc['users']
        counts, first = Counter(), {}
        for r in rows(ROOT / dc['train']):
            counts[r['target']] += 1
            if r['user_id'] not in first:
                assert len(r['history']) == 1
                first[r['user_id']] = r['history'][0]
        counts.update(first.values())
        freq = np.array([counts[i] for i in range(len(read(ROOT / dc['catalog'])['items']))])
        fit = set((ROOT / dc['c1_training_users']).read_text().splitlines())
        groups = {'all_validation': users,
                  f'target_frequency_le{dc["q1"]}': [u for u in users if counts[validation[u]['target']] <= dc['q1']],
                  f'target_frequency_gt{dc["q1"]}': [u for u in users if counts[validation[u]['target']] > dc['q1']],
                  'outside_c1_training_users': [u for u in users if u not in fit],
                  'inside_c1_training_users': [u for u in users if u in fit]}
        known = ranks(ROOT / dc['known_ranks'])
        summaries = {'2023': read(ROOT / dc['known_summary'])}
        with np.load(ROOT / dc['c1_cf']) as stored:
            assert stored['users'].tolist() == users
            ids, cf = stored['candidate_ids'], stored['scores']
        c1 = {r['user_id']: r['ranked_item_ids'] for r in rows(ROOT / dc['c1'])}
        assert all(ids[i].tolist() == c1[u] and len(set(c1[u])) == 50 for i, u in enumerate(users))
        targets = np.array([validation[u]['target'] for u in users])
        for seed in (2024, 2025):
            folder = OUT / domain / f'seed{seed}'
            s, p = read(folder / 'summary.json'), read(folder / 'protocol.json')
            summaries[str(seed)] = s
            assert s['state'] == 'COMPLETED' and s['new_optimizer_steps'] == 30 and s['cohort_seed'] == 2023
            assert not s['test_read'] and not s['new_CF_scoring'] and not s['new_beam_inference']
            assert p['config'] == cfg and p['input_sha256'] == dc['input_sha256']
            for path, expected in p['loaded_source_sha256'].items():
                assert sha(ROOT / path) == expected, path
                checked['loaded_source_hash_entries'] += 1
            reuse = read(folder / 'cache_reuse.json')
            assert reuse['cache_sha256'] == manifest['cache_sha256'] and reuse['cohort'] == dc['frozen_cohort']
            assert reuse['cohort_seed'] == 2023 and reuse['residual_initialization_seed'] == seed
            records, reference = ranks(folder / 'per_user.tsv'), ranks(ROOT / dc['reference_ranks'][str(seed)])
            assert set(records) == set(reference) == set(known) == set(users)
            for u in users:
                assert records[u]['B0'] == reference[u]['fixed']
                assert records[u]['seed_matched_full'] == reference[u]['full']
                for a in ('original_gram', 'original_pcrf', 'c1', 'c1_pcrf', 'gacr_v6_pcrf'):
                    assert records[u][a] == known[u][a]
                assert all(1 <= r <= 51 for r in records[u].values())
                checked['matched_baseline_user_rows'] += 1
            for group, selected in groups.items():
                for a, expected in s['groups'][group].items():
                    assert expected['users'] == len(selected)
                    for k in KEYS:
                        close(math.fsum(value(records[u][a], k) for u in selected) / len(selected), expected[k])
                        checked['metric_values'] += 1
                for name, values in s['comparisons'].items():
                    a, b = name.split('_minus_'); expected = values[group]
                    assert expected['users'] == len(selected)
                    for k in ('ndcg@10', 'hit@10'):
                        close(math.fsum(value(records[u][a], k) - value(records[u][b], k) for u in selected) / len(selected), expected[k]['delta'])
                    for k in (5, 10, 20, 50):
                        assert sum(records[u][a] <= k < records[u][b] for u in selected) == expected[f'hit{k}_gained']
                        assert sum(records[u][b] <= k < records[u][a] for u in selected) == expected[f'hit{k}_lost']
                    checked['paired_comparisons'] += 1
            for arm in ('B0', 'B1'):
                with np.load(folder / f'{arm}.validation_scores.npz') as stored:
                    assert stored['users'].tolist() == users and np.array_equal(stored['candidate_ids'], ids)
                    g, recorded_rel = stored['residual_scores'], stored['reliability']
                assert np.isfinite(g).all()
                order = np.argsort(-g, axis=1, kind='stable')
                ordered_ids = np.take_along_axis(ids, order, 1)
                pop = freq[ordered_ids]
                rel = 1 - (pop[:, :10] <= dc['q1']).mean(1)
                assert np.array_equal(rel, recorded_rel)
                score = z(np.take_along_axis(g, order, 1)) + rel[:, None] * z(z(np.take_along_axis(cf, order, 1)) - .5 * z(np.log1p(pop)))
                final_ids = np.take_along_axis(ordered_ids, np.argsort(-score, axis=1, kind='stable'), 1)
                for label, selected_ids in ((arm + '_raw', ordered_ids), (arm, final_ids)):
                    hit = selected_ids == targets[:, None]
                    actual = np.where(hit.any(1), hit.argmax(1) + 1, 51)
                    assert np.array_equal(actual, [records[u][label] for u in users])
                    assert np.array_equal(actual <= 50, [known[u]['c1'] <= 50 for u in users])
                checked['replayed_user_arm_scores'] += len(users)
            t = s['training']
            assert t == read(folder / 'B1.training.json')
            assert t['initialization_sha256'] == sha(ROOT / dc['residuals'][str(seed)])
            assert t['checkpoint_sha256'] == sha(folder / 'B1.pt')
            assert t['optimizer_steps'] == 30 and t['checkpoint_selection'] == 'fixed_last_step'
            assert not t['train_only_calibration']['used_for_selection']
            assert [p['step'] for p in t['curve']] == list(range(1, 31))
            assert all(math.isfinite(v) for p in t['curve'] for v in p.values())
            assert t['curve'][-1]['parameter_change_l2'] > 0
            checked['training_trajectories'] += 1
            print(domain, seed, 'PASSED', flush=True)
        dr = result['domains'][domain]
        for seed, s in summaries.items():
            assert dr['seed_metrics'][seed] == s['groups']
            assert dr['seed_sources'][seed] == ('previously_observed' if seed == '2023' else 'new_continuation')
        for group, arms in dr['aggregate'].items():
            for a, values in arms.items():
                b = 'B0_raw' if a.endswith('_raw') else 'B0'
                for key, expected in values.items():
                    xs = [s['groups'][group][a][key] for s in summaries.values()]
                    ds = [s['groups'][group][a][key] - s['groups'][group][b][key] for s in summaries.values()]
                    for k, actual in [('mean', statistics.mean(xs)), ('min', min(xs)), ('max', max(xs)),
                                      ('sample_sd_across_residual_initializations', statistics.stdev(xs)),
                                      ('mean_delta_vs_seed_matched_B0', statistics.mean(ds))]:
                        close(actual, expected[k])
                    assert sum(v > 0 for v in ds) == expected['positive_initialization_count_vs_B0']
                    checked['aggregate_metric_records'] += 1
        for name, expected in dr['decision'].items():
            key, baseline = name.split('_vs_')
            ds = [s['groups']['all_validation']['B1'][key] - s['groups']['all_validation'][baseline][key] for s in summaries.values()]
            assert sum(v > 0 for v in ds) == expected['positive_count']
            close(statistics.mean(ds), expected['mean_delta']); close(min(ds), expected['min_delta'])
        details[domain] = {'users': len(users), 'known_2023_summary_sha256': sha(ROOT / dc['known_summary']),
                           'new_summaries_sha256': {str(seed): sha(OUT / domain / f'seed{seed}/summary.json') for seed in (2024, 2025)}}
    output = {'state': 'PASSED', 'verified_at': datetime.now().astimezone().isoformat(), 'checks': dict(checked),
              'domains': details, 'summary_sha256': sha(OUT / 'summary.json'), 'unique_hashed_files': sha.cache_info().currsize,
              'metric_tolerance': 1e-12, 'verification_status': 'ANALYZED', 'new_training': False, 'test_read': False,
              'scope': 'Independent rank statistics, cached PCRF replay, matched baselines, checkpoint/source/input hashes, immutable cohort and seed aggregates.',
              'limits': 'Does not repeat model forward, optimizer execution or bootstrap intervals. Original 2023 statistics were audited previously and are hash-checked/reused here.'}
    target = Path(__file__).with_name('b1_multiseed_completion_verification.json')
    import json
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
