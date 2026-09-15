"""Independent cache audit: NumPy/stdlib only, no experiment/model imports or training."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/phase20/gacr_fusion_train_v1'
HERE = Path(__file__).resolve().parent
KEYS = [f'{kind}@{k}' for kind in ('ndcg', 'hit') for k in (5, 10, 20, 50)]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


@lru_cache(None)
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(values):
    return hashlib.sha256('\n'.join(sorted(values)).encode()).hexdigest()


def ranks(path):
    with path.open() as handle:
        records = list(csv.DictReader(handle, delimiter='\t'))
    assert len(records) == len({r['user_id'] for r in records})
    return {r['user_id']: {k: int(v) for k, v in r.items() if k != 'user_id'} for r in records}


def value(rank, key):
    kind, k = key.split('@')
    return (1. if kind == 'hit' else 1. / math.log2(rank + 1)) if rank <= int(k) else 0.


def close(actual, expected):
    assert abs(actual - expected) < 1e-12, (actual, expected)


def z(x):
    return (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)


def main():
    config = read(ROOT / 'experiment/phase20/gacr_fusion_train_v1.json')
    summary, status = read(OUT / 'summary.json'), read(OUT / 'status.json')
    assert summary['state'] == status['state'] == 'COMPLETED'
    assert status['completed_tasks'] == status['total_tasks'] == 4
    assert all(t['state'] == 'COMPLETED' and t['exit_code'] == 0 for t in status['tasks'])
    assert summary['new_optimizer_steps'] == 120 and summary['backbone_optimizer_steps'] == 0
    assert not summary['test_read']
    checked, details = Counter(), {}
    for domain, cfg in config['domains'].items():
        folder = OUT / domain
        ds, protocol = read(folder / 'summary.json'), read(folder / 'protocol.json')
        assert ds == summary['domains'][domain] and protocol['config'] == config
        assert protocol['input_sha256'] == cfg['input_sha256']
        for key in ('input_sha256', 'loaded_source_sha256'):
            for path, expected in protocol[key].items():
                assert sha(ROOT / path) == expected, path
                checked[key + '_entries'] += 1
        print(domain, 'frozen hashes passed', flush=True)
        validation = {r['user_id']: r for r in rows(ROOT / cfg['validation'])}
        users = list(validation)
        train_path = next(ROOT / p for p in cfg['input_sha256'] if p.endswith('/train.jsonl'))
        catalog_path = next(ROOT / p for p in cfg['input_sha256'] if p.endswith('/catalog.json'))
        train, names = list(rows(train_path)), read(catalog_path)['items']
        counts, first, ordinals, sample_keys = Counter(), {}, Counter(), []
        for r in train:
            u = r['user_id']
            counts[r['target']] += 1
            if u not in first:
                assert len(r['history']) == 1
                first[u] = r['history'][0]
            ordinals[u] += 1
            sample_keys.append(f"{u}:{ordinals[u]}:{names[r['target']]}")
        counts.update(first.values())
        freq = np.array([counts[i] for i in range(len(names))])
        training_users = set((ROOT / cfg['c1_training_users']).read_text().splitlines())
        groups = {
            'all_validation': users,
            f'target_frequency_le{cfg["q1"]}': [u for u in users if counts[validation[u]['target']] <= cfg['q1']],
            f'target_frequency_gt{cfg["q1"]}': [u for u in users if counts[validation[u]['target']] > cfg['q1']],
            'outside_c1_training_users': [u for u in users if u not in training_users],
            'inside_c1_training_users': [u for u in users if u in training_users],
        }
        records, reference = ranks(folder / 'per_user.tsv'), ranks(ROOT / cfg['control_ranks'])
        assert set(records) == set(validation) == set(reference) and len(users) == cfg['users']
        for u, record in records.items():
            assert all(1 <= r <= 51 for r in record.values())
            assert record['B0'] == reference[u]['c1_fixed_candidates_residual_pcrf']
            for arm in ('original_gram', 'original_pcrf', 'c1', 'c1_pcrf', 'gacr_v6_pcrf'):
                assert record[arm] == reference[u][arm]
            checked['baseline_user_parity'] += 1
        for group, selected in groups.items():
            for arm, expected in ds['groups'][group].items():
                assert expected['users'] == len(selected)
                for key in KEYS:
                    close(math.fsum(value(records[u][arm], key) for u in selected) / len(selected), expected[key])
                    checked['metric_values'] += 1
            for comparison, result in ds['comparisons'].items():
                a, b = comparison.split('_minus_')
                expected = result[group]
                assert expected['users'] == len(selected)
                for key in ('ndcg@10', 'hit@10'):
                    close(math.fsum(value(records[u][a], key) - value(records[u][b], key) for u in selected) / len(selected), expected[key]['delta'])
                for k in (5, 10, 20, 50):
                    assert sum(records[u][a] <= k < records[u][b] for u in selected) == expected[f'hit{k}_gained']
                    assert sum(records[u][b] <= k < records[u][a] for u in selected) == expected[f'hit{k}_lost']
                checked['paired_comparisons'] += 1
        old = np.load(ROOT / cfg['c1_cf'])
        ids, cf = old['candidate_ids'], old['scores']
        assert old['users'].tolist() == users and ids.shape == (len(users), 50)
        c1 = {r['user_id']: r['ranked_item_ids'] for r in rows(ROOT / cfg['c1'])}
        assert all(c1[u] == ids[i].tolist() and len(set(c1[u])) == 50 for i, u in enumerate(users))
        targets = np.array([validation[u]['target'] for u in users])
        raw_ranks = {}
        for arm in ('B0', 'B1', 'B2'):
            data = np.load(folder / f'{arm}.validation_scores.npz')
            assert data['users'].tolist() == users and np.array_equal(data['candidate_ids'], ids)
            g = data['residual_scores']
            assert g.shape == cf.shape and np.isfinite(g).all() and np.isfinite(cf).all()
            order = np.argsort(-g, axis=1, kind='stable')
            sorted_ids = np.take_along_axis(ids, order, 1)
            population = freq[sorted_ids]
            reliability = 1. - (population[:, :10] <= cfg['q1']).mean(1)
            assert np.array_equal(reliability, data['reliability'])
            combined = z(np.take_along_axis(g, order, 1)) + reliability[:, None] * z(z(np.take_along_axis(cf, order, 1)) - .5 * z(np.log1p(population)))
            final_ids = np.take_along_axis(sorted_ids, np.argsort(-combined, axis=1, kind='stable'), 1)
            for label, ranked in ((arm + '_raw', sorted_ids), (arm, final_ids)):
                hit = ranked == targets[:, None]
                result = np.where(hit.any(1), hit.argmax(1) + 1, 51)
                assert np.array_equal(result, [records[u][label] for u in users]), (domain, label)
                assert np.array_equal(result <= 50, [records[u]['c1'] <= 50 for u in users])
            raw_ranks[arm] = sorted_ids
            checked['replayed_user_arm_scores'] += len(users)
        print(domain, 'metrics and PCRF replay passed', flush=True)
        with np.load(folder / 'training_cache.npz') as stored:
            cache = {k: stored[k] for k in stored.files}
        manifest = ds['cache']
        assert sha(folder / 'training_cache.npz') == manifest['cache_sha256']
        assert sha(folder / 'train_candidates.jsonl') == manifest['candidate_jsonl_sha256']
        assert all(np.isfinite(v).all() for v in cache.values())
        assert cache['candidate_ids'].shape == (manifest['rows'], 50)
        assert np.array_equal(cache['frequencies'], freq[cache['candidate_ids']])
        ordered = sorted(training_users, key=lambda u: hashlib.sha256(f'2023|{domain}|p0-user-split|{u}'.encode()).hexdigest())
        fit_users, cal_users = set(ordered[:round(.8 * len(ordered))]), set(ordered[round(.8 * len(ordered)):])
        cohort = manifest['cohort']
        assert digest(fit_users) == cohort['fit_user_sha256'] and digest(cal_users) == cohort['calibration_user_sha256']
        seen = {'fit': [], 'calibration': []}
        for i, r in enumerate(rows(folder / 'train_candidates.jsonl')):
            source = train[r['train_index']]
            assert r['split'] == 'train' and all(r[k] == source[k] for k in ('user_id', 'history', 'target'))
            assert r['sample_key'] == sample_keys[r['train_index']]
            partition = r['partition']; seen[partition].append(r['sample_key'])
            assert r['user_id'] in (fit_users if partition == 'fit' else cal_users)
            candidates = r['union_ids'][:50]
            assert len(candidates) == len(set(candidates)) == 50
            assert candidates == cache['candidate_ids'][i].tolist()
            assert np.array_equal(np.array(r['features'][:50], dtype=np.float32), cache['features'][i])
            assert np.array_equal(np.array(r['base'][:50], dtype=np.float32), cache['base'][i])
            assert cache['fit'][i] == (partition == 'fit') and cache['head'][i] == (r['target_group'] == 'head')
            position = candidates.index(r['target']) if r['target'] in candidates else -1
            assert position == r['target_position'] == cache['positions'][i]
            checked['training_cache_rows'] += 1
        assert i + 1 == manifest['rows']
        for partition in seen:
            assert len(seen[partition]) == cohort[partition + '_samples']
            assert digest(seen[partition]) == cohort[partition + '_sample_sha256']
        covered = cache['positions'] >= 0
        for name, mask in [('fit_covered', cache['fit'] & covered), ('fit_uncovered', cache['fit'] & ~covered), ('calibration_covered', ~cache['fit'] & covered)]:
            assert int(mask.sum()) == manifest[name]
        for arm in ('B1', 'B2'):
            t = ds['training'][arm]
            assert t == read(folder / f'{arm}.training.json') and t['optimizer_steps'] == 30
            assert t['initialization_sha256'] == sha(ROOT / cfg['residual'])
            assert t['checkpoint_sha256'] == sha(folder / f'{arm}.pt')
            assert t['checkpoint_selection'] == 'fixed_last_step' and not t['train_only_calibration']['used_for_selection']
            assert [p['step'] for p in t['curve']] == list(range(1, 31))
            assert all(math.isfinite(v) for p in t['curve'] for v in p.values())
            assert t['curve'][-1]['parameter_change_l2'] > 0
            for flag, key in ((True, 'covered_head'), (False, 'covered_tail')):
                assert t[key] == int((cache['fit'] & covered & (cache['head'] == flag)).sum())
            checked['training_trajectories'] += 1
        details[domain] = {'users': len(users), 'summary_sha256': sha(folder / 'summary.json'),
                           'per_user_sha256': sha(folder / 'per_user.tsv'),
                           'B1_B2_same_raw_top10_set_users': sum(set(a[:10]) == set(b[:10]) for a, b in zip(raw_ranks['B1'], raw_ranks['B2']))}
        print(domain, 'training cache audit passed', flush=True)
    result = {'state': 'PASSED', 'verified_at': datetime.now().astimezone().isoformat(),
              'checks': dict(checked), 'domains': details, 'metric_tolerance': 1e-12,
              'summary_sha256': sha(OUT / 'summary.json'), 'unique_hashed_files': sha.cache_info().currsize,
              'scope': 'Independent rank statistics, cached PCRF replay, training cache labels/partitions/membership, trajectories and frozen hashes.',
              'limits': 'No backbone, CF or residual model forward rerun; no bootstrap interval rerun or independent training replication. Cache audit alone does not prove target-independent generation; original GPU profile covers sampled target changes.',
              'new_training': False, 'new_beam_inference': False, 'test_read': False, 'verification_status': 'ANALYZED'}
    (HERE / 'fusion_completion_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
