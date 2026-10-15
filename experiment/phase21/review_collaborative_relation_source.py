"""Independent read-only audit of saved relation-source experiment outputs."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment.phase21 import single_fusion_core as C
from experiment.phase21.candidate_gate_screen import sha, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/phase21/collaborative_relation_source_20261015_v1'
OLD = ROOT/'artifacts/phase21/single_fusion_20261014_v1'
SOURCE = ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'


def read(path):
    return json.loads(path.read_text())


def utility(r, k):
    return np.where(r <= k, 1/np.log2(r+1), 0.)


def main():
    torch.set_num_threads(2)
    assert read(OUT/'status.json')['state'] == 'COMPLETED'
    dest = OUT/'review'
    dest.mkdir(exist_ok=False)
    hashes = dict(read(OUT/'manifest.json')['sha256'])
    for d in ('Toys', 'Beauty'):
        for name in ('training_manifest.json', 'output_manifest.json'):
            for path, digest in read(OUT/d/name)['sha256'].items():
                assert path not in hashes or hashes[path] == digest
                hashes[path] = digest
    for path, digest in hashes.items():
        assert sha(ROOT/path) == digest, path
    counts = dict(prediction_records=0, metric_values=0, paired_contrasts=0,
                  independent_numpy_full_catalog_forwards=0, source_mass_checks=0)
    details = {}
    for d in ('Toys', 'Beauty'):
        p = OUT/d
        samples = read(p/'samples.json')
        rows = [json.loads(s) for s in (p/'predictions.jsonl').read_text().splitlines()]
        saved, summary = np.load(p/'ranks.npz'), read(p/'summary.json')
        assert len(rows) == len(samples) == 512
        assert saved['users'].tolist() == [s['user_id'] for s in samples]
        assert read(p/'lease_status.json')['state'] == 'released'
        ranks = {k: [] for k in summary['metrics']}
        for row, sample in zip(rows, samples):
            assert all(row[k] == sample[k] for k in ('user_id', 'history', 'target'))
            for key in ranks:
                pred = row[key]
                assert len(pred['ids']) == len(set(pred['ids'])) == len(pred['scores']) == 50
                ids, scores = np.asarray(pred['ids']), np.asarray(pred['scores'])
                valid = ~np.isin(ids, sample['history'])
                order = ids[valid][np.lexsort((ids[valid], -scores[valid]))]
                match = np.flatnonzero(order == sample['target'])
                ranks[key].append(int(match[0])+1 if len(match) else 2**30)
                counts['prediction_records'] += 1
        ranks = {k: np.asarray(v) for k, v in ranks.items()}
        for key, values in ranks.items():
            np.testing.assert_array_equal(values, saved[key])
            for k in (5, 10, 20, 50):
                assert abs(float(utility(values, k).mean())-summary['metrics'][key]['N'+str(k)]) < 1e-14
                assert abs(float((values <= k).mean())-summary['metrics'][key]['H'+str(k)]) < 1e-14
                counts['metric_values'] += 2
        for key, comp in summary['comparisons'].items():
            a, b = key.split('_vs_')
            assert abs(float((utility(ranks[a], 10)-utility(ranks[b], 10)).mean())-comp['delta_N10']) < 1e-14
            assert int(((ranks[a] <= 10) & (ranks[b] > 10)).sum()) == comp['top10_gained']
            assert int(((ranks[a] > 10) & (ranks[b] <= 10)).sum()) == comp['top10_lost']
            counts['paired_contrasts'] += 1
        fit_users = [s['user_id'] for s in read(SOURCE/d/'samples.json') if s['role'] == 'ranker_fit']
        assert set(fit_users).isdisjoint(s['user_id'] for s in samples)
        signals = np.load(p/'signals.npz')
        for arm in ('cf_full', 'cf_prompt', 'cf_matched'):
            ck = torch.load(OUT/'models'/arm/d/'user.pt', map_location='cpu')
            assert ck['fit_users'] == fit_users
            assert ck['config']['experts'] == ['ItemCF', 'BGE']
            assert ck['selected_epochs'] == min(ck['selection_curve'], key=lambda v: (v['loss'], v['epoch']))['epoch']
            users = np.load(p/'features'/arm/'validation/user.npy', mmap_mode='r')
            items = np.load(p/'features'/arm/'validation/item.npy', mmap_mode='r')
            for j in (0, 127, 255, 383, 511):
                actual, _ = C.numpy_score(ck['state_dict'], users[j], items[j], ck['mean'], ck['std'], 'user')
                np.testing.assert_allclose(actual, signals[arm][j], atol=3e-6, rtol=2e-5)
                counts['independent_numpy_full_catalog_forwards'] += 1
        np.testing.assert_array_equal(signals['sas_text'], np.load(OLD/d/'signals.npz')['user'])
        np.testing.assert_array_equal(signals['text_only'], np.load(OLD/d/'validation/item.npy', mmap_mode='r')[:, :, 1])
        np.testing.assert_array_equal(signals['cf_only'], np.load(p/'scores/validation/cf_full.npy'))
        graphs = {arm: dict(np.load(p/(arm+'_graph.npz'))) for arm in ('cf_full', 'cf_prompt', 'cf_matched', 'cf_outside')}
        hints = np.load(p/'hint_ids.npy')
        for j in range(len(hints)):
            g = {}
            for arm, a in graphs.items():
                start, stop = a['indptr'][j:j+2]
                g[arm] = dict(zip(a['indices'][start:stop].tolist(), a['weights'][start:stop].tolist()))
            full, prompt, matched, outside = (g[a] for a in ('cf_full', 'cf_prompt', 'cf_matched', 'cf_outside'))
            assert not (set(prompt) & set(outside)) and set(prompt) | set(outside) == set(full)
            assert set(prompt) == {i for i in full if i+1 in hints[j]}
            assert {**prompt, **outside} == full
            assert len(prompt) == len(matched)
            assert set(matched) == set(sorted(full, key=lambda i: (-full[i], i))[:len(prompt)])
            np.testing.assert_allclose(sum(prompt.values()), sum(matched.values()), atol=1e-7, rtol=2e-7)
            counts['source_mass_checks'] += 1
        details[d] = dict(all_graph_partitions_and_matched_edges_verified=True,
            comparisons={k: v for k, v in summary['comparisons'].items() if k.startswith('cf_full_guided_vs_')},
            relative_full_vs_text_only=100*(summary['metrics']['cf_full_guided']['N10']/summary['metrics']['text_only_guided']['N10']-1))
    write(dest/'diagnostic.json', details)
    write(dest/'verification.json', dict(state='VERIFIED_SAVED_OUTPUTS', hashes=len(hashes), **counts,
        test_read=False, new_model_inference=False, intervals_recomputed=False,
        note='Saved predictions and model forwards recomputed; not a fresh GPU repeat.'))
    print(json.dumps(read(dest/'verification.json')))


if __name__ == '__main__':
    main()
