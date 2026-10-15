"""Read-only recheck of frozen candidate-text outputs; writes a new review receipt."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment.phase21 import single_fusion_core as C
from experiment.phase21.candidate_gate_screen import sha, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/phase21/candidate_text_source_20261015_v1'
OLD = ROOT/'artifacts/phase21/single_fusion_20261014_v1'
SOURCE = ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'


def read(p):
    return json.loads(p.read_text())


def utility(r, k):
    return np.where(r <= k, 1/np.log2(r+1), 0.)


def main():
    torch.set_num_threads(2)
    assert read(OUT/'status.json')['state'] == 'COMPLETED'
    dest = OUT/'review'
    dest.mkdir(exist_ok=False)
    hashes = dict(read(OUT/'manifest.json')['sha256'])
    for d in ('Toys', 'Beauty'):
        for file in ('training_manifest.json', 'output_manifest.json'):
            for path, digest in read(OUT/d/file)['sha256'].items():
                if path in hashes:
                    assert hashes[path] == digest
                hashes[path] = digest
    for path, digest in hashes.items():
        assert sha(ROOT/path) == digest, path
    diagnostics = {}
    records = models = metrics = contrasts = 0
    for d in ('Toys', 'Beauty'):
        p = OUT/d
        samples = read(p/'samples.json')
        rows = [json.loads(s) for s in (p/'predictions.jsonl').read_text().splitlines()]
        saved = np.load(p/'ranks.npz')
        summary = read(p/'summary.json')
        assert len(rows) == len(samples) == 512
        assert saved['users'].tolist() == [s['user_id'] for s in samples]
        ranks = {k: [] for k in summary['metrics']}
        for row, sample in zip(rows, samples):
            assert all(row[k] == sample[k] for k in ('user_id', 'history', 'target'))
            for key in ranks:
                pred = row[key]
                assert len(pred['ids']) == len(set(pred['ids'])) == len(pred['scores']) == 50
                values = {int(i): float(v) for i, v in zip(pred['ids'], pred['scores']) if i not in sample['history']}
                y = sample['target']
                r = 2**30 if y not in values else 1+sum(v > values[y] or (v == values[y] and i < y) for i, v in values.items())
                ranks[key].append(r)
                records += 1
        ranks = {k: np.asarray(v) for k, v in ranks.items()}
        for key, values in ranks.items():
            np.testing.assert_array_equal(values, saved[key])
            for cutoff in (5, 10, 20, 50):
                assert abs(float(utility(values, cutoff).mean())-summary['metrics'][key]['N'+str(cutoff)]) < 1e-14
                assert abs(float((values <= cutoff).mean())-summary['metrics'][key]['H'+str(cutoff)]) < 1e-14
                metrics += 2
        for key, comp in summary['comparisons'].items():
            a, b = key.split('_vs_')
            delta = utility(ranks[a], 10)-utility(ranks[b], 10)
            assert abs(float(delta.mean())-comp['delta_N10']) < 1e-14
            assert int(((ranks[a] <= 10) & (ranks[b] > 10)).sum()) == comp['top10_gained']
            assert int(((ranks[a] > 10) & (ranks[b] <= 10)).sum()) == comp['top10_lost']
            contrasts += 1
        fit_users = [s['user_id'] for s in read(SOURCE/d/'samples.json') if s['role'] == 'ranker_fit']
        assert set(fit_users).isdisjoint(s['user_id'] for s in samples)
        signals = np.load(p/'signals.npz')
        selected = {}
        for arm in ('candidate_id', 'both_id'):
            ck = torch.load(OUT/'models'/arm/d/'user.pt', map_location='cpu')
            assert ck['fit_users'] == fit_users
            assert ck['selected_epochs'] == min(ck['selection_curve'], key=lambda v: (v['loss'], v['epoch']))['epoch']
            users = np.load(p/'features'/arm/'validation/user.npy', mmap_mode='r')
            items = np.load(p/'features'/arm/'validation/item.npy', mmap_mode='r')
            for j in (0, 127, 255, 383, 511):
                actual, _ = C.numpy_score(ck['state_dict'], users[j], items[j], ck['mean'], ck['std'], 'user')
                np.testing.assert_allclose(actual, signals[arm][j], atol=3e-6, rtol=2e-5)
                models += 1
            selected[arm] = ck['selected_epochs']
        np.testing.assert_array_equal(signals['full'], np.load(OLD/d/'signals.npz')['user'])
        np.testing.assert_array_equal(signals['sas_only'], np.load(OLD/d/'validation/item.npy', mmap_mode='r')[:, :, 0])
        diagnostics[d] = dict(selected_epochs=selected,
            relative_N10={a: 100*(summary['metrics']['full_guided']['N10']/summary['metrics'][a+'_guided']['N10']-1)
                          for a in ('candidate_id', 'both_id', 'sas_only')},
            comparisons={k: v for k, v in summary['comparisons'].items() if k.startswith('full_guided_vs_')})
    write(dest/'diagnostic.json', diagnostics)
    write(dest/'verification.json', dict(state='VERIFIED_SAVED_OUTPUTS', hashes=len(hashes),
        prediction_records=records, metric_values=metrics, paired_contrasts=contrasts,
        independent_numpy_full_catalog_forwards=models, test_read=False, new_inference=False))
    print(json.dumps(read(dest/'verification.json')))


if __name__ == '__main__':
    main()
