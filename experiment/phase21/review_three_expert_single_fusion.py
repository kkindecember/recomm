"""CPU audit of saved 21.30 outputs; no training, generation or test-set access."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase21/three_expert_single_fusion_20261015_v1'
OLD = ROOT / 'artifacts/phase21/single_fusion_20261014_v1'
KEYS = ('native', 'two_late', 'two_guided', 'three_late', 'three_guided')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def rank(ids, scores, target, history, missing):
    ids, scores = np.asarray(ids), np.asarray(scores)
    keep = ~np.isin(ids, history)
    ids, scores = ids[keep], scores[keep]
    ordered = ids[np.lexsort((ids, -scores))]
    found = np.flatnonzero(ordered == target)
    return int(found[0]) + 1 if len(found) else missing


def ndcg(ranks, k=10):
    return np.where(ranks <= k, 1 / np.log2(ranks + 1), 0.)


def main():
    torch.set_num_threads(2)
    assert read(OUT / 'status.json')['state'] == 'COMPLETED'
    assert not (OUT / 'review').exists(), 'Preserve the existing review receipt.'
    hashes, diagnostics = {}, {}
    records = pure_records = metric_count = contrasts = forward_users = 0
    for domain in ('Toys', 'Beauty'):
        folder = OUT / domain
        assert read(folder / 'status.json')['state'] == 'COMPLETED'
        assert read(folder / 'lease_status.json')['state'] == 'released'
        for filename in ('output_manifest.json', 'training_manifest.json',
                         'train/manifest.json', 'validation/manifest.json'):
            for name, expected in read(folder / filename)['sha256'].items():
                actual = hashes.get(name) or sha(ROOT / name)
                assert actual == expected, name
                hashes[name] = actual
        samples = read(folder / 'samples.json')
        rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        users = [s['user_id'] for s in samples]
        assert len(samples) == len(rows) == len(set(users)) == 512
        summary = read(folder / 'summary.json')
        assert summary == read(OUT / 'summary.json')['domains'][domain]
        saved = np.load(folder / 'ranks.npz')
        np.testing.assert_array_equal(saved['users'], users)
        ranks = {key: [] for key in KEYS}
        for sample, row in zip(samples, rows):
            for key in ('user_id', 'target', 'history'):
                assert sample[key] == row[key]
            for arm in ('two', 'three'):
                assert row[arm + '_late']['ids'] == row['native']['ids']
            for key in KEYS:
                pred = row[key]
                assert len(pred['ids']) == len(set(pred['ids'])) == len(pred['scores']) == 50
                assert np.isfinite(pred['scores']).all()
                ranks[key].append(rank(pred['ids'], pred['scores'], sample['target'], sample['history'], 2**30))
                records += 1
        ranks = {key: np.asarray(value) for key, value in ranks.items()}

        def check_metrics(r, expected):
            nonlocal metric_count
            for k in (5, 10, 20, 50):
                assert abs(ndcg(r, k).mean() - expected['N' + str(k)]) < 1e-14
                assert abs((r <= k).mean() - expected['H' + str(k)]) < 1e-14
                metric_count += 2

        def check_contrast(a, b, expected):
            nonlocal contrasts
            assert abs((ndcg(a) - ndcg(b)).mean() - expected['delta_N10']) < 1e-14
            assert int(((a <= 10) & (b > 10)).sum()) == expected['top10_gained']
            assert int(((a > 10) & (b <= 10)).sum()) == expected['top10_lost']
            contrasts += 1

        for key, r in ranks.items():
            np.testing.assert_array_equal(r, saved[key])
            check_metrics(r, summary['metrics'][key])
        for name, expected in summary['comparisons'].items():
            a, b = name.split('_vs_')
            check_contrast(ranks[a], ranks[b], expected)
        prior = np.load(OLD / domain / 'ranks.npz')
        np.testing.assert_array_equal(prior['users'], users)
        for key, old in (('native', 'native'), ('two_late', 'user_late'), ('two_guided', 'user_guided')):
            np.testing.assert_array_equal(ranks[key], prior[old])

        fit_users = [s['user_id'] for s in read(folder / 'train/users.json')]
        assert set(fit_users).isdisjoint(users)
        pure = np.load(folder / 'pure_ranks.npz')
        signals = np.load(folder / 'signals.npz')
        weights = np.load(folder / 'weights.npz')
        np.testing.assert_array_equal(pure['users'], users)
        np.testing.assert_array_equal(weights['users'], users)
        models = {}
        for arm, source, n_experts in (('two', OLD / domain, 2), ('three', folder, 3)):
            user = np.load(source / 'validation/user.npy', mmap_mode='r')
            item = np.load(source / 'validation/item.npy', mmap_mode='r')
            ck = torch.load(source / 'user.pt', map_location='cpu')
            assert ck['fit_users'] == fit_users and ck['seed'] == 0
            assert ck['selected_epochs'] == min(ck['selection_curve'], key=lambda x: (x['loss'], x['epoch']))['epoch']
            # Independently evaluate the user-only network for every development user.
            sd = {k: v.detach().numpy() for k, v in ck['state_dict'].items()}
            du = user.shape[1]
            normalized = np.clip((user - np.asarray(ck['mean'])[:du]) / np.asarray(ck['std'])[:du], -10, 10)
            hidden = np.maximum(normalized @ sd['input.weight'][:, :du].T + sd['input.bias'], 0)
            logits = hidden @ sd['output.weight'].T + sd['output.bias']
            predicted_weights = np.exp(logits - logits.max(1, keepdims=True))
            predicted_weights /= predicted_weights.sum(1, keepdims=True)
            np.testing.assert_allclose(predicted_weights, weights[arm], atol=3e-6, rtol=2e-5)
            np.testing.assert_allclose(weights[arm].sum(1), 1., atol=2e-6)
            assert (weights[arm] >= 0).all()
            values = signals[arm]
            rr = []
            for j, sample in enumerate(samples):
                predicted = (item[j, :, :n_experts] * predicted_weights[j]).sum(-1)
                np.testing.assert_allclose(predicted, values[j], atol=3e-6, rtol=2e-5)
                rr.append(rank(np.arange(1, len(predicted) + 1), values[j], sample['target'], sample['history'], len(predicted) + 1))
                pure_records += 1
                forward_users += 1
            np.testing.assert_array_equal(rr, pure[arm])
            check_metrics(np.asarray(rr), summary['training']['pure_metrics'][arm])
            models[arm] = dict(selected_epochs=ck['selected_epochs'], fit_users=len(fit_users),
                               mean_weights=weights[arm].mean(0).tolist())
        check_contrast(pure['three'], pure['two'], summary['pure_three_vs_two'])
        metrics = summary['metrics']
        diagnostics[domain] = dict(
            relative_three_vs_two=metrics['three_guided']['N10'] / metrics['two_guided']['N10'] - 1,
            relative_three_vs_native=metrics['three_guided']['N10'] / metrics['native']['N10'] - 1,
            delta_N10_by_stage=dict(pure=float((ndcg(pure['three']) - ndcg(pure['two'])).mean()),
                late=metrics['three_late']['N10'] - metrics['two_late']['N10'],
                guided=metrics['three_guided']['N10'] - metrics['two_guided']['N10']),
            hits={key: {str(k): int((r <= k).sum()) for k in (5, 10, 20, 50)} for key, r in ranks.items()},
            guided_contrast=summary['comparisons']['three_guided_vs_two_guided'], models=models)
    review = OUT / 'review'
    review.mkdir()
    receipt = dict(state='VERIFIED_SAVED_OUTPUTS', prediction_records_recomputed=records,
        pure_ranking_records_recomputed=pure_records, model_user_forwards_recomputed=forward_users,
        aggregate_metrics_recomputed=metric_count, paired_contrasts_rechecked=contrasts,
        training_feature_and_output_hashes_rechecked=len(hashes),
        fit_development_users_disjoint=True, old_dual_and_native_ranks_reproduced=True,
        new_training=False, new_generation=False, test_read=False,
        intervals='Existing exploratory intervals retained; no bootstrap rerun.',
        scope='Saved-output audit; not an independent experimental replication.')
    for filename, value in (('verification.json', receipt), ('diagnostic.json', diagnostics),
                            ('manifest.json', dict(sha256=hashes, reviewer_sha256=sha(Path(__file__))))):
        (review / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == '__main__':
    main()
