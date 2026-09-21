"""Read saved outputs independently of the screen's metric implementation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiment.phase20.conditional_reference_core import Correction


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def independent_rank(scores, positions):
    # Count strictly greater scores, then earlier columns that tie.
    result = np.full(len(scores), 51, dtype=np.int64)
    for i, p in enumerate(positions):
        if p >= 0:
            value = scores[i, p]
            result[i] = 1 + np.count_nonzero(scores[i] > value) + np.count_nonzero(scores[i, :p] == value)
    return result


def main():
    torch.set_num_threads(4)
    root = ROOT / 'artifacts/phase20/cf_exposure_toys_v1'
    config_path = ROOT / 'experiment/phase20/cf_exposure_toys_v1.json'
    config = json.loads(config_path.read_text())
    config_hash = digest(config_path)
    checked = 0
    for mode in ('prepare', 'profile', 'train', 'evaluate'):
        manifest = json.loads((root / mode / 'manifest.json').read_text())
        for name, value in manifest.items():
            assert digest(root / mode / name) == value, (mode, name)
            checked += 1
        assert json.loads((root / mode / 'summary.json').read_text())['config_sha256'] == config_hash
    for name, value in dict(config['input_sha256'], **config['source_sha256']).items():
        assert digest(ROOT / name) == value, name
    q = np.load(root / 'prepare/queries.npz')
    n = int(q['ntrain'])
    scores = np.load(root / 'evaluate/validation_scores.npz')
    result = json.loads((root / 'evaluate/summary.json').read_text())
    with (root / 'evaluate/per_user.tsv').open() as handle:
        records = list(csv.DictReader(handle, delimiter='\t'))
    assert [r['user_id'] for r in records] == q['users'][n:].tolist()
    assert all(len(set(ids)) == 50 for ids in q['candidate_ids'])
    ranks = {}
    for arm in scores.files:
        ranks[arm] = independent_rank(scores[arm], q['positions'][n:])
        np.testing.assert_array_equal(ranks[arm], [int(r[arm]) for r in records])
        for k in (5, 10, 20, 50):
            r = ranks[arm]
            hit = np.mean(r <= k)
            ndcg = sum(1. / np.log2(int(x) + 1) for x in r if x <= k) / len(r)
            assert abs(hit - result['groups']['all'][arm][f'hit@{k}']) < 1e-12
            assert abs(ndcg - result['groups']['all'][arm][f'ndcg@{k}']) < 1e-12
        if arm != 'baseline':
            model = Correction(.5).eval()
            model.load_state_dict(torch.load(root / 'evaluate' / f'{arm}.pt', map_location='cpu'))
            assert sum(p.numel() for p in model.parameters()) == 129
            predicted = q['original_base'][n:].copy()
            with torch.no_grad():
                for start in range(0, len(predicted), 512):
                    predicted[start:start + 512] += model(torch.from_numpy(
                        q['original_features'][n + start:n + start + 512])).numpy()
            np.testing.assert_array_equal(predicted, scores[arm])
    np.testing.assert_array_equal(ranks['baseline'], q['known_ranks'])
    for arm in scores.files:
        np.testing.assert_array_equal(ranks[arm] <= 50, ranks['baseline'] <= 50)
    train = np.load(root / 'prepare/cf_train.npz')
    selected = [json.loads(line) for line in (root / 'prepare/selected_events.jsonl').read_text().splitlines()]
    selected_ids = {r['train_index'] for r in selected}
    assert selected_ids == set(np.flatnonzero(train['excluded']))
    last = {}
    with (ROOT / config['data']['train']).open() as handle:
        for i, line in enumerate(handle):
            r = json.loads(line)
            last[r['user_id']] = i
            if i in selected_ids:
                np.testing.assert_array_equal(train['histories'][i][train['histories'][i] > 0], r['history'])
                assert int(train['targets'][i]) == r['target']
    assert all(last[r['user_id']] == r['train_index'] for r in selected)
    summary = dict(state='PASSED', manifests_checked=checked, validation_users=len(records),
                   independent_ranks_and_eight_metrics=True, three_fuser_checkpoint_scores_exact=True,
                   shared_original_inference=True, baseline_and_hit50_exact=True,
                   terminal_exclusion_reconstructed=True, test_read=False,
                   config_sha256=config_hash, audit_source_sha256=digest(Path(__file__)))
    with (root / 'independent_audit.json').open('x') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
