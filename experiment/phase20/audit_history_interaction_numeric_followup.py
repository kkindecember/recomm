"""Follow up the original CPU numerical tolerance miss, without changing its verdict.

Original frozen audit failed: 3 of 970600 B_scores CPU/GPU scores differed by
more than 2e-6, max 2.16066837e-6. This followup retains that failed criterion,
checks exact CPU ranks, and additionally requires bit-exact GPU score replay.
No refit, new metric selection or tolerance widening of the original audit.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.history_interaction_core import build_model

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def count_ranks(scores, positions):
    """No argsort: count larger scores and preceding ties in canonical order."""
    assert np.isfinite(scores).all()
    ranks = np.full(len(positions), 51, dtype=np.int64)
    rows = np.flatnonzero(positions >= 0)
    p = positions[rows]
    selected = scores[rows]
    target = selected[np.arange(len(rows)), p, None]
    ranks[rows] = 1 + (selected > target).sum(1) + (
        (selected == target) & (np.arange(scores.shape[1])[None] < p[:, None])).sum(1)
    return ranks


def individual_ndcg(ranks, k=10):
    out = np.zeros(len(ranks), dtype=np.float64)
    hit = ranks <= k
    out[hit] = np.log(2.) / np.log(ranks[hit] + 1.)
    return out


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config_path = ROOT / 'experiment/phase20/history_interaction_toys_v2.json'
    c = read(config_path)
    root = ROOT / c['output']
    result_path = root / 'independent_audit_numeric_followup.json'
    assert not result_path.exists(), 'Preserve previous audit'
    config_hash = digest(config_path)
    verified_files = 0
    for name, expected in {**c['input_sha256'], **c['source_sha256']}.items():
        assert digest(ROOT / name) == expected, name
        verified_files += 1
    for mode in ('prepare', 'profile', 'train', 'evaluate'):
        assert read(root / mode / 'status.json')['state'] == 'COMPLETED'
        assert read(root / mode / 'summary.json')['config_sha256'] == config_hash
        for name, expected in read(root / mode / 'manifest.json').items():
            assert digest(root / mode / name) == expected, (mode, name)
            verified_files += 1
    with np.load(root / 'prepare/queries.npz') as z:
        a = {k: z[k] for k in z.files}
    with np.load(root / 'train/validation_scores.npz') as z:
        saved = {k: z[k] for k in z.files}
    n = int(a['ntrain'])
    arms = ['A_frozen'] + c['arms']
    assert list(saved) == arms
    positions = a['positions'][n:]
    ranks = {k: count_ranks(v, positions) for k, v in saved.items()}
    np.testing.assert_array_equal(saved['A_frozen'], a['base_scores'][n:])
    np.testing.assert_array_equal(ranks['A_frozen'], a['known_ranks'])
    assert all(np.array_equal(r <= 50, positions >= 0) for r in ranks.values())
    masks = {'all': np.ones(len(positions), dtype=bool),
             'low_frequency': a['target_frequencies'] <= c['q1'],
             'other_frequency': a['target_frequencies'] > c['q1']}
    report = read(root / 'evaluate/summary.json')
    for group, mask in masks.items():
        for arm in arms:
            r = ranks[arm][mask]
            given = report['groups'][group][arm]
            assert len(r) == given['users']
            for k in (5, 10, 20, 50):
                assert abs((r <= k).mean() - given[f'hit@{k}']) < 1e-14
                assert abs(individual_ndcg(r, k).mean() - given[f'ndcg@{k}']) < 1e-14
        for key, groups in report['comparisons'].items():
            arm, ref = key.split('_minus_')
            ar, br = ranks[arm][mask], ranks[ref][mask]
            delta = individual_ndcg(ar) - individual_ndcg(br)
            row = groups[group]
            assert row['hit10_gained'] == int(((ar <= 10) & (br > 10)).sum())
            assert row['hit10_lost'] == int(((ar > 10) & (br <= 10)).sum())
            assert row['ndcg10_improved_users'] == int((delta > 0).sum())
            assert row['ndcg10_worsened_users'] == int((delta < 0).sum())
            assert abs(row['ndcg@10']['delta'] - delta.mean()) < 1e-14
            assert abs(row['hit@10']['delta'] - ((ar <= 10).mean() - (br <= 10).mean())) < 1e-14
    with (root / 'evaluate/per_user.tsv').open() as f:
        per_user = list(csv.DictReader(f, delimiter='\t'))
    assert len(per_user) == len(positions)
    for j, record in enumerate(per_user):
        assert record['user_id'] == a['users'][n+j]
        assert int(record['target_frequency']) == a['target_frequencies'][j]
        assert all(int(record[k]) == ranks[k][j] for k in arms)
    print('All stored ranks, group metrics, paired means and TSV records verified.', flush=True)

    assets = torch.load(root / 'prepare/assets.pt', map_location='cpu')
    catalog = read(ROOT / c['data']['catalog'])['items']
    text = torch.load(ROOT / c['data']['text_embeddings'], map_location='cpu')
    lookup = dict(zip(text['item_ids'], text['embeddings']))
    expected = torch.stack([lookup[item] for item in catalog[1:]])
    expected = F.layer_norm(expected, (c['text_dim'],))
    torch.testing.assert_close(assets['text'][1:], expected, rtol=0, atol=0)
    cf = torch.load(ROOT / c['data']['cf_checkpoint'], map_location='cpu')['model_state_dict']['item_embedding.weight']
    expected = F.layer_norm(F.normalize(cf, dim=-1), (c['cf_dim'],))
    torch.testing.assert_close(assets['cf'][1:], expected[1:], rtol=0, atol=0)
    assert all(not x[0].count_nonzero() and not x.requires_grad for x in assets.values())
    train_report = read(root / 'train/summary.json')
    covered_fit = int((a['fit'] & (a['positions'][:n] >= 0)).sum())
    steps = c['epochs'] * int(np.ceil(covered_fit / c['batch_size']))
    assert train_report['optimizer_steps'] == steps * len(c['arms'])
    assert torch.cuda.is_available(), 'CUDA required for original-device checkpoint replay'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(c['cuda_memory_fraction'], 0)
    assert torch.cuda.mem_get_info()[0]/2**20 >= c['minimum_free_mib']
    gpu_assets = {k:v.cuda() for k,v in assets.items()}
    replay = {}
    tensor = {k: torch.from_numpy(a[k][n:]) for k in
              ('features', 'candidate_ids', 'histories', 'history_attributes')}
    for arm in c['arms']:
        info = train_report['arms'][arm]
        assert info['optimizer_steps'] == steps and info['selected_epoch'] == c['epochs']
        assert info['selection'] == 'fixed_last_epoch'
        curve = read(root / 'train' / f'{arm}.curve.json')
        assert [r['epoch'] for r in curve] == list(range(1, c['epochs']+1))
        model = build_model(c, arm)
        model.load_state_dict(torch.load(root / 'train' / f'{arm}.pt', map_location='cpu'))
        model.eval()
        actual = []
        arm_start, last_log = time.monotonic(), time.monotonic()
        with torch.no_grad():
            for start in range(0, len(positions), c['batch_size']):
                sl = slice(start, start+c['batch_size'])
                residual = model(tensor['features'][sl], tensor['candidate_ids'][sl],
                                 tensor['histories'][sl], tensor['history_attributes'][sl],
                                 assets['text'], assets['cf']).numpy().astype(np.float64)
                actual.append(a['base_scores'][n+start:n+start+len(residual)] + residual)
                if time.monotonic()-last_log > 20:
                    print(f'CPU replay {arm}: {start+len(residual)}/{len(positions)}', flush=True)
                    last_log = time.monotonic()
                assert time.monotonic()-started < 1200, 'Audit exceeds bounded CPU budget'
        actual = np.concatenate(actual)
        error = np.abs(actual-saved[arm])
        original_tolerance_passed = bool((error <= 2e-6).all())
        np.testing.assert_array_equal(count_ranks(actual, positions), ranks[arm])
        gpu_model = model.cuda()
        gpu_tensor = {k:v.cuda() for k,v in tensor.items()}
        gpu_actual = []
        with torch.no_grad():
            for start in range(0, len(positions), c['batch_size']):
                sl = slice(start, start+c['batch_size'])
                residual = gpu_model(gpu_tensor['features'][sl], gpu_tensor['candidate_ids'][sl],
                                     gpu_tensor['histories'][sl], gpu_tensor['history_attributes'][sl],
                                     gpu_assets['text'], gpu_assets['cf']).cpu().numpy().astype(np.float64)
                gpu_actual.append(a['base_scores'][n+start:n+start+len(residual)] + residual)
        gpu_actual = np.concatenate(gpu_actual)
        np.testing.assert_array_equal(gpu_actual, saved[arm])
        del gpu_model, gpu_tensor
        residual = saved[arm] - saved['A_frozen']
        replay[arm] = {'seconds': time.monotonic()-arm_start,
                       'max_score_error': float(error.max()),
                       'original_cpu_atol_2e6_passed': original_tolerance_passed,
                       'cpu_scores_exceeding_original_atol': int((error > 2e-6).sum()),
                       'gpu_all_scores_bit_exact': True,
                       'all_validation_ranks_exact': True,
                       'mean_absolute_residual': float(np.abs(residual).mean()),
                       'fraction_near_residual_bound': float((np.abs(residual) > .99*c['correction_bound']).mean()),
                       'covered_users_with_target_rank_change': int(((positions >= 0) & (ranks[arm] != ranks['A_frozen'])).sum())}
        print(json.dumps({arm: replay[arm]}), flush=True)
    output = {'state': 'VERIFIED_GPU_SCORES_AND_CPU_RANKS_WITH_CPU_TOLERANCE_EXCEPTION',
              'original_frozen_audit_state': 'FAILED_CPU_SCORE_TOLERANCE',
              'original_tolerance_changed': False, 'config_sha256': config_hash,
              'audit_source_sha256': digest(__file__), 'verified_files': verified_files,
              'rank_algorithm': 'count_strictly_greater_plus_preceding_ties',
              'all_users': len(positions), 'groups': {k: int(v.sum()) for k,v in masks.items()},
              'group_metric_values_checked': 3*4*8, 'paired_mean_comparisons_checked': 3*6*2,
              'bootstrap_recomputed': False, 'embedding_mapping_and_normalization_recomputed': True,
              'replay': replay, 'elapsed_seconds': time.monotonic()-started,
              'test_read': False, 'refit_or_checkpoint_selection': False, 'gpu_uuid': c['gpu_uuid'],
              'original_audit_source_sha256': digest(ROOT/'experiment/phase20/audit_history_interaction.py')}
    with result_path.open('x') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
