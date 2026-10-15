"""Independently audit saved stage21.23 predictions; no training or generation.

Use NumPy sorting, separate from the runner's Python rank implementation.
The stage comparison is descriptive on the same reused 512 users / head seed0.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/phase21/dual_candidate_generation_20261014_v1'
SCREEN = ROOT / 'artifacts/phase21/dual_candidate_gate_20261014_v1'
FULL = ROOT / 'artifacts/phase21/full_validation_20260924_v1'
ARMS = ('baseline', 'user', 'candidate', 'scalar')
KEYS = ('native', 'legacy_dual', 'three') + tuple(
    a + suffix for a in ARMS for suffix in ('_late', '_guided'))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rank(prediction, target, history):
    ids = np.asarray(prediction['ids'], dtype=np.int64)
    scores = np.asarray(prediction['scores'], dtype=np.float64)
    assert len(ids) == len(scores) == len(np.unique(ids)) == 50
    assert np.isfinite(scores).all()
    keep = ~np.isin(ids, history)
    ids, scores = ids[keep], scores[keep]
    order = ids[np.lexsort((ids, -scores))]
    hit = np.flatnonzero(order == target)
    return int(hit[0]) + 1 if len(hit) else 2**30


def utility(ranks, k=10):
    return np.where(ranks <= k, 1 / np.log2(ranks + 1), 0.)


def decompose(a, b):
    delta = utility(a) - utility(b)
    both, gained, lost = (a <= 10) & (b <= 10), (a <= 10) & (b > 10), (a > 10) & (b <= 10)
    parts = {name: float(delta[mask].sum() / len(delta)) for name, mask in (
        ('both_hit_rank_changes', both), ('new_hits', gained), ('lost_hits', lost))}
    assert abs(sum(parts.values()) - delta.mean()) < 1e-14
    return dict(delta_N10=float(delta.mean()), relative_N10=float(utility(a).mean() / utility(b).mean() - 1),
                top10_gained=int(gained.sum()), top10_lost=int(lost.sum()),
                top50_net_hits=int((a <= 50).sum() - (b <= 50).sum()),
                outcome_defined_descriptive_decomposition=parts)


def main():
    status = read(SOURCE / 'status.json')
    assert status['state'] == 'COMPLETED' and status['test_read'] is False
    manifest = read(SOURCE / 'manifest.json')['sha256']
    hashes = {}

    def verify(path, expected=None):
        relative = str(path.relative_to(ROOT))
        if relative not in hashes:
            actual = digest(path)
            assert actual == (manifest[relative] if expected is None else expected), relative
            hashes[relative] = actual

    results = {}
    predictions_checked = metrics_checked = contrasts_checked = 0
    for domain in ('Toys', 'Beauty'):
        folder = SOURCE / domain
        assert read(folder / 'status.json')['state'] == 'COMPLETED'
        assert read(folder / 'lease_status.json')['state'] == 'released'
        runtime = read(folder / 'runtime_checks.json')
        assert runtime['model_digest_unchanged'] and runtime['native_reproduced_users'] == 512
        assert runtime['optimizer_steps'] == 0
        for relative, expected in read(folder / 'output_manifest.json')['sha256'].items():
            verify(ROOT / relative, expected)
        for name in ('samples.json', 'data_config.json', 'catalog.json', 'references.json'):
            verify(folder / name)
        config = read(folder / 'data_config.json')
        verify(ROOT / config['parent_checkpoint'], config['parent_sha256'])
        for arm in ARMS[1:]:
            verify(SCREEN / domain / (arm + '_0.pt'))
        verify(SCREEN / domain / 'ranks.npz')
        verify(FULL / domain / 'samples.json')

        samples = read(folder / 'samples.json')
        rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        refs = read(folder / 'references.json')
        full = read(FULL / domain / 'samples.json')
        saved = np.load(folder / 'ranks.npz')
        summary = read(folder / 'summary.json')
        assert summary['users'] == 512 and summary['head_seed'] == 0
        assert len(rows) == len(samples) == len(refs) == 512
        assert len({s['user_id'] for s in samples}) == 512
        assert set(saved.files) == set(KEYS) | {'users'}
        rr = {key: [] for key in KEYS}
        np.testing.assert_array_equal(saved['users'], [s['user_id'] for s in samples])
        identical = 0
        for j, (row, sample, ref) in enumerate(zip(rows, samples, refs)):
            assert sample['validation_index'] == j and sample['role'] == 'development'
            for key in ('user_id', 'target', 'history'):
                assert row[key] == sample[key] == full[sample['full_validation_index']][key]
            assert sample['full_history'] == full[sample['full_validation_index']]['full_history']
            assert ref['user_id'] == sample['user_id']
            for key, oldkey in (('native', 'native'), ('legacy_dual', 'guided'), ('three', 'three')):
                assert row[key]['ids'] == ref[oldkey]['ids']
            identical += row['baseline_guided']['ids'] == row['legacy_dual']['ids']
            for arm in ARMS:
                assert set(row[arm + '_late']['ids']) == set(row['native']['ids'])
            for key in KEYS:
                rr[key].append(rank(row[key], row['target'], row['history']))
                predictions_checked += 1
        assert identical == 512
        rr = {key: np.asarray(value, dtype=np.int64) for key, value in rr.items()}
        np.testing.assert_array_equal(rr['baseline_guided'], rr['legacy_dual'])
        for key, ranks in rr.items():
            np.testing.assert_array_equal(ranks, saved[key])
            for k in (5, 10, 20, 50):
                assert abs(float(utility(ranks, k).mean()) - summary['metrics'][key]['N' + str(k)]) < 1e-14
                assert abs(float((ranks <= k).mean()) - summary['metrics'][key]['H' + str(k)]) < 1e-14
                metrics_checked += 2
        for name, comparison in summary['comparisons'].items():
            a, b = name.split('_vs_')
            independent = decompose(rr[a], rr[b])
            for key in ('delta_N10', 'top10_gained', 'top10_lost'):
                assert abs(independent[key] - comparison[key]) < 1e-14
            assert abs(float(((rr[a] <= 10).astype(float) - (rr[b] <= 10)).mean()) - comparison['delta_H10']) < 1e-14
            contrasts_checked += 1
        for arm in ARMS:
            r, g = rr[arm + '_guided'], rr['native']
            assert summary['coverage'][arm] == dict(
                beam_H50=float((r <= 50).mean()),
                new_top10_outside_native50=int(((r <= 10) & (g > 50)).sum()),
                top50_gained=int(((r <= 50) & (g > 50)).sum()),
                top50_lost=int(((r > 50) & (g <= 50)).sum()))

        pure = np.load(SCREEN / domain / 'ranks.npz')
        ix = np.asarray([s['full_validation_index'] for s in samples])
        transfer = {arm: dict(
            pure_expert_N10=float(utility(pure['baseline' if arm == 'baseline' else arm + '_0'][ix]).mean()),
            fixed_native50_late_N10=float(utility(rr[arm + '_late']).mean()),
            guided_N10=float(utility(rr[arm + '_guided']).mean())) for arm in ARMS}
        candidate_vs_user_stages = {stage: transfer['candidate'][stage] - transfer['user'][stage]
                                    for stage in transfer['candidate']}
        results[domain] = dict(users=512, head_seed=0, baseline_identical_lists=identical,
            transfer=transfer, candidate_minus_user_by_stage=candidate_vs_user_stages,
            candidate_vs_controls={b: decompose(rr['candidate_guided'], rr[b]) for b in
                ('native', 'baseline_guided', 'user_guided', 'scalar_guided', 'three')},
            hits={key: {str(k): int((r <= k).sum()) for k in (10, 50)} for key, r in rr.items()})

    for path in (ROOT / 'experiment/phase21/dual_candidate_generation.py',
                 ROOT / 'experiment/phase21/dual_candidate_gate.py', SOURCE / 'config.json'):
        verify(path)
    out = SOURCE / 'review'
    out.mkdir(exist_ok=False)
    write(out / 'diagnostic.json', results)
    write(out / 'manifest.json', dict(sha256=hashes, reviewer_sha256=digest(Path(__file__))))
    write(out / 'verification.json', dict(state='VERIFIED_SAVED_PREDICTIONS',
        prediction_records_recomputed=predictions_checked, rank_arrays_recomputed=22,
        aggregate_metrics_rechecked=metrics_checked, paired_contrasts_rechecked=contrasts_checked,
        source_and_output_hashes_rechecked=len(hashes), same_user_order_targets_and_history=True,
        all_expected_users_present=True, same_seed0_at_all_three_stages=True,
        baseline_replay_identical_both_domains=True, elapsed_generation_seconds=status['elapsed_seconds'],
        new_training=False, new_generation=False, test_read=False))
    scan = {
        'Simpson_paradox': 'Separate domains; candidate-vs-baseline N10 signs differ, candidate-vs-user signs agree.',
        'ecological_fallacy': 'Paired user ranks; average effects do not imply every user benefits.',
        'Berkson_selection': 'Repeated exploratory cohort of 512 per domain; not an independent confirmation set.',
        'collider_bias': 'All fixed users retained; hit decomposition is explicitly outcome-defined and descriptive.',
        'base_rate_neglect': 'Absolute N10 and counts reported with relative differences; H50 gain vs baseline is 2 users per domain.',
        'regression_to_mean': 'No best seed selected; prior exploratory selection and noise still limit conclusions.',
        'survivorship_bias': 'All 1024 expected records complete across all arms; no outcome filtering.',
        'look_elsewhere': 'All planned arms retained; unadjusted intervals; H50 does not replace primary N10.',
        'forking_paths': 'Fixed seed0 and protocol before decoding; stage diagnostic is post hoc, not a preregistered hypothesis test.',
        'causal_language': 'Transfer failure is observed before beam changes; does not identify GRAM defect or isolate score calibration as cause.',
        'reverse_causality': 'Targets used for outcome calculations only; saved protocol excluded target-based candidate insertion.'}
    write(out / 'interpretation_checks.json', dict(coverage='11/11', checks=scan,
        decision='Current candidate-weight branch is not supported as the preferred mechanism; retain dual anchor as provisional baseline.'))
    print(json.dumps(read(out / 'verification.json'), ensure_ascii=False, indent=2))
    print(json.dumps({d: r['candidate_minus_user_by_stage'] for d, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
