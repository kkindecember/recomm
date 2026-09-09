"""Review four completed full_r2 runs from saved validation predictions, on CPU only."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from screen_saved_reference import ROOT, close, digest, jsonl, keyed, metrics, read, require


KEYS = ('ndcg@10', 'hit@10', 'hit@50')


def values(ranks):
    ranks = np.asarray(ranks)
    return np.column_stack((np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0),
                            ranks <= 10, ranks <= 50))


def paired(candidate, reference, repetitions, seed):
    difference = values(candidate) - values(reference)
    rng = np.random.default_rng(seed)
    samples = np.empty((repetitions, len(KEYS)))
    for start in range(0, repetitions, 32):
        stop = min(start + 32, repetitions)
        indices = rng.integers(0, len(candidate), (stop - start, len(candidate)))
        samples[start:stop] = difference[indices].mean(axis=1)
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    return {key: dict(delta=float(difference[:, i].mean()),
                      percentile_95ci=intervals[:, i].tolist(),
                      improved_users=int((difference[:, i] > 0).sum()),
                      worsened_users=int((difference[:, i] < 0).sum()),
                      unchanged_users=int((difference[:, i] == 0).sum()))
            for i, key in enumerate(KEYS)}


def run_dataset(dataset, repetitions, seed):
    ref = ROOT / 'artifacts/phase9/p9s_multiseed' / dataset / 'seed2023/validation'
    summary = read(ref / 'summary.json')
    require(summary['split'] == 'validation' and summary['dataset'] == dataset
            and summary['test_read'] is False and summary['integrity']['baseline_identity'],
            'Wrong historical reference')
    require(digest(ref / 'per_user.tsv') == summary['artifacts']['per_user_sha256'],
            'Historical rank hash changed')
    with (ref / 'per_user.tsv').open() as handle:
        cached = keyed(list(csv.DictReader(handle, delimiter='\t')))
    users = sorted(cached)
    require(len(users) == summary['integrity']['users'], 'Reference population mismatch')
    ranks = {name: [int(cached[u][field]) for u in users] for name, field in
             [('gram', 'baseline_rank'), ('gram_pcrf', 'pcrf_rank')]}
    baseline = {name: metrics(row) for name, row in ranks.items()}
    for name, key in [('gram', 'baseline'), ('gram_pcrf', 'pcrf')]:
        close(baseline[name], {k.lower(): v for k, v in summary[key].items()
                               if k.lower() in baseline[name]})
    native = ROOT / 'artifacts/phase18/diffgrm_native' / dataset.lower() / 'run_v1'
    diff = ROOT / ('artifacts/phase18/diff_gram/train_v1' if dataset == 'Toys'
                   else 'artifacts/phase18/diff_gram/beauty/train_v1')
    cohort = read(native / 'full_r2/trend_cohort.json')['user_ids']
    require(cohort == read(diff / 'full_r2/trend_cohort.json')['user_ids']
            and len(set(cohort)) == 2000 and set(cohort) <= set(users), 'Wrong trend cohort')
    q1 = summary['frozen_pcrf']['q1']
    history = np.array([int(cached[u]['history_length']) for u in users])
    tail = np.array([int(cached[u]['target_frequency']) <= q1 for u in users])
    in_cohort = np.isin(users, cohort)
    groups = dict(selection_subset=in_cohort, outside_selection_subset=~in_cohort,
                  history_1_5=history <= 5, history_6_10=(history > 5) & (history <= 10),
                  history_11_20=history > 10, target_tail=tail, target_non_tail=~tail)
    old_audit = read(ROOT / 'artifacts/phase3/s0' / dataset / 'validation/summary.json')
    candidates, canonical_targets = [], None
    for family, run in [('diffgrm', native), ('diff_gram', diff)]:
        directory = run / 'full_r2'
        result = read(directory / 'result.json')
        require(result['state'] == 'COMPLETED' and result['revision'] == 'full_r2',
                'Run is not completed full_r2')
        full = result['full_validation']
        require(full['epoch'] == result['selected_absolute_epoch']
                and full['stage_epoch'] == result['selected_stage_epoch'], 'Selected epoch mismatch')
        prediction = directory / f"full_validation_stage_epoch_{full['stage_epoch']:02d}.jsonl"
        metadata = read(prediction.with_suffix('.json'))
        require(digest(prediction) == full['predictions_sha256'] == metadata['predictions_sha256'],
                'Final prediction hash changed')
        prepared = (run / 'prepared' if family == 'diffgrm' else
                    ROOT / read(run / 'manifest.json')['config']['prepared'])
        manifest = read(prepared / 'manifest.json')
        require(manifest['test_examples_constructed'] == 0, 'Prepared manifest contains test examples')
        hashes = manifest.get('prepared_sha256', manifest.get('prepared_files'))
        names_file = 'item_names.json' if family == 'diffgrm' else 'catalog.json'
        for filename in ('validation.jsonl', names_file):
            require(digest(prepared / filename) == hashes[filename], 'Prepared file hash changed')
        names = read(prepared / names_file)
        if family == 'diff_gram':
            names = names['items']
        expected = keyed(jsonl(prepared / 'validation.jsonl'))
        require(set(expected) == set(users), 'Prepared population mismatch')
        require(all(min(20, len(expected[u]['history'])) == int(cached[u]['history_length'])
                    for u in users), 'Validation history lengths mismatch')
        targets = {u: names[expected[u]['target']] for u in users}
        if canonical_targets is None:
            canonical_targets = targets
        require(targets == canonical_targets, 'Cross-model target identity mismatch')
        sequence = f'GRAM/rec_datasets/{dataset}/user_sequence.txt'
        require(manifest['sources'][sequence] == old_audit['audit']['input_sha256']['user_sequence'],
                'Frozen historical/current sequence provenance differs')
        close(baseline['gram'], manifest['baseline_validation'])
        rows = keyed(jsonl(prediction))
        require(set(rows) == set(users) and len(users) == full['examples'] == metadata['examples'],
                'Final prediction population mismatch')
        candidate_ranks, lengths = [], []
        for user in users:
            row = rows[user]
            require(row['split'] == 'validation' and row['scope'] == 'full_validation'
                    and row['gold_item_id'] == expected[user]['target'], 'Prediction target/split mismatch')
            items = row['ranked_item_ids']
            require(len(items) <= 50 and len(set(items)) == len(items)
                    and all(0 < item < len(names) for item in items), 'Invalid returned items')
            candidate_ranks.append(items.index(row['gold_item_id']) + 1
                                   if row['gold_item_id'] in items else 51)
            lengths.append(len(items))
        measured = metrics(candidate_ranks)
        close(measured, full['metrics'])
        close(measured, metadata['metrics'])
        close(baseline['gram'], full['historical_gram'])
        close(baseline['gram_pcrf'], full['historical_gram_pcrf'])
        require(abs(np.mean(lengths) - full['mean_returned_items']) < 1e-12
                and abs(np.mean(np.array(lengths) < 50) - full['fewer_than_50_fraction']) < 1e-12,
                'Returned-item summary mismatch')
        subgroup_results = {}
        for group, mask in groups.items():
            require(mask.any(), 'Empty subgroup')
            sub = metrics(np.array(candidate_ranks)[mask].tolist())
            base = metrics(np.array(ranks['gram'])[mask].tolist())
            subgroup_results[group] = dict(users=int(mask.sum()), metrics=sub, gram=base,
                delta={k: sub[k] - base[k] for k in sub})
        trajectory = []
        for path in directory.glob('trend_subset_stage_epoch_*.json'):
            data = read(path)
            trajectory.append(dict(epoch=data['epoch'], stage_epoch=data['stage_epoch'],
                                   metrics=data['metrics'], source=str(path.relative_to(ROOT))))
        trajectory.sort(key=lambda row: row['epoch'])
        # Full-batch and subset inference can differ numerically. Preserve both values.
        candidates.append(dict(family=family, result_path=str((directory / 'result.json').relative_to(ROOT)),
            result_sha256=digest(directory / 'result.json'), run_result=result,
            predictions_path=str(prediction.relative_to(ROOT)), predictions_sha256=digest(prediction),
            prepared_manifest_path=str((prepared / 'manifest.json').relative_to(ROOT)),
            prepared_manifest_sha256=digest(prepared / 'manifest.json'),
            metrics=measured, relative_ndcg10_vs_gram=measured['ndcg@10']/baseline['gram']['ndcg@10']-1,
            paired={name: paired(candidate_ranks, base, repetitions, seed) for name, base in ranks.items()},
            subgroups=subgroup_results, trajectory=trajectory,
            mean_returned_items=float(np.mean(lengths)), fewer_than_50_users=int((np.array(lengths)<50).sum())))
    return dict(dataset=dataset, users=len(users), historical=baseline, candidates=candidates,
        reference_path=str((ref / 'per_user.tsv').relative_to(ROOT)), reference_sha256=digest(ref / 'per_user.tsv'),
        reference_summary_sha256=digest(ref / 'summary.json'), tail_q1=q1,
        cohort_sha256=digest(native / 'full_r2/trend_cohort.json'),
        historical_gold_note='Historical rank TSV has no gold column: identity relies on frozen validated '
                            'sequence/cache lineage. Current gold IDs checked against prepared validation.',
        integrity='PASS: hashes, completed revision, selected epoch, exact user sets/current targets, '
                  'history lengths, catalog validity/uniqueness, all six metrics, baseline identity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite an existing review snapshot')
    require(args.repetitions >= 1000, 'Use at least 1000 bootstrap draws')
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), verification_status='ANALYZED',
        training_runs=0, model_inference_runs=0, test_read=False, device='cpu',
        bootstrap=dict(unit='paired user', method='percentile', repetitions=args.repetitions, seed=args.seed,
            confidence=0.95, multiple_comparison_adjusted=False,
            limitation='Conditional on selected checkpoint and seed; ignores model-selection/training variance '
                       'and shared-item dependence. Exploratory validation, not independent confirmation.'),
        software=dict(numpy=np.__version__, script_sha256=digest(Path(__file__))),
        datasets=[run_dataset(ds, args.repetitions, args.seed) for ds in ('Toys', 'Beauty')])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for dataset in result['datasets']:
        print(dataset['dataset'], dataset['integrity'])
        for row in dataset['candidates']:
            print(row['family'], json.dumps(row['paired']['gram'], ensure_ascii=False))


if __name__ == '__main__':
    main()
