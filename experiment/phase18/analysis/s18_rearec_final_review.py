"""CPU audit of saved ReaRec Beauty results and the user-stopped Toys screen."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiment.phase18.analysis.screen_saved_reference import close, keyed, metrics, require

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def inputs(dataset):
    config = read(ROOT / f'experiment/phase18/config/s18_rearec_{dataset.lower()}.json')
    profile = read(ROOT / config['profile_output'] / 'profile.json')
    require(profile['state'] == 'PASSED' and profile['budget_admitted'], 'Unadmitted profile')
    for path, sha in profile['source_sha256'].items():
        require(digest(ROOT / path) == sha, f'Profile source drift: {path}')
    review = read(ROOT / config['input_review'])
    require(review['dataset'] == dataset and review['test_read'] is False, 'Wrong review')
    for path, sha in review['files'].items():
        require(digest(ROOT / path) == sha, f'Frozen input drift: {path}')
    records = keyed(jsonl(ROOT / config['prepared'] / 'validation.jsonl'))
    names = read(ROOT / config['prepared'] / 'catalog.json')['items']
    cohort = read(ROOT / config['cohort'])['user_ids']
    ref_path = ROOT / config['reference_predictions']
    summary = read(ref_path.with_name('summary.json'))
    require(summary['dataset'] == dataset and summary['split'] == 'validation'
            and summary['test_read'] is False and summary['integrity']['baseline_identity'], 'Invalid baseline')
    require(digest(ref_path) == summary['artifacts']['per_user_sha256'], 'Baseline rank hash mismatch')
    with ref_path.open() as handle:
        reference = keyed(list(csv.DictReader(handle, delimiter='\t')))
    require(records.keys() == reference.keys() and len(records) == review['validation_users'], 'Population mismatch')
    require(len(cohort) == len(set(cohort)) == 2000 and set(cohort) <= records.keys(), 'Cohort mismatch')
    require(all(len(r['history']) == int(reference[u]['history_length']) for u, r in records.items()), 'History mismatch')
    refs = read(ROOT / config['references'])
    for scope, users in [('historical_full', list(records)), ('matched_subset', cohort)]:
        for name, field in [('gram', 'baseline_rank'), ('gram_pcrf', 'pcrf_rank')]:
            observed = metrics([int(reference[u][field]) for u in users])
            close(observed, refs[scope][name])
            if scope == 'historical_full':
                saved = summary['baseline' if name == 'gram' else 'pcrf']
                close(observed, {k.lower(): v for k, v in saved.items() if k.lower() in observed})
    return config, records, names, cohort, reference, summary, refs


def predictions(metadata, records, names, users, scope):
    path = ROOT / metadata['predictions']
    require(digest(path) == metadata['predictions_sha256'], 'Prediction hash mismatch')
    rows = jsonl(path)
    require([r['user_id'] for r in rows] == users and len(rows) == metadata['examples'], 'Prediction order/population mismatch')
    ranks = []
    for row in rows:
        target = records[row['user_id']]['target']
        require(row['gold_item_id'] == target and row['split'] == 'validation' and row['scope'] == scope, 'Target/split mismatch')
        ids, scores = row['ranked_item_ids'], row['scores']
        require(len(ids) == len(set(ids)) == len(scores) == 50
                and all(type(i) is int and 0 < i < len(names) for i in ids), 'Invalid top50')
        require(all(math.isfinite(v) for v in scores) and all(a >= b for a, b in zip(scores, scores[1:])), 'Invalid scores')
        rank = ids.index(target) + 1 if target in ids else 51
        require(row['rank'] == rank, 'Saved rank mismatch')
        ranks.append(rank)
    close(metrics(ranks), metadata['metrics'])
    return ranks


def paired(candidate, reference):
    def values(ranks):
        ranks = np.asarray(ranks)
        return np.column_stack((np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0), ranks <= 10))
    differences = values(candidate) - values(reference)
    rng = np.random.default_rng(2023)
    boot = np.empty((2000, 2))
    for start in range(0, 2000, 25):
        indices = rng.integers(0, len(candidate), (25, len(candidate)))
        boot[start:start + 25] = differences[indices].mean(1)
    ci = np.quantile(boot, [.025, .975], axis=0)
    return dict(ndcg10_delta=float(differences[:, 0].mean()), ndcg10_ci95=ci[:, 0].tolist(),
                hit10_delta=float(differences[:, 1].mean()), hit10_ci95=ci[:, 1].tolist(),
                hit10_net_users=int(differences[:, 1].sum()),
                ndcg10_improved_users=int((differences[:, 0] > 0).sum()),
                ndcg10_worsened_users=int((differences[:, 0] < 0).sum()))


def review(dataset):
    config, records, names, cohort, reference, summary, refs = inputs(dataset)
    root = ROOT / config['output']
    result = read(root / 'result.json')
    trajectories = []
    for path in sorted(root.glob('trend_epoch_*.json')):
        data = read(path)
        predictions(data, records, names, cohort, 'trend')
        trajectories.append(dict(epoch=data['epoch'], ndcg10=data['metrics']['ndcg@10'], hit10=data['metrics']['hit@10']))
    selected = max(trajectories, key=lambda r: r['ndcg10'])
    require(selected['epoch'] == result['selected_epoch'], 'Selected checkpoint differs from observed best')
    events = jsonl(root / 'events.jsonl')
    epochs = [r for r in events if r.get('phase') == 'epoch_complete']
    complete = result['training']['epoch']
    require([r['epoch'] for r in epochs] == list(range(1, complete + 1)), 'Missing complete epochs')
    require(all(r['samples'] == read(ROOT / config['input_review'])['train_examples'] for r in epochs), 'Incomplete train pass')
    require(all(math.isfinite(v) for r in epochs for v in r['losses'].values()), 'Nonfinite epoch loss')
    require(epochs[-1]['optimizer_steps'] == result['training']['optimizer_steps'], 'Update count mismatch')
    base = dict(dataset=dataset, state=result['state'], result_path=str((root / 'result.json').relative_to(ROOT)),
                result_sha256=digest(root / 'result.json'), selected_epoch=selected['epoch'],
                completed_epochs=complete, optimizer_steps=result['training']['optimizer_steps'],
                seconds=result['seconds_including_profile'], trajectory=trajectories,
                training_curve=[{k: r[k] for k in ('epoch', 'losses', 'seconds', 'samples', 'optimizer_steps')} for r in epochs],
                historical=refs['historical_full'], subset_historical=refs['matched_subset'],
                selected_trend=selected, checks='PASS: source/input/prediction hashes, users, targets, histories, catalog, scores, six metrics, selection and complete training passes')
    if dataset == 'Toys':
        require(result['state'] == 'USER_STOPPED' and result['training_complete'] is False
                and result['full_validation'] is None and complete < config['min_epochs'], 'Wrong stopped-run status')
        require(not list(root.glob('full_validation_epoch_*.jsonl')), 'Unexpected full Toys evaluation')
        for name, info in result['checkpoint_files'].items():
            require(digest(root / name) == info['sha256'], 'Stopped checkpoint changed')
        base.update(training_complete=False, interrupted_epoch=result['interrupted_epoch'],
                    stop_request=read(root / 'user_stop_request.json')['instruction'],
                    full_validation=None, minimum_planned_epochs=config['min_epochs'],
                    subset_relative_ndcg10_vs_gram=selected['ndcg10'] / refs['matched_subset']['gram']['ndcg@10'] - 1)
        return base
    require(result['state'] == 'COMPLETED' and read(root / 'training_result.json')['stopping_reason'] == 'plateau', 'Beauty did not complete normally')
    users = list(records)
    full, zero = result['full_validation'], result['zero_step_diagnostic']
    require(full['epoch'] == zero['epoch'] == selected['epoch'], 'Diagnostic checkpoint mismatch')
    ranks = predictions(full, records, names, users, 'full_validation')
    zero_ranks = predictions(zero, records, names, users, 'zero_step_full_validation')
    comparisons = {}
    for name, field in [('gram', 'baseline_rank'), ('gram_pcrf', 'pcrf_rank')]:
        other = [int(reference[u][field]) for u in users]
        comparisons[name] = paired(ranks, other)
        old = result['paired']['comparisons'][name]
        require(all(np.allclose(comparisons[name][k], v, rtol=0, atol=1e-12) for k, v in old.items()), 'Paired result mismatch')
    comparisons['same_checkpoint_zero_step'] = paired(ranks, zero_ranks)
    q1 = summary['frozen_pcrf']['q1']
    groups = {
        'selection_subset': lambda u: u in set(cohort),
        'outside_selection_subset': lambda u: u not in set(cohort),
        'history_1_5': lambda u: int(reference[u]['history_length']) <= 5,
        'history_6_10': lambda u: 5 < int(reference[u]['history_length']) <= 10,
        'history_11_20': lambda u: int(reference[u]['history_length']) > 10,
        'target_tail': lambda u: int(reference[u]['target_frequency']) <= q1,
        'target_non_tail': lambda u: int(reference[u]['target_frequency']) > q1,
    }
    subgroups = {}
    for name, predicate in groups.items():
        indices = [i for i, u in enumerate(users) if predicate(u)]
        measured = metrics([ranks[i] for i in indices])
        baseline = metrics([int(reference[users[i]]['baseline_rank']) for i in indices])
        subgroups[name] = dict(users=len(indices), ndcg10=measured['ndcg@10'],
                              gram_ndcg10=baseline['ndcg@10'], delta=measured['ndcg@10'] - baseline['ndcg@10'])
    base.update(training_complete=True, full_validation=full, zero_step_diagnostic=zero,
                paired=comparisons, subgroups=subgroups,
                relative_ndcg10_vs_gram=full['metrics']['ndcg@10'] / refs['historical_full']['gram']['ndcg@10'] - 1,
                relative_ndcg10_vs_pcrf=full['metrics']['ndcg@10'] / refs['historical_full']['gram_pcrf']['ndcg@10'] - 1)
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Do not overwrite an existing review')
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), verification_status='ANALYZED',
                  training_runs=0, model_inference_runs=0, device='cpu', test_read=False,
                  script_sha256=digest(Path(__file__)),
                  bootstrap=dict(unit='paired user', repetitions=2000, seed=2023, method='percentile',
                                 confidence=.95, multiple_comparison_adjusted=False,
                                 limitation='Conditional on checkpoint/seed; validation selection and shared-item dependence are not included. Exploratory, not independent confirmation.'),
                  historical_gold_note='Historical rank TSV contains no target column; historical target identity relies on the frozen validated input/cache lineage. Current prediction targets are checked against prepared validation.',
                  datasets={name: review(name) for name in ('Beauty', 'Toys')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for dataset, data in result['datasets'].items():
        print(dataset, data['state'], data['checks'])
        print(json.dumps({k: data[k] for k in ('selected_epoch', 'completed_epochs', 'selected_trend')}, ensure_ascii=False))
        if dataset == 'Beauty':
            print(json.dumps(dict(paired=data['paired'], subgroups=data['subgroups']), ensure_ascii=False))


if __name__ == '__main__':
    main()
