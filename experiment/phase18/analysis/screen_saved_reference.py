"""Compare completed screen predictions with cached ranks on identical users (CPU only)."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keyed(rows):
    result = {str(row['user_id']): row for row in rows}
    require(len(result) == len(rows), 'Duplicate user IDs')
    return result


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metrics(ranks):
    require(bool(ranks) and all(1 <= rank <= 51 for rank in ranks), 'Invalid ranks')
    return {f'{metric}@{k}': math.fsum(
        (1.0 if metric == 'hit' else 1 / math.log2(rank + 1)) if rank <= k else 0.0
        for rank in ranks) / len(ranks)
        for metric in ('hit', 'ndcg') for k in (5, 10, 50)}


def close(left, right):
    require(all(math.isclose(left[k], right[k], rel_tol=0, abs_tol=1e-12) for k in right),
            'Recomputed metrics disagree with saved summary')


def dataset_report(dataset):
    reference = ROOT / 'artifacts/phase9/p9s_multiseed' / dataset / 'seed2023/validation'
    summary = read(reference / 'summary.json')
    ranks_path = reference / 'per_user.tsv'
    require(summary['dataset'] == dataset and summary['split'] == 'validation'
            and summary['test_read'] is False and summary['integrity']['baseline_identity'],
            'Reference is not the historical validation baseline')
    require(digest(ranks_path) == summary['artifacts']['per_user_sha256'], 'Reference hash changed')
    with ranks_path.open() as handle:
        cached = keyed(list(csv.DictReader(handle, delimiter='\t')))
    require(len(cached) == summary['integrity']['users'], 'Reference count mismatch')
    historical = {}
    for name, field in [('gram', 'baseline_rank'), ('gram_pcrf', 'pcrf_rank')]:
        historical[name] = metrics([int(row[field]) for row in cached.values()])
        recorded = summary['baseline' if name == 'gram' else 'pcrf']
        close(historical[name], {key.lower(): value for key, value in recorded.items()
                                if key.lower() in historical[name]})

    native = ROOT / 'artifacts/phase18/diffgrm_native' / dataset.lower() / 'run_v1'
    prepared = native / 'prepared'
    manifest = read(prepared / 'manifest.json')
    validation_path = prepared / 'validation.jsonl'
    require(digest(validation_path) == manifest['prepared_sha256']['validation.jsonl'],
            'Prepared validation hash changed')
    records = keyed(jsonl(validation_path))
    require(set(records) == set(cached), 'Validation user population changed')
    require(all(min(20, len(records[user]['history'])) == int(cached[user]['history_length'])
                for user in records), 'Historical validation history lengths differ')
    close(historical['gram'], manifest['baseline_validation'])
    old_audit = read(ROOT / 'artifacts/phase3/s0' / dataset / 'validation/summary.json')
    sequence_key = f'GRAM/rec_datasets/{dataset}/user_sequence.txt'
    require(manifest['sources'][sequence_key] == old_audit['audit']['input_sha256']['user_sequence'],
            'Historical and current sequence provenance differs')
    # Only compare recorded lineage hashes; never open raw sequence/test labels here.
    names_path = prepared / 'item_names.json'
    require(digest(names_path) == manifest['prepared_sha256']['item_names.json'], 'Item mapping changed')
    names = read(names_path)
    diff = ROOT / ('artifacts/phase18/diff_gram/train_v1' if dataset == 'Toys'
                   else 'artifacts/phase18/diff_gram/beauty/train_v1')
    diff_config = read(diff / 'manifest.json')['config']
    diff_prepared = ROOT / diff_config['prepared']
    diff_names = read(diff_prepared / 'catalog.json')['items']
    diff_records = keyed(jsonl(diff_prepared / 'validation.jsonl'))
    require(set(diff_records) == set(records), 'Candidate validation populations differ')
    require(all(diff_names[diff_records[u]['target']] == names[records[u]['target']]
                for u in records), 'Candidate validation targets differ')

    cohorts = [read(path / 'screen_r1/trend_cohort.json')['user_ids'] for path in (native, diff)]
    require(cohorts[0] == cohorts[1] and len(set(cohorts[0])) == 2000, 'Trend cohorts differ')
    cohort = cohorts[0]
    subset = {name: metrics([int(cached[user][field]) for user in cohort])
              for name, field in [('gram', 'baseline_rank'), ('gram_pcrf', 'pcrf_rank')]}
    evaluations = []
    for family, path, expected in [('diffgrm', native, records), ('diff_gram', diff, diff_records)]:
        for prediction in sorted((path / 'screen_r1').glob('*.jsonl')):
            if prediction.name.endswith('.partial.jsonl') or not prediction.name.startswith(
                    ('trend_subset_stage_epoch_', 'full_validation_stage_epoch_')):
                continue
            metadata_path = prediction.with_suffix('.json')
            if not metadata_path.exists():  # A running evaluator has not committed its summary yet.
                continue
            metadata = read(metadata_path)
            require(digest(prediction) == metadata['predictions_sha256'], 'Prediction hash changed')
            rows = keyed(jsonl(prediction))
            users = cohort if metadata['scope'] == 'trend_subset' else list(records)
            require(set(rows) == set(users) and len(rows) == metadata['examples'], 'Incomplete cohort')
            ranks = []
            for user in users:
                row = rows[user]
                require(row['split'] == 'validation' and row['scope'] == metadata['scope'], 'Split mismatch')
                require(row['gold_item_id'] == expected[user]['target'], 'Candidate target mismatch')
                items = row['ranked_item_ids']
                catalog_size = len(names) if family == 'diffgrm' else len(diff_names)
                require(len(items) <= 50 and len(set(items)) == len(items)
                        and all(0 < item < catalog_size for item in items), 'Invalid candidate ranking')
                ranks.append(items.index(row['gold_item_id']) + 1 if row['gold_item_id'] in items else 51)
            result = metrics(ranks)
            close(result, metadata['metrics'])
            comparator = subset if metadata['scope'] == 'trend_subset' else historical
            evaluations.append(dict(family=family, scope=metadata['scope'], stage_epoch=metadata['stage_epoch'],
                examples=len(users), metrics=result,
                delta_vs_gram={k: result[k] - comparator['gram'][k] for k in result},
                relative_ndcg10_vs_gram=result['ndcg@10'] / comparator['gram']['ndcg@10'] - 1,
                delta_vs_gram_pcrf={k: result[k] - comparator['gram_pcrf'][k] for k in result},
                source=str(prediction.relative_to(ROOT)), source_sha256=digest(prediction)))
    return dict(dataset=dataset, historical_full=historical, matched_subset=subset,
                cohort_users=len(cohort), evaluations=evaluations,
                checks='PASS: saved hashes, full-metric identity, users, targets, histories, cohort and metrics',
                reference=str(ranks_path.relative_to(ROOT)), reference_sha256=digest(ranks_path),
                cohort_sha256=digest(native / 'screen_r1/trend_cohort.json'),
                provenance_note='Historical targets rely on the existing validated cache lineage; '
                                'current prediction targets are checked against prepared validation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite an existing analysis snapshot')
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), device='cpu',
                  training_runs=0, model_inference_runs=0, test_read=False,
                  interpretation='Exploratory comparisons on identical users; no final efficacy or '
                                 'causal attribution claim, no automatic stopping or extension.',
                  datasets=[dataset_report(dataset) for dataset in ('Toys', 'Beauty')])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for dataset in result['datasets']:
        print(dataset['dataset'], 'matched GRAM NDCG@10', dataset['matched_subset']['gram']['ndcg@10'])
        for row in dataset['evaluations']:
            print(row['family'], row['scope'], row['stage_epoch'], 'NDCG@10', row['metrics']['ndcg@10'],
                  'delta', row['delta_vs_gram']['ndcg@10'])


if __name__ == '__main__':
    main()
