"""Freeze and audit Toys train/validation inputs without reading test or CF weights."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiment.phase18.analysis.screen_saved_reference import close, keyed, metrics, require

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiment/phase18/config/s18_rearec_toys.json')
    args = parser.parse_args()
    config = read(args.config)
    require(config['dataset'] == 'Toys' and config['max_history'] == 20, 'Wrong data contract')
    output = ROOT / config['input_review']
    require(not output.exists(), 'Refuse to overwrite a frozen input review')
    previous_path = 'artifacts/phase18/etegrec_planning/input_review_toys_20260909.json'
    previous = read(previous_path)
    paths = [config['prepared'] + '/' + name for name in ('train.jsonl', 'validation.jsonl', 'catalog.json')]
    paths += [config['cohort'], config['references'], config['reference_predictions']]
    frozen = {p: digest(p) for p in paths}
    require(all(previous['files'].get(p) == sha for p, sha in frozen.items()), 'Previously verified inputs changed')
    rows = {s: jsonl(config['prepared'] + '/' + s + '.jsonl') for s in ('train', 'validation')}
    names = read(config['prepared'] + '/catalog.json')['items']
    by_user = keyed(rows['validation'])
    cohort = read(config['cohort'])['user_ids']
    require((len(rows['train']), len(by_user), len(names)) == (109361, 19412, 11925), 'Population mismatch')
    require(names[1:] == sorted(set(names[1:])), 'Catalog mapping changed')
    require(len(cohort) == len(set(cohort)) == 2000 and set(cohort) <= by_user.keys(), 'Cohort mismatch')
    for split in rows.values():
        for row in split:
            require(row['user_id'] in by_user and 1 <= len(row['history']) <= 20, 'Invalid history/user')
            require(all(type(i) is int and 0 < i < len(names) for i in row['history'] + [row['target']]), 'Invalid item')
    reference_summary_path = str(Path(config['reference_predictions']).with_name('summary.json'))
    summary = read(reference_summary_path)
    require(summary['dataset'] == 'Toys' and summary['split'] == 'validation'
            and summary['test_read'] is False and summary['integrity']['baseline_identity'], 'Invalid saved reference')
    require(frozen[config['reference_predictions']] == summary['artifacts']['per_user_sha256'], 'Reference hash mismatch')
    with (ROOT / config['reference_predictions']).open() as handle:
        reference = keyed(list(csv.DictReader(handle, delimiter='\t')))
    require(reference.keys() == by_user.keys(), 'Paired user mismatch')
    require(all(len(row['history']) == int(reference[u]['history_length']) for u, row in by_user.items()), 'History length mismatch')
    refs = read(config['references'])
    require(refs['dataset'] == 'Toys', 'Reference domain mismatch')
    for scope, users in [('historical_full', list(by_user)), ('matched_subset', cohort)]:
        for model, field in [('gram', 'baseline_rank'), ('gram_pcrf', 'pcrf_rank')]:
            observed = metrics([int(reference[u][field]) for u in users])
            close(observed, refs[scope][model])
            if scope == 'historical_full':
                recorded = summary['baseline' if model == 'gram' else 'pcrf']
                close(observed, {k.lower(): v for k, v in recorded.items() if k.lower() in observed})
    native = 'artifacts/phase18/diffgrm_native/toys/run_v1/prepared/'
    manifest = read(native + 'manifest.json')
    for name in ('validation.jsonl', 'item_names.json'):
        require(digest(native + name) == manifest['prepared_sha256'][name], 'Native validation changed')
        frozen[native + name] = digest(native + name)
    native_rows, native_names = keyed(jsonl(native + 'validation.jsonl')), read(native + 'item_names.json')
    require(native_rows.keys() == by_user.keys(), 'Native users differ')
    for user, row in by_user.items():
        other = native_rows[user]
        require(names[row['target']] == native_names[other['target']], 'Validation target differs')
        require([names[i] for i in row['history']] == [native_names[i] for i in other['history']][-20:], 'Chronological history differs')
    for p in (previous_path, reference_summary_path, native + 'manifest.json'):
        frozen[p] = digest(p)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), dataset='Toys', files=frozen,
                  train_examples=len(rows['train']), validation_users=len(by_user), trend_users=len(cohort),
                  catalog_items=len(names) - 1, baseline=refs['historical_full']['gram'],
                  pcrf=refs['historical_full']['gram_pcrf'], test_read=False,
                  history_order='chronological; left padding in ReaRec adapter',
                  embeddings='Random trainable ID embeddings; no CF or LLM features loaded',
                  checks=['frozen prepared hashes and population', 'catalog IDs and history bounds',
                          'historical full/subset six-metric identity', 'exact paired users/history lengths',
                          'native validation raw targets and chronological histories match'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
