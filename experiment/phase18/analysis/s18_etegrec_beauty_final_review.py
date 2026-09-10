"""CPU-only review of the completed ETEGRec Beauty screen from saved predictions."""
import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import json

import numpy as np

from experiment.phase18.analysis.s18_rearec_final_review import digest, jsonl, paired, predictions, read
from experiment.phase18.analysis.screen_saved_reference import close, keyed, metrics, require

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refuse to overwrite a frozen review')
    config_path = ROOT / 'experiment/phase18/config/s18_etegrec_beauty.json'
    config = read(config_path)
    root = ROOT / config['output']
    result = read(root / 'result.json')
    require(result['state'] == 'COMPLETED' and result['test_read'] is False, 'Not a completed validation run')
    require(read(root / 'config.json') == config, 'Configuration drift')
    profile = read(ROOT / config['profile_output'] / 'profile.json')
    require(profile['config_sha256'] == digest(config_path) and profile['budget_admitted'], 'Wrong profile')
    for path, sha in read(root / 'local_sources.json').items():
        require(digest(ROOT / path) == sha, f'Local source changed: {path}')
    for path, sha in profile['local_source_sha256'].items():
        require(digest(ROOT / path) == sha, f'Profiled source changed: {path}')
    source = read(ROOT / config['source_manifest'])
    require(source['commit'] == config['source_commit'], 'Author commit mismatch')
    review = read(ROOT / config['input_review'])
    require(review['dataset'] == 'Beauty' and review['test_read'] is False, 'Wrong input review')
    for path, sha in review['files'].items():
        require(digest(ROOT / path) == sha, f'Frozen input changed: {path}')
    cf = review['collaborative_checkpoint']
    require(digest(ROOT / cf['path']) == cf['sha256'], 'CF input checkpoint changed')
    records = keyed(jsonl(ROOT / config['prepared'] / 'validation.jsonl'))
    names = read(ROOT / config['prepared'] / 'catalog.json')['items']
    users = list(records)
    cohort = read(ROOT / config['cohort'])['user_ids']
    require(len(users) == 22363 and len(names) == 12102 and len(cohort) == len(set(cohort)) == 2000,
            'Population mismatch')
    ref_path = ROOT / next(p for p in review['files'] if p.endswith('/per_user.tsv'))
    summary = read(ref_path.with_name('summary.json'))
    require(summary['dataset'] == 'Beauty' and summary['split'] == 'validation' and summary['test_read'] is False
            and summary['integrity']['baseline_identity'], 'Invalid historical summary')
    require(digest(ref_path) == summary['artifacts']['per_user_sha256'], 'Historical rank hash mismatch')
    with ref_path.open() as handle:
        reference = keyed(list(csv.DictReader(handle, delimiter='\t')))
    require(reference.keys() == records.keys(), 'Paired users mismatch')
    require(all(len(records[u]['history']) == int(reference[u]['history_length']) for u in users), 'History mismatch')
    refs = read(ROOT / config['references'])
    for scope, population in [('historical_full', users), ('matched_subset', cohort)]:
        for name, field in [('gram', 'baseline_rank'), ('gram_pcrf', 'pcrf_rank')]:
            observed = metrics([int(reference[u][field]) for u in population])
            close(observed, refs[scope][name])
            if scope == 'historical_full':
                saved = summary['baseline' if name == 'gram' else 'pcrf']
                close(observed, {k.lower(): v for k, v in saved.items() if k.lower() in observed})
    full = result['full_validation']
    ranks = predictions(full, records, names, users, 'full_validation')
    close(full['metrics'], read((ROOT / full['predictions']).with_suffix('.json'))['metrics'])
    close(full['historical_gram'], refs['historical_full']['gram'])
    close(full['historical_gram_pcrf'], refs['historical_full']['gram_pcrf'])
    trajectories = []
    for phase in ('joint', 'finetune'):
        for path in sorted(root.glob(f'trend_{phase}_*.json')):
            data = read(path)
            predictions(data, records, names, cohort, 'trend')
            close(data['historical_gram'], refs['matched_subset']['gram'])
            close(data['historical_gram_pcrf'], refs['matched_subset']['gram_pcrf'])
            trajectories.append(dict(phase=phase, label=data['label'], epoch=int(data['label'].split('_')[-1]),
                                     ndcg10=data['metrics']['ndcg@10'], hit10=data['metrics']['hit@10']))
    selected = max(trajectories, key=lambda x: x['ndcg10'])
    require(selected['label'] == result['selected']['best_label'] == full['label'], 'Selected label mismatch')
    require(result['selected']['selected_file'] == 'joint_best.pt' and selected['label'] == 'joint_290', 'Unexpected selected bundle')
    stages = {s: read(root / f'{s}_result.json') for s in ('rq', 'joint', 'finetune')}
    require(all(s['stopping_reason'] == 'plateau' for s in stages.values()), 'Unexpected stopping reason')
    require(stages['joint']['epoch'] >= config['joint_min_epochs'] and
            stages['finetune']['epoch'] >= config['finetune_min_epochs'] and
            stages['rq']['epoch'] >= config['rq_min_epochs'], 'Minimum training budget not reached')
    events = jsonl(root / 'events.jsonl')
    curves = {}
    for phase in ('joint', 'finetune'):
        rows = [e for e in events if e.get('phase') == phase + '_epoch_complete']
        require([e['epoch'] for e in rows] == list(range(1, stages[phase]['epoch'] + 1)), 'Missing epochs')
        require(all(e['samples'] == 131413 and all(math.isfinite(v) for v in e['losses'].values()) for e in rows),
                'Incomplete train pass or nonfinite losses')
        curves[phase] = rows
    require(sum(e['updated_component'] == 'id' for e in curves['joint']) == 175
            and sum(e['updated_component'] == 'rec' for e in curves['joint']) == 175, 'Alternation count mismatch')
    require(result['selected']['counters'] == dict(id=175 * 257, rec=175 * 257, finetune=65 * 257), 'Update count mismatch')
    comparisons = {name: paired(ranks, [int(reference[u][field]) for u in users])
                   for name, field in [('gram', 'baseline_rank'), ('gram_pcrf', 'pcrf_rank')]}
    old = result['paired']
    for k in ('ndcg10_delta', 'ndcg10_ci95', 'hit10_net_users', 'hit10_ci95'):
        require(np.allclose(comparisons['gram'][k], old[k], rtol=0, atol=1e-12), 'Paired comparison mismatch')
    cohort_set = set(cohort)
    q1 = summary['frozen_pcrf']['q1']
    predicates = {
        'selection_subset': lambda u: u in cohort_set,
        'outside_selection_subset': lambda u: u not in cohort_set,
        'history_1_5': lambda u: int(reference[u]['history_length']) <= 5,
        'history_6_10': lambda u: 5 < int(reference[u]['history_length']) <= 10,
        'history_11_20': lambda u: int(reference[u]['history_length']) > 10,
        'target_tail': lambda u: int(reference[u]['target_frequency']) <= q1,
        'target_non_tail': lambda u: int(reference[u]['target_frequency']) > q1,
    }
    groups = {}
    for name, predicate in predicates.items():
        indices = [i for i, u in enumerate(users) if predicate(u)]
        candidate = metrics([ranks[i] for i in indices])
        gram = metrics([int(reference[users[i]]['baseline_rank']) for i in indices])
        groups[name] = dict(users=len(indices), ndcg10=candidate['ndcg@10'], gram_ndcg10=gram['ndcg@10'],
                            ndcg10_delta=candidate['ndcg@10'] - gram['ndcg@10'])
    output = dict(created_at=datetime.now(timezone.utc).isoformat(), verification_status='ANALYZED',
                  dataset='Beauty', state='COMPLETED', device='cpu', test_read=False, training_runs=0, model_inference_runs=0,
                  checks='PASS: source/input/CF hashes, user population/targets/histories, top50/scores/ranks, six metrics, selection, training passes, original paired interval',
                  sources={str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), ROOT / 'experiment/phase18/analysis/s18_rearec_final_review.py',
                           ROOT / 'experiment/phase18/analysis/screen_saved_reference.py', root / 'result.json', config_path, ref_path,
                           ref_path.with_name('summary.json'), ROOT / config['input_review'])},
                  full_validation=full, stages=stages, selected=selected, trajectory=trajectories,
                  training_curves=curves, paired=comparisons, subgroups=groups,
                  seconds_including_profile=result['seconds_including_profile'],
                  relative_ndcg10_vs_gram=full['metrics']['ndcg@10'] / full['historical_gram']['ndcg@10'] - 1,
                  relative_ndcg10_vs_pcrf=full['metrics']['ndcg@10'] / full['historical_gram_pcrf']['ndcg@10'] - 1,
                  bootstrap=dict(unit='paired user', repetitions=2000, seed=2023, confidence=.95, method='percentile',
                                 multiple_comparison_adjusted=False,
                                 limitation='Conditional on selected checkpoint and seed; ignores training/selection variability and shared-item dependence. Exploratory validation only.'),
                  historical_gold_note='Reference TSV has no gold column; historical target identity relies on the frozen validated input/cache lineage. Current targets checked against prepared validation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(output['checks'])
    print(json.dumps(dict(selected=selected, paired=comparisons, subgroups=groups), ensure_ascii=False))


if __name__ == '__main__':
    main()
