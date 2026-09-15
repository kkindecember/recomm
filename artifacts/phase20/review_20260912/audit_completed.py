"""CPU audit of existing Stage20 validation artifacts; no inference or training.

Run from repository root: python artifacts/phase20/review_20260912/audit_completed.py
Only the adjacent audit JSON/CSV are written. Original experiment files stay intact.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
P20 = ROOT / 'artifacts/phase20'
checks, hashes, predictions, tables = [], {}, {}, {}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    hashes[str(path.relative_to(ROOT))] = h.hexdigest()
    return h.hexdigest()


def read_json(path):
    digest(path)
    return json.loads(path.read_text())


def read_tsv(path):
    digest(path)
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    result = {r['user_id']: r for r in rows}
    assert len(result) == len(rows)
    return result


def ranks(rows, name, users):
    a = np.array([int(rows[u][name]) for u in users])
    assert ((a >= 1) & (a <= 51)).all()
    return a


def dcg(a, k=10):
    return np.where(a <= k, 1 / np.log2(a + 1), 0)


def metrics(a):
    result = {'users': len(a)}
    for k in (5, 10, 20, 50):
        result[f'hit@{k}'] = float((a <= k).mean())
        result[f'ndcg@{k}'] = float(dcg(a, k).mean())
    return result


def check_metrics(label, a, expected):
    actual = metrics(a)
    error = max(abs(actual[k] - expected[k]) for k in actual)
    assert error < 1e-12, (label, error)
    checks.append({'label': label, 'users': len(a), 'max_absolute_error': error})


def comparison(a, b):
    result = {'users': len(a), 'ndcg10_delta': float((dcg(a) - dcg(b)).mean()),
              'ndcg10_relative_percent': float((dcg(a).mean()/dcg(b).mean()-1)*100)}
    for k in (10, 50):
        gained = int(((a <= k) & (b > k)).sum())
        lost = int(((a > k) & (b <= k)).sum())
        result[f'hit{k}'] = {'baseline_hits': int((b <= k).sum()), 'candidate_hits': int((a <= k).sum()),
                             'gained': gained, 'lost': lost, 'net': gained-lost}
    masks = {'both_cover50': (a <= 50) & (b <= 50),
             'candidate_only_cover50': (a <= 50) & (b > 50),
             'baseline_only_cover50': (a > 50) & (b <= 50),
             'neither_cover50': (a > 50) & (b > 50)}
    result['coverage_accounting'] = {
        label: {'users': int(mask.sum()),
                'ndcg10_contribution_to_full_delta': float((dcg(a)-dcg(b))[mask].sum()/len(a)),
                'hit10_net': int(((a <= 10).astype(int)-(b <= 10))[mask].sum())}
        for label, mask in masks.items()}
    return result


prepared = ROOT / 'artifacts/phase18/diff_gram/prepared_v2_categories'
validation = {r['user_id']: r for r in map(json.loads, (prepared/'validation.jsonl').read_text().splitlines())}
catalog = read_json(prepared/'catalog.json')
manifest = read_json(P20/'sprec_gram_toys_v1/manifest.json')
for name, expected in manifest['data_manifest']['prepared_files'].items():
    assert digest(prepared/name) == expected
parent_rows = read_tsv(P20/'sprec_gram_toys_v1/parent_full.ranks.tsv')
users = list(parent_rows)
assert len(users) == len(validation) == 19412 and set(users) == set(validation)
reference = read_json(P20/'sprec_gram_toys_v1/matched_parent_reference.json')
fit_users = set((ROOT/'artifacts/phase4/gcdh_p0_splits/Toys/train_users.txt').read_text().splitlines())
tail_users = {u for u in users if int(reference[u]['target_frequency']) <= 5}
all_ranks = {'original_gram': ranks(parent_rows, 'raw_rank', users),
             'original_pcrf': ranks(parent_rows, 'pcrf_rank', users)}


def check_prediction(path, rows, rank_column):
    digest(path)
    seen = set()
    with path.open() as handle:
        for line in handle:
            r = json.loads(line)
            u, candidates = r['user_id'], r['ranked_item_ids']
            assert u not in seen and u in rows
            seen.add(u)
            assert r['split'] == 'validation' and r['gold_item_id'] == validation[u]['target']
            assert len(candidates) == len(set(candidates)) == len(r['scores']) == 50
            assert all(0 < x < len(catalog['items']) for x in candidates)
            assert np.isfinite(r['scores']).all()
            rank = candidates.index(r['gold_item_id'])+1 if r['gold_item_id'] in candidates else 51
            assert rank == int(rows[u][rank_column])
    assert seen == set(rows)
    predictions[str(path.relative_to(ROOT))] = {'users': len(seen), 'unique_legal_finite_and_raw_rank_exact': True}


check_prediction(P20/'sprec_gram_toys_v1/parent_full.predictions.jsonl', parent_rows, 'raw_rank')
sprec = read_json(P20/'sprec_gram_toys_v1/summary.json')
sft = read_json(P20/'round1_sft_full_v1/summary.json')
for name, directory, tag, summary in [
    ('sprec', 'sprec_gram_toys_v1', 'round1_dpo_full', sprec['full_validation']['round1_dpo']),
    ('sft', 'round1_sft_full_v1', 'round1_sft_full', sft['full_validation'])]:
    rows = read_tsv(P20/directory/f'{tag}.ranks.tsv')
    assert set(rows) == set(users)
    for u in users:
        assert int(rows[u]['baseline_rank']) == int(parent_rows[u]['raw_rank'])
        assert int(rows[u]['baseline_pcrf_rank']) == int(parent_rows[u]['pcrf_rank'])
    check_prediction(P20/directory/f'{tag}.predictions.jsonl', rows, 'raw_rank')
    for column, field, method in [('raw_rank','raw',name), ('pcrf_rank','pcrf',name+'_pcrf')]:
        a = ranks(rows, column, users)
        check_metrics(method, a, summary[field])
        all_ranks[method] = a
    assert np.array_equal(all_ranks[name] <= 50, all_ranks[name+'_pcrf'] <= 50)
    check_metrics(name+'/parent', all_ranks['original_gram'], summary['original_gram'])
    check_metrics(name+'/parent_pcrf', all_ranks['original_pcrf'], summary['original_gram_pcrf'])

gacr = read_json(P20/'gacr_v6_pcrf_toys_fast_v2/summary.json')
gacr_rows = read_tsv(P20/'gacr_v6_pcrf_toys_fast_v2/per_user.tsv')
assert set(gacr_rows) == set(users)
for name in gacr['metrics']:
    a = ranks(gacr_rows, name, users)
    if name in all_ranks:
        assert np.array_equal(a, all_ranks[name])
    all_ranks[name] = a
    check_metrics('gacr/'+name, a, gacr['metrics'][name])
for raw, reranked in [('c1','c1_pcrf'), ('gacr_v6','gacr_v6_pcrf')]:
    assert np.array_equal(all_ranks[raw] <= 50, all_ranks[reranked] <= 50)
for file, column in [('c1','c1'),('gacr','gacr_v6')]:
    check_prediction(P20/f'gacr_v6_pcrf_toys_fast_v2/{file}.predictions.jsonl', gacr_rows, column)
group_masks = {'all_validation': np.ones(len(users),dtype=bool),
               'outside_c1_training_users': np.array([u not in fit_users for u in users]),
               'inside_c1_training_users': np.array([u in fit_users for u in users]),
               'target_frequency_le5': np.array([u in tail_users for u in users]),
               'target_frequency_gt5': np.array([u not in tail_users for u in users])}
group_comparisons = {}
for group, mask in group_masks.items():
    for method, expected in gacr['groups'][group]['metrics'].items():
        check_metrics('gacr/'+group+'/'+method, all_ranks[method][mask], expected)
    group_comparisons[group] = comparison(all_ranks['gacr_v6_pcrf'][mask], all_ranks['original_pcrf'][mask])

selection = []
selection_indices = read_json(P20/'sprec_gram_toys_v1/selection_indices.json')
assert selection_indices == read_json(P20/'sdpo_gram_toys_v1/selection_indices.json')
selection_users = {list(validation)[i] for i in selection_indices}
for directory, tags in [('sprec_gram_toys_v1',[f'round{r}_{s}' for r in (1,2,3) for s in ('sft','dpo')]),
                        ('sdpo_gram_toys_v1',['epoch1_sdpo','epoch2_sdpo'])]:
    for tag in tags:
        summary = read_json(P20/directory/f'{tag}.metrics.json')
        rows = read_tsv(P20/directory/f'{tag}.ranks.tsv')
        selected = list(rows)
        assert len(selected) == 3000 and set(selected) <= set(users)
        assert set(selected) == selection_users
        check_prediction(P20/directory/f'{tag}.predictions.jsonl', rows, 'raw_rank')
        for u in selected:
            assert int(rows[u]['baseline_rank']) == int(parent_rows[u]['raw_rank'])
            assert int(rows[u]['baseline_pcrf_rank']) == int(parent_rows[u]['pcrf_rank'])
        for column, field in [('raw_rank','raw'),('pcrf_rank','pcrf')]:
            check_metrics(directory+'/'+tag+'/'+field, ranks(rows,column,selected),summary[field])
        assert np.array_equal(ranks(rows,'raw_rank',selected)<=50,ranks(rows,'pcrf_rank',selected)<=50)
        selection.append({'directory': directory, 'tag': tag, 'raw': summary['raw'], 'pcrf': summary['pcrf']})

auxiliary = {}
for directory in ['pcrf_component_audit_20260911_v2','personal_popularity_probe_v1/Toys']:
    summary = read_json(P20/directory/'summary.json')
    rows = read_tsv(P20/directory/'per_user.tsv')
    assert set(rows) == set(users)
    for method in ['gram','pcrf']:
        assert np.array_equal(ranks(rows,method,users), all_ranks['original_'+method])
    for group, expected_methods in summary['groups'].items():
        selected = [u for u in users if group == 'overall' or (u in tail_users) == (group == 'tail')]
        for method, expected in expected_methods.items():
            a = ranks(rows,method,selected)
            check_metrics(directory+'/'+group+'/'+method,a,expected)
            assert np.array_equal(a<=50,ranks(rows,'gram',selected)<=50)
    auxiliary[directory] = {'methods': len(summary['groups']['overall']), 'groups': 3, 'users': len(users)}

training = {}
for directory in ['sprec_gram_toys_v1','sdpo_gram_toys_v1']:
    path = P20/directory/'events.jsonl'
    digest(path)
    events = [json.loads(l) for l in path.read_text().splitlines()]
    training[directory] = [r for r in events if r['event'] in ['training_complete','mining_complete']]
config = read_json(ROOT/'experiment/phase20/gacr_v6_pcrf_toys_v1.json')
for path, expected in config['frozen_inputs'].items():
    assert digest(ROOT/path) == expected, path
assert digest(ROOT/sft['evaluation_checkpoint']) == sft['checkpoint_sha256']
receipt = read_json(P20/'sdpo_gram_toys_v1/stop_receipt.json')
assert digest(ROOT/receipt['last_checkpoint']) == receipt['last_checkpoint_sha256']

comparisons = {}
for a,b in [('sprec','original_gram'),('sprec_pcrf','original_pcrf'),('sft','original_gram'),
            ('sft_pcrf','original_pcrf'),('sprec','sft'),('sprec_pcrf','sft_pcrf'),
            ('gacr_v6','original_gram'),('gacr_v6','c1'),('gacr_v6_pcrf','original_gram'),
            ('gacr_v6_pcrf','original_pcrf'),('gacr_v6_pcrf','c1_pcrf'),('gacr_v6_pcrf','gacr_v6')]:
    comparisons[a+'_vs_'+b] = comparison(all_ranks[a],all_ranks[b])
status_files = [p for p in P20.rglob('status.json')] + [P20/'personal_popularity_probe_v1/queue_status.json']
statuses = {str(p.relative_to(ROOT)): read_json(p) for p in sorted(status_files)}
result = {'captured_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'verification_status':'ANALYZED', 'test_read':False, 'new_training_steps':0,'new_inference':False,
          'scope':'Existing ranks and predictions audited; no PCRF rescoring, retraining, GPU inference or new method arms.',
          'checks':checks,'checked_metric_cells':len(checks)*9,'prediction_integrity':predictions,
          'full_metrics':{m:metrics(a) for m,a in all_ranks.items()},'comparisons':comparisons,
          'gacr_group_comparisons':group_comparisons,'selection_metrics':selection,
          'training_complete_events':training,'auxiliary':auxiliary,'statuses':statuses,'input_sha256':hashes}
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (OUT/'full_validation_metrics.csv').open('w') as handle:
    writer=csv.DictWriter(handle,fieldnames=['method',*metrics(all_ranks['original_gram'])])
    writer.writeheader()
    for method,a in all_ranks.items(): writer.writerow({'method':method,**metrics(a)})
print(json.dumps({'metric_tables':len(checks),'metric_cells':len(checks)*9,
                  'prediction_rows':sum(r['users'] for r in predictions.values()),
                  'hashes':len(hashes),'output':str(OUT/'audit.json')},ensure_ascii=False))
