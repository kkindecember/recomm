"""Read-only, fixed-rule CPU feasibility probe; no model training or test reads."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path('/mnt/18T/jiangtangyunzhi/projects/recomm')
OUT = Path('/tmp/recomm_case_memory_probe_v1')
OUT.mkdir(exist_ok=True)
N = 2000
K = 20
SALT = 'recomm_case_memory_probe_v1:'
CONFIG = {
    'cohort': 'first 2000 validation users sorted by SHA256(salt + user_id)',
    'salt': SALT, 'candidate_budget': K,
    'case_retrieval': 'cosine of training-prefix TF-IDF histories, recency 1/(distance+1), last20; take20 distinct next items',
    'memory_exclusion': 'exclude all cases belonging to the query user',
    'controls': ['uniform random training cases', 'last-item training transition frequency', 'recency-weighted reciprocal-rank sum of static SASRec neighbors'],
    'metric': 'target coverage in retrieved20, not trained recommendation performance',
    'test_read': False, 'hyperparameter_search': False,
}
(OUT/'protocol.json').write_text(json.dumps(CONFIG, indent=2))

def read_jsonl(p):
    with p.open() as f:
        return [json.loads(line) for line in f]

def sparse_histories(examples, nitems, idf=None):
    indptr, indices, values = [0], [], []
    for e in examples:
        weights = defaultdict(float)
        for distance, item in enumerate(reversed(e['history'][-20:])):
            weights[item] += 1.0 / (distance + 1)
        for item, w in sorted(weights.items()):
            indices.append(item)
            values.append(w)
        indptr.append(len(indices))
    mat = csr_matrix((np.asarray(values, dtype=np.float32), indices, indptr), shape=(len(examples), nitems))
    if idf is None:
        df = np.bincount(mat.indices, minlength=nitems)
        idf = (1 + np.log((len(examples) + 1)/(df + 1))).astype(np.float32)
    mat = mat.multiply(idf).tocsr()
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    mat = mat.multiply((1/np.maximum(norms, 1e-12))[:, None]).tocsr()
    return mat, idf

summaries = {}
for domain, rel in [('Toys', 'artifacts/phase18/diff_gram/prepared_v2_categories'), ('Beauty','artifacts/phase18/diff_gram/beauty/prepared_v1')]:
    started = time.monotonic()
    p = ROOT/rel
    train = read_jsonl(p/'train.jsonl')
    validation = read_jsonl(p/'validation.jsonl')
    cohort = sorted(validation, key=lambda e: hashlib.sha256((SALT+e['user_id']).encode()).hexdigest())[:N]
    catalog = json.loads((p/'catalog.json').read_text())
    asin_to_id = {a:i for i,a in enumerate(catalog['items'])}
    nitems = len(asin_to_id)
    with (ROOT/f'artifacts/phase9/p9s_multiseed/{domain}/seed2023/validation/per_user.tsv').open() as f:
        refs = {e['user_id']:e for e in csv.DictReader(f, delimiter='\t')}
    assert len(refs) == len(validation)
    assert all(len(e['history']) == int(refs[e['user_id']]['history_length']) for e in cohort)
    neighbors = {}
    with (ROOT/f'GRAM/rec_datasets/{domain}/similar_item_sasrec.txt').open() as f:
        next(f)
        for line in f:
            toks = line.split()
            if toks[0] in asin_to_id:
                neighbors[asin_to_id[toks[0]]] = [asin_to_id[a] for a in toks[1:] if a in asin_to_id]
    users = np.asarray([e['user_id'] for e in train])
    labels = np.asarray([e['target'] for e in train])
    transitions = defaultdict(Counter)
    own_transitions = defaultdict(Counter)
    for e in train:
        last = e['history'][-1]
        transitions[last][e['target']] += 1
        own_transitions[(e['user_id'],last)][e['target']] += 1
    matrix, idf = sparse_histories(train, nitems)
    queries, _ = sparse_histories(cohort, nitems, idf)
    scores = (queries @ matrix.T).tocsr()
    rows = []
    rng = np.random.default_rng(20260909)
    for j,q in enumerate(cohort):
        uid, gold = q['user_id'], q['target']
        ref = refs[uid]
        sr = scores.getrow(j)
        allowed = users[sr.indices] != uid
        ix, values = sr.indices[allowed], sr.data[allowed]
        ix = ix[np.lexsort((ix, -values))]
        selected, seen = [], set()
        for case in ix:
            item = int(labels[case])
            if item in seen:
                continue
            selected.append(int(case))
            seen.add(item)
            if len(selected) == K:
                break
        assert all(users[i] != uid for i in selected)
        memory = [int(labels[i]) for i in selected]
        random_items, random_seen = [], set()
        while len(random_items) < K:
            case = int(rng.integers(len(train)))
            item = int(labels[case])
            if users[case] != uid and item not in random_seen:
                random_seen.add(item)
                random_items.append(item)
        last = q['history'][-1]
        counts = transitions[last] - own_transitions[(uid,last)]
        transition = [item for item,count in sorted(counts.items(), key=lambda x:(-x[1],x[0]))[:K]]
        static_scores = defaultdict(float)
        for distance, item in enumerate(reversed(q['history'][-20:])):
            for rank, candidate in enumerate(neighbors.get(item, [])):
                static_scores[candidate] += 1/((distance+1)*(rank+1))
        static = sorted(static_scores, key=lambda c:(-static_scores[c],c))[:K]
        candidates = {'history_case':memory, 'random_case':random_items, 'last_transition':transition, 'static_cf_neighbor':static}
        record = {'user_id':uid, 'gold':gold, 'baseline_rank':int(ref['baseline_rank']), 'pcrf_rank':int(ref['pcrf_rank']), 'case_indices':selected}
        for method, items in candidates.items():
            assert len(items) == len(set(items)) <= K
            record[method] = {'rank':items.index(gold)+1 if gold in items else K+1, 'count':len(items), 'items':items}
        rows.append(record)
    summary = {'users':len(rows), 'training_cases':len(train), 'baseline_h10_count':sum(e['baseline_rank']<=10 for e in rows), 'baseline_h50_count':sum(e['baseline_rank']<=50 for e in rows), 'pcrf_h10_count':sum(e['pcrf_rank']<=10 for e in rows), 'median_history':float(np.median([len(q['history']) for q in cohort])), 'methods':{}}
    for method in ['history_case','random_case','last_transition','static_cf_neighbor']:
        hits = [e for e in rows if e[method]['rank']<=K]
        summary['methods'][method] = {'coverage20_count':len(hits), 'coverage20':len(hits)/len(rows), 'outside_gram50_count':sum(e['baseline_rank']>50 for e in hits), 'pcrf10_miss_count':sum(e['pcrf_rank']>10 for e in hits), 'average_candidate_count':float(np.mean([e[method]['count'] for e in rows]))}
    summary['history_case_hits_missed_by_both_controls'] = sum(e['history_case']['rank']<=K and e['last_transition']['rank']>K and e['static_cf_neighbor']['rank']>K for e in rows)
    summary['outside_gram50_history_case_hits_missed_by_both_controls'] = sum(e['baseline_rank']>50 and e['history_case']['rank']<=K and e['last_transition']['rank']>K and e['static_cf_neighbor']['rank']>K for e in rows)
    summary['seconds'] = round(time.monotonic()-started,2)
    summary['train_sha256'] = hashlib.sha256((p/'train.jsonl').read_bytes()).hexdigest()
    summary['validation_sha256'] = hashlib.sha256((p/'validation.jsonl').read_bytes()).hexdigest()
    summary['cohort_sha256'] = hashlib.sha256('\n'.join(e['user_id'] for e in rows).encode()).hexdigest()
    summaries[domain] = summary
    with (OUT/f'{domain}_per_user.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
    print(domain, json.dumps(summary), flush=True)
(OUT/'summary.json').write_text(json.dumps(summaries, indent=2))
