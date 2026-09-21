"""Independent set-based support and pair-count rank reconstruction."""
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'artifacts/phase20/candidate_transition_audit_20260917'
SCREEN = ROOT / 'artifacts/phase20/direct_transition_screen_20260917'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def score_metric(r, mask):
    selected = r[mask]
    hits = selected <= 10
    return {'users': len(selected), 'N10': float(np.sum(1/np.log2(selected[hits]+1))/len(selected)),
            'H10': float(hits.sum()/len(selected)), 'H50': float(np.count_nonzero(selected <= 50)/len(selected))}


def check_values(saved, actual):
    global CELLS
    for k, value in saved.items():
        assert abs(value - actual[k]) < 1e-12, (k, value, actual[k])
        CELLS += 1


CELLS = 0


def main():
    started = time.monotonic()
    a = json.loads((AUDIT/'summary.json').read_text())
    s = json.loads((SCREEN/'summary.json').read_text())
    hashes = 0
    for folder, obj in [(AUDIT,a),(SCREEN,s)]:
        for key, value in obj['input_sha256'].items():
            assert sha(ROOT/key) == value, key
            hashes += 1
        for key, value in obj['output_sha256'].items():
            assert sha(folder/key) == value, key
            hashes += 1
    rank_values = 0
    category_values = 0
    for d, prep in [('Toys','artifacts/phase18/diff_gram/prepared_v2_categories'),
                    ('Beauty','artifacts/phase18/diff_gram/beauty/prepared_v1')]:
        # Rebuild full user training sequences; derive edges by adjacent zip.
        seqs = {}
        with (ROOT/prep/'train.jsonl').open() as f:
            for line in f:
                r = json.loads(line)
                u = r['user_id']
                if u not in seqs:
                    seqs[u] = list(r['history'])
                assert seqs[u][-len(r['history']):] == r['history']
                seqs[u].append(r['target'])
        forward = defaultdict(set)
        for seq in seqs.values():
            for x,y in zip(seq,seq[1:]):
                forward[x].add(y)
        with (ROOT/prep/'validation.jsonl').open() as f:
            rows = {r['user_id']:r for r in map(json.loads,f)}
        with np.load(AUDIT/f'{d}.per_user.npz') as p:
            users = p['users'].copy()
            categories = p['category'].copy()
            for u, category, target in zip(users,categories,p['targets']):
                h = rows[u]['history'][-20:]
                one = set().union(*(forward[x] for x in h))
                if target in forward[h[-1]]:
                    expected = 0
                elif target in one:
                    expected = 1
                else:
                    two = set().union(*(forward[x] for x in one))
                    expected = 2 if target in two else 3
                assert category == expected
                category_values += 1
            for group, cell in a['domains'][d]['groups'].items():
                mask = np.ones(len(users),bool) if group == 'all' else categories == a['groups'].index(group)
                assert int(mask.sum()) == cell['users']
                assert int(((p['positions'] >= 0) & mask).sum()) == cell['candidate_covered']
                for label, key in [('baseline','baseline_ranks'),('raw','raw_ranks'),('cf','cf_ranks')]:
                    for saved,ranks in zip(cell[label],p[key]):
                        check_values(saved,score_metric(ranks,mask))
                for k, entry in cell['outside_candidate_cf'].items():
                    hit = p['cf_ranks'] <= int(k)
                    eligible = mask & (p['positions'] < 0)
                    assert entry['per_cf_seed'] == np.sum(hit & eligible,axis=1).tolist()
                    assert entry['any_cf_oracle'] == int(np.count_nonzero(hit.any(0)&eligible))
                    assert entry['all_cf'] == int(np.count_nonzero(hit.all(0)&eligible))
        with np.load(ROOT/f'artifacts/phase20/future_factor_screen_20260917/{d}.audit_validation_cache.npz') as p:
            cf = p['cf_mean'].copy()
        for seed in s['seeds']:
            with np.load(SCREEN/f'{d}.{seed}.npz') as p, np.load(ROOT/f'artifacts/phase20/talos_candidate_screen_20260917/{d}.{seed}.predictions.npz') as old:
                names = old['raw_names'].tolist()
                g = old['raw_scores'][names.index('normalized.next_only')]
                for i,u in enumerate(users):
                    support = set().union(*(forward[x] for x in rows[u]['history'][-20:]))
                    assert p['masks'][i].tolist() == [int(x) in support for x in p['candidate_ids'][i]]
                    order = np.argsort(p['frequencies'][i],kind='stable')
                    for j in range(0,50,10):
                        idx = order[j:j+10]
                        assert p['masks'][i,idx].sum() == p['shuffled_masks'][i,idx].sum()
                score = [g+cf,g,g+np.where(p['masks'],cf,0),g+np.sum(p['masks'],axis=1)[:,None]*cf/50,
                         g+np.where(p['shuffled_masks'],cf,0),old['raw_scores'][names.index('softmax')]+cf,
                         old['raw_scores'][names.index('negative_softmax')]+cf]
                pos = p['positions']
                covered = pos >= 0
                groups = {'all':np.ones(len(users),bool),'tail':p['tail'],'non_fit':~p['in_fit']}
                groups.update({f'relation_{i}':categories == i for i in range(4)})
                for arm, values, saved_rank in zip(s['arms'],score,p['ranks']):
                    target_score = values[np.arange(len(users)),np.maximum(pos,0)]
                    larger = values > target_score[:,None]
                    tied_before = (values == target_score[:,None]) & (np.arange(50)[None,:] < pos[:,None])
                    ranks = np.where(covered,1+np.count_nonzero(larger | tied_before,axis=1),51)
                    assert np.array_equal(ranks,saved_rank), arm
                    rank_values += len(ranks)
                    for gn,gm in groups.items():
                        cell = s['domains'][d]['seeds'][str(seed)]
                        check_values(cell['metrics'][arm][gn],score_metric(ranks,gm))
                        b = p['ranks'][0]
                        rm,bm = score_metric(ranks,gm),score_metric(b,gm)
                        gain = np.count_nonzero(gm & (ranks<=10) & (b>10))
                        loss = np.count_nonzero(gm & (ranks>10) & (b<=10))
                        check_values(cell['comparisons'][arm][gn],dict(delta_N10=rm['N10']-bm['N10'],
                            relative_N10=rm['N10']/bm['N10']-1,delta_H10=rm['H10']-bm['H10'],
                            gained=int(gain),lost=int(loss),hit_net=int(gain-loss)))
    result = dict(state='VERIFIED_NUMERICAL_REPLAY', scope='Hashes, graph strata, candidate masks, ranks and metrics; not efficacy or novelty.',
                  updated_at=datetime.now().astimezone().isoformat(),hashes_checked=hashes,
                  category_values=category_values,rank_values=rank_values,metric_cells=CELLS,
                  elapsed_seconds=time.monotonic()-started,source_sha256=sha(Path(__file__)))
    for folder in [AUDIT,SCREEN]:
        (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        status = json.loads((folder/'status.json').read_text())
        status['verification'] = result['state']
        status['verification_file'] = str((folder/'verification.json').relative_to(ROOT))
        (folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
