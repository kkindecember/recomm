"""Post-hoc common-problem diagnostic from existing predictions only.

No fitted model, new generation, alpha search, expert selection, or test access.
Conditional opportunities are descriptions using labels, not usable policies.
"""
import csv
import json
from pathlib import Path
import time
import numpy as np
from experiment.phase20.catalog_scope_core import metrics, contrast
from experiment.phase21.candidate_gate_screen import sha, write

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'artifacts/phase21/single_fusion_20261014_v1'
TEXT = ROOT/'artifacts/phase21/candidate_text_source_20261015_v1'
CF = ROOT/'artifacts/phase21/collaborative_relation_source_20261015_v1'
SOURCE = ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
OUT = ROOT/'artifacts/phase21/common_scoring_diagnosis_20261015_v1'


def read(path):
    return json.loads(path.read_text())


def utility(r):
    return np.where(r <= 10, 1/np.log2(r+1), 0.)


def rank(ids, scores, target, history):
    ids, scores = np.asarray(ids), np.asarray(scores)
    keep = ~np.isin(ids, history)
    order = ids[keep][np.lexsort((ids[keep], -scores[keep]))]
    loc = np.flatnonzero(order == target)
    return int(loc[0])+1 if len(loc) else 2**30


def comparison(a, b):
    result = contrast(a, b, repeats=2000, seed=20261015)
    delta = utility(a)-utility(b)
    result.update(users_N10_better=int((delta > 1e-14).sum()),
        users_N10_worse=int((delta < -1e-14).sum()), users_N10_tied=int((abs(delta) <= 1e-14).sum()))
    return result


def main():
    begin = time.monotonic()
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='RUNNING', training=False, gpu=False, test_read=False))
    results, hashes = {}, {}
    for domain in ('Toys', 'Beauty'):
        samples = read(OLD/domain/'samples.json')
        rows = [json.loads(s) for s in (CF/domain/'predictions.jsonl').read_text().splitlines()]
        item = np.load(OLD/domain/'validation/item.npy', mmap_mode='r')
        signals = np.load(OLD/domain/'signals.npz')['user']
        frequencies = np.load(SOURCE/domain/'frequencies.npy')
        saved = np.load(CF/domain/'ranks.npz')
        text_saved = np.load(TEXT/domain/'ranks.npz')
        for source, names in ((OLD, ('samples.json', 'validation/item.npy', 'signals.npz')),
                              (TEXT, ('ranks.npz', 'summary.json')),
                              (CF, ('predictions.jsonl', 'ranks.npz', 'summary.json')),
                              (SOURCE, ('frequencies.npy',))):
            for name in names:
                path = source/domain/name
                hashes[str(path.relative_to(ROOT))] = sha(path)
        assert len(rows) == len(samples) == 512
        np.testing.assert_array_equal(saved['users'], text_saved['users'])
        np.testing.assert_array_equal(saved['sas_text_guided'], text_saved['full_guided'])
        per_user = []
        for j, (s, pred) in enumerate(zip(samples, rows)):
            assert all(s[k] == pred[k] for k in ('user_id', 'history', 'target'))
            r = dict(user_id=s['user_id'], history_length=len(s['full_history']))
            for k in saved.files:
                if k == 'users':
                    continue
                actual = rank(pred[k]['ids'], pred[k]['scores'], s['target'], s['history'])
                assert actual == saved[k][j]
                r[k] = actual
            for k in ('sas_only', 'candidate_id', 'both_id'):
                for suffix in ('_guided', '_late'):
                    r[k+suffix] = int(text_saved[k+suffix][j])
            ids = pred['native']['ids']
            assert pred['sas_text_late']['ids'] == ids
            full_ids = np.arange(1, item.shape[1]+1)
            for name, scores in (('sas', item[j, :, 0]), ('text', item[j, :, 1]),
                                 ('fused', signals[j]), ('popularity', frequencies[1:])):
                r[name+'_native_set_rank'] = rank(ids, scores[np.asarray(ids)-1], s['target'], s['history'])
                r[name+'_catalog_rank'] = rank(full_ids, scores, s['target'], s['history'])
            per_user.append(r)
        with (OUT/(domain+'_per_user.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_user[0]))
            writer.writeheader(); writer.writerows(per_user)
        ranks = {k: np.asarray([r[k] for r in per_user]) for k in per_user[0] if k not in ('user_id', 'history_length')}
        native = ranks['native']
        pairs = [('sas_text_late', 'native'), ('sas_text_guided', 'native'), ('sas_text_guided', 'sas_text_late'),
                 ('sas_text_guided', 'sas_only_guided'), ('sas_text_guided', 'text_only_guided'),
                 ('sas_only_guided', 'native'), ('text_only_guided', 'native')]
        complementarity = {}
        for source, other in (('sas', 'text'), ('text', 'sas')):
            hit = ranks[source+'_catalog_rank'] <= 10
            exclusive = hit & (ranks[other+'_catalog_rank'] > 10)
            opportunity = exclusive & (native > 10)
            complementarity[source] = dict(catalog_hit10=int(hit.sum()), exclusive_hit10=int(exclusive.sum()),
                exclusive_hit10_native_miss10=int(opportunity.sum()),
                exclusive_opportunity_history_at_most20=sum(bool(opportunity[j]) and r['history_length'] <= 20 for j, r in enumerate(per_user)))
        group_masks = {'native_hit10': native <= 10, 'native_rank11_to50': (native > 10) & (native <= 50),
                       'native_missing50': native > 50}
        decomposition = {}
        for arm in ('sas_text_late', 'sas_text_guided', 'sas_only_guided', 'text_only_guided'):
            delta = utility(ranks[arm])-utility(native)
            decomposition[arm] = {k: dict(users=int(mask.sum()), target_hit10_after=int((ranks[arm][mask] <= 10).sum()),
                delta_N10_contribution_to_all512=float(delta[mask].sum()/512),
                positive_users=int((delta[mask] > 1e-14).sum()), negative_users=int((delta[mask] < -1e-14).sum()))
                for k, mask in group_masks.items()}
            assert abs(sum(r['delta_N10_contribution_to_all512'] for r in decomposition[arm].values())-delta.mean()) < 1e-14
        short = np.asarray([r['history_length'] <= 20 for r in per_user])
        results[domain] = dict(users=512, metrics={k: metrics(v) for k, v in ranks.items()},
            comparisons={a+'_vs_'+b: comparison(ranks[a], ranks[b]) for a, b in pairs},
            same_native_candidate_set_verified_users=512, decomposition=decomposition,
            source_complementarity=complementarity,
            history_at_most20=dict(n=int(short.sum()), metrics={k: metrics(v[short]) for k, v in ranks.items()}))
        print(json.dumps(dict(domain=domain, comparisons=results[domain]['comparisons'],
            source_complementarity=complementarity, decomposition=decomposition), ensure_ascii=False), flush=True)
    write(OUT/'summary.json', dict(domains=results, scope='post-hoc, previously reused development512/domain',
        limitations=['Describes correctable ordering, not a causal proof of a token-objective defect.',
                     'Autoregressive factorization alone does not make item likelihood incorrect.',
                     'Exclusive expert hits use target labels; not a deployable routing rule.',
                     'Current positive point estimates are not independent two-domain confirmation.',
                     'All added direct-score controls reported, including negative results.',
                     'No expert, alpha, feature, checkpoint or subset selected by these diagnostics.']))
    hashes[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    write(OUT/'manifest.json', dict(sha256=hashes))
    write(OUT/'interpretation_checks.json', dict(covered=11, total=11, status='CAUTION',
        items=['domains separate', 'user-level unit', 'selected reused development cohort disclosed',
               'label-defined subgroups descriptive only', 'counts and all512 denominator retained',
               'not independent replication', 'all arms and missing-target users retained',
               'all diagnostic controls retained', 'post-hoc questions disclosed',
               'no unique causal explanation claimed', 'no temporal or causal direction inferred']))
    write(OUT/'status.json', dict(state='COMPLETED', elapsed_seconds=time.monotonic()-begin,
        training=False, gpu=False, test_read=False, checked_users=1024, new_generation=False))


if __name__ == '__main__':
    main()
