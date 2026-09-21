"""Independent saved-output, cohort and metric audit; no new fitting."""
import hashlib
import numpy as np
import torch
from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now


def load(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k] for k in z.files}


def sorted_ranks(scores, ids, targets):
    order = np.lexsort((ids, -scores), axis=1)
    match = (np.take_along_axis(ids, order, 1) == targets[:, None])
    match &= np.isfinite(np.take_along_axis(scores, order, 1))
    return np.where(match.any(1), match.argmax(1)+1, 2**30)


def metric(r, name):
    k = int(name[1:]); hit = r <= k
    return float(hit.sum()/len(r) if name[0] == 'H' else np.sum(1/np.log2(r[hit]+1))/len(r))


def main():
    folder = ROOT/'artifacts/phase20/lime_learnability_20260920_v1'
    result = read(folder/'summary.json'); c = read(folder/'config.json')
    assert result['state'] == 'COMPLETED' and result['optimizer_steps'] == 600
    inputs = dict(result['input_sha256']); outputs = dict(result['output_sha256'])
    for name, digest in {**inputs, **outputs}.items(): assert sha(ROOT/name) == digest, name
    fields = 0; rank_rows = 0; steps = 0; conclusions = {}
    for domain, summary in result['domains'].items():
        saved = load(folder/domain/'predictions.npz')
        train = load(ROOT/c['input']/domain/'train.npz')
        dev = load(ROOT/c['input']/domain/'development.npz')
        for key in ('users','targets'): np.testing.assert_array_equal(saved[key],dev[key])
        folds = np.array([int.from_bytes(hashlib.sha256(
            f"20260920|lime-learnability|{domain}|{u}".encode()).digest()[:8], 'big') % 5
            for u in dev['users']])
        np.testing.assert_array_equal(folds, saved['folds'])
        latest = {}
        for row in np.argsort(train['train_indices']): latest[str(train['users'][row])] = int(row)
        match = np.asarray([latest.get(str(u), -1) for u in dev['users']])
        rr = dict(zip(saved['arm_names'], saved['ranks']))
        for arm, score in [('B_historical','B_scores'),('C_next_interaction','C_scores')]:
            r = sorted_ranks(saved[score], dev['ids'], dev['targets'])
            np.testing.assert_array_equal(r, rr[arm]); rank_rows += len(r)
            for fold in range(5):
                cp = torch.load(folder/domain/f'fold{fold}.{arm}.pt',map_location='cpu')
                eval_ix = np.flatnonzero(folds == fold)
                ix = np.flatnonzero((folds != fold) & (match >= 0))
                source = train if arm == 'B_historical' else dev
                ix = match[ix] if arm == 'B_historical' else ix
                np.testing.assert_array_equal(cp['cohort_indices'], ix)
                ix = ix[source['positions'][ix] >= 0]
                np.testing.assert_array_equal(cp['fit_indices'], ix)
                np.testing.assert_array_equal(cp['fit_users'], source['users'][ix])
                np.testing.assert_array_equal(cp['evaluation_indices'], eval_ix)
                assert not set(cp['fit_users']) & set(dev['users'][eval_ix])
                assert len(cp['curve']) == 30; steps += len(cp['curve'])
        expected = dict(gram_lime=dev['gram_lime_ranks'],
            crossfit_gram_lime=dev['crossfit_gram_lime_ranks'],
            union_lime=sorted_ranks(np.where(dev['ids']>0,dev['prior'],-np.inf),dev['ids'],dev['targets']))
        expected.update({f'A_head{i}':r for i,r in enumerate(dev['A_ranks'])})
        for arm,r in expected.items(): np.testing.assert_array_equal(rr[arm], r)
        groups = dict(all=np.ones(len(folds),bool),low_frequency=dev['tail'],other=~dev['tail'],
            matched_cohort=match>=0,other_users=match<0,gram_covered=dev['target_in_gram'],
            lime_only=(dev['positions']>=0)&~dev['target_in_gram'])
        for g,mask in groups.items():
            np.testing.assert_array_equal(saved['mask_'+g], mask)
            for arm,r in rr.items():
                for name,value in summary['metrics'][arm][g].items():
                    assert abs(metric(r[mask],name)-value) < 1e-12; fields += 1
            for name,value in summary['metrics']['A_mean'][g].items():
                assert abs(np.mean([metric(r[mask],name) for r in dev['A_ranks']])-value) < 1e-12
                fields += 1
        a,b=rr['C_next_interaction'],rr['gram_lime']
        positive = sum(metric(a[folds==f],'N10') > metric(b[folds==f],'N10') for f in range(5))
        assert positive == summary['positive_folds']
        conclusions[domain] = dict(N10_relative_gain=metric(a,'N10')/metric(b,'N10')-1,
            H10_net_users=int((a<=10).sum()-(b<=10).sum()),positive_folds=positive,
            low_frequency_H10_net_users=int((a[dev['tail']]<=10).sum()-(b[dev['tail']]<=10).sum()),
            matched_cohort_N10={arm:metric(rr[arm][match>=0],'N10') for arm in ('C_next_interaction','gram_lime')},
            other_users_N10={arm:metric(rr[arm][match<0],'N10') for arm in ('C_next_interaction','gram_lime')})
    assert steps == 600 and rank_rows == 75358 and fields == 1008
    report = dict(state='VERIFIED',completed_at=now(),input_hashes=len(inputs),output_hashes=len(outputs),
        independent_new_arm_ranks=rank_rows,independent_metric_fields=fields,
        checked_models=20,checked_optimizer_steps=steps,cohort_isolation=True,
        conclusions=conclusions,new_training=False,test_read=False,
        source_sha256=sha(__file__))
    write(folder/'independent_verification.json',report)
    print(report)


if __name__ == '__main__': main()
