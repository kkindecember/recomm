"""Independent saved-output audit; reconstruct covariance using matrix products."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/pair_uncertainty_screen_20260917'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def norm(x):
    return (x-x.mean(-1, keepdims=True))/np.maximum(x.std(-1, keepdims=True), 1e-6)


def main():
    summary = json.loads((OUT/'summary.json').read_text())
    for p, expected in summary['input_sha256'].items(): assert sha(ROOT/p) == expected, p
    cfg = json.loads((ROOT/'experiment/phase20/gacr_fusion_train_v1.json').read_text())
    result = {}; checked = 0
    for domain, config in cfg['domains'].items():
        old = ROOT/f'artifacts/phase20/gacr_fusion_train_v1/{domain}'
        with np.load(OUT/f'{domain}.predictions.npz') as f: saved = {k:f[k] for k in f.files}
        with np.load(OUT/f'{domain}.cf_scores.npz') as f: cf = {k:f[k] for k in f.files}
        with np.load(old/'B1.validation_scores.npz') as f: b1 = {k:f[k] for k in f.files}
        np.testing.assert_array_equal(b1['users'], saved['users'])
        np.testing.assert_array_equal(b1['users'], cf['users'])
        np.testing.assert_array_equal(b1['candidate_ids'], cf['candidate_ids'])
        canon = np.argsort(-b1['residual_scores'], axis=1, kind='stable')
        np.testing.assert_array_equal(np.take_along_axis(cf['candidate_ids'], canon, 1), saved['candidate_ids'])
        arms = list(saved['arm_names'])
        with (old/'per_user.tsv').open() as f: ref={r['user_id']:int(r['B1']) for r in csv.DictReader(f, delimiter='\t')}
        np.testing.assert_array_equal(saved['ranks'][arms.index('current')], [ref[u] for u in saved['users']])
        metrics = {}
        for j, arm in enumerate(arms):
            # Count strictly greater scores and exact ties earlier in canonical order.
            scores=saved['scores'][j]; pos=saved['positions']; idx=np.maximum(pos, 0)
            gold=scores[np.arange(len(pos)), idx,None]
            rr=1+(scores>gold).sum(1)+((scores==gold)&(np.arange(50)[None]<idx[:,None])).sum(1)
            rr=np.where(pos>=0,rr,51)
            np.testing.assert_array_equal(rr,saved['ranks'][j])
            for group, mask in [('all',np.ones(len(pos),bool)),('tail',saved['tail_mask']),('non_tail',~saved['tail_mask'])]:
                expected=summary['domains'][domain]['groups'][group][arm]
                assert int(mask.sum()) == expected['users']
                for k in (5,10,20,50):
                    hit=rr[mask]<=k
                    for key,val in [(f'H{k}',hit.mean()),(f'N{k}',(hit/np.log2(rr[mask]+1)).mean())]:
                        assert abs(val-expected[key])<1e-14,(domain,arm,group,key)
                        checked+=1
            metrics[arm] = summary['domains'][domain]['groups']['all'][arm]
        # Independently reconstruct training-only item counts and current positions.
        counts = {}; initials = {}
        for line in (ROOT/config['validation']).with_name('train.jsonl').open():
            row=json.loads(line); t=row['target']; counts[t]=counts.get(t,0)+1
            if len(row['history'])==1: initials.setdefault(row['user_id'],row['history'][0])
        for t in initials.values(): counts[t]=counts.get(t,0)+1
        freq=np.vectorize(lambda t:counts.get(int(t),0))(saved['candidate_ids'])
        validation={r['user_id']:r for r in map(json.loads,(ROOT/config['validation']).open())}
        targets=np.asarray([validation[u]['target'] for u in saved['users']])
        eq=saved['candidate_ids']==targets[:,None]
        np.testing.assert_array_equal(saved['positions'],np.where(eq.any(1),eq.argmax(1),-1))
        np.testing.assert_array_equal(saved['tail_mask'],[counts.get(int(t),0)<=config['q1'] for t in targets])
        gz=norm(np.take_along_axis(b1['residual_scores'],canon,1))
        c=norm(norm(np.take_along_axis(cf['scores'],canon[None],2))-.5*norm(np.log1p(freq))[None])
        reliability=1-(freq[:,:10]<=config['q1']).mean(1)
        error=0.
        for start in range(0,len(gz),128):
            sl=slice(start,start+128); x=c[:,sl]; mu=x.mean(0); dev=x-mu
            cov=np.einsum('mbi,mbj->bij',dev,dev)/(len(x)-1)
            variance=np.diagonal(cov,axis1=1,axis2=2)
            pair_sd=np.sqrt(np.maximum(variance[:,:,None]+variance[:,None,:]-2*cov,0))
            delta=mu[:,:,None]-mu[:,None,:]
            correction=reliability[sl,None,None]*np.sign(delta)*np.maximum(np.abs(delta)-pair_sd,0)
            margin=gz[sl,:,None]-gz[sl,None,:]+correction
            pred=(1/(1+np.exp(-np.clip(margin,-60,60)))).sum(2)
            error=max(error,float(np.max(np.abs(pred-saved['scores'][arms.index('pair_cov'),sl]))))
            np.testing.assert_allclose(pred,saved['scores'][arms.index('pair_cov'),sl],atol=1e-11,rtol=0)
            np.testing.assert_allclose(gz[sl]+reliability[sl,None]*mu,saved['scores'][arms.index('mean'),sl],atol=1e-12,rtol=0)
        r0=saved['ranks'][arms.index('current')]; rm=saved['ranks'][arms.index('mean')]
        tail=summary['domains'][domain]['groups']['tail']
        result[domain]=dict(users=len(r0),covariance_score_max_error=error,
            mean_relative_N10=metrics['mean']['N10']/metrics['current']['N10']-1,
            mean_hit10_gained=int(((rm<=10)&(r0>10)).sum()),mean_hit10_lost=int(((rm>10)&(r0<=10)).sum()),
            mean_hit10_net=int((rm<=10).sum()-(r0<=10).sum()),
            mean_tail_relative_N10=tail['mean']['N10']/tail['current']['N10']-1)
    audit=dict(state='VERIFIED',input_hashes=len(summary['input_sha256']),metrics_checked=checked,
        source_sha256=sha(Path(__file__)),domains=result,training_steps=0,test_read=False)
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
