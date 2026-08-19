import json, math, numpy as np
BASE='artifacts/phase13/explore/'
rng=np.random.default_rng(20260818)

def load(p):
    return [json.loads(l) for l in open(BASE+p)]

p0=load('v1_r2_toys_p0/predictions_validation.jsonl')
p6=load('v1_r2_toys_p6_candidate_portfolio/predictions_validation.jsonl')
p6map={d['user_id']:d for d in p6}

def dcg_hit(ranklist,target,k):
    for i,it in enumerate(ranklist[:k]):
        if it==target: return 1.0, 1.0/math.log2(i+2)
    return 0.0,0.0

# rebuild unconditional portfolio@N exactly as P6 defines: protect v0 top7,
# insert first N cold candidates (from resolver top50, not in v0 top7) at tail
def build_portfolio(d, p6d, n):
    v0=d['v0_top50']
    cands=p6d['portfolio_candidates'][:n]
    if not cands: return list(v0)
    prot=v0[:7]
    start=10-n            # @2 -> ranks 9,10 (idx8,9); @3 -> ranks 8,9,10
    out=list(prot)
    rest=[x for x in v0[7:] if x not in cands]
    while len(out)<start:
        out.append(rest.pop(0))
    out.extend(cands)
    seen=set(out)
    for x in v0[7:]+d['resolver_top50']:
        if x not in seen: out.append(x); seen.add(x)
    return out

methods={}
rows=[]
for d in p0:
    u=d['user_id']; t=d['target']; cold=d['is_cold']
    p6d=p6map[u]
    variants={
      'v0':d['v0_top50'],
      'resolver_only':d['resolver_top50'],
      'P6':p6d['p6_top50'],
      'portfolio@2':build_portfolio(d,p6d,2),
      'portfolio@3':build_portfolio(d,p6d,3),
    }
    r={'cold':cold}
    for name,rl in variants.items():
        h10,n10=dcg_hit(rl,t,10)
        h50,n50=dcg_hit(rl,t,50)
        r[name]=(h10,n10,h50)
    rows.append(r)

names=['v0','resolver_only','P6','portfolio@2','portfolio@3']
cold_idx=[i for i,r in enumerate(rows) if r['cold']]
warm_idx=[i for i,r in enumerate(rows) if not r['cold']]
print(f"n_all={len(rows)} n_cold={len(cold_idx)} n_warm={len(warm_idx)}\n")

def agg(idx,name,j):
    return float(np.mean([rows[i][name][j] for i in idx]))

print(f"{'method':16s} {'coldH@50':>10s} {'events':>7s} {'coldH@10':>10s} {'ev':>5s} {'coldN@10':>10s} {'warmN@10':>10s} {'allN@10':>10s}")
for nm in names:
    cH50=agg(cold_idx,nm,2); cH10=agg(cold_idx,nm,0); cN10=agg(cold_idx,nm,1)
    wN10=agg(warm_idx,nm,1); aN10=agg(range(len(rows)),nm,1)
    print(f"{nm:16s} {cH50:10.6f} {cH50*len(cold_idx):7.0f} {cH10:10.6f} {cH10*len(cold_idx):5.0f} {cN10:10.6f} {wN10:10.6f} {aN10:10.6f}")

# paired bootstrap vs v0
def boot(idx,nm,j,B=10000):
    a=np.array([rows[i][nm][j] for i in idx]); b=np.array([rows[i]['v0'][j] for i in idx])
    d=a-b
    obs=d.mean()
    bs=rng.choice(d,size=(B,len(d)),replace=True).mean(axis=1)
    return obs, np.percentile(bs,2.5), np.percentile(bs,97.5)

print("\n=== paired bootstrap 95% CI of (method - v0), 10000 resamples ===")
print(f"{'method':16s} {'metric':10s} {'obs diff':>12s} {'CI low':>12s} {'CI high':>12s}  verdict")
for nm in ['P6','portfolio@2','portfolio@3']:
    for label,idx,j in [('coldH@50',cold_idx,2),('coldH@10',cold_idx,0),('coldN@10',cold_idx,1),
                        ('warmN@10',warm_idx,1),('allN@10',list(range(len(rows))),1)]:
        o,lo,hi=boot(idx,nm,j)
        v='PASS' if lo>0 else ('FAIL' if hi<0 else 'INCONCLUSIVE')
        print(f"{nm:16s} {label:10s} {o:+12.6f} {lo:+12.6f} {hi:+12.6f}  {v}")
    print()
