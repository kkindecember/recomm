"""Frozen-weight score versus probability pooling on full validation caches."""
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from fusion_pooling_core import checks,probability_pool,rank,top_k

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/phase21/full_validation_20260924_v1'
PRIOR=ROOT/'artifacts/phase21/full_opportunity_counts_20260925_v1'
OUT=ROOT/'artifacts/phase21/fusion_pooling_20260925_v1'
PROTOCOL=ROOT/'plan/第二十一阶段/21.10_分数融合与概率融合固定对照_20260925.md'
EXPERTS=('SASRec','ItemCF','semantic')


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def utility(r,k=10):
    r=np.asarray(r)
    return np.where(r<=k,1/np.log2(r+1),0.)


def metrics(r):
    r=np.asarray(r)
    return {f'{m}{k}':float((r<=k).mean() if m=='H' else utility(r,k).mean())
            for k in (10,50) for m in ('H','N')}


def comparison(new,old):
    new,old=np.asarray(new),np.asarray(old)
    dn=utility(new)-utility(old)
    dh=(new<=10).astype(float)-(old<=10).astype(float)
    rng=np.random.default_rng(20260925);bsn=[];bsh=[]
    for _ in range(20):
        ix=rng.integers(0,len(old),size=(100,len(old)))
        bsn.extend(dn[ix].mean(1).tolist());bsh.extend(dh[ix].mean(1).tolist())
    return dict(users=len(old),old=metrics(old),new=metrics(new),delta_N10=float(dn.mean()),
        relative_N10=float(dn.mean()/utility(old).mean()),delta_N10_ci95=np.quantile(bsn,[.025,.975]).tolist(),
        delta_H10=float(dh.mean()),delta_H10_ci95=np.quantile(bsh,[.025,.975]).tolist(),
        gained10=int(((new<=10)&(old>10)).sum()),lost10=int(((new>10)&(old<=10)).sum()),
        ndcg_improved_users=int((dn>1e-12).sum()),ndcg_worsened_users=int((dn< -1e-12).sum()))


def summarize(rows):
    old=np.asarray([r['old_rank'] for r in rows]);new=np.asarray([r['new_rank'] for r in rows])
    numerical=np.asarray([r['geometric64_rank'] for r in rows])
    eh=np.stack([[r[e+'_rank']<=10 for r in rows] for e in EXPERTS])
    opportunity=eh.any(0)&(old>10)
    out=comparison(new,old)
    out['opportunities']=dict(expert_correct_fusion_miss=int(opportunity.sum()),
        recovered=int((opportunity&(new<=10)).sum()),still_missed=int((opportunity&(new>10)).sum()),
        gained_without_expert_top10=int(((old>10)&(new<=10)&~eh.any(0)).sum()),by_expert={})
    for e,hit in zip(EXPERTS,eh):
        mask=hit&(old>10)
        out['opportunities']['by_expert'][e]=dict(available=int(mask.sum()),recovered=int((mask&(new<=10)).sum()))
    delta=utility(new)-utility(old)
    out['target_support_groups']={}
    for count in range(4):
        mask=eh.sum(0)==count
        out['target_support_groups'][str(count)]=dict(users=int(mask.sum()),
            gained10=int((mask&(new<=10)&(old>10)).sum()),lost10=int((mask&(new>10)&(old<=10)).sum()),
            contribution_to_whole_cohort_delta_N10=float(np.where(mask,delta,0).mean()),
            within_group_delta_N10=float(delta[mask].mean()) if mask.any() else None)
    out['new_top10_dominant_contribution']={e:dict(
        fraction=float(sum(r['dominant_'+e] for r in rows)/(10*len(rows))),
        mean_gate_weight=float(np.mean([r['weight_'+e] for r in rows])),
        mean_expert_peak_probability=float(np.mean([r['peak_'+e] for r in rows]))) for e in EXPERTS}
    out['numerical_control']=dict(target_rank_changed=int((numerical!=old).sum()),
        delta_N10=float((utility(numerical)-utility(old)).mean()),
        delta_H10=float(((numerical<=10).astype(float)-(old<=10).astype(float)).mean()))
    return out


def run_domain(domain,reference,manifest):
    p=SOURCE/domain
    receipt=json.loads((p/'score_cache_manifest.json').read_text())
    names=['samples.json','experts.npy','fused.npy','gate_prediction.npz','score_cache_manifest.json','ranks.npz']
    before={name:(p/name).stat() for name in names}
    for name in names:
        digest=sha(p/name);manifest[str((p/name).relative_to(ROOT))]=digest
        if name in receipt['sha256']:assert digest==receipt['sha256'][name],(domain,name)
    assert sha(SOURCE/'manifest.json')==receipt['input_manifest_sha256']
    samples=json.loads((p/'samples.json').read_text())
    experts=np.load(p/'experts.npy',mmap_mode='r',allow_pickle=False)
    old_scores=np.load(p/'fused.npy',mmap_mode='r',allow_pickle=False)
    with np.load(p/'gate_prediction.npz',allow_pickle=False) as a:weights,penalties=a['weights'],a['penalty']
    with np.load(p/'ranks.npz',allow_pickle=False) as a:users=a['users'];existing=a['previously_evaluated']
    assert len(samples)==experts.shape[0]==old_scores.shape[0]==len(weights)==len(users)
    rows=[];max_reconstruction_error=0.;max_weight_sum_error=0.
    for i,s in enumerate(samples):
        user=s['user_id'];target=s['target']
        assert user==users[i] and bool(s['previously_evaluated'])==bool(existing[i])
        ref=reference[(domain,user)];assert target==int(ref['target'])
        mask=np.zeros(old_scores.shape[1],bool);mask[np.asarray(s['full_history'],dtype=int)-1]=True
        eligible=np.ones(old_scores.shape[1],bool);eligible[np.asarray(s['history'][-20:],dtype=int)-1]=False
        assert s['history'][-20:]==s['full_history'][-20:]
        reconstructed=(weights[i,:,None]*experts[i]).sum(0)-penalties[i]*mask
        err=float(np.max(np.abs(reconstructed-old_scores[i])));max_reconstruction_error=max(err,max_reconstruction_error)
        assert err<=2e-6,(domain,i,err)
        max_weight_sum_error=max(max_weight_sum_error,abs(float(weights[i].astype(float).sum())-1))
        assert abs(float(weights[i].astype(float).sum())-1)<1e-6
        new,contributions,peaks=probability_pool(experts[i],weights[i],mask,penalties[i])
        old_rank=rank(old_scores[i],target,eligible)
        assert old_rank==int(ref['LIME'])
        score64=(weights[i].astype(float)[:,None]*experts[i].astype(float)).sum(0)-float(penalties[i])*mask
        new_rank=rank(new,target,eligible)
        new10=top_k(new,eligible)
        old10=top_k(old_scores[i],eligible)
        assert len(new10)==len(set(new10))==10 and not(set(new10)&set(s['history'][-20:]))
        if i in (0,len(samples)//2,len(samples)-1):
            ids=np.flatnonzero(eligible)+1
            for scores,rr,ten in [(new,new_rank,new10),(old_scores[i],old_rank,old10)]:
                ordered=ids[np.lexsort((ids,-scores[ids-1]))]
                np.testing.assert_array_equal(ordered[:10],ten)
                found=np.flatnonzero(ordered==target)
                assert rr==(int(found[0]+1) if len(found) else 2**30)
        dominant=np.bincount(contributions[:,new10-1].argmax(0),minlength=3)
        row=dict(domain=domain,user_id=user,target=target,previously_evaluated=bool(existing[i]),
            training_holdout_role=s.get('training_holdout_role'),old_rank=old_rank,new_rank=new_rank,
            geometric64_rank=rank(score64,target,eligible),old_target_score=float(old_scores[i,target-1]),
            new_target_score=float(new[target-1]),old10_boundary_item=int(old10[-1]),
            new10_boundary_item=int(new10[-1]),top10_overlap=len(set(old10)&set(new10)))
        for e,k in zip(EXPERTS,range(3)):
            row[e+'_rank']=int(ref[e]);row['weight_'+e]=float(weights[i,k])
            row['peak_'+e]=float(peaks[k]);row['dominant_'+e]=int(dominant[k])
        rows.append(row)
        if (i+1)%2048==0:
            print(domain,i+1,'/',len(samples),flush=True)
            write(OUT/'status.json',dict(state='RUNNING',domain=domain,stage='fixed_pooling',completed=i+1,total=len(samples),pid=os.getpid()))
    groups={'all':rows,'previous512':[r for r in rows if r['previously_evaluated']],
            'remaining_validation_events':[r for r in rows if not r['previously_evaluated']]}
    assert len(groups['previous512'])==512
    result={name:summarize(part) for name,part in groups.items()}
    result['verification']=dict(source_score_hashes_match=True,all_original_ranks_match=True,
        max_fp32_reconstruction_error=max_reconstruction_error,max_weight_sum_error=max_weight_sum_error,
        full_sort_rank_and_top10_checks=True,recent_history_excluded=True)
    baseline=json.loads((PRIOR/'summary.json').read_text())['domains'][domain]
    for name,part in groups.items():
        for metric in ('H10','N10'):
            assert abs(result[name]['old'][metric]-baseline[name]['metrics']['LIME'][metric])<1e-12
    for name in names:
        a=(p/name).stat();b=before[name]
        assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
    write(OUT/(domain+'_summary.json'),result)
    with (OUT/(domain+'_per_user.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def main():
    started=time.monotonic();OUT.mkdir(exist_ok=False)
    (OUT/'protocol.md').write_text(PROTOCOL.read_text())
    files=[Path(__file__),Path(__file__).with_name('fusion_pooling_core.py'),PROTOCOL,PRIOR/'per_user.csv']
    manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
    for path in files[:2]:(OUT/path.name).write_text(path.read_text())
    write(OUT/'status.json',dict(state='RUNNING',stage='numerical_checks',pid=os.getpid(),optimizer_steps=0,gpu_inferences=0,test_read=False))
    try:
        numerical_checks=checks();write(OUT/'numerical_checks.json',numerical_checks)
        with (PRIOR/'per_user.csv').open() as f:reference={(r['domain'],r['user_id']):r for r in csv.DictReader(f)}
        results={}
        for domain in ('Toys','Beauty'):
            write(OUT/'status.json',dict(state='RUNNING',stage='source_hashes',domain=domain,pid=os.getpid()))
            results[domain]=run_domain(domain,reference,manifest)
        primary={d:r['remaining_validation_events'] for d,r in results.items()}
        gate=all(r['delta_N10']>0 and r['delta_N10_ci95'][0]>0 for r in primary.values())
        summary=dict(state='COMPLETED',domains=results,seconds=time.monotonic()-started,
            users=41775,optimizer_steps=0,gpu_inferences=0,test_read=False,
            actual_new_GRAM_evaluation=False,exploratory=True,independent_confirmation=False,
            dual_domain_primary_screen_passed=gate,automatic_gpu_launch=False,numerical_checks=numerical_checks)
        write(OUT/'manifest.json',manifest);write(OUT/'summary.json',summary)
        write(OUT/'status.json',{k:v for k,v in summary.items() if k not in ('domains','numerical_checks')})
        print(json.dumps({k:v for k,v in summary.items() if k not in ('domains','numerical_checks')},ensure_ascii=False),flush=True)
    except BaseException as error:
        (OUT/'failure.txt').write_text(traceback.format_exc())
        write(OUT/'status.json',dict(state='FAILED',error=repr(error),pid=os.getpid()))
        raise


if __name__=='__main__':main()
