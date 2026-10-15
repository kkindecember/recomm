"""Count full-validation expert opportunities without training or rule search."""
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/phase21/full_validation_20260924_v1'
OUT=ROOT/'artifacts/phase21/full_opportunity_counts_20260925_v1'
EXPERTS=('SASRec','ItemCF','semantic')
ABSENT=2**30


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def rank(score,target,eligible):
    if not eligible[target-1]:return ABSENT
    value=score[target-1]
    return int(1+np.count_nonzero(eligible&(score>value))+
               np.count_nonzero(eligible[:target-1]&(score[:target-1]==value)))


def summarize(rows):
    r={name:np.asarray([row[name] for row in rows]) for name in (*EXPERTS,'LIME','GRAM','late','guided')}
    h=np.stack([r[e]<=10 for e in EXPERTS]);fusion=r['LIME']<=10
    opportunity=(r['GRAM']>10)&h.any(0);pending=opportunity&(r['guided']>10)
    out=dict(users=len(rows),ineligible_targets=sum(x['ineligible_target'] for x in rows),
             expert_hit_fusion_miss=int((h.any(0)&~fusion).sum()),
             GRAM_miss_expert_hit=int(opportunity.sum()),
             opportunities_realized=int((opportunity&(r['guided']<=10)).sum()),
             opportunities_unrealized=int(pending.sum()),
             pending_fusion_outside10=int((pending&~fusion).sum()),
             pending_fusion_inside10_guided11_50=int((pending&fusion&(r['guided']<=50)).sum()),
             pending_fusion_inside10_guided_outside50=int((pending&fusion&(r['guided']>50)).sum()),
             metrics={name:dict(H10=float((values<=10).mean()),
                 N10=float(np.where(values<=10,1/np.log2(values+1),0.).mean())) for name,values in r.items()},experts={})
    for e,hits in zip(EXPERTS,h):
        out['experts'][e]=dict(hit10=int(hits.sum()),hit_fusion_miss=int((hits&~fusion).sum()),
            hit_fusion_miss_guided_miss=int((hits&~fusion&(r['guided']>10)).sum()),
            GRAM_miss_expert_hit=int(((r['GRAM']>10)&hits).sum()),
            exclusive_expert_hit_fusion_miss=int((hits&(h.sum(0)==1)&~fusion).sum()))
    assert out['opportunities_unrealized']==sum(out[k] for k in (
        'pending_fusion_outside10','pending_fusion_inside10_guided11_50','pending_fusion_inside10_guided_outside50'))
    return out


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    (OUT/'source.py').write_text(Path(__file__).read_text())
    protocol='''# 全量样本量与机会审计

沿用21.7固定的Top10/Top50定义，只扩大既有缓存的统计范围。
不拟合、不调阈值、不运行新模型，也不据此更换机制。
全量/原512/其余验证事件分别统计；其余事件不自动称为独立test或用户外泛化。
目标在最近20条历史中时保留于总分母，专家排名记为不可推荐；不剔除。
对每个用户只读缓存专家分数、融合分数和原生成排名。
固定先/中/末行完整排序复核全目录目标排名，原512逐用户核对旧审计。
输出排名CSV、摘要、源文件大小/时间及原有评分哈希清单。
两个CPU线程，15分钟硬超时，原始数据只读。用户质疑小案例依据，本轮只查扩大后的有效案例数。
'''
    (OUT/'protocol.md').write_text(protocol)
    write(OUT/'status.json',dict(state='RUNNING',pid=os.getpid(),test_read=False,optimizer_steps=0,gpu_inferences=0))
    with (ROOT/'artifacts/phase21/expert_signal_loss_20260924_v1/per_user.csv').open() as f:
        previous={(r['domain'],r['user_id']):r for r in csv.DictReader(f) if r['role']=='development'}
    domains={};all_rows=[];source_files={}
    try:
        for domain in ('Toys','Beauty'):
            p=SOURCE/domain
            names=['experts.npy','fused.npy','samples.json','ranks.npz','score_cache_manifest.json']
            before={name:(p/name).stat() for name in names}
            samples=json.loads((p/'samples.json').read_text())
            experts=np.load(p/'experts.npy',mmap_mode='r',allow_pickle=False)
            fused=np.load(p/'fused.npy',mmap_mode='r',allow_pickle=False)
            with np.load(p/'ranks.npz',allow_pickle=False) as z:cache={k:z[k] for k in z.files}
            assert experts.shape[0]==len(samples)==fused.shape[0]==len(cache['users'])
            rows=[]
            for i,s in enumerate(samples):
                user=s['user_id'];target=s['target'];assert user==cache['users'][i]
                assert bool(s['previously_evaluated'])==bool(cache['previously_evaluated'][i])
                assert s['history'][-20:]==s['full_history'][-20:]
                eligible=np.ones(fused.shape[1],bool)
                eligible[np.asarray(s['history'][-20:],dtype=int)-1]=False
                row=dict(domain=domain,user_id=user,target=target,previously_evaluated=bool(s['previously_evaluated']),
                         training_holdout_role=s.get('training_holdout_role'),ineligible_target=not bool(eligible[target-1]))
                for name,score in [*zip(EXPERTS,experts[i]),('LIME',fused[i])]:
                    row[name]=rank(score,target,eligible)
                    if i in (0,len(samples)//2,len(samples)-1):
                        ids=np.flatnonzero(eligible)+1
                        ids=ids[np.lexsort((ids,-score[ids-1]))]
                        expected=np.flatnonzero(ids==target)
                        assert row[name]==(int(expected[0]+1) if len(expected) else ABSENT)
                for name in ('GRAM','late','guided'):row[name]=int(cache[name][i])
                if row['previously_evaluated']:
                    ref=previous[(domain,user)];assert target==int(ref['target'])
                    for name in (*EXPERTS,'LIME','GRAM','late','guided'):
                        old={'late':'fixed_late','guided':'fixed_decode'}.get(name,name)
                        a,b=row[name],int(ref[old])
                        assert a==b or (name in ('GRAM','late','guided') and a>50 and b>50)
                rows.append(row)
                if (i+1)%2048==0:
                    print(domain,i+1,'/',len(samples),flush=True)
                    write(OUT/'status.json',dict(state='RUNNING',domain=domain,completed=i+1,total=len(samples),pid=os.getpid()))
            cohorts={'all':rows,'previous512':[r for r in rows if r['previously_evaluated']],
                     'remaining_validation_events':[r for r in rows if not r['previously_evaluated']]}
            assert len(cohorts['previous512'])==512
            domains[domain]={name:summarize(part) for name,part in cohorts.items()}
            assert domains[domain]['previous512']['experts']['semantic']['hit_fusion_miss']==13
            all_rows.extend(rows)
            for name in names:
                after=(p/name).stat();initial=before[name]
                assert (initial.st_size,initial.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                source_files[str((p/name).relative_to(ROOT))]=dict(bytes=after.st_size,mtime_ns=after.st_mtime_ns)
            source_files[domain+'_original_score_manifest']=json.loads((p/'score_cache_manifest.json').read_text())
        with (OUT/'per_user.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
        write(OUT/'source_files.json',source_files)
        result=dict(state='COMPLETED',domains=domains,users=len(all_rows),seconds=time.monotonic()-start,
            optimizer_steps=0,gpu_inferences=0,test_read=False,exploratory=True,
            previous512_all_ranks_verified=True,full_sort_rank_spot_checks_passed=True,
            source_unmodified=True,rule_effects_evaluated=False)
        write(OUT/'summary.json',result)
        write(OUT/'status.json',{k:v for k,v in result.items() if k!='domains'})
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as error:
        (OUT/'failure.txt').write_text(traceback.format_exc())
        write(OUT/'status.json',dict(state='FAILED',error=repr(error)))
        raise


if __name__=='__main__':main()
