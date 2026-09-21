"""Cost-bounded recovery with explicitly audited FP32 cache tolerance."""
import json
from pathlib import Path
import sys
import time

import numpy as np

from experiment.phase20 import lime_decode_screen as base
from experiment.phase20 import lime_decode_screen_fast as fast
from experiment.phase20.lime_decode_screen_core import metrics,contrast

base.OUT=fast.OLD/'bounded_v3'
fast.OUT=base.OUT
fast.MODULE='experiment.phase20.lime_decode_screen_pilot'
OUT=base.OUT


class PilotWorker(fast.FastWorker):
    def smoke(self):
        import torch
        self.stage='bounded_smoke';self.status();checks=[];timings=[]
        for j in range(3):
            ctx=self.context(j)
            self.model._diff_cross_cache_reuse=False
            native=self.decode(ctx,native_entry=True,record=False)
            self.model._diff_cross_cache_reuse=True
            zero=self.decode(ctx);timings.append(self.seconds[-1])
            assert zero['paths']==native['paths'], 'Cache changed beam paths/order in engineering check'
            np.testing.assert_allclose(zero['scores'],native['scores'],atol=5e-6,rtol=5e-6)
            q=self.index.conditional_logs(self.signals['lime'][j])
            self.decode(ctx,q,.3);timings.append(self.seconds[-1])
            zero2=self.decode(ctx,q,0.);timings.append(self.seconds[-1])
            assert zero2['paths']==zero['paths']
            np.testing.assert_array_equal(zero2['scores'],zero['scores'])
            checks.append(dict(user_id=self.samples[j]['user_id'],cross_cache_paths_exact=True,
                max_cross_cache_score_error=float(np.max(np.abs(zero['scores']-native['scores']))),
                zero_strength_exact=True,path_scores_reconstructed=True))
            self.status(completed_smoke_users=j+1,optimized_decode_seconds=timings)
            del ctx,native,zero,zero2
        longest=max(range(len(self.data)),key=lambda i:len(self.data.records[i]['history']))
        ctx=self.context(0,longest);self.decode(ctx);timings.append(self.seconds[-1]);del ctx
        assert self.digest()==self.initial_digest
        # Engineering-cost-only selection before any tuning/evaluation labels.
        self.nt,self.ne=0,0
        for nt,ne in ((128,512),(64,256),(32,128)):
            estimate=max(timings)*(5*nt+3*ne)+time.monotonic()-self.started
            if estimate<self.config['worker_timeout_seconds']-600:
                self.nt,self.ne=nt,ne;break
        base.write(self.folder/'smoke.json',dict(checks=checks,max_history_checked=True,
            optimized_decode_seconds=timings,all_decode_seconds=self.seconds,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,initial_digest=self.initial_digest))
        if not self.nt:raise TimeoutError('Even smallest fixed pilot exceeds two-hour budget')
        base.write(self.folder/'budget.json',dict(tuning_users=self.nt,evaluation_users=self.ne,
            projected_formal_seconds=max(timings)*(5*self.nt+3*self.ne),
            selection_basis='max optimized engineering time only; no outcome used',selected_at=base.stamp()))
        self.status(tuning_users=self.nt,evaluation_users=self.ne,projected_formal_seconds=estimate)

    def run(self):
        self.smoke();self.stage='tuning';self.status();tuning=[]
        with (self.folder/'tuning.jsonl').open('x') as handle:
            for k,j in enumerate(range(3,3+self.nt),1):
                row=self.one(j,{s:base.ALPHAS for s in base.SIGNALS},'tuning');tuning.append(row)
                handle.write(json.dumps(row)+'\n');handle.flush()
                if k%4==0:self.status(completed_users=k,total_users=self.nt)
        choices={};table={}
        for s in base.SIGNALS:
            table[s]={str(a):metrics([r['arms']['GRAM' if a==0 else f'{s}_{a}']['rank'] for r in tuning]) for a in (0.,)+base.ALPHAS}
            choices[s]=max((0.,)+base.ALPHAS,key=lambda a:(table[s][str(a)]['N10'],-a))
        base.write(self.folder/'selection.json',dict(selected_alpha=choices,tuning=table,selected_at=base.stamp(),
            criterion='N10; ties choose smaller alpha; zero is original cached-GRAM fallback'))
        self.stage='evaluation';self.status(selection=choices);evaluation=[]
        with (self.folder/'evaluation.jsonl').open('x') as handle:
            for k,j in enumerate(range(131,131+self.ne),1):
                row=self.one(j,{s:[choices[s]] for s in base.SIGNALS},'evaluation');evaluation.append(row)
                handle.write(json.dumps(row)+'\n');handle.flush()
                if k%4==0:self.status(completed_users=k,total_users=self.ne,selection=choices)
        ranks={s:np.asarray([r['arms'][s]['rank'] for r in evaluation]) for s in evaluation[0]['arms']}
        comps={s+'_vs_GRAM':contrast(rr,ranks['GRAM']) for s,rr in ranks.items() if s!='GRAM'}
        comps.update({s+'_vs_late':contrast(ranks[s],ranks[s+'_late']) for s in base.SIGNALS})
        for other in ('sasrec','LIME_after_GRAM','LIME'):comps['lime_vs_'+other]=contrast(ranks['lime'],ranks[other])
        assert self.digest()==self.initial_digest
        for p,digest in self.loaded_source.items():assert base.sha(base.ROOT/p)==digest,('changed source',p)
        base.write(self.folder/'summary.json',dict(domain=self.domain,selected_alpha=choices,
            tuning_users=self.nt,evaluation_users=self.ne,metrics={s:metrics(rr) for s,rr in ranks.items()},comparisons=comps,
            elapsed_seconds=time.monotonic()-self.started,completed_decodes=self.completed,model_unchanged=True,
            test_read=False,development_diagnostic=True))
        self.stage='done';self.status('COMPLETED',selection=choices)


def verify():
    summaries={d:base.read(OUT/d/'summary.json') for d in base.DOMAINS}
    hashes=base.read(OUT/'manifest.json')['sha256']
    for p,digest in hashes.items():assert base.sha(base.ROOT/p)==digest,('changed input',p)
    signal=all(s['comparisons']['lime_vs_GRAM']['relative_N10']>=.01 and s['comparisons']['lime_vs_GRAM']['net_H10']>=0 for s in summaries.values())
    extra=all(s['comparisons']['lime_vs_late']['delta_N10']>=0 for s in summaries.values()) and any(s['comparisons']['lime_vs_late']['delta_N10']>0 for s in summaries.values())
    decision='PROMISING_DEVELOPMENT_SIGNAL' if signal and extra else 'PERFORMANCE_ONLY_SIGNAL' if signal else 'NO_CONTINUATION_SIGNAL'
    base.write(OUT/'summary.json',dict(decision=decision,domains=summaries,search_additional_value=extra,test_read=False,
        caveat='Small single-seed reused-development screen; FP32 cache tolerance disclosed; not a null-effect proof.'))
    from experiment.phase20 import verify_lime_decode_screen as independent
    old=sys.argv;sys.argv=['audit','--output',str(OUT.relative_to(base.ROOT))]
    try:independent.main()
    finally:sys.argv=old
    base.write(OUT/'verification.json',dict(all_input_hashes_unchanged=True,independent_metrics_audit=True,verified_at=base.stamp()))


def prepare():
    fast.prepare()
    p=OUT/'protocol.md'
    p.write_text(p.read_text()+'''\n\n## 成本与数值补充 v3（效果揭示前）\n\n缓存复用v2首个Toys工程例路径/顺序相同，但FP32分数最大差1.79e-6，因此严格逐位分数检查停止，未运行效果评价。v3明确将跨缓存实现分数容差设为atol=rtol=5e-6，仍要求三例全部50路径和顺序完全相同；同一缓存实现下的alpha=0继续逐位一致。该微小数值变化如实披露，不宣称所有用户与原实现逐位相同。\n\n在任何tuning/evaluation效果之前，按优化后工程decode最大耗时选择预算可容纳的最大一档：128/512、64/256、32/128人。tuning均取原tuning名单前n人，evaluation取原独立evaluation名单前n人，不重抽用户、不按目标选人。预测总成本加已用时间须低于两小时减10分钟余量。各域可因共享负载选不同档，完整披露；哪怕最小档也超预算则停止。所有信号、权重搜索、模型、原始样本排序和继续标准不变；缩小样本会降低判断精度，负向/零信号仅支持停止本配方扩展。\n''')
    m=base.read(OUT/'manifest.json')
    for path in (Path(__file__).resolve(),base.ROOT/'experiment/phase20/verify_lime_decode_screen.py',p):
        m['sha256'][str(path.relative_to(base.ROOT))]=base.sha(path)
    m['numeric_and_cost_revision']='Cross-cache FP32 tolerance 5e-6; same-mode zero exact; cohort tier chosen only on engineering cost before effects.'
    base.write(OUT/'manifest.json',m)


fast.FastWorker=PilotWorker
base.verify=verify


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
    else:fast.main()
