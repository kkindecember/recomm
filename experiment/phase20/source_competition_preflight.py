"""Real-data parity/profile only; no development model selection."""
import json
import time
import unittest
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now
from experiment.phase20.catalog_scope_core import ranks, topk
from experiment.phase20.lime_comparison_core import left_pad, semantic_users
from experiment.phase20.source_competition_core import ARMS, ResidualRanker, pair_masks, target_positions, row_losses
from experiment.phase20.source_competition_data import DataBuilder, arrays, LIME, OLD
from experiment.phase20.test_source_competition import SourceCompetitionTests


def main():
    config=read(ROOT/'experiment/phase20/source_competition_20260920_v2.json')
    torch.set_num_threads(4); threadpool_limits(4)
    out=ROOT/config['preflight_output']
    out.mkdir(exist_ok=True)
    assert not (ROOT/config['output']).exists(), 'Do not overwrite checks after formal launch'
    if (out/'checks.json').exists():
        previous=out/f'checks.previous.{time.time_ns()}.json'
        (out/'checks.json').rename(previous)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SourceCompetitionTests)
    checks=unittest.TextTestRunner(verbosity=1).run(suite)
    assert checks.wasSuccessful()
    reports={}; updated=0
    for domain in config['domains']:
        started=time.monotonic(); b=DataBuilder(domain,config,lambda p:p,lambda *a,**kw:None)
        cache=arrays(OLD/domain/'training_cache.npz')
        fit=np.flatnonzero(cache['fit']); chosen=set(fit[np.linspace(0,len(fit)-1,128,dtype=int)].tolist())
        metadata=[]; gi=[]
        with (OLD/domain/'train_candidates.jsonl').open() as f:
            for i,line in enumerate(f):
                if i in chosen:
                    r=json.loads(line)
                    metadata.append({k:r[k] for k in ('train_index','user_id','history','target')}); gi.append(i)
        histories=[b.history(r['user_id'],b.train_index['position'][r['train_index']]) for r in metadata]
        assert all(h[-20:]==r['history'] for h,r in zip(histories,metadata))
        tic=time.monotonic()
        with torch.no_grad(): sequential=b.sas.predict(torch.from_numpy(left_pad(histories)))[:,1:].numpy()
        raw=(sequential,np.stack([b.cf.score('',[b.source_ids[j-1] for j in h]) for h in histories]),
            semantic_users(histories,b.emb)@b.emb.T)
        a,_,_=b.build_rows(cache['candidate_ids'][gi],histories,raw)
        prefix_seconds=time.monotonic()-tic
        pos=target_positions(a['ids'],np.array([r['target'] for r in metadata])); keep=pos>=0
        pairs=pair_masks(a['prior'][keep],a['ids'][keep]>0,a['ing'][keep],pos[keep])
        x=a['x'][keep]; valid=a['ids'][keep]>0
        mu=x[valid].mean(0); std=np.maximum(x[valid].std(0),1e-5)
        x=(x-mu)/std; x[~valid]=0
        times=[]
        for arm in ARMS:
            torch.manual_seed(20260920); model=ResidualRanker(); opt=torch.optim.AdamW(model.parameters(),lr=.01)
            for step in range(2):
                tic=time.monotonic(); opt.zero_grad()
                score=model(torch.from_numpy(x),torch.from_numpy(a['prior'][keep]))
                loss=row_losses(score,torch.from_numpy(valid),torch.from_numpy(pos[keep]),
                    None if arm=='ordinary_ce' else torch.from_numpy(pairs[arm])).mean()
                assert torch.isfinite(loss); loss.backward(); opt.step(); updated+=1
                times.append(time.monotonic()-tic)
        old=arrays(b.path/'evaluation_seed0_cal2048.npz')
        pick=np.linspace(0,len(old['users'])-1,128,dtype=int); ix=old['indices'][pick]
        hs=[b.history(u) for u in old['users'][pick]]
        files=[b.path/'sasrec_scores_seed0.npy',LIME/domain/'itemcf_scores.npy',b.path/'semantic_scores.npy']
        a,lime,fused=b.build_rows(b.q['candidate_ids'][ix],hs,
            [np.asarray(np.load(p,mmap_mode='r')[ix]) for p in files],old['weights'][pick],old['penalty'][pick])
        names=old['arm_names'].tolist()
        np.testing.assert_array_equal(lime,old['top50'][names.index('LIME'),pick])
        gram=b.q['candidate_ids'][ix]
        rr=ranks(np.take_along_axis(fused,gram-1,1),gram,old['targets'][pick])
        np.testing.assert_array_equal(rr,old['ranks'][names.index('LIME_scores_G50'),pick])
        reports[domain]=dict(real_train_prefixes=128,covered_prefixes=int(keep.sum()),
            evaluation_parity_users=128,ordered_top50_parity=True,G50_rank_parity=True,
            prefix_seconds=prefix_seconds,mean_batch_update_seconds=float(np.mean(times)),
            estimated_30_step_head_seconds=float(np.mean(times)/max(int(keep.sum()),1)*len(fit)*30),
            discarded_small_model_updates=6,total_seconds=time.monotonic()-started)
        print(json.dumps(dict(domain=domain,**reports[domain]),ensure_ascii=False),flush=True)
    result=dict(state='PASSED',completed_at=now(),unit_checks=checks.testsRun,domains=reports,
        discarded_small_model_updates=updated,formal_training_started=False,
        development_metrics_used_for_selection=False,test_read=False,gpu_used=True,
        gpu_purpose='frozen_LIME_gate_only',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiment/phase20').glob('*source_competition*') if p.is_file()})
    write(out/'checks.json',result)


if __name__=='__main__': main()
