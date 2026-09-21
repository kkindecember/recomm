"""Post-screen, frozen matched-label controls. No extra fitting of the selected arm."""
import json
import os
import signal
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiment.phase20.future_factor_screen import Factor, old_score, z, ranks, metrics, sha

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase20/future_label_specificity_20260917'
PARENT=ROOT/'artifacts/phase20/future_factor_screen_20260917'
CONTRACT=ROOT/'plan/第二十阶段/未来软标签收益来源_匹配控制合同_20260917.md'
ARMS=('uniform_mass','shuffle_identity','score_matched_shuffle')


def labels_for_controls(positions,distances,prior):
    immediate=np.arange(50)[None]==positions[:,None]
    actual=((distances>0)&(distances<=3)&(~immediate))/(np.maximum(distances,0)+1)
    uniform=(~immediate)*actual.sum(1,keepdims=True)/49
    shuffled=actual.copy();matched=actual.copy();rng=np.random.RandomState(20260917)
    for i,p in enumerate(positions):
        negatives=np.flatnonzero(np.arange(50)!=p);shuffled[i,negatives]=rng.permutation(actual[i,negatives])
        ordered=negatives[np.argsort(-prior[i,negatives],kind='stable')]
        for block in ordered.reshape(7,7):matched[i,block]=rng.permutation(actual[i,block])
    values=dict(uniform_mass=uniform,shuffle_identity=shuffled,score_matched_shuffle=matched)
    for name,y in values.items():
        np.testing.assert_allclose(y.sum(1),actual.sum(1),atol=1e-12,rtol=0)
        assert not y[immediate].any() and np.isfinite(y).all() and (y>=0).all() and (y<=1).all()
        if name!='uniform_mass':np.testing.assert_array_equal(np.sort(y,axis=1),np.sort(actual,axis=1))
    return {k:y+immediate for k,y in values.items()},actual+immediate,{
        k:dict(changed_entries=int((np.abs(y-actual)>1e-12).sum()),
            retained_true_future_mass=float((y*(actual>0)).sum()/np.maximum(actual.sum(),1e-12))) for k,y in values.items()}


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);OUT.mkdir(exist_ok=False)
    start=time.monotonic();steps=0;hashes={}
    def use(p):hashes[str(p.relative_to(ROOT))]=sha(p);return p
    def status(state,**kw):
        p=OUT/'status.tmp';p.write_text(json.dumps(dict(state=state,elapsed_seconds=time.monotonic()-start,
            updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),pid=os.getpid(),training_steps=steps,
            gpu_used=False,test_read=False,automatic_next_experiment=False,**kw),indent=2)+'\n');p.replace(OUT/'status.json')
    def timeout(sig,frame):raise TimeoutError('10 minute CPU wall budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600);result={}
    try:
        status('RUNNING',stage='verify');use(Path(__file__));use(CONTRACT);use(ROOT/'experiment/phase20/future_factor_screen.py')
        parent=json.loads(use(PARENT/'summary.json').read_text());audit=json.loads(use(PARENT/'verification.json').read_text())
        assert audit['state']=='VERIFIED'
        for group in (parent['input_sha256'],audit['output_sha256']):
            for p,h in group.items():assert sha(use(ROOT/p))==h,p
        (OUT/'protocol.json').write_text(json.dumps(dict(contract=str(CONTRACT.relative_to(ROOT)),arms=ARMS,
            source_sha256=sha(Path(__file__)),selected_arm_retrained=False,seed=20260917,steps_per_arm=30),indent=2)+'\n')
        for domain in ('Toys','Beauty'):
            status('RUNNING',dataset=domain,stage='load')
            old=ROOT/f'artifacts/phase20/gacr_fusion_train_v1/{domain}'
            with np.load(use(old/'training_cache.npz')) as f:a={k:f[k] for k in f.files}
            with np.load(use(ROOT/f'artifacts/phase20/future_horizon_audit_20260917/{domain}.labels.npz')) as f:b={k:f[k] for k in f.files}
            for k in ('candidate_ids','fit','positions'):np.testing.assert_array_equal(a[k],b[k])
            valid=a['fit']&(a['positions']>=0)&(b['remaining']>=3);head=a['head'][valid]
            xall=torch.as_tensor(a['features']);g=old_score(xall,torch.as_tensor(a['base']),torch.load(use(old/'B1.pt'),map_location='cpu'))
            prior=z(g[valid])-np.log(49);targets,actual,diagnostics=labels_for_controls(a['positions'][valid],b['distances'][valid],prior)
            np.savez_compressed(OUT/f'{domain}.training_labels.npz',actual=actual,**targets)
            x=xall[valid];prior=torch.as_tensor(prior);weights=torch.as_tensor(np.where(head,1/(2*head.sum()),1/(2*(~head).sum())))
            models={};curves={}
            for arm in ARMS:
                torch.manual_seed(20260917);m=Factor('next_only');optim=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.01)
                y=torch.as_tensor(targets[arm]);curve=[]
                for step in range(1,31):
                    optim.zero_grad(set_to_none=True);total=0.
                    for offset in range(0,len(x),256):
                        sl=slice(offset,offset+256);score,_=m(x[sl],prior[sl]);loss=(F.binary_cross_entropy_with_logits(score,y[sl],reduction='none').mean(1)*weights[sl]).sum()
                        assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
                    grad=torch.nn.utils.clip_grad_norm_(m.parameters(),10.,error_if_nonfinite=True);optim.step();steps+=1
                    curve.append(dict(step=step,loss=total,grad_norm=float(grad)))
                    status('RUNNING',dataset=domain,stage='training',arm=arm,step=step,total_steps=30)
                models[arm]=m.eval();curves[arm]=curve;torch.save(m.state_dict(),OUT/f'{domain}.{arm}.pt')
            (OUT/f'{domain}.training.json').write_text(json.dumps(dict(curves=curves,label_diagnostics=diagnostics),indent=2)+'\n')
            with np.load(use(PARENT/f'{domain}.audit_validation_cache.npz')) as f:v={k:f[k] for k in f.files}
            with np.load(use(PARENT/f'{domain}.predictions.npz')) as f:previous={k:f[k] for k in f.files}
            np.testing.assert_array_equal(v['users'],previous['users']);np.testing.assert_array_equal(v['candidate_ids'],previous['candidate_ids'])
            fitusers=set(b['users'][valid]);in_fit=np.asarray([u in fitusers for u in v['users']]);out_scores={};out_ranks={}
            for arm,m in models.items():
                status('RUNNING',dataset=domain,stage='validation',arm=arm);pred=[]
                with torch.no_grad():
                    for offset in range(0,len(v['prior']),256):
                        sl=slice(offset,offset+256);ss,_=m(torch.as_tensor(v['features'][sl]),torch.as_tensor(v['prior'][sl]));pred.append(ss.numpy())
                raw=z(np.concatenate(pred))
                for fusion in ('current','mean'):
                    name=arm+'.'+fusion;out_scores[name]=raw+v['cf_'+fusion];out_ranks[name]=ranks(out_scores[name],v['positions'])
            for arm in ('soft_future','next_only','frozen'):
                for fusion in ('current','mean'):
                    name=arm+'.'+fusion;j=list(previous['arm_names']).index(name);out_scores[name]=previous['scores'][j];out_ranks[name]=previous['ranks'][j]
            group_masks=dict(all=np.ones(len(in_fit),bool),tail=v['tail_mask'],non_tail=~v['tail_mask'],nonfit=~in_fit)
            met={name:{group:metrics(rr[mask]) for group,mask in group_masks.items()} for name,rr in out_ranks.items()}
            comparisons={}
            for arm in ARMS:
                true=out_ranks['soft_future.mean'];control=out_ranks[arm+'.mean']
                comparisons[arm]=dict(relative_N10=met['soft_future.mean']['all']['N10']/met[arm+'.mean']['all']['N10']-1,
                    nonfit_relative_N10=met['soft_future.mean']['nonfit']['N10']/met[arm+'.mean']['nonfit']['N10']-1,
                    hit_gained=int(((true<=10)&(control>10)).sum()),hit_lost=int(((true>10)&(control<=10)).sum()))
            passed=all(r['relative_N10']>=.005 and r['nonfit_relative_N10']>0 for r in comparisons.values())
            result[domain]=dict(metrics=met,comparisons=comparisons,label_diagnostics=diagnostics,passed=passed)
            names=list(out_scores);np.savez_compressed(OUT/f'{domain}.predictions.npz',users=v['users'],candidate_ids=v['candidate_ids'],
                positions=v['positions'],tail_mask=v['tail_mask'],in_fit=in_fit,arm_names=np.asarray(names),
                scores=np.stack([out_scores[n] for n in names]),ranks=np.stack([out_ranks[n] for n in names]))
            (OUT/f'{domain}.summary.json').write_text(json.dumps(result[domain],indent=2)+'\n')
        for p,h in hashes.items():assert sha(ROOT/p)==h,p
        decision='FUTURE_IDENTITY_ADDS_SCREENING_EVIDENCE' if all(v['passed'] for v in result.values()) else 'FUTURE_IDENTITY_ADVANTAGE_NOT_ESTABLISHED'
        (OUT/'summary.json').write_text(json.dumps(dict(decision=decision,domains=result,input_sha256=hashes,
            development_reused=True,posthoc_followup=True,test_read=False),indent=2)+'\n')
        status('COMPLETED',decision=decision);print(json.dumps({d:r['comparisons'] for d,r in result.items()},indent=2))
    except Exception as e:status('FAILED',error=repr(e));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
