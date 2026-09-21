"""Bounded train-only future-label screen; same capacity including a flat 3-class control."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase20/future_factor_screen_20260917'
CONTRACT=ROOT/'plan/第二十阶段/近期会买与下一件先买_最小监督筛查合同_20260917.md'
ARMS=('next_only','soft_future','flat_three','conditional')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def z(x):
    return (x-x.mean(-1,keepdims=True))/np.maximum(x.std(-1,keepdims=True),1e-6)


def old_score(features,base,state):
    with torch.no_grad():
        hidden=torch.tanh(F.linear(features,state['network.0.weight'],state['network.0.bias']))
        return (base+.2*torch.tanh(F.linear(hidden,state['network.2.weight'],state['network.2.bias'])).squeeze(-1)).numpy().astype(np.float64)


class Factor(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(6,16),nn.Tanh(),nn.Linear(16,1)) for _ in range(2)])
        for h in self.heads:nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)

    def forward(self,x,prior):
        # If a=b=root, logit(sigmoid(a)*sigmoid(b)) == prior.
        root=prior+torch.log1p(torch.sqrt(1+torch.exp(-prior)))
        delta=[2*torch.tanh(h(x).squeeze(-1)).double() for h in self.heads]
        if self.arm=='flat_three':
            log_a=F.logsigmoid(root);log_not_a=F.logsigmoid(-root)
            logits=torch.stack([2*log_a-log_not_a+delta[0],log_a+delta[1],torch.zeros_like(prior)],-1)
            lp=F.log_softmax(logits,-1)
        else:
            a,b=root+delta[0],root+delta[1]
            lp=torch.stack([F.logsigmoid(a)+F.logsigmoid(b),F.logsigmoid(a)+F.logsigmoid(-b),F.logsigmoid(-a)],-1)
        next_logit=lp[:,:,0]-torch.logsumexp(lp[:,:,1:],-1)
        return next_logit,lp


def loss_per_row(model,x,prior,positions,distances):
    s,lp=model(x,prior)
    immediate=torch.arange(50)[None]==positions[:,None]
    later=(distances>0)&(distances<=3)&(~immediate)
    if model.arm in ('flat_three','conditional'):
        classes=torch.where(immediate,torch.zeros_like(distances),torch.where(later,torch.ones_like(distances),torch.full_like(distances,2)))
        return -lp.gather(2,classes[:,:,None]).squeeze(-1).mean(1)
    target=immediate.double()
    if model.arm=='soft_future':
        target=target+later.double()/(distances.clamp_min(0).double()+1)
    return F.binary_cross_entropy_with_logits(s,target,reduction='none').mean(1)


def checks():
    torch.manual_seed(9);x=torch.randn(3,50,6);p=torch.randn(3,50,dtype=torch.float64)-math.log(49)
    pos=torch.tensor([2,4,8]);d=torch.full((3,50),-1,dtype=torch.long);d[:,12]=1;d[:,13]=3
    initial=[]
    for arm in ARMS:
        torch.manual_seed(20260917);m=Factor(arm)
        assert sum(t.numel() for t in m.parameters())==258
        s,lp=m(x,p);np.testing.assert_allclose(s.detach().numpy(),p.numpy(),atol=1e-12,rtol=0)
        np.testing.assert_allclose(lp.detach().exp().sum(-1).numpy(),1,atol=1e-12,rtol=0)
        initial.append(lp.detach().numpy())
        loss=loss_per_row(m,x,p,pos,d).mean();loss.backward()
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters())
        if arm=='next_only':
            np.testing.assert_allclose(loss_per_row(m,x,p,pos,d).detach().numpy(),loss_per_row(m,x,p,pos,torch.zeros_like(d)).detach().numpy(),atol=0,rtol=0)
    for a in initial[1:]:np.testing.assert_allclose(initial[0],a,atol=1e-12,rtol=0)
    # Nonzero logits: hierarchical joint likelihood equals BCE(A)+I(A)*BCE(B).
    a=torch.randn(3,50,dtype=torch.float64,requires_grad=True);b=torch.randn(3,50,dtype=torch.float64,requires_grad=True)
    im=torch.arange(50)[None]==pos[:,None];future=(d>0)&(d<=3)&~im;window=im|future
    explicit=(F.binary_cross_entropy_with_logits(a,window.double(),reduction='none')+window*F.binary_cross_entropy_with_logits(b,im.double(),reduction='none')).mean()
    joint=-(im*(F.logsigmoid(a)+F.logsigmoid(b))+future*(F.logsigmoid(a)+F.logsigmoid(-b))+(~window)*F.logsigmoid(-a)).mean()
    np.testing.assert_allclose(explicit.detach().numpy(),joint.detach().numpy(),atol=1e-12,rtol=0)
    for e,j in zip(torch.autograd.grad(explicit,(a,b)),torch.autograd.grad(joint,(a,b))):np.testing.assert_allclose(e.numpy(),j.numpy(),atol=1e-12,rtol=0)
    return dict(initial_parity=True,normalized_probabilities=True,parameter_count=258,
        finite_gradients=True,next_only_ignores_future=True,joint_likelihood_gradient_identity=True)


def ranks(scores,positions):
    inv=np.argsort(np.argsort(-scores,axis=1,kind='stable'),axis=1)
    return np.where(positions>=0,inv[np.arange(len(inv)),np.maximum(positions,0)]+1,51)


def metrics(rr):
    return dict(users=len(rr),**{f'{s}{k}':float(((rr<=k) if s=='H' else (rr<=k)/np.log2(rr+1)).mean()) for k in (5,10,20,50) for s in ('N','H')})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    health=checks()
    if args.check_only:print(json.dumps(health));return
    OUT.mkdir(exist_ok=False);start=time.monotonic();hashes={};steps=0
    def use(path):hashes[str(path.relative_to(ROOT))]=sha(path);return path
    def status(state,**kw):
        elapsed=time.monotonic()-start
        if state=='RUNNING' and elapsed>900:raise TimeoutError('15 minute CPU budget')
        tmp=OUT/'status.tmp';tmp.write_text(json.dumps(dict(state=state,elapsed_seconds=elapsed,
            updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),pid=os.getpid(),training_steps=steps,
            gpu_used=False,test_read=False,automatic_next_experiment=False,**kw),indent=2)+'\n');tmp.replace(OUT/'status.json')
    def timeout(sig,frame):raise TimeoutError('15 minute CPU wall budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    status('RUNNING',stage='load');use(Path(__file__));use(CONTRACT)
    cfg=json.loads(use(ROOT/'experiment/phase20/gacr_fusion_train_v1.json').read_text())
    audit=json.loads(use(ROOT/'artifacts/phase20/future_horizon_audit_20260917/summary.json').read_text())
    for p,h in audit['input_sha256'].items():assert sha(use(ROOT/p))==h,p
    (OUT/'protocol.json').write_text(json.dumps(dict(arms=ARMS,contract=str(CONTRACT.relative_to(ROOT)),
        source_sha256=sha(Path(__file__)),checks=health,seed=20260917,steps_per_arm=30,window=4),indent=2)+'\n')
    result={}
    try:
        for domain,config in cfg['domains'].items():
            status('RUNNING',dataset=domain,stage='train_data')
            folder=ROOT/f'artifacts/phase20/gacr_fusion_train_v1/{domain}'
            with np.load(use(folder/'training_cache.npz')) as f:cache={k:f[k] for k in f.files}
            with np.load(use(ROOT/f'artifacts/phase20/future_horizon_audit_20260917/{domain}.labels.npz')) as f:labels={k:f[k] for k in f.files}
            for k in ('candidate_ids','positions','fit'):np.testing.assert_array_equal(labels[k],cache[k])
            state=torch.load(use(folder/'B1.pt'),map_location='cpu')
            xall=torch.as_tensor(cache['features']);g=old_score(xall,torch.as_tensor(cache['base']),state)
            valid=cache['fit']&(cache['positions']>=0)&(labels['remaining']>=3)
            assert int(valid.sum())==audit['domains'][domain]['3']['covered_fit_prefixes']
            assert set(labels['users'][valid]).isdisjoint(set(labels['users'][~cache['fit']]))
            x=xall[valid];prior=torch.as_tensor(z(g[valid])-math.log(49));pos=torch.as_tensor(cache['positions'][valid]);dist=torch.as_tensor(labels['distances'][valid])
            head=cache['head'][valid];assert head.any() and (~head).any()
            weights=torch.as_tensor(np.where(head,1/(2*head.sum()),1/(2*(~head).sum())))
            models={};curves={}
            # All models fit before any new development outcome is computed.
            for arm in ARMS:
                torch.manual_seed(20260917);model=Factor(arm);optim=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01);curve=[]
                for step in range(1,31):
                    optim.zero_grad(set_to_none=True);total=0.
                    for offset in range(0,len(x),256):
                        sl=slice(offset,offset+256);loss=(loss_per_row(model,x[sl],prior[sl],pos[sl],dist[sl])*weights[sl]).sum()
                        assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optim.step();steps+=1
                    curve.append(dict(step=step,loss=total,grad_norm=float(grad)))
                    status('RUNNING',dataset=domain,stage='training',arm=arm,step=step,total_steps=30)
                models[arm]=model.eval();curves[arm]=curve;torch.save(model.state_dict(),OUT/f'{domain}.{arm}.pt')
            (OUT/f'{domain}.training.json').write_text(json.dumps(dict(curves=curves,fit_prefixes=int(valid.sum()),fit_users=len(set(labels['users'][valid]))),indent=2)+'\n')
            status('RUNNING',dataset=domain,stage='validation_features')
            source=ROOT/'artifacts/phase20/pair_uncertainty_screen_20260917'
            with np.load(use(source/f'{domain}.predictions.npz')) as f:ref={k:f[k] for k in f.files}
            with np.load(use(folder/'B1.validation_scores.npz')) as f:b1={k:f[k] for k in f.files}
            np.testing.assert_array_equal(b1['users'],ref['users'])
            canonical=np.argsort(-b1['residual_scores'],axis=1,kind='stable')
            np.testing.assert_array_equal(np.take_along_axis(b1['candidate_ids'],canonical,1),ref['candidate_ids'])
            lookup={u:i for i,u in enumerate(b1['users'])};features=np.empty((*b1['candidate_ids'].shape,6),np.float32);seen=set()
            for row in map(json.loads,use(ROOT/config['features']).open()):
                u=row['user_id'];assert u not in seen;seen.add(u);i=lookup[u]
                np.testing.assert_array_equal(row['union_ids'][:50],b1['candidate_ids'][i]);features[i]=row['features'][:50]
            assert seen==set(lookup)
            features=np.take_along_axis(features,canonical[:,:,None],1)
            vg=z(np.take_along_axis(b1['residual_scores'],canonical,1));vp=vg-math.log(49)
            cf_parts={name:ref['scores'][list(ref['arm_names']).index(name)]-vg for name in ('current','mean')}
            # No-update deployment must reproduce every frozen target rank.
            torch.manual_seed(20260917);untrained=Factor('conditional').eval()
            with torch.no_grad():
                zero,_=untrained(torch.as_tensor(features),torch.as_tensor(vp))
            for fusion in ('current','mean'):
                np.testing.assert_array_equal(ranks(z(zero.numpy())+cf_parts[fusion],ref['positions']),
                    ref['ranks'][list(ref['arm_names']).index(fusion)])
            output_scores={};output_ranks={};all_metrics={}
            for arm,model in models.items():
                status('RUNNING',dataset=domain,stage='validation',arm=arm)
                pred=[]
                with torch.no_grad():
                    for offset in range(0,len(vg),256):
                        sl=slice(offset,offset+256);s,_=model(torch.as_tensor(features[sl]),torch.as_tensor(vp[sl]));pred.append(s.numpy())
                raw=z(np.concatenate(pred))
                for fusion in ('raw','current','mean'):
                    name=f'{arm}.{fusion}';scores=raw if fusion=='raw' else raw+cf_parts[fusion]
                    output_scores[name]=scores;output_ranks[name]=ranks(scores,ref['positions'])
            for fusion in ('current','mean'):
                name=f'frozen.{fusion}';idx=list(ref['arm_names']).index(fusion)
                output_scores[name]=ref['scores'][idx];output_ranks[name]=ref['ranks'][idx]
            for name,rr in output_ranks.items():
                all_metrics[name]={group:metrics(rr[mask]) for group,mask in [('all',np.ones(len(rr),bool)),('tail',ref['tail_mask']),('non_tail',~ref['tail_mask'])]}
            names=list(output_scores)
            np.savez_compressed(OUT/f'{domain}.predictions.npz',users=ref['users'],candidate_ids=ref['candidate_ids'],
                positions=ref['positions'],tail_mask=ref['tail_mask'],arm_names=np.asarray(names),
                scores=np.stack([output_scores[n] for n in names]),ranks=np.stack([output_ranks[n] for n in names]))
            m=all_metrics;main_metric=m['conditional.mean']['all'];baseline=m['frozen.mean']['all']
            passed=(main_metric['N10']/baseline['N10']-1>=.005 and main_metric['H10']>=baseline['H10'] and
                all(main_metric['N10']>m[a+'.mean']['all']['N10'] for a in ARMS if a!='conditional'))
            result[domain]=dict(metrics=m,passed=passed,relative_N10_vs_mean=main_metric['N10']/baseline['N10']-1)
            (OUT/f'{domain}.summary.json').write_text(json.dumps(result[domain],indent=2)+'\n')
        for p,h in hashes.items():assert sha(ROOT/p)==h,p
        decision='SUPERVISION_DECOMPOSITION_PASSES_SCREEN' if all(v['passed'] for v in result.values()) else 'DECOMPOSITION_ADVANTAGE_NOT_ESTABLISHED'
        (OUT/'summary.json').write_text(json.dumps(dict(domains=result,decision=decision,input_sha256=hashes,
            checks=health,development_reused=True,test_read=False),indent=2)+'\n')
        status('COMPLETED',decision=decision);print(json.dumps({d:{k:v for k,v in s.items() if k!='metrics'} for d,s in result.items()},indent=2))
    except Exception as e:status('FAILED',error=repr(e));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
