"""Beauty sibling of the immutable Toys runner; same fixed-epoch training recipe."""
import contextlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiment.phase20.guided_decoding_job import ROOT,read,write,sha,now
from experiment.phase20.lime_clean_upstream_core import validate_rows,cosine_neighbors,tensor_digest
sys.path.insert(0,str(ROOT/'third_party/LIME-Rec'))
from lime_rec.models import SASRec
from scripts.training.train_sasrec import SeqDataset

CONFIG=ROOT/'experiment/phase20/lime_clean_sasrec_beauty_v1.json'


class Run:
    def __init__(self):
        self.c=read(CONFIG);self.out=ROOT/self.c['output'];self.out.mkdir(exist_ok=False)
        self.start=time.monotonic();self.steps=0;self.epoch=0;self.inputs={};self.curve=[]
        torch.set_num_threads(4);torch.set_num_interop_threads(1);self.limits=threadpool_limits(4)

    def status(self,stage,state='RUNNING',**extra):
        elapsed=time.monotonic()-self.start
        write(self.out/'status.json',dict(state=state,stage=stage,updated_at=now(),pid=os.getpid(),
            elapsed_seconds=elapsed,epoch=self.epoch,max_epochs=self.c['epochs'],optimizer_steps=self.steps,
            device='cpu',cpu_threads=4,training_only=True,development_labels_read=False,test_read=False,**extra))
        print(now(),stage,'epoch',self.epoch,'steps',self.steps,extra,flush=True)

    def use(self,path):
        rel=str(path.relative_to(ROOT));assert 'validation' not in rel and 'test.json' not in rel
        self.inputs[rel]=sha(path);return path

    def checkpoint(self,model,opt,complete=False):
        path=self.out/('sasrec.pt' if complete else 'latest.pt');tmp=path.with_suffix('.tmp')
        torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),epoch=self.epoch,
            optimizer_steps=self.steps,config=self.c,input_sha256=self.inputs,
            torch_rng_state=torch.get_rng_state(),python_rng_state=random.getstate(),numpy_rng_state=np.random.get_state(),
            seed=self.c['seed'],num_items=self.v,maxlen=50,hidden=64,num_layers=2,num_heads=2,dropout=.2,
            architecture='legacy',loss='ce',label_smoothing=.1,train_complete=complete,training_only=True,
            development_labels_read=False,test_read=False,probe_only=False),tmp)
        tmp.replace(path)

    def run(self):
        self.status('核对截断训练数据与代码')
        audit=read(self.use(ROOT/'artifacts/phase20/lime_clean_beauty_preflight_20260920_v1/verification.json'))
        assert audit['state']=='VERIFIED' and audit['dataset_boundary_checked']
        parent=read(self.use(ROOT/'artifacts/phase20/lime_clean_training_design_20260920_v1/summary.json'))
        data=ROOT/self.c['data']
        for name in ('upstream_sequences_without_terminal.jsonl','upstream_train_without_terminal.jsonl'):
            p=self.use(data/name);assert self.inputs[str(p.relative_to(ROOT))]==parent['output_sha256'][str(p.relative_to(ROOT))]
        sequences={r['user_id']:r['history'] for r in map(json.loads,(data/'upstream_sequences_without_terminal.jsonl').open())}
        rows=list(map(json.loads,(data/'upstream_train_without_terminal.jsonl').open()))
        assert validate_rows(rows,sequences)==127328;del rows
        items=list(map(json.loads,self.use(ROOT/'artifacts/phase20/nearest_comparison_20260918/Beauty.items.jsonl').open()))
        ids=[r['source_item_id'] for r in items];self.v=len(ids)
        assert self.v==12101 and len(sequences)==22363
        self.use(CONFIG);self.use(Path(__file__))
        self.use(ROOT/'experiment/phase20/lime_clean_upstream_core.py')
        self.use(ROOT/'third_party/LIME-Rec/scripts/training/train_sasrec.py')
        for p in (ROOT/'third_party/LIME-Rec/lime_rec/models').glob('*.py'):self.use(p)
        for rel in self.inputs:
            p=ROOT/rel
            if p.suffix in ('.py','.json'):
                dst=self.out/'sources'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        write(self.out/'config.json',self.c);write(self.out/'input_manifest.json',self.inputs)
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        model=SASRec(self.v,64,50,2,2,.2,architecture='legacy')
        initial=tensor_digest(model.state_dict())
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
        dataset=SeqDataset([sequences[u] for u in sorted(sequences)],50,self.v)
        assert len(dataset)==22363
        loader=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=True,num_workers=0)
        expected_tokens=sum(min(len(s)-1,50) for s in sequences.values())
        for epoch in range(1,self.c['epochs']+1):
            self.epoch=epoch;model.train();tic=time.monotonic();total=0.;tokens=0;users=0
            for step,(x,y,_) in enumerate(loader,1):
                hidden=model(x);valid=y!=0;logits=hidden[valid]@model.item_emb.weight.T
                loss=torch.nn.functional.cross_entropy(logits,y[valid],label_smoothing=.1)
                assert torch.isfinite(loss);opt.zero_grad();loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
                opt.step();self.steps+=1
                n=int(valid.sum());total+=float(loss.detach())*n;tokens+=n;users+=len(x)
                if step%50==0:self.status('SASRec正式训练',batch=step,batches=len(loader),loss=float(loss.detach()))
            assert tokens==expected_tokens and users==22363
            entry=dict(epoch=epoch,seconds=time.monotonic()-tic,loss=total/tokens,users=users,
                supervised_positions=tokens,optimizer_steps=self.steps)
            self.curve.append(entry);write(self.out/'training_curve.json',self.curve)
            if epoch==1 or epoch%self.c['checkpoint_every_epochs']==0:self.checkpoint(model,opt)
            rate=float(np.mean([r['seconds'] for r in self.curve[-5:]]))
            self.status('SASRec本轮完成',loss=entry['loss'],seconds_per_epoch=rate,
                estimated_remaining_seconds=(self.c['epochs']-epoch)*rate)
        assert self.steps==len(loader)*200 and tensor_digest(model.state_dict())!=initial
        self.checkpoint(model,opt,complete=True)
        saved=torch.load(self.out/'sasrec.pt',map_location='cpu');assert saved['train_complete']
        assert tensor_digest(saved['state_dict'])==tensor_digest(model.state_dict())
        self.status('生成正式相似商品提示')
        neighbors=cosine_neighbors(model.item_emb.weight.detach().numpy()[1:],k=10)
        np.save(self.out/'neighbor_ids.npy',neighbors)
        with (self.out/'similar_item_sasrec.txt').open('x') as f:
            for item,ns in zip(ids,neighbors):f.write(item+' '+' '.join(ids[i-1] for i in ns)+'\n')
        for p,h in self.inputs.items():assert sha(ROOT/p)==h,p
        write(self.out/'summary.json',dict(state='COMPLETED',completed_at=now(),elapsed_seconds=time.monotonic()-self.start,
            epochs=200,optimizer_steps=self.steps,users=22363,num_items=self.v,training_only=True,
            heldout_labels_used_for_selection=False,development_labels_read=False,test_read=False,
            checkpoint_sha256=sha(self.out/'sasrec.pt'),neighbors_sha256=sha(self.out/'neighbor_ids.npy'),
            input_sha256=self.inputs,checkpoint_exact_state_replay=True,next_stage_started=False))
        self.status('SASRec和正式提示已完成',state='COMPLETED')


def main():
    runner=Run()
    def timeout(*_):raise TimeoutError('Fixed two-hour SASRec training budget exceeded; last checkpoint retained')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(runner.c['max_seconds'])
    with (runner.out/'run.log').open('x',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:runner.run()
        except BaseException as e:
            runner.status('训练中断，保留最近检查点',state='FAILED',error=repr(e));traceback.print_exc();raise
        finally:signal.alarm(0)


if __name__=='__main__':main()
