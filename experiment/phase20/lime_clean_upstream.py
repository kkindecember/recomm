"""Prepare isolated upstream inputs and measure a bounded, disposable GPU probe.

No development/test reader, old trained checkpoint or old CF hint is used.
Probe-trained SASRec hints measure the interface only, not recommendation quality.
"""
import argparse
import contextlib
import gc
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import unittest

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiment.phase20.guided_decoding_job import ROOT,read,write,sha,now
from experiment.phase20.lime_clean_upstream_core import (
    validate_rows,cosine_neighbors,item_inputs,gram_sample,runtime_args,new_gram,CleanRows,tensor_digest)

CONFIG=ROOT/'experiment/phase20/lime_clean_upstream_toys_v1.json'
T5=ROOT/'.cache/huggingface/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4'
for p in (ROOT/'GRAM/src',ROOT/'third_party/LIME-Rec'):
    sys.path.insert(0,str(p))


def rows(path): return [json.loads(line) for line in Path(path).open()]
def line_map(path): return dict(line.rstrip('\n').split(' ',1) for line in Path(path).open())


class Job:
    def __init__(self,action):
        self.c=read(CONFIG);self.action=action;self.root=ROOT/self.c['output'];self.out=self.root/action
        self.out.mkdir(parents=True,exist_ok=False)
        self.start=time.monotonic();self.inputs={};self.timings={}
        torch.set_num_threads(self.c['cpu_threads']);torch.set_num_interop_threads(1)
        self.limits=threadpool_limits(self.c['cpu_threads'])

    def status(self,stage,state='RUNNING',**extra):
        result=dict(state=state,stage=stage,pid=os.getpid(),updated_at=now(),
            seconds=time.monotonic()-self.start,development_labels_read=False,test_read=False,**extra)
        write(self.out/'status.json',result);print(result,flush=True)

    def use(self,path):
        path=Path(path);rel=str(path.relative_to(ROOT))
        assert 'test.json' not in rel and 'validation.json' not in rel
        assert path.name not in ('user_sequence.txt','similar_item_sasrec.txt')
        if rel not in self.inputs:self.inputs[rel]=sha(path)
        return path

    def load_data(self):
        parent=read(self.use(ROOT/'artifacts/phase20/lime_clean_training_design_20260920_v1/summary.json'))
        for p,h in parent['output_sha256'].items():
            if '/Toys/' in p: assert sha(self.use(ROOT/p))==h
        d=ROOT/self.c['data'];self.train=rows(d/'upstream_train_without_terminal.jsonl')
        self.sequences={r['user_id']:r['history'] for r in rows(d/'upstream_sequences_without_terminal.jsonl')}
        self.held=rows(d/'terminal_holdout.jsonl')
        assert validate_rows(self.train,self.sequences)==105867
        assert len(self.held)==3494 and len(self.sequences)==19412
        items=rows(self.use(ROOT/'artifacts/phase20/nearest_comparison_20260918/Toys.items.jsonl'))
        assert [r['item_id'] for r in items]==list(range(1,len(items)+1))
        self.ids=[r['source_item_id'] for r in items];self.v=len(self.ids)
        self.text=line_map(self.use(ROOT/'GRAM/rec_datasets/Toys/item_plain_text.txt'))
        self.lex=line_map(self.use(ROOT/'GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt'))
        assert set(self.ids)==set(self.lex)==set(self.text)
        self.count=np.bincount(np.concatenate(list(self.sequences.values())),minlength=self.v+1)
        assert int(self.count.sum())==105867+19412
        assert all(r['history']==self.sequences[r['user_id']][-20:] for r in self.held)
        self.c1_users=set(self.use(ROOT/'artifacts/phase4/gcdh_p0_splits/Toys/train_users.txt').read_text().splitlines())
        self.c1_rows=[i for i,r in enumerate(self.train) if r['user_id'] in self.c1_users]
        self.use(CONFIG)
        for p in (ROOT/'experiment/phase20').glob('*lime_clean_upstream*'):
            if p.is_file():self.use(p)
        self.use(ROOT/'experiment/phase4/gcdh_p0.py')
        for sub in ('GRAM/src/model','GRAM/src/processor'):
            for p in (ROOT/sub).glob('*.py'):self.use(p)
        for name in ('lime_rec/itemcf.py','scripts/training/train_sasrec.py'):
            self.use(ROOT/'third_party/LIME-Rec'/name)
        for p in (ROOT/'third_party/LIME-Rec/lime_rec/models').glob('*.py'):self.use(p)
        for name in ('config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','spiece.model'):
            self.use(T5/name)

    def finish(self,extra):
        for p,h in self.inputs.items(): assert sha(ROOT/p)==h,p
        write(self.out/'summary.json',dict(state='COMPLETED',action=self.action,completed_at=now(),
            seconds=time.monotonic()-self.start,input_sha256=self.inputs,timings=self.timings,
            development_labels_read=False,test_read=False,formal_training_started=False,**extra))
        self.status('本阶段完成',state='COMPLETED')

    def prepare(self):
        self.status('核对训练前缀和全部依赖');self.load_data()
        from experiment.phase20.test_lime_clean_upstream import CleanUpstreamTests
        log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(CleanUpstreamTests))
        (self.out/'checks.log').write_text(log.getvalue());assert result.wasSuccessful(),log.getvalue()
        self.status('重建截断训练序列的ItemCF和频次')
        from lime_rec.itemcf import build_itemcf_model
        start=time.monotonic()
        history={u:[self.ids[i-1] for i in seq] for u,seq in self.sequences.items()}
        cf=build_itemcf_model(dataset_name='Toys',histories=history,item_ids=self.ids,top_k=100,window_size=20)
        assert cf['num_training_interactions']==int(self.count.sum())
        write(self.out/'itemcf.json',cf);np.save(self.out/'frequencies.npy',self.count)
        self.timings['itemcf_build_seconds']=time.monotonic()-start
        lengths=np.array([len(r['history']) for r in self.train])
        # Quantile-spaced lengths for timing; no target/coverage selection.
        order=np.argsort(lengths,kind='stable')
        ix=order[np.linspace(0,len(order)-1,self.c['gram_profile_rows'],dtype=int)]
        c1=np.array(self.c1_rows);c1=c1[np.argsort(lengths[c1],kind='stable')]
        cix=c1[np.linspace(0,len(c1)-1,8,dtype=int)]
        np.savez(self.out/'profile_rows.npz',gram=ix,c1=cix)
        for p in self.inputs:
            source=ROOT/p
            if source.suffix in ('.py','.json') and not p.startswith('.cache'):
                dst=self.out/'sources'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
        self.finish(dict(unit_tests=result.testsRun,train_rows=len(self.train),train_users=len(self.sequences),
            training_interactions=int(self.count.sum()),heldout_events=len(self.held),catalog_items=self.v,
            c1_train_users=len(self.c1_users),c1_train_rows=len(self.c1_rows),
            history_length_counts={str(k):int((lengths==k).sum()) for k in np.unique(lengths)},
            profile_rows=ix.tolist(),profile_history_lengths=lengths[ix].tolist(),
            outputs_sha256={p.name:sha(p) for p in self.out.iterdir() if p.is_file() and p.name not in ('status.json','run.log')},
            retained_lexical_ids='Static catalog asset reused; text-only construction notebook inspected, exact historical generation receipt unavailable.',
            new_neighbor_recipe='Cosine of newly trained SASRec item embeddings, self excluded, numeric ID tie-break. Original neighbor-builder recipe not available; this is a declared replacement, not exact reproduction.'))

    def profile(self):
        use_gpu=self.action=='profile'
        self.status('核对准备产物与运行设备');self.load_data()
        prepared=read(self.use(self.root/'prepare/summary.json'));assert prepared['state']=='COMPLETED'
        for n,h in prepared['outputs_sha256'].items():assert sha(self.use(self.root/'prepare'/n))==h
        query='CPU; GPU timing not measured'
        if use_gpu:
            query=subprocess.run(['nvidia-smi','--id='+self.c['gpu_uuid'],'--query-gpu=memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout.strip()
            free=int(query.split(',')[0]);assert free>=self.c['minimum_free_mib'],f'Insufficient free GPU memory: {query}'
        os.environ['CUDA_VISIBLE_DEVICES']=self.c['gpu_uuid'] if use_gpu else ''
        if use_gpu:
            assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(self.c['cuda_memory_fraction'])
        self.sync=torch.cuda.synchronize if use_gpu else lambda:None
        self.peak=(lambda:torch.cuda.max_memory_allocated()/2**20) if use_gpu else lambda:None
        self.reset_peak=torch.cuda.reset_peak_memory_stats if use_gpu else lambda:None
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        random.seed(self.c['seed']);np.random.seed(self.c['seed']);torch.manual_seed(self.c['seed'])
        if use_gpu:torch.cuda.manual_seed_all(self.c['seed'])
        self.device=torch.device('cuda' if use_gpu else 'cpu')
        self.status('SASRec运行检查，仅截断训练序列',device=str(self.device),device_start=query)
        from lime_rec.models import SASRec
        from scripts.training.train_sasrec import SeqDataset
        from torch.utils.data import DataLoader
        model=SASRec(self.v,64,50,2,2,.2,architecture='legacy').to(self.device)
        initial=tensor_digest(model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=.001)
        data=SeqDataset([self.sequences[u] for u in sorted(self.sequences)],50,self.v)
        loader=DataLoader(data,batch_size=self.c['sasrec_batch_size'],shuffle=True,num_workers=0)
        records=[];model.train()
        for step,(x,y,negative) in enumerate(loader):
            if step>=self.c['sasrec_profile_batches']:break
            self.sync();start=time.monotonic();x=x.to(self.device);y=y.to(self.device)
            hidden=model(x);mask=y!=0;logits=hidden[mask]@model.item_emb.weight.T
            loss=torch.nn.functional.cross_entropy(logits,y[mask],label_smoothing=.1)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
            self.sync();records.append(dict(step=step+1,seconds=time.monotonic()-start,users=len(x),loss=float(loss)))
            if step%10==0:self.status('SASRec速度检查',step=step+1,total=self.c['sasrec_profile_batches'])
        assert tensor_digest(model.state_dict())!=initial
        torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
            probe_only=True,train_complete=False,steps=len(records),input_sha256=self.inputs),self.out/'sasrec_probe.pt')
        emb=model.item_emb.weight.detach().cpu().numpy()[1:].copy()
        sas_peak=self.peak()
        del model,opt,hidden,logits,loss,x,y,negative;gc.collect();torch.cuda.empty_cache()
        self.status('生成独立相似商品提示，仅供接口检查')
        near=cosine_neighbors(emb,k=5);inputs=item_inputs(self.ids,self.text,self.lex,near)
        with (self.out/'probe_similar_items.txt').open('x') as f:
            for item,ns in zip(self.ids,near):f.write(item+' '+' '.join(self.ids[i-1] for i in ns)+'\n')
        np.save(self.out/'probe_neighbor_ids.npy',near)
        # All inference text is assembled independently of the query target.
        selected=np.load(self.root/'prepare/profile_rows.npz')
        rows_ix=selected['gram'];c1_ix=selected['c1'];selected.close()
        self.status('从外部T5初始化GRAM，检查前向、反向和保存重放')
        model,tokenizer=new_gram(T5);model.set_checkpoint(self.c['gram_checkpointing']);model.to(self.device)
        from processor import CollatorGRAM
        collator=CollatorGRAM(tokenizer=tokenizer,args=runtime_args(),mode='train')
        dataset=CleanRows(self.train,self.ids,inputs,self.lex)
        self.reset_peak();initial=tensor_digest(model.state_dict())
        from transformers import AdamW
        groups=[{'params':[p for n,p in model.named_parameters() if not any(z in n for z in ('bias','LayerNorm.weight'))],'weight_decay':.01},
                {'params':[p for n,p in model.named_parameters() if any(z in n for z in ('bias','LayerNorm.weight'))],'weight_decay':0.}]
        opt=AdamW(groups,lr=.001,eps=1e-6)
        gram=self.measure(model,opt,collator,dataset,rows_ix,c1=False)
        assert tensor_digest(model.state_dict())!=initial
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            probe_only=True,formal_parent=False,profile_rows=rows_ix,config=self.c),self.out/'gram_probe.pt')
        # Replay a saved state in the same architecture after perturbing weights.
        batch=collator([dataset[int(rows_ix[-1])]])
        batch={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in batch.items()}
        model.eval()
        with torch.no_grad():
            before=model(input_ids=batch['item_text_ids'],attention_mask=batch['item_text_masks'],labels=batch['target_ids']).logits.cpu()
            model.position_embedding.weight.add_(.2)
            cp=torch.load(self.out/'gram_probe.pt',map_location='cpu');model.load_state_dict(cp['state_dict']);del cp
            after=model(input_ids=batch['item_text_ids'],attention_mask=batch['item_text_masks'],labels=batch['target_ids']).logits.cpu()
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        gram_peak=self.peak()
        del opt,groups,before,after,batch;gc.collect();torch.cuda.empty_cache()
        self.status('C1双头接口和梯度检查，使用截断训练标签')
        from experiment.phase4.gcdh_p0 import CatalogDualHead
        c1=CatalogDualHead(model,tokenizer,self.ids,self.lex,torch.tensor(np.log(self.count[1:]+1),dtype=torch.float32)).to(self.device)
        opt=torch.optim.AdamW(c1.parameters(),lr=.0001,weight_decay=.01)
        self.reset_peak();c1_records=self.measure(c1,opt,collator,dataset,c1_ix,c1=True)
        c1_peak=self.peak()
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in c1.state_dict().items()},probe_only=True,
            formal_parent=False,profile_rows=c1_ix),self.out/'c1_probe.pt')
        seconds=np.mean([r['seconds'] for r in records[5:]])
        gram_cost=sum(r['seconds'] for r in gram[1:])/sum(r['rows'] for r in gram[1:])
        c1_cost=sum(r['seconds'] for r in c1_records[1:])/sum(r['rows'] for r in c1_records[1:])
        self.finish(dict(device=str(self.device),gpu_uuid=self.c['gpu_uuid'] if use_gpu else None,device_start=query,profile_only=True,
            sasrec=records,gram=gram,c1=c1_records,
            peaks_mib=dict(sasrec=sas_peak,gram=gram_peak,c1=c1_peak),checkpoint_replay_exact=True,
            estimates=dict(sasrec_200_epoch_training_hours=seconds*math.ceil(len(data)/128)*200/3600,
                gram_microbatch1_30_epoch_training_hours=gram_cost*len(self.train)*30/3600,
                c1_microbatch1_5_epoch_training_hours=c1_cost*len(self.c1_rows)*5/3600,
                device=str(self.device),gpu_timing_measured=use_gpu,
                limitations='Device-specific tiny quantile sample; different profile accumulation and cold models. CPU timings cannot estimate GPU cost. No validation, beam, checkpoint IO or formal-batch throughput included; not an end-to-end ETA.'),
            outputs_sha256={p.name:sha(p) for p in self.out.iterdir() if p.is_file() and p.name not in ('status.json','run.log')}))

    def measure(self,model,opt,collator,dataset,indices,c1):
        records=[];model.train();acc=self.c['gram_profile_accumulation']
        for start in range(0,len(indices),acc):
            group=indices[start:start+acc];opt.zero_grad(set_to_none=True);self.sync();tic=time.monotonic()
            total=0.;shapes=[]
            for i in group:
                sample=dataset[int(i)];batch=collator([sample])
                for k in ('item_text_ids','item_text_masks','target_ids'):batch[k]=batch[k].to(self.device)
                if c1:
                    loss,parts,logits=model.forward_loss(batch,torch.tensor([sample['target_item_id']-1],device=self.device),.2)
                    assert torch.isfinite(logits).all();del parts,logits
                else:loss=model(input_ids=batch['item_text_ids'],attention_mask=batch['item_text_masks'],labels=batch['target_ids']).loss
                assert torch.isfinite(loss);(loss/len(group)).backward();total+=float(loss.detach())
                shapes.append(list(batch['item_text_ids'].shape));del loss,batch
            gradients=[p.grad for p in model.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
            self.sync();records.append(dict(rows=len(group),seconds=time.monotonic()-tic,
                mean_loss=total/len(group),shapes=shapes,gradient_norm=float(norm)))
            self.status('C1小批量检查' if c1 else 'GRAM小批量检查',done=start+len(group),total=len(indices))
        return records


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','profile','cpu_smoke'));args=p.parse_args()
    job=Job(args.action)
    def timeout(*_):raise TimeoutError('Predeclared 900-second engineering budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(job.c['profile_timeout_seconds'])
    with (job.out/'run.log').open('x',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:job.prepare() if args.action=='prepare' else job.profile()
        except BaseException as e:
            job.status('检查未完成，保留输出',state='FAILED',error=repr(e));traceback.print_exc();raise
        finally:signal.alarm(0)


if __name__=='__main__':main()
