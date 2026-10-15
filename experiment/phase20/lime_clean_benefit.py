"""Clean-supervision scoring diagnostic. Frozen upstream, bounded two-domain run."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from experiment.phase20 import lime_decode_screen as base
from experiment.phase20.lime_clean_benefit_core import make_features

ROOT=base.ROOT
OUT=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
PROTOCOL=ROOT/'plan/第二十阶段/LIME干净监督_增量收益可学性检查_20260922.md'
MODULE='experiment.phase20.lime_clean_benefit'
GPUS={'Toys':'GPU-e2e37406-aaaa-da68-cf74-1a487b1c93e2','Beauty':'GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253'}


def rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]


def prepare():
    from experiment.phase20.lime_clean_gram_core import load_inputs
    OUT.mkdir(exist_ok=False)
    base.write(OUT/'status.json',dict(state='PREPARING',updated_at=base.stamp()))
    config=dict(domains=list(GPUS),gpus=GPUS,worker_timeout_seconds=5400,minimum_free_mib=12288,
                allocation_limit_gib=11,development_users=512,alpha=.3,beam_size=50,test_read=False,
                development_labels_used_for_fit=False,upstream_optimizer_steps=0)
    base.write(OUT/'config.json',config);shutil.copy2(PROTOCOL,OUT/'protocol.md')
    manifest={}
    def use(path):
        path=Path(path);rel=str(path.relative_to(ROOT));assert '/test/' not in rel and 'test.json' not in rel
        manifest[rel]=base.sha(path);return path
    counts={}
    for domain in GPUS:
        folder=OUT/domain;folder.mkdir()
        data=load_inputs(domain);manifest.update(data['manifest'])
        training=ROOT/f'artifacts/phase20/lime_parallel_{domain.lower()}_20260921_v1'
        assert base.read(use(training/'status.json'))['state']=='COMPLETED'
        summary=base.read(use(training/'gram/summary.json'))
        checkpoint=use(training/'gram/final.pt')
        assert base.sha(checkpoint)==summary['checkpoint_sha256']
        for rel,digest in summary['input_sha256'].items():
            if rel in data['manifest']:assert data['manifest'][rel]==digest,rel
        cfg=base.read(use(ROOT/f'experiment/phase18/config/s18_diff_gram_{domain.lower()}.json'))
        old=ROOT/cfg['prepared'];catalog=base.read(use(old/'catalog.json'))
        assert catalog['items'][1:]==data['ids']
        assert catalog['lexical_ids'][1:]==[data['lexical'][x] for x in data['ids']]
        catalog['passages']=['']+[data['dataset'].inputs[x] for x in data['ids']]
        base.write(folder/'catalog.json',catalog)
        design=ROOT/'artifacts/phase20/lime_clean_training_design_20260920_v1'/domain
        held=rows(use(design/'terminal_holdout.jsonl'))
        sequences={r['user_id']:r['history'] for r in rows(use(design/'upstream_sequences_without_terminal.jsonl'))}
        excluded={r['user_id'] for r in held}
        q=base.arrays(use(ROOT/f'artifacts/phase20/catalog_scope_20260919_v1/{domain}/queries.npz'))
        eligible={str(u) for u in q['users'][q['evaluation_indices']]}-excluded
        validation={r['user_id']:r for r in rows(use(old/'validation.jsonl'))}
        chosen=sorted(eligible,key=lambda u:hashlib.sha256(f'20260922|clean-benefit-evaluation|{domain}|{u}'.encode()).digest())[:512]
        assert len(chosen)==512
        samples=[]
        for r in held+[dict(validation[u],role='development') for u in chosen]:
            h=sequences[r['user_id']]
            assert r['history']==h[-20:]
            samples.append(dict(r,full_history=h,validation_index=len(samples)))
        assert len({s['user_id'] for s in samples})==len(samples)
        base.write(folder/'samples.json',samples);np.save(folder/'frequencies.npy',data['counts'])
        itemcf=ROOT/'artifacts/phase20'/('lime_clean_upstream_toys_20260920_v1/prepare/itemcf.json' if domain=='Toys' else 'lime_clean_beauty_preflight_20260920_v1/itemcf.json')
        cf=base.read(use(itemcf));assert cf['training_only']
        assert cf['num_training_interactions']==int(data['counts'].sum())
        sas=use(ROOT/f'artifacts/phase20/lime_clean_sasrec_{domain.lower()}_20260920_v1/sasrec.pt')
        embedding=use(ROOT/f'artifacts/phase20/lime_comparison_20260920_v1/recovery_v2/{domain}/embeddings.npz')
        config_d=dict(parent_checkpoint=str(checkpoint.relative_to(ROOT)),parent_sha256=base.sha(checkpoint),
            sasrec=str(sas.relative_to(ROOT)),itemcf=str(itemcf.relative_to(ROOT)),embedding=str(embedding.relative_to(ROOT)),
            original_config=cfg['original_config'],prepared=str(folder.relative_to(ROOT)))
        use(ROOT/cfg['original_config']);base.write(folder/'data_config.json',config_d)
        counts[domain]={role:sum(s['role']==role for s in samples) for role in ('gate_calibration','ranker_fit','development')}
        for name in ('catalog.json','samples.json','frequencies.npy','data_config.json'):use(folder/name)
        base.write(folder/'status.json',dict(state='PREPARED',counts=counts[domain]))
    paths=[PROTOCOL,OUT/'config.json',OUT/'protocol.md']
    paths+=list((ROOT/'experiment/phase20').glob('*lime_clean_benefit*'))
    paths += [ROOT/'experiment/phase20'/n for n in ('lime_decode_screen.py','lime_decode_screen_core.py','lime_clean_upstream_core.py',
        'lime_clean_gram_core.py','lime_comparison_core.py','source_competition_core.py','catalog_scope_core.py')]
    paths += list((ROOT/'GRAM/src').rglob('*.py'))
    paths += [ROOT/'experiment/phase18/core/diff_generation_cache.py',ROOT/'experiment/phase18/core/diff_data.py']
    paths += list((ROOT/'third_party/LIME-Rec/lime_rec').rglob('*.py'))
    paths += [ROOT/'third_party/LIME-Rec/scripts/evaluation/run_repeat_aware_gate.py']
    for path in paths:
        if path.is_file():
            use(path)
            if path.suffix in ('.py','.sh','.md'):
                dest=OUT/'sources'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    base.write(OUT/'manifest.json',dict(sha256=manifest,created_at=base.stamp()))
    base.write(OUT/'status.json',dict(state='PREPARED',counts=counts,updated_at=base.stamp(),test_read=False))
    print(json.dumps(counts),flush=True)


class Worker(base.Worker):
    def __init__(self,domain):
        import torch
        sys.path.insert(0,str(ROOT/'GRAM/src'));sys.path.insert(0,str(ROOT/'third_party/LIME-Rec'))
        from experiment.phase18.core.diff_data import DiffDataset
        from experiment.phase20.lime_clean_upstream_core import new_gram,runtime_args
        from experiment.phase20.lime_clean_gram_core import T5,SPECS
        from experiment.phase20.lime_decode_screen_core import ExpertTrie
        from processor import CollatorGRAM
        from utils import generation_trie
        self.domain=domain;self.folder=OUT/domain;self.started=time.monotonic();self.completed=0;self.seconds=[]
        self.stage='load';self.config=base.read(OUT/'config.json');self.cfg=base.read(self.folder/'data_config.json')
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2023)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
        self.samples=base.read(self.folder/'samples.json');self.status()
        self.catalog=base.read(self.folder/'catalog.json');self.data=DiffDataset(self.samples,self.catalog)
        self.model,self.tokenizer=new_gram(T5)
        args=runtime_args();args.hierarchical_id_type=SPECS[domain]['lexical'];self.collator=CollatorGRAM(self.tokenizer,args=args,mode='train')
        saved=torch.load(ROOT/self.cfg['parent_checkpoint'],map_location='cpu')
        assert saved['position']['epoch']==31
        self.model.load_state_dict(saved['state_dict'],strict=True);del saved
        self.model.eval().requires_grad_(False).to(self.device)
        self.index=ExpertTrie(self.catalog['decoder_paths'],range(1,len(self.catalog['items'])))
        self.max_length=max(map(len,self.catalog['decoder_paths']))
        self.trie=generation_trie.Trie(self.catalog['decoder_paths']);self.prefix_fn=generation_trie.prefix_allowed_tokens_fn(self.trie)
        self.initial_digest=self.digest()
        self.users_completed=0;self.total_queries=sum(s['role']!='gate_calibration' for s in self.samples)

    def status(self,state='RUNNING',**kw):
        super().status(state,users_completed=getattr(self,'users_completed',0),users_total=getattr(self,'total_queries',None),**kw)

    def signals(self):
        import torch
        from lime_rec.models import SASRec
        from lime_rec.experts import ItemCFExpert,LLMSemanticExpert
        from lime_rec.evaluation import normalize_score_rows
        from scripts.evaluation.run_repeat_aware_gate import RepeatAwareGate,features
        from experiment.phase20.lime_comparison_core import left_pad,semantic_users
        self.stage='clean_expert_scores';self.status()
        n=len(self.samples);v=len(self.catalog['items'])-1;source=self.catalog['items'][1:]
        histories=[s['full_history'] for s in self.samples]
        e=base.arrays(ROOT/self.cfg['embedding']);assert e['item_ids'].tolist()==source
        emb=e['embeddings'];semantic=LLMSemanticExpert(source,{},embeddings=emb)
        user_emb=semantic_users(histories,emb)
        cf=ItemCFExpert(str(ROOT/self.cfg['itemcf']),source)
        checkpoint=torch.load(ROOT/self.cfg['sasrec'],map_location='cpu')
        assert checkpoint['train_complete'] and checkpoint['epoch']==200 and checkpoint['training_only']
        sas=SASRec(v,64,50,2,2,0.,architecture='legacy').to(self.device)
        sas.load_state_dict(checkpoint['state_dict'],strict=True);sas.eval();del checkpoint
        scores=np.lib.format.open_memmap(self.folder/'experts.npy',mode='w+',dtype='float32',shape=(n,3,v))
        hist=np.zeros((n,v),bool);lengths=np.array(list(map(len,histories)),dtype=np.float32)
        with torch.no_grad():
            for start in range(0,n,64):
                stop=min(start+64,n);hs=histories[start:stop]
                raw_s=sas.predict(torch.from_numpy(left_pad(hs)).to(self.device))[:,1:].cpu().numpy()
                raw_c=np.stack([cf.score(self.samples[start+i]['user_id'],[source[x-1] for x in h]) for i,h in enumerate(hs)])
                raw_b=user_emb[start:stop]@semantic.item_embeddings_norm.T
                scores[start:stop]=np.stack([normalize_score_rows(a) for a in (raw_s,raw_c,raw_b)],axis=1)
                for j,h in enumerate(hs):hist[start+j,np.asarray(h,dtype=np.int64)-1]=True
                if start%512==0:self.status(score_rows=stop,score_total=n)
        scores.flush();del sas,cf;gc.collect();torch.cuda.empty_cache()
        # Independent author scorer parity on deterministic first/middle/last rows.
        for i in (0,n//2,n-1):
            expected=semantic.score(self.samples[i]['user_id'],[source[x-1] for x in histories[i]])
            np.testing.assert_allclose(normalize_score_rows(expected[None])[0],scores[i,2],atol=5e-6,rtol=1e-5)
        gate_ix=np.array([i for i,s in enumerate(self.samples) if s['role']=='gate_calibration'])
        gate_ix=np.array(sorted(gate_ix,key=lambda i:self.samples[i]['user_id']))
        assert len(gate_ix)==dict(Toys=721,Beauty=775)[self.domain]
        self.stage='fit_clean_LIME_gate';self.status(gate_users=len(gate_ix))
        torch.manual_seed(0);gate=RepeatAwareGate(14,[.6,.15,.25],.1).to(self.device)
        opt=torch.optim.AdamW(gate.parameters(),lr=.02,weight_decay=.001);curve=[]
        def batch(ix):
            return (torch.from_numpy(np.array(scores[ix])).to(self.device),torch.from_numpy(lengths[ix]).to(self.device),
                    torch.from_numpy(hist[ix]).to(self.device))
        for epoch in range(4):
            losses=[]
            for start in range(0,len(gate_ix),256):
                ix=gate_ix[start:start+256];s,l,h=batch(ix);w,p=gate(features(s,l,h))
                y=torch.tensor([self.samples[i]['target']-1 for i in ix],device=self.device)
                loss=torch.nn.functional.cross_entropy(20*((w[:,:,None]*s).sum(1)-p[:,None]*h),y)
                assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
            curve.append(losses);self.status(gate_epoch=epoch+1,loss=float(np.mean(losses)))
        torch.save(dict(state_dict={k:v.cpu() for k,v in gate.state_dict().items()},fit_indices=gate_ix,
            fit_users=[self.samples[i]['user_id'] for i in gate_ix],curve=curve,epochs=4,
            development_labels_used_for_fit=False),self.folder/'gate.pt')
        fused=np.lib.format.open_memmap(self.folder/'lime.npy',mode='w+',dtype='float32',shape=(n,v))
        all_features=[];all_weights=[];all_penalties=[];gate.eval()
        with torch.no_grad():
            for start in range(0,n,64):
                ix=np.arange(start,min(start+64,n));s,l,h=batch(ix);f=features(s,l,h);w,p=gate(f)
                fused[ix]=((w[:,:,None]*s).sum(1)-p[:,None]*h).cpu().numpy()
                all_features.append(f.cpu().numpy());all_weights.append(w.cpu().numpy());all_penalties.append(p.cpu().numpy())
        fused.flush();self.gate_features=np.concatenate(all_features)
        np.savez_compressed(self.folder/'gate_prediction.npz',features=self.gate_features,weights=np.concatenate(all_weights),penalty=np.concatenate(all_penalties))
        self.experts=scores;self.fused=fused
        del gate,opt,hist;gc.collect();torch.cuda.empty_cache()

    def run(self):
        import torch
        self.signals();counts=np.load(self.folder/'frequencies.npy')
        self.stage='native_cache_smoke';self.status();checks=[]
        js=[i for i,s in enumerate(self.samples) if s['role']=='ranker_fit'][:3]
        for j in js:
            ctx=self.context(j);plain=self.decode(ctx,native_entry=True,record=False);zero=self.decode(ctx)
            assert plain['paths']==zero['paths'];np.testing.assert_array_equal(plain['scores'],zero['scores'])
            q=self.index.conditional_logs(self.fused[j]);zeroq=self.decode(ctx,q,0.)
            assert zeroq['paths']==zero['paths'];np.testing.assert_array_equal(zeroq['scores'],zero['scores'])
            checks.append(dict(user=self.samples[j]['user_id'],native_wrapper_zero_exact=True,cross_attention_cache_reuse=False))
        base.write(self.folder/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage='cache_native_paths_and_matched_guidance';self.status()
        arrays={k:[] for k in ('users','targets','roles','sample_indices','ids','native','mixed','x','query_x','valid')}
        with (self.folder/'candidates.jsonl').open('x') as log:
            for j,sample in enumerate(self.samples):
                if sample['role']=='gate_calibration':continue
                ctx=self.context(j);q=self.index.conditional_logs(self.fused[j]);native=self.decode(ctx)
                mixed=native['guide'].rescore(native['paths'],q,.3)
                record=dict(sample_index=j,user_id=sample['user_id'],role=sample['role'],target=sample['target'],
                    ids=native['ids'],paths=native['paths'],native=native['scores'].tolist(),mixed=mixed.tolist())
                if sample['role']=='development':
                    guided=self.decode(ctx,q,.3)
                    record['guided']=dict(ids=guided['ids'],paths=guided['paths'],scores=guided['scores'].tolist(),
                        native=guided['guide'].rescore(guided['paths']).tolist())
                x,qx,valid=make_features(record['ids'],record['native'],mixed,self.experts[j],self.fused[j],counts,
                    len(sample['full_history']),self.gate_features[j],sample['history'])
                values=(sample['user_id'],sample['target'],sample['role'],j,record['ids'],record['native'],mixed,x,qx,valid)
                for key,value in zip(arrays,values):arrays[key].append(value)
                log.write(json.dumps(record)+'\n');log.flush();self.users_completed+=1
                if self.users_completed%25==0 or self.users_completed==1:self.status()
        np.savez_compressed(self.folder/'data.npz',**{k:np.asarray(v) for k,v in arrays.items()})
        assert self.digest()==self.initial_digest
        base.write(self.folder/'cache_summary.json',dict(state='COMPLETED',users=self.users_completed,
            decodes=self.completed,model_unchanged=True,model_digest=self.initial_digest,
            seconds=time.monotonic()-self.started,test_read=False,upstream_optimizer_steps=0))
        self.stage='cache_complete';self.status(state='COMPLETED')


def worker(domain):
    try:
        manifest=base.read(OUT/'manifest.json')['sha256']
        for rel,digest in manifest.items():assert base.sha(ROOT/rel)==digest,rel
        job=Worker(domain);job.run()
    except BaseException as e:
        (OUT/domain/'failure.txt').write_text(traceback.format_exc())
        base.write(OUT/domain/'status.json',dict(state='FAILED',error=repr(e),pid=os.getpid(),updated_at=base.stamp()))
        raise


def supervise():
    cfg=base.read(OUT/'config.json');processes={};started=time.monotonic()
    available=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    free={line.split(',')[0].strip():int(line.split(',')[1]) for line in available.splitlines()}
    for domain,gpu in GPUS.items():assert free[gpu]>=cfg['minimum_free_mib'],(domain,free[gpu])
    for domain,gpu in GPUS.items():
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
        with (OUT/domain/'worker.log').open('x') as log:
            processes[domain]=subprocess.Popen([sys.executable,'-m',MODULE,'worker','--domain',domain],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    base.write(OUT/'launch.json',dict(pid=os.getpid(),workers={d:p.pid for d,p in processes.items()},started_at=base.stamp()))
    while any(p.poll() is None for p in processes.values()):
        states={d:base.read(OUT/d/'status.json') for d in GPUS}
        base.write(OUT/'status.json',dict(state='RUNNING',stage='clean_score_cache',domains=states,
            elapsed_seconds=time.monotonic()-started,pid=os.getpid(),updated_at=base.stamp()))
        if time.monotonic()-started>cfg['worker_timeout_seconds']:
            for p in processes.values():
                if p.poll() is None:p.terminate()
            raise TimeoutError('90 minute fixed cache budget exceeded')
        time.sleep(15)
    codes={d:p.wait() for d,p in processes.items()}
    ok=all(code==0 for code in codes.values())
    base.write(OUT/'status.json',dict(state='CACHE_COMPLETED' if ok else 'FAILED',exit_codes=codes,updated_at=base.stamp()))
    if ok:
        with (OUT/'analysis.log').open('x') as log:
            subprocess.run([sys.executable,'-m','experiment.phase20.analyze_lime_clean_benefit'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','launch','supervise','worker']);parser.add_argument('--domain',choices=list(GPUS));args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='worker':worker(args.domain)
    elif args.action=='supervise':
        try:supervise()
        except BaseException as e:
            (OUT/'failure.txt').write_text(traceback.format_exc());base.write(OUT/'status.json',dict(state='FAILED',error=repr(e),updated_at=base.stamp()));raise
    else:
        assert base.read(OUT/'status.json')['state']=='PREPARED'
        with (OUT/'supervisor.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-m',MODULE,'supervise'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,output=str(OUT))),flush=True)


if __name__=='__main__':main()
