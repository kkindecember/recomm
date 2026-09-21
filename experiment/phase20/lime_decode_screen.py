"""Bounded, frozen original-GRAM/LIME decoding screen; no test or fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/phase20/lime_decode_screen_20260921_v1'
PROTOCOL = ROOT/'plan/第二十阶段/LIME信号引导原GRAM生成_限时筛查协议_20260921.md'
DOMAINS = ('Toys', 'Beauty')
GPU = {'Toys': 'GPU-cfd866c2-ccc1-8a3c-00b8-594d7a8cd059',
       'Beauty': 'GPU-7952ad6e-2db7-fb2e-581b-34be56287640'}
ALPHAS = (.1, .3)
SIGNALS = ('sasrec', 'lime')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def arrays(path):
    with np.load(path, allow_pickle=False) as x: return {k:x[k] for k in x.files}


def prepare():
    import torch
    from experiment.phase20.catalog_scope_core import topk
    sys.path.insert(0, str(ROOT/'third_party/LIME-Rec'))
    from lime_rec.evaluation import normalize_score_rows
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp(), test_read=False))
    config = dict(domains=DOMAINS, gpus=GPU, alphas=ALPHAS, tuning_users=128,
        evaluation_users=512, smoke_users=3, score_scale=20., seed=20260921,
        beam_size=50, length_penalty=1., worker_timeout_seconds=7200,
        resource_wait_seconds=600, minimum_free_mib=11264, allocation_limit_gib=11,
        no_model_training=True, test_read=False)
    write(OUT/'config.json', config)
    manifest = {}
    def use(p):
        p=Path(p); rel=str(p.relative_to(ROOT))
        assert '/test/' not in rel and 'test.json' not in rel
        if rel not in manifest: manifest[rel] = sha(p)
        return p
    for domain in DOMAINS:
        folder=OUT/domain; folder.mkdir()
        cfg_path=ROOT/f'experiment/phase18/config/s18_diff_gram_{domain.lower()}.json'
        cfg=read(use(cfg_path)); prepared=ROOT/cfg['prepared']
        assert sha(use(ROOT/cfg['parent_checkpoint'])) == cfg['parent_sha256']
        use(ROOT/cfg['original_config'])
        catalog=read(use(prepared/'catalog.json'))
        validation=[json.loads(l) for l in use(prepared/'validation.jsonl').read_text().splitlines()]
        by_user={r['user_id']:i for i,r in enumerate(validation)}
        query=arrays(use(ROOT/f'artifacts/phase20/catalog_scope_20260919_v1/{domain}/queries.npz'))
        recovery=ROOT/f'artifacts/phase20/lime_comparison_20260920_v1/recovery_v2/{domain}'
        old=arrays(use(recovery/'evaluation_seed0_cal2048.npz'))
        gate=torch.load(use(recovery/'gate_seed0_cal2048.pt'), map_location='cpu')
        np.testing.assert_array_equal(np.sort(gate['fit_indices']),query['calibration_indices'])
        np.testing.assert_array_equal(old['users'],query['users'][old['indices']])
        np.testing.assert_array_equal(old['targets'],query['targets'][old['indices']])
        assert set(old['indices']).isdisjoint(gate['fit_indices'])
        records=[json.loads(l) for l in use(ROOT/f'artifacts/phase20/nearest_comparison_20260918/{domain}.items.jsonl').read_text().splitlines()]
        assert [r['source_item_id'] for r in records] == catalog['items'][1:]
        order=sorted(range(len(old['users'])), key=lambda j: hashlib.sha256(
            f'20260921|lime-decode|{domain}|{old["users"][j]}'.encode()).digest())
        chosen=order[:643]
        samples=[]
        for j,pos in enumerate(chosen):
            qi=int(old['indices'][pos]); user=str(old['users'][pos]); vi=by_user[user]
            row=validation[vi]; h=query['history'][qi]; h=h[h>0].tolist()
            assert row['history']==h and row['target']==int(old['targets'][pos])
            samples.append(dict(user_id=user, target=row['target'], history=h, validation_index=vi,
                query_index=qi, evaluation_cache_index=pos,
                split='smoke' if j<3 else 'tuning' if j<131 else 'evaluation'))
        assert len(set(r['user_id'] for r in samples)) == len(samples)
        ix=np.asarray([r['query_index'] for r in samples]); weights=old['weights'][chosen].astype('float32')
        penalty=old['penalty'][chosen].astype('float32')
        np.testing.assert_allclose(weights.sum(1),1.,atol=2e-7)
        paths=[recovery/'sasrec_scores_seed0.npy',
            ROOT/f'artifacts/phase20/lime_comparison_20260920_v1/{domain}/itemcf_scores.npy',
            recovery/'semantic_scores.npy']
        raw=[np.asarray(np.load(use(p),mmap_mode='r')[ix]) for p in paths]
        normalized=np.stack([normalize_score_rows(s) for s in raw],axis=1)
        # Match author's FP32 fusion order and verify against saved LIME results.
        fused=(torch.from_numpy(weights)[:,:,None]*torch.from_numpy(normalized)).sum(1)
        bank=arrays(use(ROOT/f'artifacts/phase20/nearest_comparison_20260918/{domain}.train_sequences.npz'))
        ui={str(u):i for i,u in enumerate(bank['users'])}
        for j,r in enumerate(samples):
            k=ui[r['user_id']]; h=bank['items'][bank['offsets'][k]:bank['offsets'][k+1]]
            assert h[-20:].tolist()==r['history']
            fused[j,np.unique(h)-1]-=penalty[j]
        fused=fused.numpy(); masked=fused.copy()
        for j,r in enumerate(samples): masked[j,np.asarray(r['history'])-1]=-np.inf
        ids=np.arange(1,len(catalog['items']))
        tops=topk(masked,ids)
        ai=old['arm_names'].tolist().index('LIME')
        np.testing.assert_array_equal(tops,old['top50'][ai,chosen])
        np.savez_compressed(folder/'signals.npz', sasrec=normalized[:,0], lime=fused,
            lime_top50=tops, weights=weights, penalty=penalty)
        write(folder/'samples.json',samples)
        write(folder/'data_config.json',cfg)
        write(folder/'preflight.json',dict(samples=len(samples), lime_top50_exact=True,
            gate_fit_disjoint=True, catalog_and_history_aligned=True, parent_sha256=cfg['parent_sha256']))
        for name in ('signals.npz','samples.json','data_config.json','preflight.json'): use(folder/name)
    for p in (PROTOCOL,Path(__file__),ROOT/'experiment/phase20/lime_decode_screen_core.py',
              ROOT/'experiment/phase20/test_lime_decode_screen.py',ROOT/'experiment/phase20/run_lime_decode_screen.sh'):
        use(p)
    use(OUT/'config.json')
    # Freeze only loaded/intended dependencies, never change existing running jobs.
    for p in list((ROOT/'GRAM/src').rglob('*.py'))+[
            ROOT/'experiment/phase17/core/full_latte_gram_backend.py',
            ROOT/'experiment/phase18/core/diff_data.py',
            ROOT/'experiment/phase20/guided_decoding_core.py',
            ROOT/'experiment/phase20/catalog_scope_core.py']:
        use(p)
    write(OUT/'manifest.json',dict(sha256=manifest, created_at=stamp(), config=config))
    shutil.copy2(PROTOCOL,OUT/'protocol.md')
    write(OUT/'status.json',dict(state='PREPARED',updated_at=stamp(),domains=DOMAINS,test_read=False))
    print(json.dumps(dict(prepared=str(OUT),inputs=len(manifest))),flush=True)


class Worker:
    def __init__(self, domain):
        import torch
        from experiment.phase18.core.diff_data import DiffDataset, native_collator
        from experiment.phase17.core.full_latte_gram_backend import create_fresh_gram_model
        from experiment.phase20.lime_decode_screen_core import ExpertTrie
        self.domain=domain; self.folder=OUT/domain; self.started=time.monotonic()
        self.config=read(OUT/'config.json'); self.cfg=read(self.folder/'data_config.json')
        self.completed=0; self.seconds=[]; self.stage='load'
        torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.manual_seed(2023)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory,0)
        self.device=torch.device('cuda:0'); self.status()
        self.samples=read(self.folder/'samples.json'); self.signals=arrays(self.folder/'signals.npz')
        self.catalog=read(ROOT/self.cfg['prepared']/'catalog.json')
        self.data=DiffDataset.load(ROOT/self.cfg['prepared'],'validation',self.catalog)
        self.tokenizer,self.collator=native_collator(ROOT,self.cfg)
        self.model=create_fresh_gram_model(ROOT,'G0_GRAM_B0_FRESH',self.tokenizer,seed=2023)
        assert sha(ROOT/self.cfg['parent_checkpoint'])==self.cfg['parent_sha256']
        state=torch.load(ROOT/self.cfg['parent_checkpoint'],map_location='cpu')
        self.model.resize_token_embeddings(state['shared.weight'].size(0))
        self.model.load_state_dict(state,strict=True); del state
        self.model.eval().requires_grad_(False).to(self.device)
        assert not self.model.config.cf0_enabled and not self.model.config.hi_gram_enabled and not self.model.config.s17_modules
        self.index=ExpertTrie(self.catalog['decoder_paths'],range(1,len(self.catalog['items'])))
        self.max_length=max(map(len,self.catalog['decoder_paths']))
        from utils import generation_trie
        self.trie=generation_trie.Trie(self.catalog['decoder_paths'])
        self.prefix_fn=generation_trie.prefix_allowed_tokens_fn(self.trie)
        self.initial_digest=self.digest()
        self.loaded_source={}
        for module in list(sys.modules.values()):
            name=getattr(module,'__file__',None)
            if name and name.endswith('.py'):
                path=Path(name).resolve()
                if ROOT in path.parents: self.loaded_source[str(path.relative_to(ROOT))]=sha(path)
        write(self.folder/'loaded_sources.json',self.loaded_source)

    def digest(self):
        h=hashlib.sha256()
        for k,v in self.model.state_dict().items():
            h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def status(self,state='RUNNING',**kw):
        import torch
        write(self.folder/'status.json',dict(state=state,stage=self.stage,pid=os.getpid(),
            domain=self.domain,updated_at=stamp(),elapsed_seconds=round(time.monotonic()-self.started,1),
            completed_decodes=self.completed,mean_decode_seconds=float(np.mean(self.seconds[-32:])) if self.seconds else None,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,test_read=False,optimizer_steps=0,**kw))

    def context(self, j, override_index=None):
        import torch
        from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
        sample=self.data[self.samples[j]['validation_index'] if override_index is None else override_index]
        sample=dict(sample,output=self.catalog['lexical_ids'][1])
        # Targets/history labels from the collator are not passed to the model.
        batch=self.collator([sample]); ids=batch['item_text_ids'].to(self.device); mask=batch['item_text_masks'].to(self.device)
        with torch.no_grad():
            self.model.encoder.n_passages=ids.size(1)
            self.model.encoder.set_migration_context(None,None)
            encoded=self.model.encoder(input_ids=ids.reshape(1,-1),attention_mask=mask.reshape(1,-1),return_dict=True)[0]
        return dict(ids=ids,mask=mask,hidden=encoded)

    def decode(self, ctx, q=None, alpha=0., native_entry=False, record=True):
        import torch
        from transformers import LogitsProcessorList
        from model.gram import GRAM
        from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
        from experiment.phase20.lime_decode_screen_core import ProbabilityGuide
        if time.monotonic()-self.started>self.config['worker_timeout_seconds']: raise TimeoutError('Domain time budget')
        begin=time.monotonic(); guide=ProbabilityGuide(self.index,q,alpha)
        common=dict(max_length=self.max_length,prefix_allowed_tokens_fn=self.prefix_fn,
            num_beams=50,num_return_sequences=50,return_dict_in_generate=True,output_scores=True,length_penalty=1.)
        if record: common['logits_processor']=LogitsProcessorList([guide])
        with torch.no_grad():
            if native_entry:
                output=self.model.generate(input_ids=ctx['ids'],attention_mask=ctx['mask'],**common)
            else:
                # Fresh wrapper: HF expands/mutates encoder_outputs during beam search.
                encoded=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=ctx['hidden'])
                output=super(GRAM,self.model).generate(input_ids=ctx['ids'].reshape(1,-1),
                    attention_mask=ctx['mask'].reshape(1,-1),encoder_outputs=encoded,**common)
        paths=[p[:p.index(1)+1] for p in output.sequences.cpu().tolist()]
        ids=[self.index.path_to_item[tuple(p)] for p in paths]
        scores=output.sequences_scores.detach().cpu().double().numpy()
        assert len(set(ids))==50 and np.isfinite(scores).all()
        if record: np.testing.assert_allclose(guide.rescore(paths,q,alpha),scores,atol=2e-6,rtol=1e-5)
        del output
        torch.cuda.synchronize(); self.seconds.append(time.monotonic()-begin); self.completed+=1
        return dict(ids=ids,paths=paths,scores=scores,guide=guide)

    def smoke(self):
        import torch
        self.stage='smoke'; self.status(); checks=[]
        for j in range(3):
            ctx=self.context(j)
            native=self.decode(ctx,native_entry=True,record=False)
            zero=self.decode(ctx)
            assert native['paths']==zero['paths']
            np.testing.assert_array_equal(native['scores'],zero['scores'])
            q=self.index.conditional_logs(self.signals['lime'][j])
            guided=self.decode(ctx,q,.3)
            # Zero strength with expert supplied must remain exact.
            zero2=self.decode(ctx,q,0.)
            assert zero2['paths']==zero['paths']
            np.testing.assert_array_equal(zero2['scores'],zero['scores'])
            checks.append(dict(user_id=self.samples[j]['user_id'],native_cache_exact=True,
                zero_strength_exact=True,completed_path_scores_reconstructed=True))
            del ctx,native,zero,zero2,guided
        longest=max(range(len(self.data)),key=lambda i:len(self.data.records[i]['history']))
        ctx=self.context(0,longest); self.decode(ctx); del ctx
        assert self.digest()==self.initial_digest
        estimate=max(self.seconds)*2176
        write(self.folder/'smoke.json',dict(checks=checks,max_history_checked=True,
            projected_formal_seconds=estimate,decode_seconds=self.seconds,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,initial_digest=self.initial_digest))
        if estimate>self.config['worker_timeout_seconds']-300:
            raise TimeoutError(f'Profile exceeds formal budget: {estimate}')
        self.status(projected_formal_seconds=estimate)

    def one(self,j,choices,split):
        from experiment.phase20.lime_decode_screen_core import ordered,target_rank
        row=self.samples[j]; ctx=self.context(j); native=self.decode(ctx)
        history=row['history']; target=row['target']; arms={}; saved={}
        def add(name,ids,score):
            order=ordered(ids,score,history)[:50]
            arms[name]=dict(items=order,rank=target_rank(order,target))
            saved[name]=dict(ids=list(map(int,ids)),scores=np.asarray(score).tolist())
        add('GRAM',native['ids'],native['scores'])
        all_ids=np.arange(1,len(self.catalog['items']))
        add('LIME',all_ids,self.signals['lime'][j])
        add('LIME_after_GRAM',native['ids'],self.signals['lime'][j][np.asarray(native['ids'])-1])
        for signal,alphas in choices.items():
            q=self.index.conditional_logs(self.signals[signal][j])
            for alpha in alphas:
                label=signal if split=='evaluation' else f'{signal}_{alpha}'
                if alpha:
                    out=self.decode(ctx,q,alpha)
                    add(label,out['ids'],out['scores'])
                    late=native['guide'].rescore(native['paths'],q,alpha)
                    add(label+'_late',native['ids'],late)
                else:
                    arms[label]=dict(arms['GRAM']); arms[label+'_late']=dict(arms['GRAM'])
                    saved[label]=saved['GRAM']; saved[label+'_late']=saved['GRAM']
        del ctx
        # Full-catalog LIME scores are already frozen in signals.npz.
        del saved['LIME']
        return dict(user_id=row['user_id'],target=target,history=history,sample_index=j,
            split=split,arms=arms,candidate_scores=saved,baseline_paths=native['paths'])

    def run(self):
        from experiment.phase20.lime_decode_screen_core import metrics,contrast
        self.smoke(); self.stage='tuning'; self.status()
        tuning=[]
        with (self.folder/'tuning.jsonl').open('x') as handle:
            for j in range(3,131):
                row=self.one(j,{s:ALPHAS for s in SIGNALS},'tuning'); tuning.append(row)
                handle.write(json.dumps(row)+'\n'); handle.flush()
                if (j-2)%8==0: self.status(completed_users=j-2,total_users=128)
        choices={}; table={}
        for s in SIGNALS:
            table[s]={str(a):metrics([r['arms']['GRAM' if a==0 else f'{s}_{a}']['rank'] for r in tuning])
                      for a in (0.,)+ALPHAS}
            choices[s]=max((0.,)+ALPHAS,key=lambda a:(table[s][str(a)]['N10'],-a))
        write(self.folder/'selection.json',dict(selected_alpha=choices,tuning=table,selected_at=stamp(),
            criterion='N10; exact ties choose smaller alpha; zero is baseline fallback'))
        self.stage='evaluation'; self.status(selection=choices)
        evaluation=[]
        with (self.folder/'evaluation.jsonl').open('x') as handle:
            for j in range(131,len(self.samples)):
                row=self.one(j,{s:[choices[s]] for s in SIGNALS},'evaluation'); evaluation.append(row)
                handle.write(json.dumps(row)+'\n'); handle.flush()
                if (j-130)%8==0: self.status(completed_users=j-130,total_users=512,selection=choices)
        names=list(evaluation[0]['arms'])
        ranks={s:np.asarray([r['arms'][s]['rank'] for r in evaluation]) for s in names}
        comparisons={s+'_vs_GRAM':contrast(ranks[s],ranks['GRAM']) for s in names if s!='GRAM'}
        comparisons.update({s+'_vs_late':contrast(ranks[s],ranks[s+'_late']) for s in SIGNALS})
        comparisons['lime_vs_sasrec']=contrast(ranks['lime'],ranks['sasrec'])
        comparisons['lime_vs_LIME_after_GRAM']=contrast(ranks['lime'],ranks['LIME_after_GRAM'])
        comparisons['lime_vs_LIME']=contrast(ranks['lime'],ranks['LIME'])
        assert self.digest()==self.initial_digest
        for p,digest in self.loaded_source.items(): assert sha(ROOT/p)==digest,('changed source',p)
        write(self.folder/'summary.json',dict(domain=self.domain,selected_alpha=choices,
            metrics={s:metrics(rr) for s,rr in ranks.items()},comparisons=comparisons,
            elapsed_seconds=time.monotonic()-self.started,completed_decodes=self.completed,
            model_unchanged=True,test_read=False,development_diagnostic=True))
        self.stage='done'; self.status('COMPLETED',selection=choices)


def gpu_free(uuid):
    raw=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
    return int(raw.strip())


def supervise():
    config=read(OUT/'config.json'); active={}; done={}; started=time.monotonic()
    pending=set(DOMAINS)
    while pending or active:
        for domain in list(pending):
            if gpu_free(GPU[domain])>=config['minimum_free_mib']:
                log=(OUT/domain/'worker.log').open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU[domain],PYTHONUNBUFFERED='1',
                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                p=subprocess.Popen([sys.executable,'-m','experiment.phase20.lime_decode_screen','worker','--domain',domain],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                log.close(); active[domain]=(p,time.monotonic()); pending.remove(domain)
            elif time.monotonic()-started>config['resource_wait_seconds']:
                done[domain]='RESOURCE_TIMEOUT'; pending.remove(domain)
                write(OUT/domain/'status.json',dict(state='RESOURCE_TIMEOUT',updated_at=stamp()))
        for domain,(p,begin) in list(active.items()):
            rc=p.poll()
            if rc is not None:
                done[domain]=rc; del active[domain]
            elif time.monotonic()-begin>config['worker_timeout_seconds']+180:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
                done[domain]='TIMEOUT'; del active[domain]
                write(OUT/domain/'status.json',dict(state='TIMEOUT',updated_at=stamp()))
        write(OUT/'status.json',dict(state='RUNNING' if active or pending else
            'COMPLETED' if all(v==0 for v in done.values()) else 'INCOMPLETE',updated_at=stamp(),
            supervisor_pid=os.getpid(),active={d:p.pid for d,(p,_) in active.items()},pending=sorted(pending),
            completed=done,test_read=False,elapsed_seconds=time.monotonic()-started))
        if pending or active:time.sleep(5)
    if all(v==0 for v in done.values()):
        subprocess.run([sys.executable,'-m','experiment.phase20.lime_decode_screen','verify'],cwd=ROOT,check=True)


def verify():
    from experiment.phase20.lime_decode_screen_core import ordered,target_rank,metrics,contrast
    checks={}; summaries={}
    for domain in DOMAINS:
        folder=OUT/domain; samples=read(folder/'samples.json'); sig=arrays(folder/'signals.npz')
        selection=read(folder/'selection.json'); tuning=[json.loads(l) for l in (folder/'tuning.jsonl').read_text().splitlines()]
        evaluation=[json.loads(l) for l in (folder/'evaluation.jsonl').read_text().splitlines()]
        assert len(tuning)==128 and len(evaluation)==512
        assert set(r['user_id'] for r in tuning).isdisjoint(r['user_id'] for r in evaluation)
        for signal in SIGNALS:
            scores={a:metrics([r['arms']['GRAM' if a==0 else f'{signal}_{a}']['rank'] for r in tuning])['N10'] for a in (0.,)+ALPHAS}
            assert selection['selected_alpha'][signal]==max(scores,key=lambda a:(scores[a],-a))
        count=0
        for row in tuning+evaluation:
            j=row['sample_index']; assert samples[j]['user_id']==row['user_id'] and samples[j]['target']==row['target']
            for name,values in row['candidate_scores'].items():
                ids=ordered(values['ids'],values['scores'],row['history'])[:50]
                assert ids==row['arms'][name]['items']
                assert target_rank(ids,row['target'])==row['arms'][name]['rank']; count+=1
            assert row['arms']['LIME']['items']==sig['lime_top50'][j].tolist()
            assert row['arms']['LIME']['rank']==target_rank(sig['lime_top50'][j].tolist(),row['target'])
        summary=read(folder/'summary.json')
        for name,expected in summary['metrics'].items():
            actual=metrics([r['arms'][name]['rank'] for r in evaluation]); assert actual==expected
        summaries[domain]=summary
        checks[domain]=dict(predictions_recomputed=count,selection_recomputed=True,metrics_exact=True,
            cohort_disjoint=True,model_unchanged=summary['model_unchanged'])
    changed=[]
    for name,digest in read(OUT/'manifest.json')['sha256'].items():
        if sha(ROOT/name)!=digest: changed.append(name)
    assert not changed,changed
    signal=all(s['comparisons']['lime_vs_GRAM']['relative_N10']>=.01 and s['comparisons']['lime_vs_GRAM']['net_H10']>=0 for s in summaries.values())
    search_value=all(s['comparisons']['lime_vs_late']['delta_N10']>=0 for s in summaries.values()) and any(s['comparisons']['lime_vs_late']['delta_N10']>0 for s in summaries.values())
    decision='PROMISING_DEVELOPMENT_SIGNAL' if signal and search_value else 'PERFORMANCE_ONLY_SIGNAL' if signal else 'NO_CONTINUATION_SIGNAL'
    write(OUT/'verification.json',dict(verified_at=stamp(),checks=checks,all_input_hashes_unchanged=True))
    write(OUT/'summary.json',dict(decision=decision,domains=summaries,search_additional_value=search_value,
        caveat='Single-seed reused-development screen, not independent test or proof of null effect.',test_read=False))
    print(json.dumps(dict(decision=decision,verification=str(OUT/'verification.json')),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','supervise','worker','verify']);p.add_argument('--domain',choices=DOMAINS)
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='launch':
        assert read(OUT/'status.json')['state']=='PREPARED'
        assert not (OUT/'launch.json').exists()
        with (OUT/'supervisor.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m','experiment.phase20.lime_decode_screen','supervise'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(OUT/'launch.json',dict(pid=child.pid,started_at=stamp()))
        print(json.dumps(dict(pid=child.pid,status=str(OUT/'status.json'))),flush=True)
    elif args.action=='supervise':supervise()
    elif args.action=='verify':verify()
    else:
        try:Worker(args.domain).run()
        except Exception:
            detail=traceback.format_exc();print(detail,flush=True)
            write(OUT/args.domain/'failure.json',dict(at=stamp(),error=detail))
            previous=read(OUT/args.domain/'status.json') if (OUT/args.domain/'status.json').exists() else {}
            write(OUT/args.domain/'status.json',dict(previous,state='FAILED',updated_at=stamp(),error=detail))
            raise


if __name__=='__main__':main()
