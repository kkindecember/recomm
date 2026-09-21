"""Frozen global beam intervention with disjoint tuning/evaluation users."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from transformers import LogitsProcessorList

from experiment.phase20.guided_decoding_job import ROOT, CONFIG, read, write, sha, now, model_paths, cf_paths, verify
from experiment.phase20.guided_decoding_core import PrefixIndex, PrefixGuidance, rewards, standardize as z, rank_of, summarize_ranks, paired_ranks
from experiment.phase20.gacr_fusion_train import CandidateBuilder, gt
from experiment.phase20.gacr_features import vector_features
from experiment.phase20.future_factor_screen import Factor, old_score
from experiment.phase20.nearest_matched_head import OrdinaryHead
from experiment.phase20.future_fusion_alignment import CF0B2ItemHead
from experiment.phase20.personal_popularity_core import reconstruct_training
from experiment.phase20.gacr_list_distill_core import encode_history, all_scores
from experiment.phase20.preference_core import path_labels


def digest_modules(modules):
    h=hashlib.sha256()
    for module in modules:
        for key,value in module.state_dict().items():
            h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FrozenHeads:
    def __init__(self, domain, config, q1):
        paths=model_paths(domain, config); self.q1=q1
        self.b1=torch.load(ROOT/paths['B1'],map_location='cpu')
        self.models={}
        for key in ('softmax','normalized','linear','mlp'):
            model=(Factor('next_only') if key in ('softmax','normalized') else
                   OrdinaryHead('linear_residual' if key=='linear' else 'ordinary_mlp32'))
            model.load_state_dict(torch.load(ROOT/paths[key],map_location='cpu'),strict=True)
            model.eval().requires_grad_(False); self.models[key]=model

    @torch.no_grad()
    def score(self, features, base, cf, freq):
        x=torch.as_tensor(np.asarray(features),dtype=torch.float32)
        b1=old_score(x,torch.as_tensor(np.asarray(base),dtype=torch.float32),self.b1)
        order=np.argsort(-b1,kind='stable')
        reliability=1-float(np.mean(np.asarray(freq)[order[:10]]<=self.q1))
        component=reliability*z(z(cf)-.5*z(np.log1p(freq))).mean(0)
        prior=torch.tensor(z(b1)-math.log(49),dtype=torch.float64)[None]
        raw={key:model(x[None],prior)[0][0].numpy() for key,model in self.models.items()}
        final={key:z(value)+component for key,value in raw.items()}
        final['b1_cf']=z(b1)+component
        return b1,raw,final,reliability


class Worker:
    def __init__(self,domain,mode):
        self.config=read(CONFIG); self.domain=domain; self.mode=mode
        self.root=ROOT/self.config['output']; self.out=self.root/domain
        self.started=time.monotonic(); self.decoding_seconds=[]; self.completed=0
        self.memory_wait_seconds=0.
        self.status('加载冻结模型')
        torch.set_num_threads(self.config['cpu_threads']); torch.set_num_interop_threads(1)
        torch.manual_seed(2023); np.random.seed(2023)
        assert torch.cuda.is_available(), 'CUDA unavailable; worker must run on admitted host GPU'
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(self.config['cuda_memory_fraction'],0)
        self.device=torch.device('cuda:0')
        self.domain_config=read(ROOT/'experiment/phase20/gacr_fusion_train_v1.json')['domains'][domain]
        self.cohorts=read(self.out/'cohorts.json')
        self.builder=CandidateBuilder(read(ROOT/self.domain_config['transfer_config']),self.device)
        self.index=PrefixIndex(self.builder.decoder.paths,range(1,len(self.builder.catalog['items'])))
        _,sequences,_,_=reconstruct_training(self.builder.train.records)
        self.freq=np.bincount([i for seq in sequences.values() for i in seq],minlength=len(self.builder.catalog['items']))
        del sequences
        expected=torch.tensor([self.freq[self.builder.numeric[n]]+1 for n in self.builder.dual.catalog],device=self.device,dtype=torch.float32).log()
        torch.testing.assert_close(self.builder.dual.log_counts,expected,rtol=0,atol=0)
        self.heads=FrozenHeads(domain,self.config,self.domain_config['q1'])
        self.cf_models=[]
        for path in cf_paths(domain):
            checkpoint=torch.load(ROOT/path,map_location='cpu'); c=checkpoint['model_config']
            model=CF0B2ItemHead(c['num_items'],c['max_history'],c['d_model'],c['num_layers'],c['num_heads'],c['dropout'],c['temperature_initial'])
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            model.eval().requires_grad_(False).to(self.device); self.cf_models.append(model)
        self.modules=[self.builder.dual,*self.cf_models,*self.heads.models.values()]
        assert all(not p.requires_grad for m in self.modules for p in m.parameters())
        self.fingerprint=digest_modules(self.modules)
        self.status('模型加载完成')

    def status(self,stage,state='RUNNING',**kw):
        write(self.out/f'{self.mode}_status.json',dict(state=state,stage=stage,updated_at=now(),pid=os.getpid(),
            dataset=self.domain,mode=self.mode,elapsed_seconds=round(time.monotonic()-self.started,1),
            completed_decodes=self.completed,optimizer_steps=0,test_read=False,
            memory_wait_seconds=round(self.memory_wait_seconds,1),
            peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1048576,1) if torch.cuda.is_initialized() else 0,**kw))

    def inference_batch(self,index,alternative=False):
        b=self.builder; sample=b.sample(index,'validation')
        # Collator output/targets are never passed to generate or CF encoders.
        sample=dict(sample,output=b.prepared['item2lexid'][b.prepared['catalog'][-1 if alternative else 0]])
        return b.collator([sample])

    def wait_memory(self):
        started=time.monotonic()
        while True:
            free,_=torch.cuda.mem_get_info()
            # Unused allocator blocks are already ours and can be reused.
            usable=free+torch.cuda.memory_reserved()-torch.cuda.memory_allocated()
            if usable/1048576>=self.config['worker_minimum_free_mib']: break
            self.status('等待其他任务释放显存',state='WAITING_GPU',available_mib=round(usable/1048576),
                required_mib=self.config['worker_minimum_free_mib'])
            if time.monotonic()-started>self.config['resource_wait_seconds']:
                raise TimeoutError('Worker GPU memory wait budget exhausted')
            time.sleep(10)
        self.memory_wait_seconds+=time.monotonic()-started

    @torch.no_grad()
    def context(self,index):
        self.wait_memory()
        started=time.monotonic(); b=self.builder
        row=b.validation.records[index]; batch=self.inference_batch(index)
        hidden,flatmask=encode_history(b.model,batch,self.device)
        ids=batch['item_text_ids'].to(self.device); mask=batch['item_text_masks'].to(self.device)
        pooled=b.dual.pool_coarse(hidden,mask,ids.shape[-1])[0]
        logits=b.dual.catalog_head(pooled)
        seen=[b.dual.item_to_index[b.catalog['items'][i]] for i in row['history'][-20:]]
        logits[seen]=-torch.inf
        catalog=[b.dual.catalog[i] for i in torch.topk(logits,k=50).indices.cpu().tolist()]
        cf=[]
        for model in self.cf_models:
            history=torch.zeros((1,20),device=self.device,dtype=torch.long)
            values=row['history'][-20:]; history[0,:len(values)]=torch.tensor(values,device=self.device)
            user=model.encode(history,history.ne(0))
            weights=F.normalize(model.item_embedding.weight,dim=-1)
            cf.append((model.logit_scale.exp().clamp(max=100)*F.linear(user,weights))[0].cpu().numpy())
        cf=np.stack(cf).astype(np.float64)
        signals,permutation=rewards(cf,self.freq,row['user_id'],self.config['selection_seed'])
        torch.cuda.synchronize()
        return dict(index=index,row=row,batch=batch,ids=ids,mask=mask,pooled=pooled,logits=logits,catalog=catalog,
                    cf=cf,signals=signals,permutation=permutation,context_seconds=time.monotonic()-started)

    @torch.no_grad()
    def decode(self,c,signal=None,strength=0.,trace=False,explicit_zero=False):
        self.wait_memory()
        started=time.monotonic(); b=self.builder
        guidance=None
        kwargs={}
        if signal is not None or explicit_zero:
            guidance=PrefixGuidance(self.index,c['signals'][signal or 'real_cf'],strength,trace)
            kwargs['logits_processor']=LogitsProcessorList([guidance])
        output=b.model.generate(input_ids=c['ids'],attention_mask=c['mask'],max_length=b.max_length,
            prefix_allowed_tokens_fn=gt.prefix_allowed_tokens_fn(b.trie),num_beams=50,num_return_sequences=50,
            return_dict_in_generate=True,output_scores=True,length_penalty=1.,**kwargs)
        paths=[s[:s.index(1)+1] for s in output.sequences.cpu().tolist()]
        ids=np.asarray([self.index.path_to_item[tuple(p)] for p in paths])
        completed_scores=output.sequences_scores.cpu().numpy().astype(np.float64)
        assert len(set(ids))==50 and np.isfinite(completed_scores).all()
        native=completed_scores-(strength*c['signals'][signal][ids] if signal else 0.)
        del output
        # Remove the intervention from feature ranks; original generation scores
        # define the rank feature even on the changed candidate set.
        order=np.argsort(-native,kind='stable'); ids=ids[order]; native=native[order]; completed_scores=completed_scores[order]
        gram=[b.catalog['items'][i] for i in ids]
        union=list(dict.fromkeys(gram+c['catalog']))
        gr={x:i+1 for i,x in enumerate(gram)}; cr={x:i+1 for i,x in enumerate(c['catalog'])}
        indices=torch.tensor([b.dual.item_to_index[x] for x in union],device=self.device)
        features=vector_features(c['logits'],c['pooled'],b.dual.catalog_head.weight,indices,
            [gr.get(x,0) for x in union],[cr.get(x,0) for x in union]).cpu().numpy()[:50]
        base=np.asarray([1/(i+1) for i in range(50)],dtype=np.float32)
        cf=c['cf'][:,ids]; freq=self.freq[ids]
        b1,raw,final,reliability=self.heads.score(features,base,cf,freq)
        torch.cuda.synchronize(); elapsed=time.monotonic()-started
        self.decoding_seconds.append(elapsed); self.completed+=1
        b1order=np.argsort(-b1,kind='stable')
        # Use canonical B1 order as the stable tiebreaker, matching old caches.
        ids=ids[b1order]
        scores={key:values[b1order] for key,values in {**final,'native':native,'guided':completed_scores}.items()}
        target=c['row']['target']
        record=dict(user_id=c['row']['user_id'],index=c['index'],target=target,
            target_frequency=int(self.freq[target]),candidate_ids=ids.tolist(),
            features=features[b1order].tolist(),base=base[b1order].tolist(),cf_scores=cf[:,b1order].tolist(),
            frequencies=freq[b1order].tolist(),b1_scores=b1[b1order].tolist(),
            raw_head_scores={k:v[b1order].tolist() for k,v in raw.items()},
            scores={k:v.tolist() for k,v in scores.items()},
            ranks={k:rank_of(ids,v,target) for k,v in scores.items()},
            rewards={k:v[ids].tolist() for k,v in c['signals'].items()},signal=signal,strength=strength,
            reliability=reliability,decode_and_heads_seconds=elapsed,context_seconds=c['context_seconds'],
            guidance_calls=guidance.calls if guidance else 0)
        if trace and guidance: record['active_prefixes']=guidance.trace
        return record

    def frozen_check(self):
        assert all(not p.requires_grad and p.grad is None for m in self.modules for p in m.parameters())
        assert digest_modules(self.modules)==self.fingerprint, 'Frozen model changed in memory'
        verify(self.root)

    def smoke(self):
        checks=[]; selected=self.cohorts['smoke']
        users={r['user_id'] for r in selected}
        def matching(path):
            result={}
            with (ROOT/path).open() as f:
                for line in f:
                    r=json.loads(line)
                    if r['user_id'] in users: result[r['user_id']]=r
            assert set(result)==users
            return result
        self.status('读取旧结果核对基线')
        old_features=matching(self.domain_config['features']); old_pred=matching(self.domain_config['c1'])
        old_cf=np.load(ROOT/f'artifacts/phase20/pair_uncertainty_screen_20260917/{self.domain}.cf_scores.npz')
        old_scores=np.load(ROOT/f'artifacts/phase20/talos_candidate_screen_20260917/{self.domain}.{self.config["head_seed"]}.predictions.npz')
        cf_users={str(u):i for i,u in enumerate(old_cf['users'])}; score_users={str(u):i for i,u in enumerate(old_scores['users'])}
        cf_ids,cf_values=old_cf['candidate_ids'],old_cf['scores']
        score_ids,raw_scores,old_ranks=old_scores['candidate_ids'],old_scores['raw_scores'],old_scores['ranks']
        raw_names=old_scores['raw_names'].tolist(); rank_names=old_scores['rank_names'].tolist()
        with (self.out/'smoke_predictions.jsonl').open('x') as f:
            for entry in selected:
                self.status('零引导、旧候选、冻结后处理与完成分数核对',user_id=entry['user_id'])
                c=self.context(entry['index']); user=c['row']['user_id']
                alternative=self.inference_batch(entry['index'],True)
                for key in ('item_text_ids','item_text_masks'):
                    torch.testing.assert_close(c['batch'][key],alternative[key],rtol=0,atol=0)
                vanilla=self.decode(c); zero=self.decode(c,explicit_zero=True)
                assert vanilla['candidate_ids']==zero['candidate_ids']
                np.testing.assert_array_equal(vanilla['scores']['native'],zero['scores']['native'])
                nativeorder=np.argsort(-np.array(vanilla['scores']['native']),kind='stable')
                nativeids=np.array(vanilla['candidate_ids'])[nativeorder]
                assert nativeids.tolist()==old_pred[user]['ranked_item_ids']
                np.testing.assert_allclose(np.array(vanilla['features'])[nativeorder],np.array(old_features[user]['features'])[:50],rtol=0,atol=1e-6)
                i=cf_users[user]; mapping={int(item):j for j,item in enumerate(cf_ids[i])}
                aligned=cf_values[:,i,[mapping[item] for item in vanilla['candidate_ids']]]
                np.testing.assert_allclose(vanilla['cf_scores'],aligned,rtol=1e-5,atol=3e-5)
                j=score_users[user]
                assert vanilla['candidate_ids']==score_ids[j].tolist()
                for head,label in [('softmax','softmax'),('normalized','normalized.next_only')]:
                    np.testing.assert_allclose(z(vanilla['raw_head_scores'][head]),raw_scores[raw_names.index(label),j],rtol=1e-6,atol=2e-6)
                    assert vanilla['ranks'][head]==int(old_ranks[rank_names.index(label+'.mean'),j])
                guided=self.decode(c,'real_cf',.3,trace=True)
                hidden,mask=encode_history(self.builder.model,c['batch'],self.device)
                labels=path_labels(self.builder.decoder.paths,guided['candidate_ids'],self.device)
                direct=all_scores(self.builder.model,hidden,mask,labels,4).detach().cpu().numpy()
                np.testing.assert_allclose(guided['scores']['native'],direct,rtol=0,atol=5e-5)
                del hidden,mask,labels
                for r in (vanilla,zero,guided): f.write(json.dumps(r,allow_nan=False)+'\n')
                f.flush()
                checks.append(dict(user_id=user,zero_identity=True,old_candidates_features_and_heads=True,
                    target_independent_inputs=True,terminal_score_error=float(np.max(np.abs(np.array(guided['scores']['native'])-direct))),
                    overlap=len(set(vanilla['candidate_ids'])&set(guided['candidate_ids']))/50))
        # Separate engineering-only longest-history record is not used for selection.
        self.status('最长历史显存检查')
        longest=self.context(self.cohorts['longest_history']['index'])
        self.decode(longest,'real_cf',.3)
        self.frozen_check()
        calls=self.config['tuning_users']*10+self.config['evaluation_users']*4
        projection=float(np.mean(self.decoding_seconds)*calls*1.10)
        write(self.out/'smoke.json',dict(state='PASSED',checks=checks,model_fingerprint=self.fingerprint,
            frozen_parameters_unchanged=True,mean_decode_seconds=float(np.mean(self.decoding_seconds)),
            planned_decodes=calls,projected_domain_seconds=projection,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,torch_version=torch.__version__,
            gpu_name=torch.cuda.get_device_name(0),new_training_steps=0,test_read=False))
        self.status('工程检查通过',state='COMPLETED',projected_domain_seconds=projection)

    def run_group(self,group,arms):
        rows=[]; total=len(self.cohorts[group])*len(arms); started=time.monotonic()
        with (self.out/f'{group}.jsonl').open('x') as f:
            for number,entry in enumerate(self.cohorts[group]):
                c=self.context(entry['index']); vanilla=None
                for name,signal,strength in arms:
                    r=self.decode(c,signal,strength,trace=number==0 and signal is not None)
                    r['arm']=name; r['group']=group
                    if signal is None:
                        vanilla=r
                        for aname,asignal,lam in arms[1:]:
                            key='late_'+aname
                            score=np.array(r['scores']['native'])+lam*np.array(r['rewards'][asignal])
                            r['scores'][key]=score.tolist(); r['ranks'][key]=rank_of(r['candidate_ids'],score,r['target'])
                    assert vanilla is not None
                    r['overlap_with_vanilla']=len(set(r['candidate_ids'])&set(vanilla['candidate_ids']))/50.
                    f.write(json.dumps(r,allow_nan=False)+'\n'); f.flush(); rows.append(r)
                    done=len(rows); elapsed=time.monotonic()-started
                    self.status('选择引导强度' if group=='tuning' else '固定参数比较',group=group,
                        users_completed=number,users_total=len(self.cohorts[group]),arm=name,progress=dict(completed=done,total=total),
                        group_eta_seconds=round(elapsed*(total-done)/done,1),last_user_id=entry['user_id'])
        return rows

    def run(self):
        arms=[('vanilla',None,0.)]+[(f'{s}@{v:g}',s,v) for s in self.config['signals'] for v in self.config['strengths']]
        tuning=self.run_group('tuning',arms)
        metrics={a:summarize_ranks([r['ranks']['softmax'] for r in tuning if r['arm']==a]) for a,_,_ in arms}
        selected={}
        for signal in self.config['signals']:
            selected[signal]=max(self.config['strengths'],key=lambda v:(metrics[f'{signal}@{v:g}']['N10'],metrics[f'{signal}@{v:g}']['H10'],-v))
        write(self.out/'selection.json',dict(created_at=now(),selected=selected,tuning_metrics=metrics,
            selected_by='tuning N10, then H10, then smaller strength; primary frozen softmax + 3CF',
            evaluation_decodes_so_far=0,tuning_sha256=sha(self.out/'tuning.jsonl')))
        del tuning
        evaluation=self.run_group('evaluation',[('vanilla',None,0.)]+[(s,s,selected[s]) for s in self.config['signals']])
        self.status('缓存排名与冻结小头独立重放')
        summary=self.summarize(evaluation,selected)
        self.replay_verify()
        self.frozen_check()
        summary.update(model_fingerprint=self.fingerprint,peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,
            elapsed_seconds=time.monotonic()-self.started)
        write(self.out/'summary.json',summary)
        self.status('本域完成，重放检查通过',state='COMPLETED',summary=str((self.out/'summary.json').relative_to(ROOT)))

    def summarize(self,rows,selected):
        metrics={}; ranksets={}; timing={}; overlaps={}; tails={}
        for arm in ('vanilla',*self.config['signals']):
            subset=[r for r in rows if r['arm']==arm]
            assert [r['user_id'] for r in subset]==[e['user_id'] for e in self.cohorts['evaluation']]
            for head in subset[0]['ranks']:
                key=arm+'.'+head; rr=np.array([r['ranks'][head] for r in subset]); ranksets[key]=rr
                metrics[key]=summarize_ranks(rr)
                tail=np.array([r['target_frequency']<=self.domain_config['q1'] for r in subset])
                tails[key]=dict(users=int(tail.sum()),metrics=summarize_ranks(rr[tail]) if tail.any() else None)
            timing[arm]=dict(mean_decode_and_heads_seconds=float(np.mean([r['decode_and_heads_seconds'] for r in subset])),
                p95_decode_and_heads_seconds=float(np.percentile([r['decode_and_heads_seconds'] for r in subset],95)),
                mean_context_seconds=float(np.mean([r['context_seconds'] for r in subset])))
            overlaps[arm]=float(np.mean([r['overlap_with_vanilla'] for r in subset]))
        contrasts={}; rng=np.random.RandomState(self.config['selection_seed'])
        samples=rng.randint(0,self.config['evaluation_users'],(self.config['bootstrap_repeats'],self.config['evaluation_users']))
        pairs=[(s+'.softmax','vanilla.softmax') for s in self.config['signals']]
        pairs += [('real_cf.softmax',s+'.softmax') for s in ('shuffled_cf','popularity')]
        pairs += [(s+'.guided','vanilla.late_'+s) for s in self.config['signals']]
        for current,reference in pairs:
            a,b=ranksets[current],ranksets[reference]; comparison=paired_ranks(a,b)
            for name,difference in [('N10',(a<=10)/np.log2(a+1)-(b<=10)/np.log2(b+1)),('H10',(a<=10).astype(float)-(b<=10))]:
                comparison['descriptive_95pct_'+name]=np.percentile(difference[samples].mean(1),[2.5,97.5]).tolist()
            contrasts[current+'__vs__'+reference]=comparison
        return dict(dataset=self.domain,users=self.config['evaluation_users'],selected_strengths=selected,metrics=metrics,
            low_frequency_metrics=tails,contrasts=contrasts,mean_candidate_overlap=overlaps,timing=timing,
            bootstrap_repeats=self.config['bootstrap_repeats'],head_seed=self.config['head_seed'],cf_seeds=[2023,2024,2025],
            test_read=False,optimizer_steps=0,validation_reused=True,intervals='descriptive paired bootstrap, no multiplicity-adjusted confirmatory claim')

    def replay_verify(self):
        count=0; max_error=0.
        for group in ('tuning','evaluation'):
            with (self.out/f'{group}.jsonl').open() as f:
                for line in f:
                    r=json.loads(line); ids=np.array(r['candidate_ids']); target=r['target']
                    assert len(set(ids))==50 and all(tuple(self.builder.decoder.paths[i-1]) in self.index.path_to_item for i in ids)
                    for key,score in r['scores'].items():
                        # Independently count strictly higher scores and preceding ties.
                        score=np.array(score); position=np.flatnonzero(ids==target)
                        expected=51 if not len(position) else 1+int(np.sum(score>score[position[0]]))+int(np.sum(score[:position[0]]==score[position[0]]))
                        assert expected==r['ranks'][key]
                    b1,raw,final,reliability=self.heads.score(r['features'],r['base'],np.array(r['cf_scores']),np.array(r['frequencies']))
                    for key,score in final.items():
                        error=float(np.max(np.abs(score-r['scores'][key]))); max_error=max(max_error,error)
                        assert error<1e-6
                    if r['signal']:
                        np.testing.assert_allclose(np.array(r['scores']['native'])+r['strength']*np.array(r['rewards'][r['signal']]),r['scores']['guided'],atol=1e-12,rtol=0)
                    count+=1
        expected=self.config['tuning_users']*10+self.config['evaluation_users']*4
        assert count==expected
        write(self.out/'verification.json',dict(state='CACHE_REPLAY_VERIFIED',records=count,max_replay_score_error=max_error,
            scope='All cached ranks and frozen CPU heads replayed; GPU generation validated on smoke users, not fully rerun',
            output_sha256={g+'.jsonl':sha(self.out/(g+'.jsonl')) for g in ('tuning','evaluation')}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domain',choices=['Toys','Beauty'],required=True)
    parser.add_argument('--mode',choices=['smoke','run'],required=True);args=parser.parse_args()
    try:
        worker=Worker(args.domain,args.mode)
        worker.smoke() if args.mode=='smoke' else worker.run()
    except BaseException as exc:
        out=ROOT/read(CONFIG)['output']/args.domain
        (out/f'{args.mode}_failure.txt').write_text(traceback.format_exc())
        write(out/f'{args.mode}_status.json',dict(state='FAILED',stage='失败现场已保存',updated_at=now(),pid=os.getpid(),error=str(exc)))
        raise


if __name__=='__main__': main()
