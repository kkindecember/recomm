"""Bounded two-arm item-order transfer; immutable old experiments are reused."""
import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from experiment.phase20.gacr_list_distill import Work, tensors_hash, read, write, metric, paired
from experiment.phase20.gacr_list_distill_core import encode_history, all_scores
from experiment.phase20.item_order_core import sample_backward, calibration_gradients, flat_gradient
from experiment.phase20.preference_core import path_labels
from experiment.phase18.protocol.s18_diff_gram import seed_everything
from experiment.phase18.core.diff_data import sha256
from experiment.phase20.gacr_b1_multiseed import load_ranks
from experiment.phase20.gacr_controls_multiseed import group_masks

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase20/item_order_toys_v1'
SOURCE=ROOT/'experiment/phase20/gacr_list_distill_toys_recovery_v3.json'
Q0=ROOT/'artifacts/phase20/gacr_list_distill_toys_q0_memory_v3/panel.json'
ARMS=('CE_anchor','order_teacher')
REPORT=ROOT/'report/第二十阶段/Stage20_Toys商品顺序监督进展与结果_20260916.md'


def optional_read(path):
    return read(path) if path.exists() else {}


def verify(mapping):
    for p,h in mapping.items(): assert sha256(ROOT/p)==h,('changed_frozen_file',p)


def prepare(gpus):
    assert len(gpus)==len(set(gpus))==2
    original=read(SOURCE)
    verify({**original['source_sha256'],**original['input_sha256']})
    OUT.mkdir(parents=True,exist_ok=False)
    source_cache=ROOT/original['output']/'cache'
    records=[json.loads(line) for line in (ROOT/original['train_candidate_jsonl']).read_text().splitlines()]
    with np.load(source_cache/'distillation_cache.npz') as z:
        arrays={k:z[k] for k in z.files}
    order=np.load(source_cache/'training_order.npy')
    assert sorted(order.tolist())==np.flatnonzero(arrays['fit']).tolist()
    panel=read(Q0)
    gradients=panel['gradient_indices']
    assert len(gradients)==32 and all(arrays['fit'][i] and arrays['positions'][i]>=0 for i in gradients)
    assert len({records[i]['user_id'] for i in gradients})==32
    fit=sorted(order.tolist(),key=lambda i:(len(records[i]['history']),records[i]['sample_key']))
    profile=[fit[j] for j in np.linspace(0,len(fit)-1,8).astype(int)]
    panel['profile_indices']=profile
    write(OUT/'panel.json',panel)
    paths=[SOURCE,Q0,OUT/'panel.json']+[source_cache/p for p in ('manifest.json','training_order.npy','distillation_cache.npz')]
    paths += [ROOT/'experiment/phase20'/p for p in ('item_order_core.py','item_order_transfer.py','test_item_order.py','run_item_order_transfer.sh')]
    frozen={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    cfg=copy.deepcopy(original)
    cfg.update(id='stage20_item_order_toys_v1',output=str(OUT.relative_to(ROOT)),
        config_path=str((OUT/'design.json').relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)),
        learning_rate=1e-5,minimum_free_mib=15000,cuda_memory_fraction=.25,
        gpus=dict(zip(ARMS,gpus)),source_cache=str(source_cache.relative_to(ROOT)),
        source_config=str(SOURCE.relative_to(ROOT)),resource_wait_seconds=900,
        profile_budget_seconds=1800,worker_budget_seconds=41400,aggregate_budget_seconds=1800,
        total_task_budget_seconds=86400,bootstrap_repeats=2000,bootstrap_seed=20260916,
        scope='Two-arm adaptation of LOHRec Eq14 only; CE+parent vs CE+parent+item order; true unforced50; no test',
        grade_rule={'relative_ndcg10':.01,'reference':'larger N10 of C1 and CE; H10>=same chosen reference',
                    'automatic_followup':False},
        gradient_rule='alpha=norm(mean CE decoder gradient)/norm(mean normalized mean-pair hinge decoder gradient); 32 fixed covered FIT users at C1, no clipping/search',
        margin_origin='Fixed .01 per teacher rank distance: local adaptation choice, not claimed author hyperparameter',
        candidate_correction='No forced insertion, no gold-first reordering; teacher order only',
        source_revision='Equal CE/order initial gradient scale and lr1e-5 differ from historical KL; not a pure KL-vs-order ablation')
    cfg['loss'].update(margin=.01,order_weight=1.)
    cfg['source_sha256'].update({p:h for p,h in frozen.items() if p.endswith(('.py','.sh'))})
    cfg['input_sha256'].update({p:h for p,h in frozen.items() if not p.endswith(('.py','.sh'))})
    write(OUT/'design.json',cfg)
    (OUT/'workers').mkdir()
    write(OUT/'status.json',dict(state='PREPARED',updated_at=time.strftime('%FT%T%z'),new_optimizer_steps=0,test_read=False))
    print(json.dumps({'state':'PREPARED','source_rows':len(records),'fit_rows':len(order),'covered_fit':int((arrays['fit']&(arrays['positions']>=0)).sum()),'gpus':cfg['gpus']}))


class OrderWork(Work):
    def __init__(self,*args,**kwargs):
        self.mutex=threading.RLock()
        self.last_extra={}
        self.terminal=False
        super().__init__(*args,**kwargs)

    def status(self,*args,**kwargs):
        with self.mutex:
            if self.terminal: return
            if kwargs.get('stage',self.stage)!=self.stage: self.last_extra={}
            self.last_extra.update(kwargs)
            super().status(*args,**self.last_extra)
            if args and args[0] in ('COMPLETED','PASSED','FAILED','STOPPED'): self.terminal=True

    def load_cache(self):
        path=ROOT/self.config['source_cache']
        manifest=read(path/'manifest.json')
        assert manifest['cohort']==self.config['frozen_cohort']
        assert sha256(path/'distillation_cache.npz')==manifest['distillation_cache_sha256']
        with np.load(path/'distillation_cache.npz') as z: self.cache={k:z[k] for k in z.files}
        for k in ('candidate_ids','fit','positions'): assert np.array_equal(self.cache[k],self.arrays[k])
        self.order=np.load(path/'training_order.npy')
        assert sorted(self.order.tolist())==np.flatnonzero(self.arrays['fit']).tolist()
        self.panel=read(OUT/'panel.json')

    def backward(self,i,divisor=1,arm=None):
        return sample_backward(self.model,self.batch(i),self.decoder.paths,self.cache['candidate_ids'][i].tolist(),
             self.cache['parent_scores'][i],self.cache['true_probability'][i],self.cache['positions'][i]>=0,
             arm or self.arm,self.settings,self.device,divisor)

    def profile(self):
        self.data(); self.load_model(); self.load_cache(); self.snapshot()
        initial={n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if p.requires_grad}
        initial_hash=tensors_hash(initial.items())
        accum=None
        self.status(stage='calibrate_order_weight_on_32_FIT_users')
        per_sample=[]
        for j,i in enumerate(self.panel['gradient_indices']):
            ce,rank=calibration_gradients(self.model,self.batch(i),self.decoder.paths,self.cache['candidate_ids'][i].tolist(),
                     self.cache['true_probability'][i],self.settings,self.device)
            if accum is None: accum=[torch.zeros_like(ce),torch.zeros_like(rank)]
            accum[0].add_(ce/32); accum[1].add_(rank/32)
            per_sample.append({'cache_index':i,'ce_norm':float(ce.norm()),'normalized_rank_norm':float(rank.norm())})
            self.progress(j+1,32)
        norms=[float(x.double().norm()) for x in accum]
        assert min(norms)>1e-8 and all(math.isfinite(x) for x in norms)
        weight=norms[0]/norms[1]
        assert 1e-4 < weight < 1e4,('degenerate_gradient_scale',weight)
        self.settings['order_weight']=weight
        cosine=float(torch.dot(accum[0].double(),accum[1].double())/(norms[0]*norms[1]))
        del accum,ce,rank
        timings={a:[] for a in ARMS}; norms_profile={a:[] for a in ARMS}
        self.status(stage='profile_both_objectives_and_cache_parity')
        for i in self.panel['profile_indices']:
            scores=self.parent_scores(i)
            np.testing.assert_allclose(scores,self.cache['parent_scores'][i],rtol=0,atol=1e-6)
            for arm in ARMS:
                self.model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); start=time.monotonic()
                self.backward(i,arm=arm)
                norm=float(torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
                torch.cuda.synchronize(); timings[arm].append(time.monotonic()-start); norms_profile[arm].append(norm)
        self.frozen_unchanged()
        optimizer=torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],lr=self.config['learning_rate'],weight_decay=.01)
        optimizer.step(); self.engineering_steps=1
        assert tensors_hash((n,p) for n,p in self.model.named_parameters() if p.requires_grad)!=initial_hash
        checkpoint=self.path/'discarded_engineering_update.pt'
        self.save_parameters(checkpoint,discarded_engineering_update=True)
        updated_hash=tensors_hash((n,p) for n,p in self.model.named_parameters() if p.requires_grad)
        self.load_parameters(checkpoint)
        assert tensors_hash((n,p) for n,p in self.model.named_parameters() if p.requires_grad)==updated_hash
        with torch.no_grad():
            for n,p in self.model.named_parameters():
                if p.requires_grad: p.copy_(initial[n])
        del optimizer,initial
        self.model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        assert tensors_hash((n,p) for n,p in self.model.named_parameters() if p.requires_grad)==initial_hash
        self.frozen_unchanged()
        self.status(stage='frozen_C1_native_beam_parity_and_timing')
        native={r['user_id']:r for r in map(json.loads,(ROOT/self.config['c1_predictions']).read_text().splitlines())}
        ordered=sorted(range(len(self.validation)),key=lambda i:(len(self.validation.records[i]['history']),self.validation.records[i]['user_id']))
        val_indices=[ordered[j] for j in np.linspace(0,len(ordered)-1,8).astype(int)]
        infer=[]
        with torch.no_grad():
            for i in val_indices:
                batch=self.collator([self.validation[i]])
                torch.cuda.synchronize(); start=time.monotonic()
                ids,scores=self.decoder.generate(self.model,{'input_ids':batch['item_text_ids'].to(self.device),'attention_mask':batch['item_text_masks'].to(self.device)})[0]
                torch.cuda.synchronize(); infer.append(time.monotonic()-start)
                reference=native[self.validation.records[i]['user_id']]
                assert ids==reference['ranked_item_ids']
                np.testing.assert_allclose(scores,reference['scores'],rtol=0,atol=1e-6)
        estimate={a:1.15*(float(np.mean(t))*21839+float(np.mean(infer))*len(self.validation))+1200 for a,t in timings.items()}
        result=dict(state='PASSED',order_weight=weight,ce_mean_gradient_norm=norms[0],rank_mean_gradient_norm=norms[1],
            ce_rank_cosine=cosine,per_sample_gradients=per_sample,gradient_selection='fixed32 covered FIT users; no validation gradient',
            normalized_weight_is_fixed=True,training_seconds_per_sample=timings,gradient_norms=norms_profile,
            inference_seconds_per_user=infer,estimated_worker_seconds=estimate,validation_parity_users=val_indices,
            initial_decoder_sha256=initial_hash,discarded_engineering_steps=1,restored_exact=True,formal_optimizer_steps=0,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            test_read=False,source_config_sha256=sha256(ROOT/self.config['config_path']))
        result['budget_passed']=all(t<self.config['worker_budget_seconds'] for t in estimate.values())
        if not result['budget_passed']: result['state']='OVER_BUDGET'
        self.verify(); write(self.path/'profile.json',result)
        assert result['budget_passed'],('profile_exceeds_budget',estimate)
        self.status('PASSED',stage='finished')

    def train_arm(self):
        self.data(); self.load_model(); self.load_cache(); self.snapshot()
        initial_hash=tensors_hash((n,p) for n,p in self.model.named_parameters() if p.requires_grad)
        assert initial_hash==read(OUT/'profile/profile.json')['initial_decoder_sha256']
        parameters=[p for p in self.model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=self.config['learning_rate'],weight_decay=.01,eps=1e-8)
        scheduler=get_linear_schedule_with_warmup(optimizer,9,171)
        curve=[]; self.status(stage='decoder_training')
        for start in range(0,len(self.order),128):
            batch_indices=self.order[start:start+128]; optimizer.zero_grad(set_to_none=True); totals={}
            for j,i in enumerate(batch_indices):
                for k,v in self.backward(i,divisor=len(batch_indices)).items(): totals[k]=totals.get(k,0.)+v/len(batch_indices)
                if (j+1)%16==0: self.progress(start+j+1,len(self.order))
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            lr=optimizer.param_groups[0]['lr']; optimizer.step(); scheduler.step(); self.steps+=1
            point=dict(step=self.steps,examples=start+len(batch_indices),learning_rate=lr,gradient_norm_before_clip=norm,**totals)
            curve.append(point)
            with (self.path/'training_curve.jsonl').open('a') as f: f.write(json.dumps(point)+'\n')
            if self.steps%50==0: self.save_parameters(self.path/'last_progress.pt')
            self.progress(point['examples'],len(self.order))
        assert self.steps==171
        self.frozen_unchanged(); self.verify()
        checkpoint=self.path/'last.pt'; self.save_parameters(checkpoint)
        state=optimizer.state_dict()
        for item in state['state'].values():
            for k,v in item.items():
                if torch.is_tensor(v): item[k]=v.detach().cpu()
        torch.save(dict(optimizer=state,scheduler=scheduler.state_dict(),step=self.steps,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),self.path/'optimizer.pt')
        del optimizer,state
        self.status(stage='fixed_train_calibration_list_diagnostics')
        indices=self.panel['fit_indices']+self.panel['calibration_indices']
        scores=[]
        for j,i in enumerate(indices):
            scores.append(self.parent_scores(i))
            if (j+1)%16==0: self.progress(j+1,len(indices))
        np.savez(self.path/'fixed_panel_scores.npz',indices=np.array(indices),scores=np.array(scores),
                 parent_scores=self.cache['parent_scores'][indices],teacher_probability=self.cache['true_probability'][indices],positions=self.cache['positions'][indices])
        result=dict(state='COMPLETED',arm=self.arm,curve=curve,optimizer_steps=171,examples=len(self.order),epochs=1,
            checkpoint_selection='fixed_last_step',checkpoint_sha256=sha256(checkpoint),frozen_parameters_unchanged=True,
            initial_decoder_sha256=initial_hash,training_order_sha256=sha256(ROOT/self.config['source_cache']/'training_order.npy'),test_read=False)
        write(self.path/'summary.json',result); self.status('COMPLETED',stage='finished')


def terminate(signum,frame): raise TimeoutError('item order worker received signal '+str(signum))


def worker(mode,arm=None):
    cfg=read(OUT/('design.json' if mode=='profile' else 'config.json'))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM): signal.signal(sig,terminate)
    torch.set_num_threads(4); torch.cuda.set_device(0)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['gpus'][ARMS[0] if mode=='profile' else arm]
    start=time.monotonic(); stable=0
    while stable<2:
        free,_=torch.cuda.mem_get_info(); stable=stable+1 if free/2**20>=cfg['minimum_free_mib'] else 0
        if time.monotonic()-start>cfg['resource_wait_seconds']: raise TimeoutError('GPU admission exhausted')
        if stable<2: time.sleep(10)
    torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']); seed_everything(cfg['seed'])
    signal.alarm(cfg['profile_budget_seconds'] if mode=='profile' else cfg['worker_budget_seconds'])
    for stage in (('profile',) if mode=='profile' else ('train','evaluate')):
        job=OrderWork(cfg,stage,arm)
        done=threading.Event()
        def heartbeat():
            while not done.wait(30): job.status()
        thread=threading.Thread(target=heartbeat,daemon=True); thread.start()
        try:
            job.verify(); job.snapshot()
            if stage=='profile': job.profile()
            elif stage=='train': job.train_arm()
            else: job.evaluate()
        except BaseException as e:
            done.set(); thread.join(timeout=5)
            job.status('FAILED',error=str(e),traceback=traceback.format_exc()); raise
        finally: done.set(); thread.join(timeout=5)
        del job; gc.collect(); torch.cuda.empty_cache()
    signal.alarm(0)


def render(status):
    text='# Stage20 Toys商品顺序监督：进展与结果\n\n'
    text+='Origin Skill：academic-research-suite。这是LOHRec排序目标的适配，不是完整复现或新颖性声明。\n\n'
    text+=f'状态：**{status["state"]}**；更新：{status["updated_at"]}。完整结果出来前不判断效果。\n\n'
    text+='两臂同C1起点、21839条FIT前缀、171步、AdamW峰值lr1e-5；CE与父保持相同，排序臂新增全50候选的1225对有序hinge。含EOS、不含BOS，路径log概率按标签长度归一；排序项不做列表标准化。margin=0.01为本地固定选择，求均值后系数由32名FIT用户的初始decoder梯度确定。原论文还使用目标置首、不同候选数及层级训练，本实验未采用。\n\n'
    text+='教师排序不读取验证目标；原训练事件暴露仍存在。完整开发验证每臂19412人，原test不读取。固定旧列表诊断与重新生成成绩分别报告。\n\n'
    profile=optional_read(OUT/'profile/profile.json')
    if profile:
        text+=f'初始排序系数：{profile["order_weight"]:.8g}；实测外推每臂训练加评估：'+', '.join(f'{a} {v/3600:.2f}小时' for a,v in profile['estimated_worker_seconds'].items())+'。共享负载下的估算，非完工承诺。\n\n'
    text+='| 任务 | 状态 | 阶段 | 进度 |\n|---|---|---|---|\n'
    for name,d in status.get('details',{}).items():
        text+=f'| {name} | {d.get("state","等待")} | {d.get("stage","")} | {d.get("progress",{})} |\n'
    summary=optional_read(OUT/'summary.json')
    if summary:
        text+='\n完整开发结果（单次训练；探索性比较）：\n\n| 方法 | N10 | H10 | H50 |\n|---|---:|---:|---:|\n'
        for name,m in summary['all_metrics'].items(): text+=f'| {name} | {m["ndcg@10"]:.8f} | {m["hit@10"]:.8f} | {m["hit@50"]:.8f} |\n'
        text+='\n判定：`'+summary['decision']+'`；没有自动接续实验。完整分组、配对区间、核验见产物summary.json。\n'
    text+='\n[冻结设计](../../artifacts/phase20/item_order_toys_v1/design.json)、[状态](../../artifacts/phase20/item_order_toys_v1/status.json)、[实施前方案](../../plan/第二十阶段/下一项唯一实验_先验证商品顺序监督_20260916.md)。\n'
    REPORT.write_text(text)


def aggregate():
    cfg=read(OUT/'config.json'); summaries={a:read(OUT/a/'evaluate/summary.json') for a in ARMS}
    assert all(s['state']=='COMPLETED' for s in summaries.values())
    tables={a:load_ranks(OUT/a/'evaluate/per_user.tsv') for a in ARMS}
    users=sorted(tables[ARMS[0]]); assert len(users)==19412 and set(users)==set(tables[ARMS[1]])
    values={a+'_'+kind:np.array([tables[a][u][kind] for u in users]) for a in ARMS for kind in ('raw','pcrf')}
    for ref in ('original_gram','original_pcrf','c1','c1_pcrf','B1'):
        values[ref]=np.array([tables[ARMS[0]][u][ref] for u in users])
        assert np.array_equal(values[ref],[tables[ARMS[1]][u][ref] for u in users])
    all_metrics={k:metric(v) for k,v in values.items()}
    job=OrderWork(cfg,'aggregate'); job.verify(); job.data()
    lookup={r['user_id']:r for r in job.validation.records}
    masks=group_masks(users,lookup,job.frequency,set((ROOT/cfg['c1_training_users']).read_text().splitlines()),cfg['pcrf']['q1'])
    comparisons={kind:{g:paired(values['order_teacher_'+kind][m],values['CE_anchor_'+kind][m],cfg)
                      for g,m in masks.items()} for kind in ('raw','pcrf')}
    # Recount saved fresh predictions independently of decoder outputs/runner ranks.
    verified=0
    for arm in ARMS:
        seen=set()
        for line in (OUT/arm/'evaluate/predictions.jsonl').open():
            r=json.loads(line); ids=r['ranked_item_ids']; u=r['user_id']
            assert u not in seen and len(ids)==len(set(ids))==50 and np.isfinite(r['scores']).all()
            assert r['gold_item_id']==lookup[u]['target'] and r['split']=='validation'
            seen.add(u); rank=ids.index(r['gold_item_id'])+1 if r['gold_item_id'] in ids else 51
            assert rank==tables[arm][u]['raw']; verified+=1
        assert seen==set(users)
        with np.load(OUT/arm/'evaluate/pcrf_scores.npz') as z:
            for u,ids,seq,cf,reliability in zip(z['users'].tolist(),z['candidate_ids'],z['seq'],z['cf'],z['reliability']):
                def normalize(x): return (x-x.mean())/max(float(x.std()),1e-6)
                pop=job.frequency[ids]
                expected_reliability=1.-float(np.mean(pop[:10]<=cfg['pcrf']['q1']))
                assert abs(reliability-expected_reliability)<1e-12
                joint=normalize(seq)+reliability*normalize(normalize(cf)-.5*normalize(np.log1p(pop)))
                ranked=ids[np.argsort(-joint,kind='stable')].tolist(); target=lookup[u]['target']
                rank=ranked.index(target)+1 if target in ranked else 51
                assert rank==tables[arm][u]['pcrf']
        for group,mask in masks.items():
            for kind in ('raw','pcrf'):
                ranks=values[arm+'_'+kind][mask]
                for k in (5,10,20,50):
                    hit=sum(int(r<=k) for r in ranks)/len(ranks)
                    ndcg=sum(1/math.log2(r+1) if r<=k else 0 for r in ranks)/len(ranks)
                    assert abs(hit-summaries[arm]['groups'][group][kind][f'hit@{k}'])<1e-12
                    assert abs(ndcg-summaries[arm]['groups'][group][kind][f'ndcg@{k}'])<1e-12
    baseline=max(('c1','CE_anchor_raw'),key=lambda k:all_metrics[k]['ndcg@10'])
    method=all_metrics['order_teacher_raw']; ref=all_metrics[baseline]
    gate=method['ndcg@10']>=ref['ndcg@10']*1.01 and method['hit@10']>=ref['hit@10']
    verify({**cfg['source_sha256'],**cfg['input_sha256']})
    result=dict(state='COMPLETED',all_metrics=all_metrics,arms=summaries,matched_comparisons=comparisons,
        gate_reference=baseline,relative_raw_ndcg10_vs_gate_reference=method['ndcg@10']/ref['ndcg@10']-1,
        decision='REVIEW_VALUE_OF_NEXT_METHOD_DESIGN' if gate else 'STOP_THIS_ORDER_TRANSFER_RECIPE',
        independently_recounted_predictions=verified,formal_optimizer_steps=342,discarded_engineering_steps=1,
        test_read=False,automatic_next_experiment=False,ci_caveat='Single decoder seed, repeatedly used development validation; not independent test.')
    write(OUT/'summary.json',result)
    job.status('COMPLETED',stage='finished',independently_recounted_predictions=verified)


def controller():
    cfg=read(OUT/'design.json'); started=time.monotonic(); active={}; task_records={}
    with (OUT/'controller_started.json').open('x') as f: json.dump({'pid':os.getpid(),'started_at':time.strftime('%FT%T%z')},f)
    def launch(task,arm=None):
        gpu=cfg['gpus'][ARMS[0] if task=='profile' else arm]
        env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        command=[sys.executable,'-u','-m','experiment.phase20.item_order_transfer','--mode',task]
        if arm: command+=['--arm',arm]
        name=arm or task
        log=(OUT/'workers'/(name+'.log')).open('x')
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[name]=(child,log,time.monotonic())
        task_records[name]=dict(pid=child.pid,gpu_uuid=gpu,command=command,start_time=time.strftime('%FT%T%z'))
        write(OUT/'workers'/(name+'.launch.json'),task_records[name])
    def status(state,reason=None):
        details={}
        for name in task_records:
            if name=='profile': details[name]=optional_read(OUT/'profile/status.json')
            else:
                for stage in ('train','evaluate'): details[name+'_'+stage]=optional_read(OUT/name/stage/'status.json')
        value=dict(state=state,updated_at=time.strftime('%FT%T%z'),pid=os.getpid(),elapsed_seconds=time.monotonic()-started,
            details=details,active_processes={n:p.pid for n,(p,_,_) in active.items()},tasks=task_records,
            new_optimizer_steps=sum(d.get('new_optimizer_steps',0) for n,d in details.items() if n.endswith('_train')),
            planned_optimizer_steps=342,test_read=False,automatic_next_experiment=False,reason=reason)
        write(OUT/'status.json',value); render(value)
    def wait_all():
        while active:
            for name,(child,log,begin) in list(active.items()):
                rc=child.poll()
                budget=cfg['profile_budget_seconds'] if name=='profile' else cfg['worker_budget_seconds']
                if rc is None and time.monotonic()-begin>budget+cfg['resource_wait_seconds']+60:
                    raise TimeoutError('owned worker hard timeout '+name)
                if rc is not None:
                    log.close(); task_records[name].update(exit_code=rc,elapsed_seconds=time.monotonic()-begin); del active[name]
                    assert rc==0,('worker_failed',name,rc)
            status('RUNNING')
            if active: time.sleep(30)
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,terminate)
    try:
        launch('profile'); wait_all()
        profile=read(OUT/'profile/profile.json'); assert profile['state']=='PASSED'
        cfg['loss']['order_weight']=profile['order_weight']
        cfg['config_path']=str((OUT/'config.json').relative_to(ROOT))
        cfg['input_sha256'][str((OUT/'profile/profile.json').relative_to(ROOT))]=sha256(OUT/'profile/profile.json')
        cfg['input_sha256'][str((OUT/'design.json').relative_to(ROOT))]=sha256(OUT/'design.json')
        write(OUT/'config.json',cfg)
        verify({**cfg['source_sha256'],**cfg['input_sha256']})
        for arm in ARMS: launch('worker',arm)
        wait_all()
        signal.signal(signal.SIGALRM,terminate); signal.alarm(cfg['aggregate_budget_seconds'])
        aggregate(); signal.alarm(0); status('COMPLETED','Both arms completed; no automatic next experiment')
    except BaseException as e:
        for child,log,_ in active.values():
            if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
        for child,log,_ in active.values():
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
            log.close()
        active.clear(); status('FAILED',str(e)); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=('prepare','profile','worker','controller'),required=True)
    parser.add_argument('--arm',choices=ARMS); parser.add_argument('--gpus',nargs=2)
    args=parser.parse_args()
    if args.mode=='prepare': prepare(args.gpus)
    elif args.mode=='controller': controller()
    else: worker(args.mode,args.arm)
