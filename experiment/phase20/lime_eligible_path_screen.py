"""Frozen filtering controls and path-mixture screen, run in background."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from experiment.phase20 import lime_decode_screen as base
from experiment.phase20.lime_decode_screen_core import metrics, contrast, ordered, target_rank
from experiment.phase20.lime_eligible_core import EligibleTrie
from experiment.phase20.lime_path_mixture_core import PathMixtureGuide, rescore_paths

ROOT = base.ROOT
OUT = ROOT / 'artifacts/phase20/lime_eligible_path_20260922_v1'
PARENT = ROOT / 'artifacts/phase20/lime_decode_screen_20260921_v1/full_parallel_v4'
PRIORS = [PARENT, ROOT / 'artifacts/phase20/lime_mechanism_screen_20260921_v1',
          ROOT / 'artifacts/phase20/lime_prefix_expert_20260921_v1']
PROTOCOL = ROOT / 'plan/第二十阶段/LIME过滤与路径融合_后台筛查_20260922.md'
REPORT = ROOT / 'report/第二十阶段/LIME过滤与路径融合_后台筛查_20260922.md'
MODULE = 'experiment.phase20.lime_eligible_path_screen'
GPUS = {'Toys': 'GPU-2afcc1df-7df8-1f3f-8970-a252a06c6160',
        'Beauty': 'GPU-07374397-b213-09c1-2988-718acb6ed231'}
METHODS = ('hard_filter', 'eligible_fixed', 'path_rho05', 'path_rho10')
base.OUT = OUT


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare():
    import torch
    sys.path.insert(0, str(ROOT / 'third_party/LIME-Rec'))
    from lime_rec.evaluation import normalize_score_rows
    from experiment.phase20.catalog_scope_core import topk
    torch.set_num_threads(2)
    assert all(base.read(p / 'status.json')['state'] == 'COMPLETED' for p in PRIORS)
    OUT.mkdir(exist_ok=False)
    base.write(OUT / 'status.json', dict(state='PREPARING', updated_at=base.stamp(), test_read=False))
    config = dict(domains=base.DOMAINS, gpus=GPUS, evaluation_users=512, smoke_users=3,
        alpha=.3, rhos=[0., .5, 1.], score_scale=20., beam_size=50, length_penalty=1.,
        minimum_free_mib=12288, allocation_limit_gib=11, worker_timeout_seconds=None,
        filtering='Exactly existing final-filter input history; no lifetime-history expansion',
        selection='All arms frozen before evaluation; no outcome-based parameter selection',
        no_model_training=True, test_read=False, development_diagnostic=True,
        authorization='User explicitly authorized autonomous background launch and status-file observation.')
    base.write(OUT / 'config.json', config)
    (OUT / 'protocol.md').write_text(PROTOCOL.read_text())
    manifest = dict(base.read(PARENT / 'manifest.json')['sha256'])
    def use(path):
        path = Path(path); rel = str(path.relative_to(ROOT)); digest = base.sha(path)
        assert '/test/' not in rel and not path.name.startswith('test.json')
        if rel in manifest:
            assert manifest[rel] == digest, ('Changed inherited input', rel)
        manifest[rel] = digest
        return path
    for domain in base.DOMAINS:
        folder = OUT / domain; folder.mkdir()
        previous = [base.read(use(p / domain / 'samples.json')) for p in PRIORS]
        excluded = {r['user_id'] for cohort in previous for r in cohort}
        cfg = base.read(use(PARENT / domain / 'data_config.json'))
        catalog = base.read(use(ROOT / cfg['prepared'] / 'catalog.json'))
        validation = rows(use(ROOT / cfg['prepared'] / 'validation.jsonl'))
        vi = {r['user_id']: i for i, r in enumerate(validation)}
        query = base.arrays(use(ROOT / f'artifacts/phase20/catalog_scope_20260919_v1/{domain}/queries.npz'))
        recovery = ROOT / f'artifacts/phase20/lime_comparison_20260920_v1/recovery_v2/{domain}'
        old = base.arrays(use(recovery / 'evaluation_seed0_cal2048.npz'))
        gate = torch.load(use(recovery / 'gate_seed0_cal2048.pt'), map_location='cpu')
        np.testing.assert_array_equal(np.sort(gate['fit_indices']), query['calibration_indices'])
        assert set(old['indices']).isdisjoint(gate['fit_indices'])
        np.testing.assert_array_equal(old['users'], query['users'][old['indices']])
        by_user = {str(u): i for i,u in enumerate(old['users'])}
        chosen = sorted([i for i,u in enumerate(old['users']) if str(u) not in excluded],
            key=lambda i: hashlib.sha256(f'20260922|lime-eligible-path-v1|{domain}|{old["users"][i]}'.encode()).digest())[:512]
        assert len(chosen) == 512
        positions = [by_user[r['user_id']] for r in previous[0][:3]] + chosen
        samples = []
        for j, pos in enumerate(positions):
            qi = int(old['indices'][pos]); user = str(old['users'][pos]); record = validation[vi[user]]
            history = query['history'][qi]; history = history[history > 0].tolist()
            assert record['history'] == history and record['target'] == int(old['targets'][pos])
            samples.append(dict(user_id=user, target=record['target'], history=history,
                validation_index=vi[user], query_index=qi, evaluation_cache_index=pos,
                split='smoke' if j < 3 else 'evaluation'))
        assert samples[:3] == previous[0][:3]
        assert len({s['user_id'] for s in samples}) == 515
        ix = np.array([r['query_index'] for r in samples])
        paths = [recovery / 'sasrec_scores_seed0.npy',
                 ROOT / f'artifacts/phase20/lime_comparison_20260920_v1/{domain}/itemcf_scores.npy',
                 recovery / 'semantic_scores.npy']
        normalized = np.stack([normalize_score_rows(np.asarray(np.load(use(p), mmap_mode='r')[ix])) for p in paths], axis=1)
        weights = old['weights'][positions].astype('float32')
        penalty = old['penalty'][positions].astype('float32')
        fused = (torch.from_numpy(weights)[:, :, None] * torch.from_numpy(normalized)).sum(1)
        bank = base.arrays(use(ROOT / f'artifacts/phase20/nearest_comparison_20260918/{domain}.train_sequences.npz'))
        ui = {str(u): i for i,u in enumerate(bank['users'])}
        for j,sample in enumerate(samples):
            k = ui[sample['user_id']]
            history = bank['items'][bank['offsets'][k]:bank['offsets'][k+1]]
            assert history[-20:].tolist() == sample['history']
            fused[j, np.unique(history)-1] -= penalty[j]
        fused = fused.numpy(); masked = fused.copy()
        for j,sample in enumerate(samples):
            masked[j, np.asarray(sample['history'])-1] = -np.inf
        tops = topk(masked, np.arange(1,len(catalog['items'])))
        np.testing.assert_array_equal(tops, old['top50'][old['arm_names'].tolist().index('LIME'),positions])
        np.testing.assert_array_equal(fused[:3], base.arrays(use(PARENT / domain / 'signals.npz'))['lime'][:3])
        np.savez_compressed(folder / 'signals.npz', lime=fused, lime_top50=tops)
        base.write(folder / 'samples.json', samples)
        base.write(folder / 'data_config.json', cfg)
        base.write(folder / 'selection.json', dict(alpha=.3, rhos=[0., .5, 1.],
            evaluation_users_read=0, selected_at=base.stamp(), new_tuning_users=0,
            criterion='Predeclared arms; no selection among new rho settings in this screen'))
        base.write(folder / 'preflight.json', dict(samples=515, evaluation_users=512,
            all_three_previous_cohorts_excluded=True, gate_fit_disjoint=True,
            LIME_top50_exact=True, history_calibration_retained=True, test_read=False,
            caveat='Historical upstream models and development data; not a final clean-data claim'))
        base.write(folder / 'status.json', dict(state='PENDING', domain=domain, stage='pending',
            updated_at=base.stamp(), evaluation_users_written=0, evaluation_users_total=512))
        for name in ('signals.npz','samples.json','data_config.json','selection.json','preflight.json'):
            use(folder / name)
    for name in ('lime_eligible_path_screen.py','lime_eligible_core.py','lime_path_mixture_core.py',
                 'test_lime_eligible.py','test_lime_path_mixture.py','verify_lime_eligible_path.py',
                 'run_lime_eligible_path.sh','verify_lime_mechanism.py'):
        use(ROOT / 'experiment/phase20' / name)
    for p in (PROTOCOL, OUT / 'protocol.md', OUT / 'config.json',
              ROOT / 'experiment/phase18/core/diff_generation_cache.py'):
        use(p)
    base.write(OUT / 'manifest.json', dict(sha256=manifest, created_at=base.stamp()))
    base.write(OUT / 'status.json', dict(state='PREPARED', updated_at=base.stamp(),
        domains={d:base.read(OUT/d/'status.json') for d in base.DOMAINS},
        frozen_inputs=len(manifest), test_read=False))
    print(json.dumps(dict(prepared=str(OUT), frozen_inputs=len(manifest))), flush=True)


def comparison_pairs(names):
    pairs = {n+'_vs_GRAM': (n,'GRAM') for n in names if n != 'GRAM'}
    for name in ('fixed',) + METHODS:
        pairs[name+'_vs_late'] = (name,name+'_late')
    for a,b in [('GRAM_filter','GRAM'), ('hard_filter','fixed'), ('eligible_fixed','hard_filter'),
                ('eligible_fixed','fixed'), ('path_rho05','eligible_fixed'), ('path_rho10','eligible_fixed'),
                ('path_rho05','GRAM_filter'), ('path_rho10','GRAM_filter')]:
        pairs[a+'_vs_'+b] = (a,b)
    return pairs


class Worker(base.Worker):
    def __init__(self, domain):
        super().__init__(domain)
        from experiment.phase18.core.diff_generation_cache import install_cross_cache_reuse
        install_cross_cache_reuse(self.model, 50)
        self.original_index, self.original_prefix = self.index, self.prefix_fn
        self.user_seconds = []
        self.selection_hash = base.sha(self.folder/'selection.json')
        for name in ('lime_eligible_path_screen.py','lime_eligible_core.py','lime_path_mixture_core.py'):
            path = ROOT/'experiment/phase20'/name
            self.loaded_source[str(path.relative_to(ROOT))] = base.sha(path)
        base.write(self.folder/'loaded_sources.json', self.loaded_source)

    def status(self, state='RUNNING', **kw):
        import torch
        path = self.folder/'status.json'
        previous = base.read(path) if path.exists() else {}
        value = dict(previous, state=state, stage=self.stage, domain=self.domain, pid=os.getpid(),
            updated_at=base.stamp(), elapsed_seconds=time.monotonic()-self.started,
            completed_decodes=self.completed, current_sample=getattr(self,'current_sample',None),
            current_arm=getattr(self,'current_arm',None),
            mean_decode_seconds=float(np.mean(self.seconds[-16:])) if self.seconds else None,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            test_read=False, optimizer_steps=0, alpha=.3)
        value.update(kw); base.write(path,value)

    def decode_view(self, ctx, index, prefix_fn, q=None, alpha=0., rho=0.):
        import torch
        from transformers import LogitsProcessorList
        from model.gram import GRAM
        from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
        begin=time.monotonic(); guide=PathMixtureGuide(index,q,alpha,rho)
        common=dict(max_length=self.max_length, prefix_allowed_tokens_fn=prefix_fn,
            num_beams=50,num_return_sequences=50,return_dict_in_generate=True,output_scores=True,
            length_penalty=1.,logits_processor=LogitsProcessorList([guide]))
        with torch.no_grad():
            encoded=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=ctx['hidden'])
            output=super(GRAM,self.model).generate(input_ids=ctx['ids'].reshape(1,-1),
                attention_mask=ctx['mask'].reshape(1,-1),encoder_outputs=encoded,**common)
        paths=[p[:p.index(1)+1] for p in output.sequences.cpu().tolist()]
        ids=[index.path_to_item[tuple(p)] for p in paths]
        scores=output.sequences_scores.detach().cpu().double().numpy()
        assert len(set(ids))==50 and np.isfinite(scores).all()
        np.testing.assert_allclose(guide.rescore(paths), scores, atol=2e-6, rtol=1e-5)
        del output
        torch.cuda.synchronize(); elapsed=time.monotonic()-begin
        self.seconds.append(elapsed); self.completed+=1
        return dict(ids=ids,paths=paths,scores=scores,guide=guide,seconds=elapsed)

    def one(self,j,smoke=False):
        self.current_sample=j; sample=self.samples[j]; ctx=self.context(j)
        eligible=EligibleTrie(self.original_index,sample['history'])
        full_q=self.original_index.conditional_logs(self.signals['lime'][j])
        legal_q=eligible.conditional_logs(self.signals['lime'][j])
        hard_q=eligible.condition_original_logs(full_q)
        arms,saved,timing,diagnostics,paths = {},{},{},{},{}
        def add(name,ids,scores):
            result=ordered(ids,scores,sample['history'])[:50]
            arms[name]=dict(items=result,rank=target_rank(result,sample['target']))
            saved[name]=dict(ids=list(map(int,ids)),scores=np.asarray(scores).tolist())
        def decode(name,index,prefix,q=None,alpha=0.,rho=0.):
            self.current_arm=name; self.status()
            out=self.decode_view(ctx,index,prefix,q,alpha,rho)
            add(name,out['ids'],out['scores']); timing[name]=out['seconds']
            diagnostics[name]=out['guide'].diagnostics(); paths[name]=out['paths']
            if index is eligible:
                assert set(out['ids']).isdisjoint(sample['history']) and len(arms[name]['items'])==50
            return out
        native=decode('GRAM',self.original_index,self.original_prefix)
        fixed=decode('fixed',self.original_index,self.original_prefix,full_q,.3)
        late=rescore_paths(native['guide'].trace,native['paths'],full_q,.3,0.)
        add('fixed_late',native['ids'],late)
        def check_shared(out,reference,values):
            lookup=dict(zip(reference['ids'],values))
            for item,score in zip(out['ids'],out['scores']):
                if item in lookup:
                    np.testing.assert_allclose(score,lookup[item],atol=5e-6,rtol=1e-5)
        check_shared(fixed,native,late)
        filtered=decode('GRAM_filter',eligible,eligible.allowed_tokens)
        settings={'hard_filter':(hard_q,0.),'eligible_fixed':(legal_q,0.),
                  'path_rho05':(legal_q,.5),'path_rho10':(legal_q,1.)}
        shift=j%4
        for name in METHODS[shift:]+METHODS[:shift]:
            q,rho=settings[name]
            out=decode(name,eligible,eligible.allowed_tokens,q,.3,rho)
            late=rescore_paths(filtered['guide'].trace,filtered['paths'],q,.3,rho)
            add(name+'_late',filtered['ids'],late); check_shared(out,filtered,late)
        add('LIME_after_GRAM',native['ids'],self.signals['lime'][j][np.array(native['ids'])-1])
        add('LIME_after_GRAM_filter',filtered['ids'],self.signals['lime'][j][np.array(filtered['ids'])-1])
        ids=self.signals['lime_top50'][j].tolist()
        arms['LIME']=dict(items=ids,rank=target_rank(ids,sample['target']))
        if smoke:
            for label,index,prefix,q,reference in [('zero_original',self.original_index,self.original_prefix,full_q,native),
                                                 ('zero_filtered',eligible,eligible.allowed_tokens,legal_q,filtered)]:
                self.current_arm=label; self.status()
                zero=self.decode_view(ctx,index,prefix,q,0.,1.)
                assert zero['paths']==reference['paths']
                np.testing.assert_array_equal(zero['scores'],reference['scores'])
        return dict(user_id=sample['user_id'],target=sample['target'],history=sample['history'],
            sample_index=j,split=sample['split'],arms=arms,candidate_scores=saved,
            decode_seconds=timing,diagnostics=diagnostics,paths=paths,
            filtered_candidates_all_eligible=True,shared_path_scores_match=True,
            raw_history_counts={name:len(set(saved[name]['ids'])&set(sample['history'])) for name in ('GRAM','fixed')})

    def run(self):
        self.stage='engineering_smoke'
        checks=[self.one(j,smoke=True) for j in range(3)]
        assert self.digest()==self.initial_digest
        base.write(self.folder/'smoke.json',dict(users=3,zero_exact=True,checks=checks,
            initial_digest=self.initial_digest,all_filtered_paths_eligible=True))
        self.stage='evaluation'; self.status(engineering_checks_passed=True)
        evaluation=[]
        with (self.folder/'evaluation.jsonl').open('x') as f:
            for k,j in enumerate(range(3,515),1):
                begin=time.monotonic(); row=self.one(j); evaluation.append(row)
                f.write(json.dumps(row)+'\n'); f.flush()
                self.user_seconds.append(time.monotonic()-begin)
                per_user=float(np.mean(self.user_seconds[-16:]))
                self.status(evaluation_users_written=k,evaluation_users_total=512,
                    mean_recent_user_seconds=per_user,projected_remaining_hours=(512-k)*per_user/3600,
                    eta_caveat='Measured complete-user runtime; contention may change the estimate')
        assert base.sha(self.folder/'selection.json')==self.selection_hash
        assert self.digest()==self.initial_digest
        for path,digest in self.loaded_source.items():
            assert base.sha(ROOT/path)==digest,('Changed loaded source',path)
        ranks={n:np.array([r['arms'][n]['rank'] for r in evaluation]) for n in evaluation[0]['arms']}
        pairs=comparison_pairs(ranks)
        base.write(self.folder/'summary.json',dict(domain=self.domain,alpha=.3,rhos=[0.,.5,1.],
            evaluation_users=512,new_tuning_users=0,metrics={n:metrics(v) for n,v in ranks.items()},
            comparisons={n:contrast(ranks[a],ranks[b]) for n,(a,b) in pairs.items()},comparison_pairs=pairs,
            completed_decodes=self.completed,selection_sha256=self.selection_hash,
            model_unchanged=True,test_read=False,optimizer_steps=0,development_diagnostic=True,
            elapsed_seconds=time.monotonic()-self.started,
            mean_decode_seconds_by_arm={n:float(np.mean([r['decode_seconds'][n] for r in evaluation])) for n in evaluation[0]['decode_seconds']},
            mean_eligible_candidate_count={n:float(np.mean([len(r['arms'][n]['items']) for r in evaluation])) for n in ranks},
            mean_decoder_steps={n:float(np.mean([r['diagnostics'][n]['decoder_steps'] for r in evaluation])) for n in evaluation[0]['diagnostics']}))
        self.stage='done'; self.status('COMPLETED',projected_remaining_hours=0.)


def finish():
    summaries={d:base.read(OUT/d/'summary.json') for d in base.DOMAINS}
    decisions={}
    for method in ('path_rho05','path_rho10'):
        passing=all(s['comparisons'][method+'_vs_eligible_fixed']['relative_N10']>=.005
            and s['comparisons'][method+'_vs_eligible_fixed']['net_H10']>=0 for s in summaries.values())
        decisions[method]='CANDIDATE_FOR_CLEAN_RETEST' if passing else 'NO_CLEAR_DUAL_DOMAIN_INCREMENT'
    base.write(OUT/'summary.json',dict(domains=summaries,decisions=decisions,
        test_read=False,independent_verification='verification.json',automatic_new_experiments=False,
        caveat='Development screening only; threshold is resource triage, not significance or publication evidence'))
    lines=['# LIME 过滤与路径融合筛查','','状态：COMPLETED / VERIFIED。',
        '每域512名用户，排除前三轮解码评价用户和门控拟合用户；仍属于历史开发数据。',
        '固定 alpha=0.3；rho=0、0.5、1 全部报告；无本轮选参、无模型训练、未读取测试集。','']
    for domain,s in summaries.items():
        lines += [f'## {domain}','','| 方法 | NDCG@10 | Hit@10 | Hit@50 |','|---|---:|---:|---:|']
        for name,m in s['metrics'].items():
            lines.append(f'| {name} | {m["N10"]:.6f} | {m["H10"]:.6f} | {m["H50"]:.6f} |')
        lines.append('')
    lines += ['路径机制与 eligible_fixed 比较；过滤收益独立报告。',json.dumps(decisions,ensure_ascii=False),
              '保留同规则后重排与直接 LIME 重排；没有自动追加实验。']
    REPORT.write_text('\n'.join(lines)+'\n')
    return decisions


def supervise():
    config=base.read(OUT/'config.json'); pending=set(base.DOMAINS); active={}; done={}; started=time.monotonic()
    while pending or active:
        for d in sorted(pending):
            if base.gpu_free(GPUS[d])<config['minimum_free_mib']:
                continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[d],PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',
                     MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            with (OUT/d/'worker.log').open('x') as log:
                active[d]=subprocess.Popen([sys.executable,'-m',MODULE,'worker','--domain',d],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            pending.remove(d)
        for d,child in list(active.items()):
            if child.poll() is not None:
                done[d]=child.returncode; del active[d]
                if child.returncode:
                    previous=base.read(OUT/d/'status.json')
                    base.write(OUT/d/'status.json',dict(previous,state='FAILED',exit_code=child.returncode,updated_at=base.stamp()))
        status=dict(state='RUNNING',updated_at=base.stamp(),supervisor_pid=os.getpid(),background=True,
            active={d:p.pid for d,p in active.items()},pending=sorted(pending),completed=done,
            elapsed_seconds=time.monotonic()-started,domains={d:base.read(OUT/d/'status.json') for d in base.DOMAINS},
            test_read=False,automatic_new_experiments=False,wall_clock_timeout_seconds=None,
            result_path=str((OUT/'summary.json').relative_to(ROOT)),report_path=str(REPORT.relative_to(ROOT)))
        for d in pending:
            status['domains'][d]['state']='WAITING_FOR_MEMORY'
        base.write(OUT/'status.json',status)
        if pending or active:
            time.sleep(15)
    if len(done)==2 and all(rc==0 for rc in done.values()):
        with (OUT/'audit.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m','experiment.phase20.verify_lime_eligible_path'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                status.update(state='VERIFYING',updated_at=base.stamp(),audit_pid=child.pid)
                base.write(OUT/'status.json',status); time.sleep(5)
        if child.returncode==0:
            status.update(state='COMPLETED',decisions=finish(),independently_verified=True)
        else:
            status.update(state='AUDIT_FAILED',audit_exit_code=child.returncode)
    else:
        status['state']='INCOMPLETE'
    status.update(updated_at=base.stamp(),elapsed_seconds=time.monotonic()-started)
    base.write(OUT/'status.json',status)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','supervise','worker'])
    parser.add_argument('--domain',choices=base.DOMAINS)
    args=parser.parse_args()
    if args.action=='prepare':
        prepare()
    elif args.action=='supervise':
        assert base.read(OUT/'status.json')['state']=='PREPARED'
        with (OUT/'launch.json').open('x') as f:
            json.dump(dict(pid=os.getpid(),started_at=base.stamp(),host_tmux=True),f)
        try:
            supervise()
        except Exception:
            base.write(OUT/'status.json',dict(state='SUPERVISOR_FAILED',updated_at=base.stamp(),
                error=traceback.format_exc(),test_read=False)); raise
    else:
        try:
            Worker(args.domain).run()
        except Exception:
            detail=traceback.format_exc(); print(detail,flush=True)
            previous=base.read(OUT/args.domain/'status.json')
            base.write(OUT/args.domain/'status.json',dict(previous,state='FAILED',updated_at=base.stamp(),
                error=detail,automatic_retry=False)); raise


if __name__=='__main__':
    main()
