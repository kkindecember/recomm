"""One bounded CPU experiment; launch is durable, all scientific choices frozen."""
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
import torch
from threadpoolctl import threadpool_limits

from experiment.phase20.guided_decoding_job import ROOT, read, write, now, sha
from experiment.phase20.catalog_scope_core import ranks, metrics, utilities
from experiment.phase20.source_competition_core import ARMS, FEATURES, ResidualRanker, pair_masks, row_losses, independent_ranks
from experiment.phase20.source_competition_data import DataBuilder, arrays

CONFIG = ROOT/'experiment/phase20/source_competition_20260920_v2.json'


def fingerprint(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


class Experiment:
    def __init__(self, out, config):
        self.out, self.c = out, config
        self.start = time.monotonic(); self.inputs = {}; self.steps = 0
        torch.set_num_threads(config['cpu_threads'])
        torch.set_num_interop_threads(1)
        self.pool = threadpool_limits(config['cpu_threads'])

    def status(self, stage, state='RUNNING', **details):
        write(self.out/'status.json', dict(state=state, stage=stage, updated_at=now(),
            pid=os.getpid(), elapsed_seconds=round(time.monotonic()-self.start, 2),
            completed_optimizer_steps=self.steps, expected_optimizer_steps=540,
            gpu_used=self.c.get('gate_feature_device') == 'cuda', gpu_purpose='frozen_LIME_gate_only',
            peak_cuda_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,2) if torch.cuda.is_initialized() else 0,
            test_read=False, **details))

    def use(self, path):
        path = Path(path); key = str(path.relative_to(ROOT))
        assert '/test/' not in key and 'test.json' not in key
        if key not in self.inputs: self.inputs[key] = sha(path)
        return path

    def freeze(self):
        preflight=ROOT/self.c['preflight_output']/'checks.json'
        receipt=read(self.use(preflight))
        assert receipt['state'] == 'PASSED'
        for path,digest in receipt['source_sha256'].items(): assert sha(ROOT/path) == digest, path
        sources = list((ROOT/'experiment/phase20').glob('*source_competition*'))
        sources += list((ROOT/'third_party/LIME-Rec/lime_rec').rglob('*.py'))
        sources += [ROOT/'third_party/LIME-Rec/scripts/evaluation/run_repeat_aware_gate.py']
        sources += [ROOT/'experiment/phase20'/n for n in
            ('catalog_scope_core.py', 'lime_comparison_core.py', 'guided_decoding_job.py')]
        sources += [ROOT/self.c['contract']]
        for p in sources:
            if p.is_file():
                self.use(p); dst=self.out/'sources'/p.relative_to(ROOT)
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
        write(self.out/'config.json', self.c)
        shutil.copy2(ROOT/self.c['contract'], self.out/'protocol.md')

    def prepare_training(self, data):
        fit = data['positions'] >= 0
        assert fit.any() and (fit & ~data['target_in_gram']).any()
        x = data['x'][fit]; valid = data['ids'][fit] > 0
        mean = x[valid].mean(0, dtype=np.float64).astype(np.float32)
        std = np.maximum(x[valid].std(0, dtype=np.float64), 1e-5).astype(np.float32)
        x = (x-mean)/std; x[~valid] = 0
        users, counts = np.unique(data['users'][fit], return_counts=True)
        sizes = dict(zip(users, counts))
        w = np.array([1./(len(users)*sizes[u]) for u in data['users'][fit]], np.float32)
        assert abs(w.sum()-1) < 1e-6
        prior = data['prior'][fit]; pos = data['positions'][fit]; ing = data['ing'][fit]
        pairs = pair_masks(prior, valid, ing, pos, self.c['max_pairs'])
        tensors = dict(x=torch.from_numpy(x), prior=torch.from_numpy(prior), valid=torch.from_numpy(valid),
            pos=torch.from_numpy(pos), weights=torch.from_numpy(w),
            **{k:torch.from_numpy(v) for k,v in pairs.items()})
        metadata = dict(fit_examples=int(fit.sum()), fit_users=len(users), features=list(FEATURES),
            feature_mean=mean.tolist(), feature_std=std.tolist(),
            pair_counts_equal=True, max_pairs=int(pairs['cross_source'].sum(1).max()),
            pair_masks_differ_fraction=float((pairs['cross_source'] != pairs['hard_negative']).any(1).mean()),
            zero_pair_rows=int((pairs['cross_source'].sum(1) == 0).sum()))
        return tensors, mean, std, metadata

    def train(self, folder, train, mean, std):
        outputs = {}; initial = {}; training_times = []
        for seed in self.c['head_seeds']:
            for arm in self.c['arms']:
                torch.manual_seed(seed)
                model = ResidualRanker(hidden=self.c['hidden'], bound=self.c['residual_bound'])
                initial[seed] = initial.get(seed, fingerprint(model))
                assert fingerprint(model) == initial[seed]
                opt = torch.optim.AdamW(model.parameters(), lr=self.c['learning_rate'], weight_decay=self.c['weight_decay'])
                curve = []; tic = time.monotonic()
                for step in range(self.c['steps']):
                    step_start = time.monotonic(); opt.zero_grad(); total = 0.
                    for start in range(0, len(train['x']), self.c['chunk_size']):
                        sl=slice(start, start+self.c['chunk_size'])
                        score=model(train['x'][sl], train['prior'][sl])
                        pair = None if arm == 'ordinary_ce' else train[arm][sl]
                        loss=(row_losses(score, train['valid'][sl], train['pos'][sl], pair,
                            self.c['pair_weight'])*train['weights'][sl]).sum()
                        assert torch.isfinite(loss)
                        loss.backward(); total += float(loss.detach())
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    opt.step(); self.steps += 1
                    curve.append(total); training_times.append(time.monotonic()-step_start)
                    if step % 5 == 0 or step+1 == self.c['steps']:
                        self.status('训练匹配小排序器', domain=folder.name, arm=arm, seed=seed,
                            step=step+1, steps=self.c['steps'], loss=total,
                            seconds_per_update=float(np.mean(training_times[-10:])),
                            estimated_training_remaining_seconds=float(np.mean(training_times[-10:])*(540-self.steps)))
                assert fingerprint(model) != initial[seed]
                checkpoint=folder/f'{seed}.{arm}.pt'
                torch.save(dict(state_dict=model.state_dict(), seed=seed, arm=arm, steps=self.c['steps'],
                    feature_mean=mean, feature_std=std, config=self.c, training_curve=curve,
                    initial_sha256=initial[seed], final_sha256=fingerprint(model)), checkpoint)
                outputs[f'{seed}.{arm}']=dict(checkpoint=str(checkpoint.relative_to(ROOT)),
                    initial_loss=curve[0], last_preupdate_loss=curve[-1], seconds=time.monotonic()-tic,
                    parameters=sum(p.numel() for p in model.parameters()))
        write(folder/'training.json', outputs)
        return outputs

    def predict(self, checkpoint, data):
        saved=torch.load(checkpoint, map_location='cpu')
        model=ResidualRanker(hidden=self.c['hidden'], bound=self.c['residual_bound'])
        model.load_state_dict(saved['state_dict']); model.eval()
        assert fingerprint(model) == saved['final_sha256']
        result=[]
        with torch.no_grad():
            for i in range(0, len(data['x']), self.c['chunk_size']):
                sl=slice(i,i+self.c['chunk_size']); valid=data['ids'][sl] > 0
                x=(data['x'][sl]-saved['feature_mean'])/saved['feature_std']; x[~valid]=0
                score=model(torch.from_numpy(x),torch.from_numpy(data['prior'][sl])).numpy()
                assert np.isfinite(score).all()
                assert np.max(np.abs(score-data['prior'][sl])) <= self.c['residual_bound']+2e-6
                result.append(np.where(valid, score, -np.inf))
        return np.concatenate(result)

    def evaluation(self, folder, data, training):
        groups=dict(all=np.ones(len(data['users']), bool), low_frequency=data['tail'],
            other=~data['tail'], gram_covered=data['target_in_gram'],
            lime_only=(data['positions'] >= 0) & ~data['target_in_gram'],
            neither=data['positions'] < 0)
        assert all(g.any() for g in groups.values())
        rr=dict(full_lime=data['full_lime_ranks'], gram_lime=data['gram_lime_ranks'],
            union_lime=ranks(np.where(data['ids'] > 0, data['prior'], -np.inf), data['ids'], data['targets']),
            crossfit_gram_lime=data['crossfit_gram_lime_ranks'])
        for j,r in enumerate(data['A_ranks']): rr[f'A_head{j}']=r
        for key, record in training.items():
            self.status('评价与独立名次核验', domain=folder.name, head=key)
            score=self.predict(ROOT/record['checkpoint'], data)
            rank=ranks(score, data['ids'], data['targets'])
            np.testing.assert_array_equal(rank, independent_ranks(score, data['ids'], data['targets']))
            rr[key]=rank
            np.savez_compressed(folder/f'{key}.predictions.npz', scores=score, ranks=rank)
        # Train labels never enter model.forward; these are retrospective groups.
        covered=data['target_in_gram']
        assert (rr['union_lime'][covered] >= rr['gram_lime'][covered]).all()
        names=list(rr)
        np.savez_compressed(folder/'ranks.npz', users=data['users'], targets=data['targets'],
            names=np.asarray(names), ranks=np.stack([rr[k] for k in names]),
            **{f'mask_{k}':v for k,v in groups.items()})
        table={k:{g:metrics(r[m]) for g,m in groups.items()} for k,r in rr.items()}
        aggregate={arm:{g:{metric:float(np.mean([table[f'{s}.{arm}'][g][metric] for s in self.c['head_seeds']]))
            for metric in table['gram_lime'][g]} for g in groups} for arm in ARMS}
        aggregate['A_mean']={g:{m:float(np.mean([table[f'A_head{j}'][g][m] for j in range(len(data['A_ranks']))]))
            for m in table['gram_lime'][g]} for g in groups}
        contrasts={}
        cross=np.stack([rr[f'{s}.cross_source'] for s in self.c['head_seeds']])
        for control in ('ordinary_ce','hard_negative','gram_lime','crossfit_gram_lime','A_mean'):
            if control in ARMS: ref=np.stack([rr[f'{s}.{control}'] for s in self.c['head_seeds']])
            elif control == 'A_mean': ref=data['A_ranks']
            else: ref=rr[control][None]
            dn=utilities(cross).mean(0)-utilities(ref).mean(0)
            dh=(cross <= 10).mean(0)-(ref <= 10).mean(0)
            rng=np.random.RandomState(20260920); boot=np.empty((self.c['bootstrap_repeats'], 2))
            for b in range(len(boot)):
                ix=rng.randint(0,len(dn),len(dn)); boot[b]=dn[ix].mean(),dh[ix].mean()
            contrasts[control]=dict(delta_N10=float(dn.mean()), delta_H10=float(dh.mean()),
                descriptive_N10_interval=np.percentile(boot[:,0],[2.5,97.5]).tolist(),
                descriptive_H10_interval=np.percentile(boot[:,1],[2.5,97.5]).tolist())
        movements={}
        for key,r in rr.items():
            base=rr['gram_lime']
            movements[key]=dict(gained=int(((r<=10)&(base>10)).sum()),
                lost=int(((r>10)&(base<=10)).sum()),
                covered_N10_contribution=float((utilities(r)-utilities(base))[covered].sum()/len(r)),
                outside_N10_contribution=float((utilities(r)-utilities(base))[~covered].sum()/len(r)))
        c=aggregate['cross_source']; threshold=self.c['minimum_relative_N10_gain']
        checks={}
        for control in ('ordinary_ce','hard_negative','gram_lime'):
            ref=(aggregate if control in ARMS else table)[control]['all']['N10']
            checks[f'N10_over_{control}']=c['all']['N10'] >= ref*(1+threshold)
        for control in ('ordinary_ce','hard_negative'):
            checks[f'all_seeds_over_{control}']=all(table[f'{s}.cross_source']['all']['N10'] >
                table[f'{s}.{control}']['all']['N10'] for s in self.c['head_seeds'])
        checks['H10_not_below_gram_lime']=c['all']['H10'] >= table['gram_lime']['all']['H10']
        checks['tail_H10_tolerance']=c['low_frequency']['H10'] >= table['gram_lime']['low_frequency']['H10']-self.c['tail_H10_absolute_tolerance']
        checks['N10_over_A_mean']=c['all']['N10'] > aggregate['A_mean']['all']['N10']
        checks['N10_over_crossfit_gram_lime']=c['all']['N10'] > table['crossfit_gram_lime']['all']['N10']
        result=dict(metrics=table, aggregate=aggregate, group_sizes={k:int(v.sum()) for k,v in groups.items()},
            cross_source_contrasts=contrasts, movements_vs_gram_lime=movements,
            continuation_checks=checks, continue_recipe=all(checks.values()),
            test_read=False, development_reused=True, sasrec_seed=0, gate='cal2048')
        write(folder/'summary.json', result)
        return result

    def verify(self):
        checked=0
        for domain in self.c['domains']:
            folder=self.out/domain; data=arrays(folder/'development.npz')
            records=read(folder/'training.json')
            for key,record in records.items():
                saved=arrays(folder/f'{key}.predictions.npz')
                score=self.predict(ROOT/record['checkpoint'],data)
                np.testing.assert_array_equal(score,saved['scores'])
                np.testing.assert_array_equal(independent_ranks(score,data['ids'],data['targets']),saved['ranks'])
                checked+=len(score)
            self.status('保存模型重放通过', domain=domain, checked_rankings=checked)
        for path,digest in self.inputs.items(): assert sha(ROOT/path) == digest, path
        return dict(state='VERIFIED', all_checkpoints_replayed=True, independent_rankings=checked,
            input_hashes_unchanged=True, frozen_input_count=len(self.inputs), gpu_used=True,
            gpu_purpose='frozen_LIME_gate_only', new_gpu_optimizer_steps=0, test_read=False)

    def report(self, results, verification):
        lines=['# 候选补漏与排序保持首轮筛查结果', '',
            'Origin Skill: academic-research-suite。固定上游seed0/cal2048，双域三小头种子。开发集探索性筛查，无test读取。', '',
            '| 方法 | Toys N10 | Beauty N10 | Toys H10 | Beauty H10 |', '|---|---:|---:|---:|---:|']
        for arm in ('A_mean','crossfit_gram_lime','full_lime','gram_lime','union_lime',*ARMS):
            m=[(results[d]['aggregate'] if arm in (*ARMS,'A_mean') else results[d]['metrics'])[arm]['all'] for d in self.c['domains']]
            lines.append(f'| {arm} | {m[0]["N10"]:.8f} | {m[1]["N10"]:.8f} | {m[0]["H10"]:.8f} | {m[1]["H10"]:.8f} |')
        decision=all(r['continue_recipe'] for r in results.values())
        lines += ['', f'预定继续条件：{"通过，可考虑扩大验证；尚非新方法确立" if decision else "未通过，停止扩大当前损失配方；不自动调参"}。',
            f'完成小头更新{self.steps}次；独立重放{verification["independent_rankings"]}条名次。',
            '逐域分组、三种子、命中得失和各继续条件见各域summary.json。',
            'A与crossfit外部参照监督来源不同；各新训练臂的候选、输入、容量、优化预算相同。',
            '统计解释11项均检查：保留双域/低频差异；按用户统计；报告校准用户排除和训练覆盖筛选；覆盖分组仅为事后描述；完整用户分母与命中进出；未挑最好种子；全部臂保留；不作显著性声明；冻结权重且不自动搜参；不作因果解释；不从标签分组推断推理时可知目标。',
            '限制：一个上游SASRec种子；训练内历史前缀表示；反复使用的开发集；普通负例选择仍是核心可替代解释；bootstrap区间仅描述，不校正历史多重探索。']
        (self.out/'report.md').write_text('\n'.join(lines)+'\n')

    def run(self):
        self.freeze(); results={}; preparation={}
        # Finish all fits before revealing any new development metrics.
        for domain in self.c['domains']:
            folder=self.out/domain; folder.mkdir()
            self.status('加载冻结上游与核对训练边界', domain=domain)
            builder=DataBuilder(domain,self.c,self.use,self.status)
            train,meta=builder.train(); np.savez_compressed(folder/'train.npz',**train)
            del train
            data,dev_meta=builder.development()
            np.savez_compressed(folder/'development.npz',**data)
            preparation[domain]=dict(training=meta,development=dev_meta)
            write(folder/'preparation.json',preparation[domain]); del builder,data
        # All users in both domains pass reference parity before any new fitting.
        self.status('双域全量原排名复现通过，开始训练')
        for domain in self.c['domains']:
            folder=self.out/domain; train=arrays(folder/'train.npz')
            tensors,mean,std,fit_meta=self.prepare_training(train)
            preparation[domain]['fit']=fit_meta
            write(folder/'preparation.json',preparation[domain])
            self.train(folder,tensors,mean,std)
            del tensors,train
        for domain in self.c['domains']:
            folder=self.out/domain
            results[domain]=self.evaluation(folder,arrays(folder/'development.npz'),read(folder/'training.json'))
        self.status('核验全部输入与保存模型')
        verification=self.verify(); write(self.out/'verification.json',verification)
        outputs={str(p.relative_to(ROOT)):sha(p) for p in self.out.rglob('*')
            if p.is_file() and p.suffix in ('.pt','.npz','.json') and p.name not in ('status.json','launch.json')}
        summary=dict(state='COMPLETED', completed_at=now(), elapsed_seconds=time.monotonic()-self.start,
            optimizer_steps=self.steps, expected_optimizer_steps=540, new_upstream_updates=0,
            continue_recipe=all(r['continue_recipe'] for r in results.values()),
            input_sha256=self.inputs, output_sha256=outputs, verification=verification,
            preparation=preparation, domains=results, test_read=False, gpu_used=True,
            gpu_purpose='frozen_LIME_gate_only', new_gpu_optimizer_steps=0)
        assert self.steps == 540
        write(self.out/'summary.json',summary); self.report(results,verification)
        self.status('本轮完成，查看report.md',state='COMPLETED',continue_recipe=summary['continue_recipe'])


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('launch','run')); args=p.parse_args()
    config=read(CONFIG); out=ROOT/config['output']
    if args.action == 'launch':
        out.mkdir(parents=True,exist_ok=False)
        write(out/'status.json',dict(state='STARTING',updated_at=now(),test_read=False))
        with (out/'run.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-m','experiment.phase20.source_competition_screen','run'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=config['gate_gpu_uuid']))
        write(out/'launch.json',dict(pid=child.pid,started_at=now(),max_seconds=config['max_seconds'],
            command='python -B -m experiment.phase20.source_competition_screen run'))
        print(json.dumps(dict(pid=child.pid,status=str(out/'status.json')),ensure_ascii=False)); return
    def timeout(*_): raise TimeoutError('Fixed 3600-second whole-experiment budget exhausted')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(config['max_seconds'])
    exp=Experiment(out,config)
    try: exp.run()
    except BaseException as error:
        exp.status('任务停止，保留所有输出',state='FAILED',error=repr(error))
        traceback.print_exc(); raise
    finally: signal.alarm(0)


if __name__ == '__main__': main()
