"""Frozen CPU diagnostic: match users, change historical vs next-query supervision.

C fits other users' development labels. This is exploratory OOF evidence, not
an untouched test, a causal exposure experiment, or a new recommendation method.
"""
import argparse
import contextlib
import hashlib
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now
from experiment.phase20.catalog_scope_core import ranks, metrics, contrast
from experiment.phase20.source_competition_core import ResidualRanker, row_losses, target_positions, independent_ranks

CONFIG = ROOT/'experiment/phase20/lime_learnability_20260920_v1.json'
ARMS = ('B_historical', 'C_next_interaction')


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def user_folds(users, domain, seed=20260920, count=5):
    return np.asarray([int.from_bytes(hashlib.sha256(
        f'{seed}|lime-learnability|{domain}|{u}'.encode()).digest()[:8], 'big') % count for u in users])


def last_prefix_indices(users, indices):
    selected = {}
    for row, (user, index) in enumerate(zip(users, indices)):
        user = str(user)
        if user not in selected or index > indices[selected[user]]:
            selected[user] = row
    return selected


def cohorts(train, dev, domain, config):
    latest = last_prefix_indices(train['users'], train['train_indices'])
    assert len(set(dev['users'])) == len(dev['users'])
    assert set(latest) <= set(map(str, dev['users']))
    match = np.asarray([latest.get(str(u), -1) for u in dev['users']])
    folds = user_folds(dev['users'], domain, config['fold_seed'], config['folds'])
    result = []
    for fold in range(config['folds']):
        fit_next = np.flatnonzero((match >= 0) & (folds != fold))
        fit_hist = match[fit_next]
        evaluation = np.flatnonzero(folds == fold)
        np.testing.assert_array_equal(train['users'][fit_hist], dev['users'][fit_next])
        assert not set(dev['users'][fit_next]) & set(dev['users'][evaluation])
        assert not set(train['users'][fit_hist]) & set(dev['users'][evaluation])
        result.append(dict(fold=fold, B_historical=fit_hist, C_next_interaction=fit_next, evaluation=evaluation))
    np.testing.assert_array_equal(np.sort(np.concatenate([x['evaluation'] for x in result])), np.arange(len(folds)))
    return folds, match, result


def training_bundle(data, indices):
    used = np.asarray(indices)[data['positions'][indices] >= 0]
    assert len(used) and len(set(data['users'][used])) == len(used)
    valid = data['ids'][used] > 0
    x = data['x'][used]
    mean = x[valid].mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x[valid].std(0, dtype=np.float64), 1e-5).astype(np.float32)
    x = (x-mean)/std; x[~valid] = 0.
    bundle = dict(x=torch.from_numpy(x), prior=torch.from_numpy(data['prior'][used]),
        valid=torch.from_numpy(valid), pos=torch.from_numpy(data['positions'][used]))
    assert all(torch.isfinite(v).all() for v in bundle.values())
    return bundle, mean, std, used


def fingerprint(model):
    h = hashlib.sha256()
    for k,v in model.state_dict().items(): h.update(k.encode()); h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def fit(bundle, config, progress=None, steps=None):
    """Receives training tensors only; no evaluation labels or callback selection."""
    torch.manual_seed(config['seed'])
    model = ResidualRanker(hidden=config['hidden'], bound=config['residual_bound'])
    assert sum(p.numel() for p in model.parameters()) == 385
    initial = fingerprint(model)
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    curve = []; n = len(bundle['x']); started = time.monotonic()
    for step in range(config['steps'] if steps is None else steps):
        opt.zero_grad(); total = 0.
        for start in range(0, n, config['chunk_size']):
            sl = slice(start,start+config['chunk_size'])
            scores = model(bundle['x'][sl],bundle['prior'][sl])
            loss = row_losses(scores,bundle['valid'][sl],bundle['pos'][sl]).sum()/n
            assert torch.isfinite(loss)
            loss.backward(); total += float(loss.detach())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step(); curve.append(total)
        if progress is not None: progress(step+1,total)
    assert fingerprint(model) != initial
    return model, dict(initial_sha256=initial, final_sha256=fingerprint(model),
        loss_before_last_update=curve[-1], curve=curve, seconds=time.monotonic()-started, training_users=n)


def predict(model, x, prior, ids, mean, std, chunk=128):
    """Inference API has no target, coverage, fold-label or user-label input."""
    output = []; model.eval()
    with torch.no_grad():
        for start in range(0,len(x),chunk):
            sl=slice(start,start+chunk); valid=ids[sl] > 0
            values=(x[sl]-mean)/std; values[~valid]=0.
            scores=model(torch.from_numpy(values),torch.from_numpy(prior[sl])).numpy()
            assert np.isfinite(scores).all()
            assert np.max(np.abs(scores-prior[sl])) <= model.bound+2e-6
            output.append(np.where(valid,scores,-np.inf))
    return np.concatenate(output)


class Run:
    def __init__(self, config, output):
        self.c,self.out=config,output
        self.start=time.monotonic(); self.steps=0; self.inputs={}
        torch.set_num_threads(config['cpu_threads']); torch.set_num_interop_threads(1)
        self.thread_limit=threadpool_limits(config['cpu_threads'])

    def status(self, stage, state='RUNNING', **extra):
        data=dict(state=state,stage=stage,updated_at=now(),pid=os.getpid(),
            elapsed_seconds=round(time.monotonic()-self.start,2),optimizer_steps=self.steps,
            expected_optimizer_steps=600,gpu_used=False,test_read=False,development_labels_used_for_fit=True,**extra)
        write(self.out/'status.json',data)
        print(data,flush=True)

    def use(self,path,expected=None):
        path=Path(path); rel=str(path.relative_to(ROOT))
        assert 'test.json' not in rel and '/test/' not in rel
        if rel not in self.inputs:self.inputs[rel]=sha(path)
        if expected is not None:assert self.inputs[rel]==expected,rel
        return path

    def freeze(self):
        receipt=read(self.use(ROOT/self.c['preflight_output']/'checks.json'))
        assert receipt['state']=='PASSED'
        for rel,digest in receipt['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
        paths=list((ROOT/'experiment/phase20').glob('*lime_learnability*'))
        paths += [ROOT/'experiment/phase20'/x for x in ('source_competition_core.py','catalog_scope_core.py','guided_decoding_job.py')]
        paths += [ROOT/self.c['contract']]
        for path in paths:
            if path.is_file():
                self.use(path); dst=self.out/'sources'/path.relative_to(ROOT)
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
        write(self.out/'config.json',self.c)
        shutil.copy2(ROOT/self.c['contract'],self.out/'protocol.md')
        self.parent=read(self.use(ROOT/self.c['input']/'summary.json'))
        assert self.parent['state']=='COMPLETED' and self.parent['verification']['state']=='VERIFIED'

    def load_domain(self,domain):
        data=[]
        for name in ('train.npz','development.npz'):
            path=ROOT/self.c['input']/domain/name
            digest=self.parent['output_sha256'][str(path.relative_to(ROOT))]
            d=arrays(self.use(path,digest))
            np.testing.assert_array_equal(target_positions(d['ids'],d['targets']),d['positions'])
            data.append(d)
        train,dev=data
        assert len(dev['users'])==dict(Toys=17364,Beauty=20315)[domain]
        folds,match,parts=cohorts(train,dev,domain,self.c)
        assert (match>=0).sum()==dict(Toys=3494,Beauty=4085)[domain]
        counts=[dict(fold=p['fold'],fit_users=len(p['C_next_interaction']),
            historical_covered=int((train['positions'][p['B_historical']]>=0).sum()),
            next_covered=int((dev['positions'][p['C_next_interaction']]>=0).sum()),evaluation_users=len(p['evaluation'])) for p in parts]
        return train,dev,folds,match,parts,counts

    def train_domain(self,domain):
        folder=self.out/domain; folder.mkdir()
        self.status('核对缓存和按用户留出边界',domain=domain)
        train,dev,folds,match,parts,counts=self.load_domain(domain)
        write(folder/'cohorts.json',dict(matched_users=int((match>=0).sum()),folds=counts,
            matched_precoverage_users=True,coverage_counts_not_matched=True,fold_seed=self.c['fold_seed']))
        np.savez_compressed(folder/'mapping.npz',users=dev['users'],folds=folds,historical_rows=match)
        records={}
        for p in parts:
            fold=p['fold']; initial=None
            for arm in ARMS:
                data=train if arm=='B_historical' else dev
                bundle,mean,std,used=training_bundle(data,p[arm])
                assert not set(data['users'][used]) & set(dev['users'][p['evaluation']])
                def progress(step,loss):
                    self.steps+=1
                    if step%10==0 or step==1:
                        self.status('拟合普通排序器',domain=domain,fold=fold,arm=arm,step=step,loss=loss,
                            fit_covered_users=len(used))
                model,meta=fit(bundle,self.c,progress)
                if initial is None:initial=meta['initial_sha256']
                assert initial==meta['initial_sha256']
                path=folder/f'fold{fold}.{arm}.pt'
                torch.save(dict(state_dict=model.state_dict(),feature_mean=mean,feature_std=std,
                    fit_indices=used,cohort_indices=p[arm],fit_users=data['users'][used],
                    evaluation_indices=p['evaluation'],domain=domain,fold=fold,arm=arm,config=self.c,**meta),path)
                records[f'fold{fold}.{arm}']=dict(checkpoint=str(path.relative_to(ROOT)),**meta)
        write(folder/'training.json',records)

    def evaluate_domain(self,domain):
        folder=self.out/domain
        train,dev,folds,match,parts,counts=self.load_domain(domain)
        n=len(dev['users']);rr={arm:np.full(n,-1,dtype=np.int64) for arm in ARMS}
        predicted={arm:np.zeros(n,dtype=np.int8) for arm in ARMS};all_scores={arm:np.empty_like(dev['prior']) for arm in ARMS}
        replayed=0
        for p in parts:
            for arm in ARMS:
                self.status('折外预测与保存模型重放',domain=domain,fold=p['fold'],arm=arm)
                cp=torch.load(folder/f'fold{p["fold"]}.{arm}.pt',map_location='cpu')
                source=train if arm=='B_historical' else dev
                _,mu,sd,used=training_bundle(source,p[arm])
                for a,b in ((used,cp['fit_indices']),(mu,cp['feature_mean']),(sd,cp['feature_std']),
                    (cp['evaluation_indices'],p['evaluation'])):np.testing.assert_array_equal(a,b)
                ix=p['evaluation'];assert not set(cp['fit_users']) & set(dev['users'][ix])
                model=ResidualRanker(hidden=self.c['hidden'],bound=self.c['residual_bound'])
                model.load_state_dict(cp['state_dict']);assert fingerprint(model)==cp['final_sha256']
                score=predict(model,dev['x'][ix],dev['prior'][ix],dev['ids'][ix],mu,sd)
                r=ranks(score,dev['ids'][ix],dev['targets'][ix])
                np.testing.assert_array_equal(r,independent_ranks(score,dev['ids'][ix],dev['targets'][ix]))
                # Different prediction chunking guards against row/batch assembly mistakes.
                replay=predict(model,dev['x'][ix],dev['prior'][ix],dev['ids'][ix],mu,sd,chunk=97)
                np.testing.assert_allclose(score,replay,rtol=1e-6,atol=2e-6)
                np.testing.assert_array_equal(r,independent_ranks(replay,dev['ids'][ix],dev['targets'][ix]))
                rr[arm][ix]=r;predicted[arm][ix]+=1;all_scores[arm][ix]=score;replayed+=len(ix)
        for arm in ARMS:
            assert (predicted[arm]==1).all() and (rr[arm]>0).all()
        rr.update(gram_lime=dev['gram_lime_ranks'],union_lime=ranks(np.where(dev['ids']>0,dev['prior'],-np.inf),dev['ids'],dev['targets']),
            crossfit_gram_lime=dev['crossfit_gram_lime_ranks'])
        for j,r in enumerate(dev['A_ranks']):rr[f'A_head{j}']=r
        groups=dict(all=np.ones(n,bool),low_frequency=dev['tail'],other=~dev['tail'],matched_cohort=match>=0,
            other_users=match<0,gram_covered=dev['target_in_gram'],lime_only=(dev['positions']>=0)&~dev['target_in_gram'])
        table={arm:{g:metrics(r[m]) for g,m in groups.items()} for arm,r in rr.items()}
        table['A_mean']={g:{k:float(np.mean([table[f'A_head{j}'][g][k] for j in range(len(dev['A_ranks']))])) for k in table['gram_lime'][g]} for g in groups}
        byfold={str(f):{arm:metrics(r[folds==f]) for arm,r in rr.items()} for f in range(self.c['folds'])}
        C=table['C_next_interaction'];B=table['B_historical'];G=table['gram_lime']
        positive=sum(byfold[str(f)]['C_next_interaction']['N10']>byfold[str(f)]['gram_lime']['N10'] for f in range(self.c['folds']))
        checks=dict(N10_over_gram_lime_one_percent=C['all']['N10']>=G['all']['N10']*(1+self.c['minimum_relative_N10']),
            H10_not_below_gram_lime=C['all']['H10']>=G['all']['H10'],four_positive_folds=positive>=self.c['minimum_positive_folds'],
            N10_over_historical=C['all']['N10']>B['all']['N10'],
            low_frequency_H10_tolerance=C['low_frequency']['H10']>=G['low_frequency']['H10']-self.c['tail_H10_tolerance'])
        comparisons={control:contrast(rr['C_next_interaction'],rr[control],self.c['bootstrap_repeats'],20260920)
            for control in ('B_historical','gram_lime','union_lime')}
        # Independent metric reconstruction from ranks and complete denominators.
        fields=0
        for arm,r in rr.items():
            for group,mask in groups.items():
                for k in (5,10,20,50):
                    hit=r[mask]<=k;v=r[mask][hit]
                    values={'H':float(hit.sum()/mask.sum()),'N':float(np.sum(1/np.log2(v+1))/mask.sum())}
                    for name,value in values.items():
                        assert abs(value-table[arm][group][f'{name}{k}'])<1e-12;fields+=1
        np.savez_compressed(folder/'predictions.npz',users=dev['users'],targets=dev['targets'],folds=folds,
            arm_names=np.asarray(list(rr)),ranks=np.stack(list(rr.values())),
            B_scores=all_scores[ARMS[0]],C_scores=all_scores[ARMS[1]],**{f'mask_{k}':v for k,v in groups.items()})
        result=dict(metrics=table,fold_metrics=byfold,positive_folds=positive,continuation_checks=checks,
            continue_to_data_design=all(checks.values()),comparisons=comparisons,
            group_sizes={k:int(v.sum()) for k,v in groups.items()},cohorts=counts,
            verification=dict(independent_rankings=replayed,independent_metric_fields=fields,
                checkpoint_replay_different_chunks=True,each_user_predicted_once_per_arm=True,
                evaluation_users_excluded_from_fit=True,fit_only_standardization_recomputed=True),
            test_read=False,development_labels_used_for_fit=True,formal_method_evaluation=False)
        write(folder/'summary.json',result)
        return result

    def run(self):
        self.freeze()
        # Neither new arm's development metrics are revealed until all 20 fits finish.
        for domain in self.c['domains']:self.train_domain(domain)
        assert self.steps==600
        results={d:self.evaluate_domain(d) for d in self.c['domains']}
        self.status('复核输入哈希与全部结果')
        for rel,digest in self.inputs.items():assert sha(ROOT/rel)==digest,rel
        decision=all(r['continue_to_data_design'] for r in results.values())
        report=['# LIME新候选源：按用户留出诊断结果','','Origin Skill: academic-research-suite。使用其他用户的开发标签拟合C，只作投入诊断，不是正式test或新方法成绩。','',
            '| 方法 | Toys N10 | Beauty N10 | Toys H10 | Beauty H10 |','|---|---:|---:|---:|---:|']
        for arm in ('gram_lime','union_lime',*ARMS,'A_mean','crossfit_gram_lime'):
            a,b=[results[d]['metrics'][arm]['all'] for d in self.c['domains']]
            report.append(f'| {arm} | {a["N10"]:.8f} | {b["N10"]:.8f} | {a["H10"]:.8f} | {b["H10"]:.8f} |')
        report+=['',f'预定继续条件：{"通过，只进入合法训练数据设计和成本评估" if decision else "未通过，停止当前10维冻结分数＋轻量重排配方的损失/门控/阈值搜索"}。',
            '20个模型/600次更新全部完成，所有评价用户均且仅有一次本折外的新头预测，保存模型以不同batch重放通过。',
            'B/C拟合前的用户和查询数一致，但已覆盖训练样本数量、时间、目标频次不同；不能把差异单独归因为消除训练记忆。',
            '原上游曾反复用开发集选型；bootstrap为保存OOF预测的用户级描述区间，未计入折之间的训练依赖、单初始化和多次研究选择。',
            '统计解释11项覆盖：报告域与低频分层；用户级分析；披露样本选择；覆盖分组只描述；完整用户分母与命中进出；不挑最好折；全20模型保留；不作显著性声明；不扫参；不作因果归因；遵守训练与预测标签边界。']
        (self.out/'report.md').write_text('\n'.join(report)+'\n')
        outputs={str(p.relative_to(ROOT)):sha(p) for p in self.out.rglob('*') if p.is_file() and p.suffix in ('.json','.pt','.npz','.md') and p.name not in ('status.json','launch.json')}
        write(self.out/'summary.json',dict(state='COMPLETED',completed_at=now(),elapsed_seconds=time.monotonic()-self.start,
            optimizer_steps=self.steps,new_upstream_updates=0,domains=results,continue_to_data_design=decision,
            input_sha256=self.inputs,output_sha256=outputs,gpu_used=False,test_read=False,development_labels_used_for_fit=True,
            formal_method_evaluation=False))
        self.status('检查完成，查看report.md',state='COMPLETED',continue_to_data_design=decision)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','preflight'));args=p.parse_args()
    if args.action=='preflight':
        from experiment.phase20.test_lime_learnability import preflight
        preflight();return
    c=read(CONFIG);out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=False)
    write(out/'launch.json',dict(pid=os.getpid(),started_at=now(),max_seconds=c['max_seconds'],cpu_threads=c['cpu_threads']))
    runner=Run(c,out)
    def timeout(*_):raise TimeoutError('Fixed 900-second budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(c['max_seconds'])
    with (out/'run.log').open('x',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:runner.run()
        except BaseException as e:
            runner.status('任务失败，保留全部输出',state='FAILED',error=repr(e));traceback.print_exc();raise
        finally:signal.alarm(0)


if __name__=='__main__':main()
