"""Frozen cached controls: score transform, residual on fixed candidates, union.

Only prepared train/validation and existing validation caches are read. There is
no optimizer, negative mining, beam generation, hyperparameter search or test.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from experiment.phase20.audit_pcrf_components import digest, read_rows, metric, paired
from experiment.phase18.core.diff_data import write_json
from experiment.phase18.core.raserec_data import training_frequencies
from experiment.phase4.gacr_s0 import BoundedResidualRanker
from experiment.phase9.eval_cf0_b3_beamfusion import score_item_head, standardize
from experiment.phase9.eval_cf0_b4_reliability import rank_matrix

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def residual_orders(row, ranker):
    """Target-free deterministic transform of saved features, retaining tie order."""
    ids = np.asarray(row['union_ids'], dtype=np.int64)
    device = next(ranker.parameters()).device
    base = torch.tensor(row['base'], dtype=torch.float32, device=device)
    features = torch.tensor(row['features'], dtype=torch.float32, device=device)
    assert len(ids) == len(set(ids)) == len(base) and features.shape == (len(ids), 6)
    assert len(ids) >= 50 and torch.isfinite(features).all() and torch.isfinite(base).all()
    scores = (base + ranker(features)).cpu().numpy()
    assert np.isfinite(scores).all()
    full = np.argsort(-scores, kind='stable')[:50]
    fixed = np.argsort(-scores[:50], kind='stable')
    return (ids[full], scores[full]), (ids[fixed], scores[fixed])


def reciprocal_scores(size=50):
    # Match the FP32 base used by the original residual ranker, then use the
    # original PCRF float64 input conversion.
    return np.asarray([1 / (i + 1) for i in range(size)], dtype=np.float32).astype(np.float64)


def pcrf_ranks(records, frequencies, config):
    pop = np.asarray([frequencies[r['candidate_ids']] for r in records])
    positions = np.asarray([r['target_position'] for r in records])
    result, _ = rank_matrix(
        np.stack([standardize(r['seq']) for r in records]),
        np.stack([standardize(r['cf']) for r in records]),
        np.stack([standardize(np.log1p(p)) for p in pop]),
        (pop[:, :10] <= config['q1']).mean(1), positions,
        (config['lambda'], config['beta'], config['gamma']))
    assert np.array_equal(result <= 50, positions >= 0)
    return result


def metric_table(methods):
    keys = [f'{m}@{k}' for m in ('ndcg', 'hit') for k in (5, 10, 20, 50)]
    lines = ['| 方法 | N5 | N10 | N20 | N50 | H5 | H10 | H20 | H50 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    lines += ['| ' + name + ' | ' + ' | '.join(f'{values[k]:.6f}' for k in keys) + ' |'
              for name, values in methods.items()]
    return '\n'.join(lines)


def report(config, state, stage, summary=None, error=None):
    path = ROOT / config['report']
    text = '# Stage20 Toys GACR分数与候选固定对照报告\n\n'
    text += '## Material Passport\n\n- Origin Skill: academic-research-suite / experiment-agent\n'
    text += '- Origin Mode: run / validate；Verification Status: ANALYZED（完成后仍仅为开发性消融）\n'
    text += f'- 更新：{time.strftime("%Y-%m-%d %H:%M:%S %z")}；状态：**{state}**；步骤：`{stage}`。\n\n'
    text += '固定Toys全量19,412用户、原FP32缓存、residual与item-head seed2023；CPU4线程，30分钟硬上限，零新训练与beam推理，不读test。三个新增臂为原GRAM倒数rank+PCRF、C1倒数rank+PCRF、固定C1候选残差+PCRF。原六臂逐用户一致性检查通过前不发布新方法指标。\n\n'
    if config.get('residual_device') == 'cuda:0':
        text += '工程修订v2：CPU重建与原GPU分数相差约3e-8即可改变Top50边界，首次attempt在新指标产生前停止并保留。现仅将129参数冻结残差前向放回FP32/TF32关闭的GPU，CF打分、PCRF及统计仍在CPU；不放松候选/逐用户rank精确一致要求。\n\n'
    text += '主比较为完整GACR+PCRF减c1_reciprocal_pcrf的N10；lambda1/beta0.5/gamma1/q1=5固定，稳定排序；每臂按实际候选顺序重算tail_mass。倒数rank采用原GACR的FP32数值后进入PCRF的float64接口。\n\n'
    if error:
        text += f'错误：`{error}`。保留attempt，无自动重试。\n\n'
    if summary:
        text += '## 全量结果\n\n' + metric_table(summary['groups']['all_validation']) + '\n\n'
        text += '## 固定比较\n\n| 比较（前者减后者） | N10增量 | 描述性95%区间 | H10增量 | 新增/丢失Top10 |\n|---|---:|---|---:|---:|\n'
        for name, groups in summary['comparisons'].items():
            x=groups['all_validation']; n=x['ndcg@10']; h=x['hit@10']
            text += f"| {name} | {n['delta']:+.6f} | [{n['ci95'][0]:+.6f}, {n['ci95'][1]:+.6f}] | {h['delta']:+.6f} | {x['hit10_gained']}/{x['hit10_lost']} |\n"
        for group in ['target_frequency_le5','target_frequency_gt5','outside_c1_training_users','inside_c1_training_users']:
            text += f'\n## 分组：{group}（{next(iter(summary["groups"][group].values()))["users"]:,}人）\n\n'+metric_table(summary['groups'][group])+'\n'
        text += '\n原六臂的逐用户rank精确一致；全量重建GACR top50顺序与原记录一致。新增控制的实际候选、base分数与逐用户rank已保存。\n'
    text += '\n3000次配对用户bootstrap、seed20260912；validation反复用于开发，多比较未校正；区间不包含训练seed或选择历史，不是独立确认。统计解释沿用[终态复盘11项检查](第20阶段_终态复盘与下一步决策_20260912.md)：按用户配对、全量/分组并列、保留全部臂及工程失败，不将关联或分组记账写成因果证明。\n\n'
    text += f'证据目录：`{config["output"]}`；[冻结配置](../../{config["config_path"]})；[执行计划](../../plan/第二十阶段/第20阶段收尾后的验证与方法决策计划.md)。\n'
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(config['cpu_threads'])
    use_gpu = config.get('residual_device') == 'cuda:0'
    if use_gpu:
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(0.1)
    elif torch.cuda.is_available():
        raise RuntimeError('CPU-only attempt requires CUDA_VISIBLE_DEVICES empty')

    def status(state, stage, **extra):
        write_json(output/'status.json', {'state':state,'stage':stage,'pid':os.getpid(),
                   'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   'elapsed_seconds':round(time.time()-started,2),'test_read':False,'gpu_used':use_gpu,**extra})
        print(json.dumps({'state':state,'stage':stage,**extra}),flush=True)
        report(config,state,stage,error=extra.get('error'))

    def stop(signum, frame):
        raise TimeoutError('Hard budget expired' if signum == signal.SIGALRM else f'Signal {signum}')

    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    signal.alarm(config['max_wall_seconds'])
    try:
        status('RUNNING','input_hashes')
        inputs = {k:ROOT/p for k,p in config['inputs'].items()}
        inputs['config'] = ROOT/args.config
        inputs['code'] = Path(__file__)
        hashes = {k:digest(p) for k,p in inputs.items()}
        for key,expected in config['expected_sha256'].items():
            assert hashes[key] == expected, key
        write_json(output/'protocol.json',{'config':config,'input_sha256':hashes,'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        validation = read_rows(inputs['validation'])
        lookup = {r['user_id']:r for r in validation}
        users = list(lookup)
        assert len(users) == len(validation) == config['users']
        frequencies = training_frequencies(read_rows(inputs['train']),config['catalog_size_with_padding'])
        with inputs['reference_ranks'].open() as f:
            reference = {r['user_id']:r for r in csv.DictReader(f,delimiter='\t')}
        assert set(users) == set(reference)
        ranker=BoundedResidualRanker(6,16,0.2).eval().requires_grad_(False)
        ranker.load_state_dict(torch.load(inputs['residual'],map_location='cpu'),strict=True)
        ranker.to(config.get('residual_device','cpu'))
        caches={key:{r['user_id']:r for r in read_rows(inputs[key])} for key in ('parent','c1','gacr')}
        assert all(set(cache)==set(users) for cache in caches.values())
        for cache in caches.values():
            for u,r in cache.items():
                assert r['split']=='validation' and r['gold_item_id']==lookup[u]['target']
                assert len(r['ranked_item_ids'])==len(set(r['ranked_item_ids']))==len(r['scores'])==50
                assert all(0<i<len(frequencies) for i in r['ranked_item_ids']) and np.isfinite(r['scores']).all()
        status('RUNNING','frozen_residual_parity')
        fixed={}; seen=set(); max_error=0.
        with inputs['features'].open() as handle:
            for line in handle:
                row=json.loads(line); u=row['user_id']; assert u not in seen and u in lookup; seen.add(u)
                assert row['union_ids'][:50]==caches['c1'][u]['ranked_item_ids']
                np.testing.assert_array_equal(np.asarray(row['base'][:50]),reciprocal_scores())
                assert np.count_nonzero(row['base'][50:])==0
                (ids,scores),(fixed_ids,fixed_scores)=residual_orders(row,ranker)
                assert ids.tolist()==caches['gacr'][u]['ranked_item_ids'], f'GACR order mismatch: {u}'
                error=float(np.max(np.abs(scores-np.asarray(caches['gacr'][u]['scores']))))
                assert error<=1e-6, (u,error)
                max_error=max(max_error,error)
                fixed[u]={'ranked_item_ids':fixed_ids.tolist(),'scores':fixed_scores.astype(float).tolist()}
        assert seen==set(users)
        records={}
        for key in ['parent','c1','gacr']:
            records[key]=[{'user':u,'history':lookup[u]['history'][-20:],
                           'candidate_ids':caches[key][u]['ranked_item_ids'],
                           'seq':np.asarray(caches[key][u]['scores'],dtype=np.float64),
                           'target_position':caches[key][u]['ranked_item_ids'].index(lookup[u]['target'])
                            if lookup[u]['target'] in caches[key][u]['ranked_item_ids'] else -1} for u in users]
            status('RUNNING','reference_cf_scoring',source=key)
            score_item_head(records[key],inputs['item_head'],256)
        values={}
        for key,raw,combo in [('parent','original_gram','original_pcrf'),('c1','c1','c1_pcrf'),('gacr','gacr_v6','gacr_v6_pcrf')]:
            values[raw]=np.asarray([r['target_position']+1 if r['target_position']>=0 else 51 for r in records[key]])
            values[combo]=pcrf_ranks(records[key],frequencies,config)
            for name in (raw,combo):
                assert np.array_equal(values[name],np.asarray([int(reference[u][name]) for u in users])),name
        write_json(output/'parity.json',{'state':'PASSED','users':len(users),'six_reference_arms_rank_exact':True,
            'reconstructed_gacr_order_exact':True,'max_reconstructed_score_error':max_error})
        status('RUNNING','new_fixed_controls',exact_reference_users=len(users))
        for key,name in [('parent','original_gram_reciprocal_pcrf'),('c1','c1_reciprocal_pcrf')]:
            changed=[{**r,'seq':reciprocal_scores()} for r in records[key]]
            values[name]=pcrf_ranks(changed,frequencies,config)
        fixed_records=[]
        with (output/'fixed_candidates_residual.predictions.jsonl').open('x') as f:
            for u,original in zip(users,records['c1']):
                new=fixed[u]; ids=new['ranked_item_ids']; target=lookup[u]['target']
                order=[original['candidate_ids'].index(i) for i in ids]
                assert set(ids)==set(original['candidate_ids'])
                fixed_records.append({**original,'candidate_ids':ids,'seq':np.asarray(new['scores'],dtype=np.float64),
                    'cf':original['cf'][order],'target_position':ids.index(target) if target in ids else -1})
                f.write(json.dumps({'user_id':u,'split':'validation','gold_item_id':target,**new})+'\n')
        values['c1_fixed_candidates_residual_pcrf']=pcrf_ranks(fixed_records,frequencies,config)
        for name,source in [('original_gram_reciprocal_pcrf','original_gram'),('c1_reciprocal_pcrf','c1'),('c1_fixed_candidates_residual_pcrf','c1')]:
            assert np.array_equal(values[name]<=50,values[source]<=50)
        fit_users=set(inputs['c1_training_users'].read_text().splitlines())
        freq=np.asarray([frequencies[lookup[u]['target']] for u in users])
        masks={'all_validation':np.ones(len(users),dtype=bool),'target_frequency_le5':freq<=5,'target_frequency_gt5':freq>5,
               'outside_c1_training_users':np.asarray([u not in fit_users for u in users]),
               'inside_c1_training_users':np.asarray([u in fit_users for u in users])}
        groups={g:{name:metric(r[mask]) for name,r in values.items()} for g,mask in masks.items()}
        with (output/'per_user.tsv').open('w') as f:
            writer=csv.writer(f,delimiter='\t'); writer.writerow(['user_id',*values])
            for i,u in enumerate(users): writer.writerow([u,*[int(r[i]) for r in values.values()]])
        comparisons={}
        for a,b in config['comparisons']:
            status('RUNNING','descriptive_bootstrap',comparison=a+'_minus_'+b)
            comparisons[a+'_minus_'+b]={g:paired(values[a][mask],values[b][mask],config) for g,mask in masks.items()}
        assert hashes=={k:digest(p) for k,p in inputs.items()}
        result={'state':'COMPLETED','verification_status':'ANALYZED','dataset':'Toys','users':len(users),
            'groups':groups,'comparisons':comparisons,'input_sha256':hashes,'new_training_steps':0,'new_beam_inference':False,
            'gpu_used':use_gpu,'gpu_scope':'frozen 129-parameter residual only' if use_gpu else 'none',
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if use_gpu else 0,
            'test_read':False,'elapsed_seconds':round(time.time()-started,2),
            'parity':json.loads((output/'parity.json').read_text()),'config':config}
        write_json(output/'summary.json',result)
        status('COMPLETED','finished')
        report(config,'COMPLETED','finished',summary=result)
        signal.alarm(0)
    except BaseException as error:
        status('STOPPED' if isinstance(error,TimeoutError) else 'FAILED','aborted',error=repr(error))
        (output/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':
    main()
