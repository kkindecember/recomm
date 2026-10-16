"""Frozen expert reselection audit; CPU only, no final history mask or fitting."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from experiment.phase21 import expert_reselection_core as C
from experiment.phase21 import tisasrec_core as B
from experiment.phase20.catalog_scope_core import metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/phase21/expert_reselection_20261016_v2'
SOURCE = ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
FULL = ROOT/'artifacts/phase21/full_validation_20260924_v1'
QWEN = ROOT/'artifacts/phase21/qwen_text_expert_20261015_v1'
TIME = ROOT/'artifacts/phase21/time_behavior_expert_20261015_v1'
PLAN = ROOT/'plan/第二十一阶段/21.42B_专家重选缓存顺序适配修复_20261016.md'
REPORT = ROOT/'report/第二十一阶段/专家重选与去自匹配内容评分筛查_20261016.md'
MODULE = 'experiment.phase21.expert_reselection_v2'
DOMAINS = ('Toys', 'Beauty')
PARTS = ('fit', 'old512', 'wide4096')
EXPERTS = ('SASRec', 'TiSASRec', 'ItemCF', 'BGE', 'Qwen4',
           'BGE_loo', 'BGE_uniform', 'Qwen4_loo', 'Qwen4_uniform')
TEXTS = EXPERTS[3:]
RAW = dict(Toys=Path('/tmp/recomm_toys_5_timeaudit_20261015.json.gz'),
           Beauty=Path('/tmp/recomm_beauty_5_timeaudit_20261015.json.gz'))
CFG = dict(seed=20261016, wide_users=4096, score_scale=20., topk_diagnostic=10,
    features=C.NAMES, feature_dimension=6, future_fuser_parameters=323,
    experts=list(EXPERTS), max_parallel=2, threads_per_worker=2, worker_timeout_seconds=7200,
    final_history_filter=False, history_scope='full visible history; recent20 separately diagnosed',
    experts_frozen=True, new_expert_inference=True, new_expert_training=False,
    gram_generation=False, fusion_training=False, test_read=False, scientific_auto_retry=False,
    stop_after='expert diagnostics and six-input caches only; no auto-selected final method')


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def progress(domain, stage, **kw):
    value = dict(state='RUNNING', stage=stage, updated_at=stamp(), pid=os.getpid(),
                 test_read=False, gpu_used=False, **kw)
    write(OUT/domain/'status.json', value)
    print(json.dumps(dict(domain=domain, **value)), flush=True)


def feature_source(base, domain, part):
    variant = 'qwen4' if base == QWEN else 'tisas'
    return base/domain/'features'/variant/('train' if part == 'fit' else 'validation')


def prepare():
    checks = C.tests()
    for base in (FULL, QWEN, TIME):
        assert read(base/'status.json')['state'] == 'COMPLETED'
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp(), test_read=False))
    write(OUT/'config.json', CFG); write(OUT/'tests.json', checks)
    (OUT/'protocol.md').write_text(PLAN.read_text())
    sources = {}; domain_inputs = {}
    def use(path):
        rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        if rel not in sources:
            sources[rel] = sha(path)
        return rel
    common = [use(path) for path in (Path(__file__), ROOT/'experiment/phase21/expert_reselection_core.py',
        ROOT/'experiment/phase21/tisasrec_core.py', ROOT/'experiment/phase20/lime_comparison_core.py',
        ROOT/'experiment/phase20/catalog_scope_core.py', ROOT/'experiment/phase21/run_expert_reselection_v2.sh', PLAN, OUT/'config.json')]
    for domain in DOMAINS:
        p = OUT/domain; p.mkdir()
        full, old, fit_bank = [read(base/domain/'samples.json') for base in (FULL, QWEN, SOURCE)]
        fit_ix = [j for j,s in enumerate(fit_bank) if s['role'] == 'ranker_fit']
        fit = [fit_bank[j] for j in fit_ix]
        full_ix = {s['user_id']:j for j,s in enumerate(full)}
        for sample in old:
            reference = full[full_ix[sample['user_id']]]
            assert all(sample[k] == reference[k] for k in ('user_id','target','history','full_history'))
        eligible = [j for j,s in enumerate(full) if not s['previously_evaluated'] and s['training_holdout_role'] == 'none']
        eligible.sort(key=lambda j:hashlib.sha256(f'expert-reselection-20261016|{domain}|{full[j]["user_id"]}'.encode()).digest())
        wide_ix = eligible[:CFG['wide_users']]; wide = [full[j] for j in wide_ix]
        assert len(old) == 512 and len(wide) == 4096
        sets = [{s['user_id'] for s in rows} for rows in (fit, old, wide)]
        assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i))
        catalog = read(QWEN/domain/'catalog.json')
        assert catalog['items'] == read(FULL/domain/'catalog.json')['items'] == read(TIME/domain/'catalog.json')['items']
        dc = read(QWEN/domain/'data_config.json')
        paths = [SOURCE/domain/'samples.json', SOURCE/domain/'experts.npy',
            FULL/domain/'samples.json', FULL/domain/'experts.npy', FULL/domain/'score_cache_manifest.json',
            QWEN/domain/'samples.json', QWEN/domain/'catalog.json', QWEN/domain/'data_config.json',
            ROOT/dc['embedding'], QWEN/domain/'qwen4_embeddings.npz', QWEN/domain/'qwen4_encoding.json',
            TIME/domain/'behavior/tisas/model.pt', TIME/domain/'validation_histories.npz', TIME/domain/'fit_histories.npz', RAW[domain]]
        for part in ('fit', 'old512'):
            for base in (QWEN, TIME):
                paths += [feature_source(base, domain, part)/name for name in ('item.npy', 'metadata.json')]
        domain_inputs[domain] = common+[use(path) for path in paths]
        write(p/'catalog_items.json', catalog['items'])
        for part, rows, indices in (('fit', fit, fit_ix), ('old512', old, [full_ix[s['user_id']] for s in old]),
                                    ('wide4096', wide, wide_ix)):
            assert all(s['target'] not in s['full_history'] and s['history'] == s['full_history'][-20:] for s in rows)
            write(p/(part+'_samples.json'), rows)
            write(p/(part+'_metadata.json'), dict(users=len(rows), source_indices=indices,
                base='fit' if part == 'fit' else 'full', users_order=[s['user_id'] for s in rows],
                full_validation_seen_before=True, target_used_for_selection=False))
            (p/'features'/part).mkdir(parents=True)
            domain_inputs[domain] += [use(p/(part+'_samples.json')),use(p/(part+'_metadata.json'))]
        domain_inputs[domain].append(use(p/'catalog_items.json'))
        write(p/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    write(OUT/'manifest.json', dict(sha256=sources, domain_inputs=domain_inputs))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# 专家重选与去自匹配内容评分筛查\n\n状态：PREPARED，尚无新结果。\n')
    print(json.dumps(dict(state='PREPARED', input_files=len(sources), tests=checks)), flush=True)


def time_scores(domain, samples):
    p = OUT/domain; catalog = read(p/'catalog_items.json')
    ck = torch.load(TIME/domain/'behavior/tisas/model.pt', map_location='cpu')
    assert ck['train_complete'] and ck['training_only'] and ck['epoch'] == 200
    config = ck['config']
    model = B.TiSASRec(ck['num_items'], **{k:config[k] for k in
        ('hidden', 'maxlen', 'num_layers', 'num_heads', 'dropout', 'time_span')}).eval()
    model.load_state_dict(ck['state_dict']); model.requires_grad_(False)
    def predict(items, days):
        relation = B.intervals(items, days, cap=config['time_span'])
        with torch.no_grad():
            raw = model.predict(torch.from_numpy(items), torch.from_numpy(relation)).numpy()[:,1:]
        return (raw-raw.min(1, keepdims=True))/(np.ptp(raw, axis=1, keepdims=True)+1e-8)
    # Replay CPU inference against the previously exported GPU score cache.
    original = np.load(TIME/domain/'validation_histories.npz')
    replay = predict(original['items'][:16], original['days'][:16])
    expected = np.load(feature_source(TIME, domain, 'old512')/'item.npy', mmap_mode='r')[:16,:,0]
    np.testing.assert_allclose(replay, expected, atol=8e-6, rtol=3e-5)
    wanted = {(s['user_id'], catalog[item]) for s in samples for item in s['full_history'][-50:]}
    dates = {}
    progress(domain, 'visible_time_alignment', wanted_pairs=len(wanted))
    with gzip.open(RAW[domain], 'rt') as stream:
        for line in stream:
            row = json.loads(line); key = (row['reviewerID'], row['asin'])
            if key in wanted:
                assert key not in dates
                value = row['unixReviewTime']; assert value > 0 and value % 86400 == 0
                dates[key] = value//86400
    assert set(dates) == wanted
    dest = p/'features/wide4096/TiSASRec.npy'
    array = np.lib.format.open_memmap(dest, mode='w+', dtype='float32', shape=(len(samples), ck['num_items']))
    for start in range(0, len(samples), 16):
        batch = samples[start:start+16]
        items, days = np.zeros((len(batch), 50), np.int64), np.zeros((len(batch), 50), np.int64)
        for j,sample in enumerate(batch):
            history = sample['full_history'][-50:]
            values = [dates[(sample['user_id'], catalog[i])] for i in history]
            assert all(a <= b for a,b in zip(values, values[1:]))
            items[j,-len(history):], days[j,-len(history):] = history, values
        array[start:start+len(batch)] = predict(items, days)
        if start % 128 == 0 or start+len(batch) == len(samples):
            progress(domain, 'tisas_cpu_scores', completed=start+len(batch), total=len(samples))
    array.flush()
    write(p/'time_checks.json', dict(visible_pairs=len(wanted), missing=0, ambiguous=0,
        target_dates_retained=False, old512_cpu_gpu_score_parity=True, no_expert_training=True))


def topk(scores, k=10):
    threshold = np.partition(scores, len(scores)-k)[len(scores)-k]
    above = np.flatnonzero(scores > threshold)
    ties = np.flatnonzero(scores == threshold)[:k-len(above)]
    ids = np.r_[above, ties]
    return ids[np.lexsort((ids, -scores[ids]))]


def worker(domain):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    p = OUT/domain; manifest = read(OUT/'manifest.json')
    for j,rel in enumerate(manifest['domain_inputs'][domain]):
        progress(domain, 'verify_inputs', completed=j, total=len(manifest['domain_inputs'][domain]), file=rel)
        assert sha(Path(rel) if rel.startswith('/') else ROOT/rel) == manifest['sha256'][rel], rel
    time_scores(domain, read(p/'wide4096_samples.json'))
    source3 = {name:np.load(base/domain/'experts.npy', mmap_mode='r') for name,base in (('fit',SOURCE), ('full',FULL))}
    dc = read(QWEN/domain/'data_config.json')
    embeddings = {'BGE':np.load(ROOT/dc['embedding'])['embeddings'],
                  'Qwen4':np.load(QWEN/domain/'qwen4_embeddings.npz')['embeddings']}
    expected_ids = read(p/'catalog_items.json')[1:]
    for path in (ROOT/dc['embedding'], QWEN/domain/'qwen4_embeddings.npz'):
        assert np.load(path)['item_ids'].tolist() == expected_ids
    v = len(expected_ids)
    all_results = {}
    for part in PARTS:
        samples = read(p/(part+'_samples.json')); meta = read(p/(part+'_metadata.json'))
        indices = meta['source_indices']; base = source3[meta['base']]
        folder = p/'features'/part; n = len(samples)
        cached = {}
        if part != 'wide4096':
            for name, source in (('TiSASRec',TIME), ('Qwen4',QWEN)):
                src = feature_source(source, domain, part)
                metadata = read(src/'metadata.json')
                assert metadata['users'] == len(samples)
                if source == TIME:
                    order_path = TIME/domain/('fit_histories.npz' if part == 'fit' else 'validation_histories.npz')
                    with np.load(order_path) as packed:
                        assert packed['users'].tolist() == [s['user_id'] for s in samples]
                        for sample, items in zip(samples, packed['items']):
                            assert items[items != 0].tolist() == sample['full_history'][-50:]
                else:
                    assert metadata['users_order'] == [s['user_id'] for s in samples]
                cached[name] = np.load(src/'item.npy', mmap_mode='r')
        else:
            cached['TiSASRec'] = np.load(folder/'TiSASRec.npy', mmap_mode='r')
        for text in ('BGE', 'Qwen4'):
            arrays = {kind:np.lib.format.open_memmap(folder/(text+'_'+kind+'.npy'), mode='w+',
                dtype='float32', shape=(n,v)) for kind in ('loo','uniform')}
            if text == 'Qwen4' and part == 'wide4096':
                raw_array = np.lib.format.open_memmap(folder/'Qwen4.npy', mode='w+', dtype='float32', shape=(n,v))
            for start in range(0,n,32):
                stop = min(start+32,n); histories = [s['full_history'] for s in samples[start:stop]]
                reference = None
                if text == 'BGE':
                    reference = np.asarray(base[np.asarray(indices[start:stop]),2,:])
                elif part != 'wide4096':
                    reference = np.asarray(cached['Qwen4'][start:stop,:,1])
                raw, loo = C.self_excluded_scores(histories, embeddings[text], reference)
                arrays['loo'][start:stop] = loo
                arrays['uniform'][start:stop] = np.stack([C.uniform_mass_control(a,b,h) for a,b,h in zip(raw,loo,histories)])
                if text == 'Qwen4' and part == 'wide4096':
                    raw_array[start:stop] = raw
                if start % 128 == 0 or stop == n:
                    progress(domain, 'text_self_exclusion', part=part, text=text, completed=stop, total=n)
            for value in arrays.values(): value.flush()
            if text == 'Qwen4' and part == 'wide4096': raw_array.flush()
        new_arrays = {name:np.load(folder/(name+'.npy'), mmap_mode='r') for name in EXPERTS if '_' in name}
        if part == 'wide4096': cached['Qwen4'] = np.load(folder/'Qwen4.npy', mmap_mode='r')
        pairs = [(behavior,text) for behavior in ('SASRec','TiSASRec') for text in TEXTS]
        feature_arrays = {a+'__'+b:np.empty((n,6),np.float32) for a,b in pairs}
        ranks = {name:np.empty(n,np.int64) for name in EXPERTS}
        measures = {name:np.empty((n,4),np.float64) for name in EXPERTS}
        for j,sample in enumerate(samples):
            row = {name:np.asarray(base[indices[j],channel,:]) for name,channel in (('SASRec',0),('ItemCF',1),('BGE',2))}
            for name,channel in (('TiSASRec',0),('Qwen4',1)):
                row[name] = np.asarray(cached[name][j] if part == 'wide4096' else cached[name][j,:,channel])
            row.update({name:np.asarray(value[j]) for name,value in new_arrays.items()})
            history = sample['full_history']; ids = np.unique(np.asarray(history,np.int64))-1
            recent = np.unique(np.asarray(sample['history'],np.int64))-1
            target = sample['target']-1
            for name,scores in row.items():
                assert scores.shape == (v,) and np.isfinite(scores).all()
                prob = C.probabilities(scores); order = topk(scores)
                ranks[name][j] = 1+int((scores > scores[target]).sum())+int(((scores == scores[target]) & (np.arange(v) < target)).sum())
                measures[name][j] = [int(np.argmax(scores) in ids), np.isin(order,ids).sum(), prob[ids].sum(), prob[recent].sum()]
            for a,b in pairs:
                feature_arrays[a+'__'+b][j] = C.features(np.stack([row[a],row[b]]),history)
            for text in ('BGE','Qwen4'):
                mask = np.isin(np.arange(v),ids)
                np.testing.assert_array_equal(row[text][~mask],row[text+'_loo'][~mask])
                np.testing.assert_array_equal(row[text][~mask],row[text+'_uniform'][~mask])
                assert (row[text+'_loo'][mask] <= row[text][mask]).all()
                np.testing.assert_allclose(measures[text+'_loo'][j,2],measures[text+'_uniform'][j,2],atol=2e-7,rtol=3e-6)
            if j % 128 == 0 or j+1 == n:
                progress(domain, 'expert_diagnostics_and_features', part=part, completed=j+1,total=n)
        np.savez_compressed(p/(part+'_ranks.npz'),users=np.asarray([s['user_id'] for s in samples]),**ranks)
        np.savez_compressed(p/(part+'_measurements.npz'),**measures)
        for name,value in feature_arrays.items():
            np.save(folder/(name+'_features6.npy'),value)
        result = {name:dict(metrics=metrics(ranks[name]), history_top1_fraction=float(m[:,0].mean()),
            history_top10_count_mean=float(m[:,1].mean()), history_probability_mean=float(m[:,2].mean()),
            recent20_probability_mean=float(m[:,3].mean()), history_probability_quantiles=np.quantile(m[:,2],[0,.1,.5,.9,1]).tolist())
            for name,m in measures.items()}
        all_results[part] = dict(users=n,experts=result,feature_pairs=list(feature_arrays),
            feature_names=C.NAMES,feature_targets_used=False,gram_state_used=False,
            checks=dict(nonhistorical_scores_exact=True,history_mass_control_all_users=True,
                historical_self_excluded_scores_nonincreasing=True))
        write(p/(part+'_summary.json'),all_results[part])
    paths = list((p/'features').glob('*/*.npy'))+list(p.glob('*ranks.npz'))+list(p.glob('*measurements.npz'))
    write(p/'output_manifest.json',dict(sha256={str(path.relative_to(ROOT)):sha(path) for path in paths}))
    write(p/'summary.json',dict(parts=all_results,experts_frozen=True,fusion_trained=False,
        gram_generation=False,test_read=False))
    write(p/'status.json',dict(state='COMPLETED',updated_at=stamp(),test_read=False,gpu_used=False))


def summarize():
    result = {d:read(OUT/d/'summary.json') for d in DOMAINS}
    write(OUT/'summary.json',dict(domains=result,tests=read(OUT/'tests.json'),test_read=False,
        fusion_trained=False,gram_generation=False,automatic_final_expert_selection=False))
    lines = ['# 专家重选与去自匹配内容评分筛查','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: validate',
        '- Origin Date: 2026-10-16','- Verification Status: ANALYZED；冻结专家评分和特征缓存，无新融合训练或生成',
        '- Version Label: expert_reselection_v2','',
        '最终指标均不过滤历史。wide4096排除了原512及早期融合训练/校准用户，但仍属于此前已经看过总体结果的validation，不是独立test。',
        '所有指标是专家直接对全目录排序，不是GRAM引导后的最终模型成绩；不能直接据此定专家组合。', '']
    for part in PARTS:
        lines += ['## '+part,'','| 数据集 | 专家/评分 | N10 | H10 | 历史Top1比例 | 平均前10历史数 | 完整历史概率质量 |',
            '|---|---|---:|---:|---:|---:|---:|']
        for d in DOMAINS:
            for name,value in result[d]['parts'][part]['experts'].items():
                m=value['metrics'];lines.append(f'| {d} | {name} | {m["N10"]:.6f} | {m["H10"]:.6f} | '
                    f'{value["history_top1_fraction"]:.4f} | {value["history_top10_count_mean"]:.3f} | {value["history_probability_mean"]:.4f} |')
    lines += ['', 'loo去掉被评分历史商品自身的文本证据；非历史候选分数逐元素不变。uniform均匀降历史分，匹配loo的历史概率总量。商品没有被从目录或树中排除。',
        '在当前没有重复目标的数据中，仅降低历史分就可能机械性提高纯专家排序，不能以此证明新颖机制。需进一步与uniform、GRAM同种校准和实际生成匹配对照。',
        '六项输入：log历史长度、两项历史概率质量、两项非历史分布熵、一项非历史专家JS。它们分别描述证据规模、回看倾向、非历史候选集中程度、意见分歧；不是校准正确率。',
        '6→32→beta/alpha共323参数，仅实现与核验，没有声称比原10/7维更好。后续需按同预算拟合及生成验证。',
        '本轮没有输出自动赢家，也没有恢复/删除ItemCF的组合消融结论。全部来源、既有LIME衍生评分以及历史降分思路的归属保留。']
    REPORT.write_text('\n'.join(lines)+'\n')


def supervise():
    assert read(OUT/'status.json')['state'] == 'PREPARED'
    start=time.monotonic(); jobs={}
    try:
        for domain in DOMAINS:
            log=(OUT/domain/'worker.log').open('x')
            process=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--domain',domain],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            jobs[domain]=dict(process=process,log=log)
        while any(j['process'].poll() is None for j in jobs.values()):
            if time.monotonic()-start > CFG['worker_timeout_seconds']:
                raise TimeoutError('Two-hour CPU audit limit')
            write(OUT/'status.json',dict(state='RUNNING',updated_at=stamp(),elapsed_seconds=time.monotonic()-start,
                workers={d:dict(pid=j['process'].pid,exit_code=j['process'].poll(),status=read(OUT/d/'status.json')) for d,j in jobs.items()},
                gpu_used=False,test_read=False))
            time.sleep(10)
        complete=all(j['process'].returncode == 0 for j in jobs.values())
        if complete:summarize()
        write(OUT/'status.json',dict(state='COMPLETED' if complete else 'INCOMPLETE',updated_at=stamp(),
            elapsed_seconds=time.monotonic()-start,test_read=False,gpu_used=False,
            workers={d:dict(exit_code=j['process'].returncode,status=read(OUT/d/'status.json')) for d,j in jobs.items()},
            report=str(REPORT.relative_to(ROOT))))
    finally:
        for job in jobs.values():
            if job['process'].poll() is None:
                os.killpg(job['process'].pid,signal.SIGTERM)
            job['log'].close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('test','prepare','worker','supervise'))
    parser.add_argument('--domain',choices=DOMAINS);args=parser.parse_args()
    try:
        if args.mode == 'test':print(json.dumps(C.tests()))
        elif args.mode == 'prepare':prepare()
        elif args.mode == 'worker':worker(args.domain)
        else:supervise()
    except BaseException as exc:
        if OUT.exists():
            dest=OUT/args.domain if args.domain else OUT
            write(dest/'status.json',dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),
                updated_at=stamp(),test_read=False))
        raise


if __name__ == '__main__':main()
