"""Bounded, background-capable two-domain direction evidence screen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from scipy import sparse
from experiment.phase20.catalog_scope_core import ranks, topk, metrics, contrast
from experiment.phase21.relation_direction_core import (
    ARMS, transition_counts, normalized_neighbors, history_matrix,
    fold_ids, fit_head, predict_head,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase21/relation_direction_20260922_v1'
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
DESIGN = ROOT / 'artifacts/phase20/lime_clean_training_design_20260920_v1'
PLAN = ROOT / 'plan/第二十一阶段/21.1_候选卡与关系方向首轮探索_20260922.md'
REPORT = ROOT / 'report/第二十一阶段/关系方向首轮探索_20260922.md'
MODULE = 'experiment.phase21.relation_direction'


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def status(stage, **kw):
    value = dict(state='RUNNING', stage=stage, pid=os.getpid(), updated_at=stamp(),
                 test_read=False, gpu_used=False, **kw)
    write(OUT / 'worker_status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run_domain(domain):
    start = time.monotonic()
    folder = OUT / domain
    folder.mkdir()
    src = SOURCE / domain
    manifest = {}

    def use(path, expected=None):
        rel = str(path.relative_to(ROOT))
        assert 'test.json' not in rel and '/test/' not in rel
        digest = sha(path)
        if expected is not None:
            assert digest == expected, ('changed input', rel)
        manifest[rel] = digest
        return path

    status('input_validation', domain=domain)
    receipt = read(use(DESIGN / 'summary.json'))
    seqpath = DESIGN / domain / 'upstream_sequences_without_terminal.jsonl'
    sequences = {r['user_id']: r['history'] for r in rows(use(seqpath, receipt['output_sha256'][str(seqpath.relative_to(ROOT))]))}
    # Check event boundaries through the frozen design receipt, never rebuilding from full sequences.
    samples = read(use(src / 'samples.json'))
    catalog = read(use(src / 'catalog.json'))
    size = len(catalog['items'])
    cfg = read(use(src / 'data_config.json'))
    cache = read(use(src / 'cache_manifest.json'))['sha256']
    with np.load(use(src / 'data.npz', cache['data.npz'])) as z:
        data = {k: z[k] for k in z.files}
    experts = np.load(use(src / 'experts.npy', cache['experts.npy']), mmap_mode='r')
    lime = np.load(use(src / 'lime.npy', cache['lime.npy']), mmap_mode='r')
    candidate_rows = rows(use(src / 'candidates.jsonl', cache['candidates.jsonl']))
    counts = np.bincount(np.concatenate(list(sequences.values())), minlength=size)
    np.testing.assert_array_equal(counts, np.load(use(src / 'frequencies.npy')))
    selected = [samples[j] for j in data['sample_indices']]
    assert len({s['user_id'] for s in samples}) == len(samples)
    for i, sample in enumerate(selected):
        assert sample['user_id'] == data['users'][i] == candidate_rows[i]['user_id']
        assert sample['role'] == data['roles'][i]
        assert sample['full_history'] == sequences[sample['user_id']]
        assert sample['history'] == sample['full_history'][-20:]
        assert sample['target'] == data['targets'][i]
        np.testing.assert_array_equal(data['ids'][i], candidate_rows[i]['ids'])
    train = np.flatnonzero(data['roles'] == 'ranker_fit')
    dev = np.flatnonzero(data['roles'] == 'development')
    assert len(dev) == 512 and len(train) == {'Toys': 2773, 'Beauty': 3310}[domain]
    histories = [s['history'] for s in selected]
    itemcf = read(use(ROOT / cfg['itemcf']))
    assert itemcf['training_only'] and itemcf['num_training_interactions'] == int(counts.sum())
    mapping = {s: i for i, s in enumerate(catalog['items'])}
    rr, cc, vv = [], [], []
    for a, near in itemcf['neighbors'].items():
        for row in near:
            rr.append(mapping[a]); cc.append(mapping[row['item_id']]); vv.append(row['score'])
    cfgraph = sparse.csr_matrix((vv, (rr, cc)), shape=(size, size))
    del itemcf, rr, cc, vv
    status('build_direction_graphs', domain=domain, training_users=len(sequences), interactions=int(counts.sum()))
    forward = transition_counts(sequences.values(), size)
    graphs = {'forward_recent': normalized_neighbors(forward),
              'symmetric_recent': normalized_neighbors(forward + forward.T),
              'reverse_recent': normalized_neighbors(forward.T.tocsr())}
    binary = graphs['forward_recent'].copy(); binary.data[:] = 1.
    graphs['forward_binary'] = binary
    graphs['forward_uniform'] = graphs['forward_recent']
    for name in ('forward_recent', 'symmetric_recent', 'reverse_recent'):
        sparse.save_npz(folder / f'{name}.npz', graphs[name])
    graph_info = dict(raw_directed_edges=forward.nnz, weighted_events=float(forward.sum()),
                      retained_edges={k: v.nnz for k, v in graphs.items()})
    del forward
    recent = history_matrix(histories, size)
    uniform = history_matrix(histories, size, recent=False)
    relation_features = {name: np.empty(data['ids'].shape, dtype=np.float64) for name in graphs}
    top_lists = {name: np.empty((len(selected), 50), dtype=np.int64)
                 for name in (*graphs, 'sasrec', 'itemcf', 'semantic', 'lime')}
    catalogue_ranks = {name: np.empty(len(selected), dtype=np.int64) for name in top_lists}
    positive_counts = {name: np.empty(len(selected), dtype=np.int64) for name in graphs}
    parity_max = 0.
    all_ids = np.arange(1, size)
    for begin in range(0, len(selected), 64):
        end = min(begin + 64, len(selected)); sl = slice(begin, end)
        source_indices = data['sample_indices'][sl]
        cached = np.array(experts[source_indices])
        # Matrix reconstruction checks that the existing baseline already has edge strength + recency.
        cf = (recent[sl] @ cfgraph).toarray()[:, 1:]
        cf = (cf - cf.min(1, keepdims=True)) / (cf.max(1, keepdims=True) - cf.min(1, keepdims=True) + 1e-8)
        error = float(np.max(np.abs(cf - cached[:, 1])))
        parity_max = max(parity_max, error)
        np.testing.assert_allclose(cf, cached[:, 1], atol=5e-6, rtol=1e-5)
        scores = dict(sasrec=cached[:, 0], itemcf=cached[:, 1], semantic=cached[:, 2],
                      lime=np.array(lime[source_indices]))
        for name, graph in graphs.items():
            h = uniform if name == 'forward_uniform' else recent
            raw = (h[sl] @ graph).toarray()[:, 1:]
            norm = raw / np.maximum(raw.max(1, keepdims=True), 1e-12)
            relation_features[name][sl] = np.take_along_axis(norm, data['ids'][sl] - 1, axis=1)
            scores[name] = raw
        for name, score in scores.items():
            for j, history in enumerate(histories[begin:end]):
                score[j, np.asarray(history) - 1] = -np.inf
            if name in graphs:
                positive_counts[name][sl] = (score > 0).sum(1)
                score[score <= 0] = -np.inf
            top_lists[name][sl] = topk(score, all_ids, 50)
            catalogue_ranks[name][sl] = ranks(score, all_ids, data['targets'][sl])
        if begin % 512 == 0 or end == len(selected):
            status('catalogue_evidence', domain=domain, completed=end, total=len(selected))
    np.savez_compressed(folder / 'relation_features.npz', **relation_features)
    np.savez_compressed(folder / 'catalogue_top50.npz', **top_lists)
    np.savez_compressed(folder / 'catalogue_ranks.npz', **catalogue_ranks)
    np.savez_compressed(folder / 'catalogue_positive_counts.npz', **positive_counts)
    write(folder / 'input_manifest.json', dict(sha256=manifest, itemcf_parity_max_error=parity_max,
          graph=graph_info, test_read=False, graph_labels_used=False, histories_match_allowed_sequences=True))

    prior = 10 * data['mixed']
    folds = fold_ids(data['users'], domain)
    pred_ranks = dict(gram=ranks(np.where(data['valid'], data['native'], -np.inf), data['ids'], data['targets']),
                      fixed_late=ranks(np.where(data['valid'], prior, -np.inf), data['ids'], data['targets']))
    full_predictions = {}
    model_meta = {}
    for name in ARMS:
        x = data['x'].astype(np.float64)
        if name != 'base':
            x = np.concatenate((x, relation_features[name][:, :, None]), axis=2)
        predictions = np.full(prior.shape, np.nan)
        model_meta[name] = []
        for fold in range(6):
            fit = train if fold == 5 else train[folds[train] != fold]
            evaluation = dev if fold == 5 else train[folds[train] == fold]
            head = fit_head(x, prior, data['valid'], data['ids'], data['targets'], fit)
            predictions[evaluation] = predict_head(head, x[evaluation], prior[evaluation])
            np.savez_compressed(folder / f'head_{name}_{fold}.npz', **head)
            model_meta[name].append(dict(fold=fold, fit_users=len(fit), positive_users=len(head['used']),
                                        iterations=head['iterations'], gradient_max=head['gradient_max']))
            status('fit_calibration', domain=domain, arm=name, fold=fold,
                   completed=6 * ARMS.index(name) + fold + 1, total=6 * len(ARMS))
        assert np.isfinite(predictions).all()
        full_predictions[name] = predictions
        pred_ranks[name] = ranks(np.where(data['valid'], predictions, -np.inf), data['ids'], data['targets'])
    np.savez_compressed(folder / 'predictions.npz', **full_predictions)
    np.savez_compressed(folder / 'ranks.npz', **pred_ranks)
    np.savez_compressed(folder / 'alignment.npz', users=data['users'], targets=data['targets'], roles=data['roles'], folds=folds)

    summary = dict(domain=domain, graph=graph_info, itemcf_parity_max_error=parity_max,
                   models=model_meta, groups={})
    for role, ix in (('oof', train), ('development', dev)):
        group = dict(users=len(ix), ranking={k: metrics(v[ix]) for k, v in pred_ranks.items()},
                     catalogue={k: metrics(v[ix]) for k, v in catalogue_ranks.items()},
                     primary_comparisons={k: contrast(pred_ranks['forward_recent'][ix], pred_ranks[k][ix])
                                          for k in ('base', 'symmetric_recent', 'reverse_recent', 'fixed_late')})
        group['coverage'] = {}
        for name, top in top_lists.items():
            new_native = 0; new_existing = 0; union25 = 0; supported = 0
            for i in ix:
                y = data['targets'][i]
                native = data['ids'][i][data['valid'][i]]
                existing = set(native)
                for baseline in ('sasrec', 'itemcf', 'semantic', 'lime'):
                    existing.update(top_lists[baseline][i])
                guided = candidate_rows[i].get('guided')
                if guided is not None:
                    existing.update(item for item in guided['ids'] if item not in histories[i])
                hit = y in top[i]
                new_native += int(hit and y not in native)
                new_existing += int(hit and y not in existing)
                order = np.lexsort((data['ids'][i], -data['native'][i]))
                native25 = [item for item in data['ids'][i][order] if item not in histories[i]][:25]
                union25 += int(y in set(native25) | set(top[i, :25]))
                supported += int(hit)
            group['coverage'][name] = dict(top50_hits=supported, new_vs_gram50=new_native,
                new_vs_all_existing=new_existing, gram25_plus_source25_hits=union25,
                includes_guided_baseline=(role == 'development'))
        summary['groups'][role] = group
    controls = ('base', 'symmetric_recent', 'reverse_recent')
    fold_results = []
    for fold in range(5):
        ix = train[folds[train] == fold]
        table = {name: metrics(pred_ranks[name][ix]) for name in ARMS}
        win = all(table['forward_recent']['N10'] > table[c]['N10'] for c in controls)
        fold_results.append(dict(fold=fold, users=len(ix), ranking=table, primary_wins_all_controls=win))
    summary['folds'] = fold_results
    d = summary['groups']['development']['ranking']; t = summary['groups']['oof']['ranking']
    gates = dict(development_relative_N10=all(d['forward_recent']['N10'] >= 1.01 * d[c]['N10'] for c in controls),
                 development_H10=d['forward_recent']['H10'] >= d['base']['H10'],
                 oof_N10=all(t['forward_recent']['N10'] > t[c]['N10'] for c in controls),
                 fold_consistency=sum(f['primary_wins_all_controls'] for f in fold_results) >= 4,
                 new_opportunities=summary['groups']['development']['coverage']['forward_recent']['new_vs_all_existing'] >= 3)
    summary.update(gates=gates, passes=all(gates.values()), seconds=time.monotonic()-start,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   test_read=False, new_generation=False, upstream_training=False)
    write(folder / 'summary.json', summary)
    status('domain_complete', domain=domain, passes=summary['passes'], seconds=summary['seconds'])
    return summary


def report(result):
    lines = ['# 第21阶段：关系方向首轮探索', '', '## Material Passport', '',
             f'- 完成时间：{stamp()}；Origin Skill：academic-research-suite，内联执行。',
             '- 状态：COMPLETED；探索性开发证据。CPU运行，未新生成、未训练上游、未读取test。',
             '- 关系图仅由移除训练末次留出事件后的允许序列构建；校准按用户五折外预测。', '',
             '**决策：' + ('通过冻结双域门槛，可进入生成接口验证。' if result['passes'] else '未通过冻结双域门槛，本轮不启动方向图的生成大实验。') + '**', '',
             '唯一主臂是前向加权近期关系；不按开发结果改选方法。下面结果只评价候选证据和后重排，不证明生成收益。', '',
             '| 域/集合 | 方法 | NDCG@10 | H10 |', '|---|---|---:|---:|']
    for domain, summary in result['domains'].items():
        for role, group in summary['groups'].items():
            for arm, metric in group['ranking'].items():
                lines.append(f"| {domain}/{role} | {arm} | {metric['N10']:.6f} | {metric['H10']:.6f} |")
    lines += ['', '## 全目录补充机会', '',
              '| 域/开发 | 信号 | top50命中 | GRAM50外新增 | 全部现有来源外新增 | GRAM25+信号25覆盖 |',
              '|---|---|---:|---:|---:|---:|']
    for domain, summary in result['domains'].items():
        for arm, cov in summary['groups']['development']['coverage'].items():
            lines.append(f"| {domain} | {arm} | {cov['top50_hits']} | {cov['new_vs_gram50']} | {cov['new_vs_all_existing']} | {cov['gram25_plus_source25_hits']} |")
    lines += ['', '以上是目标覆盖机会，合并集合的命中不是可实现推荐指标。全部现有来源包括原生50、引导50、三专家及LIME各自top50；所有来源排除最近20项历史。', '', '## 收益、误伤与投入条件', '']
    for domain, summary in result['domains'].items():
        c = summary['groups']['development']['primary_comparisons']['base']
        lines += [f"- {domain}：主臂相对普通校准 ΔN10={c['delta_N10']:+.6f}；描述性95%区间 {c['descriptive_95pct_N10']}；新增/丢失前十命中={c['top10_gained']}/{c['top10_lost']}。",
                  f"  门槛：{json.dumps(summary['gates'], ensure_ascii=False)}。用时{summary['seconds']/60:.2f}分钟；累计峰值RSS {summary['peak_rss_mib']:.0f} MiB。"]
    lines += ['', '## 解释边界', '',
              '- 现有ItemCF本身已有关系强度和近期加权，本轮重算逐行与缓存核对；额外假设仅是局部方向。',
              '- 固定原候选上的校准只从目标覆盖者学习；全体用户进入评价分母。图漏召回与排序误伤分别报告。',
              '- 无向、反向、二值边、均匀历史是固定对照；不能把任一单域好结果写成主假设成功。',
              '- 开发池已反复研究，bootstrap是固定预测的用户条件区间；没有上游多seed、独立测试或多比较显著性声明。',
              '- 本轮负结果只约束当前窗口、图归一化与线性使用方式，不否定所有关系模型。', '',
              '计划：[冻结候选卡与协议](../../plan/第二十一阶段/21.1_候选卡与关系方向首轮探索_20260922.md)。',
              '完整指标、折结果、覆盖、模型和输入哈希：[实验目录](../../artifacts/phase21/relation_direction_20260922_v1/)。', '']
    REPORT.write_text('\n'.join(lines))
    (OUT / 'report.md').write_text('\n'.join(lines))


def worker():
    try:
        for rel, digest in read(OUT / 'launch_manifest.json')['sha256'].items():
            assert sha(ROOT / rel) == digest, rel
        domains = {domain: run_domain(domain) for domain in ('Toys', 'Beauty')}
        result = dict(state='COMPLETED', domains=domains, passes=all(s['passes'] for s in domains.values()),
                      completed_at=stamp(), test_read=False)
        write(OUT / 'summary.json', result)
        report(result)
        write(OUT / 'worker_status.json', dict(state='COMPLETED', stage='finished', updated_at=stamp(), pid=os.getpid(), passes=result['passes']))
    except BaseException as error:
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        write(OUT / 'worker_status.json', dict(state='FAILED', updated_at=stamp(), pid=os.getpid(), error=repr(error)))
        raise


def supervise():
    started = time.monotonic()
    with (OUT / 'worker.log').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-m', MODULE, 'worker'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    while proc.poll() is None:
        detail = read(OUT / 'worker_status.json') if (OUT / 'worker_status.json').exists() else {}
        write(OUT / 'status.json', dict(state='RUNNING', supervisor_pid=os.getpid(), worker_pid=proc.pid,
              updated_at=stamp(), elapsed_seconds=time.monotonic()-started, detail=detail))
        if time.monotonic() - started > 2700:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            write(OUT / 'status.json', dict(state='TIMED_OUT', updated_at=stamp(), error='45 minute hard timeout'))
            return
        time.sleep(15)
    detail = read(OUT / 'worker_status.json') if (OUT / 'worker_status.json').exists() else {}
    write(OUT / 'status.json', dict(state='COMPLETED' if proc.returncode == 0 else 'FAILED',
          exit_code=proc.returncode, updated_at=stamp(), elapsed_seconds=time.monotonic()-started, detail=detail))


def launch():
    OUT.mkdir(exist_ok=False)
    shutil.copy2(PLAN, OUT / 'protocol.md')
    sources = [PLAN, Path(__file__), ROOT / 'experiment/phase21/relation_direction_core.py',
               ROOT / 'experiment/phase20/catalog_scope_core.py']
    write(OUT / 'launch_manifest.json', dict(sha256={str(p.relative_to(ROOT)): sha(p) for p in sources}))
    for path in sources:
        dest = OUT / 'sources' / path.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write(OUT / 'status.json', dict(state='LAUNCHING', updated_at=stamp(), hard_timeout_seconds=2700))
    with (OUT / 'supervisor.log').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-m', MODULE, 'supervise'], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(OUT / 'launch.json', dict(pid=proc.pid, started_at=stamp(), command=f'bash experiment/phase21/run_relation_direction.sh'))
    print(json.dumps(dict(supervisor_pid=proc.pid, output=str(OUT))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('launch', 'supervise', 'worker'))
    action = parser.parse_args().action
    {'launch': launch, 'supervise': supervise, 'worker': worker}[action]()
