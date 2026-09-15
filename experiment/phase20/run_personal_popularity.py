"""Bounded background PP probe, with one incremental human-readable report."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

from experiment.phase20.audit_pcrf_components import digest, read_rows, metric, paired
from experiment.phase20.personal_popularity_core import (
    reconstruct_training, incidence, neighborhoods, shuffled_users, donor_neighbors,
    candidate_popularity, standardize_rows, ranks_from_scores, user_folds, crossfit,
)
from experiment.phase18.core.raserec_data import training_frequencies
from experiment.phase9.eval_cf0_b3_beamfusion import (
    load_cached_beams, normalize_lexical_id, score_item_head,
)
from experiment.phase9.eval_cf0_b4_reliability import rank_matrix

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'experiment/phase20/personal_popularity_probe.json'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S %z')


def report(config, text):
    path = ROOT / config['report']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f'\n{stamp()}　{text}\n')
        handle.flush()


def source_hashes(config_path):
    paths = [config_path, Path(__file__), ROOT / 'experiment/phase20/personal_popularity_core.py',
             ROOT / 'experiment/phase20/audit_pcrf_components.py',
             ROOT / 'experiment/phase18/core/raserec_data.py',
             ROOT / 'experiment/phase9/eval_cf0_b3_beamfusion.py',
             ROOT / 'experiment/phase9/eval_cf0_b4_reliability.py',
             ROOT / 'experiment/phase9/train_cf0_b2_item_head.py',
             ROOT / 'experiment/phase20/test_personal_popularity.py',
             ROOT / 'experiment/phase20/run_personal_popularity.sh']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def load_predictions(dc):
    if dc['prediction_format'] == 'jsonl':
        raw = read_rows(ROOT / dc['predictions'])
        assert all(row['split'] == 'validation' for row in raw)
        result = {row['user_id']: {'candidates': row['ranked_item_ids'],
                                  'scores': row['scores'], 'gold': row['gold_item_id']} for row in raw}
    else:
        catalog = json.loads((ROOT / dc['catalog']).read_text())
        lexical = catalog['lexical_ids']
        assert len(lexical) == dc['catalog_size_with_padding']
        ids = {normalize_lexical_id(value): index for index, value in enumerate(lexical) if index}
        assert len(ids) == len(lexical)-1
        raw, _ = load_cached_beams(ROOT / dc['predictions'])
        result = {user: {'candidates': [ids[normalize_lexical_id(v)] for v in row['candidates']],
                         'scores': row['seq'], 'gold': ids[normalize_lexical_id(row['gold'])]}
                  for user, row in raw.items()}
    assert len(result) == len(raw) == dc['users']
    if dc['reference_format'] == 'json':
        reference = json.loads((ROOT / dc['reference']).read_text())
    else:
        with (ROOT / dc['reference']).open() as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        reference = {row['user_id']: row for row in rows}
        assert len(rows) == len(reference)
    assert set(result) == set(reference)
    return result, reference


def feature_correlation(a, b):
    valid = (a.std(axis=1) > 1e-6) & (b.std(axis=1) > 1e-6)
    return {'eligible_users': int(valid.sum()),
            'mean_within_user_pearson': float((standardize_rows(a[valid]) *
                 standardize_rows(b[valid])).mean()) if valid.any() else None}


def worker(config_path, domain, mode):
    config = json.loads(config_path.read_text())
    dc = config['domains'][domain]
    torch.set_num_threads(config['cpu_threads'])
    started = time.monotonic()
    name = domain + ('_smoke' if mode == 'smoke' else '')
    output = ROOT / config['output'] / name
    output.mkdir(parents=True, exist_ok=False)

    def status(state, stage, **extra):
        write_json(output / 'status.json', {'state': state, 'stage': stage, 'pid': os.getpid(),
                   'updated_at': stamp(), 'elapsed_seconds': round(time.monotonic()-started, 2),
                   'dataset': domain, 'mode': mode, 'gpu_used': False, **extra})

    def event(stage, message):
        status('RUNNING', stage)
        report(config, f'**{name} / {stage}**：{message}')

    def interrupted(signum, frame):
        raise InterruptedError(f'Received signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        hashes = source_hashes(config_path)
        for key in ('train', 'validation', 'predictions', 'reference', 'item_head', 'catalog'):
            if key in dc:
                hashes[dc[key]] = digest(ROOT / dc[key])
        manifest = {'config': config, 'domain': domain, 'mode': mode, 'input_sha256': hashes,
                    'started_at': stamp(), 'test_read': False}
        write_json(output / 'protocol.json', manifest)
        event('inputs', '读取prepared train/validation及冻结候选，检查用户、商品和父参考。')
        train = read_rows(ROOT / dc['train'])
        validation = read_rows(ROOT / dc['validation'])
        users, sequences, recent, complete = reconstruct_training(train, config['max_history'])
        lookup = {row['user_id']: row for row in validation}
        assert len(lookup) == len(validation) == len(users) == dc['users']
        assert set(lookup) == set(users)
        for user in users:
            assert sequences[user][-config['max_history']:] == lookup[user]['history']
        frequency = training_frequencies(train, dc['catalog_size_with_padding'])
        bank = incidence(recent, len(frequency))
        full_bank = incidence(complete, len(frequency))
        global_pop = np.asarray(full_bank.sum(axis=0)).ravel() / len(users)
        predictions, reference = load_predictions(dc)
        assert set(users) == set(predictions)
        indices = np.arange(min(config['smoke_users'], len(users)) if mode == 'smoke' else len(users))
        query_users = [users[i] for i in indices]
        records, positions, target_frequency = [], [], []
        for user in query_users:
            row, pred = lookup[user], predictions[user]
            candidates = pred['candidates']
            assert len(candidates) == len(set(candidates)) == len(pred['scores']) == 50
            assert all(0 < i < len(frequency) for i in candidates)
            assert int(pred['gold']) == row['target']
            assert np.isfinite(pred['scores']).all()
            assert int(reference[user]['target_frequency']) == frequency[row['target']]
            records.append({'user': user, 'history': row['history'], 'candidate_ids': candidates,
                            'seq': np.asarray(pred['scores'], dtype=np.float64)})
            positions.append(candidates.index(row['target']) if row['target'] in candidates else -1)
            target_frequency.append(frequency[row['target']])
        candidates = np.asarray([r['candidate_ids'] for r in records])
        positions, target_frequency = np.asarray(positions), np.asarray(target_frequency)
        event('neighbors', '仅用训练历史构造Jaccard邻居，固定k=30；保留零相似度诊断。')
        neighbor_started = time.monotonic()
        # Smoke computes its shuffled donors too, using the full bank throughout.
        shuffle = shuffled_users(users, config['shuffle_seed'])
        needed = np.unique(np.r_[indices, shuffle[indices]])
        neighbor_ids, neighbor_scores, positive = neighborhoods(
            bank, needed, config['neighbors']+1, config['neighbor_block_size'],
            lambda done, total: status('RUNNING', 'neighbors', completed=done, total=total))
        by_bank = {int(bank_index): row for row, bank_index in enumerate(needed)}
        real_ids = np.asarray([neighbor_ids[by_bank[int(i)], :config['neighbors']] for i in indices])
        donor_rows = np.asarray([neighbor_ids[by_bank[int(i)]] for i in shuffle[indices]])
        shuffled_ids = donor_neighbors(donor_rows, np.arange(len(indices)), indices, config['neighbors'])
        assert not (real_ids == indices[:, None]).any()
        assert not (shuffled_ids == indices[:, None]).any()
        pp = candidate_popularity(bank, real_ids, candidates, config['neighbor_block_size'])
        shuffled_pp = candidate_popularity(bank, shuffled_ids, candidates, config['neighbor_block_size'])
        gp = global_pop[candidates]
        neighbor_seconds = time.monotonic()-neighbor_started
        positive_queries = np.asarray([positive[by_bank[int(i)]] for i in indices])
        real_sims = np.asarray([neighbor_scores[by_bank[int(i)], :config['neighbors']] for i in indices])
        event('reference_scoring', 'CPU复算原CF head；原GRAM/PCRF逐用户rank必须与已保存参考一致。')
        scoring_started = time.monotonic()
        score_item_head(records, ROOT / dc['item_head'], config['batch_size'])
        scoring_seconds = time.monotonic()-scoring_started
        seq = standardize_rows(np.stack([r['seq'] for r in records]))
        cf = standardize_rows(np.stack([r['cf'] for r in records]))
        pop = standardize_rows(np.log1p(frequency[candidates]))
        tail_mass = (frequency[candidates[:, :10]] <= dc['q1']).mean(axis=1)
        weight, beta, gamma = dc['pcrf']
        joint = seq + weight * ((1-tail_mass)**gamma)[:, None] * standardize_rows(cf-beta*pop)
        raw_ranks = ranks_from_scores(seq, positions)
        pcrf_ranks = ranks_from_scores(joint, positions)
        expected, _ = rank_matrix(seq, cf, pop, tail_mass, positions, tuple(dc['pcrf']))
        assert np.array_equal(pcrf_ranks, expected)
        for actual, key in ((raw_ranks, 'baseline_rank'), (pcrf_ranks, 'pcrf_rank')):
            assert np.array_equal(actual, np.asarray([int(reference[u][key]) for u in query_users]))
        diagnostics = {
            'users': len(indices), 'bank_users': len(users), 'k': config['neighbors'],
            'users_with_positive_similarity_neighbor': int((positive_queries > 0).sum()),
            'users_with_at_least_k_positive_neighbors': int((positive_queries >= config['neighbors']).sum()),
            'mean_positive_neighbor_pool_size': float(positive_queries.mean()),
            'mean_selected_neighbor_jaccard': float(real_sims.mean()),
            'selected_zero_similarity_fraction': float((real_sims == 0).mean()),
            'candidate_pp_all_zero_users': int((pp.max(axis=1) == 0).sum()),
            'candidate_pp_constant_users': int((pp.std(axis=1) <= 1e-6).sum()),
            'pp_cf_correlation': feature_correlation(pp, cf),
            'pp_global_pop_correlation': feature_correlation(pp, gp),
            'neighbor_seconds': round(neighbor_seconds, 3), 'cf_scoring_seconds': round(scoring_seconds, 3),
            'exact_reference_users': len(indices),
            'max_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2),
        }
        if mode == 'smoke':
            assert hashes == {p: digest(ROOT / p) for p in hashes}
            write_json(output / 'smoke.json', {'state': 'PASSED', 'diagnostics': diagnostics,
                                              'input_sha256': hashes, 'new_method_metrics_computed': False})
            status('COMPLETED', 'smoke_passed')
            report(config, f'**{name}检查通过**：{len(indices)}人父参考逐用户一致；'
                   f'邻居计算{neighbor_seconds:.1f}s、CF评分{scoring_seconds:.1f}s。未计算新增方法准确率。')
            return
        event('evaluation', '原PCRF、PP独排及三种增量对照；Toys五折开发，Beauty使用冻结系数。')
        etas = config['etas']
        features = {'real_pp': pp, 'shuffled_pp': shuffled_pp, 'global_pop': gp}
        rank_grids = {name: [ranks_from_scores(joint + eta * standardize_rows(feature), positions)
                            for eta in etas] for name, feature in features.items()}
        for values in rank_grids.values():
            assert np.array_equal(values[0], pcrf_ranks)
        folds = user_folds(query_users, config['fold_seed'], config['fold_count'])
        ranks = {'gram': raw_ranks, 'pcrf': pcrf_ranks, 'pp_only': ranks_from_scores(pp, positions)}
        selections, transfer_etas = {}, {}
        if domain == 'Toys':
            for name, grid in rank_grids.items():
                ranks['pcrf_plus_' + name], chosen, full = crossfit(grid, folds, etas)
                selections[name] = {'folds': chosen, 'full_toys_selection_for_transfer': full}
                transfer_etas[name] = full['eta']
        else:
            toys_path = ROOT / config['output'] / 'Toys' / 'summary.json'
            toys = json.loads(toys_path.read_text())
            if not toys['promising_for_beauty']:
                raise ValueError('Beauty prerequisite not satisfied')
            transfer_etas = toys['transfer_etas']
            manifest['toys_summary_sha256'] = digest(toys_path)
            write_json(output / 'protocol.json', manifest)
            for name, grid in rank_grids.items():
                ranks['pcrf_plus_' + name] = grid[etas.index(transfer_etas[name])]
                selections[name] = {'frozen_toys_eta': transfer_etas[name]}
        masks = {'overall': np.ones(len(indices), dtype=bool),
                 'tail': target_frequency <= dc['q1'], 'non_tail': target_frequency > dc['q1']}
        groups = {group: {name: metric(r[mask]) for name, r in ranks.items()}
                  for group, mask in masks.items() if mask.any()}
        comparisons = {}
        for baseline in ('pcrf', 'pcrf_plus_shuffled_pp', 'pcrf_plus_global_pop'):
            comparisons['real_pp_minus_' + baseline] = {
                group: paired(ranks['pcrf_plus_real_pp'][mask], ranks[baseline][mask], config)
                for group, mask in masks.items() if mask.any()}
        assert all(np.array_equal(r <= 50, raw_ranks <= 50) for r in ranks.values())
        assert hashes == {p: digest(ROOT / p) for p in hashes}
        delta = groups['overall']['pcrf_plus_real_pp']['ndcg@10'] - groups['overall']['pcrf']['ndcg@10']
        promising = bool(delta > 0 and transfer_etas['real_pp'] > 0)
        np.savez_compressed(output / 'features.npz', pp=pp, shuffled_pp=shuffled_pp, gp=gp,
                            cf=cf, joint=joint, candidates=candidates, real_neighbors=real_ids,
                            shuffled_neighbors=shuffled_ids, target_positions=positions,
                            users=np.asarray(query_users), shuffle_donors=np.asarray(users)[shuffle[indices]])
        with (output / 'per_user.tsv').open('w') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['user_id', 'fold', *ranks])
            for i, user in enumerate(query_users):
                writer.writerow([user, int(folds[i]), *(int(r[i]) for r in ranks.values())])
        grid_metrics = ({name: {str(eta): metric(r) for eta, r in zip(etas, grid)}
                         for name, grid in rank_grids.items()} if domain == 'Toys' else {})
        summary = {'state': 'COMPLETED', 'dataset': domain, 'verification_status': 'ANALYZED',
                   'groups': groups, 'paired_comparisons': comparisons, 'selections': selections,
                   'transfer_etas': transfer_etas, 'full_development_grid_metrics': grid_metrics,
                   'diagnostics': diagnostics, 'promising_for_beauty': promising if domain == 'Toys' else None,
                   'input_sha256': hashes, 'test_read': False, 'new_training_steps': 0, 'gpu_used': False,
                   'elapsed_seconds': round(time.monotonic()-started, 2),
                   'limitations': 'Reused validation; fixed parent/head; PP-inspired additive signal only; conditional user bootstrap does not refit eta or cover head seeds; no multiple-comparison correction.'}
        write_json(output / 'summary.json', summary)
        status('COMPLETED', 'finished', summary=str(output.relative_to(ROOT) / 'summary.json'))
        lines = [f'**{domain}完成**：{len(indices):,}人，{summary["elapsed_seconds"]:.1f}s；'
                 + ('五折OOF开发结果。' if domain == 'Toys' else 'Toys系数冻结后跨域评估，无Beauty调参。'),
                 '', '| 方法 | NDCG@10 | Hit@10 | 低频NDCG@10 |', '|---|---:|---:|---:|']
        for name, values in groups['overall'].items():
            tail = groups.get('tail', {}).get(name, {}).get('ndcg@10', float('nan'))
            lines.append(f'| {name} | {values["ndcg@10"]:.6f} | {values["hit@10"]:.6f} | {tail:.6f} |')
        ci = comparisons['real_pp_minus_pcrf']['overall']['ndcg@10']['ci95']
        lines += ['', f'真实PP对PCRF：NDCG@10 Δ={delta:+.6f}，描述性95%区间'
                  f'[{ci[0]:+.6f}, {ci[1]:+.6f}]。转移系数：`{transfer_etas}`。',
                  f'候选PP全零用户：{diagnostics["candidate_pp_all_zero_users"]}/{len(indices)}；'
                  f'选中邻居零相似度比例：{diagnostics["selected_zero_similarity_fraction"]:.1%}。',
                  f'证据：`{output.relative_to(ROOT)}/summary.json`。仅代表当前PP信号配置，未训练完整PPAC。']
        report(config, '\n'.join(lines))
    except BaseException as error:
        state = 'STOPPED' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'FAILED'
        status(state, 'terminated', error=repr(error))
        report(config, f'**{name} {state}**：`{error!r}`；产物保留，无自动重试。')
        raise


def supervise(config_path):
    config = json.loads(config_path.read_text())
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'queue_status.json').exists():
        raise FileExistsError('Existing queue attempt; choose a new protocol for a rerun')
    child = None
    def save(state, **extra):
        write_json(output / 'queue_status.json', {'state': state, 'supervisor_pid': os.getpid(),
                   'updated_at': stamp(), 'automatic_retry': False, 'gpu_used': False, **extra})
    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise InterruptedError(f'Queue received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        expected_sources = source_hashes(config_path)
        for domain in ('Toys', 'Beauty'):
            smoke = json.loads((output / f'{domain}_smoke/smoke.json').read_text())
            if smoke['state'] != 'PASSED' or any(smoke['input_sha256'].get(k) != v
                                                for k, v in expected_sources.items()):
                raise ValueError(f'{domain} smoke missing or source changed')
        save('STARTING')
        report(config, '**后台队列启动**：CPU最多4线程，每域30分钟硬上限；不持续人工轮询。')
        completed = []
        for domain in ('Toys', 'Beauty'):
            if domain == 'Beauty':
                toys = json.loads((output / 'Toys/summary.json').read_text())
                if not toys['promising_for_beauty']:
                    report(config, '**下一步**：Toys未触发冻结跨域条件，Beauty不启动；当前队列结束。'
                           '完整PPAC与EAGER训练需依据该结果另定具体配置。')
                    save('COMPLETED', completed_domains=completed, beauty='NOT_TRIGGERED')
                    return
            if (output / domain).exists():
                raise FileExistsError(f'Will not overwrite existing {domain}')
            with (output / f'{domain}.log').open('x') as handle:
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                           MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
                command = [sys.executable, '-u', '-m', 'experiment.phase20.run_personal_popularity',
                           '--config', str(config_path), '--mode', 'run', '--domain', domain]
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                save('RUNNING', dataset=domain, child_pid=child.pid, completed_domains=completed)
                try:
                    code = child.wait(timeout=config['max_wall_seconds_per_domain'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise TimeoutError(f'{domain} exceeded the 30-minute hard limit')
            if code != 0:
                raise RuntimeError(f'{domain} exited {code}; inspect {domain}.log')
            assert json.loads((output / domain / 'summary.json').read_text())['state'] == 'COMPLETED'
            completed.append(domain)
        report(config, '**后台队列完成**：Toys及冻结Beauty结果已记录；未自动启动完整PPAC或EAGER训练。')
        save('COMPLETED', completed_domains=completed)
    except BaseException as error:
        save('STOPPED' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'FAILED', error=repr(error))
        report(config, f'**队列结束**：`{error!r}`；无自动重试。')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--mode', choices=('smoke', 'run', 'supervise'), required=True)
    parser.add_argument('--domain', choices=('Toys', 'Beauty'), default='Toys')
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.mode == 'supervise':
        supervise(config_path)
    else:
        worker(config_path, args.domain, args.mode)
