"""Bounded background queue for four cached B1 continuations and two profiles."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

from experiment.phase20.gacr_fusion_queue import read, write

ROOT = Path(__file__).resolve().parents[2]
CHILD = None
STOPPING = False
KEYS = [f'{m}@{k}' for m in ('ndcg', 'hit') for k in (5, 10, 20, 50)]


def stop(signum, frame):
    global STOPPING
    STOPPING = True
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)


def aggregate(config):
    result = {'state': 'COMPLETED', 'domains': {}, 'new_optimizer_steps': 120,
              'backbone_optimizer_steps': 0, 'test_read': False, 'new_beam_inference': False,
              'verification_status': 'ANALYZED', 'automatic_next_experiment': False,
              'scope': 'Only residual initialization differs; 2023 is reused observed evidence, not a new replication.'}
    for d, cfg in config['domains'].items():
        summaries = {'2023': read(ROOT / cfg['known_summary'])}
        summaries.update({str(seed): read(ROOT / config['output'] / d / f'seed{seed}/summary.json') for seed in config['new_seeds']})
        assert all(s['state'] == 'COMPLETED' for s in summaries.values())
        assert sum(summaries[str(seed)]['new_optimizer_steps'] for seed in config['new_seeds']) == 60
        aggregates = {}
        for group in summaries['2023']['groups']:
            group_result = aggregates[group] = {}
            for arm in ('B0_raw', 'B0', 'B1_raw', 'B1'):
                values = group_result[arm] = {}
                for k in KEYS:
                    xs = [s['groups'][group][arm][k] for s in summaries.values()]
                    deltas = [s['groups'][group][arm][k] - s['groups'][group]['B0_raw' if arm.endswith('_raw') else 'B0'][k] for s in summaries.values()]
                    values[k] = {'mean': statistics.mean(xs), 'min': min(xs), 'max': max(xs),
                        'sample_sd_across_residual_initializations': statistics.stdev(xs),
                        'mean_delta_vs_seed_matched_B0': statistics.mean(deltas),
                        'positive_initialization_count_vs_B0': sum(x > 0 for x in deltas)}
        decision = {}
        for metric in ('ndcg@10', 'hit@10'):
            for baseline in ('B0', 'original_pcrf'):
                ds = [s['groups']['all_validation']['B1'][metric] - s['groups']['all_validation'][baseline][metric] for s in summaries.values()]
                decision[f'{metric}_vs_{baseline}'] = {'positive_count': sum(v > 0 for v in ds),
                                                       'mean_delta': statistics.mean(ds), 'min_delta': min(ds)}
        result['domains'][d] = {'seed_sources': {seed: ('previously_observed' if seed == '2023' else 'new_continuation') for seed in summaries},
            'seed_metrics': {seed: s['groups'] for seed, s in summaries.items()}, 'aggregate': aggregates,
            'decision': decision, 'known_2023_source': cfg['known_summary']}
    return result


def report(config, status):
    text = '# Stage20 B1缓存复用多初始化验证进展与结果\n\n## Material Passport\n\n'
    text += '- Origin Skill：academic-research-suite / experiment-agent；Origin Mode：run / validate。\n'
    text += f'- 更新时间：{status["updated_at"]}；状态：**{status["state"]}**。\n'
    text += '- 用户已授权继续B1的2024/2025残差初始化验证；2023沿用已观察结果，不重训骨干、head或生成候选，不读test。\n\n'
    text += f'统一观察：`{config["output"]}/status.json`，后台30秒心跳；当前阶段进度和ETA见`child_status`，整轮粗估见`projected_remaining_seconds`。\n\n'
    text += '| 任务 | 状态 | 步骤 |\n|---|---|---|\n'
    for t in status['tasks']:
        text += f'| {t["id"]} | {t["state"]} | {t.get("stage", "")} |\n'
    text += '\n## 固定协议与资源\n\n'
    text += '两域各从既有残差seed2024/2025继续B1 30个全量fit更新。仅改变起始残差checkpoint，fit/calibration始终复用上轮seed2023划分与NPZ；129参数、6维特征、固定C1 50项、49负例pairwise、covered训练head/tail各半权重、AdamW lr0.01/weight decay0.01、clip10、128条梯度累积块及最后一步选取均固定。候选外目标不参与pairwise，全部validation用户保留。训练预算合计120步，profile只反向计算不执行optimizer。\n\n'
    text += '先完成两域profile：精确复现已发表2023 B0/B1的全量raw/PCRF rank及残差分数，核对缓存哈希，测一次完整fit梯度和旧结果bootstrap耗时；两域通过才按Toys2024→Toys2025→Beauty2024→Beauty2025执行。每个新初始化的B0须逐用户精确对齐上一轮同seed fixed结果，随后才训练。\n\n'
    text += f'GPU UUID `{config["gpu_uuid"]}`；空闲至少{config["minimum_free_mib"]} MiB、allocator上限{config["cuda_memory_fraction"]:.0%}、CPU{config["cpu_threads"]}线程；每profile上限{config["profile_budget_seconds"]/60:.0f}分钟，每个正式seed上限{config["task_budget_seconds"]/60:.0f}分钟，每任务等资源最多{config["resource_wait_seconds"]/60:.0f}分钟。失败/超时保存终态并停止后续，不自动重试，不干预其他进程。\n\n'
    for d in config['execution_order']:
        profile = read(ROOT / config['output'] / d / 'profile/profile.json')
        if profile:
            text += f'- {d} profile：{profile["state"]}；完整fit梯度{profile["full_fit_gradient_seconds"]:.2f}秒，单seed投影{profile["estimated_seconds_per_new_seed"]/60:.1f}分钟，峰值allocated {profile["peak_allocated_mib"]:.0f} MiB。工期为共享GPU负载下估计，含保守加载余量，不是硬承诺。\n'
    for d, cfg in config['domains'].items():
        text += f'\n## {d}结果\n\n| 初始化 | 来源 | B0 N10 | B1 N10 | B1 H10 | ΔN10对B0 | 配对95%区间 | ΔH10对B0 | Top10新增/丢失 |\n|---|---|---:|---:|---:|---:|---|---:|---:|\n'
        for seed in (2023, 2024, 2025):
            summary_path = cfg['known_summary'] if seed == 2023 else f'{config["output"]}/{d}/seed{seed}/summary.json'
            s = read(ROOT / summary_path)
            if not s:
                continue
            g, c = s['groups']['all_validation'], s['comparisons']['B1_minus_B0']['all_validation']
            ci = c['ndcg@10']['ci95']
            text += f'| {seed} | {"已观察复用" if seed == 2023 else "本轮新训练"} | {g["B0"]["ndcg@10"]:.6f} | {g["B1"]["ndcg@10"]:.6f} | {g["B1"]["hit@10"]:.6f} | {c["ndcg@10"]["delta"]:+.6f} | [{ci[0]:+.6f},{ci[1]:+.6f}] | {c["hit@10"]["delta"]:+.6f} | {c["hit10_gained"]}/{c["hit10_lost"]} |\n'
            text += f'\n' if seed == 2025 else ''
        text += f'\n各新seed全部八指标、五分组、四配对比较、训练轨迹、checkpoint和逐用户rank见`{config["output"]}/{d}/seed2024/`及`seed2025/`。\n'
    result = read(ROOT / config['output'] / 'summary.json')
    if result:
        text += '\n## 三初始化描述性汇总\n\n| 域 | 臂 | N5 | N10 | N20 | N50 | H5 | H10 | H20 | H50 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for d, s in result['domains'].items():
            for arm in ('B0_raw', 'B0', 'B1_raw', 'B1'):
                m = s['aggregate']['all_validation'][arm]
                text += f'| {d} | {arm} | ' + ' | '.join(f'{m[k]["mean"]:.6f}' for k in KEYS) + ' |\n'
        text += '\n| 域 | 比较 | 正增量初始化数 | 平均增量 | 最差增量 |\n|---|---|---:|---:|---:|\n'
        for d, s in result['domains'].items():
            for name, v in s['decision'].items():
                text += f'| {d} | {name} | {v["positive_count"]}/3 | {v["mean_delta"]:+.6f} | {v["min_delta"]:+.6f} |\n'
    text += '\n## 解释边界\n\n这是固定父模型、C1、head和训练划分下的残差初始化验证，不是整模型三个独立seed重训。2023已被观察，不能称三次新独立确认。3000次用户配对bootstrap、seed20260912，未校正多重比较，不含训练初始化和开发历史的不确定性；三初始化均值、范围、标准差为描述性统计。B1与每个初始化自己的B0比较，不能挑最差B0或最好B1。H50固定于C1集合；低频覆盖代价保留。下阶段内部化或其他控制不自动触发。\n\n'
    text += '[前轮B1/B2完成报告](Stage20_固定候选最终融合训练进展与结果.md)、[方法推进计划](../../plan/第二十阶段/GACR_PCRF双域提点后的方法推进计划.md)。\n'
    target = ROOT / config['report']
    temp = target.with_suffix('.tmp.md'); temp.write_text(text); temp.replace(target)


def main():
    global CHILD
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    args = parser.parse_args(); config = read(ROOT / args.config)
    out = ROOT / config['output']; out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'queue.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'status.json').exists(), 'Existing attempt; no overwrite/restart'
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    tasks = [{'id': f'{d}_profile', 'dataset': d, 'seed': 2023, 'mode': 'profile', 'state': 'PENDING'} for d in config['execution_order']]
    tasks += [{'id': f'{d}_seed{seed}', 'dataset': d, 'seed': seed, 'mode': 'run', 'state': 'PENDING'} for d in config['execution_order'] for seed in config['new_seeds']]
    started = time.monotonic()
    def save(state, **extra):
        statuses = {t['id']: read(out / t['dataset'] / ('profile' if t['mode'] == 'profile' else f'seed{t["seed"]}') / 'status.json') for t in tasks}
        profiles = {d: read(out / d / 'profile/profile.json') for d in config['execution_order']}
        remaining = None
        if all(p.get('state') == 'PASSED' for p in profiles.values()):
            remaining = 0.
            for t in tasks:
                if t['mode'] == 'run' and t['state'] in ('PENDING', 'RUNNING'):
                    estimate = profiles[t['dataset']]['estimated_seconds_per_new_seed']
                    remaining += max(0., estimate - statuses[t['id']].get('elapsed_seconds', 0))
            remaining = round(remaining, 1)
        row = {'state': state, 'pid': os.getpid(), 'pid_start_ticks': Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
               'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'heartbeat_interval_seconds': 30,
               'elapsed_seconds': round(time.monotonic()-started, 1), 'tasks': tasks,
               'completed_tasks': sum(t['state'] == 'COMPLETED' for t in tasks), 'total_tasks': len(tasks),
               'completed_new_training_tasks': sum(t['state'] == 'COMPLETED' and t['mode'] == 'run' for t in tasks),
               'total_new_training_tasks': 4, 'task_status': statuses, 'gpu_uuid': config['gpu_uuid'],
               'projected_remaining_seconds': remaining, 'new_optimizer_steps': sum(s.get('new_optimizer_steps', 0) for s in statuses.values()),
               'planned_new_optimizer_steps': 120, 'backbone_optimizer_steps': 0, 'test_read': False,
               'new_beam_inference': False, 'automatic_retry': False, 'automatic_internalization': False, **extra}
        write(out / 'status.json', row); report(config, row)
    try:
        save('STARTING', reason='Frozen-cache parity and runtime profiles, followed by four B1 continuations')
        for task in tasks:
            waiting = time.monotonic()
            while not STOPPING:
                if time.monotonic() - waiting > config['resource_wait_seconds']:
                    task['state'] = 'STOPPED'; save('STOPPED', reason='Resource wait timeout'); return
                free = int(subprocess.check_output(['nvidia-smi', '--id='+config['gpu_uuid'], '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=20).strip())
                if free >= config['minimum_free_mib']:
                    break
                save('WAITING_FOR_RESOURCE', current_task=task['id'], free_mib=free, required_mib=config['minimum_free_mib']); time.sleep(30)
            if STOPPING:
                save('STOPPED', reason='User stop before task'); return
            if task['mode'] == 'run':
                assert all(read(out / d / 'profile/profile.json').get('state') == 'PASSED' for d in config['execution_order'])
            child_out = out / task['dataset'] / ('profile' if task['mode'] == 'profile' else f'seed{task["seed"]}')
            assert not child_out.exists(), ('existing_child_attempt', str(child_out))
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=config['gpu_uuid'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            command = [sys.executable, '-u', '-m', 'experiment.phase20.gacr_b1_multiseed', '--config', args.config,
                       '--dataset', task['dataset'], '--seed', str(task['seed']), '--mode', task['mode']]
            task['state'] = 'RUNNING'
            with (out / f'{task["id"]}.log').open('x') as handle:
                CHILD = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                launched, termination = time.monotonic(), None
                limit = config['profile_budget_seconds'] if task['mode'] == 'profile' else config['task_budget_seconds']
                while CHILD.poll() is None:
                    detail = read(child_out / 'status.json'); task['stage'] = detail.get('stage', 'initializing')
                    save('STOPPING' if STOPPING else 'RUNNING', current_task=task['id'], child_pid=CHILD.pid, child_alive=True,
                         child_status=detail, log=str(handle.name), child_budget_seconds=limit)
                    if (STOPPING or time.monotonic() - launched > limit + 60) and termination is None:
                        os.killpg(CHILD.pid, signal.SIGTERM); termination = time.monotonic()
                    if termination is not None and time.monotonic() - termination > 60:
                        os.killpg(CHILD.pid, signal.SIGKILL)
                    try:
                        CHILD.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            detail = read(child_out / 'status.json')
            success = CHILD.returncode == 0 and detail.get('state') == ('PASSED' if task['mode'] == 'profile' else 'COMPLETED')
            task['state'] = 'COMPLETED' if success else 'STOPPED' if STOPPING or detail.get('state') == 'STOPPED' else 'FAILED'
            task['exit_code'] = CHILD.returncode
            if not success:
                save(task['state'], current_task=task['id'], child_status=detail, reason='Owned task stopped/failed; no retry or subsequent task'); return
            save('RUNNING', reason='Task completed; continuing fixed queue')
        result = aggregate(config); write(out / 'summary.json', result)
        save('COMPLETED', reason='All four new B1 continuations completed; 2023 reused; no automatic next experiment')
    except BaseException as error:
        if CHILD is not None and CHILD.poll() is None:
            os.killpg(CHILD.pid, signal.SIGTERM)
            try:
                CHILD.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(CHILD.pid, signal.SIGKILL); CHILD.wait()
        save('FAILED', reason=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
