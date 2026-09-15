"""Observable, bounded B1/B2 queue; profiles first, then both domains regardless of accuracy."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
CHILD = None
STOPPING = False


def read(path):
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def write(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def stop(signum, frame):
    global STOPPING
    STOPPING = True
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)


def decision(summaries):
    outcomes = {}
    for domain, summary in summaries.items():
        methods = summary['groups']['all_validation']
        outcomes[domain] = {arm: {f'delta_{metric}_vs_{baseline}': methods[arm][metric] - methods[baseline][metric]
            for baseline in ('B0', 'original_pcrf', 'gacr_v6_pcrf') for metric in ('ndcg@10', 'hit@10')}
            for arm in ('B1', 'B2')}
        outcomes[domain]['B2_above_B1_N10_H10'] = all(methods['B2'][k] > methods['B1'][k] for k in ('ndcg@10', 'hit@10'))
        outcomes[domain]['B2_above_B0_N10_H10'] = all(methods['B2'][k] > methods['B0'][k] for k in ('ndcg@10', 'hit@10'))
    supported = all(d['B2_above_B1_N10_H10'] and d['B2_above_B0_N10_H10'] for d in outcomes.values())
    return {'domains': outcomes, 'both_domain_fusion_training_point_estimates_supported': supported,
            'scope': 'Descriptive matched single-initialization comparison; uncertainty and all metrics remain in summaries.',
            'automatic_next_experiment': False}


def report(config, status):
    text = '# Stage20 固定候选最终融合训练进展与结果\n\n'
    text += '## Material Passport\n\n- Origin Skill：academic-research-suite / experiment-agent；Origin Mode：run / validate。\n'
    text += f'- 更新时间：{status["updated_at"]}；状态：**{status["state"]}**。\n'
    text += '- 用户已授权继续B1/B2；C1生成器和两类head固定，只训练129参数残差；不读test，不自动启动内部化。\n\n'
    text += f'统一观察：`{config["output"]}/status.json`，监督器30秒心跳；失败/超时保留产物，不自动重试。\n\n'
    text += '| 任务 | 状态 | 步骤 |\n|---|---|---|\n'
    for task in status['tasks']:
        text += f'| {task["id"]} | {task["state"]} | {task.get("stage", "")} |\n'
    detail = status.get('child_status', {})
    if detail:
        text += f'\n当前：{detail.get("dataset", "")} / {detail.get("stage", "")}；新优化步数{detail.get("new_optimizer_steps", 0)}。\n'
        if detail.get('progress'):
            p = detail['progress']
            text += f'当前步骤：{p["completed"]:,}/{p["total"]:,}；此步骤预计剩余{detail.get("stage_eta_seconds")}秒。\n'
    if status.get('reason'):
        text += '\n说明：' + status['reason'] + '\n'
    text += '\n## 固定设计\n\n'
    text += 'B0为原seed2023 fixed组合；B1/B2都从相同seed2023权重出发，在相同train-only fit前缀、固定C1生成50项上训练。B1优化标准化残差分数；B2优化标准化PCRF最终分数。全49负例pairwise softplus，covered head/tail各半权重，AdamW lr0.01/weight decay0.01，30个全量梯度更新，clip10。固定最后一步，两臂全量validation一起公开。\n\n'
    text += 'Toys fit21,839条、Beauty fit26,486条，另各256条train-only calibration；候选外正例不插入、不参与pairwise拟合，仍计覆盖及全量评估。q1沿用5/6，lambda1/beta0.5/gamma1；排序及tail_mass停止梯度，分数标准化可微。训练FP32小MLP、FP64融合；评估保留原逐用户完整并集残差前向形状和PCRF稳定排序。\n\n'
    text += '生成器权重未更新；本次新beam只生成训练前缀缓存，validation复用既有候选/特征/CF。两臂共享一次缓存。训练记录与validation只通过不同字段进入不同步骤，验证目标不参与训练。\n'
    text += '\n## 工程检查与资源\n\n'
    for d in config['execution_order']:
        profile = read(ROOT / config['output'] / f'{d}_profile_v1/profile.json')
        if profile:
            text += f'- {d}：{profile.get("state")}；按历史长度分位取{len(profile.get("candidate_seconds", []))}条训练前缀计时，候选缓存粗估{profile.get("candidate_estimated_seconds", 0)/3600:.2f}小时；峰值allocated {profile.get("peak_allocated_mib", 0):.0f} MiB。\n'
    text += f'\n单个profile最多{config["profile_budget_seconds"]/60:.0f}分钟；每域候选缓存最多{config["candidate_budget_seconds"]/3600:.0f}小时、包含训练评估总上限{config["domain_budget_seconds"]/3600:.0f}小时；每任务等资源最多{config["resource_wait_seconds"]/3600:.0f}小时。GPU UUID `{config["gpu_uuid"]}`，空闲至少{config["minimum_free_mib"]}MiB，allocator上限{config["cuda_memory_fraction"]:.0%}，CPU4线程。计时只是共享GPU当前负载下的投影，status中的步骤ETA随进度更新。\n'
    for d in config['execution_order']:
        summary = read(ROOT / config['output'] / d / 'summary.json')
        if not summary:
            continue
        text += f'\n## {d}全量validation结果\n\n'
        text += '| 方法 | N5 | N10 | N20 | N50 | H5 | H10 | H20 | H50 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        keys = [f'{m}@{k}' for m in ('ndcg', 'hit') for k in (5, 10, 20, 50)]
        for name, values in summary['groups']['all_validation'].items():
            text += '| ' + name + ' | ' + ' | '.join(f'{values[k]:.6f}' for k in keys) + ' |\n'
        text += '\n| 比较 | ΔN10 | 用户配对95%区间 | ΔH10 | Top10新增/丢失 |\n|---|---:|---|---:|---:|\n'
        for name, groups in summary['comparisons'].items():
            v = groups['all_validation']; n = v['ndcg@10']
            text += f'| {name} | {n["delta"]:+.6f} | [{n["ci95"][0]:+.6f}, {n["ci95"][1]:+.6f}] | {v["hit@10"]["delta"]:+.6f} | {v["hit10_gained"]}/{v["hit10_lost"]} |\n'
        text += f'\n[本域完整分组、训练轨迹和summary](../../{config["output"]}/{d}/summary.json)。\n'
    text += '\n## 解释边界\n\nB2超过B0且超过B1，才支持本预算下将融合纳入训练带来额外提点；两臂都提高但接近，应保留收益而不独归融合目标。低于B0则没有新增提点。H50由固定C1集合决定，不会由本轮重排恢复。30步如未收敛，记预算限制，不能判整个思路失效。配对bootstrap3000次、seed20260912，仅为反复开发validation的用户不确定性，未覆盖独立训练seed或多重比较。完整两域结果公开后再做研究判断。\n\n'
    text += '[完整方法计划](../../plan/第二十阶段/GACR_PCRF双域提点后的方法推进计划.md)。\n'
    write_target = ROOT / config['report']
    temp = write_target.with_suffix('.tmp.md')
    temp.write_text(text); temp.replace(write_target)


def main():
    global CHILD
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    out = ROOT / config['output']
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'status.json').exists(), 'Existing attempt; no overwrite/restart'
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    started = time.monotonic()
    tasks = [{'id': f'{d}_{mode}', 'dataset': d, 'mode': mode, 'state': 'PENDING'}
             for mode in ('profile', 'run') for d in config['execution_order']]
    def save(state, **extra):
        statuses = {d: read(out / d / 'status.json') for d in config['execution_order']}
        row = {'state': state, 'scope': 'B1_B2_matched_final_fusion_training', 'pid': os.getpid(),
               'pid_start_ticks': Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
               'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'heartbeat_interval_seconds': 30,
               'elapsed_seconds': round(time.monotonic() - started, 1), 'tasks': tasks,
               'completed_tasks': sum(t['state'] == 'COMPLETED' for t in tasks), 'total_tasks': len(tasks),
               'domain_status': statuses, 'gpu_uuid': config['gpu_uuid'],
               'new_optimizer_steps': sum(s.get('new_optimizer_steps', 0) for s in statuses.values()),
               'backbone_optimizer_steps': 0, 'test_read': False, 'automatic_retry': False,
               'automatic_internalization': False, **extra}
        write(out / 'status.json', row)
        report(config, row)
    try:
        save('STARTING', reason='Both-domain engineering profiles precede shared-cache matched training')
        for task in tasks:
            waiting = time.monotonic()
            while not STOPPING:
                if time.monotonic() - waiting > config['resource_wait_seconds']:
                    task['state'] = 'STOPPED'
                    save('STOPPED', reason='Resource wait budget reached; no automatic retry')
                    return
                free = int(subprocess.check_output(['nvidia-smi', '--id=' + config['gpu_uuid'],
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=20).strip())
                if free >= config['minimum_free_mib']:
                    break
                save('WAITING_FOR_RESOURCE', current_task=task['id'], free_mib=free, required_mib=config['minimum_free_mib'])
                time.sleep(30)
            if STOPPING:
                save('STOPPED', reason='User stop before next task'); return
            if task['mode'] == 'run':
                assert all(read(out / f'{d}_profile_v1/profile.json').get('state') == 'PASSED' for d in config['execution_order'])
            child_out = out / (f'{task["dataset"]}_profile_v1' if task['mode'] == 'profile' else task['dataset'])
            assert not child_out.exists(), ('existing_child_attempt', str(child_out))
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=config['gpu_uuid'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            command = [sys.executable, '-u', '-m', 'experiment.phase20.gacr_fusion_train', '--config', args.config,
                       '--dataset', task['dataset'], '--mode', task['mode']]
            task['state'] = 'RUNNING'
            with (out / f'{task["id"]}.log').open('x') as handle:
                CHILD = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                launched, termination = time.monotonic(), None
                limit = config['profile_budget_seconds'] if task['mode'] == 'profile' else config['domain_budget_seconds']
                while CHILD.poll() is None:
                    detail = read(child_out / 'status.json')
                    task['stage'] = detail.get('stage', 'initializing')
                    save('STOPPING' if STOPPING else 'RUNNING', current_task=task['id'], child_pid=CHILD.pid,
                         child_alive=True, child_status=detail, log=str(handle.name), child_budget_seconds=limit)
                    if (STOPPING or time.monotonic() - launched > limit + 60) and termination is None:
                        os.killpg(CHILD.pid, signal.SIGTERM); termination = time.monotonic()
                    if termination is not None and time.monotonic() - termination > 60:
                        os.killpg(CHILD.pid, signal.SIGKILL)
                    try:
                        CHILD.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            detail = read(child_out / 'status.json')
            terminal = 'PASSED' if task['mode'] == 'profile' else 'COMPLETED'
            success = CHILD.returncode == 0 and detail.get('state') == terminal
            task['state'] = 'COMPLETED' if success else 'STOPPED' if STOPPING or detail.get('state') == 'STOPPED' else 'FAILED'
            task['exit_code'] = CHILD.returncode
            if not success:
                save(task['state'], current_task=task['id'], child_status=detail,
                     reason='Owned task failed/stopped; preserve outputs and do not launch remaining tasks')
                return
            save('RUNNING', reason='Task completed; continuing the fixed queue')
        summaries = {d: read(out / d / 'summary.json') for d in config['execution_order']}
        result = {'state': 'COMPLETED', 'domains': summaries, 'decision': decision(summaries),
                  'verification_status': 'ANALYZED', 'new_optimizer_steps': sum(s['new_optimizer_steps'] for s in summaries.values()),
                  'backbone_optimizer_steps': 0, 'test_read': False}
        write(out / 'summary.json', result)
        save('COMPLETED', decision=result['decision'], reason='Both-domain matched training/evaluation completed; no automatic next experiment')
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
