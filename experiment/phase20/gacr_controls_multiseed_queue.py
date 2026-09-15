"""Bounded background A1/A2 queue; its atomic status is the user entry point."""
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
    except (OSError, ValueError):
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


def refresh_report(config, status):
    text = '# Stage20 Beauty固定控制与双域残差多seed进展\n\n'
    text += '## Material Passport\n\n- Origin Skill：academic-research-suite / experiment-agent；Origin Mode：run / validate。\n'
    text += f'- 更新时间：{status["updated_at"]}；队列状态：**{status["state"]}**。\n'
    text += '- 用户2026-09-12授权继续；本轮A1/A2复用预测和冻结权重，零新训练、零新beam、不读test。\n\n'
    text += f'统一状态：`{config["output"]}/status.json`；每30秒更新。\n\n'
    text += '| 任务 | 状态 | 步骤 |\n|---|---|---|\n'
    for task in status['tasks']:
        text += f'| {task["id"]} | {task["state"]} | {task.get("stage", "")} |\n'
    detail = status.get('child_status', {})
    if detail:
        text += f'\n当前：{detail.get("dataset", "")} / {detail.get("phase", "")} / {detail.get("stage", "")}。\n'
        if detail.get('progress'):
            p = detail['progress']
            text += f'进度：{p["completed"]:,}/{p["total"]:,}人；此步骤估计剩余{detail.get("stage_eta_seconds")}秒。\n'
    if status.get('reason'):
        text += '\n队列说明：' + status['reason'] + '\n'
    for domain in config['execution_order']:
        out = ROOT / config['output'] / domain
        controls = read(out / 'controls_seed2023/summary.json')
        summary = read(out / 'summary.json')
        text += f'\n## {domain}\n\n'
        if controls:
            text += '| seed2023方法 | N10 | H10 | H50 |\n|---|---:|---:|---:|\n'
            for name, values in controls['groups']['all_validation'].items():
                text += f'| {name} | {values["ndcg@10"]:.6f} | {values["hit@10"]:.6f} | {values["hit@50"]:.6f} |\n'
        if summary:
            text += '\n| 残差seed | 方案 | N10 | H10 | H50 |\n|---|---|---:|---:|---:|\n'
            for seed, methods in summary['seed_metrics'].items():
                for name, m in methods.items():
                    text += f'| {seed} | {name} | {m["ndcg@10"]:.6f} | {m["hit@10"]:.6f} | {m["hit@50"]:.6f} |\n'
            text += '\n三seed均值及范围、全部分组和配对统计见本域summary及各子目录。\n'
        elif not controls:
            text += '尚未完成正式结果，smoke不作为提点证据。\n'
        text += f'\n本域状态：`{config["output"]}/{domain}/status.json`。\n'
    text += '\n## 协议与后续\n\n'
    text += 'Beauty q1=6、Toys q1=5；lambda1/beta0.5/gamma1。CPU4线程，GPU仅用于129参数残差的原FP32前向。'
    text += '先精确重建seed2023原候选、分数、六臂rank；Toys还核对已有九臂。每个控制/seed阶段30分钟硬预算，失败保留attempt且不自动重试。\n\n'
    text += 'bootstrap为3000次、seed20260912，仅描述反复开发validation的用户不确定性。三个seed仅改变残差权重，C1与item-head固定；不选最好seed、不把重复用户当独立样本。\n\n'
    text += '本队列完成后写出是否支持进一步方法训练的判据；B1/B2需要依据A结果固定候选版本和训练资源，当前队列不自动启动B/C。'
    text += '完整方法计划见[方法推进计划](../../plan/第二十阶段/GACR_PCRF双域提点后的方法推进计划.md)。\n'
    (ROOT / config['report']).write_text(text)


def decision(summaries):
    # A prospective descriptive resource-ranking rule, not a significance gate.
    result = {'rule': 'Both domains: three-seed mean N10 and H10 above original PCRF, '
              'at least two positive N10 seeds, and worst N10 no lower than -0.5% relative to PCRF.'}
    for method in ('full', 'fixed'):
        cells = {}
        for domain, summary in summaries.items():
            n, h = summary['aggregate'][method]['ndcg@10'], summary['aggregate'][method]['hit@10']
            baseline = summary['baseline']['ndcg@10']
            cells[domain] = {'mean_ndcg_positive': n['mean_delta_vs_original_pcrf'] > 0,
                             'mean_hit_positive': h['mean_delta_vs_original_pcrf'] > 0,
                             'positive_ndcg_seeds': n['positive_seed_count_vs_original_pcrf'],
                             'worst_relative_ndcg_delta': n['min'] / baseline - 1}
        result[method] = {'domains': cells, 'supports_next_training': all(
            c['mean_ndcg_positive'] and c['mean_hit_positive'] and c['positive_ndcg_seeds'] >= 2
            and c['worst_relative_ndcg_delta'] >= -.005 for c in cells.values())}
    result['next_step'] = ('PREPARE_MATCHED_FINAL_FUSION_TRAINING_ON_FIXED_CANDIDATES' if result['fixed']['supports_next_training']
                          else 'REVIEW_FULL_COMBINATION_AND_FREEZE_TRAINING_PROTOCOL' if result['full']['supports_next_training']
                          else 'REVIEW_SEED_VARIATION_BEFORE_NEW_TRAINING')
    result['automatic_training_started'] = False
    return result


def main():
    global CHILD
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiment/phase20/gacr_controls_multiseed_v1.json')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    out = ROOT / config['output']
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'status.json').exists():
        raise FileExistsError('Existing queue attempt: no restart or overwrite')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    started = time.monotonic()
    tasks = [{'id': d + '_smoke', 'dataset': d, 'mode': 'smoke', 'state': 'PENDING'} for d in config['execution_order']]
    tasks += [{'id': d + '_controls_and_seeds', 'dataset': d, 'mode': 'run', 'state': 'PENDING'} for d in config['execution_order']]

    def save(state, **extra):
        statuses = {d: read(out / d / 'status.json') for d in config['execution_order']}
        row = {'state': state, 'scope': 'A1_Beauty_controls_and_A2_two_domain_residual_seeds',
               'pid': os.getpid(), 'pid_start_ticks': Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
               'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'heartbeat_interval_seconds': 30,
               'elapsed_seconds': round(time.monotonic() - started, 1), 'tasks': tasks,
               'completed_tasks': sum(t['state'] == 'COMPLETED' for t in tasks), 'total_tasks': len(tasks),
               'completed_scientific_phases': sum(len(s.get('completed_phases', [])) for s in statuses.values()),
               'total_scientific_phases': 6, 'domain_status': statuses, 'gpu_uuid': config['gpu_uuid'],
               'automatic_retry': False, 'test_read': False, 'new_optimizer_steps': 0,
               'next_training_state': 'PENDING_A_RESULTS_AND_PROTOCOL', **extra}
        write(out / 'status.json', row)
        refresh_report(config, row)

    try:
        save('STARTING', reason='Queue started; engineering parity precedes formal results')
        for task in tasks:
            wait_start = time.monotonic()
            while not STOPPING:
                if time.monotonic() - wait_start > config['max_resource_wait_seconds']:
                    task['state'] = 'STOPPED'
                    save('STOPPED', reason='Resource wait budget reached; no automatic retry')
                    return
                free = int(subprocess.check_output(['nvidia-smi', '--id=' + config['gpu_uuid'],
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=20).strip())
                if free >= config['minimum_free_mib']:
                    break
                save('WAITING_FOR_RESOURCE', current_task=task['id'], free_mib=free,
                     required_mib=config['minimum_free_mib'])
                time.sleep(30)
            if STOPPING:
                save('STOPPED', reason='Stop requested before next task')
                return
            if task['mode'] == 'run':
                for domain in config['execution_order']:
                    assert read(out / f'{domain}_smoke_v1/smoke.json').get('state') == 'PASSED'
            directory = out / (task['dataset'] + '_smoke_v1' if task['mode'] == 'smoke' else task['dataset'])
            assert not directory.exists(), ('existing_child_output', str(directory))
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=config['gpu_uuid'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1')
            command = [sys.executable, '-u', '-m', 'experiment.phase20.gacr_controls_multiseed',
                       '--config', args.config, '--dataset', task['dataset'], '--mode', task['mode']]
            task['state'] = 'RUNNING'
            with (out / (task['id'] + '.log')).open('x') as log:
                CHILD = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                launched, termination = time.monotonic(), None
                limit = config['smoke_budget_seconds'] + 60 if task['mode'] == 'smoke' else 3 * config['phase_budget_seconds'] + 120
                while CHILD.poll() is None:
                    detail = read(directory / 'status.json')
                    task['stage'] = detail.get('phase', '') + '/' + detail.get('stage', '')
                    save('STOPPING' if STOPPING else 'RUNNING', current_task=task['id'],
                         child_pid=CHILD.pid, child_status=detail, log=str(log.name),
                         child_alive=True, child_budget_seconds=limit)
                    if (time.monotonic() - launched > limit or STOPPING) and termination is None:
                        save('STOPPING', reason='User stop or child hard budget; terminating owned child', child_pid=CHILD.pid)
                        os.killpg(CHILD.pid, signal.SIGTERM)
                        termination = time.monotonic()
                    if termination is not None and time.monotonic() - termination > 60:
                        os.killpg(CHILD.pid, signal.SIGKILL)
                    try:
                        CHILD.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            detail = read(directory / 'status.json')
            terminal = 'PASSED' if task['mode'] == 'smoke' else 'COMPLETED'
            success = CHILD.returncode == 0 and detail.get('state') == terminal
            task['state'] = 'COMPLETED' if success else 'STOPPED' if STOPPING or detail.get('state') == 'STOPPED' else 'FAILED'
            task['exit_code'] = CHILD.returncode
            if not success:
                save(task['state'], current_task=task['id'], child_status=detail,
                     reason='Task failed/stopped; partial artifacts retained, remaining tasks not launched')
                return
            save('RUNNING', current_task=task['id'], child_status=detail, child_exit_code=0,
                 reason='Task complete; next frozen task will start')
        summaries = {d: read(out / d / 'summary.json') for d in config['execution_order']}
        assert all(s.get('state') == 'COMPLETED' for s in summaries.values())
        result = {'state': 'COMPLETED', 'domains': summaries, 'decision': decision(summaries),
                  'verification_status': 'ANALYZED', 'new_optimizer_steps': 0, 'test_read': False}
        write(out / 'summary.json', result)
        save('COMPLETED', decision=result['decision'], reason='A1/A2 completed; matched new training is the next research step')
    except BaseException as error:
        if CHILD is not None and CHILD.poll() is None:
            os.killpg(CHILD.pid, signal.SIGTERM)
            try:
                CHILD.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(CHILD.pid, signal.SIGKILL)
                CHILD.wait()
        (out / 'queue_traceback.txt').write_text(traceback.format_exc())
        save('STOPPED' if STOPPING else 'FAILED', reason=repr(error))
        raise


if __name__ == '__main__':
    main()
