"""Bounded, observable background queue for a frozen three-arm decoder pilot."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from experiment.phase20.gacr_fusion_queue import read, write

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('CE_anchor', 'true_teacher', 'shuffled_teacher')
KEYS = [f'{m}@{k}' for m in ('ndcg', 'hit') for k in (5, 10, 20, 50)]
CHILD = None
STOPPING = False


def stop(signum, frame):
    global STOPPING
    STOPPING = True
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)


def child_path(out, task):
    return out / (f'{task["arm"]}/{task["mode"]}' if task['arm'] else task['mode'])


def report(config, status):
    text = '# Stage20 Toys完整列表内部蒸馏进展与结果\n\n'
    text += '## Material Passport\n\n- Origin Skill：academic-research-suite / experiment-agent；Origin Mode：run / validate。\n'
    text += f'- 更新时间：{status["updated_at"]}；状态：**{status["state"]}**；test未读取。\n\n'
    text += f'观察文件：`{config["output"]}/status.json`。30秒心跳，`child_status`含当前阶段、进度与阶段ETA，`projected_remaining_seconds`为profile外推剩余时间；共享GPU负载可能使估算变化。\n\n'
    text += '## 实验问题与固定协议\n\n'
    text += '检验已有效的B1+PCRF商品排序能否通过完整50候选列表蒸馏，进入C1解码器并改善重新生成的候选和排序。此次确实更新生成器；编码器、共享词嵌入、LM head、B1和CF head保持冻结。\n\n'
    text += 'Toys先做三个匹配臂：CE_anchor（CE+父C1列表KL保持）、true_teacher（再加真实B1+PCRF分布KL）、shuffled_teacher（加每个前缀固定打乱的教师分布）。三臂相同C1起点、数据、顺序、优化器和预算，不按中途验证结果筛选。\n\n'
    text += f'21,839个fit前缀/3,882个用户，12,399个目标在候选50内、9,440个在外；源缓存另256个calibration前缀不训练。每臂1个epoch、有效batch128、{config["optimizer_steps_per_arm"]}步，AdamW lr={config["learning_rate"]:g}，5% warmup，线性衰减，clip1；仅固定末步checkpoint。\n\n'
    text += 'CE覆盖全部fit前缀，父C1完整列表KL覆盖全部fit前缀；教师KL只用于候选内目标的前缀，按21839/12399归一。权重CE=1、父KL=1、教师KL=0.1，温度1。教师分布来自B1 seed2023的最终PCRF分数标准化后softmax，打乱保持概率集合和熵。候选不强行加入真值。\n\n'
    text += '完整商品分数为含EOS、不含BOS的token对数概率之和除以标签长度。全50候选统一标准化及KL；先计算50个分数的梯度，再每4条候选重放反传，CPU测试验证与完整图梯度一致。FP32骨干、FP64列表概率/损失，TF32和dropout关闭。父保持使用同接口实际teacher-forced分数缓存，记录其与native beam分数的数值差异。\n\n'
    text += '真实GPU profile检查原C1新生成候选一致性、分数、完整列表重放梯度、冻结参数与一次丢弃的工程更新，并测量显存和各阶段耗时；不产生正式训练步。通过后依次生成父保持缓存、训练三臂、对三臂各完整生成19,412名验证用户beam50并加固定PCRF、汇总匹配对照。\n\n'
    text += '## 进度与资源\n\n| 任务 | 状态 | 阶段 |\n|---|---|---|\n'
    for task in status['tasks']:
        text += f'| {task["id"]} | {task["state"]} | {task.get("stage", "")} |\n'
    text += f'\nGPU：`{config["gpu_uuid"]}`；正式解码器优化步：{status["new_optimizer_steps"]}/513。无自动重试；技术错误、超时或资源等待超时会停止队列，保存已有日志。\n'
    profile = read(ROOT / config['output'] / 'profile/profile.json')
    if profile:
        text += f'\n实测profile状态：{profile["state"]}；GPU峰值已分配 {profile["peak_allocated_mib"]:.0f} MiB，保留 {profile["peak_reserved_mib"]:.0f} MiB。\n'
        text += f'\n外推：父列表缓存 {profile["estimated_cache_seconds"]/3600:.2f}小时；三臂训练合计 {sum(profile["estimated_training_seconds"].values())/3600:.2f}小时；每臂完整验证 {profile["estimated_evaluation_seconds_per_arm"]/3600:.2f}小时；profile后队列约 {profile["estimated_queue_seconds_after_profile"]/3600:.2f}小时。这是8个历史长度分位样本外推，非完成承诺。\n'
    summaries = {a: read(ROOT / config['output'] / a / 'evaluate/summary.json') for a in ARMS}
    available = {a: s for a, s in summaries.items() if s.get('state') == 'COMPLETED'}
    if available:
        text += '\n## 全量验证八指标\n\n| 方法 | N5 | N10 | N20 | N50 | H5 | H10 | H20 | H50 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        reference = next(iter(available.values()))['groups']['all_validation']
        for name in ('original_gram', 'original_pcrf', 'c1', 'c1_pcrf', 'B1'):
            text += f'| {name} | ' + ' | '.join(f'{reference[name][k]:.6f}' for k in KEYS) + ' |\n'
        for arm, summary in available.items():
            for kind in ('raw', 'pcrf'):
                text += f'| {arm}/{kind} | ' + ' | '.join(f'{summary["groups"]["all_validation"][kind][k]:.6f}' for k in KEYS) + ' |\n'
        text += '\n| 臂 | 候选集合发生变化用户数 | 与C1平均重合商品数/50 |\n|---|---:|---:|\n'
        for arm, summary in available.items():
            text += f'| {arm} | {summary["fresh_candidate_sets_changed_users"]} | {summary["mean_overlap_items_vs_frozen_C1"]:.3f} |\n'
    result = read(ROOT / config['output'] / 'summary.json')
    if result.get('state') == 'COMPLETED':
        text += '\n## 真教师相对匹配对照\n\n| 比较 | ΔN10 | 用户配对95%CI | ΔH10 | 用户配对95%CI | H50净增命中用户 |\n|---|---:|---|---:|---|---:|\n'
        for name, groups in result['matched_arm_comparisons'].items():
            row = groups['all_validation']; n, h = row['ndcg@10'], row['hit@10']
            text += f'| {name} | {n["delta"]:+.6f} | [{n["ci95"][0]:+.6f},{n["ci95"][1]:+.6f}] | {h["delta"]:+.6f} | [{h["ci95"][0]:+.6f},{h["ci95"][1]:+.6f}] | {row["hit50_gained"]-row["hit50_lost"]:+d} |\n'
    text += '\n## 判断与边界\n\n结果尚未完成时不判断有效。需要同时看真教师与CE/打乱对照、当前冻结B1+PCRF强基线的比较；raw结果判断内部迁移程度，重新生成后的H50及低频H50判断候选覆盖。只改变候选、只优于弱父模型、或只靠验证选择最优超参数，都不足以支持达到当前提点目标。\n\n'
    text += '所有八指标、五分组、N10/H10的3000次用户配对bootstrap与H5/10/20/50 gained/lost保存在各臂summary和最终summary中。单次训练、重复使用验证集、未校正多重比较；用户区间不包含训练随机性和历次开发选择的不确定性。B1教师沿用已观察seed2023，不挑最佳初始化。\n\n'
    text += '此次训练仍限于既有fit用户/前缀，教师只覆盖12,399个候选内样本，未彻底消除旧内部迁移实验的分布差异。与旧PCPS的区别是完整实际候选分布、跨父前缀竞争、父保持和参数范围；旧PCPS已有部分完整序列损失，不能把“完整序列训练”本身声称为首次创新。打乱检验完整教师的商品对应关系，不能单独归因CF。\n\n'
    text += '[预登记方案](../../plan/第二十阶段/Toys完整候选列表内部蒸馏实验计划.md)；[B1多初始化结果](Stage20_B1缓存复用多初始化验证进展与结果.md)。\n'
    target = ROOT / config['report']; target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.tmp.md'); temp.write_text(text); temp.replace(target)


def main():
    global CHILD
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    args = parser.parse_args(); config = read(ROOT / args.config)
    out = ROOT / config['output']; out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'queue.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'status.json').exists(), 'Existing attempt; no overwrite/restart'
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
    tasks = [{'id': mode, 'mode': mode, 'arm': None, 'state': 'PENDING'} for mode in ('profile', 'cache')]
    tasks += [{'id': f'{a}_{m}', 'mode': m, 'arm': a, 'state': 'PENDING'} for m in ('train', 'evaluate') for a in ARMS]
    tasks += [{'id': 'aggregate', 'mode': 'aggregate', 'arm': None, 'state': 'PENDING'}]
    started = time.monotonic()

    def save(state, **extra):
        details = {t['id']: read(child_path(out, t) / 'status.json') for t in tasks}
        profile = read(out / 'profile/profile.json'); remaining = None
        if profile.get('state') == 'PASSED':
            remaining = 0.
            for task in tasks:
                if task['state'] not in ('PENDING', 'RUNNING') or task['mode'] == 'profile': continue
                estimate = {'cache': profile['estimated_cache_seconds'], 'train': profile['estimated_training_seconds'].get(task['arm'], 0),
                            'evaluate': profile['estimated_evaluation_seconds_per_arm'], 'aggregate': config['aggregate_allowance_seconds']}[task['mode']]
                remaining += max(0., estimate-details[task['id']].get('elapsed_seconds', 0))
        row = {'state': state, 'pid': os.getpid(), 'pid_start_ticks': Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'heartbeat_interval_seconds': 30,
            'elapsed_seconds': round(time.monotonic()-started, 1), 'tasks': tasks,
            'completed_tasks': sum(t['state'] == 'COMPLETED' for t in tasks), 'total_tasks': len(tasks),
            'task_status': details, 'gpu_uuid': config['gpu_uuid'],
            'projected_remaining_seconds': round(remaining, 1) if remaining is not None else None,
            'new_optimizer_steps': sum(s.get('new_optimizer_steps', 0) for s in details.values()),
            'planned_new_optimizer_steps': 3*config['optimizer_steps_per_arm'],
            'engineering_optimizer_steps_discarded': details['profile'].get('engineering_optimizer_steps', 0),
            'test_read': False, 'automatic_retry': False, 'automatic_next_experiment': False, **extra}
        write(out / 'status.json', row); report(config, row)

    try:
        save('STARTING', reason='Frozen profile -> parent list cache -> three matched decoder runs -> full validation -> paired summary')
        for task in tasks:
            waiting = time.monotonic()
            while task['mode'] != 'aggregate' and not STOPPING:
                if time.monotonic()-waiting > config['resource_wait_seconds']:
                    task['state'] = 'STOPPED'; save('STOPPED', reason='Resource wait timeout', current_task=task['id']); return
                free = int(subprocess.check_output(['nvidia-smi', '--id='+config['gpu_uuid'], '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=20).strip())
                if free >= config['minimum_free_mib']: break
                save('WAITING_FOR_RESOURCE', current_task=task['id'], free_mib=free, required_mib=config['minimum_free_mib']); time.sleep(30)
            if STOPPING:
                save('STOPPED', reason='User stop before task'); return
            if task['mode'] != 'profile': assert read(out / 'profile/profile.json').get('state') == 'PASSED'
            target = child_path(out, task); assert not target.exists(), ('existing_child_attempt', str(target))
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=config['gpu_uuid'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            command = [sys.executable, '-u', '-m', 'experiment.phase20.gacr_list_distill', '--config', args.config, '--mode', task['mode']]
            if task['arm']: command += ['--arm', task['arm']]
            task['state'] = 'RUNNING'
            with (out / f'{task["id"]}.log').open('x') as handle:
                CHILD = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                launched, termination = time.monotonic(), None
                limit = config[task['mode']+'_budget_seconds']
                while CHILD.poll() is None:
                    detail = read(target / 'status.json'); task['stage'] = detail.get('stage', 'initializing')
                    save('STOPPING' if STOPPING else 'RUNNING', current_task=task['id'], child_pid=CHILD.pid, child_alive=True,
                         child_status=detail, log=str(handle.name), child_budget_seconds=limit)
                    if (STOPPING or time.monotonic()-launched > limit+60) and termination is None:
                        os.killpg(CHILD.pid, signal.SIGTERM); termination = time.monotonic()
                    if termination is not None and time.monotonic()-termination > 60: os.killpg(CHILD.pid, signal.SIGKILL)
                    try: CHILD.wait(timeout=30)
                    except subprocess.TimeoutExpired: pass
            detail = read(target / 'status.json')
            success = CHILD.returncode == 0 and detail.get('state') == ('PASSED' if task['mode'] == 'profile' else 'COMPLETED')
            task['state'] = 'COMPLETED' if success else 'STOPPED' if STOPPING or detail.get('state') == 'STOPPED' else 'FAILED'
            task['exit_code'] = CHILD.returncode
            if not success:
                save(task['state'], current_task=task['id'], child_alive=False, child_status=detail, reason='Owned task stopped/failed; no retry or subsequent task'); return
            save('RUNNING', current_task=task['id'], child_alive=False, child_status=detail, reason='Task completed; continuing fixed queue')
        save('COMPLETED', current_task=None, child_alive=False, reason='Three arms fully evaluated and summarized; no automatic next experiment')
    except BaseException as error:
        if CHILD is not None and CHILD.poll() is None:
            os.killpg(CHILD.pid, signal.SIGTERM)
            try: CHILD.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(CHILD.pid, signal.SIGKILL); CHILD.wait()
        save('FAILED', child_alive=False, reason=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
