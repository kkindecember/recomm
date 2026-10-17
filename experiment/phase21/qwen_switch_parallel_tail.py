"""Add a disjoint tail worker while the frozen serial recovery keeps running.

Only scheduling and checkpoint assembly change. The original generation,
score construction, smoke tests, manifests and final summarizer are reused.
"""
import argparse
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
from experiment.phase21 import qwen_switch_controls_recovery as R

Q = R.Q
ROOT, ORIGINAL, RECOVERY = Q.ROOT, Q.OUT, R.OUT
OUT = RECOVERY/'parallel_tail_20261017_v1'
JOB = 'Beauty_1'
MODULE = 'experiment.phase21.qwen_switch_parallel_tail'
PLAN = ROOT/'plan/第二十一阶段/21.59_Qwen4剩余用户并行收尾_20261017.md'
read, write, sha, stamp = Q.read, Q.write, Q.sha, Q.stamp


def rows(path):
    """A running writer can expose an incomplete final JSON line."""
    if not path.exists():
        return []
    data = path.read_bytes()
    return [json.loads(line) for line in data[:data.rfind(b'\n')+1].splitlines()]


def command(pid):
    path = Path(f'/proc/{pid}/cmdline')
    return path.read_bytes().replace(b'\0', b' ').decode() if path.exists() else ''


def parent(pid):
    return int(next(line.split()[1] for line in Path(f'/proc/{pid}/status').read_text().splitlines()
                    if line.startswith('PPid:')))


def validate(records, expected):
    assert [r['sample_index'] for r in records] == expected
    samples = read(ORIGINAL/'Beauty/samples.json')
    raw = np.load(ORIGINAL/'Beauty/raw_experts.npy', mmap_mode='r')
    loo = np.load(ORIGINAL/'Beauty/loo_text.npy', mmap_mode='r')
    for row in records:
        j = row['sample_index']; sample = samples[j]
        assert all(row[k] == sample[k] for k in ('user_id', 'target', 'history', 'evaluation_block'))
        for arm in Q.C.FRESH:
            p = row[arm]
            assert len(p['ids']) == len(set(p['ids'])) == len(p['scores']) == 50
            assert np.isfinite(p['scores']).all()
        for arm, value in Q.C.scores(raw[j], loo[j], sample['full_history']).items():
            assert Q.F.full_rank(value, sample['target']) == row['standalone_ranks'][arm]


def prepare():
    assert read(ORIGINAL/'status.json')['state'] == 'RECOVERING_SHARED_GPU_OOM'
    job = RECOVERY/'jobs'/JOB
    assert read(job/'prefix_replay.json')['state'] == 'PASSED'
    original = read(RECOVERY/'resume_protocol.json')
    saved = rows(job/'predictions.jsonl')
    missing = original['missing_indices']
    validate(saved, missing[:len(saved)])
    # Balance one warm worker with one new worker; allow six users of startup.
    tail_count = max(1, (len(missing)-len(saved)-6)//2)
    split = len(missing)-tail_count
    assert len(saved) < split < len(missing)
    current = read(job/'status.json')
    generator = current['pid']; worker = parent(generator); supervisor = parent(worker)
    assert 'qwen_switch_controls_recovery generate' in command(generator)
    assert 'qwen_switch_controls_recovery worker' in command(worker)
    assert 'qwen_switch_controls_recovery supervise' in command(supervisor)
    OUT.mkdir(exist_ok=False)
    (OUT/'jobs'/JOB).mkdir(parents=True)
    (OUT/'Beauty').mkdir()
    shutil.copy2(ORIGINAL/'config.json', OUT/'config.json')
    shutil.copy2(PLAN, OUT/'protocol.md')
    protocol = dict(assigned_indices=missing[split:], split_position=split,
        all_missing_indices=missing, serial_users_at_prepare=len(saved),
        original_recovery_users=len(missing), original_recovery_supervisor=supervisor,
        original_recovery_worker=worker, original_recovery_generator=generator,
        replay_indices=[1,45], same_frozen_scientific_configuration=True,
        never_stop_serial_before_complete_union=True, test_read=False)
    write(OUT/'tail_protocol.json', protocol)
    manifest = read(RECOVERY/'manifest.json')
    hashes = dict(manifest['sha256'])
    inputs = list(manifest['domain_inputs']['Beauty'])
    for name in ('samples.json','catalog.json','data_config.json','teacher_checks.json',
                 'references.json','raw_experts.npy','loo_text.npy'):
        path = OUT/'Beauty'/name
        path.symlink_to(ORIGINAL/'Beauty'/name)
        rel = str(path.relative_to(ROOT)); hashes[rel] = sha(path); inputs.append(rel)
    for path in (Path(__file__).resolve(), ROOT/'experiment/phase21/run_qwen_switch_parallel_tail.sh',
                 PLAN, OUT/'config.json', OUT/'tail_protocol.json'):
        rel = str(path.relative_to(ROOT)); hashes[rel] = sha(path); inputs.append(rel)
    for rel in inputs:
        assert sha(ROOT/rel) == hashes[rel], rel
    write(OUT/'manifest.json', dict(sha256=hashes, domain_inputs={'Beauty':inputs}))
    for path in (Path(__file__).resolve(), ROOT/'experiment/phase21/run_qwen_switch_parallel_tail.sh'):
        destination = OUT/'runtime_sources'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    initial = dict(state='PREPARED', users_completed=0, users_total=tail_count,
                   updated_at=stamp(), test_read=False)
    write(OUT/'status.json', initial); write(OUT/'jobs'/JOB/'status.json', initial)
    write(ORIGINAL/'parallel_tail_handoff.json', dict(current_status=str((OUT/'status.json').relative_to(ROOT)),
        tmux_session='s21_qwen_switch_parallel_tail_20261017_v1',
        serial_recovery_kept_running=True, updated_at=stamp(), test_read=False))
    print(json.dumps(dict(state='PREPARED', preserved_serial_users=len(saved),
        tail_users=tail_count, serial_prefix_needed=split)), flush=True)


class Generation(Q.Generation):
    decode = R.Generation.decode

    def __init__(self, *args, **kwargs):
        self.protocol = read(OUT/'tail_protocol.json')
        self.memory_wait_seconds = 0.
        super().__init__(*args, **kwargs)

    def status(self, state='RUNNING', **extra):
        assert self.lease.poll() is None
        total = len(self.protocol['assigned_indices'])
        rate = float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(self.jobfolder/'status.json', dict(state=state, stage=self.stage,
            job=JOB, domain='Beauty', pid=os.getpid(), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
            users_completed=self.users_completed, users_total=total,
            fresh_generations_completed=self.users_completed*6,
            median_seconds_per_user=rate,
            remaining_generation_seconds_estimate=(total-self.users_completed)*rate if rate else None,
            memory_wait_seconds=self.memory_wait_seconds, updated_at=stamp(), test_read=False, **extra))

    def run(self):
        prefix = {r['sample_index']:r for r in rows(RECOVERY/'original_failure'/JOB/'predictions.jsonl')}
        self.stage = 'cross_gpu_preserved_prediction_replay'; self.status()
        for j in self.protocol['replay_indices']:
            ctx = self.context(j)
            for arm, value in self.scores(j).items():
                if arm in Q.C.FRESH:
                    prediction = self.decode(ctx, self.index.conditional_logs(value), .3)
                    Q.B.Generation.equal(dict(ids=prediction['ids'], scores=prediction['scores'].tolist()), prefix[j][arm])
        write(self.jobfolder/'prefix_replay.json', dict(state='PASSED',
            users=len(self.protocol['replay_indices']), arms_per_user=6,
            all_ids_and_scores_match_original=True, original_predictions_kept=True))
        Q.Generation.run(self)
        validate(rows(self.jobfolder/'predictions.jsonl'), self.protocol['assigned_indices'])
        checks = read(self.jobfolder/'runtime_checks.json')
        checks.update(users=len(self.protocol['assigned_indices']),
            scientific_source_unchanged=True, cross_gpu_replay_passed=True,
            per_decode_headroom_guard=True, memory_wait_seconds=self.memory_wait_seconds)
        write(self.jobfolder/'runtime_checks.json', checks)
        self.status('COMPLETED')


def worker():
    Q.F.OUT, Q.F.CFG, Q.F.JOBS, Q.F.MODULE = OUT, Q.CFG, {JOB:Q.JOBS[JOB]}, MODULE
    return Q.F.worker(JOB)


def finish():
    protocol = read(OUT/'tail_protocol.json')
    serial = RECOVERY/'jobs'/JOB; tail = OUT/'jobs'/JOB
    assert read(tail/'runtime_checks.json')['state'] == 'PASSED'
    assert read(tail/'prefix_replay.json')['state'] == 'PASSED'
    assert read(tail/'lease_status.json')['state'] == 'released'
    head_rows = rows(serial/'predictions.jsonl'); tail_rows = rows(tail/'predictions.jsonl')
    missing = protocol['all_missing_indices']; split = protocol['split_position']
    validate(head_rows, missing[:len(head_rows)])
    validate(tail_rows, missing[split:])
    assert len(head_rows) >= split
    by_head = {r['sample_index']:r for r in head_rows}
    for row in tail_rows:
        if row['sample_index'] in by_head:
            for arm in Q.C.FRESH:
                Q.B.Generation.equal(row[arm], by_head[row['sample_index']][arm])
    # Preserve both original generators until complete coverage is proven.
    if read(ORIGINAL/'status.json')['state'] == 'COMPLETED':
        write(OUT/'status.json', dict(state='COMPLETED_BY_SERIAL', updated_at=stamp(), test_read=False))
        return
    supervisor = protocol['original_recovery_supervisor']
    assert 'qwen_switch_controls_recovery supervise' in command(supervisor)
    write(OUT/'coverage_before_administrative_stop.json', dict(state='PASSED',
        missing_users=len(missing), serial_prefix_users=len(head_rows), tail_users=len(tail_rows),
        complete_union=True, original_1804_predictions_kept=True, updated_at=stamp()))
    os.kill(supervisor, signal.SIGINT)
    began = time.monotonic()
    while command(supervisor) or command(protocol['original_recovery_worker']) or command(protocol['original_recovery_generator']):
        assert time.monotonic()-began < 50, 'Serial cleanup timeout; do not commit while writer is alive'
        time.sleep(1)
    assert read(serial/'lease_status.json')['state'] == 'released'
    head_rows = rows(serial/'predictions.jsonl')
    validate(head_rows, missing[:len(head_rows)])
    archive = OUT/'serial_checkpoint_before_merge'
    shutil.copytree(serial, archive)
    for name in ('status.json','failure.json'):
        path = RECOVERY/name
        if path.exists():
            shutil.copy2(path, OUT/('serial_supervisor_'+name))
    assert read(archive/'prefix_replay.json')['state'] == 'PASSED'
    prefix_data = (archive/'predictions.jsonl').read_bytes().splitlines(keepends=True)
    data = b''.join(prefix_data[:split])+(tail/'predictions.jsonl').read_bytes()
    combined = [json.loads(line) for line in data.splitlines()]
    validate(combined, missing)
    temporary = serial/'parallel_merged.jsonl.tmp'; temporary.write_bytes(data)
    temporary.replace(serial/'predictions.jsonl')
    for folder in (serial, RECOVERY):
        failure = folder/'failure.json'
        if failure.exists():
            failure.replace(folder/'interruption_for_parallel_merge.json')
    checks = dict(state='PASSED', users=len(missing), all_six_fresh_arms=True,
        gram_digest_unchanged=None, targets_only_for_metrics=True, history_filter=False,
        training_steps=0, test_read=False, scientific_source_unchanged=True,
        preserved_prefix_replay_passed=True, per_decode_headroom_guard=True,
        parallel_tail=True, serial_checkpoint=str(archive.relative_to(ROOT)),
        tail_runtime_checks=str((tail/'runtime_checks.json').relative_to(ROOT)))
    # An administratively stopped prefix has no end-of-loop digest receipt.
    # Preserve that limitation explicitly instead of borrowing the tail's check.
    checks['tail_gram_digest_unchanged'] = read(tail/'runtime_checks.json')['gram_digest_unchanged']
    checks['serial_end_digest'] = 'not collected because the inference-only serial generator was administratively stopped'
    checks['serial_prefix_origin'] = 'same continuously running frozen inference-only generator; original replay and smoke PASSED'
    write(serial/'runtime_checks.json', checks)
    write(serial/'status.json', dict(state='COMPLETED', users_completed=len(missing),
        users_total=len(missing), stage='parallel_union_validated', updated_at=stamp(),
        median_seconds_per_user=None, remaining_generation_seconds_estimate=0, test_read=False))
    (serial/'successful_worker.log').write_text(
        'Administrative parallel completion; individual logs and receipts retained.\n'+
        f'Serial checkpoint: {archive}\nTail worker: {tail}\n')
    write(OUT/'merge_receipt.json', dict(state='PASSED', complete_users=len(combined),
        original_seven_shards_unchanged=True, original_12_prefix_unchanged=True,
        serial_prefix_users_used=split, tail_users_used=len(tail_rows),
        serial_complete_rows_preserved_in_archive=len(head_rows),
        both_leases_released=True, scientific_sources_unchanged=True, test_read=False))
    R.commit_and_summarize()
    write(OUT/'status.json', dict(state='COMPLETED', original_experiment_complete=True,
        users_completed=len(tail_rows), users_total=len(tail_rows), updated_at=stamp(), test_read=False))


def supervise():
    protocol = read(OUT/'tail_protocol.json'); task = None; attempt = 0
    try:
        with (OUT/'memory_trace.jsonl').open('a') as trace:
            while True:
                root = read(ORIGINAL/'status.json')
                if root['state'] == 'COMPLETED':
                    write(OUT/'status.json', dict(state='COMPLETED_BY_SERIAL', updated_at=stamp(), test_read=False))
                    return
                if task is None:
                    assert 'qwen_switch_controls_recovery supervise' in command(protocol['original_recovery_supervisor'])
                    eligible = sorted((g for g in Q.B.S.OLD.gpu_inventory() if g['free_mib'] >= Q.CFG['minimum_free_mib']),
                                      key=lambda g:(-g['free_mib'],g['utilization'],g['index']))
                    if eligible:
                        gpu = eligible[0]; attempt += 1
                        log_path = OUT/'jobs'/JOB/f'worker_attempt{attempt}.log'; log = log_path.open('x')
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='2',
                            MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
                        process = subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--job',JOB],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        task = dict(process=process, log=log, log_path=log_path, gpu=gpu)
                        write(OUT/'launch.json', dict(gpu=gpu, pid=process.pid, updated_at=stamp(),
                            admission_attempt=attempt, project_gpu_task_cap=None))
                live = read(OUT/'jobs'/JOB/'status.json')
                head = rows(RECOVERY/'jobs'/JOB/'predictions.jsonl')
                tail = rows(OUT/'jobs'/JOB/'predictions.jsonl')
                union = len({r['sample_index'] for r in head+tail})
                write(OUT/'status.json', dict(state='RUNNING' if task else 'WAITING_FOR_GPU_MEMORY',
                    serial_completed=len(head), tail_completed=len(tail), recovery_union_completed=union,
                    recovery_users_total=len(protocol['all_missing_indices']),
                    total_users_completed=1804+union, users_total=2048, tail_job_status=live,
                    updated_at=stamp(), test_read=False))
                if task and task['process'].poll() is not None:
                    code = task['process'].returncode
                    if code == 75 and live['state'] == 'ADMISSION_DEFERRED':
                        task['log'].close(); task = None
                        continue
                    assert code == 0, 'Tail worker failed; serial recovery left running, no scientific retry'
                    if len(head) >= protocol['split_position']:
                        finish(); return
                trace.write(json.dumps(dict(updated_at=stamp(), serial_completed=len(head),
                    tail_completed=len(tail), union_completed=union, tail_stage=live.get('stage'),
                    gpu_inventory=Q.B.S.OLD.gpu_inventory()))+'\n'); trace.flush()
                time.sleep(5)
    finally:
        if task:
            Q.B.stop(task['process']); task['log'].close()


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('prepare','supervise','worker','generate'))
    p.add_argument('--job', choices=(JOB,)); p.add_argument('--lease-pid', type=int)
    args = p.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('Parallel tail worker interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGALRM, interrupted)
    try:
        if args.mode == 'prepare': prepare()
        elif args.mode == 'supervise': supervise()
        elif args.mode == 'worker': return worker()
        else:
            assigned = read(OUT/'tail_protocol.json')['assigned_indices']
            Q.OUT, Q.A.OUT, Q.A.JOBS = OUT, OUT, Q.JOBS
            Q.indices = lambda shard:assigned
            Generation(JOB, args.lease_pid).run()
        return 0
    except BaseException as exc:
        if OUT.exists():
            folder = OUT/'jobs'/JOB if args.mode in ('worker','generate') else OUT
            value = dict(state='FAILED', error=repr(exc), traceback=traceback.format_exc(),
                         serial_recovery_not_automatically_stopped=True, updated_at=stamp(), test_read=False)
            write(folder/'failure.json', value); write(folder/'status.json', value)
        raise


if __name__ == '__main__':
    sys.exit(main())
