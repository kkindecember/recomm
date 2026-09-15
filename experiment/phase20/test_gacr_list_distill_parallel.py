"""Dependency, exclusivity and live-process handoff checks; no model/GPU runs."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from experiment.phase20 import gacr_list_distill_parallel as parallel


def tasks():
    rows = [dict(id='profile', mode='profile', arm=None, state='COMPLETED'),
            dict(id='cache', mode='cache', arm=None, state='RUNNING')]
    rows += [dict(id=a+'_'+m, mode=m, arm=a, state='PENDING')
             for m in ('train', 'evaluate') for a in parallel.ARMS]
    rows.append(dict(id='aggregate', mode='aggregate', arm=None, state='PENDING'))
    return rows


def test_cache_barrier_independent_arm_chains_and_single_aggregate():
    rows = tasks()
    assert parallel.ready_tasks(rows) == []
    rows[1]['state'] = 'COMPLETED'
    assert {t['id'] for t in parallel.ready_tasks(rows)} == {a+'_train' for a in parallel.ARMS}
    for row in rows[2:5]:
        row['state'] = 'RUNNING'
    assert parallel.ready_tasks(rows) == []
    rows[2]['state'] = 'COMPLETED'
    assert [t['id'] for t in parallel.ready_tasks(rows)] == ['CE_anchor_evaluate']
    for row in rows[2:8]:
        row['state'] = 'COMPLETED'
    assert [t['id'] for t in parallel.ready_tasks(rows)] == ['aggregate']
    rows[7]['state'] = 'FAILED'
    assert parallel.ready_tasks(rows) == []


def test_distinct_boards_threshold_fallback_and_sticky_assignment():
    boards = [dict(index=i, uuid=f'gpu{i}', name='NVIDIA RTX A6000', free=41000)
              for i in range(8)]
    bindings = {}
    for arm in parallel.ARMS:
        board = parallel.choose_gpu(arm, bindings, boards, 30000)
        bindings[arm] = board
    assert {a: b['index'] for a, b in bindings.items()} == parallel.PREFERRED
    assert len({b['uuid'] for b in bindings.values()}) == 3
    boards[4]['free'] = 29999
    assert parallel.choose_gpu('CE_anchor', bindings, boards, 30000) is None
    boards[4]['free'] = 30000
    assert parallel.choose_gpu('CE_anchor', bindings, boards, 30000)['index'] == 4
    boards[6]['free'] = 29999
    del bindings['shuffled_teacher']
    assert parallel.choose_gpu('shuffled_teacher', bindings, boards, 30000)['index'] == 0
    boards[0]['name'] = 'different GPU'
    assert parallel.choose_gpu('shuffled_teacher', bindings, boards, 30000)['index'] == 3


def test_resource_revision_cannot_change_scientific_settings():
    original = json.loads((parallel.ROOT / parallel.CONFIG).read_text())
    before = copy.deepcopy(original)
    changed = parallel.worker_config(original, dict(uuid='gpu4', index=4), 'parallel/CE.json')
    assert original == before
    assert {k for k in original if original[k] != changed[k]} == {'gpu_uuid', 'config_path', 'resource_revision'}


def test_reject_handoff_after_training_started():
    status = dict(state='RUNNING', current_task='cache', new_optimizer_steps=0, tasks=tasks())
    parallel.validate_handoff(status)
    status['tasks'][2]['state'] = 'RUNNING'
    with pytest.raises(AssertionError):
        parallel.validate_handoff(status)


def test_supervisor_pause_does_not_pause_independent_cache_and_pid_guard(tmp_path):
    # A disposable real parent/child pair emulates the existing queue topology.
    tick = tmp_path / 'tick'
    child_code = (
        'import time; from pathlib import Path\n'
        f'p=Path({str(tick)!r})\n'
        'while True:\n p.write_text(str(time.monotonic())); time.sleep(.02)\n'
    )
    parent_code = ('import subprocess,sys\n'
                   f'p=subprocess.Popen([sys.executable,"-c",{child_code!r}],start_new_session=True)\n'
                   'print(p.pid,flush=True)\np.wait()\n')
    parent = subprocess.Popen([sys.executable, '-u', '-c', parent_code], stdout=subprocess.PIPE, text=True)
    child_pid = int(parent.stdout.readline())
    child = parallel.process(child_pid)
    old = parallel.process(parent.pid)
    try:
        assert child['ppid'] == old['pid'] and child['pgid'] == child_pid
        bad = dict(old, start_ticks='-1')
        with pytest.raises(RuntimeError):
            parallel.checked_signal(bad, signal.SIGSTOP)
        parallel.checked_signal(child, signal.SIGSTOP)
        for _ in range(100):
            if parallel.process(child_pid)['state'] == 'T':
                break
            time.sleep(.01)
        assert parallel.process(child_pid)['state'] == 'T'
        assert parallel.same_process(child)
        # Retiring only the supervisor leaves the paused cache alive; then the
        # exact same cache process is resumed, without repeating its work.
        parallel.checked_signal(old, signal.SIGKILL)
        parent.wait(timeout=5)
        parallel.checked_signal(child, signal.SIGCONT)
        for _ in range(100):
            if tick.exists() and tick.read_text():
                break
            time.sleep(.01)
        before = tick.read_text()
        time.sleep(.1)
        assert tick.read_text() != before
        assert parallel.same_process(child)
    finally:
        if parallel.same_process(child):
            parallel.checked_signal(child, signal.SIGTERM, group=True)
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)


def test_preflight_error_preserves_original_cache(tmp_path, monkeypatch):
    obj = object.__new__(parallel.Controller)
    obj.owns_status, obj.cache_paused, obj.meta = False, False, tmp_path
    monkeypatch.setattr(parallel, 'checked_signal', lambda *args, **kwargs: pytest.fail('Must not signal on preflight failure'))
    obj.cleanup(RuntimeError('preflight failed'))
    assert parallel.read(tmp_path / 'error.json')['original_queue_preserved']


def test_complete_takeover_transfers_lock_and_keeps_cache_progress(tmp_path):
    out = tmp_path
    meta = out / 'parallel_v2'; meta.mkdir()
    (out / 'cache').mkdir()
    (out / 'cache/status.json').write_text(json.dumps(dict(state='RUNNING', elapsed_seconds=0)))
    tick = out / 'cache/tick'
    child_code = ('import time; from pathlib import Path\n'
                  f'p=Path({str(tick)!r})\n'
                  'while True:\n p.write_text(str(time.monotonic())); time.sleep(.02)\n')
    child_args = [sys.executable, '-u', '-c', child_code, '--config', parallel.CONFIG, '--mode', 'cache']
    parent_code = ('import subprocess,fcntl\n'
                   f'lock=open({str(out / "queue.lock")!r},"a");fcntl.flock(lock,fcntl.LOCK_EX)\n'
                   f'p=subprocess.Popen({child_args!r},start_new_session=True)\n'
                   'print(p.pid,flush=True);p.wait()\n')
    parent = subprocess.Popen([sys.executable, '-u', '-c', parent_code,
        'experiment.phase20.gacr_list_distill_queue', '--config', parallel.CONFIG],
        stdout=subprocess.PIPE, text=True)
    child_pid = int(parent.stdout.readline())
    child_identity = parallel.process(child_pid)
    old = parallel.process(parent.pid)
    before = dict(state='RUNNING', current_task='cache', new_optimizer_steps=0,
                  tasks=tasks(), pid=parent.pid, pid_start_ticks=old['start_ticks'], child_pid=child_pid)
    (out / 'status.json').write_text(json.dumps(before))
    obj = object.__new__(parallel.Controller)
    obj.config = dict(source_sha256={}, input_sha256={}, cache_budget_seconds=60)
    obj.out, obj.meta = out, meta
    obj.source_hash, obj.config_hash = 'fixture', 'fixture'
    obj.identity = parallel.process(os.getpid())
    obj.paused = obj.retired = obj.cache_paused = obj.owns_status = False
    obj.save = lambda *args, **kwargs: None
    try:
        obj.take_over()
        parent.wait(timeout=5)
        assert obj.owns_status and obj.retired and not obj.cache_paused
        assert parallel.same_process(child_identity)
        assert obj.receipt['cache_same_pid_resumed'] and not obj.receipt['cache_recomputed']
        assert obj.tasks == before['tasks']
        for _ in range(100):
            if tick.exists() and tick.read_text():
                break
            time.sleep(.01)
        earlier = tick.read_text()
        time.sleep(.1)
        assert tick.read_text() != earlier
        assert json.loads((meta / 'before_handoff.json').read_text()) == before
    finally:
        if parallel.same_process(child_identity):
            if parallel.process(child_pid)['state'] == 'T':
                parallel.checked_signal(child_identity, signal.SIGCONT)
            parallel.checked_signal(child_identity, signal.SIGTERM, group=True)
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
