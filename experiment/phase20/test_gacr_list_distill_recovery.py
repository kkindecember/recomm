"""Resource regression and real subprocess scheduling tests without GPU/model work."""
import copy
import json
from pathlib import Path
import sys

import pytest

from experiment.phase20 import gacr_list_distill_recovery as recovery


def test_low_post_cuda_memory_waits_then_admits_without_restarting():
    now = [100.]
    observed = []
    samples = iter([29970., 29999., 30100.])
    free = recovery.wait_for_memory(lambda: next(samples), 30000, 140,
        lambda *values: observed.append(values), clock=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
    assert free == 30100 and now[0] == 120
    assert [row[1] for row in observed] == [False, False, True]
    assert [row[2] for row in observed] == [40., 30., 20.]


def test_resource_timeout_is_cumulative_and_cannot_admit_after_deadline():
    now = [0.]
    observed = []
    with pytest.raises(TimeoutError, match='resource wait'):
        recovery.wait_for_memory(lambda: 29999, 30000, 15, lambda *x: observed.append(x),
            clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0]+dt))
    assert now[0] == 15 and len(observed) == 2
    with pytest.raises(TimeoutError):
        recovery.wait_for_memory(lambda: pytest.fail('must not query after deadline'),
                                 30000, 15, lambda *x: None, clock=lambda: 15)


def test_one_mib_headroom_is_rejected_and_stability_resets_on_dip():
    def boards(free):
        return [dict(index=4, uuid='gpu4', name='NVIDIA RTX A6000', free=free)]
    eligible, previous = recovery.stable_boards(boards(30001), set(), 31024)
    assert eligible == [] and previous == set()
    eligible, previous = recovery.stable_boards(boards(32000), previous, 31024)
    assert eligible == []
    eligible, previous = recovery.stable_boards(boards(30999), previous, 31024)
    assert eligible == [] and previous == set()
    eligible, previous = recovery.stable_boards(boards(32000), previous, 31024)
    assert eligible == []
    eligible, previous = recovery.stable_boards(boards(32000), previous, 31024)
    assert recovery.choose_gpu('CE_anchor', {}, eligible, 31024)['index'] == 4


def test_new_config_preserves_all_scientific_fields_and_old_hashes():
    original = recovery.read(recovery.ROOT / recovery.ORIGINAL)
    before = copy.deepcopy(original)
    changed = recovery.recovery_config(original)
    assert original == before
    assert {k for k in original if original[k] != changed[k]} <= {
        'id', 'config_path', 'output', 'resource_revision', 'source_sha256'}
    assert changed['input_sha256'] == original['input_sha256']
    for path, value in original['source_sha256'].items():
        assert changed['source_sha256'][path] == value


class FakeController(recovery.Controller):
    """Use real child processes/status files with no dependency on CUDA."""
    fail_arm = None

    def boards(self):
        return [dict(index=i, uuid=f'gpu{i}', name='NVIDIA RTX A6000', free=40000) for i in (1, 4, 6)]

    def render(self, value):
        pass

    def command(self, task, config_path, deadline):
        target = recovery.child_path(self.out, task)
        code = ('import json,os,time; from pathlib import Path\n'
                f'p=Path({str(target)!r}); p.mkdir(parents=True,exist_ok=False)\n'
                's={"state":"RUNNING","stage":"fake_compute","new_optimizer_steps":0}\n'
                '(p/"status.json").write_text(json.dumps(s))\n')
        if self.fail_arm and task['arm'] != self.fail_arm:
            code += 'time.sleep(60)\n'
        elif task['arm'] == self.fail_arm and self.fail_arm:
            code += ('time.sleep(.1); s.update(state="FAILED", error="injected compute failure")\n'
                     '(p/"status.json").write_text(json.dumps(s)); raise SystemExit(1)\n')
        else:
            code += ('time.sleep(.08); s.update(state="COMPLETED", stage="finished", '
                     f'new_optimizer_steps={171 if task["mode"] == "train" else 0})\n'
                     '(p/"status.json").write_text(json.dumps(s))\n')
        return [sys.executable, '-c', code]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    (out / 'profile').mkdir(parents=True)
    (out / 'cache').mkdir()
    for name in ('profile', 'cache'):
        recovery.write(out / name / 'status.json', {'state': 'COMPLETED', 'new_optimizer_steps': 0})
    recovery.write(out / 'profile/profile.json', {
        'estimated_training_seconds': {a: 100 for a in recovery.ARMS}, 'estimated_evaluation_seconds_per_arm': 50})
    recovery.write(tmp_path / 'config.json', {
        'output': 'out', 'minimum_free_mib': 30000, 'resource_wait_seconds': 30,
        'train_budget_seconds': 10, 'evaluate_budget_seconds': 10, 'aggregate_budget_seconds': 10,
        'resource_revision': {'launch_margin_mib': 1024, 'previous_output': 'old'},
    })
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(recovery, 'verify_sources', lambda _: None)
    return out


def test_full_three_gpu_chains_aggregate_and_config_reuse(prepared):
    controller = FakeController('config.json', poll_seconds=.02)
    try:
        controller.execute()
        status = recovery.read(prepared / 'status.json')
        assert status['state'] == 'COMPLETED' and status['new_optimizer_steps'] == 513
        assert status['active_children'] == {} and status['projected_remaining_seconds'] == 0
        assert len({b['uuid'] for b in status['arm_gpu_bindings'].values()}) == 3
        assert len(list((prepared / 'workers').glob('*.launch.json'))) == 7
        for arm in recovery.ARMS:
            train = recovery.read(prepared / 'workers' / (arm+'_train.launch.json'))
            evaluate = recovery.read(prepared / 'workers' / (arm+'_evaluate.launch.json'))
            assert train['config_sha256'] == evaluate['config_sha256']
            assert train['CUDA_VISIBLE_DEVICES'] == evaluate['CUDA_VISIBLE_DEVICES']
    finally:
        controller.lock.close()


def test_real_failure_stops_only_own_workers_clears_active_and_eta(prepared):
    controller = FakeController('config.json', poll_seconds=.02)
    controller.fail_arm = 'true_teacher'
    try:
        with pytest.raises(RuntimeError, match='injected compute failure'):
            try:
                controller.execute()
            except BaseException as error:
                controller.cleanup(error)
                raise
        status = recovery.read(prepared / 'status.json')
        assert status['state'] == 'FAILED'
        assert status['active_children'] == {} and status['projected_remaining_seconds'] is None
        assert status['current_task'] is None
        assert not list(prepared.glob('*/evaluate'))
        for path in (prepared / 'workers').glob('*.launch.json'):
            assert not recovery.same_process(recovery.read(path)['identity'])
    finally:
        controller.lock.close()


def test_existing_attempt_cannot_restart_or_overwrite(prepared):
    recovery.write(prepared / 'status.json', {'state': 'FAILED'})
    before = (prepared / 'status.json').read_bytes()
    with pytest.raises(AssertionError, match='no overwrite'):
        FakeController('config.json')
    assert (prepared / 'status.json').read_bytes() == before
