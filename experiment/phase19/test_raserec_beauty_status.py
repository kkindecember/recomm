import json
from pathlib import Path

import pytest

from experiment.phase19 import raserec_beauty as runner


def test_overview_replaces_legacy_link_without_overwriting_toys(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    toys = tmp_path/'artifacts/phase18/toys/status.json'
    monkeypatch.setattr(runner, 'TOYS', toys)
    runner.write(toys, {'state': 'complete', 'pid': 0})
    overview = tmp_path/'artifacts/phase19/status.json'
    overview.parent.mkdir(parents=True)
    overview.symlink_to(toys)
    original = toys.read_bytes()
    runner.publish({'output': 'artifacts/phase19/beauty'}, {'state': 'RUNNING'})
    assert toys.read_bytes() == original
    assert not overview.is_symlink()
    assert runner.read(overview)['experiments']['beauty']['state'] == 'RUNNING'


@pytest.mark.parametrize('returncode,raw_state,expected,wait', [
    (1, None, 'FAILED', False), (0, None, 'FAILED', False),
    (0, 'complete', 'COMPLETED', False), (0, 'complete', 'COMPLETED', True)])
def test_supervisor_checks_exit_and_raw_completion(tmp_path, monkeypatch, returncode, raw_state, expected, wait):
    original_root = runner.ROOT
    config = runner.read(original_root/runner.DEFAULT_CONFIG)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'TOYS', tmp_path/'toys_status.json')
    config_path = tmp_path/'config.json'
    runner.write(config_path, config)
    runner.write(tmp_path/config['runtime']['preflight'], {
        'state': 'passed', 'config_sha256': runner.digest(config_path), 'source_sha256': {}})
    monkeypatch.setattr(runner.signal, 'signal', lambda *args: None)
    free_values = iter([config['minimum_free_mib']-1, 21000] if wait else [21000])
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: f'1, {config["gpu_uuid"]}, {next(free_values)}, 0\n')
    monkeypatch.setattr(runner.time, 'sleep', lambda *args: None)
    states = []
    original_publish = runner.publish

    def publish(config, live):
        states.append(live['state'])
        original_publish(config, live)

    monkeypatch.setattr(runner, 'publish', publish)

    class Child:
        pid = 123

        def __init__(self, *args, **kwargs):
            self.returncode = returncode
            if raw_state:
                runner.write(tmp_path/config['output']/'status.json', {'state': raw_state})

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, 'Popen', Child)
    exitcode = runner.supervise(config, config_path)
    live = runner.read(tmp_path/config['output']/'live_status.json')
    assert live['state'] == expected
    assert live['process_alive'] is False
    assert live['returncode'] == returncode
    assert exitcode == (0 if expected == 'COMPLETED' else 1)
    if expected == 'COMPLETED':
        assert live['stage'] == 'complete'
    assert ('WAITING_FOR_GPU' in states) is wait
