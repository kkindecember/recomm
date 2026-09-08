"""Finish the current epoch, then preserve its state and extend to the full horizon."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import torch

from experiment.phase18.analysis.screen_saved_reference import dataset_report
from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase18.core.full_budget import schedule_for
from experiment.phase18.core.screen_budget import optimizer_steps
from experiment.phase18.protocol.s18_diffgrm_full_resume import save_training, train_original
from experiment.phase18.protocol.s18_screen_resume import (
    ROOT, Backend, Run, alive, load_checkpoint, now, process_command, restore_probe,
)


def validate_source(source, config, spec):
    original = source.get('config', source.get('original_config'))
    if original != config or source['revision_config']['revision'] != 'screen_r1':
        raise ValueError('Source checkpoint/config/revision mismatch')
    if not source['model'] or not ('cpu_rng' in source and 'cuda_rng' in source):
        raise ValueError('Source must include trained weights, Adam and training RNG')
    epoch = int(source['epoch'])
    if not 0 < epoch < spec['total_epochs'] or optimizer_steps(source['optimizer']) <= 0:
        raise ValueError('Invalid source training position')
    if ('expected_steps_per_epoch' in spec and optimizer_steps(source['optimizer']) !=
            epoch * spec['expected_steps_per_epoch']):
        raise ValueError('Source Adam state is not at a complete epoch boundary')
    if any(group['lr'] <= 0 for group in source['optimizer']['param_groups']):
        raise ValueError('Short LR has already reached zero; requires a separate restart decision')
    source['config'] = original  # Normalize the in-memory schema, never edit the source file.
    return source


def verify_process(pid, config_path):
    if pid <= 1 or not alive(pid) or Path(f'/proc/{pid}').stat().st_uid != os.getuid():
        raise RuntimeError('Previous process is missing or belongs to another user')
    arguments = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if (b'experiment.phase18.protocol.s18_screen_resume' not in arguments
            or str(config_path).encode() not in arguments):
        raise RuntimeError(f'Previous process command does not match: {process_command(pid)}')
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def handoff(run, config):
    old = json.loads((run.root / 'status.json').read_text())
    if old.get('revision') != 'screen_r1' or old['state'] not in ('TRAINING', 'VALIDATING'):
        raise ValueError('Expected an active short-run trainer')
    pid = int(old['pid'])
    if pid != run.spec['previous_pid'] or old['active_config'] != run.spec['previous_config']:
        raise ValueError('Previous process identity changed')
    identity = verify_process(pid, run.spec['previous_config'])
    target_epoch = int(old['epoch'])
    checkpoint = run.root / 'screen_r1/last.pt'
    notice = dict(state='WAITING_FOR_EPOCH_CHECKPOINT', requested_at=now(), revision=run.spec['revision'],
                  target_epoch=target_epoch, coordinator_pid=os.getpid(), previous_pid=pid,
                  config=run.config_path, training_policy='full_horizon_sparse_validation_early_stop')
    write_json(run.root / 'pending_revision.json', notice)
    write_json(run.directory / 'status_before_switch.json', old)
    run.event('handoff_requested', **notice)
    started, stamp = time.monotonic(), None
    while True:
        if verify_process(pid, run.spec['previous_config']) != identity:
            raise RuntimeError('Previous process start time changed')
        if checkpoint.exists() and checkpoint.stat().st_mtime_ns != stamp:
            stamp = checkpoint.stat().st_mtime_ns
            saved = validate_source(load_checkpoint(checkpoint), config, run.spec)
            reached = saved['epoch'] >= target_epoch
            del saved
            if reached:
                break
        if time.monotonic() - started > run.spec['handoff_timeout_seconds']:
            raise TimeoutError('Handoff timeout; original trainer has not been interrupted')
        time.sleep(2)
    # A single open file descriptor pins the atomic checkpoint even if last.pt is replaced.
    snapshot = run.directory / 'source_checkpoint.pt'
    with checkpoint.open('rb') as origin, snapshot.open('xb') as destination:
        shutil.copyfileobj(origin, destination)
    source = validate_source(load_checkpoint(snapshot), config, run.spec)
    if source['epoch'] < target_epoch:
        raise ValueError('Snapshot is older than the completed handoff epoch')
    if verify_process(pid, run.spec['previous_config']) != identity:
        raise RuntimeError('Previous process identity changed before the handoff signal')
    observed = json.loads((run.root / 'status.json').read_text())
    os.kill(pid, signal.SIGINT)
    for _ in range(60):
        if not alive(pid):
            break
        time.sleep(1)
    if alive(pid):
        raise RuntimeError('Previous trainer is still alive; refusing duplicate GPU training')
    run.switched = True
    write_json(run.directory / 'status_after_interrupt.json', json.loads((run.root / 'status.json').read_text()))
    report = dict(previous_pid=pid, source_checkpoint=str(snapshot.relative_to(ROOT)),
                  source_checkpoint_sha256=sha256(snapshot), source_epoch=source['epoch'],
                  inherited_optimizer_steps=optimizer_steps(source['optimizer']),
                  requested_epoch=target_epoch, old_process_exited=True,
                  last_observed_status=observed,
                  purpose='User-authorized full-budget continuation; short prefix preserved',
                  boundary_note='The current full epoch is retained; only batches after checkpoint commit may replay.')
    write_json(run.directory / 'handoff.json', report)
    run.event('handoff_completed', **{k:v for k,v in report.items() if k != 'last_observed_status'})
    run.spec['additional_epochs'] = run.spec['total_epochs'] - source['epoch']
    run.context.update(stage_epochs=run.spec['additional_epochs'], inherited_epoch=source['epoch'],
                       inherited_optimizer_steps=report['inherited_optimizer_steps'])
    write_json(run.directory / 'resolved_config.json', run.spec)
    notice.update(state='APPLIED', applied_at=now(), source_epoch=source['epoch'])
    write_json(run.root / 'pending_revision.json', notice)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / args.config).read_text())
    if (spec['family'], spec['dataset']) not in (('diff_gram','Toys'), ('diff_gram','Beauty'), ('diffgrm','Toys')):
        raise ValueError('This handoff only applies to the three remaining short experiments')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['gpu_uuid']:
        raise ValueError('GPU UUID does not match authorized continuation')
    run = Run(spec, args.config)
    run.switched = False
    lock = (run.root / '.full_budget_resume.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if run.directory.exists():
        raise FileExistsError(f'Revision already exists: {run.directory}')
    config = (json.loads((run.root / 'manifest.json').read_text())['config'] if spec['family']=='diff_gram'
              else json.loads((run.root / 'config.json').read_text()))
    expected = ((config['training']['warmup_epochs'] + config['training']['joint_epochs'],
                 config['training']['warmup_epochs'] + config['training']['min_joint_epochs'],
                 config['training']['patience'], config['training']['warmup_epochs'])
                if spec['family']=='diff_gram' else (config['epochs'],config['minimum_epochs'],config['patience'],0))
    if expected != tuple(spec[k] for k in ('total_epochs','minimum_epochs','patience','evaluation_epoch_offset')):
        raise ValueError('Full budget or stopping rule differs from the original plan')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; previous trainer left running')
    run.references = dataset_report(spec['dataset'])  # CPU-only preflight before any interruption.
    run.directory.mkdir()
    write_json(run.directory / 'config.json', spec)
    write_json(run.directory / 'historical_references.json', run.references)
    sources = [Path(__file__).resolve(), ROOT / args.config,
        ROOT / 'experiment/phase18/core/full_budget.py',
        ROOT / 'experiment/phase18/protocol/s18_diffgrm_full_resume.py',
        ROOT / 'experiment/phase18/protocol/s18_screen_resume.py',
        ROOT / 'experiment/phase18/protocol/s18_diff_gram.py',
        ROOT / 'experiment/phase18/protocol/s18_diffgrm_native.py',
        ROOT / 'experiment/phase18/core/screen_budget.py', ROOT / 'experiment/phase18/core/diff_data.py',
        ROOT / 'experiment/phase18/analysis/screen_saved_reference.py',
        ROOT / 'experiment/phase18/run_stage18_full_budget_resume.sh']
    for path in sources:
        destination = run.directory / 'sources' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write_json(run.directory / 'source_manifest.json', {str(p.relative_to(ROOT)):sha256(p) for p in sources})
    run.context.update(total_epochs=spec['total_epochs'], minimum_epochs=spec['minimum_epochs'],
        patience=spec['patience'], gpu_uuid=spec['gpu_uuid'],
        result_path=str((run.directory / 'result.json').relative_to(ROOT)),
        training_policy='full_horizon_sparse_validation_early_stop',
        schedule_provenance='Preserved short-run prefix; one-epoch bridge, then original long LR curve',
        hard_timeout_seconds=spec['max_wall_seconds'])
    def timeout_handler(signum, frame):
        raise TimeoutError('Full continuation exceeded the recorded wall-time limit')
    try:
        source = handoff(run, config)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, spec['max_wall_seconds'])
        run.status('RESUMING', phase='restore_preserved_checkpoint')
        backend = Backend(run, source)
        restore_probe(backend, run, source)
        scheduler = schedule_for(backend, spec)
        if scheduler.get_last_lr() != backend.start_lrs:
            raise ValueError('Transition changed the restored first optimizer LR')
        report = dict(state='PASSED', **scheduler.state_dict(), source_epoch=backend.inherited_epoch,
            source_steps=backend.initial_steps, optimizer_updates_during_probe=0,
            first_update_lrs=scheduler.get_last_lr(), lrs_after_next_update=scheduler.at(backend.initial_steps+1),
            lrs_after_bridge=scheduler.at(backend.initial_steps+backend.per_epoch),
            lrs_at_full_horizon=scheduler.at(scheduler.total_steps),
            original_trajectory_reproduced=False, short_prefix_preserved=True)
        write_json(run.directory / 'schedule_restore_check.json', report)
        run.event('full_schedule_joined', **report)
        save_training(run.directory / 'restored_start.pt', backend, run, scheduler,
                      backend.inherited_epoch, backend.initial_steps)
        del source
        train_original(backend, run, scheduler)
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc(),
                       switched=run.switched, automatic_retry=False)
        write_json(run.directory / 'failure.json', failure)
        write_json(run.root / 'pending_revision.json', dict(state='FAILED',revision=spec['revision'],**failure))
        if run.switched:
            run.status('TIMED_OUT' if isinstance(error, TimeoutError) else 'FAILED', **failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
