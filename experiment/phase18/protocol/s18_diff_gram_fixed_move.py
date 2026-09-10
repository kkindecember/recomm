"""Move the active Beauty fixed schedule at a saved epoch boundary, retaining its deadline."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

import torch
from transformers import get_linear_schedule_with_warmup

from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase18.core.screen_budget import optimizer_steps
from experiment.phase18.protocol import s18_diff_gram as original
from experiment.phase18.protocol.s18_diff_gram_fixed import (
    FixedRun, FreshBackend, frozen_references, matched_validation, preserve_training_rng,
)
from experiment.phase18.protocol.s18_diffgrm_full_resume import save_training
from experiment.phase18.protocol.s18_screen_resume import load_checkpoint, now

ROOT = original.ROOT


def read(path):
    return json.loads(Path(path).read_text())


def verify_boundary(source,spec,config,per_epoch=1027,cuda_rng_count=1):
    epoch = source['epoch']
    joint_step = (epoch-1)*per_epoch
    total = config['training']['joint_epochs']*per_epoch
    warmup = max(1,int(total*config['training']['warmup_fraction']))
    factor = joint_step/warmup if joint_step < warmup else (total-joint_step)/(total-warmup)
    expected = [config['training'][key]*factor for key in ('new_learning_rate','backbone_learning_rate')]
    actual = [g['lr'] for g in source['optimizer']['param_groups']]
    if (source['config'] != config or source['revision_config'] != spec or
            source['phase'] != 'joint' or source['phase_epoch'] != epoch-1 or
            optimizer_steps(source['optimizer']) != epoch*per_epoch or
            source['scheduler']['last_epoch'] != joint_step or
            source['scheduler']['_last_lr'] != actual or len(actual) != 2 or
            len(source['cuda_rng']) != cuda_rng_count or
            not torch.is_tensor(source['cpu_rng']) or
            any(not math.isclose(a,b,abs_tol=1e-14,rel_tol=0) for a,b in zip(actual,expected))):
        raise ValueError('Boundary checkpoint cannot reproduce the frozen training state')


def restore_joint(backend, source):
    settings = backend.config['training']
    completed = source['epoch'] - settings['warmup_epochs']
    total = settings['joint_epochs'] * backend.per_epoch
    if (source['phase'] != 'joint' or source['phase_epoch'] != completed or
            source['epoch'] * backend.per_epoch != optimizer_steps(source['optimizer'])):
        raise ValueError('Expected a complete fixed-schedule joint epoch')
    backend.model.load_state_dict(source['model'], strict=True)
    backend.model.set_backbone_frozen(False)
    scheduler = get_linear_schedule_with_warmup(backend.optimizer,
        max(1, int(total*settings['warmup_fraction'])), total)
    scheduler.load_state_dict(source['scheduler'])
    backend.optimizer.load_state_dict(source['optimizer'])
    expected = [b*f(completed*backend.per_epoch) for b,f in zip(scheduler.base_lrs,scheduler.lr_lambdas)]
    actual = [group['lr'] for group in backend.optimizer.param_groups]
    if (scheduler.last_epoch != completed*backend.per_epoch or actual != scheduler.get_last_lr() or
            any(not math.isclose(a,b,abs_tol=1e-14,rel_tol=0) for a,b in zip(actual,expected))):
        raise ValueError('Optimizer and original linear schedule do not agree')
    torch.set_rng_state(source['cpu_rng'])
    if torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all(source['cuda_rng'])
    return scheduler


def train_remaining(backend, run, cohort, source):
    scheduler = restore_joint(backend,source)
    settings = backend.config['training']
    steps = optimizer_steps(source['optimizer'])
    best, best_epoch = source['best_score'], source['best_epoch']
    initial_epoch = source['epoch']
    write_json(run.directory/'schedule_restore_check.json',dict(state='PASSED',
        source_epoch=initial_epoch,optimizer_steps=steps,scheduler_step=scheduler.last_epoch,
        learning_rates=scheduler.get_last_lr(),learning_rate_restarted=False,
        remaining_epochs=run.spec['total_epochs']-initial_epoch,original_deadline_retained=True))
    started = time.monotonic()
    for epoch in range(initial_epoch+1,run.spec['total_epochs']+1):
        phase_epoch = epoch-settings['warmup_epochs']
        backend.model.train()
        generator = torch.Generator().manual_seed(backend.config['seed']+epoch)
        loader = backend.loader(backend.training,backend.microbatch,generator)
        backend.optimizer.zero_grad(set_to_none=True)
        seen = group = 0
        total_loss = 0.
        epoch_start = last_update = time.monotonic()
        run.status('TRAINING',phase='joint',epoch=epoch,phase_epoch=phase_epoch,
            seen=0,train_total=len(backend.training),optimizer_steps=steps,
            learning_rates=scheduler.get_last_lr(),best_epoch=best_epoch)
        for index,batch in enumerate(loader):
            loss = backend.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite fixed-schedule loss after device move')
            count = backend.size(batch)
            (loss*count).backward()
            seen += count
            group += count
            total_loss += float(loss.detach())*count
            if group == backend.effective or index+1 == len(loader):
                for parameter in backend.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group)
                torch.nn.utils.clip_grad_norm_(backend.model.parameters(),backend.clip,error_if_nonfinite=True)
                backend.optimizer.step()
                scheduler.step()
                backend.optimizer.zero_grad(set_to_none=True)
                steps += 1
                group = 0
            current = time.monotonic()
            if current-last_update >= 30:
                progress = dict(epoch=epoch,phase_epoch=phase_epoch,seen=seen,train_total=len(backend.training),
                    optimizer_steps=steps,loss=total_loss/seen,epoch_elapsed_seconds=current-epoch_start,
                    epoch_eta_seconds=(len(backend.training)-seen)*(current-epoch_start)/seen,
                    learning_rates=scheduler.get_last_lr(),best_epoch=best_epoch)
                run.status('TRAINING',phase='joint',**progress)
                run.event('training_progress',**progress)
                last_update = current
        if (group or seen != len(backend.training) or steps != epoch*backend.per_epoch or
                scheduler.last_epoch != phase_epoch*backend.per_epoch):
            raise ValueError('Data/optimizer/scheduler boundary differs from fixed training')
        if phase_epoch%settings['evaluation_interval'] == 0 or epoch == run.spec['total_epochs']:
            with preserve_training_rng():
                result = matched_validation(backend,cohort,run,epoch,False)
            if result['metrics']['ndcg@10'] > best:
                best,best_epoch = result['metrics']['ndcg@10'],epoch
                save_training(run.directory/'best_trend.pt',backend,run,scheduler,epoch,steps,
                              phase='joint',selection=result)
            write_json(run.directory/'selection_state.json',dict(epoch=epoch,best_epoch=best_epoch,
                best_score=best,early_stopping_enabled=False))
        save_training(run.directory/'last.pt',backend,run,scheduler,epoch,steps,
            phase='joint',phase_epoch=phase_epoch,best_epoch=best_epoch,best_score=best)
        run.event('epoch_completed',phase='joint',epoch=epoch,optimizer_steps=steps,
            train_examples_seen=seen,loss=total_loss/seen,
            seconds_including_selection=time.monotonic()-epoch_start,
            learning_rates=scheduler.get_last_lr(),best_epoch=best_epoch)
    selected = load_checkpoint(run.directory/'best_trend.pt')
    backend.model.load_state_dict(selected['model'],strict=True)
    del selected
    with preserve_training_rng():
        full = matched_validation(backend,backend.validation,run,best_epoch,True)
    result = dict(state='COMPLETED',revision=run.spec['revision'],source_epoch=0,
        device_move_after_epoch=initial_epoch,selected_absolute_epoch=best_epoch,
        selected_stage_epoch=best_epoch,epochs_completed=epoch,optimizer_steps=steps,
        stopping_reason='epoch_limit',early_stopping_enabled=False,automatic_additional_training=False,
        full_validation=full,seconds_after_device_move=time.monotonic()-started,
        interpretation='Original fixed 1+10 schedule retained across device move; same seed, '
        'no LR restart or deadline extension; exploratory validation, not independent test.')
    write_json(run.directory/'result.json',result)
    write_json(run.root/'result.json',result)
    run.status('COMPLETED',phase='fixed_schedule_complete',result=result)
    run.event('training_completed',**result)


def process_identity(pid, config):
    proc = Path(f'/proc/{pid}')
    if proc.stat().st_uid != os.getuid():
        raise ValueError('Trainer belongs to a different user')
    arguments = (proc/'cmdline').read_bytes().split(b'\0')
    if (b'experiment.phase18.protocol.s18_diff_gram_fixed' not in arguments or
            config.encode() not in arguments):
        raise ValueError('Trainer command differs from the expected fixed run')
    return (proc/'stat').read_text().rsplit(')',1)[1].split()[19]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    move = read(ROOT/args.config)
    if sha256(ROOT/move['original_exec_config']) != move['original_exec_sha256']:
        raise ValueError('Original execution contract changed')
    spec = read(ROOT/move['original_exec_config'])
    config = read(ROOT/spec['model_config'])
    if sha256(ROOT/spec['model_config']) != spec['model_config_sha256']:
        raise ValueError('Frozen Beauty model configuration changed')
    if (spec['dataset'],spec['total_epochs'],spec['seed']) != ('Beauty',11,2023):
        raise ValueError('Only the active Beauty fixed run may be moved')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != move['target_gpu_uuid']:
        raise ValueError('Destination GPU mismatch')
    for path,expected in move['probe_sha256'].items():
        if sha256(ROOT/path) != expected or not read(ROOT/path)['exact_output_identity']:
            raise ValueError('Destination/original fixed-case identity check changed')
    canonical = ROOT/spec['run_directory']
    directory = canonical/move['revision']
    directory.mkdir(exist_ok=False)
    frozen = read(canonical/'manifest.json')['source_sha256'].copy()
    for path in [str(Path(__file__).resolve().relative_to(ROOT)),args.config,
                 'experiment/phase18/run_stage18_diff_gram_fixed_move.sh']:
        frozen[path] = sha256(ROOT/path)
    for path,expected in frozen.items():
        if sha256(ROOT/path) != expected:
            raise ValueError(f'Original or migration source changed: {path}')
        destination = directory/'watch_sources'/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/path,destination)
    write_json(directory/'watch_source_manifest.json',frozen)
    pending_path = canonical/'migration_pending.json'
    deadline = datetime.fromisoformat(move['deadline_utc']).timestamp()
    identity = process_identity(move['old_pid'],move['original_exec_config'])
    pending = dict(state='WAITING_FOR_EPOCH_CHECKPOINT',updated_at=now(),pid=os.getpid(),
        old_pid=move['old_pid'],target_gpu_uuid=move['target_gpu_uuid'],
        minimum_epoch=move['minimum_epoch'],deadline_utc=move['deadline_utc'])
    write_json(directory/'config.json',move)
    moved = False
    run = None
    def timeout(signum,frame):
        raise TimeoutError('Original Beauty 36-hour deadline reached')
    signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,max(1,deadline-time.time()))
    try:
        last_stamp = None
        while True:
            if process_identity(move['old_pid'],move['original_exec_config']) != identity:
                raise ValueError('Original process identity changed')
            pending.update(updated_at=now())
            write_json(pending_path,pending)
            checkpoint = canonical/'last.pt'
            stamp = checkpoint.stat().st_mtime_ns
            if stamp != last_stamp:
                source = load_checkpoint(checkpoint)
                last_stamp = stamp
                candidate = move['minimum_epoch'] <= source['epoch'] < spec['total_epochs']
                free = int(subprocess.check_output(['nvidia-smi','--id='+move['target_gpu_uuid'],
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if candidate and free >= move['minimum_free_mib'] and time.time()-checkpoint.stat().st_mtime < 20:
                    boundary_epoch = source['epoch']
                    break
                pending.update(last_complete_epoch=source['epoch'],destination_free_mib=free)
                del source
            time.sleep(2)
        # Stop only this verified trainer, after a freshly committed full epoch.
        verify_boundary(source,spec,config)
        for path,expected in frozen.items():
            if sha256(ROOT/path) != expected:
                raise ValueError(f'Source changed while waiting; original trainer left running: {path}')
        write_json(directory/'status_before_interrupt.json',read(canonical/'status.json'))
        if process_identity(move['old_pid'],move['original_exec_config']) != identity:
            raise ValueError('Original process changed before boundary handoff')
        os.kill(move['old_pid'],signal.SIGINT)
        for _ in range(60):
            try:
                process_identity(move['old_pid'],move['original_exec_config'])
            except FileNotFoundError:
                break
            time.sleep(1)
        else:
            raise RuntimeError('Old trainer still alive; refusing duplicate training')
        moved = True
        write_json(directory/'status_after_interrupt.json',read(canonical/'status.json'))
        source_path = directory/'source_checkpoint.pt'
        shutil.copy2(checkpoint,source_path)
        source = load_checkpoint(source_path)
        if source['config'] != config or source['epoch'] != boundary_epoch:
            raise ValueError('Captured checkpoint differs from the verified epoch boundary')
        best = load_checkpoint(canonical/'best_trend.pt')
        if best['epoch'] != source['best_epoch'] or best['selection']['metrics']['ndcg@10'] != source['best_score']:
            raise ValueError('Historical best selection differs from checkpoint')
        shutil.copy2(canonical/'best_trend.pt',directory/'best_trend.pt')
        write_json(directory/'selection_state.json',read(canonical/'selection_state.json'))
        active = dict(spec,revision=move['revision'],gpu_uuid=move['target_gpu_uuid'])
        run = FixedRun(active,args.config)
        run.directory = directory
        run.context.update(gpu_uuid=move['target_gpu_uuid'],device_move_after_epoch=source['epoch'],
            original_deadline_utc=move['deadline_utc'],
            result_path=str((directory/'result.json').relative_to(ROOT)),
            latest_trend_validation=best['selection'])
        del best
        run.status('RESUMING',phase='same_schedule_new_gpu')
        original.seed_everything(spec['seed'])
        backend = FreshBackend(config,spec['microbatch'],torch.device('cuda:0'))
        if (len(backend.training),len(backend.validation),backend.per_epoch) != (131413,22363,1027):
            raise ValueError('Beauty full-data contract changed')
        cohort = frozen_references(backend,run)
        original.snapshot(directory,config,args,backend.manifest,backend.model)
        write_json(directory/'device_move.json',dict(state='RESUMED',old_pid=move['old_pid'],
            new_pid=os.getpid(),target_gpu_uuid=move['target_gpu_uuid'],source_epoch=source['epoch'],
            source_checkpoint_sha256=sha256(source_path),deadline_utc=move['deadline_utc'],
            optimizer_and_scheduler_restarted=False,original_total_epochs=11))
        pending.update(state='RESUMED',updated_at=now(),new_pid=os.getpid(),source_epoch=source['epoch'])
        write_json(pending_path,pending)
        train_remaining(backend,run,cohort,source)
    except BaseException as error:
        failure = dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,
                       old_trainer_interrupted=moved)
        write_json(directory/'failure.json',failure)
        pending.update(state='FAILED',updated_at=now(),**failure)
        write_json(pending_path,pending)
        if moved and run is not None:
            run.status('TIMED_OUT' if isinstance(error,TimeoutError) else 'FAILED',**failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__ == '__main__':
    main()
