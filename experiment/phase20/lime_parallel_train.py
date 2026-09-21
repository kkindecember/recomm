"""Checkpoint-preserving fixed-batch multi-GPU continuation of clean GRAM/C1."""
import gc
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiment.phase20.lime_clean_gram_core import (
    ROOT, SPECS, read, write, sha, now, load_inputs, build_model, seed_all, cpu_tree,
    epoch_order, update_windows, make_optimizer, total_updates, atomic_save,
)
from experiment.phase20.lime_parallel_core import CachedCollator, parallel_update
from experiment.phase20.lime_parallel_job import BASE_CONFIG, runtime, gpu_uuids
from experiment.phase20.lime_clean_gram_worker import check_manifest
from experiment.phase20.lime_clean_upstream_core import tensor_digest


def local_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=[torch.cuda.get_rng_state(0)])


def set_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'][0], 0)


def verify_replicas(model, world):
    digest = tensor_digest(cpu_tree(model.state_dict()))
    values = [None]*world
    dist.all_gather_object(values, digest)
    assert len(set(values)) == 1, values
    return digest


def resume_check_worker(rank, uuids, out_string):
    out = Path(out_string)
    runtime(rank, uuids, out/'check.rendezvous')
    world = len(uuids)
    config = read(BASE_CONFIG)
    data = load_inputs('Toys')
    parent, reports = None, {}
    for stage in ('gram', 'c1'):
        seed_all(2023)
        model, original = build_model(data, stage, config, torch.device('cuda'), parent)
        collator = CachedCollator(original)
        optimizer, scheduler = make_optimizer(model, stage, config, total_updates(data, stage, config))
        dataset = data['dataset' if stage == 'gram' else 'c1_dataset']
        indices = sorted(range(len(dataset)), key=lambda i: len(dataset.rows[i]['history']))
        first = [[indices[0]], [indices[-1]], [indices[1]]]
        second = [[indices[2]], [indices[-2]], [indices[3]]]
        seed_all(2023+rank)
        parallel_update(model, optimizer, scheduler, collator, dataset, first, stage, config, torch.device('cuda'), rank, world)
        states = [None]*world
        dist.all_gather_object(states, local_rng())
        path = out/(stage+'_resume.pt')
        if rank == 0:
            atomic_save(path, dict(state_dict=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                scheduler=scheduler.state_dict(), parallel_rng=states, probe_only=True, formal_parent=False))
        dist.barrier()
        continuous = parallel_update(model, optimizer, scheduler, collator, dataset, second, stage, config, torch.device('cuda'), rank, world)
        expected_model = tensor_digest(cpu_tree(model.state_dict()))
        expected_optimizer = cpu_tree(optimizer.state_dict())
        saved = torch.load(path, map_location='cpu')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        set_rng(saved['parallel_rng'][rank])
        del saved
        repeated = parallel_update(model, optimizer, scheduler, collator, dataset, second, stage, config, torch.device('cuda'), rank, world)
        assert repeated['loss'] == continuous['loss']
        assert tensor_digest(cpu_tree(model.state_dict())) == expected_model
        actual_optimizer = cpu_tree(optimizer.state_dict())
        for key, value in expected_optimizer['state'].items():
            for name, expected in value.items():
                actual = actual_optimizer['state'][key][name]
                assert torch.equal(expected, actual) if torch.is_tensor(expected) else expected == actual
        assert scheduler.last_epoch == 2
        verify_replicas(model, world)
        reports[stage] = dict(state='PASSED', exact_model_optimizer_loss_replay=True,
            dropout_enabled=True, scheduler_step=2, two_rank_rng_restored=True,
            checkpoint_sha256=sha(path), probe_only=True)
        if stage == 'gram':
            parent = cpu_tree(model.state_dict())
        del model, original, collator, optimizer, scheduler, expected_optimizer, actual_optimizer
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        write(out/'summary.json', dict(state='PASSED', reports=reports, gpus=uuids,
            sources={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ROOT/'experiment/phase20/lime_parallel_core.py')},
            formal_training_started=False, test_read=False))
    dist.destroy_process_group()


def train_worker(rank, uuids, out_string, resume):
    out = Path(out_string)
    try:
        _train_worker(rank, uuids, out, resume)
    except BaseException:
        (out/f'failure_rank{rank}_{os.getpid()}.txt').write_text(traceback.format_exc())
        raise


def _train_worker(rank, uuids, out, resume):
    runtime(rank, uuids, out/f'rendezvous_{os.environ["LIME_ATTEMPT_ID"]}')
    world = len(uuids)
    assert world > 1
    prepared = read(out/'prepared.json')
    manifest = prepared['input_sha256']
    check_manifest(manifest)
    config = read(BASE_CONFIG)
    domain = prepared['domain']
    data = load_inputs(domain)
    assert all(manifest.get(k) == v for k, v in data['manifest'].items())
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def status(state, **extra):
        if rank == 0:
            write(out/'status.json', dict(state=state, domain=domain, updated_at=now(),
                rank0_pid=os.getpid(), world_size=world, gpus=uuids, test_read=False,
                development_labels_read=False, **extra))

    completed, previous_seconds = {}, 0.
    for stage in ('gram', 'c1'):
        directory = out/stage
        if rank == 0:
            directory.mkdir(exist_ok=resume)
        dist.barrier()
        if (directory/'summary.json').exists():
            assert resume
            summary = read(directory/'summary.json')
            assert summary['state'] == 'COMPLETED' and sha(directory/'final.pt') == summary['checkpoint_sha256']
            completed[stage] = summary
            previous_seconds += summary['training_seconds']
            continue
        parent = None
        if stage == 'c1':
            saved_parent = torch.load(out/'gram/final.pt', map_location='cpu')
            assert saved_parent['formal_parent'] and saved_parent['input_sha256'] == manifest
            assert saved_parent['position']['epoch'] == config['gram_epochs']+1
            parent = saved_parent['state_dict']
            del saved_parent
        seed_all(config['seed'])
        model, original = build_model(data, stage, config, torch.device('cuda'), parent)
        del parent
        collator = CachedCollator(original)
        optimizer, scheduler = make_optimizer(model, stage, config, total_updates(data, stage, config))
        dataset = data['dataset' if stage == 'gram' else 'c1_dataset']
        position = dict(epoch=1, window=0, updates=0)
        old_seconds = 0.
        seed_all(config['seed']+rank)
        load_path = directory/'latest.pt' if resume and (directory/'latest.pt').exists() else None
        migration = False
        if load_path is None and stage == 'gram' and prepared.get('migration'):
            load_path = out/'migration.pt'
            migration = True
        if load_path:
            checkpoint = torch.load(load_path, map_location='cpu')
            assert checkpoint['formal_parent'] and not checkpoint['probe_only']
            assert checkpoint['stage'] == stage and checkpoint['training_config'] == config
            expected_manifest = prepared['base_manifest'] if migration else manifest
            assert checkpoint['input_sha256'] == expected_manifest
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.zero_grad(set_to_none=True)
            position = checkpoint['position'].copy()
            old_seconds = checkpoint['training_seconds']
            assert scheduler.last_epoch == position['updates']
            if migration:
                if rank == 0:
                    set_rng(checkpoint['rng'])
            else:
                assert checkpoint['world_size'] == world
                set_rng(checkpoint['parallel_rng'][rank])
            del checkpoint
        digest = verify_replicas(model, world)
        if migration:
            assert digest == prepared['migration']['model_digest']
        stage_start = time.monotonic()
        def elapsed():
            return old_seconds + time.monotonic()-stage_start

        def save(name='latest.pt'):
            states = [None]*world
            dist.all_gather_object(states, local_rng())
            if rank == 0:
                assert all(p.grad is None for p in model.parameters())
                atomic_save(directory/name, dict(state_dict=cpu_tree(model.state_dict()),
                    optimizer=cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                    rng=states[0], parallel_rng=states, position=position.copy(),
                    input_sha256=manifest, training_config=config, stage=stage, created_at=now(),
                    world_size=world, training_seconds=elapsed(), probe_only=False, formal_parent=True,
                    domain=domain, catalog=data['ids'], lexical_type=SPECS[domain]['lexical'],
                    migration=prepared.get('migration'), parallel_execution='SUM gradients of fixed original microbatches'))
            dist.barrier()

        save()
        status('RUNNING', stage=stage, next_position=position.copy(), restored_model_digest=digest,
               scheduler_step=scheduler.last_epoch, expected_updates=total_updates(data, stage, config))
        if rank == 0:
            write(directory/f'attempt_{os.getpid()}.json', dict(started_at=now(), position=position.copy(),
                model_digest=digest, scheduler_step=scheduler.last_epoch, migration=migration,
                rank0_cuda_rng_preserved=bool(migration), other_rank_seed=config['seed']+1,
                resume=resume, gpus=uuids))
        log = (directory/'updates.jsonl').open('a', buffering=1) if rank == 0 else None
        last_report = 0.
        first_update = True
        try:
            while position['epoch'] <= config[f'{stage}_epochs']:
                epoch = position['epoch']
                windows = list(update_windows(epoch_order(len(dataset), config['seed'], epoch), 16, 8))
                for index in range(position['window'], len(windows)):
                    flag = torch.tensor(int(stop or previous_seconds+elapsed() > config['training_timeout_seconds']), device='cuda')
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                    if int(flag):
                        save()
                        status('STOPPED', stage=stage, next_position=position.copy())
                        dist.destroy_process_group()
                        return
                    start = time.monotonic()
                    record = parallel_update(model, optimizer, scheduler, collator, dataset, windows[index],
                        stage, config, torch.device('cuda'), rank, world)
                    torch.cuda.synchronize()
                    duration = torch.tensor(time.monotonic()-start, device='cuda')
                    dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                    position.update(epoch=epoch, window=index+1, updates=position['updates']+1)
                    if index+1 == len(windows):
                        position.update(epoch=epoch+1, window=0)
                    record.update(epoch=epoch, window=index, update=position['updates'],
                        seconds=float(duration), lr=optimizer.param_groups[0]['lr'], world_size=world)
                    if log:
                        log.write(json.dumps(record)+'\n')
                    if first_update:
                        verify_replicas(model, world)
                    if first_update or position['updates'] % config['checkpoint_every_updates'] == 0 or position['window'] == 0:
                        save()
                    first_update = False
                    if time.monotonic()-last_report >= 30 or position['window'] == 0:
                        status('RUNNING', stage=stage, next_position=position.copy(), last_loss=record['loss'],
                            last_update_seconds=record['seconds'], training_seconds=previous_seconds+elapsed(),
                            expected_updates=total_updates(data, stage, config), scheduler_step=scheduler.last_epoch)
                        last_report = time.monotonic()
                check_manifest(manifest)
                if epoch % 5 == 0 and rank == 0:
                    atomic_save(directory/f'epoch_{epoch:02d}.pt', dict(state_dict=cpu_tree(model.state_dict()),
                        epoch=epoch, stage=stage, input_sha256=manifest, formal_parent=True, domain=domain))
                dist.barrier()
            assert position['updates'] == total_updates(data, stage, config)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            save('final.pt')
            if rank == 0:
                write(directory/'summary.json', dict(state='COMPLETED', completed_at=now(), stage=stage,
                    epochs=config[f'{stage}_epochs'], optimizer_updates=position['updates'], training_seconds=elapsed(),
                    checkpoint_sha256=sha(directory/'final.pt'), input_sha256=manifest,
                    fixed_final_epoch=True, test_read=False, development_labels_read=False))
            dist.barrier()
            completed[stage] = read(directory/'summary.json')
            previous_seconds += completed[stage]['training_seconds']
        finally:
            if log:
                log.close()
        del model, original, collator, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        write(out/'summary.json', dict(state='COMPLETED', stages=completed, input_sha256=manifest,
            next_step='candidate generation and clean train-only gate/ranker', new_recommendation_metrics=False))
    status('COMPLETED')
    dist.destroy_process_group()


def launch(args):
    out = ROOT/args.out
    assert not out.exists()
    devices = gpu_uuids(args.gpus)
    assert len(devices) == 2, 'This measured continuation is fixed to two ranks'
    profile_path = ROOT/args.profile
    profile = read(profile_path)
    assert profile['state'] == 'PASSED' and profile['domain'] == args.domain
    parallel = profile['arms']['parallel_cached']
    assert parallel['gpus'] == devices and parallel['projected_hours'] <= 60
    assert parallel['profile_weights_discarded']
    assert all(r['gradient_check']['state'] == 'PASSED' for r in parallel['reports'].values())
    check = read(ROOT/'artifacts/phase20/lime_parallel_checks_20260921_v1/collation.json')
    assert check['state'] == 'PASSED'
    assert check['sources']['experiment/phase20/lime_parallel_core.py'] == sha(ROOT/'experiment/phase20/lime_parallel_core.py')
    resume_check = read(ROOT/'artifacts/phase20/lime_parallel_resume_check_20260921_v1/summary.json')
    assert resume_check['state'] == 'PASSED'
    check_manifest(resume_check['sources'])
    config = read(BASE_CONFIG)
    base_root = ROOT/config['output']/args.domain
    base_manifest = read(base_root/'input_manifest.json')
    check_manifest(base_manifest)
    assert read(base_root/'cpu_check/summary.json')['state'] == 'PASSED'
    assert shutil.disk_usage(ROOT).free > 50*2**30
    out.mkdir(parents=True)
    migration = None
    if args.stop_pid:
        # Only the precisely identified old Toys worker can receive this signal.
        receipt = read(base_root/'train.launch.json')
        status = read(base_root/'train/status.json')
        assert receipt['pid'] == args.stop_pid == status['pid']
        command = Path(f'/proc/{args.stop_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert 'experiment.phase20.lime_clean_gram_worker train' in command
        assert '--domain '+args.domain in command
        assert status['state'] == 'RUNNING' and status['training_stage'] == 'gram'
        os.kill(args.stop_pid, signal.SIGTERM)
        start = time.monotonic()
        while Path(f'/proc/{args.stop_pid}').exists():
            assert time.monotonic()-start < 180, 'Old worker did not stop; not starting duplicate training'
            time.sleep(1)
        old_path = base_root/'train/gram/latest.pt'
        shutil.copy2(old_path, out/'migration.pt')
        checkpoint = torch.load(out/'migration.pt', map_location='cpu')
        assert checkpoint['input_sha256'] == base_manifest and checkpoint['training_config'] == config
        assert checkpoint['formal_parent'] and not checkpoint['probe_only']
        assert checkpoint['scheduler']['last_epoch'] == checkpoint['position']['updates']
        for value in checkpoint['state_dict'].values():
            assert torch.isfinite(value).all()
        migration = dict(source=str(old_path.relative_to(ROOT)), stopped_pid=args.stop_pid,
            checkpoint_sha256=sha(out/'migration.pt'), position=checkpoint['position'],
            model_digest=tensor_digest(checkpoint['state_dict']),
            preserves='model, optimizer, scheduler, next global update, rank0 RNG',
            change='rank1 seed=2024; dropout assignment and floating reduction change; no bitwise equivalence claim')
        del checkpoint
    elif args.domain == 'Toys':
        raise AssertionError('Toys requires explicit checkpoint continuation')
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
        ROOT/'experiment/phase20/lime_parallel_core.py', ROOT/'experiment/phase20/lime_parallel_job.py',
        ROOT/'experiment/phase20/test_lime_parallel.py', ROOT/'experiment/phase20/run_lime_parallel.sh']}
    manifest = dict(base_manifest, **sources)
    manifest[str(profile_path.relative_to(ROOT))] = sha(profile_path)
    if migration:
        manifest[str((out/'migration.pt').relative_to(ROOT))] = sha(out/'migration.pt')
    for relative in sources:
        target = out/'sources'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
    write(out/'prepared.json', dict(domain=args.domain, input_sha256=manifest, base_manifest=base_manifest,
        migration=migration, gpus=devices, profile=str(profile_path.relative_to(ROOT)), created_at=now(),
        effective_batch=128, microbatch=16, local_microbatches=4, world_size=2,
        model_epochs_unchanged=True, no_evaluation_labels=True))
    command = [sys.executable, '-B', '-m', 'experiment.phase20.lime_parallel_train', 'supervise', '--out', args.out]
    with (out/'host.log').open('x', buffering=1) as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    write(out/'launch.json', dict(pid=child.pid, created_at=now(), command=command, migration=migration))
    print(dict(pid=child.pid, output=str(out), migration=migration), flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'launch', 'supervise'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--domain', choices=('Toys', 'Beauty'))
    parser.add_argument('--gpus')
    parser.add_argument('--profile')
    parser.add_argument('--stop-pid', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'check':
        out = ROOT/args.out
        out.mkdir(parents=True, exist_ok=False)
        devices = gpu_uuids(args.gpus)
        mp.spawn(resume_check_worker, args=(devices, str(out)), nprocs=len(devices), join=True)
    elif args.action == 'launch':
        launch(args)
    else:
        out = ROOT/args.out
        prepared = read(out/'prepared.json')
        os.environ['LIME_ATTEMPT_ID'] = str(os.getpid())
        try:
            mp.spawn(train_worker, args=(prepared['gpus'], str(out), args.resume), nprocs=len(prepared['gpus']), join=True)
        except BaseException:
            (out/f'supervisor_failure_{os.getpid()}.txt').write_text(traceback.format_exc())
            write(out/'status.json', dict(state='FAILED', updated_at=now(), supervisor_pid=os.getpid()))
            raise


if __name__ == '__main__':
    main()
