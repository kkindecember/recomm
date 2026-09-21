"""Measured acceleration of the frozen clean GRAM experiment; no evaluation data."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiment.phase20.lime_clean_gram_core import (
    ROOT, read, write, sha, now, load_inputs, build_model, seed_all, cpu_tree,
    epoch_order, update_windows, make_optimizer, optimizer_update, total_updates,
)
from experiment.phase20.lime_parallel_core import CachedCollator, parallel_update, sum_gradients

BASE_CONFIG = ROOT/'experiment/phase20/lime_clean_gram_20260920_native_v2.json'


def runtime(rank, uuids, rendezvous):
    os.environ['CUDA_VISIBLE_DEVICES'] = uuids[rank]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(25600/(torch.cuda.get_device_properties(0).total_memory/2**20))
    if len(uuids) > 1:
        dist.init_process_group('nccl', init_method='file://'+str(rendezvous), rank=rank,
                                world_size=len(uuids), timeout=timedelta(seconds=300))


def collation_check(data, original):
    cached = CachedCollator(original)
    ds = data['dataset']
    sorted_rows = sorted(range(len(ds)), key=lambda i: len(ds.rows[i]['history']))
    rng = np.random.RandomState(2023)
    batches = [sorted_rows[:1], sorted_rows[-16:], sorted_rows[:16]]
    batches += [rng.choice(len(ds), size=n, replace=False).tolist() for n in (1, 3, 7, 16)*5]
    original_seconds = cached_seconds = 0.
    for indices in batches:
        samples = [ds[i] for i in indices]
        start = time.monotonic()
        expected = original(samples)
        original_seconds += time.monotonic()-start
        start = time.monotonic()
        actual = cached(samples)
        cached_seconds += time.monotonic()-start
        assert actual.keys() == expected.keys()
        for key in expected:
            if torch.is_tensor(expected[key]):
                assert torch.equal(actual[key], expected[key]), key
            else:
                assert actual[key] == expected[key], key
        repeated = cached(samples)
        assert all(torch.equal(repeated[k], expected[k]) for k in expected if torch.is_tensor(expected[k]))
    return dict(state='PASSED', batches=len(batches), original_seconds=original_seconds,
                cold_cache_seconds=cached_seconds, every_returned_tensor_exact=True)


def real_gradient_check(model, collator, dataset, stage, config, rank, world):
    """Compare distributed SUM with serial global gradients, disabling dropout only here."""
    model.eval()
    indices = sorted(range(len(dataset)), key=lambda i: len(dataset.rows[i]['history']))[:3]
    def backward(i):
        batch = collator([dataset[i]])
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        if stage == 'gram':
            loss = model(input_ids=batch['item_text_ids'], attention_mask=batch['item_text_masks'], labels=batch['target_ids']).loss
        else:
            loss, _, _ = model.forward_loss(batch, batch['target_item_ids']-1, config['lambda_item'])
        (loss / len(indices)).backward()
    for i in indices:
        backward(i)
    reference = [p.grad.detach().cpu().clone() for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    for i in indices[rank::world]:
        backward(i)
    sum_gradients(model, world)
    numerator = denominator = 0.
    for p, expected in zip((p for p in model.parameters() if p.requires_grad), reference):
        actual = p.grad.detach().cpu()
        numerator += float((actual.double()-expected.double()).square().sum())
        denominator += float(expected.double().square().sum())
    relative = (numerator/max(denominator, 1e-30))**.5
    assert relative < 1e-5, relative
    model.zero_grad(set_to_none=True)
    return dict(state='PASSED', relative_gradient_l2=relative, dropout_disabled=True, microbatches=3)


def profile_worker(rank, uuids, out_string, domain, arm):
    out = Path(out_string)
    runtime(rank, uuids, out/(arm+'.rendezvous'))
    world = len(uuids)
    config = read(BASE_CONFIG)
    data = load_inputs(domain)
    reports, parent = {}, None
    for stage in ('gram', 'c1'):
        seed_all(2023)
        model, original = build_model(data, stage, config, torch.device('cuda'), parent)
        collator = original if arm == 'single_original' else CachedCollator(original)
        dataset = data['dataset' if stage == 'gram' else 'c1_dataset']
        equivalence = real_gradient_check(model, collator, dataset, stage, config, rank, world)
        optimizer, scheduler = make_optimizer(model, stage, config, total_updates(data, stage, config))
        windows = list(update_windows(epoch_order(len(dataset), 2023, 1), 16, 8))[:5]
        longest = sorted(range(len(dataset)), key=lambda i: len(dataset.rows[i]['history']), reverse=True)[:128]
        stress = list(update_windows(longest, 16, 8))[0]
        records = []
        seed_all(2023+rank)
        torch.cuda.reset_peak_memory_stats()
        for j, batches in enumerate([stress]+windows):
            if world > 1:
                dist.barrier()
            torch.cuda.synchronize()
            start = time.monotonic()
            if arm == 'single_original':
                result = optimizer_update(model, optimizer, scheduler, collator, dataset, batches, stage, config, torch.device('cuda'), stress=j == 0)
            else:
                result = parallel_update(model, optimizer, scheduler, collator, dataset, batches, stage, config, torch.device('cuda'), rank, world, stress=j == 0)
            torch.cuda.synchronize()
            duration = torch.tensor(time.monotonic()-start, device='cuda')
            if world > 1:
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            result.update(seconds=float(duration), stress=j == 0)
            records.append(result)
            if rank == 0:
                write(out/'status.json', dict(state='PROFILING', arm=arm, stage=stage, window=j, seconds=float(duration), updated_at=now()))
                print(arm, stage, j, float(duration), flush=True)
        peaks = torch.tensor([torch.cuda.max_memory_allocated()/2**20, torch.cuda.max_memory_reserved()/2**20], device='cuda')
        if world > 1:
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        reports[stage] = dict(gradient_check=equivalence, records=records, peak_allocated_mib=float(peaks[0]),
            peak_reserved_mib=float(peaks[1]), mean_seconds=float(np.mean([r['seconds'] for r in records[1:]])),
            projected_hours=max(r['seconds'] for r in records[1:])*total_updates(data, stage, config)/3600)
        if stage == 'gram':
            parent = cpu_tree(model.state_dict())
        del model, original, collator, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        write(out/(arm+'.json'), dict(state='PASSED', domain=domain, world_size=world, gpus=uuids,
            reports=reports, projected_hours=sum(r['projected_hours'] for r in reports.values()),
            input_sha256=data['manifest'], profile_weights_discarded=True, test_read=False))
    if world > 1:
        dist.destroy_process_group()


def check(out):
    out.mkdir(parents=True, exist_ok=False)
    config = read(BASE_CONFIG)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    results = {}
    for domain in ('Toys', 'Beauty'):
        data = load_inputs(domain)
        model, collator = build_model(data, 'gram', config, torch.device('cpu'))
        results[domain] = collation_check(data, collator)
        del model, collator, data
        gc.collect()
    write(out/'collation.json', dict(state='PASSED', domains=results,
        sources={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), Path(__file__).with_name('lime_parallel_core.py'))}))
    print(results, flush=True)


def gpu_uuids(indices):
    result = []
    for index in indices.split(','):
        raw = subprocess.check_output(['nvidia-smi', '--id='+index, '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
        uuid, free = [s.strip() for s in raw.split(',')]
        assert int(free) >= 28000, f'GPU {index} has {free} MiB free; need 28000 for bounded probe'
        result.append(uuid)
    assert len(set(result)) == len(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'profile'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--domain', choices=('Toys', 'Beauty'), default='Toys')
    parser.add_argument('--gpus', default='4,5')
    parser.add_argument('--parallel-only', action='store_true')
    args = parser.parse_args()
    out = ROOT/args.out
    if args.action == 'check':
        check(out)
        return
    out.mkdir(parents=True, exist_ok=False)
    uuids = gpu_uuids(args.gpus)
    write(out/'protocol.json', dict(domain=args.domain, gpus=uuids, global_batch=128,
        microbatch=16, tf32=True, dropout_streams_change_with_world_size=True,
        no_bitwise_equivalence_claim=True, new_recommendation_metrics=False,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), Path(__file__).with_name('lime_parallel_core.py'))}))
    arms = [] if args.parallel_only else [('single_original', uuids[:1]), ('single_cached', uuids[:1])]
    arms.append(('parallel_cached', uuids))
    for arm, devices in arms:
        mp.spawn(profile_worker, args=(devices, str(out), args.domain, arm), nprocs=len(devices), join=True)
    write(out/'summary.json', dict(state='PASSED', domain=args.domain,
        arms={arm: read(out/(arm+'.json')) for arm, _ in arms}, completed_at=now()))


if __name__ == '__main__':
    main()
