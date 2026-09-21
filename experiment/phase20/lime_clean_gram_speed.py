"""Bounded implementation-speed comparison, no validation or recommendation training."""
import gc
import os
import signal
import subprocess
import time

import numpy as np
import torch

from experiment.phase20.lime_clean_gram_core import (
    ROOT, read, write, sha, now, load_inputs, build_model, cpu_tree, seed_all,
    epoch_order, update_windows, make_optimizer, optimizer_update, total_updates,
)
from experiment.phase20.guided_decoding_job import gpu_candidates


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/phase20/lime_clean_gram_speed_20260920_v1')
    parser.add_argument('--mode', choices=('strict_comparison', 'native_baseline'), default='strict_comparison')
    args = parser.parse_args()
    config = read(ROOT / 'experiment/phase20/lime_clean_gram_20260920_v1.json')
    out = ROOT / args.output
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    def status(state, **extra):
        write(out / 'status.json', dict(state=state, updated_at=now(), pid=os.getpid(),
              seconds=time.monotonic()-start, profile_only=True, test_read=False, **extra))
    def timeout(*_):
        raise TimeoutError('20-minute implementation-speed check budget')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1200)
    arms = ([('checkpoint_fp32', True, False), ('checkpoint_tf32', True, True), ('native_tf32', False, True)]
            if args.mode == 'strict_comparison' else [('checkpoint_tf32', True, True), ('native_tf32', False, True)])
    write(out / 'protocol.json', dict(arms=[a[0] for a in arms], numerical_reference=arms[0][0], mode=args.mode,
        seed=2023, batch_size=16, accumulation=8, domain='Toys', random_windows=2,
        numerical_relative_loss_tolerance=.005, numerical_relative_gradient_l2_tolerance=.02,
        minimum_free_mib=30720, allocator_limit_mib=25600,
        motivation='CPU collation costs only 1.5-1.7 seconds per 15-16 second update. Torch1.11 native TF32 and no checkpointing were not disabled by old GRAM runner.',
        source_sha256=sha(__file__), new_effectiveness_metrics=False))
    try:
        status('WAITING_GPU')
        while True:
            available = [g for g in gpu_candidates() if g['free_mib'] >= 30720]
            if available:
                gpu = available[0]
                break
            if time.monotonic()-start > 600:
                status('RESOURCE_WAIT_EXPIRED')
                return
            time.sleep(20)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(25600/(torch.cuda.get_device_properties(0).total_memory/2**20))
        data = load_inputs('Toys')
        dataset = data['dataset']
        groups = list(update_windows(epoch_order(len(dataset), 2023, 1), 16, 8))[:2]
        longest = sorted(range(len(dataset)), key=lambda i: len(dataset.rows[i]['history']), reverse=True)[:128]
        stress = list(update_windows(longest, 16, 8))[0]
        reference_grad = None
        reference_loss = None
        reports = {}
        for arm, checkpointing, tf32 in arms:
            status('RUNNING', arm=arm, gpu=gpu)
            c = dict(config, encoder_checkpointing=checkpointing)
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
            seed_all(2023)
            model, collator = build_model(data, 'gram', c, torch.device('cuda'))
            # Compare loss and full parameter gradients before any update, with identical dropout RNG.
            batch = collator([dataset[i] for i in groups[0][0]])
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            seed_all(991)
            model.train()
            loss = model(input_ids=batch['item_text_ids'], attention_mask=batch['item_text_masks'], labels=batch['target_ids']).loss
            loss.backward()
            current_loss = float(loss.detach())
            grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
            assert all(torch.isfinite(g).all() for g in grads.values())
            if reference_grad is None:
                reference_grad, reference_loss = grads, current_loss
            difference = sum(float((grads[k].double()-reference_grad[k].double()).square().sum()) for k in grads)
            magnitude = sum(float(g.double().square().sum()) for g in reference_grad.values())
            relative_gradient = (difference/max(magnitude, 1e-30))**.5
            relative_loss = abs(current_loss-reference_loss)/abs(reference_loss)
            numerical_pass = relative_loss <= .005 and relative_gradient <= .02
            assert numerical_pass, f'Unexpected numerical change: {arm} {relative_loss} {relative_gradient}'
            model.zero_grad(set_to_none=True)
            del grads, batch, loss
            gc.collect()
            torch.cuda.empty_cache()
            optimizer, scheduler = make_optimizer(model, 'gram', c, total_updates(data, 'gram', c))
            seed_all(2023)
            records = []
            torch.cuda.reset_peak_memory_stats()
            for index, batches in enumerate([stress]+groups):
                torch.cuda.synchronize()
                started = time.monotonic()
                r = optimizer_update(model, optimizer, scheduler, collator, dataset, batches, 'gram', c,
                                     torch.device('cuda'), stress=(index == 0))
                torch.cuda.synchronize()
                r.update(seconds=time.monotonic()-started, stress=(index == 0))
                records.append(r)
                status('RUNNING', arm=arm, completed_windows=index+1, gpu=gpu)
            reports[arm] = dict(records=records, relative_loss=relative_loss, relative_gradient_l2=relative_gradient,
                numerical_check_passed=numerical_pass, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                conservative_gram_hours=max(r['seconds'] for r in records if not r['stress'])*total_updates(data,'gram',c)/3600)
            write(out / f'{arm}.json', reports[arm])
            del model, optimizer, scheduler, collator
            gc.collect()
            torch.cuda.empty_cache()
        for p, h in data['manifest'].items():
            assert sha(ROOT/p) == h, p
        write(out/'summary.json',dict(state='COMPLETED', gpu=gpu, reports=reports,
              input_sha256=data['manifest'], source_sha256=sha(__file__), seconds=time.monotonic()-start,
              test_read=False, development_labels_read=False, formal_training_started=False))
        status('COMPLETED', gpu=gpu)
    except BaseException as error:
        import traceback
        (out/'failure.txt').write_text(traceback.format_exc())
        status('FAILED', error=repr(error))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
