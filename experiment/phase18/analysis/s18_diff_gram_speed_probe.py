"""Time unchanged Beauty predictions on fixed saved cases; never update weights."""

import argparse
import json
import os
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import Subset

from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase18.protocol.s18_diff_gram_fixed import FreshBackend
from experiment.phase18.protocol.s18_screen_resume import load_checkpoint
from experiment.phase18.protocol import s18_diff_gram as original

ROOT = original.ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu or not torch.cuda.is_available():
        raise RuntimeError('Expected the declared visible GPU')
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=False)
    def timeout(signum, frame):
        raise TimeoutError('Fixed-case timing exceeded 10 minutes')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL,600)
    try:
        base = ROOT/'artifacts/phase18/diff_gram/beauty/confirm_v1'
        checkpoint = base/'best_trend.pt'
        checkpoint_sha = sha256(checkpoint)
        saved = load_checkpoint(checkpoint)
        if saved['epoch'] != 1 or saved['config']['reuse_cross_attention_cache']:
            raise ValueError('Timing contract is saved epoch1 with original generation')
        config = saved['config']
        original.seed_everything(config['seed'])
        backend = FreshBackend(config,8,torch.device('cuda:0'))
        backend.model.load_state_dict(saved['model'],strict=True)
        backend.model.eval()
        del saved
        metadata = json.loads((base/'trend_subset_stage_epoch_01.json').read_text())
        predictions = base/'trend_subset_stage_epoch_01.jsonl'
        if sha256(predictions) != metadata['predictions_sha256']:
            raise ValueError('Saved prediction identity changed')
        expected = {str(row['user_id']):row for row in
                    (json.loads(line) for line in predictions.read_text().splitlines())}
        cohort = json.loads((base/'trend_cohort.json').read_text())
        records = {str(row['user_id']):i for i,row in enumerate(backend.records)}
        users = cohort['user_ids'][:12]
        lengths = sorted(users,key=lambda u:len(backend.records[records[u]]['history']))
        longest = max(expected,key=lambda u:len(backend.records[records[u]]['history']))
        users = list(dict.fromkeys(users+[longest]))
        loader = iter(backend.loader(Subset(backend.validation,[records[u] for u in users]),1))
        timings = []
        wall_start = time.monotonic()
        for user in users:
            start = time.monotonic()
            batch = next(loader)
            wait_seconds = time.monotonic()-start
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            output = backend.predict(batch)[0]
            torch.cuda.synchronize()
            seconds = time.monotonic()-start
            key,gold,items,scores = output
            old = expected[user]
            row = dict(user_id=user,history_length=len(backend.records[records[user]]['history']),
                data_wait_seconds=wait_seconds,generation_seconds=seconds,
                targets_equal=(key==user and gold==old['gold_item_id']),
                items_equal=items==old['ranked_item_ids'],scores_equal=scores==old['sequence_scores'],
                max_score_difference=max(abs(a-b) for a,b in zip(scores,old['sequence_scores'])),
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            timings.append(row)
            with (out/'cases.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if sha256(checkpoint) != checkpoint_sha:
            raise ValueError('Timing checkpoint changed during the probe')
        elapsed = time.monotonic()-wall_start
        result = dict(state='COMPLETED',gpu_uuid=args.gpu,checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=checkpoint_sha,checkpoint_epoch=1,users=len(users),
            seconds_per_user=elapsed/len(users),wall_seconds=elapsed,
            generation_seconds_per_user=sum(r['generation_seconds'] for r in timings)/len(users),
            data_wait_seconds_per_user=sum(r['data_wait_seconds'] for r in timings)/len(users),
            exact_output_identity=all(r['targets_equal'] and r['items_equal'] and r['scores_equal'] for r in timings),
            precision='float32_tf32_off',beam=50,optimizer_updates=0,test_read=False,
            source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha256(Path(__file__))},
            note='Fixed engineering cases include a longest history; timing is not a population ETA.')
        write_json(out/'result.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as error:
        write_json(out/'failure.json',dict(error=repr(error),automatic_retry=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__ == '__main__':
    main()
