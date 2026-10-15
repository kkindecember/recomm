"""Independent reconstruction of the CPU deletion screen from frozen inputs."""
import json
import os
from pathlib import Path
import time

import numpy as np

from experiment.phase20.verify_lime_mechanism import digest, calculate, compare

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/lime_expert_ablation_20260921_v1'


def main():
    read = lambda p: json.loads(Path(p).read_text())
    frozen = read(OUT / 'manifest.json')['sha256']
    for path, expected in frozen.items():
        assert digest(ROOT / path) == expected, ('Changed input', path)
    variants = read(OUT / 'config.json')['variants']
    results = {}
    for domain in ('Toys', 'Beauty'):
        folder = OUT / domain
        data = np.load(folder / 'inputs.npz', allow_pickle=False)
        samples = read(folder / 'samples.json')
        predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        summary = read(folder / 'summary.json')
        assert len(predictions) == len(samples) == 512
        ranks = {n: {scope: [] for scope in ('catalog', 'GRAM50', 'guided50')} for n in variants}
        entropy = {n: [] for n in variants}
        count = 0
        for i, (row, sample) in enumerate(zip(predictions, samples)):
            assert row['user_id'] == sample['user_id'] and row['target'] == sample['target'] == data['targets'][i]
            assert row['sample_index'] == sample['sample_index'] == 131 + i
            for name, rule in variants.items():
                selected = rule['experts']
                w = np.zeros(3, dtype=np.float32)
                if rule['equal_weights']:
                    for e in selected:
                        w[e] = np.float32(1 / len(selected))
                elif len(selected) == 3:
                    w = data['weights'][i].copy()
                else:
                    total = np.sum(data['weights'][i, selected])
                    for e in selected:
                        w[e] = data['weights'][i, e] / total
                np.testing.assert_array_equal(w, np.asarray(row['arms'][name]['effective_weights'], dtype=np.float32))
                # Explicit independent scalar-component accumulation in FP32.
                scores = data['experts'][i, 0] * w[0]
                scores = scores + data['experts'][i, 1] * w[1]
                scores = scores + data['experts'][i, 2] * w[2]
                if rule['history_penalty']:
                    scores[np.asarray(sample['full_history_unique']) - 1] -= data['penalty'][i]
                assert np.isfinite(scores).all()
                if name == 'full':
                    np.testing.assert_allclose(scores, data['reference_full'][i], atol=2e-7, rtol=0)
                z = 20. * (scores.astype('float64') - float(scores.max()))
                p = np.exp(z); p /= p.sum()
                entropy[name].append(float(-(p * np.log(np.maximum(p, 1e-300))).sum()))
                for scope in ranks[name]:
                    candidates = (np.arange(1, len(scores) + 1) if scope == 'catalog'
                                  else data['gram_ids' if scope == 'GRAM50' else 'guided_ids'][i])
                    history = set(sample['history'])
                    ids = np.asarray([int(item) for item in candidates if int(item) not in history])
                    order = ids[np.lexsort((ids, -scores[ids - 1]))][:50].tolist()
                    target = sample['target']
                    rank = order.index(target) + 1 if target in order else 51
                    assert row['arms'][name]['scopes'][scope] == dict(items=order, rank=rank)
                    ranks[name][scope].append(rank)
                    count += 1
        for name in ranks:
            for scope, values in ranks[name].items():
                for key, value in calculate(values).items():
                    np.testing.assert_allclose(value, summary['metrics'][name][scope][key], atol=1e-14, rtol=0)
                if name != 'full':
                    for key, value in compare(values, ranks['full'][scope]).items():
                        np.testing.assert_allclose(value, summary['comparisons'][name][scope][key], atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.mean(entropy[name]), summary['catalog_signal_entropy'][name], atol=1e-12, rtol=0)
        results[domain] = dict(users=512, variants=len(variants), predictions_rebuilt=count,
            scores_independently_reconstructed=True, summary_sha256=digest(folder / 'summary.json'))
    result = dict(state='VERIFIED', verified_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), domains=results,
                  frozen_inputs_checked=len(frozen), independent_of_ablation_runner=True, test_read=False)
    temp = OUT / f'verification.{os.getpid()}.tmp'
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(OUT / 'verification.json')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
