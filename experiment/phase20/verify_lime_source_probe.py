"""Independent saved-ranking, full-denominator and checkpoint audit."""
import json
from pathlib import Path

import numpy as np
import torch

from experiment.phase20.guided_decoding_job import ROOT, read, sha, write, now


def sorted_ranks(scores, ids, targets):
    result = []
    for row, items, target in zip(scores, ids, targets):
        order = np.lexsort((items, -row))
        found = np.flatnonzero((items[order] == target) & np.isfinite(row[order]))
        result.append(int(found[0])+1 if len(found) else 2**30)
    return np.asarray(result, dtype=np.int64)


def main():
    torch.set_num_threads(4)
    root = ROOT/'artifacts/phase20/lime_source_probe_20260921_v1'
    summary = read(root/'summary.json')
    assert summary['state'] == 'COMPLETED'
    for group in ('input_sha256','output_sha256'):
        for path, digest in summary[group].items():
            assert sha(ROOT/path) == digest, path
    fields = predictions = models = 0
    for domain in ('Toys','Beauty'):
        with np.load(ROOT/f'artifacts/phase20/source_competition_20260920_v2/{domain}/development.npz') as f:
            ids, targets, ing, inl = (f[k] for k in ('ids','targets','ing','inl'))
        with np.load(root/domain/'all_ranks.npz') as f:
            users, names, values = f['users'], f['arm_names'], f['ranks']
            groups = {k[5:]:f[k] for k in f.files if k.startswith('mask_')}
        rr = dict(zip(names, values))
        for p in (root/domain).glob('*.predictions.npz'):
            key = p.name.removesuffix('.predictions.npz')
            with np.load(p) as f:
                score, saved = f['scores'], f['ranks']
                np.testing.assert_array_equal(users, f['users'])
            calculated = sorted_ranks(score, ids, targets)
            np.testing.assert_array_equal(calculated, saved)
            np.testing.assert_array_equal(calculated, rr[key])
            predictions += len(saved)
            if key.startswith('full_mlp.'):
                for source, mask in (('gram',ing),('lime',inl)):
                    calculated = sorted_ranks(np.where(mask,score,-np.inf), ids, targets)
                    np.testing.assert_array_equal(calculated, rr[f'{key}.{source}_only'])
                    predictions += len(calculated)
        result = summary['domains'][domain]
        for key, ranking in rr.items():
            for g, mask in groups.items():
                for k in (5,10,20,50):
                    selected = ranking[mask]
                    hit = selected <= k
                    expected = dict(H=float(hit.sum()/len(selected)),
                        N=float(np.sum(1/np.log2(selected[hit]+1))/len(selected)))
                    for metric, value in expected.items():
                        assert abs(value-result['metrics'][key][g][f'{metric}{k}']) < 1e-12
                        fields += 1
        for source, rows in result['candidate_restriction'].items():
            for row in rows:
                key = f'full_mlp.{row["seed"]}'
                a, b = rr[key], rr[f'{key}.{source}_only']
                gained = int(((a<=10)&(b>10)).sum())
                lost = int(((a>10)&(b<=10)).sum())
                assert gained == row['top10_gained'] and lost == row['top10_lost']
                assert gained-lost == row['top10_net']
                contribution = row['full_population_N10_contributions']
                assert abs(sum(contribution[g] for g in ('gram_only_target','lime_only_target','both_target','uncovered'))-row['delta_N10']) < 1e-12
        for p in (root/domain).glob('*.pt'):
            checkpoint = torch.load(p,map_location='cpu')
            assert len(checkpoint['curve']) == 30 and np.isfinite(checkpoint['curve']).all()
            assert not set(checkpoint['fit_indices']) & set(checkpoint['evaluation_indices'])
            assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())
            models += 1
    assert models == 100
    result = dict(state='VERIFIED', completed_at=now(), model_checkpoints=models,
        reconstructed_rankings=predictions, reconstructed_metric_fields=fields,
        all_input_output_hashes_match=True, top10_gain_loss_and_group_contributions_match=True,
        all_recorded_losses_and_weights_finite=True, test_read=False,
        verifier_sha256=sha(__file__), development_diagnostic_only=True)
    write(root/'independent_verification.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
