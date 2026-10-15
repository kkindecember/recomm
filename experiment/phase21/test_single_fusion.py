"""Bounded CPU integration check using eight existing ranker_fit rows."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from experiment.phase21 import single_fusion as S


def tests():
    checks = S.C.tests()
    source = S.SOURCE/'Toys'
    samples = S.read(source/'samples.json')
    ix = [j for j, s in enumerate(samples) if s['role'] == 'ranker_fit'][:8]
    raw = np.load(source/'experts.npy', mmap_mode='r')
    counts = np.load(source/'frequencies.npy')
    values = [S.C.features(S.C.retained(raw[j]), samples[j]['full_history'], counts) for j in ix]
    u = np.stack([v[0] for v in values]); c = np.stack([v[1] for v in values])
    data = SimpleNamespace(domain='Toys', n=8, v=raw.shape[-1], user=u, item=c,
        targets=np.asarray([samples[j]['target']-1 for j in ix], dtype=np.int64), folds=np.arange(8)%5,
        samples=[samples[j] for j in ix], m1=c.mean(1, dtype=np.float64), m2=np.square(c.astype(np.float64)).mean(1))
    data.norm = lambda jj: S.Data.norm(data, jj)
    mean, std = data.norm(np.arange(8))
    expanded = np.concatenate((np.broadcast_to(u[:, None, :], (8, data.v, S.C.DU)), c), axis=-1).reshape(-1, S.C.WIDTH)
    np.testing.assert_allclose(mean, expanded.mean(0, dtype=np.float64), atol=1e-6, rtol=1e-6)
    direct_std = expanded.std(0, dtype=np.float64); direct_std[direct_std < 1e-5] = 1.
    np.testing.assert_allclose(std, direct_std, atol=1e-6, rtol=1e-6)
    original_out, original_cfg = S.OUT, S.CFG
    try:
        with tempfile.TemporaryDirectory(prefix='single-fusion-preflight-') as td:
            S.OUT = Path(td)
            S.CFG = dict(original_cfg, epochs=1, checkpoints=[0, 1], batch_size=4)
            for arm in ('user', 'candidate'):
                model, norm, info = S.fit(data, arm, torch.device('cpu'))
                scores, audit = S.export(data, model, norm, arm, torch.device('cpu'))
                ck = torch.load(S.OUT/'Toys'/(arm+'.pt'), map_location='cpu')
                assert scores.shape == (8, data.v) and np.isfinite(scores).all()
                assert ck['selected_epochs'] in (0, 1) and ck['old_fusion_used'] is False
                f, _ = S.C.numpy_score(ck['state_dict'], u[0], c[0], ck['mean'], ck['std'], arm)
                np.testing.assert_allclose(f, scores[0], atol=3e-6, rtol=2e-5)
                assert audit['independent_numpy_full_catalog_users'] == [0, 4, 7]
    finally:
        S.OUT, S.CFG = original_out, original_cfg
    checks.update(real_cache_feature_compatibility=True, exact_full_catalog_normalization=True,
        training_selection_refit_export_roundtrip=True, saved_model_numpy_reload=True,
        preflight_users=8, preflight_catalog_items=int(raw.shape[-1]))
    return checks


if __name__ == '__main__':
    print(json.dumps(tests(), indent=2))
