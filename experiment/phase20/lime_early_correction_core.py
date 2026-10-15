"""Feature-availability ablation. The correction API never receives a prior."""
import numpy as np
import torch

from experiment.phase20.lime_clean_benefit_core import new_head
from experiment.phase20.source_competition_core import row_losses, target_positions

GROUPS = {'full': tuple(range(10)), 'early': (2, 3, 4, 5, 8, 9),
          'late': (0, 1, 6, 7)}


def standardized_slots(features, mean, std, columns):
    """Only supplied features are read; unavailable slots are constant zeros."""
    assert features.shape[-1] == len(columns)
    assert np.isfinite(features).all()
    z = np.zeros((*features.shape[:-1], 10), dtype=np.float32)
    z[..., list(columns)] = (features - mean) / std
    return z


def fit(data, indices, group, seed):
    indices = np.asarray(indices)
    columns = GROUPS[group]
    positions = target_positions(np.where(data['valid'][indices], data['ids'][indices], 0),
                                 data['targets'][indices])
    used = indices[positions >= 0]
    assert len(used)
    valid = data['valid'][used]
    raw = data['x'][used][..., list(columns)]
    mean = raw[valid].mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(raw[valid].std(0, dtype=np.float64), 1e-5).astype(np.float32)
    x = torch.from_numpy(standardized_slots(raw, mean, std, columns))
    v, p = torch.from_numpy(valid), torch.from_numpy(positions[positions >= 0])
    prior = torch.from_numpy((10 * data['mixed'][used]).astype(np.float32))
    model = new_head('mlp', seed)
    opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.001)
    curve = []
    for step in range(30):
        opt.zero_grad()
        total = 0.
        for start in range(0, len(used), 128):
            sl = slice(start, start + 128)
            loss = row_losses(model(x[sl], prior[sl]), v[sl], p[sl]).sum() / len(used)
            assert torch.isfinite(loss)
            loss.backward()
            total += float(loss.detach())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
        curve.append(total)
    return dict(state_dict=model.state_dict(), mean=mean, std=std, columns=columns,
                group=group, seed=seed, used=used, fit_indices=indices,
                optimizer_steps=30, curve=curve)


def correction(saved, available_features, chunk=128):
    """Pointwise item correction; early accepts exactly 6 pre-generation values."""
    model = new_head('mlp', saved['seed'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(available_features), chunk):
            raw = available_features[start:start + chunk]
            z = torch.from_numpy(standardized_slots(raw, saved['mean'], saved['std'], saved['columns']))
            result.append((2 * torch.tanh(model.network(z).squeeze(-1))).numpy())
    return np.concatenate(result)


def add_prior(mixed, delta):
    """Late evaluation wrapper, deliberately separate from correction inference."""
    return (10 * mixed).astype(np.float32) + delta
