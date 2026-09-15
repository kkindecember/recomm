"""Train-only cohort construction and differentiable fixed-candidate PCRF loss."""
import hashlib
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


def cohort(records, catalog, training_users, dataset, seed=2023, calibration_per_group=128):
    """Reconstruct historical sample keys from prepared train events, without held-out events."""
    ordered = sorted(training_users, key=lambda u: hashlib.sha256(
        f'{seed}|{dataset}|p0-user-split|{u}'.encode()).hexdigest())
    cut = round(.8 * len(ordered))
    fit_users, calibration_users = set(ordered[:cut]), set(ordered[cut:])
    counts, histories, positions, indexed = Counter(), {}, Counter(), []
    names = catalog['items']
    for index, row in enumerate(records):
        user, history, target = row['user_id'], row['history'], row['target']
        assert 0 < target < len(names) and history and len(history) <= 20
        if user not in histories:
            assert len(history) == 1, ('missing_initial_prefix', user)
            histories[user] = list(history)
            counts[history[0]] += 1
        assert history == histories[user][-20:], ('nonsequential_train_prefix', user, index)
        counts[target] += 1
        positions[user] += 1
        histories[user].append(target)
        if user in training_users:
            indexed.append({'index': index, 'user_id': user, 'position': positions[user],
                            'sample_key': f'{user}:{positions[user]}:{names[target]}',
                            'positive_item': names[target]})
    assert training_users <= set(histories)
    pop_order = sorted(counts, key=lambda item: (-counts[item], names[item]))
    head_ids = set(pop_order[:max(1, int(np.ceil(len(pop_order) * .2)))])
    for row in indexed:
        row['group'] = 'head' if records[row['index']]['target'] in head_ids else 'tail'
    fit = sorted([r for r in indexed if r['user_id'] in fit_users], key=lambda r: (r['user_id'], r['position']))
    pool = sorted([r for r in indexed if r['user_id'] in calibration_users], key=lambda r: hashlib.sha256(
        f'{seed}|{dataset}|gacr-p0-calibration|{r["sample_key"]}'.encode()).hexdigest())
    calibration = []
    for group in ('head', 'tail'):
        selected = [r for r in pool if r['group'] == group][:calibration_per_group]
        assert len(selected) == calibration_per_group, ('insufficient_calibration', group)
        calibration.extend(selected)
    calibration.sort(key=lambda r: r['sample_key'])
    assert not fit_users & calibration_users
    def digest(values):
        return hashlib.sha256('\n'.join(sorted(values)).encode()).hexdigest()
    metadata = {'fit_users': len(fit_users), 'calibration_users': len(calibration_users),
                'fit_samples': len(fit), 'calibration_samples': len(calibration),
                'fit_sample_sha256': digest(r['sample_key'] for r in fit),
                'calibration_sample_sha256': digest(r['sample_key'] for r in calibration),
                'fit_user_sha256': digest(fit_users), 'calibration_user_sha256': digest(calibration_users),
                'fit_calibration_user_overlap': 0, 'test_read': False}
    return fit, calibration, np.asarray([counts[i] for i in range(len(names))], dtype=np.int64), metadata


def standardize_tensor(values):
    """Population standard deviation and original PCRF floor; float64 outside MLP."""
    values = values.to(torch.float64)
    centered = values - values.mean(-1, keepdim=True)
    # Clamp variance before sqrt: finite backward at a constant score vector.
    denominator = centered.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    return centered / denominator


def training_scores(ranker, features, base, cf, frequencies, q1, arm):
    g = (base + ranker(features)).to(torch.float64)
    g_z = standardize_tensor(g)
    if arm == 'B1':
        return g_z
    if arm != 'B2':
        raise ValueError(arm)
    with torch.no_grad():
        # The frozen reproduction environment predates torch.argsort(stable=...).
        # This discrete decision already has stopped gradients; use deployment's sorter.
        order = torch.as_tensor(np.argsort(-g.detach().cpu().numpy(), axis=-1, kind='stable'), device=g.device)
        pop_top10 = frequencies.gather(1, order[:, :10])
        reliability = 1.0 - (pop_top10 <= q1).to(torch.float64).mean(-1, keepdim=True)
        cf_adjusted = standardize_tensor(standardize_tensor(cf) - .5 * standardize_tensor(torch.log1p(frequencies.to(torch.float64))))
    return standardize_tensor(g_z + reliability * cf_adjusted)


def pairwise_losses(scores, positions):
    """Exactly all 49 negatives. Call only on covered prefixes; never insert targets."""
    assert scores.shape[1] == 50 and torch.all((positions >= 0) & (positions < 50))
    positive = scores.gather(1, positions[:, None])
    losses = F.softplus(scores - positive)
    mask = torch.ones_like(losses, dtype=torch.bool)
    mask.scatter_(1, positions[:, None], False)
    return losses.masked_fill(~mask, 0).sum(-1) / 49


def numpy_ranks(g, cf, frequencies, positions, q1):
    """Exact historical PCRF interface, including stable ties after residual sorting."""
    from experiment.phase9.eval_cf0_b3_beamfusion import standardize
    from experiment.phase9.eval_cf0_b4_reliability import rank_matrix
    order = np.argsort(-g, axis=1, kind='stable')
    inverse = np.argsort(order, axis=1)
    positions = np.asarray(positions)
    new_positions = np.where(positions >= 0, inverse[np.arange(len(g)), np.maximum(positions, 0)], -1)
    sorted_g = np.take_along_axis(g, order, 1)
    sorted_cf = np.take_along_axis(cf, order, 1)
    sorted_pop = np.take_along_axis(frequencies, order, 1)
    ranks, reliability = rank_matrix(np.stack([standardize(x) for x in sorted_g]),
        np.stack([standardize(x) for x in sorted_cf]),
        np.stack([standardize(np.log1p(x)) for x in sorted_pop]),
        (sorted_pop[:, :10] <= q1).mean(1), new_positions, (1., .5, 1.))
    raw = np.where(new_positions >= 0, new_positions + 1, 51)
    return raw, ranks, reliability
