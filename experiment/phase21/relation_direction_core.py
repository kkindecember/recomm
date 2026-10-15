"""Target-free relation construction and fold-local linear calibration."""
import hashlib
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp

ARMS = ('base', 'symmetric_recent', 'reverse_recent', 'forward_binary',
        'forward_uniform', 'forward_recent')


def transition_counts(sequences, size, window=3):
    source, target, weight = [], [], []
    for history in sequences:
        for gap in range(1, window + 1):
            for a, b in zip(history[:-gap], history[gap:]):
                if a != b:
                    source.append(a); target.append(b); weight.append(1. / gap)
    out = sparse.coo_matrix((weight, (source, target)), shape=(size, size)).tocsr()
    out.sum_duplicates(); out.sort_indices()
    return out


def normalized_neighbors(counts, k=100):
    out = np.asarray(counts.sum(1)).ravel()
    inc = np.asarray(counts.sum(0)).ravel()
    rows, cols, values = [], [], []
    for i in range(counts.shape[0]):
        start, stop = counts.indptr[i:i+2]
        ids = counts.indices[start:stop]
        weights = counts.data[start:stop] / np.sqrt(out[i] * inc[ids])
        chosen = np.lexsort((ids, -weights))[:k]
        rows.extend([i] * len(chosen)); cols.extend(ids[chosen]); values.extend(weights[chosen])
    return sparse.csr_matrix((values, (rows, cols)), shape=counts.shape)


def history_matrix(histories, size, recent=True):
    rows, cols, values = [], [], []
    for i, history in enumerate(histories):
        h = history[-20:]
        for j, item in enumerate(h, 1):
            rows.append(i); cols.append(item); values.append(j / len(h) if recent else 1.)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(histories), size))


def fold_ids(users, domain):
    return np.array([int.from_bytes(hashlib.sha256(
        f'20260922|clean-benefit-fold|{domain}|{u}'.encode()).digest()[:8], 'big') % 5 for u in users])


def fit_head(x, prior, valid, ids, targets, indices):
    """Only selected training labels/features may affect fitting or scaling."""
    indices = np.asarray(indices)
    matches = (ids[indices] == targets[indices, None]) & valid[indices]
    used = indices[matches.any(1)]
    assert len(used)
    y = matches[matches.any(1)].argmax(1)
    mean = x[used][valid[used]].mean(0)
    std = np.maximum(x[used][valid[used]].std(0), 1e-5)
    z = (x[used] - mean) / std
    p, v = prior[used], valid[used]

    def objective(w):
        logits = np.where(v, p + z @ w, -np.inf)
        partition = logsumexp(logits, axis=1)
        loss = np.mean(partition - logits[np.arange(len(y)), y]) + .005 * (w @ w)
        residual = np.exp(logits - partition[:, None])
        residual[np.arange(len(y)), y] -= 1.
        grad = np.einsum('nc,ncd->d', residual, z) / len(y) + .01 * w
        return loss, grad

    result = minimize(objective, np.zeros(x.shape[2]), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=200, gtol=1e-6, ftol=1e-10))
    assert np.isfinite(result.fun) and np.isfinite(result.x).all()
    if not result.success:
        raise RuntimeError(f'Calibration did not converge: {result.message}')
    return dict(mean=mean, std=std, coef=result.x, used=used, fit_indices=indices,
                iterations=int(result.nit), objective=float(result.fun),
                gradient_max=float(np.abs(result.jac).max()), message=str(result.message))


def predict_head(head, x, prior):
    return prior + ((x - head['mean']) / head['std']) @ head['coef']
