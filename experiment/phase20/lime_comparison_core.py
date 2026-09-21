"""Data adaptation and ranking helpers for the pinned LIME implementation."""
import hashlib
import numpy as np


def gate_folds(users, evaluation_indices, count=5, seed=20260920):
    order = sorted(map(int, evaluation_indices), key=lambda i: hashlib.sha256(
        f'{seed}|lime_gate|{users[i]}'.encode()).digest())
    return [np.sort(np.asarray(order[j::count], dtype=np.int64)) for j in range(count)]


def histories_from_bank(bank, users):
    source = {str(u): i for i, u in enumerate(bank['users'])}
    assert len(source) == len(bank['users']) == len(users)
    result = []
    for u in users:
        i = source[str(u)]
        result.append(bank['items'][bank['offsets'][i]:bank['offsets'][i+1]].astype(int).tolist())
    return result


def left_pad(histories, width=50):
    out = np.zeros((len(histories), width), dtype=np.int64)
    for i, h in enumerate(histories):
        tail = h[-width:]
        if tail:
            out[i, -len(tail):] = tail
    return out


def unit_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def semantic_users(histories, embeddings):
    out = np.zeros((len(histories), embeddings.shape[1]), dtype=np.float32)
    for i, h in enumerate(histories):
        w = np.asarray([np.exp(-0.1*(len(h)-1-j)) for j in range(len(h))], dtype=np.float32)
        out[i] = (embeddings[np.asarray(h)-1]*w[:, None]).sum(0)/max(w.sum(), 1e-8)
    return unit_rows(out)
