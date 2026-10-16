"""TiSASRec attention, with causal prefix-only time preprocessing.

Independent PyTorch implementation of Li, Wang & McAuley, WSDM 2020:
https://doi.org/10.1145/3336191.3371786
Equations checked against the authors' TensorFlow model/modules.py:
https://github.com/JiachengLi1995/TiSASRec

Both key AND value time/absolute-position embeddings are implemented. This is
not a scalar time bias added to the legacy SASRec block. Differences from the
released recipe: full-catalog CE, matched project hyperparameters, and each
query's normalization scale uses only its visible prefix (never the target or
later timestamps). The order control replaces dates by equally spaced events.
"""
import math
import numpy as np
import torch
from torch import nn


def intervals(items, days, cap=256, order_only=False):
    """Return B,Q,K interval IDs; each query uses its own prefix time scale.

    Round normalized coordinates as in the reference preprocessing, then take
    their difference. All inputs are left-padded, chronological visible events.
    Padded/future entries are zero and are independently masked by attention.
    """
    items, days = np.asarray(items), np.asarray(days)
    assert items.ndim == 2 and days.shape == items.shape
    valid = items != 0
    assert valid.any(1).all()
    assert not (valid[:, :-1] & ~valid[:, 1:]).any(), 'Expected left padding'
    if order_only:
        days = np.broadcast_to(np.arange(items.shape[1]), items.shape)
    gaps = np.diff(days, axis=1)
    pairs = valid[:, :-1] & valid[:, 1:]
    assert (gaps[pairs] >= 0).all()
    huge = 2**50
    positive = np.where(pairs & (gaps > 0), gaps, huge)
    scale = np.minimum.accumulate(np.pad(positive, ((0, 0), (1, 0)), constant_values=huge), axis=1)
    scale = np.where(scale == huge, 1, scale)
    origin = days[np.arange(len(days)), valid.argmax(1)]
    coordinates = np.rint((days[:, None, :] - origin[:, None, None]) / scale[:, :, None])
    query_coordinates = np.diagonal(coordinates, axis1=1, axis2=2)
    values = np.abs(query_coordinates[:, :, None] - coordinates).clip(0, cap)
    allowed = valid[:, :, None] & valid[:, None, :] & np.tri(items.shape[1], dtype=bool)
    return np.where(allowed, values, 0).astype(np.int64)


class TimeAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        assert width % heads == 0
        self.heads, self.head_width = heads, width // heads
        self.query, self.key, self.value = [nn.Linear(width, width) for _ in range(3)]
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, x, pos_key, pos_value, time_key, time_value, valid):
        b, length, width = x.shape
        def split(value):
            return value.reshape(b, length, self.heads, self.head_width).permute(0, 2, 1, 3)
        q, k, v = split(self.query(query)), split(self.key(x)), split(self.value(x))
        pk, pv = split(pos_key.expand(b, -1, -1)), split(pos_value.expand(b, -1, -1))
        tk = time_key.reshape(b, length, length, self.heads, self.head_width).permute(0, 3, 1, 2, 4)
        tv = time_value.reshape(b, length, length, self.heads, self.head_width).permute(0, 3, 1, 2, 4)
        logits = (q @ (k + pk).transpose(-1, -2) + torch.einsum('bhqd,bhqkd->bhqk', q, tk)) / math.sqrt(self.head_width)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        mask = causal[None, None] & valid[:, None, None, :]
        # A padding query may attend to its own slot, then is zeroed. Real
        # queries can never attend to padding or future keys.
        diagonal = torch.eye(length, dtype=torch.bool, device=x.device)[None, None]
        mask = mask | (diagonal & ~valid[:, None, :, None])
        weights = self.dropout(logits.masked_fill(~mask, -torch.inf).softmax(-1))
        output = weights @ (v + pv) + torch.einsum('bhqk,bhqkd->bhqd', weights, tv)
        return output.transpose(1, 2).reshape(b, length, width) * valid.unsqueeze(-1)


class TimeBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width, eps=1e-8)
        self.attention = TimeAttention(width, heads, dropout)
        self.norm_ff = nn.LayerNorm(width, eps=1e-8)
        self.ff = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(width, width), nn.Dropout(dropout))

    def forward(self, x, *args):
        q = self.norm_attention(x)
        x = q + self.attention(q, x, *args)
        x = self.norm_ff(x)
        return x + self.ff(x)


class TiSASRec(nn.Module):
    def __init__(self, num_items, hidden=64, maxlen=50, num_layers=2, num_heads=2, dropout=.2, time_span=256):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, hidden, padding_idx=0)
        self.pos_key, self.pos_value = [nn.Embedding(maxlen, hidden) for _ in range(2)]
        self.time_key, self.time_value = [nn.Embedding(time_span + 1, hidden) for _ in range(2)]
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([TimeBlock(hidden, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden, eps=1e-8)
        # Xavier matrices, zero linear biases, unit LayerNorm as in a standard
        # reference initialization. Identical for real-time and order control.
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, items, relations):
        valid = items != 0
        positions = torch.arange(items.shape[1], device=items.device)[None]
        x = self.dropout(self.item_emb(items) * math.sqrt(self.item_emb.embedding_dim)) * valid.unsqueeze(-1)
        pk, pv = self.dropout(self.pos_key(positions)), self.dropout(self.pos_value(positions))
        tk, tv = self.dropout(self.time_key(relations)), self.dropout(self.time_value(relations))
        for layer in self.layers:
            x = layer(x, pk, pv, tk, tv, valid) * valid.unsqueeze(-1)
        return self.norm(x) * valid.unsqueeze(-1)

    def predict(self, items, relations):
        return self(items, relations)[:, -1] @ self.item_emb.weight.T


def tests():
    torch.set_num_threads(2)
    torch.manual_seed(71)
    items = np.array([[0, 1, 2, 3, 4], [0, 0, 0, 5, 6]])
    dates = np.array([[0, 100, 100, 106, 107], [0, 0, 0, 20, 30]])
    r = intervals(items, dates)
    # The future one-day gap cannot change the earlier six-day normalization.
    np.testing.assert_array_equal(r[0, 3], [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(r[0, 4], [0, 7, 7, 1, 0])
    np.testing.assert_array_equal(r, intervals(items, dates + (items != 0)*4000))
    np.testing.assert_array_equal(intervals(items, dates, order_only=True), intervals(items, dates*100, order_only=True))
    assert intervals(items, dates, cap=2).max() <= 2
    # Compare every query against independent scalar prefix preprocessing.
    for b in range(len(items)):
        for i in np.flatnonzero(items[b]):
            positions = np.flatnonzero(items[b, :i+1])
            prefix = dates[b, positions].tolist()
            positive = [y-x for x, y in zip(prefix, prefix[1:]) if y > x]
            scale = min(positive) if positive else 1
            coords = [round((t-prefix[0])/scale) for t in prefix]
            for j, coord in zip(positions, coords):
                assert r[b, i, j] == min(256, abs(coords[-1]-coord))
    model = TiSASRec(15, hidden=8, maxlen=5, num_heads=2, dropout=0).eval()
    x, rel = torch.from_numpy(items), torch.from_numpy(r)
    out = model(x, rel)
    changed_items, changed_days = items.copy(), dates.copy()
    changed_items[0, -1], changed_days[0, -1] = 9, 600
    changed = model(torch.from_numpy(changed_items), torch.from_numpy(intervals(changed_items, changed_days)))
    torch.testing.assert_close(out[0, :-1], changed[0, :-1], rtol=0, atol=0)
    assert torch.count_nonzero(out[x == 0]) == 0
    # Independent scalar multihead attention computation for valid query slots.
    attn = model.layers[0].attention
    q, k = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    pk, pv, tk, tv = torch.randn(1, 5, 8), torch.randn(1, 5, 8), torch.randn(2, 5, 5, 8), torch.randn(2, 5, 5, 8)
    got = attn(q, k, pk, pv, tk, tv, x != 0).detach().numpy()
    qv, kv, vv = [z.detach().numpy() for z in (attn.query(q), attn.key(k), attn.value(k))]
    for b in range(2):
        for i in np.flatnonzero(items[b]):
            js = np.flatnonzero(items[b, :i+1])
            for h in range(2):
                part = slice(h*4, (h+1)*4)
                logits = np.array([np.dot(qv[b, i, part], kv[b, j, part]+pk[0, j, part].numpy()+tk[b, i, j, part].numpy())/2 for j in js])
                weights = np.exp(logits-logits.max()); weights /= weights.sum()
                expected = sum(w*(vv[b, j, part]+pv[0, j, part].numpy()+tv[b, i, j, part].numpy()) for w, j in zip(weights, js))
                np.testing.assert_allclose(got[b, i, part], expected, atol=5e-7, rtol=2e-6)
    model.train()
    loss = torch.nn.functional.cross_entropy(model.predict(x, rel), torch.tensor([7, 8]), label_smoothing=.1)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.time_key.weight.grad.abs().sum() > 0 and model.time_value.weight.grad.abs().sum() > 0
    return dict(state='PASSED', prefix_normalization=True, future_timestamp_invariance=True,
                attention_scalar_reference=True, padding_finite=True, time_key_value_gradients=True,
                order_control_ignores_dates=True, target_dates_not_in_api=True)
