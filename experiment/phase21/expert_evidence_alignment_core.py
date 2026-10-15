"""Exact score-margin attribution at fixed inputs; never accepts a target label."""
import numpy as np

EXPERTS = ('sas', 'cf', 'text')


def vote(margin, tolerance=1e-6):
    return 'A' if margin > tolerance else 'B' if margin < -tolerance else 'tie'


def normalized(scores):
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


def summarize_contributions(values):
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    mass = float(np.abs(values).sum())
    tol = max(1e-12, mass * 1e-6)
    pos, neg = int((values > tol).sum()), int((values < -tol).sum())
    # Delete only one contribution, keeping the other event weights unchanged.
    flip = bool(abs(total) > tol and np.any((total - values) * np.sign(total) < -tol))
    return dict(raw_margin=total, absolute_mass=mass, positive_events=pos,
                negative_events=neg, conflicting_events=bool(pos and neg),
                no_contribution=mass <= 1e-12,
                cancellation=None if mass <= 1e-12 else float(1 - abs(total) / mass),
                single_event_flip=flip)


class Attribution:
    def __init__(self, item_ids, embeddings, cf_model):
        assert len(item_ids) == len(embeddings)
        self.emb = np.asarray(embeddings, dtype=np.float32)
        self.unit = self.emb / (np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-8)
        self.index = {v: i + 1 for i, v in enumerate(item_ids)}
        assert cf_model['training_only'] and cf_model['scoring_history_length'] == 20
        self.cf = {self.index[h]: {self.index[v['item_id']]: float(np.float32(v['score']))
                   for v in rows if v['item_id'] in self.index}
                   for h, rows in cf_model['neighbors'].items() if h in self.index}

    def pair(self, a, b, history):
        assert history
        ids = np.asarray(history, dtype=int) - 1
        w = np.asarray([np.exp(-.1 * (len(history) - 1 - j))
                        for j in range(len(history))], dtype=np.float32)
        # Use original float32 pooling denominator and unnormalized history vectors.
        mean = (self.emb[ids] * w[:, None]).sum(0) / max(w.sum(), 1e-8)
        denom = float(max(w.sum(), 1e-8)) * float(np.linalg.norm(mean) + 1e-8)
        pair_units = self.unit[[a - 1, b - 1]].astype(np.float64)
        text = (self.emb[ids].astype(np.float64) @ pair_units.T) * w[:, None] / denom
        cf = np.zeros((len(history), 2), dtype=np.float64)
        offset = max(0, len(history) - 20)
        for j, h in enumerate(history[offset:]):
            weight = (j + 1) / min(len(history), 20)
            cf[offset + j] = [weight * self.cf.get(h, {}).get(c, 0.) for c in (a, b)]
        return dict(cf=cf, text=text)


def support_card(history, values, titles):
    unique = list(dict.fromkeys(history))
    rows = []
    for j, h in enumerate(unique, 1):
        ix = [k for k, v in enumerate(history) if v == h]
        ab = values[ix].sum(0)
        rows.append(dict(ref=f'H{j}', item=h, title=titles[h],
                         positions=[k + 1 for k in ix], a=float(ab[0]), b=float(ab[1]),
                         difference=float(ab[0] - ab[1])))
    return rows


def top_evidence(rows, role, k=3):
    sign = 1 if role == 'A' else -1
    total = sum(abs(r['difference']) for r in rows)
    tol = max(1e-12, total * 1e-6)
    return sorted([r for r in rows if r['difference'] * sign > tol],
                  key=lambda r: (-r['difference'] * sign, r['ref']))[:k]
