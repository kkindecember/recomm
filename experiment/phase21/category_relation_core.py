"""Explicit catalogue-category/history relations. No outcome enters these APIs."""
from collections import Counter
import hashlib

import numpy as np

from analyze_history_evidence import fields

LEVEL_NAMES = ['known', 'log_distinct_support', 'history_share', 'log_catalog_frequency']
NAMES = [f'{level}_{side}_{name}' for level in ('coarse', 'fine')
         for side, names in [('a', LEVEL_NAMES), ('b', LEVEL_NAMES), ('difference', LEVEL_NAMES),
                             ('pair', ['same_category'])] for name in names]


def parse_category(raw, domain):
    value = ' '.join(raw.lower().split())
    if not value or value in ('na', 'none', 'unknown'): return '', '', 'missing'
    parts = tuple(p.strip() for p in value.split(','))
    root = 'toys games' if domain == 'Toys' else 'beauty'
    if parts[0] != root: return '', '', 'unexpected_root'
    if len(parts) > 8 or len(value) > 220 or any(len(p) > 60 or not p for p in parts):
        return '', '', 'malformed_length'
    if ':' in value or ';' in value: return '', '', 'embedded_field_or_description'
    coarse = ', '.join(parts[:2]) if len(parts) >= 2 else ''
    fine = ', '.join(parts) if len(parts) >= 3 else ''
    return coarse, fine, 'fine_available' if fine else 'coarse_only' if coarse else 'root_only'


class CategoryCatalog:
    def __init__(self, catalog, domain):
        self.catalog, self.domain = catalog, domain
        self.fields = [fields(p) for p in catalog['passages']]
        parsed = [parse_category(f.get('categories', ''), domain) for f in self.fields]
        self.coarse = [r[0] for r in parsed]; self.fine = [r[1] for r in parsed]
        self.reasons = [r[2] for r in parsed]
        self.df = [Counter(x for x in levels[1:] if x) for levels in (self.coarse, self.fine)]

    def paired(self, items, b, history):
        unique = set(history); denom = max(len(unique), 1); values = []
        for levels, df in zip((self.coarse, self.fine), self.df):
            count = Counter(levels[h] for h in unique if levels[h])
            def one(i):
                label = levels[i]
                return [bool(label), np.log1p(count[label]) if label else 0.,
                        count[label]/denom if label else 0., np.log1p(df[label]) if label else 0.]
            a = np.array([one(int(i)) for i in items], dtype=np.float32).reshape(-1, 4)
            bv = np.repeat(np.array([one(b)], np.float32), len(items), axis=0)
            same = np.array([bool(levels[int(i)]) and levels[int(i)] == levels[b] for i in items], np.float32)
            values.append(np.column_stack([a, bv, a-bv, same]))
        return np.concatenate(values, axis=1)

    def card(self, item, history=None):
        result = dict(item=int(item), asin=self.catalog['items'][item],
                      title=self.fields[item].get('title', ''), raw_category=self.fields[item].get('categories', ''),
                      coarse=self.coarse[item], fine=self.fine[item], reason=self.reasons[item])
        if history is not None:
            result['fine_support_items'] = sorted(h for h in set(history)
                if self.fine[item] and self.fine[h] == self.fine[item])
        return result

    def audit(self):
        selected = sorted(range(1, len(self.fine)), key=lambda i: hashlib.sha256(
            f'21.13|{self.domain}|{self.catalog["items"][i]}'.encode()).digest())[:32]
        return dict(items=len(self.fine)-1, reasons=dict(Counter(self.reasons[1:])),
                    coarse_known=sum(bool(s) for s in self.coarse[1:]), fine_known=sum(bool(s) for s in self.fine[1:]),
                    coarse_categories=len(self.df[0]), fine_categories=len(self.df[1]),
                    blind_catalog_sample=[self.card(i) for i in selected],
                    unavailable=[self.card(i) for i in range(1, len(self.fine)) if not self.fine[i]])


def qualifying(x):
    a = np.rint(np.expm1(x[:, 14])).astype(int)
    b = np.rint(np.expm1(x[:, 18])).astype(int)
    return (x[:, 13] > 0) & (x[:, 17] > 0) & (a >= 2) & (a > b)


def rule_choice(items, category, fused_scores):
    ok = np.flatnonzero(qualifying(category))
    if not len(ok): return 0
    support = np.rint(np.expm1(category[:, 14])).astype(int)
    j = min(ok, key=lambda j: (-support[j], -float(fused_scores[j]), int(items[j])))
    return int(items[j])


def top_items(scores, excluded, k=10):
    eligible = np.ones(len(scores), bool)
    eligible[np.asarray(excluded, int)-1] = False
    ix = np.flatnonzero(eligible)
    values = scores[ix]
    threshold = np.partition(values, len(values)-k)[len(values)-k]
    higher = ix[values > threshold]
    equal = ix[values == threshold][:k-len(higher)]
    selected = np.concatenate([higher, equal])
    return (selected[np.lexsort((selected, -scores[selected]))]+1).astype(np.int32)
