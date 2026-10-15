"""Observable prefix features computed before pruning; no targets/final list."""
import numpy as np
from experiment.phase20.lime_decode_screen_core import mixed_log_values


def prefix_features(steps, expert_logs, depth, beam_size=50):
    assert depth in (2, 3) and len(steps) >= depth
    # Deliberately never read a later step.
    snapshots = steps[:depth]
    traces = {}
    for expected, step in enumerate(snapshots, 1):
        assert step['depth'] == expected
        for prefix, parent_score, row in zip(step['prefixes'], step['parent_scores'], step['rows']):
            if parent_score <= -1e8:
                continue
            key = tuple(prefix)
            value = (np.asarray(row['native']), row['legal_log_mass'], row['tokens'])
            if key in traces:
                np.testing.assert_array_equal(traces[key][0], value[0])
            traces[key] = value

    def mixed_parent(prefix):
        total = 0.
        for j in range(1, len(prefix)):
            parent, token = tuple(prefix[:j]), prefix[j]
            values, mass, tokens = traces[parent]
            total += float(mixed_log_values(values[tokens.index(token)], mass,
                                           expert_logs[parent][token], .3))
        return total

    current = snapshots[-1]
    expanded = {}
    for prefix, parent_score, row in zip(current['prefixes'], current['parent_scores'], current['rows']):
        if parent_score <= -1e8:
            continue
        prefix = tuple(prefix)
        values = np.asarray(row['native'])
        q = np.asarray([expert_logs[prefix][token] for token in row['tokens']])
        mixed = mixed_parent(prefix) + mixed_log_values(values, row['legal_log_mass'], q, .3)
        native = (np.float32(parent_score) + values.astype('float32')).astype('float64')
        for token, ns, ms in zip(row['tokens'], native, mixed):
            key = prefix + (int(token),)
            pair = (float(ns), float(ms))
            if key in expanded:
                assert expanded[key] == pair
            expanded[key] = pair
    assert expanded
    order = sorted(expanded, key=lambda p: (-expanded[p][0], p))
    rank = {p: i + 1 for i, p in enumerate(order)}
    cutoff = sorted((m for n, m in expanded.values()), reverse=True)[min(beam_size, len(expanded)) - 1]
    return {p: np.array([n / depth, (m - n) / depth, 1 / rank[p], (m - cutoff) / depth], dtype='float32')
            for p, (n, m) in expanded.items()}
