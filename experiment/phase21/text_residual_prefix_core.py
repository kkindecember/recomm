"""User-isolated prefix augmentation and matched optimizer-step schedules."""
import hashlib
import numpy as np


def expand(samples, folds, domain, maximum=8):
    rows, terminal, schedules = [], [], []
    for j, (sample, fold) in enumerate(zip(samples, folds)):
        h = sample['full_history']
        terminal.append(len(rows))
        rows.append(dict(user_id=sample['user_id'], user_index=j, fold=int(fold),
                         position=len(h), full_history=h, target=sample['target'], terminal=True))
        earlier = list(range(1, len(h))) if fold != 0 else []
        earlier.sort(key=lambda p: hashlib.sha256(
            f'prefix-residual|{domain}|{sample["user_id"]}|{p}'.encode()).digest())
        schedule = [terminal[-1]]
        for position in earlier[:maximum-1]:
            schedule.append(len(rows))
            rows.append(dict(user_id=sample['user_id'], user_index=j, fold=int(fold),
                             position=position, full_history=h[:position],
                             target=h[position], terminal=False))
        schedules.append(schedule)
    return rows, np.asarray(terminal), schedules


def epoch_examples(users, terminal, schedules, epoch, augmented):
    # Every arm takes one example per user per epoch: identical step budgets.
    # Epoch 0 is identical across the two data regimes at a fixed capacity.
    return np.asarray([schedules[j][epoch % len(schedules[j])] if augmented else terminal[j]
                       for j in users], np.int64)


def tests():
    samples = [dict(user_id='a', full_history=[1, 2, 3, 4], target=5),
               dict(user_id='b', full_history=[6, 7, 8], target=9)]
    rows, terminal, schedules = expand(samples, [2, 0], 'synthetic')
    assert len(rows) == 5 and len(schedules[0]) == 4 and len(schedules[1]) == 1
    assert rows[terminal[0]]['target'] == 5
    for row in rows:
        s = samples[row['user_index']]
        assert row['full_history'] == s['full_history'][:row['position']]
        assert row['target'] == (s['target'] if row['terminal'] else s['full_history'][row['position']])
    for augmented in (False, True):
        np.testing.assert_array_equal(epoch_examples([0, 1], terminal, schedules, 0, augmented), terminal)
    seen = set(epoch_examples([0], terminal, schedules, e, True)[0] for e in range(4))
    assert seen == set(schedules[0])
    return dict(state='PASSED', prefix_no_future_features=True, calibration_users_not_augmented=True,
                equal_users_and_steps_per_epoch=True, identical_first_epoch=True,
                deterministic_complete_prefix_cycle=True)
