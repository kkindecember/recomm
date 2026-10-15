"""Validation-only sample construction; no test or training entry points."""


def build_samples(validation, upstream, heldout, prior_samples):
    assert len({r['user_id'] for r in validation}) == len(validation)
    held = {r['user_id']: r for r in heldout}
    assert len(held) == len(heldout)
    assert {r['user_id'] for r in validation} == set(upstream)
    prior = {r['user_id']: r for r in prior_samples if r['role'] == 'development'}
    samples = []
    for i, row in enumerate(validation):
        user = row['user_id']
        history = list(upstream[user])
        if user in held:
            assert held[user]['history'] == history[-20:]
            # This is the last TRAINING event, already observed before validation.
            # The validation target is never appended to the inference history.
            history.append(held[user]['target'])
        assert history[-20:] == row['history'], user
        sample = dict(row, full_history=history, validation_index=i,
                      role='development', previously_evaluated=user in prior,
                      training_holdout_role=held.get(user, {}).get('role', 'none'))
        if user in prior:
            for key in ('target', 'history', 'full_history'):
                assert sample[key] == prior[user][key], (user, key)
        samples.append(sample)
    assert set(prior) <= {s['user_id'] for s in samples}
    return samples
