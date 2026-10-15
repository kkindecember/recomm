"""Fast real-cache integration check; no validation-based method selection."""
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from experiment.phase21 import dual_candidate_gate as D


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    D.OUT = Path(tempfile.mkdtemp(prefix='dual_gate_smoke_', dir='/tmp'))
    (D.OUT/'Toys').mkdir()
    D.CFG['epochs'] = 4
    D.CFG['checkpoints'] = [0, 4]
    unit = D.tests()
    checks = {}; begin = time.monotonic()
    for domain in D.DOMAINS:
        ck = torch.load(D.DUAL/domain/'gate.pt', map_location='cpu')
        anchor = D.Anchor(ck)
        checks[domain] = D.check_anchor(domain, anchor)
        print(domain, checks[domain], flush=True)
        if domain == 'Toys':
            toys_anchor = anchor
    train = D.Data('Toys', True, toys_anchor, limit=64)
    models = {}; dense_checks = {}
    for arm in D.ARMS:
        model, norm, info = D.fit(train, arm, 0)
        models[arm] = model, norm
        ix = np.arange(4)
        unchanged_delta = train.delta(model, ix, arm, norm)[0].detach().clone()
        # Selection can legitimately choose epoch zero on this tiny smoke split.
        # Force nonzero logits only in this test to exercise correction paths.
        with torch.no_grad():
            model.output.weight.normal_(0, .005)
            model.output.bias.add_(torch.tensor([.01, -.01]))
        delta, _ = train.delta(model, ix, arm, norm)
        assert delta.abs().max() > 0
        f = torch.from_numpy(np.array(train.a['fused'][ix], copy=True)).double()*20
        changed = f.clone()
        for j, k in enumerate(ix):
            mask = train.a['mask'][k]; ids = train.a['ids'][k, mask]
            changed[j, ids] += 20*delta[j, mask].double()
        p = f.softmax(1)
        dense = torch.nn.functional.cross_entropy(changed, torch.from_numpy(train.targets[ix]))
        dense += .2*(p*(f.log_softmax(1)-changed.log_softmax(1))).sum(1).mean()
        compressed = train.objective(model, ix, arm, norm)
        assert abs(float(dense)-float(compressed)) < 4e-6, (arm, dense, compressed)
        saved = torch.load(D.OUT/'Toys'/f'{arm}_0.pt', map_location='cpu')
        assert saved['fit_users'] == [s['user_id'] for s in train.samples]
        reloaded = D.Head(); reloaded.load_state_dict(saved['state_dict'])
        with torch.no_grad():
            expected = train.delta(reloaded, ix, arm, (saved['mean'], saved['std']))[0]
        torch.testing.assert_close(expected, unchanged_delta, atol=0, rtol=0)
        dense_checks[arm] = dict(dense=float(dense), compressed=float(compressed), checkpoint_reload_exact=True,
                                nonzero_correction_exercised=True)
    valid = D.Data('Toys', False, toys_anchor, limit=32)
    eval_checks = {}
    for arm, (model, norm) in models.items():
        ranks, checks_for_arm = D.evaluate(valid, model, norm, arm, 1)
        assert len(ranks) == 32
        eval_checks[arm] = checks_for_arm
    result = dict(state='PASSED', temporary_output=str(D.OUT), elapsed_seconds=time.monotonic()-begin,
                  unit=unit, anchor_checks=checks, dense_checks=dense_checks, evaluation=eval_checks)
    D.write(D.OUT/'smoke_receipt.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
