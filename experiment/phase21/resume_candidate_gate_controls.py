"""Finish missing controls with fixed old settings; retain original outputs."""
import json
from pathlib import Path
import shutil
import signal
import time
import traceback

import numpy as np
import torch

from experiment.phase21 import candidate_gate_controls as C
from experiment.phase21 import candidate_gate_screen as S
from experiment.phase21.candidate_gate_core import utilities

OUT = S.ROOT / 'artifacts/phase21/candidate_gate_controls_resume_20261014_v1'


def main():
    for rel, digest in S.json.loads((C.OUT/'manifest.json').read_text())['sha256'].items():
        assert S.sha(S.ROOT/rel) == digest, rel
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    S.CFG.update(timeout_seconds=21600, threads=2)
    S.write(OUT/'recovery.json', dict(source=str(C.OUT.relative_to(S.ROOT)), original_stop='1800 second shell timeout', scientific_settings_unchanged=True))
    S.write(OUT/'manifest.json',dict(sha256={str(p.relative_to(S.ROOT)):S.sha(p) for p in [Path(__file__), C.OUT/'manifest.json']}))
    results = {}
    for control in ('unbounded_gate', 'direct_residual'):
        direct = control == 'direct_residual'
        S.OUT = OUT/control
        S.OUT.mkdir()
        S.CFG['eta'] = .1 if direct else 1.
        S.Gate = C.Scalar if direct else C.Gate
        for domain in ('Toys', 'Beauty'):
            dst = S.OUT/domain
            dst.mkdir()
            src = C.OUT/control/domain
            if (src/'summary.json').exists():
                for p in src.glob('*'):
                    if p.is_file(): shutil.copy2(p, dst/p.name)
                result = json.loads((dst/'summary.json').read_text())
                rr = np.load(dst/'ranks.npz')
                valid = C.CachedData(domain, False, direct)
                for seed in S.CFG['seeds']:
                    for name in ('primary','all'):
                        mask = valid.cohorts=='primary' if name=='primary' else np.ones(valid.n,bool)
                        assert abs(utilities(rr[str(seed)][mask]).mean()-result['cohorts'][name]['metrics'][f'control_{seed}']['N10']) < 1e-14
                result['recovery'] = 'existing complete domain metrics rechecked'
            else:
                train = C.CachedData(domain, True, direct)
                valid = C.CachedData(domain, False, direct)
                predictions, fits, diagnostics = {}, {}, {}
                for seed in S.CFG['seeds']:
                    old = src/f'score_seed{seed}.pt'
                    if old.exists() and old.with_suffix('.json').exists():
                        ck = torch.load(old,map_location='cpu')
                        assert ck['fit_users'] == [s['user_id'] for s in train.samples]
                        assert ck['config']['eta'] == S.CFG['eta']
                        model = S.Gate(); model.load_state_dict(ck['state_dict'])
                        norm = ck['mean'], ck['std']
                        info = json.loads(old.with_suffix('.json').read_text())
                        shutil.copy2(old,dst/old.name)
                        shutil.copy2(old.with_suffix('.json'),dst/old.with_suffix('.json').name)
                    else:
                        model,norm,info = S.fit(train,'score',seed)
                    fits[str(seed)] = info
                    predictions[str(seed)],diagnostics[str(seed)] = C.evaluate(valid,model,norm)
                    np.save(dst/f'ranks_seed{seed}.npy',predictions[str(seed)])
                np.savez_compressed(dst/'ranks.npz',**predictions)
                old = np.load(C.SOURCE_OUT/domain/'ranks.npz')
                cohorts = {}
                for name in ('primary','all'):
                    mask = valid.cohorts=='primary' if name=='primary' else np.ones(valid.n,bool)
                    ranks = {**{f'control_{s}':predictions[str(s)][mask] for s in S.CFG['seeds']},
                             **{f'candidate_{s}':old[f'score_{s}'][mask] for s in S.CFG['seeds']}, 'original':old['original'][mask]}
                    metrics = {k:dict(N10=float(utilities(v).mean()),H10=float((v<=10).mean()),H50=float((v<=50).mean())) for k,v in ranks.items()}
                    cu = np.mean([utilities(ranks[f'control_{s}']) for s in S.CFG['seeds']],axis=0)
                    ca = np.mean([utilities(ranks[f'candidate_{s}']) for s in S.CFG['seeds']],axis=0)
                    diff = np.stack([cu-ca,cu-utilities(ranks['original'])],axis=1)
                    rng = np.random.default_rng(20261010)
                    boots = [diff[rng.integers(0,len(cu),size=(50,len(cu)))].mean(1) for _ in range(40)]
                    ci = np.quantile(np.concatenate(boots),[.025,.975],axis=0)
                    cohorts[name] = dict(users=len(cu),metrics=metrics,control_mean_N10=float(cu.mean()),candidate_mean_N10=float(ca.mean()),delta_vs_candidate=float(diff[:,0].mean()),ci95_vs_candidate=ci[:,0].tolist(),delta_vs_original=float(diff[:,1].mean()),ci95_vs_original=ci[:,1].tolist())
                result = dict(control=control,domain=domain,fits=fits,diagnostics=diagnostics,cohorts=cohorts)
            S.write(dst/'summary.json',result)
            results.setdefault(control,{})[domain]=result
            S.write(OUT/'summary.json',results)
            S.write(OUT/'status.json',dict(state='RUNNING',control=control,domain=domain,pid=S.os.getpid()))
    S.write(OUT/'status.json',dict(state='COMPLETE',elapsed_seconds=time.monotonic()-S.START,test_read=False))


if __name__ == '__main__':
    def interrupted(signum,frame): raise TimeoutError('Control recovery interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try: main()
    except BaseException as e:
        if OUT.exists(): S.write(OUT/'status.json',dict(state='FAILED',error=repr(e),traceback=traceback.format_exc()))
        raise
