"""Q0: read-only fixed-list and component-gradient diagnosis of frozen models."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
import traceback

import numpy as np
import torch

from experiment.phase20.gacr_list_distill import Work, tensors_hash
from experiment.phase20.gacr_list_distill_core import encode_history, all_scores, decode_scores, list_loss, sample_backward
from experiment.phase20.preference_core import path_labels
from experiment.phase18.protocol.s18_diff_gram import seed_everything
from experiment.phase18.core.diff_data import sha256

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiment/phase20/gacr_list_distill_toys_recovery_v3.json'
OUTPUT = ROOT / 'artifacts/phase20/gacr_list_distill_toys_q0_v1'
MODEL_NAMES = ('C1', 'CE_anchor', 'true_teacher', 'shuffled_teacher')
TERMS = ('ce', 'anchor', 'teacher')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sample_hash(record):
    return hashlib.sha256(f"20260915|{record['sample_key']}|q0".encode()).hexdigest()


def select_panel(records, fit, positions):
    seen, selected = set(), []
    # Covered stratum first; exclude its selected users from the other stratum.
    for covered, count in ((True, 192), (False, 64)):
        pool = sorted([i for i in range(len(records)) if fit[i] and bool(positions[i] >= 0) == covered],
                      key=lambda i: (sample_hash(records[i]), records[i]['sample_key']))
        part = []
        for i in pool:
            if records[i]['user_id'] not in seen:
                seen.add(records[i]['user_id']); part.append(i)
            if len(part) == count:
                break
        selected.extend(part)
    calibration = np.flatnonzero(~fit).tolist()
    gradient = sorted([i for i in selected if positions[i] >= 0], key=lambda i: sample_hash(records[i]))[:32]
    return selected, calibration, gradient


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    original = read(SOURCE)
    records = [json.loads(x) for x in (ROOT / original['train_candidate_jsonl']).read_text().splitlines()]
    run = ROOT / original['output']
    with np.load(run/'cache/distillation_cache.npz') as cache:
        fit, pos = cache['fit'], cache['positions']
        selected, calibration, gradient = select_panel(records, fit, pos)
    assert len(selected) == len({records[i]['user_id'] for i in selected}) == 256
    assert len(calibration) == 256 and len(gradient) == 32
    assert sum(pos[i] >= 0 for i in selected) == 192
    ordered = sorted(range(len(records)), key=lambda i: (len(records[i]['history']), records[i]['sample_key']))
    profile = [ordered[i] for i in np.linspace(0, len(ordered)-1, 8).astype(int)]
    manifest = dict(fit_indices=selected, calibration_indices=calibration, gradient_indices=gradient,
        profile_indices=profile, selection='SHA256(20260915|sample_key|q0); covered stratum first; one FIT prefix/user',
        samples=[dict(cache_index=i, **{k:records[i][k] for k in ('sample_key','user_id','partition','target','train_index')},
                      history_length=len(records[i]['history']), covered=bool(pos[i]>=0)) for i in selected+calibration])
    write(OUTPUT/'panel.json', manifest)
    hashes = {str(SOURCE.relative_to(ROOT)):sha256(SOURCE),
              str(Path(__file__).resolve().relative_to(ROOT)):sha256(Path(__file__)),
              str((OUTPUT/'panel.json').relative_to(ROOT)):sha256(OUTPUT/'panel.json')}
    for filename in ('run_gacr_list_distill_diagnostic.sh','test_gacr_list_distill_diagnostic.py'):
        path=Path(__file__).resolve().with_name(filename)
        hashes[str(path.relative_to(ROOT))]=sha256(path)
    for arm in MODEL_NAMES[1:]:
        for p in (run/arm/'train/last.pt', run/'workers'/f'{arm}.json', run/arm/'train/summary.json'):
            hashes[str(p.relative_to(ROOT))] = sha256(p)
    config = dict(source_config=str(SOURCE.relative_to(ROOT)), source_output=original['output'],
        output=str(OUTPUT.relative_to(ROOT)), model_names=MODEL_NAMES, new_optimizer_steps=0,
        bootstrap=False, test_read=False, minimum_free_mib=24000, memory_fraction=.42,
        resource_wait_seconds=3600, hard_timeout_seconds=5400, profile_budget_seconds=600,
        cpu_threads=4, score_atol=1e-6, gradient_identity_rtol=1e-4, gradient_identity_atol=2e-6,
        frozen_hashes=hashes,
        authorization='User requested Q0 diagnosis and a value-of-continuing decision; limited experiment time',
        resource_revision='Q0 only: no optimizer, component gradients accumulated on CPU; 24000 MiB admission, 42% allocator cap; full profile before scoring')
    write(OUTPUT/'config.json', config)
    print(json.dumps({'state':'PREPARED','output':config['output'],'fit':256,'calibration':256,'gradient':32}))


def flat_gradient(model):
    return torch.cat([(p.grad.detach().cpu().reshape(-1) if p.grad is not None else torch.zeros(p.numel()))
                      for p in model.parameters() if p.requires_grad])


def component_gradients(model, batch, paths, candidates, parent, probability, settings, device):
    """Return separately weighted gradients of the original three-term objective.

    Complete-list score derivatives are replayed in chunks; no parameter updates.
    All returned vectors live on CPU. The caller controls panel averaging.
    """
    hidden, mask = encode_history(model, batch, device)
    labels = path_labels(paths, candidates, device)
    scores = all_scores(model, hidden, mask, labels, settings['candidate_chunk'])
    leaf = scores.detach().requires_grad_(True)
    weight = settings['teacher_weight'] * settings['covered_normalizer']
    _, anchor, teacher = list_loss(leaf, torch.as_tensor(parent, device=device),
                                   torch.as_tensor(probability, device=device), settings['anchor_weight'], weight, settings['temperature'])
    derivatives = {
        'anchor': torch.autograd.grad(settings['anchor_weight']*anchor, leaf, retain_graph=True)[0].detach(),
        'teacher': torch.autograd.grad(weight*teacher, leaf)[0].detach(),
    }
    model.zero_grad(set_to_none=True)
    ce = -decode_scores(model, hidden, mask, batch['target_ids'].to(device)).mean()
    (settings['ce_weight']*ce).backward()
    grads = {'ce': flat_gradient(model)}
    for name in ('anchor', 'teacher'):
        model.zero_grad(set_to_none=True)
        for start in range(0, 50, settings['candidate_chunk']):
            stop = min(start+settings['candidate_chunk'],50)
            replay = decode_scores(model, hidden, mask, labels[start:stop])
            torch.testing.assert_close(replay.detach(), scores[start:stop], rtol=0, atol=1e-6)
            (replay * derivatives[name][start:stop]).sum().backward()
        grads[name] = flat_gradient(model)
    model.zero_grad(set_to_none=True)
    assert all(torch.isfinite(v).all() for v in grads.values())
    return grads, dict(ce=float(ce.detach()), anchor_kl=float(anchor.detach()), teacher_kl=float(teacher.detach()))


def gradient_summary(grads):
    vectors = {k:v.double() for k,v in grads.items()}
    norms = {k:float(v.norm()) for k,v in vectors.items()}
    cos = {}
    for a,b in (('ce','anchor'),('ce','teacher'),('anchor','teacher')):
        # Near-zero anchor at C1 identity has no interpretable direction.
        cos[a+'__'+b] = (float(torch.dot(vectors[a],vectors[b]))/(norms[a]*norms[b])
                        if min(norms[a],norms[b]) > 1e-7 else None)
    total = sum(vectors.values())
    return dict(norms=norms, cosines=cos, total_norm=float(total.norm()),
                teacher_over_ce=norms['teacher']/max(norms['ce'],1e-30),
                anchor_over_teacher=norms['anchor']/max(norms['teacher'],1e-30),
                teacher_alignment_with_total=(float(torch.dot(total,vectors['teacher'])) /
                    max(float(total.norm())*norms['teacher'],1e-30)))


class Diagnosis:
    def __init__(self):
        self.config, self.panel = read(OUTPUT/'config.json'), read(OUTPUT/'panel.json')
        self.original = read(ROOT/self.config['source_config'])
        self.run = ROOT/self.original['output']
        self.started=time.monotonic(); self.compute_started=None
        self.lock=threading.Lock(); self.finished=threading.Event()
        self.info=dict(state='RUNNING',stage='initialization',pid=os.getpid(),new_optimizer_steps=0,test_read=False)
        if (OUTPUT/'status.json').exists():
            raise FileExistsError('Existing Q0 run; refusing overwrite or implicit retry')
        self.status()
        threading.Thread(target=self.heartbeat,daemon=True).start()

    def status(self, **values):
        with self.lock:
            self.info.update(values)
            self.info.update(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),elapsed_seconds=round(time.monotonic()-self.started,2))
            write(OUTPUT/'status.json',self.info)

    def heartbeat(self):
        while not self.finished.wait(30): self.status()

    def verify(self):
        for path,h in self.config['frozen_hashes'].items():
            assert sha256(ROOT/path)==h,('changed_q0_input',path)

    def setup(self):
        self.verify()
        torch.set_num_threads(self.config['cpu_threads'])
        self.status(stage='post_cuda_resource_check')
        torch.cuda.set_device(0)
        deadline=time.monotonic()+self.config['resource_wait_seconds']
        while True:
            free,total=torch.cuda.mem_get_info()
            self.status(free_mib=free/2**20,gpu_name=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
            if free/2**20>=self.config['minimum_free_mib']:break
            if time.monotonic()>=deadline:raise TimeoutError('Q0 resource wait budget exhausted')
            time.sleep(10)
        torch.cuda.set_per_process_memory_fraction(self.config['memory_fraction'],0)
        self.compute_started=time.monotonic()
        signal.alarm(self.config['hard_timeout_seconds'])
        seed_everything(self.original['seed'])
        cfg=copy.deepcopy(self.original);cfg['output']=str((OUTPUT/'work').relative_to(ROOT))
        self.work=Work(cfg,'profile');self.work.verify();self.work.snapshot();self.work.data();self.work.load_model()
        self.work.snapshot()  # Include any modules loaded during model creation.
        self.initial={n:p.detach().cpu().clone() for n,p in self.work.model.named_parameters() if p.requires_grad}
        with np.load(self.run/'cache/distillation_cache.npz') as z:
            self.cache={k:z[k] for k in z.files}
        assert np.array_equal(self.cache['candidate_ids'], self.work.arrays['candidate_ids'])
        self.set_model('C1')

    def set_model(self,name):
        self.work.model.zero_grad(set_to_none=True)
        if name=='C1':state=self.initial
        else:
            p=self.run/name/'train/last.pt';payload=torch.load(p,map_location='cpu')
            assert payload['config_sha256']==sha256(self.run/'workers'/f'{name}.json')
            assert payload['C1_checkpoint_sha256']==self.original['input_sha256'][self.original['c1_checkpoint']]
            assert payload['step']==171 and payload['arm']==name and payload['trainable_names']==self.work.trainable
            assert payload['frozen_parameters_sha256']==self.work.frozen_hash
            state=payload['decoder_parameters']
        assert set(state)==set(self.initial)
        with torch.no_grad():
            for n,p in self.work.model.named_parameters():
                if p.requires_grad:p.copy_(state[n])
        self.current=name
        self.model_hash=tensors_hash((n,p) for n,p in self.work.model.named_parameters() if p.requires_grad)
        self.work.frozen_unchanged()

    def unchanged(self):
        self.work.model.zero_grad(set_to_none=True)
        assert self.model_hash==tensors_hash((n,p) for n,p in self.work.model.named_parameters() if p.requires_grad)
        self.work.frozen_unchanged()

    @torch.no_grad()
    def score(self,i):
        batch=self.work.batch(i);hidden,mask=encode_history(self.work.model,batch,self.work.device)
        labels=path_labels(self.work.decoder.paths,self.cache['candidate_ids'][i].tolist(),self.work.device)
        scores=all_scores(self.work.model,hidden,mask,labels,self.work.settings['candidate_chunk'])
        ce=-decode_scores(self.work.model,hidden,mask,batch['target_ids'].to(self.work.device)).mean()
        assert torch.isfinite(scores).all() and torch.isfinite(ce)
        return scores.cpu().numpy(),float(ce)

    def gradients(self,i):
        return component_gradients(self.work.model,self.work.batch(i),self.work.decoder.paths,
            self.cache['candidate_ids'][i].tolist(),self.cache['parent_scores'][i],self.cache['true_probability'][i],
            self.work.settings,self.work.device)

    def profile(self):
        self.status(stage='profile_8_cases');started=time.monotonic();times=[];errors=[]
        for i in self.panel['profile_indices']:
            torch.cuda.synchronize();before=time.monotonic()
            scores,_=self.score(i);torch.cuda.synchronize();times.append(time.monotonic()-before)
            error=float(np.max(np.abs(scores-self.cache['parent_scores'][i])));errors.append(error)
            assert error<=self.config['score_atol'],('parent_score_parity',i,error)
            if time.monotonic()-started>self.config['profile_budget_seconds']:raise TimeoutError('Q0 profile budget exhausted')
        # Include longest profile history in the backward resource measurement.
        i=max(self.panel['profile_indices'],key=lambda i:len(self.work.records[i]['history']))
        self.set_model('true_teacher')
        # Profile at an actual covered sample with greatest history for objective parity.
        covered=np.flatnonzero(self.cache['fit'] & (self.cache['positions']>=0))
        i=max(covered.tolist(),key=lambda j:(len(self.work.records[j]['history']),self.work.records[j]['sample_key']))
        torch.cuda.synchronize();before=time.monotonic();g,_=self.gradients(i);torch.cuda.synchronize()
        gradient_seconds=time.monotonic()-before
        # Sum of split gradients must recover the exact original training loss.
        self.work.model.zero_grad(set_to_none=True)
        sample_backward(self.work.model,self.work.batch(i),self.work.decoder.paths,self.cache['candidate_ids'][i].tolist(),
            self.cache['parent_scores'][i],self.cache['true_probability'][i],True,'true_teacher',self.work.settings,self.work.device)
        original=flat_gradient(self.work.model);separate=sum(g.values())
        torch.testing.assert_close(separate,original,rtol=self.config['gradient_identity_rtol'],atol=self.config['gradient_identity_atol'])
        relative=float((separate-original).norm())/max(float(original.norm()),1e-30)
        del g,original,separate
        self.unchanged()
        estimated=4*len(self.panel['samples'])*float(np.mean(times))+64*gradient_seconds+120
        result=dict(state='PASSED',score_seconds=times,parent_parity_max_abs=max(errors),
            gradient_seconds=gradient_seconds,gradient_identity_relative_error=relative,
            gradient_profile_history_length=len(self.work.records[i]['history']),
            estimated_remaining_seconds=estimated,elapsed_seconds=time.monotonic()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            optimizer_steps=0)
        write(OUTPUT/'profile.json',result)
        if estimated+time.monotonic()-self.compute_started>self.config['hard_timeout_seconds']:
            raise TimeoutError('Profile predicts Q0 exceeds fixed compute budget')
        self.status(profile=result)
        print(json.dumps(result),flush=True)

    def run_all(self):
        self.setup();self.profile()
        indices=self.panel['fit_indices']+self.panel['calibration_indices']
        outputs={}
        for name in MODEL_NAMES:
            self.set_model(name);scores=[];ces=[];started=time.monotonic()
            self.status(stage='fixed_list_scoring',model=name,completed=0,total=len(indices))
            with (OUTPUT/f'{name}.scores.jsonl').open('x') as handle:
                for done,i in enumerate(indices,1):
                    s,ce=self.score(i)
                    if name=='C1':assert np.max(np.abs(s-self.cache['parent_scores'][i]))<=self.config['score_atol']
                    scores.append(s);ces.append(ce)
                    handle.write(json.dumps(dict(cache_index=i,scores=s.tolist(),ce=ce))+'\n')
                    if done%16==0 or done==len(indices):
                        handle.flush();self.status(completed=done,stage_elapsed_seconds=time.monotonic()-started,
                            stage_eta_seconds=(time.monotonic()-started)*(len(indices)-done)/done)
            self.unchanged()
            np.savez(OUTPUT/f'{name}.scores.npz',indices=np.asarray(indices),scores=np.asarray(scores),ce=np.asarray(ces))
            outputs[name]=dict(seconds=time.monotonic()-started,model_parameter_sha256=self.model_hash,samples=len(indices))
        grad_outputs={}
        for name in ('C1','true_teacher'):
            self.set_model(name);acc=None;sample_rows=[];started=time.monotonic()
            self.status(stage='component_gradient_diagnosis',model=name,completed=0,total=32)
            with (OUTPUT/f'{name}.gradients.jsonl').open('x') as handle:
                for done,i in enumerate(self.panel['gradient_indices'],1):
                    grads,loss=self.gradients(i)
                    if acc is None:acc={k:torch.zeros_like(v,dtype=torch.float64) for k,v in grads.items()}
                    for k,v in grads.items():acc[k].add_(v.double()/len(self.panel['gradient_indices']))
                    row=dict(cache_index=i,loss=loss,**gradient_summary(grads));sample_rows.append(row)
                    handle.write(json.dumps(row)+'\n');handle.flush()
                    del grads
                    self.status(completed=done,stage_elapsed_seconds=time.monotonic()-started,
                        stage_eta_seconds=(time.monotonic()-started)*(32-done)/done)
            grad_outputs[name]=dict(mean_gradient=gradient_summary(acc),samples=sample_rows,seconds=time.monotonic()-started)
            del acc;self.unchanged();gc.collect()
        self.verify();self.work.verify()
        result=dict(state='COMPLETED',fixed_list_models=outputs,gradients=grad_outputs,
            elapsed_seconds=time.monotonic()-self.started,compute_seconds=time.monotonic()-self.compute_started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            all_model_parameters_unchanged_during_diagnosis=True,new_optimizer_steps=0,test_read=False,
            verification_status='ANALYZED',next_training_started=False)
        write(OUTPUT/'summary.json',result);self.status(state='COMPLETED',stage='finished',completed=32,total=32)
        signal.alarm(0);self.finished.set();print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare();return
    job=Diagnosis()
    def stop(sig,frame):raise TimeoutError(f'Q0 interrupted by signal {sig}')
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,stop)
    try:job.run_all()
    except BaseException as error:
        job.status(state='STOPPED' if isinstance(error,TimeoutError) else 'FAILED',error=str(error),traceback=traceback.format_exc())
        job.finished.set();raise


if __name__=='__main__':main()
