"""Bounded formula checks of pinned public DNR code; no third-party execution.

These small fixtures assess code paths, not DNR benchmark effectiveness.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import torch


def main():
    started=time.monotonic()
    source=Path('/tmp/recomm_dnr_review_20260916')
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert commit=='40a00c0b0095f5b70581ed0d566a05680eecf7a9'
    out=Path('artifacts/phase20/dnr_reference_audit_20260916')
    out.mkdir(parents=True,exist_ok=False)
    files=['README.md','model/DNR.py','model/stage1_MF.py','stage2_trainer_our.py','stage2_main_our.py','utils.py']
    hashes={}
    for name in files:
        p=source/name; q=out/'sources'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    trainer=(source/'stage2_trainer_our.py').read_text()
    assert 'x_t.clone().detach()' in trainer
    assert 'criterion(x_0.clone().detach(), labels)' in trainer
    model_text=(source/'model/DNR.py').read_text()
    assert 'self.fc2 = nn.Linear(self.hidden_dims, 1)' in model_text
    assert 'self.lambda_noise*x_t.unsqueeze(1).repeat(1,self.stage1_choose,1)' in model_text
    retrieval=(source/'model/stage1_MF.py').read_text()
    assert 'torch.cat((slate.float(), top_remaining_indices),dim=1)' in retrieval
    torch.set_num_threads(2);torch.manual_seed(46)
    generator=torch.nn.Linear(3,4);denoiser=torch.nn.Linear(4,4)
    x=torch.randn(5,3);labels=torch.randint(0,2,(5,4)).float()
    noisy=torch.sigmoid(generator(x))
    denoised=torch.sigmoid(denoiser(noisy.clone().detach()))
    regularizer=noisy.square().mean()
    adversarial=torch.nn.functional.binary_cross_entropy(denoised.clone().detach(),labels)
    only=torch.autograd.grad(regularizer,list(generator.parameters()),retain_graph=True)
    combined=torch.autograd.grad(regularizer-.1*adversarial,list(generator.parameters()),retain_graph=True)
    assert all(torch.equal(a,b) for a,b in zip(only,combined))
    connected=torch.nn.functional.binary_cross_entropy(torch.sigmoid(denoiser(noisy)),labels)
    gradients=torch.autograd.grad(connected,list(generator.parameters()))
    assert sum(g.abs().sum() for g in gradients)>0
    # Public final scoring head is linear on candidate + common request vectors.
    candidate=torch.randn(7,50,12); user=torch.randn(7,12)
    score_projection=torch.nn.Linear(50,12); final=torch.nn.Linear(12,1)
    scores1,scores2=torch.randn(7,50),torch.randn(7,50)
    def logits(scores):
        return final(candidate+.1*user[:,None]+.1*score_projection(scores)[:,None]).squeeze(-1)
    l1,l2=logits(scores1),logits(scores2)
    torch.testing.assert_close((l1-l2),(l1-l2)[:,:1].expand_as(l1),rtol=0,atol=3e-7)
    assert torch.equal(torch.argsort(l1,dim=-1),torch.argsort(l2,dim=-1))
    assert not torch.equal(torch.sigmoid(l1),torch.sigmoid(l2))
    # Observed slate is concatenated even when its items are outside pure top-K.
    catalog_scores=torch.tensor([[.9,.8,.7,.6,.2,.1]])
    observed_slate=torch.tensor([[4,5]])
    remaining=torch.topk(catalog_scores,3).indices
    candidates=torch.cat([observed_slate,remaining],dim=1)
    assert (candidates==5).any() and not (torch.topk(catalog_scores,5).indices==5).any()
    result={'state':'CODE_PATH_CHECKS_CONFIRMED_NOT_BENCHMARK_REPRODUCTION','commit':commit,
        'repository':'https://github.com/maowenyu-11/DNR','source_sha256':hashes,
        'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'detached_adversarial_term_adds_zero_generator_gradient':True,
        'connected_comparison_adversarial_gradient_nonzero':True,
        'retriever_score_path_common_offset_preserves_within_query_order_given_fixed_candidate_states':True,
        'retriever_score_path_can_change_probabilities':True,
        'observed_slate_is_forced_into_candidate_list':True,
        'no_claim_of_paper_or_benchmark_invalidity':True,
        'public_code_executed':False,'project_data_read':False,'gpu_training':False,
        'elapsed_seconds':time.monotonic()-started}
    with (out/'summary.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
