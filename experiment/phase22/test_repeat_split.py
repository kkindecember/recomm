"""Independent leaf-sum, gradient, real beam, and durable-output checks."""
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch
from experiment.phase20.lime_decode_screen_core import ExpertTrie, ProbabilityGuide
from experiment.phase22 import repeat_split_core as C


def tests():
    torch.set_num_threads(2)
    paths = [[0,2,4,1],[0,2,5,1],[0,3,6,1],[0,3,7,1]]
    tree = ExpertTrie(paths, range(1,5))
    scores, history = np.array([.9,.3,.7,.5]), [1,1,3]
    groups, R = C.components(scores,history)
    np.testing.assert_allclose(groups.sum(1), [1,1],atol=1e-15)
    for rho in (0.,.1,R,1.):
        leaves = C.item_mass(scores,history,rho)
        np.testing.assert_allclose(leaves.sum(),1.,atol=1e-15)
        np.testing.assert_allclose(leaves[[0,2]].sum(),rho,atol=1e-15)
        if rho < 1:
            np.testing.assert_allclose(leaves[1]/leaves[3],math.exp(20*(scores[1]-scores[3])),rtol=1e-14)
        logs = C.conditional_logs(tree,scores,history,rho)
        for prefix, actual in logs.items():
            parent = [i for i,p in enumerate(paths) if tuple(p[:len(prefix)])==prefix]
            denom = leaves[parent].sum()
            if not denom:
                assert actual is None
                continue
            for token, value in actual.items():
                child = [i for i in parent if paths[i][len(prefix)]==token]
                np.testing.assert_allclose(math.exp(value),leaves[child].sum()/denom,atol=1e-15)
    original, restored = tree.conditional_logs(scores), C.conditional_logs(tree,scores,history,R)
    for prefix in original:
        np.testing.assert_allclose(list(original[prefix].values()),list(restored[prefix].values()),atol=1e-13)
    for h in ([],[1,2,3,4]):
        for rho in (0.,.3,1.):
            q = C.conditional_logs(tree,scores,h,rho)
            for prefix in original:
                np.testing.assert_allclose(list(original[prefix].values()),list(q[prefix].values()),atol=1e-13)
    counts = C.novel_counts(tree,history)
    for node, prefix in enumerate(tree.nodes):
        expected = sum(i+1 not in history and tuple(p[:len(prefix)])==prefix for i,p in enumerate(paths))
        assert counts[node] == expected
    # Entire-history-only subtree gets None and must retain native scores.
    zero = C.conditional_logs(tree,scores,[1,2],0.)
    assert zero[(0,2)] is None
    native = torch.full((1,10),-torch.inf)
    native[0,4:6] = torch.tensor([-1.3,-2.4])
    actual = C.SplitGuide(tree,zero,.3)(torch.tensor([[0,2]]),native.clone())
    torch.testing.assert_close(actual,native,rtol=0,atol=0)
    # Differentiable training agrees with decoder at every target path step.
    arrays = [C.path_components(tree,scores,history,p,3) for p in paths]
    parent, child, valid = [np.stack([a[k] for a in arrays]) for k in (0,1,2)]
    pt, ct = torch.from_numpy(parent), torch.from_numpy(child)
    theta = torch.tensor(-1.4,dtype=torch.double,requires_grad=True)
    q, supported = C.path_logs(pt,ct,theta)
    reference = C.conditional_logs(tree,scores,history,float(theta.sigmoid()))
    for i,path in enumerate(paths):
        for d in range(1,len(path)):
            np.testing.assert_allclose(q[i,d-1].detach().numpy(),reference[tuple(path[:d])][path[d]],atol=1e-13)
    n, m, v = torch.full_like(q,-3.),torch.full_like(q,-.4),torch.from_numpy(valid)
    assert torch.autograd.gradcheck(lambda t:C.training_nll(n,m,pt,ct,v,t),(theta,),eps=1e-6,atol=1e-5)
    loss = C.training_nll(n,m,pt,ct,v,theta)
    loss.backward(); assert torch.isfinite(theta.grad)
    # Real HF beam search checks processor placement and sequence scoring.
    from transformers import T5Config,T5ForConditionalGeneration,LogitsProcessorList
    torch.manual_seed(7)
    model = T5ForConditionalGeneration(T5Config(vocab_size=24,d_model=16,d_kv=8,d_ff=24,
        num_layers=1,num_decoder_layers=1,num_heads=2,dropout_rate=0.,decoder_start_token_id=0,
        eos_token_id=1,pad_token_id=0)).eval()
    beam_paths = [[0,2+i//8,5+i%8,1] for i in range(16)]
    ix = ExpertTrie(beam_paths,range(1,17))
    def allowed(batch, ids):
        node = ix.lookup.get(tuple(ids.tolist()))
        return list(ix.children[node]) if node is not None else []
    scores16 = np.linspace(0,1,16)
    h16 = list(range(1,9))
    qold = ix.conditional_logs(scores16)
    qR = C.conditional_logs(ix,scores16,h16,C.components(scores16,h16)[1])
    common = dict(input_ids=torch.tensor([[13,14,1]]),max_length=4,num_beams=4,
        num_return_sequences=4,prefix_allowed_tokens_fn=allowed,return_dict_in_generate=True,
        output_scores=True,length_penalty=1.)
    def decode(guide):
        with torch.no_grad():
            return model.generate(**common,logits_processor=LogitsProcessorList([guide]))
    old = decode(ProbabilityGuide(ix,qold,.3))
    newguide = C.SplitGuide(ix,qold,.3)
    new = decode(newguide)
    restored = decode(C.SplitGuide(ix,qR,.3))
    torch.testing.assert_close(old.sequences,new.sequences,rtol=0,atol=0)
    torch.testing.assert_close(old.sequences_scores,new.sequences_scores,rtol=0,atol=0)
    torch.testing.assert_close(old.sequences,restored.sequences,rtol=0,atol=0)
    torch.testing.assert_close(old.sequences_scores,restored.sequences_scores,rtol=0,atol=0)
    np.testing.assert_allclose(newguide.rescore(new.sequences.tolist(),qold,.3),new.sequences_scores.numpy(),atol=1e-6)
    zeroA, plain = decode(C.SplitGuide(ix,qR,0.)),decode(ProbabilityGuide(ix,None,0.))
    torch.testing.assert_close(zeroA.sequences,plain.sequences,rtol=0,atol=0)
    torch.testing.assert_close(zeroA.sequences_scores,plain.sequences_scores,rtol=0,atol=0)
    nrguide = C.SplitGuide(ix,counts=C.novel_counts(ix,h16))
    nr = decode(nrguide)
    assert all(ix.path_to_item[tuple(p)] not in h16 for p in nr.sequences.tolist())
    np.testing.assert_allclose(nrguide.rescore(nr.sequences.tolist()),nr.sequences_scores.numpy(),atol=1e-6)
    # Durable-prefix validation: truncate partial tail, reject completed bad data.
    from experiment.phase22.repeat_split import recover,ARMS
    with tempfile.TemporaryDirectory(prefix='phase22_',dir='/tmp') as tmp:
        p = Path(tmp)/'rows.jsonl'
        sample = dict(user_id='u',target=99,history=[1])
        record = dict(sample_index=0,**sample,methods={a:dict(ids=list(range(1,51)),scores=list(range(50))) for a in ARMS})
        line = json.dumps(record)+'\n'
        p.write_text(line+'{"sample_')
        assert recover(p,[sample]) == 1 and p.read_text() == line
        assert recover(p,[sample]) == 1
        bad = dict(record,user_id='wrong')
        p.write_text(json.dumps(bad)+'\n')
        try:
            recover(p,[sample])
            raise RuntimeError('Completed corrupt line was accepted')
        except AssertionError:
            pass
    return dict(state='PASSED',independent_leaf_probabilities=True,rho_R_identity=True,
        missing_group_edges=True,explore_relative_odds=True,zero_support_native_fallback=True,
        novel_descendant_counts=True,training_decoder_paths_match=True,rho_gradient_check=True,
        real_t5_old_guidance_exact=True,real_t5_rho_R_exact=True,real_t5_alpha0_exact=True,
        real_t5_norepeat=True,sequence_rescoring=True,durable_partial_tail_recovery=True,
        corrupt_completed_row_rejected=True)


if __name__ == '__main__':
    print(json.dumps(tests()))
