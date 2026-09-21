"""Reproduce the author's CPU/CUDA top-k tie difference without new fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='GPU-cfd866c2-ccc1-8a3c-00b8-594d7a8cd059'
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiment.phase20.source_competition_data import DataBuilder, arrays, ROOT, LIME
from experiment.phase20.guided_decoding_job import read, write, now
from lime_rec.evaluation import normalize_score_rows
from scripts.evaluation.run_repeat_aware_gate import features


def main():
    torch.set_num_threads(4); threadpool_limits(4)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.02)
    torch.backends.cuda.matmul.allow_tf32=False
    c=read(ROOT/'experiment/phase20/source_competition_20260920_v1.json')
    b=DataBuilder('Toys',c,lambda p:p,lambda *a,**kw:None)
    old=arrays(b.path/'evaluation_seed0_cal2048.npz'); ix=old['indices'][128:256]
    hs=[b.history(u) for u in old['users'][128:256]]
    files=[b.path/'sasrec_scores_seed0.npy',LIME/'Toys/itemcf_scores.npy',b.path/'semantic_scores.npy']
    scores=torch.from_numpy(np.stack([normalize_score_rows(np.asarray(np.load(p,mmap_mode='r')[ix])) for p in files],1))
    hist=torch.zeros((len(ix),b.v),dtype=torch.bool)
    for i,h in enumerate(hs):hist[i,np.asarray(h)-1]=True
    lengths=torch.tensor(list(map(len,hs)),dtype=torch.float32)
    cpu_features=features(scores,lengths,hist)
    with torch.no_grad(): cpu_weights,_=b.gate(cpu_features)
    b.gate.to('cuda')
    gpu_features=features(scores.cuda(),lengths.cuda(),hist.cuda())
    with torch.no_grad(): gpu_weights,_=b.gate(gpu_features)
    np.testing.assert_allclose(gpu_weights.cpu(),old['weights'][128:256],rtol=1e-6,atol=1e-7)
    changed=(cpu_features-gpu_features.cpu()).abs()
    significant=np.argwhere(changed.numpy()>1e-5)
    assert significant.tolist()==[[71,11]]
    ids_cpu=scores[71,1].topk(5).indices
    ids_gpu=scores.cuda()[71,1].topk(5).indices.cpu()
    report=dict(state='CAUSE_REPRODUCED',updated_at=now(),test_read=False,new_optimizer_steps=0,
        cause='CPU/CUDA select different item IDs at an exact ItemCF fifth-place tie; history contamination feature changes',
        feature_index=11,cpu_feature=float(cpu_features[71,11]),cuda_feature=float(gpu_features[71,11]),
        max_cpu_weight_error=float(np.max(np.abs(cpu_weights.numpy()-old['weights'][128:256]))),
        max_cuda_weight_error=float(np.max(np.abs(gpu_weights.cpu().numpy()-old['weights'][128:256]))),
        cuda_top5_contains_history=bool(hist[71,ids_gpu].any()),cpu_top5_contains_history=bool(hist[71,ids_cpu].any()),
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    write(ROOT/'artifacts/phase20/source_competition_20260920_preflight/tie_audit.json',report)
    print(report)


if __name__=='__main__': main()
