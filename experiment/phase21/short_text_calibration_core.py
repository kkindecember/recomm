"""Calibration and complete compact predictions on fixed candidate pools."""
import numpy as np
from experiment.phase20.catalog_scope_core import metrics, ranks


def corrected(base, signal, strength):
    base = np.asarray(base,float)
    if strength==0:
        return base.copy()
    return (base-base.max(1,keepdims=True))/np.maximum(base.std(1,keepdims=True),1e-3)+float(strength)*signal


def choose(ids,base,signal,targets,strengths):
    grid = [dict(strength=float(g),**metrics(ranks(corrected(base,signal,g),ids,targets))) for g in strengths]
    winner = max(grid,key=lambda row:(row['N10'],row['H10'],-row['strength']))
    return winner['strength'],grid


def prediction(ids,base,signal,strength,targets):
    score = corrected(base,signal,strength)
    order = np.lexsort((ids,-score),axis=1)
    sorted_ids = np.take_along_axis(ids,order,axis=1)
    matches = sorted_ids==np.asarray(targets)[:,None]
    independent = np.where(matches.any(1),matches.argmax(1)+1,2**30)
    rr = ranks(score,ids,targets)
    np.testing.assert_array_equal(independent,rr)
    return score,order.astype(np.uint8),rr


def tests():
    ids = np.array([[4,2,1,3],[1,2,3,4]])
    base = np.array([[1.,1.,1.,0.],[4.,3.,2.,1.]])
    signal = np.array([[0.,0.,0.,0.],[-1.,0.,0.,1.]])
    score,order,rr = prediction(ids,base,signal,0.,[1,7])
    np.testing.assert_array_equal(score,base)
    np.testing.assert_array_equal(ids[0,order[0]],[1,2,4,3])
    np.testing.assert_array_equal(rr,[1,2**30])
    gamma,grid = choose(ids,base,signal,[1,7],[0.,.1,1.])
    assert gamma==0. and len(grid)==3
    assert corrected(base,signal,1.).shape==base.shape
    return dict(state='PASSED',zero_exact=True,tie_order_by_item_id=True,
                absent_targets_retained=True,calibration_ties_prefer_zero=True,independent_ranks=True)
