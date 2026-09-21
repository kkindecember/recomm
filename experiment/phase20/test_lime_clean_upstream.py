import copy
import unittest
import numpy as np
from experiment.phase20.lime_clean_upstream_core import validate_rows,cosine_neighbors,item_inputs,gram_sample


class CleanUpstreamTests(unittest.TestCase):
    def test_truncated_sequences_match_exactly_and_future_event_is_rejected(self):
        rows=[dict(user_id='a',history=[1],target=2),dict(user_id='b',history=[4],target=5)]
        self.assertEqual(validate_rows(rows,dict(a=[1,2],b=[4,5])),2)
        with self.assertRaises(AssertionError):
            validate_rows(rows+[dict(user_id='a',history=[1,2],target=3)],dict(a=[1,2],b=[4,5]))

    def test_target_change_cannot_modify_prediction_inputs(self):
        ids=['x','y','z'];lex=dict(x='|one',y='|two',z='|three');text={k:'title: '+k for k in ids}
        inputs=item_inputs(ids,text,lex,np.array([[2,3],[1,3],[1,2]]))
        row=dict(user_id='a',history=[1,2],target=3)
        first=gram_sample(row,ids,inputs,lex);changed=copy.deepcopy(row);changed['target']=1
        second=gram_sample(changed,ids,inputs,lex)
        self.assertEqual(first['input'],second['input'])
        self.assertEqual(first['history_item_ids'],[2,1])
        self.assertNotEqual(first['output'],second['output'])

    def test_neighbors_exclude_self_ties_and_chunk_invariance(self):
        e=np.array([[1,0],[1,0],[0,1],[0,1]],dtype=np.float32)
        a=cosine_neighbors(e,k=2,block=1);b=cosine_neighbors(e,k=2,block=3)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a,[[2,3],[1,3],[4,1],[3,1]])


if __name__=='__main__':unittest.main()
