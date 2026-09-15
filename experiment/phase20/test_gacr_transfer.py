import json
from pathlib import Path
import tempfile
import unittest

from experiment.phase20.evaluate_gacr_transfer import compare_parent_predictions


class FrozenParentParityTest(unittest.TestCase):
    def test_full_reference_requires_every_smoke_user_and_exact_candidate_order(self):
        with tempfile.TemporaryDirectory() as directory:
            smoke,full=Path(directory)/'smoke.jsonl',Path(directory)/'full.jsonl'
            a={'user_id':'a','ranked_item_ids':[1,2,3],'scores':[.3,.2,.1]}
            b={'user_id':'b','ranked_item_ids':[3,1,2],'scores':[.8,.4,.2]}
            smoke.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n')
            full.write_text(json.dumps(b)+'\n'+json.dumps(a)+'\n')
            self.assertEqual(compare_parent_predictions(smoke,full),2)
            full.write_text(json.dumps(a)+'\n')
            with self.assertRaises(AssertionError):compare_parent_predictions(smoke,full)
            full.write_text(json.dumps(a)+'\n'+json.dumps({**b,'ranked_item_ids':[1,3,2]})+'\n')
            with self.assertRaises(AssertionError):compare_parent_predictions(smoke,full)

    def test_equal_ranks_do_not_hide_score_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            smoke,full=Path(directory)/'smoke.jsonl',Path(directory)/'full.jsonl'
            row={'user_id':'a','ranked_item_ids':[1,2,3],'scores':[.3,.2,.1]}
            smoke.write_text(json.dumps(row)+'\n')
            full.write_text(json.dumps({**row,'scores':[.6,.4,.2]})+'\n')
            with self.assertRaises(AssertionError):compare_parent_predictions(smoke,full)


if __name__=='__main__':unittest.main()
