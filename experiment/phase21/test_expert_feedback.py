import unittest
import numpy as np

from experiment.phase21.expert_feedback_core import CardBuilder, select_candidates, shuffle_evidence, pack_cards


class FeedbackBoundaryTests(unittest.TestCase):
    def test_repeat_build_does_not_mutate_cache_and_evidence_can_be_older(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return dict(input_ids=[20+sum(map(ord, word))%100 for word in text.split()])
        catalog=dict(items=['','a','b','c','d'], lexical_ids=['','a','b','c','d'],
            decoder_paths=[[0,201,1],[0,202,1],[0,203,1],[0,204,1]],
            passages=['']+[f'title: item {x}; categories: sample; brand: brand' for x in 'abcd'])
        emb=np.array([[1.,0.],[0.,1.],[1.,0.],[.2,.8]])
        cf=dict(neighbors={'a':[dict(item_id='c',score=.9)],'b':[dict(item_id='c',score=.1)]})
        builder=CardBuilder(catalog,Tokenizer(),emb,cf,per_expert=1)
        scores=np.array([[0.,0.,1.,.5]]*3)
        before=builder.tokens('sequence rank 1; item: ',16).copy()
        a=builder.build([1,2],scores);b=builder.build([1,2],scores)
        self.assertEqual(a,b)
        self.assertEqual(before,builder.tokens('sequence rank 1; item: ',16))
        self.assertTrue(all(203 in prefix for prefix in a[0]))
        self.assertEqual(a[2][0]['support_items'],[])
        self.assertEqual(a[2][1]['support_items'],[1,2])
        self.assertEqual(a[2][2]['support_items'],[1,2])

    def test_candidates_use_scores_and_observed_history_only(self):
        scores = np.array([[1., 4., 4., 2., 0.]] * 3)
        np.testing.assert_array_equal(select_candidates(scores, [2], 3), [[3, 4, 1]] * 3)
        changed = scores.copy()
        changed[:, 4] = 5.
        self.assertEqual(int(select_candidates(changed, [2], 3)[0, 0]), 5)

    def test_placebo_preserves_candidates_scores_lengths_and_split(self):
        prefixes = [[10+i, 25] for i in range(8)]
        tails = [[40+i, 60+i] for i in range(8)]
        users = ['a','b','c','d','e','f','g','h']
        roles = ['fit']*4 + ['dev']*4
        experts = [0,0,1,1]*2
        cards, donors = shuffle_evidence(prefixes, tails, users, roles, experts)
        for i, d in enumerate(donors):
            self.assertNotEqual(i, d)
            self.assertEqual(cards[i][:2], prefixes[i])
            self.assertEqual(cards[i][2:-1], tails[d])
            self.assertEqual((roles[i],experts[i]),(roles[d],experts[d]))
        bank = pack_cards(cards, 2, 4, 8)
        self.assertEqual(bank.shape, (2,4,8))
        np.testing.assert_array_equal((bank != 0).sum(-1), np.full((2,4),5))

    def test_single_user_stratum_is_reported_unchanged(self):
        cards, donors = shuffle_evidence([[2],[3]], [[4],[5]], ['u','u'], ['fit']*2, [0]*2)
        np.testing.assert_array_equal(donors, [0,1])
        self.assertEqual(cards, [[2,4,1],[3,5,1]])


if __name__ == '__main__':
    unittest.main()
