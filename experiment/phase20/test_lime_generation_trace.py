"""Check diagnostic failure modes, especially local versus global competition."""
import inspect
import unittest
from experiment.phase20.lime_generation_trace_core import BeamObserver, target_path_analysis, validate_steps


def step(prefixes,parents,rows,indices,tokens,scores,finished=()):
    return dict(depth=len(prefixes[0]),prefixes=prefixes,parent_scores=parents,rows=rows,
        selected=dict(next_beam_indices=indices,next_beam_tokens=tokens,next_beam_scores=scores),
        top=dict(indices=indices,tokens=tokens,scores=scores),finished=list(finished),done=False)


class TraceTests(unittest.TestCase):
    def test_cross_parent_pruning_despite_local_top1(self):
        rows=[dict(tokens=[7,8],native=[-1.,-2.],used=[-1.,-2.],legal_log_mass=-.6867383124817772),
              dict(tokens=[9],native=[-.1],used=[-.1],legal_log_mass=-.1)]
        s=step([[0,2],[0,3]],[-4.,-.2],rows,[1],[9],[-.30000001192092896])
        a=target_path_analysis([s],[0,2,7,1],[2],1,51,{(0,2):{7:-.1,8:-2.35}},[1.,0.],[])
        self.assertEqual(a['outcome'],'prefix_pruned')
        self.assertEqual(a['first_loss']['native_child_rank'],1)
        self.assertEqual(a['first_loss']['global_rank_min'],2)
        self.assertGreater(a['first_loss']['expert_log_ratio'],0)
        validate_steps([s])

    def test_forced_step_is_not_confidence_evidence(self):
        r=dict(tokens=[7],native=[-.8],used=[-.8],legal_log_mass=-.8)
        s=step([[0,2]],[-4.],[r],[],[],[])
        a=target_path_analysis([s],[0,2,7,1],[2],1,51,{(0,2):{7:0.}},[1.,0.],[])
        self.assertTrue(a['first_loss']['forced'])
        self.assertAlmostEqual(a['first_loss']['expert_log_ratio'],0.)

    def test_eos_not_admitted_is_not_ordinary_prefix_loss(self):
        r=dict(tokens=[1],native=[-.2],used=[-.2],legal_log_mass=-.2)
        s=step([[0,7]],[-4.],[r],[],[],[])
        a=target_path_analysis([s],[0,7,1],[2],1,51,{(0,7):{1:0.}},[1.,0.],[])
        self.assertEqual(a['outcome'],'eos_not_admitted');self.assertIsNone(a['first_loss'])

    def test_returned_below_top10_is_ranking_not_pruning(self):
        a=target_path_analysis([], [0,7,1], [1],1,12,{},[1.,0.],[])
        self.assertEqual(a['outcome'],'returned_below_top10')
        self.assertIsNone(a['first_loss'])

    def test_history_filter_is_separate(self):
        a=target_path_analysis([], [0,7,1], [1],1,51,{},[1.,0.],[1])
        self.assertEqual(a['outcome'],'excluded_by_existing_history_rule')

    def test_observer_has_no_target_interface(self):
        self.assertEqual(list(inspect.signature(BeamObserver).parameters),['guide','beams'])
        self.assertEqual(list(inspect.signature(BeamObserver.__call__).parameters),['self','input_ids','scores'])

    def test_expert_diagnostics_are_json_serializable(self):
        import json
        import numpy as np
        from types import SimpleNamespace
        from experiment.phase20.lime_decode_screen_core import ExpertTrie
        from experiment.phase20.lime_generation_trace import Worker
        fake=SimpleNamespace(index=ExpertTrie([[0,2,1],[0,3,1]],range(1,3)),
            signals={k:np.array([[.2,.3]],dtype=np.float32) for k in ('SASRec','ItemCF','BGE')},
            samples=[dict(history=[],target=1)])
        result=Worker.expert_target_paths(fake,0,[0,2,1])
        self.assertEqual(json.loads(json.dumps(result))['ItemCF']['catalog_rank'],2)

    def test_observer_preserves_real_hf_beam_and_eos(self):
        import torch
        from transformers import T5Config, T5ForConditionalGeneration, LogitsProcessorList
        from experiment.phase20.lime_decode_screen_core import ExpertTrie
        from experiment.phase20.lime_path_mixture_core import PathMixtureGuide
        torch.set_num_threads(1);torch.manual_seed(3)
        model=T5ForConditionalGeneration(T5Config(vocab_size=12,d_model=8,d_ff=16,
            num_layers=1,num_heads=2,decoder_start_token_id=0,eos_token_id=1,pad_token_id=0)).eval()
        paths=[[0,2,1],[0,3,4,1],[0,3,5,1],[0,6,7,1]]
        index=ExpertTrie(paths,range(1,5));q=index.conditional_logs([.1,.2,.3,.4])
        def allowed(batch,prefix):
            node=index.lookup.get(tuple(prefix.tolist()))
            return list(index.children[node]) if node is not None else []
        kwargs=dict(input_ids=torch.tensor([[8,9]]),num_beams=2,num_return_sequences=2,
                    max_length=5,output_scores=True,return_dict_in_generate=True,
                    prefix_allowed_tokens_fn=allowed)
        with torch.no_grad():
            plain=model.generate(**kwargs,logits_processor=LogitsProcessorList([PathMixtureGuide(index,q,.3)]))
            guide=PathMixtureGuide(index,q,.3);observer=BeamObserver(guide,2)
            with observer.installed():
                traced=model.generate(**kwargs,logits_processor=LogitsProcessorList([guide,observer]))
        self.assertTrue(torch.equal(plain.sequences,traced.sequences))
        self.assertTrue(torch.equal(plain.sequences_scores,traced.sequences_scores))
        validate_steps(observer.steps)


if __name__=='__main__':unittest.main()
