import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment.phase18.core import etegrec_adapter as a


class EtegrecTests(unittest.TestCase):
    def setUp(self):
        a.seed_all(2023)
        self.config = json.loads((a.ROOT / 'experiment/phase18/config/s18_etegrec_beauty.json').read_text())
        self.config['model'].update(semantic_hidden_size=12, encoder_layers=1, decoder_layers=1,
                                   d_model=16, d_ff=32, num_heads=2, d_kv=8, dropout_rate=0.0,
                                   code_num=4, num_emb_list=[4, 4, 4], layers=[16], e_dim=8,
                                   kmeans_init=False)
        self.rec, self.rq = a.build_models(self.config, F.normalize(torch.randn(10, 12), dim=-1), 'cpu')
        self.codes = torch.tensor([[-1] * 4] + [[0, i // 4, i % 4, 0] for i in range(9)])
        self.batch = a.collate([{'user_id': str(i), 'history': [1, 2][:1 + i % 2], 'target': t}
                                for i, t in enumerate([3, 4, 3, 5])])

    def test_alignment_gradients_and_frozen_networks(self):
        a.set_phase(self.rec, self.rq, 'id')
        rec_before = copy.deepcopy(self.rec.state_dict())
        for key in ('kl', 'contrastive'):
            self.rq.zero_grad(set_to_none=True)
            _, values = a.losses(self.rec, self.rq, self.codes, self.batch, 'id')
            values[key].backward()
            grads = [p.grad for p in self.rq.encoder.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        opt = torch.optim.AdamW(self.rq.parameters(), lr=0.001)
        before = copy.deepcopy(self.rq.state_dict())
        opt.step()
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in self.rq.state_dict().items()))
        self.assertTrue(all(torch.equal(rec_before[k], v) for k, v in self.rec.state_dict().items()))
        a.set_phase(self.rec, self.rq, 'rec')
        a.losses(self.rec, self.rq, self.codes, self.batch, 'rec')[0].backward()
        self.assertIsNone(self.rec.semantic_embedding.weight.grad)
        self.assertTrue(all(p.grad is None for p in self.rq.parameters()))
        self.assertGreater(self.rec.token_embeddings[0].weight.grad.abs().sum().item(), 0)

    def test_alignment_formulas_match_author_functions(self):
        manifest = json.loads((a.ROOT / self.config['source_manifest']).read_text())
        source = (a.ROOT / manifest['source'] / 'trainer.py').read_text()
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
        fns = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('compute_discrete_contrastive_loss_kl', 'compute_contrastive_loss')]
        for n in fns:
            n.decorator_list = []
        namespace = {'torch': torch, 'F': F}
        exec(compile(ast.fix_missing_locations(ast.Module(body=fns, type_ignores=[])), '<author functions>', 'exec'), namespace)
        x, y = torch.randn(3, 3, 4), torch.randn(3, 3, 4)
        kl = namespace['compute_discrete_contrastive_loss_kl']
        self.assertTrue(torch.equal(a.symmetric_kl(x, y), kl(x, y) + kl(y, x)))
        x, y = torch.randn(3, 12), torch.randn(3, 12)
        cl = namespace['compute_contrastive_loss']
        self.assertTrue(torch.allclose(a.symmetric_contrastive(x, y), cl(x, y, gathered=False) + cl(y, x, gathered=False)))

    def test_catalog_beam_matches_exhaustive_sequence_scores(self):
        self.rec.eval()
        trie = a.CatalogTrie(self.codes, 4)
        items, scores = a.generate_items(self.rec, self.codes, trie, self.batch, beam=9)
        exact = []
        with torch.no_grad():
            for item in range(1, 10):
                x = self.codes[self.batch['ids']].flatten(1).clone()
                labels = self.codes[item].expand(len(x), -1)
                out = self.rec(input_ids=x, attention_mask=x.ne(-1), labels=labels)
                exact.append(out.logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1).mean(-1))
        exact = torch.stack(exact, -1)
        order = exact.argsort(dim=-1, descending=True, stable=True)
        self.assertTrue(torch.equal(items, order + 1))
        self.assertTrue(torch.allclose(scores, exact.gather(1, order), atol=2e-6))

    def test_checkpoint_resume_matches_continuous_adam_updates(self):
        a.set_phase(self.rec, self.rq, 'rec')
        opt = torch.optim.AdamW(self.rec.parameters(), lr=0.001)
        sched = a.scheduler(opt, 6, 2)
        def step(rec, rq, optimizer, schedule):
            a.set_phase(rec, rq, 'rec')
            loss, _ = a.losses(rec, rq, self.codes, self.batch, 'rec')
            loss.backward()
            optimizer.step()
            schedule.step()
        for _ in range(3):
            step(self.rec, self.rq, opt, sched)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'checkpoint.pt'
            a.save_bundle(p, self.rec, self.rq, self.codes, {'rec': opt}, {'rec': sched}, {'steps': 3})
            for _ in range(3):
                step(self.rec, self.rq, opt, sched)
            restored, rq = a.build_models(self.config, torch.randn(10, 12), 'cpu')
            resumed_opt = torch.optim.AdamW(restored.parameters(), lr=0.001)
            resumed_sched = a.scheduler(resumed_opt, 6, 2)
            codes, progress = a.load_bundle(p, restored, rq, {'rec': resumed_opt}, {'rec': resumed_sched})
            self.assertTrue(torch.equal(codes, self.codes))
            self.assertEqual(progress['steps'], 3)
            self.assertTrue(all(q.initted for q in rq.rq.vq_layers))
            for _ in range(3):
                step(restored, rq, resumed_opt, resumed_sched)
            self.assertEqual(resumed_opt.param_groups[0]['lr'], 0)
            self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k, v in self.rec.state_dict().items()))

    def test_patience_does_not_carry_across_minimum_budget(self):
        from experiment.phase18.protocol.s18_etegrec import plateau
        state = {}
        for _ in range(20):
            self.assertEqual(plateau(.08, state, False, .0001), 0)
        self.assertEqual(plateau(.07, state, True, .0001), 0)
        for expected in range(1, 6):
            self.assertEqual(plateau(.07, state, True, .0001), expected)
        self.assertEqual(plateau(.081, state, True, .0001), 0)


if __name__ == '__main__':
    unittest.main()
