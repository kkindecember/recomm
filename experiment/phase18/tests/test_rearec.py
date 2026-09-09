import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from experiment.phase18.core import rearec_adapter as a


class ReaRecTests(unittest.TestCase):
    def setUp(self):
        a.seed_all(2023)
        self.config = json.loads((a.ROOT / 'experiment/phase18/config/s18_rearec_beauty.json').read_text())
        self.config['max_history'] = 4
        self.config['model'].update(emb_size=16, inner_size=24, dropout=0.0)
        self.model = a.ReaRec(self.config, 13)
        self.batch = a.collate([dict(user_id=str(i), history=h, target=t)
                                for i, (h, t) in enumerate([([1], 3), ([3, 2], 4),
                                                           ([1, 4, 2], 5), ([6, 2, 4, 1], 7)])], 4)

    def test_cached_reasoning_matches_full_recomputation(self):
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.batch)['model_output']
            embeddings = self.model.input_embeddings(self.batch['ids'])
            states = []
            for step in range(3):
                wrapper = self.model.model
                padding = wrapper._prepare_padding_mask(self.batch['lengths'], 'cpu', step)
                mask = wrapper._prepare_attention_mask(4, 4 + step, 'cpu', padding)
                layers, _ = self.model.trm_encoder(wrapper.LayerNorm(embeddings), mask, kv_caches=[None] * 2)
                last = layers[-1][:, -1:, :]
                states.append(last)
                if step < 2:
                    embeddings = torch.cat([embeddings, last + wrapper.reason_pos_emb.weight[step]], 1)
            self.assertTrue(torch.allclose(expected, torch.cat(states, 1), atol=2e-6, rtol=2e-6))
            # A zero-step query must be exactly the initial state of the same run.
            zero = self.model(self.batch, reason_steps=0)['model_output']
            self.assertTrue(torch.equal(zero[:, 0], expected[:, 0]))

    def test_prl_formula_matches_author_with_noise(self):
        manifest = json.loads((a.ROOT / self.config['source_manifest']).read_text())
        source = (a.ROOT / manifest['source'] / 'src/models/PRL.py').read_text()
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'PRL')
        fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'loss'))
        namespace = dict(torch=torch, F=F, ITEM_ID='item_id')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<pinned PRL loss>', 'exec'), namespace)
        # Avoid exact FP32 saturation in a four-example formula fixture.
        self.model.temperature = 1.0
        self.model.train()
        out = self.model(self.batch, epoch=3)
        loss, parts = self.model.loss(out)
        self.model.loss_fct = torch.nn.CrossEntropyLoss()
        author = namespace['loss'](self.model, dict(out, item_id=out['labels']))
        self.assertTrue(torch.allclose(loss, author, atol=1e-6))
        self.assertGreater(parts['contrastive'].item(), 0)

    def test_warmup_and_reasoning_gradients(self):
        self.model.train()
        for epoch in (1, 3):
            self.model.zero_grad(set_to_none=True)
            loss, parts = self.model.loss(self.model(self.batch, epoch=epoch))
            self.assertTrue(torch.isfinite(loss))
            if epoch == 1:
                self.assertEqual(parts['contrastive'].item(), 0)
            loss.backward()
            self.assertGreater(self.model.model.reason_pos_emb.weight.grad.abs().sum().item(), 0)
            self.assertEqual(self.model.item_emb.weight.grad[0].abs().sum().item(), 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_padding_is_invisible_and_scores_map_to_valid_items(self):
        self.model.eval()
        with torch.no_grad():
            before = self.model(self.batch)['prediction']
            self.model.item_emb.weight[0].fill_(1000)
            self.model.pos_emb.weight[4].fill_(-2000)
            after = self.model(self.batch)['prediction']
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(before.shape, (4, 12))
        self.assertTrue(torch.equal(self.model(self.batch)['labels'] + 1, self.batch['targets']))

    def test_checkpoint_restores_rng_and_adam_trajectory(self):
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        def step(model, optimizer):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.loss(model(self.batch, epoch=3))
            loss.backward()
            optimizer.step()
        step(self.model, opt)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            a.save_checkpoint(path, self.model, opt, dict(epoch=1, optimizer_steps=1))
            step(self.model, opt)
            restored = a.ReaRec(self.config, 13)
            optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
            progress = a.load_checkpoint(path, restored, optimizer)
            self.assertEqual(progress['optimizer_steps'], 1)
            step(restored, optimizer)
            self.assertTrue(all(torch.equal(value, restored.state_dict()[k]) for k, value in self.model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
