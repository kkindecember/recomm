"""Independent objective/gradient and parameter-scope checks for list migration."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiment.phase20.preference_core import path_labels
from experiment.phase20.gacr_list_distill_core import (
    teacher_distribution, freeze_encoder_and_embeddings, encode_history,
    decode_scores, list_loss, sample_backward,
)
from experiment.phase9.eval_cf0_b3_beamfusion import standardize
from experiment.phase9.eval_cf0_b4_reliability import rank_matrix

torch.set_num_threads(4)


class Encoder(nn.Module):
    def __init__(self, shared):
        super().__init__(); self.embed = shared; self.linear = nn.Linear(8, 8)

    def set_migration_context(self, *args):
        pass

    def forward(self, input_ids, **kwargs):
        return (self.linear(self.embed(input_ids)),)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Embedding(64, 8)
        self.encoder = Encoder(self.shared)
        self.decoder = nn.ModuleDict({'embed': self.shared, 'linear': nn.Linear(8, 8)})
        self.lm_head = nn.Linear(8, 64, bias=False)
        self.lm_head.weight = self.shared.weight

    def forward(self, encoder_outputs, labels, **kwargs):
        shifted = torch.cat([torch.zeros_like(labels[:, :1]), labels[:, :-1].clamp_min(0)], 1)
        hidden = self.shared(shifted) + encoder_outputs[0].mean(1, keepdim=True)
        logits = self.lm_head(torch.tanh(self.decoder['linear'](hidden)))
        return SimpleNamespace(logits=logits)


def fixture():
    torch.manual_seed(123)
    model = TinyModel(); freeze_encoder_and_embeddings(model)
    paths = [[0, i+2, *([61] if i % 2 else []), 1] for i in range(50)]
    batch = {'item_text_ids': torch.tensor([[[2, 3, 4], [5, 6, 7]]]),
             'item_text_masks': torch.ones(1, 2, 3, dtype=torch.long),
             'target_ids': path_labels(paths, [9], 'cpu')}
    return model, paths, batch


def test_teacher_matches_original_pcrf_and_shuffle_preserves_entropy():
    rng = np.random.RandomState(4)
    g, cf = rng.normal(size=50), rng.normal(size=50)
    pop = rng.randint(0, 20, 50)
    p, q = teacher_distribution(g, cf, pop, 'train-user-prefix')
    order = np.argsort(-g, kind='stable')
    ranks = []
    for original_index in range(50):
        rank, _ = rank_matrix(standardize(g[order])[None], standardize(cf[order])[None],
            standardize(np.log1p(pop[order]))[None], np.array([(pop[order][:10] <= 5).mean()]),
            np.array([int(np.flatnonzero(order == original_index)[0])]), (1., .5, 1.))
        ranks.append(rank[0])
    np.testing.assert_array_equal(np.argsort(-p, kind='stable'), np.argsort(ranks))
    np.testing.assert_array_equal(np.sort(p), np.sort(q))
    assert not np.array_equal(p, q) and np.isclose(p.sum(), 1.)
    np.testing.assert_array_equal(q, teacher_distribution(g, cf, pop, 'train-user-prefix')[1])
    assert not np.array_equal(q, teacher_distribution(g, cf, pop, 'another-prefix')[1])


def test_paths_exclude_bos_include_eos_and_mask_padding():
    labels = path_labels([[0, 3, 1], [0, 4, 5, 1]], [1, 2], 'cpu')
    assert labels.tolist() == [[3, 1, -100], [4, 5, 1]]
    with pytest.raises(ValueError):
        path_labels([[0, 3, 0, 1]], [1], 'cpu')


@pytest.mark.parametrize('arm,covered,divisor', [
    ('CE_anchor', True, 1), ('true_teacher', True, 128),
    ('shuffled_teacher', True, 79), ('true_teacher', False, 1),
])
def test_chunk_replay_matches_direct_complete_list_gradient(arm, covered, divisor):
    replay, paths, batch = fixture(); direct = copy.deepcopy(replay)
    candidates = list(range(1, 51))
    parent = np.random.RandomState(5).normal(size=50).astype(np.float32)
    p, q = teacher_distribution(parent, parent[::-1], np.arange(50), 'gradient-test')
    probability = q if arm == 'shuffled_teacher' else p
    settings = dict(candidate_chunk=4, temperature=1., anchor_weight=1., teacher_weight=.1,
                    covered_normalizer=21839/12399, ce_weight=1.)
    values = sample_backward(replay, batch, paths, candidates, parent, probability,
                             covered, arm, settings, 'cpu', divisor)
    hidden, mask = encode_history(direct, batch, 'cpu')
    scores = decode_scores(direct, hidden, mask, path_labels(paths, candidates, 'cpu'))
    weight = .1 * settings['covered_normalizer'] if covered and arm != 'CE_anchor' else 0.
    aux, _, _ = list_loss(scores, torch.tensor(parent), torch.tensor(probability), 1., weight, 1.)
    ce = -decode_scores(direct, hidden, mask, batch['target_ids']).mean()
    ((ce + aux)/divisor).backward()
    assert abs(values['loss'] - float(ce+aux)) < 1e-5
    for (name, a), (_, b) in zip(replay.named_parameters(), direct.named_parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)
        else:
            assert a.grad is None and b.grad is None, name


def test_optimizer_changes_only_decoder_and_parent_anchor_zero_at_identity():
    model, paths, batch = fixture()
    old = {n: p.detach().clone() for n, p in model.named_parameters()}
    hidden, mask = encode_history(model, batch, 'cpu')
    scores = decode_scores(model, hidden, mask, path_labels(paths, list(range(1, 51)), 'cpu'))
    loss, anchor, _ = list_loss(scores, scores.detach(), torch.ones(50)/50, 1., 0., 1.)
    assert abs(float(anchor)) < 1e-12
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert float(p.grad.abs().max()) < 1e-6
    model.zero_grad(set_to_none=True)
    (-decode_scores(model, hidden, mask, batch['target_ids']).mean()).backward()
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3).step()
    changes = [n for n, p in model.named_parameters() if not torch.equal(p, old[n])]
    assert changes and all(n.startswith('decoder.') for n in changes)
    assert not model.training and all(not p.requires_grad for p in model.encoder.parameters())
