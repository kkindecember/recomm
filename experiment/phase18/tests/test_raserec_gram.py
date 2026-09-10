"""Checks for split safety, author fidelity, and the native generation bridge."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiment.phase18.core.raserec_author import DuoRec
from experiment.phase18.core.raserec_data import PositiveCases, sequence_tensors, training_frequencies
from experiment.phase18.core.raserec_gram import RaSeRecGram, RaSeRecReader, ReaderConfig, exact_case_neighbors
from experiment.phase18.tests.test_diff_gram import tiny_model, batch

ROOT = Path(__file__).resolve().parents[3]
torch.set_num_threads(2)


def test_vendored_neural_layers_and_pretraining_match_author_source():
    names = {'MultiHeadAttention', 'CrossMultiHeadAttention', 'FeedForward', 'TransformerLayer', 'TransformerEncoder', 'DuoRec'}
    def bodies(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name in names}
    original = bodies(ROOT/'third_party/RaSeRec/recbole/model/layers.py')
    original.update(bodies(ROOT/'third_party/RaSeRec/recbole/model/sequential_recommender/duorec.py'))
    assert bodies(ROOT/'experiment/phase18/core/raserec_author.py') == original


def test_retrieval_excludes_all_same_user_cases_and_is_chunk_independent():
    queries = torch.tensor([[1., 0.], [0., 1.]])
    keys = torch.tensor([[10., 0.], [1., .2], [.5, .5], [0., 2.], [.1, 1.]])
    query_users = torch.tensor([1, 2])
    key_users = torch.tensor([1, 1, 3, 2, 4])
    found = exact_case_neighbors(queries, keys, query_users, key_users, 2, chunk_size=1)
    assert found.tolist() == [[2, 4], [4, 2]]
    assert torch.equal(found, exact_case_neighbors(queries, keys, query_users, key_users, 2, chunk_size=2))
    with pytest.raises(ValueError, match='Not enough'):
        exact_case_neighbors(queries[:1], keys[:2], query_users[:1], key_users[:2], 1)


def test_query_history_independent_of_gold_and_training_event_counts():
    rows = [{'user_id': 'a', 'history': [1], 'target': 2},
            {'user_id': 'a', 'history': [1, 2], 'target': 3},
            {'user_id': 'b', 'history': [2], 'target': 3}]
    changed = copy.deepcopy(rows)
    changed[0]['target'] = 4
    h, lengths, _ = sequence_tensors(rows)
    h2, lengths2, _ = sequence_tensors(changed)
    assert torch.equal(h, h2) and torch.equal(lengths, lengths2)
    assert training_frequencies(rows, 5).tolist() == [0, 1, 2, 2, 0]
    sampled = PositiveCases(rows).sample([0, 1, 2], np.random.default_rng(1))
    assert sampled.tolist() == [0, 2, 1]


def test_author_pretraining_loss_backward_and_padding():
    config = json.loads((ROOT/'experiment/phase18/config/s18_raserec_gram_toys.json').read_text())['pretrain']
    config.update(train_batch_size=3, hidden_dropout_prob=0., attn_dropout_prob=0.)
    model = DuoRec(config, SimpleNamespace(num_items=10))
    h = torch.tensor([[1, 2], [3, 4], [4, 5]])
    lengths = torch.tensor([2, 2, 2])
    targets = torch.tensor([3, 5, 6])
    torch.testing.assert_close(model(h, lengths), model(F_pad(h), lengths))
    loss = model.calculate_loss({'history': h, 'length': lengths, 'target': targets,
                                'sem_aug': h.flip(0), 'sem_aug_lengths': lengths})
    loss.backward()
    assert torch.isfinite(loss) and model.trm_encoder.layer[0].multi_head_attention.query.weight.grad.abs().sum() > 0


def F_pad(ids):
    return torch.cat([ids, torch.zeros(ids.size(0), 2, dtype=ids.dtype)], 1)


def memory_model_and_batch():
    torch.manual_seed(42)
    gram = tiny_model().gram
    cfg = ReaderConfig(hidden_size=16, inner_size=32, heads=2, dropout=0., topk=3)
    model = RaSeRecGram(gram, RaSeRecReader(cfg), torch.randn(6, 16))
    data = batch()
    data.update(memory_query=torch.randn(2, 16), memory_keys=torch.randn(2, 3, 16),
                memory_values=torch.randn(2, 3, 16))
    return model, data


def test_reader_singleton_and_use_of_retrieved_values():
    model, data = memory_model_and_batch()
    reader = model.reader.eval()
    q, k, v = [data[name] for name in ('memory_query', 'memory_keys', 'memory_values')]
    output = reader(q, k, v)
    torch.testing.assert_close(output[:1], reader(q[:1], k[:1], v[:1]))
    assert not torch.allclose(output, reader(q, k, v+3))


def test_native_identity_memory_changes_gradients_and_reload(tmp_path):
    model, data = memory_model_and_batch()
    model.eval()
    model.memory_enabled = False
    result = model(**data)
    original = model.gram(input_ids=data['input_ids'], attention_mask=data['attention_mask'],
                          labels=data['labels'], use_cache=False, return_dict=True)
    torch.testing.assert_close(result.logits, original.logits, atol=0, rtol=0)
    torch.testing.assert_close(result.loss, original.loss, atol=0, rtol=0)
    model.memory_enabled = True
    model.set_backbone_frozen(True)
    model.train()
    output = model(**data)
    output.loss.backward()
    assert model.memory_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.gram.parameters())
    model.zero_grad(set_to_none=True)
    model.set_backbone_frozen(False)
    model(**data).loss.backward()
    assert model.gram.shared.weight.grad.abs().sum() > 0
    model.eval()
    before = model(**data).logits.detach()
    torch.save(model.state_dict(), tmp_path/'state.pt')
    restored, _ = memory_model_and_batch()
    restored.load_state_dict(torch.load(tmp_path/'state.pt'), strict=True)
    restored.eval()
    torch.testing.assert_close(restored(**data).logits, before, atol=0, rtol=0)


def test_native_generation_bridge():
    model, data = memory_model_and_batch()
    model.eval()
    data.pop('labels')
    data.pop('target_item_ids')
    generated = model.generate(**data, max_new_tokens=3, num_beams=2, num_return_sequences=2)
    assert generated.shape[0] == 4
