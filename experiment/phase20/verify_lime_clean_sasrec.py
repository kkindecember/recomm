"""Read-only completion audit of the two fixed-epoch clean SASRec parents."""
import argparse
import json
import math
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiment.phase20.guided_decoding_job import ROOT, now, read, sha, write

sys.path.insert(0, str(ROOT / 'third_party/LIME-Rec'))
from lime_rec.models import SASRec


def verify(domain):
    started = time.monotonic()
    folder = ROOT / f'artifacts/phase20/lime_clean_sasrec_{domain.lower()}_20260920_v1'
    status = read(folder / 'status.json')
    assert status['state'] == 'COMPLETED', status
    summary = read(folder / 'summary.json')
    config = read(folder / 'config.json')
    manifest = read(folder / 'input_manifest.json')
    assert summary['state'] == 'COMPLETED'
    assert summary['input_sha256'] == manifest
    assert config['epochs'] == 200 and config['seed'] == 0
    assert not config['automatic_next_stage'] and not summary['next_stage_started']
    for name, digest in manifest.items():
        assert sha(ROOT / name) == digest, name
    checkpoint_path = folder / 'sasrec.pt'
    assert sha(checkpoint_path) == summary['checkpoint_sha256']
    assert sha(folder / 'neighbor_ids.npy') == summary['neighbors_sha256']
    cp = torch.load(checkpoint_path, map_location='cpu')
    last = torch.load(folder / 'latest.pt', map_location='cpu')
    assert cp['config'] == config and cp['input_sha256'] == manifest
    assert cp['train_complete'] and not cp['probe_only']
    assert cp['epoch'] == last['epoch'] == status['epoch'] == summary['epochs'] == 200
    assert cp['state_dict'].keys() == last['state_dict'].keys()
    for key, tensor in cp['state_dict'].items():
        assert torch.isfinite(tensor).all(), key
        assert torch.equal(tensor, last['state_dict'][key]), key
    for record in (cp, last, summary, status):
        assert record['training_only']
        assert not record['development_labels_read'] and not record['test_read']
    assert not summary['heldout_labels_used_for_selection']
    for key in ('torch_rng_state', 'python_rng_state', 'numpy_rng_state', 'optimizer'):
        assert key in cp
    sequences = [json.loads(line)['history'] for line in
                 (ROOT / config['data'] / 'upstream_sequences_without_terminal.jsonl').open()]
    positions = sum(min(len(seq) - 1, 50) for seq in sequences)
    batches = math.ceil(len(sequences) / config['batch_size'])
    expected_steps = batches * 200
    assert cp['optimizer_steps'] == last['optimizer_steps'] == summary['optimizer_steps'] == expected_steps
    for state in cp['optimizer']['state'].values():
        assert int(state['step']) == expected_steps
        for value in state.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all()
    curve = read(folder / 'training_curve.json')
    assert len(curve) == 200 and len(sequences) == summary['users']
    for epoch, entry in enumerate(curve, 1):
        assert entry['epoch'] == epoch and entry['optimizer_steps'] == epoch * batches
        assert entry['users'] == len(sequences) and entry['supervised_positions'] == positions
        assert math.isfinite(entry['loss']) and entry['seconds'] > 0
    items = [json.loads(line) for line in
             (ROOT / f'artifacts/phase20/nearest_comparison_20260918/{domain}.items.jsonl').open()]
    ids = [item['source_item_id'] for item in items]
    assert len(ids) == len(set(ids)) == cp['num_items'] == summary['num_items']
    assert [item['item_id'] for item in items] == list(range(1, len(ids) + 1))
    neighbors = np.load(folder / 'neighbor_ids.npy', allow_pickle=False)
    k = 5 if domain == 'Toys' else 10
    assert neighbors.shape == (len(ids), k)
    assert np.issubdtype(neighbors.dtype, np.integer)
    assert neighbors.min() >= 1 and neighbors.max() <= len(ids)
    assert not np.any(neighbors == np.arange(1, len(ids) + 1)[:, None])
    assert all(len(set(row)) == k for row in neighbors)
    # Independent sort implementation: stable descending sort preserves
    # ascending numeric catalog IDs for equal cosine scores.
    embedding = cp['state_dict']['item_emb.weight'][1:].numpy().copy()
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    for start in range(0, len(ids), 256):
        scores = embedding[start:start + 256] @ embedding.T
        scores[np.arange(len(scores)), np.arange(start, start + len(scores))] = -np.inf
        expected = np.argsort(-scores, axis=1, kind='stable')[:, :k] + 1
        np.testing.assert_array_equal(expected, neighbors[start:start + len(scores)])
    text_rows = [line.split() for line in (folder / 'similar_item_sasrec.txt').open()]
    assert len(text_rows) == len(ids)
    for item, near, row in zip(ids, neighbors, text_rows):
        assert row == [item] + [ids[j - 1] for j in near]
    # Check that the saved model is loadable and actually scores short and
    # long legal training histories without nonfinite outputs.
    order = sorted(range(len(sequences)), key=lambda i: (len(sequences[i]), i))
    selected = [order[i] for i in np.linspace(0, len(order) - 1, 8, dtype=int)]
    inputs = torch.zeros((len(selected), 50), dtype=torch.long)
    for row, index in enumerate(selected):
        seq = sequences[index][-50:]
        inputs[row, -len(seq):] = torch.tensor(seq)
    model = SASRec(cp['num_items'], cp['hidden'], cp['maxlen'], cp['num_layers'],
                   cp['num_heads'], cp['dropout'], architecture=cp['architecture'])
    model.load_state_dict(cp['state_dict'], strict=True)
    model.eval()
    with torch.no_grad():
        logits = model.predict(inputs)
        assert logits.shape == (len(selected), len(ids) + 1) and torch.isfinite(logits).all()
        model.load_state_dict(last['state_dict'], strict=True)
        assert torch.equal(logits, model.predict(inputs))
    # Detect changes made while this audit was reading the completed run.
    for name, digest in manifest.items():
        assert sha(ROOT / name) == digest, name
    assert sha(checkpoint_path) == summary['checkpoint_sha256']
    result = dict(state='VERIFIED', domain=domain, completed_at=now(),
                  seconds=time.monotonic() - started, training_completed_at=summary['completed_at'],
                  training_minutes=summary['elapsed_seconds'] / 60, epochs=200,
                  optimizer_steps=expected_steps, training_users=len(sequences),
                  supervised_positions_per_epoch=positions, input_hashes_checked=len(manifest),
                  neighbor_rows_checked=len(ids), neighbor_pairs_checked=int(neighbors.size),
                  neighbor_recomputed_from_final_embeddings=True, hint_text_mapping_verified=True,
                  final_latest_state_equal=True, checkpoint_optimizer_rng_present=True,
                  checkpoint_forward_users=len(selected), checkpoint_forward_finite=True,
                  checkpoint_forward_replay_exact=True, training_curve_epochs_checked=len(curve),
                  development_labels_read=False, test_read=False, new_effectiveness_metrics=False,
                  next_stage_started=False, source_sha256=sha(__file__),
                  input_sha256=manifest, outputs_sha256={name: sha(folder / name) for name in (
                      'sasrec.pt', 'latest.pt', 'neighbor_ids.npy', 'similar_item_sasrec.txt',
                      'training_curve.json', 'status.json', 'summary.json', 'config.json')})
    destination = ROOT / 'artifacts/phase20/lime_clean_sasrec_verification_20260920_v1'
    destination.mkdir(exist_ok=True)
    assert not (destination / f'{domain}.json').exists(), 'Keep existing audit receipts immutable'
    write(destination / f'{domain}.json', result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('input_sha256', 'outputs_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('domains', choices=('Toys', 'Beauty'), nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with threadpool_limits(2):
        for domain in args.domains:
            verify(domain)
