"""Prepare comparable train/development records and audit pinned SEAR code.

No optimizer, recommender forward, external entry point, or test input is used.
Run with the existing gram-repro Python; --verify checks saved inputs/outputs.
"""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/nearest_comparison_20260918'
SEAR = ROOT / 'third_party/SEAR'
COMMIT = '4dc184400d96a93d74d14270f5f8c869b667a835'
DATA = {'Toys': 'artifacts/phase18/diff_gram/prepared_v2_categories',
        'Beauty': 'artifacts/phase18/diff_gram/beauty/prepared_v1'}
SEEDS = (20260917, 20260918, 20260919)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def environment():
    result = {}
    for name, dist in [('torch', 'torch'), ('numpy', 'numpy'), ('recbole', 'recbole'),
                       ('mamba_ssm', 'mamba-ssm'), ('causal_conv1d', 'causal-conv1d'),
                       ('sentence_transformers', 'sentence-transformers'), ('transformers', 'transformers')]:
        try:
            version = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            version = None
        result[name] = dict(available=importlib.util.find_spec(name) is not None, version=version)
    return result


def sear_boundary_check():
    """Execute the inspected, unchanged augmentation indexing prefix on fake IDs."""
    source = SEAR / 'seq_dataset.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CustomizedSeqDataset')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'data_augmentation')
    stop = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'new_length' for t in n.targets))
    method.body = method.body[:stop] + [ast.parse('return target_index, target_list_index').body[0]]
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = dict(np=np, torch=torch)
    exec(compile(module, str(source), 'exec'), namespace)
    reports = []
    for max_history in (20, 100):
        sizes = [max_history + 6, 8]
        ids = torch.tensor([1] * sizes[0] + [2] * sizes[1])
        fake = SimpleNamespace(config={'MAX_ITEM_LIST_LENGTH': max_history, 'k': 4},
            uid_field='uid', time_field='time', inter_feat={'uid': ids},
            logger=SimpleNamespace(debug=lambda *a: None), _aug_presets=lambda: None,
            _check_field=lambda *a: None, sort=lambda **kw: None)
        indices, slices = namespace['data_augmentation'](fake)
        affected = []
        for index, sl in zip(indices.tolist(), slices.tolist()):
            user = int(ids[index]); start = 0 if user == 1 else sizes[0]
            user_end = start + sizes[user - 1]; train_end = user_end - 2
            if index >= train_end:
                continue  # validation/test target rows are not training examples
            actual = list(range(sl.start, min(sl.stop, len(ids))))
            expected = list(range(index, min(index + 4, train_end)))
            if actual != expected:
                affected.append(dict(target_position=index - start,
                    expected_positions=[v - start for v in expected],
                    actual_positions=[v - start for v in actual],
                    includes_heldout=any(train_end <= v < user_end for v in actual),
                    crosses_user=any(int(ids[v]) != user for v in actual)))
        assert affected and any(x['includes_heldout'] for x in affected)
        assert any(x['crosses_user'] for x in affected)
        reports.append(dict(max_history=max_history, synthetic_user_lengths=sizes, affected=affected))
    return dict(state='REPRODUCED_ON_SYNTHETIC_DATA', source=str(source.relative_to(ROOT)),
        cases=reports, original_paper_result_impact='NOT_MEASURED',
        explanation='seq_start moves with the history window but is reused as the fixed user origin for the future-label stop.')


def sear_loss_check():
    """Run only the inspected author's KL class, without importing the model."""
    path = SEAR / 'model/loss.py'
    spec = importlib.util.spec_from_file_location('pinned_sear_loss', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    logits = torch.tensor([[.1, .2, -.3, .4, .5], [.5, -.1, .3, .2, -.4]])
    ids = torch.tensor([[1, 2, 1, 0], [3, 4, 0, 0]])
    ratings = torch.tensor([[5., 4., 2., 0.], [2., 5., 0., 0.]])
    loss = module.KLDivergenceLoss(lambda_weight=2)(logits, ids, ratings)
    q = np.zeros((2, 5), dtype=np.float64)
    for row in range(2):
        weights = [float(ratings[row, j]) * np.exp(-2 * j) if ids[row, j] else 0. for j in range(4)]
        for j, weight in enumerate(weights):
            q[row, int(ids[row, j])] += weight / sum(weights)
    lp = torch.log_softmax(logits.double(), -1).numpy(); positive = q > 0
    manual = ((q[positive] * (np.log(q[positive]) - lp[positive])).sum() / 2)
    assert abs(float(loss) - manual) < 1e-6
    uniform = module.KLDivergenceLoss(lambda_weight=2)(logits, ids, torch.ones_like(ratings))
    assert abs(float(loss) - float(uniform)) > 1e-4
    empty = module.KLDivergenceLoss(lambda_weight=2)(logits[:1], torch.zeros((1, 4), dtype=torch.long), torch.zeros((1, 4)))
    return dict(rating_weighted_formula_verified=True, repeated_item_mass_accumulates=True,
        absolute_error=abs(float(loss) - manual), replacing_ratings_changes_loss=True,
        all_padding_target_weight_denominator_zero=True,
        all_padding_loss_on_installed_torch=float(empty) if bool(torch.isfinite(empty)) else None,
        installed_torch=torch.__version__, all_padding_training_occurrence='NOT_CLAIMED')


def padded(sequence, width):
    values = sequence[-width:]
    return values + [0] * (width - len(values))


def prepare_domain(domain, use):
    directory = ROOT / DATA[domain]
    catalog = json.loads(use(directory / 'catalog.json').read_text())
    train = [json.loads(line) for line in use(directory / 'train.jsonl').open()]
    valid = [json.loads(line) for line in use(directory / 'validation.jsonl').open()]
    sequences, user_order, rows = {}, [], []
    for index, row in enumerate(train):
        user = row['user_id']
        if user not in sequences:
            assert len(row['history']) == 1
            sequences[user] = list(row['history']); user_order.append(user)
        sequence = sequences[user]
        assert sequence[-20:] == row['history'], (domain, index, 'chronology mismatch')
        rows.append((user, len(sequence), index))
        sequence.append(row['target'])
    mapping = {u: i for i, u in enumerate(user_order)}
    assert len(valid) == len(sequences) and len({r['user_id'] for r in valid}) == len(valid)
    for row in valid:
        assert row['history'] == sequences[row['user_id']][-20:]
    # Preserve complete train sequences as a reusable bank. No raw sequence file,
    # validation target, or synthetic held-out placeholder enters this bank.
    offsets = [0]; tokens = []
    for u in user_order:
        tokens.extend(sequences[u]); offsets.append(len(tokens))
    np.savez_compressed(OUT / f'{domain}.train_sequences.npz', users=np.asarray(user_order),
        offsets=np.asarray(offsets), items=np.asarray(tokens, dtype=np.int32))
    data = dict(user_index=[], position=[], source_row=[], history20=[], history100=[], target=[], future4=[])
    for user, position, index in rows:
        seq = sequences[user]
        data['user_index'].append(mapping[user]); data['position'].append(position); data['source_row'].append(index)
        for width in (20, 100): data[f'history{width}'].append(padded(seq[:position], width))
        data['target'].append(seq[position])
        future = seq[position:position + 4]
        data['future4'].append(future + [0] * (4 - len(future)))
    arrays = {key: np.asarray(value, dtype=np.int32) for key, value in data.items()}
    assert (arrays['future4'][:, 0] == arrays['target']).all()
    np.savez_compressed(OUT / f'{domain}.train.npz', **arrays)
    np.savez_compressed(OUT / f'{domain}.validation.npz', users=np.asarray([r['user_id'] for r in valid]),
        user_index=np.asarray([mapping[r['user_id']] for r in valid], dtype=np.int32),
        history20=np.asarray([padded(sequences[r['user_id']], 20) for r in valid], dtype=np.int32),
        history100=np.asarray([padded(sequences[r['user_id']], 100) for r in valid], dtype=np.int32),
        target=np.asarray([r['target'] for r in valid], dtype=np.int32))
    text = {}
    for line in use(ROOT / f'GRAM/rec_datasets/{domain}/item_plain_text.txt').open():
        item, body = line.rstrip('\n').split(' ', 1)
        assert item not in text
        fields = {}
        for fragment in body.split(';'):
            if ':' in fragment:
                key, value = fragment.split(':', 1)
                if key.strip() in ('title', 'categories', 'brand', 'price', 'description'):
                    fields[key.strip()] = value.strip()
        text[item] = fields
    counts = Counter()
    with (OUT / f'{domain}.items.jsonl').open('w') as handle:
        for i, item in enumerate(catalog['items'][1:], 1):
            fields = text[item]
            counts.update(k for k, v in fields.items() if v.lower() not in ('', 'na', 'n/a', 'none'))
            handle.write(json.dumps(dict(item_id=i, source_item_id=item, fields=fields), ensure_ascii=False) + '\n')
    # Check current complete method against stored ranks and checkpoint identities.
    controls = json.loads(use(ROOT / 'artifacts/phase20/strong_baseline_20260917/list_loss_controls.json').read_text())
    refs, prior_users, prior_candidates = [], None, None
    for j, seed in enumerate(SEEDS):
        checkpoint = ROOT / controls['arms']['softmax'][domain]['checkpoint_paths'][j]
        use(checkpoint)
        predictions = use(ROOT / f'artifacts/phase20/talos_candidate_screen_20260917/{domain}.{seed}.predictions.npz')
        with np.load(predictions, allow_pickle=False) as v:
            assert len(v['users']) == len(valid)
            targets = {r['user_id']: r['target'] for r in valid}
            expected_pos = np.asarray([next((k for k, item in enumerate(items) if item == targets[str(u)]), -1)
                                       for u, items in zip(v['users'], v['candidate_ids'])])
            np.testing.assert_array_equal(expected_pos, v['positions'])
            if prior_users is not None:
                np.testing.assert_array_equal(prior_users, v['users'])
                np.testing.assert_array_equal(prior_candidates, v['candidate_ids'])
            prior_users, prior_candidates = v['users'].copy(), v['candidate_ids'].copy()
            rank = v['ranks'][list(v['rank_names']).index('softmax.mean')]
            h = float((rank <= 10).mean()); n = float(np.where(rank <= 10, 1 / np.log2(rank + 1), 0).mean())
            for key, actual in [('N10', n), ('H10', h)]:
                expected = controls['arms']['softmax'][domain]['aggregate_metrics']['all'][key]['values'][j]
                assert abs(expected - actual) < 1e-14
            refs.append(dict(head_seed=seed, checkpoint=str(checkpoint.relative_to(ROOT)), N10=n, H10=h))
    lengths = [len(s) for s in sequences.values()]
    return dict(train_prefixes=len(train), train_users=len(sequences), validation_users=len(valid),
        catalog_items=len(catalog['items']) - 1, train_events=len(tokens),
        train_history_max=max(lengths), train_users_longer_than100=sum(x > 100 for x in lengths),
        train_record_fields=sorted(set().union(*(r.keys() for r in train))),
        real_ratings_available=False, real_timestamps_available=False,
        item_fields_present_nonempty=dict(counts), train_and_validation_histories_exact=True,
        future_labels_train_only=True, full_window4_rows=int((arrays['future4'] > 0).all(1).sum()),
        validation_repeat_target_users=sum(r['target'] in sequences[r['user_id']] for r in valid),
        validation_repeat_target_within20_users=sum(r['target'] in r['history'] for r in valid),
        reference_head_runs=refs, N10_head_mean=float(np.mean([r['N10'] for r in refs])),
        H10_head_mean=float(np.mean([r['H10'] for r in refs])),
        rank_metric_replay_only=True, checkpoints_reforwarded=False)


def verify():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for group in ('inputs', 'outputs'):
        for name, expected in manifest[group].items():
            assert sha(ROOT / name) == expected, name
    total = 0
    for domain in DATA:
        with np.load(OUT / f'{domain}.train_sequences.npz', allow_pickle=False) as archive:
            bank = {k: archive[k] for k in archive.files}
        with np.load(OUT / f'{domain}.train.npz', allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        for row, (uid, pos) in enumerate(zip(data['user_index'], data['position'])):
            seq = bank['items'][bank['offsets'][uid]:bank['offsets'][uid + 1]].tolist()
            assert data['target'][row] == seq[pos]
            for width in (20, 100):
                np.testing.assert_array_equal(data[f'history{width}'][row], padded(seq[:pos], width))
            future = seq[pos:pos+4]
            np.testing.assert_array_equal(data['future4'][row], future + [0]*(4-len(future)))
            total += 1
    result = dict(state='VERIFIED_PREPARATION', hashes=True, train_prefixes_rechecked=total,
                  training_updates=0, test_read=False, method_performance_verified=False)
    write(OUT / 'verification.json', result)
    print(json.dumps(result))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--verify', action='store_true'); args = ap.parse_args()
    torch.set_num_threads(4)
    if args.verify:
        return verify()
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    started = time.monotonic(); inputs = {}
    def use(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return path
    try:
        write(OUT / 'status.json', dict(state='PREPARING', training_updates=0, test_read=False))
        commit = subprocess.check_output(['git', '-C', str(SEAR), 'rev-parse', 'HEAD'], text=True).strip()
        assert commit == COMMIT
        assert not subprocess.check_output(['git', '-C', str(SEAR), 'status', '--porcelain'], text=True).strip()
        paths = subprocess.check_output(['git', '-C', str(SEAR), 'ls-files'], text=True).splitlines()
        for p in paths: use(SEAR / p)
        use(Path(__file__).resolve())
        for name in ('talos_candidate_screen.py', 'future_factor_screen.py', 'gacr_features.py', 'future_fusion_alignment.py'):
            use(ROOT / 'experiment/phase20' / name)
        audit = dict(commit=commit, environment=environment(), boundary=sear_boundary_check(), loss=sear_loss_check())
        write(OUT / 'sear_source_audit.json', audit)
        domains = {name: prepare_domain(name, use) for name in DATA}
        result = dict(state='PREPARED', elapsed_seconds=time.monotonic()-started, domains=domains,
            sear_original_run_ready=False, missing=['aligned real ratings', 'aligned real timestamps for native data loader',
                'SEAR runtime dependencies', 'MPNet checkpoint/ID-aligned embeddings'],
            qualification='Prepared shared records and evidence, not a SEAR reproduction or an accuracy comparison.',
            training_updates=0, model_forwards=0, test_read=False, development_reused=True)
        write(OUT / 'summary.json', result)
        write(OUT / 'status.json', dict(state='PREPARED', training_updates=0, test_read=False,
                                       next='matched control plan; original SEAR needs data/runtime completion'))
        for name, value in inputs.items(): assert sha(ROOT / name) == value, name
        outputs = {str(p.relative_to(ROOT)): sha(p) for p in OUT.iterdir() if p.is_file()}
        write(OUT / 'manifest.json', dict(inputs=inputs, outputs=outputs, external_commit=commit))
        print(json.dumps(result, ensure_ascii=False))
    except BaseException as exc:
        write(OUT / 'status.json', dict(state='FAILED', error=repr(exc), training_updates=0, test_read=False))
        raise


if __name__ == '__main__':
    main()
