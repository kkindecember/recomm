"""Fixed train-only TiSASRec fit; neither heldout labels nor dates are read."""
import gc
import hashlib
import random
import time
import numpy as np
import torch
from experiment.phase21.tisasrec_core import TiSASRec, intervals


def train(folder, variant, config, progress, write):
    """Same train sequences/next-item CE budget as the existing SASRec parent."""
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    raw = np.load(folder/'upstream.npz')
    x, y, days = raw['items'], raw['targets'], raw['days']
    relations = intervals(x, days, cap=config['time_span'], order_only=variant == 'tisas_order')
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(relations))
    generator = torch.Generator().manual_seed(config['seed'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                         generator=generator, num_workers=0)
    model = TiSASRec(int(raw['num_items']), **{k: config[k] for k in
        ('hidden', 'maxlen', 'num_layers', 'num_heads', 'dropout', 'time_span')}).cuda()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    initial_hash = hashlib.sha256()
    for key, value in initial.items():
        initial_hash.update(key.encode()); initial_hash.update(value.numpy().tobytes())
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    path = folder/'behavior'/variant
    path.mkdir(parents=True, exist_ok=False)
    expected = int((y != 0).sum())
    begin, steps, curve = time.monotonic(), 0, []
    for epoch in range(1, config['epochs']+1):
        model.train()
        tic, total, tokens, users = time.monotonic(), 0., 0, 0
        for step, (items, targets, times) in enumerate(loader, 1):
            items, targets, times = items.cuda(), targets.cuda(), times.cuda()
            features = model(items, times)
            valid = targets != 0
            logits = features[valid] @ model.item_emb.weight.T
            loss = torch.nn.functional.cross_entropy(logits, targets[valid], label_smoothing=config['label_smoothing'])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
            assert torch.isfinite(grad)
            optimizer.step()
            n = int(valid.sum()); total += float(loss.detach())*n
            tokens += n; users += len(items); steps += 1
            if step == 1 or step % 50 == 0:
                progress('behavior_training', epoch=epoch, epochs=config['epochs'], batch=step,
                         batches=len(loader), loss=float(loss.detach()), elapsed_seconds=time.monotonic()-begin,
                         estimated_remaining_seconds=(config['epochs']-epoch)*np.mean([r['seconds'] for r in curve[-5:]]) if curve else None)
        assert tokens == expected and users == len(x)
        curve.append(dict(epoch=epoch, loss=total/tokens, seconds=time.monotonic()-tic,
                          supervised_positions=tokens, users=users, optimizer_steps=steps))
        write(path/'curve.json', curve)
        if epoch == 1 or epoch % 10 == 0 or epoch == config['epochs']:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint = dict(state_dict=state, optimizer=optimizer.state_dict(), epoch=epoch,
                config=config, variant=variant, num_items=int(raw['num_items']), optimizer_steps=steps,
                parameters=sum(p.numel() for p in model.parameters()), train_complete=epoch == config['epochs'],
                torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state(),
                shuffle_rng_state=generator.get_state(), training_only=True, test_read=False,
                heldout_labels_read=False, target_timestamps_used=False)
            target = path/('model.pt' if epoch == config['epochs'] else 'latest.pt')
            temporary = target.with_suffix('.tmp')
            torch.save(checkpoint, temporary); temporary.replace(target)
        progress('behavior_epoch_complete', epoch=epoch, epochs=config['epochs'], loss=total/tokens,
                 seconds_per_epoch=float(np.mean([r['seconds'] for r in curve[-5:]])),
                 estimated_remaining_seconds=(config['epochs']-epoch)*float(np.mean([r['seconds'] for r in curve[-5:]])),
                 elapsed_seconds=time.monotonic()-begin)
    assert steps == len(loader)*config['epochs']
    assert any(not torch.equal(v, initial[k]) for k, v in state.items())
    restored = torch.load(path/'model.pt', map_location='cpu')
    assert all(torch.equal(v, restored['state_dict'][k]) for k, v in state.items())
    write(path/'summary.json', dict(state='COMPLETED', training_only=True, users=users,
        supervised_positions_per_epoch=expected, epochs=epoch, optimizer_steps=steps,
        elapsed_seconds=time.monotonic()-begin, checkpoint_exact_replay=True,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, parameters=checkpoint['parameters'],
        initial_state_sha256=initial_hash.hexdigest(),
        shuffle_rng_final_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
        heldout_labels_read=False, target_timestamps_used=False, test_read=False))
    del model, optimizer, logits, features, loss, checkpoint, restored
    gc.collect(); torch.cuda.empty_cache()
