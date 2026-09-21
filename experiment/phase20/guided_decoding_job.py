"""Frozen experiment inputs and durable launch control; imports no torch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'experiment/phase20/guided_decoding_20260919_v1.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def model_paths(domain, config):
    seed = config['head_seed']
    prefix = 'artifacts/phase20'
    return dict(B1=f'{prefix}/gacr_fusion_train_v1/{domain}/B1.pt',
        softmax=f'{prefix}/talos_candidate_screen_20260917/{domain}.{seed}.softmax.pt',
        normalized=f'{prefix}/future_fusion_alignment_20260917/{domain}.{seed}.normalized.next_only.pt',
        linear=f'{prefix}/nearest_matched_head_20260918/{domain}.{seed}.linear_residual.pt',
        mlp=f'{prefix}/nearest_matched_head_20260918/{domain}.{seed}.ordinary_mlp32.pt')


def cf_paths(domain):
    first = ('artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt' if domain == 'Toys'
             else 'artifacts/phase9/p9x_beauty_item_head/best_item_head.pt')
    return [first] + [f'artifacts/phase9/p9s_multiseed/{domain}/seed{s}/item_head/best_item_head.pt' for s in (2024, 2025)]


def prepare(config, out):
    baseline = read(ROOT / 'experiment/phase20/gacr_fusion_train_v1.json')
    inputs = {str(CONFIG.relative_to(ROOT)), config['contract'], 'experiment/phase20/gacr_fusion_train_v1.json'}
    sources = set(ROOT.glob('experiment/**/*.py')) | set(ROOT.glob('GRAM/src/**/*.py'))
    sources |= set(ROOT.glob('experiment/phase20/*guided_decoding*.sh'))
    manifest = {}
    for domain in config['domains']:
        d = baseline['domains'][domain]
        folder = out / domain; folder.mkdir()
        rows = [json.loads(line) for line in (ROOT / d['validation']).read_text().splitlines()]
        order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(
            f'{config["selection_seed"]}|{domain}|{rows[i]["user_id"]}'.encode()).digest())
        cohorts = {}; offset = 0
        for key, n in [('smoke', config['smoke_users']), ('tuning', config['tuning_users']), ('evaluation', config['evaluation_users'])]:
            cohorts[key] = [dict(index=i, user_id=rows[i]['user_id']) for i in order[offset:offset+n]]
            offset += n
        cohorts['longest_history'] = dict(index=max(range(len(rows)), key=lambda i: len(rows[i]['history'])))
        assert len({x['user_id'] for k in ('smoke','tuning','evaluation') for x in cohorts[k]}) == offset
        write(folder / 'cohorts.json', cohorts)
        manifest[str((folder/'cohorts.json').relative_to(ROOT))] = sha(folder/'cohorts.json')
        inputs |= set(model_paths(domain, config).values()) | set(cf_paths(domain))
        inputs |= {d['transfer_config'], d['features'], d['c1']}
        transfer = read(ROOT / d['transfer_config'])
        inputs |= {transfer['data_config'], transfer['c1_checkpoint'], transfer['c1_catalog_file'], transfer['c1_training_users']}
        # Covers parent, prepared train/validation/catalog, and old parity caches.
        inputs |= {p for p in d['input_sha256'] if not p.startswith('experiment/')}
        inputs |= {f'artifacts/phase20/{folder_name}/{domain}{suffix}' for folder_name, suffix in (
            ('talos_candidate_screen_20260917', f'.{config["head_seed"]}.predictions.npz'),
            ('pair_uncertainty_screen_20260917', '.cf_scores.npz'))}
    for path in sorted(sources):
        rel = str(path.relative_to(ROOT)); target = out/'sources'/rel
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        manifest[rel] = sha(path)
    for rel in sorted(inputs):
        assert 'test.json' not in rel and '/test/' not in rel, rel
        manifest[rel] = sha(ROOT/rel)
    write(out/'manifest.json', dict(created_at=now(), sha256=manifest,
        config=config, no_model_training=True, test_read=False, development_validation_reused=True))
    shutil.copy2(CONFIG, out/'config.json')
    shutil.copy2(ROOT/config['contract'], out/'protocol.md')


def verify(out):
    for path, expected in read(out/'manifest.json')['sha256'].items():
        assert sha(ROOT/path) == expected, f'Frozen input changed: {path}'


def gpu_candidates():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    return sorted([dict(uuid=p[0].strip(), free_mib=int(p[1]), utilization=int(p[2]))
        for line in raw.splitlines() for p in [line.split(',')]], key=lambda r: -r['free_mib'])


def supervise(config, out):
    from experiment.phase20.guided_decoding_parallel import supervise_parallel
    supervise_parallel(config, out)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action', choices=['launch','supervise','stop'])
    args=parser.parse_args(); config=read(CONFIG); out=ROOT/config['output']
    if args.action=='stop':
        receipt=read(out/'launch.json'); pid=receipt['pid']; proc=Path(f'/proc/{pid}')
        if proc.exists():
            command=(proc/'cmdline').read_bytes().split(b'\0')
            assert b'experiment.phase20.guided_decoding_job' in command and b'supervise' in command, 'PID is not this supervisor'
            assert (proc/'cwd').resolve()==ROOT, 'Wrong process workspace'
            os.kill(pid,signal.SIGTERM)
            for _ in range(50):
                if not proc.exists() or ') Z ' in (proc/'stat').read_text(): break
                time.sleep(.2)
            else: raise RuntimeError('Supervisor did not stop; leaving output untouched')
        print(json.dumps(dict(stopped_supervisor=pid,reason='User requested less conservative parallel scheduling')))
    elif args.action=='launch':
        out.mkdir(parents=True, exist_ok=False)
        with (out/'supervisor.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m','experiment.phase20.guided_decoding_job','supervise'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONUNBUFFERED='1'), start_new_session=True)
        write(out/'launch.json',dict(pid=child.pid,started_at=now(),status=str((out/'status.json').relative_to(ROOT))))
        print(json.dumps(dict(pid=child.pid,status=str(out/'status.json'))))
    else: supervise(config,out)


if __name__=='__main__': main()
