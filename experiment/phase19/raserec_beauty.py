"""Beauty input checks and a status supervisor for the unchanged RaSeRec trainer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = 'experiment/phase19/config/s19_raserec_gram_beauty.json'
TOYS = ROOT/'artifacts/phase18/raserec_gram/toys/v1/status.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def preflight(config, config_path):
    # This phase uses CPU and existing train/validation artifacts only.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from experiment.phase18.protocol.s18_raserec_gram import Run
    from experiment.phase9.eval_cf0_b3_beamfusion import load_cached_beams, normalize_lexical_id

    torch.set_num_threads(4)
    toys = read(ROOT/'experiment/phase18/config/s18_raserec_gram_toys.json')
    for key in ('seed', 'pretrain', 'reader', 'reader_training', 'gram_training', 'max_wall_seconds'):
        if config[key] != toys[key]:
            raise AssertionError(f'Unexpected change from Toys: {key}')
    if config['dataset'] != 'Beauty':
        raise AssertionError('Expected Beauty')
    if not 0 < config['cuda_memory_fraction'] <= 1 or config['minimum_free_mib'] <= 0:
        raise AssertionError('Invalid resource settings')
    run = Run(config, config_path, ROOT/config['output'], torch.device('cpu'))
    if (len(run.train), len(run.validation), run.num_items-1) != (131413, 22363, 12101):
        raise AssertionError('Beauty data dimensions differ')
    if digest(ROOT/run.data_config['parent_checkpoint']) != run.data_config['parent_sha256']:
        raise AssertionError('Beauty parent hash differs')
    if set(run.reference) != {r['user_id'] for r in run.validation.records}:
        raise AssertionError('Validation users differ from the fixed reference')
    summary = read(ROOT/config['pcrf']['summary'])
    if {k: config['pcrf'][k] for k in ('lambda', 'beta', 'gamma', 'q1')} != summary['frozen_pcrf']:
        raise AssertionError('Beauty PCRF parameters differ from the fixed reference')
    frequencies = sorted(int(run.frequencies[r['target']]) for r in run.validation.records)
    if frequencies[len(frequencies)//4] != config['pcrf']['q1']:
        raise AssertionError('Beauty frequency quartile differs')
    beams, _ = load_cached_beams(ROOT/config['runtime']['baseline_predictions'])
    lexical = {normalize_lexical_id(v): i for i, v in enumerate(run.catalog['lexical_ids']) if i}
    predictions = []
    for row in run.validation.records[:64]:
        beam = beams[row['user_id']]
        if lexical[normalize_lexical_id(beam['gold'])] != row['target']:
            raise AssertionError('Baseline gold item differs')
        items = [lexical[normalize_lexical_id(v)] for v in beam['candidates']]
        rank = items.index(row['target'])+1 if row['target'] in items else 51
        if rank != int(run.reference[row['user_id']]['baseline_rank']):
            raise AssertionError('Baseline rank differs')
        predictions.append({'user_id': row['user_id'], 'ranked_item_ids': items, 'scores': beam['seq']})
    ranks = run.pcrf(predictions)
    if ranks != [int(run.reference[r['user_id']]['pcrf_rank']) for r in predictions]:
        raise AssertionError('Beauty PCRF baseline-beam ranks differ')
    paths = [config_path, ROOT/config['data_config'], ROOT/config['pcrf']['checkpoint'],
             ROOT/config['pcrf']['reference'], ROOT/config['pcrf']['summary'],
             ROOT/'experiment/phase18/protocol/s18_raserec_gram.py', Path(__file__),
             ROOT/'experiment/phase19/run_stage19_raserec_beauty.sh']
    result = dict(state='passed', updated_at=now(), config_sha256=digest(config_path),
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
                  train_examples=len(run.train), validation_examples=len(run.validation),
                  catalog_items=run.num_items-1, parent_sha256=run.data_config['parent_sha256'],
                  training_frequencies_match_all_validation_references=True,
                  pcrf_original_beam_rank_parity_users=len(predictions), pcrf_rank_parity=True,
                  frozen_pcrf=summary['frozen_pcrf'], primary_comparator='original_gram',
                  original_gram=summary['baseline'], original_gram_pcrf=summary['pcrf'],
                  gpu_smoke='Runs in the background after pretrain/bank/reader, before GRAM training',
                  test_read=False)
    write(ROOT/config['runtime']['preflight'], result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'original_gram', 'original_gram_pcrf')}))


def publish(config, live):
    write(ROOT/config['output']/'live_status.json', live)
    toys = read(TOYS)
    toys_pid = toys.get('pid')
    toys['process_alive'] = alive(toys_pid)
    if toys.get('state') == 'running' and not toys['process_alive']:
        toys['observed_state'] = 'STALE_PROCESS_NOT_ALIVE'
    write(ROOT/'artifacts/phase19/status.json', {
        'updated_at': now(), 'primary_comparator': 'original_gram',
        'overview_heartbeat': 'Updated by the Beauty supervisor while it runs; individual status files remain canonical.',
        'experiments': {'toys': toys, 'beauty': live},
        'individual_status_paths': {'toys': str(TOYS.relative_to(ROOT)),
                                    'beauty': config['output']+'/live_status.json'}})


def supervise(config, config_path):
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    if (output/'live_status.json').exists() or (output/'manifest.json').exists():
        raise FileExistsError('Existing Beauty attempt; no automatic restart')
    checked = read(ROOT/config['runtime']['preflight'])
    if checked.get('state') != 'passed' or checked.get('config_sha256') != digest(config_path):
        raise RuntimeError('A passing preflight for this exact configuration is required')
    for path, expected in checked['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise RuntimeError(f'Source changed after preflight: {path}')
    started = time.time()
    meta = dict(experiment_id=config['name'], dataset='Beauty', seed=config['seed'],
                primary_comparator='original_gram', tmux_session=config['runtime']['tmux_session'],
                launcher_pid=os.getpid(), physical_gpu=config['runtime']['physical_gpu'],
                gpu_uuid=config['gpu_uuid'], started_at=now(),
                deadline_utc=datetime.fromtimestamp(started+config['max_wall_seconds'], timezone.utc).isoformat(),
                hard_timeout_seconds=config['max_wall_seconds'], config=str(config_path.relative_to(ROOT)),
                config_sha256=digest(config_path), log_path=config['output']+'/run.log',
                workload_status_path=config['output']+'/status.json',
                output_directory=config['output'], gram_total_epochs=11,
                train_examples=131413, validation_examples=22363,
                minimum_free_mib=config['minimum_free_mib'], cuda_memory_fraction=config['cuda_memory_fraction'],
                automatic_retry=False, test_read=False)
    child = None
    requested_signal = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: requested_signal.append(signum))
    state, error, returncode = 'STARTING', None, None
    try:
        publish(config, dict(meta, state=state, updated_at=now(), stage='initializing', process_alive=False))
        candidates = config['runtime'].get('gpu_candidates', [config['gpu_uuid']])
        while True:
            gpu_text = subprocess.check_output(['nvidia-smi', '-i', ','.join(candidates),
                        '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            gpus = []
            for line in gpu_text.strip().splitlines():
                index, uuid, free, utilization = [v.strip() for v in line.split(',')]
                if uuid not in candidates:
                    raise RuntimeError('GPU query returned an unexpected UUID')
                gpus.append(dict(physical_gpu=int(index), uuid=uuid, free_mib=int(free), utilization=int(utilization)))
            eligible = [g for g in gpus if g['free_mib'] >= config['minimum_free_mib']]
            if eligible:
                selected = max(eligible, key=lambda g: (g['free_mib'], -g['utilization']))
                break
            state = 'WAITING_FOR_GPU'
            publish(config, dict(meta, state=state, updated_at=now(), heartbeat_at=now(),
                                 stage='gpu_admission', process_alive=False, supervisor_alive=True,
                                 waiting_elapsed_seconds=round(time.time()-started, 1),
                                 gpu_candidates=gpus, scientific_training_started=False))
            if requested_signal:
                raise InterruptedError(f'Interrupted by signal {requested_signal[0]}')
            if time.time()-started > config['runtime'].get('admission_wait_seconds', 21600):
                raise TimeoutError('No eligible GPU during the bounded admission wait; training never started')
            time.sleep(config['runtime']['heartbeat_seconds'])
        config = json.loads(json.dumps(config))
        config['gpu_uuid'] = selected['uuid']
        config['runtime']['physical_gpu'] = selected['physical_gpu']
        execution_config = output/'execution_config.json'
        write(execution_config, config)
        write(output/'gpu_admission.json', dict(observed_at=now(), **selected))
        queued_at = meta['started_at']
        queue_seconds = time.time()-started
        started = time.time()
        meta.update(queued_at=queued_at, waiting_elapsed_seconds=round(queue_seconds, 1),
                    started_at=now(), gpu_uuid=config['gpu_uuid'], physical_gpu=selected['physical_gpu'],
                    deadline_utc=datetime.fromtimestamp(started+config['max_wall_seconds'], timezone.utc).isoformat(),
                    execution_config=str(execution_config.relative_to(ROOT)),
                    execution_config_sha256=digest(execution_config))
        command = [sys.executable, '-u', '-m', 'experiment.phase18.protocol.s18_raserec_gram',
                   '--config', str(execution_config.relative_to(ROOT)), '--stage', 'all']
        write(output/'launch.json', dict(meta, command=command, preflight=config['runtime']['preflight']))
        with (output/'run.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                returncode = child.poll()
                raw = read(output/'status.json')
                state = 'RUNNING' if returncode is None else ('COMPLETED' if returncode == 0 and raw.get('state') == 'complete' else 'FAILED')
                if raw.get('state') == 'failed' and returncode is None:
                    state = 'FAILING'
                live = dict(meta, state=state, updated_at=now(), heartbeat_at=now(),
                            workload_pid=child.pid, process_alive=returncode is None,
                            elapsed_seconds=round(time.time()-started, 1), returncode=returncode,
                            stage=raw.get('stage', 'complete' if state == 'COMPLETED' else 'initializing'),
                            workload_status=raw)
                for key in ('epoch', 'seen', 'total', 'event', 'tag'):
                    if key in raw:
                        live[key] = raw[key]
                publish(config, live)
                if returncode is not None:
                    break
                if requested_signal:
                    raise InterruptedError(f'Interrupted by signal {requested_signal[0]}')
                if time.time()-started > config['max_wall_seconds']+120:
                    raise TimeoutError('Beauty workload exceeded its fixed wall budget and shutdown grace')
                time.sleep(config['runtime']['heartbeat_seconds'])
    except BaseException as exc:
        state = 'INTERRUPTED' if isinstance(exc, InterruptedError) else 'FAILED'
        error = repr(exc)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        returncode = child.returncode if child is not None else None
    finally:
        raw = read(output/'status.json')
        publish(config, dict(meta, state=state, updated_at=now(), heartbeat_at=now(),
                             ended_at=now(), elapsed_seconds=round(time.time()-started, 1),
                             stage=raw.get('stage', 'complete' if state == 'COMPLETED' else 'initializing'), workload_status=raw,
                             workload_pid=child.pid if child is not None else None,
                             process_alive=False, returncode=returncode, error=error or raw.get('error')))
    return 0 if state == 'COMPLETED' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--mode', choices=('preflight', 'run'), required=True)
    args = parser.parse_args()
    config_path = (ROOT/args.config).resolve()
    config = read(config_path)
    if args.mode == 'preflight':
        preflight(config, config_path)
        return 0
    return supervise(config, config_path)


if __name__ == '__main__':
    raise SystemExit(main())
