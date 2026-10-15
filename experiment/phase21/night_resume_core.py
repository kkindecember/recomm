"""Durable continuation of sequential per-user generation records."""
import json
import math
import os
from pathlib import Path


def durable_file(path):
    with Path(path).open('rb') as handle:
        os.fsync(handle.fileno())


def prediction_prefix(path, expected, repair=False):
    """Keep complete validated rows; only a torn final JSON record is repairable."""
    path = Path(path)
    if not path.exists(): return []
    records = []; offset = 0; torn = None; add_newline = False
    with path.open('rb') as handle:
        while True:
            line = handle.readline()
            if not line: break
            try: record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b'\n') or handle.read(1): raise ValueError('Corrupt non-final prediction record')
                torn = line; break
            assert len(records) < len(expected), 'Too many saved users'
            index, sample = expected[len(records)]
            for key in ('user_id','target','history'):
                assert record[key] == sample[key], (len(records), key)
            assert record['sample_index'] == index
            for name in ('native','late','guided'):
                value = record[name]
                assert len(value['ids']) == len(value['scores']) > 0
                assert len(set(value['ids'])) == len(value['ids'])
                assert all(math.isfinite(float(s)) for s in value['scores'])
            records.append(record); offset += len(line)
            add_newline = not line.endswith(b'\n')
    if repair and torn is not None:
        backup = path.with_name(path.name + f'.torn.{os.getpid()}.bin')
        with backup.open('xb') as handle: handle.write(torn); handle.flush(); os.fsync(handle.fileno())
        with path.open('r+b') as handle: handle.truncate(offset); handle.flush(); os.fsync(handle.fileno())
    if repair and add_newline:
        with path.open('ab') as handle: handle.write(b'\n'); handle.flush(); os.fsync(handle.fileno())
    return records


def process_identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        if fields[0] == 'Z': return None
        return dict(pid=int(pid), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), start=fields[19])
    except (FileNotFoundError, ProcessLookupError): return None


def identity_alive(identity):
    return bool(identity) and process_identity(identity['pid']) == identity
