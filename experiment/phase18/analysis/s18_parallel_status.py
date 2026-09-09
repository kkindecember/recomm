"""Read-only monitor of the four authorized runs; writes a separate overview."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'artifacts/phase18/parallel_screen_20260909/status.json'
JOBS = {
    'diff_gram_beauty': ('artifacts/phase18/diff_gram/beauty/confirm_v1', 's18_diff_gram_fixed'),
    'etegrec_beauty': ('artifacts/phase18/etegrec/beauty/run_v1', 'protocol.s18_etegrec'),
    'etegrec_toys': ('artifacts/phase18/etegrec/toys/run_v1', 's18_etegrec_toys'),
    'rearec_beauty': ('artifacts/phase18/rearec/beauty/run_v1', 's18_rearec'),
}


def latest_event(path):
    if not path.exists():
        return None
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size - 65536))
        lines = f.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def snapshot():
    observed = datetime.now(timezone.utc)
    jobs = {}
    for name, (directory, module) in JOBS.items():
        path = ROOT / directory / 'status.json'
        status = json.loads(path.read_text())
        event = latest_event(path.parent / 'events.jsonl')
        pid = status.get('pid')
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
            alive = module in command
        except FileNotFoundError:
            alive = False
        except PermissionError:
            alive = None
        timestamps = [status.get('updated_at', status.get('started_at'))]
        if event:
            timestamps.append(event.get('time'))
        activity = max(datetime.fromisoformat(s) for s in timestamps if s)
        jobs[name] = dict(status_path=str(path.relative_to(ROOT)), state=status['state'],
                          process_alive=alive, seconds_since_activity=(observed - activity).total_seconds(),
                          raw_status=status, latest_event=event)
    terminal = all(j['process_alive'] is False for j in jobs.values())
    result = dict(updated_at=observed.isoformat(), monitor_running=not terminal, jobs=jobs,
                  note='RQ progress may update events before raw status; inspect latest_event. This monitor never controls training.')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(OUTPUT)
    return terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        if snapshot() or not args.watch:
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
