"""Read-only experiment observer; writes one separate overview/status.json.

Does not modify, restart, or signal either workload. State reflects execution,
not whether a method improves recommendation accuracy.
"""
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'artifacts/phase20/overview'
TERMINAL = {'COMPLETED', 'FAILED', 'STOPPED', 'ORPHANED'}


def read(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def recent_events(path):
    if not path.exists():
        return []
    with path.open('rb') as f:
        size = f.seek(0, 2)
        f.seek(max(0, size-131072))
        lines = f.read().splitlines()
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            continue
    return result


def progress(detail, directory):
    result = {'stage': detail.get('stage', 'initializing'), 'round_or_epoch': detail.get('round'),
              'examples': detail.get('examples'), 'total': detail.get('total'),
              'optimizer_step': detail.get('step'),
              'last_update': detail.get('time', detail.get('updated_at'))}
    if result['examples'] is not None and result['total']:
        result['current_step_percent'] = round(100*result['examples']/result['total'], 2)
    events = [e for e in recent_events(directory/'events.jsonl')
              if e.get('stage') == detail.get('stage') and e.get('round') == detail.get('round')
              and e.get('total') == detail.get('total') and e.get('examples') is not None]
    if len(events) >= 2:
        a, b = events[max(0, len(events)-8)], events[-1]
        dt = b.get('elapsed_seconds', 0)-a.get('elapsed_seconds', 0)
        dn = b['examples']-a['examples']
        if dt > 0 and dn > 0:
            rate = dn/dt
            result['recent_examples_per_second'] = round(rate, 3)
            if detail.get('total'):
                result['estimated_current_step_remaining_hours'] = round(
                    max(0, detail['total']-detail['examples'])/rate/3600, 2)
                result['eta_scope'] = 'current step only; not the complete experiment'
    return result


def job(name, status_path, default_output, prior):
    outer = read(status_path)
    if name == 'SPRec':
        if 'Beauty' in outer.get('completed_domains', []):
            default_output = 'artifacts/phase20/sprec_gram_beauty_v1'
        directory = ROOT/outer.get('output', default_output)
        detail = read(directory/'status.json') or outer.get('child_status', {})
    else:
        directory = ROOT/default_output
        detail = outer
    state = outer.get('state', 'NOT_STARTED')
    pid = outer.get('supervisor_pid', detail.get('pid'))
    if state in ('RUNNING', 'STOPPING') and pid and not Path(f'/proc/{pid}').exists():
        state = 'ORPHANED'
    result = {'state': state, 'finished': state in TERMINAL,
              'status_file': str(status_path.relative_to(ROOT)),
              'detail_file': str((directory/'status.json').relative_to(ROOT)),
              'summary_file': str((directory/'summary.json').relative_to(ROOT)),
              'dataset': outer.get('dataset', detail.get('dataset', 'Toys')),
              'progress': progress(detail, directory),
              'rough_total_time_estimate': prior,
              'execution_success_is_not_accuracy_gain': True}
    summary = read(directory/'summary.json')
    if summary:
        result['available_result'] = {k: summary.get(k) for k in (
            'raw_ndcg_delta_vs_gram', 'combined_ndcg_delta_vs_original_pcrf', 'decision', 'positive_signal')}
    if outer.get('error'):
        result['error'] = outer['error']
    return result


def snapshot():
    jobs = {
        'SPRec': job('SPRec', ROOT/'artifacts/phase20/status.json', 'artifacts/phase20/sprec_gram_toys_v1',
            {'Toys_hours': [24, 72], 'with_conditional_Beauty_days': [2, 6],
             'basis': 'planning window revised for observed throughput fluctuation; no complete training round timed yet; deadline may stop an incomplete run', 'per_domain_deadline_hours': 72}),
        'S-DPO': job('S-DPO', ROOT/'artifacts/phase20/sdpo_gram_toys_v1/status.json', 'artifacts/phase20/sdpo_gram_toys_v1',
            {'Toys_hours': [24, 48], 'basis': 'planning window revised for observed throughput fluctuation; no complete epoch timed yet; deadline may stop an incomplete run', 'deadline_hours': 48})}
    finished = all(j['finished'] for j in jobs.values())
    success = finished and all(j['state'] == 'COMPLETED' for j in jobs.values())
    value = {'state': ('COMPLETED' if success else 'FINISHED_WITH_ERRORS') if finished else 'RUNNING',
             'all_finished': finished, 'all_succeeded': success,
             'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'observer_pid': os.getpid(),
             'refresh_seconds': 30, 'jobs': jobs,
             'note': 'Only all_finished=true means both configured jobs have ended. COMPLETED is execution completion, not a positive research result.'}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT/'status.tmp.json'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(OUTPUT/'status.json')
    return finished


def main():
    started = time.time()
    while True:
        if snapshot():
            return
        if time.time()-started > 7*86400:
            path = OUTPUT/'status.json'
            value = read(path)
            value['observer_stopped'] = '7-day observer limit; inspect workload files directly'
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
