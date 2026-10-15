"""Finish supervisor handoff after inherited workers exit; no new experiments."""
import os
from pathlib import Path
import signal
import time

from experiment.phase21 import expert_selection as E
from experiment.phase21.rebalance_expert_selection import proc


def main():
    receipt=E.read(E.OUT/'resource_override.json')
    old_pid=receipt['old_supervisor_pid'];new_pid=receipt['new_supervisor_pid']
    inherited=receipt['running_workers_preserved'];started=time.monotonic()
    note=E.OUT/'handoff_finalizer_status.json'
    E.write(note,dict(state='WAITING_FOR_INHERITED_WORKER_EXIT',updated_at=E.stamp()))
    # Do not kill the old session leader while its worker still owns the terminal.
    while any((p:=proc(pid)) is not None and p['state']!='Z' for pid in inherited):
        if time.monotonic()-started>86400:raise TimeoutError('Inherited worker handoff exceeded 24 hours')
        time.sleep(10)
    old=proc(old_pid)
    if old and old['state']!='Z' and old['args'][-3:]==['-m','experiment.phase21.expert_selection','supervise']:
        os.kill(old_pid,signal.SIGTERM)
        if old['state'] in ('T','t'):os.kill(old_pid,signal.SIGCONT)
    E.write(note,dict(state='WAITING_FOR_NEW_SUPERVISOR',old_supervisor_retired=True,updated_at=E.stamp()))
    while (p:=proc(new_pid)) is not None and p['state']!='Z':
        if time.monotonic()-started>86400:raise TimeoutError('New supervisor handoff exceeded 24 hours')
        time.sleep(10)
    states={a+'/'+d:E.read(E.OUT/a/d/'status.json') for a in E.ARMS for d in ('Toys','Beauty')}
    # A non-parent supervisor can lose access to an exit code after reaping by
    # the original parent. Require each worker's independently verified result.
    complete=all(s['state']=='COMPLETED' for s in states.values())
    if complete:
        E.bind()
        for a in E.ARMS:
            for d in ('Toys','Beauty'):
                r=E.read(E.OUT/a/d/'summary.json')
                assert r['users']==512 and r['model_unchanged'] and r['verified_by_sort_and_count']
                assert E.read(E.OUT/a/d/'lease_status.json')['state']=='released'
                E.T.verify_domain(a,d)
        E.summarize()
        previous=E.read(E.OUT/'status.json')
        E.write(E.OUT/'status.json',dict(state='COMPLETED',scope='expert_selection_only',updated_at=E.stamp(),
            resource_override='resource_override.json',worker_results_verified=True,
            inherited_exit_code_may_be_unavailable=True,previous_supervisor_result=previous,
            test_read=False,candidate_weights_scheduled=False))
    E.write(note,dict(state='COMPLETED' if complete else 'INCOMPLETE',all_worker_results_complete=complete,updated_at=E.stamp()))


if __name__=='__main__':main()
