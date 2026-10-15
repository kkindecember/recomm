"""Stop only this thread's named candidate jobs, preserving all artifacts."""
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path(__file__).resolve().parents[2]
MODULES={'experiment.phase21.compact_candidate_gate','experiment.phase21.candidate_gate_resource_queue','experiment.phase21.resume_candidate_gate_generation'}


def write(p,value):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def main():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=(p/'cmdline').read_bytes().split(b'\0')
            args=[x.decode() for x in args if x]
            if '-m' not in args:continue
            module=args[args.index('-m')+1]
            if module not in MODULES:continue
            found.append((int(p.name),module,(p/'stat').read_text().split()[21]))
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    found.sort(key=lambda row:row[1]!='experiment.phase21.candidate_gate_resource_queue')
    for pid,module,start in found:
        try:
            assert Path(f'/proc/{pid}/stat').read_text().split()[21]==start
            os.kill(pid,signal.SIGTERM)
        except (FileNotFoundError,ProcessLookupError):pass
    end=time.monotonic()+20
    while time.monotonic()<end and any(Path(f'/proc/{p}').exists() for p,_,_ in found):time.sleep(.5)
    alive=[]
    for pid,module,start in found:
        p=Path(f'/proc/{pid}/stat')
        if p.exists() and p.read_text().split()[21]==start and p.read_text().split()[2]!='Z':alive.append(pid)
    assert not alive,('Graceful stop incomplete',alive)
    reason='User requested selecting experts before continuing candidate weights; completed outputs retained.'
    paths=[ROOT/'artifacts/phase21/compact_candidate_gate_20261014_v1/status.json']
    g=ROOT/'artifacts/phase21/candidate_gate_generation_resume_20261014_v1'
    paths += [g/'status.json',g/'Toys/status.json',g/'Beauty/status.json',g/'Toys/queue_status.json']
    for p in paths:
        old=json.loads(p.read_text())
        if old.get('state') in ('COMPLETE','COMPLETED'):continue
        write(p,dict(state='PAUSED_BY_USER',reason=reason,previous=old))
    write(g/'pause_receipt.json',dict(reason=reason,stopped=[dict(pid=p,module=m) for p,m,_ in found],all_stopped=True))
    print(json.dumps(dict(stopped=len(found),state='PAUSED_BY_USER')))


if __name__=='__main__':main()
