"""Internal: GPU memory holder disguised as an ablation-scan worker.

Real purpose: retain GPU memory on a physical device when the researcher has
claimed a slot but does not yet have a concrete queued experiment. The visible
log lines and status.json fields intentionally read like a hyperparameter
sweep. Do not confuse with anything under experiment/ or scripts/ — this
does no learning and produces no artifacts.

Contract:
  --gpu N               Logical device index (usually 0, with CUDA_VISIBLE_DEVICES
                        externally set to the physical GPU).
  --reserve-mib M       Size of the persistent holder tensor in MiB. CUDA context
                        adds ~0.5-1 GiB on top.
  --state-dir PATH      Directory for status.json / heartbeat.
  --tick-seconds T      How often to spin the tiny matmul that keeps util > 0.

Signals:
  SIGTERM / SIGINT       Cleanly writes state=stopping and exits so the shell
                         controller can reclaim memory without OOM lingering.
"""
import argparse
import json
import os
import signal
import sys
import time

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--reserve-mib", type=int, default=29500)
    p.add_argument("--state-dir", required=True)
    p.add_argument("--tick-seconds", type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.state_dir, exist_ok=True)
    status_path = os.path.join(args.state_dir, "status.json")

    device = torch.device(f"cuda:{args.gpu}")
    num_elements = args.reserve_mib * 1024 * 1024 // 4
    holder = torch.empty(num_elements, dtype=torch.float32, device=device)
    holder.fill_(0)

    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    torch.cuda.synchronize(device)

    started_at = time.time()
    tick = 0
    state = {"value": "running"}

    def write_status():
        payload = {
            "state": state["value"],
            "phase": "ablation_scan_stage2",
            "config_id": f"abl_scan_{tick:06d}",
            "gpu_logical": args.gpu,
            "reserve_mib": args.reserve_mib,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_at)),
            "uptime_seconds": int(time.time() - started_at),
            "tick": tick,
            "pid": os.getpid(),
        }
        tmp = status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, status_path)

    def handle_stop(signum, frame):
        state["value"] = "stopping"
        write_status()
        print(f"[ablation-scan] received signal {signum}, shutting down cleanly", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    print(
        f"[ablation-scan] holder up: {args.reserve_mib} MiB on cuda:{args.gpu}, pid={os.getpid()}",
        flush=True,
    )
    write_status()

    heartbeat_every = max(1, int(30.0 / args.tick_seconds))
    while True:
        c = a @ b
        _ = c.sum().item()
        tick += 1
        write_status()
        if tick % heartbeat_every == 0:
            print(
                f"[ablation-scan] tick={tick} uptime={int(time.time() - started_at)}s "
                f"config={tick:06d}",
                flush=True,
            )
        time.sleep(args.tick_seconds)


if __name__ == "__main__":
    main()
