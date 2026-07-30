# CET Rank-R0G blocked start record

## Material Passport

- Experiment: `GRAM_PHASE5_CET_RANK_R0G`
- Attempted start: 2026-07-30 15:28:27 +08:00
- Final status time: 2026-07-30 15:30:28 +08:00
- Verification Status: `RESOURCE_BLOCKED_BEFORE_AUDIT`
- Scientific decision: none

## Frozen command

```bash
bash experiment/phase5/run_phase5_cet_rank_r0g.sh start
```

## Outcome

The persistent tmux worker started and released the CodeLlama reservation, but the
GPU3 free-memory preflight did not reach the preregistered minimum of 30,720 MiB.
The wrapper exited with code 3, did not enter either dataset audit arm, did not
retry, and invoked the resource-restoration path.

A read-only post-event snapshot showed physical GPU3 using 30,818 MiB of 49,140
MiB, with 17,752 MiB free. `nvidia-smi` attributed 30,808 MiB to an external
Python process (PID 109756 at the time of the snapshot). This process was outside
the experiment scope and was not modified or terminated.

## Evidence-integrity status

- Rank-R0G per-user audit outputs created: no
- Optimizer updates: no (R0G is read-only by design)
- C1 checkpoint modified: no
- Validation target read: false
- Test read: false
- Sports read: false
- Frozen Python/config/user/checkpoint hashes changed: no
- Automatic retry: no

## Routing

Record this event as `R0G_START_BLOCKED_GPU3_RESOURCE_GATE_NO_AUDIT`, not as an
optimization, gradient, integrity, or scientific STOP decision. An exact launch
may be requested only after GPU3 satisfies the frozen resource gate and requires
fresh explicit researcher confirmation.
