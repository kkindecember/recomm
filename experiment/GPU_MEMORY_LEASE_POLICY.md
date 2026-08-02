# GPU memory lease policy

Every new long-running experiment must hold a **total 30 GiB GPU lease** for
its complete runtime.  The workload's own CUDA allocations count toward that
lease; the runner reserves only the difference with
`experiment/gpu_memory_lease.py`.

Before launch, record a conservative `expected_workload_peak_mib` in the
experiment's frozen configuration.  The runner starts the sidecar first with:

```bash
python experiment/gpu_memory_lease.py --gpu 0 --total-lease-mib 30720 \
  --expected-workload-peak-mib <recorded-peak> --status-path <output>/gpu_lease_status.json
```

The sidecar reservation is `30720 - expected_workload_peak_mib`.  The runner
must keep the sidecar alive until the workload exits, record its status path,
and reject plans whose declared peak exceeds 30 GiB.  The GPU admission gate
must still require enough free memory for both the workload peak and sidecar.
