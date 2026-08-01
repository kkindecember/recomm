## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-31T20:16:19+08:00
- Verification Status: VERIFIED
- Version Label: `phase6_gacr_v3_failed_start_v1`

## Experiment Result

- **ID**: GRAM_PHASE6_GACR_V3_ATTEMPT_1
- **Status**: crashed before workload
- **Command**: `bash experiment/phase6/run_phase6_gacr_v3.sh start`
- **Exit signal**: HUP (129)
- **Scientific status**: `EXECUTION_INVALID_NO_SCIENTIFIC_RESULT`

The experiment runner exported `SESSION=gram_phase6_gacr_v3` and then invoked
`run_codellama.sh stop`. The resource script uses `SESSION` for its own tmux name,
so it killed the experiment session instead of the independent `codellama` session.
The worker received HUP before telemetry, candidate construction, training, or
validation began. It then restarted CodeLlama under the wrong
`gram_phase6_gacr_v3` name.

No workload PID or `summary.json` was produced. Sports and test were not read.
The runner was amended to invoke every resource operation with an explicit
`SESSION=codellama`; shell checks, generic status tests, and all 9 Phase 6 Python
tests pass. Per the no-automatic-retry policy, cleanup and restart require the
researcher's explicit confirmation.
