# GCGD-P0 failed start record — 2026-08-02

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-02T13:36:30+08:00
- Verification Status: UNVERIFIED
- Version Label: `gcgd_p0_failed_start_v1`

## Experiment Result

- **ID**: `GRAM_PHASE7_GCGD_P0_LINEAGE_V1`
- **Type**: generic / CPU-only lineage audit
- **Status**: crashed
- **Command**: `bash experiment/phase7/run_phase7_gcgd_p0.sh start`
- **Working Directory**: `/home/jiangtangyunzhi/projects/recomm`
- **Exit Code**: `1`
- **Failure point**: Python module import, before tokenizer or dataset loading
- **Error**: `ModuleNotFoundError: No module named 'experiment'`

## Integrity and resources

- No input dataset was read and no scientific summary was produced.
- No checkpoint, fresh validation, test prediction, or Sports data was read.
- CodeLlama remained running on physical GPU0 with the requested reservation.
- The runner did not retry automatically.

## Repair scope

The direct-script entry point now inserts the repository root into `sys.path` before importing
`experiment.phase7.gcgd_v1`. No graph definition, train slice, tokenizer rule, metric, input,
output, timeout, or resource policy changed. A new implementation SHA must be locked before a
researcher-approved retry.
