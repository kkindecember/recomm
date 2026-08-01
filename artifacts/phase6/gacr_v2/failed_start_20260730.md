## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-30T22:24:34+08:00
- Verification Status: VERIFIED
- Version Label: phase6_gacr_v2_failed_start_v1

## Experiment Result

- **ID**: GRAM_PHASE6_GACR_V2_ATTEMPT_1
- **Type**: training
- **Status**: crashed
- **Command**: `bash experiment/phase6/run_phase6_gacr_v2.sh start`
- **Working Directory**: `/home/jiangtangyunzhi/projects/recomm`
- **Exit Code**: 1

### Failure

The process stopped before candidate generation while opening
`artifacts/phase4/gcdh_p0/Toys/C1/model.pt`. Both Toys and Beauty C1 checkpoint
files were absent. Their historical training summaries and expected SHA256
values remained present.

No development cohort, test data, or Sports data was read. No scientific result
was produced and `summary.json` was not created.

### Recovery Authorization

The researcher explicitly authorized continuation. Recovery is limited to exact
reconstruction of the two missing C1 checkpoints from the locked GCDH-P0 config
on physical GPU6. GACR-v2 may continue only after exact historical SHA256
matches.
