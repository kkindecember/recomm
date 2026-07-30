## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-30T21:11:34+08:00
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

## Experiment Result

- **ID**: GRAM_PHASE5_CET_RANK_R0G
- **Type**: analysis
- **Status**: completed
- **Command**: `bash experiment/phase5/run_phase5_cet_rank_r0g.sh start`
- **Working Directory**: `/home/jiangtangyunzhi/projects/recomm`
- **Duration**: 227 seconds
- **Exit Code**: 0
- **Physical GPU**: 6
- **Machine Decision**: `CET_R0G_DIRECT_RANK_GRADIENT_DISTINCT`

### Output Files

| File | Size |
|------|------|
| `artifacts/phase5/cet_rank_r0g/summary.json` | 3,999 bytes |
| `artifacts/phase5/cet_rank_r0g/Toys/summary.json` | 1,400 bytes |
| `artifacts/phase5/cet_rank_r0g/Toys/per_user.csv` | 8,387 bytes |
| `artifacts/phase5/cet_rank_r0g/Beauty/summary.json` | 1,404 bytes |
| `artifacts/phase5/cet_rank_r0g/Beauty/per_user.csv` | 8,338 bytes |
| `artifacts/phase5/cet_rank_r0g/gpu_telemetry.csv` | 2,068 bytes |

### Output Summary

- Toys: 34 masked users; median gradient cosine 0.0882; 95% bootstrap interval
  for mean cosine [0.0515, 0.2186].
- Beauty: 33 masked users; median gradient cosine 0.1614; 95% bootstrap interval
  for mean cosine [0.0299, 0.2741].
- Both domains had 100% rank-loss signal coverage and 100% direct-rank nonzero-gradient
  coverage. All registered integrity checks passed.
- Validation, test, and Sports targets remained unread.
- CodeLlama reservation was restored to physical GPU6 after exit.

### Anomalies Detected

None during this run. The prior GPU3 blocked start remains a separate historical record.
