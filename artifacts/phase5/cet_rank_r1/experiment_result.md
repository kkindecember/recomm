## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-30T21:43:21+08:00
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

## Experiment Result

- **ID**: GRAM_PHASE5_CET_RANK_R1
- **Type**: training
- **Status**: completed
- **Command**: `bash experiment/phase5/run_phase5_cet_rank_r1.sh start`
- **Working Directory**: `/home/jiangtangyunzhi/projects/recomm`
- **Duration**: 833 seconds
- **Exit Code**: 0
- **Physical GPU**: 6
- **Machine Decision**: `STOP_CET_RANK_NOT_OPTIMIZABLE`

### Output Files

| File | Size |
|------|------|
| `artifacts/phase5/cet_rank_r1/summary.json` | 7,107 bytes |
| `artifacts/phase5/cet_rank_r1/Toys/summary.json` | 3,085 bytes |
| `artifacts/phase5/cet_rank_r1/Toys/decoder_last_layer.pt` | 16,788,030 bytes |
| `artifacts/phase5/cet_rank_r1/Beauty/summary.json` | 3,102 bytes |
| `artifacts/phase5/cet_rank_r1/Beauty/decoder_last_layer.pt` | 16,788,030 bytes |
| `artifacts/phase5/cet_rank_r1/gpu_telemetry.csv` | 7,553 bytes |

### Output Summary

- Toys: gamma 33.0465; rank-JS decrease 0.1938%; clean CE change +0.0806%;
  top-10 overlap change 0.
- Beauty: gamma 137.2267; rank-JS decrease 1.9307%; clean CE change +0.0231%;
  top-10 overlap change -1.5152 percentage points.
- Both domains passed every registered integrity and clean-CE safety check, but
  failed the rank-JS improvement and top-10 overlap gates.
- Validation, test, and Sports targets remained unread.
- CodeLlama reservation was restored to physical GPU6 after exit.

### Anomalies Detected

None. The negative machine decision is a valid scientific result, not an execution
anomaly.
