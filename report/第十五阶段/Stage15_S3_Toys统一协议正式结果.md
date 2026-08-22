# Stage15 S3：Toys 统一协议结果（进行中）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-22
- Verification Status: PARTIALLY_VERIFIED（S15-3A B3 admission 已裁决；B2 admission 尚未执行）
- Version Label: stage15_s3_toys_v0_admission_interim

## 当前结论

- B3：`FAIL_B3_S15_3A_EDIT_STATE_ADMISSION`
- B2：state build PASS；独立 512-event admission 待执行
- held evaluation：尚未打开
- test：未打开

当前结果不包含 efficacy 结论。B3 失败发生在 held evaluation 前的 edit-state Gate。

## B3 admission 证据

| Attempt | Layer rule | Position z-success | 结果 |
|---|---|---|---|
| attempt-1 | 复用 v0 probe `[5,5,5,5,5,4]` | positions 0–3=`[2,1,2,1]/4`；position 4=`0/4` | rc=1，held 未打开 |
| attempt-2 | clean-base train-only 6×6 probe；selected=`[5,5,3,5,0,0]` | positions 0–3=`[2,1,2,1]/4`；position 4=`0/4` | rc=1，held 未打开 |

attempt-2 probe 中 positions 4/5 在全部 6 层的 token accuracy 均为 0。更换到 clean-base probe 层后失败完全复现，因此 attempt-1 的失败不能再归因于单一 layer-map 迁移错误。

## 裁决纪律

以下 rescue 均不执行：放宽 legal probability threshold=`0.3`、增加每位置 requests、修改浅层 tie-break、换 seed 或事后选择 request。B3 不进入 S15-3B full validation。

B2 auxiliary drafter 在两次 admission attempt 中均得到相同 finite loss `9.36309439 → 9.24616081`，state 正常改变。为避免 B3 的方法级失败阻塞 B2，下一 Gate 拆为独立 B0+B2 512-event admission；仍只检查完整 beam/draft/verifier/redraft 路径、finite score、唯一 catalog top-50、资源与 artifact schema，不作 efficacy promotion。
