# TCDR-P0 Frozen Preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Registered: 2026-07-29
- Parent Evidence: TCDR-N1 + TCDR-S0
- Test/Sports: forbidden
- Validation policy: conditional one-time read after training-only mechanism gate

## Matched training

For each domain, C0 and C1 start from the identical frozen GCDH-P0 C0
checkpoint.

- Fit: 256 unique latest training-prefix users, 128 head and 128 tail.
- Calibration: 128 additional unique users, 64 head and 64 tail; disjoint from
  fit.
- Both controls update only `decoder.block[-1]`.
- Two epochs, batch 16, AdamW, learning rate `2e-5`, no weight decay, gradient
  clipping 1.0.
- C0 objective: ordinary lexical CE.
- C1 objective: `lexical CE + 0.1 * TCDR`.
- C1 uses four deterministic TCDR-N1 matched pairs per batch, cycling through
  all 64 pairs. Correlations are computed over the 16 users in that batch.
- Both controls use identical users, epoch order, batches, steps and starting
  checkpoint. Epoch 2 is fixed; no checkpoint selection.

## Training-only mechanism gate

After training, evaluate C0 and C1 on the 128 calibration users and pair indices
0–15:

- C1 mean paired correlation excess must be at least 10% lower than C0;
- C1 lexical CE may exceed C0 by at most 1%;
- exact score, Trie, mapping and finite rates must be 100%;
- fit/calibration user overlap must be zero;
- source checkpoint SHA must remain unchanged.

This conjunction must pass in both domains. Failure gives
`STOP_TCDR_MECHANISM_GATE_FAILED` and validation remains unread.

## Locked validation

Only after the mechanism gate passes, select 512 users/domain by SHA-256 of
`tcdr-p0-validation-v1|dataset|user_id` from the pre-existing GCDH validation
user pool. User selection occurs before target access.

Before reading that user file, exercise the exact C0/C1 beam-50 generation and
candidate-mapping path on two calibration training-prefix users; both mapping
rates must equal 1.0.

C0 and C1 use the same beam 50, Trie, mapping, maximum length and length
penalty. Bootstrap unit is user with 1,000 replicates.

Both domains must satisfy:

- overall NDCG@10 relative gain `>= 1%`;
- overall NDCG relative-gain bootstrap lower bound `>= 0`;
- Recall@10 absolute gain `>= 0`;
- tail NDCG@10 relative gain `>= 1%`;
- tail NDCG relative-gain bootstrap lower bound `>= 0`;
- C0-hit/C1-miss rate `<= 0.5%`.

Pass: `TCDR_FREEZE_FOR_CONFIRMATION`. Any valid effect failure:
`STOP_TCDR_EFFECT_GATE_FAILED`. No loss-weight, epoch, pair, cohort or threshold
rescue is permitted.
