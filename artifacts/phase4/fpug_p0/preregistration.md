# FPUG-P0 Frozen Preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Registered: 2026-07-29
- Parent Evidence: FPUG-N1 + FPUG-S0
- Sports/test: forbidden
- Validation policy: one locked Toys/Beauty read after training-prefix calibration

## Method

The frozen GCDH-P0 C0 backbone is retained. FPUG adds one domain-trained bounded
detail-passage gate:

```text
g_i = 1 + 0.5 * tanh(Linear([q, p_i, q*p_i, recency_i]))
```

Only gate parameters are optimized with the original lexical CE. The coarse
history passage is never gated. The decoder, lexical IDs, Trie, beam size and
candidate mapping are unchanged.

## Training and calibration

- 1,024 fit samples/domain (512 head, 512 tail), unique users.
- 256 calibration samples/domain (128 head, 128 tail), unique users and disjoint
  from fit users.
- All samples are training prefixes with at least five history items.
- Five epochs, batch 16, AdamW, learning rate `1e-3`, no weight decay.
- Backbone remains frozen.
- Each epoch is evaluated on training-prefix calibration lexical CE.
- A single shared epoch is selected by maximum mean relative calibration CE
  decrease across Toys and Beauty. Ties choose the earlier epoch.

## Locked validation cohort

For each domain, select the first 512 users by SHA-256 of
`fpug-p0-validation-v1|dataset|user_id` from the pre-existing GCDH validation-user
pool. Selection uses user IDs only and occurs before reading `sequence[-2]`.

Baseline and FPUG use beam 50, identical Trie, maximum length and length penalty.
The bootstrap unit is user, 1,000 replicates with seed 2023. Tail is defined from
training-prefix item popularity.

Before reading the validation-user file, the identical baseline/gated beam-50
generation path is exercised on two deterministic training-prefix samples (one
head and one tail). All returned sequences must map to candidates.

## Frozen effect conjunction

Both domains must satisfy:

- overall NDCG@10 relative gain `>= 1%`;
- overall NDCG relative-gain bootstrap lower bound `>= 0`;
- Recall@10 absolute gain `>= 0`;
- tail NDCG@10 relative gain `>= 1%`;
- tail NDCG relative-gain bootstrap lower bound `>= 0`;
- baseline-hit/gated-miss rate `<= 0.5%`.

Pass: `FPUG_FREEZE_FOR_CONFIRMATION`. Any valid failure:
`STOP_FPUG_EFFECT_GATE_FAILED`. No epoch, gate bound, feature, loss, seed, cohort
or threshold rescue is permitted.
