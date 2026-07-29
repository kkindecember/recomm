# TCDR-S0 Frozen Correctness Preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Registered: 2026-07-29
- Parent Evidence: TCDR-N1
- Validation/test/Sports: forbidden

## Scope

S0 tests only whether the TCDR loss is a correct differentiable training
primitive. It does not estimate recommendation effect or select a production
training recipe.

## Frozen cohort and loss

- Per domain, reuse the first 8 deterministic TCDR-N1 training-prefix users.
- Reuse pair indices 0–7 from the frozen N1 pair CSV, without selecting on their
  observed correlation excess.
- Encode each user context once and detach the frozen encoder state.
- Compute differentiable length-normalized legal-child log scores for every
  close/control endpoint using the unchanged lexical Trie.
- For each matched pair:

```text
L_pair = relu(corr_user(close_left, close_right)
              - corr_user(far_left, far_right))
L_TCDR = mean(L_pair)
```

Pearson correlation uses epsilon `1e-8`.

## Optimization smoke

- Freeze the encoder, embeddings, LM head and all decoder layers except the
  final decoder block.
- Optimize `L_TCDR` for exactly 5 steps with AdamW, learning rate `1e-4`, no
  weight decay and gradient clipping at 1.0.
- Ordinary lexical CE is monitored but is not part of the five-step smoke
  objective.
- The intended later method remains ordinary lexical CE plus a frozen TCDR
  weight; S0 does not choose that weight.

## Correctness conjunction

Both domains must satisfy:

- `lambda=0` total-loss identity max difference `<= 1e-7`;
- initial TCDR loss is finite and strictly positive;
- trainable gradient is finite and has norm `> 0`;
- all legal-path scores and correlations are finite;
- final TCDR loss decreases by at least 1%;
- monitored lexical CE relative increase is at most 10%;
- at least one trainable parameter changes;
- source checkpoint SHA is unchanged;
- pair/user counts are exactly 8/8;
- no validation/test/Sports read.

Pass: `TCDR_S0_CORRECTNESS_PASS`. Any valid failure:
`STOP_TCDR_S0_CORRECTNESS_FAILED`. No automatic retry or hyperparameter rescue.

