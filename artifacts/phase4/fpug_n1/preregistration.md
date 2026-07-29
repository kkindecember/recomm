# FPUG-N1 Frozen Preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Registered: 2026-07-28
- Data Scope: unique-user training prefixes only
- Parent: `artifacts/phase4/fpug_n0/mechanism_brief.md`
- Validation/test/Sports access: forbidden

## Frozen audit

- Each domain uses 512 unique users, balanced 256 head / 256 tail.
- Samples must have at least five training-prefix history items.
- The frozen GCDH-P0 C0 checkpoint is evaluated without parameter updates.
- The encoder runs once per batch. Each counterfactual masks exactly one
  fine-grained item passage only at decoder cross-attention; the coarse lexical
  history passage is byte-identical and remains fully visible.
- Gold loss is legal-child CE averaged across competitive non-EOS Trie steps.
- A passage is harmful when its removal improves mean legal-child CE by at least
  0.05 nats.
- Passage index 1 is most recent because GRAM serializes detailed history in reverse
  chronological order; the last active detail passage is oldest.
- Dynamic utility must beat the fixed oldest-removal baseline and harmful passages
  must cover at least three preregistered recency quartiles.

Both domains must pass every gate in the locked JSON. A valid scientific failure
closes FPUG; only a proven integrity error permits an exact rerun.
