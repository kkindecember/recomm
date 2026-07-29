# LNDR-N1 Frozen Preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Registered: 2026-07-28
- Data Scope: catalog metadata + training prefixes only
- Parent: `artifacts/phase4/lndr_n0/mechanism_brief.md`
- Validation/test/Sports access: forbidden

## Audit question

LNDR-N1 tests whether the frozen GRAM checkpoint simultaneously exhibits:

1. same lexical token at same-depth Trie nodes whose descendant-item metadata meanings differ;
2. decoder-state separation between those node occurrences;
3. a shared-readout gold-sibling margin deficit concentrated on high-polysemy edges.

All three chains must pass independently on Toys and Beauty. No model parameter is
updated. Native T5 input-embedding means are used only as a frozen catalog metadata
representation; decoder states and lexical logits come from the frozen GCDH-P0 C0
checkpoint.

## Cohorts and controls

- A node is the edge `(parent prefix, child lexical token)`.
- Semantic comparisons are restricted to the same token and same depth.
- Each node must cover at least three descendant catalog items.
- High polysemy is frozen at mean cross-node centroid cosine distance `>= 0.10`.
- Controls are structurally unique eligible token-depth nodes or reused nodes with
  semantic distance `<= 0.05`; intermediate nodes are excluded from the contrast.
- Readout controls are exactly balanced within depth × head/tail × legal-child-count
  bin (`2`, `3–4`, `5–8`, `9+`) using the frozen hash rule.
- A margin deficit means the frozen gold lexical logit is no greater than the best
  legal nongold sibling logit.
- State separation uses cosine distance, comparing same-node with different-node
  pairs for an identical lexical token and depth. AUC `0.5` is chance.

## Frozen scientific conjunction

Per domain, all gates in
`artifacts/phase4/configs/lndr_n1_preregistered.json` must pass. The principal
effect gates are median semantic distance `>= 0.10`, state-separation AUC `>= 0.65`,
high-polysemy deficit rate `>= 0.25`, and matched deficit-rate difference
`>= 0.05`, with the preregistered support minima.

The dual-domain conjunction produces `LNDR_S0_DESIGN_ALLOWED`. Any scientifically
valid failure produces `STOP_LNDR_NO_NODE_POLYSEMY_DEFICIT`; thresholds, cohorts,
representations, sample size, checkpoint, and seed may not be changed in response.
Only a proven integrity failure permits an exact rerun.

## Frozen execution

- Seed: `2023`
- Samples: 1,024 unique users/domain, balanced 512 head and 512 tail
- Device: physical GPU 4, exposed as process-local CUDA 0
- Optimizer steps: zero
- Timeout: 4 hours
- Code and configuration SHA are recorded in the locked JSON configuration.
