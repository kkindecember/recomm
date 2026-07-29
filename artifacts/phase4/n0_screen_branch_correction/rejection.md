# N0 Rejection: Static Trie Branching Correction

Date: 2026-07-29

Candidate: apply a deterministic path-score correction based on the number of
legal Trie children at each lexical-ID prefix.

The direction fails search-bounded novelty. *Towards Mitigating Length Bias in
Large Language Models for Recommendation* (LBR) explicitly accumulates path
uncertainty from the valid Trie branching set using Hartley entropy
`log2 |V_k|`. Related work also treats tree expressiveness and position
information gain in semantic-ID generation:

- https://arxiv.org/abs/2607.04270
- https://arxiv.org/abs/2605.06331
- https://arxiv.org/abs/2607.12425

Applying the same static correction to GRAM native lexical IDs would be a
backbone transfer rather than an independent method contribution.

Fixed decision: **`STOP_N0_BRANCH_CORRECTION_PRIOR_ART_OVERLAP`**. No N1 was
implemented and no validation, test, or Sports data were read.
