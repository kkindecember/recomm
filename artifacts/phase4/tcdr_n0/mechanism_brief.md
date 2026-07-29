# TCDR-N0 Mechanism Brief

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: deep-research → plan
- Created: 2026-07-29
- Validation/test/Sports: forbidden
- Parent evidence: independent tree-expressiveness literature; not FPUG rescue

## Research question

GRAM assigns every item one unique native-token path and scores items
autoregressively through a shared lexical Trie. Does that tree force items with
long common prefixes to have overly correlated score responses across users,
even when their training-only collaborative incidence is nearly unrelated?

## Mechanism

**Tree-Coupling Decorrelation Regularization (TCDR)** targets excess
cross-user score covariance:

```text
for catalog items (i, j):
  tree_close(i, j) and collaborative_similarity(i, j) <= epsilon

L_tcdr = max(0, corr_u[s(i|u), s(j|u)] - calibrated_far_pair_corr - delta)
```

The eventual loss would be training-only and auxiliary to ordinary lexical CE.
It would not create a new catalog head, rerank frozen candidates, alter lexical
IDs, add latent tokens, or change Trie-constrained inference.

## Independent basis and novelty boundary

- GRAM represents each item by one unique hierarchical native-token identifier
  and generates the identifier autoregressively:
  https://aclanthology.org/2025.acl-long.1596/
- Hou et al. show that autoregressive semantic-ID trees can couple the
  probabilities of tree-near items across users and propose Latte, which adds a
  latent token to form multiple trees:
  https://arxiv.org/abs/2605.06331
- The hourglass literature separately shows that tree/token utilization can
  constrain generative retrieval:
  https://aclanthology.org/2024.emnlp-industry.50/

TCDR does **not** claim that tree coupling is newly discovered. Its bounded
method difference is to retain GRAM's single native lexical Trie and penalize
only excess cross-user response covariance for tree-close,
collaboratively-dissimilar pairs, calibrated against popularity/collaboration
matched tree-far controls.

## Separation from prior phase-4 directions

- CHPR compares gold and proposer negatives inside one user at their first Trie
  divergence; TCDR compares two items' score vectors across a fixed user panel.
- LNDR studies reuse of one lexical token under semantically different parents;
  TCDR does not require token reuse or a new readout.
- SCDL changes sibling token assignment; TCDR retains identifiers unchanged.
- GCDH/GACR/CCRR add or fit ranking capacity; TCDR is a training-time structural
  regularizer on the original generator.
- FPUG modifies encoder passage utility; TCDR concerns decoder-tree
  expressiveness and uses no passage-removal evidence.

## Falsifiable N1 premise

On frozen Toys and Beauty checkpoints, tree-close item pairs with training-only
collaborative cosine at most 0.05 must show materially stronger cross-user exact
path-score correlation than matched tree-far controls. If this is not present
in both domains, TCDR stops before implementation.

