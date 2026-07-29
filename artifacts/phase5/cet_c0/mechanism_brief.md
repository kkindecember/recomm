# CET-C0 Mechanism Brief

## Material Passport

- Created: 2026-07-29
- Scope: search-bounded mechanism and novelty review
- Verification Status: ANALYZED
- Validation/Test/Sports Read: false
- Decision: `CET_C0_PASS_WITH_TRANSFER_NARROWING`

## Research question

Can a GRAM generator be trained to preserve its legal Trie-child belief under structured removal of
non-essential fine-grained history passages, while keeping the original full-evidence inference path?

## Independent basis

1. [GRAM](https://aclanthology.org/2025.acl-long.1596/) uses semantic-to-lexical translation and
   multi-granular late fusion; it does not report a clean-view/partial-view consistency objective.
2. [R-Drop](https://arxiv.org/abs/2106.14448) shows that constraining predictions from stochastic
   subnetworks can reduce train/inference inconsistency, but its perturbation is ordinary model
   dropout and its output support is not a recommendation Trie.
3. [Passage-Mask](https://aclanthology.org/2022.emnlp-main.260/) treats passage masking as reader
   regularization, but learns masks for retrieval-reader tasks and does not preserve a clean-only
   generative recommender at inference.
4. [CORD](https://aclanthology.org/2025.naacl-short.66/) uses context perturbation, consistency and
   rank distillation for RAG robustness; its task, perturbation and output space differ from GRAM's
   ordered interaction passages and legal item-path children.
5. Recent generative-recommendation work already covers information-gain token weighting and
   multi-target objectives, while reinforced preference optimization covers constrained-beam hard
   negatives and ranking rewards. CET therefore must not be presented as a general fix for token
   weighting, negative modeling or beam exposure.

## Minimal method

For each training prefix, construct a clean full-passage view and one perturbed view. Always retain
the coarse passage and newest fine passage; independently mask each remaining fine passage with
fixed probability `q`. At each gold prefix, normalize logits only over legal Trie children and use
the clean distribution as a stopped-gradient anchor:

```text
L = CE(clean) + alpha * CE(perturbed)
  + beta * KL(stopgrad(p_clean_legal) || p_perturbed_legal)
```

Inference is exactly the original clean GRAM path.

## Non-claims

CET does not claim the first use of:

- dropout or consistency regularization;
- passage masking;
- RAG context perturbation;
- KL distillation;
- Trie-constrained generation.

The only candidate contribution is their task-specific composition: structured historical-evidence
subsampling with a clean anchor and legal-child consistency for multi-granular generative
recommendation, plus evidence about when that transfer does or does not improve ranking.

## Closest-overlap judgment

The components have strong prior art, which is desirable for effect prior but requires a narrow
claim. In the bounded primary-source review, no work was found that simultaneously uses:

1. GRAM-style coarse plus per-item fine passages;
2. training-only structured fine-passage subsampling;
3. clean-to-perturbed distribution consistency restricted to current catalog-Trie children; and
4. unchanged full-evidence inference.

This is a search-bounded statement, not an absolute first claim. A broader systematic literature
review is required before manuscript submission.

## Falsifiers

- augmentation-only matches CET, so legal-child consistency adds no value;
- full-vocabulary KL matches CET, so Trie localization adds no value;
- consistency improves perturbed-view CE but not clean-view Recall/NDCG;
- gains require changing inference passages or dataset-specific hyperparameters;
- compute-matched R-Drop explains the effect.

## Fixed decision

**`CET_C0_PASS_WITH_TRANSFER_NARROWING`**

CET may proceed to correctness smoke. This decision does not claim recommendation improvement and
does not authorize validation-driven hyperparameter search.
