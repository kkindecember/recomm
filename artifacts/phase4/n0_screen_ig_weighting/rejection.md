# N0 Rejection: Information-Gain Token Weighting

## Material Passport

- Origin Skill: academic-research-suite / deep-research + experiment-agent
- Created: 2026-07-29
- Status: search-bounded N0 rejection
- Experiment/data access: none

## Candidate

Use catalog-subtree information gain to assign fixed weights to GRAM lexical-ID
token CE, emphasizing identifier decisions that reduce more candidate
uncertainty. Inference would remain unchanged.

## Rejection evidence

This is not sufficiently independent:

- *Token-Weighted Multi-Target Learning for Generative Recommenders with
  Curriculum Learning* explicitly identifies equal token weighting as a
  semantic-ID mismatch and proposes information-gain-based token weighting,
  including front-greater and frequency weighting:
  https://arxiv.org/abs/2601.17787
- *Where Reasoning Matters* explicitly measures semantic-ID position
  information gain and allocates more computation to high-IG positions:
  https://arxiv.org/abs/2607.12425
- LOHRec already exploits hierarchy/order-aware generative recommendation:
  https://aclanthology.org/2025.findings-emnlp.977/

Applying the same principle to GRAM native lexical identifiers would be an
implementation transfer, not a sufficiently distinct mechanism contribution.

## Fixed decision

**`STOP_N0_IG_WEIGHTING_PRIOR_ART_OVERLAP`**

No premise audit, training, validation, test or Sports access is authorized.

