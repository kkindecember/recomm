# CPIA-N0 Mechanism Brief

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: deep-research → plan
- Created: 2026-07-29
- Validation/test/Sports: forbidden
- Parent evidence: GRAM information-linking claim and independent alignment literature

## Research question

GRAM places every history item's native lexical ID in both the coarse user
prompt and its corresponding fine item prompt. The paper calls that repeated ID
an information-linking bridge, while training uses only next-ID
sequence-to-sequence CE. Do the two contextual occurrences actually identify
each other inside the frozen encoder?

## Mechanism

**Cross-Passage Identifier Alignment (CPIA)** would add a training-only
within-example contrastive loss:

```text
c_i = pooled encoder states of item i's lexical-ID span in the coarse prompt
f_i = pooled encoder states of the same span in item i's fine prompt

L_cpia = -log exp(sim(c_i, f_i) / tau)
              / sum_j exp(sim(c_i, f_j) / tau)
```

Negatives are other history items from the same user example. The ordinary
lexical recommendation CE, IDs, prompt text, FiD layout, decoder, and
Trie-constrained inference remain unchanged. The method is deterministic and
non-adaptive at inference because the auxiliary loss is removed after training.

## Independent basis and novelty boundary

- GRAM explicitly states that the repeated item ID bridges coarse and fine
  prompts, but its stated training objective is only teacher-forced
  sequence-to-sequence CE:
  https://aclanthology.org/2025.acl-long.1596/
- RA-Rec aligns pretrained collaborative ID embeddings with an LLM space:
  https://arxiv.org/abs/2402.04527
- GENPLUGIN aligns separate language and ID views during pretraining:
  https://arxiv.org/abs/2507.03568
- Tree-structured identifier work applies contrastive objectives to identifier
  construction/structure:
  https://arxiv.org/abs/2309.13375

CPIA does not claim contrastive alignment itself is new. Its bounded difference
is an intra-example, same-native-ID, coarse-to-fine contextual alignment loss
for GRAM's FiD passages. It does not introduce external ID embeddings, optimize
the identifier tree, align metadata to a separate SID encoder, or add an
inference-time scorer.

## Separation from prior phase-4 directions

- FPUG estimates whether a fine passage is harmful; CPIA asks whether the
  intended coarse/fine identity link is represented at all.
- TCDR concerns decoder score covariance between catalog paths; CPIA concerns
  encoder states of the same item across two input passages.
- LNDR concerns prefix-specific output-token readout; CPIA changes no decoder
  readout.
- GCDH/GACR add catalog-ranking capacity; CPIA retains the original generator
  and only supplies an auxiliary representation constraint.

## Falsifiable N1 premise

Using only frozen Toys and Beauty training-prefix examples, exact lexical-ID
span mapping must be complete. Within each user, coarse-ID states must carry
some same-item signal above deterministic chance, but must still fail to
reliably retrieve their corresponding fine passage. If either domain already
has a strong bridge, has no measurable bridge signal, or fails exact span
mapping, CPIA stops before implementation.
