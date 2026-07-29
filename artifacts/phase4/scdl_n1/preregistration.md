# SCDL-N1 Frozen Preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Registered: 2026-07-28
- Data Scope: catalog metadata and frozen lexical IDs only
- Parent: `artifacts/phase4/scdl_n0/mechanism_brief.md`
- Interaction targets/checkpoint/validation/test/Sports access: forbidden

## Frozen audit

The audit reconstructs a native T5-token TF-IDF matrix from `item_plain_text.txt`
(`max_tokens=128`, smooth IDF, L2-normalized item rows). For every competitive
Trie parent with at least two nonempty child subtrees:

- current margin is the selected child token's subtree-centroid weight minus its
  largest sibling-centroid weight;
- each child contributes its top 32 representative native tokens;
- a Hungarian one-to-one assignment maximizes `own weight + sibling margin`;
- assignment outside a child's own top-32 is forbidden;
- no hierarchy, item membership, text field, tokenizer, or identifier length changes.

Both domains must pass every support, deficit, improvement, and semantic-retention
gate in the locked JSON configuration. A valid failure closes SCDL. Only a proven
integrity failure allows an exact rerun.
