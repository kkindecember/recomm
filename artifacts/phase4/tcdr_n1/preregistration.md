# TCDR-N1 Frozen Preregistration

## Scope

Frozen-checkpoint, training-prefix-only premise audit. No optimizer, validation,
test, or Sports access.

## Cohorts

- Per domain, select 128 train users with at least five prefix-history items by
  SHA-256 of `2023|dataset|tcdr-n1-users-v1|user_id`.
- Build item incidence only from those users' permitted training portions
  (`sequence[:-2]`), supplemented by all predefined train users for stable item
  frequency/incidence estimates.
- Eligible items require at least three train-user incidences.
- Select 64 tree-close item pairs by deterministic hash. A close pair shares at
  least two lexical tokens and has collaborative cosine `<= 0.05`.
- For every close pair, select one tree-far control with zero common lexical
  prefix, the same log2-frequency bins for both endpoints, and collaborative
  cosine `<= 0.05`.

## Frozen measurements

For each user, encode the ordinary GRAM training-prefix context once and compute
length-normalized exact lexical-path log scores for every selected item. For
each pair compute Pearson correlation across the 128-user score vectors.

Primary paired quantities:

- close-pair score correlation;
- matched far-pair score correlation;
- correlation excess = close − far;
- positive excess rate.

## Scientific conjunction

Both Toys and Beauty must satisfy:

- 64 eligible close/control pair matches;
- median close-pair correlation `>= 0.50`;
- median paired correlation excess `>= 0.10`;
- positive excess rate `>= 0.65`;
- bootstrap 95% lower bound of mean paired excess `> 0`.

All must pass for `TCDR_S0_DESIGN_ALLOWED`; otherwise
`STOP_TCDR_NO_TREE_COUPLING_DEFICIT`.

## Integrity

Mapping, Trie membership and finite rates must equal 1.0; selected users are
unique; near/control endpoint frequency bins match; optimizer steps are zero;
checkpoint SHA is unchanged; validation/test/Sports are unread. Any failure is
`EXECUTION_INVALID`.

