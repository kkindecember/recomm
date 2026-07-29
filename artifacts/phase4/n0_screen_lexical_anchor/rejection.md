# N0 Rejection: Lexical Semantic Anchor Preservation

Date: 2026-07-29

Candidate: penalize movement of native lexical-ID token embeddings away from
their pretrained T5 values.

A training-only structural probe compared the reproduced GRAM checkpoint with
the base T5 embedding table. Native lexical-ID tokens had larger relative drift
than a frequency-matched control vocabulary:

- Toys: mean cosine drift `3.733e-5` versus `9.956e-6` (3.75×);
- Beauty: `4.817e-5` versus `1.085e-5` (4.44×).

The ratio is not an actionable premise because the absolute cosine movement is
below `5e-5` in both domains. An anchor loss would therefore target a
numerically negligible displacement, and reporting only the relative ratio
would inflate the evidence.

Fixed decision: **`STOP_N0_LEXICAL_ANCHOR_NO_ABSOLUTE_DRIFT`**. No validation,
test, or Sports data were read.
