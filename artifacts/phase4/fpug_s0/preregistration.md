# FPUG-S0 Frozen Correctness Preregistration

## Scope

- Toys and Beauty training prefixes only; validation/test/Sports forbidden.
- Eight unique users/domain, four head and four tail, history length at least five.
- Frozen GCDH-P0 C0 backbone; only the new passage gate is optimized.

## Frozen gate

For pooled coarse state `q`, pooled detailed passage state `p_i`, and normalized
recency `r_i`:

```text
g_i = 1 + 0.5 * tanh(Linear([q, p_i, q*p_i, r_i]))
```

The linear layer is initialized to all zeros, so every `g_i=1` exactly. The coarse
passage is never multiplied by a gate. Detailed encoder states are multiplied by
`g_i` before the unchanged decoder.

## Correctness conjunction

Both domains must pass:

- zero-init logits exactly match the frozen baseline within `1e-6`;
- coarse encoder states remain exactly unchanged;
- all detail gates remain in `[0.5, 1.5]`;
- gate gradient is finite and nonzero while all backbone gradients remain absent;
- 20 gate-only steps reduce same-batch lexical CE by at least 1%;
- trained gates are non-identity;
- save/reload reproduces logits within `1e-6`;
- mapping, finite, head/tail presence and data-firewall checks pass.

Pass: `FPUG_S0_CORRECTNESS_PASS`. Failure: `STOP_FPUG_S0_CORRECTNESS_FAILED`.
