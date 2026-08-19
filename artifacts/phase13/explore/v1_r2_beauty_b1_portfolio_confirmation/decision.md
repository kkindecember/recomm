# v1-R² Beauty B1 Decision

- Experiment: `GRAM_PHASE13_V1_R2_BEAUTY_B1_PORTFOLIO_CONFIRMATION`
- Scientific status: `completed`
- Frozen verdict: **PASS**
- Transition: **`PASS_TO_PUBLICATION_PREPARATION`**
- Publication novelty checkpoint: **`METHOD_NOVELTY_NOT_YET_CLEARED`**
- Follow-up selection (2026-08-19): **`METHOD_ROUTE_SELECTED_R2_V2_NOT_STARTED`**
- Primary candidate: `unconditional_portfolio2`
- Main evidence:
  - overall NDCG@10: `0.03893624 → 0.04055018` (`+4.15%`), paired 95% CI of delta `[+0.00085109, +0.00239939]`
  - cold H@50: `0.01305088 → 0.03253263` (`2.49×`, 69→172 events), paired 95% CI of delta `[+0.01569888, +0.02326934]`
  - warm NDCG@10 retention: `94.74%` (`−5.26%`), reported as a tradeoff rather than a Gate
- Event-density guard: not triggered (`69 >= 30`)
- Test access: Beauty/Toys test predictions were not opened by this experiment
- Retuning: forbidden on Beauty validation

## Claim boundary

- Supported: the unconditional portfolio gain over v0 transfers from Toys validation to Beauty validation.
- Supported on Toys only: unconditional portfolio outperforms the explored P1–P7 learned gating variants.
- Not supported by B1: a direct Beauty comparison against domain-local P6, because `p6_comparison_included=false`.
- Not supported: a cross-domain guarantee of warm retention `>=95%`; Beauty achieved `94.74%`.

## Next action

1. Close the exploratory record and freeze `portfolio@2` as the conservative primary point; retain `portfolio@3` as the aggressive Pareto point.
2. Do not start P8 or retune any portfolio/gating parameter on Toys or Beauty.
3. Rewrite the publication plan around cold reachability, exact resolver ceiling, and the cold–warm Pareto frontier.
4. Resolve the method-novelty question before starting a full publication matrix.

Follow-up: the user selected the method-paper route on 2026-08-19. A new, single trainable `R²-v2` CBSA has been preregistered in the Phase 13 exploratory plan. Toys/Beauty are development domains for that new method, Sports remains the untouched confirmation domain, and no R²-v2 experiment was started by this documentation update. This does not alter the frozen B1 verdict above.

The method is not training-free: Toys and Beauty each trained a separate warm-only residual user projector for 12 epochs. The novelty risk is that the projector is architecturally simple and the final passing integration component is a fixed portfolio rule.

## Evidence

- `summary.json`
- `status.json`
- `run.log`
- `gpu_telemetry.csv`
- `../v1_r2_beauty_p0/{summary.json,config.json,resolver.pt,predictions_validation.jsonl}`
- `report/第十三阶段/GRAM_第十三阶段_v1-R2_Beauty-B1_无条件portfolio跨域确认报告.md`
