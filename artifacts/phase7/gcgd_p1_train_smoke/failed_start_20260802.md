# GCGD P1 train-only smoke failed start record

- Started: `2026-08-02T14:20:24+08:00`
- Terminal status: failed in `Toys`, before A/B/C beam generation
- Failure: an all-zero prefix mass caused `math.log(0)` after masked-item scores underflowed during exponentiation
- Data integrity: train-only; fresh validation, test predictions, and Sports were not read
- Retry policy: no automatic retry occurred
- Resource restoration: CodeLlama was restored to physical GPU 0 after the failed exit
- Repair: omit zero-mass children, let the graph branch abstain on wholly zero-mass prefixes, and mask only the sample-time visible history
- Scientific configuration: unchanged
- Restart authority: the researcher explicitly requested a manual repair and restart
