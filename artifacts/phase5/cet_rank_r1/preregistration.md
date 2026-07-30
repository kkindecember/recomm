## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PLANNED
- Version Label: cet_rank_r1_preregistration_v1

## CET Rank-R1 Preregistration

- **Experiment ID**: `GRAM_PHASE5_CET_RANK_R1`
- **Precondition**: `CET_R0G_DIRECT_RANK_GRADIENT_DISTINCT`
- **Purpose**: Test whether direct sequence-rank JS can be optimized on fresh,
  fit/evaluation-disjoint training-prefix users without harming clean lexical CE.
- **Physical GPU**: 6
- **GPU Run Status**: Not started; awaiting explicit confirmation.

### Frozen Design

- Toys/Beauty: 64 fit + 64 evaluation users per domain.
- Optimizer: AdamW, learning rate `1e-4`, batch size 2, 32 steps.
- Trainable parameters: `decoder.block[-1]` only.
- Objective: `CE_clean + CE_perturbed + gamma * rank-JS`.
- Gamma: `0.1 × median(||g_CE|| / ||g_rank-JS||)` over actually masked fit users.
- Candidate support: detached clean/perturbed top-4 union, at most 8 candidates.
- Evaluation gates, required in both domains:
  - masked-user rank-JS relative decrease `>=10%`;
  - clean lexical CE relative increase `<=1%`;
  - masked-user mean top-10 overlap absolute change `>0`;
  - at least 24 masked fit and 24 masked evaluation users;
  - all integrity checks pass.

### Frozen Hashes

- Code SHA256:
  `7ea46d35eeb36e51310672fc0d759d804523379fbebcbc7bc5d39cf66b3d4f95`
- Config SHA256:
  `b48a5ee3324941531d1ae106b5273ffa41e85d865f797cc9846aced4771f29c9`
- Frozen split manifest SHA256:
  `aece5f0e3ed81491d46c760044337eb2382a5f14ca4d60daf4a7ba1525c3b128`
- Toys C1 checkpoint SHA256:
  `5661d9fbdeb25f49d0b36a18b7df6d9ffcc72e18a34736d5c41e0c7f6deb68ff`
- Beauty C1 checkpoint SHA256:
  `75dba35dd176f9a4ad4fd602f70629baed65844e28e9227546aa556d7f4e9db3`

### Frozen User Sets

| Domain | Subset | Users | User-set SHA256 | Prior overlap |
|---|---:|---:|---|---:|
| Toys | fit | 64 | `22f32c4334f0f5cf7696cc843398a67545ae2657e46565ae83e383515924b8b7` | 0 |
| Toys | evaluation | 64 | `99275ea984fee042312246f138272ec0ff7cbf84c5629dd5dabe72e55f425e46` | 0 |
| Beauty | fit | 64 | `20fefc661957eee62dd7146c1549b2d02f54376d8287d7b57b4c0c683a01dc63` | 0 |
| Beauty | evaluation | 64 | `634376b136cde1d5d493e4c5b522965a3b7049acf832b5260fddf101cf1cf93b` | 0 |

### Unique Start Command

```bash
bash experiment/phase5/run_phase5_cet_rank_r1.sh start
```

No automatic retry is permitted. Rank-R2 remains unauthorized unless the frozen
Rank-R1 machine decision is `CET_R1_RANK_CONSISTENCY_PASS`.
