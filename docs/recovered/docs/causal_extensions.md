# Causal Extensions

This document defines the eight-cell mapping used by the causal benchmark integration.

| Cell | Z Exposed | U Exposed | `pi_b` Known | Online/Offline |
| ---- | --------- | --------- |--------------| -------------- |
| 1    | Yes       | Yes       | -            | Online         |
| 2    | No        | No        | -            | Online         |
| 3    | Yes       | Yes       | Yes          | Offline        |
| 4    | Yes       | No        | Yes          | Offline        |
| 5    | Yes       | Yes       | No           | Offline        |
| 6    | Yes       | No        | No           | Offline        |
| 7    | Yes       | No        | Yes          | Offline        |
| 8    | No        | No        | No           | Offline        |

Notes:

- `Z` is latent context.
- `U` is hidden confounder.
- `pi_b` is the behavior policy used for data collection.
- Replay buffers store optional `behavior_logprob` and optional `latent` according to this table.

Recommended env-sets:

- `causal_8cells` (sepsis family)
- `causal_blockmdp_8cells` (synthetic block-MDP family)

Important:

- `delta_*` divergences are computed only for `CausalEnv` wrappers.
- Standard Gymnasium envs keep `delta_*` columns empty.
