# Offline-budget probe v2 — CartPole-v1 sigma=0, base=cql
MC behaviour-anchor (per RE): RE3000=20.12.  eval-derived improved scale ~= 5.0x anchor.
ratio=apparent_Q/anchor (1=behaviour scale=under-trained; ~10-12=improved scale; >>that=diverged)

| target | steps | RE | anchor | obs_relerr | prox_relerr | obs_ratio | prox_ratio | oracle_ratio | eval_ret | n |
|--|--|--|--|--|--|--|--|--|--|--|
| 5k | 5000 | 3000 | 20.12 | 0.286 | 0.028 | 1.45 | 1.14 | 1.15 | 984.0 | 3 |
| 20k | 19000 | 3000 | 20.12 | 0.384 | 0.048 | 3.58 | 2.43 | 2.41 | 990.9 | 3 |
| 50k | 50000 | 3000 | 20.12 | 0.42 | 0.139 | 5.66 | 4.0 | 3.58 | 987.0 | 3 |
| 100k | 50000 | 3000 | 20.12 | 0.42 | 0.139 | 5.66 | 4.0 | 3.58 | 987.0 | 3 |
| 256k | 50000 | 3000 | 20.12 | 0.42 | 0.139 | 5.66 | 4.0 | 3.58 | 987.0 | 3 |
