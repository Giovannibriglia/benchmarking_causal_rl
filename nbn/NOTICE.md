# NOTICE — vendored NeuralBayesianNetworks (NBN)

This directory is a vendored snapshot of the NeuralBayesianNetworks library.

- **Source:** https://github.com/Giovannibriglia/NeuralBayesianNetworks
- **Version:** `0.7.dev37+ga5f97d5d7.d20260522` (see `_version.py`; upstream commit `a5f97d5d7`, snapshot dated 2026-05-22)
- **Vendored into this repository:** 2026-08 (committed 2026-08-12 on `feat/grace-critic`)
- **Author:** Giovanni Briglia (author of both this repository and the upstream library)
- **License:** GPLv3 — see `nbn/LICENSE` (same license as the host repository)

Local conventions:

- The snapshot is **pinned**: it is excluded from this repository's pre-commit
  formatters/linters (`exclude: ^nbn/` in `.pre-commit-config.yaml`) so the tree
  stays byte-diffable against upstream. Do not reformat or "fix" files here.
- **Deferred-sync TODO:** upstream has moved past this snapshot (v0.14 line).
  Syncing was deliberately deferred so GRACE ships against the audited API
  surface; see `docs/grace.md` for the re-audit checklist to run when syncing.
- Known sharp edges of this snapshot (do-operator vs exact VE, `load()`,
  VE cache keyed on `id(model)`, …) are catalogued in `docs/grace.md`; GRACE
  code works around them and does not patch this directory.
