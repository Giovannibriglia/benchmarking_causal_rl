# Recovered prior causal implementation — ARCHIVE ONLY

These files are the earlier, uncommitted causal-cells implementation of this
repo (sepsis/BlockMDP-based eight-cell experiments, `confounded_dqn`,
`causal_metrics` estimators/gap, `offline_collector`). They were moved to the
desktop Trash on **2026-06-05 at 15:27–15:30** (file-manager deletion; the
whole `runs/` directory went at 15:30:29 and
`runs/causal_8cells_20260429_092500` was restored back in place from Trash).
Everything here was copied verbatim from
`~/.local/share/Trash/files/` on 2026-06-05; the Trash itself was left intact.

**Do not import or integrate this code.** It references modules and registry
entries that no longer exist (`a2c_cc`, `vanilla_cc`, `causal_8cells` env set)
and predates the Phase-0 golden/invariant contract. It is archived purely as
design reference for Phases 3–4 of the causal-RL refactor (offline collection,
confounded behavior policies, gap estimators).

Directory layout mirrors the original repo-relative paths
(`src/...`, `tests/...`, `reproducibility/...`, `docs/...`).

Additional fallback: PyCharm Local History
(`~/.cache/JetBrains/PyCharm2026.1/LocalHistory/`, last written 2026-06-05
15:55) contains revisions of these files; recover via the IDE
(*right-click → Local History → Show History*) if a version older than the
trashed one is ever needed. The binary store is not parsed here.
