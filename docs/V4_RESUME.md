# V4 — interrupted 2026-08-25, HOW TO RESUME after reboot

State: `results/v4/report.json` holds 36/48 completed rows (the ENTIRE
interval block — d_a_null + all five D-D sweep points). Remaining: the 12
bounds rows (D-E dual, D-B-prime walk). The runner skips completed rows on
relaunch (resume support in commit `ad89978`; determinism makes any redone
row bitwise identical, so nothing can be lost or repeated).

Resume with exactly:

    cd ~/PycharmProjects/bcrl-grace-v2 && mkdir -p results/v4 && \
    nohup systemd-inhibit --what=sleep:idle --why="V4 resume: bounds rows" \
      .venv/bin/python tools/run_v4.py >> results/v4/driver.log 2>&1 &

Then watch for "V4 RUN COMPLETE" in `results/v4/driver.log` (~5–8 h: the 12
bounds rows are walk-based). Verdict assembly after completion: coverage
aggregate vs nominal 90% (running 25/29 at interruption — consistent),
widths/collapse table, D-E instrument-value gap, weak-end procedural shares
(d010 Acrobot s1 already showing the 113% path-chaos signature), binding
aggregate from the persisted diagnostics.
