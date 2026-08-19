"""Sharp-edge audit of the vendored nbn/ snapshot. Run at every sync.

Two-tier argument, matching how the snapshot actually changes:

1. BYTE-IDENTITY carry-over: subtrees untouched since the last audited tag
   keep their audit rows without re-measurement (a byte-identical file cannot
   have changed behaviour). The script states which subtrees those are for the
   current sync; verify with ``git diff --stat <prev>..<cur> -- nbn/`` upstream.
2. EMPIRICAL re-checks for everything the sync touched, run against the
   VENDORED copy (this repo's import path, this repo's torch pin), so the
   audit certifies the code actually shipped here.

At v0.14.0 -> v0.15.0 the diff is learning/ + mechanisms/ + a fit-threading
kwarg in core/network.py; inference/, sampling/, update/ and core/dag.py are
byte-identical, so the engine rows (VE do-operator, batched do, cache keying)
carry over.
"""

from __future__ import annotations

import copy
import sys

import torch


def check(name, fn):
    try:
        detail = fn()
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
        return True
    except Exception as e:  # noqa: BLE001 -- an audit reports, never crashes
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        return False


def main() -> int:
    from nbn import ContinuousVariable, NeuralBayesianNetwork
    from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism
    from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism
    from nbn.mechanisms.parametric.mdn import MDNMechanism

    torch.manual_seed(0)
    results = []

    def _chain_net(n=4096, noise=0.1):
        b = torch.randn(n, 1)
        data = {"B": b, "R": b + noise * torch.randn(n, 1)}
        net = NeuralBayesianNetwork(
            [("B", "R")],
            {"B": ContinuousVariable("B"), "R": ContinuousVariable("R")},
            device="cpu",
        )
        net.set_mechanism("B", LinearGaussianMechanism())
        net.set_mechanism("R", LinearGaussianMechanism())
        net.fit(data, consolidate=False)
        return net

    # --- differentiable interventional path (network.py was touched) --------
    def sample_do_grad():
        net = _chain_net()
        v = torch.tensor([[2.0]], requires_grad=True)
        net.sample(512, do={"B": v})["R"].mean().backward()
        g = float(v.grad)
        assert abs(g - 1.0) < 0.1, f"grad {g} vs analytic 1.0"
        return f"grad {g:.4f} vs analytic 1.0"

    results.append(
        check("sample(do=) differentiable through caller tensor", sample_do_grad)
    )

    # --- batched do through the engines (row exercises network.query_batch) --
    def query_batch_do():
        net = _chain_net()
        vals = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        w, s = net.query_batch(["R"], {}, do={"B": vals})
        est = (w.unsqueeze(-1) * s).sum(dim=1) / w.sum(dim=1, keepdim=True)
        err = float((est.squeeze() - torch.tensor([0.0, 1.0, 2.0, 3.0])).abs().max())
        assert err < 0.15, f"max err {err}"
        return f"E[R|do(B)] max err {err:.3f} vs truth [0,1,2,3]"

    results.append(check("query_batch(do=) matches truth", query_batch_do))

    # --- save/load round-trip (network.py was touched) ------------------------
    def save_load():
        import tempfile

        net = _chain_net(n=2048)
        before = float(
            net.sample(4096, do={"B": torch.tensor([[1.0]])})["R"].mean().detach()
        )
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            net.save(f.name)
            net2 = NeuralBayesianNetwork.load(f.name)
        after = float(
            net2.sample(4096, do={"B": torch.tensor([[1.0]])})["R"].mean().detach()
        )
        assert abs(before - after) < 0.15, f"{before} vs {after}"
        return f"E[R|do(B=1)] {before:.3f} vs reloaded {after:.3f}"

    results.append(check("save/load round-trips fitted mechanisms", save_load))

    # --- constraint rows -------------------------------------------------------
    def update_local_refuses_weights():
        m = LinearGaussianMechanism()
        x, pa = torch.randn(64, 1), torch.randn(64, 1)
        m.fit_local(x, pa, consolidate=False)
        try:
            m.update_local(x, pa, weights=torch.ones(64))
        except (NotImplementedError, TypeError) as e:
            return f"refused: {type(e).__name__}"
        raise AssertionError("update_local accepted weights")

    results.append(
        check("update_local refuses weights (N2 stands)", update_local_refuses_weights)
    )

    def is_fitted_after_fit():
        for cls, kw in (
            (LinearGaussianMechanism, {}),
            (MDNMechanism, {"num_components": 2}),
        ):
            m = cls(**kw)
            assert not m.is_fitted
            m.fit_local(
                torch.randn(64, 1),
                torch.randn(64, 1),
                consolidate=False,
                **({"epochs": 2} if cls is MDNMechanism else {}),
            )
            assert m.is_fitted, cls.__name__
        return "LG + MDN"

    results.append(check("is_fitted true after fit", is_fitted_after_fit))

    # --- NEW at v0.15.0: the warm-start contract -------------------------------
    def warm_epochs0_bitwise():
        m = MDNMechanism(num_components=2)
        x, pa = torch.randn(200, 1), torch.randn(200, 2)
        m.fit_local(x, pa, epochs=3, consolidate=False)
        before = copy.deepcopy(m.state_dict())  # state_dict ALIASES
        info = m.fit_local(x, pa, epochs=0, consolidate=False, warm_start=True)
        after = m.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)
        assert info["warm_started"] is True
        return "theta bitwise unchanged, warm_started=True"

    results.append(
        check("warm_start=True, epochs=0 is a bitwise no-op", warm_epochs0_bitwise)
    )

    def warm_shape_mismatch_raises():
        m = MDNMechanism(num_components=2)
        m.fit_local(
            torch.randn(200, 1), torch.randn(200, 2), epochs=2, consolidate=False
        )
        try:
            m.fit_local(
                torch.randn(200, 1),
                torch.randn(200, 3),
                epochs=0,
                consolidate=False,
                warm_start=True,
            )
        except Exception as e:  # the contract: raise, NEVER silently rebuild
            return f"raised {type(e).__name__}"
        raise AssertionError("shape mismatch silently accepted")

    results.append(
        check("warm_start shape mismatch raises", warm_shape_mismatch_raises)
    )

    def warm_never_fitted_cold_builds():
        m = MDNMechanism(num_components=2)
        info = m.fit_local(
            torch.randn(200, 1),
            torch.randn(200, 2),
            epochs=2,
            consolidate=False,
            warm_start=True,
        )
        assert m.is_fitted and info["warm_started"] is False
        return "cold-built, warm_started=False"

    results.append(
        check(
            "never-fitted + warm_start=True cold-builds, observable",
            warm_never_fitted_cold_builds,
        )
    )

    def warm_noop_on_closed_form():
        m = LinearGaussianMechanism()
        assert m.warm_start_is_noop is True
        x, pa = torch.randn(64, 1), torch.randn(64, 1)
        m.fit_local(x, pa, consolidate=False)
        info = m.fit_local(x, pa, consolidate=False, warm_start=True)
        assert info["warm_started"] is False
        return "accepted + reported warm_started=False"

    results.append(
        check("closed-form branch: warm_start accepted no-op", warm_noop_on_closed_form)
    )

    # --- NEW at v0.15.0: FlexCode root branch honours weights ------------------
    def flexcode_root_weights():
        y = torch.cat([torch.zeros(400, 1), torch.full((400, 1), 10.0)])
        w = torch.cat([torch.ones(400), torch.zeros(400)])
        m = FlexCodeMechanism(n_basis=7, hidden=(8,), epochs=3)
        m.fit_local(y, None, weights=w, consolidate=False)
        lp0 = float(m.log_prob(torch.zeros(1, 1), None).detach())
        lp10 = float(m.log_prob(torch.full((1, 1), 10.0), None).detach())
        assert lp0 > lp10 + 2.0, f"log_prob(0)={lp0:.2f} vs log_prob(10)={lp10:.2f}"
        return f"log_prob(0)={lp0:.2f} >> log_prob(10)={lp10:.2f} under zeroed weights"

    results.append(
        check("FlexCode ROOT branch honours per-sample weights", flexcode_root_weights)
    )

    n_fail = results.count(False)
    print(
        f"\n  {len(results) - n_fail}/{len(results)} checks pass"
        + ("" if not n_fail else "  <-- AUDIT FAILED")
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
