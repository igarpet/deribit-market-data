"""
Synthetic recovery test for the HSMM engine.

    python src/data_engineering/modeling/hsmm/selftest.py

1. Recovery at realistic strength: data simulated from a known 2-state HSMM
   (opposite displacement slopes, scales differing by 35%, shared lag terms,
   Student-t noise, grid gaps): slopes, scales, shared terms and the state
   path must be recovered. Mean durations are reported, not checked: with
   weakly separated states, misclassified bars split long episodes, so
   durations are biased short (a property of the data, not the engine).
1b. Recovery at strong separation: mean durations must be recovered.
2. No regimes: data with one regime must not produce two states with
   clearly different slopes.
3. Prediction: out-of-sample predictive log score of the 2-state model must
   beat the 1-state model on regime data.
Exits with a non-zero status if any check fails.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_engineering.modeling.hsmm.engine import (  # noqa: E402
    HSMMData, HSMMParams, HSMMSpec, duration_pmf, fit, predictive_scores, simulate_states,
)

D_MAX = 144
TRUE = dict(beta_D=np.array([-0.30, 0.25]), mu=np.array([0.0, 0.05]), sigma=np.array([0.85, 1.15]),
            nu=6.0, shared=np.array([-0.04, -0.02, -0.01]), dur=[(3.0, 24.0), (2.0, 40.0)])


def simulate(T: int, rng, regimes: bool = True, gap_every: int = 1500):
    pmf = np.vstack([duration_pmf("negbin", D_MAX, r, r / (r + m - 1)) for r, m in TRUE["dur"]])
    params = HSMMParams(pi=np.array([0.5, 0.5]), A=np.array([[0.0, 1.0], [1.0, 0.0]]), pmf=pmf,
                        dur=[("negbin",) + d for d in TRUE["dur"]], beta_s=np.zeros((2, 2)),
                        beta_c=TRUE["shared"], sigma=TRUE["sigma"], nu=TRUE["nu"])
    lab = simulate_states(params, T, rng) if regimes else np.zeros(T, int)
    phi = 0.95
    Dx = np.zeros(T)
    y = np.zeros(T)
    for t in range(1, T):
        Dx[t] = phi * Dx[t - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal()
    eps = rng.standard_t(TRUE["nu"], T) * np.sqrt((TRUE["nu"] - 2) / TRUE["nu"])
    for t in range(3, T):
        k = lab[t]
        beta_D = TRUE["beta_D"][k] if regimes else 0.0
        sig = TRUE["sigma"][k] if regimes else 1.0
        mu = TRUE["mu"][k] if regimes else 0.0
        y[t] = mu + beta_D * Dx[t - 1] + TRUE["shared"] @ y[t - 3:t][::-1] + sig * eps[t]
    t_idx = np.arange(3, T)
    Xs = np.column_stack([np.ones(len(t_idx)), Dx[t_idx - 1]])
    Xc = np.column_stack([y[t_idx - 1], y[t_idx - 2], y[t_idx - 3]])
    starts = np.arange(0, len(t_idx), gap_every)
    ends = np.r_[starts[1:], len(t_idx)]
    data = HSMMData(y[t_idx], Xs, Xc, starts, ends, ["const", "D"], ["z1", "z2", "z3"])
    return data, lab[t_idx]


def split(data: HSMMData, frac: float = 0.5):
    cut = int(data.T * frac)
    def part(a, b):
        st = [max(s, a) - a for s, e in zip(data.starts, data.ends) if e > a and s < b]
        en = [min(e, b) - a for s, e in zip(data.starts, data.ends) if e > a and s < b]
        return HSMMData(data.y[a:b], data.Xs[a:b], data.Xc[a:b], np.array(st), np.array(en),
                        data.names_s, data.names_c)
    return part(0, cut), part(cut, data.T)


def main() -> int:
    rng = np.random.default_rng(11)
    failures = []

    data, lab = simulate(6000, rng)
    t0 = time.perf_counter()
    res = fit(data, HSMMSpec(K=2, d_max=D_MAX, n_init=3, seed=1))
    secs = time.perf_counter() - t0
    p = res.params
    acc = float((res.gamma.argmax(1) == lab).mean())
    md = p.mean_duration()
    true_md = np.array([m for _, m in TRUE["dur"]])
    print(f"[1] recovery: {res.iterations} EM iterations, {secs:.1f}s, converged={res.converged}")
    print(f"    beta_D  {np.round(p.beta_s[:, 1], 3)}  true {TRUE['beta_D']}")
    print(f"    sigma   {np.round(p.sigma, 3)}  true {TRUE['sigma']}")
    print(f"    shared  {np.round(p.beta_c, 3)}  true {TRUE['shared']}")
    print(f"    nu {p.nu}  true {TRUE['nu']} | mean duration {np.round(md, 1)}  true {true_md}")
    print(f"    state accuracy {acc:.3f}")
    if np.max(np.abs(p.beta_s[:, 1] - TRUE["beta_D"])) > 0.08:
        failures.append("slopes not recovered")
    # simulated noise has unit variance, so the true t-scale is sigma * sqrt((nu - 2) / nu)
    true_scale = TRUE["sigma"] * np.sqrt((TRUE["nu"] - 2) / TRUE["nu"])
    if np.max(np.abs(p.sigma / true_scale - 1)) > 0.10:
        failures.append("scales not recovered")
    if np.max(np.abs(p.beta_c - TRUE["shared"])) > 0.04:
        failures.append("shared lag terms not recovered")
    if acc < 0.75:
        failures.append("state path not recovered")
    print(f"    (durations at realistic strength are biased short: {np.round(md / true_md, 2)} of true)")

    saved = {k: TRUE[k].copy() for k in ("sigma", "beta_D")}
    TRUE["sigma"], TRUE["beta_D"] = np.array([0.5, 2.0]), np.array([-0.6, 0.6])
    strong, lab_s = simulate(8000, rng, gap_every=10 ** 9)
    TRUE.update(saved)
    rs = fit(strong, HSMMSpec(K=2, d_max=D_MAX, n_init=2, seed=4))
    runs = []
    t = 0
    while t < len(lab_s):
        e = t
        while e < len(lab_s) and lab_s[e] == lab_s[t]:
            e += 1
        runs.append((lab_s[t], e - t))
        t = e
    true_runs = np.array([np.mean([L for k_, L in runs if k_ == k]) for k in (0, 1)])
    md_s = rs.params.mean_duration()
    acc_s = float((rs.gamma.argmax(1) == lab_s).mean())
    print(f"[1b] strong separation: mean duration {np.round(md_s, 1)} vs realised {np.round(true_runs, 1)}, accuracy {acc_s:.3f}")
    if np.max(np.abs(md_s / true_runs - 1)) > 0.15 or acc_s < 0.95:
        failures.append("durations not recovered at strong separation")

    train, test = split(data)
    r1 = fit(train, HSMMSpec(K=1))
    r2 = fit(train, HSMMSpec(K=2, d_max=D_MAX, n_init=3, seed=2))
    g = predictive_scores(r2.params, test).mean() - predictive_scores(r1.params, test).mean()
    print(f"[3] out-of-sample log score gain K=2 vs K=1: {g:.4f} per bar")
    if g <= 0:
        failures.append("no out-of-sample gain on regime data")

    null, _ = simulate(6000, rng, regimes=False)
    rn = fit(null, HSMMSpec(K=2, d_max=D_MAX, n_init=3, seed=3))
    r1n = fit(null, HSMMSpec(K=1))
    spread = float(np.ptp(rn.params.beta_s[:, 1]))
    lr = 2 * (rn.loglik - r1n.loglik)
    print(f"[2] no regimes: slope spread {spread:.3f} (regime data: {np.ptp(p.beta_s[:, 1]):.3f}), "
          f"LR vs K=1 {lr:.1f} (regime data: {2 * (res.loglik - fit(data, HSMMSpec(K=1)).loglik):.1f})")
    if spread > 0.25:
        failures.append("spurious slope separation without regimes")

    if failures:
        print("FAILED:", "; ".join(failures))
        return 1
    print("HSMM SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
