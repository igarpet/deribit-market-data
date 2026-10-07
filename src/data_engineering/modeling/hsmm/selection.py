"""
Choose the number of structural regimes K from the data (framework v2.1, Step 6).

Blocked, expanding-window cross-validation over whole sessions:

    fold i: train on every session before test block i, test on block i

The criterion is the summed one-step predictive log density of y_t on the test
rows (identical rows for every K within a fold):

    L_OOS(K) = sum_{folds} sum_{t in test} log sum_k P(R_t = k | F_{t-1}) p_k(y_t | F_{t-1})

Rule, fixed in advance:
  1. one-standard-error rule: the smallest K whose L_OOS is within one standard
     error of the best K, where the standard error is that of the summed
     per-session score difference to the best K;
  2. support: every state of a full-sample fit at that K must have at least
     MIN_EPISODES posterior-mode episodes. The selected K is the smallest K that
     satisfies both rules; if none does, the best-scoring K with support.

In-sample likelihood and BIC are reported but not used (framework §8).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from data_engineering.modeling.hsmm.design import Design
from data_engineering.modeling.hsmm.engine import HSMMSpec, episodes, fit, predictive_scores


MIN_EPISODES = 20


def blocked_folds(session_dates: list[str], n_folds: int = 3, min_train_share: float = 0.45) -> list[tuple[set, set]]:
    """Expanding-window folds over sorted session dates."""
    d = sorted(session_dates)
    n0 = max(2, int(math.ceil(len(d) * min_train_share)))
    rest = d[n0:]
    if len(rest) < n_folds:
        n_folds = max(1, len(rest))
    blocks = np.array_split(np.array(rest), n_folds)
    folds = []
    for i, blk in enumerate(blocks):
        if len(blk) == 0:
            continue
        train = set(d[:n0]) | set(np.concatenate(blocks[:i]).tolist() if i else [])
        folds.append((train, set(blk.tolist())))
    return folds


def select_K(design: Design, K_values=(1, 2, 3, 4, 5, 6), n_folds: int = 3, spec_kw: dict | None = None,
             seed: int = 0, verbose: bool = True, data_kw: dict | None = None) -> tuple[int, pd.DataFrame, dict]:
    """data_kw: passed to Design.data (xs, xc, with_block, block_weight), so
    factor-augmented and block models are selected with the same rule."""
    spec_kw = dict(spec_kw or {})
    data_kw = dict(data_kw or {})
    fr = design.frame
    dates = sorted(fr["session_date"].unique())
    folds = blocked_folds(dates, n_folds)
    sess_scores = {K: {} for K in K_values}
    fold_rows = []
    for fi, (train, test) in enumerate(folds):
        tr = fr["session_date"].isin(train).to_numpy()
        te = fr["session_date"].isin(test).to_numpy()
        rows_te = design.rows_of(te)
        if len(rows_te) == 0 or len(design.rows_of(tr)) == 0:
            continue
        tr_data, te_data = design.data(rows=tr, **data_kw), design.data(rows=te, **data_kw)
        sess = fr["session_date"].to_numpy()[rows_te]
        for K in K_values:
            t0 = time.perf_counter()
            res = fit(tr_data, HSMMSpec(K=K, seed=seed + fi, **spec_kw))
            s = predictive_scores(res.params, te_data)
            for sd, v in pd.Series(s).groupby(sess).sum().items():
                sess_scores[K][sd] = sess_scores[K].get(sd, 0.0) + float(v)
            fold_rows.append({"fold": fi, "K": K, "train_sessions": len(train), "test_sessions": len(test),
                              "test_rows": len(s), "oos_logscore": float(s.sum()),
                              "train_loglik": res.loglik, "train_bic": res.bic, "seconds": time.perf_counter() - t0})
            if verbose:
                print(f"  {design.asset} fold {fi} K={K}: OOS {s.sum():.1f}  ({time.perf_counter() - t0:.0f}s)")
    folds_df = pd.DataFrame(fold_rows)
    S = pd.DataFrame(sess_scores)                     # sessions x K
    total = S.sum()
    best = int(total.idxmax())
    rows = []
    for K in K_values:
        d = S[K] - S[best]
        se = float(d.std(ddof=1) * math.sqrt(len(d))) if len(d) > 1 else np.nan
        rows.append({"K": K, "oos_logscore": float(total[K]), "diff_vs_best": float(total[K] - total[best]),
                     "se_diff": se, "within_1se": bool(K == best or total[K] >= total[best] - (se if np.isfinite(se) else 0.0))})
    table = pd.DataFrame(rows).set_index("K")

    # support: every state needs MIN_EPISODES posterior-mode episodes in a full-sample fit
    support = {1: {"episodes_per_state": None, "ok": True}}
    full = design.data(**data_kw)
    for K in [k for k in K_values if k >= 2]:
        res = fit(full, HSMMSpec(K=K, seed=seed, **spec_kw))
        ep = pd.DataFrame(episodes(res.gamma, full))
        counts = ep.groupby("state").size().reindex(range(K), fill_value=0).tolist()
        support[K] = {"episodes_per_state": counts, "ok": min(counts) >= MIN_EPISODES}
    admissible = [K for K in sorted(table.index) if table.loc[K, "within_1se"] and support[K]["ok"]]
    if admissible:
        chosen = admissible[0]
    else:
        # nothing within one standard error has support: best-scoring supported K
        supported = [K for K in table.index if support[K]["ok"]]
        chosen = int(table.loc[supported, "oos_logscore"].idxmax())
    table["episodes_per_state"] = [support.get(K, {}).get("episodes_per_state") for K in table.index]
    table["support_ok"] = [support.get(K, {}).get("ok") for K in table.index]
    table["selected"] = table.index == chosen
    return chosen, table, {"folds": folds, "fold_scores": folds_df, "session_scores": S}
