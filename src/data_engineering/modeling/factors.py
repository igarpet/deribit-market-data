"""
Factor inputs for the factor-augmented switching regression (framework v3.1,
structural layer step A: static factors, training-only).

Market measurements are grouped into feature FAMILIES. Each feature is
clock-adjusted (modeling.clock_baseline, fitted on the training rows only),
standardized with training moments and winsorized; each family is
compressed to its first principal component on the training rows:

    x~_{j,t} = clip((clock_adj_{j,t} - m_j) / s_j, -5, 5)      (training m_j, s_j)
    f_{g,t}  = sum_{j in g} w_{g,j} x~_{j,t} / sqrt(n_obs_{g,t} / n_g)

w_g is the leading eigenvector of the training correlation matrix of family
g (missing entries set to 0 = the training mean, i.e. "no information"),
signed so that the family's anchor feature has a positive loading. The
partial-row rescaling keeps the factor variance comparable when some of
the family's features are missing. A factor is missing when every feature
of its family is missing.

Two channels (they answer different questions):

  directional families (flow, book, basis, skew): signed pressure that can
      predict the SIGN of the next index return. They enter the price
      equation as state-specific regressors, gamma_k' F_{t-1}.
  state families (activity, volatility, liquidity, option level): symmetric
      measurements that cannot predict the sign but may help identify the
      regime. They enter as the masked emission block, with a tempered
      weight so that they cannot take over the states.

Nothing here uses future information: clock baselines, moments, loadings
and signs come from the training rows; rolling features are trailing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_engineering.modeling.clock_baseline import ClockBaselineSet, FeatureSpec, transformed_values


FACTOR_VERSION = "0.1"
WINSOR = 5.0
BAR = pd.Timedelta(minutes=5)


@dataclass(frozen=True)
class FeatureDef:
    name: str
    column: str
    transform: str = "identity"          # identity | log | log1p
    scope: str | None = None             # clock_baseline scope (perp | fit | option)
    clock: str = "new_york"


@dataclass(frozen=True)
class Family:
    name: str
    channel: str                         # directional | state
    anchor: str                          # feature with a positive loading
    features: tuple


FAMILIES: tuple[Family, ...] = (
    Family("flow", "directional", "tfi", (
        FeatureDef("tfi", "trade_flow_imbalance"),
        FeatureDef("tfi_30m", "tfi_30m"),
    )),
    Family("book", "directional", "obi_20", (
        FeatureDef("obi_5", "book_twa_obi_5", scope="perp"),
        FeatureDef("obi_10", "book_twa_obi_10", scope="perp"),
        FeatureDef("obi_20", "book_twa_obi_20", scope="perp"),
        FeatureDef("obi_20_30m", "obi_20_30m", scope="perp"),
        FeatureDef("microprice_bps", "book_twa_microprice_vs_reference_bps", scope="perp"),
    )),
    Family("basis", "directional", "basis_bps", (
        FeatureDef("basis_bps", "perpetual_mid_vs_index_bps", scope="perp"),
    )),
    Family("skew", "directional", "skew", (
        FeatureDef("skew", "fit_skew_rel", scope="fit", clock="expiry_utc"),
        FeatureDef("skew_change_30m", "fit_skew_change_30m", scope="fit", clock="expiry_utc"),
    )),
    Family("activity", "state", "log_volume", (
        FeatureDef("log_volume", "log1p_volume_usd"),
        FeatureDef("log_trades", "trade_count", transform="log1p"),
        FeatureDef("log_book_updates", "book_update_count", transform="log1p", scope="perp"),
    )),
    Family("volatility", "state", "log_rv_1h", (
        FeatureDef("log_rv_1h", "realized_vol_1h_annualized", transform="log"),
        FeatureDef("log_rv_30m", "realized_vol_30m_annualized", transform="log"),
        FeatureDef("log_range", "intrabar_range_bps", transform="log"),
    )),
    Family("liquidity", "state", "log_spread", (
        FeatureDef("log_spread", "book_twa_spread_bps", transform="log", scope="perp"),
        FeatureDef("log_depth", "book_depth_20_usd", transform="log", scope="perp"),
    )),
    Family("option_level", "state", "log_atm_iv", (
        FeatureDef("log_atm_iv", "atm_iv_quote", transform="log", clock="expiry_utc"),
        FeatureDef("iv_minus_rv", "iv_minus_rv", clock="expiry_utc"),
        FeatureDef("curvature", "fit_curvature_rel", scope="fit", clock="expiry_utc"),
    )),
)

DIRECTIONAL = tuple(f.name for f in FAMILIES if f.channel == "directional")
STATE = tuple(f.name for f in FAMILIES if f.channel == "state")


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.map({"True": True, "False": False, "true": True, "false": False, True: True, False: False})
    return s.fillna(False).astype(bool)


def _trailing_mean(x: pd.Series, times: pd.Series, window: int, min_obs: int) -> pd.Series:
    """Mean of the last `window` bars on a contiguous grid (gaps restart)."""
    run = (times.diff() != BAR).cumsum()
    return x.groupby(run.to_numpy()).transform(lambda v: v.rolling(window, min_periods=min_obs).mean())


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Derived raw features (per asset, causal)."""
    out = []
    for _, g in df.groupby("underlying", sort=False):
        g = g.sort_values("state_time").copy()
        t = pd.to_datetime(g["state_time"], utc=True)
        if "trade_flow_imbalance" in g:
            g["tfi_30m"] = _trailing_mean(pd.to_numeric(g["trade_flow_imbalance"], errors="coerce"), t, 6, 4)
        if "book_twa_obi_20" in g:
            g["obi_20_30m"] = _trailing_mean(pd.to_numeric(g["book_twa_obi_20"], errors="coerce"), t, 6, 4)
        if {"book_twa_bid_depth_20", "book_twa_ask_depth_20"} <= set(g.columns):
            g["book_depth_20_usd"] = (pd.to_numeric(g["book_twa_bid_depth_20"], errors="coerce")
                                      + pd.to_numeric(g["book_twa_ask_depth_20"], errors="coerce"))
        if "option_iv_atm_forward" in g:
            iv = pd.to_numeric(g["option_iv_atm_forward"], errors="coerce")
            if "option_atm_quote_ok" in g:
                iv = iv.where(_as_bool(g["option_atm_quote_ok"]))
            if "in_settlement_window" in g:
                iv = iv.where(~_as_bool(g["in_settlement_window"]))
            g["atm_iv_quote"] = iv
            if "realized_vol_1h_annualized" in g:
                rv = pd.to_numeric(g["realized_vol_1h_annualized"], errors="coerce")
                # IV in percent on Deribit, RV annualized as a fraction
                ivf = iv / 100.0 if iv.dropna().median() > 3 else iv
                g["iv_minus_rv"] = np.log(ivf.where(ivf > 0)) - np.log(rv.where(rv > 0))
        out.append(g)
    return pd.concat(out).sort_index()


@dataclass
class FactorSet:
    factors: pd.DataFrame                 # index = input rows; one column per family (NaN = missing)
    loadings: pd.DataFrame                # underlying, family, feature, loading
    summary: pd.DataFrame                 # underlying, family, n_features, explained, coverage, stability
    missing_columns: list = field(default_factory=list)
    clock_skipped: list = field(default_factory=list)

    def directional(self, rows=None) -> pd.DataFrame:
        cols = [f"F_{f}" for f in DIRECTIONAL if f"F_{f}" in self.factors]
        return self.factors.loc[rows if rows is not None else self.factors.index, cols]

    def state(self, rows=None) -> pd.DataFrame:
        cols = [f"F_{f}" for f in STATE if f"F_{f}" in self.factors]
        return self.factors.loc[rows if rows is not None else self.factors.index, cols]


def _first_pc(X: np.ndarray, anchor_idx: int) -> tuple[np.ndarray, float]:
    C = np.corrcoef(X, rowvar=False) if X.shape[1] > 1 else np.ones((1, 1))
    C = np.nan_to_num(C, nan=0.0)
    vals, vecs = np.linalg.eigh(C)
    w = vecs[:, -1]
    if w[anchor_idx] < 0:
        w = -w
    return w, float(vals[-1] / max(vals.sum(), 1e-12))


def build_factors(df: pd.DataFrame, train_mask: pd.Series, families: tuple[Family, ...] = FAMILIES) -> FactorSet:
    """
    df: state rows with state_id, state_time, underlying, is_weekend and the
    raw columns (missing columns are reported and skipped). train_mask: rows
    used for clock baselines, moments and loadings.
    """
    df = add_derived(df)
    train_mask = train_mask.reindex(df.index).fillna(False).astype(bool)

    feats, missing = [], []
    for fam in families:
        for fd in fam.features:
            (feats if fd.column in df else missing).append(fd)
    specs = tuple(FeatureSpec(fd.name, fd.column, transform=fd.transform, clock=fd.clock,
                              scope=fd.scope, scale="constant") for fd in feats)
    base_cols = ["state_id", "state_time", "underlying", "is_weekend", "perp_integrity_ok", "fit_ok",
                 "in_settlement_window"]
    frame = df[[c for c in base_cols if c in df] + sorted({fd.column for fd in feats})].copy()
    if "fit_ok" not in frame:
        specs = tuple(s for s in specs if s.scope != "fit")
    clock = ClockBaselineSet.fit(frame, train_mask, specs=specs)
    adj = clock.transform(frame)
    adj.index = frame.index

    # standardized, winsorized inputs (per asset, training moments)
    Z = pd.DataFrame(index=df.index)
    for fd in feats:
        col = f"{fd.name}__clock_adj"
        if col in adj:
            x = adj[col]
        else:      # no clock fit (too little training data): unadjusted transform
            spec = next(sp for sp in specs if sp.name == fd.name) if any(sp.name == fd.name for sp in specs) else None
            x = transformed_values(frame, spec) if spec is not None else pd.Series(np.nan, index=df.index)
        z = pd.Series(np.nan, index=df.index)
        for asset, idx in df.groupby("underlying").groups.items():
            xa = x.loc[idx]
            tr = xa[train_mask.loc[idx]]
            m, s = tr.mean(), tr.std()
            if not np.isfinite(s) or s <= 0:
                continue
            z.loc[idx] = ((xa - m) / s).clip(-WINSOR, WINSOR)
        Z[fd.name] = z

    fac = pd.DataFrame(index=df.index)
    load_rows, sum_rows = [], []
    for fam in families:
        names = [fd.name for fd in fam.features if fd.name in Z]
        col = f"F_{fam.name}"
        fac[col] = np.nan
        if not names:
            continue
        anchor_idx = names.index(fam.anchor) if fam.anchor in names else 0
        for asset, idx in df.groupby("underlying").groups.items():
            Xa = Z.loc[idx, names]
            obs = Xa.notna()
            if obs.to_numpy().sum() == 0:
                continue
            tr = train_mask.loc[idx].to_numpy()
            keep = [j for j, n in enumerate(names) if obs.loc[tr, n].mean() > 0.2]
            if not keep:
                continue
            kn = [names[j] for j in keep]
            a_idx = kn.index(names[anchor_idx]) if names[anchor_idx] in kn else 0
            Xtr = Xa.loc[tr, kn]
            Xtr = Xtr[Xtr.notna().any(axis=1)].fillna(0.0).to_numpy()
            if len(Xtr) < 50:
                continue
            w, expl = _first_pc(Xtr, a_idx)
            Xall = Xa[kn].fillna(0.0).to_numpy()
            n_obs = Xa[kn].notna().sum(axis=1).to_numpy()
            score = (Xall @ w) / np.sqrt(np.maximum(n_obs, 1) / len(kn))
            score = np.where(n_obs > 0, score, np.nan)
            sd = np.nanstd(score[tr]) if np.isfinite(score[tr]).sum() > 10 else 1.0
            fac.loc[idx, col] = score / (sd if sd > 0 else 1.0)
            # loading stability: first vs second half of the training rows
            tri = np.flatnonzero(tr)
            stab = np.nan
            if len(tri) >= 200 and len(kn) > 1:
                h = len(tri) // 2
                parts = []
                for sel in (tri[:h], tri[h:]):
                    Xh = Xa.iloc[sel][kn]
                    Xh = Xh[Xh.notna().any(axis=1)].fillna(0.0).to_numpy()
                    parts.append(_first_pc(Xh, a_idx)[0] if len(Xh) >= 50 else None)
                if all(p is not None for p in parts):
                    stab = float(abs(parts[0] @ parts[1]))
            for n, lw in zip(kn, w):
                load_rows.append({"underlying": asset, "family": fam.name, "channel": fam.channel,
                                  "feature": n, "loading": float(lw)})
            sum_rows.append({"underlying": asset, "family": fam.name, "channel": fam.channel,
                             "n_features": len(kn), "explained_pc1": expl,
                             "coverage": float(np.isfinite(fac.loc[idx, col]).mean()),
                             "loading_stability": stab})
    return FactorSet(fac, pd.DataFrame(load_rows), pd.DataFrame(sum_rows),
                     [fd.column for fd in missing], list(clock.skipped))
