"""PSH02: preregistered volatility × signed-flow perpetual return experiment.

Historical, exploratory, read-only.  The PSH01 Student-t HMM is re-fit using
its training-only procedure at K=4.  PSH01 itself and frozen models are unchanged.
No orders, no exchange requests, no Deribit execution claim.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from data_engineering.modeling import predictive_state_hmm as psh
from data_engineering.modeling import candle_streak_perp_target as csp

VERSION = "PSH02-0.1.0"
ASSETS = ("BTC", "ETH", "SOL", "XRP")
BASE = ("now", "ema_30m", "ema_240m", "clock_sin", "clock_cos")
CONT = BASE + ("flow_5m",)
INTER = CONT + ("flow_x_slow", "flow_x_slope")
ARMS = ("price_vol", "plus_flow", "continuous_interaction",
        "hmm_additive", "hmm_flow_interaction", "nonlinear_continuous")
COMPS = (("plus_flow", "price_vol"),
         ("continuous_interaction", "plus_flow"),
         ("hmm_flow_interaction", "continuous_interaction"),
         ("hmm_flow_interaction", "hmm_additive"),
         ("hmm_flow_interaction", "nonlinear_continuous"))
STEP = pd.Timedelta(minutes=5)
CUTOFF = pd.Timestamp("2026-10-17 00:00:00", tz="UTC")


@dataclass(frozen=True)
class Spec:
    version: str = VERSION
    assets: tuple[str, ...] = ASSETS
    k_states: int = 4                  # fixed from PSH01, NOT retuned here
    origin_step_minutes: int = 30      # same anchor support as PSH01
    horizons_minutes: tuple[int, ...] = (5, 30)
    primary_horizon_minutes: int = 5   # matches CS03c's predictive horizon
    logistic_C: float = 0.25
    ridge_alpha: float = 100.0
    nonlinear_leaf_nodes: int = 7
    nonlinear_iterations: int = 80
    roundtrip_cost_bp: float = 12.0
    top_decile_fraction: float = 0.1
    bootstrap_reps: int = 2000
    familywise_alpha: float = 0.05
    min_test_days: int = 30
    seed: int = 2618
    status: str = "HISTORICAL_EXPLORATORY"


def registration(cfg: Spec = Spec()) -> dict:
    return asdict(cfg) | {
        "source": "Binance USD-M 1min contract candles and taker buy quote volume; "
                  "Binance spot 5min price for VD12 EMA states",
        "cross_branch": "PSH01 K=4 historical structure; CS03c signed flow evidence",
        "feature_admission": ["ema_30m", "ema_240m", "flow_5m"],
        "primary_metric": "next-5m nonzero perp direction logloss",
        "comparisons": [list(p) for p in COMPS],
        "familywise_one_sided_quantile": 1 - cfg.familywise_alpha/(len(COMPS)*2),
        "historical_test_reused": True, "options_economics": "NOT_TESTED",
        "trade_status": "NO_EXECUTION_APPROVAL"}


def read_perp_panel(path: Path, *, end: pd.Timestamp = CUTOFF) -> pd.DataFrame:
    """Compute completed 5-minute taker flow and forward perp price returns.

    Futures minute open time m contributes ONLY to the completed bucket ending
    ceil5(m+1min).  All labels use later completed perp buckets. Missing
    or incomplete buckets invalidate the full future window, never get filled.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    returns, flow = csp.read_perp(path, end=end)
    r = returns[["state_time", "return_5m"]].copy().sort_values("state_time")
    if r.state_time.duplicated().any():
        raise ValueError("Repeated futures 5m timestamp")
    out = r.merge(flow[["state_time", "flow_5m", "flow_valid"]],
                  on="state_time", how="left", validate="one_to_one")
    times = pd.to_datetime(out.state_time, utc=True)
    arr = pd.to_numeric(out.return_5m, errors="coerce").to_numpy(float)
    for horizon in (5, 30):
        steps = horizon // 5
        total = np.zeros(len(out), float)
        valid = np.ones(len(out), bool)
        for k in range(1, steps + 1):
            future = np.r_[arr[k:], np.full(k, np.nan)]
            time_future = times.shift(-k)
            valid &= np.isfinite(future) & time_future.eq(times + k*STEP).to_numpy()
            total += np.nan_to_num(future, nan=0.)
        out[f"perp_fwd_{horizon}m"] = np.where(valid, total, np.nan)
    return out.drop(columns=["return_5m"])


def assemble(asset: pd.DataFrame, future: pd.DataFrame, cfg: Spec) -> pd.DataFrame:
    """Join source-exact timestamps without allowing future return as feature."""
    a = asset.merge(future, on="state_time", how="left", validate="one_to_one")
    a["flow_5m"] = pd.to_numeric(a.flow_5m, errors="coerce").where(
        a.flow_valid.fillna(False).astype(bool))
    # Products of TRAIN-scaled inputs are created in build_design, not here.
    for h in cfg.horizons_minutes:
        label = f"perp_fwd_{h}m"
        for sp in ("train", "valid", "test"):
            rows = a.split.eq(sp)
            if not rows.any():
                continue
            next_split_start = (a.loc[a.split.eq({"train":"valid","valid":"test",
                                                  "test":"_none"}[sp]), "state_time"].min()
                                if sp != "test" else CUTOFF)
            if pd.notna(next_split_start):
                a.loc[rows & (a.state_time + pd.Timedelta(minutes=h) >= next_split_start), label] = np.nan
    return a


def train_transform(train: pd.DataFrame, other: pd.DataFrame,
                    cols: tuple[str, ...], *, normalizer: StandardScaler | None = None):
    """Build train-only standardized continuous design; cross-terms in z units."""
    sc = normalizer or StandardScaler().fit(train[list(CONT)].to_numpy(float))
    def one(f):
        z = sc.transform(f[list(CONT)].to_numpy(float))
        base = pd.DataFrame(z, columns=CONT, index=f.index)
        base["flow_x_slow"] = base["flow_5m"] * base["ema_240m"]
        base["flow_x_slope"] = base["flow_5m"] * (base["ema_30m"]-base["ema_240m"])
        return base[list(cols)].to_numpy(float)
    return one(train), one(other), sc


def design_for_arm(train: pd.DataFrame, other: pd.DataFrame,
                   ptr: np.ndarray, pte: np.ndarray, arm: str):
    """Common features shared by the fair nonlinear and HMM comparisons."""
    cols = BASE if arm == "price_vol" else CONT if arm in (
        "plus_flow", "nonlinear_continuous") else INTER
    tr, te, _ = train_transform(train, other, cols)
    if arm in ("hmm_additive", "hmm_flow_interaction"):
        # K-1 coordinates remove probability simplex collinearity.
        tr = np.column_stack([tr, ptr[:, :-1]])
        te = np.column_stack([te, pte[:, :-1]])
    if arm == "hmm_flow_interaction":
        # Standardized flow coordinate is the CONT column "flow_5m".
        ztr, zte, _ = train_transform(train, other, CONT)
        j = CONT.index("flow_5m")
        tr = np.column_stack([tr, ptr[:, :-1] * ztr[:, j, None]])
        te = np.column_stack([te, pte[:, :-1] * zte[:, j, None]])
    # Scale expanded features on TRAIN alone, no later fold information.
    sc = StandardScaler().fit(tr)
    return sc.transform(tr), sc.transform(te)


def loss_binary(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-7, 1-1e-7)
    return -(y*np.log(p) + (1-y)*np.log1p(-p))


def day_bootstrap(delta: np.ndarray, days: np.ndarray, cfg: Spec) -> dict:
    """Paired daily resampling; joint asset-wide multiplicity set separately."""
    d = np.asarray(delta, float)
    u, ix = np.unique(days, return_inverse=True)
    sums = np.bincount(ix, weights=d, minlength=len(u))
    counts = np.bincount(ix, minlength=len(u))
    rng = np.random.default_rng(cfg.seed)
    draw = rng.integers(len(u), size=(cfg.bootstrap_reps, len(u)))
    sampled = sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
    q = 1-cfg.familywise_alpha/(len(COMPS)*2)
    return {"mean_delta": float(d.mean()), "upper95": float(np.quantile(sampled,.95)),
            "upper_familywise": float(np.quantile(sampled,q)),
            "lower95": float(np.quantile(sampled,.05)), "n_days": int(len(u)),
            "favorable_bonferroni": bool(len(u)>=cfg.min_test_days and d.mean()<0 and np.quantile(sampled,q)<0)}


def evaluate_horizon(parts: dict, posterior: dict, cfg: Spec, horizon: int):
    """Identical nonzero-return origin support for all binary probability arms."""
    col = f"perp_fwd_{horizon}m"
    good = {}
    for name in ("train", "valid", "test"):
        f = parts[name]
        good[name] = (f.flow_5m.notna() & np.isfinite(f[col].to_numpy(float)) &
                      f.state_ready.astype(bool))
    train = parts["train"].loc[good["train"]].copy()
    test = parts["test"].loc[good["test"]].copy()
    ptrain = posterior["train"]["hmm"][good["train"].to_numpy()]
    ptest = posterior["test"]["hmm"][good["test"].to_numpy()]
    if len(train) < 600 or len(test) < 100:
        raise ValueError(f"Insufficient complete common futures+flow data: {len(train)}/{len(test)}")
    # Sign model has binary conditional-on-nonzero estimand; zero bars have
    # no class, but all valid bars remain in the independent mean/P&L test.
    nztr = np.abs(train[col].to_numpy(float)) > 0.
    nzte = np.abs(test[col].to_numpy(float)) > 0.
    yt = (train.loc[nztr, col].to_numpy(float) > 0).astype(int)
    ye = (test.loc[nzte, col].to_numpy(float) > 0).astype(int)
    if len(np.unique(yt)) < 2 or len(ye) < 100:
        raise ValueError("Insufficient valid sign samples")
    train_nz, test_nz = train.loc[nztr], test.loc[nzte]
    ptrain_nz, ptest_nz = ptrain[nztr], ptest[nzte]
    preds = {}; gross = {}; metrics = []; econ = []
    for arm in ARMS:
        xa, xb = design_for_arm(train_nz, test_nz, ptrain_nz, ptest_nz, arm)
        if arm == "nonlinear_continuous":
            model = HistGradientBoostingClassifier(
                max_leaf_nodes=cfg.nonlinear_leaf_nodes,
                max_iter=cfg.nonlinear_iterations, learning_rate=.05,
                l2_regularization=10., random_state=cfg.seed)
        else:
            model = LogisticRegression(C=cfg.logistic_C, max_iter=350,
                                       solver="lbfgs")
        model.fit(xa, yt)
        probability = model.predict_proba(xb)[:, 1]
        preds[arm] = probability
        metrics.append(dict(model=arm, horizon_min=horizon,
            n_nonzero_test=int(len(ye)), n_positive_test=int(ye.sum()),
            logloss=float(loss_binary(ye,probability).mean()),
            brier=float(np.mean((probability-ye)**2))))
        if arm == "nonlinear_continuous":
            continue  # No comparable nonlinear expected-return regression in this module
        # Independent Ridge expectation in bp for a cost-gated perp scenario.
        ra, rb = design_for_arm(train,test,ptrain,ptest,arm)
        target = 1e4*train[col].to_numpy(float)
        realized = 1e4*test[col].to_numpy(float)
        pred_bp = Ridge(alpha=cfg.ridge_alpha).fit(ra,target).predict(rb)
        chosen = np.abs(pred_bp) > cfg.roundtrip_cost_bp
        side = np.sign(pred_bp[chosen])
        pnl = side*realized[chosen]-cfg.roundtrip_cost_bp
        # TRAIN prediction quantile is a second FIXED research scenario;
        # its trade count cannot be optimized on TEST.
        # Reported only as diagnostic, not a trade recommendation.
        fit_forecast = Ridge(alpha=cfg.ridge_alpha).fit(ra,target).predict(ra)
        cut = float(np.quantile(np.abs(fit_forecast),1-cfg.top_decile_fraction))
        decile = np.abs(pred_bp) >= cut
        side2 = np.sign(pred_bp[decile])
        net2 = side2*realized[decile]-cfg.roundtrip_cost_bp
        econ.append(dict(model=arm,horizon_min=horizon,n_test=int(len(test)),
            return_mse_bp2=float(np.mean((pred_bp-realized)**2)),
            return_mae_bp=float(np.mean(np.abs(pred_bp-realized))),
            zero_forecast_mse_bp2=float(np.mean(realized**2)),
            n_cost_gated=int(chosen.sum()),gross_bp_per_cost_trade=
                float((side*realized[chosen]).mean()) if chosen.any() else None,
            net_bp_per_cost_trade=float(pnl.mean()) if chosen.any() else None,
            n_top_decile=int(decile.sum()),
            net_bp_per_top_decile_trade=float(net2.mean()) if decile.any() else None,
            gross_bp_per_top_decile_trade=float((side2*realized[decile]).mean()) if decile.any() else None,
            pnl_research_only=True))
    days = test_nz.state_time.dt.floor("D").to_numpy()
    comps = []
    for left, right in COMPS:
        b = day_bootstrap(loss_binary(ye,preds[left])-
                          loss_binary(ye,preds[right]),days,cfg)
        comps.append(dict(left=left,right=right,horizon_min=horizon,**b))
    return metrics, comps, econ, {"n_train":int(len(train)),
                                  "n_test":int(len(test)),"n_nonzero_test":int(len(ye)),
                                  "days_test":int(len(np.unique(days)))}


def run(price_dir: Path, futures_dir: Path, out_dir: Path, cfg: Spec=Spec()) -> dict:
    """No downloads or mutations outside PSH02's numbered output folder."""
    out = Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    pspec = psh.Spec(assets=cfg.assets, hmm_states=(cfg.k_states,),
                     seed=2617, anchor_minutes=cfg.origin_step_minutes)
    panel, _ = psh.load_panel(Path(price_dir),pspec)
    all_scores=[]; all_comps=[]; all_econ=[]; all_coverage=[]; decisions=[]
    for asset in cfg.assets:
        source=Path(futures_dir)/f"{asset}USDT_um_1m.parquet"
        if not source.exists():
            decisions.append({"asset":asset,"verdict":"INSUFFICIENT_PERP_SOURCE",
                              "prospective_confirmation":False})
            continue
        futures=read_perp_panel(source)
        f=assemble(panel.loc[panel.underlying.eq(asset)].copy(),futures,cfg)
        parts=psh._partition(f)
        if min(len(v) for v in parts.values())<300:
            raise ValueError(f"{asset}: insufficient state-ready chronology")
        model, posterior, _, occup=psh._hmm_fit(parts,cfg.k_states,pspec)
        if float(occup.min())<.01:
            raise ValueError(f"{asset}: PSH01 K=4 insufficient TRAIN occupancy")
        outcomes={}
        for h in cfg.horizons_minutes:
            scores,comparisons,econ,coverage=evaluate_horizon(parts,posterior,cfg,h)
            for obj in scores: all_scores.append(dict(asset=asset,**obj))
            for obj in comparisons: all_comps.append(dict(asset=asset,**obj))
            for obj in econ: all_econ.append(dict(asset=asset,**obj))
            outcomes[str(h)] = coverage
            all_coverage.append(dict(asset=asset,horizon_min=h,
                                     min_hmm_occupancy=float(occup.min()),**coverage))
        decisions.append({"asset":asset,"verdict":"PENDING_POOLED_DECISION",
                          "coverage":outcomes,"prospective_confirmation":False,
                          "execution_approved":False})
    # Only the preregistered BTC/ETH 5-minute tests form the primary 10-test
    # Bonferroni family: no post-hoc adoption from SOL/XRP or 30-minute cells.
    primary={(r["asset"],r["left"],r["right"]):r for r in all_comps
             if r["asset"] in ("BTC","ETH") and r["horizon_min"]==cfg.primary_horizon_minutes}
    by_asset={x["asset"]:x for x in decisions}
    for asset in ("BTC","ETH"):
        if asset not in by_asset:
            continue
        rows=[primary.get((asset,l,r)) for l,r in COMPS]
        by_asset[asset]["primary_checks"]=[dict(left=l,right=r,
                   passed=bool(z and z["favorable_bonferroni"]),
                   delta=z["mean_delta"] if z else None,
                   upper_familywise=z["upper_familywise"] if z else None)
                   for (l,r),z in zip(COMPS,rows)]
        by_asset[asset]["verdict"]=("EXPLORATORY_STATE_INTERACTION_GAIN"
            if all(z and z["favorable_bonferroni"] for z in rows)
            else "NO_CONFIRMED_INCREMENT")
    for asset in ("SOL","XRP"):
        if asset in by_asset:
            by_asset[asset]["verdict"]="SECONDARY_DESCRIPTIVE_ONLY"
    pd.DataFrame(all_scores).to_csv(out/"02_forecast_scores.csv",index=False)
    pd.DataFrame(all_comps).to_csv(out/"02_paired_day_comparisons.csv",index=False)
    pd.DataFrame(all_econ).to_csv(out/"02_perp_cost_scenarios.csv",index=False)
    pd.DataFrame(all_coverage).to_csv(out/"02_coverage.csv",index=False)
    (out/"02_decision.json").write_text(json.dumps(decisions,indent=2))
    (out/"02_manifest.json").write_text(json.dumps({
        "spec":registration(cfg),"historical_cutoff_exclusive":str(CUTOFF),
        "assets":list(cfg.assets),"synthetic":False,"prospective_confirmation":False,
        "outcome_data":"Binance USD-M future CLOSE returns; no fills/funding",
        "sources_modified":False},indent=2))
    return {"decisions":decisions,"scores":pd.DataFrame(all_scores),
            "comparisons":pd.DataFrame(all_comps),
            "perp_scenarios":pd.DataFrame(all_econ)}
