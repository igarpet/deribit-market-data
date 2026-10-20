"""LA-08: Frozen A03b 1:1 strategy evaluated using Claude's Deribit-native signals and filters.

No new exit simulation, threshold search, portfolio claims, or use of outcomes in eligibility.
Inputs are the already executed EXT A03b event replay and V2 signal-time CVD/profile
context. Also compares the pre-existing V2 TRAIL_1M/no CUT15 outcome on original IDs.

This is an EXPLORATORY methodology-bridge on an already inspected history,
NOT an untouched confirmation. Missing or censored trades are never manufactured.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = "LA-BRIDGE-08-0.1.0"
CLOSED = ("TARGET", "STOP", "TIMEOUT")
TARGET_OK = "PROFILE_LVN_TO_HVN"
POLICIES = (
    "A03B_ALL",
    "A03B_CVD_ONLY",
    "A03B_PROFILE_ONLY",
    "A03B_CVD_AND_PROFILE",
    "CLAUDE_V2_CVD_PROFILE_VOL_TRAIL",
)
INPUTS = {
    "base": "EXT/03b/03b_1to1_events.csv",
    "signals": "V2/V2_signals.csv",
    "v2": "V2/V2_events.csv",
}
PERIODS = ("H1", "H2", "INSAMPLE_JUL_SEP_2026", "HOLDOUT_INSPECTED")

@dataclass(frozen=True)
class Spec:
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 20261020
    bootstrap_alpha: float = .05
    base_1to1_tolerance: float = 1e-6
    require_complete_signal_match: bool = True


def utc_ms(series: pd.Series) -> pd.Series:
    t = pd.to_datetime(series, utc=True, errors="raise")
    if t.isna().any():
        raise ValueError("Null signal datetime")
    return t.dt.as_unit("ms").astype("int64")


def require_unique(df: pd.DataFrame, name: str) -> None:
    cols = ["a02_candidate_row", "symbol", "side", "signal_time_utc"]
    missing = set(cols).difference(df)
    if missing:
        raise ValueError(f"{name}: missing identification columns: {sorted(missing)}")
    if df.a02_candidate_row.isna().any() or df.a02_candidate_row.duplicated().any():
        raise ValueError(f"{name}: invalid/duplicated candidate ID")
    if not df.symbol.isin(("BTCUSDT", "ETHUSDT")).all() or not df.side.isin(("long", "short")).all():
        raise ValueError(f"{name}: invalid symbol or side")
    if not len(df):
        raise ValueError(f"{name}: no signals")


def classify_period(dates: pd.Series) -> pd.Series:
    """Exact Claude LA-024 holdout date cells; previously INSPECTED, not fresh OOS."""
    t = pd.to_datetime(dates, utc=True)
    ins = (t >= pd.Timestamp("2026-07-01", tz="UTC")) & (t < pd.Timestamp("2026-10-01", tz="UTC"))
    return pd.Series(np.where(ins, "INSAMPLE_JUL_SEP_2026",
                        np.where(t < pd.Timestamp("2025-04-01", tz="UTC"), "H1", "H2")), index=dates.index)


def load_inputs(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(repo) / "notebooks" / "Liquidity_Absorption"
    paths = {k: root / v for k, v in INPUTS.items()}
    missing = [str(v) for v in paths.values() if not v.is_file()]
    if missing:
        raise FileNotFoundError("Run/obtain Claude EXT and V2 outputs first; missing: " + "; ".join(missing))
    a = pd.read_csv(paths["base"], low_memory=False)
    b = pd.read_csv(paths["signals"], low_memory=False)
    v = pd.read_csv(paths["v2"], low_memory=False)
    return a, b, v


def cost_reprice(events: pd.DataFrame) -> pd.Series:
    """Use the same cost bridge as LA-EXT, but only for COMPLETED simulated trades."""
    from src.data_engineering.modeling.liquidity_absorption_extended import deribit_net
    return deribit_net(events)


def ensure_invariants(b: pd.DataFrame, s: pd.DataFrame, v: pd.DataFrame,
                      spec: Spec = Spec()) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_unique(b, "EXT A03b baseline")
    require_unique(s, "V2 signal-time context")
    if len(b) != len(s):
        raise ValueError(f"EXT/V2 candidate universe mismatch: {len(b)} vs {len(s)}")
    keys = set(b.a02_candidate_row.astype(int))
    if keys != set(s.a02_candidate_row.astype(int)):
        raise ValueError("EXT/V2 candidate IDs do not match")
    reqb = {"reason", "entry_print", "target", "initial_stop", "target_rr_initial",
            "entry_time_ms", "exit_time_ms", "net_return_bp_assumed_costs", "gross_return_bp",
            "entry_fill_assumed", "exit_fill_assumed", "entry_fee_bp", "exit_fee_bp", "signal_time_utc"}
    reqs = {"cvd_ok", "cvd_flow", "v2_target_source", "v2_target", "signal_time_utc", "symbol", "side"}
    if reqb - set(b) or reqs - set(s):
        raise ValueError(f"EXT/V2 required fields missing: base {sorted(reqb-set(b))}, V2 {sorted(reqs-set(s))}")
    # Preserve the EXT event row as the canonical original trade. Suffix V2 fields.
    sig = s[["a02_candidate_row", "symbol", "side", "signal_time_utc", "cvd_ok", "cvd_flow",
             "v2_target_source", "v2_target"]].copy()
    joined = b.merge(sig, on="a02_candidate_row", how="left", validate="one_to_one", suffixes=("", "_v2"))
    for col in ("symbol", "side"):
        if not joined[col].eq(joined[f"{col}_v2"]).all():
            raise ValueError(f"Signal identity drift: {col}")
    if not np.array_equal(utc_ms(joined.signal_time_utc).to_numpy(),utc_ms(joined.signal_time_utc_v2).to_numpy()):
        raise ValueError("EXT/V2 signal-time drift")
    cvd = joined.cvd_ok.astype(str).str.lower()
    if not cvd.isin(("true", "false")).all():
        raise ValueError("V2 CVD eligibility missing/malformed")
    joined["cvd_ok"] = cvd.eq("true")
    joined["profile_available"] = joined.v2_target_source.eq(TARGET_OK)
    # V2 must not modify original A03b entries, target or mirrored stops.
    closed = joined.reason.isin(CLOSED)
    if not closed.any():
        raise ValueError("No resolved A03b baseline trades")
    z = joined.loc[closed].copy()
    for col in ("entry_print", "target", "initial_stop", "entry_time_ms", "exit_time_ms"):
        if not np.isfinite(pd.to_numeric(z[col], errors="coerce")).all():
            raise ValueError(f"Closed original trade lacks {col}")
    sign = np.where(z.side.eq("long"), 1., -1.)
    entry = z.entry_print.to_numpy(float);target = z.target.to_numpy(float);stop = z.initial_stop.to_numpy(float)
    target_r = sign * (target - entry)
    stop_r = sign * (entry - stop)
    if np.any(target_r <= 0) or np.any(stop_r <= 0) or not np.allclose(target_r,stop_r,rtol=spec.base_1to1_tolerance,atol=spec.base_1to1_tolerance):
        raise ValueError("A03b original target/stop geometric 1:1 invariant broken")
    # V2 is a different stop/target/management replay. Keep exactly no-CUT15 TRAIL_1M.
    if "v2_variant" not in v:
        raise ValueError("V2 events lack v2_variant")
    vv = v.loc[v.v2_variant.astype(str).eq("TRAIL_1M")].copy()
    require_unique(vv, "V2 no-CUT15 TRAIL_1M")
    if set(vv.a02_candidate_row.astype(int)) != keys:
        raise ValueError("V2 comparator candidate universe mismatch")
    for col in ("reason", "entry_print", "target", "initial_stop", "gross_return_bp",
                "net_return_bp_assumed_costs", "entry_fill_assumed", "exit_fill_assumed",
                "entry_fee_bp", "exit_fee_bp"):
        if col not in vv:
            raise ValueError(f"V2 comparator missing {col}")
    cmp = joined[["a02_candidate_row", "symbol", "side", "signal_time_utc", "cvd_ok", "profile_available"]].merge(
        vv[["a02_candidate_row", "symbol", "side", "signal_time_utc", "reason", "entry_print", "target", "initial_stop",
            "gross_return_bp", "net_return_bp_assumed_costs", "entry_fill_assumed", "exit_fill_assumed",
            "entry_fee_bp", "exit_fee_bp"]], on="a02_candidate_row", validate="one_to_one", suffixes=("", "_v2"))
    for col in ("symbol", "side"):
        if not cmp[col].eq(cmp[f"{col}_v2"]).all():
            raise ValueError(f"V2 comparator differs on {col}")
    if not np.array_equal(utc_ms(cmp.signal_time_utc).to_numpy(),utc_ms(cmp.signal_time_utc_v2).to_numpy()):
        raise ValueError("V2 replay differs on signal time")
    return joined.sort_values("a02_candidate_row").reset_index(drop=True),cmp.sort_values("a02_candidate_row").reset_index(drop=True)


def day_bootstrap(x: pd.Series, days: pd.Series, spec: Spec = Spec()) -> tuple[float,float]:
    """Day cluster resampling: overlap-aware but cannot address already-inspected selection."""
    z = pd.DataFrame({"x":pd.to_numeric(x, errors="coerce").to_numpy(),
                      "day":pd.to_datetime(days, utc=True).dt.floor("D").to_numpy()}).dropna()
    if len(z) < 5:
        return float("nan"),float("nan")
    gr = z.groupby("day").x.agg(["sum","count"])
    counts=gr["count"].to_numpy(float);sums=gr["sum"].to_numpy(float)
    rng=np.random.default_rng(spec.bootstrap_seed)
    draws=rng.integers(0,len(gr),size=(spec.bootstrap_iterations,len(gr)))
    estimates=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return float(sums.sum()/counts.sum()),float(np.quantile(estimates,spec.bootstrap_alpha))


def prepare_rows(original: pd.DataFrame, cmp: pd.DataFrame,
                 spec: Spec = Spec()) -> pd.DataFrame:
    """Build one row per original candidate × policy; eligibility is strictly at signal time."""
    primary=original.copy(); secondary=cmp.copy()
    orig_closed=primary.reason.isin(CLOSED)
    sec_closed=secondary.reason.isin(CLOSED)
    primary["net_deribit"] = np.nan
    secondary["net_deribit"] = np.nan
    primary.loc[orig_closed,"net_deribit"] = cost_reprice(primary.loc[orig_closed]).to_numpy(float)
    secondary.loc[sec_closed,"net_deribit"] = cost_reprice(secondary.loc[sec_closed]).to_numpy(float)
    if (not np.isfinite(primary.loc[orig_closed,"net_deribit"]).all()
            or not np.isfinite(secondary.loc[sec_closed,"net_deribit"]).all()):
        raise ValueError("Invalid net Deribit repricing")
    primary["period"] = classify_period(primary.signal_time_utc)
    secondary["period"] = classify_period(secondary.signal_time_utc)
    rows=[]
    for policy in POLICIES:
        v2=policy.startswith("CLAUDE_")
        d=(secondary if v2 else primary).copy()
        if policy=="A03B_ALL": eligible=pd.Series(True,index=d.index)
        elif policy=="A03B_CVD_ONLY": eligible=d.cvd_ok.astype(bool)
        elif policy=="A03B_PROFILE_ONLY": eligible=d.profile_available.astype(bool)
        else:eligible=d.cvd_ok.astype(bool)&d.profile_available.astype(bool)
        d["policy"] = policy
        d["policy_eligible"] = eligible
        d["policy_outcome"] = np.where(eligible,d.reason,"FILTERED_OUT")
        closed=d.reason.isin(CLOSED)&eligible
        d["policy_closed"] = closed
        d["policy_censored"] = eligible & d.reason.astype(str).str.startswith("CENSORED")
        d["policy_skipped"] = eligible & (~d.reason.isin(CLOSED)) & (~d["policy_censored"])
        d["policy_realized_net_bp"] = np.where(closed,d.net_deribit,0.)
        d["policy_realized_gross_bp"] = np.where(closed,pd.to_numeric(d.gross_return_bp,errors="coerce"),0.)
        d["executed_entry"] = eligible & pd.to_numeric(d.entry_print,errors="coerce").notna()
        # do not expose after-the-fact outcome fields as signal-time filter inputs
        rows.append(d[["a02_candidate_row","symbol","side","signal_time_utc","period","policy",
            "cvd_ok","profile_available","policy_eligible","policy_outcome", "policy_closed",
            "policy_censored","policy_skipped","executed_entry","policy_realized_net_bp","policy_realized_gross_bp"]])
    out=pd.concat(rows,ignore_index=True)
    if out.duplicated(["a02_candidate_row","policy"]).any() or len(out)!=len(primary)*len(POLICIES):
        raise AssertionError("Policy candidate accounting lost or repeated rows")
    return out


def summarize(data: pd.DataFrame,spec:Spec=Spec())->pd.DataFrame:
    """Net per *closed* trade and realized per original candidate; no censored-imputed payoffs."""
    records=[]
    for (policy,symbol), ss in data.groupby(["policy","symbol"],sort=True):
        for period in PERIODS:
            s=ss[ss.period!= "INSAMPLE_JUL_SEP_2026"] if period=="HOLDOUT_INSPECTED" else ss[ss.period==period]
            if s.empty: continue
            closed=s[s.policy_closed]
            net=closed.policy_realized_net_bp
            m,lo=day_bootstrap(net,closed.signal_time_utc,spec)
            # A zero contribution is attributed to non-entry or unresolved trades ONLY
            # in the realized-closed contribution metric, not as the economic payoff.
            contribution,contr_lo=day_bootstrap(s.policy_realized_net_bp,s.signal_time_utc,spec)
            records.append({"policy":policy,"symbol":symbol,"period":period,"n_original_signals":len(s),
                "n_eligible":int(s.policy_eligible.sum()),"n_entries":int(s.executed_entry.sum()),
                "n_closed":len(closed),"n_filtered":int((~s.policy_eligible).sum()),
                "n_censored":int(s.policy_censored.sum()),"n_skipped":int(s.policy_skipped.sum()),
                "n_target":int((s.policy_outcome=="TARGET").sum()),"n_stop":int((s.policy_outcome=="STOP").sum()),
                "n_timeout":int((s.policy_outcome=="TIMEOUT").sum()),
                "mean_net_bp_per_closed_trade":m,"net_closed_day_boot_lo95":lo,
                "realized_closed_net_bp_per_original_signal":contribution,
                "realized_closed_contribution_day_boot_lo95":contr_lo,
                "note":"Not independent validation; unresolved exits not imputed and fixed-notional contributions omit portfolio constraints"})
    return pd.DataFrame(records)


def paired_selection_effect(data:pd.DataFrame,spec:Spec=Spec())->pd.DataFrame:
    """Paired realized net contributions vs A03b, excluding *either-side censored* IDs.

    An original skip is a valid zero. An open-but-censored trade is NOT a known
    zero. Exclude both-source unresolved positions from each matched analysis.
    """
    p=data.pivot(index="a02_candidate_row",columns="policy",values="policy_realized_net_bp")
    cens=data.pivot(index="a02_candidate_row",columns="policy",values="policy_censored")
    base=data[data.policy=="A03B_ALL"].set_index("a02_candidate_row")
    rows=[]
    for policy in POLICIES[1:]:
        eligible_pair=~(cens[policy].astype(bool)|cens["A03B_ALL"].astype(bool))
        df=base.loc[eligible_pair,["signal_time_utc","symbol","period"]].copy()
        df["paired_contribution_delta_bp"]=(p.loc[eligible_pair,policy]-p.loc[eligible_pair,"A03B_ALL"])
        for symbol in ("BTCUSDT","ETHUSDT","ALL"):
            sub=df if symbol=="ALL" else df[df.symbol==symbol]
            for per in PERIODS:
                z=sub[sub.period!="INSAMPLE_JUL_SEP_2026"] if per=="HOLDOUT_INSPECTED" else sub[sub.period==per]
                if z.empty:continue
                m,lo=day_bootstrap(z.paired_contribution_delta_bp,z.signal_time_utc,spec)
                rows.append({"policy":policy,"symbol":symbol,"period":per,"n_resolved_paired_signals":len(z),
                             "n_excluded_censored":int((~eligible_pair & (base.symbol.eq(symbol) if symbol!='ALL' else pd.Series(True,index=base.index)) &
                                 (base.period.ne('INSAMPLE_JUL_SEP_2026') if per=='HOLDOUT_INSPECTED' else base.period.eq(per))).sum()),
                             "mean_paired_realized_contribution_delta_bp":m,"day_boot_lo95":lo,
                             "n_positive":int((z.paired_contribution_delta_bp>0).sum()),
                             "n_negative":int((z.paired_contribution_delta_bp<0).sum()),
                             "n_unchanged":int((z.paired_contribution_delta_bp==0).sum())})
    return pd.DataFrame(rows)


def run(repo:Path,out:Path, spec:Spec=Spec())->dict:
    base,sig,v2=load_inputs(repo)
    primary,cmp=ensure_invariants(base,sig,v2,spec)
    policy=prepare_rows(primary,cmp,spec)
    summary=summarize(policy,spec)
    deltas=paired_selection_effect(policy,spec)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    policy.to_csv(out/"08_policy_candidate_panel.csv",index=False)
    summary.to_csv(out/"08_policy_comparison.csv",index=False)
    deltas.to_csv(out/"08_paired_selection_effects.csv",index=False)
    # Document actual source files and decisions; no model fitted and no tuning.
    source={"version":VERSION,"status":"EXPLORATORY_ALREADY_INSPECTED_NOT_NEW_OOS",
        "origin":"EXT A03b original 1:1 payoff replay, joined 1:1 with V2 causal CVD/profile signal features",
        "original_signals":len(primary),"v2_same_signal_replay":len(cmp),
        "policies":list(POLICIES),"spec":asdict(spec),
        "original_target_and_stop":"A03b unchanged under four overlay filters",
        "v2_comparator":"Existing TRAIL_1M without CUT15: different target, stop and exits",
        "no_trade_unresolved":"Unresolved/censored positions are not imputed. Per-signal realized contribution is not economic EV; paired effects exclude either-side censorship.",
        "data_inspection":"Full EXT/V2 2024-2026 datasets already inspected; H1/H2 period labels do not imply pristine holdout.",
        "economic_limits":"Executed trade-price assumption only; 5bp/side synthetic taker fee, adverse fill offsets, no funding, book, queues or capital netting.",
        "candidate_outcomes":primary.reason.astype(str).value_counts().to_dict(),
        "v2_outcomes":cmp.reason.astype(str).value_counts().to_dict(),
        "cvd_true_share":float(primary.cvd_ok.mean()),
        "profile_available_share":float(primary.profile_available.mean()),
        "both_share":float((primary.cvd_ok&primary.profile_available).mean())}
    (out/"08_decision.json").write_text(json.dumps(source,indent=2)+"\n",encoding="utf-8")
    _plot(summary,out)
    return source


def _plot(summary:pd.DataFrame,out:Path)->None:
    import matplotlib.pyplot as plt
    rows=summary[(summary.period=="HOLDOUT_INSPECTED")].copy()
    if rows.empty:return
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,sym in zip(axes,("BTCUSDT","ETHUSDT")):
        sub=rows[rows.symbol==sym].set_index("policy").reindex(POLICIES)
        ax.barh(range(len(POLICIES)),sub.mean_net_bp_per_closed_trade)
        ax.set_yticks(range(len(POLICIES)),[p.replace("A03B_","").replace("CLAUDE_","") for p in POLICIES])
        ax.axvline(0,color="gray",linewidth=.8)
        ax.set(xlabel="Assumed-cost net bp / completed trade",title=sym)
        ax.invert_yaxis()
    fig.suptitle("Frozen A03b 1:1 with V2 signal-time filters; V2 trail is a different execution policy")
    fig.tight_layout();fig.savefig(out/"08_comparison_net_bp.png",dpi=150,bbox_inches="tight");plt.close(fig)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--out",type=Path)
    opts=parser.parse_args(argv)
    root=opts.repo or Path(__file__).resolve().parents[3]
    out=opts.out or root/"notebooks"/"Liquidity_Absorption"/"08"
    print(json.dumps(run(root,out),indent=2))

if __name__=="__main__":main()
