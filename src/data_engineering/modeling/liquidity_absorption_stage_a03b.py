"""LA A03b: keep the pre-existing A03 structural take-profit; mirror the initial stop.

A02/A02b population and A03 target selection remain unchanged. At the first
post-signal print, choose the target with the original A03 provisional stop-risk
basis, then set stop = 2 * entry PRINT - target. This is *exactly 1:1* in
uncosted price distance, not guaranteed +1R / -1R net of fees, gaps or timeouts.

Only the FIXED policy is reported. Under the original A03 rule, break-even
activation is +1 original R, which coincides with a 1R target in this variant;
BE and HH/HL trailing therefore never engage before the target. The inherited
engine still evaluates the other modes; we assert their outcome equivalence.

Read-only Binance historical aggTrades, no quote/depth data and no funding.
This is exploratory, NOT independent confirmation or an implementable portfolio.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_engineering.modeling.liquidity_absorption_stage_a import binance_root, repo_root
from src.data_engineering.modeling.liquidity_absorption_stage_a03 import (
    Config as A03Config, EventSimulation, prepare_bars, load_candidates,
    scan_event_tapes, MODES,
)

VERSION = "LA-A03b-0.1.0"
NOTIONAL_USDT = 10_000.0
CLOSED = {"STOP", "TARGET", "TIMEOUT"}


class MirroredStopSimulation(EventSimulation):
    """Select exactly the A03 target THEN replace only the initial stop distance."""

    def _start(self, tm: int, px: float) -> None:
        super()._start(tm, px)
        if self.entry_ms is None:
            return
        old_a03_stop = self.initial_stop
        target_distance = self.d * (self.target - self.entry_print)
        if not (np.isfinite(target_distance) and target_distance > 0):
            raise ValueError("Invalid pre-existing structural target geometry")
        mirrored_stop = float(2 * self.entry_print - self.target)
        if mirrored_stop <= 0 or not np.isfinite(mirrored_stop):
            raise ValueError("Nonpositive mirrored stop: refuse unphysical execution")
        self.a03_original_stop = float(old_a03_stop)
        self.a03_original_initial_risk_price = float(self.risk)
        self.initial_stop = mirrored_stop  # no tick rounding: exact 1:1 before costs
        self.risk = float(target_distance)
        for state in self.states.values():
            state.stop = mirrored_stop
            state.initial_risk_price = float(target_distance)

    def fixed_record(self) -> dict:
        records = self.records()
        fixed = next(x for x in records if x["mode"] == "FIXED")
        for other in records:
            if other["mode"] == "FIXED":
                continue
            # At +1R the fixed 1R target is hit. No prior BE/trailing policy
            # divergence should occur under unchanged trigger rules.
            if other["reason"] != fixed["reason"]:
                raise AssertionError("1:1 policy unexpected divergence in reason")
            for key in ("exit_time_ms", "exit_reference_px", "gross_r", "net_r_assumed_costs"):
                a, b = fixed[key], other[key]
                if (pd.isna(a) and pd.isna(b)):
                    continue
                if not np.isclose(float(a), float(b), rtol=1e-10, atol=1e-8):
                    raise AssertionError(f"1:1 policy diverges at {key}: {a} vs {b}")
            if other["trail_updates"]:
                raise AssertionError("1:1 study cannot activate +1R trailing before 1R TP")
        fixed["strategy"] = "A03B_MIRRORED_STOP_FIXED_1TO1"
        fixed["a03_original_stop"] = getattr(self, "a03_original_stop", np.nan)
        fixed["a03_original_risk_price"] = getattr(self, "a03_original_initial_risk_price", np.nan)
        fixed["symmetric_1to1_before_costs"] = bool(
            self.entry_ms is not None and np.isclose(fixed["target_rr_initial"], 1.0, rtol=0, atol=1e-10)
        )
        return fixed


def check_against_original(a03: pd.DataFrame, mirrored: pd.DataFrame) -> pd.DataFrame:
    """Ensure the target selection, eligibility and entry are identical to A03.

    Report paired statuses; do not copy historic A03 outcomes into this variant.
    """
    old = a03[a03["mode"] == "FIXED"].copy()
    if old.a02_candidate_row.duplicated().any() or mirrored.a02_candidate_row.duplicated().any():
        raise AssertionError("Duplicate candidate IDs")
    if len(old) != len(mirrored):
        raise AssertionError("Candidate counts differ from original A03")
    join = mirrored.merge(old, on="a02_candidate_row", how="outer", validate="one_to_one",
                          suffixes=("_1to1", "_old"), indicator=True)
    if not join["_merge"].eq("both").all():
        raise AssertionError("Unpaired candidate IDs")
    for field in ("symbol", "side", "signal_time_utc", "target_source"):
        if not join[f"{field}_1to1"].equals(join[f"{field}_old"]):
            raise AssertionError(f"A03 target/event drift: {field}")
    for field in ("entry_time_ms", "entry_print", "entry_fill_assumed", "target"):
        v = pd.to_numeric(join[f"{field}_1to1"], errors="coerce").to_numpy(float)
        w = pd.to_numeric(join[f"{field}_old"], errors="coerce").to_numpy(float)
        if field == "entry_time_ms":
            agree = np.isclose(v, w, rtol=0, atol=0, equal_nan=True)
        else:
            agree = np.isclose(v, w, rtol=1e-12, atol=1e-7, equal_nan=True)
        if not np.all(agree):
            raise AssertionError(f"A03 target/entry/eligibility drift: {field}")
    skip_1to1 = join["reason_1to1"].isin(["NO_STRUCTURAL_TARGET", "NO_PROMPT_ENTRY_PRINT", "INVALID_ENTRY_STOP_GEOMETRY"])
    skip_old = join["reason_old"].isin(["NO_STRUCTURAL_TARGET", "NO_PROMPT_ENTRY_PRINT", "INVALID_ENTRY_STOP_GEOMETRY"])
    if not skip_1to1.equals(skip_old):
        raise AssertionError("Structural-target availability drift")
    joined = join.drop(columns="_merge")
    joined["paired_closed"] = joined.reason_1to1.isin(CLOSED) & joined.reason_old.isin(CLOSED)
    return joined


def _metric_stats(series: pd.Series) -> dict:
    values = series.dropna().astype(float)
    return {"n": int(len(values)), "mean": float(values.mean()) if len(values) else None,
            "median": float(values.median()) if len(values) else None}


def build_coverage(frame: pd.DataFrame, paired: pd.DataFrame) -> list[dict]:
    out = []
    for sym, sub in frame.groupby("symbol"):
        closed = sub[sub.reason.isin(CLOSED)]
        cmp = paired[(paired.symbol_1to1 == sym) & paired.paired_closed]
        out.append({
            "symbol": sym, "candidates": int(len(sub)), "closed": int(len(closed)),
            "stop": int((sub.reason == "STOP").sum()),
            "target": int((sub.reason == "TARGET").sum()),
            "timeout": int((sub.reason == "TIMEOUT").sum()),
            "no_structural_target": int((sub.reason == "NO_STRUCTURAL_TARGET").sum()),
            "other_no_entry": int(sub.reason.isin(["NO_PROMPT_ENTRY_PRINT", "INVALID_ENTRY_STOP_GEOMETRY", "NO_ENTRY_TAPE"]).sum()),
            "censored": int((sub.reason == "CENSORED_TAPE_END").sum()),
            "gross_r_mean": _metric_stats(closed.gross_r)["mean"],
            "net_r_mean": _metric_stats(closed.net_r_assumed_costs)["mean"],
            "net_return_bp_mean": _metric_stats(closed.net_return_bp_assumed_costs)["mean"],
            "profitable_closed_share": float((closed.net_r_assumed_costs > 0).mean()) if len(closed) else None,
            "initial_risk_bp_median": _metric_stats(closed.initial_risk_bps)["median"],
            "paired_closed_original_a03_fixed": int(len(cmp)),
            "original_a03_fixed_paired_net_return_bp_mean": _metric_stats(cmp.net_return_bp_assumed_costs_old)["mean"],
            "mirrored_paired_net_return_bp_mean": _metric_stats(cmp.net_return_bp_assumed_costs_1to1)["mean"],
        })
    return out


def equity_paths(paired: pd.DataFrame, notional: float = NOTIONAL_USDT) -> pd.DataFrame:
    """Closed-event, exit-time-aligned virtual P&L, *not* an executable equity curve.

    Each event contributes fixed USDT notional once, with no capital, turnover,
    funding or overlapping exposure constraints. Only the *same* events closed
    under both original and 1:1 simulations are used for a paired comparison.
    """
    if not (np.isfinite(notional) and notional > 0):
        raise ValueError("Invalid fixed comparison notional")
    sub = paired.loc[paired.paired_closed].copy()
    records = []
    for universe in ("BTCUSDT", "ETHUSDT", "COMBINED"):
        keep = sub if universe == "COMBINED" else sub[sub.symbol_1to1 == universe]
        for label, suffix in (("A03_ORIGINAL_FIXED", "old"), ("A03B_MIRRORED_1TO1", "1to1")):
            if keep.empty:
                continue
            a = keep[["a02_candidate_row", f"exit_time_ms_{suffix}",
                      f"gross_return_bp_{suffix}", f"net_return_bp_assumed_costs_{suffix}",
                      f"gross_r_{suffix}", f"net_r_assumed_costs_{suffix}"]].copy()
            a.columns = ["a02_candidate_row", "exit_time_ms", "gross_bp", "net_bp", "gross_r", "net_r"]
            if a.isna().any().any():
                raise AssertionError("Paired closed equity data unexpectedly missing")
            a = a.sort_values(["exit_time_ms", "a02_candidate_row"]).reset_index(drop=True)
            a["universe"] = universe
            a["model"] = label
            a["ordinal_closed"] = np.arange(1, len(a) + 1, dtype=int)
            a["time_utc"] = pd.to_datetime(a.exit_time_ms, unit="ms", utc=True)
            a["cum_gross_usdt"] = a.gross_bp.cumsum() / 10000 * notional
            a["cum_net_usdt"] = a.net_bp.cumsum() / 10000 * notional
            a["cum_gross_r"] = a.gross_r.cumsum()
            a["cum_net_r"] = a.net_r.cumsum()
            records.append(a)
    if not records:
        return pd.DataFrame(columns=["universe", "model", "time_utc", "cum_gross_usdt", "cum_net_usdt", "cum_gross_r", "cum_net_r"])
    return pd.concat(records, ignore_index=True)


def plot_equity(paths: pd.DataFrame, out: Path, notional: float = NOTIONAL_USDT) -> None:
    import matplotlib.pyplot as plt
    for name, quantity, title, filename in (
        ("net", "cum_net_usdt", f"Cumulative net P&L • USDT {notional:,.0f} fixed notional / trade", "03b_equity_net.png"),
        ("gross", "cum_gross_usdt", f"Cumulative gross P&L • USDT {notional:,.0f} fixed notional / trade", "03b_equity_gross.png"),
    ):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for ax, universe in zip(axes, ["BTCUSDT", "ETHUSDT", "COMBINED"]):
            block = paths.loc[paths.universe == universe]
            for policy in ("A03_ORIGINAL_FIXED", "A03B_MIRRORED_1TO1"):
                values = block.loc[block.model == policy].sort_values("time_utc")
                if not values.empty:
                    ax.step(values.time_utc, values[quantity], where="post", label=policy)
            ax.axhline(0, linewidth=.8)
            ax.set_ylabel(f"{universe}\nP&L (USDT)")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="best", fontsize=8)
        fig.suptitle(title + "\nPaired completed historical events only; exits ordered by actual exit time")
        axes[-1].set_xlabel("UTC exit date • overlapping hypothetical positions allowed")
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.savefig(out / filename, dpi=155, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, universe in zip(axes, ["BTCUSDT", "ETHUSDT", "COMBINED"]):
        values = paths.loc[(paths.universe == universe) & (paths.model == "A03B_MIRRORED_1TO1")].sort_values("time_utc")
        if not values.empty:
            ax.step(values.time_utc, values.cum_gross_r, where="post", label="1:1 gross R")
            ax.step(values.time_utc, values.cum_net_r, where="post", label="1:1 net R")
        ax.axhline(0, linewidth=.8)
        ax.set_ylabel(f"{universe}\nCumulative R")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle("Mirrored-stop 1:1 • cumulative initial-risk units\nBefore vs after assumed transaction costs")
    axes[-1].set_xlabel("UTC exit date • not a capital-constrained or funded equity curve")
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(out / "03b_1to1_r_curve.png", dpi=155, bbox_inches="tight")
    plt.close(fig)


def run_a03b(repo: Path, data_root: Path, out: Path, cfg: A03Config | None = None) -> dict:
    """Execute exact symmetric-stop counterfactual on local historical tape."""
    cfg = cfg or A03Config()
    if cfg.break_even_initial_r != 1.0:
        raise ValueError("A03b requires the original +1R BE rule unchanged")
    if cfg.hold_minutes <= 0:
        raise ValueError("Invalid holding time")
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((repo / "notebooks/Liquidity_Absorption/03/03_decision.json").read_text(encoding="utf-8"))
    if meta.get("version") != "LA-A03-0.1.0" or meta.get("source_events") != 452:
        raise ValueError("Unexpected baseline A03 decision contract")
    a = load_candidates(repo)
    sims = []
    for sym, group in a.groupby("symbol"):
        bars = prepare_bars(data_root, sym)
        for event in group.to_dict("records"):
            sims.append(MirroredStopSimulation(event, bars, cfg))
    tape = scan_event_tapes(data_root, sims)
    new = pd.DataFrame([sim.fixed_record() for sim in sims])
    if len(new) != len(a) or new.a02_candidate_row.duplicated().any():
        raise AssertionError("A02 candidate universe drift")
    old = pd.read_csv(repo / "notebooks/Liquidity_Absorption/03/03_execution_events.csv")
    paired = check_against_original(old, new)
    if not new.loc[new.reason.isin(CLOSED), "symmetric_1to1_before_costs"].all():
        raise AssertionError("At least one entered trade was not exactly 1:1 before costs")
    new.to_csv(out / "03b_1to1_events.csv", index=False)
    paired.to_csv(out / "03b_paired_comparison.csv", index=False)
    paths = equity_paths(paired)
    paths.to_csv(out / "03b_equity_paths.csv", index=False)
    if not paths.empty:
        plot_equity(paths, out)
    coverage = build_coverage(new, paired)
    pd.DataFrame(coverage).to_csv(out / "03b_coverage.csv", index=False)
    contract = {
        "version": VERSION, "parent": "LA-A03-0.1.0", "source_events": int(len(new)),
        "config": asdict(cfg),
        "target": "IDENTICAL original A03 selection: VWAP if >=2 ORIGINAL-stop R favorable, else opposite fully completed prior-day extreme; no LVN",
        "entry": "Original first aggTrade print at or after signal close, same 60-second limit",
        "stop": "Exactly 2 * entry_print - original_A03_target, unrounded to preserve 1:1 PRE-COST price distance; order tick validity is not claimed",
        "fixed_target_rr": 1.0, "exit_mode": "FIXED only; original +1R BE/HH-HL trailing degenerate with 1R target and are verified equivalent",
        "timeout": "240min unchanged; timeouts can realize non-1R returns; missing tape censored",
        "fees_impact": "Identical A03 fees/assumed adverse execution, before funding; actual executable quotes not known",
        "equity": "Exit-time-ordered realized standalone trade outcomes, paired completed events only; USDT 10000 fixed hypothetical notional per trade, overlapping positions allowed without leverage/capital/portfolio constraints",
        "status": "EXPLORATORY_COMPARISON_ONLY_NOT_CONFIRMED_OR_INVESTABLE",
    }
    (out / "03b_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    report = {"version": VERSION, "status": contract["status"], "source_events": int(len(new)),
              "baseline_version": "LA-A03-0.1.0", "coverage": coverage,
              "paired_closed_events": int(paired.paired_closed.sum()),
              "print_tape": tape,
              "charts": ["03b_equity_net.png", "03b_equity_gross.png", "03b_1to1_r_curve.png"] if not paths.empty else [],
              "caution": "Original structural targets kept; ONLY stop mirrored to entry/TP distance. Gross RR exactly 1 before exit gaps/timeouts; net returns include assumed transaction costs (no funding). Equity curves are NOT a capital-constrained portfolio."}
    (out / "03b_decision.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--binance-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    repo = args.repo or repo_root()
    data_root = args.binance_root or binance_root(repo)
    out = args.out or repo / "notebooks/Liquidity_Absorption/03b"
    report = run_a03b(repo, data_root, out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
