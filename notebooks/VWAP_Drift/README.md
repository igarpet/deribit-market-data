# VWAP Drift research branch

**Question:** does price drift toward the anchored VWAP after a displacement measured in volatility-engine units, beyond placebo anchors and by more than costs, at 15/60/240 min or to the 08:00 option expiry, and does a high-volatility regime or a recent jump change that? Discussions: `../../discussions/VWAP_Drift/`.

| Notebook | Content | Status |
|---|---|---|
| `01_vwap_drift.ipynb` | Perp-only conditional drift test: $Z_h=D/\sqrt{v_h}$ from frozen VD18 HAR_4; 144 real cells; DEV (< 2026-04-01) selection, TEST gates G1 hour-matched real − placebo, G2 > 7 bp; options gate 12 bp + spread/Δ; S1 high-vol hypothesis; hit rate vs driftless benchmark | Pre-registered (VWD 001); awaiting first run |

Code: `src/data_engineering/modeling/vwap_drift.py`. Tests: `tests/test_vwap_drift.py`. Outputs: `01/01_*`.

## Running 01 (PC, Windows CMD)

Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python tools\download_binance_um_klines.py --months 25 --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT
python -m unittest discover -s tests -p "test_vwap_drift.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\VWAP_Drift\01_vwap_drift.ipynb
git add notebooks\VWAP_Drift
git commit -m "VWAP Drift 01: run"
git push
```

Optional before the `jupyter` line: `set VWD01_ASSETS=BTC,ETH`.
