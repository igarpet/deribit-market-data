# VWAP Events discussions

Is a touch of an anchored VWAP (session or 4-hour) followed by an unusually violent move, beyond what touches of placebo levels produce? This tests Ivan's observation from discretionary MNQ trading on crypto before VWAP is used as a special level or regime feature. Notebooks: `../../notebooks/VWAP_Events/`.

Independent branch: Binance USD-M perpetual 1-minute klines (`tools/download_binance_um_klines.py`), no other branch's outputs.

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Data, anchors, touch detector, outcomes, offset and phase placebos, dev/test split, Bonferroni gates, interpretation table written before results |
| `002_claude_vwap01_readout.md` | VWAP 01 result: no anchor passes; real-anchor touches = random-anchor touches (ratio 0.995–1.007, CI within ±2%); offset placebo biased; v1.1 fixes -inf in secondary outputs |

Next turn in this branch: `003`.

Procedure and numbering rules: `../README.md`.
- `003_claude_ve02_descriptive_readout.md` — VE02 counts: ~41 session / ~95 4H crosses per day; median return after a cross 2–5 min; real anchors identical to placebo anchors.
- `004_claude_ve03_preregistration.md` — VE03 VWAP fade scalp: primary k=1, exit at VWAP touch, stop 20 bp; G1 hour-matched real − placebo, G2 gross > 7 bp.
- `005_claude_ve03_readout.md` — VE03 **NO_EDGE**: primary −0.16 bp [−0.43, +0.11]; whole grid −1.2…+0.8 bp, identical at placebo anchors; win rates follow S/(S+T). Branch closed.
