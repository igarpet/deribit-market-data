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
