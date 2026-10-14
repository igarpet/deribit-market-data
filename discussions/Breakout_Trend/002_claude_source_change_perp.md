# BT 002 — Claude: data source changed to Binance USD-M perpetual (before any real run)

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-14 |
| Branch | Breakout Trend |
| Responds to | BT 001 §2 (data); Ivan (chat): "I don't care about AVAX, TRX or HYPE, let's do it with the perp data I already have" |
| Status | Amendment to the pre-registration, made before any real-data BT 01 output |

## 1. Change

| | BT 001 (registered) | BT 002 (now primary) |
|---|---|---|
| Source | Binance **spot** 5m OHLC (`history/ohlc_5m`); HYPE from Deribit perp | Binance **USD-M perpetual** 1m klines (`<parent of DERIBIT_DATA_ROOT>/Binance/<ASSET>USDT_um_1m.parquet`), aggregated to 5m OHLC by `bt.ohlc5_from_um_1m` |
| Assets | BTC, ETH, SOL, XRP, AVAX, TRX, HYPE | **BTC, ETH, SOL, XRP** |
| Start | 2024-08-25 (100 h warm-up before dev start 2024-09-01) | 2024-10-01. Development effectively starts about 2024-10-05 after warm-up. |

## 2. Unchanged

- Strategy rules and parameters.
- Costs: perp taker 0.05% + 1 bp per side; funding ignored.
- Options leg.
- Windows: development to 2025-12-31, test from 2026-01-01.
- 200-run coin-flip null and the pooled test-window decision rule.

## 3. Construction

- 5-minute bar = minutes `[t, t+5)` of the 1m klines, labelled by its END time (as `download_ohlc_history.py`).
- A bar is kept only if all five minutes are present.
- Test: `TestUmSource.test_ohlc5_from_um_1m`.

## 4. Effect on interpretation

- Perpetual prices are closer to the instrument the strategy would trade, which is a slight improvement for the perp leg.
- The development window loses about one month.
- Fewer assets means the pooled test decision rests on 4 assets instead of 7.
- The spot path is kept (`BT01_SOURCE=ohlc_5m`) if Ivan later wants the registered version.
