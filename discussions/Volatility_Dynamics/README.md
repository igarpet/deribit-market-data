# Volatility Dynamics — ChatGPT / Claude research dialogue

Research branch for volatility dynamics developed independently from every other research branch.

- **001**: `001_chatgpt_module01_design.md` — historical first design; **superseded** because it reused prior-branch HAR/volatility logic.
- **002**: `002_chatgpt_all_available_data.md` — historical data-window decision; superseded as part of the full reset.
- **003**: `003_chatgpt_full_independence_reset.md` — authoritative reset: return-only Module 01, no imported research outputs or assumptions.
- **004**: `004_chatgpt_timestamp_unit_bugfix.md` — Pandas datetime-unit bugfix for the standalone engine.
- **005**: `005_chatgpt_native_5m_module02.md` — native five-minute squared-return analysis; no rolling 30m block.
- **006**: `006_chatgpt_q99_precursor_module03.md` — full-schema q99 precursor discovery with strict pre-event information boundary and matched controls.
- **007**: `007_chatgpt_module03_2h_and_conditional_module04.md` — extends Module 03 anatomy to ±120m and adds native-volatility-conditioned q99 precursor Module 04.
- **Next turn:** `008`.

Module 01 remains historical baseline work. **Module 02 is authoritative for primitive volatility persistence/duration questions:** `notebooks/Volatility_Dynamics/02_native_5m_volatility.ipynb`.

Follow `../README.md`: one immutable discussion file per turn; evidence before interpretation.
