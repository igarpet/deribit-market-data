# Jump Fade discussions

Does a passive ±1.5σ bracket around the price at each 5-minute boundary, with a passive take-profit back at that price (plus a 3σ stop and a 60-minute time stop), earn more than the sign-flip null, and net of Deribit costs? Notebooks: `../../notebooks/Jump_Fade/`.

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Rule (1m simulation, conservative fill ordering), splits all / burst / recent_jump, sign-flip null (199), G1 reversion, G2 net (Deribit), grid descriptive |

| `002_claude_jf01_readout.md` | JF 01 (85a989b): NO_REVERSION in all splits (TEST gross −0.91 bp vs null −0.95; net −4.1 bp/trade at Deribit, −154 bp/day); recent_jump p 0.02 but gross < 0 on 4/4 assets; exploratory mirror (breakout) gross +0.92 = null, net −8.0 bp/trade |

Next turn in this branch: `003`.

Procedure and numbering rules: `../README.md`.
