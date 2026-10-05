"""
Quote-based smile fit in standardized moneyness, and exit-liquidity counts.

One row per:
    state_id x option_family_id

Why this layer exists (Notebook 03, 2026-10-05):

  - The fixed log-moneyness grid looks available all session only because
    it falls back to Deribit's mark IV. In the last hours of a 0DTE session
    89-100% of core nodes have a mark-sourced side.
  - Near expiry, strikes are several expected moves apart, so two-point
    node interpolation manufactures curvature and noise.
  - Missing IV means missing market, not missing Deribit IV: fewer than
    0.3% of near-the-money contracts have a two-sided price without a
    Deribit bid/ask IV. Bids disappear near expiry while asks remain.

The fit therefore uses ONLY quote-sourced contracts (bid/ask-IV midpoint)
and is reported as missing when quotes cannot support it. Mark IV is never
used to fill the smile shape.

Model, per state and family:

    sigma(z) = a + b z + c z^2,
    z = ln(K / F) / (sigma_ATM * sqrt(tau)),

fitted by weighted least squares on OTM contracts with |z| <= FIT_Z_MAX,
weights w = 1 / max(IV spread, FIT_SPREAD_FLOOR_IV)^2. The normal
equations are solved in closed form (Cramer's rule). The z scale uses the
preferred-surface ATM IV of the same family (k = 0 node); it only sets the
unit of z.

Acceptance (fit_ok):
    n quotes >= FIT_MIN_QUOTES,
    min z <= -FIT_MIN_SIDE_Z and max z >= +FIT_MIN_SIDE_Z (both wings),
    normal-equation determinant not degenerate (scale-free test).

Exit-liquidity counts (all surface-eligible OTM contracts, quoted or not,
within |z| <= 1 and |z| <= 2): how many have a bid and how many an ask.
These are inputs for the later option-selection, valuation and backtest
layers, where a contract without a bid cannot be exited before settlement.

Units: IVs are in Deribit percent; IV_PERCENT_TO_DECIMAL converts for the
z scale only. All fit outputs (a, b, c, RMSE) stay in IV percent.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


SMILE_FIT_VERSION = "0.1"

FIT_Z_MAX = 2.0
FIT_MIN_QUOTES = 5
FIT_MIN_SIDE_Z = 0.5
FIT_SPREAD_FLOOR_IV = 0.5
FIT_MISSING_SPREAD_IV = 2.0
FIT_DET_REL_TOL = 1e-9

IV_PERCENT_TO_DECIMAL = 0.01
MINUTES_PER_YEAR_365 = 365.0 * 24.0 * 60.0


def create_quote_smile_fit_state(connection) -> None:

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW quote_smile_fit_state AS

        WITH scale AS (

            SELECT
                state_id,
                state_time,
                underlying,
                session_id,
                minutes_to_expiry,
                option_family_id,
                iv_atm_forward
                    AS scale_atm_iv,

                iv_atm_forward
                * {IV_PERCENT_TO_DECIMAL}
                * SQRT(
                    minutes_to_expiry
                    / {MINUTES_PER_YEAR_365}
                )
                    AS sigma_sqrt_t

            FROM option_smile_state

            WHERE
                surface_source = 'preferred'
                AND iv_atm_forward > 0
                AND minutes_to_expiry > 0
        ),

        points AS (

            SELECT
                n.state_id,
                n.option_family_id,
                n.preferred_iv_source,
                n.iv_mid,
                n.iv_spread,
                n.best_bid_price,
                n.best_ask_price,

                n.log_moneyness
                / s.sigma_sqrt_t
                    AS z

            FROM normalized_option_chain n

            JOIN scale s
                ON n.state_id = s.state_id
                AND n.option_family_id = s.option_family_id

            WHERE
                n.surface_base_eligible
                AND n.log_moneyness IS NOT NULL
        ),

        window_points AS (

            SELECT *
            FROM points
            WHERE ABS(z) <= {FIT_Z_MAX}
        ),

        liquidity AS (

            SELECT
                state_id,
                option_family_id,

                COUNT(*)
                    AS otm_contracts_z2,

                SUM(CASE WHEN best_bid_price > 0 THEN 1 ELSE 0 END)
                    AS otm_with_bid_z2,

                SUM(CASE WHEN best_ask_price > 0 THEN 1 ELSE 0 END)
                    AS otm_with_ask_z2,

                SUM(CASE WHEN ABS(z) <= 1.0 THEN 1 ELSE 0 END)
                    AS otm_contracts_z1,

                SUM(
                    CASE
                        WHEN ABS(z) <= 1.0 AND best_bid_price > 0
                        THEN 1 ELSE 0
                    END
                )
                    AS otm_with_bid_z1,

                SUM(
                    CASE
                        WHEN ABS(z) <= 1.0 AND best_ask_price > 0
                        THEN 1 ELSE 0
                    END
                )
                    AS otm_with_ask_z1

            FROM window_points

            GROUP BY
                state_id,
                option_family_id
        ),

        quotes AS (

            SELECT
                *,

                1.0
                / POWER(
                    GREATEST(
                        COALESCE(
                            iv_spread,
                            {FIT_MISSING_SPREAD_IV}
                        ),
                        {FIT_SPREAD_FLOOR_IV}
                    ),
                    2
                )
                    AS w

            FROM window_points

            WHERE
                preferred_iv_source = 'bid_ask_iv_mid'
                AND iv_mid > 0
        ),

        sums AS (

            SELECT
                state_id,
                option_family_id,

                COUNT(*) AS n_quotes,
                SUM(CASE WHEN ABS(z) <= 1.0 THEN 1 ELSE 0 END)
                    AS n_quotes_z1,
                MIN(z) AS z_min,
                MAX(z) AS z_max,

                SUM(w) AS s0,
                SUM(w * z) AS s1,
                SUM(w * z * z) AS s2,
                SUM(w * z * z * z) AS s3,
                SUM(w * z * z * z * z) AS s4,

                SUM(w * iv_mid) AS t0,
                SUM(w * iv_mid * z) AS t1,
                SUM(w * iv_mid * z * z) AS t2

            FROM quotes

            GROUP BY
                state_id,
                option_family_id
        ),

        determinants AS (

            SELECT
                *,

                s0 * (s2 * s4 - s3 * s3)
                - s1 * (s1 * s4 - s3 * s2)
                + s2 * (s1 * s3 - s2 * s2)
                    AS det_m,

                t0 * (s2 * s4 - s3 * s3)
                - s1 * (t1 * s4 - s3 * t2)
                + s2 * (t1 * s3 - s2 * t2)
                    AS det_a,

                s0 * (t1 * s4 - s3 * t2)
                - t0 * (s1 * s4 - s3 * s2)
                + s2 * (s1 * t2 - t1 * s2)
                    AS det_b,

                s0 * (s2 * t2 - t1 * s3)
                - s1 * (s1 * t2 - t1 * s2)
                + t0 * (s1 * s3 - s2 * s2)
                    AS det_c

            FROM sums
        ),

        solved AS (

            SELECT
                state_id,
                option_family_id,
                n_quotes,
                n_quotes_z1,
                z_min,
                z_max,

                (
                    n_quotes >= {FIT_MIN_QUOTES}
                    AND z_min <= -{FIT_MIN_SIDE_Z}
                    AND z_max >= {FIT_MIN_SIDE_Z}
                    AND ABS(det_m)
                        > {FIT_DET_REL_TOL} * POWER(s0, 3)
                )
                    AS fit_ok,

                det_a / NULLIF(det_m, 0) AS a_raw,
                det_b / NULLIF(det_m, 0) AS b_raw,
                det_c / NULLIF(det_m, 0) AS c_raw

            FROM determinants
        ),

        residuals AS (

            SELECT
                q.state_id,
                q.option_family_id,

                SQRT(
                    SUM(
                        q.w
                        * POWER(
                            q.iv_mid
                            - (
                                s.a_raw
                                + s.b_raw * q.z
                                + s.c_raw * q.z * q.z
                            ),
                            2
                        )
                    )
                    / SUM(q.w)
                )
                    AS fit_wrmse

            FROM quotes q

            JOIN solved s
                ON q.state_id = s.state_id
                AND q.option_family_id = s.option_family_id

            WHERE s.fit_ok

            GROUP BY
                q.state_id,
                q.option_family_id
        )

        SELECT
            sc.state_id,
            sc.state_time,
            sc.underlying,
            sc.session_id,
            sc.minutes_to_expiry,
            sc.option_family_id,
            sc.scale_atm_iv,
            sc.sigma_sqrt_t,

            COALESCE(so.fit_ok, FALSE)
                AS fit_ok,

            COALESCE(so.n_quotes, 0)
                AS fit_n_quotes,

            COALESCE(so.n_quotes_z1, 0)
                AS fit_n_quotes_z1,

            so.z_min AS fit_z_min,
            so.z_max AS fit_z_max,

            CASE WHEN so.fit_ok THEN so.a_raw END
                AS fit_atm_iv,

            CASE WHEN so.fit_ok THEN so.b_raw END
                AS fit_skew,

            CASE WHEN so.fit_ok THEN so.c_raw END
                AS fit_curvature,

            CASE
                WHEN so.fit_ok AND so.a_raw > 0
                THEN so.b_raw / so.a_raw
            END
                AS fit_skew_rel,

            CASE
                WHEN so.fit_ok AND so.a_raw > 0
                THEN so.c_raw / so.a_raw
            END
                AS fit_curvature_rel,

            r.fit_wrmse,

            COALESCE(l.otm_contracts_z1, 0) AS otm_contracts_z1,
            COALESCE(l.otm_with_bid_z1, 0) AS otm_with_bid_z1,
            COALESCE(l.otm_with_ask_z1, 0) AS otm_with_ask_z1,
            COALESCE(l.otm_contracts_z2, 0) AS otm_contracts_z2,
            COALESCE(l.otm_with_bid_z2, 0) AS otm_with_bid_z2,
            COALESCE(l.otm_with_ask_z2, 0) AS otm_with_ask_z2,

            CASE
                WHEN l.otm_contracts_z2 > 0
                THEN l.otm_with_bid_z2::DOUBLE / l.otm_contracts_z2
            END
                AS otm_bid_share_z2

        FROM scale sc

        LEFT JOIN solved so
            ON sc.state_id = so.state_id
            AND sc.option_family_id = so.option_family_id

        LEFT JOIN residuals r
            ON sc.state_id = r.state_id
            AND sc.option_family_id = r.option_family_id

        LEFT JOIN liquidity l
            ON sc.state_id = l.state_id
            AND sc.option_family_id = l.option_family_id
        """
    )


def validate_quote_smile_fit(connection) -> None:

    row = connection.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(*) - COUNT(DISTINCT state_id || '#' || option_family_id)
                AS duplicates,
            SUM(CASE WHEN fit_ok THEN 1 ELSE 0 END)
                AS accepted,
            SUM(
                CASE
                    WHEN fit_ok
                    AND (
                        fit_n_quotes < {FIT_MIN_QUOTES}
                        OR fit_atm_iv IS NULL
                        OR fit_skew IS NULL
                        OR fit_curvature IS NULL
                    )
                    THEN 1 ELSE 0
                END
            )
                AS inconsistent_accepted,
            SUM(
                CASE
                    WHEN fit_ok AND fit_atm_iv <= 0
                    THEN 1 ELSE 0
                END
            )
                AS non_positive_atm,
            SUM(
                CASE
                    WHEN otm_with_bid_z2 > otm_contracts_z2
                    OR otm_with_ask_z2 > otm_contracts_z2
                    THEN 1 ELSE 0
                END
            )
                AS invalid_liquidity_counts
        FROM quote_smile_fit_state
        """
    ).fetchone()

    print("\nQUOTE SMILE FIT VALIDATION")
    print("=" * 80)
    print(f"Family-states:                 {row[0]:,}")
    print(f"Duplicate family-states:       {row[1]:,}")
    print(f"Accepted fits:                 {int(row[2] or 0):,}")
    print(f"Inconsistent accepted fits:    {int(row[3] or 0):,}")
    print(f"Accepted fits with ATM <= 0:   {int(row[4] or 0):,}")
    print(f"Invalid liquidity counts:      {int(row[5] or 0):,}")

    if row[1]:
        raise RuntimeError("quote_smile_fit_state has duplicate family-states.")
    if row[3]:
        raise RuntimeError("quote_smile_fit_state has accepted fits with missing parameters.")
    if row[5]:
        raise RuntimeError("quote_smile_fit_state has bid/ask counts above contract counts.")
    # A non-positive fitted ATM level is reported, not fatal: it is a
    # measurement outcome that downstream quality filters must handle.
