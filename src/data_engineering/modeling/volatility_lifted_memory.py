"""Volatility Dynamics 11: causal empirical finite-exponential volatility-memory study.

This is NOT a calibration of latent lifted-Heston stochastic variance factors.
Only native observed 5m returns are used; no other research-branch artifacts.
No repository database is ever mutated by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gamma, log
import numpy as np
import pandas as pd
from scipy.special import expit

BAR_MIN = 5
HORIZONS = (30, 60, 90, 120)
PRIMARY_HORIZON = 60
H_GRID = (0.05, 0.10, 0.20, 0.30, 0.40)
N_GRID = (4, 6, 8)
RIDGE_GRID = (1e-4, 1e-3, 1e-2)
MIN_MEMORY_BARS = 288
LAG_BARS = (1, 2, 6, 12, 24)


def dense_panel(raw: pd.DataFrame, *, min_days: int = 12) -> pd.DataFrame:
    """An explicit 5m calendar per asset; a missing bar is NaN, not bridged."""
    req = {'state_time', 'underlying', 'return_5m'}
    if req - set(raw):
        raise ValueError(f'Missing source fields: {sorted(req - set(raw))}')
    f = raw[list(req)].copy()
    f['state_time'] = pd.to_datetime(f.state_time, utc=True)
    f['underlying'] = f.underlying.astype(str)
    f['return_5m'] = pd.to_numeric(f.return_5m, errors='coerce')
    if f[['underlying', 'state_time']].duplicated().any():
        raise ValueError('Duplicate underlying x state_time')
    if (f.state_time.dt.minute % 5).any() or (f.state_time.dt.second != 0).any():
        raise ValueError('Timestamp off 5m grid')
    pieces = []
    for asset, a in f.groupby('underlying', sort=True):
        a = a.set_index('state_time').sort_index()
        idx = pd.date_range(a.index.min(), a.index.max(), freq='5min', tz='UTC')
        a = a.reindex(idx)
        a['underlying'] = asset
        a.index.name = 'state_time'
        pieces.append(a.reset_index())
    out = pd.concat(pieces, ignore_index=True)
    out['date'] = out.state_time.dt.floor('D')
    dates = np.sort(out.date.unique())
    if len(dates) < min_days:
        raise ValueError(f'Need >= {min_days} UTC calendar days, got {len(dates)}')
    t = max(1, int(np.floor(.60 * len(dates))))
    v = max(t + 1, int(np.floor(.80 * len(dates))))
    if v >= len(dates):
        raise ValueError('Not enough calendar days for fit/validation/test')
    out['split'] = np.where(out.date < dates[t], 'train',
                    np.where(out.date < dates[v], 'valid', 'test'))
    out['hour'] = out.state_time.dt.hour.astype(int)
    out['slot'] = (out.state_time.dt.hour * 12 + out.state_time.dt.minute // 5).astype(int)
    out['v'] = out.return_5m.pow(2)
    return out


def training_clock(panel: pd.DataFrame, min_n_hour: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn epsilon and deterministic hourly medians on training history only."""
    p = panel.copy()
    rows = []
    for asset, a in p.groupby('underlying', sort=True):
        tr = a[a.split == 'train']
        positive = tr.loc[np.isfinite(tr.v) & (tr.v > 0), 'v'].to_numpy(float)
        if len(positive) < 100:
            raise ValueError(f'Not enough positive native variance in train: {asset}')
        eps = float(np.median(positive) * 1e-3)
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError(f'Invalid variance epsilon: {asset}')
        local = np.log(tr.v.to_numpy(float) + eps)
        global_med = float(np.nanmedian(local))
        for hour in range(24):
            vh = local[tr.hour.to_numpy() == hour]
            vh = vh[np.isfinite(vh)]
            val = float(np.median(vh)) if len(vh) >= min_n_hour else global_med
            rows.append({'underlying': asset, 'hour': hour, 'epsilon': eps,
                         'clock_center': val, 'n_train_hour': len(vh),
                         'clock_source': 'hour' if len(vh) >= min_n_hour else 'asset'})
    par = pd.DataFrame(rows)
    p = p.merge(par[['underlying', 'hour', 'epsilon', 'clock_center']],
                on=['underlying', 'hour'], how='left', validate='many_to_one', sort=False)
    p['logv'] = np.log(p.v + p.epsilon)
    p['y'] = p.logv - p.clock_center
    return p, par


def regularity_mask(p: pd.DataFrame, min_history: int = MIN_MEMORY_BARS) -> np.ndarray:
    """True only after min_history consecutively observed bars INCLUDING current bar."""
    good = np.zeros(len(p), dtype=bool)
    for _, a in p.groupby('underlying', sort=False):
        ix = a.index.to_numpy()
        run = 0
        for k, z in enumerate(a.y.to_numpy(float)):
            run = run + 1 if np.isfinite(z) else 0
            good[ix[k]] = run >= min_history
    return good


def fixed_lags(p: pd.DataFrame) -> pd.DataFrame:
    out = p[['y']].rename(columns={'y': 'now'}).copy()
    g = p.groupby('underlying', sort=False)['y']
    for k in LAG_BARS:
        out[f'lag_{k * 5}m'] = g.shift(k)
    return out


def ema_prior(p: pd.DataFrame, half_lives_min: np.ndarray) -> np.ndarray:
    """Only y_{t-1} and earlier enter the state at the end of bar t.

    Returns NaN on the first bar of a new contiguous finite stretch.
    Missing values reset *all* factors. No direct lookahead or future imputation.
    """
    tau = np.asarray(half_lives_min, dtype=float)
    if tau.ndim != 1 or len(tau) < 1 or np.any(tau <= 0):
        raise ValueError('Positive one-dimensional half lives required')
    decay = np.exp(-np.log(2.0) * BAR_MIN / tau)
    out = np.full((len(p), len(tau)), np.nan)
    for _, a in p.groupby('underlying', sort=False):
        state = None
        prev = np.nan
        for row_id, cur in zip(a.index.to_numpy(), a.y.to_numpy(float)):
            if np.isfinite(cur):
                if np.isfinite(prev):
                    if state is None:
                        state = np.full(len(tau), prev)
                    else:
                        state = decay * state + (1.0 - decay) * prev
                    out[row_id] = state
                else:
                    state = None
            else:
                state = None
            prev = cur
    return out


def quadrature(H: float, N: int, min_half_life=5.0, max_half_life=1440.0):
    """Positive geometric rate-cell quadrature of a fractional power kernel.

    Fixed integration interval across N permits a meaningful approximation-error
    comparison as N varies; each rate is the geometric midpoint of a cell.
    Extremely fast and slow kernel tails remain truncated and are reported.
    """
    if not (0 < H < .5) or N < 2:
        raise ValueError('0 < H < 0.5 and N >= 2 required')
    edges = np.geomspace(log(2)/min_half_life, log(2)/max_half_life, N+1)
    rates = np.sqrt(edges[:-1] * edges[1:])
    alpha = H + .5
    exponent = 1 - alpha
    weights = (edges[:-1]**exponent - edges[1:]**exponent) / (
        exponent * gamma(alpha) * gamma(1-alpha))
    return rates, weights


def kernel_error(H: float, rates: np.ndarray, weights: np.ndarray):
    time = np.geomspace(5, 1440, 120)
    alpha = H + .5
    true = time ** (alpha - 1) / gamma(alpha)
    approx = np.exp(-np.outer(time, rates)) @ weights
    # Scale-free error: shape approximation in finite range, not error in amplitude.
    norm = float(np.dot(true, approx) / np.dot(approx, approx))
    err = float(np.sqrt(np.mean(((norm*approx - true)/true)**2)))
    return err, time, true, norm*approx


def lifted_features(p: pd.DataFrame, H: float, N: int, *, precomputed=None) -> tuple[pd.DataFrame, dict]:
    rates, weights = quadrature(H, N)
    halves = log(2) / rates
    em = ema_prior(p, halves) if precomputed is None else precomputed
    if em.shape != (len(p), N):
        raise ValueError('precomputed EMA shape mismatch')
    # Constant-bar input integrated against each exponential: mass w_j / x_j.
    masses = weights / rates
    masses = masses / masses.sum()
    agg = em @ masses
    fast = em[:, :max(1, N//3)].mean(axis=1)
    slow = em[:, -max(1, N//3):].mean(axis=1)
    gap = fast - slow
    err, *_ = kernel_error(H, rates, weights)
    return pd.DataFrame({'now': p.y.to_numpy(float), 'kernel_memory': agg,
                         'fast_minus_slow': gap}, index=p.index), {
                             'H': H, 'N': N, 'half_lives_min': halves.tolist(),
                             'mean_reversion_rates_per_min': rates.tolist(),
                             'quadrature_weights': weights.tolist(),
                             'memory_mass_weights': masses.tolist(),
                             'kernel_relative_rms_error': err}


def equal_exponential_features(p: pd.DataFrame, N: int, *, precomputed=None) -> pd.DataFrame:
    """Non-rough matched-scale null: equal weights across the SAME factor grid.

    Rate support is identical to the rough lift for the chosen N. The rough lift
    must beat this baseline before a kernel-specific interpretation is warranted.
    """
    rates, _ = quadrature(.10, N)
    em = ema_prior(p, log(2)/rates) if precomputed is None else precomputed
    if em.shape != (len(p), N):
        raise ValueError('precomputed EMA shape mismatch')
    fast = em[:, :max(1,N//3)].mean(axis=1)
    slow = em[:, -max(1,N//3):].mean(axis=1)
    return pd.DataFrame({'now': p.y.to_numpy(float),
                         'equal_memory': em.mean(axis=1),
                         'fast_minus_slow': fast - slow}, index=p.index)


def simple_exponential_features(p: pd.DataFrame) -> pd.DataFrame:
    e = ema_prior(p, np.asarray([30., 240.]))
    return pd.DataFrame({'now': p.y.to_numpy(float), 'ema_30m': e[:, 0],
                         'ema_240m': e[:, 1]}, index=p.index)


def q99_labels(p: pd.DataFrame, horizons=HORIZONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-asset TRAIN-only threshold and exact-future, uncensored q99 onset labels.

    A label is valid only if all bars in (t,t+h] have finite observations,
    and the entire future horizon belongs to the SAME chrono split.
    """
    out = p[['underlying', 'state_time', 'date', 'split']].copy()
    thresholds = {}
    for a, df in p.groupby('underlying'):
        x = df.loc[(df.split == 'train') & np.isfinite(df.v), 'v']
        if len(x) < 200:
            raise ValueError(f'Insufficient training variance: {a}')
        thresholds[a] = float(np.quantile(x, .99))
    out['q99'] = np.zeros(len(p), dtype=bool)
    out['onset'] = np.zeros(len(p), dtype=bool)
    out['eligible_now'] = False
    for h in horizons:
        out[f'y_{h}m'] = np.nan
    for asset, df in p.groupby('underlying', sort=False):
        idx = df.index.to_numpy()
        v = df.v.to_numpy(float)
        good = np.isfinite(v)
        q = good & (v >= thresholds[asset])
        prev = np.r_[False, q[:-1]]
        onset = q & ~prev
        out.loc[idx, 'q99'] = q
        out.loc[idx, 'onset'] = onset
        out.loc[idx, 'eligible_now'] = good & ~q
        n = len(df)
        cgood = np.r_[0, np.cumsum(good)]
        cevent = np.r_[0, np.cumsum(onset)]
        split = df.split.to_numpy()
        for h in horizons:
            k = h//5
            origin = np.arange(n)
            endpoint = origin + k
            in_bounds = endpoint < n
            end = np.minimum(endpoint, n-1)
            future_complete = (cgood[end + 1]-cgood[origin + 1] == k)
            split_ok = split[end] == split[origin]
            valid = in_bounds & good & future_complete & split_ok
            yy = (cevent[end+1]-cevent[origin+1] > 0).astype(float)
            yy[~valid] = np.nan
            out.loc[idx, f'y_{h}m'] = yy
    return out, pd.DataFrame({'underlying': list(thresholds),
                              'train_q99_threshold_v': list(thresholds.values())})


def design(train: pd.DataFrame, other: pd.DataFrame, feature_names: list[str]):
    """One-hot asset fixed effects + harmonics + TRAIN-standardized inputs."""
    assets = sorted(train.underlying.unique())
    ztr = train[feature_names].to_numpy(float)
    mu = ztr.mean(axis=0)
    sd = ztr.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.)

    def build(frame):
        xs = [np.ones(len(frame))]
        for a in assets[1:]:
            xs.append((frame.underlying.to_numpy() == a).astype(float))
        slot = frame.slot.to_numpy(float)
        xs.extend([np.sin(2*np.pi*slot/288), np.cos(2*np.pi*slot/288)])
        arr = (frame[feature_names].to_numpy(float)-mu)/sd
        xs.extend([arr[:,j] for j in range(arr.shape[1])])
        return np.column_stack(xs)
    return build(train), build(other)


def ridge_logit(X, y, strength=1e-3, max_iter=40):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError('Nonfinite training data in regression')
    n, d = X.shape
    if len(np.unique(y)) < 2:
        raise ValueError('Need both binary classes in training')
    beta = np.zeros(d)
    prob = np.clip(y.mean(), 1e-5, 1-1e-5)
    beta[0] = log(prob/(1-prob))
    ridge = strength * np.eye(d)
    ridge[0,0] = 0
    for _ in range(max_iter):
        pred = expit(np.clip(X@beta, -30, 30))
        w = np.maximum(pred*(1-pred), 1e-7)
        grad = X.T@(pred-y)/n + ridge@beta
        hess = (X.T*w)@X/n + ridge + 1e-9*np.eye(d)
        step = np.linalg.solve(hess, grad)
        step = np.clip(step, -3, 3)
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


def score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1-1e-6)
    ll = -(y*np.log(pred)+(1-y)*np.log1p(-pred))
    return {'logloss': float(ll.mean()), 'brier': float(np.mean((pred-y)**2)),
            'positive_origin_count': int(y.sum()), 'n_origins': len(y)}, ll


def fit_predict(frame: pd.DataFrame, names: list[str], ridge: float, h: int,
                train_split='train', eval_split='valid'):
    col = f'y_{h}m'
    must = names + [col]
    eligible = frame.eligible_now.to_numpy(dtype=bool) & np.isfinite(frame[must].to_numpy(float)).all(axis=1)
    tr = frame[(frame.split == train_split) & eligible]
    te = frame[(frame.split == eval_split) & eligible]
    if len(tr) < 200 or len(te) < 30:
        raise ValueError('Insufficient complete common observations')
    Xtr, Xte = design(tr, te, names)
    beta = ridge_logit(Xtr, tr[col].to_numpy(float), ridge)
    preds = expit(np.clip(Xte@beta, -30, 30))
    stats, ll = score(te[col], preds)
    return stats, pd.DataFrame({'underlying':te.underlying.to_numpy(),
                                'date':te.date.to_numpy(),
                                'state_time':te.state_time.to_numpy(),
                                'y':te[col].to_numpy(float),
                                'probability':preds, 'loss':ll}, index=te.index)


def bootstrap_day_diff(compare: pd.DataFrame, B=500, seed=1105):
    """Paired all-asset UTC-day bootstrap. Negative means left model is better."""
    delta = (compare.loss_left-compare.loss_right).to_numpy(float)
    days = compare.date.to_numpy()
    ud, inverse = np.unique(days, return_inverse=True)
    sums = np.bincount(inverse, weights=delta, minlength=len(ud))
    counts = np.bincount(inverse, minlength=len(ud))
    rng = np.random.default_rng(seed)
    pick = rng.integers(len(ud), size=(B,len(ud)))
    samples = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return {'delta_logloss': float(delta.mean()),
            'bootstrap_q025': float(np.quantile(samples,.025)),
            'bootstrap_q975': float(np.quantile(samples,.975)),
            'test_day_count': int(len(ud))}


def align_comparison(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Restrict ALL models to identical test indices; preserve paired loss."""
    common = left.index.intersection(right.index)
    a, b = left.loc[common], right.loc[common]
    if len(a) == 0 or not np.array_equal(a.y.to_numpy(), b.y.to_numpy()):
        raise ValueError('Prediction windows or outcomes do not align')
    return pd.DataFrame({'date': a.date.to_numpy(), 'loss_left': a.loss.to_numpy(),
                         'loss_right': b.loss.to_numpy()}, index=common)


def event_factor_path(p: pd.DataFrame, labels: pd.DataFrame,
                      factor: pd.Series, window_bars=24) -> pd.DataFrame:
    """Descriptive test-onset alignment, strictly -120m through -5m."""
    lines=[]
    for asset, aa in p.groupby('underlying', sort=False):
        rows=aa.index.to_numpy()
        ev=rows[(labels.loc[rows,'split']=='test').to_numpy() &
                labels.loc[rows,'onset'].to_numpy(bool)]
        for row in ev:
            j=int(np.searchsorted(rows,row))
            prior=rows[max(0,j-window_bars):j]
            if len(prior)!=window_bars or not np.all(np.isfinite(factor.loc[prior])):
                continue
            for m, r in enumerate(prior):
                lines.append({'event_time':p.loc[row,'state_time'],
                              'underlying':asset,'relative_min':-5*(window_bars-m),
                              'factor':float(factor.loc[r])})
    if not lines:
        return pd.DataFrame(columns=['relative_min','mean_factor','n_event_instances','n_unique_event_times'])
    tab = pd.DataFrame(lines)
    # Every unique timestamp receives equal total weight across synchronous assets.
    tab['weight'] = 1/tab.groupby(['event_time','relative_min'])['underlying'].transform('count')
    return tab.groupby('relative_min',as_index=False).apply(
        lambda v: pd.Series({'mean_factor':np.average(v.factor,weights=v.weight),
                             'n_event_instances':v[['event_time','underlying']].drop_duplicates().shape[0],
                             'n_unique_event_times':v.event_time.nunique()}),
        include_groups=False).reset_index(drop=True).assign(
            relative_min=sorted(tab.relative_min.unique()))
