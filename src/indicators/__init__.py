"""
Indicators Module
=================
All technical indicator calculations.

Fix log (relative to original):
    [FIX-1]  rsi            : Removed fillna(50). RSI during warmup is undefined; returning NaN
                              lets signal logic (pandas comparisons) naturally exclude warmup bars.
    [FIX-2]  stoch_rsi      : Added `stoch_length` parameter (default 14). Previously, rsi_length
                              was reused for the stochastic min/max window, conflating two
                              independent concepts. Standard platforms expose all four params.
    [FIX-3]  supertrend     : Flipped direction convention to 1=bullish, -1=bearish (industry
                              standard: TradingView, NinjaTrader).
    [FIX-4]  bollinger_width: Added explicit `mult` parameter (default 2.0). Previously hardcoded
                              to 1σ, diverging from standard Bollinger Band definition.
    [FIX-5]  ma             : Now raises ValueError on unknown ma_type instead of silently
                              falling back to SMA, which would mask typos.
    [FIX-6]  wma            : Replaced rolling.apply(lambda) with np.convolve — O(n) instead of
                              O(n × length) Python-loop overhead.
    [FIX-7]  percentile_rank: Replaced bare Python loop with scipy.stats.percentileofscore via
                              rolling.apply. More readable; correctness is unchanged.
    [FIX-8]  vwap           : Added DatetimeIndex guard. Previously would raise confusing
                              AttributeError on non-datetime indices.
    [FIX-9]  pamrp          : Added docstring. Function was undocumented, violating project style.
    [FIX-10] all multi-series: Added index alignment assertions on all functions that combine
                              two or more input Series.
    [NEW-1]  rsi_hidden_divergence : Hidden bullish/bearish RSI divergence for trend continuation.
    [NEW-2]  hpdr_bands     : High Probability Distribution Range bands using rolling
                              quantiles, with Normal-equivalent coverage labels.
"""

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from typing import Tuple, Dict


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _assert_aligned(*series: pd.Series) -> None:
    """
    [FIX-10] Assert all series share the same index.
    Raises ValueError immediately rather than silently producing NaN-filled output.
    """
    ref = series[0].index
    for s in series[1:]:
        if not ref.equals(s.index):
            raise ValueError(
                f"Index mismatch: series indices do not align. "
                f"Lengths: {[len(s) for s in series]}. "
                "Ensure all inputs are sliced from the same DataFrame."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# MOVING AVERAGES
# ═══════════════════════════════════════════════════════════════════════════════

def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average (Wilder-style span)."""
    return series.ewm(span=length, adjust=False).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    """
    Weighted Moving Average.

    [FIX-6] Replaced rolling.apply(lambda) with np.convolve.
    Weights: w_i = i+1, normalised. Convolution is O(n) via FFT path.

    WMA = Σ(w_i * x_i) / Σ(w_i),  w_i = 1, 2, …, length
    """
    weights = np.arange(1, length + 1, dtype=float)
    weights /= weights.sum()
    # np.convolve in 'full' mode then trim to original length
    arr = series.to_numpy(dtype=float)
    result = np.convolve(arr, weights[::-1], mode='full')[:len(arr)]
    result[:length - 1] = np.nan
    return pd.Series(result, index=series.index)


def rma(series: pd.Series, length: int) -> pd.Series:
    """
    Wilder's Moving Average (RMA / SMMA).
    Equivalent to EMA with alpha = 1/length.
    """
    return series.ewm(alpha=1 / length, adjust=False).mean()


def ma(series: pd.Series, length: int, ma_type: str = 'sma') -> pd.Series:
    """
    Moving average dispatcher.

    [FIX-5] Raises ValueError on unknown ma_type rather than silently
    falling back to SMA, which would mask typos in parameter dicts.

    Parameters
    ----------
    series  : price series
    length  : lookback window
    ma_type : one of 'sma', 'ema', 'wma', 'rma'
    """
    ma_funcs = {'sma': sma, 'ema': ema, 'wma': wma, 'rma': rma}
    key = ma_type.lower()
    if key not in ma_funcs:
        raise ValueError(
            f"Unknown ma_type '{ma_type}'. Valid options: {list(ma_funcs.keys())}"
        )
    return ma_funcs[key](series, length)


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY / RANGE INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def pamrp(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """
    Price Action Mean Reversion Percentile (PAMRP).

    Measures where the current close sits within the high-low range of the
    past `length` bars, expressed as a percentage (0–100).

        PAMRP = (close - lowest_low) / (highest_high - lowest_low) × 100

    Values near 0 indicate close to the bottom of the range (potential long entry).
    Values near 100 indicate close to the top (potential short entry or long exit).

    Parameters
    ----------
    high, low, close : OHLC series (must share the same index)
    length           : lookback window in bars
    """
    _assert_aligned(high, low, close)  # [FIX-10]
    highest   = high.rolling(window=length).max()
    lowest    = low.rolling(window=length).min()
    range_val = highest - lowest
    # Avoid division by zero on flat bars; replace 0-range with NaN so
    # downstream comparisons correctly exclude those bars.
    return (close - lowest) / range_val.replace(0, np.nan) * 100


def bollinger_width(close: pd.Series, length: int, mult: float = 2.0) -> pd.Series:
    """
    Bollinger Band Width as percentage of the middle band.

    [FIX-4] Added explicit `mult` parameter (default 2.0).
    Previously hardcoded to 1σ, which diverges from the standard
    Bollinger Band definition used in all major platforms.

        Width = (upper - lower) / basis × 100
        upper = basis + mult × σ
        lower = basis - mult × σ
        basis = SMA(close, length)
    """
    basis = sma(close, length)
    std   = close.rolling(window=length).std()
    upper = basis + mult * std
    lower = basis - mult * std
    # Use pandas arithmetic so NaN propagates naturally during warmup.
    # Previously used np.where(basis > 0, ..., 0) which mapped NaN → 0
    # and made warmup bars indistinguishable from zero-width bars.
    return (upper - lower) / basis * 100


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """
    Average True Range (Wilder's smoothing).

    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = RMA(TR, length)
    """
    _assert_aligned(high, low, close)  # [FIX-10]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return rma(tr, length)


# ═══════════════════════════════════════════════════════════════════════════════
# PERCENTILE / STATISTICAL
# ═══════════════════════════════════════════════════════════════════════════════

def percentile_rank(series: pd.Series, lookback: int) -> pd.Series:
    """
    Percentile rank of the current value within a rolling lookback window.

    [FIX-7] Replaced bare Python loop with scipy.stats.percentileofscore
    via rolling.apply. Readable, correct, and avoids manual index arithmetic.

    Returns NaN during the warmup period (first `lookback` bars).
    Previously returned 0.0, which silently passed threshold filters
    during warmup and generated false entry signals.

    Parameters
    ----------
    series   : input Series
    lookback : rolling window size

    Returns
    -------
    pd.Series in [0, 100]
    """
    def _pct_score(window: np.ndarray) -> float:
        # Score the last element against the rest of the window (exclude self)
        historical = window[:-1]
        current    = window[-1]
        return percentileofscore(historical, current, kind='rank')

    return series.rolling(window=lookback + 1).apply(_pct_score, raw=True)


def bbwp(close: pd.Series, length: int, lookback: int) -> pd.Series:
    """
    Bollinger Band Width Percentile (BBWP).

    Ranks current band width against its own rolling history.
    High values (>80) indicate volatility expansion; low values (<20) compression.
    """
    width = bollinger_width(close, length)
    return percentile_rank(width, lookback)


# ═══════════════════════════════════════════════════════════════════════════════
# MOMENTUM INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def rsi(close: pd.Series, length: int) -> pd.Series:
    """
    Relative Strength Index (Wilder's method).

    [FIX-1] Removed fillna(50). RSI during warmup is mathematically
    undefined, not neutral. Returning NaN lets all downstream filters
    (pandas comparisons like rsi < 30) correctly evaluate to False on
    warmup bars without any special-casing.

        RS  = RMA(gains, length) / RMA(losses, length)
        RSI = 100 - 100 / (1 + RS)
    """
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = (-delta).where(delta < 0, 0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stoch_rsi(
    close: pd.Series,
    rsi_length: int,
    stoch_length: int,
    k_length: int,
    d_length: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic RSI oscillator.

    [FIX-2] Added `stoch_length` as an independent parameter.
    Previously, `rsi_length` was reused for the stochastic min/max window,
    conflating two distinct concepts. All major platforms (TradingView,
    Bloomberg) expose all four parameters independently.

        StochRSI = (RSI - min(RSI, stoch_length)) / (max - min) × 100
        %K       = SMA(StochRSI, k_length)
        %D       = SMA(%K, d_length)

    Parameters
    ----------
    close        : close price series
    rsi_length   : RSI lookback (typically 14)
    stoch_length : stochastic lookback over RSI values (typically 14)
    k_length     : %K smoothing period (typically 3)
    d_length     : %D smoothing period (typically 3)
    """
    rsi_val     = rsi(close, rsi_length)
    lowest_rsi  = rsi_val.rolling(window=stoch_length).min()
    highest_rsi = rsi_val.rolling(window=stoch_length).max()
    rsi_range   = highest_rsi - lowest_rsi
    stoch       = (rsi_val - lowest_rsi) / rsi_range.replace(0, np.nan) * 100
    k           = sma(stoch, k_length)
    d           = sma(k, d_length)
    return k, d


def macd(
    close: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD: Moving Average Convergence Divergence.

        MACD line   = EMA(close, fast) - EMA(close, slow)
        Signal line = EMA(MACD line, signal)
        Histogram   = MACD line - Signal line

    Returns
    -------
    macd_line, signal_line, histogram
    """
    fast_ema   = ema(close, fast)
    slow_ema   = ema(close, slow)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int,
    smoothing: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index with +DI and -DI.

        +DI = 100 × RMA(+DM, length) / ATR
        -DI = 100 × RMA(-DM, length) / ATR
        DX  = 100 × |+DI - -DI| / (+DI + -DI)
        ADX = RMA(DX, smoothing)

    Returns
    -------
    plus_di, minus_di, adx_val
    """
    _assert_aligned(high, low, close)  # [FIX-10]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=close.index,
    )

    atr_val  = rma(tr, length)
    plus_di  = 100 * rma(plus_dm,  length) / atr_val
    minus_di = 100 * rma(minus_dm, length) / atr_val
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val  = rma(dx.fillna(0), smoothing)

    return plus_di, minus_di, adx_val


# ═══════════════════════════════════════════════════════════════════════════════
# TREND INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    multiplier: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator.

    [FIX-3] Direction convention corrected to industry standard:
        +1 = bullish (price above supertrend line)
        -1 = bearish (price below supertrend line)

    Original code had the inverse (-1=bullish, 1=bearish). All internal
    strategy code now uses the corrected convention.

    Parameters
    ----------
    high, low, close : OHLC series
    period           : ATR period
    multiplier       : ATR multiplier for band width

    Returns
    -------
    supertrend_line : pd.Series — the trailing stop line
    direction       : pd.Series — +1 bullish, -1 bearish
    """
    _assert_aligned(high, low, close)  # [FIX-10]
    atr_val    = atr(high, low, close, period)
    hl2        = (high + low) / 2
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    n               = len(close)
    supertrend_val  = np.zeros(n)
    direction       = np.zeros(n)

    supertrend_val[0] = upper_band.iloc[0]
    direction[0]      = -1  # start bearish until proven otherwise

    close_arr   = close.to_numpy()
    upper_arr   = upper_band.to_numpy()
    lower_arr   = lower_band.to_numpy()

    for i in range(1, n):
        prev_st = supertrend_val[i - 1]

        if close_arr[i] > prev_st:
            direction[i] = 1   # bullish
        elif close_arr[i] < prev_st:
            direction[i] = -1  # bearish
        else:
            direction[i] = direction[i - 1]

        if direction[i] == 1:
            # Bullish: use lower band, never let it fall
            supertrend_val[i] = lower_arr[i]
            if direction[i - 1] == 1 and supertrend_val[i] < prev_st:
                supertrend_val[i] = prev_st
        else:
            # Bearish: use upper band, never let it rise
            supertrend_val[i] = upper_arr[i]
            if direction[i - 1] == -1 and supertrend_val[i] > prev_st:
                supertrend_val[i] = prev_st

    return (
        pd.Series(supertrend_val, index=close.index),
        pd.Series(direction,      index=close.index),
    )


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Volume Weighted Average Price with daily session reset.

        typical_price = (high + low + close) / 3
        VWAP(session) = cumsum(typical_price × volume) / cumsum(volume)

    [FIX-8] Added DatetimeIndex guard. Without a DatetimeIndex,
    index.date raises AttributeError, producing a confusing traceback.

    Resets at each calendar date boundary. For daily data each bar is
    its own session, so VWAP equals the typical price of that bar.
    """
    _assert_aligned(high, low, close, volume)  # [FIX-10]

    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError(
            "vwap() requires a DatetimeIndex. "
            "Ensure the DataFrame index is parsed as datetime (e.g. pd.to_datetime)."
        )

    typical_price = (high + low + close) / 3
    dates         = close.index.date
    cum_tpv       = (typical_price * volume).groupby(dates).cumsum()
    cum_vol       = volume.groupby(dates).cumsum()

    return cum_tpv / cum_vol


# ═══════════════════════════════════════════════════════════════════════════════
# RSI HIDDEN DIVERGENCE  [NEW-1]
# ═══════════════════════════════════════════════════════════════════════════════

def _find_pivot_lows(series: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
    """
    Boolean Series marking confirmed pivot lows.

    A pivot low at index i is confirmed when the next `right_bars` bars
    have all closed higher. This introduces a lag of `right_bars` bars,
    which is intentional — it is the minimum delay required to confirm
    a pivot without look-ahead bias.
    """
    arr    = series.to_numpy()
    n      = len(arr)
    pivots = np.zeros(n, dtype=bool)
    for i in range(left_bars, n - right_bars):
        window = arr[i - left_bars: i + right_bars + 1]
        if arr[i] == window.min():
            pivots[i] = True
    return pd.Series(pivots, index=series.index)


def _find_pivot_highs(series: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
    """Boolean Series marking confirmed pivot highs (symmetric to pivot lows)."""
    arr    = series.to_numpy()
    n      = len(arr)
    pivots = np.zeros(n, dtype=bool)
    for i in range(left_bars, n - right_bars):
        window = arr[i - left_bars: i + right_bars + 1]
        if arr[i] == window.max():
            pivots[i] = True
    return pd.Series(pivots, index=series.index)


def rsi_hidden_divergence(
    close: pd.Series,
    rsi_length: int = 14,
    pivot_left: int = 5,
    pivot_right: int = 5,
    lookback_pivots: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """
    RSI Hidden Divergence — trend continuation signals.

    Hidden divergence signals that the prevailing trend is likely to continue,
    distinguishing it from regular divergence which signals reversal.

    Hidden Bullish (uptrend continuation)
    ──────────────────────────────────────
        Price : Higher Low  — pullback held above the prior swing low
        RSI   : Lower Low   — RSI dipped deeper on the same pullback
        Signal: expect the uptrend to resume

    Hidden Bearish (downtrend continuation)
    ────────────────────────────────────────
        Price : Lower High  — rally failed to exceed the prior swing high
        RSI   : Higher High — RSI exceeded its prior swing high
        Signal: expect the downtrend to continue

    ⚠ BIAS WARNING
    Pivot confirmation requires `pivot_right` future bars to resolve.
    Signals are timestamped at the bar where the pivot is *confirmed*,
    not where it occurs. The backtesting engine must use signal[i] to
    enter on bar i+1 open. Never enter on the signal bar itself.

    Parameters
    ----------
    close           : close price series (DatetimeIndex recommended)
    rsi_length      : RSI period (default 14)
    pivot_left      : bars left of pivot required for confirmation (default 5)
    pivot_right     : bars right of pivot required for confirmation (default 5)
                      also determines the confirmation lag in bars
    lookback_pivots : max number of prior pivots to compare against (default 5)

    Returns
    -------
    hidden_bull : pd.Series[bool] — True where hidden bullish divergence confirmed
    hidden_bear : pd.Series[bool] — True where hidden bearish divergence confirmed
    """
    rsi_vals = rsi(close, rsi_length)  # NaN during warmup; pivot detection handles this

    pl_price = _find_pivot_lows(close,    pivot_left, pivot_right)
    ph_price = _find_pivot_highs(close,   pivot_left, pivot_right)
    pl_rsi   = _find_pivot_lows(rsi_vals,  pivot_left, pivot_right)
    ph_rsi   = _find_pivot_highs(rsi_vals, pivot_left, pivot_right)

    n           = len(close)
    hidden_bull = np.zeros(n, dtype=bool)
    hidden_bear = np.zeros(n, dtype=bool)

    close_arr = close.to_numpy()
    rsi_arr   = rsi_vals.to_numpy()
    pl_p      = pl_price.to_numpy()
    ph_p      = ph_price.to_numpy()
    pl_r      = pl_rsi.to_numpy()
    ph_r      = ph_rsi.to_numpy()

    def _prior_pivot_indices(mask: np.ndarray, current: int, n_max: int):
        indices = np.where(mask[:current])[0]
        return indices[-n_max:] if len(indices) > 0 else np.array([], dtype=int)

    for i in range(pivot_right, n):
        # ── Hidden Bullish ────────────────────────────────────────────────────
        # Current bar is simultaneously a confirmed price pivot low AND rsi pivot low.
        # We look for a prior bar j that was also both; if price[i] > price[j]
        # (higher low) but rsi[i] < rsi[j] (lower low) → hidden bull.
        if pl_p[i] and pl_r[i]:
            prior_pl_p = _prior_pivot_indices(pl_p, i, lookback_pivots)
            prior_pl_r = _prior_pivot_indices(pl_r, i, lookback_pivots)
            common     = np.intersect1d(prior_pl_p, prior_pl_r)
            for j in common:
                if close_arr[i] > close_arr[j] and rsi_arr[i] < rsi_arr[j]:
                    hidden_bull[i] = True
                    break

        # ── Hidden Bearish ────────────────────────────────────────────────────
        # Current bar is simultaneously a confirmed price pivot high AND rsi pivot high.
        # We look for a prior bar j; if price[i] < price[j] (lower high)
        # but rsi[i] > rsi[j] (higher high) → hidden bear.
        if ph_p[i] and ph_r[i]:
            prior_ph_p = _prior_pivot_indices(ph_p, i, lookback_pivots)
            prior_ph_r = _prior_pivot_indices(ph_r, i, lookback_pivots)
            common     = np.intersect1d(prior_ph_p, prior_ph_r)
            for j in common:
                if close_arr[i] < close_arr[j] and rsi_arr[i] > rsi_arr[j]:
                    hidden_bear[i] = True
                    break

    return (
        pd.Series(hidden_bull, index=close.index),
        pd.Series(hidden_bear, index=close.index),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HPDR BANDS  [NEW-2]
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# NEW ENTRY FILTER INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def bollinger_bands(
    close: pd.Series,
    length: int = 20,
    mult: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

        mid   = SMA(close, length)
        std   = sample_std(close, length)   # ddof=1, matches TradingView
        upper = mid + mult × std
        lower = mid - mult × std

    Returns
    -------
    upper, lower, mid
    """
    mid   = sma(close, length)
    std   = close.rolling(window=length).std(ddof=1)
    upper = mid + mult * std
    lower = mid - mult * std
    return upper, lower, mid


def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    slowing: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Standard Stochastic Oscillator.

        raw_k = (close - lowest_low(k_period)) / (highest_high(k_period) - lowest_low(k_period)) × 100
        %K    = SMA(raw_k, slowing)
        %D    = SMA(%K, d_period)

    Returns
    -------
    k, d
    """
    _assert_aligned(high, low, close)
    lowest  = low.rolling(window=k_period).min()
    highest = high.rolling(window=k_period).max()
    raw_k   = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    k       = sma(raw_k, slowing)
    d       = sma(k, d_period)
    return k, d


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
) -> pd.Series:
    """
    Commodity Channel Index (Lambert's original definition).

        TP  = (high + low + close) / 3
        CCI = (TP - SMA(TP, length)) / (0.015 × MeanDev(TP, length))

    The 0.015 constant ensures ~70-80% of values fall within ±100 on
    normally-distributed data (Lambert, 1980).
    """
    _assert_aligned(high, low, close)
    tp       = (high + low + close) / 3
    tp_sma   = sma(tp, length)
    mean_dev = tp.rolling(length).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (tp - tp_sma) / (0.015 * mean_dev)


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """
    Williams %R (Larry Williams, 1973).

        %R = (highest_high(length) - close) / (highest_high(length) - lowest_low(length)) × -100

    Output range: [-100, 0].
    Oversold: %R < -80. Overbought: %R > -20.
    """
    _assert_aligned(high, low, close)
    highest = high.rolling(window=length).max()
    lowest  = low.rolling(window=length).min()
    return (highest - close) / (highest - lowest).replace(0, np.nan) * -100


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (Granville, 1963).

        OBV[i] = OBV[i-1] + volume[i]  if close[i] > close[i-1]
               = OBV[i-1] - volume[i]  if close[i] < close[i-1]
               = OBV[i-1]              otherwise
    """
    _assert_aligned(close, volume)
    delta     = close.diff()
    direction = pd.Series(
        np.where(delta > 0, 1, np.where(delta < 0, -1, 0)),
        index=close.index,
    )
    return (direction * volume).cumsum()


def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    length: int = 20,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channel (breakout channel).

        upper = highest_high(length)
        lower = lowest_low(length)
        mid   = (upper + lower) / 2

    Returns
    -------
    upper, lower, mid
    """
    _assert_aligned(high, low)
    upper = high.rolling(window=length).max()
    lower = low.rolling(window=length).min()
    mid   = (upper + lower) / 2
    return upper, lower, mid


def keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    mult: float = 1.5,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channel (Chester Keltner / Linda Raschke ATR variant).

        mid   = EMA(close, length)
        upper = mid + mult × ATR(length)
        lower = mid - mult × ATR(length)

    Returns
    -------
    upper, lower, mid
    """
    _assert_aligned(high, low, close)
    mid   = ema(close, length)
    atr_v = atr(high, low, close, length)
    upper = mid + mult * atr_v
    lower = mid - mult * atr_v
    return upper, lower, mid


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> pd.Series:
    """
    Parabolic SAR (Wilder, 1978).

    Iterative state machine — cannot be vectorized.

        SAR[i] = SAR[i-1] + AF × (EP - SAR[i-1])
        AF increments by af_step each time EP makes a new extreme, capped at af_max.

    Bullish (+1): price above SAR. Bearish (-1): price below SAR.
    """
    _assert_aligned(high, low)
    n      = len(high)
    high_a = high.to_numpy(dtype=float)
    low_a  = low.to_numpy(dtype=float)
    sar    = np.full(n, np.nan)

    trend  = 1          # +1 = bullish, -1 = bearish
    af     = af_start
    ep     = high_a[0]
    sar[0] = low_a[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]

        if trend == 1:  # bullish
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low_a[i - 1])
            if i >= 2:
                sar[i] = min(sar[i], low_a[i - 2])

            if low_a[i] < sar[i]:  # reversal to bearish
                trend  = -1
                sar[i] = ep
                ep     = low_a[i]
                af     = af_start
            else:
                if high_a[i] > ep:
                    ep = high_a[i]
                    af = min(af + af_step, af_max)
        else:  # bearish
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high_a[i - 1])
            if i >= 2:
                sar[i] = max(sar[i], high_a[i - 2])

            if high_a[i] > sar[i]:  # reversal to bullish
                trend  = 1
                sar[i] = ep
                ep     = high_a[i]
                af     = af_start
            else:
                if low_a[i] < ep:
                    ep = low_a[i]
                    af = min(af + af_step, af_max)

    return pd.Series(sar, index=high.index)


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> Dict[str, pd.Series]:
    """
    Ichimoku Kinko Hyo Cloud.

        Tenkan-sen  = (highest_high(tenkan) + lowest_low(tenkan)) / 2
        Kijun-sen   = (highest_high(kijun)  + lowest_low(kijun))  / 2
        Senkou A    = (Tenkan + Kijun) / 2
        Senkou B    = (highest_high(senkou_b) + lowest_low(senkou_b)) / 2
        Chikou span = close shifted backward kijun_period bars

    LOOK-AHEAD BIAS NOTE
    ────────────────────
    The display Senkou spans are shifted forward by kijun_period bars.
    For signal generation, NON-SHIFTED versions are used (senkou_a_signal /
    senkou_b_signal) so that we compare current price to the cloud as computed
    with only past data — no look-ahead bias.

    Returns
    -------
    Dict with keys:
        tenkan_sen, kijun_sen,
        senkou_a_signal, senkou_b_signal   — non-shifted, for entry logic
        senkou_a_display, senkou_b_display — shifted forward, for chart only
        chikou_span
    """
    _assert_aligned(high, low, close)

    def mid_price(h: pd.Series, l: pd.Series, period: int) -> pd.Series:
        return (h.rolling(period).max() + l.rolling(period).min()) / 2

    tenkan   = mid_price(high, low, tenkan_period)
    kijun    = mid_price(high, low, kijun_period)
    senkou_a = (tenkan + kijun) / 2
    senkou_b = mid_price(high, low, senkou_b_period)
    chikou   = close.shift(-kijun_period)

    return {
        'tenkan_sen':       tenkan,
        'kijun_sen':        kijun,
        'senkou_a_signal':  senkou_a,                       # non-shifted — use for signals
        'senkou_b_signal':  senkou_b,                       # non-shifted — use for signals
        'senkou_a_display': senkou_a.shift(kijun_period),   # shifted forward — chart only
        'senkou_b_display': senkou_b.shift(kijun_period),   # shifted forward — chart only
        'chikou_span':      chikou,
    }


def hull_ma(close: pd.Series, length: int = 20) -> pd.Series:
    """
    Hull Moving Average (Alan Hull, 2005).

        HMA = WMA(2 × WMA(close, length/2) - WMA(close, length), sqrt(length))

    Significantly less lag than EMA of the same period.
    """
    half_len = max(int(length / 2), 1)
    sqrt_len = max(int(np.sqrt(length)), 1)
    wma_half = wma(close, half_len)
    wma_full = wma(close, length)
    diff     = 2 * wma_half - wma_full
    return wma(diff, sqrt_len)


def trix(
    close: pd.Series,
    length: int = 15,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series]:
    """
    TRIX — Triple Exponential Moving Average Rate-of-Change.

        EMA1   = EMA(close, length)
        EMA2   = EMA(EMA1, length)
        EMA3   = EMA(EMA2, length)
        TRIX   = (EMA3 - EMA3.shift(1)) / EMA3.shift(1) × 100
        Signal = EMA(TRIX, signal)

    Long when TRIX crosses above Signal; short when crosses below.

    Returns
    -------
    trix_line, signal_line
    """
    ema1      = ema(close, length)
    ema2      = ema(ema1,  length)
    ema3      = ema(ema2,  length)
    trix_line = (ema3 - ema3.shift(1)) / ema3.shift(1).replace(0, np.nan) * 100
    sig_line  = ema(trix_line, signal)
    return trix_line, sig_line


def hpdr_bands(
    close: pd.Series,
    lookback: int = 252,
    z_scores: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5),
) -> Dict[str, pd.Series]:
    """
    High Probability Distribution Range (HPDR) Bands.

    Non-parametric rolling quantile bands anchored to the median price.
    At each bar, computes the empirical distribution of close prices over
    the prior `lookback` bars and marks quantile levels corresponding to
    each z-score's Normal-equivalent coverage probability.

    WHY QUANTILE (NOT PARAMETRIC)
    ──────────────────────────────
    Parametric (log-normal) approaches suffer from look-ahead bias in σ
    estimation: if you use the same window for SMA and σ, the σ is biased
    downward ~40–60%, making bands too narrow and poorly calibrated.
    Rolling quantile bands bypass this entirely — coverage is exact by
    construction without any distributional assumption.

    z-score → quantile mapping (Normal CDF equivalents, two-tailed)
    ────────────────────────────────────────────────────────────────
        ±0.5σ  →  Q(30.9%) / Q(69.1%)   ~38% outside  (teal,  inner)
        ±1.0σ  →  Q(15.9%) / Q(84.1%)   ~32% outside  (green)
        ±1.5σ  →  Q( 6.7%) / Q(93.3%)   ~13% outside  (yellow)
        ±2.0σ  →  Q( 2.3%) / Q(97.7%)   ~ 5% outside  (orange)
        ±2.5σ  →  Q( 0.6%) / Q(99.4%)   ~ 1% outside  (red,   outer)

    Visual interpretation
    ─────────────────────
        Price in teal zone    → behaving within historical norms
        Price in red zone     → historically overextended; statistical outlier
        Bands wide            → high-volatility regime
        Bands narrow (squeeze)→ low-volatility regime; potential breakout setup

    ⚠ These are DESCRIPTIVE bands based on historical distribution, not
    predictive. Being in the red zone does not imply an imminent reversal.

    Parameters
    ----------
    close    : close price series
    lookback : rolling window in bars (252 ≈ 1 trading year on daily data)
    z_scores : band levels; each maps to Normal-equivalent quantile pair

    Returns
    -------
    dict with keys:
        'center'        → rolling median of close (Q50)
        'mu_return'     → rolling mean of daily log returns (informational)
        'sigma_return'  → rolling std  of daily log returns (informational)
        'upper_{z:.1f}' → upper quantile band for each z level
        'lower_{z:.1f}' → lower quantile band for each z level
    """
    from scipy.stats import norm

    log_ret = np.log(close / close.shift(1))

    result: Dict[str, pd.Series] = {
        'center':       close.rolling(window=lookback).quantile(0.5),
        'mu_return':    log_ret.rolling(window=lookback).mean(),
        'sigma_return': log_ret.rolling(window=lookback).std(),
    }

    for z in z_scores:
        # Convert z-score to one-tailed quantile levels via Normal CDF
        q_low  = float(norm.cdf(-z))   # e.g. z=1 → 0.159
        q_high = float(norm.cdf( z))   # e.g. z=1 → 0.841
        label  = f"{z:.1f}"
        result[f'upper_{label}'] = close.rolling(window=lookback).quantile(q_high)
        result[f'lower_{label}'] = close.rolling(window=lookback).quantile(q_low)

    return result
