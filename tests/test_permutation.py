import numpy as np
import pandas as pd
import pytest

from src.permutation import _permute_prices


def _make_ohlcv(n=400, seed=7):
    """Synthetic OHLCV with real gaps, varied bar shapes, and AR(1) return
    autocorrelation — so the signal-destruction tests are meaningful."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.6 * r[i - 1] + rng.normal(0, 0.01)
    close = 100.0 * np.exp(np.cumsum(r))
    gap = rng.normal(0, 0.004, n)
    openp = np.empty(n); openp[0] = close[0]
    openp[1:] = close[:-1] * np.exp(gap[1:])
    up = np.abs(rng.normal(0, 0.006, n))
    dn = np.abs(rng.normal(0, 0.006, n))
    high = np.maximum(openp, close) * np.exp(up)
    low  = np.minimum(openp, close) * np.exp(-dn)
    vol  = rng.integers(1_000, 5_000, n).astype(float)
    idx  = pd.date_range("2021-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _decompose(df):
    o = df["open"].values; h = df["high"].values
    l = df["low"].values;  c = df["close"].values
    gap  = np.log(o[1:] / c[:-1])
    body = np.log(c[1:] / o[1:])
    hi   = np.log(h[1:] / o[1:])
    lo   = np.log(l[1:] / o[1:])
    return gap, body, hi, lo


# 1. The original bug is gone: bar-shape ratios are no longer time-pinned.
# Old code: new_high[i]/new_close[i] == orig_high[i]/orig_close[i] exactly (ratio
# at each index preserved). New code shuffles shape tuples independently, so
# corr(orig_ratio, new_ratio) should be near zero.
def test_intrabar_no_longer_pinned_to_close():
    df = _make_ohlcv()
    rng = np.random.default_rng(42)
    s = _permute_prices(df, rng)
    for col in ("open", "high", "low"):
        orig_ratio = np.log(df[col].values / df["close"].values)
        new_ratio  = np.log(s[col].values  / s["close"].values)
        corr = np.corrcoef(orig_ratio, new_ratio)[0, 1]
        assert corr < 0.5, f"{col}/close ratio is still time-pinned (corr={corr:.4f})"


# 2. OHLC validity holds on every surrogate, across many seeds.
def test_ohlc_validity():
    df = _make_ohlcv()
    for seed in range(25):
        s = _permute_prices(df, np.random.default_rng(seed))
        o, h, l, c = (s[k].values for k in ("open", "high", "low", "close"))
        assert np.all(h >= np.maximum(o, c) - 1e-9)
        assert np.all(l <= np.minimum(o, c) + 1e-9)
        assert np.all(h >= l)
        assert np.all(s.values[:, :4] > 0)


# 3. Marginals preserved exactly: the surrogate is a true permutation of the
#    decomposed components (gaps + shape tuples), not a corruption of them.
def test_marginals_preserved_exactly():
    df = _make_ohlcv()
    g0, b0, hi0, lo0 = _decompose(df)
    s = _permute_prices(df, np.random.default_rng(42))
    g1, b1, hi1, lo1 = _decompose(s)
    assert np.allclose(np.sort(g0), np.sort(g1), atol=1e-9)
    key0 = np.lexsort((lo0, hi0, b0)); key1 = np.lexsort((lo1, hi1, b1))
    assert np.allclose(b0[key0],  b1[key1],  atol=1e-9)
    assert np.allclose(hi0[key0], hi1[key1], atol=1e-9)
    assert np.allclose(lo0[key0], lo1[key1], atol=1e-9)


# 4. Temporal structure destroyed: surrogate return autocorrelation ~ 0,
#    even though the real series is strongly autocorrelated.
def test_autocorrelation_destroyed():
    df = _make_ohlcv()
    def lag1(c):
        r = np.diff(np.log(c)); return np.corrcoef(r[:-1], r[1:])[0, 1]
    real_ac = lag1(df["close"].values)
    s = _permute_prices(df, np.random.default_rng(42))
    surr_ac = lag1(s["close"].values)
    assert abs(real_ac) > 0.3
    assert abs(surr_ac) < 0.1


# 5. Anchor bar is unchanged.
def test_anchor_bar_unchanged():
    df = _make_ohlcv()
    s = _permute_prices(df, np.random.default_rng(42))
    for col in ("open", "high", "low", "close"):
        assert s[col].iloc[0] == pytest.approx(df[col].iloc[0])


# 6. Determinism: same seed -> identical surrogate.
def test_determinism():
    df = _make_ohlcv()
    a = _permute_prices(df, np.random.default_rng(123))
    b = _permute_prices(df, np.random.default_rng(123))
    pd.testing.assert_frame_equal(a, b)


# 7. Volume untouched.
def test_volume_unchanged():
    df = _make_ohlcv()
    s = _permute_prices(df, np.random.default_rng(42))
    assert np.array_equal(s["volume"].values, df["volume"].values)


# 8. Positivity guard fires.
def test_nonpositive_raises():
    df = _make_ohlcv()
    df.iloc[5, df.columns.get_loc("low")] = -1.0
    with pytest.raises(ValueError):
        _permute_prices(df, np.random.default_rng(0))
