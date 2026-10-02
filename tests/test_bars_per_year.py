import pandas as pd
import pytest

from src.backtest import _estimate_bars_per_year


def _session_index(freq: str, start="09:30", end="15:59", days=30):
    out = []
    for d in pd.bdate_range("2024-01-02", periods=days):
        out.extend(pd.date_range(f"{d.date()} {start}", f"{d.date()} {end}", freq=freq))
    return pd.DatetimeIndex(out)


def _bars_per_date(idx: pd.DatetimeIndex) -> int:
    return int(pd.Series(idx.normalize()).value_counts().median())


def _bpy(idx) -> int:
    return _estimate_bars_per_year(pd.DataFrame(index=idx))


def test_session_helper_counts():
    assert _bars_per_date(_session_index("30min")) == 13
    assert _bars_per_date(_session_index("1h")) == 7


@pytest.mark.parametrize(
    "idx, expected",
    [
        (_session_index("1min"), 98280),
        (_session_index("30min"), 3276),
        (_session_index("1h"), 1764),
        (pd.date_range("2024-01-01", periods=24 * 30, freq="h"), 8760),
        (pd.date_range("2024-01-01", periods=6 * 30, freq="4h"), 2190),
        (pd.bdate_range("2020-01-01", periods=400), 252),
        (pd.date_range("2020-01-01", periods=400, freq="D"), 365),
        (pd.date_range("2020-01-06", periods=100, freq="W-MON"), 52),
        (pd.date_range("2020-01-01", periods=60, freq="MS"), 12),
        (pd.date_range("2020-01-01", periods=1, freq="D"), 252),
    ],
    ids=[
        "1min_equity", "30min_equity", "1h_equity", "1h_crypto", "4h_crypto",
        "daily_equity", "daily_crypto", "weekly", "monthly", "single_bar",
    ],
)
def test_bars_per_year(idx, expected):
    assert _bpy(idx) == expected


def test_non_datetime_index_raises():
    with pytest.raises(TypeError):
        _estimate_bars_per_year(pd.DataFrame(index=pd.RangeIndex(10)))
