"""
Five reference backtest configurations that pin behavioral snapshots.

Each entry: (config_name, params_dict, fixture_filename)

StrategyParams defaults to watch out for:
    pamrp_enabled=True, bbwp_enabled=True, stop_loss_enabled=True,
    pamrp_exit_enabled=True — must be explicitly overridden when not wanted.

Fixtures 1/2/4/5 use real yfinance daily data (committed as parquet).
Fixture 3 (ETH-USD 1h style) uses synthetic generate_sample_data output
because yfinance limits 1h data to ~730 calendar days of rolling history.
"""

CONFIGS = [
    (
        "pamrp_long_spy",
        {
            "trade_direction": "long_only",
            "entry_operator": "and",
            "exit_operator": "or",
            # PAMRP entry on
            "pamrp_enabled": True,
            "pamrp_entry_ma_length": 20,
            "pamrp_entry_lookback": 350,
            "pamrp_entry_ma_type": "sma",
            "pamrp_entry_long": 20,
            "pamrp_entry_short": 80,
            # PAMRP exit on
            "pamrp_exit_enabled": True,
            "pamrp_exit_ma_length": 20,
            "pamrp_exit_lookback": 350,
            "pamrp_exit_ma_type": "sma",
            "pamrp_exit_long": 70,
            "pamrp_exit_short": 30,
            # Stop loss (default exits)
            "stop_loss_enabled": True,
            "stop_loss_pct_long": 3.0,
            # Disable all others
            "bbwp_enabled": False,
            "adx_enabled": False,
            "ma_trend_enabled": False,
            "rsi_enabled": False,
            "volume_enabled": False,
            "supertrend_enabled": False,
            "vwap_enabled": False,
            "macd_enabled": False,
            "take_profit_enabled": False,
            "trailing_stop_enabled": False,
            "atr_trailing_enabled": False,
            "stoch_rsi_exit_enabled": False,
            "time_exit_enabled": False,
            "ma_exit_enabled": False,
            "bbwp_exit_enabled": False,
        },
        "spy_1d_2020_2024.parquet",
    ),
    (
        "pamrp_bbwp_long_qqq",
        {
            "trade_direction": "long_only",
            "entry_operator": "and",
            "exit_operator": "or",
            # PAMRP entry on
            "pamrp_enabled": True,
            "pamrp_entry_ma_length": 20,
            "pamrp_entry_lookback": 252,
            "pamrp_entry_ma_type": "sma",
            "pamrp_entry_long": 20,
            # PAMRP exit on
            "pamrp_exit_enabled": True,
            "pamrp_exit_ma_length": 20,
            "pamrp_exit_lookback": 252,
            "pamrp_exit_ma_type": "sma",
            "pamrp_exit_long": 70,
            # BBWP entry filter on
            "bbwp_enabled": True,
            "bbwp_length": 13,
            "bbwp_lookback": 252,
            "bbwp_sma_length": 5,
            "bbwp_threshold_long": 50,
            "bbwp_ma_filter": "disabled",
            # Stop loss + Take profit
            "stop_loss_enabled": True,
            "stop_loss_pct_long": 3.0,
            "take_profit_enabled": True,
            "take_profit_pct_long": 5.0,
            # Disable others
            "adx_enabled": False,
            "ma_trend_enabled": False,
            "rsi_enabled": False,
            "volume_enabled": False,
            "supertrend_enabled": False,
            "vwap_enabled": False,
            "macd_enabled": False,
            "trailing_stop_enabled": False,
            "atr_trailing_enabled": False,
            "stoch_rsi_exit_enabled": False,
            "time_exit_enabled": False,
            "ma_exit_enabled": False,
            "bbwp_exit_enabled": False,
        },
        "qqq_1d_2018_2024.parquet",
    ),
    (
        "rsi_both_eth_trailing",
        {
            "trade_direction": "both",
            "entry_operator": "and",
            "exit_operator": "or",
            "entry_conflict_mode": "skip",
            # RSI mean-reversion
            "rsi_enabled": True,
            "rsi_length": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            # Trailing stop + hard SL backup (prevents open-ended positions
            # on a downtrending synthetic asset where activation never fires)
            "stop_loss_enabled": True,
            "stop_loss_pct_long": 8.0,
            "stop_loss_pct_short": 8.0,
            "trailing_stop_enabled": True,
            "trailing_stop_pct": 3.0,
            "trailing_stop_activation": 1.0,
            # No PAMRP
            "pamrp_enabled": False,
            "pamrp_exit_enabled": False,
            # Disable others
            "bbwp_enabled": False,
            "adx_enabled": False,
            "ma_trend_enabled": False,
            "volume_enabled": False,
            "supertrend_enabled": False,
            "vwap_enabled": False,
            "macd_enabled": False,
            "take_profit_enabled": False,
            "atr_trailing_enabled": False,
            "stoch_rsi_exit_enabled": False,
            "time_exit_enabled": False,
            "ma_exit_enabled": False,
            "bbwp_exit_enabled": False,
        },
        # SYNTHETIC — yfinance 1h data is limited to ~730 rolling calendar days,
        # making the fixture non-reproducible across time. Using generate_sample_data
        # with seed=7, vol=0.025, drift=0.0002 to simulate a crypto-like asset.
        "eth_1h_synthetic.parquet",
    ),
    (
        "adx_matrend_long_aapl_timeexit",
        {
            "trade_direction": "long_only",
            "entry_operator": "and",
            "exit_operator": "or",
            # ADX trend-strength filter
            "adx_enabled": True,
            "adx_length": 14,
            "adx_smoothing": 14,
            "adx_threshold": 20,
            "adx_require_di": True,
            # MA trend direction filter
            "ma_trend_enabled": True,
            "ma_fast_length": 50,
            "ma_slow_length": 200,
            "ma_type": "sma",
            # Time exit
            "time_exit_enabled": True,
            "time_exit_bars_long": 20,
            # No signal-based exits, no fixed SL
            "stop_loss_enabled": False,
            "pamrp_exit_enabled": False,
            # Disable others
            "pamrp_enabled": False,
            "bbwp_enabled": False,
            "rsi_enabled": False,
            "volume_enabled": False,
            "supertrend_enabled": False,
            "vwap_enabled": False,
            "macd_enabled": False,
            "take_profit_enabled": False,
            "trailing_stop_enabled": False,
            "atr_trailing_enabled": False,
            "stoch_rsi_exit_enabled": False,
            "ma_exit_enabled": False,
            "bbwp_exit_enabled": False,
        },
        "aapl_1d_2015_2024.parquet",
    ),
    (
        "macd_volume_both_tsla_pamrpexit",
        {
            "trade_direction": "both",
            "entry_operator": "and",
            "exit_operator": "or",
            "entry_conflict_mode": "skip",
            # MACD signal
            "macd_enabled": True,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "macd_mode": "histogram",
            # Volume confirmation
            "volume_enabled": True,
            "volume_ma_length": 20,
            "volume_multiplier": 1.0,
            # Stop loss
            "stop_loss_enabled": True,
            "stop_loss_pct_long": 3.0,
            "stop_loss_pct_short": 3.0,
            # PAMRP exit
            "pamrp_exit_enabled": True,
            "pamrp_exit_ma_length": 20,
            "pamrp_exit_lookback": 252,
            "pamrp_exit_ma_type": "sma",
            "pamrp_exit_long": 70,
            "pamrp_exit_short": 30,
            # No PAMRP entry
            "pamrp_enabled": False,
            # Disable others
            "bbwp_enabled": False,
            "adx_enabled": False,
            "ma_trend_enabled": False,
            "rsi_enabled": False,
            "supertrend_enabled": False,
            "vwap_enabled": False,
            "take_profit_enabled": False,
            "trailing_stop_enabled": False,
            "atr_trailing_enabled": False,
            "stoch_rsi_exit_enabled": False,
            "time_exit_enabled": False,
            "ma_exit_enabled": False,
            "bbwp_exit_enabled": False,
        },
        "tsla_1d_2020_2024.parquet",
    ),
]
