"""
Strategy Module
===============
Generates entry and exit signals from StrategyParams + indicator layer.

Changes from previous version
──────────────────────────────
[FIX-ST]  supertrend direction convention corrected in generate_entry_signals.
          Old code checked `st_direction < 0` for long (legacy inverted convention).
          Now correctly checks `st_direction > 0` (1 = bullish, industry standard).
          The indicators layer handles the actual flip; this file just uses the
          correct sign.

[FIX-SR]  stoch_rsi call updated to pass `stoch_length` as the new second argument.
          Previously the call was stoch_rsi(close, rsi_length, k, d).
          Now:                       stoch_rsi(close, rsi_length, stoch_length, k, d).
          stoch_length defaults to stoch_rsi_length (same parameter) since no
          separate UI param exists yet — behaviour is unchanged for existing configs.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Iterable
from enum import Enum

from ..indicators import (
    pamrp, bbwp, sma, ema, rsi, stoch_rsi, adx, atr,
    supertrend, vwap, macd, ma,
    bollinger_bands, stochastic_oscillator, cci, williams_r,
    obv, donchian_channel, keltner_channel, parabolic_sar,
    ichimoku, hull_ma, trix,
)


class TradeDirection(Enum):
    LONG_ONLY  = "long_only"
    SHORT_ONLY = "short_only"
    BOTH       = "both"


class ConditionOperator(str, Enum):
    AND = "and"
    OR = "or"


class EntryConflictMode(str, Enum):
    SKIP = "skip"
    PREFER_LONG = "prefer_long"
    PREFER_SHORT = "prefer_short"


@dataclass
class StrategyParams:
    """All strategy parameters — fully exposed."""

    trade_direction: TradeDirection = TradeDirection.LONG_ONLY
    entry_operator: ConditionOperator = ConditionOperator.AND
    exit_operator: ConditionOperator = ConditionOperator.OR
    allow_same_bar_exit: bool = True
    allow_same_bar_reversal: bool = False
    entry_conflict_mode: EntryConflictMode = EntryConflictMode.SKIP

    # Position Sizing
    position_size_pct: float = 100.0
    use_kelly: bool = False
    kelly_fraction: float = 0.5

    # PAMRP
    pamrp_enabled: bool = True
    pamrp_entry_length: int = 21
    pamrp_entry_long: int = 20
    pamrp_entry_short: int = 80
    pamrp_exit_length: int = 21
    pamrp_exit_long: int = 70
    pamrp_exit_short: int = 30

    # BBWP
    bbwp_enabled: bool = True
    bbwp_length: int = 13
    bbwp_lookback: int = 252
    bbwp_sma_length: int = 5
    bbwp_threshold_long: int = 50
    bbwp_threshold_short: int = 50
    bbwp_ma_filter: str = "disabled"

    # ADX
    adx_enabled: bool = False
    adx_length: int = 14
    adx_smoothing: int = 14
    adx_threshold: int = 20
    adx_require_di: bool = False

    # MA Trend
    ma_trend_enabled: bool = False
    ma_fast_length: int = 50
    ma_slow_length: int = 200
    ma_type: str = "sma"

    # RSI
    rsi_enabled: bool = False
    rsi_length: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70

    # Volume
    volume_enabled: bool = False
    volume_ma_length: int = 20
    volume_multiplier: float = 1.0

    # Supertrend
    supertrend_enabled: bool = False
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0

    # VWAP
    vwap_enabled: bool = False

    # MACD
    macd_enabled: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_mode: str = "histogram"

    # Bollinger Bands Entry
    bb_enabled: bool = False
    bb_length: int = 20
    bb_mult: float = 2.0
    bb_mode: str = 'squeeze'   # 'squeeze' or 'breakout'

    # Stochastic Entry
    stoch_entry_enabled: bool = False
    stoch_entry_k_period: int = 14
    stoch_entry_d_period: int = 3
    stoch_entry_slowing: int = 3
    stoch_entry_oversold: int = 20
    stoch_entry_overbought: int = 80

    # CCI Entry
    cci_enabled: bool = False
    cci_length: int = 20
    cci_oversold: int = -100
    cci_overbought: int = 100

    # Williams %R Entry
    willr_enabled: bool = False
    willr_length: int = 14
    willr_oversold: int = -80
    willr_overbought: int = -20

    # OBV Entry
    obv_enabled: bool = False
    obv_ma_length: int = 20

    # Donchian Channel Entry
    donchian_enabled: bool = False
    donchian_length: int = 20

    # Keltner Channel Entry
    keltner_enabled: bool = False
    keltner_length: int = 20
    keltner_mult: float = 1.5

    # Parabolic SAR Entry
    psar_enabled: bool = False
    psar_af_start: float = 0.02
    psar_af_step: float = 0.02
    psar_af_max: float = 0.2

    # Ichimoku Cloud Entry
    ichi_enabled: bool = False
    ichi_tenkan: int = 9
    ichi_kijun: int = 26
    ichi_senkou_b: int = 52

    # Hull MA Entry
    hull_enabled: bool = False
    hull_length: int = 20

    # TRIX Entry
    trix_enabled: bool = False
    trix_length: int = 15
    trix_signal: int = 9

    # Stop Loss
    stop_loss_enabled: bool = True
    stop_loss_pct_long: float = 3.0
    stop_loss_pct_short: float = 3.0

    # Take Profit
    take_profit_enabled: bool = False
    take_profit_pct_long: float = 5.0
    take_profit_pct_short: float = 5.0

    # Trailing Stop
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 2.0
    trailing_stop_activation: float = 1.0

    # ATR Trailing
    atr_trailing_enabled: bool = False
    atr_length: int = 14
    atr_multiplier: float = 2.0

    # PAMRP Exit
    pamrp_exit_enabled: bool = True

    # Stoch RSI Exit
    stoch_rsi_exit_enabled: bool = False
    stoch_rsi_length: int = 14
    stoch_rsi_k: int = 3
    stoch_rsi_d: int = 3
    stoch_rsi_overbought: int = 80
    stoch_rsi_oversold: int = 20

    # Time Exit
    time_exit_enabled: bool = False
    time_exit_bars_long: int = 20
    time_exit_bars_short: int = 20

    # MA Exit
    ma_exit_enabled: bool = False
    ma_exit_fast: int = 10
    ma_exit_slow: int = 20

    # BBWP Exit
    bbwp_exit_enabled: bool = False
    bbwp_exit_threshold: int = 80

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            result[k] = v.value if isinstance(v, Enum) else v
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StrategyParams':
        d = d.copy()
        if 'trade_direction' in d and isinstance(d['trade_direction'], str):
            d['trade_direction'] = TradeDirection(d['trade_direction'])
        if 'entry_operator' in d and isinstance(d['entry_operator'], str):
            d['entry_operator'] = ConditionOperator(d['entry_operator'].lower())
        if 'exit_operator' in d and isinstance(d['exit_operator'], str):
            d['exit_operator'] = ConditionOperator(d['exit_operator'].lower())
        if 'entry_conflict_mode' in d and isinstance(d['entry_conflict_mode'], str):
            d['entry_conflict_mode'] = EntryConflictMode(d['entry_conflict_mode'].lower())
        legacy_pamrp_length = d.pop('pamrp_length', None)
        if legacy_pamrp_length is not None:
            d.setdefault('pamrp_entry_length', legacy_pamrp_length)
            d.setdefault('pamrp_exit_length', legacy_pamrp_length)
        valid_fields = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


class SignalGenerator:
    """Generate entry and exit signals."""

    def __init__(self, params: StrategyParams):
        self.params = params

    @staticmethod
    def _combine_condition_masks(
        masks: Iterable[pd.Series],
        operator: ConditionOperator,
        index: pd.Index,
    ) -> pd.Series:
        masks = [mask.fillna(False) for mask in masks]
        if not masks:
            return pd.Series(False, index=index)

        if operator == ConditionOperator.AND:
            result = pd.Series(True, index=index)
            for mask in masks:
                result = result & mask
            return result

        result = pd.Series(False, index=index)
        for mask in masks:
            result = result | mask
        return result

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p  = self.params

        # PAMRP entry/exit can use different lookbacks.
        if p.pamrp_enabled:
            df['pamrp_entry'] = pamrp(df['high'], df['low'], df['close'], p.pamrp_entry_length)
        else:
            df['pamrp_entry'] = 50.0

        if p.pamrp_exit_enabled:
            if p.pamrp_enabled and p.pamrp_entry_length == p.pamrp_exit_length:
                df['pamrp_exit'] = df['pamrp_entry']
            else:
                df['pamrp_exit'] = pamrp(df['high'], df['low'], df['close'], p.pamrp_exit_length)
        else:
            df['pamrp_exit'] = 50.0

        # Keep the legacy column for callers that still expect one PAMRP series.
        if p.pamrp_enabled:
            df['pamrp'] = df['pamrp_entry']
        elif p.pamrp_exit_enabled:
            df['pamrp'] = df['pamrp_exit']
        else:
            df['pamrp'] = 50.0

        # BBWP
        if p.bbwp_enabled or p.bbwp_exit_enabled:
            df['bbwp']     = bbwp(df['close'], p.bbwp_length, p.bbwp_lookback)
            df['bbwp_sma'] = sma(df['bbwp'], p.bbwp_sma_length)
        else:
            df['bbwp']     = 50.0
            df['bbwp_sma'] = 50.0

        # ADX
        if p.adx_enabled:
            df['di_plus'], df['di_minus'], df['adx'] = adx(
                df['high'], df['low'], df['close'], p.adx_length, p.adx_smoothing
            )

        # MA Trend
        if p.ma_trend_enabled:
            df['ma_fast'] = ma(df['close'], p.ma_fast_length, p.ma_type)
            df['ma_slow'] = ma(df['close'], p.ma_slow_length, p.ma_type)

        # RSI
        if p.rsi_enabled:
            df['rsi'] = rsi(df['close'], p.rsi_length)

        # Stoch RSI
        # [FIX-SR] stoch_length added as explicit second argument.
        # Defaults to stoch_rsi_length (same value) — no behaviour change
        # for existing configs. Expose a separate UI param in the future if needed.
        if p.stoch_rsi_exit_enabled:
            df['stoch_k'], df['stoch_d'] = stoch_rsi(
                df['close'],
                p.stoch_rsi_length,   # rsi_length
                p.stoch_rsi_length,   # stoch_length  ← [FIX-SR]
                p.stoch_rsi_k,
                p.stoch_rsi_d,
            )

        # Volume
        if p.volume_enabled and 'volume' in df.columns:
            df['volume_ma'] = sma(df['volume'], p.volume_ma_length)

        # Supertrend
        if p.supertrend_enabled:
            df['supertrend'], df['st_direction'] = supertrend(
                df['high'], df['low'], df['close'],
                p.supertrend_period, p.supertrend_multiplier
            )

        # VWAP
        if p.vwap_enabled and 'volume' in df.columns:
            df['vwap'] = vwap(df['high'], df['low'], df['close'], df['volume'])

        # MACD
        if p.macd_enabled:
            df['macd'], df['macd_signal'], df['macd_hist'] = macd(
                df['close'], p.macd_fast, p.macd_slow, p.macd_signal
            )

        # Bollinger Bands
        if p.bb_enabled:
            df['bb_upper'], df['bb_lower'], df['bb_mid'] = bollinger_bands(
                df['close'], p.bb_length, p.bb_mult
            )

        # Stochastic Entry
        if p.stoch_entry_enabled:
            df['stoch_entry_k'], df['stoch_entry_d'] = stochastic_oscillator(
                df['high'], df['low'], df['close'],
                p.stoch_entry_k_period, p.stoch_entry_d_period, p.stoch_entry_slowing,
            )

        # CCI
        if p.cci_enabled:
            df['cci'] = cci(df['high'], df['low'], df['close'], p.cci_length)

        # Williams %R
        if p.willr_enabled:
            df['willr'] = williams_r(df['high'], df['low'], df['close'], p.willr_length)

        # OBV
        if p.obv_enabled and 'volume' in df.columns:
            df['obv'] = obv(df['close'], df['volume'])
            df['obv_ma'] = sma(df['obv'], p.obv_ma_length)

        # Donchian Channel
        if p.donchian_enabled:
            df['donchian_upper'], df['donchian_lower'], df['donchian_mid'] = donchian_channel(
                df['high'], df['low'], p.donchian_length
            )

        # Keltner Channel
        if p.keltner_enabled:
            df['keltner_upper'], df['keltner_lower'], df['keltner_mid'] = keltner_channel(
                df['high'], df['low'], df['close'], p.keltner_length, p.keltner_mult
            )

        # Parabolic SAR
        if p.psar_enabled:
            df['psar'] = parabolic_sar(
                df['high'], df['low'], p.psar_af_start, p.psar_af_step, p.psar_af_max
            )

        # Ichimoku Cloud
        if p.ichi_enabled:
            ichi_result = ichimoku(
                df['high'], df['low'], df['close'],
                p.ichi_tenkan, p.ichi_kijun, p.ichi_senkou_b,
            )
            df['ichi_tenkan_sen']    = ichi_result['tenkan_sen']
            df['ichi_kijun_sen']     = ichi_result['kijun_sen']
            df['ichi_senkou_a']      = ichi_result['senkou_a_signal']   # non-shifted for signals
            df['ichi_senkou_b_line'] = ichi_result['senkou_b_signal']   # non-shifted for signals
            df['ichi_senkou_a_disp'] = ichi_result['senkou_a_display']  # shifted for chart
            df['ichi_senkou_b_disp'] = ichi_result['senkou_b_display']  # shifted for chart
            df['ichi_chikou']        = ichi_result['chikou_span']

        # Hull MA
        if p.hull_enabled:
            df['hull_ma'] = hull_ma(df['close'], p.hull_length)

        # TRIX
        if p.trix_enabled:
            df['trix_line'], df['trix_signal_line'] = trix(
                df['close'], p.trix_length, p.trix_signal
            )

        # ATR (for trailing stop)
        if p.atr_trailing_enabled:
            df['atr'] = atr(df['high'], df['low'], df['close'], p.atr_length)

        # MA Exit
        if p.ma_exit_enabled:
            df['exit_ma_fast'] = ema(df['close'], p.ma_exit_fast)
            df['exit_ma_slow'] = ema(df['close'], p.ma_exit_slow)

        return df

    def generate_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p  = self.params

        long_masks: list[pd.Series] = []
        short_masks: list[pd.Series] = []

        if p.pamrp_enabled:
            long_masks.append(df['pamrp_entry'] < p.pamrp_entry_long)
            short_masks.append(df['pamrp_entry'] > p.pamrp_entry_short)

        if p.bbwp_enabled:
            long_mask = df['bbwp'] < p.bbwp_threshold_long
            short_mask = df['bbwp'] > p.bbwp_threshold_short

            if p.bbwp_ma_filter == 'decreasing':
                bbwp_ma_ok   = df['bbwp_sma'] < df['bbwp_sma'].shift(1)
                long_mask = long_mask & bbwp_ma_ok
                short_mask = short_mask & bbwp_ma_ok
            elif p.bbwp_ma_filter == 'increasing':
                bbwp_ma_ok   = df['bbwp_sma'] > df['bbwp_sma'].shift(1)
                long_mask = long_mask & bbwp_ma_ok
                short_mask = short_mask & bbwp_ma_ok

            long_masks.append(long_mask)
            short_masks.append(short_mask)

        if p.adx_enabled and 'adx' in df.columns:
            long_mask = df['adx'] > p.adx_threshold
            short_mask = long_mask
            if p.adx_require_di:
                long_mask = long_mask & (df['di_plus'] > df['di_minus'])
                short_mask = short_mask & (df['di_minus'] > df['di_plus'])
            long_masks.append(long_mask)
            short_masks.append(short_mask)

        if p.ma_trend_enabled and 'ma_fast' in df.columns:
            long_masks.append(df['ma_fast'] > df['ma_slow'])
            short_masks.append(df['ma_fast'] < df['ma_slow'])

        if p.rsi_enabled and 'rsi' in df.columns:
            long_masks.append(df['rsi'] < p.rsi_oversold)
            short_masks.append(df['rsi'] > p.rsi_overbought)

        if p.volume_enabled and 'volume_ma' in df.columns:
            vol_ok = df['volume'] > df['volume_ma'] * p.volume_multiplier
            long_masks.append(vol_ok)
            short_masks.append(vol_ok)

        if p.supertrend_enabled and 'st_direction' in df.columns:
            # [FIX-ST] Corrected convention: +1 = bullish, -1 = bearish
            # Old code (wrong): st_direction < 0 for long, > 0 for short
            long_masks.append(df['st_direction'] > 0)
            short_masks.append(df['st_direction'] < 0)

        if p.vwap_enabled and 'vwap' in df.columns:
            long_masks.append(df['close'] > df['vwap'])
            short_masks.append(df['close'] < df['vwap'])

        if p.macd_enabled and 'macd_hist' in df.columns:
            if p.macd_mode == 'histogram':
                long_mask = df['macd_hist'] > 0
                short_mask = df['macd_hist'] < 0
            elif p.macd_mode == 'crossover':
                long_mask = df['macd'] > df['macd_signal']
                short_mask = df['macd'] < df['macd_signal']
            else:  # zero-line
                long_mask = df['macd'] > 0
                short_mask = df['macd'] < 0
            long_masks.append(long_mask)
            short_masks.append(short_mask)

        # Bollinger Bands
        if p.bb_enabled and 'bb_upper' in df.columns:
            if p.bb_mode == 'squeeze':
                long_masks.append(df['close'] < df['bb_lower'])
                short_masks.append(df['close'] > df['bb_upper'])
            else:  # breakout
                long_masks.append(df['close'] > df['bb_upper'])
                short_masks.append(df['close'] < df['bb_lower'])

        # Stochastic Entry — cross into oversold/overbought
        if p.stoch_entry_enabled and 'stoch_entry_k' in df.columns:
            k = df['stoch_entry_k']
            long_masks.append((k < p.stoch_entry_oversold) & (k > k.shift(1)))
            short_masks.append((k > p.stoch_entry_overbought) & (k < k.shift(1)))

        # CCI
        if p.cci_enabled and 'cci' in df.columns:
            long_masks.append(df['cci'] < p.cci_oversold)
            short_masks.append(df['cci'] > p.cci_overbought)

        # Williams %R
        if p.willr_enabled and 'willr' in df.columns:
            long_masks.append(df['willr'] < p.willr_oversold)
            short_masks.append(df['willr'] > p.willr_overbought)

        # OBV
        if p.obv_enabled and 'obv_ma' in df.columns:
            long_masks.append(df['obv'] > df['obv_ma'])
            short_masks.append(df['obv'] < df['obv_ma'])

        # Donchian Channel — breakout
        if p.donchian_enabled and 'donchian_upper' in df.columns:
            long_masks.append(df['close'] > df['donchian_upper'].shift(1))
            short_masks.append(df['close'] < df['donchian_lower'].shift(1))

        # Keltner Channel — mean reversion
        if p.keltner_enabled and 'keltner_upper' in df.columns:
            long_masks.append(df['close'] < df['keltner_lower'])
            short_masks.append(df['close'] > df['keltner_upper'])

        # Parabolic SAR
        if p.psar_enabled and 'psar' in df.columns:
            long_masks.append(df['close'] > df['psar'])
            short_masks.append(df['close'] < df['psar'])

        # Ichimoku Cloud — price vs non-shifted cloud + tenkan/kijun cross
        if p.ichi_enabled and 'ichi_senkou_a' in df.columns:
            cloud_top    = df[['ichi_senkou_a', 'ichi_senkou_b_line']].max(axis=1)
            cloud_bottom = df[['ichi_senkou_a', 'ichi_senkou_b_line']].min(axis=1)
            long_masks.append(
                (df['close'] > cloud_top) & (df['ichi_tenkan_sen'] > df['ichi_kijun_sen'])
            )
            short_masks.append(
                (df['close'] < cloud_bottom) & (df['ichi_tenkan_sen'] < df['ichi_kijun_sen'])
            )

        # Hull MA — direction
        if p.hull_enabled and 'hull_ma' in df.columns:
            long_masks.append(df['hull_ma'] > df['hull_ma'].shift(1))
            short_masks.append(df['hull_ma'] < df['hull_ma'].shift(1))

        # TRIX — signal line crossover
        if p.trix_enabled and 'trix_line' in df.columns:
            trix_above = df['trix_line'] > df['trix_signal_line']
            trix_above_prev = df['trix_line'].shift(1) > df['trix_signal_line'].shift(1)
            long_masks.append(trix_above & ~trix_above_prev)
            short_masks.append(~trix_above & trix_above_prev)

        long_signal = self._combine_condition_masks(long_masks, p.entry_operator, df.index)
        short_signal = self._combine_condition_masks(short_masks, p.entry_operator, df.index)

        if p.trade_direction == TradeDirection.LONG_ONLY:
            short_signal = pd.Series(False, index=df.index)
        elif p.trade_direction == TradeDirection.SHORT_ONLY:
            long_signal  = pd.Series(False, index=df.index)

        df['entry_long']  = long_signal.fillna(False)
        df['entry_short'] = short_signal.fillna(False)

        return df

    def generate_exit_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p  = self.params

        exit_long_masks: list[pd.Series] = []
        exit_short_masks: list[pd.Series] = []

        if p.pamrp_exit_enabled:
            exit_long_masks.append(df['pamrp_exit'] > p.pamrp_exit_long)
            exit_short_masks.append(df['pamrp_exit'] < p.pamrp_exit_short)

        if p.stoch_rsi_exit_enabled and 'stoch_k' in df.columns:
            exit_long_masks.append(df['stoch_k'] > p.stoch_rsi_overbought)
            exit_short_masks.append(df['stoch_k'] < p.stoch_rsi_oversold)

        if p.ma_exit_enabled and 'exit_ma_fast' in df.columns:
            exit_long_masks.append(df['exit_ma_fast'] < df['exit_ma_slow'])
            exit_short_masks.append(df['exit_ma_fast'] > df['exit_ma_slow'])

        if p.bbwp_exit_enabled and 'bbwp' in df.columns:
            exit_long_masks.append(df['bbwp'] > p.bbwp_exit_threshold)
            exit_short_masks.append(df['bbwp'] < p.bbwp_exit_threshold)

        exit_long = self._combine_condition_masks(exit_long_masks, p.exit_operator, df.index)
        exit_short = self._combine_condition_masks(exit_short_masks, p.exit_operator, df.index)

        df['exit_long_signal']  = exit_long.fillna(False)
        df['exit_short_signal'] = exit_short.fillna(False)

        return df

    def generate_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df = self.generate_entry_signals(df)
        df = self.generate_exit_signals(df)
        return df
