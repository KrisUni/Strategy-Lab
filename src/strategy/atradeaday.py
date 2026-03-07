"""
src/strategy/atradeaday.py
==========================
A Trade A Day — standalone backtest engine.

Strategy rules:
  1. Each day, identify the FIRST candle (or the candle at entry_time if set).
     Mark its high and low. That's the analysis.
  2. From the NEXT candle onwards, look for a Fair Value Gap that breaks
     through day_high (bullish) or day_low (bearish).
     FVG = 3-candle pattern: c0.high < c2.low (bull) or c0.low > c2.high (bear)
     The middle candle must cross the day level.
  3. Wait for price to pull back into the FVG zone.
  4. Wait for an engulfing candle that completely covers the pullback candle.
  5. Enter at the close of the engulfing candle.
  6. SL = low/high of the first FVG candle.
  7. TP = entry +/- risk x rr_ratio (default 3:1).
  8. One trade per day maximum.

Works with any timeframe: 1m, 5m, 15m, 30m, 60m, 1d.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List

from src.backtest import BacktestResults, Trade


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ATradeADayParams:
    rr_ratio: float = 3.0
    risk_per_trade: float = 100.0
    commission_pct: float = 0.05
    use_first_candle: bool = True   # If True, always use first candle of day
    entry_time: str = "09:30"       # Only used if use_first_candle is False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_opening_bar(day_df: pd.DataFrame, params: ATradeADayParams) -> Optional[pd.Series]:
    """
    Get the opening bar for the day.
    If use_first_candle=True, just take the first bar.
    Otherwise try to match entry_time — fall back to first bar if not found.
    """
    if params.use_first_candle or day_df.empty:
        return day_df.iloc[0]

    mask = day_df.index.strftime('%H:%M') == params.entry_time
    if mask.any():
        return day_df[mask].iloc[0]

    # Fallback: first candle of the day
    return day_df.iloc[0]


def _find_fvg_break(
    bars: pd.DataFrame,
    day_high: float,
    day_low: float,
) -> Optional[Tuple[str, int, float, float, float]]:
    """
    Scan for a Fair Value Gap confirming a break of day_high or day_low.

    FVG conditions (relaxed to catch more setups):
      Bullish: c0.high < c2.low  (gap between wicks)
               AND (c1 touched day_high OR c2 closed above day_high)
      Bearish: c0.low  > c2.high
               AND (c1 touched day_low  OR c2 closed below day_low)

    Returns (direction, c2_index, fvg_top, fvg_bottom, sl_price) or None.
    """
    for i in range(2, len(bars)):
        c0 = bars.iloc[i - 2]
        c1 = bars.iloc[i - 1]
        c2 = bars.iloc[i]

        # ── Bullish FVG ───────────────────────────────────────────────────────
        if c2['low'] > c0['high']:
            if c1['high'] >= day_high or c2['close'] > day_high or c1['close'] > day_high:
                fvg_top    = c2['low']
                fvg_bottom = c0['high']
                sl_price   = c0['low']
                return ('long', i, fvg_top, fvg_bottom, sl_price)

        # ── Bearish FVG ───────────────────────────────────────────────────────
        if c2['high'] < c0['low']:
            if c1['low'] <= day_low or c2['close'] < day_low or c1['close'] < day_low:
                fvg_top    = c0['low']
                fvg_bottom = c2['high']
                sl_price   = c0['high']
                return ('short', i, fvg_top, fvg_bottom, sl_price)

    return None


def _find_engulfing_entry(
    bars: pd.DataFrame,
    direction: str,
    fvg_top: float,
    fvg_bottom: float,
) -> Optional[Tuple[int, pd.Series, pd.Series]]:
    """
    Look for pullback into FVG zone then engulfing candle entry.

    Relaxed: engulf covers the body (open-close) of the pullback candle,
    not necessarily the full wicks — catches more real-world setups.
    """
    in_pullback  = False
    pullback_bar = None

    for i in range(len(bars)):
        bar = bars.iloc[i]

        if not in_pullback:
            if direction == 'long':
                if bar['low'] <= fvg_top and bar['close'] >= fvg_bottom:
                    in_pullback  = True
                    pullback_bar = bar
            else:
                if bar['high'] >= fvg_bottom and bar['close'] <= fvg_top:
                    in_pullback  = True
                    pullback_bar = bar
        else:
            pb_body_high = max(pullback_bar['open'], pullback_bar['close'])
            pb_body_low  = min(pullback_bar['open'], pullback_bar['close'])

            if direction == 'long':
                # Full engulf OR strong close above pullback high
                if (bar['close'] > pb_body_high and bar['open'] < pb_body_low) or \
                   (bar['close'] > pullback_bar['high'] and bar['open'] < pb_body_high):
                    return (i, pullback_bar, bar)
                # Reset if price leaves the zone
                if bar['low'] > fvg_top * 1.002:
                    in_pullback  = False
                    pullback_bar = None
            else:
                # Full engulf OR strong close below pullback low
                if (bar['close'] < pb_body_low and bar['open'] > pb_body_high) or \
                   (bar['close'] < pullback_bar['low'] and bar['open'] > pb_body_low):
                    return (i, pullback_bar, bar)
                # Reset if price leaves the zone
                if bar['high'] < fvg_bottom * 0.998:
                    in_pullback  = False
                    pullback_bar = None

    return None


def _simulate_exit(
    bars: pd.DataFrame,
    direction: str,
    sl_price: float,
    tp_price: float,
) -> Tuple[float, str, int]:
    """
    Walk through remaining bars to find SL or TP hit.
    Gap-aware: if bar opens past the level, fill at open.
    Returns (exit_price, exit_reason, bars_held).
    """
    for i, (_, bar) in enumerate(bars.iterrows()):
        sl_hit = tp_hit = False
        sl_fill = tp_fill = None

        if direction == 'long':
            if bar['low'] <= sl_price:
                sl_hit  = True
                sl_fill = min(sl_price, bar['open'])
            if bar['high'] >= tp_price:
                tp_hit  = True
                tp_fill = max(tp_price, bar['open'])
        else:
            if bar['high'] >= sl_price:
                sl_hit  = True
                sl_fill = max(sl_price, bar['open'])
            if bar['low'] <= tp_price:
                tp_hit  = True
                tp_fill = min(tp_price, bar['open'])

        if sl_hit and tp_hit:
            sl_dist = abs(bar['open'] - sl_fill)
            tp_dist = abs(bar['open'] - tp_fill)
            if tp_dist <= sl_dist:
                return tp_fill, 'take_profit', i + 1
            else:
                return sl_fill, 'stop_loss', i + 1
        elif tp_hit:
            return tp_fill, 'take_profit', i + 1
        elif sl_hit:
            return sl_fill, 'stop_loss', i + 1

    last_close = bars.iloc[-1]['close'] if len(bars) > 0 else sl_price
    return last_close, 'end_of_data', len(bars)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_bars_per_year(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 252
    deltas  = pd.Series(df.index).diff().dropna()
    seconds = deltas.median().total_seconds()
    if seconds <= 120:      return 252 * 390
    elif seconds <= 600:    return 252 * 78
    elif seconds <= 1800:   return 252 * 26
    elif seconds <= 3600:   return 252 * 13
    elif seconds <= 7200:   return 252 * 7
    elif seconds <= 172800: return 252
    elif seconds <= 864000: return 52
    else:                   return 12


def _calculate_metrics(
    trades: List[Trade],
    equity_curve: pd.Series,
    realized_equity: pd.Series,
    initial_capital: float,
    commission_pct: float,
    bars_per_year: int,
) -> BacktestResults:

    if not trades:
        return BacktestResults(
            trades=[],
            equity_curve=equity_curve,
            realized_equity=realized_equity,
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            bars_per_year=bars_per_year,
        )

    winners = [t for t in trades if t.pnl > 0]
    losers  = [t for t in trades if t.pnl <= 0]

    total_return     = equity_curve.iloc[-1] - initial_capital
    total_return_pct = total_return / initial_capital * 100

    n_bars = len(equity_curve)
    cagr   = 0.0
    if n_bars > 1 and equity_curve.iloc[-1] > 0:
        cagr = ((equity_curve.iloc[-1] / initial_capital) ** (bars_per_year / n_bars) - 1) * 100

    num_trades    = len(trades)
    win_rate      = len(winners) / num_trades * 100
    gross_profit  = sum(t.pnl for t in winners) if winners else 0.0
    gross_loss    = abs(sum(t.pnl for t in losers)) if losers else 0.0
    profit_factor = min(gross_profit / gross_loss, 999.99) if gross_loss > 0 else (999.99 if gross_profit > 0 else 0.0)

    avg_winner     = float(np.mean([t.pnl     for t in winners])) if winners else 0.0
    avg_loser      = float(np.mean([t.pnl     for t in losers]))  if losers  else 0.0
    avg_winner_pct = float(np.mean([t.pnl_pct for t in winners])) if winners else 0.0
    avg_loser_pct  = float(np.mean([t.pnl_pct for t in losers]))  if losers  else 0.0
    avg_trade      = float(np.mean([t.pnl     for t in trades]))
    avg_bars       = float(np.mean([t.bars_held for t in trades]))

    wr_frac    = len(winners) / num_trades
    expectancy = wr_frac * avg_winner - (1.0 - wr_frac) * abs(avg_loser)
    payoff     = min(avg_winner / abs(avg_loser), 999.99) if avg_loser != 0 else 999.99

    peak       = equity_curve.expanding().max()
    drawdown   = equity_curve - peak
    max_dd     = float(drawdown.min())
    max_dd_pct = float((drawdown / peak).min() * 100) if peak.max() > 0 else 0.0

    in_dd      = drawdown < 0
    dd_groups  = (~in_dd).cumsum()
    longest_dd = int(in_dd.groupby(dd_groups).sum().max()) if in_dd.any() else 0

    returns        = equity_curve.pct_change().dropna()
    active_returns = returns[returns != 0]
    n_total        = len(returns)
    n_active       = len(active_returns)

    active_bpy = n_active * (bars_per_year / n_total) if n_total > 0 and n_active > 0 else bars_per_year

    sharpe  = float((active_returns.mean() / active_returns.std()) * np.sqrt(active_bpy)) \
              if n_active > 1 and active_returns.std() > 0 else 0.0

    neg_active = active_returns[active_returns < 0]
    sortino    = float((active_returns.mean() / neg_active.std()) * np.sqrt(active_bpy)) \
                 if len(neg_active) > 1 and neg_active.std() > 0 else sharpe

    calmar = abs(cagr / max_dd_pct) if max_dd_pct != 0 else 0.0

    max_cl = max_cw = cur_l = cur_w = 0
    for t in trades:
        if t.pnl <= 0:
            cur_l += 1; cur_w = 0; max_cl = max(max_cl, cur_l)
        else:
            cur_w += 1; cur_l = 0; max_cw = max(max_cw, cur_w)

    pct_in_market = (sum(t.bars_held for t in trades) / n_bars * 100) if n_bars > 0 else 0.0

    return BacktestResults(
        trades=trades,
        equity_curve=equity_curve,
        realized_equity=realized_equity,
        total_return=total_return,
        total_return_pct=total_return_pct,
        cagr=cagr,
        num_trades=num_trades,
        winners=len(winners),
        losers=len(losers),
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        payoff_ratio=payoff,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_winner=avg_winner,
        avg_loser=avg_loser,
        avg_winner_pct=avg_winner_pct,
        avg_loser_pct=avg_loser_pct,
        avg_trade=avg_trade,
        avg_bars_held=avg_bars,
        max_consecutive_losses=max_cl,
        max_consecutive_wins=max_cw,
        longest_drawdown_bars=longest_dd,
        pct_time_in_market=pct_in_market,
        avg_mae=0.0,
        avg_mfe=0.0,
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        bars_per_year=bars_per_year,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_atradeaday(df: pd.DataFrame, params: ATradeADayParams) -> BacktestResults:
    """
    Run the A Trade A Day strategy on any OHLCV DataFrame.

    Works with any timeframe loaded in the sidebar (1m, 5m, 15m, 60m, 1d etc).
    One trade per calendar day maximum.
    """
    if df is None or df.empty:
        empty = pd.Series(dtype=float)
        return BacktestResults(trades=[], equity_curve=empty, realized_equity=empty)

    df = df.copy()

    bars_per_year   = _estimate_bars_per_year(df)
    initial_capital = params.risk_per_trade * 50
    capital         = float(initial_capital)
    trades: List[Trade] = []

    equity_index = df.index.tolist()
    running_eq   = {ts: initial_capital for ts in equity_index}

    df['_date'] = df.index.normalize()

    for date, day_df in df.groupby('_date'):
        if len(day_df) < 4:
            continue

        opening_bar = _get_opening_bar(day_df, params)
        if opening_bar is None:
            continue

        day_high = opening_bar['high']
        day_low  = opening_bar['low']

        post_open = day_df[day_df.index > opening_bar.name]
        if len(post_open) < 3:
            continue

        fvg_result = _find_fvg_break(post_open, day_high, day_low)
        if fvg_result is None:
            continue

        direction, fvg_c2_pos, fvg_top, fvg_bottom, sl_price_raw = fvg_result

        post_fvg = post_open.iloc[fvg_c2_pos + 1:]
        if len(post_fvg) < 1:
            continue

        entry_result = _find_engulfing_entry(post_fvg, direction, fvg_top, fvg_bottom)
        if entry_result is None:
            continue

        _, _, engulf_bar = entry_result
        entry_price = float(engulf_bar['close'])
        sl_price    = float(sl_price_raw)

        if direction == 'long':
            risk_per_unit = entry_price - sl_price
            tp_price      = entry_price + risk_per_unit * params.rr_ratio
        else:
            risk_per_unit = sl_price - entry_price
            tp_price      = entry_price - risk_per_unit * params.rr_ratio

        # Skip if SL is nonsensical
        if risk_per_unit <= 0 or risk_per_unit > entry_price * 0.15:
            continue

        position_size_dollars = params.risk_per_trade / (risk_per_unit / entry_price)

        remaining = post_fvg[post_fvg.index > engulf_bar.name]

        if len(remaining) == 0:
            exit_price  = entry_price
            exit_reason = 'end_of_data'
            bars_held   = 0
            exit_ts     = engulf_bar.name
        else:
            exit_price, exit_reason, bars_held = _simulate_exit(
                remaining, direction, sl_price, tp_price
            )
            exit_idx = min(bars_held - 1, len(remaining) - 1)
            exit_ts  = remaining.index[exit_idx]

        if direction == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        pnl_pct -= params.commission_pct * 2
        pnl      = position_size_dollars * (pnl_pct / 100)
        capital += pnl

        try:
            entry_iloc = equity_index.index(engulf_bar.name)
            exit_iloc  = equity_index.index(exit_ts)
        except ValueError:
            entry_iloc = exit_iloc = 0

        trade = Trade(
            entry_idx    = entry_iloc,
            entry_date   = engulf_bar.name,
            entry_price  = entry_price,
            direction    = direction,
            size_dollars = position_size_dollars,
            exit_idx     = exit_iloc,
            exit_date    = exit_ts,
            exit_price   = exit_price,
            exit_reason  = exit_reason,
            pnl          = pnl,
            pnl_pct      = pnl_pct,
            bars_held    = bars_held,
            mae          = 0.0,
            mfe          = 0.0,
        )
        trades.append(trade)

        # Update equity from exit bar onwards
        updating = False
        for ts in equity_index:
            if ts == exit_ts:
                updating = True
            if updating:
                running_eq[ts] = capital

    equity_curve    = pd.Series(running_eq, index=equity_index)
    realized_equity = equity_curve.copy()

    return _calculate_metrics(
        trades, equity_curve, realized_equity,
        initial_capital, params.commission_pct, bars_per_year,
    )
