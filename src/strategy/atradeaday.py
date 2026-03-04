"""
src/strategy/atradeaday.py
==========================
A Trade A Day — standalone backtest engine.

Strategy rules (exact, from spec):
  1. Every morning, identify the first 5-min candle at `entry_time` (default 09:30).
  2. Mark that candle's high (day_high) and low (day_low). Analysis done.
  3. On the 1-min / 5-min chart from the NEXT candle onwards:
     a. Watch for a Fair Value Gap (FVG) that BREAKS through day_high or day_low.
        - FVG = 3-candle pattern where candle[i-2].high < candle[i].low  (bullish)
                                    or candle[i-2].low  > candle[i].high (bearish)
        - The middle candle (momentum candle) must cross the day level.
     b. Wait for price to PULL BACK into the FVG gap zone.
     c. Wait for an ENGULFING candle that completely covers the pullback candle.
     d. Enter at the CLOSE of the engulfing candle.
     e. Stop Loss   = low  of candle[i-2] (first FVG candle) for longs
                    = high of candle[i-2]                     for shorts
     f. Take Profit = entry ± (entry - SL) × rr_ratio  (default 3:1)
  4. Maximum ONE trade per calendar day. Walk away after entry.

Returns a BacktestResults object so the UI can display identically to the
main Backtest tab.
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
    rr_ratio: float = 3.0          # Risk/reward ratio for TP
    risk_per_trade: float = 100.0  # Fixed dollar risk per trade
    entry_time: str = "09:30"      # Time string of the 5-min candle to mark
    commission_pct: float = 0.05   # Round-trip commission %


# ─────────────────────────────────────────────────────────────────────────────
# Pattern detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_fvg_break(
    bars: pd.DataFrame,
    day_high: float,
    day_low: float,
) -> Optional[Tuple[str, int, float, float, float]]:
    """
    Scan `bars` for a Fair Value Gap that confirms a break of day_high or day_low.

    An FVG is a 3-candle pattern:
      - Bullish: bars[i-2].high < bars[i].low  (gap between wicks of c0 and c2)
      - Bearish: bars[i-2].low  > bars[i].high

    The middle candle (bars[i-1]) must have broken through the day level to
    confirm the break is genuine and not a random FVG elsewhere.

    Returns
    -------
    (direction, fvg_c2_pos, fvg_top, fvg_bottom, sl_price)
      direction   : 'long' or 'short'
      fvg_c2_pos  : integer position (iloc) of the 3rd FVG candle in `bars`
      fvg_top     : upper bound of the FVG zone (retest target)
      fvg_bottom  : lower bound of the FVG zone
      sl_price    : stop loss price (worst wick of first FVG candle)

    Returns None if no qualifying FVG is found.
    """
    for i in range(2, len(bars)):
        c0 = bars.iloc[i - 2]
        c1 = bars.iloc[i - 1]  # momentum candle
        c2 = bars.iloc[i]

        # ── Bullish FVG ───────────────────────────────────────────────────────
        # Gap: c0.high < c2.low  (no overlap between wicks)
        # Confirmation: middle candle broke above day_high
        if c2['low'] > c0['high']:
            if c1['high'] > day_high or c1['close'] > day_high:
                fvg_top    = c2['low']    # top of the gap zone
                fvg_bottom = c0['high']   # bottom of the gap zone
                sl_price   = c0['low']    # SL below the first candle of the FVG
                return ('long', i, fvg_top, fvg_bottom, sl_price)

        # ── Bearish FVG ───────────────────────────────────────────────────────
        # Gap: c0.low > c2.high
        # Confirmation: middle candle broke below day_low
        if c2['high'] < c0['low']:
            if c1['low'] < day_low or c1['close'] < day_low:
                fvg_top    = c0['low']    # top of the gap zone
                fvg_bottom = c2['high']   # bottom of the gap zone
                sl_price   = c0['high']   # SL above the first candle of the FVG
                return ('short', i, fvg_top, fvg_bottom, sl_price)

    return None


def _find_engulfing_entry(
    bars: pd.DataFrame,
    direction: str,
    fvg_top: float,
    fvg_bottom: float,
) -> Optional[Tuple[int, pd.Series, pd.Series]]:
    """
    After the FVG is identified, scan for a pullback retest followed by
    an engulfing candle.

    A pullback is when price trades INTO the FVG zone:
      - Long:  bar.low  <= fvg_top    (price dips back into the gap)
      - Short: bar.high >= fvg_bottom (price bounces back into the gap)

    An engulfing candle completely covers the prior (pullback) candle:
      - Bullish engulf: open < pullback.low  AND close > pullback.high
      - Bearish engulf: open > pullback.high AND close < pullback.low

    Returns (bar_position, pullback_candle, engulf_candle) or None.
    """
    in_pullback = False
    pullback_bar = None

    for i in range(len(bars)):
        bar = bars.iloc[i]

        if not in_pullback:
            # Detect pullback into FVG zone
            if direction == 'long':
                if bar['low'] <= fvg_top:
                    in_pullback = True
                    pullback_bar = bar
            else:
                if bar['high'] >= fvg_bottom:
                    in_pullback = True
                    pullback_bar = bar

        else:
            # Look for engulfing candle over the pullback
            if direction == 'long':
                # Bullish engulf: open below pullback low, close above pullback high
                if bar['open'] < pullback_bar['low'] and bar['close'] > pullback_bar['high']:
                    return (i, pullback_bar, bar)
                # Reset pullback if price moves away from zone without engulfing
                if bar['low'] > fvg_top * 1.001:
                    in_pullback = False
                    pullback_bar = None
            else:
                # Bearish engulf: open above pullback high, close below pullback low
                if bar['open'] > pullback_bar['high'] and bar['close'] < pullback_bar['low']:
                    return (i, pullback_bar, bar)
                # Reset pullback if price moves away from zone without engulfing
                if bar['high'] < fvg_bottom * 0.999:
                    in_pullback = False
                    pullback_bar = None

    return None


def _simulate_exit(
    bars: pd.DataFrame,
    direction: str,
    sl_price: float,
    tp_price: float,
) -> Tuple[float, str, int]:
    """
    Walk through remaining bars and find when SL or TP is hit.

    Returns (exit_price, exit_reason, bars_held).
    If neither is hit before end of data, exits at last close (end_of_data).

    Uses gap-aware fills: if bar opens past the level, fill at open.
    """
    for i, (_, bar) in enumerate(bars.iterrows()):
        sl_hit = tp_hit = False
        sl_fill = tp_fill = None

        if direction == 'long':
            if bar['low'] <= sl_price:
                sl_hit  = True
                sl_fill = min(sl_price, bar['open'])  # gap-aware: open could gap below
            if bar['high'] >= tp_price:
                tp_hit  = True
                tp_fill = max(tp_price, bar['open'])  # gap-aware: open could gap above
        else:
            if bar['high'] >= sl_price:
                sl_hit  = True
                sl_fill = max(sl_price, bar['open'])
            if bar['low'] <= tp_price:
                tp_hit  = True
                tp_fill = min(tp_price, bar['open'])

        if sl_hit and tp_hit:
            # Both hit same bar: whichever is closer to the open wins
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

    # No exit hit — close at last available price
    last_close = bars.iloc[-1]['close'] if len(bars) > 0 else sl_price
    return last_close, 'end_of_data', len(bars)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (mirrors BacktestEngine._calculate_metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_metrics(
    trades: List[Trade],
    equity_curve: pd.Series,
    realized_equity: pd.Series,
    initial_capital: float,
    commission_pct: float,
) -> BacktestResults:
    """Build a full BacktestResults from a list of completed trades."""

    bars_per_year = 252 * 78  # 5-min bars in a trading year

    if not trades:
        return BacktestResults(
            trades=[],
            equity_curve=equity_curve if len(equity_curve) else pd.Series(dtype=float),
            realized_equity=realized_equity if len(realized_equity) else pd.Series(dtype=float),
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            bars_per_year=bars_per_year,
        )

    winners = [t for t in trades if t.pnl > 0]
    losers  = [t for t in trades if t.pnl <= 0]

    total_return     = equity_curve.iloc[-1] - initial_capital
    total_return_pct = total_return / initial_capital * 100

    n_bars = len(equity_curve)
    if n_bars > 1 and equity_curve.iloc[-1] > 0:
        cagr = (equity_curve.iloc[-1] / initial_capital) ** (bars_per_year / n_bars) - 1
        cagr *= 100
    else:
        cagr = 0.0

    num_trades = len(trades)
    win_rate   = len(winners) / num_trades * 100

    gross_profit = sum(t.pnl for t in winners) if winners else 0.0
    gross_loss   = abs(sum(t.pnl for t in losers)) if losers else 0.0
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

    # Drawdown
    peak          = equity_curve.expanding().max()
    drawdown      = equity_curve - peak
    max_dd        = float(drawdown.min())
    max_dd_pct    = float((drawdown / peak).min() * 100) if peak.max() > 0 else 0.0

    in_dd     = drawdown < 0
    dd_groups = (~in_dd).cumsum()
    longest_dd = int(in_dd.groupby(dd_groups).sum().max()) if in_dd.any() else 0

    # Sharpe / Sortino on active returns only
    returns        = equity_curve.pct_change().dropna()
    active_returns = returns[returns != 0]
    n_total        = len(returns)
    n_active       = len(active_returns)

    if n_total > 0 and n_active > 0:
        active_bpy = n_active * (bars_per_year / n_total)
    else:
        active_bpy = bars_per_year

    if n_active > 1 and active_returns.std() > 0:
        sharpe = float((active_returns.mean() / active_returns.std()) * np.sqrt(active_bpy))
    else:
        sharpe = 0.0

    neg_active = active_returns[active_returns < 0]
    sortino    = float((active_returns.mean() / neg_active.std()) * np.sqrt(active_bpy)) \
                 if len(neg_active) > 1 and neg_active.std() > 0 else sharpe

    calmar = abs(cagr / max_dd_pct) if max_dd_pct != 0 else 0.0

    # Consecutive wins/losses
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
    Run the A Trade A Day strategy on 5-min OHLCV data.

    Parameters
    ----------
    df     : pd.DataFrame with DatetimeIndex and columns open/high/low/close/volume.
             Expected to be 5-min bars covering at least one trading session.
    params : ATradeADayParams

    Returns
    -------
    BacktestResults — identical schema to what BacktestEngine returns, so the
    UI can render it using the exact same metric widgets.
    """
    if df is None or df.empty:
        empty = pd.Series(dtype=float)
        return BacktestResults(trades=[], equity_curve=empty, realized_equity=empty)

    df = df.copy()

    # Capital tracking
    initial_capital = params.risk_per_trade * 50  # Scale capital so risk = 2% per trade
    capital         = initial_capital
    trades: List[Trade] = []

    # Equity arrays keyed by bar timestamp
    equity_index  = df.index.tolist()
    mtm_equity    = {ts: initial_capital for ts in equity_index}
    real_equity   = {ts: initial_capital for ts in equity_index}

    # ── Iterate day by day ────────────────────────────────────────────────────
    df['_date'] = df.index.normalize()

    for date, day_df in df.groupby('_date'):

        # Find the opening candle at entry_time
        opening_mask = day_df.index.strftime('%H:%M') == params.entry_time
        if not opening_mask.any():
            continue

        opening_bar = day_df[opening_mask].iloc[0]
        day_high    = opening_bar['high']
        day_low     = opening_bar['low']

        # All bars AFTER the opening candle, same day
        post_open = day_df[day_df.index > opening_bar.name]
        if len(post_open) < 4:
            continue

        # Step 1 — Find an FVG that breaks through the day level
        fvg_result = _find_fvg_break(post_open, day_high, day_low)
        if fvg_result is None:
            continue

        direction, fvg_c2_pos, fvg_top, fvg_bottom, sl_price_raw = fvg_result

        # Step 2 — Scan bars after the FVG for pullback + engulfing entry
        post_fvg = post_open.iloc[fvg_c2_pos + 1:]
        if len(post_fvg) < 2:
            continue

        entry_result = _find_engulfing_entry(post_fvg, direction, fvg_top, fvg_bottom)
        if entry_result is None:
            continue

        _, _, engulf_bar = entry_result

        # Step 3 — Calculate trade levels
        entry_price = engulf_bar['close']
        sl_price    = sl_price_raw

        if direction == 'long':
            risk_per_unit = entry_price - sl_price
            tp_price      = entry_price + risk_per_unit * params.rr_ratio
        else:
            risk_per_unit = sl_price - entry_price
            tp_price      = entry_price - risk_per_unit * params.rr_ratio

        if risk_per_unit <= 0:
            continue

        # Step 4 — Position sizing: risk exactly risk_per_trade dollars
        position_size_dollars = params.risk_per_trade / (risk_per_unit / entry_price)

        # Step 5 — Simulate exit in remaining bars of the day (walk away)
        remaining = post_fvg[post_fvg.index > engulf_bar.name]
        if len(remaining) == 0:
            # No bars left in the day — exit at engulf close
            exit_price  = entry_price
            exit_reason = 'end_of_data'
            bars_held   = 0
        else:
            exit_price, exit_reason, bars_held = _simulate_exit(
                remaining, direction, sl_price, tp_price
            )

        # Step 6 — Calculate P&L
        if direction == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        # Subtract commission (round-trip)
        pnl_pct -= params.commission_pct * 2

        pnl     = position_size_dollars * (pnl_pct / 100)
        capital += pnl

        # Step 7 — Record trade
        exit_date = remaining.index[min(bars_held, len(remaining) - 1)] \
                    if len(remaining) > 0 else engulf_bar.name

        trade = Trade(
            entry_idx    = equity_index.index(engulf_bar.name) if engulf_bar.name in equity_index else 0,
            entry_date   = engulf_bar.name,
            entry_price  = entry_price,
            direction    = direction,
            size_dollars = position_size_dollars,
            exit_idx     = equity_index.index(exit_date) if exit_date in equity_index else 0,
            exit_date    = exit_date,
            exit_price   = exit_price,
            exit_reason  = exit_reason,
            pnl          = pnl,
            pnl_pct      = pnl_pct,
            bars_held    = bars_held,
            mae          = 0.0,
            mfe          = 0.0,
        )
        trades.append(trade)

        # Update realized equity from the exit bar onwards
        update_from = exit_date
        updating    = False
        running_capital = capital
        for ts in equity_index:
            if ts == update_from:
                updating = True
            if updating:
                real_equity[ts] = running_capital
                mtm_equity[ts]  = running_capital

    # Build equity curve series
    equity_curve    = pd.Series(mtm_equity,  index=equity_index)
    realized_equity = pd.Series(real_equity, index=equity_index)

    return _calculate_metrics(
        trades, equity_curve, realized_equity,
        initial_capital, params.commission_pct
    )
