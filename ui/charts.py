"""
ui/charts.py
============
Pure chart-factory functions.  Every function accepts data and returns a
Plotly Figure object.  No Streamlit calls live here — all st.plotly_chart()
calls stay in the tab modules.

NOTE: create_heatmap lives in ui/tabs/heatmap.py because it depends on
      BacktestEngine and params_to_strategy, avoiding a circular import.

CHANGES (TradingView-style interaction):
  - _chart_layout() now sets dragmode='pan', crosshair spikes, hovermode
  - PLOTLY_CONFIG / PLOTLY_CONFIG_STATIC exported for tab modules
  - fixedrange=True removed from all time-series charts
  - Range selector buttons added to price + equity charts
  - Thin rangeslider on price chart for overview navigation
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

from src.montecarlo import MonteCarloResult
from src.indicators import (
    rsi_hidden_divergence,
    hpdr_bands,
    rsi as compute_rsi,
)


# ─────────────────────────────────────────────────────────────────────────────
# Plotly config dicts — import and pass to every st.plotly_chart() call
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_CONFIG = {
    'scrollZoom': True,                # scroll wheel = zoom
    'displayModeBar': True,            # always show toolbar
    'modeBarButtonsToAdd': [
        'drawline', 'drawopenpath', 'eraseshape',
    ],
    'modeBarButtonsToRemove': [
        'lasso2d', 'select2d', 'autoScale2d',
    ],
    'displaylogo': False,
}

# For small categorical charts (DOW, walkforward, heatmaps) where zoom is pointless
PLOTLY_CONFIG_STATIC = {
    'scrollZoom': False,
    'displayModeBar': False,
    'displaylogo': False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Crosshair spike defaults
# ─────────────────────────────────────────────────────────────────────────────

_SPIKE_DEFAULTS = dict(
    spikemode='across',
    spikethickness=0.5,
    spikedash='solid',
    spikecolor='#64748b',
    spikesnap='cursor',
    showspikes=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared layout helper
# ─────────────────────────────────────────────────────────────────────────────

def _chart_layout(height: int = 280, crosshair: bool = False, **kwargs) -> dict:
    """
    Base Plotly layout with TradingView-style defaults.

    Args:
        height:    chart height in px
        crosshair: if True, enable spike lines (crosshair) on x/y axes
        **kwargs:  override any default (e.g. showlegend=True, hovermode='closest')
    """
    defaults = dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,14,20,0.8)',
        height=height,
        showlegend=False,
        margin=dict(l=50, r=20, t=10, b=30),
        font=dict(size=9),
        # ── Interaction ──
        dragmode='pan',             # drag = pan; shift+drag = zoom box
        hovermode='closest',      # unified tooltip on time-series
    )

    if crosshair:
        defaults['xaxis'] = defaults.get('xaxis', {})
        defaults['yaxis'] = defaults.get('yaxis', {})
        defaults['xaxis'].update(**_SPIKE_DEFAULTS)
        defaults['yaxis'].update(**_SPIKE_DEFAULTS)

    defaults.update(kwargs)
    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# Range selector presets (reusable)
# ─────────────────────────────────────────────────────────────────────────────

_RANGE_SELECTOR_STYLE = dict(
    bgcolor='rgba(26,31,46,0.8)',
    activecolor='#3b82f6',
    bordercolor='#2d3548',
    borderwidth=1,
    font=dict(size=9, color='#e2e8f0'),
)

_RANGE_BUTTONS_FULL = [
    dict(count=1, label='1M', step='month', stepmode='backward'),
    dict(count=3, label='3M', step='month', stepmode='backward'),
    dict(count=6, label='6M', step='month', stepmode='backward'),
    dict(count=1, label='YTD', step='year', stepmode='todate'),
    dict(count=1, label='1Y', step='year', stepmode='backward'),
    dict(step='all', label='All'),
]

_RANGE_BUTTONS_SHORT = [
    dict(count=3, label='3M', step='month', stepmode='backward'),
    dict(count=6, label='6M', step='month', stepmode='backward'),
    dict(count=1, label='1Y', step='year', stepmode='backward'),
    dict(step='all', label='All'),
]


# ─────────────────────────────────────────────────────────────────────────────
# Price chart
# ─────────────────────────────────────────────────────────────────────────────

def create_price_chart_with_trades(df: pd.DataFrame, trades=None, bands=None) -> go.Figure:
    """
    Price chart with optional trade markers and HPDR rainbow cone overlay.

    Y-axis is always locked to the actual price range so HPDR band traces
    (which extend well beyond price on trending instruments) never push
    the axis into unusable territory.
    """
    fig = go.Figure()

    # ── HPDR rainbow cone — drawn BEFORE candles so it sits behind ────────────
    if bands is not None:
        idx = df.index

        ZONES = [
            ('2.0', '2.5', 'rgba(180,30,30,0.18)'),
            ('1.5', '2.0', 'rgba(220,110,30,0.18)'),
            ('1.0', '1.5', 'rgba(200,185,0,0.18)'),
            ('0.5', '1.0', 'rgba(60,190,80,0.18)'),
            ('0.0', '0.5', 'rgba(0,195,180,0.20)'),
        ]

        def _get(key):
            s = bands.get(key)
            return s.reindex(idx) if s is not None else None

        center = _get('center')
        upper = {z: _get(f'upper_{z}') for z in ['0.5', '1.0', '1.5', '2.0', '2.5']}
        lower = {z: _get(f'lower_{z}') for z in ['0.5', '1.0', '1.5', '2.0', '2.5']}

        for inner_z, outer_z, fill_rgba in ZONES:
            u_inner = upper.get(inner_z) if inner_z != '0.0' else center
            u_outer = upper.get(outer_z)
            if u_inner is None or u_outer is None:
                continue
            fig.add_trace(go.Scatter(x=idx, y=u_outer, mode='lines',
                line=dict(width=0.5, color=fill_rgba.replace('0.18', '0.4').replace('0.20', '0.4')),
                showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=idx, y=u_inner, mode='lines',
                line=dict(width=0), fill='tonexty', fillcolor=fill_rgba,
                showlegend=False, hoverinfo='skip'))

        for inner_z, outer_z, fill_rgba in ZONES:
            l_inner = lower.get(inner_z) if inner_z != '0.0' else center
            l_outer = lower.get(outer_z)
            if l_inner is None or l_outer is None:
                continue
            fig.add_trace(go.Scatter(x=idx, y=l_inner, mode='lines',
                line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=idx, y=l_outer, mode='lines',
                line=dict(width=0.5, color=fill_rgba.replace('0.18', '0.4').replace('0.20', '0.4')),
                fill='tonexty', fillcolor=fill_rgba, showlegend=False, hoverinfo='skip'))

        if center is not None:
            fig.add_trace(go.Scatter(x=idx, y=center, mode='lines',
                line=dict(color='rgba(255,255,255,0.35)', width=1, dash='dot'),
                name='HPDR Center', showlegend=True, hoverinfo='skip'))

        for z, color in [('0.5', 'rgba(0,195,180,0.6)'), ('1.0', 'rgba(60,190,80,0.5)'),
                          ('1.5', 'rgba(200,185,0,0.5)'), ('2.0', 'rgba(220,110,30,0.5)'),
                          ('2.5', 'rgba(180,30,30,0.5)')]:
            pct = {'0.5': '38%', '1.0': '68%', '1.5': '87%', '2.0': '95%', '2.5': '99%'}[z]
            for side, arr in [('upper', upper), ('lower', lower)]:
                s = arr.get(z)
                if s is not None:
                    fig.add_trace(go.Scatter(x=idx, y=s, mode='lines',
                        line=dict(color=color, width=0.8),
                        name=f'HPDR ±{z}σ ({pct})' if side == 'upper' else None,
                        showlegend=(side == 'upper'),
                        hovertemplate=f'±{z}σ: %{{y:.2f}}<extra></extra>'))

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#10b981', decreasing_line_color='#ef4444',
        increasing_fillcolor='rgba(16,185,129,0.3)', decreasing_fillcolor='rgba(239,68,68,0.3)',
        name='Price'))

# ── Trade vertical lines (2 traces, not N shapes) ─────────────────────────
    #   Draws all entries as one scatter trace, all exits as another,
    #   using None-gap segments for vertical lines. This is O(1) traces
    #   instead of O(N) shapes — massive rendering speedup for 50+ trades.
    if trades:
        price_min = float(df['low'].min())
        price_max = float(df['high'].max())
        pad = (price_max - price_min) * 0.05
        y_lo, y_hi = price_min - pad, price_max + pad

        same_bar_dates = set()
        for t in trades:
            if t.exit_date and t.entry_date == t.exit_date:
                same_bar_dates.add(t.entry_date)

        # Build entry segments: for each trade, 2 points + None gap
        entry_x, entry_y, entry_colors, entry_hover = [], [], [], []
        exit_x, exit_y, exit_hover = [], [], []

        for t in trades:
            # Entry segment
            entry_x.extend([t.entry_date, t.entry_date, None])
            entry_y.extend([y_lo, y_hi, None])
            c = '#10b981' if t.direction == 'long' else '#ef4444'
            entry_colors.append(c)
            label = f"{'▲ Long' if t.direction == 'long' else '▼ Short'} @ ${t.entry_price:.2f}"
            entry_hover.extend([label, label, None])

            # Exit segment
            if t.exit_date:
                exit_x.extend([t.exit_date, t.exit_date, None])
                exit_y.extend([y_lo, y_hi, None])
                label = f"Exit @ ${t.exit_price:.2f} | ${t.pnl:+.2f} ({t.exit_reason})"
                exit_hover.extend([label, label, None])

        # Split entries into long vs short traces (different colors)
        long_x, long_y, long_h = [], [], []
        short_x, short_y, short_h = [], [], []
        for t in trades:
            is_same = t.entry_date in same_bar_dates
            dst_x = long_x if t.direction == 'long' else short_x
            dst_y = long_y if t.direction == 'long' else short_y
            dst_h = long_h if t.direction == 'long' else short_h
            dst_x.extend([t.entry_date, t.entry_date, None])
            dst_y.extend([y_lo, y_hi, None])
            label = f"{'▲ Long' if t.direction == 'long' else '▼ Short'} @ ${t.entry_price:.2f}"
            dst_h.extend([label, label, None])

        if long_x:
            fig.add_trace(go.Scattergl(
                x=long_x, y=long_y, mode='lines',
                line=dict(color='rgba(16,185,129,0.5)', width=1),
                text=long_h, hoverinfo='text',
                name='Long Entries', showlegend=False,
            ))
        if short_x:
            fig.add_trace(go.Scattergl(
                x=short_x, y=short_y, mode='lines',
                line=dict(color='rgba(239,68,68,0.5)', width=1),
                text=short_h, hoverinfo='text',
                name='Short Entries', showlegend=False,
            ))
        if exit_x:
            fig.add_trace(go.Scattergl(
                x=exit_x, y=exit_y, mode='lines',
                line=dict(color='rgba(245,158,11,0.5)', width=1),
                text=exit_hover, hoverinfo='text',
                name='Exits', showlegend=False,
            ))
    # ── Y-axis locked to price range — HPDR bands must not expand this ────────
    price_min = float(df['low'].min())
    price_max = float(df['high'].max())
    pad = (price_max - price_min) * 0.05
    y_range = [price_min - pad, price_max + pad]

    show_legend = bands is not None
    fig.update_layout(
        **_chart_layout(400, showlegend=show_legend,
                        legend=dict(orientation='h', y=1.06, font=dict(size=8), traceorder='normal')),
        xaxis_rangeslider_visible=False,
        xaxis_rangeslider=dict(
            visible=False,
            thickness=0.04,
            bgcolor='rgba(10,14,20,0.5)',
            bordercolor='#2d3548',
            borderwidth=1,
        ),
        xaxis_rangeselector=dict(
            buttons=_RANGE_BUTTONS_FULL,
            x=0, y=1.06,
            **_RANGE_SELECTOR_STYLE,
        ),
    )
    fig.update_xaxes(gridcolor='rgba(45,53,72,0.3)')
    fig.update_yaxes(gridcolor='rgba(45,53,72,0.3)', range=y_range)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RSI Hidden Divergence chart
# ─────────────────────────────────────────────────────────────────────────────

def create_rsi_divergence_chart(df: pd.DataFrame, p: dict) -> go.Figure:
    """
    Two-panel chart:
      Row 1 (65%) — price candlesticks with bull/bear divergence arrows
      Row 2 (35%) — RSI line with OB/OS bands and divergence dots

    Signals are placed at the *confirmation* bar (pivot_right bars after the
    actual swing), not retroactively — zero look-ahead bias.
    """
    length = p['rsi_div_length']
    rsi_series = compute_rsi(df['close'], length)
    hidden_bull, hidden_bear = rsi_hidden_divergence(
        df['close'], 
        rsi_length=length,
        pivot_left=p['rsi_div_pivot_left'],
        pivot_right=p['rsi_div_pivot_right'],
        lookback_pivots=p.get('rsi_div_lookback_pivots', 3),
    )

    bull = hidden_bull.values.astype(bool)
    bear = hidden_bear.values.astype(bool)
    bull_idx = df.index[bull]
    bear_idx = df.index[bear]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.65, 0.35])

    # Row 1 — price
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#10b981', decreasing_line_color='#ef4444',
        increasing_fillcolor='rgba(16,185,129,0.3)', decreasing_fillcolor='rgba(239,68,68,0.3)',
        name='Price', showlegend=True), row=1, col=1)

    offset = (df['high'].max() - df['low'].min()) * 0.015
    if len(bull_idx) > 0:
        fig.add_trace(go.Scatter(x=bull_idx, y=df['low'][bull] - offset, mode='markers',
            marker=dict(color='#10b981', size=10, symbol='triangle-up', line=dict(width=1, color='white')),
            name='Hidden Bull', showlegend=True,
            hovertemplate='Hidden Bull<br>%{x}<extra></extra>'), row=1, col=1)

    if len(bear_idx) > 0:
        fig.add_trace(go.Scatter(x=bear_idx, y=df['high'][bear] + offset, mode='markers',
            marker=dict(color='#ef4444', size=10, symbol='triangle-down', line=dict(width=1, color='white')),
            name='Hidden Bear', showlegend=True,
            hovertemplate='Hidden Bear<br>%{x}<extra></extra>'), row=1, col=1)

    # Row 2 — RSI
    fig.add_trace(go.Scatter(x=df.index, y=rsi_series, mode='lines',
        line=dict(color='#f59e0b', width=1.5), name='RSI', showlegend=True), row=2, col=1)

    fig.add_hline(y=p.get('rsi_div_ob', 70), line_dash='dot', line_color='rgba(239,68,68,0.4)', row=2, col=1)
    fig.add_hline(y=p.get('rsi_div_os', 30), line_dash='dot', line_color='rgba(16,185,129,0.4)', row=2, col=1)
    fig.add_hrect(y0=p.get('rsi_div_os', 30), y1=p.get('rsi_div_ob', 70),
                  fillcolor='rgba(100,116,139,0.05)', line_width=0, row=2, col=1)

    if len(bull_idx) > 0:
        fig.add_trace(go.Scatter(x=bull_idx, y=rsi_series[bull], mode='markers',
            marker=dict(color='#10b981', size=8, symbol='circle', line=dict(width=1, color='white')),
            showlegend=False, hovertemplate='Hidden Bull<br>RSI: %{y:.1f}<br>%{x}<extra></extra>'), row=2, col=1)

    if len(bear_idx) > 0:
        fig.add_trace(go.Scatter(x=bear_idx, y=rsi_series[bear], mode='markers',
            marker=dict(color='#ef4444', size=8, symbol='circle', line=dict(width=1, color='white')),
            showlegend=False, hovertemplate='Hidden Bear<br>RSI: %{y:.1f}<br>%{x}<extra></extra>'), row=2, col=1)

    fig.update_layout(**_chart_layout(400, showlegend=True,
        legend=dict(orientation='h', y=1.06, font=dict(size=8))),
        xaxis_rangeslider_visible=False)
    fig.update_xaxes(gridcolor='rgba(45,53,72,0.3)')
    fig.update_yaxes(gridcolor='rgba(45,53,72,0.3)')
    fig.update_yaxes(title_text='RSI', range=[0, 100], row=2, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Equity / drawdown charts
# ─────────────────────────────────────────────────────────────────────────────

def create_equity_chart(results) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=results.equity_curve.index, y=results.equity_curve.values, mode='lines',
        line=dict(color='#3b82f6', width=2), fill='tozeroy', fillcolor='rgba(59,130,246,0.1)', name='MTM Equity'), row=1, col=1)
    fig.add_trace(go.Scatter(x=results.realized_equity.index, y=results.realized_equity.values, mode='lines',
        line=dict(color='#64748b', width=1, dash='dot'), name='Realized'), row=1, col=1)
    peak = results.equity_curve.expanding().max()
    dd = (results.equity_curve - peak) / peak * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode='lines', line=dict(color='#ef4444', width=2),
        fill='tozeroy', fillcolor='rgba(239,68,68,0.15)', name='Drawdown %'), row=2, col=1)

    fig.update_layout(**_chart_layout(340, showlegend=True, legend=dict(orientation='h', y=1.12, font=dict(size=8))))
    fig.update_xaxes(rangeselector=dict(buttons=_RANGE_BUTTONS_SHORT, **_RANGE_SELECTOR_STYLE))
    fig.update_xaxes(gridcolor='rgba(45,53,72,0.3)')
    fig.update_yaxes(gridcolor='rgba(45,53,72,0.3)')
    return fig


def create_stitched_equity_chart(equity: pd.Series) -> go.Figure:
    if equity is None or len(equity) == 0:
        return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode='lines',
        line=dict(color='#10b981', width=2), fill='tozeroy',
        fillcolor='rgba(16,185,129,0.1)', name='Stitched OOS'), row=1, col=1)
    peak = equity.expanding().max()
    dd = (equity - peak) / peak * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode='lines',
        line=dict(color='#ef4444', width=2), fill='tozeroy',
        fillcolor='rgba(239,68,68,0.15)', name='Drawdown'), row=2, col=1)

    fig.update_layout(**_chart_layout(280, showlegend=True, legend=dict(orientation='h', y=1.12, font=dict(size=8))))
    fig.update_xaxes(rangeselector=dict(buttons=_RANGE_BUTTONS_SHORT, **_RANGE_SELECTOR_STYLE))
    
    fig.update_xaxes(gridcolor='rgba(45,53,72,0.3)')
    fig.update_yaxes(gridcolor='rgba(45,53,72,0.3)')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Optimisation charts
# ─────────────────────────────────────────────────────────────────────────────

def create_walkforward_chart(wf_folds) -> go.Figure:
    """Train vs OOS per fold. Categorical — no zoom needed."""
    if not wf_folds:
        return go.Figure()
    folds = [f"Fold {f.fold_num}" for f in wf_folds]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=folds, y=[f.train_value for f in wf_folds], name='Train', marker_color='#3b82f6', opacity=0.8))
    fig.add_trace(go.Bar(x=folds, y=[f.test_value for f in wf_folds], name='OOS', marker_color='#10b981', opacity=0.8))
    fig.update_layout(**_chart_layout(250, crosshair=False, showlegend=True, barmode='group',
                                      legend=dict(orientation='h', y=1.12), hovermode='closest'))
    fig.update_xaxes(type='category', categoryorder='array', categoryarray=folds)
    return fig


def create_optimization_chart(trials_df: pd.DataFrame, metric: str) -> go.Figure:
    if trials_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trials_df['trial_number'], y=trials_df['value'], mode='markers',
                             marker=dict(color='rgba(59,130,246,0.4)', size=5), name='Trial'))
    fig.add_trace(go.Scatter(x=trials_df['trial_number'], y=trials_df['value'].expanding().max(),
                             mode='lines', line=dict(color='#10b981', width=2), name='Best so far'))
    fig.update_layout(**_chart_layout(220, showlegend=True,
                      legend=dict(orientation='h', y=1.12, font=dict(size=8))),
                      xaxis_title='Trial', yaxis_title=metric)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Multi-asset
# ─────────────────────────────────────────────────────────────────────────────

def create_multi_asset_chart(results_dict: dict) -> go.Figure:
    fig = go.Figure()
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    for i, (sym, res) in enumerate(results_dict.items()):
        norm = (res.equity_curve / res.equity_curve.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode='lines',
                                 name=sym, line=dict(color=colors[i % len(colors)], width=2)))
    fig.update_layout(**_chart_layout(300, showlegend=True, legend=dict(orientation='h', y=1.1)),
                      yaxis_title='Return %')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo charts
# ─────────────────────────────────────────────────────────────────────────────

def create_mc_confidence_chart(mc: MonteCarloResult) -> go.Figure:
    fig = go.Figure()
    n_steps = len(mc.equity_bands.get('50%', []))
    x = list(range(n_steps))
    if '5%' in mc.equity_bands and '95%' in mc.equity_bands:
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['95%'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['5%'], mode='lines', line=dict(width=0),
                                 fill='tonexty', fillcolor='rgba(59,130,246,0.15)', name='5–95%'))
    if '25%' in mc.equity_bands and '75%' in mc.equity_bands:
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['75%'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['25%'], mode='lines', line=dict(width=0),
                                 fill='tonexty', fillcolor='rgba(59,130,246,0.3)', name='25–75%'))
    if '50%' in mc.equity_bands:
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['50%'], mode='lines',
                                 line=dict(color='#3b82f6', width=2), name='Median'))
    fig.update_layout(**_chart_layout(280, showlegend=True, legend=dict(orientation='h', y=1.12, font=dict(size=8))))
    fig.update_xaxes(title_text='Step', gridcolor='rgba(45,53,72,0.3)')
    fig.update_yaxes(title_text='Equity $', gridcolor='rgba(45,53,72,0.3)')
    return fig


def create_mc_histogram(values, title: str = '', xaxis_title: str = '') -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=50, marker_color='#3b82f6', opacity=0.7))
    p5 = np.percentile(values, 5)
    p50 = np.percentile(values, 50)
    p95 = np.percentile(values, 95)
    for val, label, color in [(p5, '5%', '#64748b'), (p50, '50%', '#f59e0b'), (p95, '95%', '#64748b')]:
        fig.add_vline(x=val, line_dash='dash', line_color=color,
                      annotation_text=f"{label}: {val:,.0f}" if abs(val) > 100 else f"{label}: {val:.1f}%")
    fig.update_layout(**_chart_layout(220, hovermode='closest'), xaxis_title=xaxis_title)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Calendar charts
# ─────────────────────────────────────────────────────────────────────────────

def create_dow_chart(dow_df: pd.DataFrame) -> go.Figure:
    """
    Day-of-week bar chart with significance markers (* p<0.05, ** p<0.01)
    and win rate error bars showing 95% Wilson CI.
    """
    if dow_df.empty:
        return go.Figure()
    colors = ['#ef4444' if v < 0 else '#10b981' for v in dow_df['Avg %']]
    win_rates = [float(w.replace('%', '')) for w in dow_df['Win Rate']]
    # Build bar labels: value + significance marker
    has_sig = 'Sig' in dow_df.columns
    bar_text = [
        f"{v:+.4f}%{s}" for v, s in zip(
            dow_df['Avg %'],
            dow_df['Sig'] if has_sig else [''] * len(dow_df),
        )
    ]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.65, 0.35], subplot_titles=['Average Daily Return %  (* p<0.05  ** p<0.01)', 'Win Rate % with 95% CI'])
    fig.add_trace(go.Bar(x=dow_df['Day'], y=dow_df['Avg %'], marker_color=colors,
        text=bar_text, textposition='outside', name='Avg Return'), row=1, col=1)
    # Win rate bars with Wilson CI error bars
    wr_error_minus = wr_error_plus = None
    if 'WR CI Low' in dow_df.columns:
        ci_lows  = [float(str(v).replace('%', '')) for v in dow_df['WR CI Low']]
        ci_highs = [float(str(v).replace('%', '')) for v in dow_df['WR CI High']]
        wr_error_minus = [wr - lo for wr, lo in zip(win_rates, ci_lows)]
        wr_error_plus  = [hi - wr for wr, hi in zip(win_rates, ci_highs)]
    error_y = dict(type='data', symmetric=False,
                   array=wr_error_plus, arrayminus=wr_error_minus,
                   color='#94a3b8', thickness=1.5, width=4) if wr_error_plus else None
    fig.add_trace(go.Bar(x=dow_df['Day'], y=win_rates, marker_color='#3b82f6', opacity=0.7,
        text=[f"{w:.1f}%" for w in win_rates], textposition='outside',
        error_y=error_y,
        name='Win Rate'), row=2, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='#64748b', row=2, col=1)
    fig.update_layout(**_chart_layout(400, crosshair=False, showlegend=False, hovermode='closest',
                                      margin=dict(l=50, r=20, t=40, b=30)), bargap=0.3)
    fig.update_xaxes(type='category')
    return fig


def create_monthly_bar_chart(monthly_df: pd.DataFrame) -> go.Figure:
    """Monthly seasonality bar chart with significance markers and win rate CIs."""
    if monthly_df.empty:
        return go.Figure()
    colors = ['#ef4444' if v < 0 else '#10b981' for v in monthly_df['Avg %']]
    win_rates = [float(w.replace('%', '')) for w in monthly_df['Win Rate']]
    has_sig = 'Sig' in monthly_df.columns
    bar_text = [
        f"{v:+.2f}%{s}" for v, s in zip(
            monthly_df['Avg %'],
            monthly_df['Sig'] if has_sig else [''] * len(monthly_df),
        )
    ]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.65, 0.35], subplot_titles=['Average Monthly Return %  (* p<0.05  ** p<0.01)', 'Win Rate % with 95% CI'])
    fig.add_trace(go.Bar(x=monthly_df['Month'], y=monthly_df['Avg %'], marker_color=colors,
        text=bar_text, textposition='outside', name='Avg Return'), row=1, col=1)
    fig.add_trace(go.Bar(x=monthly_df['Month'], y=win_rates, marker_color='#3b82f6', opacity=0.7,
        text=[f"{w:.1f}%" for w in win_rates], textposition='outside', name='Win Rate'), row=2, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='#64748b', row=2, col=1)
    fig.update_layout(**_chart_layout(400, crosshair=False, showlegend=False, hovermode='closest',
                                      margin=dict(l=50, r=20, t=40, b=30)), bargap=0.3)
    fig.update_xaxes(type='category')
    return fig


def create_monthly_heatmap(heatmap_df: pd.DataFrame) -> go.Figure:
    """Year × Month heatmap. Fixed-range is appropriate here."""
    if heatmap_df.empty:
        return go.Figure()
    z = heatmap_df.values
    z_display = np.where(np.isnan(z.astype(float)), None, z)
    text = np.where(np.isnan(z.astype(float)), '',
        [[f"{v:.1f}%" if v is not None else '' for v in row] for row in z])
    fig = go.Figure(data=go.Heatmap(z=z_display, x=heatmap_df.columns.tolist(),
        y=[str(y) for y in heatmap_df.index], colorscale='RdYlGn', zmid=0,
        text=text, texttemplate='%{text}', showscale=True,
        colorbar=dict(title='Return %', thickness=12)))
    fig.update_layout(**_chart_layout(max(250, len(heatmap_df) * 38 + 60),
                                      crosshair=False, hovermode='closest'))
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def create_dom_chart(dom_df: pd.DataFrame) -> go.Figure:
    """Day-of-month chart. Categorical — no zoom needed."""
    if dom_df.empty:
        return go.Figure()
    colors = ['#ef4444' if v < 0 else '#10b981' for v in dom_df['Avg %']]
    win_rates = [float(w.replace('%', '')) for w in dom_df['Win Rate']]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=dom_df['Day of Month'], y=dom_df['Avg %'],
        marker_color=colors, name='Avg Return %', opacity=0.85), secondary_y=False)
    fig.add_trace(go.Scatter(x=dom_df['Day of Month'], y=win_rates, mode='lines+markers',
        line=dict(color='#f59e0b', width=2), marker=dict(size=5), name='Win Rate %'), secondary_y=True)
    fig.add_hline(y=0, line_dash='dot', line_color='#64748b', secondary_y=False)
    fig.update_layout(**_chart_layout(280, crosshair=False, showlegend=True,
                                      legend=dict(orientation='h', y=1.1), hovermode='closest'))
    fig.update_xaxes(title_text='Day of Month', dtick=1)
    fig.update_yaxes(title_text='Avg Return %', secondary_y=False)
    fig.update_yaxes(title_text='Win Rate %', secondary_y=True)
    return fig


def create_hourly_chart(hourly_df: pd.DataFrame) -> go.Figure:
    """Hourly returns bar chart. Categorical — no zoom needed."""
    if hourly_df is None or hourly_df.empty:
        return go.Figure()
    colors = ['#ef4444' if v < 0 else '#10b981' for v in hourly_df['Avg %']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hourly_df['Hour'], y=hourly_df['Avg %'], marker_color=colors,
        text=[f"{v:+.4f}%" for v in hourly_df['Avg %']], textposition='outside'))
    fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
    fig.update_layout(**_chart_layout(260, crosshair=False, hovermode='closest'), bargap=0.2)
    fig.update_xaxes(type='category')
    return fig


def create_return_distribution_chart(dist) -> go.Figure:
    """Return distribution histogram with VaR lines."""
    if not dist.bins:
        return go.Figure()
    colors = ['#ef4444' if b < 0 else '#10b981' for b in dist.bins]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dist.bins, y=dist.counts, marker_color=colors, opacity=0.75, name='Frequency'))
    vlines = [
        (dist.mean,             f'Mean {dist.mean:+.3f}%',                    '#f59e0b'),
        (dist.mean - dist.std,  f'−1σ {dist.mean - dist.std:.3f}%',           '#94a3b8'),
        (dist.mean + dist.std,  f'+1σ {dist.mean + dist.std:.3f}%',           '#94a3b8'),
        (dist.var_95,           f'VaR 95% {dist.var_95:.3f}%',                '#ef4444'),
        (dist.var_99,           f'VaR 99% {dist.var_99:.3f}%',                '#dc2626'),
    ]
    for val, label, color in vlines:
        fig.add_vline(x=val, line_dash='dash', line_color=color,
                      annotation_text=label, annotation_position='top')
    fig.update_layout(**_chart_layout(300, crosshair=False, showlegend=False, hovermode='closest',
                                      margin=dict(l=50, r=20, t=40, b=30)),
                      xaxis_title='Daily Return %', yaxis_title='Frequency')
    return fig


def create_quarterly_chart(quarterly_df: pd.DataFrame) -> go.Figure:
    """Q1–Q4 seasonality bar chart with win rate panel."""
    if quarterly_df.empty:
        return go.Figure()
    colors = ['#ef4444' if v < 0 else '#10b981' for v in quarterly_df['Avg %']]
    win_rates = [float(w.split('%')[0]) for w in quarterly_df['Win Rate']]
    sig_text = [
        f"{v:+.4f}%{s}" for v, s in zip(quarterly_df['Avg %'], quarterly_df['Sig'])
    ]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.65, 0.35], subplot_titles=['Average Daily Return % (by Quarter)', 'Win Rate %'])
    fig.add_trace(go.Bar(x=quarterly_df['Quarter'], y=quarterly_df['Avg %'],
        marker_color=colors, text=sig_text, textposition='outside', name='Avg Return'), row=1, col=1)
    fig.add_trace(go.Bar(x=quarterly_df['Quarter'], y=win_rates, marker_color='#3b82f6', opacity=0.7,
        text=[f"{w:.1f}%" for w in win_rates], textposition='outside', name='Win Rate'), row=2, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='#64748b', row=2, col=1)
    fig.update_layout(**_chart_layout(360, crosshair=False, showlegend=False, hovermode='closest',
                                      margin=dict(l=50, r=20, t=40, b=30)), bargap=0.3)
    fig.update_xaxes(type='category')
    return fig


def create_yearly_chart(yearly_df: pd.DataFrame) -> go.Figure:
    """Year-by-year total return bars with Sharpe overlay."""
    if yearly_df.empty:
        return go.Figure()
    colors = ['#ef4444' if v < 0 else '#10b981' for v in yearly_df['Total Return %']]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.6, 0.4], subplot_titles=['Annual Total Return %', 'Annualised Sharpe'])
    fig.add_trace(go.Bar(
        x=yearly_df['Year'].astype(str), y=yearly_df['Total Return %'],
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in yearly_df['Total Return %']],
        textposition='outside', name='Total Return'), row=1, col=1)
    sharpe_colors = ['#ef4444' if v < 0 else '#10b981' for v in yearly_df['Sharpe (ann.)']]
    fig.add_trace(go.Bar(
        x=yearly_df['Year'].astype(str), y=yearly_df['Sharpe (ann.)'],
        marker_color=sharpe_colors, opacity=0.8,
        text=[f"{v:.2f}" for v in yearly_df['Sharpe (ann.)']],
        textposition='outside', name='Sharpe'), row=2, col=1)
    fig.add_hline(y=0, line_dash='dot', line_color='#64748b', row=1, col=1)
    fig.add_hline(y=0, line_dash='dot', line_color='#64748b', row=2, col=1)
    fig.update_layout(**_chart_layout(400, crosshair=False, showlegend=False, hovermode='closest',
                                      margin=dict(l=50, r=20, t=40, b=30)), bargap=0.2)
    fig.update_xaxes(type='category')
    return fig


def create_rolling_dow_chart(rolling_dow_df: pd.DataFrame) -> go.Figure:
    """
    Year-over-year DOW mean return — one line per day.
    A flat or reversing line = edge is unreliable.
    """
    if rolling_dow_df.empty:
        return go.Figure()
    palette = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4']
    fig = go.Figure()
    for i, col in enumerate(rolling_dow_df.columns):
        series = rolling_dow_df[col].dropna()
        if series.empty:
            continue
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=series.index.astype(str), y=series.values,
            mode='lines+markers', name=col,
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
        ))
    fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
    fig.update_layout(**_chart_layout(300, crosshair=False, showlegend=True,
                                      legend=dict(orientation='h', y=1.12),
                                      hovermode='closest'),
                      xaxis_title='Year', yaxis_title='Avg Daily Return %')
    return fig


def create_autocorr_chart(autocorr) -> go.Figure:
    """ACF bar chart with 95% confidence band."""
    if not autocorr.lags:
        return go.Figure()
    colors = ['#ef4444' if abs(v) > abs(autocorr.conf_upper) else '#3b82f6'
              for v in autocorr.acf_values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=autocorr.lags, y=autocorr.acf_values,
        marker_color=colors, opacity=0.8, name='ACF',
    ))
    # 95% CI bands
    n = len(autocorr.lags)
    fig.add_trace(go.Scatter(
        x=autocorr.lags + autocorr.lags[::-1],
        y=[autocorr.conf_upper] * n + [autocorr.conf_lower] * n,
        fill='toself', fillcolor='rgba(100,116,139,0.15)',
        line=dict(color='rgba(100,116,139,0.4)', dash='dash'),
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_hline(y=0, line_dash='solid', line_color='#64748b', line_width=0.5)
    fig.update_layout(**_chart_layout(260, crosshair=False, showlegend=False, hovermode='closest'),
                      xaxis_title='Lag (days)', yaxis_title='Autocorrelation',
                      xaxis=dict(dtick=1))
    return fig


def create_day_hour_heatmap(day_hour_df: pd.DataFrame) -> go.Figure:
    """Day-of-week × Hour mean return heatmap (intraday)."""
    if day_hour_df is None or day_hour_df.empty:
        return go.Figure()
    z = day_hour_df.values.astype(float)
    text = [[f"{v:.4f}%" if not np.isnan(v) else '' for v in row] for row in z]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[f"{int(h):02d}:00" for h in day_hour_df.columns],
        y=day_hour_df.index.tolist(),
        colorscale='RdYlGn', zmid=0,
        text=text, texttemplate='%{text}',
        showscale=True, colorbar=dict(title='Avg %', thickness=12),
    ))
    fig.update_layout(**_chart_layout(max(200, len(day_hour_df) * 45 + 60),
                                      crosshair=False, hovermode='closest'))
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig