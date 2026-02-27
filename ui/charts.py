"""
ui/charts.py
============
Pure chart-factory functions.  Every function accepts data and returns a
Plotly Figure object.  No Streamlit calls live here — all st.plotly_chart()
calls stay in the tab modules.

NOTE: create_heatmap lives in ui/tabs/heatmap.py because it depends on
      BacktestEngine and params_to_strategy, avoiding a circular import.
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
# Shared layout helper
# ─────────────────────────────────────────────────────────────────────────────

def _chart_layout(height: int = 280, **kwargs) -> dict:
    defaults = dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,14,20,0.8)',
        height=height,
        showlegend=False,
        margin=dict(l=50, r=20, t=10, b=30),
        font=dict(size=9),
    )
    defaults.update(kwargs)
    return defaults


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

    # ── Trade markers ─────────────────────────────────────────────────────────
    if trades:
        ed = [t.entry_date for t in trades]
        ep = [t.entry_price for t in trades]
        ec = ['#10b981' if t.direction == 'long' else '#ef4444' for t in trades]
        es = ['triangle-up' if t.direction == 'long' else 'triangle-down' for t in trades]
        el = [f"{'▲ Long' if t.direction=='long' else '▼ Short'} @ ${t.entry_price:.2f}" for t in trades]
        fig.add_trace(go.Scatter(x=ed, y=ep, mode='markers',
            marker=dict(color=ec, size=9, symbol=es, line=dict(width=1, color='white')),
            text=el, hoverinfo='text', name='Entries'))
        xt = [t for t in trades if t.exit_date]
        xd = [t.exit_date for t in xt]
        xp = [t.exit_price for t in xt]
        xc = ['#10b981' if t.pnl >= 0 else '#ef4444' for t in xt]
        xl = [f"Exit @ ${t.exit_price:.2f} | ${t.pnl:+.2f} ({t.exit_reason})" for t in xt]
        fig.add_trace(go.Scatter(x=xd, y=xp, mode='markers',
            marker=dict(color=xc, size=8, symbol='x', line=dict(width=1, color='white')),
            text=xl, hoverinfo='text', name='Exits'))

    # ── Y-axis locked to price range — HPDR bands must not expand this ────────
    price_min = float(df['low'].min())
    price_max = float(df['high'].max())
    pad = (price_max - price_min) * 0.05
    y_range = [price_min - pad, price_max + pad]

    show_legend = bands is not None
    fig.update_layout(
        **_chart_layout(320, showlegend=show_legend,
                        legend=dict(orientation='h', y=1.06, font=dict(size=8), traceorder='normal')),
        xaxis_rangeslider_visible=False)
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
    length = p.get('rsi_div_length', 14)
    piv_left = p.get('rsi_div_pivot_left', 5)
    piv_right = p.get('rsi_div_pivot_right', 5)

    rsi_series = compute_rsi(df['close'], length)
    bull, bear = rsi_hidden_divergence(df['close'], rsi_length=length,
                                       pivot_left=piv_left, pivot_right=piv_right)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06, row_heights=[0.65, 0.35])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#10b981', decreasing_line_color='#ef4444',
        increasing_fillcolor='rgba(16,185,129,0.3)', decreasing_fillcolor='rgba(239,68,68,0.3)',
        name='Price', showlegend=False), row=1, col=1)

    bull_idx = df.index[bull]
    if len(bull_idx) > 0:
        fig.add_trace(go.Scatter(x=bull_idx, y=df['low'][bull] * 0.998,
            mode='markers+text',
            marker=dict(symbol='triangle-up', color='#10b981', size=11, line=dict(width=1, color='white')),
            text=['HB'] * len(bull_idx), textposition='bottom center',
            textfont=dict(size=7, color='#10b981'), name='Hidden Bull', showlegend=True,
            hovertemplate='Hidden Bullish Div<br>%{x}<extra></extra>'), row=1, col=1)

    bear_idx = df.index[bear]
    if len(bear_idx) > 0:
        fig.add_trace(go.Scatter(x=bear_idx, y=df['high'][bear] * 1.002,
            mode='markers+text',
            marker=dict(symbol='triangle-down', color='#ef4444', size=11, line=dict(width=1, color='white')),
            text=['HBr'] * len(bear_idx), textposition='top center',
            textfont=dict(size=7, color='#ef4444'), name='Hidden Bear', showlegend=True,
            hovertemplate='Hidden Bearish Div<br>%{x}<extra></extra>'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=rsi_series, mode='lines',
        line=dict(color='#f59e0b', width=1.5), name='RSI', showlegend=False), row=2, col=1)

    for level, color in [(70, 'rgba(239,68,68,0.4)'), (30, 'rgba(16,185,129,0.4)'), (50, 'rgba(100,116,139,0.3)')]:
        fig.add_hline(y=level, line_dash='dot', line_color=color, line_width=1, row=2, col=1)

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
        fill='tozeroy', fillcolor='rgba(239,68,68,0.15)'), row=2, col=1)
    fig.update_layout(**_chart_layout(280, showlegend=True, legend=dict(orientation='h', y=1.12, font=dict(size=8))))
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
    fig.update_xaxes(gridcolor='rgba(45,53,72,0.3)')
    fig.update_yaxes(gridcolor='rgba(45,53,72,0.3)')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Optimisation charts
# ─────────────────────────────────────────────────────────────────────────────

def create_walkforward_chart(wf_folds) -> go.Figure:
    if not wf_folds:
        return go.Figure()
    folds = [f"Fold {f.fold_num}" for f in wf_folds]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=folds, y=[f.train_value for f in wf_folds], name='Train', marker_color='#3b82f6', opacity=0.8))
    fig.add_trace(go.Bar(x=folds, y=[f.test_value for f in wf_folds], name='OOS', marker_color='#10b981', opacity=0.8))
    fig.update_layout(**_chart_layout(250, showlegend=True, barmode='group',
                                      legend=dict(orientation='h', y=1.12), dragmode=False))
    fig.update_xaxes(type='category', fixedrange=True, categoryorder='array', categoryarray=folds)
    fig.update_yaxes(fixedrange=True)
    return fig


def create_optimization_chart(trials_df: pd.DataFrame, metric: str) -> go.Figure:
    if trials_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trials_df['trial_number'], y=trials_df['value'], mode='markers',
                             marker=dict(color='rgba(59,130,246,0.4)', size=5)))
    fig.add_trace(go.Scatter(x=trials_df['trial_number'], y=trials_df['value'].expanding().max(),
                             mode='lines', line=dict(color='#10b981', width=2)))
    fig.update_layout(**_chart_layout(200), xaxis_title="Trial", yaxis_title=metric)
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
                      yaxis_title="Return %")
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
                                 fill='tonexty', fillcolor='rgba(59,130,246,0.15)', name='5-95%'))
    if '25%' in mc.equity_bands and '75%' in mc.equity_bands:
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['75%'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['25%'], mode='lines', line=dict(width=0),
                                 fill='tonexty', fillcolor='rgba(59,130,246,0.3)', name='25-75%'))
    if '50%' in mc.equity_bands:
        fig.add_trace(go.Scatter(x=x, y=mc.equity_bands['50%'], mode='lines',
                                 line=dict(color='#3b82f6', width=2), name='Median'))
    fig.update_layout(**_chart_layout(250, showlegend=True, legend=dict(orientation='h', y=1.12, font=dict(size=8))))
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
    fig.update_layout(**_chart_layout(220), xaxis_title=xaxis_title)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Calendar charts
# ─────────────────────────────────────────────────────────────────────────────

def create_dow_chart(dow_df: pd.DataFrame) -> go.Figure:
    if dow_df.empty:
        return go.Figure()
    colors = ['#ef4444' if v < 0 else '#10b981' for v in dow_df['Avg %']]
    win_rates = [float(w.replace('%', '')) for w in dow_df['Win Rate']]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.65, 0.35], subplot_titles=['Average Daily Return %', 'Win Rate %'])
    fig.add_trace(go.Bar(x=dow_df['Day'], y=dow_df['Avg %'], marker_color=colors,
        text=[f"{v:+.4f}%" for v in dow_df['Avg %']], textposition='outside', name='Avg Return'), row=1, col=1)
    fig.add_trace(go.Bar(x=dow_df['Day'], y=win_rates, marker_color='#3b82f6', opacity=0.7,
        text=[f"{w:.1f}%" for w in win_rates], textposition='outside', name='Win Rate'), row=2, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='#64748b', row=2, col=1)
    fig.update_layout(**_chart_layout(380, showlegend=False), bargap=0.3)
    fig.update_xaxes(type='category', fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def create_monthly_bar_chart(monthly_df: pd.DataFrame) -> go.Figure:
    if monthly_df.empty:
        return go.Figure()
    colors = ['#ef4444' if v < 0 else '#10b981' for v in monthly_df['Avg %']]
    win_rates = [float(w.replace('%', '')) for w in monthly_df['Win Rate']]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.65, 0.35], subplot_titles=['Average Monthly Return %', 'Win Rate %'])
    fig.add_trace(go.Bar(x=monthly_df['Month'], y=monthly_df['Avg %'], marker_color=colors,
        text=[f"{v:+.2f}%" for v in monthly_df['Avg %']], textposition='outside', name='Avg Return'), row=1, col=1)
    fig.add_trace(go.Bar(x=monthly_df['Month'], y=win_rates, marker_color='#3b82f6', opacity=0.7,
        text=[f"{w:.1f}%" for w in win_rates], textposition='outside', name='Win Rate'), row=2, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='#64748b', row=2, col=1)
    fig.update_layout(**_chart_layout(380, showlegend=False), bargap=0.3)
    fig.update_xaxes(type='category', fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def create_monthly_heatmap(heatmap_df: pd.DataFrame) -> go.Figure:
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
    fig.update_layout(**_chart_layout(max(250, len(heatmap_df) * 38 + 60)))
    return fig


def create_dom_chart(dom_df: pd.DataFrame) -> go.Figure:
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
    fig.update_layout(**_chart_layout(280, showlegend=True, legend=dict(orientation='h', y=1.1)))
    fig.update_xaxes(title_text='Day of Month', dtick=1, fixedrange=True)
    fig.update_yaxes(title_text='Avg Return %', fixedrange=True, secondary_y=False)
    fig.update_yaxes(title_text='Win Rate %', fixedrange=True, secondary_y=True)
    return fig


def create_hourly_chart(hourly_df: pd.DataFrame) -> go.Figure:
    if hourly_df is None or hourly_df.empty:
        return go.Figure()
    colors = ['#ef4444' if v < 0 else '#10b981' for v in hourly_df['Avg %']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hourly_df['Hour'], y=hourly_df['Avg %'], marker_color=colors,
        text=[f"{v:+.4f}%" for v in hourly_df['Avg %']], textposition='outside'))
    fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
    fig.update_layout(**_chart_layout(260), bargap=0.2)
    fig.update_xaxes(type='category', fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def create_return_distribution_chart(dist) -> go.Figure:
    if not dist.bins:
        return go.Figure()
    colors = ['#ef4444' if b < 0 else '#10b981' for b in dist.bins]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dist.bins, y=dist.counts, marker_color=colors, opacity=0.75, name='Frequency'))
    for val, label, color in [
        (dist.mean, f'Mean {dist.mean:+.3f}%', '#f59e0b'),
        (dist.mean - dist.std, f'−1σ {dist.mean - dist.std:.3f}%', '#94a3b8'),
        (dist.mean + dist.std, f'+1σ {dist.mean + dist.std:.3f}%', '#94a3b8'),
    ]:
        fig.add_vline(x=val, line_dash='dash', line_color=color,
                      annotation_text=label, annotation_position='top')
    fig.update_layout(**_chart_layout(280, showlegend=False),
                      xaxis_title='Daily Return %', yaxis_title='Frequency')
    return fig
