"""
ui/lightweight_chart.py
=======================
HTML generator for the main TradingView-style price chart.

This uses TradingView's Lightweight Charts via CDN inside a Streamlit HTML
component so the primary zoom/pan interaction feels closer to a native trading
terminal than Plotly candlesticks do inside Streamlit.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd

from ui.state_migration import migrate_legacy_pamrp_params


_CHART_HEIGHT = 520


def _is_intraday(index: pd.Index) -> bool:
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        return False
    return not index.normalize().equals(index)


def _chart_time(ts: pd.Timestamp, intraday: bool) -> str | int:
    ts = pd.Timestamp(ts)
    if intraday:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp())
    return ts.strftime("%Y-%m-%d")


def _series_points(index: Iterable[pd.Timestamp], values: Iterable[Any], intraday: bool) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for ts, value in zip(index, values):
        if pd.isna(value):
            continue
        points.append({"time": _chart_time(pd.Timestamp(ts), intraday), "value": float(value)})
    return points


def create_lightweight_price_chart_html(
    df: pd.DataFrame,
    trades=None,
    params: dict[str, Any] | None = None,
    indicator_df: pd.DataFrame | None = None,
) -> str:
    p = migrate_legacy_pamrp_params(params or {})
    intraday = _is_intraday(df.index)

    candles = [
        {
            "time": _chart_time(pd.Timestamp(ts), intraday),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for ts, row in df[["open", "high", "low", "close"]].iterrows()
    ]

    overlays: list[dict[str, Any]] = []
    legend_items = ["Price"]
    idf = indicator_df

    if idf is not None and not idf.empty:
        if p.get("ma_trend_enabled") and {"ma_fast", "ma_slow"}.issubset(idf.columns):
            ma_type = p.get("ma_type", "sma").upper()
            overlays.extend([
                {
                    "name": f"{ma_type}({p.get('ma_fast_length', 50)})",
                    "color": "#3b82f6",
                    "lineWidth": 2,
                    "data": _series_points(idf.index, idf["ma_fast"], intraday),
                },
                {
                    "name": f"{ma_type}({p.get('ma_slow_length', 200)})",
                    "color": "#f59e0b",
                    "lineWidth": 2,
                    "data": _series_points(idf.index, idf["ma_slow"], intraday),
                },
            ])
            legend_items.extend([overlays[-2]["name"], overlays[-1]["name"]])

        if p.get("supertrend_enabled") and {"supertrend", "st_direction"}.issubset(idf.columns):
            bull = idf["supertrend"].where(idf["st_direction"] > 0)
            bear = idf["supertrend"].where(idf["st_direction"] < 0)
            overlays.extend([
                {
                    "name": "Supertrend Bull",
                    "color": "#10b981",
                    "lineWidth": 2,
                    "data": _series_points(idf.index, bull, intraday),
                },
                {
                    "name": "Supertrend Bear",
                    "color": "#ef4444",
                    "lineWidth": 2,
                    "data": _series_points(idf.index, bear, intraday),
                },
            ])
            legend_items.extend(["Supertrend Bull", "Supertrend Bear"])

        if p.get("vwap_enabled") and "vwap" in idf.columns:
            overlays.append({
                "name": "VWAP",
                "color": "#a855f7",
                "lineWidth": 2,
                "data": _series_points(idf.index, idf["vwap"], intraday),
            })
            legend_items.append("VWAP")

    markers: list[dict[str, Any]] = []
    if trades:
        for trade in trades:
            entry_position = "belowBar" if trade.direction == "long" else "aboveBar"
            entry_shape = "arrowUp" if trade.direction == "long" else "arrowDown"
            entry_color = "#10b981" if trade.direction == "long" else "#ef4444"
            markers.append({
                "time": _chart_time(pd.Timestamp(trade.entry_date), intraday),
                "position": entry_position,
                "shape": entry_shape,
                "color": entry_color,
                "text": "Long" if trade.direction == "long" else "Short",
            })

            if trade.exit_date:
                exit_position = "aboveBar" if trade.direction == "long" else "belowBar"
                exit_text = f"Exit {trade.pnl:+.0f}"
                markers.append({
                    "time": _chart_time(pd.Timestamp(trade.exit_date), intraday),
                    "position": exit_position,
                    "shape": "circle",
                    "color": "#f59e0b",
                    "text": exit_text,
                })

    payload = {
        "height": _CHART_HEIGHT,
        "candles": candles,
        "overlays": overlays,
        "markers": markers,
        "legend": legend_items,
        "intraday": intraday,
    }

    payload_json = json.dumps(payload, separators=(",", ":"))

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: transparent;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #e5edf7;
    }}
    .wrap {{
      background: rgba(10, 14, 20, 0.88);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 12px;
      overflow: hidden;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 10px 12px 6px 12px;
      font-size: 12px;
      color: #a4b4ca;
      border-bottom: 1px solid rgba(148, 163, 184, 0.12);
      background: rgba(7, 11, 18, 0.72);
    }}
    .legend span::before {{
      content: "•";
      margin-right: 6px;
      color: #4f8cff;
    }}
    #chart {{
      width: 100%;
      height: { _CHART_HEIGHT }px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="legend" id="legend"></div>
    <div id="chart"></div>
  </div>
  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <script>
    const payload = {payload_json};
    const legend = document.getElementById("legend");
    payload.legend.forEach((item) => {{
      const span = document.createElement("span");
      span.textContent = item;
      legend.appendChild(span);
    }});

    const container = document.getElementById("chart");
    if (typeof LightweightCharts === "undefined") {{
      container.innerHTML = '<div style="padding:16px;color:#fca5a5;">Unable to load the TradingView-style chart component. The detailed Plotly chart is still available below.</div>';
      throw new Error("Lightweight Charts failed to load");
    }}
    const chart = LightweightCharts.createChart(container, {{
      width: container.clientWidth,
      height: payload.height,
      layout: {{
        background: {{ color: "rgba(10, 14, 20, 0.88)" }},
        textColor: "#a4b4ca",
        fontSize: 12,
      }},
      grid: {{
        vertLines: {{ color: "rgba(45,53,72,0.25)" }},
        horzLines: {{ color: "rgba(45,53,72,0.25)" }},
      }},
      crosshair: {{
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: {{ color: "rgba(100,116,139,0.8)", width: 1, style: 0 }},
        horzLine: {{ color: "rgba(100,116,139,0.5)", width: 1, style: 0 }},
      }},
      rightPriceScale: {{
        borderColor: "rgba(148,163,184,0.18)",
        autoScale: true,
      }},
      timeScale: {{
        borderColor: "rgba(148,163,184,0.18)",
        rightOffset: 6,
        barSpacing: payload.intraday ? 7 : 9,
        minBarSpacing: 2,
        fixLeftEdge: false,
        fixRightEdge: false,
        lockVisibleTimeRangeOnResize: false,
        timeVisible: payload.intraday,
        secondsVisible: false,
      }},
      handleScroll: {{
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      }},
      handleScale: {{
        mouseWheel: true,
        pinch: true,
        axisPressedMouseMove: true,
        axisDoubleClickReset: true,
      }},
    }});

    const candleSeries = chart.addCandlestickSeries({{
      upColor: "#10b981",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#10b981",
      wickDownColor: "#ef4444",
      priceLineVisible: false,
      lastValueVisible: true,
    }});
    candleSeries.setData(payload.candles);

    if (payload.markers.length) {{
      candleSeries.setMarkers(payload.markers);
    }}

    payload.overlays.forEach((overlay) => {{
      const line = chart.addLineSeries({{
        color: overlay.color,
        lineWidth: overlay.lineWidth || 2,
        priceLineVisible: false,
        lastValueVisible: false,
      }});
      line.setData(overlay.data);
    }});

    chart.timeScale().fitContent();

    const ro = new ResizeObserver((entries) => {{
      const rect = entries[0].contentRect;
      chart.applyOptions({{ width: Math.floor(rect.width), height: payload.height }});
    }});
    ro.observe(container);
  </script>
</body>
</html>
"""
