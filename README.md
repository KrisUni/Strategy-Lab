# Strategy Lab

Quantitative trading strategy backtesting and optimization platform. Built in Python/Streamlit. Free and open-source stack — no paid APIs, no black-box libraries.

## What It Does

- **Backtest** strategies with institutional-grade fill simulation (gap-through stops, entry-bar SL/TP, slippage, commission)
- **Optimize** parameters via Bayesian optimization (Optuna TPE) with rolling or anchored walk-forward validation
- **Validate** edge with in-sample permutation testing (p-value against randomized data)
- **Stress test** via Monte Carlo simulation (trade shuffle, return bootstrap, noise injection)
- **Analyze** calendar seasonality (day-of-week, monthly, hourly, day-of-month, return distribution)
- **Compare** strategy vs buy-and-hold with beta/alpha/correlation
- **Scan** parameter sensitivity via 2D heatmaps
- **Run** the same strategy across multiple assets simultaneously

## Project Structure

```
app.py                      # Streamlit entry point (orchestrator only)
src/
├── __init__.py              # Package version
├── backtest/__init__.py     # Backtest engine (fill sim, equity tracking, metrics)
├── strategy/__init__.py     # StrategyParams, SignalGenerator, entry/exit logic
├── indicators/__init__.py   # Technical indicators (PAMRP, BBWP, RSI, MACD, ADX, etc.)
├── optimize/__init__.py     # Bayesian optimizer, walk-forward, pinned params
├── montecarlo/__init__.py   # Monte Carlo simulation (3 methods)
├── permutation/__init__.py  # In-sample permutation test
├── analytics/__init__.py    # Calendar analytics
├── data/__init__.py         # Data fetching (yfinance), validation, sample generation
ui/
├── session.py               # Session state + default params
├── sidebar.py               # Sidebar controls (data, indicators, exits)
├── helpers.py               # params_to_strategy, display helpers, beta/alpha
├── styles.py                # Global CSS
├── charts.py                # Plotly chart factories
├── tabs/
│   ├── backtest.py
│   ├── optimize.py
│   ├── compare.py
│   ├── montecarlo.py
│   ├── calendar.py
│   ├── heatmap.py
│   └── multi_asset.py
tests/
│   └── test_toolkit.py
scripts/
│   ├── run_backtest.py
│   ├── run_optimization.py
│   └── generate_report.py
Makefile
requirements.txt
```

## Setup

```bash
make setup       # creates venv, installs deps
make run-ui      # launches Streamlit app
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Dependencies

Python 3.10+. Core dependencies:

| Package    | Purpose              |
|------------|----------------------|
| streamlit  | UI framework         |
| pandas     | Data manipulation    |
| numpy      | Numerical compute    |
| plotly     | Charts               |
| optuna     | Bayesian optimization|
| yfinance   | Market data (free)   |
| scipy      | Statistics           |

Full list in `requirements.txt`.

## CLI Usage

```bash
make run-backtest                                    # SPY, daily, 2020-2024
make run-backtest SYMBOL=TSLA INTERVAL=15m           # custom
make run-optimize SYMBOL=QQQ TRIALS=500              # Bayesian optimization
make run-quick                                       # fast optimization (50 trials)
make test                                            # run unit tests
```

## Available Indicators

**Entry filters:** PAMRP, BBWP, ADX, MA Trend (SMA/EMA/WMA), RSI, Volume, Supertrend, VWAP, MACD

**Exit methods:** Stop Loss, Take Profit, Trailing Stop, ATR Trailing, PAMRP Exit, Stoch RSI Exit, Time Exit, MA Exit, BBWP Exit

**Visual overlays:** HPDR Bands, RSI Hidden Divergence

All indicators off by default. Enable what you need in the sidebar.

## Backtest Engine

Key execution rules:

- **Entries:** previous bar's signal → current bar's open + slippage
- **Stop orders:** trigger on high/low, fill at order level or bar open if gap-through (whichever is worse)
- **Entry-bar stops:** SL/TP checked on the same bar as entry
- **Time exits:** execute at bar open (no look-ahead)
- **SL/TP conflict:** closer to open wins
- **Open position at EOD:** force-closed at last close
- **Kelly sizing:** uses realized trade stats after 20 trades, not theoretical parameters

Metrics: total return, CAGR, Sharpe, Sortino, Calmar, max drawdown, profit factor, win rate, expectancy, payoff ratio, MAE/MFE, consecutive wins/losses, time in market.

## Optimization

- **Simple split:** train/test with configurable split ratio
- **Walk-forward:** rolling (fixed window) or anchored (expanding window), N folds
- **Deployment params:** last fold's best params (not best OOS fold — avoids selection bias)
- **Pinned params:** freeze specific parameters to reduce search space dimensionality
- **Diagnostics:** efficiency ratio, parameter stability CV, trial budget warnings, failed trial %

## Permutation Test

Tests whether the optimized strategy has a real edge or is just overfitting noise.

1. Optimize on real data → record metric
2. Shuffle returns N times, re-optimize each → build null distribution
3. p-value = fraction of permuted metrics ≥ real metric

p < 0.05 → statistically significant edge.

## Data Sources

| Source        | Coverage                         | Cost |
|---------------|----------------------------------|------|
| Yahoo Finance | Equities, ETFs, crypto, FX       | Free |
| CSV upload    | Anything with OHLCV columns      | —    |
| Sample data   | Synthetic GBM with configurable vol | —  |

yfinance interval limits: 1m (7d), 5m (60d), 1h (730d), 1d+ (full history).


