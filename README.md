# Strategy Lab

> Rigorous quantitative backtesting and optimization in Python — walk-forward, permutation testing, Monte Carlo, and a Claude MCP server so you can drive the whole research loop from an LLM agent.

![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-22c55e?style=flat-square)
![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Optuna](https://img.shields.io/badge/optuna-TPE-0084ff?style=flat-square)
![MCP](https://img.shields.io/badge/MCP-Claude-7c3aed?style=flat-square)
![Backtesting](https://img.shields.io/badge/backtesting-engine-0ea5e9?style=flat-square)
![Walk-Forward](https://img.shields.io/badge/walk--forward-validation-f59e0b?style=flat-square)
![Permutation Test](https://img.shields.io/badge/permutation-test-ec4899?style=flat-square)
![Status](https://img.shields.io/badge/status-beta-eab308?style=flat-square)
![PRs](https://img.shields.io/badge/PRs-welcome-22c55e?style=flat-square)

**Free, open-source, zero paid APIs, zero black-box libraries.** Every backtest rule is in the repo — including the unpleasant ones.

---

## Why this exists

Strategy Lab is built for researchers who want to know whether their edge is real. The platform's job is to give you an honest answer — through realistic fills, leakage-free signals, walk-forward selection that doesn't introduce hindsight bias, and a permutation test that destroys signal cleanly while preserving bar structure.
Honest backtests are harder to fake an edge in. That's the point

---

## Features

### Backtesting
- Gap-aware stop fills (stops trigger on high/low, fill at order level or bar open if gapped through — whichever is worse)
- Entry-bar SL/TP evaluation (no free first bar)
- Explicit slippage and commission per fill — no implicit fills anywhere
- SL/TP conflict resolution (closer to open wins, deterministic)
- End-of-data force-close for equity-curve integrity (filtered from metric calculations)
- Kelly position sizing from realized trade stats, not theoretical assumptions

### Optimization
- **Bayesian (Optuna TPE)** over the active parameter space
- **Walk-forward validation** — rolling (fixed window) or anchored (expanding)
- **Deployment params = last fold's params**, not best-OOS-fold (which would re-introduce selection bias)
- **Pinned parameters** — freeze any subset to reduce search dimensionality
- **Diagnostics**: efficiency ratio, parameter stability CV, trial-budget warnings, failed-trial rate
- Pure Optuna for the fit — no LLM does any actual fitting

### Validation
- **In-sample permutation test** — shuffles close-to-close returns while preserving intra-bar structure, then re-optimizes on each shuffle. p-value vs. the null of "no real edge."
- **Monte Carlo** — trade shuffle, block bootstrap of returns, noise injection. Ruin-threshold probability included.
- **Trade-log regression tests** — bit-identical trade hashes across releases, not just metric-level equality.

### Analytics
- Calendar seasonality (day-of-week, monthly, hourly, day-of-month, return distribution)
- Strategy vs. buy-and-hold with beta/alpha/correlation
- 2D parameter heatmaps
- Multi-asset portfolio runs

### MCP server (Claude Desktop / Claude Code)
The platform exposes itself as an [MCP](https://modelcontextprotocol.io) server. Claude can drive the full research loop:

```
load_data → get_market_characterization → set_params → run_backtest
         → run_optimize → run_permutation_test → log_research_result
```

Includes `get_research_history` so the agent can avoid re-testing already-disqualified ideas, and `register_indicator` for runtime indicator authoring (sandboxed: import whitelist, syntax check, schema validation, smoke test on synthetic data).

A `CLAUDE.md` protocol file in the repo turns the model into an experienced quantitative researcher — characterize regime first, then hypothesize, then test. Two entry indicators max. Costs assumed 3% round-trip. Edge must survive permutation testing.

---

## Indicators

Built-in entry filters: **PAMRP, BBWP, ADX, MA Trend (SMA/EMA/WMA), RSI, Volume, Supertrend, VWAP, MACD, Stochastic RSI, Choppiness, Donchian, RSI Hidden Divergence.**

Exit methods: **Stop Loss, Take Profit, Trailing Stop, ATR Trailing, PAMRP Exit, Stoch RSI Exit, Time Exit, MA Exit, BBWP Exit.**

Visual overlays: **HPDR Bands** (rolling-quantile range bands with Normal-equivalent coverage labels), **RSI Hidden Divergence**.

### Adding your own

Each indicator is **one file** in `src/indicators/specs/`. Adding one means:

1. Write a spec file (`compute`, `long_signal`, `short_signal`, parameter declarations)
2. Import it in `src/indicators/specs/__init__.py`
3. Add defaults to `ui/session.py → get_default_params()`

Sidebar widgets, Optuna search ranges, pin UI, plotting, signal pipeline integration — all wired automatically from the spec. No if-else chains, no widget boilerplate, no optimizer blocks. See [`docs/contributing/adding-an-indicator.md`](docs/contributing/adding-an-indicator.md).

---

## Quick start

```bash
git clone https://github.com/YOUR_USER/strategy-lab.git
cd strategy-lab
make run
```

`make run` creates the venv on first run, installs dependencies, and launches the Streamlit app at `http://localhost:8501`.

Manual setup:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run tests:

```bash
make test           # unit tests
make regression     # trade-log hash regression suite
```

Reset the environment:

```bash
make clear          # removes venv/
```

---

## MCP server setup (Claude Desktop)

Add this to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, `~/.config/claude/claude_desktop_config.json` on Linux):

```json
{
  "mcpServers": {
    "strategy-lab": {
      "command": "/absolute/path/to/strategy-lab/venv/bin/python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/absolute/path/to/strategy-lab"
    }
  }
}
```

Restart Claude Desktop. The `strategy-lab` tool group will appear. Open `CLAUDE.md` in the project root for the research protocol the agent follows.

---

## Project structure

```
app.py                              Streamlit orchestrator (thin)
src/
├── backtest/                       Fill simulation, equity tracking, metrics
├── strategy/                       StrategyParams, signal generation, entry/exit logic
├── indicators/
│   ├── registry.py                 IndicatorSpec / ParamSpec / PlotSpec
│   └── specs/                      One file per indicator
├── optimize/                       Optuna TPE, walk-forward, pinned params, diagnostics
├── permutation/                    In-sample permutation test
├── montecarlo/                     Three MC methods + ruin probability
├── analytics/                      Calendar seasonality
├── data/                           yfinance + CSV loaders, sample data, validation
└── mcp_server.py                   FastMCP server for Claude Desktop
ui/
├── session.py                      Default params, session state init
├── sidebar.py                      Sidebar (data, indicators, exits) — registry-driven
├── charts.py                       Plotly factories
├── helpers.py
├── styles.py
└── tabs/                           One file per tab
tests/
├── regression/                     Trade-log hash fixtures
└── ...
docs/contributing/                  How to add indicators, etc.
CLAUDE.md                           Research protocol the MCP agent follows
Makefile
requirements.txt
```

---

## Backtest execution rules (the unpleasant details)

| Event | Rule |
|---|---|
| Entry fill | Previous bar's signal → current bar's open + slippage |
| Stop orders | Trigger on high/low; fill at order level **or** bar open if gap-through (whichever is worse) |
| Entry-bar SL/TP | Checked on the same bar as entry (no free bar) |
| Time exits | Execute at bar open (no look-ahead) |
| SL/TP conflict on same bar | Closer to open wins, deterministic |
| Open position at end of data | Force-closed at last close; `exit_reason == 'end_of_data'` filtered from metrics |
| Slippage on gap-aware exits | Not double-charged (`GAP_AWARE_EXITS` exclusion set) |
| Kelly sizing | Realized stats after 20 trades, not theoretical inputs |

---

## Walk-forward and selection bias

The walk-forward implementation has one rule that distinguishes it from most tutorials:

> **The deployment parameter set is the *last fold's* best parameters, not the best-OOS-fold's.**

Picking the best OOS fold is a form of selection bias — you're choosing across out-of-sample windows, which itself becomes an in-sample procedure. The last fold's params are the most regime-recent and were never chosen on the basis of their OOS score.

Walk-forward also reports an **efficiency ratio** (mean OOS metric / mean IS metric). Values above 1.0 are unusual and warrant scrutiny — they often indicate the strategy is exploiting a data artifact rather than a real edge.

---

## Permutation test — why row-wise OHLC shuffling is wrong

The naive permutation test shuffles entire OHLC rows. This breaks price continuity (the close of bar *t* no longer leads to the open of bar *t+1*) and makes the permuted series structurally distinguishable from any real price series — destroying the test's validity.

Strategy Lab decomposes each bar into:

- a **close-to-close return** (the temporal signal)
- a set of **scale-independent intra-bar ratios** (open/high/low relative to close)

Returns are shuffled. Bar structures are shuffled as units. The continuous price series is then reconstructed. This preserves return distribution, volatility, and bar-shape distribution while destroying temporal autocorrelation and any genuine signal.

p-value = fraction of permuted optimizations whose best metric ≥ the real optimization's best metric. p < 0.05 → statistically significant edge.

---

## Data sources

| Source | Coverage | Cost |
|---|---|---|
| Yahoo Finance (yfinance) | Equities, ETFs, crypto, FX | Free |
| CSV upload | Anything with OHLCV columns | — |
| Sample data | Synthetic GBM, configurable vol/drift/seed | — |

yfinance interval limits: 1m (7d), 5m (60d), 1h (730d), 1d+ (full history). For higher-frequency crypto/DEX data the `mcp_server` can also read local parquet/CSV files via `load_data_from_file`.

---

## Dependencies

| Package | Purpose |
|---|---|
| streamlit | UI framework |
| pandas, numpy | Data and numerical compute |
| scipy | Statistics |
| plotly | Charts |
| optuna | Bayesian (TPE) optimization |
| yfinance | Market data (free) |
| fastmcp | MCP server for Claude Desktop |
| pytest | Tests |
| joblib, tqdm | Parallelism and progress |

Python 3.10+. Full list in `requirements.txt`.

---

## Roadmap

- Volume-bubble order-flow integration via parquet bridge (companion repo)
- Bar-signal validity windows and historical bar-index confirmation
- AND-A / AND-B / OR condition groups for indicator composition
- Break-even stop modes and weekday/session filters
- Generic threshold/range conditions
- kNN-based strategy (Pine Script → Python conversion)

See [issues](../../issues) for the live list.

---

## Contributing

Contributions welcome. Useful entry points:

- Add an indicator — [step-by-step walkthrough](docs/contributing/adding-an-indicator.md)
- Add a metric — propose it as a `BacktestResults` attribute; reference the formula in the docstring
- Improve fill realism — open an issue describing the case the current engine gets wrong; include a fixture if possible
- Add tests — regression tests use bit-identical trade hashes; unit tests live in `tests/`

Style: pure functions, vectorized where possible, no UI/logic mixing, explicit formulas in docstrings, no silent exception swallowing. See `Development Guidelines` in the repo for the full philosophy.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Acknowledgements

Built on the open-source quantitative-finance Python stack: pandas, numpy, scipy, Optuna, Streamlit, Plotly, yfinance, FastMCP. None of this exists without them.

---

**Keywords:** quantitative finance, algorithmic trading, backtesting, walk-forward optimization, Optuna, Bayesian optimization, permutation test, Monte Carlo simulation, Streamlit, MCP, Model Context Protocol, Claude Desktop, agentic trading research, Python, open source, no paid APIs, gap-aware fills, look-ahead bias, signal validation, trade-log regression, yfinance.
