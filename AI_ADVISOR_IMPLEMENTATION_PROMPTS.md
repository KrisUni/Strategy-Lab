# Strategy Lab — AI Advisor & Indicator Builder: Implementation Prompts

Each numbered step is a self-contained prompt for a fresh Claude Code session. Run them in order; every step lists its hard prerequisites. Touch points were verified against the current tree.

## Sequencing decision (phasing was left open in the issue board — committing to this)

Order follows the dependency graph, not the epic numbering. Four phases, mapped to the board's natural cut points:

- **Phase 0 — Substrate** (SL-001, SL-002): fix/confirm rigor and harden the log before any confident voice sits on top of it. Non-negotiable first.
- **Phase 1 — Talk (Cut A)** (SL-601, SL-602, SL-201, SL-202, SL-203, SL-204, + SL-501/SL-503 since their deps are met here): a grounded, skeptical, read-only copilot. Ships the differentiator first at lowest risk — no state mutation.
- **Phase 2 — Act (Cut B)** (SL-101 keystone, SL-102, SL-301, SL-302, SL-303, SL-304, + SL-502): authorization and the compute-cost guard land *with* the ability to act, never after.
- **Phase 3 — Build indicators (Cut C)** (SL-401, SL-402, SL-403, SL-404): depends on the keystone and the action surface.

Rationale: Cut A delivers user-visible value behind a clean off-by-default switch with zero mutation risk; the keystone (SL-101) is deferred until the act phase actually needs it, so Phase 1 isn't blocked on the largest refactor. SL-403 (look-ahead check) gates Phase 3 shipping.

**Cross-cutting invariant to repeat in every prompt:** one canonical stateless logic core; no front-end re-implements logic; state is per-process (MCP `_state` and `st.session_state` never sync — only logic is shared); every mutating action is preview→confirm→execute; advisor is off by default and the platform is fully functional without a key.

**Conflict to watch:** the generic dev-guidelines doc says "use float32 if precision allows." Do **not** apply that anywhere in this work. Float32 breaks the bit-identical trade-log hash standard; everything stays float64.

---

## Phase 0 — Substrate

### P0.1 · Verify & close out rigor baseline (SL-001) · S
**Depends on:** nothing. **Blocks:** P1.5, P1.7, P1.8 (anything that shows/reasons over metrics).

Context: The advisor will surface these metrics to non-sophisticated users. A confident advisor on top of stale or biased numbers is the worst case for an honesty-first product. The permutation fix and the active-return Sharpe correction were drafted as patches — confirm they actually landed, then re-validate.

Touch points: `src/permutation/__init__.py` (`_permute_prices`), `src/backtest/__init__.py` (Sharpe `in_position_arr`), `tests/regression/` (`generate_hashes.py`, `expected_hashes.json`, `make regression`), `RESEARCH_LOG.md`.

Task:
1. Write a fixture-based test asserting that after `_permute_prices`, the Pearson correlation between the permuted close series and the original close is ≪ 1.0 (assert `< 0.2` on the synthetic ETH fixture). If the buggy `new_close[i]/close[i]` intrabar-scaling is still present, this test must fail.
2. Confirm the active-return Sharpe off-by-one in `in_position_arr` is fixed (add/confirm a unit test on a hand-constructed in-position mask).
3. Run `make regression`. If the Sharpe fix legitimately changed any trade log, re-pin via `python -m tests.regression.generate_hashes` and justify each changed hash in the commit message. If hashes did **not** move, do not re-pin.
4. Re-run the permutation test on every strategy in `RESEARCH_LOG.md` that carries a p-value (their old p-values were computed under the buggy permuter). Record new results in the log; flag any strategy whose verdict flips (e.g. VIABLE→DISCARD).

Acceptance:
- Permuted-vs-original close correlation `< 0.2` on the fixture.
- `make regression` green; hashes re-pinned only if a deliberate trade-logic change occurred, with justification.
- Revalidation sweep results appended to `RESEARCH_LOG.md`; flipped verdicts explicitly flagged.

Edge cases: if a strategy's recomputed p-value crosses 0.05 in either direction, do not silently overwrite — keep the old entry and append a new dated one noting the permuter fix as the cause.

---

### P0.2 · Migrate RESEARCH_LOG to structured JSONL (SL-002) · M
**Depends on:** nothing. **Blocks:** P1.5, P2.7 (write-back), P1.7 (multiple-comparisons count).

Context: The advisor reasons over history programmatically. The current parser in `get_research_history` does `text.split("\n---\n")` then `header[3:].split(" — ")` — any thesis containing `" — "`, `"##"`, or newlines corrupts on parse. Too fragile to depend on.

Touch points: `src/mcp_server.py` (`log_research_result`, `get_research_history`, `_LOG_PATH`).

Task:
1. Introduce a JSONL store (`RESEARCH_LOG.jsonl`) as the source of truth. One JSON object per line: symbol, interval, date, verdict, thesis, indicators, regime, date_range, n_bars, metrics dict, efficiency_ratio, p_value, failure_reason, iteration_history, notes, and a `source` field (default `"mcp"`).
2. Rewrite `log_research_result` to append a JSON line. Keep the human-readable `RESEARCH_LOG.md` as a *generated render* from the JSONL (regenerate on each write, or expose a small render function).
3. Rewrite `get_research_history` to read JSONL and filter structurally (no string-splitting).
4. One-time migration: parse the existing markdown log into JSONL losslessly. Where a field can't be recovered, store `null`, never drop the entry.

Acceptance:
- A thesis containing `" — "`, `"##"`, and an embedded newline round-trips byte-intact through write→read.
- Existing markdown log migrates with entry count preserved.
- `get_research_history` returns structured dicts; markdown view still available and matches.
- New entries carry a `source` field.

Edge cases: malformed legacy entries (missing the 3-part header) go into JSONL with best-effort fields plus `notes: "migrated, header unparseable"`, not discarded.

---

## Phase 1 — Talk (read-only advisor)

### P1.1 · Settings panel: key, model, CLAUDE.md override (SL-601) · M
**Depends on:** nothing. **Blocks:** P1.3, P1.4.

Context: BYO key, opt-in, off by default. No hosting of keys or inference.

Touch points: `ui/tabs/` (new settings tab or section), `ui/session.py` (`get_default_params` / session init), repo-root `CLAUDE.md`.

Task: Add a settings panel with (a) API-key entry held in `st.session_state` only by default, persisted to disk **only** on an explicit opt-in checkbox; (b) model selector; (c) a `CLAUDE.md` override editor that loads the repo file as the default and lets the user edit it for their session. Absent key ⇒ advisor disabled.

Acceptance:
- No key written to disk unless the user opts in; verify by inspecting the persistence path after entering a key without opting in.
- Override editor pre-fills from repo `CLAUDE.md`; clearing it falls back to the repo file, never to a hardcoded copy.
- With no key set, advisor state resolves to disabled cleanly (no exceptions).

Edge cases: invalid/blank key must not crash settings render; trim whitespace; never echo the key back in logs or `st.write`.

---

### P1.2 · Privacy disclosure (SL-602) · S
**Depends on:** nothing. **Blocks:** P1.4.

Touch points: settings/advisor enable path from P1.1.

Task: Before the advisor can be enabled, require a one-time acknowledgement that enabling it sends params/metrics/indicator descriptions to the Anthropic API. Opt-in checkbox; record acknowledgement in session state.

Acceptance: advisor cannot transition to enabled without the disclosure being acknowledged once; acknowledgement persists for the session.

---

### P1.3 · Right-rail panel scaffold (SL-201) · M
**Depends on:** P1.1. **Blocks:** P1.4.

Context: Collapsible right rail, same registry-driven logic conventions as the left rail. Off by default.

Touch points: `app.py` (layout/orchestrator), new `ui/advisor/` module, `ui/styles.py`.

Task: Render a collapsible right rail with a conversation area, an input box, and an enable/disable state. With no key it shows a non-intrusive "advisor disabled" state and the app works normally. Keep all logic out of this layer (§8) — it only collects input and displays output.

Acceptance: rail collapses/expands; layout doesn't fight the active tab at common widths; disabled is the default and unobtrusive.

Edge cases: rail must not steal focus or reflow the main tabs when collapsed; no advisor code path runs while disabled.

---

### P1.4 · Anthropic API client + conversation state (SL-202) · M
**Depends on:** P1.1, P1.2, P1.3.

Touch points: `ui/advisor/` (new client module), session state.

Task: Streaming Anthropic client. Conversation history kept in `st.session_state` and re-sent in full each turn (the API is stateless). Key pulled from settings; absent/invalid key ⇒ disabled with a friendly message, never a stack trace. Support cancel/stop.

Acceptance:
- Multi-turn conversation persists across Streamlit reruns.
- Missing or invalid key surfaces a clean message, no traceback to the user.
- Stop/cancel interrupts a stream without corrupting history.

Edge cases: network/timeout errors render as a recoverable message; a partial stream that errors mid-way leaves a coherent history entry.

---

### P1.5 · Read-only context assembler (SL-203) · M
**Depends on:** P0.1, P0.2, P1.4. (Note: depends on the metric extractor; if P2.1/SL-101 hasn't landed yet, read metrics via the existing `_results_to_dict`/`last_metrics` path — do not re-implement metric math here.)

Context: Feed the model only what's actually on screen. Never invent metrics.

Touch points: `ui/advisor/`, `st.session_state` (`params`, `backtest_results`/`last_metrics`), `get_active_filters_display` in `ui/helpers.py`, `get_research_history` (JSONL, from P0.2).

Task: Assemble read-only context: current tab params, last computed metrics, active indicators (via `get_active_filters_display`), and a research-history summary for the current symbol pulled from the JSONL store. No actions.

Acceptance: advisor correctly answers "why is my Sharpe negative when win rate is 60%?" against the real on-screen numbers; it never reports a metric that isn't present in state (test by asking about a metric not yet computed — it must say it isn't available).

Edge cases: no data loaded / no backtest run ⇒ context says so explicitly rather than fabricating zeros.

---

### P1.6 · System prompt = CLAUDE.md, skeptical-but-educational (SL-204) · S
**Depends on:** P1.4. Related: P1.1 (override editor).

Task: Load the (possibly overridden) `CLAUDE.md` as the base system prompt; layer a tone directive: skeptic by default, but explains *why* for a non-sophisticated audience rather than dismissing. Use the loaded file, never a hardcoded copy.

Acceptance: advisor refuses to cheerlead a weak result but explains the reasoning; swapping the override editor content visibly changes behavior, proving the live file is used.

---

### P1.7 · Multiple-comparisons surfacing (SL-501) · M
**Depends on:** P0.2, P1.5.

Context: Core defense against the advisor becoming a p-hacking co-signer.

Touch points: `ui/advisor/` context assembler, JSONL log.

Task: On open (and when relevant), state the variant/run count for the current symbol from the log ("this is your 39th configuration on SPY") and have the advisor adjust its significance language accordingly.

Acceptance: count matches the log exactly; a p=0.04 reached after many configurations is explicitly framed as weak, with the count cited.

Edge cases: symbol never tested ⇒ "first configuration," no false count.

---

### P1.8 · Block blessing of untested results (SL-503) · S
**Depends on:** P1.5.

Task: The advisor refuses "looks good / ship it" verdicts when no permutation p-value exists for the current config; it directs the user to run the permutation test first and explains why.

Acceptance: positive verdict withheld until a p-value is present in context; the refusal states the reason (edge unproven against the null).

---

## Phase 2 — Act (authorized)

### P2.1 · Extract stateless `session_core` (SL-101, KEYSTONE) · M–L
**Depends on:** P0.1 (clean hashes to refactor against). **Blocks:** P2.2, P2.4, P3.1.

Context: One implementation of run/optimize/permutation/metrics glue makes divergence structurally impossible. Drift is the documented scar tissue (trigger-vs-filter, entry-into-active-exit). This is a **metrics-neutral refactor — trade-log hashes must not move.**

Touch points: new `src/session_core.py` (or `src/core/`); `src/mcp_server.py` (`run_backtest`, `run_optimize`, `run_permutation_test`, `_results_to_dict`); `ui/helpers.py` (`params_to_strategy`).

Task:
1. Lift the run/optimize/permutation/metrics glue into a **stateless** module: explicit inputs (df, params dict, capital, commission, slippage, optimize/permutation config), explicit outputs. No module-global state — `_state` stays in the MCP layer, `st.session_state` stays in the UI layer; only the *logic* moves.
2. Promote `_results_to_dict` into the core as the single `results_to_metrics` definition; MCP calls the core's version.
3. Re-point the MCP tools to thin-adapt onto the core (read from `_state` → call core → write back to `_state`).
4. Leave the UI tabs on their current `params_to_strategy → BacktestEngine` path for now (fast-follow); only the cost-defaulting and metric-stashing are drift surfaces and the hash suite guards them.

Acceptance:
- `session_core` holds no module-global state (grep confirms; functions are pure given inputs).
- Running `tests/regression` CONFIGS through the core yields **bit-identical** hashes — `make regression` green, **no re-pin**.
- An MCP `run_backtest` and a direct core call produce an identical metrics dict for the same config (add an explicit dict-equality test — metrics are not in the hash, so this guard is required).

Edge cases / notes: a long optimize blocking the Streamlit rerun thread is acceptable for v1 (background runner is a later optimization). Float64 throughout — do not introduce float32.

---

### P2.2 · In-app action → immediate UI reflection (SL-102) · M
**Depends on:** P2.1.

Context: Writing `st.session_state.params` alone does not update already-instantiated widgets. Must mirror what `apply_best_params_callback` already does.

Touch points: `ui/helpers.py` (`apply_best_params_callback`, `_STRATEGY_WIDGET_KEYS`), `ui/sidebar_renderer.py` (`widget_{param.name}` convention), `ui/advisor/` action adapter.

Task: Add a shared sync helper (UI-layer only) that, given a params delta, writes both `st.session_state.params[k]` and the widget key `_STRATEGY_WIDGET_KEYS.get(k, f"widget_{k}")`, mirrors strategy-level keys (`tdir`, `ecm`, `eop`, etc.), then triggers a rerun. The advisor action path calls this helper. This sync step lives in the UI adapter, **never** in `session_core` (the MCP process has no widgets and skips it).

Acceptance:
- Advisor sets RSI window ⇒ the sidebar slider visibly moves on next rerun with no manual refresh.
- No stale-widget desync after two consecutive advisor actions.

Edge cases: a param with no corresponding widget (advisor-only field) updates `params` without erroring on the missing widget key.

---

### P2.3 · Action tool surface, session-bound (SL-301) · M
**Depends on:** P2.1. **Blocks:** P2.4, P3.1, P2.6.

Task: Expose `set_params` / `run_backtest` / `run_optimize` / `run_permutation_test` / `log_research_result` to the model as tools, bound to the in-process session core — the *same* functions the MCP tools call, no private code path.

Acceptance: a backtest run via the advisor produces the same trade-log hash as the same config run via the UI; each tool routes through the SL-101 core.

---

### P2.4 · Authorization gate: diff + confirm (SL-302) · M
**Depends on:** P2.3. **Blocks:** P2.5, P2.7.

Context: Every action is authorized.

Task: Every action renders a preview before execution — a param diff, or "will run an N-trial optimization" — and requires an explicit confirm. Rejecting leaves state untouched.

Acceptance: no tool executes without a confirm click; rejecting a proposal leaves `params`/state byte-identical to before; the diff is legible to a non-sophisticated user (old→new per field).

Edge cases: a proposal the user ignores (neither confirms nor rejects) must not execute on the next unrelated rerun.

---

### P2.5 · Compute-cost guard (SL-303) · M
**Depends on:** P2.4. Related: P2.8.

Context: The single failure mode most worth defending against — burning optimization compute.

Task: Optimization and permutation actions must display an estimated cost (trials × folds × est. runtime) and require a *separate* explicit confirm beyond the SL-302 gate. The advisor may **propose** but never auto-launch a sweep. Add a configurable hard trial-budget ceiling that warns/blocks above threshold, and throttle rapid repeated optimize proposals.

Acceptance:
- No multi-trial run triggers without the cost preview + its own confirm.
- A run above the configured trial budget fires a hard warning/block.
- Repeated optimize proposals within a short window are throttled.

Edge cases: the advisor cannot chain "optimize → permutation → optimize" autonomously; each leg needs its own confirm.

---

### P2.6 · Iteration-fatigue warning (SL-502) · S
**Depends on:** P2.3. Related: P2.5.

Task: Detect repeated re-optimization on the same data/symbol within a session and warn about overfitting before proposing another run. Tie into the SL-303 throttle.

Acceptance: N re-optimizations within a session triggers an overfitting warning before the next proposal; the warning references the count.

---

### P2.7 · Research-log write-back, tagged (SL-304) · S
**Depends on:** P0.2, P2.4.

Task: The advisor can append a verdict to the JSONL research log **only on explicit confirm**, tagged `source = "in-app"`, so in-app entries are distinguishable from the external agent's (`source = "mcp"`) history.

Acceptance: writes only after confirm; `source` tag present and correct; external-agent and in-app entries are filterable apart.

---

## Phase 3 — Build indicators

> **Visualization scope (read first).** The chart layer (`create_price_chart_with_trades` in `ui/charts.py`) is registry-driven: it draws a spec only if the spec is in the *in-process* registry, has a `PlotSpec.render`, has `enable_param=True`, and its `compute` populated the indicator df. Consequence:
> - **MCP-authored** indicators (`register_indicator`, external process) are **data-only, never drawn** — the MCP process doesn't share its registry with Streamlit (invariant 15) and has no chart layer. This is intended for the external-agent loop.
> - **In-app advisor-authored** indicators run in the Streamlit process and **can** be drawn. To make that real and safe: the model does **not** author render code; the builder auto-synthesizes a default `PlotSpec` from declared `outputs` (P3.1), and P3.5 verifies a provisional indicator actually renders.
> - To make an AI-built indicator durable across restarts / both processes, persist its spec source to `src/indicators/specs/` via the promote path (P3.4) — never live cross-process registry sync.

### P3.1 · NL → IndicatorSpec generation (SL-401) · M
**Depends on:** P2.3. **Blocks:** P3.2.

Context: Reuse the registry's self-describing unit shape; produce a *draft* shown to the user before anything runs.

Touch points: `src/indicators/registry.py` (`IndicatorSpec`, `ParamSpec`), the `register_indicator` source conventions in `src/mcp_server.py` (no import statements; `pd`, `np`, `IndicatorSpec`, `ParamSpec`, `register`, and `src.indicators` primitives pre-loaded; named callables only, no lambdas).

Task: Turn a plain-language description into a draft `IndicatorSpec` source string (compute, long_signal, short_signal, ParamSpecs, optimizer metadata), following the exact sandbox source conventions. The model authors **compute and signal functions only** — it must **not** write the `PlotSpec.render` callable (that would be unsandboxed plotly-mutating code in the UI process). Instead, the builder auto-synthesizes a default `PlotSpec` from the declared `outputs`: overlay a line on the price panel for any output column whose value range overlaps the price range, otherwise give the column its own panel. Show the full draft (including the auto-generated plot) to the user before any execution.

Acceptance:
- "an RSI that only fires when volume is above its 20-bar average" produces a structurally valid spec draft (named functions, declared `outputs`, declared ParamSpecs, an `*_enabled` bool param) presented for review before registration.
- The draft contains an auto-generated `PlotSpec` (the model's source never defines `render`); a price-scaled output overlays the price panel, a bounded oscillator output (e.g. 0–100) gets its own panel.

---

### P3.2 · Harden the sandbox for AI-authored code (SL-402) · M
**Depends on:** P3.1. **Blocks:** P3.3, P3.4.

Context: AI-generated code is executed; for naive users the sandbox is the only safety net.

Touch points: `register_indicator` in `src/mcp_server.py` (existing import whitelist, restricted `__builtins__`, exec, `validate_registry`, 100-bar synthetic smoke test). Lift the shared validation into a reusable function the advisor calls (don't duplicate it in the UI).

Task: Reuse the existing whitelist / syntax check / schema validation / synthetic smoke test, and add execution-time and memory limits plus a clear rejection path with human-readable reasons.

Acceptance: disallowed imports, an infinite loop, and a malformed spec are each rejected with a plain-language reason; nothing executes outside the sandbox before validation passes.

Edge cases: a spec that registers but raises on the 100-bar smoke data is rolled back from `INDICATOR_REGISTRY` (mirror the existing append→validate→pop-on-failure pattern).

---

### P3.3 · Look-ahead / leakage check (SL-403, THESIS-CRITICAL) · M
**Depends on:** P3.2. **Gates Phase 3 shipping.**

Context: A non-sophisticated user cannot detect look-ahead. CLAUDE.md states the validator does **not** yet catch look-ahead — this step closes that gap. Shipping a leaky AI-built indicator on an honesty-first platform is an existential contradiction. A single future-bar perturbation is **not** sufficient: a value-dependent leak (e.g. "use the future max only when it exceeds the current bar") can survive one perturbation at one index. The check below proves causality rather than spot-checking it.

Touch points: the smoke-test stage of the hardened sandbox (P3.2).

Causal property to enforce: the output at index `i` depends only on inputs at indices `≤ i`. Apply **every** layer below to **both** the `compute` output columns **and** the realized `long_signal` / `short_signal` boolean series (a leak can live in a signal function independently of compute). Reject on the first failing layer and name it in plain language.

1. **Static AST scan (cheap pre-filter).** Reject any negative shift (`.shift(-N)`), index/time reversal feeding a windowed op (`[::-1]`, `.iloc[::-1]`, `.sort_index(ascending=False)`), forward positional indexing (`.iloc[i+…]`, `values[i+1:]`), and division/normalisation by a whole-series aggregate computed per-row. AST flags intent; it is not the proof — suspicious-but-unprovable patterns fall through to the behavioural layers rather than being trusted.

2. **Determinism precondition.** Run compute + signals twice on identical synthetic data; require bit-identical output. Non-determinism violates §4.3 and makes the causality tests unreliable → reject `non_deterministic`.

3. **Prefix/truncation invariance (the airtight proof).** On synthetic data of length N (≥300 bars), for a set of cut points K — sampled across the series *and weighted toward the end* (e.g. 25%, 50%, 75%, 90%, N−2, N−5, plus several random k past the indicator's warm-up) — compute outputs on the full series and on the prefix `df.iloc[:k]`. For every index `i < k`: require the prefix value to be non-NaN exactly where the full-series value is non-NaN, and **exactly equal** (bit-identical, no tolerance — a causal pure function recomputes past values identically when future bars are absent). Any mismatch, including a prefix-NaN where the full series has a value at `i<k`, means future data altered a past output → reject `lookahead_truncation`. Removing future data entirely is the strongest possible perturbation, which is what makes this airtight.

4. **Future-block perturbation (defense-in-depth for value-dependent leaks).** For several cut points k, replace **all** bars at index `≥ k` (not one bar) and recompute, using multiple perturbation patterns applied independently: ×1.5 scale, ×0.5 scale, reversed future block, additive random noise, and a flat-constant block. Require all outputs at `i < k` unchanged versus the original full-series outputs. Multiple patterns catch leaks sensitive to specific future value relationships that fixed cut points in layer 3 might not excite → reject `lookahead_perturbation`.

NaN/alignment handling: compare only on the intersection of `i < k` and non-NaN full-series indices; NaN masks must match on that range. If an indicator yields zero non-NaN outputs at every tested k (can't be validated), reject `insufficient_output` — do not admit an unvalidatable indicator.

Acceptance:
- A `close.shift(-1)` indicator is rejected at layer 1 with a named reason.
- A value-dependent leak that *passes* a single-bar single-pattern perturbation is rejected by layer 3 or 4 (include such a fixture: e.g. compute uses `close.iloc[i:].max()`).
- A non-deterministic compute is rejected at layer 2.
- A correct causal indicator (e.g. `rsi(close, 14)`) passes all layers.
- Every rejection names the offending behaviour in plain language and identifies whether it was found in compute or in a signal.

Edge cases: legitimate warm-up NaNs at the series head must not be misflagged (handled by the non-NaN intersection rule); `ewm`/causal rolling must pass layer 3 exactly (they do, since past windows are identical under truncation).

---

### P3.4 · Provisional lifecycle (SL-404) · S
**Depends on:** P3.2.

Touch points: `_PROVISIONAL_KEYS` (registry), `list_indicators` (already reports `provisional`), `src.persistence` (save/load strategy), UI labels.

Task: Register AI-built indicators provisionally via `_PROVISIONAL_KEYS`; flag them visibly in the UI; exclude them from saved strategies until the user promotes them; allow discard. Promotion stays a human gate (per CLAUDE.md — do not automate). Promotion materialises the spec source to a file under `src/indicators/specs/` (and adds the import) so it survives restart and is visible to both processes — this is the only architecture-consistent way an AI-built indicator reaches the MCP process, replacing any notion of live cross-process registry sync.

Acceptance: provisional indicators are labeled in the UI; promote and discard both work; a saved strategy never silently depends on an unpromoted provisional indicator (saving with a provisional enabled either blocks or strips it with a clear message); a promoted indicator persists across an app restart.

---

### P3.5 · Provisional indicator renders on the chart (NEW — closes the visualization gap) · S
**Depends on:** P3.1, P3.2.

Context: Registering a spec is not the same as drawing it. `create_price_chart_with_trades` re-collects `enabled_specs(params)` each render, so a provisional spec *will* be picked up — but only if its `enable_param` is set true, its `compute` runs into the indicator df, and it carries a valid auto-generated `PlotSpec` (P3.1). This step verifies the end-to-end visual path inside the Streamlit process.

Touch points: `ui/charts.py` (`_collect_plot_rows`, `create_price_chart_with_trades`), the indicator-df build path that feeds `idf`, the auto-`PlotSpec` from P3.1.

Task: After an in-app indicator is registered and enabled, confirm the chart includes its trace/panel without code edits to `ui/charts.py` (registry-driven by design). Confirm the auto-generated overlay-vs-panel heuristic places price-scaled outputs on the price panel and bounded oscillators on their own panel.

Acceptance:
- An in-app-built, enabled provisional indicator appears on the price chart (overlay) or as its own panel on the next rerun, with no manual edits to the chart module.
- A provisional indicator with no enabled state does not draw.
- Discarding the provisional indicator (P3.4) removes its trace on the next rerun.

Edge cases: a compute that writes an output column not declared in `outputs` must still not crash the renderer — the auto-`PlotSpec` renders only declared `outputs`.

---

## Definition of done per phase

- **Phase 0:** permutation correlation test passes; `make regression` green; revalidation sweep logged; JSONL round-trip test passes.
- **Phase 1:** advisor toggles on with a key, off by default, answers grounded questions, refuses to bless untested results, surfaces the comparison count — all read-only.
- **Phase 2:** advisor mutates state only through preview→confirm; optimize/permutation gated behind a cost confirm; advisor-run backtests hash-match UI runs; `make regression` still green with no re-pin from SL-101.
- **Phase 3:** NL→spec drafts (with auto-generated `PlotSpec`, no AI-authored render) render for review; sandbox rejects leaky/malicious/malformed specs with reasons; the causality check rejects truncation/perturbation/AST/non-determinism leaks (incl. a value-dependent fixture that survives single-bar perturbation); an enabled in-app indicator draws on the chart; provisional indicators are labeled and excluded from saves until promoted; promotion persists to disk across restart.
