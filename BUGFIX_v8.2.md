# AlphaYantra v8.2 — Bug Fix Patch

## Overview
Deep-dive code audit of the full v8.1 codebase across all 20+ modules.
8 bugs identified and fixed. No changes to model architecture or strategies.

---

## BUG-1 — SHORT unrealised P&L always computed as LONG
**File:** `paper_trading/engine.py` → `_tick_all_trades()`  
**Problem:** `unrealised_pnl = (price - entry_price) * quantity` — same formula for both directions. A SHORT trade profits when price FALLS, so this showed the wrong sign.  
**Fix:** Multiply by `direction_mult = 1 if LONG else -1`.

---

## BUG-2 — pnl_pct sign wrong for SHORT trades
**File:** `paper_trading/engine.py` → `_close()`  
**Problem:** `pnl_pct = (exit / entry - 1) * 100` always returns positive for price > entry. For a SHORT, price rising means a LOSS.  
**Fix:** Multiply by direction sign: `* (1 if LONG else -1)`.

---

## BUG-3 — max_drawdown_rs calculated incorrectly (Python `and` short-circuit)
**File:** `backtest/engine.py`  
**Problem:** `round(abs(drawdown.idxmin() and eq_series.max() - eq_series.min()), 2)` — Python `and` evaluates to the second operand when the first is truthy. `drawdown.idxmin()` is a Timestamp (always truthy), so the entire expression returns `eq_series.max() - eq_series.min()` (the total equity range), NOT the rupee value of the max drawdown.  
**Fix:** `round(abs(roll_max[drawdown.idxmin()] - eq_series[drawdown.idxmin()]), 2)` — peak equity at the drawdown point minus trough equity.

---

## BUG-4 — /train background task blocks asyncio event loop
**File:** `api/main.py`  
**Problem:** `_do_train` was defined as `async def`. FastAPI's `background_tasks.add_task()` runs async callables as coroutines in the event loop, not in a thread pool. The synchronous blocking calls inside (yfinance, sklearn, LightGBM) would freeze the entire FastAPI event loop during training.  
**Fix:** Changed to `def _do_train()` (plain function). FastAPI now runs it in a thread pool executor via `anyio.to_thread`.

---

## BUG-5 — Daily state refresh discarded at market open
**File:** `scheduler/tasks.py`  
**Problem:** `risk._load_or_init_state()` was called but its return value was discarded. `_load_or_init_state()` returns a new `DailyState` object but does NOT mutate `self.state` internally. The risk manager's in-memory state was never refreshed to the new day — it kept operating on yesterday's counters.  
**Fix:** `risk.state = risk._load_or_init_state()`.

---

## BUG-6 — STRONG SELL has no confirmation gate (asymmetric with STRONG BUY)
**File:** `strategies/indicators.py`  
**Problem:** `STRONG BUY` required `confirmations >= min_confirmations AND score >= 75` (two gates). `STRONG SELL` only required `score <= 25` (one gate). A single indicator pile-on could trigger STRONG SELL with zero bearish confirmations.  
**Fix:** Added `bearish_confirmations` counter tracking: RSI overbought, MACD bearish cross, BB upper touch, EMA death cross. `STRONG SELL` now requires `bearish_confirmations >= min_confirmations AND score <= 25`.

---

## BUG-7 — run.py and scheduler pass `use_triple_barrier=True`
**Files:** `run.py`, `scheduler/tasks.py`  
**Problem:** v8 analysis showed triple-barrier labels produce ~41% positive rate (near coin-flip AUC). The fix was the `trend_quality_labels` function. But the callers still passed `use_triple_barrier=True`, silently overriding the v8 default.  
**Fix:** Both callers now pass `use_triple_barrier=False`.

---

## BUG-8 — SELL signal closes wrong quantity (kelly-sized, not actual position)
**File:** `broker/paper_trader.py` → `execute_signal()`  
**Problem:** When a SELL signal arrives to close a LONG position, `quantity = int(trade_value / fill_price)` recomputes a kelly-sized quantity from current capital. If capital has changed since the position was opened, this quantity differs from the held position size — resulting in partial close or over-selling attempt.  
**Fix:** For SELL signals, use `self.positions[ticker].quantity` (the actual held quantity) instead of a freshly computed value.

---

## Files Changed
- `paper_trading/engine.py` — BUG-1, BUG-2
- `backtest/engine.py` — BUG-3
- `api/main.py` — BUG-4
- `scheduler/tasks.py` — BUG-5, BUG-7
- `strategies/indicators.py` — BUG-6
- `run.py` — BUG-7
- `broker/paper_trader.py` — BUG-8
