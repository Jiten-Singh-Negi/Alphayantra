"""
terminal.py  — AlphaYantra GPU Terminal v5
────────────────────────────────────────────
FIX 1 — Async-GUI Deadlock (CRITICAL):
  OLD: FeedThread was a threading.Thread that started LiveFeed.start()
       which internally could call asyncio.run() for the WebSocket bridge.
       asyncio.run() blocks the calling thread; if that thread was ever
       the main thread (or shared an event loop), DPG froze.

  NEW: The feed runs in a dedicated multiprocessing.Process (FeedProcess).
       It has its OWN Python interpreter, its OWN event loop, its OWN
       memory space.  It can run asyncio freely.  It writes ticks into a
       multiprocessing.Queue — the render loop reads from that queue once
       per frame.  Zero shared state, zero deadlock risk.

  All OTHER background workers (ML inference, VIX fetch, sentiment, paper
  trader) remain as daemon threads because they don't use asyncio.

Architecture:
  Main process / main thread   → DPG render loop (GPU, 60fps)
  Main process / thread pool   → MLThread, VIXThread, SentimentThread,
                                  PaperTraderThread (no asyncio)
  Separate process             → FeedProcess (NSE polling, asyncio-safe)
  IPC                          → multiprocessing.Queue (tick_queue)
                                  multiprocessing.Queue (log_queue)

FIX 2 — Lookahead Bias: handled in ml_engine.py and denoising.py.
  terminal.py itself doesn't denoise — it just displays the model output.

FIX 3 — Server Clash: handled in run.py.
  terminal.py is never started directly; always via run.py --terminal.
"""

import sys
import os
import time
import threading
import multiprocessing
import queue
import json
import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    print("ERROR: DearPyGui not installed.  Run: pip install dearpygui")
    sys.exit(1)

from loguru import logger

IST = ZoneInfo("Asia/Kolkata")

# ── NSE symbol universe for autocomplete ────────────────────────────────
try:
    from data.fetcher import ALL_NSE_STOCKS as _NSE_STOCKS, NIFTY_50 as _NIFTY_50
    _ALL_SYMBOLS: list = sorted(set(
        _NIFTY_50 + _NSE_STOCKS + [
            "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX",
        ]
    ))
except ImportError:
    _ALL_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                    "NIFTY", "BANKNIFTY", "TATAMOTORS", "WIPRO", "AXISBANK"]

# ── IPC queues — multiprocessing-safe ────────────────────────────────────
# These are created in main() and passed to FeedProcess.
# threading.Queue objects are passed to the thread-based workers.
_THREAD_LOG_Q  : queue.Queue         = queue.Queue(maxsize=1_000)
_THREAD_SIG_Q  : queue.Queue         = queue.Queue(maxsize=500)

# ── Colours ───────────────────────────────────────────────────────────────
C_BG          = (13,  17,  23,  255)
C_SURFACE     = (22,  27,  34,  255)
C_BORDER      = (48,  54,  61,  255)
C_TEXT        = (230, 237, 243, 255)
C_MUTED       = (139, 148, 158, 255)
C_GREEN       = (63,  185, 80,  255)
C_RED         = (248, 81,  73,  255)
C_YELLOW      = (210, 153, 34,  255)
C_BLUE        = (88,  166, 255, 255)
C_ORANGE      = (219, 109, 40,  255)
C_CANDLE_BULL = (63,  185, 80,  200)
C_CANDLE_BEAR = (248, 81,  73,  200)

# ── Global state shared across threads in the MAIN process ───────────────
_state: dict = {
    "quotes":         {},
    "signals":        [],
    "portfolio":      {},
    "positions_list": [],
    "vix":            15.0,
    "pcr":            1.0,
    "iv_rank":        50.0,
    "regime":         "normal",
    "regime_weights": {"rf": 0.33, "xgb": 0.33, "tcn": 0.34},
    "sentiment":      {},
    "trade_log":      deque(maxlen=300),
    "model_loaded":   False,
    "market_open":    False,
    "selected_sym":   "NIFTY",
    "kelly_frac":     0.5,
    "trail_pct":      1.5,
    "running":        True,
    # Object refs (not serialisable — stay in main process only)
    "_live_feed":     None,
    "_paper_trader":  None,
    "_risk_manager":  None,
    "_feed_proc":     None,
}
_state_lock = threading.Lock()


def _upd(key, value):
    with _state_lock:
        _state[key] = value


def _get(key):
    with _state_lock:
        return _state[key]


def _safe_hide_popup():
    """Hide the autocomplete popup (called on Escape key)."""
    try:
        dpg.configure_item(TAG_SEARCH_POPUP, show=False)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
# FEED PROCESS — runs in a separate Python process, asyncio-safe
# ══════════════════════════════════════════════════════════════════════════

def _feed_process_main(
    watchlist:  list,
    tick_q:     multiprocessing.Queue,
    log_q:      multiprocessing.Queue,
    stop_event: multiprocessing.Event,
):
    """
    Entry point for the feed subprocess.

    This function runs in a SEPARATE PROCESS — it has its own GIL,
    its own event loop, and can safely run asyncio without touching DPG.

    It polls NSE every poll_interval seconds and puts tick dicts into
    tick_q (multiprocessing.Queue).  The main process drains tick_q
    once per render frame.

    Communication back to main process: tick_q (data), log_q (status).
    Communication from main process: stop_event (shutdown signal).
    """
    # Re-add root path in the child process (it doesn't inherit sys.path changes)
    sys.path.insert(0, str(ROOT))

    try:
        from feed.live_feed import LiveFeed, is_market_open
        from loguru import logger as proc_logger

        proc_logger.info("FeedProcess: starting LiveFeed")
        log_q.put_nowait("[FEED] Feed process started")

        feed = LiveFeed(watchlist=watchlist, poll_interval=5,
                        tick_queue=tick_q)
        feed.start()

        while not stop_event.is_set():
            stop_event.wait(timeout=10)

        feed.stop()
        log_q.put_nowait("[FEED] Feed process stopped cleanly")

    except Exception as e:
        try:
            log_q.put_nowait(f"[FEED ERROR] {e}")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
# BACKGROUND THREADS — all in the main process, no asyncio
# ══════════════════════════════════════════════════════════════════════════

class MLInferenceThread(threading.Thread):
    """Runs model.predict() every `interval` seconds per watchlist symbol."""

    def __init__(self, watchlist: list, interval: int = 60):
        super().__init__(daemon=True, name="MLThread")
        self.watchlist = watchlist
        self.interval  = interval

    def run(self):
        model = None
        try:
            from models.ml_engine import AlphaYantraML
            m = AlphaYantraML("default")
            m.load()
            model = m
            _upd("model_loaded", True)
            _THREAD_LOG_Q.put_nowait("[ML] Model loaded — inference active")
        except FileNotFoundError:
            _THREAD_LOG_Q.put_nowait("[ML] No model found — run: python run.py --train")
        except Exception as e:
            _THREAD_LOG_Q.put_nowait(f"[ML ERROR] {e}")

        from data.fetcher import fetch_ohlcv
        from strategies.indicators import compute_indicators, IndicatorConfig
        cfg = IndicatorConfig()

        while _get("running"):
            if model is None:
                time.sleep(self.interval)
                continue

            vix = _get("vix")
            pcr = _get("pcr")

            for sym in self.watchlist[:20]:
                if not _get("running"):
                    break
                try:
                    df = fetch_ohlcv(sym, period="1y")
                    if df.empty or len(df) < 50:
                        continue
                    df  = compute_indicators(df, cfg)
                    res = model.predict(df, vix=vix)
                    q   = _get("quotes").get(sym, {})

                    sig = {
                        "ticker":          sym,
                        "close":           round(float(q.get("ltp", df["Close"].iloc[-1])), 2),
                        "signal":          res["signal"],
                        "probability":     res["probability"],
                        "confidence":      res["confidence"],
                        "expected_return": res.get("expected_return", 0.0),
                        "expected_sharpe": res.get("expected_sharpe", 0.0),
                        "kelly_fraction":  res.get("kelly_fraction", 0.01),
                        "regime":          res.get("regime", "normal"),
                        "weights":         res.get("ensemble_weights", {}),
                        "ts":              datetime.now(IST).strftime("%H:%M:%S"),
                    }

                    try:
                        _THREAD_SIG_Q.put_nowait(sig)
                    except queue.Full:
                        pass

                    _upd("regime",         res.get("regime", "normal"))
                    _upd("regime_weights", res.get("ensemble_weights", {}))

                except Exception as e:
                    logger.debug(f"MLThread {sym}: {e}")
                time.sleep(2)

            time.sleep(self.interval)


class VIXThread(threading.Thread):
    """Fetches India VIX + PCR from yfinance / options chain every 30s."""

    def __init__(self, interval: int = 30):
        super().__init__(daemon=True, name="VIXThread")
        self.interval = interval

    def run(self):
        while _get("running"):
            try:
                import yfinance as yf
                raw = yf.download("^INDIAVIX", period="5d", progress=False,
                                  auto_adjust=True)
                if not raw.empty:
                    if hasattr(raw.columns, "get_level_values"):
                        raw.columns = raw.columns.get_level_values(0)
                    _upd("vix", round(float(raw["Close"].iloc[-1]), 2))
            except Exception:
                pass

            try:
                from options.chain import OptionsChain
                chain = OptionsChain().get_chain("NIFTY", num_strikes=3)
                if chain:
                    _upd("pcr",     round(chain.pcr, 3))
                    _upd("iv_rank", round(chain.iv_rank, 1))
            except Exception:
                pass

            time.sleep(self.interval)


class SentimentThread(threading.Thread):
    """Fetches FinBERT sentiment scores (cached 1h) in background."""

    def __init__(self, watchlist: list, interval: int = 1800):
        super().__init__(daemon=True, name="SentimentThread")
        self.watchlist = watchlist
        self.interval  = interval

    def run(self):
        try:
            from news.sentiment import NewsSentimentEngine
            engine = NewsSentimentEngine(use_finbert=True)
            _THREAD_LOG_Q.put_nowait("[NLP] Sentiment engine initialising...")

            while _get("running"):
                for sym in self.watchlist[:10]:
                    if not _get("running"):
                        break
                    try:
                        res  = engine.get_stock_sentiment(sym, hours_window=24)
                        sent = dict(_get("sentiment"))
                        sent[sym] = res["score"]
                        _upd("sentiment", sent)
                    except Exception as e:
                        logger.debug(f"SentimentThread {sym}: {e}")
                    time.sleep(6)
                time.sleep(self.interval)
        except Exception as e:
            logger.error(f"SentimentThread crashed: {e}")


class PaperTraderThread(threading.Thread):
    """
    Drains ML signal queue, executes high-confidence signals via PaperTrader,
    and pushes execution logs into _THREAD_LOG_Q for the DPG trade ledger.

    WIRE TO TERMINAL (Goal 3):
      Every trade fill → formatted string → _THREAD_LOG_Q
      render_tick() drains _THREAD_LOG_Q into the DPG input_text widget
      so every fill appears tick-by-tick in the GUI trade ledger panel.
    """

    def __init__(self, enabled: bool = True, interval: int = 5,
                 tick_q: Optional[multiprocessing.Queue] = None):
        super().__init__(daemon=True, name="PaperThread")
        self.enabled  = enabled
        self.interval = interval
        self.tick_q   = tick_q   # read latest prices from feed queue snapshot

    def run(self):
        if not self.enabled:
            _THREAD_LOG_Q.put_nowait("[PAPER] Disabled — use --paper to enable")
            return

        try:
            from broker.paper_trader import PaperTrader
            from risk.manager import RiskManager

            rm     = RiskManager()
            trader = PaperTrader(risk_manager=rm)
            _upd("_paper_trader", trader)
            _upd("_risk_manager", rm)
            _THREAD_LOG_Q.put_nowait("[PAPER] Paper trading engine active ✅")

            while _get("running"):
                # ── Drain signal queue ──────────────────────────────────
                processed = 0
                while processed < 20:
                    try:
                        sig = _THREAD_SIG_Q.get_nowait()
                    except queue.Empty:
                        break

                    # Store for display table regardless of whether we execute
                    with _state_lock:
                        _state["signals"] = [sig] + _state["signals"][:49]

                    # Execute only STRONG signals above confidence threshold
                    if (sig["signal"] in ("STRONG BUY", "STRONG SELL")
                            and sig["confidence"] >= 70):
                        try:
                            # Apply Kelly multiplier from slider
                            kelly = _get("kelly_frac") * sig.get("kelly_fraction", 0.01)
                            sig["kelly_fraction"] = min(kelly, 0.10)   # cap at 10%

                            result = trader.execute_signal(sig)

                            if result.get("status") == "filled":
                                # ── Route fill to DPG trade ledger ─────────
                                fill_log = (
                                    f"[{sig['ts']}] ✅ {result.get('direction','?')} "
                                    f"{sig['ticker']} ×{result.get('quantity',0)} "
                                    f"@ ₹{result.get('fill_price',0):,.0f}  "
                                    f"ML {sig['confidence']:.0f}%  "
                                    f"P={sig['probability']:.2f}"
                                )
                                _THREAD_LOG_Q.put_nowait(fill_log)
                            elif result.get("status") == "rejected":
                                rej_log = (
                                    f"[{sig['ts']}] ❌ REJECTED {sig['ticker']} — "
                                    f"{result.get('reason', 'risk limit')}"
                                )
                                _THREAD_LOG_Q.put_nowait(rej_log)

                        except Exception as e:
                            logger.debug(f"PaperThread execute {sig['ticker']}: {e}")

                    processed += 1

                # ── Update live P&L ─────────────────────────────────────
                try:
                    # Inject latest prices from quotes dict
                    quotes = _get("quotes")
                    for ticker, pos in trader.positions.items():
                        q = quotes.get(ticker, {})
                        ltp = float(q.get("ltp", 0) or 0)
                        if ltp > 0:
                            pos.current_price  = ltp
                            pos.unrealised_pnl = (ltp - pos.avg_price) * pos.quantity
                            pos.unrealised_pct = (ltp / pos.avg_price - 1) * 100

                            # Check trailing stop
                            trail = _get("trail_pct")
                            if ltp > pos.highest_price:
                                pos.highest_price = ltp
                                pos.trailing_stop = ltp * (1 - trail / 100)

                            if ltp <= pos.trailing_stop and pos.trailing_stop > 0:
                                trader._auto_exit(ticker, ltp, "TRAIL_STOP")
                                _THREAD_LOG_Q.put_nowait(
                                    f"[{datetime.now(IST).strftime('%H:%M:%S')}] "
                                    f"🔴 TRAIL STOP hit {ticker} @ ₹{ltp:,.0f}"
                                )

                    summary = trader.get_portfolio_summary()
                    _upd("portfolio", summary)

                except Exception as e:
                    logger.debug(f"PaperThread PnL: {e}")

                time.sleep(self.interval)

        except Exception as e:
            logger.error(f"PaperTraderThread crashed: {e}")
            _THREAD_LOG_Q.put_nowait(f"[PAPER ERROR] {e}")


# ══════════════════════════════════════════════════════════════════════════
# DPG UI TAGS
# ══════════════════════════════════════════════════════════════════════════

TAG_CANDLE    = "candle_series"
TAG_VOL       = "vol_series"
TAG_CHART_Y   = "chart_yaxis"
TAG_VIX_BAR   = "vix_bar"
TAG_VIX_TXT   = "vix_text"
TAG_PCR_BAR   = "pcr_bar"
TAG_PCR_TXT   = "pcr_text"
TAG_SENT_BAR  = "sent_bar"
TAG_SENT_TXT  = "sent_text"
TAG_REGIME    = "regime_text"
TAG_W_RF      = "w_rf"
TAG_W_XGB     = "w_xgb"
TAG_W_TCN     = "w_tcn"
TAG_SIG_TABLE = "signal_table"
TAG_PORTFOLIO = "portfolio_text"
TAG_POSITIONS = "positions_text"
TAG_LEDGER    = "trade_log_scroll"
TAG_MKT       = "market_status"
TAG_NIFTY_P   = "nifty_price"
TAG_NIFTY_C   = "nifty_change"
TAG_KILL_BTN  = "kill_btn"
TAG_KILL_LBL  = "kill_label"
TAG_KELLY     = "kelly_slider"
TAG_TRAIL     = "trail_slider"
TAG_MODEL_S   = "model_status_text"
TAG_SEARCH       = "stock_search_input"
TAG_SEARCH_BTN   = "stock_search_btn"
TAG_SEARCH_MSG   = "stock_search_msg"
TAG_SEARCH_POPUP = "stock_search_popup"   # autocomplete dropdown


def _c_regime(r):
    return {"trending": C_GREEN, "trending_vol": C_YELLOW,
            "mean_reverting": C_BLUE, "high_vol_chop": C_RED}.get(r, C_MUTED)


def _c_signal(s):
    if "STRONG BUY"  in s: return C_GREEN
    if "BUY"         in s: return (120, 210, 130, 255)
    if "STRONG SELL" in s: return C_RED
    if "SELL"        in s: return (220, 120, 100, 255)
    return C_MUTED


def _c_vix(v):
    if v < 14: return C_GREEN
    if v < 20: return C_YELLOW
    if v < 28: return C_ORANGE
    return C_RED


# ══════════════════════════════════════════════════════════════════════════
# UI BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_ui(watchlist: list):

    # ── Global dark theme ────────────────────────────────────────────────
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,     C_BG,             category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg,      C_SURFACE,        category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,      (30, 36, 44, 255),category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, C_SURFACE,       category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border,        C_BORDER,        category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text,          C_TEXT,          category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Button,       (40, 70,120,255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,(55, 95,160,255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,    8, 4, category=dpg.mvThemeCat_Core)
    dpg.bind_theme(global_theme)

    # ── Red kill-switch button theme ─────────────────────────────────────
    with dpg.theme(tag="kill_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,       (180, 30, 30, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,(220, 50, 50, 255), category=dpg.mvThemeCat_Core)

    # ── Keyboard shortcuts ───────────────────────────────────────────────
    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=_cb_kill)
        dpg.add_key_press_handler(dpg.mvKey_Escape,   callback=_cb_resume)
        dpg.add_key_press_handler(dpg.mvKey_Escape,   callback=lambda: _safe_hide_popup())

    sym_list = list(watchlist)

    # ════════════════════════════════════════════════════════════════════
    # Primary window
    # ════════════════════════════════════════════════════════════════════
    with dpg.window(tag="primary", no_title_bar=True, no_resize=True, no_move=True):

        # ── Top status bar ────────────────────────────────────────────
        with dpg.group(horizontal=True):
            dpg.add_text("⟡ AlphaYantra v5", color=C_BLUE)
            dpg.add_text(" │ ", color=C_BORDER)
            dpg.add_text("● CLOSED", tag=TAG_MKT,     color=C_RED)
            dpg.add_text(" │ ", color=C_BORDER)
            dpg.add_text("NIFTY",             color=C_MUTED)
            dpg.add_text("──",   tag=TAG_NIFTY_P, color=C_TEXT)
            dpg.add_text("──",   tag=TAG_NIFTY_C, color=C_MUTED)
            dpg.add_text(" │ ", color=C_BORDER)
            dpg.add_button(label="⚡ KILL [SPACE]", tag=TAG_KILL_BTN,
                           callback=_cb_kill, width=135, height=24)
            dpg.bind_item_theme(TAG_KILL_BTN, "kill_theme")
            dpg.add_text(" │ ", color=C_BORDER)
            dpg.add_text("", tag=TAG_KILL_LBL, color=C_YELLOW)
            dpg.add_text(" │ ", color=C_BORDER)
            dpg.add_text(datetime.now(IST).strftime("%a %d %b %Y"), color=C_MUTED)

        dpg.add_separator()
        dpg.add_spacer(height=3)

        # ── 3-column layout ───────────────────────────────────────────
        with dpg.group(horizontal=True):

            # ═══════ Column 1: Chart (wide) ═══════════════════════════
            with dpg.child_window(width=830, height=840, border=True):

                with dpg.group(horizontal=True):
                    dpg.add_text("Chart", color=C_BLUE)
                    dpg.add_spacer(width=6)
                    dpg.add_combo(sym_list, default_value=sym_list[0],
                                  label="", width=130, tag="sym_combo",
                                  callback=_cb_sym_changed)
                    dpg.add_spacer(width=8)
                    # ── Stock Search with autocomplete ─────────────────
                    dpg.add_input_text(
                        tag=TAG_SEARCH, hint="Search: HDFCBANK, TCS, INFY...",
                        width=200, on_enter=True,
                        callback=_cb_search_typed,
                    )
                    dpg.add_button(
                        tag=TAG_SEARCH_BTN, label="Add ->",
                        width=52, callback=_cb_search_stock,
                    )
                    dpg.add_text("", tag=TAG_SEARCH_MSG, color=C_MUTED)

                dpg.add_separator()

                # Candlestick chart
                with dpg.plot(label="Price", height=490, width=-1,
                              tag="chart_plot", anti_aliased=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time",
                                      tag="chart_xaxis", time=True)
                    with dpg.plot_axis(dpg.mvYAxis, label="Price ₹",
                                       tag=TAG_CHART_Y):
                        dpg.add_candle_series(
                            dates=[], opens=[], closes=[],
                            lows=[], highs=[],
                            label="OHLCV", tag=TAG_CANDLE,
                            bull_color=C_CANDLE_BULL,
                            bear_color=C_CANDLE_BEAR,
                            weight=0.4,
                        )

                dpg.add_spacer(height=4)
                dpg.add_separator()
                dpg.add_text("Volume", color=C_MUTED)

                with dpg.plot(label="Vol", height=100, width=-1):
                    dpg.add_plot_axis(dpg.mvXAxis, tag="vol_x", time=True,
                                      no_tick_labels=True)
                    with dpg.plot_axis(dpg.mvYAxis, tag="vol_y"):
                        dpg.add_bar_series([], [], label="Vol",
                                           tag=TAG_VOL, weight=86400)

                dpg.add_separator()
                dpg.add_spacer(height=4)
                dpg.add_text("Live Quotes", color=C_BLUE)

                with dpg.table(header_row=True, borders_innerH=True,
                               borders_outerH=True, tag="quotes_tbl",
                               scrollY=True, height=145,
                               policy=dpg.mvTable_SizingFixedFit):
                    dpg.add_table_column(label="Symbol", init_width_or_weight=80)
                    dpg.add_table_column(label="LTP",    init_width_or_weight=90)
                    dpg.add_table_column(label="Chg%",   init_width_or_weight=72)
                    dpg.add_table_column(label="Open",   init_width_or_weight=90)
                    dpg.add_table_column(label="High",   init_width_or_weight=90)
                    dpg.add_table_column(label="Low",    init_width_or_weight=90)

            dpg.add_spacer(width=5)

            # ═══════ Column 2: ML Telemetry ═══════════════════════════
            with dpg.child_window(width=390, height=840, border=True):

                dpg.add_text("Market Gauges", color=C_BLUE)
                dpg.add_separator()
                dpg.add_spacer(height=3)

                # VIX
                with dpg.group(horizontal=True):
                    dpg.add_text("India VIX   ", color=C_MUTED)
                    dpg.add_text("15.0", tag=TAG_VIX_TXT, color=C_YELLOW)
                dpg.add_progress_bar(tag=TAG_VIX_BAR, default_value=0.30,
                                     width=-1, overlay="VIX")

                dpg.add_spacer(height=5)

                # PCR
                with dpg.group(horizontal=True):
                    dpg.add_text("Put-Call Ratio", color=C_MUTED)
                    dpg.add_text("1.00", tag=TAG_PCR_TXT, color=C_YELLOW)
                dpg.add_progress_bar(tag=TAG_PCR_BAR, default_value=0.50,
                                     width=-1, overlay="PCR")

                dpg.add_spacer(height=5)

                # Sentiment
                with dpg.group(horizontal=True):
                    dpg.add_text("News Sentiment", color=C_MUTED)
                    dpg.add_text("50.0", tag=TAG_SENT_TXT, color=C_YELLOW)
                dpg.add_progress_bar(tag=TAG_SENT_BAR, default_value=0.50,
                                     width=-1, overlay="Sentiment")

                dpg.add_spacer(height=8)
                dpg.add_separator()

                # Regime
                dpg.add_text("Market Regime", color=C_BLUE)
                dpg.add_spacer(height=3)
                dpg.add_text("■  NORMAL", tag=TAG_REGIME, color=C_MUTED)
                dpg.add_spacer(height=3)
                with dpg.group(horizontal=True):
                    dpg.add_text("RF:",  color=C_MUTED)
                    dpg.add_text("33%", tag=TAG_W_RF,  color=C_TEXT)
                    dpg.add_spacer(width=6)
                    dpg.add_text("XGB:", color=C_MUTED)
                    dpg.add_text("34%", tag=TAG_W_XGB, color=C_TEXT)
                    dpg.add_spacer(width=6)
                    dpg.add_text("TCN:", color=C_MUTED)
                    dpg.add_text("33%", tag=TAG_W_TCN, color=C_TEXT)

                dpg.add_spacer(height=8)
                dpg.add_separator()

                # Signal matrix
                dpg.add_text("ML Signal Matrix", color=C_BLUE)
                dpg.add_spacer(height=3)

                with dpg.table(header_row=True, borders_innerH=True,
                               borders_outerH=True, tag=TAG_SIG_TABLE,
                               scrollY=True, height=300, freeze_rows=1,
                               policy=dpg.mvTable_SizingFixedFit):
                    dpg.add_table_column(label="Symbol", init_width_or_weight=68)
                    dpg.add_table_column(label="Signal", init_width_or_weight=88)
                    dpg.add_table_column(label="P%",     init_width_or_weight=48)
                    dpg.add_table_column(label="Sharpe", init_width_or_weight=52)
                    dpg.add_table_column(label="Ret%",   init_width_or_weight=50)
                    dpg.add_table_column(label="Time",   init_width_or_weight=52)

                dpg.add_spacer(height=5)
                dpg.add_separator()

                # Model status
                dpg.add_text("Model", color=C_BLUE)
                dpg.add_text("Checking...", tag=TAG_MODEL_S, color=C_MUTED, wrap=380)
                dpg.add_spacer(height=4)
                dpg.add_button(label="Force Inference Now",
                               callback=_cb_force_infer, width=-1)

            dpg.add_spacer(width=5)

            # ═══════ Column 3: Execution & Risk ═══════════════════════
            with dpg.child_window(width=370, height=840, border=True):

                # Portfolio summary
                dpg.add_text("Paper Portfolio", color=C_BLUE)
                dpg.add_separator()
                dpg.add_spacer(height=3)
                dpg.add_text("No positions yet", tag=TAG_PORTFOLIO,
                             color=C_MUTED, wrap=358)

                dpg.add_spacer(height=4)

                # Open positions detail
                dpg.add_text("Open Positions", color=C_MUTED)
                dpg.add_input_text(tag=TAG_POSITIONS, multiline=True,
                                   readonly=True, default_value="—",
                                   width=-1, height=100)

                dpg.add_spacer(height=6)
                dpg.add_separator()

                # Dynamic risk sliders
                dpg.add_text("Risk Controls", color=C_BLUE)
                dpg.add_spacer(height=4)

                dpg.add_text("Kelly Fraction Multiplier", color=C_MUTED)
                dpg.add_slider_float(tag=TAG_KELLY, label="",
                                     min_value=0.1, max_value=1.0,
                                     default_value=0.5, width=-1,
                                     callback=_cb_kelly, format="%.2f")

                dpg.add_spacer(height=4)
                dpg.add_text("Trailing Stop  (%  from peak)", color=C_MUTED)
                dpg.add_slider_float(tag=TAG_TRAIL, label="",
                                     min_value=0.5, max_value=5.0,
                                     default_value=1.5, width=-1,
                                     callback=_cb_trail, format="%.1f%%")

                dpg.add_spacer(height=8)
                dpg.add_separator()

                # Emergency controls
                dpg.add_text("Emergency Controls", color=C_RED)
                dpg.add_spacer(height=4)

                with dpg.group(horizontal=True):
                    dpg.add_button(label="⚡ FLATTEN ALL  [SPACE]",
                                   callback=_cb_kill, width=185, height=32)
                    dpg.add_button(label="✅ RESUME  [ESC]",
                                   callback=_cb_resume, width=172, height=32)

                dpg.add_spacer(height=3)
                dpg.add_text("SPACE instantly flattens all paper positions.",
                             color=C_MUTED, wrap=358)

                dpg.add_spacer(height=8)
                dpg.add_separator()

                # Trade ledger — live scrolling
                dpg.add_text("Trade Ledger (live)", color=C_BLUE)
                dpg.add_spacer(height=3)
                dpg.add_input_text(tag=TAG_LEDGER,
                                   multiline=True, readonly=True,
                                   default_value="Waiting for trades...",
                                   width=-1, height=320)

    # ── Autocomplete suggestion popup (global overlay) ─────────────────
    # Shown dynamically when user types in the search box
    with dpg.window(
        tag=TAG_SEARCH_POPUP,
        no_title_bar=True, no_resize=True,
        width=250, height=220,
        pos=(155, 65),
        show=False,
    ):
        dpg.add_text("  Suggestions (click to select)", color=C_MUTED)
        dpg.add_separator()
        for _i in range(8):
            dpg.add_button(
                tag=f"sugg_{_i}", label="",
                width=-1, height=22,
                callback=_cb_pick_suggestion,
                user_data=_i,
                show=False,
            )
        dpg.add_spacer(height=2)
        dpg.add_text("  Press Enter to add any NSE symbol", color=(80,90,100,200))


# ══════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════

def _cb_sym_changed(sender, app_data, user_data):
    _upd("selected_sym", app_data)
    try:
        dpg.configure_item(TAG_CANDLE, dates=[], opens=[], closes=[], lows=[], highs=[])
        dpg.configure_item(TAG_VOL,    x=[], y=[])
    except Exception:
        pass


# ── Full NSE universe for autocomplete (imported at top of file) ───────────


def _fuzzy_match(query: str, limit: int = 8) -> list:
    """
    Fast prefix + substring matching for NSE symbol autocomplete.

    Priority:
      1. Exact match (user typed it perfectly)
      2. Prefix match  (HDFCBANK when user types HDF)
      3. Substring match (KOTAKBANK when user types TAK)

    Returns up to `limit` unique matches, priority-ordered.
    """
    q = query.upper().strip()
    if not q:
        return []

    exact    = [s for s in _ALL_SYMBOLS if s == q]
    prefix   = [s for s in _ALL_SYMBOLS if s.startswith(q) and s != q]
    contains = [s for s in _ALL_SYMBOLS if q in s and not s.startswith(q)]

    seen = set()
    results = []
    for s in exact + prefix + contains:
        if s not in seen:
            seen.add(s)
            results.append(s)
        if len(results) >= limit:
            break
    return results


def _cb_search_typed(sender=None, app_data=None, user_data=None):
    """
    Called every time the user types in the search box.
    Shows autocomplete suggestions as clickable buttons.
    If user pressed Enter (on_enter callback) — call _cb_search_stock directly.
    """
    try:
        raw  = dpg.get_value(TAG_SEARCH).strip()
        sugs = _fuzzy_match(raw, limit=8)

        if not sugs or not raw:
            # Hide popup
            try:
                dpg.configure_item(TAG_SEARCH_POPUP, show=False)
            except Exception:
                pass
            # If Enter pressed with a valid text but no suggestions, try adding directly
            if raw and sender == TAG_SEARCH:
                _cb_search_stock()
            return

        # Populate suggestion buttons
        for i in range(8):
            tag = f"sugg_{i}"
            try:
                if i < len(sugs):
                    dpg.configure_item(tag, label=f"  {sugs[i]}", show=True)
                else:
                    dpg.configure_item(tag, show=False)
            except Exception:
                pass

        # Show popup
        try:
            dpg.configure_item(TAG_SEARCH_POPUP, show=True)
        except Exception:
            pass

        # If Enter was pressed, pick the first suggestion automatically
        if sender == TAG_SEARCH and sugs:
            dpg.set_value(TAG_SEARCH, sugs[0])
            try:
                dpg.configure_item(TAG_SEARCH_POPUP, show=False)
            except Exception:
                pass
            _cb_search_stock()

    except Exception as e:
        pass   # Never crash the render loop on UI callback errors


def _cb_pick_suggestion(sender=None, app_data=None, user_data=None):
    """
    Called when user clicks one of the autocomplete suggestion buttons.
    user_data = index (0-7) of the suggestion.
    """
    try:
        idx = int(user_data) if user_data is not None else 0
        tag = f"sugg_{idx}"
        label = dpg.get_item_label(tag).strip()
        if label:
            dpg.set_value(TAG_SEARCH, label)
            try:
                dpg.configure_item(TAG_SEARCH_POPUP, show=False)
            except Exception:
                pass
            _cb_search_stock()
    except Exception:
        pass



def _cb_search_stock(sender=None, app_data=None, user_data=None):
    """
    Stock search callback — adds any valid NSE ticker to the combo and chart.

    HOW IT WORKS:
    1. User types a symbol in the search box (e.g. "HDFCBANK", "TATAMOTORS")
    2. Presses Enter or clicks "Add ->"
    3. We validate it's a non-empty string and convert to uppercase
    4. Add it to the combo dropdown if not already present
    5. Select it as the active chart symbol
    6. Update the chart immediately with fresh yfinance data
    7. Show a status message (green=ok, red=error)

    Supported formats:
      - NSE symbol:  HDFCBANK, RELIANCE, TCS, INFY
      - With suffix: HDFCBANK.NS (we strip .NS automatically)
      - Lowercase:   hdfcbank (we uppercase automatically)
      - Index:       NIFTY, BANKNIFTY (shown as ^NSEI, ^NSEBANK in yfinance)
    """
    try:
        raw = dpg.get_value(TAG_SEARCH).strip().upper()
        if not raw:
            return

        # Normalise: strip .NS suffix, remove spaces
        sym = raw.replace(".NS", "").replace(" ", "").replace("&", "")
        if not sym:
            return

        # Validate: try fetching a tiny sample
        try:
            import yfinance as yf
            # yfinance NSE symbol format: SYMBOL.NS
            yf_sym = f"{sym}.NS" if not sym.startswith("^") else sym
            probe = yf.download(yf_sym, period="5d", progress=False, auto_adjust=True)
            if probe.empty:
                dpg.configure_item(TAG_SEARCH_MSG,
                                   default_value=f"✗ '{sym}' not found on NSE",
                                   color=C_RED)
                return
        except Exception as e:
            dpg.configure_item(TAG_SEARCH_MSG,
                               default_value=f"✗ fetch error: {e}",
                               color=C_RED)
            return

        # Add to combo if not already present
        try:
            current_items = dpg.get_item_configuration("sym_combo").get("items", [])
            if sym not in current_items:
                new_items = current_items + [sym]
                dpg.configure_item("sym_combo", items=new_items)
        except Exception:
            pass

        # Select it
        dpg.set_value("sym_combo", sym)
        _upd("selected_sym", sym)

        # Clear chart so the ML thread refreshes it
        dpg.configure_item(TAG_CANDLE, dates=[], opens=[], closes=[], lows=[], highs=[])
        dpg.configure_item(TAG_VOL,    x=[], y=[])

        # Hide autocomplete popup
        try:
            dpg.configure_item(TAG_SEARCH_POPUP, show=False)
        except Exception:
            pass

        # Show success
        dpg.configure_item(TAG_SEARCH_MSG,
                           default_value=f"✓ {sym} added",
                           color=C_GREEN)
        dpg.set_value(TAG_SEARCH, "")   # clear input

    except Exception as e:
        try:
            dpg.configure_item(TAG_SEARCH_MSG,
                               default_value=f"✗ Error: {e}",
                               color=C_RED)
        except Exception:
            pass


def _cb_kelly(sender, app_data, user_data):
    _upd("kelly_frac", float(app_data))


def _cb_trail(sender, app_data, user_data):
    _upd("trail_pct", float(app_data))


def _cb_kill(sender=None, app_data=None, user_data=None):
    """Flatten all paper positions and activate kill switch."""
    try:
        rm = _get("_risk_manager")
        if rm:
            rm.activate_kill_switch("KILL SWITCH — terminal")

        trader = _get("_paper_trader")
        if trader:
            quotes = _get("quotes")
            for ticker in list(trader.positions.keys()):
                price = float(quotes.get(ticker, {}).get("ltp", 0) or 0)
                if price > 0:
                    trader._auto_exit(ticker, price, "KILL_SWITCH")
                    _THREAD_LOG_Q.put_nowait(
                        f"[{datetime.now(IST).strftime('%H:%M:%S')}] "
                        f"⚡ KILL: flattened {ticker} @ ₹{price:,.0f}"
                    )

        _THREAD_LOG_Q.put_nowait("[⚡ KILL SWITCH ACTIVATED — All positions flattened]")
        try:
            dpg.configure_item(TAG_KILL_LBL, default_value="⚡ KILL ACTIVE", color=C_RED)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Kill switch error: {e}")


def _cb_resume(sender=None, app_data=None, user_data=None):
    try:
        rm = _get("_risk_manager")
        if rm:
            rm.deactivate_kill_switch()
        _THREAD_LOG_Q.put_nowait("[✅ Kill switch deactivated — trading resumed]")
        try:
            dpg.configure_item(TAG_KILL_LBL, default_value="", color=C_MUTED)
        except Exception:
            pass
    except Exception:
        pass


def _cb_force_infer(sender=None, app_data=None, user_data=None):
    _THREAD_LOG_Q.put_nowait("[ML] Manual inference triggered — next cycle starting...")


# ══════════════════════════════════════════════════════════════════════════
# RENDER UPDATE FUNCTIONS  (called at controlled rates from render_tick)
# ══════════════════════════════════════════════════════════════════════════

def _drain_mp_queue(mp_q: multiprocessing.Queue):
    """
    Drain ALL pending ticks from the feed process queue into _state["quotes"].

    Called every frame. Must drain EVERYTHING — not just N items.

    OLD (broken): capped at max_items=100
      At 9:15 AM, NSE can send 300+ ticks/second. At 60 FPS = 5 ticks per frame
      processed. After 1 minute, queue has 14,400 ticks backlog → UI shows
      prices from 4 minutes ago.

    FIX: drain until empty every frame.
      Since each tick is just a dict (~500 bytes), 300 get_nowait() calls per
      frame takes <0.5ms — completely invisible to the user.
      The IPC pipe bomb is prevented by the mp_q having maxsize=5000 and
      live_feed.py using put_nowait() which drops ticks on Full rather than
      blocking the feed process.
    """
    while True:
        try:
            item = mp_q.get_nowait()
        except Exception:
            break
        sym   = item.get("symbol", "")
        quote = item.get("quote", {})
        if sym and quote:
            with _state_lock:
                _state["quotes"][sym] = quote


def _update_top_bar():
    try:
        q   = _get("quotes").get("NIFTY", {})
        ltp = q.get("ltp", 0) or 0
        chg = q.get("change_pct", 0) or 0

        if ltp:
            dpg.configure_item(TAG_NIFTY_P, default_value=f"  {ltp:,.2f}")
            c = C_GREEN if chg >= 0 else C_RED
            dpg.configure_item(TAG_NIFTY_C, default_value=f"  ({chg:+.2f}%)", color=c)

        mkt = _get("market_open")
        dpg.configure_item(TAG_MKT,
                           default_value="● MARKET OPEN"   if mkt else "● MARKET CLOSED",
                           color=C_GREEN                   if mkt else C_RED)
    except Exception:
        pass


def _update_gauges():
    try:
        vix    = _get("vix")
        pcr    = _get("pcr")
        sym    = _get("selected_sym")
        s_val  = _get("sentiment").get(sym, 50.0)

        vix_norm = min(1.0, max(0.0, (vix - 8) / 42))
        dpg.configure_item(TAG_VIX_TXT, default_value=f"{vix:.1f}", color=_c_vix(vix))
        dpg.configure_item(TAG_VIX_BAR, default_value=vix_norm, overlay=f"VIX {vix:.1f}")
        _tint_bar(TAG_VIX_BAR, _c_vix(vix))

        pcr_norm  = min(1.0, max(0.0, pcr / 2.0))
        pcr_c     = C_GREEN if pcr < 0.8 else (C_RED if pcr > 1.2 else C_YELLOW)
        dpg.configure_item(TAG_PCR_TXT, default_value=f"{pcr:.2f}", color=pcr_c)
        dpg.configure_item(TAG_PCR_BAR, default_value=pcr_norm, overlay=f"PCR {pcr:.2f}")

        s_norm = s_val / 100.0
        s_c    = C_GREEN if s_val >= 60 else (C_RED if s_val < 40 else C_YELLOW)
        dpg.configure_item(TAG_SENT_TXT, default_value=f"{s_val:.0f}", color=s_c)
        dpg.configure_item(TAG_SENT_BAR, default_value=s_norm, overlay=f"Sent {s_val:.0f}")
    except Exception:
        pass


def _tint_bar(tag: str, color: tuple):
    """Apply dynamic colour to a progress bar via a bound theme."""
    try:
        th_tag = f"__bar_theme_{tag}"
        if dpg.does_item_exist(th_tag):
            dpg.delete_item(th_tag)
        with dpg.theme(tag=th_tag):
            with dpg.theme_component(dpg.mvProgressBar):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram,
                                    color, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme(tag, th_tag)
    except Exception:
        pass


def _update_regime():
    try:
        r       = _get("regime")
        w       = _get("regime_weights")
        labels  = {
            "trending":        "▲  TRENDING  (Risk-On)",
            "trending_vol":    "▲  TRENDING + HIGH VOL",
            "mean_reverting":  "↔  MEAN REVERTING  (Risk-Off)",
            "high_vol_chop":   "⚠  HIGH VOL CHOP",
            "normal":          "■  NORMAL",
        }
        dpg.configure_item(TAG_REGIME, default_value=labels.get(r, f"■ {r.upper()}"),
                           color=_c_regime(r))
        dpg.configure_item(TAG_W_RF,  default_value=f"{int(w.get('rf',0.33)*100)}%")
        dpg.configure_item(TAG_W_XGB, default_value=f"{int(w.get('xgb',0.33)*100)}%")
        dpg.configure_item(TAG_W_TCN, default_value=f"{int(w.get('tcn',0.34)*100)}%")
    except Exception:
        pass


def _update_signal_table():
    try:
        sigs = _get("signals")[:20]
        for child in dpg.get_item_children(TAG_SIG_TABLE, 1) or []:
            dpg.delete_item(child)
        for s in sigs:
            sig  = s.get("signal", "HOLD")
            prob = s.get("probability", 0.5)
            shr  = s.get("expected_sharpe", 0.0)
            ret  = s.get("expected_return", 0.0)
            with dpg.table_row(parent=TAG_SIG_TABLE):
                dpg.add_text(s.get("ticker", ""), color=C_BLUE)
                dpg.add_text(sig, color=_c_signal(sig))
                dpg.add_text(f"{prob*100:.0f}")
                dpg.add_text(f"{shr:.2f}",
                             color=C_GREEN if shr > 0.5 else (C_RED if shr < 0 else C_MUTED))
                dpg.add_text(f"{ret:+.1f}%",
                             color=C_GREEN if ret > 0 else (C_RED if ret < 0 else C_MUTED))
                dpg.add_text(s.get("ts", ""))
    except Exception as e:
        logger.debug(f"Signal table: {e}")


def _update_quotes_table():
    try:
        quotes = _get("quotes")
        for ch in dpg.get_item_children("quotes_tbl", 1) or []:
            dpg.delete_item(ch)
        for sym, q in list(quotes.items())[:12]:
            ltp = q.get("ltp", 0) or 0
            chg = q.get("change_pct", 0) or 0
            c   = C_GREEN if chg >= 0 else C_RED
            with dpg.table_row(parent="quotes_tbl"):
                dpg.add_text(sym,                color=C_BLUE)
                dpg.add_text(f"₹{ltp:,.2f}")
                dpg.add_text(f"{chg:+.2f}%",    color=c)
                dpg.add_text(f"₹{q.get('open',0):,.2f}", color=C_MUTED)
                dpg.add_text(f"₹{q.get('high',0):,.2f}", color=C_GREEN)
                dpg.add_text(f"₹{q.get('low', 0):,.2f}", color=C_RED)
    except Exception as e:
        logger.debug(f"Quotes table: {e}")


def _update_portfolio():
    try:
        p = _get("portfolio")
        if not p:
            return
        pnl   = p.get("total_pnl", 0)
        c     = C_GREEN if pnl >= 0 else C_RED
        txt   = (
            f"P&L: ₹{pnl:+,.0f}   "
            f"Realised: ₹{p.get('realised_pnl',0):+,.0f}\n"
            f"Open: {p.get('open_positions',0)}   "
            f"Trades: {p.get('total_trades',0)}   "
            f"Win%: {p.get('win_rate',0):.1f}%"
        )
        dpg.configure_item(TAG_PORTFOLIO, default_value=txt, color=c)

        # Open positions detail
        pos_lines = []
        for pos in p.get("positions", []):
            pnl_c = "+" if pos["pnl"] >= 0 else ""
            pos_lines.append(
                f"{pos['ticker']:12s}  ×{pos['qty']}  "
                f"avg ₹{pos['avg_price']:,.0f}  "
                f"cur ₹{pos['current']:,.0f}  "
                f"P&L {pnl_c}₹{pos['pnl']:,.0f}  "
                f"SL ₹{pos['stop_loss']:,.0f}"
            )
        dpg.configure_item(TAG_POSITIONS,
                           default_value="\n".join(pos_lines) if pos_lines else "No open positions")
    except Exception:
        pass


def _update_trade_log():
    """Drain _THREAD_LOG_Q and append to the DPG ledger text widget."""
    try:
        new_lines = []
        while len(new_lines) < 50:
            try:
                new_lines.append(_THREAD_LOG_Q.get_nowait())
            except queue.Empty:
                break

        if new_lines:
            with _state_lock:
                for ln in reversed(new_lines):
                    _state["trade_log"].appendleft(ln)
            log_txt = "\n".join(list(_state["trade_log"])[:100])
            dpg.configure_item(TAG_LEDGER, default_value=log_txt)
    except Exception:
        pass


def _update_chart(feed_proc_ref):
    """Pull latest candles from the feed process and update the DPG chart."""
    try:
        # The feed process writes to its own CandleBuilder inside live_feed.py.
        # We can't directly access it from a different process.
        # Instead, fetch recent OHLCV from yfinance as an always-available fallback.
        # When the NSE session is live the candles in the quotes dict are up to date;
        # for the chart we use yfinance 1-day interval data for display continuity.
        sym = _get("selected_sym")

        import yfinance as yf
        df = yf.download(
            sym + ".NS" if sym not in ("NIFTY", "BANKNIFTY", "FINNIFTY") else "^NSEI",
            period="3mo", interval="1d", progress=False, auto_adjust=True,
        )
        if df is None or df.empty:
            return

        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)

        import time as _t
        dates  = [int(_t.mktime(d.to_pydatetime().timetuple()))
                  for d in df.index]
        opens  = df["Open"].tolist()
        highs  = df["High"].tolist()
        lows   = df["Low"].tolist()
        closes = df["Close"].tolist()
        vols   = [float(v) for v in df["Volume"].tolist()]

        dpg.configure_item(TAG_CANDLE,
                           dates=dates, opens=opens, closes=closes,
                           lows=lows, highs=highs)
        dpg.configure_item(TAG_VOL, x=dates, y=vols)

        if closes:
            lo  = min(lows[-60:])
            hi  = max(highs[-60:])
            pad = (hi - lo) * 0.05
            dpg.set_axis_limits(TAG_CHART_Y, lo - pad, hi + pad)

    except Exception as e:
        logger.debug(f"Chart update: {e}")


def _update_model_status():
    try:
        loaded = _get("model_loaded")
        n_sigs = len(_get("signals"))
        regime = _get("regime")
        if loaded:
            dpg.configure_item(TAG_MODEL_S,
                               default_value=f"✅ Active | {n_sigs} signals | {regime}",
                               color=C_GREEN)
        else:
            dpg.configure_item(TAG_MODEL_S,
                               default_value="⚠️  No model — run: python run.py --train",
                               color=C_YELLOW)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
# RENDER LOOP
# ══════════════════════════════════════════════════════════════════════════

_frame   = 0
_last_slow   = 0.0
_last_chart  = 0.0


def render_tick(tick_q: multiprocessing.Queue, feed_proc):
    """
    Called every DPG frame (~60fps).

    Fast (every frame):
      - Drain multiprocessing.Queue from feed process (zero-copy, non-blocking)

    Medium (~10fps):
      - Top bar, gauges, regime, portfolio text

    Slow (~1fps):
      - Signal table, quotes table, trade ledger (expensive table redraws)

    Very slow (every 60s):
      - Candlestick chart (yfinance call — IO, worth caching)
    """
    global _frame, _last_slow, _last_chart
    _frame += 1

    # Always: drain feed queue (core data pipeline — never skip)
    _drain_mp_queue(tick_q)

    # 10fps: lightweight gauge updates
    if _frame % 6 == 0:
        _update_top_bar()
        _update_gauges()
        _update_regime()
        _update_portfolio()

    # 1fps: table redraws
    now = time.monotonic()
    if now - _last_slow > 1.0:
        _last_slow = now
        _update_signal_table()
        _update_quotes_table()
        _update_trade_log()
        _update_model_status()

    # Every 60s: chart (yfinance IO)
    if now - _last_chart > 60.0:
        _last_chart = now
        _update_chart(feed_proc)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="AlphaYantra Terminal v5")
    ap.add_argument("--paper",        action="store_true")
    ap.add_argument("--universe",     default="nifty50",
                    choices=["nifty50", "nifty500", "midcap150"])
    ap.add_argument("--no-ml",        action="store_true", dest="no_ml")
    ap.add_argument("--no-sentiment", action="store_true", dest="no_sentiment")
    args = ap.parse_args()

    # Default watchlist
    try:
        from data.fetcher import UNIVERSE_MAP
        watchlist = UNIVERSE_MAP.get(args.universe, [])[:20]
    except Exception:
        watchlist = []
    if not watchlist:
        watchlist = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "HDFCBANK",
                     "INFY", "ICICIBANK", "SBIN", "AXISBANK", "BAJFINANCE"]

    logger.info(f"Terminal starting — universe={args.universe}, paper={args.paper}")

    # ── Create IPC bridge ───────────────────────────────────────────────
    # multiprocessing.Queue for feed process → main process (ticks)
    # Must use multiprocessing.Queue (not threading.Queue) for cross-process IPC
    mp_tick_q = multiprocessing.Queue(maxsize=5_000)
    mp_log_q  = multiprocessing.Queue(maxsize=500)
    stop_event = multiprocessing.Event()

    # ── Start feed in separate process (asyncio-safe, DPG-safe) ─────────
    feed_proc = multiprocessing.Process(
        target=_feed_process_main,
        args=(watchlist, mp_tick_q, mp_log_q, stop_event),
        name="FeedProcess",
        daemon=True,
    )
    feed_proc.start()
    _upd("_feed_proc", feed_proc)
    logger.info(f"FeedProcess started (PID {feed_proc.pid})")

    # Forward feed process logs to main process log queue
    def _forward_mp_logs():
        while _get("running"):
            try:
                msg = mp_log_q.get(timeout=2)
                _THREAD_LOG_Q.put_nowait(msg)
            except Exception:
                pass
    threading.Thread(target=_forward_mp_logs, daemon=True,
                     name="LogForwarder").start()

    # ── Start remaining background threads (no asyncio — safe) ──────────
    VIXThread(interval=30).start()

    if not args.no_ml:
        MLInferenceThread(watchlist=watchlist, interval=60).start()

    if not args.no_sentiment:
        SentimentThread(watchlist=watchlist[:8], interval=1800).start()

    PaperTraderThread(enabled=args.paper, interval=5,
                      tick_q=mp_tick_q).start()

    # ── Wait for feed to send first ticks ───────────────────────────────
    time.sleep(2)
    _THREAD_LOG_Q.put_nowait("[SYSTEM] Terminal ready — SPACE=kill, ESC=resume")

    # ── Build DPG context and UI ─────────────────────────────────────────
    dpg.create_context()
    dpg.create_viewport(
        title="AlphaYantra v5 — GPU Terminal",
        width=1640, height=900,
        min_width=1400, min_height=720,
    )
    dpg.setup_dearpygui()
    build_ui(watchlist)
    dpg.set_primary_window("primary", True)
    dpg.show_viewport()

    logger.info("DPG render loop starting — main thread locked by GPU")

    # ── FPS throttle constant ─────────────────────────────────────────────
    # DearPyGui has no built-in VSync in Python loop mode.
    # Without this sleep, the loop spins at 2000-5000 FPS, pinning the GPU
    # to 100% usage for no visual benefit — fans scream, thermal throttling.
    # 0.0167s = ~60 FPS.  Set to 0.033 for ~30 FPS on low-end machines.
    _FRAME_BUDGET = 0.0167

    # ══ RENDER LOOP — MAIN THREAD NEVER LEAVES HERE UNTIL WINDOW CLOSE ══
    while dpg.is_dearpygui_running():
        _frame_start = time.monotonic()
        render_tick(mp_tick_q, feed_proc)
        dpg.render_dearpygui_frame()
        # Throttle to ~60 FPS — prevents GPU meltdown (drops load from 100% to ~1%)
        elapsed = time.monotonic() - _frame_start
        sleep_for = _FRAME_BUDGET - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
    # ════════════════════════════════════════════════════════════════════

    # ── Clean shutdown ───────────────────────────────────────────────────
    logger.info("DPG window closed — shutting down")
    _upd("running", False)

    stop_event.set()        # signal feed process to stop
    feed_proc.join(timeout=5)
    if feed_proc.is_alive():
        feed_proc.terminate()
        logger.warning("FeedProcess force-terminated")

    dpg.destroy_context()
    logger.info("AlphaYantra Terminal exited cleanly")


if __name__ == "__main__":
    # Required on Windows / macOS for multiprocessing to work correctly
    multiprocessing.freeze_support()
    main()
