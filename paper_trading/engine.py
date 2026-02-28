"""
paper_trading/engine.py
────────────────────────
Virtual Broker Paper Trading Engine.

Instead of real orders, this:
  1. Captures live bid/ask spread from the live feed
  2. Logs virtual trades to a JSON ledger (paper_trading/ledger.json)
  3. Tracks live P&L tick-by-tick from the feed
  4. Sends Telegram alerts for virtual entries/exits
  5. Generates daily P&L reports comparing virtual vs benchmark

This proves the system works in real-time before risking real capital.
When you're satisfied with paper performance, swap get_live_quote()
with your Zerodha Kite API calls — everything else stays identical.
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from loguru import logger

IST          = ZoneInfo("Asia/Kolkata")
LEDGER_FILE  = Path("paper_trading/ledger.json")
LEDGER_FILE.parent.mkdir(exist_ok=True)

# Simulated bid/ask spread (% of price) — realistic for NSE large-caps
SPREAD_PCT = {
    "RELIANCE": 0.0005, "TCS": 0.0005, "HDFCBANK": 0.0005,
    "NIFTY":    0.0002, "BANKNIFTY": 0.0002,
    "default":  0.001,  # 0.1% for mid/small caps
}

BROKERAGE_FLAT = 20.0   # ₹20 per order (Zerodha/Upstox model)
STT_PCT        = 0.001  # 0.1% on sell


@dataclass
class PaperTrade:
    id:            str
    ticker:        str
    direction:     str         # "LONG" or "SHORT"
    entry_time:    str
    entry_price:   float
    quantity:      int
    stop_loss:     float
    take_profit:   float
    signal:        str         # ML signal that triggered
    ml_confidence: float
    tech_score:    float
    regime:        str
    # Filled after close
    exit_time:     Optional[str]  = None
    exit_price:    Optional[float] = None
    exit_reason:   str            = ""
    gross_pnl:     float          = 0.0
    charges:       float          = 0.0
    net_pnl:       float          = 0.0
    pnl_pct:       float          = 0.0
    # Live tracking
    current_price: float          = 0.0
    unrealised_pnl: float         = 0.0
    status:        str            = "OPEN"   # OPEN / CLOSED


@dataclass
class PaperLedger:
    created_at:       str   = field(default_factory=lambda: datetime.now(IST).isoformat())
    initial_capital:  float = 500_000.0
    current_capital:  float = 500_000.0
    open_trades:      List[PaperTrade] = field(default_factory=list)
    closed_trades:    List[PaperTrade] = field(default_factory=list)
    total_trades:     int   = 0
    winning_trades:   int   = 0
    total_pnl:        float = 0.0
    total_charges:    float = 0.0


class PaperTradingEngine:
    """
    Virtual broker. Mirrors the full risk/execution logic using
    live quotes from the feed — no real money involved.
    """

    def __init__(
        self,
        initial_capital: float = 500_000,
        live_feed = None,       # LiveFeed instance for price updates
        monitor   = None,       # TelegramMonitor for alerts
    ):
        self.initial_capital = initial_capital
        self.feed    = live_feed
        self.monitor = monitor
        self.ledger  = self._load_ledger()
        self._lock   = threading.Lock()
        self._running = False

        if self.ledger.current_capital == 500_000 and initial_capital != 500_000:
            self.ledger.initial_capital = initial_capital
            self.ledger.current_capital = initial_capital

        logger.info(f"PaperTrading started | Capital: ₹{self.ledger.current_capital:,.0f}")

    # ── Entry ──────────────────────────────────────────────────────────

    def open_trade(
        self,
        ticker:        str,
        signal:        str,
        ml_confidence: float,
        tech_score:    float,
        stop_loss:     float,
        take_profit:   float,
        regime:        str   = "normal",
        quantity:      Optional[int] = None,
    ) -> Optional[PaperTrade]:
        """
        Log a virtual trade entry.
        Price is fetched from the live feed (with simulated spread).
        """
        with self._lock:
            # Check if already in this position
            open_tickers = {t.ticker for t in self.ledger.open_trades}
            if ticker in open_tickers:
                logger.debug(f"Paper: already in {ticker} — skipping")
                return None

            # Get live price + spread
            entry_price = self._get_entry_price(ticker, signal)
            if entry_price <= 0:
                logger.warning(f"Paper: no live price for {ticker}")
                return None

            # Position sizing: 5% of current capital
            if quantity is None:
                pos_value = self.ledger.current_capital * 0.05
                quantity  = max(1, int(pos_value / entry_price))

            cost     = quantity * entry_price
            charges  = BROKERAGE_FLAT + cost * 0.0001   # brokerage + exchange
            if cost + charges > self.ledger.current_capital:
                logger.warning(f"Paper: insufficient capital for {ticker}")
                return None

            trade = PaperTrade(
                id            = f"{ticker}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}",
                ticker        = ticker,
                direction     = "LONG" if "BUY" in signal else "SHORT",
                entry_time    = datetime.now(IST).isoformat(),
                entry_price   = round(entry_price, 2),
                quantity      = quantity,
                stop_loss     = round(stop_loss, 2),
                take_profit   = round(take_profit, 2),
                signal        = signal,
                ml_confidence = round(ml_confidence, 1),
                tech_score    = round(tech_score, 1),
                regime        = regime,
                current_price = entry_price,
                charges       = charges,
            )

            self.ledger.open_trades.append(trade)
            self.ledger.current_capital -= (cost + charges)
            self.ledger.total_trades    += 1
            self._save_ledger()

            logger.info(f"📝 PAPER ENTRY: {ticker} {signal} @ ₹{entry_price:.2f} × {quantity}")
            if self.monitor:
                self.monitor.send_message(
                    f"📝 <b>PAPER TRADE — {signal}</b>\n"
                    f"  {ticker} @ ₹{entry_price:.2f} × {quantity}\n"
                    f"  SL: ₹{stop_loss:.2f}  TP: ₹{take_profit:.2f}\n"
                    f"  ML: {ml_confidence:.1f}%  Regime: {regime}\n"
                    f"  Capital left: ₹{self.ledger.current_capital:,.0f}"
                )
            return trade

    # ── Exit ───────────────────────────────────────────────────────────

    def close_trade(
        self,
        trade_id: str,
        reason:   str = "MANUAL",
    ) -> Optional[PaperTrade]:
        """Close a virtual trade at current market price."""
        with self._lock:
            trade = next((t for t in self.ledger.open_trades if t.id == trade_id), None)
            if not trade:
                return None
            return self._close(trade, reason)

    def _close(self, trade: PaperTrade, reason: str) -> PaperTrade:
        """Internal close — call with lock held."""
        exit_price = self._get_exit_price(trade.ticker, trade.direction)
        charges    = BROKERAGE_FLAT + trade.quantity * exit_price * (STT_PCT + 0.0001)

        gross  = (exit_price - trade.entry_price) * trade.quantity
        if trade.direction == "SHORT":
            gross = -gross
        net    = gross - trade.charges - charges

        trade.exit_time    = datetime.now(IST).isoformat()
        trade.exit_price   = round(exit_price, 2)
        trade.exit_reason  = reason
        trade.gross_pnl    = round(gross, 2)
        trade.charges     += charges
        trade.net_pnl      = round(net, 2)
        trade.pnl_pct      = round((exit_price / trade.entry_price - 1) * 100, 2)
        trade.status       = "CLOSED"
        trade.unrealised_pnl = 0.0

        self.ledger.open_trades.remove(trade)
        self.ledger.closed_trades.append(trade)
        self.ledger.current_capital += trade.quantity * exit_price - charges
        self.ledger.total_pnl       += net
        self.ledger.total_charges   += trade.charges
        if net > 0:
            self.ledger.winning_trades += 1

        self._save_ledger()

        emoji = "✅" if net > 0 else "❌"
        logger.info(f"{emoji} PAPER EXIT: {trade.ticker} @ ₹{exit_price:.2f} | "
                    f"Net P&L: ₹{net:+,.0f} ({trade.pnl_pct:+.1f}%) | {reason}")
        if self.monitor:
            self.monitor.send_message(
                f"{emoji} <b>PAPER EXIT — {trade.ticker}</b>\n"
                f"  Exit: ₹{exit_price:.2f}  Reason: {reason}\n"
                f"  Net P&L: ₹{net:+,.0f} ({trade.pnl_pct:+.1f}%)\n"
                f"  Total Capital: ₹{self.ledger.current_capital:,.0f}"
            )
        return trade

    # ── Live P&L monitor ───────────────────────────────────────────────

    def start_monitor(self, check_interval: int = 30):
        """Start background thread that checks SL/TP every 30 seconds."""
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._monitor_loop, args=(check_interval,), daemon=True)
        t.start()
        logger.info("Paper trading monitor started")

    def stop_monitor(self):
        self._running = False

    def _monitor_loop(self, interval: int):
        while self._running:
            try:
                self._tick_all_trades()
            except Exception as e:
                logger.debug(f"Paper monitor tick error: {e}")
            time.sleep(interval)

    def _tick_all_trades(self):
        """Update unrealised P&L and check SL/TP for all open trades."""
        with self._lock:
            for trade in list(self.ledger.open_trades):
                price = self._get_current_price(trade.ticker)
                if price <= 0:
                    continue
                trade.current_price   = round(price, 2)
                trade.unrealised_pnl  = round((price - trade.entry_price) * trade.quantity, 2)

                # Check barriers
                if trade.direction == "LONG":
                    if price <= trade.stop_loss:
                        self._close(trade, "SL_HIT")
                    elif price >= trade.take_profit:
                        self._close(trade, "TP_HIT")
                else:  # SHORT
                    if price >= trade.stop_loss:
                        self._close(trade, "SL_HIT")
                    elif price <= trade.take_profit:
                        self._close(trade, "TP_HIT")

    # ── Reports ────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Full portfolio status snapshot."""
        with self._lock:
            unrealised = sum(t.unrealised_pnl for t in self.ledger.open_trades)
            total_val  = self.ledger.current_capital + unrealised
            ret_pct    = (total_val - self.ledger.initial_capital) / self.ledger.initial_capital * 100
            win_rate   = (self.ledger.winning_trades / max(1, len(self.ledger.closed_trades))) * 100

            return {
                "initial_capital":  self.ledger.initial_capital,
                "current_capital":  round(self.ledger.current_capital, 2),
                "unrealised_pnl":   round(unrealised, 2),
                "total_value":      round(total_val, 2),
                "total_return_pct": round(ret_pct, 2),
                "total_pnl":        round(self.ledger.total_pnl, 2),
                "total_charges":    round(self.ledger.total_charges, 2),
                "total_trades":     self.ledger.total_trades,
                "open_trades":      len(self.ledger.open_trades),
                "closed_trades":    len(self.ledger.closed_trades),
                "win_rate_pct":     round(win_rate, 1),
                "open_positions":   [
                    {"ticker": t.ticker, "entry": t.entry_price,
                     "current": t.current_price, "unrealised": t.unrealised_pnl,
                     "signal": t.signal, "sl": t.stop_loss, "tp": t.take_profit}
                    for t in self.ledger.open_trades
                ],
            }

    def daily_report(self) -> str:
        """Formatted daily paper trading report for Telegram."""
        s = self.get_status()
        today_trades = [t for t in self.ledger.closed_trades
                        if t.exit_time and t.exit_time[:10] == date.today().isoformat()]
        today_pnl = sum(t.net_pnl for t in today_trades)

        return (
            f"📊 <b>Paper Trading Report — {date.today()}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💼 Capital:     ₹{s['current_capital']:,.0f}\n"
            f"📈 Total Value: ₹{s['total_value']:,.0f} ({s['total_return_pct']:+.2f}%)\n"
            f"Today P&L:    ₹{today_pnl:+,.0f} ({len(today_trades)} trades)\n"
            f"Unrealised:   ₹{s['unrealised_pnl']:+,.0f}\n"
            f"Win Rate:     {s['win_rate_pct']:.1f}%\n"
            f"Open Pos:     {s['open_trades']}\n"
            f"Total Trades: {s['total_trades']}"
        )

    def compare_to_benchmark(self, nifty_return_pct: float) -> str:
        """Compare paper portfolio return vs Nifty benchmark."""
        s   = self.get_status()
        ret = s["total_return_pct"]
        alpha = ret - nifty_return_pct
        return (
            f"📊 <b>Alpha Report</b>\n"
            f"  Portfolio:  {ret:+.2f}%\n"
            f"  Nifty 50:   {nifty_return_pct:+.2f}%\n"
            f"  Alpha:      {alpha:+.2f}%  {'✅' if alpha > 0 else '❌'}"
        )

    # ── Price helpers ──────────────────────────────────────────────────

    def _get_entry_price(self, ticker: str, signal: str) -> float:
        """Entry price = mid-price + half spread (market order simulation)."""
        mid   = self._get_current_price(ticker)
        if mid <= 0:
            return 0.0
        spread = SPREAD_PCT.get(ticker, SPREAD_PCT["default"])
        # Buy: pay ask (mid + spread/2); Sell: receive bid (mid - spread/2)
        return mid * (1 + spread / 2) if "BUY" in signal else mid * (1 - spread / 2)

    def _get_exit_price(self, ticker: str, direction: str) -> float:
        """Exit price = mid ± half spread."""
        mid    = self._get_current_price(ticker)
        if mid <= 0:
            return 0.0
        spread = SPREAD_PCT.get(ticker, SPREAD_PCT["default"])
        return mid * (1 - spread / 2) if direction == "LONG" else mid * (1 + spread / 2)

    def _get_current_price(self, ticker: str) -> float:
        """Get latest price from live feed or fallback."""
        if self.feed:
            q = self.feed.get_quote(ticker)
            if q and q.get("ltp", 0) > 0:
                return float(q["ltp"])
        # Fallback: try yfinance (slow but works offline)
        try:
            import yfinance as yf
            t  = yf.Ticker(f"{ticker}.NS")
            fi = t.fast_info
            return float(fi.last_price) if hasattr(fi, "last_price") else 0.0
        except Exception:
            return 0.0

    # ── Persistence ────────────────────────────────────────────────────

    def _load_ledger(self) -> PaperLedger:
        if LEDGER_FILE.exists():
            try:
                with open(LEDGER_FILE) as f:
                    d = json.load(f)
                # Reconstruct trades
                open_t   = [PaperTrade(**t) for t in d.pop("open_trades", [])]
                closed_t = [PaperTrade(**t) for t in d.pop("closed_trades", [])]
                ledger   = PaperLedger(**d)
                ledger.open_trades   = open_t
                ledger.closed_trades = closed_t
                return ledger
            except Exception as e:
                logger.warning(f"Ledger load failed: {e} — starting fresh")
        return PaperLedger(initial_capital=self.initial_capital,
                           current_capital=self.initial_capital)

    def _save_ledger(self):
        data = asdict(self.ledger)
        with open(LEDGER_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
