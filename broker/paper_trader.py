"""
broker/paper_trader.py  — Virtual Broker / Paper Trading Engine  v5
────────────────────────────────────────────────────────────────────
FIX: Indian Transaction Costs added to every trade.

OLD (broken): P&L = (exit_price - entry_price) × qty
  Missing: STT 0.1% on sell, SEBI 0.0001%, stamp duty 0.015% on buy,
           exchange fee 0.00325%, brokerage ₹0 (Zerodha delivery), GST 18%.
  A profitable-looking paper trade can be net-negative after costs!

FIX: _compute_charges() mirrors engine.py exactly:
  BUY side:  stamp_duty + exchange_fee + GST on (exchange + brokerage)
  SELL side: stt + sebi + exchange_fee + GST on (exchange + brokerage)
  Brokerage: ₹0 for equity delivery (Zerodha), ₹20 flat for F&O.

These charges are deducted from realised P&L in _close_position() and
_auto_exit().  The ledger records both gross_pnl and net_pnl separately
so you can see exactly how much the government took.
"""

import json
import uuid
import threading
from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field, asdict
from typing import Optional
from loguru import logger

IST       = ZoneInfo("Asia/Kolkata")
LEDGER    = Path("paper_trades.json")
PNL_FILE  = Path("paper_pnl.json")

# ── Indian transaction cost constants (equity delivery, same as engine.py) ─
_STT_DELIVERY   = 0.001      # 0.1%  on sell value
_SEBI_CHARGES   = 0.000001   # 0.0001%
_EXCHANGE_FEE   = 0.0000325  # NSE transaction charge
_STAMP_DUTY     = 0.00015    # 0.015% on buy value
_GST_RATE       = 0.18       # 18% GST on (brokerage + exchange fee)
_BROKERAGE_FLAT = 0          # ₹0 for Zerodha equity delivery


def _compute_charges(fill_price: float, quantity: int, side: str) -> dict:
    """
    Compute all-in Indian transaction charges for one leg of a trade.

    Args:
        fill_price: price per share
        quantity:   number of shares
        side:       "BUY" or "SELL"

    Returns dict with each charge component and total_charges.
    """
    value      = fill_price * quantity
    brokerage  = _BROKERAGE_FLAT          # ₹0 for delivery
    exchange   = value * _EXCHANGE_FEE
    gst        = (brokerage + exchange) * _GST_RATE

    if side == "BUY":
        stt        = 0.0
        sebi       = 0.0
        stamp      = value * _STAMP_DUTY
    else:  # SELL
        stt        = value * _STT_DELIVERY
        sebi       = value * _SEBI_CHARGES
        stamp      = 0.0

    total = brokerage + stt + sebi + exchange + stamp + gst
    return {
        "brokerage":  round(brokerage, 2),
        "stt":        round(stt, 2),
        "sebi":       round(sebi, 2),
        "exchange":   round(exchange, 2),
        "stamp":      round(stamp, 2),
        "gst":        round(gst, 2),
        "total":      round(total, 2),
    }


@dataclass
class PaperOrder:
    order_id:      str
    ticker:        str
    direction:     str      # BUY / SELL
    quantity:      int
    order_type:    str      # MARKET / LIMIT
    limit_price:   float    # 0 for MARKET
    status:        str      # PENDING / FILLED / REJECTED / CANCELLED
    fill_price:    float    = 0.0
    fill_time:     str      = ""
    signal_source: str      = ""   # "ML+TECH", "MANUAL", etc.
    stop_loss:     float    = 0.0
    take_profit:   float    = 0.0
    ml_confidence: float    = 0.0
    tech_score:    float    = 0.0
    regime:        str      = ""
    notes:         str      = ""


@dataclass
class PaperPosition:
    ticker:        str
    quantity:      int
    avg_price:     float    # average fill price
    current_price: float
    stop_loss:     float
    take_profit:   float
    entry_time:    str
    unrealised_pnl: float   = 0.0
    unrealised_pct: float   = 0.0
    highest_price: float    = 0.0   # for trailing stop tracking
    trailing_stop: float    = 0.0


class PaperTrader:
    """
    Virtual broker.  Call execute_signal() with an ML signal dict
    and it handles everything: risk check → order → ledger → P&L.
    """

    def __init__(self, risk_manager=None, monitor=None, live_feed=None):
        self.risk   = risk_manager
        self.monitor = monitor
        self.feed   = live_feed
        self._lock  = threading.Lock()

        self.positions : dict[str, PaperPosition] = {}
        self.orders    : list[PaperOrder]          = []
        self.closed_trades: list[dict]             = []

        self._load_ledger()
        logger.info("PaperTrader initialised")

    # ── Main entry point ───────────────────────────────────────────────

    def execute_signal(self, signal: dict) -> dict:
        """
        Process an ML+Tech signal dict.  Performs full pre-trade checks,
        fills the virtual order, updates ledger, fires Telegram alert.

        signal dict must have:
          ticker, signal ("BUY"/"SELL"/"STRONG BUY"/"STRONG SELL"),
          close, stop_loss, take_profit, ml_confidence, tech_score,
          expected_return, kelly_fraction, regime
        """
        ticker     = signal.get("ticker", "")
        sig        = signal.get("signal", "HOLD")
        close      = float(signal.get("close", 0))
        sl         = float(signal.get("stop_loss", 0))
        tp         = float(signal.get("take_profit", 0))
        confidence = float(signal.get("ml_confidence", 50))
        tech_score = float(signal.get("tech_score", 50))
        kelly      = float(signal.get("kelly_fraction", 0.01))
        regime     = signal.get("regime", "normal")

        if sig not in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
            return {"status": "ignored", "reason": "HOLD signal — no trade"}

        is_buy = "BUY" in sig

        # ── Risk check ─────────────────────────────────────────────────
        if self.risk:
            capital = self.risk.config.total_capital
            # Kelly-sized position
            trade_value = capital * min(kelly * 2, self.risk.config.max_position_size)
            approved, reason = self.risk.check_trade(
                signal=sig, symbol=ticker,
                trade_value=trade_value,
                ml_confidence=confidence,
                tech_score=tech_score,
            )
            if not approved:
                logger.info(f"  PaperTrade BLOCKED: {ticker} — {reason}")
                return {"status": "blocked", "reason": reason}
        else:
            capital     = 1_000_000
            trade_value = capital * 0.05

        # ── Already have a position? ───────────────────────────────────
        if is_buy and ticker in self.positions:
            return {"status": "ignored", "reason": f"Already long {ticker}"}
        if not is_buy and ticker not in self.positions:
            return {"status": "ignored", "reason": f"No position to sell in {ticker}"}

        # ── Get live price (use feed if available, else use signal close) ─
        live_price = self._get_live_price(ticker) or close

        # Simulate bid/ask spread (0.05% each side = 0.1% round trip)
        spread = live_price * 0.0005
        fill_price = (live_price + spread) if is_buy else (live_price - spread)

        quantity = max(1, int(trade_value / fill_price))

        order = PaperOrder(
            order_id      = str(uuid.uuid4())[:8],
            ticker        = ticker,
            direction     = "BUY" if is_buy else "SELL",
            quantity      = quantity,
            order_type    = "MARKET",
            limit_price   = 0.0,
            status        = "FILLED",
            fill_price    = round(fill_price, 2),
            fill_time     = datetime.now(IST).isoformat(),
            signal_source = sig,
            stop_loss     = sl,
            take_profit   = tp,
            ml_confidence = confidence,
            tech_score    = tech_score,
            regime        = regime,
        )

        with self._lock:
            if is_buy:
                self._open_position(order, sl, tp)
                if self.risk:
                    self.risk.record_trade_open(quantity * fill_price)
            else:
                pnl = self._close_position(ticker, order)
                if self.risk:
                    self.risk.record_trade_close(pnl, quantity * fill_price)

            self.orders.append(order)
            self._save_ledger()

        # Telegram alert
        if self.monitor:
            direction_emoji = "🟢 PAPER BUY" if is_buy else "🔴 PAPER SELL"
            self.monitor.send_message(
                f"{direction_emoji} <b>{ticker}</b>\n"
                f"Fill: ₹{fill_price:,.2f} × {quantity} lots\n"
                f"ML: {confidence:.0f}%  Tech: {tech_score:.0f}/100\n"
                f"SL: ₹{sl:,.0f}  TP: ₹{tp:,.0f}\n"
                f"Regime: {regime}"
            )

        return {
            "status":     "filled",
            "order_id":   order.order_id,
            "ticker":     ticker,
            "direction":  order.direction,
            "quantity":   quantity,
            "fill_price": fill_price,
            "trade_value": round(quantity * fill_price, 2),
        }

    def update_pnl(self):
        """
        Refresh unrealised P&L for all open positions.
        Call this from the live feed callback (every tick or every 5 seconds).
        """
        with self._lock:
            for ticker, pos in self.positions.items():
                price = self._get_live_price(ticker)
                if price is None:
                    continue
                pos.current_price  = price
                pos.unrealised_pnl = round((price - pos.avg_price) * pos.quantity, 2)
                pos.unrealised_pct = round((price / pos.avg_price - 1) * 100, 3)

                # Update trailing stop
                if price > pos.highest_price:
                    pos.highest_price = price
                    atr_est = price * 0.015
                    pos.trailing_stop = max(pos.trailing_stop, price - atr_est)

                # Auto-exit checks
                if price <= pos.stop_loss:
                    self._auto_exit(ticker, price, "SL_HIT")
                elif price >= pos.take_profit:
                    self._auto_exit(ticker, price, "TP_HIT")
                elif pos.trailing_stop > pos.avg_price and price <= pos.trailing_stop:
                    self._auto_exit(ticker, price, "TRAIL_STOP")

    def get_portfolio_summary(self) -> dict:
        """Live portfolio snapshot with net P&L after all transaction costs."""
        total_value    = sum(p.quantity * p.current_price for p in self.positions.values())
        unrealised_pnl = sum(p.unrealised_pnl for p in self.positions.values())
        realised_pnl   = sum(t.get("pnl", 0) for t in self.closed_trades)       # net
        gross_pnl      = sum(t.get("gross_pnl", t.get("pnl", 0)) for t in self.closed_trades)
        total_charges  = sum(t.get("charges", 0) for t in self.closed_trades)
        win_trades     = [t for t in self.closed_trades if t.get("pnl", 0) > 0]
        win_rate       = len(win_trades) / max(1, len(self.closed_trades))

        return {
            "open_positions":      len(self.positions),
            "total_market_value":  round(total_value, 2),
            "unrealised_pnl":      round(unrealised_pnl, 2),
            "realised_pnl":        round(realised_pnl, 2),     # NET
            "gross_pnl":           round(gross_pnl, 2),
            "total_charges_paid":  round(total_charges, 2),
            "total_pnl":           round(unrealised_pnl + realised_pnl, 2),
            "total_trades":        len(self.closed_trades),
            "win_rate":            round(win_rate * 100, 1),
            "positions": [
                {
                    "ticker":        p.ticker,
                    "qty":           p.quantity,
                    "avg_price":     p.avg_price,
                    "current":       p.current_price,
                    "pnl":           p.unrealised_pnl,
                    "pnl_pct":       p.unrealised_pct,
                    "stop_loss":     p.stop_loss,
                    "take_profit":   p.take_profit,
                    "trailing_stop": p.trailing_stop,
                }
                for p in self.positions.values()
            ],
        }

    def daily_report(self) -> str:
        s = self.get_portfolio_summary()
        pnl_sym = "🟢" if s["total_pnl"] >= 0 else "🔴"
        return (
            f"📋 <b>Paper Trading Daily Report</b>\n"
            f"{pnl_sym} Total P&L: ₹{s['total_pnl']:+,.0f}\n"
            f"   Realised:   ₹{s['realised_pnl']:+,.0f}\n"
            f"   Unrealised: ₹{s['unrealised_pnl']:+,.0f}\n"
            f"📈 Trades today: {s['total_trades']}\n"
            f"✅ Win rate: {s['win_rate']:.1f}%\n"
            f"💼 Open positions: {s['open_positions']}"
        )

    # ── Private ─────────────────────────────────────────────────────────

    def _get_live_price(self, ticker: str) -> Optional[float]:
        if self.feed and ticker in self.feed.latest_quotes:
            return float(self.feed.latest_quotes[ticker].get("ltp", 0)) or None
        return None

    def _open_position(self, order: PaperOrder, sl: float, tp: float):
        self.positions[order.ticker] = PaperPosition(
            ticker        = order.ticker,
            quantity      = order.quantity,
            avg_price     = order.fill_price,
            current_price = order.fill_price,
            stop_loss     = sl,
            take_profit   = tp,
            entry_time    = order.fill_time,
            highest_price = order.fill_price,
            trailing_stop = sl,
        )
        logger.info(f"  PaperBUY  {order.ticker} × {order.quantity} @ ₹{order.fill_price:,.2f}")

    def _close_position(self, ticker: str, order: PaperOrder) -> float:
        pos = self.positions.pop(ticker, None)
        if pos is None:
            return 0.0

        gross_pnl = (order.fill_price - pos.avg_price) * pos.quantity

        # ── Indian transaction costs (government charges — always deducted) ─
        buy_charges  = _compute_charges(pos.avg_price,    pos.quantity, "BUY")
        sell_charges = _compute_charges(order.fill_price, pos.quantity, "SELL")
        total_charges = buy_charges["total"] + sell_charges["total"]
        net_pnl = gross_pnl - total_charges

        self.closed_trades.append({
            "ticker":        ticker,
            "entry_price":   pos.avg_price,
            "exit_price":    order.fill_price,
            "quantity":      pos.quantity,
            "gross_pnl":     round(gross_pnl, 2),
            "charges":       round(total_charges, 2),
            "pnl":           round(net_pnl, 2),    # NET (what you actually keep)
            "pnl_pct":       round((order.fill_price / pos.avg_price - 1) * 100, 3),
            "entry_time":    pos.entry_time,
            "exit_time":     order.fill_time,
            "exit_reason":   order.notes or "SIGNAL",
            "charge_detail": {
                "buy":  buy_charges,
                "sell": sell_charges,
            },
        })
        logger.info(
            f"  PaperSELL {ticker} × {pos.quantity} @ ₹{order.fill_price:,.2f} "
            f"→ Gross ₹{gross_pnl:+,.0f}  Charges ₹{total_charges:,.0f}  "
            f"Net ₹{net_pnl:+,.0f}"
        )
        return net_pnl

    def _auto_exit(self, ticker: str, price: float, reason: str):
        """Automatically close position when SL/TP/trailing stop hit."""
        pos = self.positions.get(ticker)
        if pos is None:
            return

        gross_pnl = (price - pos.avg_price) * pos.quantity
        buy_charges  = _compute_charges(pos.avg_price, pos.quantity, "BUY")
        sell_charges = _compute_charges(price,         pos.quantity, "SELL")
        total_charges = buy_charges["total"] + sell_charges["total"]
        net_pnl = gross_pnl - total_charges

        self.closed_trades.append({
            "ticker":        ticker,
            "entry_price":   pos.avg_price,
            "exit_price":    price,
            "quantity":      pos.quantity,
            "gross_pnl":     round(gross_pnl, 2),
            "charges":       round(total_charges, 2),
            "pnl":           round(net_pnl, 2),
            "pnl_pct":       round((price / pos.avg_price - 1) * 100, 3),
            "entry_time":    pos.entry_time,
            "exit_time":     datetime.now(IST).isoformat(),
            "exit_reason":   reason,
        })
        del self.positions[ticker]
        self._save_ledger()
        logger.info(
            f"  AUTO-EXIT {ticker} @ ₹{price:,.2f} ({reason}) "
            f"Gross ₹{gross_pnl:+,.0f}  Charges ₹{total_charges:,.0f}  "
            f"Net ₹{net_pnl:+,.0f}"
        )
        if self.monitor:
            emoji = "🛑" if "SL" in reason else ("🎯" if "TP" in reason else "🔄")
            self.monitor.send_message(
                f"{emoji} <b>Paper {reason}</b> — {ticker}\n"
                f"Exit: ₹{price:,.2f}  Gross: ₹{gross_pnl:+,.0f}\n"
                f"Charges: ₹{total_charges:,.0f}  Net: ₹{net_pnl:+,.0f}"
            )
        if self.risk:
            self.risk.record_trade_close(net_pnl, price * pos.quantity)

    def _save_ledger(self):
        try:
            data = {
                "updated":       datetime.now(IST).isoformat(),
                "positions":     {k: asdict(v) for k, v in self.positions.items()},
                "orders":        [asdict(o) for o in self.orders[-200:]],
                "closed_trades": self.closed_trades[-500:],
            }
            with open(LEDGER, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"Ledger save error: {e}")

    def _load_ledger(self):
        if not LEDGER.exists():
            return
        try:
            with open(LEDGER) as f:
                data = json.load(f)
            for ticker, pd_data in data.get("positions", {}).items():
                self.positions[ticker] = PaperPosition(**pd_data)
            self.closed_trades = data.get("closed_trades", [])
            logger.info(f"Ledger loaded: {len(self.positions)} open, "
                        f"{len(self.closed_trades)} closed trades")
        except Exception as e:
            logger.warning(f"Ledger load error: {e}")
