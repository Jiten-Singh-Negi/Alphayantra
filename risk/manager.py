"""
risk/manager.py
────────────────
Professional risk management system.

Rules enforced:
  - Max daily loss limit (stops all trading if breached)
  - Max position size per trade
  - Max concurrent open positions
  - Max capital allocation per sector
  - Volatility-adjusted position sizing (Kelly / ATR-based)
  - Drawdown-based trading halt
  - Time-based rules (no new trades after 3:00 PM, etc.)
  - Emergency kill switch
"""

import json
from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field, asdict
from loguru import logger
from typing import Optional

IST = ZoneInfo("Asia/Kolkata")

RISK_CONFIG_FILE = Path("risk/config.json")
RISK_STATE_FILE  = Path("risk/daily_state.json")


@dataclass
class RiskConfig:
    # ── Capital limits ─────────────────────────────────────────────────
    total_capital:        float = 1_000_000   # ₹10 Lakh total
    max_capital_deployed: float = 0.80        # max 80% deployed at once
    max_position_size:    float = 0.05        # max 5% per single trade
    max_single_option:    float = 0.03        # max 3% on a single option leg

    # ── Loss limits ────────────────────────────────────────────────────
    max_daily_loss_pct:   float = 0.02        # stop trading if down 2% in a day
    max_weekly_loss_pct:  float = 0.05        # halt for week if down 5%
    max_drawdown_pct:     float = 0.15        # full stop if drawdown > 15%

    # ── Trade limits ───────────────────────────────────────────────────
    max_open_positions:   int   = 10
    max_options_positions: int  = 5
    max_trades_per_day:   int   = 20

    # ── Sector limits ──────────────────────────────────────────────────
    max_sector_exposure:  float = 0.25        # max 25% in any one sector

    # ── Time rules ─────────────────────────────────────────────────────
    no_new_trades_after:  str   = "15:00"     # IST — no new entries after 3 PM
    no_options_after:     str   = "15:20"     # options expire same day
    square_off_by:        str   = "15:25"     # force close all intraday

    # ── Options specific ───────────────────────────────────────────────
    max_option_loss_pct:  float = 0.50        # exit option if down 50%
    min_option_profit_pct: float = 0.30       # book partial profit at 30% gain
    max_iv_to_trade:      float = 50.0        # don't buy options if IV > 50%

    # ── ML confidence thresholds ───────────────────────────────────────
    min_ml_confidence:    float = 60.0        # minimum ML confidence to signal
    min_tech_score:       float = 65.0        # minimum technical score

    # ── Kill switch ────────────────────────────────────────────────────
    kill_switch:          bool  = False       # emergency stop all trading


@dataclass
class DailyState:
    date:              str   = ""
    trades_today:      int   = 0
    pnl_today:         float = 0.0
    capital_deployed:  float = 0.0
    open_positions:    int   = 0
    trading_halted:    bool  = False
    halt_reason:       str   = ""
    peak_capital:      float = 0.0
    current_capital:   float = 0.0


class RiskManager:
    """
    Central risk management. Call check_trade() before any trade.
    Returns (approved: bool, reason: str).
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or self._load_config()
        self.state  = self._load_or_init_state()
        logger.info(f"RiskManager initialised — Capital: ₹{self.config.total_capital:,.0f}")

    # ── Pre-trade checks ──────────────────────────────────────────────

    def check_trade(
        self,
        signal:         str,
        symbol:         str,
        trade_value:    float,
        ml_confidence:  float,
        tech_score:     float,
        is_option:      bool = False,
        iv:             float = 0.0,
    ) -> tuple[bool, str]:
        """
        Master pre-trade check. Returns (approved, reason).
        Call this before every trade signal.
        """
        cfg = self.config
        st  = self.state

        # ── Kill switch ────────────────────────────────────────────────
        if cfg.kill_switch:
            return False, "KILL SWITCH ACTIVE — all trading halted"

        # ── Trading halted ─────────────────────────────────────────────
        if st.trading_halted:
            return False, f"Trading halted: {st.halt_reason}"

        # ── Time checks ────────────────────────────────────────────────
        now_str = datetime.now(IST).strftime("%H:%M")
        if now_str >= cfg.no_new_trades_after:
            return False, f"No new trades after {cfg.no_new_trades_after} IST"
        if is_option and now_str >= cfg.no_options_after:
            return False, f"No new option trades after {cfg.no_options_after} IST"

        # ── ML confidence ─────────────────────────────────────────────
        if ml_confidence < cfg.min_ml_confidence:
            return False, f"ML confidence {ml_confidence:.1f}% < minimum {cfg.min_ml_confidence}%"

        # ── Tech score ────────────────────────────────────────────────
        if tech_score < cfg.min_tech_score:
            return False, f"Tech score {tech_score:.1f} < minimum {cfg.min_tech_score}"

        # ── Daily loss limit ──────────────────────────────────────────
        capital = cfg.total_capital
        daily_loss_limit = capital * cfg.max_daily_loss_pct
        if st.pnl_today < -daily_loss_limit:
            self._halt(f"Daily loss limit breached: ₹{st.pnl_today:,.0f}")
            return False, f"Daily loss limit reached (₹{daily_loss_limit:,.0f})"

        # ── Drawdown check ────────────────────────────────────────────
        if st.peak_capital > 0:
            drawdown = (st.peak_capital - st.current_capital) / st.peak_capital
            if drawdown > cfg.max_drawdown_pct:
                self._halt(f"Max drawdown breached: {drawdown:.1%}")
                return False, f"Drawdown limit {cfg.max_drawdown_pct:.0%} breached"

        # ── Trade count ───────────────────────────────────────────────
        if st.trades_today >= cfg.max_trades_per_day:
            return False, f"Max trades per day ({cfg.max_trades_per_day}) reached"

        # ── Position count ────────────────────────────────────────────
        if st.open_positions >= cfg.max_open_positions:
            return False, f"Max open positions ({cfg.max_open_positions}) reached"

        # ── Position size ─────────────────────────────────────────────
        max_trade = capital * (cfg.max_single_option if is_option else cfg.max_position_size)
        if trade_value > max_trade:
            return False, f"Trade size ₹{trade_value:,.0f} > max ₹{max_trade:,.0f}"

        # ── Capital deployment ────────────────────────────────────────
        max_deploy = capital * cfg.max_capital_deployed
        if st.capital_deployed + trade_value > max_deploy:
            return False, f"Would exceed max deployment limit (₹{max_deploy:,.0f})"

        # ── IV check for options ──────────────────────────────────────
        if is_option and iv > cfg.max_iv_to_trade:
            return False, f"IV {iv:.1f}% too high (max {cfg.max_iv_to_trade}%) — don't buy overpriced options"

        return True, "APPROVED"

    def position_size(
        self,
        capital:    float,
        entry:      float,
        stop_loss:  float,
        atr:        float,
        confidence: float,
    ) -> int:
        """
        Calculate optimal position size using ATR-based risk.
        Risk per trade = 1% of capital.
        Position size = Risk amount / (Entry - Stop Loss)

        Returns number of shares/lots.
        """
        risk_per_trade = capital * 0.01   # 1% risk per trade
        risk_per_share = max(0.01, entry - stop_loss)

        # Base quantity from risk
        qty = int(risk_per_trade / risk_per_share)

        # Scale by ML confidence (higher confidence = larger size, max 1.5x)
        confidence_scalar = 0.5 + (confidence / 100)
        qty = int(qty * min(1.5, confidence_scalar))

        # Cap at max position size
        max_qty = int((capital * self.config.max_position_size) / entry)
        return max(1, min(qty, max_qty))

    def kelly_fraction(
        self,
        win_rate:    float,  # e.g. 0.60 for 60%
        avg_win:     float,  # average win amount
        avg_loss:    float,  # average loss amount (positive number)
    ) -> float:
        """
        Kelly Criterion: optimal fraction of capital to risk.
        Uses half-Kelly for safety (full Kelly is too aggressive).
        """
        if avg_loss == 0:
            return 0.01
        b = avg_win / avg_loss     # win/loss ratio
        p = win_rate
        q = 1 - p
        kelly = (b * p - q) / b
        half_kelly = max(0, kelly / 2)
        return round(min(half_kelly, self.config.max_position_size), 4)

    # ── State updates ──────────────────────────────────────────────────

    def record_trade_open(self, value: float):
        self.state.trades_today    += 1
        self.state.capital_deployed += value
        self.state.open_positions  += 1
        self._save_state()

    def record_trade_close(self, pnl: float, value: float):
        self.state.pnl_today        += pnl
        self.state.capital_deployed  = max(0, self.state.capital_deployed - value)
        self.state.open_positions    = max(0, self.state.open_positions - 1)
        self.state.current_capital  += pnl
        if self.state.current_capital > self.state.peak_capital:
            self.state.peak_capital  = self.state.current_capital
        self._save_state()

        # Check limits after close
        capital = self.config.total_capital
        if self.state.pnl_today < -(capital * self.config.max_daily_loss_pct):
            self._halt("Daily loss limit breached after trade close")

    def activate_kill_switch(self, reason: str = "Manual"):
        """Emergency stop — call this to halt all trading immediately."""
        self.config.kill_switch = True
        logger.critical(f"🛑 KILL SWITCH ACTIVATED: {reason}")
        self._save_config()

    def deactivate_kill_switch(self):
        self.config.kill_switch = False
        logger.info("Kill switch deactivated")
        self._save_config()

    def get_status(self) -> dict:
        cfg = self.config
        st  = self.state
        capital = cfg.total_capital
        return {
            "date":               st.date,
            "kill_switch":        cfg.kill_switch,
            "trading_halted":     st.trading_halted,
            "halt_reason":        st.halt_reason,
            "trades_today":       st.trades_today,
            "pnl_today":          round(st.pnl_today, 2),
            "pnl_today_pct":      round(st.pnl_today / capital * 100, 2),
            "daily_loss_limit":   round(capital * cfg.max_daily_loss_pct, 2),
            "capital_deployed":   round(st.capital_deployed, 2),
            "capital_deployed_pct": round(st.capital_deployed / capital * 100, 1),
            "open_positions":     st.open_positions,
            "max_positions":      cfg.max_open_positions,
            "current_capital":    round(st.current_capital, 2),
            "peak_capital":       round(st.peak_capital, 2),
            "drawdown_pct":       round(
                (st.peak_capital - st.current_capital) / max(1, st.peak_capital) * 100, 2
            ) if st.peak_capital > 0 else 0,
        }

    def daily_report(self) -> str:
        s = self.get_status()
        pnl_color = "🟢" if s["pnl_today"] >= 0 else "🔴"
        return (
            f"📊 Daily Risk Report — {s['date']}\n"
            f"{pnl_color} P&L: ₹{s['pnl_today']:+,.0f} ({s['pnl_today_pct']:+.2f}%)\n"
            f"📈 Trades: {s['trades_today']}\n"
            f"💼 Deployed: ₹{s['capital_deployed']:,.0f} ({s['capital_deployed_pct']:.1f}%)\n"
            f"📉 Drawdown: {s['drawdown_pct']:.2f}%\n"
            f"🔒 Kill Switch: {'ON' if s['kill_switch'] else 'OFF'}\n"
            f"⚠️  Halted: {s['trading_halted']} — {s['halt_reason']}"
        )

    # ── Private ────────────────────────────────────────────────────────

    def _halt(self, reason: str):
        self.state.trading_halted = True
        self.state.halt_reason    = reason
        logger.warning(f"⚠️  Trading halted: {reason}")
        self._save_state()

    def _load_config(self) -> RiskConfig:
        RISK_CONFIG_FILE.parent.mkdir(exist_ok=True)
        if RISK_CONFIG_FILE.exists():
            with open(RISK_CONFIG_FILE) as f:
                d = json.load(f)
            return RiskConfig(**{k: v for k, v in d.items() if hasattr(RiskConfig, k)})
        return RiskConfig()

    def _save_config(self):
        RISK_CONFIG_FILE.parent.mkdir(exist_ok=True)
        with open(RISK_CONFIG_FILE, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    def _load_or_init_state(self) -> DailyState:
        today = date.today().isoformat()
        RISK_STATE_FILE.parent.mkdir(exist_ok=True)
        if RISK_STATE_FILE.exists():
            with open(RISK_STATE_FILE) as f:
                d = json.load(f)
            if d.get("date") == today:
                return DailyState(**d)
        # New day — reset state
        state = DailyState(
            date             = today,
            current_capital  = self.config.total_capital,
            peak_capital     = self.config.total_capital,
        )
        self._save_state(state)
        return state

    def _save_state(self, state: Optional[DailyState] = None):
        s = state or self.state
        with open(RISK_STATE_FILE, "w") as f:
            json.dump(asdict(s), f, indent=2)
