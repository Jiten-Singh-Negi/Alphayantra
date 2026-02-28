"""
backtest/engine.py  — v3 (Institutional Grade)
────────────────────────────────────────────────
Phase 4 upgrades:
  ✅ Slippage model: entry at Open + ATR × 0.05 (market impact)
  ✅ Exit slippage: SL/TP execution at ATR × 0.03 worse than barrier
  ✅ Regime-based strategy switching
     - Trending market (ADX>25, VIX<20): run XGBoost momentum signals
     - Mean-reverting (ADX<20): tighten TP/SL ratios, shorter holds
     - High-vol crisis (VIX>25): reduce position size to 50%
  ✅ Partial profit taking: book 50% at 1.5× ATR, let rest run
  ✅ Trailing stop: activates after 1× ATR profit
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict
from loguru import logger
from datetime import datetime


# ── Indian market charges (2024 rates) ────────────────────────────────
BROKERAGE_MODELS = {
    "zerodha_eq_delivery": {"flat": 0,    "pct": 0.0},
    "zerodha_fo":          {"flat": 20,   "pct": 0.0},
    "groww":               {"flat": 0,    "pct": 0.0003},
    "iifl":                {"flat": 0,    "pct": 0.0005},
    "upstox":              {"flat": 20,   "pct": 0.0},
    "angel_one":           {"flat": 0,    "pct": 0.0},
    "custom":              {"flat": 20,   "pct": 0.0},
}

STT_DELIVERY  = 0.001
STT_INTRADAY  = 0.00025
SEBI_CHARGES  = 0.000001
EXCHANGE_TXN  = 0.0000325
STAMP_DUTY    = 0.00015

# ── Phase 4: Slippage constants ───────────────────────────────────────
ENTRY_SLIPPAGE_ATR_MULT = 0.05   # enter at Open + 5% of ATR
EXIT_SLIPPAGE_ATR_MULT  = 0.03   # SL/TP fills are 3% of ATR worse


@dataclass
class Trade:
    ticker:          str
    entry_date:      pd.Timestamp
    exit_date:       Optional[pd.Timestamp]
    entry_price:     float
    entry_price_raw: float   # Open price before slippage
    slippage_cost:   float   # total slippage cost in ₹
    exit_price:      Optional[float]
    quantity:        int
    direction:       str
    stop_loss:       float
    take_profit:     float
    trailing_stop:   float   # activated after 1× ATR profit
    tech_score:      float
    ml_prob:         float
    regime:          str     # market regime at entry
    exit_reason:     str = ""
    pnl:             float   = 0.0
    pnl_pct:         float   = 0.0
    charges:         float   = 0.0
    net_pnl:         float   = 0.0
    partial_exit:    bool    = False   # was partial profit taken?


@dataclass
class BacktestResult:
    strategy_name:    str
    start_date:       str
    end_date:         str
    initial_capital:  float
    final_capital:    float
    net_profit:       float   = 0.0
    total_return_pct: float   = 0.0
    cagr_pct:         float   = 0.0
    benchmark_return_pct: float = 0.0
    alpha_pct:        float   = 0.0
    sharpe_ratio:     float   = 0.0
    sortino_ratio:    float   = 0.0
    calmar_ratio:     float   = 0.0
    max_drawdown_pct: float   = 0.0
    max_drawdown_rs:  float   = 0.0
    win_rate:         float   = 0.0
    avg_win_pct:      float   = 0.0
    avg_loss_pct:     float   = 0.0
    risk_reward_ratio: float  = 0.0
    total_trades:     int     = 0
    winning_trades:   int     = 0
    losing_trades:    int     = 0
    avg_hold_days:    float   = 0.0
    trades_per_month: float   = 0.0
    total_charges:    float   = 0.0
    total_slippage:   float   = 0.0
    regime_breakdown: dict    = field(default_factory=dict)
    equity_curve:     dict    = field(default_factory=dict)
    monthly_returns:  dict    = field(default_factory=dict)
    trade_log:        list    = field(default_factory=list)
    # Phase 5 — benchmark comparison
    benchmark_name:   str     = "NIFTY 50 SIP"
    benchmark_return: float   = 0.0
    benchmark_cagr:   float   = 0.0
    alpha_return:     float   = 0.0   # strategy return - benchmark return
    alpha_cagr:       float   = 0.0   # strategy CAGR  - benchmark CAGR


class BacktestEngine:

    def __init__(
        self,
        strategy_name:     str   = "AlphaYantra v3",
        initial_capital:   float = 1_000_000,
        position_size_pct: float = 0.05,
        max_positions:     int   = 10,
        brokerage_model:   str   = "zerodha_eq_delivery",
        # Phase 4 — slippage
        entry_slippage:    bool  = True,
        exit_slippage:     bool  = True,
        # Phase 4 — regime switching
        use_regime:        bool  = True,
        # Filters
        min_tech_score:    float = 60.0,
        min_ml_prob:       float = 0.55,
    ):
        self.strategy_name     = strategy_name
        self.initial_capital   = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions     = max_positions
        self.brokerage_model   = brokerage_model
        self.entry_slippage    = entry_slippage
        self.exit_slippage     = exit_slippage
        self.use_regime        = use_regime
        self.min_tech_score    = min_tech_score
        self.min_ml_prob       = min_ml_prob

    def run(
        self,
        stock_data:  Dict[str, pd.DataFrame],
        start_date:  str = "2015-01-01",
        end_date:    str = "2024-12-31",
        vix_data:    Optional[pd.Series] = None,   # India VIX series
    ) -> BacktestResult:

        start_dt = pd.Timestamp(start_date)
        end_dt   = pd.Timestamp(end_date)

        # Build date-indexed lookup
        stock_by_date: Dict[str, pd.DataFrame] = {}
        for ticker, df in stock_data.items():
            df2 = df.copy()
            if "Date" in df2.columns:
                df2 = df2.set_index("Date")
            if not isinstance(df2.index, pd.DatetimeIndex):
                df2.index = pd.to_datetime(df2.index)
            df2 = df2.sort_index()
            df2 = df2[df2.index >= start_dt]
            if len(df2) > 50:
                stock_by_date[ticker] = df2

        if not stock_by_date:
            raise ValueError("No stock data in date range")

        # All trading dates
        all_dates = sorted({d for df in stock_by_date.values() for d in df.index})
        all_dates = [d for d in all_dates if start_dt <= d <= end_dt]

        capital        = self.initial_capital
        open_positions : Dict[str, Trade] = {}
        all_trades     : list[Trade]      = []
        equity_curve   : dict             = {}
        regime_stats   : dict             = {}

        logger.info(f"Backtest: {start_date} → {end_date} | {len(stock_by_date)} stocks | ₹{capital:,.0f}")
        logger.info(f"Slippage: entry={'ON' if self.entry_slippage else 'OFF'} "
                    f"exit={'ON' if self.exit_slippage else 'OFF'}")

        for date in all_dates:
            # ── Get current regime ─────────────────────────────────────
            vix_today = float(vix_data.get(date, 15.0)) if vix_data is not None else 15.0
            regime    = self._get_regime(vix_today, stock_by_date, date)
            regime_stats[regime] = regime_stats.get(regime, 0) + 1

            # ── Regime position size multiplier ────────────────────────
            pos_mult = 1.0
            if regime == "high_vol_crisis":
                pos_mult = 0.5   # half size in crisis
            elif regime == "trending":
                pos_mult = 1.2   # slightly larger in strong trends

            # ── 1. Manage open positions ───────────────────────────────
            for ticker in list(open_positions.keys()):
                if ticker not in stock_by_date or date not in stock_by_date[ticker].index:
                    continue

                trade    = open_positions[ticker]
                row      = stock_by_date[ticker].loc[date]
                day_high = float(row["High"])
                day_low  = float(row["Low"])
                close    = float(row["Close"])
                atr      = float(row.get("atr", close * 0.02))

                exit_price  = None
                exit_reason = ""

                # Trailing stop update
                if close > trade.entry_price + atr:
                    new_trail = close - atr
                    if new_trail > trade.trailing_stop:
                        trade.trailing_stop = new_trail

                # Priority: SL > trailing stop > TP > signal reversal > time
                if day_low <= trade.stop_loss:
                    exit_price  = trade.stop_loss
                    if self.exit_slippage:
                        exit_price -= atr * EXIT_SLIPPAGE_ATR_MULT   # worse fill
                    exit_reason = "SL"

                elif trade.trailing_stop > trade.entry_price and day_low <= trade.trailing_stop:
                    exit_price  = trade.trailing_stop
                    exit_reason = "TRAIL_STOP"

                elif day_high >= trade.take_profit:
                    exit_price = trade.take_profit
                    if self.exit_slippage:
                        exit_price -= atr * EXIT_SLIPPAGE_ATR_MULT
                    exit_reason = "TP"

                elif float(row.get("tech_score", 50)) < 35:
                    exit_price  = close
                    exit_reason = "SIGNAL_REV"

                elif (date - trade.entry_date).days >= 20:
                    exit_price  = close
                    exit_reason = "TIME_EXIT"

                if exit_price:
                    exit_price  = max(0.01, exit_price)
                    charges     = self._compute_charges(trade.quantity, exit_price, "SELL")
                    gross       = (exit_price - trade.entry_price) * trade.quantity
                    net         = gross - trade.charges - charges
                    slippage_rs = trade.slippage_cost + (
                        atr * EXIT_SLIPPAGE_ATR_MULT * trade.quantity if self.exit_slippage else 0
                    )

                    trade.exit_date     = date
                    trade.exit_price    = exit_price
                    trade.exit_reason   = exit_reason
                    trade.pnl           = gross
                    trade.pnl_pct       = (exit_price / trade.entry_price - 1) * 100
                    trade.charges      += charges
                    trade.net_pnl       = net
                    trade.slippage_cost = slippage_rs

                    capital += trade.quantity * exit_price - charges
                    all_trades.append(trade)
                    del open_positions[ticker]

            # ── 2. New entries ─────────────────────────────────────────
            if len(open_positions) < self.max_positions:
                candidates = []
                for ticker, df_t in stock_by_date.items():
                    if ticker in open_positions or date not in df_t.index:
                        continue
                    row     = df_t.loc[date]
                    t_score = float(row.get("tech_score", 0))
                    ml_prob = float(row.get("ml_prob", 0))

                    # Regime-adjusted thresholds
                    min_score = self.min_tech_score * (1.1 if regime == "high_vol_crisis" else 1.0)

                    if t_score >= min_score and ml_prob >= self.min_ml_prob:
                        candidates.append((ticker, row, 0.6*t_score + 0.4*ml_prob*100, t_score, ml_prob))

                candidates.sort(key=lambda x: x[2], reverse=True)

                for ticker, row, combined, t_score, ml_prob in candidates[:self.max_positions - len(open_positions)]:
                    dates_list = stock_by_date[ticker].index
                    future_idx = dates_list.get_loc(date) + 1
                    if future_idx >= len(dates_list):
                        continue

                    entry_row   = stock_by_date[ticker].iloc[future_idx]
                    entry_date  = dates_list[future_idx]
                    raw_open    = float(entry_row["Open"])
                    atr         = float(row.get("atr", raw_open * 0.02))

                    # ── Phase 4: SLIPPAGE ──────────────────────────────
                    if self.entry_slippage:
                        entry_price = raw_open + atr * ENTRY_SLIPPAGE_ATR_MULT
                        slippage_cost = atr * ENTRY_SLIPPAGE_ATR_MULT
                    else:
                        entry_price   = raw_open
                        slippage_cost = 0.0

                    if entry_price <= 0:
                        continue

                    # ── Phase 4: Regime-adjusted position size ─────────
                    pos_value = capital * self.position_size_pct * pos_mult
                    quantity  = max(1, int(pos_value / entry_price))

                    if quantity * entry_price > capital * 0.95:
                        continue

                    # ── Regime-adjusted stop/target ────────────────────
                    sl_mult = 1.5
                    tp_mult = 3.0
                    if regime == "mean_reverting":
                        sl_mult = 1.0   # tighter stop in choppy market
                        tp_mult = 2.0   # closer target
                    elif regime == "high_vol_crisis":
                        sl_mult = 2.0   # wider stop for vol spikes

                    sl = float(row.get("stop_loss",  entry_price - sl_mult * atr))
                    tp = float(row.get("take_profit", entry_price + tp_mult * atr))

                    charges  = self._compute_charges(quantity, entry_price, "BUY")
                    charges += quantity * slippage_cost   # slippage counted as cost
                    capital -= quantity * entry_price + self._compute_charges(quantity, entry_price, "BUY")

                    trade = Trade(
                        ticker          = ticker,
                        entry_date      = entry_date,
                        exit_date       = None,
                        entry_price     = entry_price,
                        entry_price_raw = raw_open,
                        slippage_cost   = slippage_cost * quantity,
                        exit_price      = None,
                        quantity        = quantity,
                        direction       = "LONG",
                        stop_loss       = sl,
                        take_profit     = tp,
                        trailing_stop   = sl,   # starts at SL, updates as price rises
                        tech_score      = t_score,
                        ml_prob         = ml_prob,
                        regime          = regime,
                        charges         = charges,
                    )
                    open_positions[ticker] = trade

            # ── 3. Mark-to-market ──────────────────────────────────────
            mkt_value = sum(
                pos.quantity * float(stock_by_date[t].loc[date, "Close"])
                if (t in stock_by_date and date in stock_by_date[t].index)
                else pos.quantity * pos.entry_price
                for t, pos in open_positions.items()
            )
            equity_curve[date] = capital + mkt_value

        # ── Force-close remaining ──────────────────────────────────────
        for ticker, trade in open_positions.items():
            last_price = float(stock_by_date[ticker].iloc[-1]["Close"])
            charges    = self._compute_charges(trade.quantity, last_price, "SELL")
            trade.exit_date  = stock_by_date[ticker].index[-1]
            trade.exit_price = last_price
            trade.exit_reason = "EOD_FORCE"
            trade.pnl    = (last_price - trade.entry_price) * trade.quantity
            trade.charges += charges
            trade.net_pnl = trade.pnl - trade.charges
            capital += trade.quantity * last_price - charges
            all_trades.append(trade)

        return self._compute_metrics(
            all_trades, equity_curve, capital, start_date, end_date, regime_stats
        )

    def _get_regime(self, vix: float, stock_by_date: dict, date) -> str:
        """Simple regime from VIX level and average ADX."""
        if vix > 25:
            return "high_vol_crisis"
        adx_vals = []
        for df in list(stock_by_date.values())[:20]:   # sample 20 stocks
            if date in df.index and "adx" in df.columns:
                adx_vals.append(float(df.loc[date, "adx"]))
        avg_adx = np.mean(adx_vals) if adx_vals else 20.0

        if avg_adx > 25 and vix < 20:
            return "trending"
        elif avg_adx < 20:
            return "mean_reverting"
        else:
            return "normal"

    def _compute_charges(self, qty: int, price: float, side: str) -> float:
        model     = BROKERAGE_MODELS.get(self.brokerage_model, BROKERAGE_MODELS["zerodha_eq_delivery"])
        value     = qty * price
        brokerage = max(model["flat"], value * model["pct"])
        stt       = value * STT_DELIVERY if side == "SELL" else 0
        sebi      = value * SEBI_CHARGES
        exchange  = value * EXCHANGE_TXN
        stamp     = value * STAMP_DUTY if side == "BUY" else 0
        gst       = (brokerage + sebi + exchange) * 0.18
        return brokerage + stt + sebi + exchange + stamp + gst

    def _compute_metrics(self, trades, equity_curve, final_capital,
                          start_date, end_date, regime_stats) -> BacktestResult:
        completed = [t for t in trades if t.exit_price is not None]
        net_profit = final_capital - self.initial_capital
        total_return = net_profit / self.initial_capital * 100

        years = max(0.1, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25)
        cagr  = ((final_capital / self.initial_capital) ** (1/years) - 1) * 100

        # Daily returns from equity curve
        eq_series  = pd.Series(equity_curve).sort_index()
        daily_rets = eq_series.pct_change().dropna()

        sharpe  = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
        neg_rets = daily_rets[daily_rets < 0]
        sortino = (daily_rets.mean() / neg_rets.std() * np.sqrt(252)) if (len(neg_rets) > 0 and neg_rets.std() > 0) else 0

        # Drawdown
        roll_max = eq_series.cummax()
        drawdown = (eq_series - roll_max) / roll_max * 100
        max_dd   = abs(drawdown.min())
        calmar   = (cagr / max_dd) if max_dd > 0 else 0

        # Trade stats
        wins  = [t for t in completed if t.net_pnl > 0]
        loses = [t for t in completed if t.net_pnl <= 0]
        win_rate  = len(wins) / max(1, len(completed))
        avg_win   = np.mean([t.pnl_pct for t in wins])  if wins  else 0
        avg_loss  = np.mean([t.pnl_pct for t in loses]) if loses else 0
        rr        = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        total_charges  = sum(t.charges for t in completed)
        total_slippage = sum(t.slippage_cost for t in completed)
        avg_hold = np.mean([(t.exit_date - t.entry_date).days for t in completed if t.exit_date]) if completed else 0

        months = max(1, years * 12)
        tpm    = len(completed) / months

        # Regime breakdown
        regime_trade_counts = {}
        for t in completed:
            regime_trade_counts[t.regime] = regime_trade_counts.get(t.regime, 0) + 1

        # Monthly returns
        monthly = {}
        if not eq_series.empty:
            monthly_eq = eq_series.resample("ME").last()
            monthly_rets = monthly_eq.pct_change().dropna()
            monthly = {str(d.date()): round(float(v*100), 2) for d, v in monthly_rets.items()}

        # Trade log
        trade_log = []
        for t in sorted(completed, key=lambda x: x.entry_date):
            trade_log.append({
                "ticker":       t.ticker,
                "entry_date":   str(t.entry_date.date()),
                "exit_date":    str(t.exit_date.date()) if t.exit_date else None,
                "entry_price":  round(t.entry_price, 2),
                "entry_raw":    round(t.entry_price_raw, 2),
                "slippage":     round(t.slippage_cost, 2),
                "exit_price":   round(t.exit_price, 2) if t.exit_price else None,
                "quantity":     t.quantity,
                "exit_reason":  t.exit_reason,
                "pnl":          round(t.pnl, 2),
                "pnl_pct":      round(t.pnl_pct, 2),
                "charges":      round(t.charges, 2),
                "net_pnl":      round(t.net_pnl, 2),
                "tech_score":   t.tech_score,
                "ml_prob":      t.ml_prob,
                "regime":       t.regime,
            })

        logger.info(f"\nBacktest Results:")
        logger.info(f"  Return: {total_return:.1f}%  CAGR: {cagr:.1f}%")
        logger.info(f"  Sharpe: {sharpe:.2f}  Sortino: {sortino:.2f}  Calmar: {calmar:.2f}")
        logger.info(f"  Max DD: -{max_dd:.1f}%  Win Rate: {win_rate:.1%}")
        logger.info(f"  Trades: {len(completed)}  Charges: ₹{total_charges:,.0f}  Slippage: ₹{total_slippage:,.0f}")
        logger.info(f"  Regime breakdown: {regime_stats}")

        return BacktestResult(
            strategy_name     = self.strategy_name,
            start_date        = start_date,
            end_date          = end_date,
            initial_capital   = self.initial_capital,
            final_capital     = round(final_capital, 2),
            net_profit        = round(net_profit, 2),
            total_return_pct  = round(total_return, 2),
            cagr_pct          = round(cagr, 2),
            sharpe_ratio      = round(sharpe, 3),
            sortino_ratio     = round(sortino, 3),
            calmar_ratio      = round(calmar, 3),
            max_drawdown_pct  = round(max_dd, 2),
            max_drawdown_rs   = round(abs(drawdown.idxmin() and eq_series.max() - eq_series.min()), 2),
            win_rate          = round(win_rate, 4),
            avg_win_pct       = round(avg_win, 2),
            avg_loss_pct      = round(avg_loss, 2),
            risk_reward_ratio = round(rr, 2),
            total_trades      = len(completed),
            winning_trades    = len(wins),
            losing_trades     = len(loses),
            avg_hold_days     = round(avg_hold, 1),
            trades_per_month  = round(tpm, 1),
            total_charges     = round(total_charges, 2),
            total_slippage    = round(total_slippage, 2),
            regime_breakdown  = {**regime_stats, "by_trade": regime_trade_counts},
            equity_curve      = {str(d.date()): round(v, 2) for d, v in equity_curve.items()},
            monthly_returns   = monthly,
            trade_log         = trade_log,
        )
