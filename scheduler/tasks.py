"""
scheduler/tasks.py  — v4 (collision-safe)
──────────────────────────────────────────
Fixes:
  ✅ Heavy tasks (Bhavcopy download, model retrain) BLOCKED during market hours
  ✅ All v3 ML train() kwargs — no horizon/threshold/lstm args
  ✅ Bhavcopy update runs nightly, not during trading hours
  ✅ Weekly retrain fires Sunday 11 PM only (market closed)
  ✅ Intraday refresh rate-limited to prevent CPU spike

Schedule (IST):
  09:00  Pre-market scan + Telegram alert
  09:15  Market open — live feed starts, daily risk state resets
  09:30  First intraday signal refresh
  Every 5 min (09:15–15:30)  — Signal refresh (lightweight)
  15:00  Final scan + EOD alert
  15:30  Market close — live feed stops
  18:00  EOD report
  18:30  Bhavcopy incremental update (post-market, light)
  23:00 Sun  Weekly full model retrain
  01st of month 20:00  Feature importance report
"""

import schedule
import time
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from loguru import logger
from pathlib import Path

IST = ZoneInfo("Asia/Kolkata")

# Heavy tasks must NOT run during this window
MARKET_OPEN_H,  MARKET_OPEN_M  = 9,  15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 35


def _is_market_hours() -> bool:
    now  = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    after_open  = (h, m) >= (MARKET_OPEN_H,  MARKET_OPEN_M)
    before_close = (h, m) <= (MARKET_CLOSE_H, MARKET_CLOSE_M)
    return after_open and before_close


def _weekday_only(fn):
    """Decorator: skip task on weekends."""
    def wrapper(*args, **kwargs):
        if datetime.now(IST).weekday() >= 5:
            return
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


def _no_market_hours(fn):
    """Decorator: skip task if market is open (prevents CPU starvation)."""
    def wrapper(*args, **kwargs):
        if _is_market_hours():
            logger.debug(f"Skipped {fn.__name__} — market hours")
            return
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


class AlphaYantraScheduler:

    def __init__(self, app_context: dict):
        """
        app_context keys:
          live_feed, options_chain, risk_manager, ml_model,
          monitor, signal_engine, paper_trader
        """
        self.ctx     = app_context
        self.running = False
        self._last_intraday = 0.0   # rate-limit intraday refresh

    def start(self):
        self._register_schedules()
        self.running = True
        t = threading.Thread(target=self._run_loop, daemon=True, name="Scheduler")
        t.start()
        logger.info("Scheduler started")

    def stop(self):
        self.running = False

    def _register_schedules(self):
        schedule.every().day.at("09:00").do(self._pre_market_scan)
        schedule.every().day.at("09:15").do(self._market_open)
        schedule.every(5).minutes.do(self._intraday_refresh)
        schedule.every().day.at("15:00").do(self._final_scan)
        schedule.every().day.at("15:30").do(self._market_close)
        schedule.every().day.at("18:00").do(self._eod_report)
        # Bhavcopy update: post-market, never during trading hours
        schedule.every().day.at("18:30").do(self._bhavcopy_update)
        # Heavy tasks: off-hours only
        schedule.every().sunday.at("23:00").do(self._weekly_retrain)
        schedule.every().day.at("20:00").do(self._monthly_rescan_if_needed)
        logger.info("Schedules registered (heavy tasks blocked during market hours)")

    def _run_loop(self):
        while self.running:
            schedule.run_pending()
            time.sleep(30)

    # ── Lightweight market tasks ─────────────────────────────────────────

    @_weekday_only
    def _pre_market_scan(self):
        logger.info("Pre-market scan 09:00...")
        try:
            engine = self.ctx.get("signal_engine")
            monitor = self.ctx.get("monitor")
            if engine:
                signals = engine(universe="nifty50", top_n=10)
                if monitor:
                    monitor.send_pre_market_report(signals)
        except Exception as e:
            logger.error(f"Pre-market scan: {e}")

    @_weekday_only
    def _market_open(self):
        logger.info("Market OPEN 09:15")
        try:
            feed = self.ctx.get("live_feed")
            if feed:
                feed.start()
            risk = self.ctx.get("risk_manager")
            if risk:
                risk._load_or_init_state()
            monitor = self.ctx.get("monitor")
            if monitor:
                monitor.send_message("🔔 Market Open — AlphaYantra live")
        except Exception as e:
            logger.error(f"Market open: {e}")

    def _intraday_refresh(self):
        """Lightweight signal refresh — rate-limited to 5-min minimum."""
        if not _is_market_hours():
            return
        now = time.time()
        if now - self._last_intraday < 290:   # 4m 50s guard
            return
        self._last_intraday = now
        try:
            engine = self.ctx.get("signal_engine")
            if engine:
                engine(universe="nifty50", top_n=5)
        except Exception as e:
            logger.debug(f"Intraday refresh: {e}")

    @_weekday_only
    def _final_scan(self):
        logger.info("Final pre-close scan 15:00")
        try:
            engine  = self.ctx.get("signal_engine")
            monitor = self.ctx.get("monitor")
            if engine:
                signals = engine(universe="nifty50", top_n=10)
                if monitor:
                    monitor.send_eod_signals(signals)
        except Exception as e:
            logger.error(f"Final scan: {e}")

    @_weekday_only
    def _market_close(self):
        logger.info("Market CLOSED 15:30")
        try:
            feed = self.ctx.get("live_feed")
            if feed:
                feed.stop()
            risk = self.ctx.get("risk_manager")
            if risk:
                logger.info(risk.daily_report())
        except Exception as e:
            logger.error(f"Market close: {e}")

    @_weekday_only
    def _eod_report(self):
        logger.info("EOD report 18:00")
        try:
            risk    = self.ctx.get("risk_manager")
            monitor = self.ctx.get("monitor")
            trader  = self.ctx.get("paper_trader")
            if risk and monitor:
                monitor.send_message(risk.daily_report())
            if trader and monitor:
                monitor.send_message(trader.daily_report())
        except Exception as e:
            logger.error(f"EOD report: {e}")

    # ── Heavy tasks (explicitly blocked during market hours) ─────────────

    @_weekday_only
    @_no_market_hours
    def _bhavcopy_update(self):
        """Incremental Bhavcopy download — post-market only."""
        logger.info("Bhavcopy incremental update 18:30...")
        try:
            from data.bhavcopy import BhavcopyScraper
            n = BhavcopyScraper().update()
            logger.info(f"Bhavcopy: {n} new days stored")
        except Exception as e:
            logger.error(f"Bhavcopy update: {e}")

    @_no_market_hours
    def _weekly_retrain(self):
        """Full model retrain Sunday 23:00 — will not fire if market open."""
        logger.info("Weekly retrain starting (Sunday 23:00)...")
        monitor = self.ctx.get("monitor")
        try:
            import sys, os
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from data.fetcher import fetch_universe
            from strategies.indicators import compute_indicators, IndicatorConfig
            from models.ml_engine import AlphaYantraML

            if monitor:
                monitor.send_message("🧠 Weekly retrain starting...")

            stock_data = fetch_universe("nifty500", period="15y")
            cfg        = IndicatorConfig()
            processed  = {}
            for ticker, df in stock_data.items():
                try:
                    processed[ticker] = compute_indicators(df, cfg)
                except Exception:
                    pass

            model   = AlphaYantraML("default")
            # ── Correct v3 kwargs — no horizon/threshold/lstm ──────────
            metrics = model.train(
                all_stock_dfs      = processed,
                n_cv_folds         = 4,
                skip_tcn           = False,
                tcn_max_samples    = 50_000,
                tcn_epochs         = 10,
                use_triple_barrier = True,
            )
            self.ctx["ml_model"] = model

            msg = (
                f"✅ Weekly retrain complete!\n"
                f"  Ensemble AUC: {metrics.get('ensemble_auc', 'N/A')}\n"
                f"  Walk-Forward AUC: {metrics.get('cv_mean_auc', 'N/A')}\n"
                f"  TCN: {'✅' if metrics.get('tcn_included') else '⚠️ skipped'}"
            )
            logger.info(msg)
            if monitor:
                monitor.send_message(msg)

        except Exception as e:
            logger.error(f"Weekly retrain failed: {e}")
            if monitor:
                monitor.send_message(f"❌ Retrain failed: {e}")

    @_no_market_hours
    def _monthly_rescan_if_needed(self):
        if datetime.now(IST).day != 1:
            return
        logger.info("Monthly feature importance report...")
        try:
            model   = self.ctx.get("ml_model")
            monitor = self.ctx.get("monitor")
            if model and model.trained:
                importance = model.get_feature_importance()
                top5 = importance.head(5)["feature"].tolist()
                if monitor:
                    monitor.send_message(
                        "📅 Monthly Feature Report\nTop 5 signals:\n" +
                        "\n".join(f"  {i+1}. {f}" for i, f in enumerate(top5))
                    )
        except Exception as e:
            logger.error(f"Monthly rescan: {e}")
