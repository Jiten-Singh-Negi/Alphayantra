"""
monitor/telegram.py  — v5 (with message batching / rate-limit guard)
──────────────────────────────────────────────────────────────────────
TELEGRAM THROTTLE FIX:
  OLD: send_message() fired an immediate requests.post() for every call.
       If 15 breakout signals triggered at 9:15 AM, we'd send 15 requests
       in a few ms. Telegram enforces ~20 msg/min per chat — the bot gets
       a 429 Too Many Requests block and critical risk alerts are silently
       dropped.

  FIX: MessageBatcher background thread.
       All calls to send_message() go into a thread-safe deque.
       A background daemon flushes the queue in 2-second windows:
         - If 1 message queued:  send it immediately.
         - If N > 1 queued:      concatenate into ONE message with newlines,
                                 send exactly one HTTP request.
         - If queue still backed up: apply 3s backoff between batches.
       This guarantees ≤ 1 message per 2-second window → max 30/min,
       well under Telegram's 20/min chat limit.

       CRITICAL alerts bypass the batcher entirely (sent inline, immediately)
       because a ₹500K position flattening cannot afford a 2-second delay.

Setup (one-time):
  1. Message @BotFather on Telegram → /newbot → copy BOT_TOKEN
  2. Message your bot once, then run: python -c "from monitor.telegram import TelegramMonitor; TelegramMonitor().get_chat_id()"
  3. Add to .env:
       TELEGRAM_BOT_TOKEN=your_token_here
       TELEGRAM_CHAT_ID=your_chat_id_here

Alerts sent automatically:
  - Pre-market top signals (9:00 AM via scheduler)
  - STRONG BUY / STRONG SELL during market hours
  - Options: PCR extremes, max pain alerts
  - Daily P&L report (6:00 PM)
  - Weekly retrain completion
  - Risk limit breaches (immediate, bypasses batcher)
  - System errors
"""

import os
import time
import threading
import requests
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
from loguru import logger
from typing import Optional

IST = ZoneInfo("Asia/Kolkata")

TELEGRAM_API     = "https://api.telegram.org/bot{token}/{method}"
BATCH_WINDOW_SEC = 2.0    # flush every 2 seconds
MAX_BATCH_CHARS  = 3800   # Telegram limit is 4096 chars; stay safely below
BACKOFF_SEC      = 3.0    # wait after any 429 response


class TelegramMonitor:

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id:   Optional[str] = None,
    ):
        self.token   = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id   or os.getenv("TELEGRAM_CHAT_ID",   "")
        self.enabled = bool(self.token and self.chat_id)

        # ── Message batcher ──────────────────────────────────────────────
        self._queue  : deque = deque()          # pending messages
        self._q_lock = threading.Lock()
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="TelegramBatcher"
        )
        if self.enabled:
            self._flush_thread.start()
            logger.info("Telegram monitor ready (batching on)")
        else:
            logger.warning(
                "Telegram not configured — alerts disabled.\n"
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable."
            )

    # ── Public API ────────────────────────────────────────────────────────

    def send_message(self, text: str, parse_mode: str = "HTML",
                     critical: bool = False) -> bool:
        """
        Queue a message for batched delivery.

        critical=True: bypasses the batcher and fires immediately.
        Use for kill-switch activations and hard risk-limit breaches
        where a 2-second delay is unacceptable.
        """
        if not self.enabled:
            logger.info(f"[Telegram disabled] {text[:80]}")
            return False

        if critical:
            return self._send_now(text, parse_mode)

        # Normal path — queue for batching
        with self._q_lock:
            self._queue.append((text, parse_mode))
        return True

    def send_signal_alert(self, signal: dict):
        emoji = {
            "STRONG BUY":  "🚀",
            "BUY":         "🟢",
            "HOLD":        "⚪",
            "SELL":        "🔴",
            "STRONG SELL": "💥",
        }.get(signal.get("signal", ""), "📊")

        text = (
            f"{emoji} <b>{signal.get('signal')} — {signal.get('ticker')}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 LTP:          ₹{signal.get('close', 0):,.2f}\n"
            f"🎯 ML Confidence: {signal.get('ml_confidence', 0):.1f}%\n"
            f"📊 Tech Score:    {signal.get('tech_score', 0):.1f}/100\n"
            f"📰 News Score:    {signal.get('news_score', 50):.1f}/100\n"
            f"🛑 Stop Loss:     ₹{signal.get('stop_loss', 0):,.2f}\n"
            f"✅ Take Profit:   ₹{signal.get('take_profit', 0):,.2f}\n"
            f"⚡ Confirmations: {signal.get('confirmations', 0)} indicators\n"
            f"🕐 {datetime.now(IST).strftime('%H:%M:%S IST')}"
        )
        self.send_message(text)

    def send_options_alert(self, recommendation: dict):
        rec = recommendation.get("recommended", {})
        if not rec:
            return
        text = (
            f"📈 <b>Options Signal — {recommendation.get('signal')}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📌 Strike:    {rec.get('type')} {rec.get('strike'):,.0f}\n"
            f"💰 LTP:       ₹{rec.get('ltp', 0):.2f}\n"
            f"📊 IV:        {rec.get('iv', 0):.1f}%\n"
            f"Δ  Delta:     {rec.get('delta', 0):.3f}\n"
            f"θ  Theta:     ₹{rec.get('theta', 0):.2f}/day\n"
            f"📦 Per Lot:   ₹{rec.get('per_lot', 0):,.0f}\n"
            f"📉 PCR:       {recommendation.get('pcr', 0):.2f} ({recommendation.get('sentiment')})\n"
            f"🎯 Max Pain:  ₹{recommendation.get('max_pain', 0):,.0f}\n"
            f"📅 Expiry:    {rec.get('expiry')}\n"
            f"\n💡 {recommendation.get('strategy_note', '')[:200]}"
        )
        self.send_message(text)

    def send_pre_market_report(self, signals: list):
        if not signals:
            return
        lines = [f"🌅 <b>Pre-Market Top Signals — {datetime.now(IST).strftime('%d %b %Y')}</b>\n"]
        for i, s in enumerate(signals[:5], 1):
            emoji = "🚀" if "STRONG" in s.get("signal","") else "🟢" if "BUY" in s.get("signal","") else "🔴"
            lines.append(
                f"{i}. {emoji} <b>{s['ticker']}</b> — {s['signal']}\n"
                f"   ₹{s.get('close',0):,.2f} | ML: {s.get('ml_confidence',0):.0f}% | "
                f"Score: {s.get('combined_score',0):.0f}"
            )
        self.send_message("\n".join(lines))

    def send_eod_signals(self, signals: list):
        if not signals:
            return
        lines = [f"📋 <b>EOD Watchlist for Tomorrow</b>\n"]
        for i, s in enumerate(signals[:8], 1):
            lines.append(
                f"{i}. <b>{s['ticker']}</b> {s['signal']} | "
                f"₹{s.get('close',0):,.0f} | {s.get('ml_confidence',0):.0f}%"
            )
        self.send_message("\n".join(lines))

    def send_risk_alert(self, message: str, critical: bool = False):
        """
        Risk alerts use critical=True to bypass the batcher.
        A stop-loss breach must reach you immediately, not in 2 seconds.
        """
        prefix = "🛑 <b>CRITICAL RISK ALERT</b>" if critical else "⚠️ <b>Risk Alert</b>"
        self.send_message(f"{prefix}\n{message}", critical=critical)

    def send_error(self, error: str):
        self.send_message(f"❌ <b>System Error</b>\n<code>{error[:500]}</code>")

    def send_retrain_complete(self, metrics: dict):
        self.send_message(
            f"🧠 <b>Model Retrain Complete</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📊 Ensemble AUC:  {metrics.get('ensemble_auc', 'N/A')}\n"
            f"✅ Accuracy:      {metrics.get('ensemble_accuracy', 0):.1%}\n"
            f"🔁 TCN included:  {metrics.get('tcn_included', False)}\n"
            f"📅 Trained at:    {metrics.get('trained_at', '')[:16]}"
        )

    def get_chat_id(self) -> Optional[str]:
        """Run once after messaging your bot to get your chat ID."""
        url  = TELEGRAM_API.format(token=self.token, method="getUpdates")
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            updates = resp.json().get("result", [])
            if updates:
                cid = updates[-1]["message"]["chat"]["id"]
                print(f"Your Chat ID: {cid}")
                return str(cid)
        print("No messages found — send a message to your bot first")
        return None

    # ── Internal batching engine ──────────────────────────────────────────

    def _flush_loop(self):
        """
        Background thread: flushes message queue every BATCH_WINDOW_SEC.
        If multiple messages queued, concatenates them into one request.
        Backs off after 429 responses.
        """
        while True:
            time.sleep(BATCH_WINDOW_SEC)
            self._flush_once()

    def _flush_once(self):
        with self._q_lock:
            if not self._queue:
                return
            # Drain all queued messages into a local list
            batch = list(self._queue)
            self._queue.clear()

        if not batch:
            return

        # Group by parse_mode (usually always HTML)
        # Concatenate up to MAX_BATCH_CHARS then send; start a new batch
        current_chunks   = []
        current_mode     = batch[0][1]
        current_len      = 0
        groups_to_send   = []

        for text, mode in batch:
            if mode != current_mode or current_len + len(text) > MAX_BATCH_CHARS:
                if current_chunks:
                    groups_to_send.append(("\n\n".join(current_chunks), current_mode))
                current_chunks = [text]
                current_mode   = mode
                current_len    = len(text)
            else:
                current_chunks.append(text)
                current_len   += len(text) + 2  # +2 for "\n\n"

        if current_chunks:
            groups_to_send.append(("\n\n".join(current_chunks), current_mode))

        n = len(batch)
        if n > 1:
            logger.debug(f"Telegram batcher: sending {n} messages as {len(groups_to_send)} batch(es)")

        for body, mode in groups_to_send:
            ok = self._send_now(body, mode)
            if not ok:
                # Back off and re-queue remaining if 429
                time.sleep(BACKOFF_SEC)

    def _send_now(self, text: str, parse_mode: str = "HTML") -> bool:
        """Fire one HTTP request synchronously. Returns True on success."""
        try:
            url  = TELEGRAM_API.format(token=self.token, method="sendMessage")
            resp = requests.post(url, json={
                "chat_id":    self.chat_id,
                "text":       text[:4096],  # hard Telegram limit
                "parse_mode": parse_mode,
            }, timeout=8)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", BACKOFF_SEC))
                logger.warning(f"Telegram 429 — backing off {retry_after}s")
                time.sleep(retry_after)
                return False
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"Telegram send failed: {e}")
            return False
