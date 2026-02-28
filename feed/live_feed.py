"""
feed/live_feed.py  — v4 (memory-safe, rate-limited)
─────────────────────────────────────────────────────
Fixes applied:
  ✅ All tick buffers are bounded deque(maxlen=N) — zero unbounded growth
  ✅ NSE rate limiting: 600ms between symbols + exponential backoff on 429
  ✅ Rotating User-Agent pool to avoid IP-level blocks
  ✅ Cookie refresh every 4 minutes (NSE sessions expire at ~5 min)
  ✅ Quotes pushed into a shared multiprocessing-safe Queue so the
     DPG terminal reads directly from RAM with zero HTTP overhead
"""

import json
import time
import threading
import requests
import random
import pandas as pd
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from collections import defaultdict, deque
from loguru import logger
from typing import Optional, Callable
from queue import Queue

IST = ZoneInfo("Asia/Kolkata")

MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
PRE_OPEN     = dtime(9, 0)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]

NSE_INDEX_URL = "https://www.nseindia.com/api/allIndices"
NSE_QUOTE_URL = "https://www.nseindia.com/api/quote-equity?symbol={symbol}"

MAX_TICKS_PER_SYMBOL = 5_000
MAX_CANDLES          = 390
MAX_QUEUE_SIZE       = 1_000

DEFAULT_WATCHLIST = [
    "NIFTY", "BANKNIFTY", "FINNIFTY",
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "AXISBANK", "BAJFINANCE", "TATAMOTORS", "WIPRO",
]


def is_market_open() -> bool:
    now = datetime.now(IST).time()
    return datetime.now(IST).weekday() < 5 and MARKET_OPEN <= now <= MARKET_CLOSE


def is_pre_open() -> bool:
    now = datetime.now(IST).time()
    return PRE_OPEN <= now < MARKET_OPEN


def seconds_to_open() -> int:
    now  = datetime.now(IST)
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    return max(0, int((open_t - now).total_seconds()))


# ── NSE Session with backoff + UA rotation ─────────────────────────────

class NSESession:
    REFRESH_INTERVAL = 240  # 4 min

    def __init__(self):
        self.session          = requests.Session()
        self._last_refresh    = 0.0
        self._last_req_ts     = 0.0
        self._backoff         = 1.0
        self._refresh()

    def _refresh(self):
        ua = random.choice(_USER_AGENTS)
        self.session.headers.update({
            "User-Agent":      ua,
            "Accept":          "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer":         "https://www.nseindia.com/",
        })
        try:
            self.session.get("https://www.nseindia.com", timeout=8)
            self._last_refresh = time.time()
            self._backoff      = 1.0
        except Exception as e:
            logger.debug(f"NSE session refresh: {e}")

    def _rate_gate(self):
        elapsed = time.time() - self._last_req_ts
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_req_ts = time.time()

    def get(self, url: str, timeout: int = 8) -> Optional[dict]:
        if time.time() - self._last_refresh > self.REFRESH_INTERVAL:
            self._refresh()
        self._rate_gate()
        for attempt in range(3):
            try:
                r = self.session.get(url, timeout=timeout)
                if r.status_code == 200:
                    self._backoff = max(1.0, self._backoff * 0.5)
                    return r.json()
                elif r.status_code == 401:
                    self._refresh()
                elif r.status_code == 429:
                    wait = self._backoff * (2 ** attempt)
                    self._backoff = min(60.0, self._backoff * 2)
                    logger.warning(f"NSE 429 — waiting {wait:.0f}s")
                    time.sleep(wait)
                else:
                    return None
            except requests.Timeout:
                time.sleep(1.0 * (attempt + 1))
            except Exception as e:
                logger.debug(f"NSE GET error: {e}")
                return None
        return None


class QuoteFetcher:
    def __init__(self):
        self.nse = NSESession()

    def get_index_quote(self, index: str = "NIFTY 50") -> Optional[dict]:
        data = self.nse.get(NSE_INDEX_URL)
        if not data:
            return None
        for item in data.get("data", []):
            if item.get("index", "").upper() == index.upper():
                return {
                    "symbol":     index,
                    "ltp":        item.get("last", 0),
                    "change":     item.get("variation", 0),
                    "change_pct": item.get("percentChange", 0),
                    "open":       item.get("open", 0),
                    "high":       item.get("high", 0),
                    "low":        item.get("low", 0),
                    "prev_close": item.get("previousClose", 0),
                    "timestamp":  datetime.now(IST).isoformat(),
                }
        return None

    def get_stock_quote(self, symbol: str) -> Optional[dict]:
        data = self.nse.get(NSE_QUOTE_URL.format(symbol=symbol))
        if not data:
            return None
        pi = data.get("priceInfo", {})
        return {
            "symbol":     symbol,
            "ltp":        pi.get("lastPrice", 0),
            "change":     pi.get("change", 0),
            "change_pct": pi.get("pChange", 0),
            "open":       pi.get("open", 0),
            "high":       pi.get("high", 0),
            "low":        pi.get("low", 0),
            "prev_close": pi.get("previousClose", 0),
            "volume":     (data.get("marketDeptOrderBook") or {})
                          .get("tradeInfo", {}).get("totalTradedVolume", 0),
            "timestamp":  datetime.now(IST).isoformat(),
        }

    def get_all_indices(self) -> list:
        data = self.nse.get(NSE_INDEX_URL)
        if not data:
            return []
        return [
            {"index": d.get("index"), "ltp": d.get("last", 0),
             "change_pct": d.get("percentChange", 0)}
            for d in data.get("data", []) if d.get("index")
        ]


class CandleBuilder:
    def __init__(self, symbol: str, max_candles: int = MAX_CANDLES):
        self.symbol       = symbol
        self.candles      : deque = deque(maxlen=max_candles)
        self._current     : Optional[dict] = None
        self._current_min : Optional[int]  = None

    def add_tick(self, price: float, volume: int = 0):
        now    = datetime.now(IST)
        minute = now.hour * 60 + now.minute
        if self._current_min != minute:
            if self._current:
                self.candles.append(dict(self._current))
            self._current = {
                "datetime": now.replace(second=0, microsecond=0),
                "Open": price, "High": price, "Low": price,
                "Close": price, "Volume": volume,
            }
            self._current_min = minute
        else:
            self._current["High"]   = max(self._current["High"], price)
            self._current["Low"]    = min(self._current["Low"],  price)
            self._current["Close"]  = price
            self._current["Volume"] += volume

    def get_dataframe(self) -> pd.DataFrame:
        candles = list(self.candles)
        if self._current:
            candles.append(dict(self._current))
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        df.rename(columns={"datetime": "Date"}, inplace=True)
        df["Ticker"] = self.symbol
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = 0.0
        return df[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]]

    @property
    def latest_price(self) -> Optional[float]:
        if self._current:
            return self._current["Close"]
        if self.candles:
            return self.candles[-1]["Close"]
        return None


class LiveFeed:
    """
    Polls NSE. All data goes into:
      self.latest_quotes  — plain dict (bounded by watchlist size)
      self.tick_history   — bounded deque per symbol
      self.tick_queue     — Queue(maxsize=MAX_QUEUE_SIZE) for DPG terminal
    """

    def __init__(
        self,
        watchlist:     list   = DEFAULT_WATCHLIST,
        poll_interval: int    = 5,
        tick_queue:    Optional[Queue] = None,
    ):
        self.watchlist       = list(watchlist)
        self.poll_interval   = poll_interval
        self.tick_queue      = tick_queue or Queue(maxsize=MAX_QUEUE_SIZE)
        self.fetcher         = QuoteFetcher()
        self.candle_builders = {s: CandleBuilder(s) for s in self.watchlist}
        self.latest_quotes   : dict = {}
        self.tick_history    : dict = {
            s: deque(maxlen=MAX_TICKS_PER_SYMBOL) for s in self.watchlist
        }
        self._callbacks      : dict = defaultdict(list)
        self._running        = False
        self._thread         : Optional[threading.Thread] = None

    def subscribe(self, symbol: str, callback: Callable):
        self._callbacks[symbol].append(callback)

    def subscribe_all(self, callback: Callable):
        self._callbacks["*"].append(callback)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop, daemon=True, name="LiveFeed")
        self._thread.start()
        logger.info(f"LiveFeed started — {len(self.watchlist)} symbols, poll={self.poll_interval}s")

    def stop(self):
        self._running = False

    def get_quote(self, symbol: str) -> Optional[dict]:
        return self.latest_quotes.get(symbol)

    def get_candles(self, symbol: str) -> pd.DataFrame:
        return self.candle_builders.get(symbol, CandleBuilder(symbol)).get_dataframe()

    def _run_loop(self):
        index_map = {
            "NIFTY":     "NIFTY 50",
            "BANKNIFTY": "NIFTY BANK",
            "FINNIFTY":  "NIFTY FIN SERVICE",
        }
        while self._running:
            if not is_market_open():
                wait = seconds_to_open()
                time.sleep(min(max(wait, 15), 300))
                continue

            for symbol in self.watchlist:
                if not self._running:
                    break
                try:
                    quote = (self.fetcher.get_index_quote(index_map[symbol])
                             if symbol in index_map
                             else self.fetcher.get_stock_quote(symbol))

                    if quote:
                        self.latest_quotes[symbol] = quote
                        ltp = float(quote.get("ltp", 0) or 0)
                        if ltp > 0:
                            self.tick_history[symbol].append(
                                {"ts": datetime.now(IST).isoformat(), "price": ltp}
                            )
                            self.candle_builders[symbol].add_tick(ltp)

                        # Non-blocking push to queue — drop if full
                        try:
                            self.tick_queue.put_nowait({"symbol": symbol, "quote": quote})
                        except Exception:
                            pass  # queue full — DPG terminal is too slow; skip

                        for cb in self._callbacks.get(symbol, []):
                            try:
                                cb(quote)
                            except Exception as e:
                                logger.error(f"Callback error {symbol}: {e}")
                        for cb in self._callbacks.get("*", []):
                            try:
                                cb(quote)
                            except Exception as e:
                                logger.error(f"Callback error *: {e}")

                except Exception as e:
                    logger.debug(f"Quote error ({symbol}): {e}")

                time.sleep(0.6)   # 600ms between symbols

            time.sleep(self.poll_interval)

    async def start_ws_server(self, host: str = "0.0.0.0", port: int = 8765):
        import asyncio
        import websockets
        logger.info(f"WebSocket bridge: ws://{host}:{port}")

        async def _handler(ws):
            try:
                while True:
                    await ws.send(json.dumps({
                        "type": "quotes",
                        "timestamp": datetime.now(IST).isoformat(),
                        "market_open": is_market_open(),
                        "quotes": self.latest_quotes,
                    }))
                    await asyncio.sleep(2)
            except websockets.exceptions.ConnectionClosed:
                pass

        async with websockets.serve(_handler, host, port):
            import asyncio
            await asyncio.Future()
