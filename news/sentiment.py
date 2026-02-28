"""
news/sentiment.py  — v4 (non-blocking, cached)
────────────────────────────────────────────────
Fixes:
  ✅ FinBERT runs in a ThreadPoolExecutor — never blocks asyncio event loop
  ✅ Result cache with TTL (1 hour) — same ticker doesn't re-run FinBERT
  ✅ RSS fetch with per-source timeout — one slow feed can't block others
  ✅ numpy imported at top (was at bottom, causing NameError in some paths)

Architecture:
  - get_stock_sentiment() is the synchronous API used by the DPG terminal
  - get_stock_sentiment_async() is the awaitable version for FastAPI
  Both share the same underlying cache and thread pool.
"""

import time
import threading
import feedparser
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
from dataclasses import dataclass

try:
    from transformers import pipeline as hf_pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logger.warning("transformers not installed — using VADER fallback")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    _vader = None
    VADER_AVAILABLE = False


# ── Trusted RSS feeds ─────────────────────────────────────────────────

RSS_FEEDS = {
    "Economic Times":    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol":      "https://www.moneycontrol.com/rss/MCrecentnews.xml",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
    "Mint":              "https://www.livemint.com/rss/markets",
    "Financial Express": "https://www.financialexpress.com/market/feed/",
}

SECTOR_KEYWORDS = {
    "banking":  ["bank","nbfc","credit","loan","npa","rbi","interest rate"],
    "it":       ["software","it","tech","cloud","saas","ai","infosys","tcs","wipro"],
    "pharma":   ["pharma","drug","fda","usfda","api","healthcare","hospital"],
    "auto":     ["automobile","car","ev","electric vehicle","suv","two-wheeler"],
    "energy":   ["oil","gas","power","renewable","solar","wind","coal","crude"],
    "fmcg":     ["fmcg","consumer","food","beverage","volume growth"],
    "metal":    ["steel","metal","aluminium","copper","zinc","mining"],
    "infra":    ["infrastructure","road","highway","construction","cement"],
}

# ── Cache ─────────────────────────────────────────────────────────────

_CACHE: dict = {}           # ticker → {result, expires_at}
_CACHE_TTL  = 3600          # 1 hour
_CACHE_LOCK = threading.Lock()

# Single shared thread-pool for FinBERT inference
_THREAD_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="FinBERT")


@dataclass
class SentimentResult:
    score:            float   # 0–100
    label:            str
    article_count:    int
    source_breakdown: dict
    cached:           bool = False


class NewsSentimentEngine:
    """
    Non-blocking news sentiment engine.

    FinBERT inference is pushed to a background ThreadPoolExecutor,
    so FastAPI's asyncio event loop is never blocked.
    The DPG terminal calls get_stock_sentiment() directly from its
    background inference thread (not the render thread).
    """

    def __init__(self, use_finbert: bool = True, max_articles: int = 30):
        self.max_articles = max_articles
        self._nlp         = None
        self._nlp_lock    = threading.Lock()
        self._mode        = "none"

        if use_finbert and FINBERT_AVAILABLE:
            # Load lazily in background thread — don't block __init__
            threading.Thread(target=self._load_finbert, daemon=True).start()
        elif VADER_AVAILABLE:
            self._mode = "vader"
            logger.info("Sentiment: using VADER")
        else:
            logger.warning("Sentiment: no NLP engine — returning neutral scores")

    def _load_finbert(self):
        try:
            logger.info("Loading FinBERT (background)...")
            nlp = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1,  # CPU; change to 0 for GPU
            )
            with self._nlp_lock:
                self._nlp  = nlp
                self._mode = "finbert"
            logger.info("FinBERT loaded OK")
        except Exception as e:
            logger.warning(f"FinBERT load failed: {e} — using VADER")
            with self._nlp_lock:
                self._mode = "vader" if VADER_AVAILABLE else "none"

    # ── Public sync API (used by DPG terminal background thread) ─────────

    def get_stock_sentiment(
        self,
        ticker:        str,
        hours_window:  int   = 24,
        min_confidence: float = 0.45,
    ) -> dict:
        """
        Synchronous. Safe to call from any thread.
        Returns cached result if < 1 hour old.
        """
        cache_key = f"{ticker}_{hours_window}"
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
            if cached and time.time() < cached["expires_at"]:
                r = cached["result"]
                r["cached"] = True
                return r

        result = self._compute(ticker, hours_window, min_confidence)

        with _CACHE_LOCK:
            _CACHE[cache_key] = {
                "result":     result,
                "expires_at": time.time() + _CACHE_TTL,
            }
        return result

    async def get_stock_sentiment_async(
        self,
        ticker:       str,
        hours_window: int   = 24,
    ) -> dict:
        """
        Async wrapper — runs computation in thread pool, never blocks event loop.
        Use this in FastAPI endpoints.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _THREAD_POOL,
            self.get_stock_sentiment,
            ticker,
            hours_window,
        )

    def invalidate_cache(self, ticker: str = None):
        """Clear cache for a ticker (or all if None)."""
        with _CACHE_LOCK:
            if ticker:
                for k in list(_CACHE.keys()):
                    if k.startswith(ticker):
                        del _CACHE[k]
            else:
                _CACHE.clear()

    # ── Internal computation ─────────────────────────────────────────────

    def _compute(self, ticker: str, hours_window: int, min_confidence: float) -> dict:
        """Fetch + score articles. CPU-bound but runs in thread pool."""
        articles = self._fetch_all_articles(ticker, hours_window)
        if not articles:
            return self._neutral(ticker)

        scored = []
        for art in articles[:self.max_articles]:
            text                = (art["title"] + ". " + art.get("summary", ""))[:512]
            sentiment, confidence = self._score_text(text)
            if confidence < min_confidence:
                continue
            age_h  = max(0, (datetime.now() - art["published"]).total_seconds() / 3600)
            weight = max(0.05, np.exp(-age_h / max(1, hours_window)))
            scored.append({
                "title":      art["title"][:120],
                "source":     art["source"],
                "url":        art.get("url", ""),
                "sentiment":  round(sentiment, 3),
                "confidence": round(confidence, 3),
                "weight":     round(weight, 3),
                "age_hours":  round(age_h, 1),
            })

        if not scored:
            return self._neutral(ticker)

        total_w  = sum(a["weight"] for a in scored)
        avg_sent = sum(a["sentiment"] * a["weight"] for a in scored) / total_w
        score    = (avg_sent + 1) / 2 * 100

        label = ("Strongly Bullish" if score >= 70 else
                 "Bullish"          if score >= 58 else
                 "Neutral"          if score >= 42 else
                 "Bearish"          if score >= 30 else
                 "Strongly Bearish")

        by_source: dict = {}
        for a in scored:
            s = a["source"]
            by_source.setdefault(s, []).append(a["sentiment"])
        source_breakdown = {
            s: round(sum(v) / len(v) * 50 + 50, 1)
            for s, v in by_source.items()
        }

        return {
            "ticker":           ticker,
            "score":            round(score, 1),
            "label":            label,
            "article_count":    len(scored),
            "source_breakdown": source_breakdown,
            "articles":         scored[:8],
            "cached":           False,
        }

    def _score_text(self, text: str) -> tuple:
        with self._nlp_lock:
            mode = self._mode
            nlp  = self._nlp

        if mode == "finbert" and nlp is not None:
            try:
                res   = nlp(text)[0]
                label = res["label"].lower()
                score = float(res["score"])
                if label == "positive":
                    return score, score
                elif label == "negative":
                    return -score, score
                else:
                    return 0.0, score
            except Exception:
                pass

        if mode == "vader" and VADER_AVAILABLE and _vader:
            scores = _vader.polarity_scores(text)
            return scores["compound"], abs(scores["compound"])

        return 0.0, 0.5

    def _fetch_all_articles(self, ticker: str, hours_window: int) -> list:
        """Fetch from all RSS feeds concurrently with per-source timeout."""
        cutoff   = datetime.now() - timedelta(hours=hours_window * 3)
        articles = []

        futures = {
            _THREAD_POOL.submit(self._fetch_rss, src, url): src
            for src, url in RSS_FEEDS.items()
        }
        for fut, src in futures.items():
            try:
                items = fut.result(timeout=8)
                articles.extend(items)
            except FuturesTimeout:
                logger.debug(f"RSS timeout: {src}")
            except Exception as e:
                logger.debug(f"RSS error {src}: {e}")

        # Filter relevant + recent
        return [
            a for a in articles
            if self._is_relevant(a, ticker) and a["published"] > cutoff
        ]

    def _fetch_rss(self, source: str, url: str) -> list:
        try:
            feed  = feedparser.parse(url)
            items = []
            for e in feed.entries[:15]:
                pub = self._parse_date(getattr(e, "published", ""))
                items.append({
                    "source":    source,
                    "title":     getattr(e, "title", ""),
                    "summary":   getattr(e, "summary", ""),
                    "url":       getattr(e, "link", ""),
                    "published": pub,
                })
            return items
        except Exception as e:
            logger.debug(f"RSS parse error {source}: {e}")
            return []

    def _parse_date(self, s: str) -> datetime:
        for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z",
                    "%a, %d %b %Y %H:%M:%S GMT"]:
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=None)
            except (ValueError, TypeError):
                continue
        return datetime.now()

    def _is_relevant(self, item: dict, ticker: str) -> bool:
        text  = (item["title"] + " " + item.get("summary", "")).lower()
        short = ticker.lower().replace("ind", "").replace("ltd", "")
        return ticker.lower() in text or (len(short) > 3 and short in text)

    @staticmethod
    def _neutral(ticker: str) -> dict:
        return {
            "ticker": ticker, "score": 50.0, "label": "Neutral",
            "article_count": 0, "source_breakdown": {}, "articles": [], "cached": False,
        }
