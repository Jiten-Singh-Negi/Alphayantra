"""
data/fetcher.py
───────────────
Fetches real OHLCV + volume data for Indian stocks.

CRITICAL FIX — Price Adjustment Collision:
  OLD: auto_adjust=True  (yfinance default)
    yfinance adjusts historical prices for splits/dividends.
    NSE Bhavcopy stores RAW unadjusted option strike prices.
    Result: a 2021 ₹1000 price becomes ₹100 in yfinance after a 10:1 split,
    but bhavcopy still shows OI at ₹1000 strike → ml_engine aligns garbage.

  FIX: auto_adjust=False
    We download unadjusted OHLCV so the underlying price always matches
    the historical strike prices in bhavcopy.db.
    Note: Volume remains unadjusted too, which is correct for OI correlation.

SURVIVORSHIP BIAS FIX:
  Added DELISTED_STOCKS — companies that were in Nifty indices historically
  but were delisted/went bankrupt/got acquired (Yes Bank, DHFL, Jet Airways,
  Reliance Communications, IL&FS, JSPL old, etc).
  fetch_universe() includes these when survivorship_bias_fix=True (default).
  This ensures the model learns what failing companies look like.
  Stocks that yfinance can't find (fully delisted) are silently skipped.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

# ── Cache directory ───────────────────────────────────────────────────
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Current Nifty 50 ──────────────────────────────────────────────────
NIFTY_50 = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN",
    "BHARTIARTL","ITC","KOTAKBANK","LT","AXISBANK","BAJFINANCE","MARUTI",
    "SUNPHARMA","TITAN","HCLTECH","WIPRO","ULTRACEMCO","NESTLEIND","TECHM",
    "POWERGRID","NTPC","COALINDIA","ONGC","M&M","BAJAJFINSV","DIVISLAB",
    "GRASIM","DRREDDY","ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT",
    "BAJAJ-AUTO","BPCL","BRITANNIA","CIPLA","EICHERMOT","HDFCLIFE",
    "HEROMOTOCO","HINDALCO","INDUSINDBK","JSWSTEEL","LTIM","SBILIFE",
    "SHREECEM","TATAMOTORS","TATACONSUM","TATASTEEL",
]

NIFTY_500_EXTRA = [
    "MPHASIS","PERSISTENT","COFORGE","LTIM","TATAELXSI",
    "PAYTM","NYKAA","IRCTC","CDSL",
    "HAL","BEL","BHEL","NMDC","SAIL",
    "MOIL","NATIONALUM","HINDZINC",
    "POLYCAB","HAVELLS","CROMPTON","VOLTAS","BLUESTARCO",
    "PIDILITIND","BERGERPAINTS","KANSAINER",
    "ABBINDIA","SIEMENS","ABB","CUMMINSIND","THERMAX",
    "AUROPHARMA","LUPIN","TORNTPHARM","ALKEM","IPCALAB","NATCOPHARM",
    "BANDHANBNK","FEDERALBNK","IDFCFIRSTB","RBLBANK","CANBK","UNIONBANK",
    "BANKBARODA","PNB","MAHABANK","UCOBANK",
    "ZEEL","PVRINOX",
    "INOXWIND","KAYNES","DIXON","AMBER",
    "TATACHEM","DEEPAKNTR","PIIND","CHOLAFIN",
    "MANAPPURAM","MUTHOOTFIN","BAJAJHLDNG","SBICARD","HDFCAMC",
    "MFSL","AAVAS",
    "NAUKRI","JUSTDIAL","INDIAMART",
    "MOTHERSON","BALKRISIND","APOLLOTYRE","CEAT","MRF","TVSMOTOR",
    "ASHOKLEY","ESCORTS","MAHINDCIE",
    "GLAND","LICI","METROPOLIS","THYROCARE","LALPATHLAB",
    "KARURVYSYA","DCBBANK",
    "JSWENERGY","TORNTPOWER","ADANIGREEN","ADANIPOWER","TATAPOWER",
    "NHPC","SJVN","IRFC","RECLTD","PFC","HUDCO",
    "MARICO","DABUR","EMAMILTD","GODREJCP","COLPAL",
    "TRENT","ABFRL","VEDL","HINDCOPPER",
]

# ── SURVIVORSHIP BIAS FIX — historically significant but now delisted/degraded ─
# These were Nifty 50/100/500 constituents at various points in 2010-2022.
# Including them forces the model to learn failure patterns, not just survivor bias.
# yfinance will find partial histories for most; fully gone ones are silently skipped.
DELISTED_STOCKS = [
    # Banks / NBFC — massive Nifty constituents, then collapsed
    "YESBANK",        # Yes Bank — Nifty 50 member, near-collapse 2020
    "RBLBANK",        # Still listed but heavily degraded
    "DHFL",           # DHFL Housing — delisted after default
    "ILFSTRANS",      # IL&FS Transport — fraud, delisted
    # Telecom — once Nifty heavyweights
    "RCOM",           # Reliance Communications — delisted 2021
    "IDEA",           # Vodafone Idea — still listed, 95%+ down
    # Infrastructure / Real Estate
    "UNITECH",        # Unitech — delisted after fraud
    "JAYPEEINFRA",    # Jaypee Infratech — insolvency
    "HDIL",           # HDIL — PMC Bank fraud
    # Aviation
    "JETAIRWAYS",     # Jet Airways — grounded 2019, delisted
    # Power / Infra
    "RELINFRA",       # Reliance Infrastructure — heavily distressed
    "LANCO",          # Lanco Infratech — delisted
    # Old-school IT (still listed but fallen out of favour)
    "MTNL",           # Mahanagar Telephone — near-zero
    # Metals / Mining — cyclical failures
    "JSWISPAT",       # JSW Ispat — merged out
    # Consumer
    "SPICEJET",       # SpiceJet — repeated distress
    # Sugar / agri — extreme cyclicality
    "SUCROSA",        # delisted
]

ALL_NSE_STOCKS = list(set(NIFTY_50 + NIFTY_500_EXTRA))

# Universe including delisted for survivorship-bias-free training
ALL_NSE_WITH_DELISTED = list(set(ALL_NSE_STOCKS + DELISTED_STOCKS))

UNIVERSE_MAP = {
    "nifty50":              NIFTY_50,
    "nifty500":             ALL_NSE_STOCKS,
    "nifty500_unbiased":    ALL_NSE_WITH_DELISTED,   # use for training
    "midcap150":            NIFTY_500_EXTRA[:150],
    "custom":               [],
}


def _yf_symbol(ticker: str) -> str:
    """Convert NSE symbol to Yahoo Finance format (append .NS)."""
    if "." in ticker:
        return ticker
    special = {
        "M&M":         "M&M.NS",
        "BAJAJ-AUTO":  "BAJAJ-AUTO.NS",
        "BERGERPAINTS":"BERGEPAINT.NS",
        "BLUESTARCO":  "BLUESTARCO.NS",
        "IPCALAB":     "IPCALAB.NS",
        "DIXON":       "DIXON.NS",
        "TVSMOTOR":    "TVSMOTOR.NS",
        "LTIM":        "LTIM.NS",
        "YESBANK":     "YESBANK.NS",
        "IDEA":        "IDEA.NS",
        "RCOM":        "RCOM.NS",
        "SPICEJET":    "SPICEJET.NS",
    }
    if ticker in special:
        return special[ticker]
    return f"{ticker}.NS"


def fetch_ohlcv(
    ticker: str,
    period: str = "15y",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Dual-track price fetcher — solves the fundamental indicator/options conflict.

    THE PROBLEM (why the old single-track approach destroyed ML model quality):
    ──────────────────────────────────────────────────────────────────────────
    Indian stocks frequently undergo corporate actions: 1:1 bonus issues,
    stock splits (2:1, 5:1, 10:1), rights issues. A 1:1 bonus drops the
    unadjusted share price by exactly 50% in one day.

    OLD approach (auto_adjust=False only):
      - yfinance returns unadjusted OHLCV
      - RSI sees a genuine 50% one-day drop → plunges to 2-5 (extreme oversold)
      - MACD histogram: massive downward spike (fake death cross)
      - Bollinger Bands: explode to 5× normal width for weeks
      - ATR becomes enormous, making all subsequent stop-losses comically wide
      - The ML model trains on thousands of these phantom "crashes" per stock
        and learns absolutely nothing useful → AUC ~0.48 (worse than coin flip)

    DUAL-TRACK FIX:
    ──────────────────────────────────────────────────────────────────────────
    Two separate downloads, kept as separate columns:

    ADJUSTED columns  (Open, High, Low, Close, Volume):
      auto_adjust=True — yfinance back-adjusts all history for every split/bonus
      These are smooth, continuous series with no jump discontinuities
      Used by: ALL indicators (RSI, MACD, EMA, ATR, Bollinger, ADX, etc.)

    UNADJUSTED column (Close_Raw):
      auto_adjust=False — raw NSE prices exactly as traded on that date
      Used by: Bhavcopy options strike price matching ONLY
      A ₹1000 strike in bhavcopy.db maps to ₹1000 raw price, not ₹100 adjusted

    The result: indicators are computed on clean data, options silo features
    are computed on matching raw data. Both are correct in their own domain.

    Returns DataFrame with columns:
        Date, Open, High, Low, Close, Volume, Close_Raw, Ticker
    """
    cache_file = CACHE_DIR / f"{ticker}_{period}_{interval}_dualtrack.parquet"

    if use_cache and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            logger.debug(f"Cache hit: {ticker}")
            return pd.read_parquet(cache_file)

    yf_sym = _yf_symbol(ticker)
    try:
        # ── Track A: adjusted prices (for indicators) ──────────────────
        adj = yf.download(
            yf_sym,
            period=period,
            interval=interval,
            auto_adjust=True,       # smooth, continuous — correct for indicators
            progress=False,
        )
        if adj.empty:
            logger.warning(f"No data returned for {ticker} ({yf_sym})")
            return pd.DataFrame()

        if isinstance(adj.columns, pd.MultiIndex):
            adj.columns = adj.columns.get_level_values(0)
        adj = adj.reset_index()
        adj.columns = [c.strip() for c in adj.columns]
        col_map = {"Datetime": "Date", "datetime": "Date"}
        adj.rename(columns=col_map, inplace=True)
        keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"]
                if c in adj.columns]
        adj = adj[keep].dropna(subset=["Close"])
        adj["Date"] = pd.to_datetime(adj["Date"])

        # ── Track B: raw unadjusted close (for options strike matching) ─
        try:
            raw = yf.download(
                yf_sym,
                period=period,
                interval=interval,
                auto_adjust=False,     # exact historical price for strike matching
                progress=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.reset_index()
            raw.columns = [c.strip() for c in raw.columns]
            raw.rename(columns={"Datetime": "Date", "datetime": "Date"}, inplace=True)
            raw["Date"] = pd.to_datetime(raw["Date"])
            raw = raw[["Date", "Close"]].rename(columns={"Close": "Close_Raw"})
            raw = raw.dropna(subset=["Close_Raw"])
        except Exception as e:
            logger.debug(f"Raw track failed for {ticker}: {e} — using adjusted as fallback")
            raw = adj[["Date", "Close"]].rename(columns={"Close": "Close_Raw"})

        # ── Merge both tracks on Date ───────────────────────────────────
        df = adj.merge(raw, on="Date", how="left")
        df["Close_Raw"] = df["Close_Raw"].ffill().fillna(df["Close"])
        df["Ticker"] = ticker
        df = df.sort_values("Date").reset_index(drop=True)

        df.to_parquet(cache_file, index=False)
        logger.info(f"Fetched {len(df)} rows for {ticker} (dual-track: adjusted + raw)")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def fetch_universe(
    universe: str = "nifty50",
    period: str = "15y",
    max_workers: int = 8,
    survivorship_bias_fix: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for an entire universe of stocks.

    survivorship_bias_fix=True (default for training):
      Uses nifty500_unbiased universe which includes historically significant
      but now delisted stocks. This prevents the model from only learning
      from companies that survived and succeeded.

    Returns: {ticker: DataFrame}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if survivorship_bias_fix and universe in ("nifty500",):
        universe = "nifty500_unbiased"
        logger.info("Survivorship bias fix: using nifty500_unbiased universe")

    tickers = UNIVERSE_MAP.get(universe, NIFTY_50)
    logger.info(f"Fetching {len(tickers)} stocks for universe '{universe}'...")

    results = {}
    def _fetch_one(t):
        df = fetch_ohlcv(t, period=period)
        return t, df

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            ticker, df = fut.result()
            if not df.empty and len(df) >= 200:
                results[ticker] = df
            elif not df.empty:
                logger.warning(f"  {ticker}: only {len(df)} days — skipped (need ≥200)")
            # Fully delisted with no data — silently skipped (empty DataFrame)

    logger.info(f"Successfully fetched {len(results)}/{len(tickers)} stocks")
    return results


def fetch_nse_filings(ticker: str) -> list[dict]:
    url = f"https://www.nseindia.com/api/corp-info?symbol={ticker}&subject=financialResults"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com",
    }
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        resp = session.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", [])[:10]
    except Exception as e:
        logger.warning(f"NSE filings failed for {ticker}: {e}")
    return []


def get_market_cap_category(ticker: str) -> str:
    large = set(NIFTY_50)
    mid   = set(NIFTY_500_EXTRA[:150])
    if ticker in large:
        return "large"
    elif ticker in mid:
        return "mid"
    return "small"


if __name__ == "__main__":
    df = fetch_ohlcv("RELIANCE", period="5y")
    print(df.tail(5))
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['Date'].min()} → {df['Date'].max()}")
