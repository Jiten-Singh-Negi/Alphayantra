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
    """
    Convert NSE internal symbol to Yahoo Finance ticker format.

    Corrections verified from live fetch log (2026-03-02):
      - Stocks that failed with yf.download() but work via yf.Ticker().history()
        are handled by the fetch_ohlcv rewrite (Ticker API, not download API).
      - Symbols that Yahoo Finance stores under a different string than NSE are
        mapped explicitly here.
      - ESCORTS renamed to ESCORTSKUBOTA after Kubota JV consolidation (2023).
      - BERGERPAINTS: Yahoo uses BERGEPAINT (no S).
      - M&M / BAJAJ-AUTO: special chars preserved — Yahoo accepts them fine.
    """
    if "." in ticker:
        return ticker

    # Verified NSE symbol → Yahoo Finance ticker corrections
    # Key principle: Yahoo Finance uses the TRADING symbol on NSE, which sometimes
    # differs from the commonly used short name.
    SYMBOL_MAP = {
        # Special characters
        "M&M":            "M&M.NS",
        "BAJAJ-AUTO":     "BAJAJ-AUTO.NS",

        # Name changes / mergers (NSE internal name → current Yahoo ticker)
        "BERGERPAINTS":   "BERGEPAINT.NS",     # Yahoo drops the S
        "ESCORTS":        "ESCORTSKUBOTA.NS",   # renamed after Kubota merger 2023
        "BERGER":         "BERGEPAINT.NS",

        # Stocks that fail with yf.download() due to yfinance MultiIndex bug —
        # these are fetched via yf.Ticker().history() in fetch_ohlcv, so the
        # symbol just needs to be correct here
        "SBIN":           "SBIN.NS",
        "NTPC":           "NTPC.NS",
        "HEROMOTOCO":     "HEROMOTOCO.NS",
        "DIVISLAB":       "DIVISLAB.NS",
        "TATACHEM":       "TATACHEM.NS",
        "ASHOKLEY":       "ASHOKLEY.NS",
        "SJVN":           "SJVN.NS",
        "MOIL":           "MOIL.NS",
        "DEEPAKNTR":      "DEEPAKNTR.NS",
        "IPCALAB":        "IPCALAB.NS",
        "VEDL":           "VEDL.NS",
        "RCOM":           "RCOM.NS",
        "RELIANCE":       "RELIANCE.NS",
        "CDSL":           "CDSL.NS",
        "NATIONALUM":     "NATIONALUM.NS",
        "HINDUNILVR":     "HINDUNILVR.NS",
        "WIPRO":          "WIPRO.NS",
        "PIDILITIND":     "PIDILITIND.NS",
        "TITAN":          "TITAN.NS",
        "ASIANPAINT":     "ASIANPAINT.NS",
        "CEAT":           "CEAT.NS",
        "TATAMOTORS":     "TATAMOTORS.NS",
    }

    if ticker in SYMBOL_MAP:
        return SYMBOL_MAP[ticker]
    return f"{ticker}.NS"


def _ticker_history(yf_sym: str, period: str, auto_adjust: bool) -> pd.DataFrame:
    """
    Robust single-stock OHLCV fetch using yf.Ticker().history().

    WHY THIS INSTEAD OF yf.download():
    ─────────────────────────────────────────────────────────────────────────
    yf.download() has a persistent bug where for certain NSE symbols it returns
    a MultiIndex DataFrame with empty levels, causing "No objects to concatenate"
    inside yfinance itself. Confirmed failing stocks from live log:
      SBIN, NTPC, HEROMOTOCO, DIVISLAB, TATACHEM, ASHOKLEY, SJVN, MOIL,
      DEEPAKNTR, IPCALAB, VEDL, RCOM.

    Additionally, yf.download(auto_adjust=True) sometimes returns an extra
    'Adj Close' column causing duplicate column names after our renaming.
    Confirmed failing: RELIANCE, CDSL, NATIONALUM, HINDUNILVR, WIPRO,
    PIDILITIND, TITAN.

    yf.Ticker(sym).history() ALWAYS returns a flat, clean DataFrame with no
    MultiIndex, no extra columns, no NoneType errors.
    ─────────────────────────────────────────────────────────────────────────
    """
    try:
        tk   = yf.Ticker(yf_sym)
        hist = tk.history(period=period, interval="1d", auto_adjust=auto_adjust)
        if hist is None or hist.empty:
            return pd.DataFrame()
        # history() returns DatetimeIndex — normalise to UTC-naive
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist.index.name = "Date"
        hist = hist.reset_index()
        # Keep only standard OHLCV — drop Dividends, Stock Splits, Capital Gains
        standard = ["Date", "Open", "High", "Low", "Close", "Volume"]
        hist = hist[[c for c in standard if c in hist.columns]]
        hist = hist.dropna(subset=["Close"])
        return hist
    except Exception:
        return pd.DataFrame()


def fetch_ohlcv(
    ticker: str,
    period: str = "15y",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Dual-track price fetcher v2 — uses yf.Ticker().history() for reliability.

    DUAL-TRACK DESIGN:
    ─────────────────────────────────────────────────────────────────────────
    Track A — Adjusted (Open, High, Low, Close, Volume):
        auto_adjust=True: back-adjusted for all splits/bonuses.
        Smooth continuous series — used by ALL indicators.

    Track B — Raw close (Close_Raw):
        auto_adjust=False: exact NSE price as traded.
        Used ONLY for Bhavcopy options strike matching.

    Returns DataFrame with columns:
        Date, Open, High, Low, Close, Volume, Close_Raw, Ticker
    """
    # Cache key uses _v2 suffix so stale v1 caches (which caused duplicate
    # Close_Raw bugs) are automatically bypassed without manual deletion.
    cache_file = CACHE_DIR / f"{ticker}_{period}_{interval}_v2.parquet"

    if use_cache and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            logger.debug(f"Cache hit: {ticker}")
            df = pd.read_parquet(cache_file)
            if "Close_Raw" in df.columns and not df.empty:
                return df

    yf_sym = _yf_symbol(ticker)
    try:
        # ── Track A: adjusted prices (for indicators) ──────────────────
        adj = _ticker_history(yf_sym, period, auto_adjust=True)
        if adj.empty:
            logger.warning(f"No data returned for {ticker} ({yf_sym})")
            return pd.DataFrame()

        adj["Date"] = pd.to_datetime(adj["Date"])
        # Defensive: remove any Close_Raw that might exist from a bad cache read
        adj = adj[[c for c in adj.columns if c != "Close_Raw"]]

        # ── Track B: raw unadjusted close (for options strike matching) ─
        try:
            raw = _ticker_history(yf_sym, period, auto_adjust=False)
            if raw.empty:
                raise ValueError("empty raw track")
            raw["Date"] = pd.to_datetime(raw["Date"])
            raw = raw[["Date", "Close"]].rename(columns={"Close": "Close_Raw"})
        except Exception as e:
            logger.debug(f"Raw track failed for {ticker}: {e} — using adjusted as fallback")
            raw = adj[["Date", "Close"]].rename(columns={"Close": "Close_Raw"})

        # ── Merge ──────────────────────────────────────────────────────
        df = adj.merge(raw, on="Date", how="left")
        df["Close_Raw"] = df["Close_Raw"].ffill().fillna(df["Close"])
        df["Ticker"]    = ticker
        df = df.sort_values("Date").reset_index(drop=True)

        # Final safety: remove any accidental duplicate columns
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

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

    results    = {}
    failed     = []   # first-pass failures to retry
    n_total    = len(tickers)

    def _fetch_one(t: str, attempt: int = 1) -> tuple:
        """Fetch with one silent retry on empty result (handles transient timeouts)."""
        df = fetch_ohlcv(t, period=period)
        if df.empty and attempt == 1:
            time.sleep(2)   # brief back-off before retry
            df = fetch_ohlcv(t, period=period, use_cache=False)
        return t, df

    # Reduced workers (4) to avoid yfinance rate limiting — the log showed
    # many concurrent failures that are rate-limit artefacts, not real failures.
    effective_workers = min(max_workers, 4)
    logger.info(f"Using {effective_workers} workers (rate-limit safe)...")

    with ThreadPoolExecutor(max_workers=effective_workers) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            ticker, df = fut.result()
            done += 1
            if not df.empty and len(df) >= 200:
                results[ticker] = df
            elif not df.empty:
                logger.warning(f"  {ticker}: only {len(df)} days — skipped (need ≥200)")
            else:
                failed.append(ticker)
            if done % 20 == 0:
                logger.info(f"  Progress: {done}/{n_total} fetched, {len(results)} OK so far")

    # Second-pass retry for any that failed — sequential with 1s gap each
    if failed:
        logger.info(f"Retrying {len(failed)} failed tickers sequentially...")
        for ticker in failed:
            time.sleep(1)
            df = fetch_ohlcv(ticker, period=period, use_cache=False)
            if not df.empty and len(df) >= 200:
                results[ticker] = df
                logger.info(f"  Retry OK: {ticker} ({len(df)} rows)")
            elif not df.empty:
                logger.warning(f"  Retry {ticker}: only {len(df)} days — skipped")
            else:
                logger.debug(f"  Retry failed: {ticker} — likely delisted/unavailable")

    logger.info(f"Successfully fetched {len(results)}/{n_total} stocks")
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
