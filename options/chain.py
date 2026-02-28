"""
options/chain.py
─────────────────
Live NIFTY / BANKNIFTY options chain from NSE public API.

Features:
  - Real-time option chain (CE/PE at every strike)
  - IV (Implied Volatility) per strike
  - Greeks: Delta, Gamma, Theta, Vega (Black-Scholes)
  - OI (Open Interest) analysis — OI buildup / unwinding signals
  - PCR (Put-Call Ratio) — market sentiment indicator
  - Max Pain calculation — strike where option writers lose least
  - Support / Resistance from OI clustering
  - Best strike selector for buying/selling based on ML signal
"""

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, date
from zoneinfo import ZoneInfo
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional
import time

IST = ZoneInfo("Asia/Kolkata")

NSE_OPTION_URL    = "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
NSE_EXPIRY_URL    = "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
RISK_FREE_RATE    = 0.065   # RBI repo rate approximation


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class Greeks:
    delta:  float = 0.0
    gamma:  float = 0.0
    theta:  float = 0.0
    vega:   float = 0.0
    rho:    float = 0.0


@dataclass
class OptionStrike:
    strike:        float
    expiry:        str
    ce_ltp:        float = 0.0
    pe_ltp:        float = 0.0
    ce_iv:         float = 0.0
    pe_iv:         float = 0.0
    ce_oi:         int   = 0
    pe_oi:         int   = 0
    ce_oi_change:  int   = 0
    pe_oi_change:  int   = 0
    ce_volume:     int   = 0
    pe_volume:     int   = 0
    ce_greeks:     Greeks = field(default_factory=Greeks)
    pe_greeks:     Greeks = field(default_factory=Greeks)
    pcr:           float = 0.0   # PE OI / CE OI at this strike


@dataclass
class OptionChainSummary:
    symbol:           str
    spot_price:       float
    expiry:           str
    timestamp:        str
    atm_strike:       float
    pcr:              float        # overall PCR
    max_pain:         float        # max pain strike
    iv_rank:          float        # current IV vs 52-week range (0-100)
    call_oi_total:    int
    put_oi_total:     int
    support_levels:   list[float]  # strikes with highest PE OI buildup
    resistance_levels: list[float] # strikes with highest CE OI buildup
    strikes:          list[OptionStrike] = field(default_factory=list)
    sentiment:        str = "Neutral"   # Bullish / Bearish / Neutral


# ── Black-Scholes pricing and Greeks ──────────────────────────────────

def black_scholes(S, K, T, r, sigma, option_type="CE"):
    """
    Standard Black-Scholes option pricing.
    S: spot price
    K: strike price
    T: time to expiry in years
    r: risk-free rate
    sigma: implied volatility (decimal)
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(0, S - K) if option_type == "CE" else max(0, K - S)
        return intrinsic

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "CE":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return max(0, price)


def compute_greeks(S, K, T, r, sigma, option_type="CE") -> Greeks:
    """Compute option Greeks analytically."""
    if T <= 0 or sigma <= 0:
        return Greeks()

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == "CE":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma (same for CE and PE)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta (per calendar day)
    theta_common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "CE":
        theta = (theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega (per 1% change in IV)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Rho (per 1% change in rate)
    if option_type == "CE":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return Greeks(
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        theta=round(theta, 2),
        vega=round(vega, 2),
        rho=round(rho, 4),
    )


def implied_volatility(market_price, S, K, T, r, option_type="CE") -> float:
    """
    Calculate implied volatility using Brent's method (fast, reliable).
    Returns IV as decimal (0.20 = 20% IV).
    """
    if market_price <= 0 or T <= 0:
        return 0.0

    intrinsic = max(0, S - K) if option_type == "CE" else max(0, K - S)
    if market_price < intrinsic:
        return 0.0

    try:
        iv = brentq(
            lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - market_price,
            1e-6, 10.0,   # search between 0.0001% and 1000% IV
            xtol=1e-6,
            maxiter=100,
        )
        return round(iv, 4)
    except (ValueError, RuntimeError):
        return 0.0


# ── Options chain fetcher ──────────────────────────────────────────────

class OptionsChain:
    """
    Fetches and analyses live NSE options chain for NIFTY/BANKNIFTY.
    No broker API needed — uses NSE public endpoints.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept":     "application/json",
            "Referer":    "https://www.nseindia.com",
        })
        self._session_refreshed = False

    def _refresh_session(self):
        try:
            self.session.get("https://www.nseindia.com", timeout=5)
            self._session_refreshed = True
        except Exception as e:
            logger.warning(f"Session refresh failed: {e}")

    def _fetch(self, url: str) -> Optional[dict]:
        if not self._session_refreshed:
            self._refresh_session()
        try:
            r = self.session.get(url, timeout=8)
            if r.status_code == 401:
                self._refresh_session()
                r = self.session.get(url, timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.debug(f"Fetch failed: {e}")
        return None

    def get_chain(
        self,
        symbol:     str = "NIFTY",    # "NIFTY" or "BANKNIFTY"
        expiry:     Optional[str] = None,   # "27-Feb-2025" or None = nearest
        num_strikes: int = 20,              # strikes above/below ATM
    ) -> Optional[OptionChainSummary]:
        """
        Fetch complete options chain and compute all analytics.

        Returns OptionChainSummary with:
          - All strikes with LTP, IV, OI, Greeks
          - PCR, Max Pain, Support/Resistance from OI
          - Overall market sentiment
        """
        url  = NSE_OPTION_URL.format(symbol=symbol)
        data = self._fetch(url)
        if not data:
            logger.warning(f"No option chain data for {symbol}")
            return None

        records   = data.get("records", {})
        spot      = float(records.get("underlyingValue", 0))
        expiries  = records.get("expiryDates", [])

        if not expiries:
            return None

        target_expiry = expiry or expiries[0]   # default = nearest weekly

        # ATM strike = round spot to nearest 50 (NIFTY) or 100 (BANKNIFTY)
        step = 100 if symbol == "BANKNIFTY" else 50
        atm  = round(spot / step) * step

        # Time to expiry in years
        try:
            exp_date = datetime.strptime(target_expiry, "%d-%b-%Y").date()
            T = max(0.001, (exp_date - date.today()).days / 365.0)
        except Exception:
            T = 7 / 365.0   # fallback: 1 week

        # Parse all strikes
        raw_strikes = records.get("data", [])
        strike_map  : dict[float, OptionStrike] = {}

        for row in raw_strikes:
            if row.get("expiryDate") != target_expiry:
                continue

            k = float(row.get("strikePrice", 0))
            if k == 0:
                continue

            ce = row.get("CE", {})
            pe = row.get("PE", {})

            ce_ltp = float(ce.get("lastPrice", 0))
            pe_ltp = float(pe.get("lastPrice", 0))

            # Compute IV
            ce_iv = implied_volatility(ce_ltp, spot, k, T, RISK_FREE_RATE, "CE") if ce_ltp > 0 else 0
            pe_iv = implied_volatility(pe_ltp, spot, k, T, RISK_FREE_RATE, "PE") if pe_ltp > 0 else 0
            avg_iv = (ce_iv + pe_iv) / 2 if (ce_iv and pe_iv) else (ce_iv or pe_iv)

            # Greeks
            ce_greeks = compute_greeks(spot, k, T, RISK_FREE_RATE, avg_iv or 0.15, "CE")
            pe_greeks = compute_greeks(spot, k, T, RISK_FREE_RATE, avg_iv or 0.15, "PE")

            ce_oi = int(ce.get("openInterest", 0))
            pe_oi = int(pe.get("openInterest", 0))

            strike_map[k] = OptionStrike(
                strike       = k,
                expiry       = target_expiry,
                ce_ltp       = ce_ltp,
                pe_ltp       = pe_ltp,
                ce_iv        = round(ce_iv * 100, 2),
                pe_iv        = round(pe_iv * 100, 2),
                ce_oi        = ce_oi,
                pe_oi        = pe_oi,
                ce_oi_change = int(ce.get("changeinOpenInterest", 0)),
                pe_oi_change = int(pe.get("changeinOpenInterest", 0)),
                ce_volume    = int(ce.get("totalTradedVolume", 0)),
                pe_volume    = int(pe.get("totalTradedVolume", 0)),
                ce_greeks    = ce_greeks,
                pe_greeks    = pe_greeks,
                pcr          = round(pe_oi / max(1, ce_oi), 3),
            )

        # Filter to num_strikes above/below ATM
        all_strikes   = sorted(strike_map.keys())
        atm_idx       = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - atm))
        lo            = max(0, atm_idx - num_strikes)
        hi            = min(len(all_strikes), atm_idx + num_strikes + 1)
        selected      = [strike_map[s] for s in all_strikes[lo:hi]]

        # ── Analytics ────────────────────────────────────────────────
        total_ce_oi   = sum(s.ce_oi for s in selected)
        total_pe_oi   = sum(s.pe_oi for s in selected)
        overall_pcr   = round(total_pe_oi / max(1, total_ce_oi), 3)
        max_pain      = self._compute_max_pain(strike_map, spot)
        support       = self._oi_support(selected, "PE", n=3)
        resistance    = self._oi_resistance(selected, "CE", n=3)

        # Sentiment from PCR
        if overall_pcr > 1.3:
            sentiment = "Strongly Bullish"
        elif overall_pcr > 1.0:
            sentiment = "Bullish"
        elif overall_pcr < 0.7:
            sentiment = "Strongly Bearish"
        elif overall_pcr < 1.0:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

        # IV Rank (approximate — would need historical IV for accurate rank)
        atm_strike_obj = strike_map.get(atm)
        atm_iv = atm_strike_obj.ce_iv if atm_strike_obj else 15.0
        iv_rank = min(100, max(0, (atm_iv - 10) / 40 * 100))  # rough: 10-50% IV range

        return OptionChainSummary(
            symbol            = symbol,
            spot_price        = spot,
            expiry            = target_expiry,
            timestamp         = datetime.now(IST).isoformat(),
            atm_strike        = atm,
            pcr               = overall_pcr,
            max_pain          = max_pain,
            iv_rank           = round(iv_rank, 1),
            call_oi_total     = total_ce_oi,
            put_oi_total      = total_pe_oi,
            support_levels    = support,
            resistance_levels = resistance,
            strikes           = selected,
            sentiment         = sentiment,
        )

    def get_best_strikes(
        self,
        chain:       OptionChainSummary,
        signal:      str,        # "BUY" / "SELL" / "STRONG BUY" etc from ML
        strategy:    str = "buy_option",  # "buy_option" or "sell_premium"
        budget:      float = 50000,       # max premium to spend per leg
    ) -> dict:
        """
        Given an ML signal and options chain, recommend the best strike(s).

        buy_option:    Buy CE on BUY signal, PE on SELL signal
                       → look for ATM or slightly OTM, decent delta, affordable premium
        sell_premium:  Sell far OTM options (theta decay play)
                       → collect premium, delta < 0.25, high IV
        """
        spot   = chain.spot_price
        atm    = chain.atm_strike
        step   = 100 if chain.symbol == "BANKNIFTY" else 50

        if "BUY" in signal:
            # Look for CE options — slightly OTM for leverage
            target_strikes  = [atm, atm + step, atm + 2*step]
            option_type     = "CE"
            greek_attr      = "ce_greeks"
            ltp_attr        = "ce_ltp"
            iv_attr         = "ce_iv"
        else:
            # Look for PE options
            target_strikes  = [atm, atm - step, atm - 2*step]
            option_type     = "PE"
            greek_attr      = "pe_greeks"
            ltp_attr        = "pe_ltp"
            iv_attr         = "pe_iv"

        candidates = []
        for s in chain.strikes:
            if s.strike not in target_strikes:
                continue
            ltp    = getattr(s, ltp_attr)
            greeks = getattr(s, greek_attr)
            iv     = getattr(s, iv_attr)
            if ltp <= 0:
                continue

            lot_size  = 50 if chain.symbol == "NIFTY" else 15
            premium   = ltp * lot_size
            max_lots  = max(1, int(budget / premium))

            candidates.append({
                "strike":     s.strike,
                "type":       option_type,
                "ltp":        ltp,
                "iv":         iv,
                "delta":      abs(greeks.delta),
                "theta":      greeks.theta,
                "gamma":      greeks.gamma,
                "vega":       greeks.vega,
                "oi":         s.ce_oi if option_type == "CE" else s.pe_oi,
                "volume":     s.ce_volume if option_type == "CE" else s.pe_volume,
                "lot_size":   lot_size,
                "per_lot":    round(premium, 0),
                "max_lots":   max_lots,
                "expiry":     s.expiry,
                # Score: want good delta (0.4-0.6), high volume, affordable
                "score":      abs(greeks.delta - 0.5) * -10 + min(s.ce_volume, 10000) / 1000,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0] if candidates else None

        return {
            "signal":     signal,
            "spot":       spot,
            "expiry":     chain.expiry,
            "pcr":        chain.pcr,
            "sentiment":  chain.sentiment,
            "max_pain":   chain.max_pain,
            "recommended": best,
            "alternatives": candidates[1:3],
            "strategy_note": self._strategy_note(signal, chain, best),
        }

    def _strategy_note(self, signal, chain, best) -> str:
        """Generate a human-readable trade rationale."""
        if not best:
            return "No suitable strike found."
        pcr_note = f"PCR {chain.pcr:.2f} ({chain.sentiment})"
        mp_note  = f"Max Pain ₹{chain.max_pain:,.0f}"
        iv_note  = f"IV {best['iv']:.1f}% (Rank {chain.iv_rank:.0f})"
        delta_note = f"Delta {best['delta']:.2f}"
        return (f"{signal} signal. {pcr_note}. {mp_note}. "
                f"Recommended: {best['type']} {best['strike']:.0f} @ ₹{best['ltp']:.0f}. "
                f"{delta_note}. {iv_note}. "
                f"Per lot: ₹{best['per_lot']:,.0f}")

    def _compute_max_pain(self, strike_map: dict, spot: float) -> float:
        """
        Max Pain = strike where total option writer losses are minimised.
        At expiry, this is the price that makes the most options expire worthless.
        """
        strikes = sorted(strike_map.keys())
        min_pain = float("inf")
        max_pain_strike = spot

        for exp_price in strikes:
            total_pain = 0
            for k, opt in strike_map.items():
                # CE writers lose when exp_price > strike
                ce_loss = max(0, exp_price - k) * opt.ce_oi
                # PE writers lose when exp_price < strike
                pe_loss = max(0, k - exp_price) * opt.pe_oi
                total_pain += ce_loss + pe_loss

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = exp_price

        return max_pain_strike

    def _oi_support(self, strikes: list[OptionStrike], option: str, n: int = 3) -> list[float]:
        """Strikes with highest PE OI — act as support levels."""
        sorted_s = sorted(strikes, key=lambda s: s.pe_oi, reverse=True)
        return [s.strike for s in sorted_s[:n]]

    def _oi_resistance(self, strikes: list[OptionStrike], option: str, n: int = 3) -> list[float]:
        """Strikes with highest CE OI — act as resistance levels."""
        sorted_s = sorted(strikes, key=lambda s: s.ce_oi, reverse=True)
        return [s.strike for s in sorted_s[:n]]

    def get_pcr_trend(self, symbol: str = "NIFTY", samples: int = 10) -> dict:
        """
        Return PCR trend from the last `samples` stored readings — INSTANT.

        OLD (broken):
          for i in range(10): chain = self.get_chain(); time.sleep(60)
          → Froze the calling thread for 10 MINUTES. Any ML inference or
            FastAPI endpoint that called this was dead until 9:40 AM.

        FIX:
          A background PCRWriterThread (started at server/terminal boot) fetches
          the chain once per minute and writes PCR+timestamp into a SQLite ring
          buffer (options_pcr.db).  This method queries that ring buffer and
          returns the last `samples` rows instantly — zero sleep, zero blocking.

          If the ring buffer is empty (first boot, market closed), falls back to
          a single live fetch so callers always get something useful.
        """
        import sqlite3, os
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "options_pcr.db")
        db_path = os.path.normpath(db_path)

        # ── Query ring buffer ────────────────────────────────────────────
        try:
            con = sqlite3.connect(db_path, timeout=3)
            con.execute("PRAGMA journal_mode=WAL")
            rows = con.execute(
                """
                SELECT pcr, iv_rank, fetched_at FROM pcr_history
                WHERE symbol = ?
                ORDER BY fetched_at DESC LIMIT ?
                """,
                (symbol, samples),
            ).fetchall()
            con.close()

            if rows and len(rows) >= 2:
                pcrs = [r[0] for r in rows]          # newest first
                pcrs_chrono = list(reversed(pcrs))   # oldest first for slope
                slope = (pcrs_chrono[-1] - pcrs_chrono[0]) / len(pcrs_chrono)
                return {
                    "trend":     "rising" if slope > 0.02 else "falling" if slope < -0.02 else "flat",
                    "current":   pcrs[0],     # most recent
                    "start":     pcrs[-1],    # oldest in window
                    "slope":     round(slope, 4),
                    "pcrs":      pcrs_chrono,
                    "signal":    "Bearish" if slope > 0.02 else "Bullish" if slope < -0.02 else "Neutral",
                    "source":    "ring_buffer",
                    "n_samples": len(rows),
                }
        except Exception as e:
            logger.debug(f"PCR ring buffer read failed: {e}")

        # ── Fallback: single live fetch (no sleep) ───────────────────────
        chain = self.get_chain(symbol)
        if chain:
            return {
                "trend": "unknown", "current": chain.pcr, "start": chain.pcr,
                "slope": 0.0, "pcrs": [chain.pcr], "signal": "Neutral",
                "source": "live_fallback", "n_samples": 1,
            }
        return {"trend": "unknown", "pcrs": [], "source": "no_data", "n_samples": 0}


class PCRWriterThread:
    """
    Background thread that fetches live PCR every `interval` seconds
    and writes it to a local SQLite ring buffer (data/options_pcr.db).

    Start this once at server/terminal boot:
        PCRWriterThread(symbols=["NIFTY","BANKNIFTY"], interval=60).start()

    get_pcr_trend() then reads from this DB instantly — zero blocking.
    Ring buffer is capped at 1440 rows per symbol (24h × 60min).
    """

    def __init__(self, symbols: list = None, interval: int = 60):
        import threading, os
        self.symbols  = symbols or ["NIFTY", "BANKNIFTY"]
        self.interval = interval
        self._stop    = threading.Event()
        self.db_path  = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "options_pcr.db")
        )
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="PCRWriter"
        )

    def start(self):
        self._ensure_schema()
        self._thread.start()
        logger.info(f"PCRWriterThread started (interval={self.interval}s)")

    def stop(self):
        self._stop.set()

    def _ensure_schema(self):
        import sqlite3
        try:
            con = sqlite3.connect(self.db_path, timeout=5)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("""
                CREATE TABLE IF NOT EXISTS pcr_history (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol     TEXT NOT NULL,
                    pcr        REAL NOT NULL,
                    iv_rank    REAL NOT NULL DEFAULT 50.0,
                    spot_price REAL NOT NULL DEFAULT 0.0,
                    fetched_at TEXT NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS ix_sym_ts ON pcr_history (symbol, fetched_at DESC)")
            con.commit()
            con.close()
        except Exception as e:
            logger.warning(f"PCR DB schema: {e}")

    def _run(self):
        import sqlite3
        from datetime import datetime
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
        oc  = OptionsChain()

        while not self._stop.is_set():
            for symbol in self.symbols:
                try:
                    chain = oc.get_chain(symbol, num_strikes=3)
                    if chain:
                        con = sqlite3.connect(self.db_path, timeout=5)
                        con.execute("PRAGMA journal_mode=WAL")
                        con.execute(
                            "INSERT INTO pcr_history (symbol, pcr, iv_rank, spot_price, fetched_at) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (symbol, chain.pcr, chain.iv_rank, chain.spot_price,
                             datetime.now(IST).isoformat()),
                        )
                        # Keep only last 1440 rows (24h) per symbol
                        con.execute(
                            """
                            DELETE FROM pcr_history
                            WHERE symbol = ? AND id NOT IN (
                                SELECT id FROM pcr_history WHERE symbol = ?
                                ORDER BY fetched_at DESC LIMIT 1440
                            )
                            """,
                            (symbol, symbol),
                        )
                        con.commit()
                        con.close()
                except Exception as e:
                    logger.debug(f"PCRWriter {symbol}: {e}")

            self._stop.wait(timeout=self.interval)
