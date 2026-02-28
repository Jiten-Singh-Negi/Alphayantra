"""
data/bhavcopy.py  — NSE F&O Bhavcopy Historical Options Data
──────────────────────────────────────────────────────────────
The NSE publishes daily Bhavcopy ZIP files with every F&O contract's
Open Interest, Volume, and settlement price.  This is the only free
reliable source of historical PCR and IV for Indian markets.

What this does:
  1. Downloads daily F&O Bhavcopy ZIPs from NSE archives (2015 → today)
  2. Parses NIFTY / BANKNIFTY / FINNIFTY options per strike per day
  3. Computes daily PCR (Put OI / Call OI) per symbol
  4. Estimates ATM IV from settle prices (Black-Scholes approximation)
  5. Computes max pain strike
  6. Stores everything in a local SQLite DB (data/bhavcopy.db)
  7. Exports pd.Series for injection into ml_engine.py training

CLI usage:
    python -m data.bhavcopy --backfill          # one-time, ~2-4 hours
    python -m data.bhavcopy --update            # daily incremental
    python -m data.bhavcopy --status            # show DB stats

In-code usage:
    from data.bhavcopy import BhavcopyScraper
    scraper = BhavcopyScraper()
    scraper.update()
    pcr      = scraper.get_pcr_series("NIFTY")
    iv_rank  = scraper.get_iv_rank_series("NIFTY")
    max_pain = scraper.get_max_pain_series("NIFTY")
"""

import io, time, sqlite3, zipfile
import requests, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
from loguru import logger
from typing import Optional

DB_PATH   = Path("data/bhavcopy.db")
CACHE_DIR = Path("data/bhavcopy_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NSE_FO_URL = (
    "https://archives.nseindia.com/content/historical/DERIVATIVES/"
    "{year}/{month}/fo{date_str}bhav.csv.zip"
)
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com",
}
MONTH_MAP = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
             7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}
SYMBOLS   = {"NIFTY", "BANKNIFTY", "FINNIFTY"}


class BhavcopyScraper:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self._init_db()
        self._warm_session()

    # ── Public ──────────────────────────────────────────────────────────

    def update(self) -> int:
        """Download any missing days from last stored date to yesterday."""
        last  = self._last_stored_date()
        start = (last + timedelta(days=1)) if last else date(2015, 1, 1)
        end   = date.today() - timedelta(days=1)
        if start > end:
            logger.info("Bhavcopy already up to date")
            return 0
        return self.download_range(start.isoformat(), end.isoformat())

    def download_range(self, start_date: str, end_date: str) -> int:
        """Download all trading days in range.  Returns number of days stored."""
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        stored, d  = 0, start
        while d <= end:
            if d.weekday() < 5:  # skip weekends
                try:
                    n = self._download_one_day(d)
                    if n > 0:
                        stored += 1
                        logger.info(f"  {d}: {n} F&O rows stored")
                except Exception as e:
                    logger.debug(f"  {d}: {e}")
                time.sleep(0.25)
            d += timedelta(days=1)
        logger.info(f"Bhavcopy complete: {stored} days stored")
        return stored

    def get_pcr_series(self, symbol: str = "NIFTY") -> pd.Series:
        """Daily PCR as pd.Series indexed by date."""
        df = self._query("SELECT trade_date, pcr FROM daily_pcr WHERE symbol=? ORDER BY trade_date",
                         (symbol,))
        if df.empty:
            return pd.Series(dtype=float, name="pcr")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.set_index("trade_date")["pcr"]

    def get_iv_rank_series(self, symbol: str = "NIFTY", window: int = 252) -> pd.Series:
        """IV Rank 0-100: where is today's IV vs past `window` days."""
        df = self._query(
            "SELECT trade_date, atm_iv FROM daily_pcr WHERE symbol=? AND atm_iv>0 ORDER BY trade_date",
            (symbol,))
        if df.empty:
            return pd.Series(dtype=float, name="iv_rank")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        iv = df.set_index("trade_date")["atm_iv"]
        rank = pd.Series(index=iv.index, dtype=float, name="iv_rank")
        for i in range(len(iv)):
            win = iv.iloc[max(0, i - window): i + 1]
            lo, hi = win.min(), win.max()
            rank.iloc[i] = 100.0 * (iv.iloc[i] - lo) / (hi - lo) if hi > lo else 50.0
        return rank

    def get_max_pain_series(self, symbol: str = "NIFTY") -> pd.Series:
        """Daily max pain strike as pd.Series."""
        df = self._query("SELECT trade_date, max_pain FROM daily_pcr WHERE symbol=? ORDER BY trade_date",
                         (symbol,))
        if df.empty:
            return pd.Series(dtype=float, name="max_pain")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.set_index("trade_date")["max_pain"]

    def status(self) -> dict:
        try:
            df = self._query("SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT trade_date) FROM daily_pcr")
            row = df.iloc[0]
            return {"earliest": row.iloc[0], "latest": row.iloc[1], "trading_days": int(row.iloc[2]),
                    "db_path": str(self.db_path)}
        except Exception:
            return {"trading_days": 0, "db_path": str(self.db_path)}

    # ── Private ─────────────────────────────────────────────────────────

    def _init_db(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS fo_oi (
                    trade_date TEXT, symbol TEXT, expiry_date TEXT,
                    strike REAL, option_type TEXT,
                    open_int INTEGER, oi_change INTEGER,
                    close_price REAL, settle_price REAL, volume INTEGER,
                    PRIMARY KEY (trade_date, symbol, expiry_date, strike, option_type)
                );
                CREATE TABLE IF NOT EXISTS daily_pcr (
                    trade_date TEXT, symbol TEXT,
                    pcr REAL, atm_iv REAL, max_pain REAL,
                    call_oi INTEGER, put_oi INTEGER,
                    PRIMARY KEY (trade_date, symbol)
                );
                CREATE INDEX IF NOT EXISTS idx_pcr_date ON daily_pcr(trade_date);
                CREATE INDEX IF NOT EXISTS idx_fo_sym   ON fo_oi(trade_date, symbol);
            """)

    def _warm_session(self):
        try: self.session.get("https://www.nseindia.com", timeout=8)
        except Exception: pass

    def _query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as c:
            return pd.read_sql(sql, c, params=params)

    def _last_stored_date(self) -> Optional[date]:
        try:
            df = self._query("SELECT MAX(trade_date) FROM daily_pcr")
            val = df.iloc[0, 0]
            if val:
                return date.fromisoformat(val)
        except Exception:
            pass
        return None

    def _download_one_day(self, d: date) -> int:
        cache = CACHE_DIR / f"fo_{d.strftime('%Y%m%d')}.csv"
        if cache.exists():
            df = pd.read_csv(cache, low_memory=False)
        else:
            df = self._fetch_csv(d)
            if df is None or df.empty:
                return 0
            df.to_csv(cache, index=False)

        df.columns = [c.strip().upper() for c in df.columns]
        # Keep only index options
        df = df[df.get("INSTRUMENT", pd.Series()).str.strip().isin(["OPTIDX"])].copy()
        df = df[df.get("SYMBOL",     pd.Series()).str.strip().isin(SYMBOLS)].copy()
        if df.empty:
            return 0

        date_str = d.isoformat()
        rows = []
        for _, r in df.iterrows():
            try:
                rows.append((
                    date_str,
                    str(r["SYMBOL"]).strip(),
                    str(r["EXPIRY_DT"]).strip(),
                    float(str(r["STRIKE_PR"]).replace(",","")),
                    str(r["OPTION_TYP"]).strip(),
                    int(float(str(r["OPEN_INT"]).replace(",",""))),
                    int(float(str(r.get("CHG_IN_OI",0) or 0))),
                    float(str(r["CLOSE"]).replace(",","") or 0),
                    float(str(r["SETTLE_PR"]).replace(",","") or 0),
                    int(float(str(r["CONTRACTS"]).replace(",","") or 0)),
                ))
            except (ValueError, KeyError, TypeError):
                continue

        if not rows:
            return 0

        with sqlite3.connect(self.db_path) as c:
            c.executemany("""
                INSERT OR REPLACE INTO fo_oi
                (trade_date,symbol,expiry_date,strike,option_type,open_int,oi_change,
                 close_price,settle_price,volume)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, rows)

        # Summarise per symbol
        for sym in SYMBOLS:
            self._store_summary(date_str, sym, df[df["SYMBOL"].str.strip() == sym])

        return len(rows)

    def _store_summary(self, date_str: str, symbol: str, sym_df: pd.DataFrame):
        if sym_df.empty:
            return
        try:
            sym_df = sym_df.copy()
            sym_df["_expiry"] = pd.to_datetime(sym_df["EXPIRY_DT"].str.strip(),
                                                format="%d-%b-%Y", errors="coerce")
            nearest = sym_df["_expiry"].min()
            sym_df  = sym_df[sym_df["_expiry"] == nearest]

            def to_float(col): return sym_df[col].astype(str).str.replace(",","").astype(float)
            def to_int(col):   return to_float(col).astype(int)

            call_mask = sym_df["OPTION_TYP"].str.strip() == "CE"
            put_mask  = sym_df["OPTION_TYP"].str.strip() == "PE"
            call_oi   = int(to_float("OPEN_INT")[call_mask].sum())
            put_oi    = int(to_float("OPEN_INT")[put_mask].sum())
            pcr       = round(put_oi / max(1, call_oi), 4)

            # ATM strike = highest combined OI
            sym_df["_strike"] = to_float("STRIKE_PR")
            sym_df["_oi"]     = to_float("OPEN_INT")
            strike_oi = sym_df.groupby("_strike")["_oi"].sum()
            atm_strike = float(strike_oi.idxmax()) if not strike_oi.empty else 0.0

            # ATM IV approximation
            atm_ce = sym_df[(sym_df["_strike"] == atm_strike) & call_mask]
            days_to_exp = max(1, (nearest.date() - date.fromisoformat(date_str)).days)
            T = days_to_exp / 365.0
            approx_iv = 0.0
            if not atm_ce.empty and atm_strike > 0:
                atm_price = float(str(atm_ce.iloc[0]["SETTLE_PR"]).replace(",","") or 0)
                if atm_price > 0:
                    approx_iv = float(np.clip((atm_price / atm_strike) / np.sqrt(T) * 100, 3, 120))

            max_pain = self._max_pain(sym_df, call_mask, put_mask)

            with sqlite3.connect(self.db_path) as c:
                c.execute("""
                    INSERT OR REPLACE INTO daily_pcr
                    (trade_date,symbol,pcr,atm_iv,max_pain,call_oi,put_oi)
                    VALUES (?,?,?,?,?,?,?)
                """, (date_str, symbol, pcr, approx_iv, max_pain, call_oi, put_oi))
        except Exception as e:
            logger.debug(f"  Summary error {symbol} {date_str}: {e}")

    def _max_pain(self, sym_df, call_mask, put_mask) -> float:
        try:
            calls = sym_df[call_mask][["_strike","_oi"]].values
            puts  = sym_df[put_mask][["_strike","_oi"]].values
            all_k = np.unique(np.concatenate([calls[:,0], puts[:,0]]))
            min_pain, mp = np.inf, all_k[0]
            for exp in all_k:
                pain  = sum(max(0.0, exp-k)*oi for k,oi in calls)
                pain += sum(max(0.0, k-exp)*oi for k,oi in puts)
                if pain < min_pain:
                    min_pain, mp = pain, exp
            return float(mp)
        except Exception:
            return 0.0

    def _fetch_csv(self, d: date) -> Optional[pd.DataFrame]:
        ds   = d.strftime("%d%b%Y").upper()
        url  = NSE_FO_URL.format(year=d.year, month=MONTH_MAP[d.month], date_str=ds)
        try:
            r = self.session.get(url, timeout=20)
            if r.status_code == 404:
                return None
            if r.status_code != 200:
                return None
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                return pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        except Exception as e:
            logger.debug(f"  Fetch error {d}: {e}")
            return None


# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--backfill",  action="store_true")
    p.add_argument("--update",    action="store_true")
    p.add_argument("--status",    action="store_true")
    p.add_argument("--from-date", default="2015-01-01")
    p.add_argument("--to-date",   default=date.today().isoformat())
    args = p.parse_args()

    s = BhavcopyScraper()
    if args.status:
        info = s.status()
        print(f"DB: {info['db_path']}  |  Days: {info['trading_days']}"
              f"  |  {info.get('earliest')} → {info.get('latest')}")
    elif args.backfill:
        print(f"Backfilling {args.from_date} → {args.to_date}  (~2-4 hours, safe to Ctrl+C)")
        s.download_range(args.from_date, args.to_date)
    elif args.update:
        n = s.update()
        print(f"Downloaded {n} new days")
    else:
        p.print_help()
