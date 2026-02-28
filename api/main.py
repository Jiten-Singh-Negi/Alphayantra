"""
api/main.py  — AlphaYantra v3  (all bugs fixed)
─────────────────────────────────────────────────
Fixes applied:
  ✅ /predict  — injects live VIX, PCR, IV Rank, FinBERT into model.predict()
  ✅ /backtest — fetches ^INDIAVIX history and passes vix_data to engine.run()
  ✅ /train    — uses correct v3 kwargs (no horizon/threshold/skip_lstm)
  ✅ TrainRequest — removed obsolete v2 fields
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import json, os
from pathlib import Path
from datetime import datetime
from loguru import logger
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher          import fetch_ohlcv, fetch_universe, UNIVERSE_MAP
from strategies.indicators import compute_indicators, compute_signal_score, IndicatorConfig
from backtest.engine       import BacktestEngine
from news.sentiment        import NewsSentimentEngine
from models.ml_engine      import AlphaYantraML
from options.chain         import OptionsChain
from feed.live_feed        import LiveFeed, is_market_open, DEFAULT_WATCHLIST
from risk.manager          import RiskManager
from dashboard.app         import router as dashboard_router

app = FastAPI(title="AlphaYantra v3", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(dashboard_router)

_ml_model        = None
_news_engine     = None
_options_chain   = None
_live_feed       = None
_risk_manager    = None
_training_status = {"status": "idle", "progress": 0, "message": ""}
_vix_cache       = {"series": None, "fetched_at": None}   # cached VIX series

STRATEGY_DIR = Path("strategies/saved")
STRATEGY_DIR.mkdir(parents=True, exist_ok=True)


# ── Singletons ─────────────────────────────────────────────────────────

def get_ml_model() -> AlphaYantraML:
    global _ml_model
    if _ml_model is None:
        _ml_model = AlphaYantraML()
        try:
            _ml_model.load()
        except FileNotFoundError:
            logger.info("No saved model — train first via POST /train")
    return _ml_model

def get_news_engine() -> NewsSentimentEngine:
    global _news_engine
    if _news_engine is None:
        _news_engine = NewsSentimentEngine(use_finbert=True)
    return _news_engine

def get_options_chain() -> OptionsChain:
    global _options_chain
    if _options_chain is None:
        _options_chain = OptionsChain()
    return _options_chain

def get_live_feed() -> LiveFeed:
    global _live_feed
    if _live_feed is None:
        _live_feed = LiveFeed(watchlist=DEFAULT_WATCHLIST, poll_interval=5)
        _live_feed.start()
    return _live_feed

def get_risk_manager() -> RiskManager:
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager

def get_india_vix_series() -> Optional[pd.Series]:
    """
    Returns a pd.Series of India VIX indexed by date.
    Caches in memory; refreshes once per day.
    """
    global _vix_cache
    now = datetime.now()
    if (_vix_cache["series"] is not None and _vix_cache["fetched_at"] is not None
            and (now - _vix_cache["fetched_at"]).seconds < 86400):
        return _vix_cache["series"]
    try:
        import yfinance as yf
        raw = yf.download("^INDIAVIX", period="20y", progress=False, auto_adjust=True)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        s = raw["Close"].squeeze()
        s.index = pd.to_datetime(s.index)
        _vix_cache["series"]     = s
        _vix_cache["fetched_at"] = now
        logger.info(f"India VIX loaded: {len(s)} days")
        return s
    except Exception as e:
        logger.warning(f"India VIX fetch failed: {e}")
        return None

def get_live_vix() -> float:
    """Return today's India VIX level (scalar)."""
    try:
        feed = get_live_feed()
        quote = feed.latest_quotes.get("INDIA VIX") or feed.latest_quotes.get("INDIAVIX")
        if quote:
            return float(quote.get("ltp", 15.0))
    except Exception:
        pass
    # Fallback: last value from history
    s = get_india_vix_series()
    if s is not None and not s.empty:
        return float(s.iloc[-1])
    return 15.0

def get_live_adx(ticker: str = "NIFTY") -> float:
    """Return latest ADX for NIFTY index."""
    try:
        df = fetch_ohlcv(ticker, period="1y")
        if df.empty:
            return 20.0
        cfg = IndicatorConfig()
        df  = compute_indicators(df, cfg)
        if "adx" in df.columns:
            return float(df["adx"].iloc[-1])
    except Exception:
        pass
    return 20.0


# ── Schemas ─────────────────────────────────────────────────────────────

class IndicatorWeights(BaseModel):
    rsi:float=0.25; macd:float=0.20; bb:float=0.15
    volume:float=0.15; fib:float=0.10; ema:float=0.15; news:float=0.20

class StrategyConfig(BaseModel):
    name:str="My Strategy"; use_rsi:bool=True; use_macd:bool=True; use_bb:bool=True
    use_ema:bool=True; use_volume:bool=True; use_fib:bool=True
    use_adx:bool=False; use_supertrend:bool=False; use_ichimoku:bool=False
    rsi_period:int=14; rsi_oversold:int=30; rsi_overbought:int=70
    macd_fast:int=12; macd_slow:int=26; macd_signal:int=9
    bb_period:int=20; bb_std:float=2.0; ema_fast:int=9; ema_slow:int=21
    atr_period:int=14; atr_sl_mult:float=1.5
    fib_levels:List[float]=[0.236,0.382,0.618,0.786]; fib_tolerance:float=0.015
    vol_spike_mult:float=1.5; min_confirmations:int=3
    weights:IndicatorWeights=Field(default_factory=IndicatorWeights)
    news_weight:float=0.20; news_sources:List[str]=[]; news_hours:int=24

class PredictRequest(BaseModel):
    strategy:StrategyConfig; tickers:List[str]=[]; universe:str="nifty50"; top_n:int=20

class BacktestRequest(BaseModel):
    strategy:StrategyConfig; universe:str="nifty500"; custom_tickers:List[str]=[]
    start_date:str="2010-01-01"; end_date:str="2024-12-31"
    initial_capital:float=1_000_000; brokerage_model:str="zerodha_eq_delivery"
    position_size_pct:float=0.05; max_positions:int=10

class TrainRequest(BaseModel):
    # v3 — no obsolete horizon/threshold/skip_lstm fields
    universe:       str  = "nifty500"
    period:         str  = "15y"
    n_cv_folds:     int  = 4
    skip_tcn:       bool = False
    tcn_max_samples:int  = 50_000
    tcn_epochs:     int  = 10
    use_triple_barrier: bool = True


def _strategy_to_cfg(s: StrategyConfig) -> IndicatorConfig:
    return IndicatorConfig(
        use_rsi=s.use_rsi, use_macd=s.use_macd, use_bb=s.use_bb,
        use_ema_fast=s.use_ema, use_ema_slow=s.use_ema,
        use_vol_spike=s.use_volume, use_fib=s.use_fib,
        use_adx=s.use_adx, use_supertrend=s.use_supertrend,
        use_ichimoku=s.use_ichimoku,
        rsi_period=s.rsi_period, rsi_oversold=s.rsi_oversold,
        rsi_overbought=s.rsi_overbought,
        macd_fast=s.macd_fast, macd_slow=s.macd_slow, macd_signal=s.macd_signal,
        bb_period=s.bb_period, bb_std=s.bb_std,
        ema_fast=s.ema_fast, ema_slow=s.ema_slow,
        atr_period=s.atr_period, atr_sl_mult=s.atr_sl_mult,
        fib_levels=s.fib_levels, fib_tolerance=s.fib_tolerance,
        vol_spike_mult=s.vol_spike_mult, min_confirmations=s.min_confirmations,
        weight_rsi=s.weights.rsi, weight_macd=s.weights.macd,
        weight_bb=s.weights.bb, weight_volume=s.weights.volume,
        weight_fib=s.weights.fib, weight_ema=s.weights.ema,
        weight_news=s.news_weight,
    )


# ── Core endpoints ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    m = get_ml_model()
    f = get_live_feed()
    r = get_risk_manager()
    return {
        "status":         "ok",
        "timestamp":      datetime.now().isoformat(),
        "version":        "3.0.0",
        "market_open":    is_market_open(),
        "model_trained":  m.trained,
        "model_metrics":  m.metrics if m.trained else None,
        "cv_folds":       len(m.cv_metrics),
        "feed_running":   f._running,
        "feed_symbols":   len(f.latest_quotes),
        "risk_status":    r.get_status(),
        "india_vix":      get_live_vix(),
    }


@app.get("/indicators/{ticker}")
def get_indicators(ticker: str, period: str = "1y"):
    """
    ASGI FIX: plain `def` — FastAPI auto-routes to threadpool.
    fetch_ohlcv() + compute_indicators() are synchronous CPU/IO.
    Using `async def` here blocks the ASGI event loop for 1–2s
    while yfinance downloads data, freezing every other endpoint.
    """
    df = fetch_ohlcv(ticker, period=period)
    if df.empty:
        raise HTTPException(404, f"No data for {ticker}")
    cfg = IndicatorConfig()
    df  = compute_indicators(df, cfg)
    sig = compute_signal_score(df.iloc[-1], cfg)
    return {"ticker": ticker, "close": round(float(df.iloc[-1]["Close"]), 2), "signal": sig}


@app.get("/news/{ticker}")
def get_news(ticker: str, hours: int = 24):
    """ASGI FIX: plain `def` — FinBERT inference is synchronous CPU work."""
    return {"ticker": ticker, **get_news_engine().get_stock_sentiment(ticker=ticker, hours_window=hours)}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    ASGI FIX: plain `def`.
    This endpoint loops over up to 200 stocks, each requiring a yfinance
    download (~1s I/O each) + model inference (~0.2s CPU each).
    Running it as `async def` would occupy the ASGI event loop for minutes,
    dropping all other requests. FastAPI's threadpool handles it correctly
    when defined as a plain synchronous function.
    """
    """
    BUG FIX: now injects live VIX, ADX, PCR, IV Rank and FinBERT sentiment
    into model.predict() instead of leaving all silo features at neutral defaults.
    """
    model   = get_ml_model()
    tickers = req.tickers or UNIVERSE_MAP.get(req.universe, [])
    if not tickers:
        raise HTTPException(400, "No tickers specified")

    cfg          = _strategy_to_cfg(req.strategy)
    news_engine  = get_news_engine()
    predictions  = []

    # Fetch live context once (shared across all tickers in this scan)
    vix       = get_live_vix()
    adx       = get_live_adx("NIFTY")
    vix_series = get_india_vix_series()

    # Fetch live options chain metrics once
    pcr_today = iv_rank_today = None
    try:
        oc    = get_options_chain()
        chain = oc.get_chain("NIFTY", num_strikes=5)
        if chain:
            pcr_today     = chain.pcr
            iv_rank_today = chain.iv_rank
    except Exception as e:
        logger.debug(f"Options chain for predict: {e}")

    for ticker in tickers[:200]:
        try:
            df = fetch_ohlcv(ticker, period="2y")
            if df.empty or len(df) < 50:
                continue

            df   = compute_indicators(df, cfg)
            row  = df.iloc[-1]
            tech = compute_signal_score(row, cfg)

            # ── FinBERT sentiment for this ticker ──────────────────────
            news = news_engine.get_stock_sentiment(ticker, hours_window=req.strategy.news_hours)

            # Build single-day series for the silo injections
            # (model.predict uses these to populate silo feature columns)
            today = pd.Timestamp.today().normalize()
            pcr_s  = pd.Series({today: pcr_today or 1.0})
            ivr_s  = pd.Series({today: iv_rank_today or 50.0})
            sent_s = pd.Series({today: news["score"]})

            if model.trained:
                ml = model.predict(
                    df_with_indicators = df,
                    vix                = vix,
                    adx                = adx,
                    pcr_series         = pcr_s,
                    iv_rank_series     = ivr_s,
                    sentiment_series   = sent_s,
                    vix_series         = vix_series,
                )
            else:
                ml = {"signal": "HOLD", "confidence": 50, "probability": 0.5,
                      "expected_return": 0.0, "kelly_fraction": 0.01,
                      "regime": "normal", "expected_sharpe": 0.0}

            combined = (1 - req.strategy.news_weight) * tech["score"] + req.strategy.news_weight * news["score"]

            predictions.append({
                "ticker":           ticker,
                "close":            round(float(row["Close"]), 2),
                "signal":           ml["signal"],
                "ml_confidence":    ml["confidence"],
                "tech_score":       tech["score"],
                "news_score":       news["score"],
                "news_label":       news["label"],
                "combined_score":   round(combined, 1),
                "stop_loss":        tech["stop_loss"],
                "take_profit":      tech["take_profit"],
                "rsi":              tech.get("rsi"),
                "confirmations":    tech["confirmations"],
                "expected_return":  ml.get("expected_return", 0.0),
                "expected_sharpe":  ml.get("expected_sharpe", 0.0),
                "kelly_fraction":   ml.get("kelly_fraction", 0.01),
                "regime":           ml.get("regime", "normal"),
                "ensemble_weights": ml.get("ensemble_weights", {}),
                "indicator_breakdown": tech["breakdown"],
                "vix":              vix,
                "adx":              adx,
            })
        except Exception as e:
            logger.debug(f"Predict {ticker}: {e}")

    predictions.sort(key=lambda x: x["combined_score"], reverse=True)
    return {
        "timestamp":     datetime.now().isoformat(),
        "strategy":      req.strategy.name,
        "total_scanned": len(tickers),
        "context":       {"vix": vix, "adx": adx, "pcr": pcr_today, "iv_rank": iv_rank_today},
        "predictions":   predictions[:req.top_n],
    }


@app.post("/backtest")
def run_backtest(req: BacktestRequest):
    """
    ASGI FIX: plain `def`.
    Fetches 100 stocks × 15 years each + runs full backtest engine.
    This is pure sync I/O + CPU — must NOT be async def.
    """
    tickers = req.custom_tickers or UNIVERSE_MAP.get(req.universe, [])
    if not tickers:
        raise HTTPException(400, "No tickers in universe")
    if req.initial_capital <= 0:
        raise HTTPException(400, "initial_capital must be > 0")

    cfg        = _strategy_to_cfg(req.strategy)
    stock_data = {}
    ml_model   = get_ml_model()

    for ticker in tickers[:100]:
        df = fetch_ohlcv(ticker, period="15y")
        if df.empty or len(df) < 100:
            continue
        df = compute_indicators(df, cfg)
        # Use real ML probabilities if model is trained, else baseline
        if ml_model.trained:
            try:
                df["ml_prob"] = ml_model.predict(df)["probability"]
            except Exception:
                df["ml_prob"] = 0.55
        else:
            df["ml_prob"] = 0.55
        df["tech_score"] = df.apply(lambda r: compute_signal_score(r, cfg)["score"], axis=1)
        stock_data[ticker] = df

    if not stock_data:
        raise HTTPException(400, "No valid stock data fetched")

    # ── BUG FIX: Fetch real India VIX history ─────────────────────────
    vix_data = get_india_vix_series()

    engine = BacktestEngine(
        strategy_name     = req.strategy.name,
        initial_capital   = req.initial_capital,
        position_size_pct = req.position_size_pct,
        max_positions     = req.max_positions,
        brokerage_model   = req.brokerage_model,
        entry_slippage    = True,
        exit_slippage     = True,
        use_regime        = True,
    )
    r = engine.run(
        stock_data  = stock_data,
        start_date  = req.start_date,
        end_date    = req.end_date,
        vix_data    = vix_data,
    )

    # ── Benchmark: passive NIFTY 50 SIP comparison ────────────────────
    benchmark = _compute_sip_benchmark(
        req.start_date, req.end_date, req.initial_capital
    )

    return {
        "strategy":           r.strategy_name,
        "initial_capital":    r.initial_capital,
        "final_capital":      r.final_capital,
        "net_profit":         r.net_profit,
        "total_return_pct":   r.total_return_pct,
        "cagr_pct":           r.cagr_pct,
        "sharpe_ratio":       r.sharpe_ratio,
        "sortino_ratio":      r.sortino_ratio,
        "calmar_ratio":       r.calmar_ratio,
        "max_drawdown_pct":   r.max_drawdown_pct,
        "win_rate":           r.win_rate,
        "total_trades":       r.total_trades,
        "avg_hold_days":      r.avg_hold_days,
        "total_charges":      r.total_charges,
        "total_slippage":     r.total_slippage,
        "regime_breakdown":   r.regime_breakdown,
        "monthly_returns":    r.monthly_returns,
        "equity_curve":       {str(d): v for d, v in r.equity_curve.items()},
        "trade_log":          r.trade_log[:50],
        # Alpha vs benchmark
        "benchmark": benchmark,
        "alpha_pct":  round(r.total_return_pct - benchmark.get("total_return_pct", 0), 2),
        "alpha_cagr": round(r.cagr_pct          - benchmark.get("cagr_pct", 0),        2),
    }


def _compute_sip_benchmark(start_date: str, end_date: str, capital: float) -> dict:
    """
    Compute passive NIFTY 50 SIP (lump sum) benchmark over the same period.
    Assumes buying and holding NIFTYBEES (NIFTY ETF proxy) from start to end.
    """
    try:
        df = fetch_ohlcv("NIFTYBEES", period="15y")
        if df.empty:
            # Fallback to NIFTY index
            import yfinance as yf
            raw = yf.download("^NSEI", start=start_date, end=end_date,
                              progress=False, auto_adjust=True)
            if raw.empty:
                return {}
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw.reset_index()
            if "index" in df.columns:
                df.rename(columns={"index": "Date"}, inplace=True)
            elif "Date" not in df.columns and "Datetime" in df.columns:
                df.rename(columns={"Datetime": "Date"}, inplace=True)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df[(df["Date"] >= pd.Timestamp(start_date)) &
                (df["Date"] <= pd.Timestamp(end_date))].sort_values("Date")
        if len(df) < 10:
            return {}

        entry_price = float(df.iloc[0]["Close"])
        exit_price  = float(df.iloc[-1]["Close"])
        qty         = capital / entry_price
        final       = qty * exit_price

        years   = (df.iloc[-1]["Date"] - df.iloc[0]["Date"]).days / 365.25
        total_r = (final / capital - 1) * 100
        cagr    = ((final / capital) ** (1 / max(0.1, years)) - 1) * 100

        # Daily returns for Sharpe
        daily_r = df["Close"].pct_change().dropna()
        sharpe  = (daily_r.mean() / daily_r.std() * (252 ** 0.5)) if daily_r.std() > 0 else 0

        roll_max = df["Close"].cummax()
        max_dd   = abs(((df["Close"] - roll_max) / roll_max).min() * 100)

        return {
            "name":              "NIFTY 50 Buy & Hold",
            "initial_capital":   round(capital, 2),
            "final_capital":     round(final, 2),
            "total_return_pct":  round(total_r, 2),
            "cagr_pct":          round(cagr, 2),
            "sharpe_ratio":      round(float(sharpe), 3),
            "max_drawdown_pct":  round(float(max_dd), 2),
            "note":              "Lump-sum buy on start date, sell on end date. No tax/charges.",
        }
    except Exception as e:
        logger.warning(f"Benchmark compute failed: {e}")
        return {}


@app.post("/train")
async def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    """BUG FIX: uses v3 kwargs — no horizon/threshold/skip_lstm."""
    global _training_status
    if _training_status["status"] == "running":
        return {"status": "already_running", "progress": _training_status["progress"]}

    async def _do_train():
        global _training_status, _ml_model
        try:
            _training_status = {"status": "running", "progress": 5, "message": "Fetching stock data..."}
            stock_data = fetch_universe(req.universe, period=req.period)

            _training_status["progress"] = 25
            _training_status["message"]  = f"Computing indicators for {len(stock_data)} stocks..."
            cfg       = IndicatorConfig()
            processed = {}
            for t, df in stock_data.items():
                try:
                    processed[t] = compute_indicators(df, cfg)
                except Exception:
                    pass

            _training_status["progress"] = 45
            _training_status["message"]  = f"Training ML ensemble on {len(processed)} stocks..."
            model   = AlphaYantraML("default")
            metrics = model.train(
                all_stock_dfs      = processed,
                n_cv_folds         = req.n_cv_folds,
                skip_tcn           = req.skip_tcn,
                tcn_max_samples    = req.tcn_max_samples,
                tcn_epochs         = req.tcn_epochs,
                use_triple_barrier = req.use_triple_barrier,
            )
            _ml_model       = model
            _training_status = {"status": "complete", "progress": 100,
                                 "message": "Training complete", "metrics": metrics}
        except Exception as e:
            logger.error(f"Training failed: {e}")
            _training_status = {"status": "error", "progress": 0, "message": str(e)}

    background_tasks.add_task(_do_train)
    return {"status": "started", "message": "Training started in background"}


@app.get("/train/status")
async def train_status():
    return _training_status


# ── Options ─────────────────────────────────────────────────────────────

@app.get("/options/chain")
def options_chain(symbol: str = "NIFTY", expiry: Optional[str] = None, num_strikes: int = 15):
    oc    = get_options_chain()
    chain = oc.get_chain(symbol=symbol, expiry=expiry, num_strikes=num_strikes)
    if not chain:
        raise HTTPException(503, f"Could not fetch options chain for {symbol}")
    return {
        "symbol": chain.symbol, "spot_price": chain.spot_price, "expiry": chain.expiry,
        "timestamp": chain.timestamp, "atm_strike": chain.atm_strike,
        "pcr": chain.pcr, "max_pain": chain.max_pain, "iv_rank": chain.iv_rank,
        "call_oi_total": chain.call_oi_total, "put_oi_total": chain.put_oi_total,
        "support_levels": chain.support_levels, "resistance_levels": chain.resistance_levels,
        "sentiment": chain.sentiment,
        "strikes": [{
            "strike": s.strike, "ce_ltp": s.ce_ltp, "pe_ltp": s.pe_ltp,
            "ce_iv": s.ce_iv, "pe_iv": s.pe_iv, "ce_oi": s.ce_oi, "pe_oi": s.pe_oi,
            "ce_oi_change": s.ce_oi_change, "pe_oi_change": s.pe_oi_change,
            "ce_volume": s.ce_volume, "pe_volume": s.pe_volume, "pcr": s.pcr,
            "ce_greeks": {"delta": s.ce_greeks.delta, "gamma": s.ce_greeks.gamma,
                          "theta": s.ce_greeks.theta, "vega": s.ce_greeks.vega},
            "pe_greeks": {"delta": s.pe_greeks.delta, "gamma": s.pe_greeks.gamma,
                          "theta": s.pe_greeks.theta, "vega": s.pe_greeks.vega},
        } for s in chain.strikes],
    }


@app.get("/options/signal")
def options_signal(symbol: str = "NIFTY", signal: str = "BUY",
                          budget: float = 50000, strategy: str = "buy_option"):
    oc    = get_options_chain()
    chain = oc.get_chain(symbol=symbol, num_strikes=10)
    if not chain:
        raise HTTPException(503, "Could not fetch chain")
    return oc.get_best_strikes(chain=chain, signal=signal, strategy=strategy, budget=budget)


# ── Feed ────────────────────────────────────────────────────────────────

@app.get("/feed/quotes")
async def feed_quotes():
    f = get_live_feed()
    return {"market_open": is_market_open(), "timestamp": datetime.now().isoformat(),
            "quotes": f.latest_quotes, "symbols": list(f.latest_quotes.keys())}

@app.get("/feed/candles/{symbol}")
async def feed_candles(symbol: str):
    f = get_live_feed()
    if symbol not in f.candle_builders:
        raise HTTPException(404, f"{symbol} not in watchlist")
    df = f.candle_builders[symbol].get_dataframe()
    return {"symbol": symbol, "count": len(df),
            "candles": df.to_dict(orient="records") if not df.empty else []}

@app.get("/feed/status")
async def feed_status():
    f = get_live_feed()
    return {"running": f._running, "watchlist": f.watchlist,
            "symbols_live": list(f.latest_quotes.keys()), "market_open": is_market_open()}


# ── Risk ─────────────────────────────────────────────────────────────────

@app.get("/risk/status")
async def risk_status():
    return get_risk_manager().get_status()

@app.post("/risk/kill-switch/activate")
async def activate_kill(reason: str = "Manual"):
    get_risk_manager().activate_kill_switch(reason)
    return {"kill_switch": True, "reason": reason}

@app.post("/risk/kill-switch/deactivate")
async def deactivate_kill():
    get_risk_manager().deactivate_kill_switch()
    return {"kill_switch": False}

@app.get("/risk/config")
async def get_risk_config():
    from dataclasses import asdict
    return asdict(get_risk_manager().config)


# ── Paper trading ────────────────────────────────────────────────────────

@app.get("/paper/portfolio")
def paper_portfolio():
    from broker.paper_trader import PaperTrader
    pt = PaperTrader(risk_manager=get_risk_manager(), live_feed=get_live_feed())
    return pt.get_portfolio_summary()

@app.get("/paper/report")
def paper_report():
    from broker.paper_trader import PaperTrader
    pt = PaperTrader(risk_manager=get_risk_manager(), live_feed=get_live_feed())
    return {"report": pt.daily_report()}


# ── Strategies ───────────────────────────────────────────────────────────

@app.post("/strategies")
async def save_strategy(cfg: StrategyConfig):
    path = STRATEGY_DIR / f"{cfg.name.replace(' ', '_')}.json"
    with open(path, "w") as f:
        json.dump(cfg.dict(), f, indent=2)
    return {"saved": True, "path": str(path)}

@app.get("/strategies")
async def list_strategies():
    strategies = []
    for p in STRATEGY_DIR.glob("*.json"):
        with open(p) as f:
            strategies.append(json.load(f))
    return {"strategies": strategies}


# ── TradingView webhook ───────────────────────────────────────────────────

@app.post("/tradingview/signal")
def tradingview_webhook(payload: dict):
    """
    TradingView webhook — receives JSON alerts from TradingView Pine Script.

    MUST be a plain `def` (not `async def`).
    fetch_ohlcv() uses yfinance which is synchronous requests under the hood.
    If this were async, it would block the entire ASGI event loop for 1-2s
    per request.  FastAPI automatically routes plain `def` endpoints to a
    background thread pool, keeping the event loop free.
    """
    ticker = payload.get("ticker", "").replace("NSE:", "").split(":")[0]
    if not ticker:
        return {"status": "ignored"}

    model = get_ml_model()
    df    = fetch_ohlcv(ticker, period="2y")
    if df.empty:
        return {"status": "no_data"}

    cfg  = IndicatorConfig()
    df   = compute_indicators(df, cfg)
    tech = compute_signal_score(df.iloc[-1], cfg)
    vix  = get_live_vix()
    adx  = get_live_adx()

    # ── Inject live silo features (same as /predict endpoint) ───────────
    today  = pd.Timestamp.today().normalize()
    pcr_s  = iv_s = sent_s = None
    try:
        oc    = get_options_chain()
        chain = oc.get_chain("NIFTY", num_strikes=5)
        if chain:
            pcr_s = pd.Series({today: chain.pcr})
            iv_s  = pd.Series({today: chain.iv_rank})
    except Exception:
        pass
    try:
        news  = get_news_engine().get_stock_sentiment(ticker, hours_window=24)
        sent_s = pd.Series({today: news["score"]})
    except Exception:
        pass

    if model.trained:
        ml = model.predict(
            df_with_indicators = df,
            vix                = vix,
            adx                = adx,
            pcr_series         = pcr_s,
            iv_rank_series     = iv_s,
            sentiment_series   = sent_s,
            vix_series         = get_india_vix_series(),
        )
    else:
        ml = {"signal": "HOLD", "confidence": 50}

    return {
        "ticker":        ticker,
        "tv_action":     payload.get("action", ""),
        "ml_signal":     ml["signal"],
        "ml_confidence": ml.get("confidence"),
        "tech_score":    tech["score"],
        "stop_loss":     tech["stop_loss"],
        "take_profit":   tech["take_profit"],
        "regime":        ml.get("regime", "normal"),
        "vix":           vix,
    }


# ── Bhavcopy management ───────────────────────────────────────────────────

@app.get("/bhavcopy/status")
async def bhavcopy_status():
    from data.bhavcopy import BhavcopyScraper
    return BhavcopyScraper().status()

@app.post("/bhavcopy/update")
async def bhavcopy_update(background_tasks: BackgroundTasks):
    from data.bhavcopy import BhavcopyScraper
    def _update():
        BhavcopyScraper().update()
    background_tasks.add_task(_update)
    return {"status": "started", "message": "Bhavcopy update running in background"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
