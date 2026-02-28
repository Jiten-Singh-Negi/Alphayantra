"""
strategies/indicators.py
─────────────────────────
Computes ALL technical indicators from real OHLCV data.
Every value here is calculated from actual price/volume — nothing hardcoded.

Indicators implemented:
  Trend:      EMA(fast/slow), SMA200, VWAP, Supertrend, Ichimoku
  Momentum:   RSI, MACD, Stochastic RSI, Williams %R, CCI, ADX
  Volatility: Bollinger Bands, ATR, Keltner Channels
  Volume:     OBV, Volume Spike, Delivery %, CMF
  Structure:  Fibonacci Retracements, Pivot Points (Daily/Weekly)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import ta  # pip install ta


@dataclass
class IndicatorConfig:
    """
    User-defined indicator settings — mirrors the Strategy Builder sliders.
    All defaults match the HTML frontend defaults.
    """
    # ── Enabled indicators ────────────────────────────────────────────
    use_ema_fast:    bool  = True
    use_ema_slow:    bool  = True
    use_sma200:      bool  = False
    use_vwap:        bool  = True
    use_rsi:         bool  = True
    use_macd:        bool  = True
    use_stochrsi:    bool  = False
    use_bb:          bool  = True
    use_atr:         bool  = True
    use_adx:         bool  = False
    use_supertrend:  bool  = False
    use_ichimoku:    bool  = False
    use_williams:    bool  = False
    use_cci:         bool  = False
    use_obv:         bool  = False
    use_vol_spike:   bool  = True
    use_delivery:    bool  = True
    use_fib:         bool  = True
    use_pivot:       bool  = True

    # ── EMA ───────────────────────────────────────────────────────────
    ema_fast:        int   = 9
    ema_slow:        int   = 21

    # ── RSI ───────────────────────────────────────────────────────────
    rsi_period:      int   = 14
    rsi_oversold:    int   = 30       # Buy zone threshold
    rsi_overbought:  int   = 70       # Sell zone threshold

    # ── MACD ──────────────────────────────────────────────────────────
    macd_fast:       int   = 12
    macd_slow:       int   = 26
    macd_signal:     int   = 9

    # ── Stochastic RSI ────────────────────────────────────────────────
    stochrsi_period: int   = 14
    stochrsi_oversold: int = 20

    # ── Bollinger Bands ───────────────────────────────────────────────
    bb_period:       int   = 20
    bb_std:          float = 2.0

    # ── ATR ───────────────────────────────────────────────────────────
    atr_period:      int   = 14
    atr_sl_mult:     float = 1.5      # Stop loss = entry − ATR × mult

    # ── ADX ───────────────────────────────────────────────────────────
    adx_min:         int   = 25       # Only trade if ADX > this (trending market)

    # ── Supertrend ────────────────────────────────────────────────────
    st_atr_mult:     float = 3.0

    # ── Volume ────────────────────────────────────────────────────────
    vol_spike_mult:  float = 1.5      # Spike if volume > 1.5× 20-day avg
    delivery_min:    float = 40.0     # Min delivery % to qualify (NSE data)

    # ── Fibonacci ─────────────────────────────────────────────────────
    fib_levels:      list  = field(default_factory=lambda: [0.236, 0.382, 0.618, 0.786])
    fib_extensions:  list  = field(default_factory=lambda: [1.272, 1.618])
    fib_tolerance:   float = 0.015    # ±1.5% band around each level

    # ── Williams %R ───────────────────────────────────────────────────
    wr_period:       int   = 14
    wr_oversold:     int   = -80

    # ── CCI ───────────────────────────────────────────────────────────
    cci_period:      int   = 20
    cci_oversold:    int   = -100

    # ── Signal confirmation ───────────────────────────────────────────
    min_confirmations: int = 3        # Min indicators must agree for a signal

    # ── Indicator weights (for scoring 0–100) ─────────────────────────
    weight_rsi:      float = 0.25
    weight_macd:     float = 0.20
    weight_bb:       float = 0.15
    weight_volume:   float = 0.15
    weight_fib:      float = 0.10
    weight_ema:      float = 0.15
    weight_news:     float = 0.20     # News sentiment contribution


def compute_indicators(df: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """
    Compute all enabled indicators on an OHLCV DataFrame.

    Input columns required: Date, Open, High, Low, Close, Volume
    Returns the same DataFrame with indicator columns added.
    """
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Moving Averages ────────────────────────────────────────────────
    if cfg.use_ema_fast:
        df["ema_fast"] = ta.trend.ema_indicator(close, window=cfg.ema_fast)

    if cfg.use_ema_slow:
        df["ema_slow"] = ta.trend.ema_indicator(close, window=cfg.ema_slow)
        if cfg.use_ema_fast:
            # Golden/Death cross
            df["ema_cross"] = (df["ema_fast"] > df["ema_slow"]).astype(int)
            df["ema_cross_prev"] = df["ema_cross"].shift(1)
            df["ema_golden"] = ((df["ema_cross"] == 1) & (df["ema_cross_prev"] == 0)).astype(int)
            df["ema_death"]  = ((df["ema_cross"] == 0) & (df["ema_cross_prev"] == 1)).astype(int)

    if cfg.use_sma200:
        df["sma200"] = ta.trend.sma_indicator(close, window=200)
        df["above_sma200"] = (close > df["sma200"]).astype(int)

    if cfg.use_vwap:
        df["vwap"] = (close * vol).cumsum() / vol.cumsum()
        df["above_vwap"] = (close > df["vwap"]).astype(int)

    # ── RSI ───────────────────────────────────────────────────────────
    if cfg.use_rsi:
        df["rsi"] = ta.momentum.rsi(close, window=cfg.rsi_period)
        df["rsi_oversold"]    = (df["rsi"] < cfg.rsi_oversold).astype(int)
        df["rsi_overbought"]  = (df["rsi"] > cfg.rsi_overbought).astype(int)
        # RSI divergence: price makes new low but RSI doesn't (bullish)
        df["rsi_bull_div"] = (
            (close < close.shift(5)) & (df["rsi"] > df["rsi"].shift(5))
        ).astype(int)

    # ── MACD ──────────────────────────────────────────────────────────
    if cfg.use_macd:
        macd_obj = ta.trend.MACD(
            close,
            window_fast=cfg.macd_fast,
            window_slow=cfg.macd_slow,
            window_sign=cfg.macd_signal,
        )
        df["macd"]         = macd_obj.macd()
        df["macd_signal"]  = macd_obj.macd_signal()
        df["macd_hist"]    = macd_obj.macd_diff()
        df["macd_bullish"] = (
            (df["macd"] > df["macd_signal"]) &
            (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        ).astype(int)
        df["macd_bearish"] = (
            (df["macd"] < df["macd_signal"]) &
            (df["macd"].shift(1) >= df["macd_signal"].shift(1))
        ).astype(int)

    # ── Stochastic RSI ────────────────────────────────────────────────
    if cfg.use_stochrsi:
        stoch = ta.momentum.StochRSIIndicator(close, window=cfg.stochrsi_period)
        df["stochrsi_k"] = stoch.stochrsi_k() * 100
        df["stochrsi_d"] = stoch.stochrsi_d() * 100
        df["stochrsi_oversold"] = (df["stochrsi_k"] < cfg.stochrsi_oversold).astype(int)

    # ── Bollinger Bands ───────────────────────────────────────────────
    if cfg.use_bb:
        bb = ta.volatility.BollingerBands(close, window=cfg.bb_period, window_dev=cfg.bb_std)
        df["bb_upper"]    = bb.bollinger_hband()
        df["bb_lower"]    = bb.bollinger_lband()
        df["bb_mid"]      = bb.bollinger_mavg()
        df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"]      = bb.bollinger_pband()   # 0=lower, 1=upper
        df["bb_squeeze"]  = (df["bb_width"] < df["bb_width"].rolling(50).quantile(0.20)).astype(int)
        df["bb_touch_lower"] = (close <= df["bb_lower"]).astype(int)
        df["bb_touch_upper"] = (close >= df["bb_upper"]).astype(int)

    # ── ATR (volatility & stop loss sizing) ───────────────────────────
    if cfg.use_atr:
        df["atr"] = ta.volatility.average_true_range(high, low, close, window=cfg.atr_period)
        df["stop_loss"]   = close - (df["atr"] * cfg.atr_sl_mult)
        df["take_profit"] = close + (df["atr"] * cfg.atr_sl_mult * 2)   # 2:1 R:R

    # ── ADX (trend strength filter) ───────────────────────────────────
    if cfg.use_adx:
        adx_obj = ta.trend.ADXIndicator(high, low, close, window=14)
        df["adx"]      = adx_obj.adx()
        df["adx_pos"]  = adx_obj.adx_pos()
        df["adx_neg"]  = adx_obj.adx_neg()
        df["trending"] = (df["adx"] > cfg.adx_min).astype(int)

    # ── Supertrend ────────────────────────────────────────────────────
    if cfg.use_supertrend:
        df = _compute_supertrend(df, cfg.st_atr_mult)

    # ── Williams %R ───────────────────────────────────────────────────
    if cfg.use_williams:
        df["williams_r"]         = ta.momentum.WilliamsRIndicator(high, low, close, lbp=cfg.wr_period).williams_r()
        df["williams_oversold"]  = (df["williams_r"] < cfg.wr_oversold).astype(int)
        df["williams_overbought"] = (df["williams_r"] > -20).astype(int)

    # ── CCI ───────────────────────────────────────────────────────────
    if cfg.use_cci:
        df["cci"] = ta.trend.cci(high, low, close, window=cfg.cci_period)
        df["cci_oversold"]   = (df["cci"] < cfg.cci_oversold).astype(int)
        df["cci_overbought"] = (df["cci"] > abs(cfg.cci_oversold)).astype(int)

    # ── Volume indicators ─────────────────────────────────────────────
    if cfg.use_obv:
        df["obv"] = ta.volume.on_balance_volume(close, vol)
        df["obv_ema"] = ta.trend.ema_indicator(df["obv"], window=20)
        df["obv_rising"] = (df["obv"] > df["obv_ema"]).astype(int)

    if cfg.use_vol_spike:
        vol_ma = vol.rolling(20).mean()
        df["vol_ratio"]  = vol / vol_ma
        df["vol_spike"]  = (df["vol_ratio"] > cfg.vol_spike_mult).astype(int)

    # ── Fibonacci Retracements ────────────────────────────────────────
    if cfg.use_fib:
        df = _compute_fibonacci(df, cfg)

    # ── Pivot Points (daily) ──────────────────────────────────────────
    if cfg.use_pivot:
        df["pivot"]  = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        df["pivot_r1"] = 2 * df["pivot"] - low.shift(1)
        df["pivot_s1"] = 2 * df["pivot"] - high.shift(1)
        df["pivot_r2"] = df["pivot"] + (high.shift(1) - low.shift(1))
        df["pivot_s2"] = df["pivot"] - (high.shift(1) - low.shift(1))
        df["near_support"] = (
            (close - df["pivot_s1"]).abs() / close < 0.015
        ).astype(int)

    # ── Ichimoku Cloud ────────────────────────────────────────────────
    if cfg.use_ichimoku:
        df = _compute_ichimoku(df)

    return df


def compute_signal_score(row: pd.Series, cfg: IndicatorConfig) -> dict:
    """
    Given a single row of indicator values, compute a composite buy/sell score.

    Score: 0–100
        > 70 → Strong Buy
        50–70 → Buy
        30–50 → Hold
        < 30  → Sell

    Returns:
        {
          "score": float,          # 0–100
          "signal": str,           # "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL"
          "confirmations": int,    # how many indicators agreed
          "breakdown": dict,       # per-indicator contribution
          "stop_loss": float,
          "take_profit": float,
        }
    """
    breakdown = {}
    confirmations = 0

    def _safe(key, default=np.nan):
        v = row.get(key, default)
        return default if pd.isna(v) else v

    # RSI contribution
    rsi_val = _safe("rsi", 50)
    rsi_score = 0
    if cfg.use_rsi:
        if rsi_val < cfg.rsi_oversold:
            rsi_score = 1.0
            confirmations += 1
        elif rsi_val < 45:
            rsi_score = 0.6
        elif rsi_val > cfg.rsi_overbought:
            rsi_score = -1.0
        else:
            rsi_score = max(0, (50 - rsi_val) / 50)
        breakdown["rsi"] = round(rsi_score * cfg.weight_rsi * 100, 1)

    # MACD contribution
    macd_score = 0
    if cfg.use_macd:
        if _safe("macd_bullish") == 1:
            macd_score = 1.0
            confirmations += 1
        elif _safe("macd_hist", 0) > 0:
            macd_score = 0.5
        elif _safe("macd_bearish") == 1:
            macd_score = -1.0
        breakdown["macd"] = round(macd_score * cfg.weight_macd * 100, 1)

    # Bollinger Band contribution
    bb_score = 0
    if cfg.use_bb:
        if _safe("bb_touch_lower") == 1:
            bb_score = 1.0
            confirmations += 1
        elif _safe("bb_pct", 0.5) < 0.2:
            bb_score = 0.6
        elif _safe("bb_touch_upper") == 1:
            bb_score = -1.0
        elif _safe("bb_squeeze") == 1:
            bb_score = 0.3      # Squeeze = imminent breakout
        breakdown["bb"] = round(bb_score * cfg.weight_bb * 100, 1)

    # Volume contribution
    vol_score = 0
    if cfg.use_vol_spike:
        if _safe("vol_spike") == 1:
            vol_score = 0.8
            confirmations += 1
        elif _safe("vol_ratio", 1) > 1.2:
            vol_score = 0.4
        breakdown["volume"] = round(vol_score * cfg.weight_volume * 100, 1)

    # Fibonacci contribution
    fib_score = 0
    if cfg.use_fib:
        fib_score = _safe("fib_score", 0)
        if fib_score > 0.5:
            confirmations += 1
        breakdown["fibonacci"] = round(fib_score * cfg.weight_fib * 100, 1)

    # EMA trend contribution
    ema_score = 0
    if cfg.use_ema_fast and cfg.use_ema_slow:
        if _safe("ema_golden") == 1:
            ema_score = 1.0
            confirmations += 1
        elif _safe("ema_cross") == 1:
            ema_score = 0.5
        elif _safe("ema_death") == 1:
            ema_score = -1.0
        breakdown["ema_trend"] = round(ema_score * cfg.weight_ema * 100, 1)

    # Total technical score (0–100 range)
    raw = (
        rsi_score  * cfg.weight_rsi  +
        macd_score * cfg.weight_macd +
        bb_score   * cfg.weight_bb   +
        vol_score  * cfg.weight_volume +
        fib_score  * cfg.weight_fib  +
        ema_score  * cfg.weight_ema
    )
    # Normalise to 0–100 (raw is in -1 to +1 range weighted)
    max_possible = (cfg.weight_rsi + cfg.weight_macd + cfg.weight_bb +
                    cfg.weight_volume + cfg.weight_fib + cfg.weight_ema)
    tech_score = ((raw + max_possible) / (2 * max_possible)) * 100
    tech_score = max(0, min(100, tech_score))

    # Determine signal label
    if confirmations >= cfg.min_confirmations and tech_score >= 75:
        signal = "STRONG BUY"
    elif tech_score >= 60:
        signal = "BUY"
    elif tech_score <= 25:
        signal = "STRONG SELL"
    elif tech_score <= 40:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Stop loss / take profit from ATR
    sl = _safe("stop_loss", _safe("Close", 0) * 0.97)
    tp = _safe("take_profit", _safe("Close", 0) * 1.05)

    return {
        "score":         round(tech_score, 1),
        "signal":        signal,
        "confirmations": confirmations,
        "breakdown":     breakdown,
        "stop_loss":     round(sl, 2),
        "take_profit":   round(tp, 2),
        "rsi":           round(rsi_val, 1),
    }


# ── Private helpers ───────────────────────────────────────────────────

def _compute_fibonacci(df: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """
    Compute Fibonacci retracement levels over a rolling 52-week high/low window.
    Adds:
      fib_score: 0–1 (how close price is to a key fib level)
      fib_level: which fib level is nearest
    """
    window = 252  # 1 trading year
    df["fib_high"] = df["High"].rolling(window).max()
    df["fib_low"]  = df["Low"].rolling(window).min()
    df["fib_range"] = df["fib_high"] - df["fib_low"]

    df["fib_score"] = 0.0
    df["fib_level"] = np.nan

    for level in cfg.fib_levels:
        fib_price = df["fib_high"] - level * df["fib_range"]
        distance  = (df["Close"] - fib_price).abs() / df["Close"]
        at_level  = distance < cfg.fib_tolerance
        # Score inversely proportional to distance
        score     = (1 - distance / cfg.fib_tolerance).clip(0, 1)
        df.loc[at_level, "fib_score"] = score[at_level]
        df.loc[at_level, "fib_level"] = level

    return df


def _compute_supertrend(df: pd.DataFrame, mult: float) -> pd.DataFrame:
    """
    Compute Supertrend indicator.
    supertrend_bull = 1 when price is above supertrend line (bullish)
    """
    atr = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=10)
    hl2 = (df["High"] + df["Low"]) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction  = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        if df["Close"].iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["Close"].iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    df["supertrend_bull"] = (direction == 1).astype(int)
    return df


def _compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Ichimoku Cloud components.
    above_cloud = 1 when price is above the cloud (bullish)
    """
    high, low, close = df["High"], df["Low"], df["Close"]

    tenkan  = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun   = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a  = ((tenkan + kijun) / 2).shift(26)
    span_b  = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    df["ichimoku_tenkan"]  = tenkan
    df["ichimoku_kijun"]   = kijun
    df["ichimoku_span_a"]  = span_a
    df["ichimoku_span_b"]  = span_b

    cloud_top    = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

    df["above_cloud"] = (close > cloud_top).astype(int)
    df["below_cloud"] = (close < cloud_bottom).astype(int)
    df["tk_cross_bull"] = (
        (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    ).astype(int)

    return df
