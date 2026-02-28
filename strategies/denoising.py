"""
strategies/denoising.py  — v5 (causal, no lookahead bias)
───────────────────────────────────────────────────────────
CRITICAL FIX:
  OLD: dwt_denoise() called pywt.wavedec() on the ENTIRE array at once.
       Row t's denoised value used rows t+1..n (future prices) → lookahead bias.
       A model trained on this would show 90%+ backtest accuracy but fail live.

  NEW: Every function is strictly CAUSAL:
       - dwt_denoise_causal(): rolling window, takes only [t-W+1..t] each step
       - denoise_series(): wraps dwt_denoise_causal
       - add_denoised_features(): calls denoise_series (causal)
       - dwt_denoise() retained for ONE legitimate use: live inference on a
         short recent window (e.g., last 64 candles), where no future exists.

Usage in training pipeline: always call denoise_series() or add_denoised_features().
Usage at inference time: either dwt_denoise() on the last 64 rows, or just call
add_denoised_features() — it's causal so it's safe in both contexts.
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Literal


# ── Window parameters ─────────────────────────────────────────────────
DWT_WINDOW     = 64   # rolling window for causal DWT (power of 2)
DWT_MIN_WINDOW = 16   # minimum history before attempting DWT


def dwt_denoise(
    prices:      np.ndarray,
    wavelet:     str = "db4",
    level:       int = 3,
    keep_levels: int = 1,
) -> np.ndarray:
    """
    ONE-SHOT DWT denoising of a complete array.

    ONLY USE THIS for live inference on a short lookback window
    (e.g., the last 64 candles of the current day) where NO future
    data is present in the array by definition.

    DO NOT call this during training or backtesting on a full
    historical array — it will introduce lookahead bias.
    Use dwt_denoise_causal() or denoise_series() instead.
    """
    if len(prices) < 8:
        return prices.copy()
    try:
        import pywt
    except ImportError:
        return prices.copy()
    try:
        coeffs = pywt.wavedec(prices, wavelet, level=level, mode="periodization")
        for i in range(1, keep_levels + 1):
            coeffs[i] = np.zeros_like(coeffs[i])
        denoised = pywt.waverec(coeffs, wavelet, mode="periodization")
        return denoised[: len(prices)]
    except Exception as e:
        logger.debug(f"DWT one-shot failed: {e}")
        return prices.copy()


def dwt_denoise_causal(
    prices: np.ndarray,
    window: int = DWT_WINDOW,
    wavelet: str = "db4",
    level:  int = 3,
) -> np.ndarray:
    """
    CAUSAL rolling-window DWT denoising.

    For each time step t:
      - Take window [t-window+1 .. t]  (past only, never future)
      - Apply DWT denoising to that window
      - Take the LAST element as the denoised value for row t

    This exactly mirrors what the model sees during live trading.
    Safe to use during training, backtesting, and inference.

    Time complexity: O(n × window) — about 0.3s for 3000 rows, window=64.
    """
    n        = len(prices)
    filtered = prices.copy().astype(float)

    try:
        import pywt

        for t in range(n):
            start         = max(0, t - window + 1)
            window_prices = prices[start : t + 1]
            L             = len(window_prices)

            if L < DWT_MIN_WINDOW:
                # Not enough history — exponential smoothing proxy
                alpha = 0.2
                sm    = float(window_prices[0])
                for p in window_prices[1:]:
                    sm = alpha * float(p) + (1 - alpha) * sm
                filtered[t] = sm
                continue

            try:
                coeffs = pywt.wavedec(window_prices, wavelet, level=level,
                                      mode="periodization")
                # Zero finest detail band (highest freq = noise)
                coeffs[-1][:] = 0.0
                # Soft-threshold next band
                med           = np.median(np.abs(coeffs[-2]))
                sigma         = med / 0.6745
                threshold     = sigma * np.sqrt(2 * np.log(max(L, 2)))
                coeffs[-2]    = pywt.threshold(coeffs[-2], threshold, mode="soft")
                recon         = pywt.waverec(coeffs, wavelet, mode="periodization")
                # Last element = row t's causal denoised price
                filtered[t]   = float(recon[min(L - 1, len(recon) - 1)])
            except Exception:
                # If DWT fails for this window, fall back to EMA
                alpha       = 0.15
                filtered[t] = alpha * prices[t] + (1 - alpha) * filtered[max(0, t - 1)]

    except ImportError:
        # PyWavelets not installed — causal EMA fallback
        logger.debug("PyWavelets not installed — using causal EMA for denoising")
        alpha = 0.15
        sm    = float(prices[0])
        for t in range(n):
            sm          = alpha * float(prices[t]) + (1 - alpha) * sm
            filtered[t] = sm

    return filtered


def denoise_series(
    series:  pd.Series,
    method:  Literal["dwt", "kalman", "both"] = "dwt",
    window:  int = DWT_WINDOW,
    wavelet: str = "db4",
    level:   int = 3,
) -> pd.Series:
    """
    Denoise a pandas price Series using strictly causal transforms.
    Returns a new Series with the same index and dtype=float64.

    Usage:
        clean_close = denoise_series(df["Close"])
        df["rsi"]   = ta.momentum.rsi(clean_close, window=14)
    """
    values = series.values.astype(float)

    if method in ("dwt", "both"):
        values = dwt_denoise_causal(values, window=window, wavelet=wavelet, level=level)

    if method in ("kalman", "both"):
        # Causal EMA (proxy for Kalman) — only uses past values by definition
        alpha    = 0.15
        smoothed = values.copy()
        for i in range(1, len(smoothed)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        values = smoothed

    return pd.Series(values, index=series.index, name=series.name, dtype=float)


def add_denoised_features(df: pd.DataFrame, window: int = DWT_WINDOW) -> pd.DataFrame:
    """
    Add causal DWT-denoised features to a DataFrame.

    Columns added:
      close_dwt      — causal denoised close price
      rsi_clean      — RSI computed on causal denoised price
      macd_clean_hist— MACD histogram on causal denoised price
      rsi_divergence — rsi_clean minus raw rsi (divergence signal)

    All transforms are strictly causal — safe for training and backtesting.
    """
    if len(df) < DWT_MIN_WINDOW:
        return df

    df         = df.copy()
    close_clean = denoise_series(df["Close"], method="dwt", window=window)
    df["close_dwt"] = close_clean

    try:
        import ta

        rsi_clean = ta.momentum.RSIIndicator(close_clean, window=14).rsi()
        df["rsi_clean"] = rsi_clean.values

        macd = ta.trend.MACD(close_clean, window_fast=12, window_slow=26, window_sign=9)
        df["macd_clean_hist"] = macd.macd_diff().values

        if "rsi" in df.columns:
            df["rsi_divergence"] = (df["rsi_clean"] - df["rsi"]).clip(-20, 20)

    except Exception as e:
        logger.debug(f"Denoised indicator computation failed: {e}")

    return df


def estimate_noise_level(prices: np.ndarray, window: int = 64) -> float:
    """
    Estimate the noise-to-signal ratio of a price series.
    Uses the LAST `window` rows only — causal by design.
    Higher value = noisier = denoising will help more.
    """
    segment = prices[-window:] if len(prices) >= window else prices
    try:
        import pywt
        coeffs     = pywt.wavedec(segment, "db4", level=3)
        hf_energy  = sum(np.sum(c ** 2) for c in coeffs[1:3])
        tot_energy = sum(np.sum(c ** 2) for c in coeffs)
        return float(hf_energy / max(tot_energy, 1e-8))
    except Exception:
        return 0.3
