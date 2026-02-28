# AlphaYantra — Real AI Trading System 🇮🇳

A **real** Python ML system for Indian stock market prediction.  
No fake numbers. No simulated percentages. Actual ML trained on 15 years of NSE/BSE data.

---

## Architecture

```
alphayantra/
├── data/
│   └── fetcher.py          ← Real OHLCV from NSE via yfinance
├── strategies/
│   └── indicators.py       ← RSI, MACD, Bollinger, Fibonacci, ATR, etc.
├── models/
│   └── ml_engine.py        ← Random Forest + XGBoost + LSTM ensemble
├── backtest/
│   └── engine.py           ← Real trade simulation with Indian market charges
├── news/
│   └── sentiment.py        ← FinBERT NLP on ET/Moneycontrol/BusinessLine/NSE filings
├── api/
│   └── main.py             ← FastAPI REST server
└── run.py                  ← Entry point
```

---

## Setup (Step by Step)

### 1. Python version
```bash
python --version   # needs 3.10+
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ PyTorch: if you have a GPU, install the CUDA version from pytorch.org for much faster training.

### 4. Check everything works
```bash
python run.py --check
```

### 5. Quick demo (no training needed)
```bash
python run.py --demo
```
This fetches RELIANCE data from NSE and shows you real indicator values.

### 6. Train the ML model (takes 30–90 min)
```bash
python run.py --train
```
Trains on NIFTY 500 stocks × 15 years. Saves to `models/saved/default/`.

### 7. Start the API server
```bash
python run.py
```
Open: http://localhost:8000/docs

---

## API Endpoints

### Get predictions for all NIFTY stocks
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": {"name": "My Strategy", "use_rsi": true, "use_macd": true},
    "universe": "nifty50",
    "top_n": 10
  }'
```

### Run a real backtest
```bash
curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": {"name": "Test"},
    "universe": "nifty50",
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 1000000
  }'
```
Returns actual equity curve, trade log, Sharpe ratio, etc. — computed from your capital.

### Get news sentiment for RELIANCE
```bash
curl http://localhost:8000/news/RELIANCE?hours=24
```

### Get technical indicators
```bash
curl http://localhost:8000/indicators/TCS
```

---

## Connect to TradingView

1. Start the server: `python run.py`
2. Expose it publicly (use ngrok if local): `ngrok http 8000`
3. In TradingView → Alerts → Webhook URL → `https://your-ngrok-url/tradingview/signal`
4. Alert message format:
```json
{"ticker": "{{ticker}}", "action": "{{strategy.order.action}}", "price": {{close}}}
```
The API returns ML-enriched signal back to you.

---

## How the ML works

### What it learns
The model is trained to answer: **"Will this stock gain >2% in the next 5 trading days?"**

### Features (50+ inputs per day per stock)
- All technical indicator values (RSI, MACD, Bollinger, etc.)
- Price ratios (normalised — no raw prices)
- Rolling returns (1d, 3d, 5d, 10d, 20d)
- Candle body/wick ratios
- Volume ratios

### Models
| Model | Role | Why |
|-------|------|-----|
| Random Forest | Captures non-linear indicator interactions | Robust, doesn't overfit easily |
| XGBoost | Gradient boosting on tabular data | Usually best on financial tabular data |
| LSTM | Temporal sequence learning (30-day window) | Captures patterns across multiple days |
| **Ensemble** | Weighted average: RF(35%) + XGB(40%) + LSTM(25%) | More stable than any single model |

### Walk-forward validation
Training uses **walk-forward validation** — the model is always trained on past data and tested on future data. This prevents the #1 mistake in backtesting: data leakage.

### Strategy Optimization
The system can run multiple strategies across 500 stocks and compare:
- Win rate
- Sharpe ratio  
- Max drawdown
- CAGR

POST different `StrategyConfig` objects to `/backtest` and compare results.

---

## Realistic expectations

| Period | NIFTY 50 benchmark | This system (well-tuned) |
|--------|-------------------|--------------------------|
| 1 year | ~12% avg CAGR     | 15–25% CAGR (no guarantee) |
| Sharpe | ~0.6–0.8          | 1.2–2.0 target |
| Max DD | -30 to -50%       | -15 to -25% (with SL) |

**Important**: Backtest results ≠ live trading results. Always paper trade first.

---

## Data Sources

| Source | What | Notes |
|--------|------|-------|
| Yahoo Finance (via yfinance) | OHLCV daily data for NSE | Free, 15yr history |
| NSE India API | Corporate filings, results | Free, no key needed |
| RSS feeds | ET, Moneycontrol, BusinessLine | Free public feeds |
| FinBERT | Sentiment NLP model | Free via HuggingFace |

No paid APIs needed to start. For live trading, connect Zerodha Kite API.

---

## Next steps (advanced)

1. **Live trading**: Integrate Zerodha Kite API (`kiteconnect` package)
2. **Intraday**: Switch yfinance to 5min/15min data (limited to 60 days)
3. **Options**: Add NSE F&O OI data and PCR analysis
4. **GPU training**: Install CUDA PyTorch for 10× faster LSTM training
5. **Scheduling**: Use APScheduler to retrain weekly and predict daily at 9:15 AM
