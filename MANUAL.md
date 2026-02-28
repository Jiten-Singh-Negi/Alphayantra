# AlphaYantra v8 — Complete User Manual

> **What this system does:**  
> AlphaYantra is a quantitative trading research platform for Indian equities (NSE).  
> It trains an ML model (Random Forest + LightGBM + XGBoost + TCN neural net) on 15 years of Nifty data,  
> generates live BUY/SELL signals during market hours, simulates trades via a paper broker,  
> and sends Telegram alerts. It does **not** place real orders automatically — you must click.

> **v8 Model Quality Overhaul:**  
> The v7 model scored AUC 0.48 (worse than random). v8 fixes the three root causes:  
> 1. **Unadjusted price trap** — RSI/MACD/Bollinger Bands were exploding on every stock split. Now fixed with dual-track pricing (adjusted for indicators, raw for options only).  
> 2. **Broken DWT denoising** — PyWavelets with level=3 on 16-day windows produced mathematical garbage. Replaced with Savitzky-Golay polynomial smoothing.  
> 3. **Random labels** — Triple-barrier at 2:1 ratio hits ~41% positives by pure chance. Replaced with Trend-Quality labels (ATR-normalized forward return + volume confirmation).  
> Expected AUC: **0.60–0.68** (up from 0.48).

> **v8 Terminal: Stock Search**  
> The search box now has **autocomplete** — type any partial NSE symbol (e.g. "HDFC") to see suggestions. Click any suggestion or press Enter to add it to your chart.

---

## CRITICAL: Delete old data cache before retraining

The v7 cache used wrong (unadjusted) price data. v8 uses dual-track adjusted prices.
**You must delete the old cache** or you'll retrain on garbage data:

```bash
# Windows
rmdir /s /q data\cache

# Mac/Linux
rm -rf data/cache
```

Then retrain:
```bash
python run.py --train
```

---

## Part 1 — First-Time Setup (do this once)

### Step 1: Install Python dependencies

```bash
pip install dearpygui pandas numpy yfinance ta scikit-learn xgboost lightgbm torch \
            fastapi uvicorn loguru requests feedparser pydantic scipy \
            psutil schedule websockets python-dotenv transformers sentencepiece
```

---

### Step 2: Set up Telegram alerts (10 minutes)

1. Open Telegram, search `@BotFather`, send `/newbot`
2. Name your bot anything (e.g. `AlphaYantra Bot`)
3. Copy the **BOT_TOKEN** BotFather gives you (looks like `7291837465:AAG...`)
4. Start a chat with your new bot — send any message to it
5. In your terminal:

```bash
cd alphayantra
python -c "
from monitor.telegram import TelegramMonitor
t = TelegramMonitor(bot_token='YOUR_BOT_TOKEN_HERE')
t.get_chat_id()
"
```

6. It prints something like: `Your Chat ID: 812345678`
7. Create `.env` in the project root:

```
TELEGRAM_BOT_TOKEN=7291837465:AAG...your_token...
TELEGRAM_CHAT_ID=812345678
```

**Test it:**
```bash
python -c "
from monitor.telegram import TelegramMonitor
t = TelegramMonitor()
t.send_message('✅ AlphaYantra connected!')
"
```

---

### Step 3: Download Bhavcopy historical options data (one-time, 2–4 hours)

This downloads NSE F&O daily settlement data from 2015 to today.  
This data teaches the model what PCR and IV-Rank values look like during bull/bear markets.

```bash
python run.py --backfill
```

> **Safe to interrupt.** Progress is saved. Re-run to resume from where it stopped.  
> Skip this if you want to start quickly — the model works without it, but PCR/IV features will be neutral defaults.

---

### Step 4: Train the ML model (30–60 minutes)

```bash
python run.py --train
```

This downloads 15 years of Nifty 500 data (including historically failed companies to avoid survivorship bias), computes 80+ technical indicators, and trains RF + XGB + TCN.

**Lower RAM machine? Use fast-train (skips TCN neural net):**
```bash
python run.py --fast-train    # ~10 minutes, needs ~4GB RAM
```

**After fast-train, optionally add TCN separately when RAM is free:**
```bash
python resume_training.py
```

You'll see output like:
```
Training complete! ✅
  Ensemble AUC:     0.71
  Walk-Forward AUC: 0.68 ± 0.03
  Sharpe Corr:      0.42
  TCN:              included ✅
```

> AUC of 0.68–0.72 is excellent for equity prediction. Above 0.75 likely means overfitting.

---

## Part 2 — Daily Usage

### Option A: GPU Terminal (your personal Windows/Mac/Linux machine)

```bash
python run.py --terminal              # view-only (watch signals, no auto-trades)
python run.py --terminal --paper      # paper trading mode (simulates trades automatically)
```

**Terminal controls:**
| Key / Control | Action |
|---|---|
| `SPACE` | Emergency kill — flatten ALL paper positions instantly |
| `ESC` | Resume trading after kill switch / close search suggestions |
| Symbol dropdown | Switch the price chart to a different stock |
| **Search box** | Type any NSE symbol or partial name to search (e.g. "HDFC" → shows HDFCBANK, HDFCLIFE, HDFCAMC). Click a suggestion or press Enter to add |
| Kelly slider | Adjust position sizing (0.1x = tiny, 1.0x = full Kelly formula) |
| Trail stop slider | Trailing stop distance (% from peak price) |

---

### Option B: Headless API Server (cloud/VPS without a screen)

```bash
python run.py --headless              # port 8000
python run.py --headless --port 8080  # custom port
```

Open your browser:
- **Dashboard:** `http://your-server-ip:8000/dashboard`
- **API docs:**   `http://your-server-ip:8000/docs`

---

## Part 3 — How to Read Signals

### Signal Meanings

| Signal | Meaning | What to do |
|--------|---------|-----------|
| `STRONG BUY` | Model ≥70% confident, 3+ indicators confirm | Consider entering long position |
| `BUY` | Model 55–70% confident | Weaker signal — smaller size |
| `HOLD` | Model uncertain | Do nothing |
| `SELL` | 55–70% confident bearish | Consider exiting long |
| `STRONG SELL` | ≥70% confident bearish | Exit quickly |

### ML Signal Matrix (in terminal, Column 2)

| Column | What it means |
|--------|-------------|
| Symbol | Stock ticker |
| Signal | BUY/SELL/HOLD classification |
| P% | Probability of price going up in next 10 days (>60% = bullish) |
| Sharpe | Expected Sharpe ratio of trade (>0.5 = good) |
| Ret% | Expected return % over holding period |
| Time | When signal was generated |

### Market Gauges (Column 2, top section)

| Gauge | Interpretation |
|-------|---------------|
| **India VIX < 14** | Low fear — good for momentum trades |
| **India VIX 14–20** | Normal market |
| **India VIX > 20** | High fear — reduce position sizes |
| **India VIX > 28** | Extreme fear — stay out or hedge only |
| **PCR < 0.8** | More calls than puts — market is optimistic (bullish) |
| **PCR > 1.2** | More puts than calls — hedging/fear (bearish) |
| **PCR 0.8–1.2** | Neutral |
| **Sentiment > 60** | Positive news flow for this stock |
| **Sentiment < 40** | Negative news flow |

### Regime Indicator

| Regime | Meaning |
|--------|---------|
| `▲ TRENDING (Risk-On)` | Strong trend, ADX > 25, low VIX — highest weight to TCN |
| `↔ MEAN REVERTING` | Range-bound, ADX < 15 — use contrarian signals |
| `⚠ HIGH VOL CHOP` | VIX > 28, ADX uncertain — reduce all sizes by 50% |

---

## Part 4 — Paper Trading Explained

When you run `--paper`, the system:

1. **Reads ML signals** every 60 seconds for up to 20 stocks
2. **Filters** to only STRONG BUY / STRONG SELL with ≥70% confidence
3. **Checks risk limits** (max position size, daily loss limit, kill switch)
4. **Sizes the position** using the Kelly Criterion × your Kelly slider setting
5. **Simulates a market order** with 0.05% bid/ask spread
6. **Deducts real Indian taxes**: STT 0.1% on sell, SEBI 0.0001%, stamp duty 0.015% on buy, exchange fees
7. **Updates P&L live** in the terminal's "Paper Portfolio" panel
8. **Sends Telegram alert** for every fill, stop hit, and take profit

**To see your paper performance:**
```bash
curl http://localhost:8000/paper/portfolio
```

Or just watch the terminal's Column 3 (right panel).

---

## Part 5 — API Reference (for headless mode)

All endpoints return JSON. Full docs at `/docs`.

### Scan for signals
```
POST /predict
{
  "universe": "nifty50",        // "nifty50" | "nifty500" | "midcap150"
  "top_n": 10,                  // how many to return
  "strategy": {"name": "momentum"}
}
```

### Get one stock's indicators
```
GET /indicators/RELIANCE
```

### Run a backtest
```
POST /backtest
{
  "universe": "nifty50",
  "start_date": "2020-01-01",
  "end_date": "2024-01-01",
  "initial_capital": 1000000,
  "strategy": {"name": "momentum"}
}
```

### Live quotes
```
GET /feed/quotes
```

### Options chain (live)
```
GET /options/chain?symbol=NIFTY&num_strikes=10
```

### Kill switch (emergency)
```
POST /risk/kill-switch/activate?reason=manual
POST /risk/kill-switch/deactivate
```

---

## Part 6 — Automatic Daily Schedule

The scheduler (`scheduler/tasks.py`) runs these automatically when you have the system running:

| Time (IST) | Task |
|-----------|------|
| 9:00 AM | Pre-market signal scan → Telegram |
| 9:05 AM | PCR ring buffer writer starts |
| 9:15 AM | Market open — live feed active |
| Every minute | PCR + IV Rank written to ring buffer DB |
| Every 60s | ML inference on watchlist |
| 3:00 PM | EOD signal scan → Telegram |
| 3:30 PM | Market close |
| 6:30 PM | Download today's Bhavcopy (incremental) |
| Sunday 11 PM | Weekly model retrain → Telegram when done |

---

## Part 7 — Recommended Workflow

### Day 1 (setup)
1. Run `python run.py --backfill` overnight
2. While it runs, set up Telegram (Step 2 above)

### Day 2 (first trade day)
1. Run `python run.py --train` (morning — takes 30–60 min)
2. At 9:00 AM run `python run.py --terminal --paper`
3. Watch the signals in Column 2
4. Use **view-only** for 1–2 weeks to build confidence before trusting any signal

### Weekly
- System retrains automatically Sunday 11 PM — you don't need to do anything
- Check Telegram for the retrain completion message
- If retrain notification never arrives, run `python run.py --train` manually

### If you want to go live (real money)
1. Open a Zerodha/Upstox demat account
2. Get the API key + access token
3. In `broker/paper_trader.py`, replace `_paper_fill()` with your broker's order API
4. Every other part of the system (risk manager, P&L, Telegram) stays identical

---

## Part 8 — Troubleshooting

| Problem | Fix |
|---------|-----|
| `No model found — run training first` | Run `python run.py --train` |
| **Model AUC is 0.48-0.50 (coin flip)** | You ran the old v7 model. Delete `data/cache/` folder and `models/saved/` folder, then retrain with v8 |
| **AUC 0.50-0.55 after v8 retrain** | Normal first run — LightGBM may not be installed. Run `pip install lightgbm` then retrain |
| **DWT boundary effects warning** | Harmless in v8 — DWT was removed. If you see this, you're running old code. Replace with v8 |
| Terminal window freezes | Restart — don't click/drag the window during high-frequency ticks |
| `NSE 429` in logs | NSE is rate-limiting you. Normal — system backs off automatically |
| `database is locked` | Run `python run.py --headless` then `python run.py --terminal` separately, not the same command |
| Telegram alerts not arriving | Check `.env` has correct token and chat_id. Test with the snippet in Step 2 |
| VIX shows 15.0 always | yfinance connection issue — check internet. System uses neutral default |
| PCR shows 1.0 always | Run Bhavcopy backfill, or wait 10 minutes for PCRWriterThread to populate |
| Search autocomplete doesn't show | Type at least 2 characters. All 200+ NSE symbols are in the autocomplete list |
| GPU terminal very slow | Add `--no-ml --no-sentiment` flags — skips heavy compute threads |

---

## Part 9 — File Map (where everything lives)

```
alphayantra/
├── terminal.py          GPU dashboard (DearPyGui) — with stock search + autocomplete
├── run.py               Entry point (--terminal / --headless / --train)
├── api/main.py          FastAPI REST server
├── models/ml_engine.py  RF + LightGBM + XGB + TCN model (v8: trend-quality labels)
├── feed/live_feed.py    NSE live quote polling
├── strategies/
│   ├── indicators.py    80+ technical indicators (uses adjusted prices in v8)
│   └── denoising.py     Savitzky-Golay + Kalman denoising (DWT removed in v8)
├── options/chain.py     Options chain + PCR ring buffer
├── broker/
│   └── paper_trader.py  Virtual broker (with full STT/GST/stamp duty calculation)
├── risk/manager.py      Position limits, kill switch, daily loss limit
├── news/sentiment.py    FinBERT news sentiment (cached)
├── monitor/telegram.py  Telegram alerts (batched, rate-limit safe)
├── data/
│   ├── fetcher.py       Dual-track OHLCV (adjusted for indicators + raw for options)
│   └── bhavcopy.py      NSE F&O options history scraper
├── backtest/engine.py   Full backtest with charges, slippage, VIX regime
├── scheduler/tasks.py   Automated daily/weekly task runner
└── MANUAL.md            This file
```

---

## Appendix — v8 Model Improvements Explained

### Why was v7 AUC 0.48 (worse than random)?

Three compounding bugs made the model learn pure noise:

**Bug 1: Unadjusted price trap**
When `auto_adjust=False`, a 1:1 bonus issue causes a genuine 50% price drop on the historical chart. RSI plunges to 2-5 (extreme oversold signal), MACD fires a massive fake death cross, Bollinger Bands explode. The model saw thousands of these phantom "crashes" per stock across 15 years and learned nothing real.

**Fix:** Dual-track pricing. `auto_adjust=True` for all indicators. `auto_adjust=False` only for the `Close_Raw` column used to match Bhavcopy options strikes.

**Bug 2: DWT mathematical garbage**
`pywt.wavedec(window=16, level=3)` requires at least 57 samples (2³ × 7 + 1). With window=16, PyWavelets fills with boundary effects that completely overwhelm the signal. The "denoised" prices were actually scrambled prices fed as features.

**Fix:** Savitzky-Golay polynomial smoothing (scipy). No minimum window requirement, no boundary effects, analytic derivatives for momentum/acceleration features.

**Bug 3: Random labels**
Triple-barrier with take-profit=2×ATR, stop-loss=1×ATR on equity data: hitting the take-profit barrier first is a ~33-41% probability by pure random walk (provable from Brownian motion hitting probabilities). The model tried to predict something that was essentially random → AUC ≈ theoretical random = 0.50.

**Fix:** Trend-Quality labels: Label=1 only if (a) normalized forward return > threshold, (b) the move is sustained at midpoint, (c) volume confirms. This correlates with actual momentum, not random barrier geometry.

**New models added:**
- **LightGBM**: leaf-wise tree growth outperforms XGBoost on financial tabular data with sparse regime features. ~0.03 AUC improvement typically.
- **Stacking meta-learner**: logistic regression trained on out-of-fold predictions learns the optimal ensemble weights from data, instead of hardcoded 30/45/25 fixed weights.

*AlphaYantra v8 — Model rebuilt from scratch. Happy trading.*
