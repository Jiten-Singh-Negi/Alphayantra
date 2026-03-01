# AlphaYantra v8.3 — Complete Setup & Operations Runbook

---

## The Short Answer

Yes, you need to **retrain** and **re-backfill** (once). The feature matrix changed from 62 → 79
features. Any old saved model will reject the new data shape. Here is the exact order of operations.

---

## Step 0 — Deploy the New Code

Replace your existing folder with the v8.3 code.

```bash
# Back up your old installation first
mv alphayantra  alphayantra_v8_2_backup

# Unpack v8.3
tar -xzf alphayantra_v8_3.tar.gz
cd alphayantra_v8_3
```

Delete any old saved model files — they are incompatible with the new feature count:

```bash
# Delete the old trained model (RF, XGB, LGB, TCN weights)
rm -rf models/saved/*

# Your Bhavcopy database (data/bhavcopy.db) is FINE — keep it
# Your .env file with Telegram credentials — keep it
# Your risk state (data/risk_state.json) — keep it
```

---

## Step 1 — Verify Your Environment

```bash
python run.py --check
```

Everything must be ✅. If anything is missing:

```bash
pip install pandas numpy yfinance ta scikit-learn xgboost lightgbm torch \
            loguru requests schedule psutil fastapi uvicorn websockets \
            pydantic transformers vaderSentiment python-dotenv feedparser
```

For the GPU dashboard (personal machine only):

```bash
pip install dearpygui
```

---

## Step 2 — Backfill Historical Data (One Time)

This downloads 10 years of NSE F&O Bhavcopy data. It is the raw material the model uses to
compute options features (PCR, IV rank, max pain). It takes **2–4 hours** and only needs to
run once. If your `data/bhavcopy.db` already exists from a previous version, **skip this step**.

```bash
python run.py --backfill
```

Progress is saved every session, so if it crashes you can just rerun — it will pick up where
it left off.

---

## Step 3 — Train the Model

This is the big one. The model sees all 79 features for the first time and runs purged
walk-forward cross-validation across 500 Nifty stocks × 15 years of data.

### If you have ≥ 16 GB RAM — Full train (RF + XGB + LGB + TCN):

```bash
python run.py --train
```

Duration: **45–90 minutes** depending on CPU/GPU.

### If you have < 16 GB RAM — Fast train first:

```bash
python run.py --fast-train
```

This trains RF + XGB + LGB only (skips TCN). Takes ~20 minutes. Then, separately:

```bash
python resume_training.py
```

This resumes from the checkpoint and only trains the TCN (~15–30 more minutes).

### What to look for when training finishes:

```
Training complete! ✅
  Ensemble AUC:     0.67        ← target > 0.62 (was 0.48 before v8)
  Walk-Forward AUC: 0.64 ± 0.03 ← std < 0.05 means consistent
  Sharpe Corr:      0.31        ← positive = model ranks stocks by real Sharpe
  TCN:              included ✅
  Label type:       trend_quality
```

If AUC is below 0.58, something is wrong with your data feed — check yfinance connectivity.

---

## Step 4 — First Run (Verify Everything Works)

Before going live, run the demo to verify signal generation works end-to-end:

```bash
python run.py --demo
```

You should see a RELIANCE signal, a live NIFTY quote, and options chain PCR. If anything fails,
check your internet connection (NSE API access) and that your `.env` has Telegram credentials.

---

## Step 5 — Go Live

### Option A — With the GPU Dashboard (personal machine, requires display):

```bash
python run.py --terminal --paper
```

`--paper` enables paper trading auto-execution (no real money). Remove it only when you are
confident in the signals.

To use a specific universe:

```bash
python run.py --terminal --universe nifty50   # faster, focused
python run.py --terminal --universe nifty500  # full universe scan
```

### Option B — Headless API Server (cloud / no display):

```bash
python run.py --headless
```

Then open `http://your-server-ip:8000/dashboard` in a browser. All the same data, no GPU needed.

Custom port:

```bash
python run.py --headless --port 8080
```

---

## Step 6 — Keep It Running (Automated Everything)

Once the headless server is running, the **built-in scheduler does everything automatically**.
You do not need to touch it again. Here is what runs without your intervention:

| Time (IST)       | What Happens                                              |
|------------------|-----------------------------------------------------------|
| **09:00**        | Pre-market scan → Telegram alert with top 10 signals      |
| **09:15**        | Market opens → live feed starts, daily risk state resets  |
| **Every 5 min**  | Signal refresh across Nifty 50 during market hours        |
| **15:00**        | Final pre-close scan → Telegram EOD signals               |
| **15:30**        | Market closes → live feed stops, daily P&L logged         |
| **18:00**        | Full EOD report → Telegram (P&L, open positions, risk)    |
| **18:30**        | Bhavcopy incremental update (downloads today's F&O data)  |
| **Sunday 23:00** | Full model retrain on all 500 stocks (takes ~1 hour)      |
| **1st of month** | Feature importance report → Telegram (which signals work) |

**You only ever manually retrain when you change the code.** After that, the Sunday retrain
keeps the model fresh automatically.

---

## Running on a Server 24/7

### Using systemd (Linux server — recommended):

Create `/etc/systemd/system/alphayantra.service`:

```ini
[Unit]
Description=AlphaYantra Trading System
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/alphayantra_v8_3
ExecStart=/usr/bin/python3 run.py --headless
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable alphayantra
sudo systemctl start alphayantra

# Check it's running
sudo systemctl status alphayantra

# See live logs
sudo journalctl -u alphayantra -f
```

### Using screen (simpler, any Linux):

```bash
screen -S alphayantra
cd alphayantra_v8_3
python run.py --headless
# Ctrl+A then D to detach (keeps running)
# screen -r alphayantra to re-attach
```

### On Windows — Task Scheduler:

```
Program: C:\Python311\python.exe
Arguments: C:\alphayantra_v8_3\run.py --headless
Start in: C:\alphayantra_v8_3
Run: Whether user is logged on or not
Trigger: At startup
```

---

## Telegram Setup (Highly Recommended)

Without Telegram you miss all the pre-market alerts and EOD reports. Takes 2 minutes to set up:

1. Open Telegram, search for **@BotFather**
2. Send `/newbot` → give it a name → copy the **bot token**
3. Start a chat with your new bot, then visit:
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Find `"chat":{"id":12345678}` — that is your **chat ID**
5. Edit `.env` in the project folder:

```
TELEGRAM_BOT_TOKEN=1234567890:ABCdef...
TELEGRAM_CHAT_ID=12345678
```

Restart the server. You will now get:
- 📊 Pre-market signal alerts at 09:00
- 📈 EOD report at 18:00
- ✅/❌ Paper trade execution notifications
- 🧠 Weekly retrain completion confirmation
- 📅 Monthly feature importance report

---

## After Any Future Code Change

The order is always:

```
1. Stop the server
2. Update the code
3. Delete models/saved/*   (if features changed)
4. python run.py --train   (retrain)
5. python run.py --headless (restart)
```

If you only changed non-feature code (bug fixes, risk parameters, UI) you can skip steps 3–4
and just restart the server.

---

## Monitoring & Health Checks

While the server is running, these endpoints tell you everything:

| URL                                    | What It Shows                          |
|----------------------------------------|----------------------------------------|
| `http://localhost:8000/dashboard`      | Full web UI                            |
| `http://localhost:8000/health`         | System health + uptime                 |
| `http://localhost:8000/train/status`   | Training progress (if retraining)      |
| `http://localhost:8000/paper/report`   | Paper trading P&L report               |
| `http://localhost:8000/signals/nifty50`| Current live signals                   |
| `http://localhost:8000/docs`           | Full interactive API documentation     |

---

## Quick Reference — All Commands

```bash
python run.py --check           # verify environment
python run.py --backfill        # one-time data download (2-4h)
python run.py --train           # full retrain (RF+XGB+LGB+TCN)
python run.py --fast-train      # fast retrain (skip TCN)
python resume_training.py       # resume after crash (TCN only)
python run.py --demo            # verify signals work
python run.py --terminal        # GPU dashboard (personal machine)
python run.py --terminal --paper           # GPU dashboard + paper trading
python run.py --terminal --universe nifty50  # limit scan to 50 stocks
python run.py --headless        # API server (cloud/server)
python run.py --headless --port 8080       # custom port
```
