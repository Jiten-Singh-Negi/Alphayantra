"""
run.py  — AlphaYantra v5 entry point
──────────────────────────────────────
CLEAN ROUTING — no more server clashes:

  python run.py --terminal         GPU dashboard (DearPyGui, personal machine)
  python run.py --headless         FastAPI backend only (cloud/server, no GUI)
  python run.py --train            Full retrain (RF + XGB + TCN)
  python run.py --fast-train       RF + XGB only (faster, less RAM)
  python run.py --backfill         Download Bhavcopy history (one-time, ~2-4h)
  python run.py --demo             Quick signal demo (no training needed)
  python run.py --check            Check all dependencies

Key fixes:
  ✅ --terminal and --headless are mutually exclusive — no port/DB conflicts
  ✅ SQLite WAL mode enabled before any --headless or --backfill writes
     so that a running --headless server doesn't block --backfill reads
  ✅ --terminal never starts uvicorn — DPG owns the main thread cleanly
  ✅ --headless never imports dearpygui — no GPU/display required on servers
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ── SQLite WAL mode helper ──────────────────────────────────────────────

def _enable_wal(db_path: str = "data/bhavcopy.db"):
    """
    Enable SQLite Write-Ahead Logging (WAL) mode on the Bhavcopy database.

    WAL allows concurrent readers while one writer is active, so the
    FastAPI server (reader) and the Bhavcopy updater (writer) can run
    simultaneously without 'database is locked' errors.

    Must be called BEFORE any other process opens the database.
    Safe to call multiple times (WAL persists across connections).
    """
    db = Path(db_path)
    if not db.exists():
        return   # DB will be created by bhavcopy.py with WAL mode
    try:
        import sqlite3
        con = sqlite3.connect(str(db))
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")  # faster without sacrificing safety
        con.commit()
        con.close()
        print(f"  SQLite WAL mode enabled: {db_path}")
    except Exception as e:
        print(f"  WAL mode skipped: {e}")


# ── Dependency check ────────────────────────────────────────────────────

def check_dependencies(mode: str = "all") -> bool:
    """
    mode: "all" | "terminal" | "headless"
    Terminal mode needs dearpygui. Headless mode doesn't.
    """
    print("Checking dependencies...")
    required = [
        ("pandas",   "pandas"),
        ("numpy",    "numpy"),
        ("yfinance", "yfinance"),
        ("ta",       "ta"),
        ("sklearn",  "scikit-learn"),
        ("xgboost",  "xgboost"),
        ("torch",    "torch"),
        ("loguru",   "loguru"),
        ("requests", "requests"),
        ("schedule", "schedule"),
        ("psutil",   "psutil"),
    ]
    headless_extras = [
        ("fastapi",    "fastapi"),
        ("uvicorn",    "uvicorn"),
        ("websockets", "websockets"),
        ("pydantic",   "pydantic"),
    ]
    terminal_extras = [
        ("dearpygui", "dearpygui"),
    ]
    optional = [
        ("transformers",   "transformers"),
        ("vaderSentiment", "vaderSentiment"),
        ("dotenv",         "python-dotenv"),
        ("pywt",           "PyWavelets"),
        ("feedparser",     "feedparser"),
    ]

    if mode in ("all", "headless"):
        required += headless_extras
    if mode in ("all", "terminal"):
        required += terminal_extras

    missing = []
    for mod, pkg in required:
        try:
            __import__(mod)
            print(f"  ✅  {pkg}")
        except ImportError:
            print(f"  ❌  {pkg}  MISSING")
            missing.append(pkg)
    for mod, pkg in optional:
        try:
            __import__(mod)
            print(f"  ✅  {pkg}  (optional)")
        except ImportError:
            print(f"  ⚪  {pkg}  (optional — not installed)")

    if missing:
        print(f"\nInstall missing:  pip install {' '.join(missing)}")
        return False
    print("\nAll required packages present ✅")
    return True


# ── Terminal mode ──────────────────────────────────────────────────────

def start_terminal(args):
    """
    Launch the DearPyGui GPU terminal.

    IMPORTANT: This function NEVER starts uvicorn or opens bhavcopy.db
    for writing.  The terminal reads market data from live NSE API and
    reads the trained model from disk — it does not write to any shared DB.
    DPG must own the main thread.  No asyncio event loops are started here.
    """
    _enable_wal()   # ensure any pre-existing DB allows concurrent reads

    # Dynamically import terminal only after all shared imports are done
    from terminal import main as terminal_main

    print("\n⟡ AlphaYantra GPU Terminal starting...")
    print("  Press SPACE to activate kill switch")
    print("  Press ESC  to deactivate kill switch")
    print("  Close window to exit\n")

    # Pass args through sys.argv for terminal.py's own argparse
    new_argv = [sys.argv[0]]
    if args.paper:
        new_argv.append("--paper")
    if args.universe:
        new_argv += ["--universe", args.universe]
    if args.no_ml:
        new_argv.append("--no-ml")
    if args.no_sentiment:
        new_argv.append("--no-sentiment")
    sys.argv = new_argv

    terminal_main()


# ── Headless (FastAPI) mode ─────────────────────────────────────────────

def start_headless(args):
    """
    Launch the FastAPI server only — no DearPyGui, no GPU required.
    Safe to run on a cloud server (AWS, DigitalOcean, GCP) without a display.

    SQLite WAL mode is enabled first so the Bhavcopy nightly updater
    (which runs inside scheduler/tasks.py at 18:30 IST) can write while
    FastAPI is concurrently reading option chain history.
    """
    _enable_wal()

    if not Path(".env").exists():
        Path(".env").write_text("TELEGRAM_BOT_TOKEN=\nTELEGRAM_CHAT_ID=\n")
        print("Created .env — add Telegram credentials to enable alerts")

    host = getattr(args, "host", "0.0.0.0")
    port = getattr(args, "port", 8000)

    print(f"\n⟡ AlphaYantra headless server starting...")
    print(f"  Dashboard:  http://{host}:{port}/dashboard")
    print(f"  API docs:   http://{host}:{port}/docs")
    print(f"  Press Ctrl+C to stop\n")

    reload_flag = getattr(args, "reload", False)
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", host,
        "--port", str(port),
    ]
    if reload_flag:
        cmd.append("--reload")

    subprocess.run(cmd)


# ── Training ────────────────────────────────────────────────────────────

def train_model(skip_tcn: bool = False):
    from data.fetcher import fetch_universe
    from strategies.indicators import compute_indicators, IndicatorConfig
    from models.ml_engine import AlphaYantraML

    mode = "RF + XGB only (fast)" if skip_tcn else "RF + XGB + TCN (full)"
    print(f"\nTraining: {mode}")
    if not skip_tcn:
        print("Tip: use --fast-train first if RAM < 16 GB, then resume_training.py")

    stock_data = fetch_universe("nifty500", period="15y")
    print(f"  {len(stock_data)} stocks fetched")

    cfg       = IndicatorConfig()
    processed = {}
    for i, (t, df) in enumerate(stock_data.items()):
        try:
            processed[t] = compute_indicators(df, cfg)
        except Exception:
            pass
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(stock_data)} indicators computed...")

    print(f"  {len(processed)} stocks ready — starting ML training...")
    model   = AlphaYantraML("default")
    metrics = model.train(
        all_stock_dfs      = processed,
        n_cv_folds         = 4,
        skip_tcn           = skip_tcn,
        tcn_max_samples    = 50_000,
        tcn_epochs         = 10,
        use_triple_barrier = True,
    )

    print(f"\nTraining complete! ✅")
    print(f"  Ensemble AUC:     {metrics.get('ensemble_auc', 'N/A')}")
    print(f"  Walk-Forward AUC: {metrics.get('cv_mean_auc', 'N/A')} ± {metrics.get('cv_std_auc', 'N/A')}")
    print(f"  Sharpe Corr:      {metrics.get('xgb_reg_sharpe_corr', 'N/A')}")
    print(f"  TCN:              {'included ✅' if metrics.get('tcn_included') else 'skipped (RAM limit)'}")
    print(f"  Label type:       {metrics.get('label_type', 'N/A')}")

    if not metrics.get("tcn_included") and not skip_tcn:
        print("\n  TCN auto-skipped (RAM).  Train later: python resume_training.py")

    print("\nRun: python run.py --terminal   (GPU dashboard)")
    print("     python run.py --headless    (API server)")


# ── Bhavcopy backfill ───────────────────────────────────────────────────

def backfill_bhavcopy():
    _enable_wal()   # enable WAL before writing

    from data.bhavcopy import BhavcopyScraper
    import datetime

    print("\nBhavcopy backfill — downloading NSE F&O history (2015 → today)")
    print("Takes 2-4 hours.  Progress saved — safe to Ctrl+C and resume.\n")

    s = BhavcopyScraper()
    n = s.download_range("2015-01-01", datetime.date.today().isoformat())

    print(f"\nDone: {n} days stored in data/bhavcopy.db")
    info = s.status()
    print(f"DB: {info['earliest']} → {info['latest']}  ({info['trading_days']} trading days)")
    print("\nNow retrain: python run.py --train")


# ── Demo ────────────────────────────────────────────────────────────────

def run_demo():
    from data.fetcher import fetch_ohlcv
    from strategies.indicators import compute_indicators, compute_signal_score, IndicatorConfig
    from options.chain import OptionsChain
    from feed.live_feed import QuoteFetcher

    print("\n=== DEMO: RELIANCE signal + NIFTY options ===")
    df  = fetch_ohlcv("RELIANCE", period="2y")
    cfg = IndicatorConfig()
    df  = compute_indicators(df, cfg)
    sig = compute_signal_score(df.iloc[-1], cfg)
    print(f"RELIANCE: {sig['signal']}  Score: {sig['score']}/100")
    print(f"  SL: ₹{sig.get('stop_loss', 0):.0f}   TP: ₹{sig.get('take_profit', 0):.0f}")

    try:
        q = QuoteFetcher().get_index_quote("NIFTY 50")
        if q:
            print(f"NIFTY 50: {q['ltp']:,.2f}  ({q['change_pct']:+.2f}%)")
    except Exception as e:
        print(f"Live quote: {e}")

    try:
        c = OptionsChain().get_chain("NIFTY", num_strikes=5)
        if c:
            print(f"NIFTY Options — PCR: {c.pcr:.2f} ({c.sentiment})   MaxPain: {c.max_pain:,.0f}")
    except Exception as e:
        print(f"Options chain: {e}")

    print("\nLaunch terminal: python run.py --terminal")
    print("Or API server:   python run.py --headless")


# ── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="AlphaYantra v5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --terminal                     # GPU dashboard (personal machine)
  python run.py --terminal --paper             # GPU dashboard + paper trading
  python run.py --terminal --universe nifty50  # Limit to Nifty 50 symbols
  python run.py --headless                     # FastAPI backend (cloud server)
  python run.py --headless --port 8080         # Custom port
  python run.py --train                        # Full model training
  python run.py --fast-train                   # RF+XGB only (less RAM)
  python run.py --backfill                     # One-time Bhavcopy download
  python run.py --demo                         # Quick demo, no training needed
  python run.py --check                        # Verify all dependencies
""",
    )

    # ── Mode flags (mutually exclusive) ────────────────────────────────
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--terminal",   action="store_true",
                            help="Launch GPU DearPyGui dashboard (requires display)")
    mode_group.add_argument("--headless",   action="store_true",
                            help="Launch FastAPI backend only (cloud/server mode)")
    mode_group.add_argument("--train",      action="store_true",
                            help="Full model training (RF + XGB + TCN)")
    mode_group.add_argument("--fast-train", action="store_true", dest="fast_train",
                            help="Fast training (RF + XGB only, less RAM)")
    mode_group.add_argument("--backfill",   action="store_true",
                            help="One-time Bhavcopy historical data download")
    mode_group.add_argument("--demo",       action="store_true",
                            help="Quick signal demo")
    mode_group.add_argument("--check",      action="store_true",
                            help="Check all dependencies")

    # ── Terminal-specific options ───────────────────────────────────────
    p.add_argument("--paper",          action="store_true",
                   help="Enable paper trading auto-execution in terminal")
    p.add_argument("--universe",       default="nifty50",
                   choices=["nifty50", "nifty500", "midcap150"],
                   help="Stock universe for live scanning")
    p.add_argument("--no-ml",          action="store_true", dest="no_ml",
                   help="Skip ML inference (faster startup, lower RAM)")
    p.add_argument("--no-sentiment",   action="store_true", dest="no_sentiment",
                   help="Skip FinBERT sentiment (saves ~500MB RAM)")

    # ── Headless-specific options ───────────────────────────────────────
    p.add_argument("--host",   default="0.0.0.0", help="API server host (default: 0.0.0.0)")
    p.add_argument("--port",   default=8000, type=int, help="API server port (default: 8000)")
    p.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (dev only)")

    args = p.parse_args()

    # ── Route to correct mode ───────────────────────────────────────────
    if args.check:
        mode = "terminal" if args.terminal else "headless" if args.headless else "all"
        ok   = check_dependencies(mode)
        sys.exit(0 if ok else 1)

    elif args.terminal:
        if not check_dependencies("terminal"):
            sys.exit(1)
        start_terminal(args)

    elif args.headless:
        if not check_dependencies("headless"):
            sys.exit(1)
        start_headless(args)

    elif args.train:
        check_dependencies("all")
        train_model(skip_tcn=False)

    elif args.fast_train:
        check_dependencies("all")
        train_model(skip_tcn=True)

    elif args.backfill:
        backfill_bhavcopy()

    elif args.demo:
        run_demo()

    else:
        # No flag → print help
        p.print_help()
        print("\nQuick start:")
        print("  python run.py --terminal     # GPU dashboard")
        print("  python run.py --headless     # API server")
