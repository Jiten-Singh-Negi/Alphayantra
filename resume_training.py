"""
resume_training.py  — v3
─────────────────────────
Run after a crash where RF + XGB finished but TCN did not.
The checkpoint is saved — only TCN will be re-trained (~15-30 min).

Usage:
    cd D:\alphayantra
    python resume_training.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from data.fetcher import fetch_universe
from strategies.indicators import compute_indicators, IndicatorConfig
from models.ml_engine import AlphaYantraML

def main():
    logger.info("AlphaYantra v3 — Resume after crash")
    logger.info("RF + XGBoost checkpoint found — only TCN will be trained")

    logger.info("Fetching data (cached)...")
    stock_data = fetch_universe("nifty500", period="15y")
    logger.info(f"  {len(stock_data)} stocks fetched")

    cfg       = IndicatorConfig()
    processed = {}
    for i, (ticker, df) in enumerate(stock_data.items()):
        try:
            processed[ticker] = compute_indicators(df, cfg)
        except Exception as e:
            logger.warning(f"  Skip {ticker}: {e}")
        if i % 50 == 0 and i > 0:
            logger.info(f"  {i}/{len(stock_data)} indicators computed...")

    logger.info(f"  {len(processed)} stocks ready")

    model   = AlphaYantraML("default")
    # ── All v3 kwargs — no LSTM args anywhere ──────────────────────────
    metrics = model.resume_after_crash(
        all_stock_dfs      = processed,
        n_cv_folds         = 4,
        skip_tcn           = False,        # False = train the TCN
        tcn_max_samples    = 50_000,
        tcn_epochs         = 10,
        use_triple_barrier = True,
    )

    logger.info("\n Training complete!")
    logger.info(f"  Ensemble AUC:      {metrics.get('ensemble_auc', 'N/A')}")
    logger.info(f"  Walk-Forward AUC:  {metrics.get('cv_mean_auc', 'N/A')} "
                f"± {metrics.get('cv_std_auc', 'N/A')}")
    logger.info(f"  Sharpe Corr:       {metrics.get('xgb_reg_sharpe_corr', 'N/A')}")
    logger.info(f"  TCN AUC:           {metrics.get('tcn_auc', 'N/A')}")
    logger.info(f"  TCN included:      {metrics.get('tcn_included', False)}")
    logger.info(f"  Label type:        {metrics.get('label_type', 'triple_barrier')}")
    logger.info("\nRun: python run.py")

if __name__ == "__main__":
    main()
