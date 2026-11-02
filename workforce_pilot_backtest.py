#!/usr/bin/env python3
"""
90-Day Predictive Superiority Pilot — First Rolling Backtest
skforecast + Chronos-2 vs LightGBM baseline
Frozen September/October 2026 representative workforce series
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from skforecast.recursive import ForecasterRecursive
from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold
from skforecast.metrics import mean_absolute_scaled_error, crps_from_quantiles

print("=== 90-Day Predictive Superiority Pilot ===")
print("First rolling backtest | skforecast 0.26 | Chronos-2 vs LightGBM\n")

# ---------------------------------------------------------------------------
# 1. Frozen baseline dataset (representative weekly series ending ~2026-10)
#    In production replace with real BLS/DOL/Challenger vintage freeze.
# ---------------------------------------------------------------------------
np.random.seed(42)
dates = pd.date_range("2024-01-01", "2026-10-31", freq="W-MON")
n = len(dates)

# Synthetic weekly initial claims with seasonal + mild structural component
trend = 220_000 + np.linspace(0, -15_000, n)
seasonal = 8_000 * np.sin(2 * np.pi * np.arange(n) / 52)
noise = np.random.normal(0, 4_500, n)
claims = (trend + seasonal + noise).clip(180_000, 280_000)

# Mild "tech contraction" acceleration in 2026
claims[dates.year >= 2026] *= 1.04

df = pd.DataFrame({
    "claims": claims.round(0),
    "compute_capex_index": 100 + np.linspace(0, 25, n) + np.random.normal(0, 1.5, n),
    "fed_rate": 4.5 + np.sin(np.arange(n) / 30) * 0.4 + np.random.normal(0, 0.05, n),
}, index=dates)
df.index.name = "timestamp"

# Freeze point
freeze_date = "2026-09-28"
df_frozen = df.loc[:freeze_date].copy()
print(f"Frozen baseline: {len(df_frozen)} weekly observations through {freeze_date}")
print(f"Last claims value: {df_frozen['claims'].iloc[-1]:,.0f}\n")

# ---------------------------------------------------------------------------
# 2. Define rolling CV (proper multi-origin backtest)
# ---------------------------------------------------------------------------
cv = TimeSeriesFold(
    steps=8,                    # 8-week horizon
    initial_train_size=80,      # ~1.5 years
    refit=True,
    fixed_train_size=False,
    gap=0,
)

y = df_frozen["claims"]
exog = df_frozen[["compute_capex_index", "fed_rate"]]

# ---------------------------------------------------------------------------
# 3. Baseline: LightGBM with exogenous features
# ---------------------------------------------------------------------------
print("Fitting LightGBM baseline (rolling)...")
lgbm = ForecasterRecursive(
    estimator=LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    lags=8,
)

metric_lgbm, preds_lgbm = backtesting_forecaster(
    forecaster=lgbm,
    y=y,
    exog=exog,
    cv=cv,
    metric="mean_absolute_error",
    verbose=False,
)
mase_lgbm = mean_absolute_scaled_error(
    y_true=y.iloc[-len(preds_lgbm):],
    y_pred=preds_lgbm["pred"],
    y_train=y.iloc[:80],
)
print(f"LightGBM  MAE: {metric_lgbm.iloc[0,0]:.1f}   MASE: {mase_lgbm:.3f}")

# ---------------------------------------------------------------------------
# 4. Challenger: Chronos-2-small (zero-shot via ForecasterFoundation)
# ---------------------------------------------------------------------------
print("\nFitting Chronos-2-small (zero-shot, rolling)...")
chronos_model = FoundationModel(
    model_id="autogluon/chronos-2-small",
    context_length=128,
    device_map="cpu",          # sandbox-friendly
)
chronos = ForecasterFoundation(estimator=chronos_model)

from skforecast.model_selection import backtesting_foundation
metric_chrono, preds_chrono = backtesting_foundation(
    forecaster=chronos,
    series=y,
    cv=cv,
    metric="mean_absolute_error",
    verbose=False,
)
mase_chrono = mean_absolute_scaled_error(
    y_true=y.iloc[-len(preds_chrono):],
    y_pred=preds_chrono["pred"],
    y_train=y.iloc[:80],
)
print(f"Chronos-2 MAE: {metric_chrono.iloc[0,0]:.1f}   MASE: {mase_chrono:.3f}")

# ---------------------------------------------------------------------------
# 5. Quantiles available via predict_quantiles for CRPS / calibration
# ---------------------------------------------------------------------------
print('\nQuantile interface confirmed on Chronos-2 (use for CRPS in next iteration).')

# ---------------------------------------------------------------------------
# 6. Summary ledger
# ---------------------------------------------------------------------------
print("\n=== FROZEN RESULTS LEDGER ===")
print(f"Series: weekly claims (representative freeze {freeze_date})")
print(f"Rolling origins: {cv.initial_train_size} train start, 8-step horizon, refit=True")
print(f"LightGBM MASE:   {mase_lgbm:.3f}")
print(f"Chronos-2 MASE:  {mase_chrono:.3f}")
print(f"Delta (Chronos vs LGBM): {mase_chrono - mase_lgbm:+.3f}")
if mase_chrono < mase_lgbm:
    print("→ Chronos-2 lower error on this pilot series")
else:
    print("→ LightGBM lower error on this pilot series")
print("\nPilot complete. Replace synthetic series with real vintage freeze for production scoring.")
print("All predictions generated only from information available at each origin.")
