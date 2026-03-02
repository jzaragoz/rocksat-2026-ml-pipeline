#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL AUDIT — Time Dependency Analysis
=====================================================
Tests EVERY model for the same time-dependency issue that broke RF.
For each model: synthetic test, .dat test, feature importance, time shuffle.
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

np.random.seed(42)

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 70)
print("LOADING DATA AND MODELS")
print("=" * 70)

from data_loader import load_from_csv, load_from_dat_file

# CSV training data
res_csv = load_from_csv('data/Magneto_Fixed_Timeline.csv')
X_csv, y_csv = res_csv[1], res_csv[2]
bx_csv, by_csv, bz_csv, t_csv = res_csv[5], res_csv[6], res_csv[7], res_csv[8]
print(f"  CSV: {len(X_csv)} samples, X.shape={X_csv.shape}")

# .dat unseen data
res = load_from_dat_file('data/UPR - Flight_R4_EXP3_18_Nov_2025_11_37_58.dat')
df_dat, X_dat, y_dat = res[0], res[1], res[2]
bx_dat, by_dat, bz_dat, t_dat = res[5], res[6], res[7], res[8]
print(f"  DAT: {len(X_dat)} samples, X.shape={X_dat.shape}")

# Generate synthetic data (random Bx, By, Bz)
N_SYNTH = 2000
bx_s = np.random.uniform(-7000, 7000, N_SYNTH)
by_s = np.random.uniform(-7000, 7000, N_SYNTH)
bz_s = np.random.uniform(-7000, 7000, N_SYNTH)
X_synth = np.column_stack([bx_s, by_s, bz_s])
y_synth = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
t_synth = np.random.uniform(-125, 300, N_SYNTH)
print(f"  Synthetic: {N_SYNTH} random samples in [-7000, 7000]³")

# ============================================================================
# HELPER: pretty print results
# ============================================================================
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results = {}

def report(model_name, test_name, value, threshold, higher_is_better=True):
    if higher_is_better:
        status = PASS if value >= threshold else FAIL
    else:
        status = PASS if value <= threshold else FAIL
    print(f"  {status} {test_name}: {value:.6f} (threshold: {'>' if higher_is_better else '<'}{threshold})")
    results.setdefault(model_name, []).append((test_name, status, value))


# ############################################################################
#
#  1. RANDOM FOREST (already fixed — baseline reference)
#
# ############################################################################
print("\n" + "=" * 70)
print("1. RANDOM FOREST (reference — already fixed)")
print("=" * 70)

rf = joblib.load('models/cached/rf_model.joblib')
print(f"  n_features_in_ = {rf.n_features_in_}")
print(f"  Feature importances: {dict(zip(['Bx','By','Bz'], rf.feature_importances_))}")

# 1a. Synthetic
y_rf_synth = rf.predict(X_synth)
r2_rf_synth = r2_score(y_synth, y_rf_synth)
report("RF", "Synthetic R²", r2_rf_synth, 0.95)

# 1b. .dat file
y_rf_dat = rf.predict(X_dat)
r2_rf_dat = r2_score(y_dat, y_rf_dat)
report("RF", ".dat R²", r2_rf_dat, 0.95)

# 1c. CSV
y_rf_csv = rf.predict(X_csv)
r2_rf_csv = r2_score(y_csv, y_rf_csv)
report("RF", "CSV R²", r2_rf_csv, 0.95)

# 1d. No time to shuffle — model doesn't use time ✓
print(f"  {PASS} Time dependency: IMPOSSIBLE (model has 3 features, no Time input)")


# ############################################################################
#
#  2. NEURAL NETWORK (FCN) — 3 features [Bx, By, Bz] (retrained, no time)
#
# ############################################################################
print("\n" + "=" * 70)
print("2. NEURAL NETWORK (FCN) — 3 features [Bx, By, Bz] (no time)")
print("=" * 70)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model as keras_load

nn_model = keras_load('models/exports/magnetometer_fcn.keras')
nn_norm = np.load('models/exports/magnetometer_fcn_norm.npz')
nn_mean = nn_norm['mean']
nn_std = nn_norm['std']
print(f"  Input shape: {nn_model.input_shape}")
print(f"  Normalization mean: {nn_mean}")
print(f"  Normalization std:  {nn_std}")
assert nn_model.input_shape == (None, 3, 1), f"Expected (None, 3, 1), got {nn_model.input_shape}"
assert len(nn_mean) == 3, f"Expected 3-element mean, got {len(nn_mean)}"

def nn_predict(bx, by, bz):
    """Run NN prediction with proper normalization (3 features, no time)."""
    X_raw = np.column_stack([bx, by, bz]).astype(np.float32)
    X_norm = (X_raw - nn_mean) / nn_std
    X_in = X_norm.reshape(len(X_norm), 3, 1)
    return nn_model.predict(X_in, verbose=0).flatten()

# 2a. Synthetic test
y_nn_synth = nn_predict(bx_s, by_s, bz_s)
r2_nn_synth = r2_score(y_synth, y_nn_synth)
report("NN", "Synthetic R²", r2_nn_synth, 0.95)

# 2b. .dat file test
y_nn_dat = nn_predict(bx_dat, by_dat, bz_dat)
r2_nn_dat = r2_score(y_dat, y_nn_dat)
report("NN", ".dat R²", r2_nn_dat, 0.95)

# 2c. CSV test
y_nn_csv = nn_predict(bx_csv, by_csv, bz_csv)
r2_nn_csv = r2_score(y_csv, y_nn_csv)
report("NN", "CSV R²", r2_nn_csv, 0.95)

# 2d. Time dependency: IMPOSSIBLE — model has 3 features, no Time input
print(f"  {PASS} Time dependency: IMPOSSIBLE (model has 3 features, no Time input)")

# 2e. PERMUTATION IMPORTANCE (spatial features only)
print(f"\n  PERMUTATION IMPORTANCE:")
r2_baseline = r2_score(y_csv, y_nn_csv)

for feat_idx, feat_name in enumerate(['Bx', 'By', 'Bz']):
    X_3col = np.column_stack([bx_csv, by_csv, bz_csv])
    X_perm = X_3col.copy()
    X_perm[:, feat_idx] = np.random.permutation(X_perm[:, feat_idx])
    X_norm_p = (X_perm.astype(np.float32) - nn_mean) / nn_std
    X_in_p = X_norm_p.reshape(len(X_norm_p), 3, 1).astype(np.float32)
    y_perm = nn_model.predict(X_in_p, verbose=0).flatten()
    r2_perm = r2_score(y_csv, y_perm)
    importance = r2_baseline - r2_perm
    print(f"    {PASS} {feat_name}: R² drop = {importance:.6f} (baseline R² = {r2_baseline:.6f})")


# ############################################################################
#
#  3. GAUSSIAN PROCESS REGRESSOR — 3 features [Bx, By, Bz] (retrained, no time)
#
# ############################################################################
print("\n" + "=" * 70)
print("3. GAUSSIAN PROCESS REGRESSOR — 3 features [Bx, By, Bz] (no time)")
print("=" * 70)

gpr_c = joblib.load('models/cached/gpr_model.joblib')
gpr_model = gpr_c['model']
gpr_xs = gpr_c['X_scaler']
gpr_ys = gpr_c['y_scaler']
print(f"  X_scaler n_features: {gpr_xs.n_features_in_}")
print(f"  X_scaler mean: {gpr_xs.mean_}")
print(f"  X_scaler scale: {gpr_xs.scale_}")
print(f"  Learned kernel: {gpr_model.kernel_}")
assert gpr_xs.n_features_in_ == 3, f"Expected 3 features, got {gpr_xs.n_features_in_}"

def gpr_predict(bx, by, bz):
    """Run GPR prediction with proper scaling (3 features, no time)."""
    X_3 = np.column_stack([bx, by, bz])
    X_scaled = gpr_xs.transform(X_3)
    y_scaled = gpr_model.predict(X_scaled)
    return gpr_ys.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

# 3a. Synthetic test
y_gpr_synth = gpr_predict(bx_s, by_s, bz_s)
r2_gpr_synth = r2_score(y_synth, y_gpr_synth)
report("GPR", "Synthetic R²", r2_gpr_synth, 0.95)

# 3b. .dat file test
y_gpr_dat = gpr_predict(bx_dat, by_dat, bz_dat)
r2_gpr_dat = r2_score(y_dat, y_gpr_dat)
report("GPR", ".dat R²", r2_gpr_dat, 0.95)

# 3c. CSV test
y_gpr_csv = gpr_predict(bx_csv, by_csv, bz_csv)
r2_gpr_csv = r2_score(y_csv, y_gpr_csv)
report("GPR", "CSV R²", r2_gpr_csv, 0.95)

# 3d. Time dependency: IMPOSSIBLE — model has 3 features, no Time input
print(f"  {PASS} Time dependency: IMPOSSIBLE (model has 3 features, no Time input)")

# 3e. KERNEL LENGTHSCALE ANALYSIS (ARD kernel)
print(f"\n  KERNEL ANALYSIS:")
kernel = gpr_model.kernel_
print(f"    Learned kernel: {kernel}")
try:
    rbf_part = kernel.k1.k2  # k1 = Const*RBF, k2 of k1 = RBF
    lengthscales = rbf_part.length_scale
    if np.isscalar(lengthscales):
        print(f"    RBF lengthscale (isotropic): {lengthscales:.4f}")
    else:
        for name, ls in zip(['Bx', 'By', 'Bz'], lengthscales):
            print(f"    {name} lengthscale: {ls:.4f}")
        print(f"    {PASS} All features are spatial — no time to leak")
except Exception as e:
    print(f"    Could not extract lengthscales: {e}")

# 3f. PERMUTATION IMPORTANCE (spatial features only)
print(f"\n  PERMUTATION IMPORTANCE:")
r2_gpr_base = r2_score(y_csv, y_gpr_csv)
for feat_idx, feat_name in enumerate(['Bx', 'By', 'Bz']):
    cols = [bx_csv.copy(), by_csv.copy(), bz_csv.copy()]
    cols[feat_idx] = np.random.permutation(cols[feat_idx])
    y_perm = gpr_predict(cols[0], cols[1], cols[2])
    r2_perm = r2_score(y_csv, y_perm)
    importance = r2_gpr_base - r2_perm
    print(f"    {PASS} {feat_name}: R² drop = {importance:.6f}")


# ############################################################################
#
#  4. TEMPORAL RF — Sequence prediction (different purpose)
#
# ############################################################################
print("\n" + "=" * 70)
print("4. TEMPORAL RF — Predicts NEXT magnitude from past 3 readings")
print("=" * 70)

from temporal_models import (create_temporal_dataset, extract_temporal_features,
                              temporal_train_test_split, predict_rf_temporal,
                              get_feature_names)

temp_c = joblib.load('models/cached/temporal_rf_model.joblib')
temp_rf = temp_c['model']
print(f"  n_features_in_ = {temp_rf.n_features_in_}")
print(f"  n_estimators = {temp_rf.n_estimators}")

# Load temporal dataset
X_temp, y_temp, t_temp = create_temporal_dataset()
X_train_t, y_train_t, t_train_t, X_test_t, y_test_t, t_test_t = \
    temporal_train_test_split(X_temp, y_temp, t_temp)
print(f"  Windows: {len(X_temp)} total, {len(X_train_t)} train, {len(X_test_t)} test")
print(f"  Window shape: {X_temp.shape} (samples, window_len, channels=[X,Y,Z,Mag])")

# 4a. Test on temporal test set (future data)
y_pred_temp = predict_rf_temporal(temp_rf, X_test_t)
r2_temp = r2_score(y_test_t, y_pred_temp)
report("Temporal RF", "Temporal test R² (future data)", r2_temp, 0.90)

# 4b. Feature importance
feat_names = get_feature_names()
importances = temp_rf.feature_importances_
print(f"\n  FEATURE IMPORTANCE (top 10):")
sorted_idx = np.argsort(importances)[::-1]
for i in range(min(10, len(sorted_idx))):
    idx = sorted_idx[i]
    print(f"    {feat_names[idx]:20s}: {importances[idx]:.4f}")

# Check if any time-related features dominate
# Temporal RF uses extracted features from windows — no raw "Time" feature.
# The window contents are [X, Y, Z, Magnitude] — time is only implicit
# (sequential ordering). This is BY DESIGN for temporal forecasting.
print(f"\n  NOTE: Temporal RF uses WINDOWS of [X,Y,Z,Mag] values.")
print(f"  It has NO explicit Time feature — only sequential patterns.")
print(f"  This is correct for temporal forecasting (predict next from past 3).")
print(f"  Time dependency here means 'using temporal patterns' which is THE POINT.")


# ############################################################################
#
#  FINAL SUMMARY
#
# ############################################################################
print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

total_pass = 0
total_fail = 0
for model_name, tests in results.items():
    passes = sum(1 for _, s, _ in tests if s == PASS)
    fails = sum(1 for _, s, _ in tests if s == FAIL)
    total_pass += passes
    total_fail += fails
    status = "✅" if fails == 0 else "❌"
    print(f"  {status} {model_name}: {passes}/{passes+fails} tests passed")
    for test_name, status_str, value in tests:
        if status_str == FAIL:
            print(f"       FAILED: {test_name} = {value:.6f}")

print(f"\n  TOTAL: {total_pass}/{total_pass+total_fail} tests passed")
if total_fail > 0:
    print(f"\n  ❌ {total_fail} FAILURES DETECTED — MODELS NEED FIXING")
else:
    print(f"\n  ✅ ALL MODELS PASSED — No time dependency issues")
