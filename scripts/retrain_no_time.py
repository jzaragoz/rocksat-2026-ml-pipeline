#!/usr/bin/env python3
"""
RETRAIN GPR AND NN WITHOUT TIME — 3 features [Bx, By, Bz] only.
Matches the RF fix: models should learn |B| = √(Bx²+By²+Bz²), not memorize.
"""

import sys, os, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

from data_loader import load_from_csv, load_from_dat_file

res_csv = load_from_csv('data/Magneto_Fixed_Timeline.csv')
X_csv, y_csv = res_csv[1], res_csv[2]
bx_csv, by_csv, bz_csv, t_csv = res_csv[5], res_csv[6], res_csv[7], res_csv[8]
print(f"  CSV: {len(X_csv)} samples")

res_dat = load_from_dat_file('data/UPR - Flight_R4_EXP3_18_Nov_2025_11_37_58.dat')
X_dat, y_dat = res_dat[1], res_dat[2]
bx_dat, by_dat, bz_dat, t_dat = res_dat[5], res_dat[6], res_dat[7], res_dat[8]
print(f"  DAT: {len(X_dat)} samples")

# Synthetic test data
N_SYNTH = 2000
bx_s = np.random.uniform(-7000, 7000, N_SYNTH)
by_s = np.random.uniform(-7000, 7000, N_SYNTH)
bz_s = np.random.uniform(-7000, 7000, N_SYNTH)
X_synth = np.column_stack([bx_s, by_s, bz_s])
y_synth = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)

# Generate synthetic augmentation data (same as RF retrain)
N_AUG = 50000
bx_aug = np.random.uniform(-7000, 7000, N_AUG)
by_aug = np.random.uniform(-7000, 7000, N_AUG)
bz_aug = np.random.uniform(-7000, 7000, N_AUG)
X_aug = np.column_stack([bx_aug, by_aug, bz_aug])
y_aug = np.sqrt(bx_aug**2 + by_aug**2 + bz_aug**2)

# Combined training data = real + augmented
X_combined = np.vstack([X_csv, X_aug])
y_combined = np.concatenate([y_csv, y_aug])
print(f"  Combined training: {len(X_combined)} samples (real + 50K synthetic)")


# ############################################################################
#
#  1. RETRAIN GPR — 3 features [Bx, By, Bz]
#
# ############################################################################
print("\n" + "=" * 70)
print("RETRAINING GPR — 3 features [Bx, By, Bz]")
print("=" * 70)

# GPR can't handle 67K samples — subsample intelligently
# Use stratified sampling from combined data
MAX_GPR = 500
indices = np.random.choice(len(X_combined), MAX_GPR, replace=False)
X_gpr_train = X_combined[indices]
y_gpr_train = y_combined[indices]

# Fit scalers on 3-feature data
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_gpr_scaled = X_scaler.fit_transform(X_gpr_train)
y_gpr_scaled = y_scaler.fit_transform(y_gpr_train.reshape(-1, 1)).flatten()

# Use ARD (Automatic Relevance Determination) kernel to learn per-feature importance
kernel = (
    ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
    RBF(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2)) +
    WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    random_state=42,
    n_restarts_optimizer=10,
    normalize_y=False
)

print(f"  Training on {MAX_GPR} samples (3 features)...")
t0 = time.time()
gpr.fit(X_gpr_scaled, y_gpr_scaled)
print(f"  Training time: {time.time()-t0:.1f}s")
print(f"  Learned kernel: {gpr.kernel_}")

# Save
cache_path = Path('models/cached/gpr_model.joblib')
joblib.dump({'model': gpr, 'X_scaler': X_scaler, 'y_scaler': y_scaler}, cache_path)
print(f"  ✅ Saved to {cache_path}")

# Test
def gpr_predict_3(X_3col):
    X_s = X_scaler.transform(X_3col)
    y_s = gpr.predict(X_s)
    return y_scaler.inverse_transform(y_s.reshape(-1, 1)).flatten()

r2_csv = r2_score(y_csv, gpr_predict_3(X_csv))
r2_dat = r2_score(y_dat, gpr_predict_3(X_dat))
r2_synth = r2_score(y_synth, gpr_predict_3(X_synth))

print(f"\n  GPR RESULTS (3 features):")
print(f"    CSV R²:       {r2_csv:.6f}")
print(f"    .dat R²:      {r2_dat:.6f}")
print(f"    Synthetic R²: {r2_synth:.6f}")

if r2_synth > 0.95:
    print(f"  ✅ GPR FIXED — generalizes to synthetic data")
else:
    print(f"  ❌ GPR still broken — synthetic R² = {r2_synth:.4f}")


# ############################################################################
#
#  2. RETRAIN NN — 3 features [Bx, By, Bz]
#
# ############################################################################
print("\n" + "=" * 70)
print("RETRAINING NN (FCN) — 3 features [Bx, By, Bz]")
print("=" * 70)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Normalize using combined data stats (3 features)
nn_mean = X_combined.mean(axis=0)
nn_std = X_combined.std(axis=0)
print(f"  Normalization mean: {nn_mean}")
print(f"  Normalization std:  {nn_std}")

X_train_norm = (X_combined - nn_mean) / nn_std
X_train_nn = X_train_norm.reshape(len(X_train_norm), 3, 1).astype(np.float32)

# Build FCN model — 3 features now
# Architecture: Input(3,1) → Conv1D layers → output
# Adjusted for 3-feature input (shorter sequence)
model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(3, 1), padding='same'),
    BatchNormalization(),
    Conv1D(128, kernel_size=2, activation='relu', padding='same'),
    BatchNormalization(),
    Conv1D(64, kernel_size=2, activation='relu', padding='same'),
    BatchNormalization(),
    Conv1D(32, kernel_size=1, activation='relu', padding='same'),
    BatchNormalization(),
    # Global average pooling via Conv1D with stride
    Conv1D(1, kernel_size=3, activation='linear', padding='valid'),  # (3,1) → (1,1)
    Reshape((1,))
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_nn, y_combined, test_size=0.1, random_state=42)

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
]

print(f"\n  Training on {len(X_tr)} samples (3 features)...")
t0 = time.time()
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)
train_time = time.time() - t0
print(f"  Training time: {train_time:.1f}s")

# Test NN
def nn_predict_3(X_3col):
    X_n = (X_3col - nn_mean) / nn_std
    X_in = X_n.reshape(len(X_n), 3, 1).astype(np.float32)
    return model.predict(X_in, verbose=0).flatten()

r2_csv_nn = r2_score(y_csv, nn_predict_3(X_csv))
r2_dat_nn = r2_score(y_dat, nn_predict_3(X_dat))
r2_synth_nn = r2_score(y_synth, nn_predict_3(X_synth))

print(f"\n  NN RESULTS (3 features):")
print(f"    CSV R²:       {r2_csv_nn:.6f}")
print(f"    .dat R²:      {r2_dat_nn:.6f}")
print(f"    Synthetic R²: {r2_synth_nn:.6f}")

if r2_synth_nn > 0.95:
    print(f"  ✅ NN FIXED — generalizes to synthetic data")
else:
    print(f"  ⚠️  NN synthetic R² = {r2_synth_nn:.4f} — may need architecture tuning")

# Save NN
model.save('models/exports/magnetometer_fcn.keras')
np.savez('models/exports/magnetometer_fcn_norm.npz',
         mean=nn_mean, std=nn_std, r2=r2_csv_nn)
print(f"  ✅ Saved NN to models/exports/magnetometer_fcn.keras")

# Also save as SavedModel for HEF compilation
model.export('models/exports/magnetometer_fcn_saved_model')
print(f"  ✅ Saved SavedModel for potential HEF recompilation")


# ############################################################################
#
#  FINAL VERIFICATION — Re-run all tests
#
# ############################################################################
print("\n" + "=" * 70)
print("FINAL VERIFICATION — ALL MODELS")
print("=" * 70)

rf = joblib.load('models/cached/rf_model.joblib')

print(f"\n  {'Model':<15} {'CSV R²':>10} {'DAT R²':>10} {'Synth R²':>10} {'Status':>10}")
print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

# RF
r2_rf_c = r2_score(y_csv, rf.predict(X_csv))
r2_rf_d = r2_score(y_dat, rf.predict(X_dat))
r2_rf_s = r2_score(y_synth, rf.predict(X_synth))
st = "✅" if r2_rf_s > 0.95 else "❌"
print(f"  {'RF':<15} {r2_rf_c:>10.6f} {r2_rf_d:>10.6f} {r2_rf_s:>10.6f} {st:>10}")

# GPR
r2_g_c = r2_score(y_csv, gpr_predict_3(X_csv))
r2_g_d = r2_score(y_dat, gpr_predict_3(X_dat))
r2_g_s = r2_score(y_synth, gpr_predict_3(X_synth))
st = "✅" if r2_g_s > 0.95 else "❌"
print(f"  {'GPR':<15} {r2_g_c:>10.6f} {r2_g_d:>10.6f} {r2_g_s:>10.6f} {st:>10}")

# NN
r2_n_c = r2_score(y_csv, nn_predict_3(X_csv))
r2_n_d = r2_score(y_dat, nn_predict_3(X_dat))
r2_n_s = r2_score(y_synth, nn_predict_3(X_synth))
st = "✅" if r2_n_s > 0.95 else "❌"
print(f"  {'NN':<15} {r2_n_c:>10.6f} {r2_n_d:>10.6f} {r2_n_s:>10.6f} {st:>10}")

print()
all_pass = r2_rf_s > 0.95 and r2_g_s > 0.95 and r2_n_s > 0.95
if all_pass:
    print("  ✅ ALL MODELS PASS SYNTHETIC GENERALIZATION TEST")
else:
    print("  ⚠️  Some models may need further tuning")
    if r2_n_s < 0.95:
        print(f"     NN synthetic R² = {r2_n_s:.4f} — NN may need deeper architecture for pure formula learning")
