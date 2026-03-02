"""
RockSat-X 2026 — Temporal Forecasting Models
=============================================
"""

import os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from config import ML_CONFIG

warnings.filterwarnings("ignore")

# ── TensorFlow (optional) ────────────────────────────────────────────────────
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import GRU, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    pass

# ── Constants ─────────────────────────────────────────────────────────────────
_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_LIB_DIR)
def _find_data_csv():
    """Find the best available magnetometer CSV in data/ (newest by mtime)."""
    data_dir = os.path.join(_PROJECT_ROOT, "data")
    import glob as _glob
    csvs = sorted(
        _glob.glob(os.path.join(data_dir, "*.csv")),
        key=os.path.getmtime, reverse=True
    )
    return csvs[0] if csvs else os.path.join(data_dir, "data.csv")

DATA_PATH = _find_data_csv()
MODEL_DIR = os.path.join(_PROJECT_ROOT, "output", "temporal")
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_WINDOW = 3           # 3 timesteps look-back (best MAE from window sweep)
FORECAST_HORIZON = 1         # predict 1 step ahead
FEATURES = ["X", "Y", "Z"]  # raw input features per timestep
# We also derive Magnitude in the window as a 4th channel
N_CHANNELS = 4               # X, Y, Z, Magnitude


###############################################################################
# 1. DATASET CREATION
###############################################################################

def load_sensor_series(path: str = DATA_PATH, sensor_id: int | None = None):
    """
    Load CSV → per-sensor, per-second aggregated time series.

    Returns dict  sensor_id → DataFrame(Time, X, Y, Z, Magnitude)  sorted by
    Time with no duplicates.  Gaps (boot gap T19-T45, launch gap T0-T5) are
    left as missing rows so window creation naturally avoids spanning them.

    Supports both training CSV format (Time, Sensor, X, Y, Z) and
    ML CSV format from test_main.py (mission_time_s, sensor_id, bx_raw, by_raw, bz_raw).
    """
    from data_loader import is_legacy_csv, parse_legacy_csv
    if is_legacy_csv(path):
        df = parse_legacy_csv(path)
    else:
        df = pd.read_csv(path)

    # Normalise ML CSV column names → training format
    col_map = {
        'bx_raw': 'X', 'by_raw': 'Y', 'bz_raw': 'Z',
        'mission_time_s': 'Time', 'sensor_id': 'Sensor',
        'magnitude_measured': 'Magnitude',
    }
    rename = {old: new for old, new in col_map.items()
              if old in df.columns and new not in df.columns}
    if rename:
        df = df.rename(columns=rename)

    if "Magnitude" not in df.columns:
        df["Magnitude"] = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)

    sensors = [sensor_id] if sensor_id is not None else sorted(df["Sensor"].unique())
    result = {}
    for sid in sensors:
        sub = df[df["Sensor"] == sid].copy()
        # aggregate sub-samples within each integer second
        agg = (sub.groupby("Time")
               .agg(X=("X", "mean"), Y=("Y", "mean"),
                    Z=("Z", "mean"), Magnitude=("Magnitude", "mean"))
               .reset_index()
               .sort_values("Time")
               .reset_index(drop=True))
        result[sid] = agg
    return result


def _make_windows(series_df, window: int = DEFAULT_WINDOW,
                  horizon: int = FORECAST_HORIZON):
    """
    Sliding-window generator for ONE sensor series.

    Each sample:
        X_window : (window, N_CHANNELS) — past W readings [X,Y,Z,Mag]
        y_target : scalar — Magnitude at time t (one step after window)
        t_target : scalar — the timestamp of the target

    Windows never span a time-gap > 1 s between consecutive rows.
    """
    vals = series_df[["X", "Y", "Z", "Magnitude"]].values.astype(np.float32)
    times = series_df["Time"].values.astype(np.float64)

    # Mark "continuous" stretches (consecutive rows with dt == 1)
    dt = np.diff(times)
    break_mask = dt > 1.5  # anything > 1 s is a gap/discontinuity

    # Build list of contiguous segment start/end indices
    seg_starts = [0]
    for i, is_break in enumerate(break_mask):
        if is_break:
            seg_starts.append(i + 1)
    segments = []
    for s in seg_starts:
        # find extent of this segment
        e = s
        while e < len(times) - 1 and not break_mask[e]:
            e += 1
        if break_mask[e - 1] if e > 0 and e - 1 < len(break_mask) else False:
            segments.append((s, e))
        else:
            segments.append((s, e + 1))

    # Deduplicate / merge overlapping
    merged = []
    for s, e in segments:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    Xs, ys, ts = [], [], []
    for seg_s, seg_e in merged:
        seg_len = seg_e - seg_s
        if seg_len < window + horizon:
            continue
        for i in range(seg_len - window - horizon + 1):
            idx = seg_s + i
            Xs.append(vals[idx: idx + window])
            ys.append(vals[idx + window + horizon - 1, 3])   # Magnitude
            ts.append(times[idx + window + horizon - 1])

    if not Xs:
        return np.empty((0, window, N_CHANNELS)), np.empty(0), np.empty(0)
    return np.array(Xs), np.array(ys), np.array(ts)


def create_temporal_dataset(window: int = DEFAULT_WINDOW,
                            sensors: list | None = None,
                            csv_path: str | None = None):
    """
    Build the full sliding-window dataset across all requested sensors.

    Returns
    -------
    X : ndarray (N, window, 4)
    y : ndarray (N,)
    t : ndarray (N,)
    """
    all_series = load_sensor_series(path=csv_path or DATA_PATH)
    if sensors is not None:
        all_series = {k: v for k, v in all_series.items() if k in sensors}

    all_X, all_y, all_t = [], [], []
    for sid, sdf in all_series.items():
        Xi, yi, ti = _make_windows(sdf, window)
        all_X.append(Xi)
        all_y.append(yi)
        all_t.append(ti)

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    t = np.concatenate(all_t)
    return X, y, t


def temporal_train_test_split(X, y, t, split_time: float = None):
    """
    Temporal split: train on everything before `split_time`,
    test on everything at or after `split_time`.
    If split_time is None, uses the 70th percentile of time values.
    """
    if split_time is None:
        split_time = float(np.percentile(t, 70))
    train_mask = t < split_time
    test_mask = t >= split_time
    return (X[train_mask], y[train_mask], t[train_mask],
            X[test_mask],  y[test_mask],  t[test_mask])


###############################################################################
# 2. FEATURE ENGINEERING (for Random Forest)
###############################################################################

def extract_temporal_features(X_windows: np.ndarray) -> np.ndarray:
    """
    From each (window, 4) window, extract hand-crafted temporal statistics
    that capture dynamics — NOT the current reading.

    Features per channel (X, Y, Z, Mag):
        mean, std, min, max, last_value,
        linear_slope, last_minus_first (trend)

    Plus cross-channel features:
        magnitude_velocity (slope of Mag), magnitude_acceleration,
        max_rate_of_change (Mag)
    """
    n_samples, win_len, n_ch = X_windows.shape
    per_ch = 7   # stats per channel
    extra = 3    # cross-channel
    n_feat = n_ch * per_ch + extra
    out = np.empty((n_samples, n_feat), dtype=np.float32)

    t_axis = np.arange(win_len, dtype=np.float32)

    for i in range(n_samples):
        w = X_windows[i]  # (win_len, n_ch)
        idx = 0
        for ch in range(n_ch):
            col = w[:, ch]
            out[i, idx]     = col.mean()
            out[i, idx + 1] = col.std()
            out[i, idx + 2] = col.min()
            out[i, idx + 3] = col.max()
            out[i, idx + 4] = col[-1]            # most recent value
            # linear slope via least-squares
            slope = np.polyfit(t_axis, col, 1)[0] if col.std() > 1e-9 else 0.0
            out[i, idx + 5] = slope
            out[i, idx + 6] = col[-1] - col[0]   # trend
            idx += per_ch

        # Cross-channel: magnitude dynamics
        mag = w[:, 3]
        diffs = np.diff(mag)
        out[i, idx]     = diffs[-1] if len(diffs) else 0.0          # velocity
        out[i, idx + 1] = np.diff(diffs)[-1] if len(diffs) > 1 else 0.0  # accel
        out[i, idx + 2] = np.max(np.abs(diffs)) if len(diffs) else 0.0   # max RoC
    return out


def get_feature_names() -> list[str]:
    channels = ["X", "Y", "Z", "Mag"]
    stats = ["mean", "std", "min", "max", "last", "slope", "trend"]
    names = [f"{ch}_{st}" for ch in channels for st in stats]
    names += ["mag_velocity", "mag_acceleration", "mag_max_roc"]
    return names


###############################################################################
# 3. MODELS
###############################################################################

# ── 3a  Naive baseline ───────────────────────────────────────────────────────

def naive_forecast(X_windows: np.ndarray) -> np.ndarray:
    """Predict magnitude(t) = magnitude(t-1)  (last value in window)."""
    return X_windows[:, -1, 3].copy()


# ── 3b  Random Forest with temporal features ─────────────────────────────────

def train_rf_temporal(X_train_win, y_train,
                      n_estimators=50, max_depth=12, seed=42):
    """
    Train RF on hand-crafted temporal features extracted from windows.
    Returns (model, feature_names).
    50 trees to meet 22ms inference budget on Raspberry Pi 5.
    """
    feats = extract_temporal_features(X_train_win)
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(feats, y_train)
    return rf, get_feature_names()


def predict_rf_temporal(model, X_windows):
    feats = extract_temporal_features(X_windows)
    return model.predict(feats)


# ── 3c  GRU neural network ───────────────────────────────────────────────────

def build_gru_model(window: int = DEFAULT_WINDOW, n_channels: int = N_CHANNELS):
    """
    Small GRU for temporal forecasting.
    GRU is lighter than LSTM with comparable performance on short sequences.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for GRU model")

    model = Sequential([
        Input(shape=(window, n_channels)),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def train_gru_model(X_train, y_train, X_val=None, y_val=None,
                    window=DEFAULT_WINDOW, epochs=200, batch_size=32):
    """
    Train GRU model with early stopping.
    Normalises inputs internally; returns (model, norm_params).
    """
    # Normalise per-channel
    mean = X_train.mean(axis=(0, 1))
    std  = X_train.std(axis=(0, 1))
    std[std < 1e-9] = 1.0
    X_tr_n = (X_train - mean) / std

    y_mean, y_std = y_train.mean(), y_train.std()
    y_tr_n = (y_train - y_mean) / y_std

    val_data = None
    if X_val is not None:
        X_v_n = (X_val - mean) / std
        y_v_n = (y_val - y_mean) / y_std
        val_data = (X_v_n, y_v_n)

    model = build_gru_model(window, X_train.shape[2])

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"
                      if val_data else "loss"),
        ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-6),
    ]

    history = model.fit(
        X_tr_n, y_tr_n,
        validation_data=val_data,
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=0,
    )

    norm_params = {"x_mean": mean, "x_std": std,
                   "y_mean": y_mean, "y_std": y_std}
    return model, norm_params, history


def predict_gru(model, X_windows, norm_params):
    X_n = (X_windows - norm_params["x_mean"]) / norm_params["x_std"]
    y_n = model.predict(X_n, verbose=0).ravel()
    return y_n * norm_params["y_std"] + norm_params["y_mean"]


###############################################################################
# 4. EVALUATION HELPERS
###############################################################################

def evaluate_model(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100
    residuals = y_true - y_pred
    print(f"  {label:30s}  MAE={mae:7.2f} nT  RMSE={rmse:7.2f} nT  R²={r2:.6f}  MAPE={mape:.3f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape,
            "residuals": residuals, "label": label}


def evaluate_by_phase(y_true, y_pred, times, label="Model"):
    """MAE broken down by flight phase."""
    phases = OrderedDict([
        ("Pre-launch (T<0)",       times < 0),
        ("Launch (−5<T<5)",        (times >= -5) & (times <= 5)),
        ("Ascent (0<T<200)",       (times > 0)   & (times < 200)),
        ("Apogee (195<T<205)",     (times >= 195) & (times <= 205)),
        ("Descent (T>200)",        times > 200),
    ])
    results = {}
    for name, mask in phases.items():
        n = mask.sum()
        if n == 0:
            results[name] = {"n": 0, "mae": np.nan, "r2": np.nan}
            continue
        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        r2 = r2_score(y_true[mask], y_pred[mask]) if n > 1 else np.nan
        results[name] = {"n": int(n), "mae": mae, "r2": r2}
    return results


def benchmark_inference(model_fn, X_single, n_repeats=200, label="Model"):
    """Measure single-sample inference time in milliseconds."""
    # warm up
    for _ in range(5):
        model_fn(X_single)
    times_ms = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model_fn(X_single)
        times_ms.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times_ms)
    print(f"  {label:30s}  mean={arr.mean():.2f} ms  p95={np.percentile(arr,95):.2f} ms  max={arr.max():.2f} ms")
    return {"mean_ms": arr.mean(), "p95_ms": np.percentile(arr, 95),
            "max_ms": arr.max()}


###############################################################################
# 5. CONVENIENCE: full pipeline
###############################################################################

def run_full_temporal_pipeline(window: int = DEFAULT_WINDOW,
                               split_time: float = None,
                               verbose: bool = True,
                               csv_path: str | None = None):
    """
    End-to-end: create dataset → train models → evaluate → return
    everything needed by anomaly_v2.

    Returns
    -------
    results : dict with keys
        'rf_model', 'gru_model', 'gru_norm',
        'metrics_*', 'X_test', 'y_test', 't_test',
        'pred_rf', 'pred_gru', 'pred_naive'
    """
    if verbose:
        print("=" * 65)
        print("  TEMPORAL FORECASTING PIPELINE")
        print("=" * 65)

    # ── dataset ───────────────────────────────────────────────────────────
    if verbose:
        print("\n  Creating sliding-window dataset (window=%d) …" % window)
    X_all, y_all, t_all = create_temporal_dataset(window, csv_path=csv_path)
    if verbose:
        print(f"    Total samples: {len(y_all)}")

    X_tr, y_tr, t_tr, X_te, y_te, t_te = temporal_train_test_split(
        X_all, y_all, t_all, split_time
    )
    if verbose:
        print(f"    Train (T<{split_time:.0f}): {len(y_tr)}")
        print(f"    Test  (T≥{split_time:.0f}): {len(y_te)}")

    results = {"window": window, "split_time": split_time,
               "X_train": X_tr, "y_train": y_tr, "t_train": t_tr,
               "X_test": X_te,  "y_test": y_te,  "t_test": t_te,
               "X_all": X_all,  "y_all": y_all,  "t_all": t_all}

    # ── naive baseline ────────────────────────────────────────────────────
    if verbose:
        print("\n  [1/3] Naive baseline (predict previous value)")
    pred_naive_te = naive_forecast(X_te)
    results["pred_naive"] = pred_naive_te
    results["metrics_naive"] = evaluate_model(y_te, pred_naive_te, "Naive (prev value)")
    results["phases_naive"] = evaluate_by_phase(y_te, pred_naive_te, t_te, "Naive")

    # ── Random Forest ─────────────────────────────────────────────────────
    if verbose:
        print("\n  [2/3] Random Forest with temporal features")
    t0 = time.time()
    rf_model, feat_names = train_rf_temporal(X_tr, y_tr, seed=42)
    rf_train_time = time.time() - t0
    pred_rf_te = predict_rf_temporal(rf_model, X_te)
    results["rf_model"] = rf_model
    results["rf_feat_names"] = feat_names
    results["pred_rf"] = pred_rf_te
    results["metrics_rf"] = evaluate_model(y_te, pred_rf_te, "RF temporal")
    results["phases_rf"] = evaluate_by_phase(y_te, pred_rf_te, t_te, "RF temporal")
    if verbose:
        print(f"    Training time: {rf_train_time:.1f}s")
        # Feature importance
        imp = rf_model.feature_importances_
        top = np.argsort(imp)[::-1][:10]
        print("    Top-10 feature importances:")
        for rank, idx in enumerate(top, 1):
            print(f"      {rank:2d}. {feat_names[idx]:20s} {imp[idx]:.4f}")

    # Also train on full data and predict full dataset for anomaly scoring
    rf_full, _ = train_rf_temporal(X_all, y_all, seed=42)
    results["pred_rf_all"] = predict_rf_temporal(rf_full, X_all)
    results["rf_model_full"] = rf_full

    # ── GRU ───────────────────────────────────────────────────────────────
    if TF_AVAILABLE:
        if verbose:
            print("\n  [3/3] GRU neural network")
        # Use 10% of training as validation for early stopping
        n_val = max(100, int(len(y_tr) * 0.1))
        X_tr_gru, X_val_gru = X_tr[:-n_val], X_tr[-n_val:]
        y_tr_gru, y_val_gru = y_tr[:-n_val], y_tr[-n_val:]

        t0 = time.time()
        gru_model, gru_norm, gru_hist = train_gru_model(
            X_tr_gru, y_tr_gru, X_val_gru, y_val_gru,
            window=window, epochs=200, batch_size=32,
        )
        gru_train_time = time.time() - t0

        pred_gru_te = predict_gru(gru_model, X_te, gru_norm)
        results["gru_model"] = gru_model
        results["gru_norm"] = gru_norm
        results["pred_gru"] = pred_gru_te
        results["metrics_gru"] = evaluate_model(y_te, pred_gru_te, "GRU temporal")
        results["phases_gru"] = evaluate_by_phase(y_te, pred_gru_te, t_te, "GRU temporal")
        if verbose:
            print(f"    Training time: {gru_train_time:.1f}s  "
                  f"Epochs run: {len(gru_hist.history['loss'])}")

        # Full-data predictions for anomaly scoring
        # Retrain on all data
        gru_full, gru_norm_full, _ = train_gru_model(
            X_all, y_all, window=window, epochs=150, batch_size=32,
        )
        results["pred_gru_all"] = predict_gru(gru_full, X_all, gru_norm_full)
        results["gru_model_full"] = gru_full
        results["gru_norm_full"] = gru_norm_full
    else:
        if verbose:
            print("\n  [3/3] GRU — SKIPPED (TensorFlow not available)")
        results["gru_model"] = None

    # ── Speed benchmarks ──────────────────────────────────────────────────
    if verbose:
        print("\n  Inference speed (single sample):")
    single = X_te[:1]
    results["speed_naive"] = benchmark_inference(
        lambda s: naive_forecast(s), single, label="Naive")
    results["speed_rf"] = benchmark_inference(
        lambda s: predict_rf_temporal(rf_model, s), single, label="RF temporal")
    if TF_AVAILABLE and results.get("gru_model") is not None:
        results["speed_gru"] = benchmark_inference(
            lambda s: predict_gru(gru_model, s, gru_norm), single, label="GRU temporal")

    return results


###############################################################################
# 6. CACHED TEMPORAL MODEL FOR MAIN PIPELINE
###############################################################################

_CACHE_DIR = Path(__file__).parent.parent / "models" / "cached"

def run_temporal_forecasting(mae_threshold_multiplier: float = 3.0,
                             verbose: bool = True,
                             csv_path: str | None = None):
    """
    Load or train the temporal RF model, compute prediction errors on the
    full dataset, and return anomaly flags based on prediction error.

    This is the function main.py calls to integrate temporal forecasting
    as a 5th anomaly detection method.

    The idea: if |predicted - actual| > threshold, the reading deviates
    from recent trend → anomaly.  threshold = multiplier × MAE on test set.

    Returns
    -------
    dict with keys:
        'model'          : trained RF temporal model
        'pred_all'       : predictions for full dataset (aligned to t_all)
        'y_all'          : actual magnitudes (aligned)
        't_all'          : timestamps (aligned)
        'errors_all'     : |predicted - actual| for each point
        'anomaly_mask'   : bool array — True where error > threshold
        'threshold'      : the nT threshold used
        'mae'            : MAE on test set
        'r2'             : R² on test set
        'n_samples'      : total aligned samples
    """
    cache_path = _CACHE_DIR / "temporal_rf_model.joblib"

    # ── Build sliding-window dataset ──────────────────────────────────────
    if verbose:
        print("\n" + "=" * 50)
        print("TEMPORAL FORECASTING (Prediction-Error Anomaly)")
        print("=" * 50)

    X_all, y_all, t_all = create_temporal_dataset(DEFAULT_WINDOW, csv_path=csv_path)
    X_tr, y_tr, t_tr, X_te, y_te, t_te = temporal_train_test_split(
        X_all, y_all, t_all
    )
    split_t = t_te[0] if len(t_te) > 0 else 0

    if verbose:
        print(f"   Sliding-window samples: {len(y_all)} (window={DEFAULT_WINDOW})")
        print(f"   Train (T<{split_t:.0f}): {len(y_tr)}  |  Test (T≥{split_t:.0f}): {len(y_te)}")

    # ── Load or train model ───────────────────────────────────────────────
    if cache_path.exists():
        cached = joblib.load(cache_path)
        rf_model = cached['model']
        test_mae = cached['mae']
        test_r2 = cached['r2']
        if verbose:
            print(f"   ✓ Loaded cached temporal model")
            print(f"     Cached MAE: {test_mae:.2f} nT  |  R²: {test_r2:.6f}")
    else:
        if verbose:
            print("   Training temporal RF model...")
        rf_model, _ = train_rf_temporal(X_tr, y_tr, n_estimators=50, max_depth=12)
        pred_te = predict_rf_temporal(rf_model, X_te)
        test_mae = mean_absolute_error(y_te, pred_te)
        test_r2 = r2_score(y_te, pred_te)
        # Save
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': rf_model, 'mae': test_mae, 'r2': test_r2}, cache_path)
        if verbose:
            print(f"   ✓ Trained and cached  MAE: {test_mae:.2f} nT  |  R²: {test_r2:.6f}")

    # ── Predict on FULL dataset ───────────────────────────────────────────
    # Retrain on all data for full-dataset predictions (or use same model)
    pred_all = predict_rf_temporal(rf_model, X_all)
    errors_all = np.abs(pred_all - y_all)

    # Threshold = multiplier × MAE (default: 3 × MAE ≈ 45 nT)
    threshold = mae_threshold_multiplier * test_mae
    anomaly_mask = errors_all > threshold
    n_flagged = anomaly_mask.sum()

    if verbose:
        print(f"   Threshold: {mae_threshold_multiplier}× MAE = {threshold:.1f} nT")
        print(f"   Anomalies: {n_flagged} ({100*n_flagged/len(y_all):.1f}%)")

    return {
        'model': rf_model,
        'pred_all': pred_all,
        'y_all': y_all,
        't_all': t_all,
        'errors_all': errors_all,
        'anomaly_mask': anomaly_mask,
        'threshold': threshold,
        'mae': test_mae,
        'r2': test_r2,
        'n_samples': len(y_all),
    }


###############################################################################
# 7. STANDALONE ENTRY POINT
###############################################################################

if __name__ == "__main__":
    res = run_full_temporal_pipeline(window=DEFAULT_WINDOW, split_time=200.0)

    print("\n" + "=" * 65)
    print("  TEMPORAL FORECASTING SUMMARY")
    print("=" * 65)
    for key in ["metrics_naive", "metrics_rf", "metrics_gru"]:
        m = res.get(key)
        if m:
            print(f"  {m['label']:25s}  MAE={m['mae']:.2f}  R²={m['r2']:.4f}")
    naive_mae = res["metrics_naive"]["mae"]
    rf_mae    = res["metrics_rf"]["mae"]
    improvement = (1 - rf_mae / naive_mae) * 100
    print(f"\n  RF improvement over naive: {improvement:+.1f}%")
    if res.get("metrics_gru"):
        gru_mae = res["metrics_gru"]["mae"]
        improvement_gru = (1 - gru_mae / naive_mae) * 100
        print(f"  GRU improvement over naive: {improvement_gru:+.1f}%")
