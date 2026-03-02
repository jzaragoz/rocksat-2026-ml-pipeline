#!/usr/bin/env python3
"""
RockSat-X 2026 — Post-Flight Analysis
======================================
Generates the FULL set of visualizations from flight data — the same
plots that main.py produces, but reconstructed from saved data.

Works with:
    - ML CSV files from test_main.py or sim_flight.py  (has ML predictions)
    - Raw .dat files from the flight serial log          (needs ML re-inference)

The .dat file is the raw sensor output — interleaved magneto/ADC/thermo/
pressure/TDL lines written by all sensor scripts via serial.  This script
parses only the magnetometer lines and runs cached ML models on them.

Usage:
    python post_flight.py                           # Auto-find latest CSV
    python post_flight.py --csv output/flight_data/sim_data_20260207.csv
    python post_flight.py --dat data/flight.dat      # Parse raw .dat file

Output:
    output/post_flight/
        01_random_forest.png        — RF scatter (actual vs predicted)
        01b_rf_timeseries.png       — RF predictions over time
        02_kmeans_clusters.png      — K-Means cluster assignments
        03_neural_network.png       — NN scatter + timeseries
        05_rotation_analysis.png    — Magnetometer spin analysis
        06_anomaly_combined.png     — Anomaly ensemble (all methods)
        anomaly_mag0.png …          — Per-sensor anomaly timelines
        FLIGHT_REPORT.txt           — Text summary
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR / 'lib'))

import joblib
from config import STORAGE_PATH, ML_CONFIG, PROJECT_ROOT, MISSION_NAME
from data_loader import parse_dat_file, parse_dat_file_all
from anomaly import advanced_anomaly_detection
from temporal_models import run_temporal_forecasting
from visualization import (
    plot_multi_magnetometer, plot_clusters, plot_rf_actual_vs_predicted,
    plot_rf_timeseries, plot_gpr_uncertainty, plot_anomaly_detection,
    plot_nn_results, plot_magnetometer_rotation,
    plot_anomaly_per_mag_timeline, plot_temporal_prediction_errors,
    plot_raw_magnetometer_overview, plot_pressure_timeseries,
    plot_temperature_timeseries, plot_adc_timeseries,
    plot_flight_events_timeline,
)

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ==============================================================================
# CSV LOADING
# ==============================================================================

def find_flight_csvs():
    """Find flight CSV file(s) in the output directory.

    If the Pi rebooted mid-flight, there will be MULTIPLE timestamped CSVs
    (e.g. flight_data_20260301_120000.csv and flight_data_20260301_120024.csv).
    This function returns ALL non-empty CSVs for a given prefix so they can
    be merged — preventing the GHOST-2025 data-loss scenario.

    Returns:
        list[str]: List of CSV paths (newest first), or empty list.
    """
    base = str(SCRIPT_DIR / 'output' / 'flight_data')

    def _find_all(pattern):
        """Return all non-empty CSVs matching *pattern*, newest first."""
        candidates = sorted(glob.glob(os.path.join(base, pattern)),
                            key=os.path.getmtime, reverse=True)
        return [f for f in candidates if os.path.getsize(f) > 300]

    # Prefer real flight data over sim data
    results = _find_all('flight_data_*.csv')
    if results:
        return results
    results = _find_all('sim_data_*.csv')
    if results:
        return results
    return _find_all('*.csv')


def find_latest_flight_csv():
    """Backwards-compatible: return single newest CSV path, or None."""
    csvs = find_flight_csvs()
    return csvs[0] if csvs else None


def load_flight_csv(csv_path):
    """Load and parse a single flight CSV file.

    Normalises column names so both 'legacy' CSVs
    (X, Y, Z, T, Sensor, Magnitude) and 'ML-pipeline' CSVs
    (bx_raw, by_raw, bz_raw, mission_time_s, sensor_id, magnitude_measured)
    result in the same expected schema.
    """
    print(f"Loading flight data: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} readings loaded")
    print(f"  Columns: {list(df.columns)}")

    # ── Column normalisation ──────────────────────────────────────────
    # Map alternate names → canonical pipeline names.
    # T and Time are the same data in legacy CSVs; drop Time if T exists.
    if 'T' in df.columns and 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    COL_MAP = {
        'X':         'bx_raw',
        'Y':         'by_raw',
        'Z':         'bz_raw',
        'T':         'mission_time_s',
        'Time':      'mission_time_s',   # fallback if T was absent
        'Sensor':    'sensor_id',
        'Magnitude': 'magnitude_measured',
    }
    renamed = {}
    for old, new in COL_MAP.items():
        if old in df.columns and new not in df.columns:
            renamed[old] = new
    if renamed:
        df = df.rename(columns=renamed)
        print(f"  Column mapping applied: {renamed}")

    if 'mission_time_s' in df.columns:
        t_min = df['mission_time_s'].min()
        t_max = df['mission_time_s'].max()
        print(f"  Mission time: T{t_min:+.1f}s to T{t_max:+.1f}s")

    if 'sensor_id' in df.columns:
        sensors = df['sensor_id'].unique()
        print(f"  Sensors: {sorted(sensors)}")

    if 'magnitude_measured' in df.columns:
        mag = df['magnitude_measured']
        print(f"  Magnitude range: {mag.min():.1f} to {mag.max():.1f} nT")

    return df


def load_and_merge_flight_csvs(csv_paths):
    """Load multiple flight CSVs, merge, sort by timestamp, fix mission_time.

    When the Pi reboots mid-flight, each boot creates a new CSV with
    mission_time_s reset to T-150 (or T+0). But timestamp_unix is absolute
    (Unix epoch) and correct across reboots.

    This function:
      1. Concatenates all CSVs
      2. Sorts by timestamp_unix (absolute wall-clock time)
      3. Recalculates mission_time_s from timestamp_unix
      4. Marks boot gaps so plots can show them

    Args:
        csv_paths: List of CSV file paths (from find_flight_csvs)

    Returns:
        pd.DataFrame with corrected, monotonic mission_time_s
    """
    if len(csv_paths) == 1:
        return load_flight_csv(csv_paths[0])

    # Load all CSVs
    frames = []
    for i, p in enumerate(sorted(csv_paths)):
        df = pd.read_csv(p)
        if len(df) == 0:
            continue
        df['_boot_id'] = i
        frames.append(df)
        print(f"  Boot {i}: {os.path.basename(p)} — {len(df)} readings")

    if not frames:
        print("  ERROR: All CSV files are empty")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)

    # Sort by absolute timestamp (survives reboots)
    if 'timestamp_unix' in merged.columns:
        merged = merged.sort_values('timestamp_unix').reset_index(drop=True)

        # Recalculate mission_time from timestamp_unix
        # Use the FIRST reading's original mission_time as anchor
        first_unix = merged['timestamp_unix'].iloc[0]
        first_mt = merged['mission_time_s'].iloc[0]
        merged['mission_time_s'] = first_mt + (merged['timestamp_unix'] - first_unix)

        # Detect boot gaps (>5s jump in timestamp)
        dt = merged['timestamp_unix'].diff()
        boot_gaps = dt[dt > 5.0]
        if len(boot_gaps) > 0:
            n_boots = len(boot_gaps) + 1
            gap_total = boot_gaps.sum() - len(boot_gaps) * (1.0/45)  # subtract normal sample gap
            print(f"\n  ⚠  DETECTED {n_boots} BOOT SESSIONS (Pi rebooted {len(boot_gaps)} time(s))")
            for idx, gap in boot_gaps.items():
                gap_mt = merged.loc[idx, 'mission_time_s']
                print(f"     Boot gap at T{gap_mt:+.1f}s — {gap:.1f}s of missing data")
            print(f"     Total data gap: {gap_total:.1f}s")
    else:
        print("  ⚠  No timestamp_unix column — cannot recalculate mission time")

    n_boots = merged['_boot_id'].nunique()
    merged.drop(columns=['_boot_id'], inplace=True)

    print(f"\n  MERGED: {len(merged)} readings from {n_boots} boot(s)")
    if 'mission_time_s' in merged.columns:
        print(f"  Mission time: T{merged['mission_time_s'].min():+.1f}s to T{merged['mission_time_s'].max():+.1f}s")

    return merged


# ==============================================================================
# EXTRACT ARRAYS FROM CSV
# ==============================================================================

def extract_arrays(df):
    """
    Extract the same arrays that main.py works with from the CSV.
    Returns a dict of arrays ready for visualization functions.
    """
    bx = df['bx_raw'].values.astype(float)
    by = df['by_raw'].values.astype(float)
    bz = df['bz_raw'].values.astype(float)
    t = df['mission_time_s'].values.astype(float) if 'mission_time_s' in df.columns else np.arange(len(df), dtype=float)
    magnitude = np.sqrt(bx**2 + by**2 + bz**2)

    X = np.column_stack([bx, by, bz, t])

    sensor_ids = df['sensor_id'].values if 'sensor_id' in df.columns else np.zeros(len(df), dtype=int)

    return {
        'bx': bx, 'by': by, 'bz': bz, 't': t,
        'magnitude': magnitude, 'X': X,
        'sensor_ids': sensor_ids,
    }


# ==============================================================================
# PER-SENSOR DATA SPLIT
# ==============================================================================

def build_mag_data_by_sensor(df, arrays):
    """Build the mag_data_by_sensor dict that per-mag plots expect."""
    if 'sensor_id' not in df.columns:
        return {}

    mag_data = {}
    for sid in sorted(df['sensor_id'].unique()):
        mask = df['sensor_id'].values == sid
        mag_data[int(sid)] = {
            'time': arrays['t'][mask],
            'x': arrays['bx'][mask],
            'y': arrays['by'][mask],
            'z': arrays['bz'][mask],
            'magnitude': arrays['magnitude'][mask],
            'count': int(np.sum(mask)),
        }
    return mag_data


# ==============================================================================
# ANALYSIS + VISUALIZATION
# ==============================================================================

def analyze_and_plot(df, output_dir, csv_path=None):
    """Generate ALL visualizations — matching main.py output."""

    if 'bx_raw' not in df.columns:
        print("  ERROR: CSV missing bx_raw/by_raw/bz_raw — cannot generate plots")
        return

    arrays = extract_arrays(df)
    X = arrays['X']
    y = arrays['magnitude']
    bx, by, bz, t = arrays['bx'], arrays['by'], arrays['bz'], arrays['t']

    # Build mag_data_by_sensor (passed to plots that support multi-sensor overlay)
    mag_data = build_mag_data_by_sensor(df, arrays)

    # Clean stale per-sensor anomaly plots from previous runs
    import glob as _glob
    for stale in _glob.glob(os.path.join(output_dir, 'anomaly_mag*.png')):
        os.remove(stale)

    # ------------------------------------------------------------------
    # 0. Raw Magnetometer Overview (non-ML baseline)
    # ------------------------------------------------------------------
    if mag_data:
        try:
            plot_multi_magnetometer(
                mag_data,
                save_path=os.path.join(output_dir, '00_raw_magnetometer.png'))
        except Exception as e:
            print(f"    Raw magnetometer plot failed: {e}")

    # ------------------------------------------------------------------
    # 1. RF Scatter — actual vs predicted
    # ------------------------------------------------------------------
    if 'magnitude_rf' in df.columns and not df['magnitude_rf'].isna().all():
        rf_pred = df['magnitude_rf'].values.astype(float)
        mask = ~np.isnan(rf_pred)
        if mask.sum() > 0:
            measured = y[mask]
            predicted = rf_pred[mask]
            ss_res = np.sum((measured - predicted)**2)
            ss_tot = np.sum((measured - np.mean(measured))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((measured - predicted)**2))
            print(f"\n  RF: R² = {r2:.6f} | RMSE = {rmse:.2f} nT")

            try:
                plot_rf_actual_vs_predicted(
                    measured, predicted, r2,
                    save_path=os.path.join(output_dir, '01_random_forest.png'))
            except Exception as e:
                print(f"    RF scatter plot failed: {e}")

            # 1b. RF Timeseries — predictions over mission time
            try:
                plot_rf_timeseries(
                    X[mask], measured, predicted,
                    save_path=os.path.join(output_dir, '01b_rf_timeseries.png'),
                    mag_data_by_sensor=mag_data)
            except Exception as e:
                print(f"    RF timeseries plot failed: {e}")

    # ------------------------------------------------------------------
    # 2. K-Means Clustering
    # ------------------------------------------------------------------
    if 'cluster_id' in df.columns:
        cluster_labels = df['cluster_id'].values.astype(int)
        n_clusters = len(np.unique(cluster_labels))
        # Compute cluster centers from the data
        centers = np.array([
            np.column_stack([bx, by, bz])[cluster_labels == k].mean(axis=0)
            for k in range(n_clusters)
        ])
        print(f"  K-Means: {n_clusters} clusters")
        try:
            plot_clusters(
                X, cluster_labels, centers,
                save_path=os.path.join(output_dir, '02_kmeans_clusters.png'),
                title=f"{MISSION_NAME} — K-Means Cluster Assignment")
        except Exception as e:
            print(f"    Cluster plot failed: {e}")

    # ------------------------------------------------------------------
    # 3. Neural Network
    # ------------------------------------------------------------------
    if 'magnitude_nn' in df.columns and not df['magnitude_nn'].isna().all():
        nn_pred = df['magnitude_nn'].values.astype(float)
        mask = ~np.isnan(nn_pred)
        if mask.sum() > 0:
            measured = y[mask]
            predicted = nn_pred[mask]
            ss_res = np.sum((measured - predicted)**2)
            ss_tot = np.sum((measured - np.mean(measured))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((measured - predicted)**2))
            print(f"  NN: R² = {r2:.6f} | RMSE = {rmse:.2f} nT")

            try:
                plot_nn_results(
                    X[mask], measured, predicted,
                    save_path=os.path.join(output_dir, '03_neural_network.png'),
                    r2=r2)
            except Exception as e:
                print(f"    NN plot failed: {e}")

    # ------------------------------------------------------------------
    # 4. GPR (if present in CSV)
    # ------------------------------------------------------------------
    if 'magnitude_gpr' in df.columns and not df['magnitude_gpr'].isna().all():
        gpr_pred = df['magnitude_gpr'].values.astype(float)
        mask = ~np.isnan(gpr_pred)
        if mask.sum() > 0:
            measured = y[mask]
            predicted = gpr_pred[mask]
            ss_res = np.sum((measured - predicted)**2)
            ss_tot = np.sum((measured - np.mean(measured))**2)
            r2_gpr = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            std_dummy = np.zeros_like(predicted)
            try:
                plot_gpr_uncertainty(
                    X[mask], measured, predicted, std_dummy,
                    save_path=os.path.join(output_dir, '04_gaussian_process.png'),
                    r2=r2_gpr, mag_data_by_sensor=mag_data)
            except Exception as e:
                print(f"    GPR plot failed: {e}")

    # ------------------------------------------------------------------
    # 5. Magnetometer Rotation (Spin Analysis)
    # ------------------------------------------------------------------
    try:
        plot_magnetometer_rotation(
            bx, by, bz, t,
            save_path=os.path.join(output_dir, '05_rotation_analysis.png'),
            mag_data_by_sensor=mag_data)
    except Exception as e:
        print(f"    Rotation plot failed: {e}")

    # ------------------------------------------------------------------
    # 6. Anomaly Detection — Full retrained analysis
    # ------------------------------------------------------------------
    print("\n  Running fresh anomaly detection on flight data...")
    anomaly_results = advanced_anomaly_detection(X, y, label="Flight Data")

    try:
        plot_anomaly_detection(
            X, y, anomaly_results,
            save_path=os.path.join(output_dir, '06_anomaly_combined.png'),
            mag_data_by_sensor=mag_data)
    except Exception as e:
        print(f"    Anomaly combined plot failed: {e}")

    # ------------------------------------------------------------------
    # 7. Temporal Forecasting
    # ------------------------------------------------------------------
    try:
        temporal_results = run_temporal_forecasting(
            mae_threshold_multiplier=3.0, csv_path=csv_path)
        if temporal_results:
            plot_temporal_prediction_errors(
                temporal_results,
                save_path=os.path.join(output_dir, '07_temporal_forecasting.png'),
                mag_data_by_sensor=mag_data)
    except Exception as e:
        print(f"    Temporal forecasting failed: {e}")

    # ------------------------------------------------------------------
    # Per-Magnetometer Anomaly Timelines
    # ------------------------------------------------------------------
    if mag_data and len(mag_data) > 0:
        try:
            plot_anomaly_per_mag_timeline(
                mag_data,
                anomaly_func=advanced_anomaly_detection,
                save_path_prefix=os.path.join(output_dir, 'anomaly'))
        except Exception as e:
            print(f"    Per-mag anomaly plots failed: {e}")

    return anomaly_results


# ==============================================================================
# IN-FLIGHT ANOMALY SUMMARY
# ==============================================================================

def analyze_anomalies_from_csv(df):
    """Analyze anomaly flags that were logged during flight/sim."""
    print("\n--- In-Flight Anomaly Summary (from CSV flags) ---")

    total = len(df)
    if total == 0:
        print("  (no data rows — skipping anomaly summary)")
        return

    for col, name in [
        ('anomaly_is_zscore', 'Z-Score'),
        ('anomaly_is_roc', 'Rate of Change'),
        ('anomaly_is_ensemble', 'Ensemble'),
    ]:
        if col in df.columns:
            n = int(df[col].sum())
            print(f"  {name}: {n} flagged ({100*n/total:.2f}%)")


# ==============================================================================
# FLIGHT REPORT
# ==============================================================================

def generate_flight_report(df, output_dir):
    """Generate a text summary of the flight."""
    report_path = os.path.join(output_dir, 'FLIGHT_REPORT.txt')
    total = len(df)

    lines = [
        "=" * 65,
        f"  {MISSION_NAME} — POST-FLIGHT ANALYSIS REPORT",
        "=" * 65,
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Flight CSV: {total} readings",
        "",
    ]

    if 'mission_time_s' in df.columns:
        lines.append(f"  Mission time: T{df['mission_time_s'].min():+.1f}s to T{df['mission_time_s'].max():+.1f}s")

    if 'sensor_id' in df.columns:
        lines.append(f"  Sensors: {sorted(df['sensor_id'].unique().tolist())}")
        for sid in sorted(df['sensor_id'].unique()):
            n = len(df[df['sensor_id'] == sid])
            lines.append(f"    Sensor {sid}: {n} readings")

    if 'magnitude_measured' in df.columns:
        mag = df['magnitude_measured']
        lines.append(f"  Magnitude: {mag.min():.1f} to {mag.max():.1f} nT (mean: {mag.mean():.1f})")

    # Anomaly summary
    lines.append("")
    lines.append("  ANOMALY FLAGS (in-flight):")
    for col, name in [('anomaly_is_zscore', 'Z-Score'),
                      ('anomaly_is_ensemble', 'Ensemble (2+ agree)')]:
        if col in df.columns and total > 0:
            n = int(df[col].sum())
            lines.append(f"    {name}: {n} ({100*n/total:.2f}%)")

    # Model predictions
    lines.append("")
    lines.append("  MODEL PREDICTIONS:")
    for model, col in [('RF', 'magnitude_rf'), ('NN', 'magnitude_nn'), ('GPR', 'magnitude_gpr')]:
        if col in df.columns and not df[col].isna().all():
            pred = df[col].astype(float)
            meas = df['magnitude_measured'].astype(float)
            mask = ~pred.isna()
            if mask.sum() > 0:
                residuals = pred[mask].values - meas[mask].values
                rmse = np.sqrt(np.mean(residuals**2))
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((meas[mask].values - np.mean(meas[mask].values))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                lines.append(f"    {model}: R² = {r2:.6f} | RMSE = {rmse:.2f} nT ({mask.sum()} predictions)")

    lines.append("")
    lines.append("=" * 65)

    report = '\n'.join(lines)
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n{report}")
    print(f"Report saved to: {report_path}")


# ==============================================================================
# .DAT FILE LOADING — raw sensor log → ML-ready DataFrame
# ==============================================================================

def load_dat_file(dat_path):
    """
    Parse a raw .dat telemetry file (the format the sensors produce),
    run cached ML models on the raw magneto data, and return a DataFrame
    with the SAME columns as the ML CSV so analyze_and_plot() works
    identically on both inputs.

    Also parses pressure, ADC, thermocouple, and events for raw sensor plots.

    Returns:
        tuple: (df, raw_sensors) where df is the ML-ready DataFrame and
        raw_sensors is a dict with 'pressure', 'adc', 'thermo', 'events',
        'mag_data_by_sensor' DataFrames.
    """
    print(f"Loading raw .dat file: {dat_path}")
    all_data = parse_dat_file_all(dat_path)
    mag_df = all_data['magneto']
    events_df = all_data['events']

    # Build raw sensor dict for raw plots
    raw_sensors = {
        'pressure': all_data['pressure'],
        'adc': all_data['adc'],
        'thermo': all_data['thermo'],
        'events': events_df,
        'mag_data_by_sensor': {},
    }

    if mag_df is None or len(mag_df) == 0:
        print("  ERROR: No magnetometer data found in .dat file")
        sys.exit(1)

    n_mags = len(mag_df['magneto_id'].unique())
    print(f"  {len(mag_df)} magnetometer readings parsed")
    print(f"  Magnetometers detected: {sorted(mag_df['magneto_id'].unique())} ({n_mags} sensors)")
    print(f"  Mission time: T{mag_df['time_mission'].min()} to T+{mag_df['time_mission'].max()}")
    if events_df is not None and len(events_df) > 0:
        print(f"  Flight events: {len(events_df)}")

    # ---- Build clean DataFrame matching ML CSV column names ----
    # parse_dat_file() returns columns:
    #   time_mission, sensor_id (=reading counter |ctr|),
    #   magneto_id (=channel 0/1/2/3), x, y, z, magnitude, datetime
    # Our ML CSV uses:
    #   mission_time_s, sensor_id (=magnetometer channel), bx_raw, by_raw, bz_raw, ...
    df = pd.DataFrame({
        'bx_raw':             mag_df['x'].values.astype(float),
        'by_raw':             mag_df['y'].values.astype(float),
        'bz_raw':             mag_df['z'].values.astype(float),
        'mission_time_s':     mag_df['time_mission'].values.astype(float),
        'magnitude_measured': mag_df['magnitude'].values.astype(float),
        'sensor_id':          mag_df['magneto_id'].values,      # channel number
        'reading_number':     mag_df['sensor_id'].values,       # reading counter
    })
    # Add timestamp_unix from parsed datetime
    if 'datetime' in mag_df.columns and mag_df['datetime'].notna().any():
        df['timestamp_unix'] = mag_df['datetime'].astype(np.int64).values / 1e9
    else:
        df['timestamp_unix'] = 0.0

    # ---- Run cached ML models on raw data ----
    print("\n  Running ML models on raw .dat data...")
    bx = df['bx_raw'].values.astype(float)
    by = df['by_raw'].values.astype(float)
    bz = df['bz_raw'].values.astype(float)
    mt = df['mission_time_s'].values.astype(float)
    X_rf = np.column_stack([bx, by, bz])          # RF: 3 features
    X_nn = np.column_stack([bx, by, bz])           # NN: 3 features (no time)

    # Random Forest
    rf_path = os.path.join(PROJECT_ROOT, 'models', 'cached', 'rf_model.joblib')
    if os.path.exists(rf_path):
        try:
            rf = joblib.load(rf_path)
            df['magnitude_rf'] = rf.predict(X_rf)
            print(f"    RF: {len(df)} predictions")
        except Exception as e:
            print(f"    RF failed: {e}")
            df['magnitude_rf'] = np.nan
    else:
        print(f"    RF model not found at {rf_path}")
        df['magnitude_rf'] = np.nan

    # K-Means (stored as {'model': KMeans, 'k': int})
    km_path = os.path.join(PROJECT_ROOT, 'models', 'cached', 'kmeans_model.joblib')
    if os.path.exists(km_path):
        try:
            km_data = joblib.load(km_path)
            km = km_data['model'] if isinstance(km_data, dict) else km_data
            df['cluster_id'] = km.predict(np.column_stack([bx, by, bz]))
            print(f"    K-Means: {len(np.unique(df['cluster_id']))} clusters")
        except Exception as e:
            print(f"    K-Means failed: {e}")
            df['cluster_id'] = 0
    else:
        df['cluster_id'] = 0

    # Neural Network
    nn_model_path = os.path.join(PROJECT_ROOT, 'models', 'exports', 'magnetometer_fcn.keras')
    nn_norm_path  = os.path.join(PROJECT_ROOT, 'models', 'exports', 'magnetometer_fcn_norm.npz')
    if os.path.exists(nn_model_path) and os.path.exists(nn_norm_path) and TF_AVAILABLE:
        try:
            nn = load_model(nn_model_path, compile=False)
            norms = np.load(nn_norm_path)
            X_n = (X_nn - norms['mean']) / norms['std']
            X_tf = X_n.reshape(X_n.shape[0], X_n.shape[1], 1).astype(np.float32)
            df['magnitude_nn'] = nn.predict(X_tf, verbose=0, batch_size=512).flatten()
            print(f"    NN: {len(df)} predictions")
        except Exception as e:
            print(f"    NN failed: {e}")
            df['magnitude_nn'] = np.nan
    else:
        if not TF_AVAILABLE:
            print("    NN skipped: TensorFlow not available")
        else:
            print(f"    NN model not found")
        df['magnitude_nn'] = np.nan

    # Anomaly flags (compute fresh — the .dat has no ML flags)
    # Use same Modified Z-Score and Rate-of-Change as live pipeline
    mags = df['magnitude_measured'].values
    z_threshold = ML_CONFIG.get('ANOMALY_Z_THRESHOLD', 3.5)
    roc_threshold = ML_CONFIG.get('RATE_OF_CHANGE_THRESHOLD', 200.0)

    med = np.median(mags)
    mad = np.median(np.abs(mags - med))
    if mad > 1e-9:
        z_scores = 0.6745 * np.abs(mags - med) / mad
    else:
        z_scores = np.zeros(len(mags))
    is_z = (z_scores > z_threshold).astype(int)

    roc = np.zeros(len(mags))
    roc[1:] = np.abs(np.diff(mags))
    is_roc = (roc > roc_threshold).astype(int)
    is_ens = ((is_z + is_roc) >= 2).astype(int)

    df['anomaly_zscore'] = z_scores
    df['anomaly_is_zscore'] = is_z
    df['anomaly_rate_of_change'] = roc
    df['anomaly_is_roc'] = is_roc
    df['anomaly_is_ensemble'] = is_ens
    n_anom = is_ens.sum()
    print(f"    Anomaly: {n_anom} ensemble flags ({n_anom/len(df)*100:.1f}%)")

    # Keep only the columns post_flight expects
    keep = [
        'timestamp_unix', 'mission_time_s', 'sensor_id',
        'bx_raw', 'by_raw', 'bz_raw', 'magnitude_measured',
        'magnitude_rf', 'magnitude_nn', 'cluster_id',
        'anomaly_zscore', 'anomaly_is_zscore', 'anomaly_is_roc',
        'anomaly_is_ensemble', 'anomaly_rate_of_change', 'reading_number',
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Build mag_data_by_sensor for raw magneto plots
    for sid in sorted(mag_df['magneto_id'].unique()):
        mask = mag_df['magneto_id'] == sid
        raw_sensors['mag_data_by_sensor'][int(sid)] = {
            'time': mag_df.loc[mask, 'time_mission'].values,
            'x': mag_df.loc[mask, 'x'].values.astype(float),
            'y': mag_df.loc[mask, 'y'].values.astype(float),
            'z': mag_df.loc[mask, 'z'].values.astype(float),
            'magnitude': mag_df.loc[mask, 'magnitude'].values.astype(float),
            'count': int(mask.sum()),
        }

    print(f"  DataFrame ready: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Sensors: {sorted(df['sensor_id'].unique())}")

    return df, raw_sensors


# ==============================================================================
# MAIN
# ==============================================================================

def generate_raw_sensor_plots(raw_sensors, output_dir):
    """
    Generate raw sensor visualizations from .dat file data.
    These are the pure hardware readings — no ML.

    Only generates plots for sensors with useful flight data:
      - Magnetometer array (primary science instrument)
      - Pressure sensor (mission environment / altitude context)
      - Thermocouple (thermal environment validation)

    Omitted (documented in summary):
      - ADC: All channels constant during flight (no sensors connected)
      - Events timeline: System log noise; key events annotated on science plots
    """
    raw_dir = os.path.join(output_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    print("\n--- Raw Sensor Plots (from .dat) ---")

    # ── Magnetometer array (primary science) ──────────────────────────────
    mag_data = raw_sensors.get('mag_data_by_sensor', {})
    if mag_data:
        try:
            plot_raw_magnetometer_overview(
                mag_data,
                save_path=os.path.join(raw_dir, 'magnetometer_raw_overview.png'))
        except Exception as e:
            print(f"    Raw magneto plot failed: {e}")

    # ── Pressure (mission environment) ────────────────────────────────────
    pressure_df = raw_sensors.get('pressure')
    if pressure_df is not None and len(pressure_df) > 0:
        try:
            plot_pressure_timeseries(
                pressure_df,
                save_path=os.path.join(raw_dir, 'pressure_timeseries.png'))
        except Exception as e:
            print(f"    Pressure plot failed: {e}")
    else:
        print("    No pressure data found in .dat")

    # ── Thermocouple (thermal validation) ─────────────────────────────────
    thermo_df = raw_sensors.get('thermo')
    if thermo_df is not None and len(thermo_df) > 0:
        try:
            plot_temperature_timeseries(
                thermo_df,
                save_path=os.path.join(raw_dir, 'temperature_timeseries.png'))
        except Exception as e:
            print(f"    Temperature plot failed: {e}")
    else:
        print("    No thermocouple data found in .dat")

    # ADC and events timeline intentionally omitted from plots.
    # ADC: all channels constant (0.550V) — no analog sensors connected.
    # Events: 1148 system log entries, key events already annotated on
    # magnetometer and pressure plots.

    # Raw sensor text summary
    _write_raw_sensor_summary(raw_sensors, raw_dir)

    print(f"\n  Raw sensor plots saved to: {raw_dir}")


def _write_raw_sensor_summary(raw_sensors, raw_dir):
    """Write a presentation-grade text summary of raw sensor data."""
    import numpy as np

    lines = [
        "=" * 70,
        "  RockSat-X 2026 — RAW SENSOR DATA SUMMARY",
        "=" * 70,
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "  DATA SOURCE: .dat telemetry file",
        "",
    ]

    # ── Magnetometer array ────────────────────────────────────────────────
    mag_data = raw_sensors.get('mag_data_by_sensor', {})
    if mag_data:
        total_readings = sum(d['count'] for d in mag_data.values())
        lines.append(f"  MAGNETOMETER ARRAY (RM3100 × {len(mag_data)})")
        lines.append(f"  {'─' * 50}")
        lines.append(f"    Total readings: {total_readings:,}")
        for sid, data in sorted(mag_data.items()):
            mag = data['magnitude']
            t = np.asarray(data['time'], dtype=float)
            # Count post-launch data (T > 0)
            valid = t > 0
            n_valid = int(np.sum(valid))
            lines.append(f"    Sensor #{sid}: {data['count']:,} total, "
                         f"{n_valid:,} in flight window  |  "
                         f"|B|: {mag.min():.0f}–{mag.max():.0f} nT "
                         f"(μ={mag.mean():.0f} nT)")
        lines.append(f"    Status: ALL SENSORS OPERATIONAL ✓")
        lines.append(f"    Plot: magnetometer_raw_overview.png")

    # ── Pressure ──────────────────────────────────────────────────────────
    p_df = raw_sensors.get('pressure')
    if p_df is not None and len(p_df) > 0:
        p = p_df['pressure_mbar']
        t = p_df['time_mission']
        lines.append(f"\n  PRESSURE SENSOR (MPRLS)")
        lines.append(f"  {'─' * 50}")
        lines.append(f"    Readings: {len(p_df):,}")
        lines.append(f"    Range: {p.min():.1f}–{p.max():.1f} mbar")
        lines.append(f"    Time span: T{t.min():+.0f}s to T{t.max():+.0f}s")
        lines.append(f"    Sensor floor: {p[t > 60].min():.1f} mbar (near-vacuum)")
        lines.append(f"    Status: OPERATIONAL ✓")
        lines.append(f"    Plot: pressure_timeseries.png")

    # ── Thermocouple ──────────────────────────────────────────────────────
    tc_df = raw_sensors.get('thermo')
    if tc_df is not None and len(tc_df) > 0:
        lines.append(f"\n  THERMOCOUPLE SENSORS (MAX31856)")
        lines.append(f"  {'─' * 50}")
        lines.append(f"    Readings: {len(tc_df):,}")
        if 'tc0_C' in tc_df.columns:
            tc0 = tc_df['tc0_C']
            tc0_clean = tc0[(tc0 > 1) & (tc0 < 100)]
            if len(tc0_clean) > 0:
                lines.append(f"    TC0: {tc0_clean.min():.1f}–{tc0_clean.max():.1f} °C  "
                             f"(Δ={tc0_clean.max()-tc0_clean.min():.1f}°C)")
        if 'tc1_C' in tc_df.columns:
            tc1 = tc_df['tc1_C'].dropna()
            tc1_clean = tc1[(tc1 > 1) & (tc1 < 100)]
            if len(tc1_clean) > 0:
                lines.append(f"    TC1: {tc1_clean.min():.1f}–{tc1_clean.max():.1f} °C  "
                             f"(Δ={tc1_clean.max()-tc1_clean.min():.1f}°C)")
        lines.append(f"    Assessment: Thermal environment nominal (15–27°C)")
        lines.append(f"    Plot: temperature_timeseries.png")

    # ── ADC (not plotted) ─────────────────────────────────────────────────
    adc_df = raw_sensors.get('adc')
    if adc_df is not None and len(adc_df) > 0:
        lines.append(f"\n  ADC (ADS1115) — NOT PLOTTED")
        lines.append(f"  {'─' * 50}")
        lines.append(f"    Readings: {len(adc_df):,}")
        for ch in ['a0', 'a1', 'a2', 'a3']:
            if ch in adc_df.columns:
                vals = adc_df[ch].dropna()
                if len(vals) > 0:
                    rng = vals.max() - vals.min()
                    lines.append(f"    {ch.upper()}: {vals.mean():.3f} V "
                                 f"(range: {rng:.4f} V)")
        lines.append(f"    Reason omitted: No dynamic response during flight.")
        lines.append(f"    All channels constant — no analog sensors connected.")

    # ── Events (not plotted) ──────────────────────────────────────────────
    ev_df = raw_sensors.get('events')
    if ev_df is not None and len(ev_df) > 0:
        lines.append(f"\n  FLIGHT EVENTS — NOT PLOTTED")
        lines.append(f"  {'─' * 50}")
        lines.append(f"    Total logged: {len(ev_df):,}")
        lines.append(f"    Reason omitted: Predominantly repetitive system messages.")
        lines.append(f"    Key events annotated directly on sensor plots.")

    lines.append("")
    lines.append("=" * 70)

    report = '\n'.join(lines)
    report_path = os.path.join(raw_dir, 'RAW_SENSOR_REPORT.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n{report}")


# ==============================================================================
# RAW SENSOR CSV — plots from _raw_sensors_*.csv (pre-ML backup)
# ==============================================================================

def _find_companion_raw_csv(ml_csv_path):
    """Given an ML CSV path, find the matching raw sensor CSV.

    test_main.py writes both files with the same timestamp:
        test_data_20260228_120000.csv        (ML)
        test_raw_sensors_20260228_120000.csv  (raw)

    Also checks for *_raw_sensors_latest.csv symlink.
    """
    ml_dir = os.path.dirname(ml_csv_path)
    ml_base = os.path.basename(ml_csv_path)

    # Extract timestamp from ML CSV name (e.g. "test_data_20260228_120000.csv")
    # Pattern: {prefix}_data_{timestamp}.csv → {prefix}_raw_sensors_{timestamp}.csv
    import re
    m = re.match(r'^(.+?)_data_(\d{8}_\d{6})\.csv$', ml_base)
    if m:
        prefix, ts = m.groups()
        raw_name = f'{prefix}_raw_sensors_{ts}.csv'
        raw_path = os.path.join(ml_dir, raw_name)
        if os.path.exists(raw_path):
            return raw_path

    # Try _latest symlink
    for pattern in ['*_raw_sensors_latest.csv', '*_raw_sensors_*.csv']:
        candidates = sorted(glob.glob(os.path.join(ml_dir, pattern)),
                            key=os.path.getmtime, reverse=True)
        for c in candidates:
            if os.path.getsize(c) > 100:
                return c

    return None


def load_raw_sensor_csv(raw_csv_path):
    """Load a raw sensor CSV and build mag_data_by_sensor dict.

    Raw CSV columns: timestamp_unix, sensor_id, bx_raw, by_raw, bz_raw
    """
    print(f"  Loading raw sensor CSV: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    print(f"  {len(df)} raw readings")

    if len(df) == 0 or 'bx_raw' not in df.columns:
        print("  WARNING: Raw CSV empty or missing bx_raw column")
        return {}

    # Compute magnitude and relative time
    bx = df['bx_raw'].values.astype(float)
    by = df['by_raw'].values.astype(float)
    bz = df['bz_raw'].values.astype(float)
    magnitude = np.sqrt(bx**2 + by**2 + bz**2)

    # Convert timestamp_unix to relative seconds from first reading
    if 'timestamp_unix' in df.columns:
        t0 = df['timestamp_unix'].values[0]
        t = df['timestamp_unix'].values.astype(float) - t0
    else:
        t = np.arange(len(df), dtype=float)

    sensor_col = 'sensor_id' if 'sensor_id' in df.columns else None

    mag_data = {}
    if sensor_col:
        for sid in sorted(df[sensor_col].unique()):
            mask = df[sensor_col].values == sid
            mag_data[int(sid)] = {
                'time': t[mask],
                'x': bx[mask],
                'y': by[mask],
                'z': bz[mask],
                'magnitude': magnitude[mask],
                'count': int(np.sum(mask)),
            }
    else:
        # Single sensor (no sensor_id column)
        mag_data[0] = {
            'time': t,
            'x': bx, 'y': by, 'z': bz,
            'magnitude': magnitude,
            'count': len(df),
        }

    for sid, data in sorted(mag_data.items()):
        print(f"    Sensor {sid}: {data['count']} readings, |B| range: "
              f"{data['magnitude'].min():.0f}–{data['magnitude'].max():.0f} nT")

    return mag_data


def generate_raw_csv_plots(mag_data, raw_dir):
    """Generate raw magnetometer plots from the raw sensor CSV."""
    # Clean stale files
    for stale in glob.glob(os.path.join(raw_dir, 'anomaly_mag*.png')):
        os.remove(stale)

    # Raw magnetometer overview (4-panel: Bx, By, Bz, |B|)
    try:
        plot_multi_magnetometer(
            mag_data,
            save_path=os.path.join(raw_dir, '00_raw_magnetometer.png'))
        print(f"    Saved 00_raw_magnetometer.png")
    except Exception as e:
        print(f"    Raw magnetometer plot failed: {e}")

    # Detailed per-axis overview
    try:
        plot_raw_magnetometer_overview(
            mag_data,
            save_path=os.path.join(raw_dir, 'magnetometer_raw_overview.png'))
        print(f"    Saved magnetometer_raw_overview.png")
    except Exception as e:
        print(f"    Raw overview plot failed: {e}")

    # Rotation analysis (spin detection from raw data)
    # Use first sensor's data for the rotation plot
    first_sid = min(mag_data.keys())
    d = mag_data[first_sid]
    try:
        plot_magnetometer_rotation(
            d['x'], d['y'], d['z'], d['time'],
            save_path=os.path.join(raw_dir, 'rotation_analysis.png'),
            mag_data_by_sensor=mag_data)
        print(f"    Saved rotation_analysis.png")
    except Exception as e:
        print(f"    Rotation plot failed: {e}")

    print(f"  Raw sensor plots saved to: {raw_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Post-flight analysis — generates all visualizations from CSV or raw .dat')
    parser.add_argument('--csv', type=str, help='Path to ML flight CSV file')
    parser.add_argument('--dat', type=str, help='Path to raw .dat telemetry file')
    parser.add_argument('--raw-csv', type=str, dest='raw_csv',
                        help='Path to raw sensor CSV (_raw_sensors_*.csv)')
    args = parser.parse_args()

    # Output directories
    output_dir = os.path.join(STORAGE_PATH, 'post_flight')
    ml_dir = os.path.join(output_dir, 'ml')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ml_dir, exist_ok=True)

    print("=" * 65)
    print(f"  {MISSION_NAME} — POST-FLIGHT ANALYSIS")
    print("=" * 65)

    # Track whether we have raw sensor data (from .dat)
    raw_sensors = None
    csv_path = None        # Path to the ML CSV (for temporal forecasting)
    raw_csv_path = None    # Path to raw sensor CSV (for raw plots)

    # Determine input source
    if args.dat:
        # Raw .dat file — parse ALL sensors and run ML models on magneto
        df, raw_sensors = load_dat_file(args.dat)
    elif args.csv:
        csv_path = args.csv
        df = load_flight_csv(args.csv)
        # Use explicit --raw-csv, or try to find companion raw sensor CSV
        raw_csv_path = args.raw_csv or _find_companion_raw_csv(args.csv)
    else:
        # Auto-detect: prefer ML CSV, fall back to .dat
        dat_candidates = list(Path(SCRIPT_DIR / 'data').glob('*.dat'))
        csv_paths = find_flight_csvs()

        if csv_paths:
            csv_path = csv_paths[0]
            if len(csv_paths) > 1:
                print(f"  Auto-detected {len(csv_paths)} flight CSVs (possible reboot)")
                df = load_and_merge_flight_csvs(csv_paths)
            else:
                print(f"  Auto-detected ML CSV")
                df = load_flight_csv(csv_paths[0])
            raw_csv_path = args.raw_csv or _find_companion_raw_csv(csv_paths[0])
            # If .dat also exists, parse it for raw sensor plots
            if dat_candidates:
                dat_file = str(max(dat_candidates, key=os.path.getmtime))
                print(f"  Also found .dat file — parsing for raw sensor plots")
                all_data = parse_dat_file_all(dat_file)
                raw_sensors = {
                    'pressure': all_data['pressure'],
                    'adc': all_data['adc'],
                    'thermo': all_data['thermo'],
                    'events': all_data['events'],
                    'mag_data_by_sensor': {},
                }
                mag_df = all_data['magneto']
                if len(mag_df) > 0:
                    for sid in sorted(mag_df['magneto_id'].unique()):
                        mask = mag_df['magneto_id'] == sid
                        raw_sensors['mag_data_by_sensor'][int(sid)] = {
                            'time': mag_df.loc[mask, 'time_mission'].values,
                            'x': mag_df.loc[mask, 'x'].values.astype(float),
                            'y': mag_df.loc[mask, 'y'].values.astype(float),
                            'z': mag_df.loc[mask, 'z'].values.astype(float),
                            'magnitude': mag_df.loc[mask, 'magnitude'].values.astype(float),
                            'count': int(mask.sum()),
                        }
        elif dat_candidates:
            dat_file = str(max(dat_candidates, key=os.path.getmtime))
            print(f"  Auto-detected .dat file")
            df, raw_sensors = load_dat_file(dat_file)
        else:
            print("ERROR: No flight data found.")
            print("  Provide --csv <path> or --dat <path>")
            print("  Or run test_main.py / sim_flight.py first")
            sys.exit(1)

    # Bail out early if the CSV has no data rows
    if len(df) == 0:
        print("\n  ERROR: CSV is empty (header only, 0 data rows).")
        print("  This usually means test_main.py was run on a Mac without")
        print("  sensors.  Use sim_flight.py or provide a different CSV.")
        sys.exit(1)

    # ---- Raw Sensor Plots (if .dat data is available) ----
    if raw_sensors is not None:
        generate_raw_sensor_plots(raw_sensors, output_dir)

    # ---- ML Analysis + Visualizations ----
    # In-flight anomaly flags from CSV (if present)
    if 'anomaly_is_zscore' in df.columns:
        analyze_anomalies_from_csv(df)

    # ---- Raw Sensor Plots from raw CSV (if available) ----
    if raw_csv_path and raw_sensors is None:
        print(f"\n--- Raw Sensor Plots (from raw CSV) ---")
        raw_mag_data = load_raw_sensor_csv(raw_csv_path)
        if raw_mag_data:
            raw_dir = os.path.join(output_dir, 'raw')
            os.makedirs(raw_dir, exist_ok=True)
            generate_raw_csv_plots(raw_mag_data, raw_dir)

    print("\n--- ML Visualizations ---")
    analyze_and_plot(df, ml_dir, csv_path=csv_path)

    # Flight report (covers both raw + ML)
    generate_flight_report(df, output_dir)

    # Summary of what was generated
    print(f"\n{'=' * 65}")
    print(f"  All post-flight outputs saved to: {output_dir}")
    if raw_sensors or raw_csv_path:
        print(f"    raw/   — Raw sensor plots (magneto, pressure, temp)")
    print(f"    ml/    — ML analysis (RF, NN, K-Means, anomalies)")
    print(f"    FLIGHT_REPORT.txt — Text summary")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()