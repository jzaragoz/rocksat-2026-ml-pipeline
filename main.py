#!/usr/bin/env python3
"""
RockSat-X 2026 — ML Training, Validation & Visualization
==========================================================
Universal pipeline: trains on ALL available datasets combined, then
generates per-dataset visualizations.

1. Load combined training data (Virginia 2025 + GHOST 2025)
2. Train universal models (RF, NN, GPR) on combined data
3. Load specific visualization dataset (--csv flag)
4. Apply universal models + per-dataset clustering/anomaly detection
5. Generate all visualizations for the specified dataset

Usage:
    python main.py --csv data/UPR_2025_Flight.csv          # Virginia plots
    python main.py --csv data/Magneto_Fixed_Timeline.csv    # GHOST plots
    python main.py                                           # Auto-detect

Other scripts:
    test_main.py   — FLIGHT software (Pi with sensors, CSV logging, no viz)
    sim_flight.py  — Replay CSV through the flight pipeline (Mac)
    post_flight.py — Generate visualizations from flight CSV after recovery

Models trained here deploy to the Pi for real-time inference.
"""

import os
import sys
import time
import atexit
import platform
import numpy as np
from pathlib import Path

# ==============================================================================
# SETUP
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR / 'lib'))

# ==============================================================================
# IMPORT LOCAL MODULES
# ==============================================================================

from config import (
    IS_RASPBERRY_PI, STORAGE_PATH,
    MISSION_TIMELINE, HAILO_HEF_PATH, ML_CONFIG, MISSION_NAME,
    TRAINING_DATASETS
)
from utils import setup_model_storage, find_telemetry_file
from data_loader import (
    load_from_dat_file, load_from_csv, load_from_sensor,
    load_combined_training_data, detect_time_gaps
)
from models import (
    run_random_forest, run_kmeans, run_nn_model,
    run_gaussian_process, gpr_predict_full, TENSORFLOW_AVAILABLE
)
from anomaly import advanced_anomaly_detection
from temporal_models import run_temporal_forecasting
from visualization import (
    plot_multi_magnetometer, plot_clusters, plot_rf_actual_vs_predicted,
    plot_rf_timeseries, plot_gpr_uncertainty, plot_anomaly_detection,
    plot_nn_results, plot_magnetometer_rotation,
    plot_anomaly_per_mag_timeline, plot_temporal_prediction_errors
)
from sensors import MultiMagnetometerReader
from controller import AIController, HAILO_AVAILABLE

# ==============================================================================
# SKLEARN IMPORT
# ==============================================================================
from sklearn.ensemble import RandomForestRegressor


def print_header():
    """Print startup banner."""
    print("=" * 70)
    print(f"{MISSION_NAME} — ML Telemetry Analysis")
    print("RockSat-X 2026 Pre-Flight Validation")
    print("=" * 70)
    print(f"Platform:    {'Raspberry Pi 5 (Flight)' if IS_RASPBERRY_PI else platform.system()}")
    print(f"TensorFlow:  {'Available' if TENSORFLOW_AVAILABLE else 'Not available'}")
    print(f"Hailo NPU:   {'Available' if HAILO_AVAILABLE else 'Not available (CPU mode)'}")
    print("=" * 70)


def main():
    """Main execution function."""
    import argparse
    parser = argparse.ArgumentParser(description='ML Training, Validation & Visualization')
    parser.add_argument('--csv', default=None,
                        help='Path to input CSV (auto-detects if not specified)')
    args = parser.parse_args()

    print_header()

    # =========================================================================
    # SETUP
    # =========================================================================

    print("\nCurrent Working Directory:", os.getcwd())

    storage_path = setup_model_storage()
    print(f"Model Storage Path: {storage_path}")

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    if args.csv:
        file_path = args.csv
        file_type = 'csv' if file_path.endswith('.csv') else 'dat'
    else:
        # Auto-detect telemetry file (newest CSV in data/)
        file_path, file_type = find_telemetry_file()

    if file_path is None or not os.path.exists(str(file_path)):
        print("\nERROR: No telemetry file found.")
        print("   Use --csv <path> to specify, or place a .csv in data/")
        sys.exit(1)

    print(f"\nFound telemetry file: {os.path.basename(file_path)}")
    print(f"   File type: {file_type}")

    # Initialize variables
    X = np.zeros((0, 3))
    y_values = np.zeros(0)
    mag_data_by_sensor = {}
    mag_sensor = None

    # =========================================================================
    # SENSOR MODE vs FILE MODE
    # =========================================================================

    # *********************************
    # True for sensor, False for file
    use_sensor = False
    # *********************************

    if use_sensor:
        try:
            mag_sensor = MultiMagnetometerReader()
            print('\nInitializing RM3100 sensors...')

            working_sensors = mag_sensor.verify_sensors()
            if not any(working_sensors.values()):
                print("No working sensors found. Falling back to file data.")
                use_sensor = False
            else:
                active = [c for c, status in working_sensors.items() if status]
                print(f"Working sensors: {active}")

                df, X, y_values, steps, features, x, y, z, t = load_from_sensor(
                    mag_sensor, n_samples=1000)

                if len(X) == 0:
                    print("Failed to collect initial sensor data. Falling back to file data.")
                    use_sensor = False

        except Exception as e:
            print(f'Sensor initialization failed: {e}')
            print('Using file data instead.')
            use_sensor = False

    if not use_sensor:
        print(f'\nLoading data from: {os.path.basename(file_path)}')

        if file_type == 'csv':
            df, X, y_values, steps, features, x, y, z, t, mag_data_by_sensor = load_from_csv(file_path)
        else:
            df, X, y_values, steps, features, x, y, z, t, mag_data_by_sensor = load_from_dat_file(file_path)

    # =========================================================================
    # DATA VALIDATION
    # =========================================================================

    if len(X) == 0:
        print("\nERROR: No data loaded. Exiting.")
        sys.exit(1)

    # X is 3-col [Bx, By, Bz]; t is returned separately
    t_vals = t
    # All models now use 3-feature X [Bx, By, Bz] — no time
    # Build 4-col X_viz only for visualization (time axis on plots)
    X_viz = np.column_stack([X, t_vals])
    print(f"\nLoaded {len(X):,} samples")
    print(f"   Time range: T{t_vals.min():.1f}s to T{t_vals.max():.1f}s")
    print(f"   Magnetic field: {y_values.min():.1f} to {y_values.max():.1f} nT")

    # Detect time gaps (boot resets)
    gaps = detect_time_gaps(t_vals, threshold_seconds=5.0)
    if gaps:
        print(f"   Found {len(gaps)} gap(s) in timeline")

    # =========================================================================
    # COMBINED TRAINING — Universal models from ALL datasets
    # =========================================================================
    # Models trained on combined Virginia + GHOST data generalize across
    # different magnetic environments. This enables deployment without
    # retraining at the launch site.

    combined_X, combined_y = load_combined_training_data(TRAINING_DATASETS)

    # Fall back to single-dataset training if combined loading failed
    if len(combined_X) == 0:
        print("WARNING: Combined training data empty. Training on visualization dataset only.")
        combined_X, combined_y = X, y_values

    # ----- Random Forest (trained on combined data) -----
    rf_model, _, _, _, _, r2_rf_combined = run_random_forest(combined_X, combined_y)
    if rf_model is None:
        rf_model = RandomForestRegressor()  # Dummy model

    # ----- Neural Network (trained on combined data) -----
    nn_model = None
    mse_nn = r2_nn = None

    if TENSORFLOW_AVAILABLE:
        nn_model, _, _, _, mse_nn, r2_nn, history = run_nn_model(
            combined_X, combined_y, features=3, epochs=100, batch_size=32)
    else:
        print("\nTensorFlow not available - Neural Network SKIPPED")

    # ----- Gaussian Process Regression (trained on combined data) -----
    gpr_model, _, _, _, _, _, r2_gpr_combined = run_gaussian_process(
        combined_X, combined_y)

    # =========================================================================
    # PER-DATASET EVALUATION — Apply universal models to visualization dataset
    # =========================================================================

    from sklearn.metrics import r2_score as _r2_score, mean_squared_error as _mse

    # Random Forest — per-dataset
    y_pred_rf_full = rf_model.predict(X) if rf_model is not None else np.zeros(len(X))
    r2_rf = _r2_score(y_values, y_pred_rf_full) if len(y_values) > 1 else 0.0

    # Split viz dataset for scatter plot (20% held out for visual clarity)
    from sklearn.model_selection import train_test_split as _tts
    _idx_train, _idx_test = _tts(np.arange(len(X)), test_size=0.2, random_state=42)
    X_rf_test = X[_idx_test]
    y_rf_test = y_values[_idx_test]
    y_pred_rf = rf_model.predict(X_rf_test) if rf_model is not None else np.zeros(len(X_rf_test))

    # Neural Network — per-dataset
    X_nn_test = y_nn_test = y_pred_nn = None
    t_nn_test = None
    if nn_model is not None:
        from models import load_cached_nn_model, _hailo_batch_predict, HAILO_NN_AVAILABLE
        _, nn_mean, nn_std, _ = load_cached_nn_model()
        if nn_mean is not None:
            X_nn_test = X[_idx_test]
            y_nn_test = y_values[_idx_test]
            X_nn_norm = (X_nn_test - nn_mean) / nn_std
            # Try Hailo, fall back to TF
            y_pred_nn = None
            if HAILO_NN_AVAILABLE and Path(HAILO_HEF_PATH).exists():
                y_pred_nn = _hailo_batch_predict(X_nn_norm, HAILO_HEF_PATH)
            if y_pred_nn is None:
                X_tf_in = X_nn_norm.reshape(len(X_nn_norm), -1, 1).astype(np.float32)
                y_pred_nn = nn_model.predict(X_tf_in, verbose=0).flatten()
            r2_nn = _r2_score(y_nn_test, y_pred_nn) if len(y_nn_test) > 1 else 0.0
            t_nn_test = t_vals[_idx_test]

    # GPR — per-dataset (use gpr_predict_full for smooth visualization)
    r2_gpr = r2_gpr_combined  # Use combined metric for summary
    t_gpr_test = t_vals[_idx_test] if gpr_model is not None else None

    # =========================================================================
    # PER-DATASET CLUSTERING (K-Means, optimal K via silhouette score)
    # =========================================================================
    # Cached model is used during live flight for single-point classification.
    kmeans_labels, kmeans_centers = run_kmeans(X[:, :3], k='auto', max_k=6)

    # =========================================================================
    # PER-DATASET ANOMALY DETECTION (adaptive thresholds)
    # =========================================================================

    anomaly_results = advanced_anomaly_detection(X, y_values)

    # ----- Temporal Forecasting (5th anomaly method) -----
    temporal_results = run_temporal_forecasting(mae_threshold_multiplier=3.0,
                                                csv_path=file_path)
    n_temporal = int(temporal_results['anomaly_mask'].sum()) if temporal_results else 0

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================

    n_anomalies = np.sum(anomaly_results['ensemble_anomalies']) if anomaly_results['ensemble_anomalies'] is not None else 0
    n_iso = np.sum(anomaly_results['isolation_forest']['anomalies']) if anomaly_results['isolation_forest']['anomalies'] is not None else 0
    n_zscore = np.sum(anomaly_results['z_score']['anomalies']) if anomaly_results['z_score']['anomalies'] is not None else 0
    n_lof = np.sum(anomaly_results['lof']['anomalies']) if anomaly_results['lof']['anomalies'] is not None else 0

    z_thresh = ML_CONFIG.get('ANOMALY_Z_THRESHOLD', 3.5)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    n_kmeans_clusters = len(set(kmeans_labels))

    print(f"""
Training Data:      {len(combined_X):,} combined samples ({len(TRAINING_DATASETS)} datasets)
Visualization:      {os.path.basename(file_path)} — {len(X):,} samples | T{t_vals.min():.0f}s to T{t_vals.max():.0f}s
Platform:           {'Raspberry Pi 5' if IS_RASPBERRY_PI else platform.system()} | Hailo: {'ON' if HAILO_AVAILABLE else 'OFF'}

Regression Models (universal, trained on combined data):
  Random Forest      R² = {r2_rf:.6f}  (on this dataset)
  Neural Network     {'R² = ' + f'{r2_nn:.6f}' + (' [Hailo NPU]' if nn_model and getattr(nn_model, '_nn_backend', '') == 'hailo' else ' [TF CPU]') if r2_nn else 'Skipped'}
  Gaussian Process   {'R² = ' + f'{r2_gpr:.6f}' if gpr_model else 'Failed'}

Temporal Forecasting:
  RF Temporal         R² = {temporal_results['r2']:.6f}  MAE = {temporal_results['mae']:.2f} nT

Clustering (K-Means, optimal K via silhouette score):
  K-Means Clusters   {n_kmeans_clusters} clusters

Anomaly Detection (per-dataset, adaptive thresholds):
  Isolation Forest   {n_iso:>5} flagged  ({100*n_iso/len(X):.1f}%)
  Z-Score (>{z_thresh}σ)       {n_zscore:>5} flagged  ({100*n_zscore/len(X):.2f}%)
  Local Outlier Factor {n_lof:>3} flagged  ({100*n_lof/len(X):.1f}%)
  Prediction Error   {n_temporal:>5} flagged  ({100*n_temporal/temporal_results['n_samples']:.1f}%)
  Ensemble (2+ agree)  {n_anomalies:>3} flagged  ({100*n_anomalies/len(X):.2f}%)

Output: {STORAGE_PATH}
""")
    print("=" * 60)

    # =========================================================================
    # REAL-TIME SENSOR CONTROLLER (if enabled)
    # =========================================================================

    aic = None
    if use_sensor and mag_sensor is not None:
        try:
            if nn_model is not None and rf_model is not None and cluster_model is not None:
                aic = AIController(
                    mag_sensor, nn_model, rf_model, cluster_model,
                    gpr_model=gpr_model,
                    anomaly_detectors=anomaly_results.get('models', {})
                )
                # Load NN normalization params for correct inference
                _nn_norm_path = os.path.join(str(SCRIPT_DIR), 'models', 'exports',
                                             'magnetometer_fcn_norm.npz')
                if os.path.exists(_nn_norm_path):
                    _nn_norms = np.load(_nn_norm_path)
                    aic.nn_mean = _nn_norms['mean']
                    aic.nn_std = _nn_norms['std']
                aic.start()
                print('RM3100 Live Monitoring Started')
        except Exception as e:
            print(f"Error initializing sensor controller: {e}")
            aic = None

    def shutdown_hook():
        if aic is not None:
            aic.save_models()
    atexit.register(shutdown_hook)

    # =========================================================================
    # GENERATE VISUALIZATIONS
    # =========================================================================

    # Clean stale per-sensor anomaly plots (e.g. anomaly_mag3.png from 4-sensor data
    # lingering when running 3-sensor data)
    import glob as _glob
    for stale in _glob.glob(os.path.join(STORAGE_PATH, 'anomaly_mag*.png')):
        os.remove(stale)

    print("\nGenerating visualizations...\n")

    # 0. Raw Magnetometer Overview (non-ML baseline)
    if mag_data_by_sensor:
        plot_multi_magnetometer(
            mag_data_by_sensor,
            save_path=os.path.join(STORAGE_PATH, '00_raw_magnetometer.png'))

    # 1. Random Forest — Scatter
    if y_rf_test is not None and y_pred_rf is not None:
        plot_rf_actual_vs_predicted(
            y_rf_test, y_pred_rf, r2_rf,
            save_path=os.path.join(STORAGE_PATH, '01_random_forest.png'))

    # 1b. Random Forest — Time Series (full dataset prediction)
    if rf_model is not None:
        plot_rf_timeseries(
            X_viz, y_values, y_pred_rf_full,
            save_path=os.path.join(STORAGE_PATH, '01b_rf_timeseries.png'),
            mag_data_by_sensor=mag_data_by_sensor)

    # 2. K-Means Clusters (optimal K selected by silhouette score)
    plot_clusters(
        X_viz, kmeans_labels, kmeans_centers,
        save_path=os.path.join(STORAGE_PATH, '02_kmeans_clusters.png'),
        title=f"{MISSION_NAME} — K-Means Cluster Assignment")

    # 3. Neural Network
    if TENSORFLOW_AVAILABLE and X_nn_test is not None:
        plot_nn_results(
            X_nn_test, y_nn_test, y_pred_nn,
            save_path=os.path.join(STORAGE_PATH, '03_neural_network.png'),
            time_vals=t_nn_test, r2=r2_nn)

    # 4. Gaussian Process Regression — predict on FULL dataset for smooth plot
    if gpr_model is not None:
        y_pred_gpr_full, y_std_gpr_full = gpr_predict_full(X, y_values)
        if y_pred_gpr_full is not None:
            plot_gpr_uncertainty(
                X, y_values, y_pred_gpr_full, y_std_gpr_full,
                save_path=os.path.join(STORAGE_PATH, '04_gaussian_process.png'),
                time_vals=t_vals, r2=r2_gpr, mag_data_by_sensor=mag_data_by_sensor)

    # 5. Magnetometer Rotation (Spin Analysis)
    plot_magnetometer_rotation(
        x, y, z, t,
        save_path=os.path.join(STORAGE_PATH, '05_rotation_analysis.png'),
        mag_data_by_sensor=mag_data_by_sensor)

    # 6. Anomaly Detection — All Methods Combined
    plot_anomaly_detection(
        X_viz, y_values, anomaly_results,
        save_path=os.path.join(STORAGE_PATH, '06_anomaly_combined.png'),
        mag_data_by_sensor=mag_data_by_sensor)

    # 7. Temporal Forecasting — Predicted vs Actual + Prediction Errors
    if temporal_results:
        plot_temporal_prediction_errors(
            temporal_results,
            save_path=os.path.join(STORAGE_PATH, '07_temporal_forecasting.png'),
            mag_data_by_sensor=mag_data_by_sensor)

    # Per-Magnetometer Anomaly Detection
    if mag_data_by_sensor and len(mag_data_by_sensor) > 0:
        plot_anomaly_per_mag_timeline(
            mag_data_by_sensor,
            anomaly_func=advanced_anomaly_detection,
            save_path_prefix=os.path.join(STORAGE_PATH, 'anomaly'))

    print("\nAll visualizations saved.")
    print("=" * 60)

    # =========================================================================
    # SENSOR MONITORING LOOP (only when use_sensor=True)
    # =========================================================================

    if use_sensor and aic is not None:
        print("\nEntering real-time monitoring mode...")
        print("Press Ctrl+C to stop.\n")
        try:
            while True:
                if not aic.running:
                    raise RuntimeError('Sensor monitoring stopped unexpectedly')
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            aic.stop()
        except RuntimeError as e:
            print(e)
            aic.stop()
    else:
        print("\nAnalysis complete. No live monitoring (use_sensor=False).")


if __name__ == '__main__':
    main()
