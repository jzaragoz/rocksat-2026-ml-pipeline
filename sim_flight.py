#!/usr/bin/env python3
"""
RockSat-X 2026 — Flight Simulation (Mac / Dev)
================================================
Replays the Ghost CSV through the flight ML pipeline to verify
everything works before deploying to the Pi.

This is FAST because it:
    - Uses vectorized batch predictions (not per-sample TF calls)
    - Skips GPR (too slow per-sample, not needed for validation)
    - Processes the full 17,139-row Ghost dataset in seconds

Usage:
    python sim_flight.py                    # Full simulation, default speed
    python sim_flight.py --live             # With real-time visualization window
    python sim_flight.py --live --speed 20  # Faster replay (20x real time)
    python sim_flight.py --csv other.csv    # Use a different CSV
    python sim_flight.py --no-nn            # Skip NN (fastest)

Output:
    output/flight_data/sim_data_YYYYMMDD_HHMMSS.csv
    → Feed this into post_flight.py for visualizations
"""

import os
import sys
import csv
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import deque
import argparse

SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR / 'lib'))

from config import STORAGE_PATH, ML_CONFIG, HAILO_HEF_PATH, MISSION_NAME

import joblib

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

HAILO_AVAILABLE = False
try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                                InferVStreams, ConfigureParams,
                                InputVStreamParams, OutputVStreamParams, FormatType)
    HAILO_AVAILABLE = True
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description='Flight simulation on Mac')
    # Auto-detect default CSV: newest CSV in data/ (no hardcoded preference)
    import glob as _g
    csvs = sorted(_g.glob(str(SCRIPT_DIR / 'data' / '*.csv')),
                  key=os.path.getmtime, reverse=True)
    default_csv = os.path.relpath(csvs[0], SCRIPT_DIR) if csvs else 'data/flight_data.csv'
    parser.add_argument('--csv', default=default_csv,
                        help='Input CSV to replay')
    parser.add_argument('--no-nn', action='store_true',
                        help='Skip neural network (fastest)')
    parser.add_argument('--live', action='store_true',
                        help='Open real-time anomaly detection visualization window')
    parser.add_argument('--speed', type=float, default=10.0,
                        help='Replay speed multiplier for --live (default: 10x)')
    args = parser.parse_args()

    csv_path = str(SCRIPT_DIR / args.csv)
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    print("=" * 60)
    print(f"  {MISSION_NAME} — Flight Simulation")
    print("=" * 60)

    # ── Load models ──────────────────────────────────────────────
    print("\nLoading models...")
    cache = SCRIPT_DIR / 'models' / 'cached'
    exports = SCRIPT_DIR / 'models' / 'exports'

    rf = joblib.load(cache / 'rf_model.joblib')
    print(f"  ✓ Random Forest")

    km_data = joblib.load(cache / 'kmeans_model.joblib')
    kmeans = km_data['model'] if isinstance(km_data, dict) else km_data
    print(f"  ✓ K-Means")

    nn_model, nn_mean, nn_std = None, None, None
    hailo_active = False
    hailo_ctx = None

    if not args.no_nn:
        fcn = exports / 'magnetometer_fcn.keras'
        nrm = exports / 'magnetometer_fcn_norm.npz'

        # Try Hailo NPU first (Pi with Hailo HAT+)
        hef_path = Path(HAILO_HEF_PATH)
        if HAILO_AVAILABLE and hef_path.exists() and nrm.exists():
            try:
                n = np.load(nrm)
                nn_mean, nn_std = n['mean'], n['std']
                hailo_hef = HEF(str(hef_path))
                hailo_vdevice = VDevice()
                cp = ConfigureParams.create_from_hef(
                    hailo_hef, interface=HailoStreamInterface.PCIe)
                hailo_ng = hailo_vdevice.configure(hailo_hef, cp)[0]
                hailo_in = hailo_hef.get_input_vstream_infos()
                hailo_out = hailo_hef.get_output_vstream_infos()
                hailo_ip = InputVStreamParams.make_from_network_group(
                    hailo_ng, quantized=False, format_type=FormatType.FLOAT32)
                hailo_op = OutputVStreamParams.make_from_network_group(
                    hailo_ng, quantized=False, format_type=FormatType.FLOAT32)
                hailo_ctx = {
                    'ng': hailo_ng, 'in': hailo_in, 'out': hailo_out,
                    'ip': hailo_ip, 'op': hailo_op
                }
                hailo_active = True
                print(f"  ✓ Neural Network (Hailo NPU — 26 TOPS)")
            except Exception as e:
                print(f"  ⚠ Hailo init failed: {e} — trying TF CPU")

        # Fallback to TensorFlow CPU
        if not hailo_active and TF_AVAILABLE:
            if fcn.exists() and nrm.exists():
                nn_model = load_model(fcn)
                n = np.load(nrm)
                nn_mean, nn_std = n['mean'], n['std']
                print(f"  ✓ Neural Network (TF CPU fallback)")
        elif not hailo_active:
            print(f"  ⚠ No NN backend available (no Hailo, no TF)")
    else:
        print(f"  — Neural Network skipped (--no-nn)")

    # ── Load CSV ─────────────────────────────────────────────────
    print(f"\nLoading: {os.path.basename(csv_path)}")

    # Auto-detect legacy (pre-GHOST) CSV format
    from data_loader import is_legacy_csv, parse_legacy_csv
    if is_legacy_csv(csv_path):
        print("  Detected legacy (pre-GHOST) CSV format")
        df = parse_legacy_csv(csv_path)
    else:
        df = pd.read_csv(csv_path)
    total = len(df)
    print(f"  {total} readings")

    # Normalize column names
    bx = df['X'].values if 'X' in df.columns else df['Bx'].values
    by = df['Y'].values if 'Y' in df.columns else df['By'].values
    bz = df['Z'].values if 'Z' in df.columns else df['Bz'].values
    t  = df['Time'].values if 'Time' in df.columns else df['T'].values
    sensor_ids = df['Sensor'].values if 'Sensor' in df.columns else np.zeros(total, dtype=int)
    magnitude = np.sqrt(bx**2 + by**2 + bz**2)

    # ── Batch ML predictions (FAST) ─────────────────────────────
    print("\nRunning ML pipeline...")
    t0 = time.time()

    # RF — batch (3 features: bx, by, bz — no time)
    X_rf = np.column_stack([bx, by, bz])
    rf_pred = rf.predict(X_rf)
    rf_time = time.time() - t0
    rf_r2 = 1 - np.sum((magnitude - rf_pred)**2) / np.sum((magnitude - np.mean(magnitude))**2)
    print(f"  RF:      {total} predictions in {rf_time:.2f}s  "
          f"(R² = {rf_r2:.6f}, {total/rf_time:.0f} pred/sec)")

    # K-Means — batch
    clusters = kmeans.predict(np.column_stack([bx, by, bz]))
    print(f"  K-Means: {len(np.unique(clusters))} clusters assigned")

    # NN — batch (Hailo NPU or TF CPU)
    # NN now uses 3 features [bx, by, bz] — same as RF
    X_nn = np.column_stack([bx, by, bz])
    nn_pred = np.full(total, np.nan)
    nn_r2 = float('nan')
    if hailo_active and nn_mean is not None:
        # Hailo: batch via sequential single-sample (NPU is fast per-inference)
        t1 = time.time()
        X_norm = (X_nn - nn_mean) / nn_std
        in_shape = hailo_ctx['in'][0].shape
        in_name = hailo_ctx['in'][0].name
        out_name = hailo_ctx['out'][0].name
        with hailo_ctx['ng'].activate():
            with InferVStreams(hailo_ctx['ng'], hailo_ctx['ip'], hailo_ctx['op']) as pipe:
                for i in range(total):
                    x_in = X_norm[i].astype(np.float32).reshape(1, *in_shape)
                    res = pipe.infer({in_name: x_in})
                    nn_pred[i] = float(res[out_name].flatten()[0])
        nn_time = time.time() - t1
        nn_r2 = 1 - np.sum((magnitude - nn_pred)**2) / np.sum((magnitude - np.mean(magnitude))**2)
        print(f"  NN:      {total} predictions in {nn_time:.2f}s via Hailo NPU  "
              f"(R² = {nn_r2:.6f}, {total/nn_time:.0f} pred/sec)")
    elif nn_model is not None:
        t1 = time.time()
        X_norm = (X_nn - nn_mean) / nn_std
        X_tf = X_norm.reshape(total, -1, 1).astype(np.float32)
        nn_pred = nn_model.predict(X_tf, verbose=0, batch_size=256).flatten()
        nn_time = time.time() - t1
        nn_r2 = 1 - np.sum((magnitude - nn_pred)**2) / np.sum((magnitude - np.mean(magnitude))**2)
        print(f"  NN:      {total} predictions in {nn_time:.2f}s via TF CPU  "
              f"(R² = {nn_r2:.6f}, {total/nn_time:.0f} pred/sec)")

    # ── Anomaly detection (vectorized) ───────────────────────────
    print("\n  Anomaly detection...")

    # Modified Z-Score (rolling window of 500)
    z_scores = np.zeros(total)
    is_z = np.zeros(total, dtype=int)
    z_thresh = ML_CONFIG.get('ANOMALY_Z_THRESHOLD', 3.5)
    window = 500
    for i in range(20, total):
        start = max(0, i - window)
        h = magnitude[start:i+1]
        med = np.median(h)
        mad = np.median(np.abs(h - med))
        if mad > 1e-9:
            z_scores[i] = 0.6745 * abs(magnitude[i] - med) / mad
        is_z[i] = int(z_scores[i] > z_thresh)

    # Rate of change
    roc = np.abs(np.diff(magnitude, prepend=magnitude[0]))
    roc_thresh = ML_CONFIG.get('RATE_OF_CHANGE_THRESHOLD', 200.0)
    is_roc = (roc > roc_thresh).astype(int)

    # Ensemble
    is_ens = ((is_z + is_roc) >= 2).astype(int)

    n_z = is_z.sum()
    n_roc = is_roc.sum()
    n_ens = is_ens.sum()
    print(f"    Z-Score:  {n_z} flagged ({100*n_z/total:.2f}%)")
    print(f"    RoC:      {n_roc} flagged ({100*n_roc/total:.2f}%)")
    print(f"    Ensemble: {n_ens} flagged ({100*n_ens/total:.2f}%)")

    # ── Real-time terminal output (print every reading) ──────────
    print(f"\n{'='*60}")
    print("  SIMULATED FLIGHT TELEMETRY  (Ctrl+C to skip)")
    print(f"{'='*60}")
    anomaly_count = 0
    telemetry_interrupted = False
    try:
        for i in range(total):
            mt = t[i]
            mt_int = round(mt)
            mt_sign = '+' if mt_int > 0 else ''
            sid = int(sensor_ids[i])
            ts_str = time.asctime(time.localtime())

            # Magneto line (team .dat format)
            print(f"T{mt_sign}{mt_int}: |{i+1}|: Magneto #{sid} "
                  f"X: {int(bx[i])} Y: {int(by[i])} Z: {int(bz[i])} "
                  f"( {ts_str} )")

            # ML line (team .dat format + ML results)
            rf_str = f'RF={rf_pred[i]:.1f}' if not np.isnan(rf_pred[i]) else 'RF=---'
            nn_str = f'NN={nn_pred[i]:.1f}' if not np.isnan(nn_pred[i]) else 'NN=---'
            anom_tag = ''
            if is_ens[i]:
                anomaly_count += 1
                anom_tag = f' ANOMALY#{anomaly_count}'
            print(f"T{mt_sign}{mt_int}: |{i+1}|: ML{anom_tag} "
                  f"|B|={magnitude[i]:.1f} {rf_str} {nn_str} C={int(clusters[i])} "
                  f"Z={z_scores[i]:.2f} RoC={roc[i]:.1f} "
                  f"( {ts_str} )")
    except KeyboardInterrupt:
        telemetry_interrupted = True
        anomaly_count = int(is_ens.sum())
        print(f"\n  Telemetry interrupted at reading {i+1}/{total}")

    if not telemetry_interrupted:
        print(f"{'='*60}")
    print(f"  Telemetry: {total} readings, {anomaly_count} anomalies")
    print(f"{'='*60}")

    # ── Write CSV (same format as flight main.py) ────────────────
    out_dir = os.path.join(STORAGE_PATH, 'flight_data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'sim_data_latest.csv')

    COLUMNS = [
        'timestamp_unix', 'mission_time_s', 'sensor_id',
        'bx_raw', 'by_raw', 'bz_raw',
        'magnitude_measured', 'magnitude_rf', 'magnitude_nn',
        'cluster_id',
        'anomaly_zscore', 'anomaly_is_zscore', 'anomaly_is_roc',
        'anomaly_is_ensemble', 'anomaly_rate_of_change', 'reading_number',
    ]

    now = time.time()
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i in range(total):
            w.writerow([
                f'{now + i * 0.022:.6f}',      # fake timestamps at ~45 Hz
                f'{t[i]:.4f}',
                int(sensor_ids[i]),
                bx[i], by[i], bz[i],
                f'{magnitude[i]:.2f}',
                f'{rf_pred[i]:.2f}',
                f'{nn_pred[i]:.2f}' if not np.isnan(nn_pred[i]) else '',
                int(clusters[i]),
                f'{z_scores[i]:.4f}',
                is_z[i],
                is_roc[i],
                is_ens[i],
                f'{roc[i]:.2f}',
                i + 1,
            ])

    elapsed = time.time() - t0
    print(f"\n  CSV saved: {out_path}")
    print(f"  Total time: {elapsed:.1f}s for {total} readings "
          f"({total/elapsed:.0f} readings/sec)")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Readings:    {total}")
    print(f"  RF R²:       {rf_r2:.6f}")
    if hailo_active or nn_model is not None:
        nn_backend = "Hailo NPU" if hailo_active else "TF CPU"
        print(f"  NN R²:       {nn_r2:.6f}  ({nn_backend})")
    print(f"  Anomalies:   {n_ens} ensemble ({100*n_ens/total:.2f}%)")
    print(f"  Output:      {out_path}")
    print(f"\n  Next: python post_flight.py --csv {out_path}")
    print(f"{'='*60}")

    # ── Live visualization (separate window) ─────────────────────
    if args.live:
        print(f"\n  Opening real-time visualization ({args.speed:.0f}x speed)...")
        print(f"  Close the window to exit.\n")
        try:
            from live_display import replay_simulation
            replay_simulation(
                t=t,
                magnitude=magnitude,
                rf_pred=rf_pred,
                nn_pred=nn_pred,
                z_scores=z_scores,
                roc=roc,
                is_z=is_z,
                is_roc=is_roc,
                is_ensemble=is_ens,
                clusters=clusters,
                speed=args.speed,
                z_thresh=z_thresh,
                roc_thresh=roc_thresh,
            )
        except Exception as e:
            print(f"  ⚠ Visualization failed: {e}")
            print(f"  (Make sure matplotlib + tkinter are installed: pip install matplotlib)")


if __name__ == '__main__':
    main()
