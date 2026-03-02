#!/usr/bin/env python3
"""
RockSat-X 2026 — ML MAGNETOMETER PIPELINE (Raspberry Pi 5)
===========================================================
Runs as a subprocess launched by flight_controller.py (replaces magnetometerMVprintable.py).
Can also be run standalone for bench testing.

Launched by flight_controller.py:
    python test_main.py --st <start_time> --channels 0 1 2 --prints

Standalone:
    python test_main.py                  # Auto-detects sensors
    python test_main.py --test           # Bench test mode
    python test_main.py --channels 0 1 2 # Specify MUX channels

What this does:
    - Loads CACHED pre-trained models (never retrains)
    - Reads RM3100 magnetometers via I2C MUX at ~45 Hz
    - Runs RF, NN (Hailo or CPU), K-Means, anomaly detection on every reading
    - Logs every reading + prediction + anomaly flag to CSV (flushed to disk)
    - Sends telemetry over serial port (when --prints enabled)
    - Auto-shuts down at T+336s (experiment power off)
    - Detects poweroff30 GPIO flag for graceful 30s shutdown
    - Graceful shutdown on Ctrl+C or SIGTERM
    - Optional: --live opens real-time anomaly detection visualization window
    - Prints every reading + ML results to terminal (with --prints or --test)

Other scripts:
    flight_controller.py — Master controller (replaces RSX_5TEST.py)
    main.py              — ML training, validation, visualizations (Mac or Pi)
    sim_flight.py        — Replay Norway CSV through the pipeline (Mac testing)
    post_flight.py       — Generate visualizations from flight CSV after recovery
"""

import os
import sys
import csv
import time
import atexit
import signal
import queue
import argparse
import platform
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

# ==============================================================================
# Serial Telemetry (matches utility2026.py Print())
# ==============================================================================
SERIAL_OUT = None
SERIAL_PRINTS = False

try:
    import serial as _serial
    SERIAL_OUT = _serial.Serial(
        port="/dev/ttyAMA0",
        baudrate=153600,
        bytesize=_serial.EIGHTBITS,
        parity=_serial.PARITY_NONE,
        stopbits=_serial.STOPBITS_ONE,
        timeout=1,
    )
except Exception:
    pass  # No serial on Mac — that's fine

# GPIO for poweroff30 detection (only on Pi)
POWEROFF30_BUTTON = None
try:
    from gpiozero import Button as _Button
    POWEROFF30_BUTTON = _Button(16, pull_up=True)
except Exception:
    pass


_TELEMETRY_QUEUE = queue.Queue(maxsize=500)
_TELEMETRY_DROPS = 0


def _telemetry_worker():
    """Background thread: drains telemetry queue to serial port."""
    global _TELEMETRY_DROPS
    while True:
        try:
            msg = _TELEMETRY_QUEUE.get()
            if msg is None:
                break  # Shutdown sentinel
            if SERIAL_OUT is not None:
                SERIAL_OUT.write((msg + "\r\r\n").encode())
        except Exception:
            pass


_telemetry_thread = threading.Thread(target=_telemetry_worker, daemon=True)
_telemetry_thread.start()


def telemetry(msg):
    """Send message to serial port + terminal (non-blocking)."""
    global _TELEMETRY_DROPS
    if SERIAL_PRINTS:
        print(msg)
    try:
        _TELEMETRY_QUEUE.put_nowait(msg)
    except queue.Full:
        _TELEMETRY_DROPS += 1


# ==============================================================================
# Setup
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR / 'lib'))

from config import (
    IS_RASPBERRY_PI, STORAGE_PATH, ML_CONFIG,
    HAILO_HEF_PATH, SENSOR_CONFIG, MISSION_TIMELINE, MISSION_NAME
)

import joblib

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

HAILO_AVAILABLE = False
try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                                InferVStreams, ConfigureParams,
                                InputVStreamParams, OutputVStreamParams, FormatType)
    HAILO_AVAILABLE = True
except ImportError:
    pass

from sensors import MultiMagnetometerReader


# ==============================================================================
# Flight CSV Logger
# ==============================================================================
class RawSensorLogger:
    """
    SAFETY-CRITICAL: Writes raw sensor data to CSV the instant it's read,
    BEFORE it enters the ML queue.  This file is the last-resort backup.

    If ML crashes, if the queue overflows, if the consumer thread dies —
    this file still has every successful I2C read.

    Design:
    - Called directly from the sensor producer thread (no queue in between)
    - fsync after EVERY row (maximum 22ms of data at risk, never more)
    - Minimal columns (no ML results) to keep writes as fast as possible
    - Separate file from FlightLogger so one can't corrupt the other
    """

    COLUMNS = [
        'timestamp_unix',
        'sensor_id',
        'bx_raw', 'by_raw', 'bz_raw',
    ]

    def __init__(self, output_dir, prefix='flight'):
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(output_dir, f'{prefix}_raw_sensors_{ts}.csv')
        # Also create/update a symlink for easy post-flight access
        latest = os.path.join(output_dir, f'{prefix}_raw_sensors_latest.csv')
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(self.csv_path, latest)
        except OSError:
            pass
        self.file = open(self.csv_path, 'w', newline='', buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.COLUMNS)
        self.file.flush()
        os.fsync(self.file.fileno())
        self.row_count = 0
        self.write_errors = 0
        self._closed = False
        print(f"  RAW sensor log → {self.csv_path}")

    def log(self, sensor_id, bx, by, bz, timestamp):
        """Write one row and fsync immediately. Never raises."""
        try:
            self.writer.writerow([f'{timestamp:.6f}', sensor_id, bx, by, bz])
            self.row_count += 1
            self.file.flush()
            os.fsync(self.file.fileno())
        except Exception:
            self.write_errors += 1

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
            print(f"  RAW sensor log closed — {self.row_count} rows → {self.csv_path}")
            if self.write_errors:
                print(f"  RAW sensor log write errors: {self.write_errors}")
        except Exception as e:
            print(f"  RAW sensor log close error: {e}")


class FlightLogger:
    """
    Writes every sensor reading + ML result to CSV.
    Flushes to disk after every batch — a power cut never loses more than
    a handful of rows.
    """

    COLUMNS = [
        'timestamp_unix',           # Unix epoch (float)
        'mission_time_s',           # Seconds relative to launch (T+/T-)
        'sensor_id',                # Magnetometer channel (0, 1, 2)
        'bx_raw', 'by_raw', 'bz_raw',
        'magnitude_measured',       # √(bx² + by² + bz²)
        'magnitude_rf',             # Random Forest prediction
        'magnitude_nn',             # Neural Network prediction
        'cluster_id',               # K-Means cluster
        'anomaly_zscore',           # Modified Z-score value
        'anomaly_is_zscore',        # 1 if Z-score flags it
        'anomaly_is_roc',           # 1 if rate-of-change flags it
        'anomaly_is_ensemble',      # 1 if 2+ methods agree
        'anomaly_rate_of_change',   # |Δmagnitude| from previous
        'reading_number',           # Sequential counter
    ]

    def __init__(self, output_dir, prefix='flight'):
        os.makedirs(output_dir, exist_ok=True)
        # Timestamped filename prevents data loss if Pi reboots mid-flight
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(output_dir, f'{prefix}_data_{ts}.csv')
        # Also create/update a symlink to latest for easy post_flight.py access
        latest = os.path.join(output_dir, f'{prefix}_data_latest.csv')
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(self.csv_path, latest)
        except OSError:
            pass  # Windows doesn't support symlinks — that's fine
        self.file = open(self.csv_path, 'w', newline='', buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.COLUMNS)
        self.file.flush()
        os.fsync(self.file.fileno())
        self.row_count = 0
        self.write_errors = 0
        self._closed = False
        self.flush_every_row = False
        print(f"  CSV logger → {self.csv_path}")

    def log(self, row_dict):
        """Write one row. Never raises — a disk error here must not kill ML."""
        try:
            row = [row_dict.get(c, '') for c in self.COLUMNS]
            self.writer.writerow(row)
            self.row_count += 1
            if self.flush_every_row or self.row_count % 5 == 0:
                self.file.flush()
                os.fsync(self.file.fileno())
        except Exception:
            self.write_errors += 1

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
            print(f"  CSV closed — {self.row_count} rows → {self.csv_path}")
            if self.write_errors:
                print(f"  CSV write errors: {self.write_errors}")
        except Exception as e:
            print(f"  CSV close error: {e}")


# ==============================================================================
# Cached Model Loader (FROZEN — no retraining ever)
# ==============================================================================
class CachedModels:
    """Load pre-trained models from disk. These are never modified."""

    def __init__(self):
        self.rf = None
        self.nn = None
        self.nn_mean = None
        self.nn_std = None
        self.kmeans = None
        self.temporal_rf = None
        self.temporal_mae = None
        self.loaded = False

    def load(self):
        cache_dir = SCRIPT_DIR / 'models' / 'cached'
        exports = SCRIPT_DIR / 'models' / 'exports'
        ok = True

        # Random Forest (REQUIRED)
        p = cache_dir / 'rf_model.joblib'
        if p.exists():
            self.rf = joblib.load(p)
            print(f"  ✓ Random Forest  ({p.stat().st_size // 1024} KB)")
        else:
            print(f"  ✗ Random Forest NOT FOUND — run main.py first")
            ok = False

        # K-Means
        p = cache_dir / 'kmeans_model.joblib'
        if p.exists():
            d = joblib.load(p)
            self.kmeans = d['model'] if isinstance(d, dict) else d
            print(f"  ✓ K-Means")
        else:
            print(f"  ⚠ K-Means not found")

        # Temporal RF
        p = cache_dir / 'temporal_rf_model.joblib'
        if p.exists():
            d = joblib.load(p)
            self.temporal_rf = d['model']
            self.temporal_mae = d.get('mae', 15.0)
            print(f"  ✓ Temporal RF  (MAE={self.temporal_mae:.2f} nT)")
        else:
            print(f"  ⚠ Temporal RF not found")

        # Neural Network (FCN — matches Hailo HEF)
        if TENSORFLOW_AVAILABLE:
            fcn = exports / 'magnetometer_fcn.keras'
            nrm = exports / 'magnetometer_fcn_norm.npz'
            if fcn.exists() and nrm.exists():
                self.nn = load_model(fcn)
                n = np.load(nrm)
                self.nn_mean, self.nn_std = n['mean'], n['std']
                print(f"  ✓ Neural Network (FCN)")
            else:
                print(f"  ⚠ NN not found at {fcn}")
        else:
            print(f"  ⚠ TensorFlow not available — NN skipped")

        self.loaded = ok
        return ok


# ==============================================================================
# Flight Inference Engine
# ==============================================================================
class FlightEngine:
    """Processes sensor readings through the ML pipeline and logs results."""

    # Maximum time (seconds) to allow for NN CPU-fallback inference.
    # If TF exceeds this, skip the NN prediction for this reading.
    NN_TIMEOUT_S = 0.015  # 15 ms

    def __init__(self, models, logger):
        self.models = models
        self.logger = logger
        self.reading_count = 0
        self.launch_time = None
        self.start_time = time.time()
        self.running = False
        self.shutdown_requested = False
        self.anomaly_count = 0

        # Live visualization (set externally via --live flag)
        self.live_display = None

        # Anomaly state — ring buffer + cached numpy array for fast Z-score
        self.magnitude_history = deque(maxlen=500)
        self._mag_hist_arr = np.empty(500, dtype=np.float64)
        self._mag_hist_len = 0
        self.prev_magnitude = None

        # Pipeline health counters
        self.overrun_count = 0        # Readings that exceeded the 22ms budget
        self.nn_skip_count = 0        # NN predictions skipped due to timeout
        self.dropped_readings = 0     # Sensor readings dropped (queue overflow)
        self._process_times = deque(maxlen=200)  # Recent process_reading durations

        # Hailo
        self.hailo_active = False
        self._init_hailo()

    def _init_hailo(self):
        if not HAILO_AVAILABLE or not Path(HAILO_HEF_PATH).exists():
            return
        try:
            self.hailo_hef = HEF(str(HAILO_HEF_PATH))
            self.hailo_vdevice = VDevice()
            cp = ConfigureParams.create_from_hef(self.hailo_hef,
                                                  interface=HailoStreamInterface.PCIe)
            self.hailo_ng = self.hailo_vdevice.configure(self.hailo_hef, cp)[0]
            self.hailo_in  = self.hailo_hef.get_input_vstream_infos()
            self.hailo_out = self.hailo_hef.get_output_vstream_infos()
            self.hailo_ip  = InputVStreamParams.make_from_network_group(
                self.hailo_ng, quantized=False, format_type=FormatType.FLOAT32)
            self.hailo_op  = OutputVStreamParams.make_from_network_group(
                self.hailo_ng, quantized=False, format_type=FormatType.FLOAT32)
            # Activate network group once and keep alive (matches controller.py)
            self._hailo_activation_ctx = self.hailo_ng.activate()
            self._hailo_activation_ctx.__enter__()
            self.hailo_active = True
            print(f"  ✓ Hailo NPU ready  input={self.hailo_in[0].shape}  output={self.hailo_out[0].shape}")
        except Exception as e:
            print(f"  ⚠ Hailo init failed: {e} — CPU fallback")

    # ----- timing helpers -----
    def set_launch_time(self):
        self.launch_time = time.time()
        print(f"  ⏱  LAUNCH TIME SET")

    def mission_time(self):
        if self.launch_time is None:
            return -(150 - (time.time() - self.start_time))
        return time.time() - self.launch_time

    # ----- core: process one sensor reading -----
    def process_reading(self, sensor_id, bx, by, bz, timestamp=None):
        proc_start = time.monotonic()
        timestamp = timestamp or time.time()
        mt = self.mission_time()
        self.reading_count += 1

        magnitude = np.sqrt(bx**2 + by**2 + bz**2)

        # Write magneto line to serial in EXACT magnetometerMVprintable.py format
        # so the .dat file is identical to what the team's code produced.
        mt_int = round(mt)
        mt_sign = '+' if mt_int > 0 else ''
        dat_line = (f"T{mt_sign}{mt_int}: |{self.reading_count}|: "
                    f"Magneto #{sensor_id} X: {int(bx)} Y: {int(by)} Z: {int(bz)} "
                    f"( {time.asctime(time.localtime())} )")
        telemetry(dat_line)

        # Pre-allocate input array once (reuse across models)
        X_in = np.array([[bx, by, bz]])

        # RF (3 features: bx, by, bz — no time)
        rf_pred = None
        if self.models.rf is not None:
            try:
                rf_pred = float(self.models.rf.predict(X_in)[0])
            except Exception:
                pass

        # NN (Hailo first, TF fallback with timeout guard)
        nn_pred = None
        if self.models.nn is not None and self.models.nn_mean is not None:
            try:
                X_n = (np.array([bx, by, bz]) - self.models.nn_mean) / self.models.nn_std
                if self.hailo_active:
                    nn_pred = self._hailo_predict(X_n)
                if nn_pred is None:
                    # TF CPU fallback — enforce a time budget so we don't stall
                    tf_start = time.monotonic()
                    X_tf = X_n.reshape(1, -1, 1).astype(np.float32)
                    nn_pred = float(self.models.nn.predict(X_tf, verbose=0).flatten()[0])
                    tf_elapsed = time.monotonic() - tf_start
                    if tf_elapsed > self.NN_TIMEOUT_S:
                        # Log the slow inference but keep the result this time.
                        # If it keeps happening, the status line will show nn_skip_count.
                        self.nn_skip_count += 1
            except Exception as e:
                self.ml_errors = getattr(self, 'ml_errors', 0) + 1
                if self.ml_errors <= 5:
                    print(f"  ⚠ NN error #{self.ml_errors}: {e}")

        # K-Means
        cluster = None
        if self.models.kmeans is not None:
            try:
                cluster = int(self.models.kmeans.predict(X_in)[0])
            except Exception:
                pass

        # Anomaly: Modified Z-Score — optimized with cached numpy array
        z_score = 0.0
        is_z = False
        self.magnitude_history.append(magnitude)
        n = len(self.magnitude_history)
        if n >= 20:
            # Update cached array from deque (only copy what we need)
            if n != self._mag_hist_len:
                self._mag_hist_len = n
            for i, v in enumerate(self.magnitude_history):
                self._mag_hist_arr[i] = v
            h = self._mag_hist_arr[:n]
            med = np.median(h)
            mad = np.median(np.abs(h - med))
            if mad > 1e-9:
                z_score = 0.6745 * abs(magnitude - med) / mad
            is_z = z_score > ML_CONFIG.get('ANOMALY_Z_THRESHOLD', 3.5)

        # Anomaly: Rate of Change
        roc = abs(magnitude - self.prev_magnitude) if self.prev_magnitude is not None else 0.0
        self.prev_magnitude = magnitude
        is_roc = roc > ML_CONFIG.get('RATE_OF_CHANGE_THRESHOLD', 200.0)

        # Ensemble (2+ methods)
        is_ens = (int(is_z) + int(is_roc)) >= 2
        if is_ens:
            self.anomaly_count += 1

        # ML telemetry line (team .dat format)
        rf_str = f'RF={rf_pred:.1f}' if rf_pred is not None else 'RF=---'
        nn_str = f'NN={nn_pred:.1f}' if nn_pred is not None else 'NN=---'
        cl_str = f'C={cluster}' if cluster is not None else 'C=-'
        anom_tag = f' ANOMALY#{self.anomaly_count}' if is_ens else ''
        ml_line = (f"T{mt_sign}{mt_int}: |{self.reading_count}|: "
                   f"ML{anom_tag} |B|={magnitude:.1f} {rf_str} {nn_str} {cl_str} "
                   f"Z={z_score:.2f} RoC={roc:.1f} "
                   f"( {time.asctime(time.localtime())} )")
        telemetry(ml_line)

        # Log
        self.logger.log({
            'timestamp_unix':       f'{timestamp:.6f}',
            'mission_time_s':       f'{mt:.4f}',
            'sensor_id':            sensor_id,
            'bx_raw': bx, 'by_raw': by, 'bz_raw': bz,
            'magnitude_measured':   f'{magnitude:.2f}',
            'magnitude_rf':         f'{rf_pred:.2f}' if rf_pred is not None else '',
            'magnitude_nn':         f'{nn_pred:.2f}' if nn_pred is not None else '',
            'cluster_id':           cluster if cluster is not None else '',
            'anomaly_zscore':       f'{z_score:.4f}',
            'anomaly_is_zscore':    int(is_z),
            'anomaly_is_roc':       int(is_roc),
            'anomaly_is_ensemble':  int(is_ens),
            'anomaly_rate_of_change': f'{roc:.2f}',
            'reading_number':       self.reading_count,
        })

        # Feed live visualization (non-blocking)
        if self.live_display is not None:
            self.live_display.update(
                mission_time=mt, magnitude=magnitude,
                rf_pred=rf_pred, nn_pred=nn_pred,
                z_score=z_score, roc=roc,
                is_ensemble=is_ens, cluster_id=cluster,
                sensor_id=sensor_id, reading_number=self.reading_count,
            )

        # Track processing time
        proc_elapsed = time.monotonic() - proc_start
        self._process_times.append(proc_elapsed)
        period = SENSOR_CONFIG['SAMPLE_PERIOD_MS'] / 1000.0
        if proc_elapsed > period:
            self.overrun_count += 1

    def _hailo_predict(self, X_norm):
        if not self.hailo_active:
            return None
        try:
            shape = self.hailo_in[0].shape
            X_in = X_norm.astype(np.float32).reshape(1, *shape)
            name_in = self.hailo_in[0].name
            with InferVStreams(self.hailo_ng, self.hailo_ip, self.hailo_op) as p:
                res = p.infer({name_in: X_in})
            return float(res[self.hailo_out[0].name].flatten()[0])
        except Exception as e:
            print(f"  ⚠ Hailo predict error: {e}")
            return None

    def shutdown(self):
        self.running = False
        print(f"\n{'='*60}")
        print(f"  SHUTDOWN at T{self.mission_time():+.1f}s")
        print(f"  Readings processed: {self.reading_count}")
        print(f"  Anomalies detected: {self.anomaly_count}")
        # Pipeline health report
        if self._process_times:
            avg_ms = 1000 * np.mean(list(self._process_times))
            max_ms = 1000 * max(self._process_times)
            print(f"  Avg process time:   {avg_ms:.1f} ms  (max {max_ms:.1f} ms)")
        if self.overrun_count:
            print(f"  Budget overruns:    {self.overrun_count} "
                  f"({100*self.overrun_count/max(self.reading_count,1):.1f}%)")
        if self.dropped_readings:
            print(f"  Dropped readings:   {self.dropped_readings}")
        if self.nn_skip_count:
            print(f"  NN slow inferences: {self.nn_skip_count}")
        if _TELEMETRY_DROPS:
            print(f"  Telemetry drops:    {_TELEMETRY_DROPS}")
        ml_errors = getattr(self, 'ml_errors', 0)
        if ml_errors:
            print(f"  ML processing errors: {ml_errors}")
        if self.logger.write_errors:
            print(f"  CSV write errors:   {self.logger.write_errors}")
        # Raw sensor backup stats
        raw = getattr(self, 'raw_logger', None)
        if raw is not None:
            print(f"  Raw sensor rows:    {raw.row_count}")
            if raw.write_errors:
                print(f"  Raw write errors:   {raw.write_errors}")
        self.logger.close()
        if self.live_display is not None:
            self.live_display.stop()
        # Shut down telemetry writer
        try:
            _TELEMETRY_QUEUE.put_nowait(None)
        except queue.Full:
            pass
        # Release Hailo resources
        if hasattr(self, '_hailo_activation_ctx'):
            try:
                self._hailo_activation_ctx.__exit__(None, None, None)
            except Exception:
                pass
        print(f"{'='*60}")


# ==============================================================================
# Bench test (no sensors — synthetic data through FlightEngine)
# ==============================================================================
def run_bench(engine, duration=5.0):
    """
    Feed synthetic magnetometer data through FlightEngine.process_reading().
    Tests: ML inference, Z-score, timing instrumentation, telemetry queue,
    and the shutdown health report — all without I2C hardware.

    Usage:  python test_main.py --bench --bench-duration 10
    """
    print(f"  BENCH MODE — {duration}s of synthetic data at 45 Hz")
    print(f"  No sensors needed. Testing FlightEngine pipeline.\n")

    engine.running = True
    engine.set_launch_time()

    period = SENSOR_CONFIG['SAMPLE_PERIOD_MS'] / 1000.0
    t0 = time.time()
    reading = 0

    # Simulate 3 magnetometers with slightly different baselines
    baselines = {0: (20000, -5000, 45000),
                 1: (19500, -4800, 44500),
                 2: (20200, -5200, 45500)}
    rng = np.random.default_rng(42)

    try:
        while time.time() - t0 < duration and not engine.shutdown_requested:
            loop_t = time.time()

            for ch in [0, 1, 2]:
                bx_base, by_base, bz_base = baselines[ch]
                # Add realistic noise + occasional spike
                noise = rng.normal(0, 50, 3)
                spike = 0.0
                if rng.random() < 0.005:  # 0.5% chance of anomaly
                    spike = rng.choice([-1, 1]) * rng.uniform(500, 2000)

                bx = bx_base + noise[0] + spike
                by = by_base + noise[1]
                bz = bz_base + noise[2]

                engine.process_reading(ch, bx, by, bz, time.time())
                reading += 1

            # Pace to 45 Hz
            dt = time.time() - loop_t
            if dt < period:
                time.sleep(period - dt)

            # Status every ~5 s
            if reading % (45 * 5) < 3:
                el = time.time() - t0
                hz = reading / max(el, 0.001)
                print(f"  T{engine.mission_time():+.1f}s | "
                      f"{reading} readings | {hz:.1f} Hz")

    except KeyboardInterrupt:
        print("\n  Stopped by user.")

    engine.shutdown()


# ==============================================================================
# Live sensor loop (producer-consumer architecture)
# ==============================================================================
# The sensor reading thread (producer) pushes readings into a bounded queue.
# The main thread (consumer) pulls from the queue and runs ML inference.
# This decouples the sampling rate from the processing rate:
#   - If ML inference is slow, the producer keeps sampling at 45 Hz
#   - If the queue fills up, the oldest un-processed readings are dropped
#     (logged via engine.dropped_readings counter)
#
# Queue capacity: 3 sensors * 10 cycles = 30 readings (~220 ms of data).
# This gives the consumer enough headroom to absorb occasional ML spikes
# without losing data, while bounding memory usage.
# ==============================================================================

_SENSOR_QUEUE_SIZE = 30  # ~220 ms of buffered readings at 45 Hz * 3 sensors


def _sensor_producer(mag, active, engine, sensor_queue, period, raw_logger):
    """
    Producer thread: reads sensors at a steady 45 Hz and pushes readings
    into a bounded queue.  If the queue is full (consumer can't keep up),
    the OLDEST reading is evicted so the newest data is always available.

    SAFETY: Every successful I2C read is written to raw_logger IMMEDIATELY
    (fsync'd every row) BEFORE entering the ML queue.  If anything downstream
    crashes, the raw data is already safe on disk.

    SAFETY: The entire loop body is wrapped in try/except so a single bad
    read, a transient I2C glitch, or any unexpected error cannot kill this
    thread.  The thread must survive for the entire flight.
    """
    poweroff_detected = False
    consecutive_errors = 0

    while engine.running and not engine.shutdown_requested:
        loop_t = time.time()

        try:
            for ch in active:
                try:
                    r = mag.read_magnetometer(ch)
                except Exception:
                    # I2C bus error — don't let it kill the thread
                    r = {'error': 'I2C exception'}

                if 'error' not in r:
                    consecutive_errors = 0
                    bx, by, bz, ts = r['x'], r['y'], r['z'], r['timestamp']

                    # ── SAFETY-CRITICAL: save raw data BEFORE it touches ML ──
                    raw_logger.log(ch, bx, by, bz, ts)

                    # Push to ML queue (evict oldest if full)
                    item = (ch, bx, by, bz, ts)
                    while True:
                        try:
                            sensor_queue.put_nowait(item)
                            break
                        except queue.Full:
                            try:
                                sensor_queue.get_nowait()  # Evict oldest
                                engine.dropped_readings += 1
                            except queue.Empty:
                                break
                else:
                    consecutive_errors += 1
                    # Write error line in EXACT team format for .dat compatibility
                    try:
                        mt = engine.mission_time()
                        mt_int = round(mt)
                        mt_sign = '+' if mt_int > 0 else ''
                        engine.reading_count += 1
                        err_line = (f"T{mt_sign}{mt_int}: |{engine.reading_count}|: "
                                    f"Magneto #{ch} Excpetion occured: "
                                    f"Unable to Read Measurement "
                                    f"( {time.asctime(time.localtime())} )")
                        telemetry(err_line)
                    except Exception:
                        pass  # Telemetry failure must not kill sensor thread

            # Check poweroff30 GPIO flag (30s warning before power cut)
            if POWEROFF30_BUTTON is not None and not poweroff_detected:
                try:
                    count = sum(1 for _ in range(20) if POWEROFF30_BUTTON.is_pressed)
                    if count >= 15:
                        poweroff_detected = True
                        engine.logger.flush_every_row = True
                        telemetry(f"  POWEROFF30 FLAG — 30s until power cut — flushing every row")
                except Exception:
                    pass  # GPIO failure must not kill sensor thread

        except Exception as e:
            # Catch-all: NOTHING can kill this thread during flight.
            # Log it, keep going.
            consecutive_errors += 1
            try:
                telemetry(f"  SENSOR THREAD ERROR: {e}")
            except Exception:
                pass

        # If we're getting many consecutive errors, back off slightly
        # to avoid hammering a broken I2C bus
        if consecutive_errors > 50:
            time.sleep(0.1)

        # Pace to 45 Hz
        dt = time.time() - loop_t
        if dt < period:
            time.sleep(period - dt)


def run_live(engine, channels, mode='flight'):
    mag = MultiMagnetometerReader(channels=channels)

    working = mag.verify_sensors()
    active = [ch for ch, ok in working.items() if ok]
    if not active:
        print("  No sensors detected — check I2C wiring and MUX (0x70)")
        return False

    print(f"  Sensors online: {active}")
    print(f"  Target rate: {SENSOR_CONFIG['SAMPLING_RATE_HZ']} Hz")
    print(f"  Pipeline:    producer-consumer (queue={_SENSOR_QUEUE_SIZE})")
    print(f"  Press Ctrl+C to stop\n")

    engine.running = True
    engine.set_launch_time()

    period = SENSOR_CONFIG['SAMPLE_PERIOD_MS'] / 1000.0   # 22.2 ms
    t0 = time.time()

    # Bounded queue between sensor producer and ML consumer
    sensor_queue = queue.Queue(maxsize=_SENSOR_QUEUE_SIZE)

    # Raw sensor backup logger (written in producer thread, before ML)
    raw_logger = RawSensorLogger(
        os.path.join(STORAGE_PATH, 'flight_data'),
        prefix=mode,
    )
    engine.raw_logger = raw_logger  # So shutdown() can report its stats
    atexit.register(raw_logger.close)

    # Start producer thread
    producer = threading.Thread(
        target=_sensor_producer,
        args=(mag, active, engine, sensor_queue, period, raw_logger),
        daemon=True,
    )
    producer.start()

    # Track ML processing errors (ML crashes that were caught and survived)
    ml_errors = 0

    try:
        while engine.running and not engine.shutdown_requested:
            # SAFETY: check if producer thread died unexpectedly
            if not producer.is_alive():
                telemetry("  CRITICAL: Sensor producer thread died — restarting")
                print("  CRITICAL: Sensor producer thread died — attempting restart")
                producer = threading.Thread(
                    target=_sensor_producer,
                    args=(mag, active, engine, sensor_queue, period, raw_logger),
                    daemon=True,
                )
                producer.start()

            # Block-wait for the next reading (up to 100 ms timeout to check flags)
            try:
                ch, x, y, z, ts = sensor_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # SAFETY: ML crash must not kill the consumer loop.
            # Raw data is already on disk (written by producer).
            # If process_reading fails, we lose the ML results for this
            # reading but the consumer keeps running for future readings.
            try:
                engine.process_reading(ch, x, y, z, ts)
            except Exception as e:
                ml_errors += 1
                if ml_errors <= 5:
                    telemetry(f"  ML ERROR #{ml_errors}: {e}")
                elif ml_errors == 6:
                    telemetry(f"  ML ERROR: suppressing further error messages")

            # Drain any backlog: process all queued readings before sleeping
            while not sensor_queue.empty():
                try:
                    ch, x, y, z, ts = sensor_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    engine.process_reading(ch, x, y, z, ts)
                except Exception:
                    ml_errors += 1

            # Status every ~5 s
            if engine.reading_count % (45 * 5) < len(active):
                el = time.time() - t0
                hz = engine.reading_count / max(el, 0.001)
                drops = engine.dropped_readings
                overruns = engine.overrun_count
                msg = (f"  T{engine.mission_time():+.1f}s | "
                       f"{engine.reading_count} readings | {hz:.1f} Hz"
                       f"{f' | {drops} dropped' if drops else ''}"
                       f"{f' | {overruns} overruns' if overruns else ''}"
                       f"{f' | {ml_errors} ML errors' if ml_errors else ''}")
                print(msg)
                telemetry(f"T{engine.mission_time():+.1f}s ML: "
                          f"{engine.reading_count} readings, {hz:.1f} Hz"
                          f"{f', {drops} dropped' if drops else ''}")

            # Auto-shutdown at experiments-off time
            t_off = MISSION_TIMELINE.get('T_EXPERIMENTS_OFF', 336)
            if engine.launch_time and engine.mission_time() > t_off:
                telemetry(f"  T+{t_off}s — EXPERIMENTS OFF — shutting down ML pipeline")
                break

    except KeyboardInterrupt:
        print("\n  Stopped by user.")

    # Stop producer thread
    engine.running = False
    producer.join(timeout=2)

    mag.close()
    engine.ml_errors = ml_errors
    engine.shutdown()
    raw_logger.close()
    return True


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='RockSat-X 2026 ML Magnetometer Pipeline (Raspberry Pi 5)',
        epilog='Run main.py for ML training/validation, sim_flight.py for simulation.')
    parser.add_argument('--test', action='store_true',
                        help='Bench-test mode (more terminal output)')
    parser.add_argument('--channels', type=int, nargs='+', default=[0, 1, 2],
                        help='MUX channels (default: 0 1 2)')
    parser.add_argument('--st', type=float, default=None,
                        help='Shared start time from flight_controller.py (perf_counter)')
    parser.add_argument('--prints', action='store_true',
                        help='Enable serial telemetry + terminal output')
    parser.add_argument('--live', action='store_true',
                        help='Open real-time anomaly detection visualization window')
    parser.add_argument('--bench', action='store_true',
                        help='Bench-test mode: feed synthetic data through FlightEngine '
                             '(no sensors needed, works on Mac)')
    parser.add_argument('--bench-duration', type=float, default=5.0,
                        help='Duration in seconds for --bench mode (default: 5)')
    args = parser.parse_args()

    # Enable serial telemetry if requested
    global SERIAL_PRINTS
    SERIAL_PRINTS = args.prints or args.test

    mode = 'test' if args.test else 'flight'

    print("=" * 60)
    print(f"  {MISSION_NAME} — RockSat-X Flight Software")
    print("=" * 60)
    print(f"  Mode:       {mode.upper()}")
    print(f"  Platform:   {'Raspberry Pi 5' if IS_RASPBERRY_PI else platform.system()}")
    print(f"  TensorFlow: {'Yes' if TENSORFLOW_AVAILABLE else 'No'}")
    print(f"  Hailo NPU:  {'Yes' if HAILO_AVAILABLE else 'No (CPU fallback)'}")
    print(f"  Channels:   {args.channels}")
    print(f"  Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not IS_RASPBERRY_PI:
        print("\n  ⚠  Not running on Raspberry Pi.")
        print("  Use sim_flight.py to test on Mac, or --test for bench testing.\n")

    # Load models
    print("\nLoading cached models (frozen)...")
    models = CachedModels()
    if not models.load():
        print("\n  ⛔ Required models missing. Run main.py first.")
        sys.exit(1)

    # Logger
    out_dir = os.path.join(STORAGE_PATH, 'flight_data')
    logger = FlightLogger(out_dir, prefix=mode)

    # Engine
    engine = FlightEngine(models, logger)

    # Live anomaly visualization (separate window, non-blocking)
    if args.live:
        try:
            from live_display import LiveAnomalyDisplay
            engine.live_display = LiveAnomalyDisplay(
                z_thresh=ML_CONFIG.get('ANOMALY_Z_THRESHOLD', 3.5),
                roc_thresh=ML_CONFIG.get('RATE_OF_CHANGE_THRESHOLD', 200.0),
            )
            engine.live_display.start()
            print("  ✓ Live visualization window opened")
        except Exception as e:
            print(f"  ⚠ Live visualization failed: {e}")
            print(f"    (Requires matplotlib + tkinter: pip install matplotlib)")

    # If launched by flight_controller.py with --st, use shared start time
    if args.st is not None:
        # Compute how far along the timeline we are using the shared clock
        PRE_FLIGHT = 170
        PI_BOOT = 24
        elapsed_since_start = time.perf_counter() - args.st
        mission_offset = elapsed_since_start - PRE_FLIGHT + PI_BOOT
        engine.start_time = time.time() - elapsed_since_start
        telemetry(f"  Synced with flight_controller.py (mission T{mission_offset:+.0f}s)")

    # Graceful shutdown
    def handle_signal(sig, frame):
        print(f"\n  Signal {sig} received")
        engine.shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    atexit.register(logger.close)

    # Run
    print()
    if args.bench:
        run_bench(engine, duration=args.bench_duration)
    else:
        run_live(engine, channels=args.channels, mode=mode)
    print("\nDone.")


if __name__ == '__main__':
    main()
