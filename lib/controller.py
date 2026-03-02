"""
RockSat-X 2026 - AI Controller Module
Real-time sensor monitoring and ML inference controller.
Manages data collection, predictions, and anomaly detection during flight.
"""

import os
import time
import json
import glob
import pickle
import numpy as np
import pandas as pd
import joblib
from threading import Thread, Lock
from collections import deque

from config import STORAGE_PATH, HAILO_HEF_PATH, ML_CONFIG
from anomaly import real_time_anomaly_check

# Conditional imports
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    load_model = None
    TENSORFLOW_AVAILABLE = False

# Hailo NPU imports
HAILO_AVAILABLE = False
try:
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
    from hailo_platform import InputVStreamParams, OutputVStreamParams, FormatType
    HAILO_AVAILABLE = True
except ImportError:
    pass


class AIController:
    """
    Real-time AI controller for magnetometer data processing.

    Manages:
    - Sensor data collection at 45 Hz
    - ML model inference (RF, NN, GPR, K-Means)
    - Anomaly detection
    - Hailo NPU acceleration (when available)
    - Model persistence
    """

    def __init__(self, sensors, nn_model, rf_model, cluster_model,
                 gpr_model=None, anomaly_detectors=None):
        """
        Initialize the AI Controller.

        Args:
            sensors: MultiMagnetometerReader instance
            nn_model: Trained neural network model
            rf_model: Trained Random Forest model
            cluster_model: Trained K-Means model
            gpr_model: Optional Gaussian Process model
            anomaly_detectors: Dict of anomaly detection models
        """
        self.sensors = sensors
        self.model = {
            'nn': nn_model,
            'rf': rf_model,
            'cluster': cluster_model,
            'gpr': gpr_model
        }
        self.nn_mean = None
        self.nn_std = None
        self.anomaly_detectors = anomaly_detectors or {
            'isolation_forest': None,
            'local_outlier_factor': None
        }

        # Configuration
        self.anomaly_threshold = ML_CONFIG['ANOMALY_Z_THRESHOLD']
        self.buffer_size = max(100, 25 * len(sensors.channels))
        self.update_interval = max(0.005, 0.001 * len(sensors.channels))

        # Data buffers
        self.channel_buffer = {ch: deque(maxlen=self.buffer_size) for ch in sensors.channels}
        self.decision_log = deque(maxlen=100)
        self.xyz_buffer = deque(maxlen=200)
        self.raw_time_buffer = deque(maxlen=200)
        self.cluster_buffer = deque(maxlen=200)
        self.data_buffer = deque(maxlen=200)

        # Prediction buffers
        self.time_data = deque(maxlen=200)
        self.actual_data = deque(maxlen=200)
        self.pred_rf_data = deque(maxlen=200)
        self.pred_nn_data = deque(maxlen=200)
        self.pred_gpr_data = deque(maxlen=200)
        self.pred_gpr_std = deque(maxlen=200)
        self.pred_cluster_data = deque(maxlen=200)
        self.consensus_data = deque(maxlen=200)
        self.realtime_z_scores = deque(maxlen=200)
        self.Anomaly_Points = deque(maxlen=10)

        # State
        self.running = False
        self.lock = Lock()
        self.start_time = time.time()
        self.cluster_centers = None
        self._retrain_count = 0
        self._anomaly_detector_retrain_interval = 100

        # Storage
        self.storage_path = STORAGE_PATH
        os.makedirs(self.storage_path, exist_ok=True)

        # Hailo NPU configuration
        self.use_hailo = False
        self.hailo_hef = None
        self.hailo_vdevice = None
        self.hailo_network_group = None
        self._hailo_inference_times = deque(maxlen=100)
        self._cpu_inference_times = deque(maxlen=100)

        # Initialize Hailo if available
        if HAILO_AVAILABLE and os.path.exists(HAILO_HEF_PATH):
            try:
                self._init_hailo(HAILO_HEF_PATH)
            except Exception as e:
                print(f"WARNING: Hailo initialization failed: {e}")
                print("         Falling back to CPU inference.")
                self.use_hailo = False

        print(f'AIController initialized for {len(sensors.channels)} sensors')

    # =========================================================================
    # HAILO NPU METHODS
    # =========================================================================

    def _init_hailo(self, hef_path):
        """Initialize Hailo NPU for neural network inference."""
        print(f"Initializing Hailo NPU with model: {hef_path}")

        self.hailo_hef = HEF(hef_path)
        self.hailo_vdevice = VDevice()

        configure_params = ConfigureParams.create_from_hef(
            self.hailo_hef, interface=HailoStreamInterface.PCIe)
        self.hailo_network_group = self.hailo_vdevice.configure(
            self.hailo_hef, configure_params)[0]

        # Activate the network group (must stay active for inference)
        self._hailo_activation_ctx = self.hailo_network_group.activate()
        self._hailo_activation_ctx.__enter__()

        self.hailo_input_vstreams_info = self.hailo_hef.get_input_vstream_infos()
        self.hailo_output_vstreams_info = self.hailo_hef.get_output_vstream_infos()

        self.hailo_input_params = InputVStreamParams.make_from_network_group(
            self.hailo_network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.hailo_output_params = OutputVStreamParams.make_from_network_group(
            self.hailo_network_group, quantized=False, format_type=FormatType.FLOAT32)

        self.use_hailo = True
        print(f"Hailo NPU initialized successfully")
        print(f"  Input shape: {self.hailo_input_vstreams_info[0].shape}")
        print(f"  Output shape: {self.hailo_output_vstreams_info[0].shape}")

    def _hailo_predict(self, X_input):
        """Run inference on Hailo NPU (network group already activated)."""
        if not self.use_hailo:
            return None

        try:
            start_time = time.time()

            input_data = X_input.astype(np.float32)
            # Reshape to match HEF expected input shape
            hef_shape = self.hailo_input_vstreams_info[0].shape
            batch = input_data.shape[0] if input_data.ndim > 0 else 1
            input_data = input_data.reshape(batch, *hef_shape)

            input_name = self.hailo_input_vstreams_info[0].name
            input_dict = {input_name: input_data}

            with InferVStreams(self.hailo_network_group,
                              self.hailo_input_params,
                              self.hailo_output_params) as pipeline:
                results = pipeline.infer(input_dict)

            output_name = self.hailo_output_vstreams_info[0].name
            pred = results[output_name].flatten()[0]

            inference_time = time.time() - start_time
            self._hailo_inference_times.append(inference_time)

            return float(pred)

        except Exception as e:
            print(f"Hailo inference error: {e}")
            return None

    def get_inference_stats(self):
        """Return benchmarking stats for Hailo vs CPU inference."""
        stats = {
            'hailo_enabled': self.use_hailo,
            'hailo_avg_time_ms': 0,
            'cpu_avg_time_ms': 0,
            'speedup': 0
        }

        if self._hailo_inference_times:
            stats['hailo_avg_time_ms'] = np.mean(list(self._hailo_inference_times)) * 1000
        if self._cpu_inference_times:
            stats['cpu_avg_time_ms'] = np.mean(list(self._cpu_inference_times)) * 1000
        if stats['cpu_avg_time_ms'] > 0 and stats['hailo_avg_time_ms'] > 0:
            stats['speedup'] = stats['cpu_avg_time_ms'] / stats['hailo_avg_time_ms']

        return stats

    # =========================================================================
    # DATA PROCESSING
    # =========================================================================

    def preprocess_data(self, df):
        """Convert raw data to ML features."""
        if isinstance(df, deque):
            data = list(df)
            df = pd.DataFrame(data)

        xyz = df[['x', 'y', 'z']].values
        scaling_factor = 20  # Adjust as needed
        xyz = xyz * scaling_factor

        if 'timestamp' in df.columns:
            timestamps = df['timestamp'].values
        else:
            timestamps = df['Time'].astype(float).values

        # Normalize timestamps
        timestamps = (timestamps - timestamps.min()) / (np.ptp(timestamps) + 1e-8)

        X = xyz  # 3 features [Bx, By, Bz] — no time (prevents temporal leakage)
        y = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)

        return X, y

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def start(self):
        """Start the real-time monitoring threads."""
        self.running = True
        self.data_thread = Thread(target=self._ai_loop, daemon=True)
        self.data_thread.start()
        print("AI monitoring started")

    def stop(self):
        """Stop monitoring and save models."""
        self.running = False
        if hasattr(self, 'data_thread'):
            self.data_thread.join(timeout=1)
        self.data_buffer.clear()
        print("AI monitoring stopped")

    def _ai_loop(self):
        """Main AI processing loop."""
        while self.running:
            try:
                # Read from all channels
                for channel in self.sensors.channels:
                    reading = self.sensors.read_magnetometer(channel)

                    if 'error' not in reading:
                        with self.lock:
                            self.channel_buffer[channel].append(reading)
                            self.data_buffer.append(reading)

                            x, y, z = reading['x'], reading['y'], reading['z']
                            self.xyz_buffer.append((x, y, z))
                            self.raw_time_buffer.append(reading['timestamp'])

                            # Calculate magnitude
                            magnitude = np.sqrt(x**2 + y**2 + z**2)
                            self.actual_data.append(magnitude)
                            self.time_data.append(reading['timestamp'])

                            # Run predictions if enough data
                            if len(self.data_buffer) >= 10:
                                self._run_predictions(reading)

                time.sleep(self.update_interval)

            except Exception as e:
                print(f"AI loop error: {e}")
                time.sleep(0.1)

    def _run_predictions(self, reading):
        """Run ML predictions on current reading."""
        x, y, z = reading['x'], reading['y'], reading['z']
        t = reading['timestamp'] - self.start_time
        magnitude = np.sqrt(x**2 + y**2 + z**2)

        X_rf = np.array([[x * 20, y * 20, z * 20]])  # RF: 3 features [Bx, By, Bz]
        X_nn_sample = np.array([[x * 20, y * 20, z * 20]])  # NN: 3 features (no time)

        # Random Forest prediction
        try:
            start = time.time()
            rf_pred = self.model['rf'].predict(X_rf)[0]
            self._cpu_inference_times.append(time.time() - start)
            self.pred_rf_data.append(rf_pred)
        except Exception:
            self.pred_rf_data.append(magnitude)

        # Neural Network prediction (Hailo or CPU)
        try:
            if self.use_hailo:
                X_nn = X_nn_sample.copy()
                if self.nn_mean is not None:
                    X_nn = (X_nn - self.nn_mean) / self.nn_std
                X_nn = X_nn.reshape((1, 3, 1))
                nn_pred = self._hailo_predict(X_nn)
                if nn_pred is None:
                    nn_pred = magnitude
            elif TENSORFLOW_AVAILABLE and self.model['nn'] is not None:
                X_nn = X_nn_sample.copy()
                if self.nn_mean is not None:
                    X_nn = (X_nn - self.nn_mean) / self.nn_std
                X_nn = X_nn.reshape((1, 3, 1))
                nn_pred = self.model['nn'].predict(X_nn, verbose=0)[0][0]
            else:
                nn_pred = magnitude
            self.pred_nn_data.append(nn_pred)
        except Exception:
            self.pred_nn_data.append(magnitude)

        # Cluster prediction
        try:
            cluster_label = self.model['cluster'].predict(X_rf)[0]
            self.pred_cluster_data.append(cluster_label)
            self.cluster_buffer.append(cluster_label)
        except Exception:
            self.pred_cluster_data.append(0)
            self.cluster_buffer.append(0)

        # Consensus (average of predictions)
        predictions = [self.pred_rf_data[-1], self.pred_nn_data[-1]]
        consensus = np.mean(predictions)
        self.consensus_data.append(consensus)

        # Anomaly check
        anomaly_result = real_time_anomaly_check(
            reading, self.anomaly_detectors, self.data_buffer)

        if anomaly_result['is_anomaly']:
            self.Anomaly_Points.append({
                'timestamp': reading['timestamp'],
                'actual_mag': magnitude,
                'methods': anomaly_result['methods'],
                'scores': anomaly_result['scores']
            })

        # Periodic retraining
        if len(self.data_buffer) % self._anomaly_detector_retrain_interval == 0:
            self._retrain_models()

    def _retrain_models(self):
        """Periodically retrain models on accumulated data."""
        try:
            df = pd.DataFrame(self.data_buffer)
            X, y = self.preprocess_data(df)

            # Retrain RF
            self.model['rf'].n_estimators = min(1000, self.model['rf'].n_estimators + 10)
            self.model['rf'].fit(X, y)

            # Update clusters
            self._update_cluster_centers()

            self._retrain_count += 1
            if self._retrain_count % 4 == 0:
                self.save_models()

        except Exception as e:
            print(f"Retraining error: {e}")

    def _update_cluster_centers(self):
        """Update K-Means cluster centers."""
        if len(self.xyz_buffer) < 10:
            return

        try:
            X = np.array(self.xyz_buffer)
            n_clusters = min(3, len(X) // 10)
            n_clusters = max(1, n_clusters)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            new_labels = kmeans.fit_predict(X[:, :3] if X.shape[1] > 3 else X)

            self.cluster_buffer = deque(new_labels.tolist(), maxlen=200)
            self.cluster_centers = kmeans.cluster_centers_

        except Exception as e:
            print(f"Cluster update error: {e}")

    # =========================================================================
    # MODEL PERSISTENCE
    # =========================================================================

    def save_models(self):
        """Save all models to disk."""
        try:
            timestamp = int(time.time())

            # Save RF
            rf_path = os.path.join(self.storage_path, 'rf', f'model_{timestamp}.joblib')
            joblib.dump(self.model['rf'], rf_path)

            # Save NN (if available)
            if TENSORFLOW_AVAILABLE and self.model['nn'] is not None:
                nn_path = os.path.join(self.storage_path, 'nn', f'model_{timestamp}.h5')
                self.model['nn'].save(nn_path)

            # Save cluster state
            cluster_state = {
                'cluster': self.model['cluster'],
                'cluster_centers': list(self.cluster_centers) if self.cluster_centers is not None else [],
                'anomaly_threshold': self.anomaly_threshold
            }
            cluster_path = os.path.join(self.storage_path, 'cluster', f'state_{timestamp}.pkl')
            with open(cluster_path, 'wb') as f:
                pickle.dump(cluster_state, f)

            # Save config
            config = {
                'retrain_count': self._retrain_count,
                'anomaly_threshold': self.anomaly_threshold,
                'last_save': timestamp
            }
            config_path = os.path.join(self.storage_path, 'config', 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            self._cleanup_old_models()
            print("Models saved successfully")

        except Exception as e:
            print(f"Model save error: {e}")

    def load_models(self):
        """Load previously saved models."""
        try:
            # Load RF
            rf_files = glob.glob(os.path.join(self.storage_path, 'rf', '*.joblib'))
            if rf_files:
                latest_rf = max(rf_files, key=os.path.getmtime)
                self.model['rf'] = joblib.load(latest_rf)
                print("Loaded Random Forest model")

            # Load NN
            if TENSORFLOW_AVAILABLE:
                nn_files = glob.glob(os.path.join(self.storage_path, 'nn', '*.h5'))
                if nn_files:
                    latest_nn = max(nn_files, key=os.path.getmtime)
                    self.model['nn'] = load_model(latest_nn)
                    print("Loaded Neural Network model")

            # Load cluster state
            cluster_files = glob.glob(os.path.join(self.storage_path, 'cluster', '*.pkl'))
            if cluster_files:
                latest_cluster = max(cluster_files, key=os.path.getmtime)
                with open(latest_cluster, 'rb') as f:
                    cluster_state = pickle.load(f)
                    self.cluster_centers = cluster_state.get('cluster_centers')
                    self.anomaly_threshold = cluster_state.get('anomaly_threshold', 3.0)
                print("Loaded cluster state")

            # Load config
            config_path = os.path.join(self.storage_path, 'config', 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self._retrain_count = config.get('retrain_count', 0)

            print("Previous learning state restored")
            return True

        except Exception as e:
            print(f"Model load error: {e}")
            return False

    def _cleanup_old_models(self, keep_last=10):
        """Remove old model files."""
        try:
            for model_type in ['rf', 'nn', 'cluster']:
                files = glob.glob(os.path.join(self.storage_path, model_type, '*'))
                files.sort(key=os.path.getmtime, reverse=True)
                for old_file in files[keep_last:]:
                    os.remove(old_file)
        except Exception as e:
            print(f"Cleanup error: {e}")
