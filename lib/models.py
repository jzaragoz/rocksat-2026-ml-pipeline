"""
RockSat-X 2026 - ML Models Module
Random Forest, Neural Network, K-Means, and Gaussian Process models.

Models are cached to disk after first training so runs
load instantly without retraining.
"""

import time
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

import shutil
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from config import ML_CONFIG, MISSION_TIMELINE, HAILO_HEF_PATH

# ==============================================================================
# MODEL CACHE DIRECTORY
# ==============================================================================

_CACHE_DIR = Path(__file__).parent.parent / "models" / "cached"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_DATA_HASH_PATH = _CACHE_DIR / "_data_fingerprint.txt"


def _data_fingerprint(X, y):
    """Create a quick fingerprint of the training data (shape + sample of values)."""
    return f"{X.shape}|{y.shape}|{y[:5].tolist()}|{y[-5:].tolist()}"


def _check_cache_valid(X, y):
    """Check if cached models match the current data. Invalidate if not."""
    fp = _data_fingerprint(X, y)
    if _DATA_HASH_PATH.exists():
        stored = _DATA_HASH_PATH.read_text().strip()
        if stored == fp:
            return True
        # Data changed — clear ALL cached models (joblib + NN exports)
        print("   Data changed — clearing cached models for retrain")
        for f in _CACHE_DIR.glob("*.joblib"):
            f.unlink()
        # Also clear NN exports (critical: these were NOT cleared before,
        # causing GHOST-trained NN normalization to corrupt Virginia inference)
        _nn_export_dir = Path(__file__).parent.parent / "models" / "exports"
        for pattern in ["magnetometer_fcn.keras", "magnetometer_fcn_norm.npz"]:
            p = _nn_export_dir / pattern
            if p.exists():
                p.unlink()
                print(f"   Cleared NN export: {pattern}")
        saved_model_dir = _nn_export_dir / "magnetometer_fcn_saved_model"
        if saved_model_dir.exists():
            shutil.rmtree(saved_model_dir)
            print("   Cleared NN saved_model directory")
    _DATA_HASH_PATH.write_text(fp)
    return False

# ==============================================================================
# CONDITIONAL IMPORTS - TensorFlow may not be available
# ==============================================================================

TENSORFLOW_AVAILABLE = False
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv1D, Flatten, Dense, BatchNormalization, Dropout, Reshape
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    Sequential = MagicMock
    load_model = MagicMock
    Conv1D = Flatten = Dense = BatchNormalization = Dropout = Reshape = Adam = MagicMock

# Hailo NPU for NN inference
HAILO_NN_AVAILABLE = False
try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface, InferVStreams,
        ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
    )
    HAILO_NN_AVAILABLE = True
except ImportError:
    pass


# ==============================================================================
# RANDOM FOREST REGRESSOR
# ==============================================================================

def run_random_forest(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate a Random Forest Regressor.
    Loads from cache if available, otherwise trains and saves.
    Auto-invalidates cache when input data changes.

    Args:
        X: Feature matrix (n_samples, 3) - [Bx, By, Bz] (no time feature)
        y: Target values (magnetic field magnitude)
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        tuple: (model, X_test, y_test, y_pred, mse, r2)
    """
    print("\n" + "="*50)
    print("RANDOM FOREST REGRESSOR")
    print("="*50)

    if len(X) < 10:
        print("Not enough data for Random Forest training.")
        return None, None, None, None, None, None

    # Invalidate cache if training data changed (different flight file)
    _check_cache_valid(X, y)

    cache_path = _CACHE_DIR / "rf_model.joblib"

    # Split data (always, for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Try loading cached model
    if cache_path.exists():
        rf_model = joblib.load(cache_path)
        print("✓ Loaded cached model")

        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"R² on current data: {r2:.4f}")
        print(f"MSE: {mse:.4f}")

        return rf_model, X_test, y_test, y_pred, mse, r2

    # No cache — train from scratch
    rf_model = RandomForestRegressor(
        n_estimators=ML_CONFIG['RF_N_ESTIMATORS'],
        max_depth=ML_CONFIG['RF_MAX_DEPTH'],
        random_state=random_state,
        n_jobs=-1
    )

    start_time = time.time()
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Save to cache
    joblib.dump(rf_model, cache_path)
    print(f"✓ Trained and cached ({training_time:.2f}s)")

    # Predict and evaluate
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Samples: {len(X_train)} train, {len(X_test)} test")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

    # Feature importance
    importance = rf_model.feature_importances_
    feature_names = ['Bx', 'By', 'Bz']
    print("Feature Importance:")
    for name, imp in zip(feature_names, importance):
        print(f"  {name}: {imp:.4f}")

    return rf_model, X_test, y_test, y_pred, mse, r2


# ==============================================================================
# K-MEANS CLUSTERING
# ==============================================================================

def run_kmeans(X_spatial, k='auto', max_k=6, random_state=42):
    """
    Perform K-Means clustering on spatial magnetometer data.
    Loads from cache if available, otherwise trains and saves.

    Args:
        X_spatial: Feature matrix (n_samples, 3) - [x, y, z]
        k: Number of clusters, or 'auto' to find optimal
        max_k: Maximum clusters to test when k='auto'
        random_state: Random seed

    Returns:
        tuple: (cluster_labels, cluster_centers)
    """
    print("\n" + "="*50)
    print("K-MEANS CLUSTERING")
    print("="*50)

    if len(X_spatial) < 3:
        print("Not enough data for clustering.")
        return np.zeros(len(X_spatial)), np.zeros((1, 3))

    cache_path = _CACHE_DIR / "kmeans_model.joblib"

    # Try loading cached model
    if cache_path.exists():
        cached = joblib.load(cache_path)
        kmeans_model = cached['model']
        best_k = cached['k']
        print(f"✓ Loaded cached model (k={best_k})")

        labels = kmeans_model.predict(X_spatial)
        centers = kmeans_model.cluster_centers_

        for i in range(best_k):
            cluster_size = np.sum(labels == i)
            print(f"Cluster {i+1}: {cluster_size} points ({100*cluster_size/len(labels):.1f}%)")

        return labels, centers

    # No cache — find optimal k and train
    if k == 'auto':
        max_k = min(max_k, len(X_spatial) // 100)
        max_k = max(2, max_k)

        best_k = 2
        best_score = -1

        print(f"Testing k=2 to k={max_k}...")
        for test_k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=test_k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X_spatial)

            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_spatial, labels)
                print(f"  k={test_k}: silhouette={score:.3f}")
                if score > best_score:
                    best_score = score
                    best_k = test_k

        k = best_k
        print(f"Selected k={k} (best silhouette: {best_score:.3f})")

    # Final clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_spatial)
    centers = kmeans.cluster_centers_

    # Save to cache
    joblib.dump({'model': kmeans, 'k': k}, cache_path)
    print("✓ Trained and cached")

    # Print cluster statistics
    for i in range(k):
        cluster_mask = labels == i
        cluster_size = np.sum(cluster_mask)
        print(f"Cluster {i+1}: {cluster_size} points ({100*cluster_size/len(labels):.1f}%)")

    return labels, centers


# ==============================================================================
# NEURAL NETWORK (Conv1D)
# ==============================================================================

# FCN model (fully convolutional - matches Hailo HEF exactly, 3 features only)
_FCN_MODEL_DIR = Path(__file__).parent.parent / "models" / "exports"
_FCN_MODEL_PATH = _FCN_MODEL_DIR / "magnetometer_fcn.keras"
_FCN_NORM_PATH = _FCN_MODEL_DIR / "magnetometer_fcn_norm.npz"

# Cached model directory
_CACHED_MODEL_DIR = Path(__file__).parent.parent / "models" / "cached"


def load_cached_nn_model():
    """
    Load the cached FCN neural network model (3 features: Bx, By, Bz only).
    
    IMPORTANT: Only loads the FCN model which has exactly 3 input features.
    The legacy magnitude_predictor_best.keras (4 features including Time) is
    NEVER loaded — it caused R² = -105 on the Hailo NPU due to mismatched
    input dimensions.
    
    Returns:
        tuple: (model, mean, std, r2) or (None, None, None, None) if not found
    """
    if not TENSORFLOW_AVAILABLE:
        return None, None, None, None

    if not _FCN_MODEL_PATH.exists() or not _FCN_NORM_PATH.exists():
        print("  ⚠ FCN model not found — no NN model available")
        print(f"    Expected: {_FCN_MODEL_PATH}")
        return None, None, None, None

    try:
        model = load_model(_FCN_MODEL_PATH)
        norm_data = np.load(_FCN_NORM_PATH)
        mean = norm_data['mean']
        std = norm_data['std']
        r2 = float(norm_data['r2']) if 'r2' in norm_data else None

        # Safety guard: reject if normalization has wrong number of features
        if len(mean) != 3:
            print(f"  ❌ REJECTED: normalization has {len(mean)} features, expected 3")
            print(f"    This model likely includes Time — DO NOT use with Hailo HEF")
            return None, None, None, None

        print("  Using FCN model (3 features: Bx, By, Bz — matches Hailo HEF)")
        return model, mean, std, r2
    except Exception as e:
        print(f"  FCN model load failed: {e}")
        return None, None, None, None


def build_nn_model(input_shape):
    """
    Build a fully convolutional neural network (FCN) for magnetometer prediction.
    
    Architecture for 3-feature input [Bx, By, Bz]:
    Input(3,1) → Conv1D(64) → BN → Conv1D(128) → BN →
    Conv1D(64) → BN → Conv1D(32) → BN → Conv1D(1,k=3) → Reshape(1)
    
    No Dense/Flatten layers — all Conv1D for Hailo-8 NPU compatibility.

    Args:
        input_shape: Shape of input data (features, 1), e.g. (3, 1)

    Returns:
        Compiled Keras model
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - cannot build NN model")
        return None

    model = Sequential([
        Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv1D(128, kernel_size=2, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, kernel_size=2, strides=2, activation='relu', padding='valid'),
        BatchNormalization(),
        Conv1D(32, kernel_size=2, activation='relu', padding='valid'),
        BatchNormalization(),
        Conv1D(1, kernel_size=1, activation='linear', padding='valid'),
        Reshape((1,))
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model

def _hailo_batch_predict(X_norm, hef_path):
    """
    Run batch inference on Hailo NPU using magnetometer.hef.

    Args:
        X_norm: Normalized input array (n_samples, 3) — float32
        hef_path: Path to the .hef file

    Returns:
        y_pred (ndarray) or None on failure
    """
    if not HAILO_NN_AVAILABLE or not Path(hef_path).exists():
        return None

    try:
        hef = HEF(str(hef_path))
        vdevice = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe)
        network_group = vdevice.configure(hef, configure_params)[0]

        input_info = hef.get_input_vstream_infos()
        output_info = hef.get_output_vstream_infos()
        input_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32)
        output_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32)

        hef_shape = input_info[0].shape            # e.g. (3, 1)
        input_name = input_info[0].name
        output_name = output_info[0].name

        X_in = X_norm.astype(np.float32).reshape(len(X_norm), *hef_shape)

        with network_group.activate():
            with InferVStreams(network_group, input_params, output_params) as pipeline:
                results = pipeline.infer({input_name: X_in})

        del vdevice  # release Hailo device
        return results[output_name].flatten()

    except Exception as e:
        print(f"  Hailo inference error: {e}")
        return None


def run_nn_model(X, y, features=3, epochs=None, batch_size=None, test_size=0.2, random_state=42):
    """
    Load cached Neural Network model or train a new one.
    When Hailo NPU is available, uses it for inference instead of TensorFlow.

    Args:
        X: Feature matrix (n_samples, 3) - [Bx, By, Bz] (no time feature)
        y: Target values
        features: Number of features (default 3)
        epochs: Training epochs (default from config)
        batch_size: Batch size (default from config)
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        tuple: (model, X_test, y_test, y_pred, mse, r2, history)
               model has attribute `_nn_backend` = 'hailo' | 'tensorflow'
    """
    print("\n" + "="*50)
    print("NEURAL NETWORK (Conv1D)")
    print("="*50)

    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - skipping NN training")
        return None, None, None, None, None, None, None

    if len(X) < 10:
        print("Not enough data for NN training.")
        return None, None, None, None, None, None, None

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Try to load cached FCN model
    model, cached_mean, cached_std, cached_r2 = load_cached_nn_model()
    
    if model is not None:
        print(f"✓ Loaded cached model")
        print(f"  Cached R²: {cached_r2:.4f}" if cached_r2 else "  Cached R²: unknown")
        
        # Use cached normalization parameters
        X_test_norm = (X_test - cached_mean) / cached_std

        # ── Try Hailo NPU first ──────────────────────────────────
        nn_backend = 'tensorflow'  # default
        y_pred = None

        if HAILO_NN_AVAILABLE and Path(HAILO_HEF_PATH).exists():
            t0 = time.time()
            y_pred = _hailo_batch_predict(X_test_norm, HAILO_HEF_PATH)
            if y_pred is not None:
                hailo_ms = (time.time() - t0) * 1000
                nn_backend = 'hailo'
                print(f"  ⚡ Hailo NPU inference: {len(X_test)} samples in {hailo_ms:.0f} ms")

        # ── TensorFlow CPU fallback ──────────────────────────────
        if y_pred is None:
            # Reshape to (N, 3, 1) for Conv1D input
            X_tf_in = X_test_norm.reshape(len(X_test_norm), -1, 1).astype(np.float32)
            y_pred = model.predict(X_tf_in, verbose=0).flatten()
            print(f"  🖥  TensorFlow CPU inference")

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model._nn_backend = nn_backend
        
        print(f"  R² on current test data: {r2:.4f}")
        print(f"  MSE: {mse:.4f}")
        
        return model, X_test, y_test, y_pred, mse, r2, None

    # No cached model - train from scratch
    print("No cached model found. Training from scratch...")
    
    epochs = epochs or ML_CONFIG['NN_EPOCHS']
    batch_size = batch_size or ML_CONFIG['NN_BATCH_SIZE']

    # Reshape for Conv1D: (samples, features, 1)
    X_train_nn = X_train.reshape((X_train.shape[0], features, 1))
    X_test_nn = X_test.reshape((X_test.shape[0], features, 1))

    # Build and train model
    model = build_nn_model((features, 1))
    if model is None:
        return None, None, None, None, None, None, None

    print(f"Training NN on {len(X_train)} samples...")
    start_time = time.time()

    history = model.fit(
        X_train_nn, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0
    )

    training_time = time.time() - start_time

    # Predict and evaluate
    y_pred = model.predict(X_test_nn, verbose=0).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model._nn_backend = 'tensorflow'

    print(f"Training time: {training_time:.2f}s ({epochs} epochs)")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

    return model, X_test, y_test, y_pred, mse, r2, history


# ==============================================================================
# GAUSSIAN PROCESS REGRESSOR
# ==============================================================================

def run_gaussian_process(X, y, max_samples=None, test_size=0.2, random_state=42):
    """
    Train and evaluate a Gaussian Process Regressor.
    Loads from cache if available, otherwise trains and saves.

    Args:
        X: Feature matrix (n_samples, 3) - [Bx, By, Bz] (no time feature)
        y: Target values
        max_samples: Maximum samples for GPR training (default from config)
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        tuple: (model, X_test, y_test, y_pred, y_std, mse, r2)
    """
    print("\n" + "="*50)
    print("GAUSSIAN PROCESS REGRESSOR")
    print("="*50)

    if len(X) < 20:
        print("Not enough data for GPR training.")
        return None, None, None, None, None, None, None

    max_samples = max_samples or ML_CONFIG['GPR_MAX_SAMPLES']
    cache_path = _CACHE_DIR / "gpr_model.joblib"

    # Split data (always, for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Try loading cached model
    if cache_path.exists():
        cached = joblib.load(cache_path)
        gpr_model = cached['model']
        X_scaler = cached['X_scaler']
        y_scaler = cached['y_scaler']
        print("✓ Loaded cached model")

        X_test_scaled = X_scaler.transform(X_test)
        y_pred_scaled, y_std_scaled = gpr_model.predict(X_test_scaled, return_std=True)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_std_raw = y_std_scaled * y_scaler.scale_[0]

        residuals = np.abs(y_test - y_pred)
        empirical_std = np.std(residuals)
        min_uncertainty = empirical_std * 0.5
        upper_bound = empirical_std * 5
        y_std = np.clip(np.maximum(y_std_raw, min_uncertainty), min_uncertainty, upper_bound)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"R² on current data: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"Mean uncertainty: {y_std.mean():.2f} nT")

        return gpr_model, X_test, y_test, y_pred, y_std, mse, r2

    # No cache — train from scratch
    from sklearn.preprocessing import StandardScaler

    target_samples = min(max_samples, len(X_train))

    # Time-stratified subsampling: divide training data into chunks and
    # sample proportionally from each, ensuring temporal coverage across
    # the full dataset (important for combined multi-environment training).
    if len(X_train) > target_samples:
        n_chunks = min(20, target_samples // 10)
        chunk_size = len(X_train) // n_chunks
        samples_per_chunk = target_samples // n_chunks
        indices = []
        rng = np.random.RandomState(random_state)
        for c in range(n_chunks):
            start = c * chunk_size
            end = start + chunk_size if c < n_chunks - 1 else len(X_train)
            chunk_indices = rng.choice(
                np.arange(start, end),
                size=min(samples_per_chunk, end - start),
                replace=False
            )
            indices.extend(chunk_indices)
        indices = np.array(indices)
        X_gpr = X_train[indices]
        y_gpr = y_train[indices]
    else:
        X_gpr = X_train
        y_gpr = y_train

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_gpr_scaled = X_scaler.fit_transform(X_gpr)
    y_gpr_scaled = y_scaler.fit_transform(y_gpr.reshape(-1, 1)).flatten()

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e3)) *
        RBF(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(0.1, 100.0)) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e1))
    )
    gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=random_state, n_restarts_optimizer=10)

    print(f"Training GPR on {len(X_gpr)} stratified samples...")
    start_time = time.time()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr_model.fit(X_gpr_scaled, y_gpr_scaled)

    training_time = time.time() - start_time

    # Save to cache
    joblib.dump({'model': gpr_model, 'X_scaler': X_scaler, 'y_scaler': y_scaler}, cache_path)
    print(f"✓ Trained and cached ({training_time:.2f}s)")

    # Predict with uncertainty
    X_test_scaled = X_scaler.transform(X_test)
    y_pred_scaled, y_std_scaled = gpr_model.predict(X_test_scaled, return_std=True)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_std_raw = y_std_scaled * y_scaler.scale_[0]

    residuals = np.abs(y_test - y_pred)
    empirical_std = np.std(residuals)
    print(f"Empirical prediction std: {empirical_std:.2f} nT")

    min_uncertainty = empirical_std * 0.5
    upper_bound = empirical_std * 5
    y_std_clipped = np.clip(np.maximum(y_std_raw, min_uncertainty), min_uncertainty, upper_bound)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Training time: {training_time:.2f}s")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Mean uncertainty: {y_std_clipped.mean():.2f} nT")

    return gpr_model, X_test, y_test, y_pred, y_std_clipped, mse, r2


def gpr_predict_full(X, y_values):
    """
    Predict on the full dataset using the cached GPR model.
    Used for smooth full-timeline visualization (not just the sparse test set).

    Returns:
        tuple: (y_pred, y_std) or (None, None) if cache unavailable
    """
    cache_path = _CACHE_DIR / "gpr_model.joblib"
    if not cache_path.exists():
        return None, None

    cached = joblib.load(cache_path)
    gpr_model = cached['model']
    X_scaler = cached['X_scaler']
    y_scaler = cached['y_scaler']

    X_scaled = X_scaler.transform(X)
    y_pred_scaled, y_std_scaled = gpr_model.predict(X_scaled, return_std=True)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_std_raw = y_std_scaled * y_scaler.scale_[0]

    residuals = np.abs(y_values - y_pred)
    empirical_std = np.std(residuals)
    min_uncertainty = empirical_std * 0.5
    upper_bound = empirical_std * 5
    y_std = np.clip(np.maximum(y_std_raw, min_uncertainty), min_uncertainty, upper_bound)

    return y_pred, y_std
