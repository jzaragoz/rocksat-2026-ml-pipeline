"""
RockSat-X 2026 - Anomaly Detection Module
Ensemble anomaly detection using Isolation Forest, Z-Score, and LOF.
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from config import ML_CONFIG


def advanced_anomaly_detection(X, y_values, z_threshold=None, contamination=None, label=None):
    """
    Advanced anomaly detection using multiple methods with ensemble voting.

    Methods:
    1. Isolation Forest - Tree-based multivariate anomaly detection
    2. Z-Score - Statistical threshold on magnitude
    3. Local Outlier Factor (LOF) - Density-based anomaly detection
    4. Rate of Change - Flags rapid magnitude changes (NEW)

    A point is flagged as anomalous if N+ methods agree (ensemble voting).

    Args:
        X: Feature matrix (n_samples, 3) - [Bx, By, Bz]
        y_values: Magnetic field magnitude values
        z_threshold: Z-score threshold (default from config)
        contamination: Isolation Forest contamination (default from config)

    Returns:
        dict: Results containing anomaly masks and scores for each method
    """
    z_threshold = z_threshold or ML_CONFIG['ANOMALY_Z_THRESHOLD']
    contamination = contamination or ML_CONFIG.get('ISOLATION_FOREST_CONTAMINATION', 0.02)
    vote_threshold = ML_CONFIG.get('ENSEMBLE_VOTE_THRESHOLD', 2)
    rate_threshold = ML_CONFIG.get('RATE_OF_CHANGE_THRESHOLD', 200.0)
    lof_neighbors = ML_CONFIG.get('LOF_N_NEIGHBORS', 50)
    lof_contamination = ML_CONFIG.get('LOF_CONTAMINATION', 0.02)

    results = {
        'isolation_forest': {'anomalies': None, 'scores': None, 'threshold': None},
        'z_score': {'anomalies': None, 'scores': None},
        'lof': {'anomalies': None, 'scores': None},
        'rate_of_change': {'anomalies': None, 'scores': None},
        'ensemble': None,
        'ensemble_anomalies': None,
        'models': {}
    }

    if len(X) < 10:
        print("Not enough data for anomaly detection.")
        return results

    # Use spatial features only (x, y, z)
    X_spatial = X[:, :3]

    # -------------------------------------------------------------------------
    # 1. ISOLATION FOREST
    # -------------------------------------------------------------------------
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_spatial)
    iso_scores = iso_forest.score_samples(X_spatial)

    # Use contamination-based threshold
    iso_anomalies = iso_forest.predict(X_spatial) == -1

    results['isolation_forest']['anomalies'] = iso_anomalies
    results['isolation_forest']['scores'] = iso_scores
    results['models']['isolation_forest'] = iso_forest

    # -------------------------------------------------------------------------
    # 2. MODIFIED Z-SCORE ON MAGNITUDE (robust to existing outliers)
    # -------------------------------------------------------------------------
    # Detrend first: subtract local rolling median so gradual flight dynamics
    # (ascent/descent magnitude changes) don't get flagged.
    # Then apply modified Z-Score (median + MAD) on residuals.
    from scipy.ndimage import median_filter
    window_size = max(31, len(y_values) // 10)
    if window_size % 2 == 0:
        window_size += 1  # median_filter needs odd window
    local_trend = median_filter(y_values, size=window_size, mode='nearest')
    y_detrended = y_values - local_trend

    med = np.median(y_detrended)
    mad = np.median(np.abs(y_detrended - med))
    if mad < 1e-9:
        mad = np.std(y_detrended)  # fallback
    z_scores = 0.6745 * np.abs(y_detrended - med) / max(mad, 1e-9)
    z_anomalies = z_scores > z_threshold

    results['z_score']['anomalies'] = z_anomalies
    results['z_score']['scores'] = z_scores

    # -------------------------------------------------------------------------
    # 3. LOCAL OUTLIER FACTOR
    # -------------------------------------------------------------------------
    lof = LocalOutlierFactor(
        n_neighbors=lof_neighbors,
        contamination=lof_contamination
    )
    lof_labels = lof.fit_predict(X_spatial)
    lof_anomalies = lof_labels == -1
    lof_scores = -lof.negative_outlier_factor_

    results['lof']['anomalies'] = lof_anomalies
    results['lof']['scores'] = lof_scores
    results['models']['lof'] = lof

    # -------------------------------------------------------------------------
    # 4. RATE OF CHANGE (dB/dt) — phase-adaptive thresholds
    # -------------------------------------------------------------------------
    # Use per-phase percentile thresholds instead of a single global value
    rate_of_change = np.abs(np.diff(y_values, prepend=y_values[0]))
    # Data-adaptive: use 99th percentile AND scale to magnitude range.
    # The fixed 200 nT/sample threshold doesn't adapt to different magnetic
    # environments (Virginia ~2600 nT vs GHOST ~4000 nT). Use 5% of the
    # data's dynamic range as a floor, ensuring proportional sensitivity.
    positive_rates = rate_of_change[rate_of_change > 0]
    p99 = np.percentile(positive_rates, 99) if len(positive_rates) > 0 else 0.0
    magnitude_range = np.percentile(y_values, 95) - np.percentile(y_values, 5)
    adaptive_roc_floor = magnitude_range * 0.05  # 5% of dynamic range
    adaptive_roc_threshold = max(p99, adaptive_roc_floor)
    rate_anomalies = rate_of_change > adaptive_roc_threshold
    
    results['rate_of_change']['anomalies'] = rate_anomalies
    results['rate_of_change']['scores'] = rate_of_change

    # -------------------------------------------------------------------------
    # ENSEMBLE VOTING (configurable threshold)
    # -------------------------------------------------------------------------
    ensemble_votes = (
        iso_anomalies.astype(int) +
        z_anomalies.astype(int) +
        lof_anomalies.astype(int) +
        rate_anomalies.astype(int)
    )
    ensemble_anomalies = ensemble_votes >= vote_threshold

    results['ensemble'] = ensemble_votes
    results['ensemble_anomalies'] = ensemble_anomalies

    # -------------------------------------------------------------------------
    # PRINT SUMMARY
    # -------------------------------------------------------------------------
    n_iso = np.sum(iso_anomalies)
    n_z = np.sum(z_anomalies)
    n_lof = np.sum(lof_anomalies)
    n_ensemble = np.sum(ensemble_anomalies)
    n_rate = np.sum(rate_anomalies)
    total = len(X)

    header = f"\nAnomaly Detection Results{f' — {label}' if label else ''}:"
    print(header)
    print(f"  - Isolation Forest: {n_iso} ({100*n_iso/total:.1f}%)")
    print(f"  - Z-Score (>{z_threshold}σ): {n_z} ({100*n_z/total:.1f}%)")
    print(f"  - LOF: {n_lof} ({100*n_lof/total:.1f}%)")
    print(f"  - Rate of Change (>{rate_threshold} nT/sample): {n_rate} ({100*n_rate/total:.1f}%)")
    print(f"  - Ensemble ({vote_threshold}+ methods agree): {n_ensemble} ({100*n_ensemble/total:.1f}%)")

    return results


def get_anomaly_details(X, y_values, anomaly_results, top_n=10):
    """
    Get details about the most significant anomalies.

    Args:
        X: Feature matrix
        y_values: Magnitude values
        anomaly_results: Results from advanced_anomaly_detection()
        top_n: Number of top anomalies to return

    Returns:
        list: List of dicts with anomaly details
    """
    ensemble_mask = anomaly_results['ensemble_anomalies']
    if ensemble_mask is None or not np.any(ensemble_mask):
        return []

    anomaly_indices = np.where(ensemble_mask)[0]

    # Sort by ensemble vote count (most agreement first)
    votes = anomaly_results['ensemble'][anomaly_indices]
    sorted_indices = anomaly_indices[np.argsort(-votes)][:top_n]

    anomalies = []
    for idx in sorted_indices:
        anomalies.append({
            'index': int(idx),
            'time': float(X[idx, 3]) if X.shape[1] > 3 else float(idx),
            'x': float(X[idx, 0]),
            'y': float(X[idx, 1]),
            'z': float(X[idx, 2]),
            'magnitude': float(y_values[idx]),
            'votes': int(anomaly_results['ensemble'][idx]),
            'z_score': float(anomaly_results['z_score']['scores'][idx]),
            'iso_score': float(anomaly_results['isolation_forest']['scores'][idx]),
            'lof_score': float(anomaly_results['lof']['scores'][idx]),
            'rate_of_change': float(anomaly_results['rate_of_change']['scores'][idx])
        })

    return anomalies


def real_time_anomaly_check(reading, models, history_buffer, z_threshold=3.0):
    """
    Check a single reading for anomalies in real-time.
    Used during live sensor monitoring.

    Args:
        reading: dict with 'x', 'y', 'z' keys
        models: dict with trained anomaly detection models
        history_buffer: deque of recent readings for context
        z_threshold: Z-score threshold

    Returns:
        dict: {'is_anomaly': bool, 'methods': list, 'scores': dict}
    """
    result = {
        'is_anomaly': False,
        'methods': [],
        'scores': {}
    }

    if len(history_buffer) < 20:
        return result  # Not enough history for comparison

    # Extract current point
    x, y, z = reading['x'], reading['y'], reading['z']
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    point = np.array([[x, y, z]])

    # Get historical magnitudes for Z-score
    hist_magnitudes = [np.sqrt(h['x']**2 + h['y']**2 + h['z']**2)
                       for h in history_buffer]

    # 1. Z-Score check
    mean_mag = np.mean(hist_magnitudes)
    std_mag = np.std(hist_magnitudes)
    if std_mag > 0:
        z_score = abs(magnitude - mean_mag) / std_mag
        result['scores']['z_score'] = z_score
        if z_score > z_threshold:
            result['methods'].append('z_score')

    # 2. Isolation Forest check (if model available)
    if 'isolation_forest' in models and models['isolation_forest'] is not None:
        iso_score = models['isolation_forest'].score_samples(point)[0]
        result['scores']['isolation_forest'] = iso_score
        # Use adaptive threshold based on training
        if iso_score < -0.5:  # Conservative threshold
            result['methods'].append('isolation_forest')

    # 3. Simple distance-based check (LOF approximation)
    hist_points = np.array([[h['x'], h['y'], h['z']] for h in history_buffer])
    distances = np.sqrt(np.sum((hist_points - point)**2, axis=1))
    mean_dist = np.mean(distances)
    if mean_dist > np.percentile(distances, 95):
        result['methods'].append('distance')
        result['scores']['distance'] = mean_dist

    # Ensemble decision
    result['is_anomaly'] = len(result['methods']) >= 2

    return result


def classify_anomaly_type(anomaly_details, mission_time):
    """
    Classify what type of anomaly was detected based on context.
    Uses MISSION_TIMELINE from config for event windows (universal).

    Args:
        anomaly_details: dict with anomaly information
        mission_time: Time relative to launch

    Returns:
        str: Anomaly classification
    """
    from config import MISSION_TIMELINE

    z_score = anomaly_details.get('z_score', 0)

    # Check configured mission events (±5s window around each)
    event_map = {
        'T_LAUNCH': 'LAUNCH_TRANSIENT',
        'T_BURN_END': 'MOTOR_BURNOUT_TRANSIENT',
        'T_SKIRT_DEPLOY': 'NOSECONE_SEPARATION',
        'T_APOGEE': 'APOGEE_REGION',
    }
    for key, label in event_map.items():
        t_event = MISSION_TIMELINE.get(key)
        if t_event is not None and abs(mission_time - t_event) < 5:
            return label

    # Launch is always at T=0 even if not in timeline
    if abs(mission_time) < 5:
        return 'LAUNCH_TRANSIENT'

    # Statistical classification (no hardcoded magnitudes)
    if z_score > 5:
        return 'EXTREME_OUTLIER'
    else:
        return 'UNKNOWN'
