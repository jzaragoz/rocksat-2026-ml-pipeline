"""
RockSat-X 2026 - Visualization Module
All plotting functions for ML results and sensor data.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D

import config
from config import (SENSOR_COLORS, AXIS_COLORS, GAP_COLOR, LAUNCH_COLOR,
                     MISSION_TIMELINE, MISSION_NAME, MISSION_EVENTS)

def _post_gap_start():
    """Compute plot x-axis start from detected gaps (called at plot time, not import time)."""
    if config.DETECTED_GAPS:
        return max(0, config.DETECTED_GAPS[-1][1] - 5)
    return 0

# Descriptive labels for each RM3100 magnetometer (MUX channel)
SENSOR_LABELS = {
    0: 'RM3100 #0 (MUX Ch0)',
    1: 'RM3100 #1 (MUX Ch1)',
    2: 'RM3100 #2 (MUX Ch2)',
    3: 'RM3100 #3 (MUX Ch3)',
}

def _sensor_label(sensor_id):
    """Return descriptive label for a sensor, with fallback."""
    return SENSOR_LABELS.get(sensor_id, f'RM3100 #{sensor_id}')


def _plot_per_sensor(ax, mag_data_by_sensor, s=8, alpha=0.5):
    """Plot per-sensor magnitude as colored dots (background layer)."""
    for sid in sorted(mag_data_by_sensor.keys()):
        sensor = mag_data_by_sensor[sid]
        color = SENSOR_COLORS[sid % len(SENSOR_COLORS)]
        label = f'{_sensor_label(sid)} (n={sensor["count"]})'
        ax.scatter(sensor['time'], sensor['magnitude'],
                   c=color, s=s, alpha=alpha, label=label, edgecolors='none')


def _avg_by_time(time_vals, *arrays):
    """
    Average arrays at each unique timestamp (sensor averaging).
    When multiple sensors share timestamps, averages their values
    to produce one clean time series.

    If timestamps are mostly unique (single sensor), just sorts by time.

    Returns:
        tuple: (time_sorted, arr1_avg, arr2_avg, ...)
    """
    unique_t = np.unique(time_vals)
    if len(unique_t) >= len(time_vals) * 0.8:
        # Mostly unique timestamps — no multi-sensor averaging needed
        sort_idx = np.argsort(time_vals)
        return (time_vals[sort_idx],) + tuple(a[sort_idx] for a in arrays)

    # Multiple sensors share timestamps — average at each time step
    result_arrays = [np.empty(len(unique_t)) for _ in arrays]
    for j, t in enumerate(unique_t):
        mask = time_vals == t
        for i, a in enumerate(arrays):
            result_arrays[i][j] = a[mask].mean()

    return (unique_t,) + tuple(result_arrays)



# ==============================================================================
# TIME FORMATTING HELPERS (T+/T- flight standard)
# ==============================================================================

def format_time_tplus(x, pos=None):
    """
    Format time values as T+/T- for flight standard display.
    
    Args:
        x: Time value in seconds (relative to launch, T=0)
        pos: Tick position (unused, required by FuncFormatter)
    
    Returns:
        str: Formatted time string (e.g., 'T-30', 'T+120', 'T0')
    """
    if x == 0:
        return 'T0'
    elif x > 0:
        return f'T+{int(x)}'
    else:
        return f'T{int(x)}'


def set_time_axis_format(ax, axis='x'):
    """
    Apply T+/T- formatting to an axis.
    
    Args:
        ax: Matplotlib axis object
        axis: 'x' or 'y' to specify which axis to format
    """
    formatter = FuncFormatter(format_time_tplus)
    if axis == 'x':
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel('Mission Time', color='white')
    else:
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel('Mission Time', color='white')


# ==============================================================================
# GAP SHADING HELPER
# ==============================================================================

def add_gap_shading(ax, gaps=None, color=GAP_COLOR, alpha=0.2, label='Payload Off'):
    """
    Add shaded regions for time gaps (payload OFF periods).

    Args:
        ax: Matplotlib axis
        gaps: List of (start, end) tuples. If None, uses detected gaps.
        color: Shading color
        alpha: Transparency
        label: Legend label (only added to first gap)
    """
    if gaps is None:
        gaps = config.DETECTED_GAPS

    for i, (gap_start, gap_end) in enumerate(gaps):
        lbl = label if i == 0 else None
        ax.axvspan(gap_start, gap_end, alpha=alpha, color=color, zorder=0, label=lbl)


def add_launch_marker(ax, label='Launch'):
    """Add vertical line at T=0 (launch)."""
    ax.axvline(x=0, color=LAUNCH_COLOR, linestyle='--', linewidth=2, alpha=0.7, label=label)


# ==============================================================================
# MULTI-MAGNETOMETER ARRAY PLOT
# ==============================================================================

def plot_multi_magnetometer(mag_data_by_sensor, save_path=None):
    """
    Comprehensive multi-magnetometer visualization showing all sensors.
    4-panel plot: Magnitude, X-axis, Y-axis, Z-axis comparisons.

    Args:
        mag_data_by_sensor: dict {sensor_id: {'time', 'x', 'y', 'z', 'magnitude', 'count'}}
        save_path: Path to save figure
    """
    if not mag_data_by_sensor or len(mag_data_by_sensor) == 0:
        print("   No multi-magnetometer data available")
        return None

    n_sensors = len(mag_data_by_sensor)
    plt.style.use('dark_background')

    fig = plt.figure(figsize=(16, 12), facecolor='black')

    # Panel 1: All magnetometer magnitudes overlaid
    ax1 = fig.add_subplot(2, 2, 1)
    for i, (mag_id, data) in enumerate(sorted(mag_data_by_sensor.items())):
        color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
        ax1.scatter(data['time'], data['magnitude'], c=color, alpha=0.6,
                    s=3, label=f'{_sensor_label(mag_id)} (n={data["count"]})')
    add_launch_marker(ax1)
    add_gap_shading(ax1)
    ax1.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax1, 'x')
    ax1.set_ylabel('|B| (nT)', color='white')
    ax1.set_title('All Magnetometers - Magnitude', color='white', fontsize=12, fontweight='bold')
    ax1.legend(facecolor='#111', labelcolor='white', fontsize=9, markerscale=3)
    ax1.grid(True, alpha=0.3, color='gray')
    ax1.set_facecolor('black')
    ax1.tick_params(colors='white')

    # Panel 2: X-axis comparison
    ax2 = fig.add_subplot(2, 2, 2)
    for i, (mag_id, data) in enumerate(sorted(mag_data_by_sensor.items())):
        color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
        ax2.scatter(data['time'], data['x'], c=color, alpha=0.6, s=3, label=f'{_sensor_label(mag_id)}')
    add_launch_marker(ax2)
    add_gap_shading(ax2)
    ax2.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax2, 'x')
    ax2.set_ylabel('Bx (nT)', color='white')
    ax2.set_title('X-Axis Comparison', color='#FF6B6B', fontsize=12, fontweight='bold')
    ax2.legend(facecolor='#111', labelcolor='white', fontsize=9, markerscale=3)
    ax2.grid(True, alpha=0.3, color='gray')
    ax2.set_facecolor('black')
    ax2.tick_params(colors='white')

    # Panel 3: Y-axis comparison
    ax3 = fig.add_subplot(2, 2, 3)
    for i, (mag_id, data) in enumerate(sorted(mag_data_by_sensor.items())):
        color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
        ax3.scatter(data['time'], data['y'], c=color, alpha=0.6, s=3, label=f'{_sensor_label(mag_id)}')
    add_launch_marker(ax3)
    add_gap_shading(ax3)
    ax3.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax3, 'x')
    ax3.set_ylabel('By (nT)', color='white')
    ax3.set_title('Y-Axis Comparison', color='#FFE66D', fontsize=12, fontweight='bold')
    ax3.legend(facecolor='#111', labelcolor='white', fontsize=9, markerscale=3)
    ax3.grid(True, alpha=0.3, color='gray')
    ax3.set_facecolor('black')
    ax3.tick_params(colors='white')

    # Panel 4: Z-axis comparison
    ax4 = fig.add_subplot(2, 2, 4)
    for i, (mag_id, data) in enumerate(sorted(mag_data_by_sensor.items())):
        color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
        ax4.scatter(data['time'], data['z'], c=color, alpha=0.6, s=3, label=f'{_sensor_label(mag_id)}')
    add_launch_marker(ax4)
    add_gap_shading(ax4)
    ax4.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax4, 'x')
    ax4.set_ylabel('Bz (nT)', color='white')
    ax4.set_title('Z-Axis Comparison', color='#4ECDC4', fontsize=12, fontweight='bold')
    ax4.legend(facecolor='#111', labelcolor='white', fontsize=9, markerscale=3)
    ax4.grid(True, alpha=0.3, color='gray')
    ax4.set_facecolor('black')
    ax4.tick_params(colors='white')

    fig.suptitle(f'{MISSION_NAME} — RM3100 Magnetometer Array ({n_sensors} Sensors)',
                 color='white', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


# ==============================================================================
# K-MEANS CLUSTERING PLOT
# ==============================================================================

def plot_clusters(X, cluster_labels, centers, save_path=None, title=None):
    """
    3D scatter plot of K-Means cluster assignments.

    Args:
        X: Feature matrix (n, 3+) — columns 0-2 are Bx, By, Bz; column 3 is time
        cluster_labels: Cluster assignment for each point
        centers: Cluster center coordinates in XYZ space
        save_path: Path to save figure
        title: Plot title
    """
    if len(X) == 0 or len(cluster_labels) == 0:
        print("No data to plot.")
        return None

    labels = np.asarray(cluster_labels, dtype=int)
    unique_labels = sorted(set(labels))
    has_noise = -1 in unique_labels
    n_real = len(unique_labels) - (1 if has_noise else 0)

    cluster_df = pd.DataFrame({
        'x': X[:, 0], 'y': X[:, 1], 'z': X[:, 2],
        'Time': X[:, 3] if X.shape[1] >= 4 else np.arange(len(X)),
        'Cluster': labels,
        'orig_idx': np.arange(len(X))
    })

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(n_real, 1)))

    scatter_objects = []
    color_idx = 0
    for label in unique_labels:
        mask = cluster_df['Cluster'] == label
        points = cluster_df[mask]

        if label == -1:
            sc = ax.scatter(points['x'], points['y'], points['z'],
                            c='gray', s=15, alpha=0.3, edgecolors='none',
                            label=f'Noise ({len(points)})')
        else:
            color = colors[color_idx % len(colors)]
            sc = ax.scatter(points['x'], points['y'], points['z'],
                            c=[color], s=50, alpha=0.8, edgecolors='w', linewidth=0.3,
                            label=f'Cluster {label} ({len(points)})')
            color_idx += 1

        scatter_objects.append((sc, points['orig_idx'].values))

    # Centroids - lime green X markers
    if centers is not None and len(centers) > 0:
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   c='lime', s=400, marker='X', edgecolors='black', linewidth=3,
                   label='Centroids', depthshade=False)

    # Styling
    ax.set_xlabel("Bx (nT)", color='white')
    ax.set_ylabel("By (nT)", color='white')
    ax.set_zlabel("Bz (nT)", color='white')
    _title = title if title is not None else f"{MISSION_NAME} — K-Means Clusters"
    ax.set_title(_title, color='white', fontsize=16, pad=20)

    ax.xaxis.set_pane_color((0, 0, 0, 1))
    ax.yaxis.set_pane_color((0, 0, 0, 1))
    ax.zaxis.set_pane_color((0, 0, 0, 1))
    fig.patch.set_facecolor('black')
    ax.grid(True, alpha=0.3, color='#00ff88')
    ax.tick_params(colors='white')

    ax.legend(facecolor='#111', edgecolor='lime', labelcolor='white',
              loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)

    # Hover functionality with mplcursors
    try:
        import mplcursors
        def on_hover(sel):
            artist = sel.artist
            for sc, indices_in_scatter in scatter_objects:
                if sc == artist:
                    local_idx = sel.index
                    global_idx = indices_in_scatter[local_idx]
                    row = cluster_df.iloc[global_idx]
                    t_val = row['Time']
                    c_val = row['Cluster']
                    lbl = 'Noise' if c_val == -1 else f'Regime {c_val}'
                    sel.annotation.set(text=f"Time: {t_val:.2f}s\n{lbl}",
                                       fontsize=11, color='cyan')
                    sel.annotation.get_bbox_patch().set(facecolor='black', alpha=0.95,
                                                       edgecolor='lime', linewidth=1.5)
                    return
            sel.annotation.set(text="Error", color="red")

        cursor = mplcursors.cursor([sc for sc, _ in scatter_objects], hover=True)
        cursor.connect("add", on_hover)
    except ImportError:
        pass

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


# ==============================================================================
# RANDOM FOREST PLOTS
# ==============================================================================

def plot_rf_actual_vs_predicted(y_test, y_pred, r2, save_path=None):
    """
    Scatter plot of Random Forest: Actual vs Predicted with residual analysis.
    Includes annotation explaining this is a calibration check
    (RF learning the Euclidean norm from vector components).

    Args:
        y_test: Actual values
        y_pred: Predicted values
        r2: R-squared score
        save_path: Path to save figure
    """
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='black',
                                     gridspec_kw={'width_ratios': [3, 2]})

    # Left: scatter plot
    ax1.scatter(y_test, y_pred, c='cyan', s=10, alpha=0.5, label='Predictions')

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')

    ax1.set_xlabel('Actual |B| (nT)', color='white', fontsize=12)
    ax1.set_ylabel('Predicted |B| (nT)', color='white', fontsize=12)
    ax1.set_title(f'RF Calibration Check (R\u00b2 = {r2:.6f})',
                 color='white', fontsize=14, fontweight='bold')
    ax1.set_facecolor('black')
    ax1.legend(loc='upper left', facecolor='#111', labelcolor='white')
    ax1.grid(True, alpha=0.3, color='gray')
    ax1.tick_params(colors='white')

    # Annotation explaining high R-squared
    ax1.text(0.02, 0.82,
             'Calibration: RF learns |B| = \u221a(Bx\u00b2+By\u00b2+Bz\u00b2)\n'
             'High R\u00b2 is expected (deterministic target)',
             transform=ax1.transAxes, fontsize=9, color='#FFE66D',
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', alpha=0.8))

    # Right: residual distribution
    residuals = y_test - y_pred
    mae = np.mean(np.abs(residuals))
    ax2.hist(residuals, bins=50, color='cyan', alpha=0.7, edgecolor='white', linewidth=0.3)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Residual (nT)', color='white', fontsize=11)
    ax2.set_ylabel('Count', color='white', fontsize=11)
    ax2.set_title(f'Residuals (MAE = {mae:.2f} nT)',
                 color='white', fontsize=12, fontweight='bold')
    ax2.set_facecolor('black')
    ax2.grid(True, alpha=0.3, color='gray')
    ax2.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


def plot_rf_timeseries(X_test, y_test, y_pred, save_path=None, mag_data_by_sensor=None):
    """
    Time-series plot showing RF actual vs predicted over mission time.

    Args:
        X_test: Test feature matrix (needs time in column 3)
        y_test: Actual values
        y_pred: Predicted values
        save_path: Path to save figure
        mag_data_by_sensor: Per-sensor data dict for multi-sensor display
    """
    if X_test is None or len(X_test) == 0:
        print("   No RF results to plot")
        return None

    plt.style.use('dark_background')

    time_col = X_test[:, 3] if X_test.shape[1] >= 4 else np.arange(len(X_test))

    fig, ax = plt.subplots(figsize=(16, 7), facecolor='black')

    if mag_data_by_sensor and len(mag_data_by_sensor) > 1:
        # Per-sensor actual magnitude (colored lines)
        _plot_per_sensor(ax, mag_data_by_sensor, s=12, alpha=0.5)
        # RF prediction overlay (single color, smaller dots on top)
        sort_idx = np.argsort(time_col)
        ax.scatter(time_col[sort_idx], y_pred.flatten()[sort_idx],
                   color='white', s=4, alpha=0.4, label='RF Predicted', zorder=5)
    else:
        t_avg, y_avg, p_avg = _avg_by_time(time_col, y_test.flatten(), y_pred.flatten())
        ax.scatter(t_avg, y_avg, color='#FF6B6B', s=20, alpha=0.6,
                   label='Actual |B|', edgecolors='white', linewidth=0.2)
        ax.scatter(t_avg, p_avg, color='#4ECDC4', s=20, alpha=0.6,
                   label='RF Predicted', edgecolors='white', linewidth=0.2)

    add_launch_marker(ax)
    add_gap_shading(ax)

    ax.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax, 'x')
    ax.set_ylabel('Magnetic Field Magnitude (nT)', color='white', fontsize=12)
    ax.set_title('Random Forest: Actual vs Predicted Over Time',
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#111', labelcolor='white', loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


# ==============================================================================
# GAUSSIAN PROCESS PLOT
# ==============================================================================

def plot_gpr_uncertainty(X_test, y_test, y_pred, y_std, save_path=None, time_vals=None, r2=None, mag_data_by_sensor=None):
    """
    Plot Gaussian Process predictions with 95% confidence interval.

    Args:
        X_test: Test features
        y_test: Actual values
        y_pred: GPR mean predictions
        y_std: GPR standard deviation (uncertainty)
        save_path: Path to save figure
        time_vals: Optional time values for x-axis
        r2: Optional R² score to display in title
        mag_data_by_sensor: Per-sensor data dict for multi-sensor display
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')

    # Sort by time
    if time_vals is not None:
        tv = time_vals
    elif X_test.shape[1] >= 4:
        tv = X_test[:, 3]
    else:
        tv = np.arange(len(X_test))

    # Average predictions/std for CI band (always need this for fill_between)
    t_avg, p_avg, s_avg = _avg_by_time(tv, y_pred, y_std)

    # 95% confidence interval (mean ± 1.96 * std)
    lower = p_avg - 1.96 * s_avg
    upper = p_avg + 1.96 * s_avg

    if mag_data_by_sensor and len(mag_data_by_sensor) > 1:
        # Per-sensor actual magnitude (colored dots)
        _plot_per_sensor(ax, mag_data_by_sensor, s=8, alpha=0.5)
    else:
        _, y_avg = _avg_by_time(tv, y_test)
        ax.scatter(t_avg, y_avg, c='white', s=8, alpha=0.4, label='Actual data', rasterized=True)

    ax.fill_between(t_avg, lower, upper, alpha=0.35, color='cyan', label='95% confidence band')
    ax.plot(t_avg, p_avg, color='red', linewidth=2, alpha=0.9, label='Predicted average')

    add_launch_marker(ax)
    add_gap_shading(ax)

    ax.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax, 'x')
    ax.set_ylabel('Magnitude (nT)', color='white', fontsize=12)
    r2_str = f' (R\u00b2 = {r2:.6f})' if r2 is not None else ''
    ax.set_title(f'Gaussian Process Regression{r2_str}',
                 color='white', fontsize=14, fontweight='bold')
    ax.set_facecolor('black')
    ax.legend(loc='upper right', facecolor='#111', labelcolor='white', fontsize=9)
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


# ==============================================================================
# ANOMALY DETECTION PLOT
# ==============================================================================

def plot_anomaly_detection(X, y_values, anomaly_results, save_path=None, mag_data_by_sensor=None):
    """
    Plot magnetic field timeline with ALL anomaly methods overlaid.

    Args:
        X: Feature matrix
        y_values: Magnitude values
        anomaly_results: Results from advanced_anomaly_detection()
        save_path: Path to save figure
        mag_data_by_sensor: Per-sensor data dict for multi-sensor display
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')

    time_vals = X[:, 3] if X.shape[1] >= 4 else np.arange(len(X))

    if mag_data_by_sensor and len(mag_data_by_sensor) > 1:
        # Per-sensor actual magnitude (colored dots as background)
        _plot_per_sensor(ax, mag_data_by_sensor, s=4, alpha=0.4)
    else:
        t_avg, y_avg = _avg_by_time(time_vals, y_values)
        ax.scatter(t_avg, y_avg, c='white', s=2, alpha=0.4, label='Normal')

    # --- Individual methods as faded context (plot on raw combined data) ---
    iso_mask = anomaly_results['isolation_forest']['anomalies']
    if iso_mask is not None and np.any(iso_mask):
        ax.scatter(time_vals[iso_mask], y_values[iso_mask],
                   c='red', s=8, alpha=0.25, marker='o',
                   label=f"IF ({np.sum(iso_mask)})")

    z_mask = anomaly_results['z_score']['anomalies']
    if z_mask is not None and np.any(z_mask):
        ax.scatter(time_vals[z_mask], y_values[z_mask],
                   c='yellow', s=20, alpha=0.3, marker='x', linewidths=1,
                   label=f"Z-Score ({np.sum(z_mask)})")

    lof_mask = anomaly_results['lof']['anomalies']
    if lof_mask is not None and np.any(lof_mask):
        ax.scatter(time_vals[lof_mask], y_values[lof_mask],
                   c='#FF6B6B', s=10, alpha=0.25, marker='D',
                   label=f"LOF ({np.sum(lof_mask)})")

    # --- ENSEMBLE: the hero layer ---
    ens_mask = anomaly_results['ensemble_anomalies']
    n_ens = int(np.sum(ens_mask)) if ens_mask is not None else 0
    if ens_mask is not None and np.any(ens_mask):
        ax.scatter(time_vals[ens_mask], y_values[ens_mask],
                   c='#00FFFF', s=120, alpha=0.95, marker='*',
                   edgecolors='white', linewidths=0.5, zorder=10,
                   label=f"ENSEMBLE ({n_ens})")

    add_launch_marker(ax)
    add_gap_shading(ax)

    ax.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax, 'x')
    ax.set_ylabel('Magnitude (nT)', color='white', fontsize=12)
    ax.set_title(f'Anomaly Detection — Ensemble Consensus  ({n_ens} flagged)',
                 color='#00FFFF', fontsize=14, fontweight='bold')
    ax.set_facecolor('black')
    ax.legend(loc='upper right', facecolor='#111', labelcolor='white', fontsize=9)
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


# ==============================================================================
# INDIVIDUAL ANOMALY METHOD PLOTS
# ==============================================================================

def plot_isolation_forest(X, y_values, anomaly_results, save_path=None):
    """Plot Isolation Forest anomalies on the magnetic field timeline."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')

    time_vals = X[:, 3] if X.shape[1] >= 4 else np.arange(len(X))
    iso_mask = anomaly_results['isolation_forest']['anomalies']
    n_iso = int(np.sum(iso_mask)) if iso_mask is not None else 0

    ax.scatter(time_vals, y_values, c='white', s=1, alpha=0.3, label='Normal')
    if iso_mask is not None and np.any(iso_mask):
        ax.scatter(time_vals[iso_mask], y_values[iso_mask],
                   c='red', s=15, alpha=0.7, marker='o',
                   label=f'Anomaly ({n_iso})')

    add_launch_marker(ax)
    add_gap_shading(ax)
    ax.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax, 'x')
    ax.set_ylabel('Magnitude (nT)', color='white', fontsize=12)
    ax.set_title(f'Isolation Forest  ({n_iso} anomalies, {100*n_iso/len(X):.1f}%)',
                 color='white', fontsize=14, fontweight='bold')
    ax.set_facecolor('black')
    ax.legend(loc='upper right', facecolor='#111', labelcolor='white', fontsize=9)
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


def plot_zscore(X, y_values, anomaly_results, save_path=None):
    """Plot Z-Score anomalies on the magnetic field timeline."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')

    time_vals = X[:, 3] if X.shape[1] >= 4 else np.arange(len(X))
    z_mask = anomaly_results['z_score']['anomalies']
    n_z = int(np.sum(z_mask)) if z_mask is not None else 0

    ax.scatter(time_vals, y_values, c='white', s=1, alpha=0.3, label='Normal')
    if z_mask is not None and np.any(z_mask):
        ax.scatter(time_vals[z_mask], y_values[z_mask],
                   c='yellow', s=50, alpha=0.9, marker='x', linewidths=2,
                   label=f'Anomaly ({n_z})')

    add_launch_marker(ax)
    add_gap_shading(ax)
    ax.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax, 'x')
    ax.set_ylabel('Magnitude (nT)', color='white', fontsize=12)
    ax.set_title(f'Z-Score Standard Deviation  ({n_z} anomalies)',
                 color='white', fontsize=14, fontweight='bold')
    ax.set_facecolor('black')
    ax.legend(loc='upper right', facecolor='#111', labelcolor='white', fontsize=9)
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


def plot_lof(X, y_values, anomaly_results, save_path=None):
    """Plot Local Outlier Factor anomalies on the magnetic field timeline."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')

    time_vals = X[:, 3] if X.shape[1] >= 4 else np.arange(len(X))
    lof_mask = anomaly_results['lof']['anomalies']
    n_lof = int(np.sum(lof_mask)) if lof_mask is not None else 0

    ax.scatter(time_vals, y_values, c='white', s=1, alpha=0.3, label='Normal')
    if lof_mask is not None and np.any(lof_mask):
        ax.scatter(time_vals[lof_mask], y_values[lof_mask],
                   c='#FF6B6B', s=20, alpha=0.8, marker='D',
                   label=f'Anomaly ({n_lof})')

    add_launch_marker(ax)
    add_gap_shading(ax)
    ax.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax, 'x')
    ax.set_ylabel('Magnitude (nT)', color='white', fontsize=12)
    ax.set_title(f'Local Outlier Factor  ({n_lof} anomalies, {100*n_lof/len(X):.1f}%)',
                 color='white', fontsize=14, fontweight='bold')
    ax.set_facecolor('black')
    ax.legend(loc='upper right', facecolor='#111', labelcolor='white', fontsize=9)
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


# ==============================================================================
# MISSION PROFILE PLOT
# ==============================================================================

def plot_mission_profile(t, x, y, z, save_path=None):
    """
    Plot magnetometer X/Y/Z components and magnitude over mission time.

    Args:
        t: Time values
        x, y, z: Magnetic field components
        save_path: Path to save figure
    """
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='black')

    magnitude = np.sqrt(x**2 + y**2 + z**2)

    # X component
    ax = axes[0, 0]
    ax.scatter(t, x, c=AXIS_COLORS['x'], s=1, alpha=0.6)
    add_launch_marker(ax)
    add_gap_shading(ax)
    ax.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax, 'x')
    ax.set_ylabel('Bx (nT)', color='white')
    ax.set_title('X-Axis Component', color=AXIS_COLORS['x'], fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')

    # Y component
    ax = axes[0, 1]
    ax.scatter(t, y, c=AXIS_COLORS['y'], s=1, alpha=0.6)
    add_launch_marker(ax)
    add_gap_shading(ax)
    ax.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax, 'x')
    ax.set_ylabel('By (nT)', color='white')
    ax.set_title('Y-Axis Component', color=AXIS_COLORS['y'], fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')

    # Z component
    ax = axes[1, 0]
    ax.scatter(t, z, c=AXIS_COLORS['z'], s=1, alpha=0.6)
    add_launch_marker(ax)
    add_gap_shading(ax)
    ax.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax, 'x')
    ax.set_ylabel('Bz (nT)', color='white')
    ax.set_title('Z-Axis Component', color=AXIS_COLORS['z'], fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')

    # Magnitude
    ax = axes[1, 1]
    ax.scatter(t, magnitude, c='white', s=1, alpha=0.6)
    add_launch_marker(ax)
    add_gap_shading(ax)
    ax.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax, 'x')
    ax.set_ylabel('|B| (nT)', color='white')
    ax.set_title('Total Magnitude', color='white', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')

    fig.suptitle(f'{MISSION_NAME} — Mission Profile', color='white', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


# ==============================================================================
# ROTATION ANALYSIS PLOT
# ==============================================================================

def _interpolate_rotation(t_vals, x_vals, y_vals, z_vals, max_points=3000):
    """Cubic spline interpolation for smooth rotation arcs, handling gaps."""
    n = len(t_vals)
    if n < 4:
        return t_vals, x_vals, y_vals, z_vals

    try:
        from scipy.interpolate import CubicSpline

        dt = np.diff(t_vals)
        gap_mask = dt > 5.0

        if not np.any(gap_mask):
            n_interp = min(n * 5, max_points)
            ti = np.linspace(t_vals[0], t_vals[-1], n_interp)
            return ti, CubicSpline(t_vals, x_vals)(ti), \
                   CubicSpline(t_vals, y_vals)(ti), CubicSpline(t_vals, z_vals)(ti)

        # Gaps present — interpolate each segment separately
        xp, yp, zp, tp = [], [], [], []
        seg_starts = np.concatenate([[0], np.where(gap_mask)[0] + 1])
        seg_ends = np.concatenate([np.where(gap_mask)[0] + 1, [n]])

        for s, e in zip(seg_starts, seg_ends):
            if e - s < 4:
                tp.extend(t_vals[s:e])
                xp.extend(x_vals[s:e])
                yp.extend(y_vals[s:e])
                zp.extend(z_vals[s:e])
                continue

            seg_t = t_vals[s:e]
            ni = min((e - s) * 5, max_points // max(1, len(seg_starts)))
            ti = np.linspace(seg_t[0], seg_t[-1], ni)
            tp.extend(ti)
            xp.extend(CubicSpline(seg_t, x_vals[s:e])(ti))
            yp.extend(CubicSpline(seg_t, y_vals[s:e])(ti))
            zp.extend(CubicSpline(seg_t, z_vals[s:e])(ti))

        return np.array(tp), np.array(xp), np.array(yp), np.array(zp)
    except (ImportError, Exception):
        return t_vals, x_vals, y_vals, z_vals


def plot_magnetometer_rotation(x, y, z, t, save_path=None, mag_data_by_sensor=None):
    """
    Visualize rocket rotation/spin using magnetometer phase planes with time coloring.

    When mag_data_by_sensor is provided (multi-sensor), plots each sensor as a
    separate colored trace — avoids cross-sensor averaging which destroys the
    rotation signal when sensors have different mounting orientations.

    For sparse data, applies cubic spline interpolation for smooth arcs.

    Args:
        x, y, z: Magnetic field components (fallback, all sensors concatenated)
        t: Time values (fallback)
        save_path: Path to save figure
        mag_data_by_sensor: dict of per-sensor data (preferred)
    """
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor='black')

    from matplotlib.colors import Normalize

    # Determine global time range for consistent colormap
    if mag_data_by_sensor and len(mag_data_by_sensor) > 0:
        all_t = np.concatenate([s['time'] for s in mag_data_by_sensor.values()])
    else:
        all_t = t
    norm = Normalize(vmin=all_t.min(), vmax=all_t.max())

    sc_last = None  # for colorbar

    if mag_data_by_sensor and len(mag_data_by_sensor) > 1:
        # Per-sensor rotation: each sensor traces its own clean ellipse
        for sid in sorted(mag_data_by_sensor.keys()):
            sensor = mag_data_by_sensor[sid]
            st = sensor['time']
            sx, sy, sz = sensor['x'], sensor['y'], sensor['z']

            # Sort by time
            order = np.argsort(st)
            st, sx, sy, sz = st[order], sx[order], sy[order], sz[order]

            # Interpolate for smooth arcs
            st, sx, sy, sz = _interpolate_rotation(st, sx, sy, sz)

            dot_size = max(2, 8 - len(mag_data_by_sensor))
            sc_last = axes[0].scatter(sx, sy, c=st, cmap='plasma', s=dot_size,
                                      alpha=0.6, norm=norm, edgecolors='none')
            axes[1].scatter(sx, sz, c=st, cmap='plasma', s=dot_size,
                           alpha=0.6, norm=norm, edgecolors='none')
            axes[2].scatter(sy, sz, c=st, cmap='plasma', s=dot_size,
                           alpha=0.6, norm=norm, edgecolors='none')
    else:
        # Single sensor or fallback: sort and interpolate
        order = np.argsort(t)
        ts, xs, ys, zs = t[order], x[order], y[order], z[order]
        ts, xs, ys, zs = _interpolate_rotation(ts, xs, ys, zs)

        sc_last = axes[0].scatter(xs, ys, c=ts, cmap='plasma', s=4, alpha=0.7, norm=norm)
        axes[1].scatter(xs, zs, c=ts, cmap='plasma', s=4, alpha=0.7, norm=norm)
        axes[2].scatter(ys, zs, c=ts, cmap='plasma', s=4, alpha=0.7, norm=norm)

    # Labels
    titles = ['XY Plane (Rotation View)', 'XZ Plane', 'YZ Plane']
    xlabels = ['Bx (nT)', 'Bx (nT)', 'By (nT)']
    ylabels = ['By (nT)', 'Bz (nT)', 'Bz (nT)']
    for ax, title, xl, yl in zip(axes, titles, xlabels, ylabels):
        ax.set_xlabel(xl, color='white')
        ax.set_ylabel(yl, color='white')
        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')

    # Colorbar
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    cbar = fig.colorbar(sc_last, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Mission Time', color='white', fontsize=10)
    cbar.ax.xaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_xticklabels(), color='white')
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(format_time_tplus))

    fig.suptitle(f'{MISSION_NAME} — Magnetometer Rotation Analysis', color='white', fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


# ==============================================================================
# SENSOR SUMMARY DASHBOARD
# ==============================================================================

def plot_sensor_summary(x, y, z, t, anomaly_results, save_path=None):
    """
    Summary dashboard with key statistics.

    Args:
        x, y, z: Magnetic field components
        t: Time values
        anomaly_results: Results from anomaly detection
        save_path: Path to save figure
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8), facecolor='black')

    magnitude = np.sqrt(x**2 + y**2 + z**2)

    # Panel 1: Magnitude histogram (top)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.hist(magnitude, bins=50, color='cyan', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax1.axvline(np.mean(magnitude), color='red', linestyle='--', label=f'Mean: {np.mean(magnitude):.1f}')
    ax1.axvline(np.median(magnitude), color='yellow', linestyle='--', label=f'Median: {np.median(magnitude):.1f}')
    ax1.set_xlabel('Magnitude (nT)', color='white')
    ax1.set_ylabel('Count', color='white')
    ax1.set_title('Magnitude Distribution', color='white', fontsize=12, fontweight='bold')
    ax1.legend(facecolor='#111', labelcolor='white')
    ax1.set_facecolor('black')
    ax1.tick_params(colors='white')

    # Panel 2: Time series with rolling stats (bottom)
    ax2 = fig.add_subplot(2, 1, 2)
    window = min(100, len(magnitude) // 10)
    if window > 1:
        rolling_mean = pd.Series(magnitude).rolling(window).mean()
        rolling_std = pd.Series(magnitude).rolling(window).std()

        ax2.scatter(t, magnitude, c='white', s=1, alpha=0.2, label='Raw')
        ax2.plot(t, rolling_mean, color='cyan', linewidth=2, label=f'Rolling Mean (n={window})')
        ax2.fill_between(t, rolling_mean - rolling_std, rolling_mean + rolling_std,
                        alpha=0.2, color='cyan', label='±1σ')

    add_launch_marker(ax2)
    add_gap_shading(ax2)
    ax2.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax2, 'x')
    ax2.set_ylabel('|B| (nT)', color='white')
    ax2.set_title('Magnitude Over Time', color='white', fontsize=12, fontweight='bold')
    ax2.legend(facecolor='#111', labelcolor='white', fontsize=8)
    ax2.set_facecolor('black')
    ax2.tick_params(colors='white')

    fig.suptitle(f'{MISSION_NAME} — Sensor Summary', color='white', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


# ==============================================================================
# INDIVIDUAL MAGNETOMETER ANOMALY DETECTION (CPU-BASED)
# ==============================================================================

def plot_anomaly_detection_per_magnetometer(mag_data_by_sensor, save_path_prefix=None):
    """
    Plot CPU-based anomaly detection for each magnetometer individually.

    Creates separate plots showing Isolation Forest and Z-Score anomalies
    for each magnetometer in the array.

    Args:
        mag_data_by_sensor: dict {sensor_id: {'time', 'x', 'y', 'z', 'magnitude', 'count'}}
        save_path_prefix: Path prefix for saving figures (will append sensor ID)

    Returns:
        List of saved file paths
    """
    from scipy import stats
    from sklearn.ensemble import IsolationForest
    from config import ML_CONFIG

    if not mag_data_by_sensor or len(mag_data_by_sensor) == 0:
        print("   No per-magnetometer data available")
        return []

    plt.style.use('dark_background')
    saved_paths = []

    z_threshold = ML_CONFIG.get('ANOMALY_Z_THRESHOLD', 3.5)
    contamination = ML_CONFIG.get('ISOLATION_FOREST_CONTAMINATION', 0.02)

    for sensor_id, data in sorted(mag_data_by_sensor.items()):
        print(f"   Processing {_sensor_label(sensor_id)}...")

        time_vals = data['time']
        x_vals = data['x']
        y_vals = data['y']
        z_vals = data['z']
        mag_vals = data['magnitude']

        # Skip if insufficient data
        if len(time_vals) < 10:
            print(f"      Skipping (insufficient data: {len(time_vals)} samples)")
            continue

        # Prepare features for anomaly detection
        X_spatial = np.column_stack([x_vals, y_vals, z_vals])

        # Run Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        iso_anomalies = iso_forest.fit_predict(X_spatial) == -1

        # Run Modified Z-Score detection (robust to outliers)
        med = np.median(mag_vals)
        mad = np.median(np.abs(mag_vals - med))
        if mad < 1e-9:
            mad = np.std(mag_vals)
        z_scores = 0.6745 * np.abs(mag_vals - med) / max(mad, 1e-9)
        z_anomalies = z_scores > z_threshold

        # Count anomalies
        n_iso = np.sum(iso_anomalies)
        n_z = np.sum(z_anomalies)

        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor='black')

        # ===== Top: Isolation Forest anomalies =====
        ax1 = axes[0]
        ax1.set_facecolor('black')

        # Plot normal points
        normal_mask = ~iso_anomalies
        ax1.scatter(time_vals[normal_mask], mag_vals[normal_mask],
                   c='white', s=3, alpha=0.3, label='Normal', zorder=1)

        # Plot anomalies
        if n_iso > 0:
            ax1.scatter(time_vals[iso_anomalies], mag_vals[iso_anomalies],
                       c='red', s=30, alpha=0.8, marker='o',
                       label=f'Anomalies ({n_iso}, {100*n_iso/len(time_vals):.1f}%)', zorder=2)

        add_launch_marker(ax1, label='Launch')
        add_gap_shading(ax1)
        ax1.set_xlim(left=_post_gap_start())
        set_time_axis_format(ax1, 'x')

        ax1.set_ylabel('Magnitude (nT)', color='white', fontsize=11)
        ax1.set_title(f'Isolation Forest Anomaly Detection',
                     color='white', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', facecolor='#111', labelcolor='white', fontsize=9)
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.tick_params(colors='white')

        # ===== Bottom: Z-Score anomalies =====
        ax2 = axes[1]
        ax2.set_facecolor('black')

        # Plot normal points
        normal_mask = ~z_anomalies
        ax2.scatter(time_vals[normal_mask], mag_vals[normal_mask],
                   c='white', s=3, alpha=0.3, label='Normal', zorder=1)

        # Plot anomalies
        if n_z > 0:
            ax2.scatter(time_vals[z_anomalies], mag_vals[z_anomalies],
                       c='yellow', s=50, alpha=0.9, marker='x', linewidths=2,
                       label=f'Anomalies (>{z_threshold}σ: {n_z}, {100*n_z/len(time_vals):.1f}%)', zorder=2)

        add_launch_marker(ax2, label='Launch')
        add_gap_shading(ax2)
        ax2.set_xlim(left=_post_gap_start())
        set_time_axis_format(ax2, 'x')

        ax2.set_ylabel('Magnitude (nT)', color='white', fontsize=11)
        ax2.set_title(f'Z-Score Anomaly Detection (threshold: {z_threshold}σ)',
                     color='white', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', facecolor='#111', labelcolor='white', fontsize=9)
        ax2.grid(True, alpha=0.3, color='gray')
        ax2.tick_params(colors='white')

        # Overall title
        fig.suptitle(f'CPU Anomaly Detection — {_sensor_label(sensor_id)} ({data["count"]} samples)',
                    color='white', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.985])

        if save_path_prefix:
            save_path = f"{save_path_prefix}_mag{sensor_id}.png"
            plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
            print(f"      Saved: {save_path}")
            saved_paths.append(save_path)
        plt.close()

    return saved_paths


# ==============================================================================
# NEURAL NETWORK RESULTS PLOT
# ==============================================================================

def plot_nn_results(X_test, y_test, y_pred, save_path=None, time_vals=None, r2=None):
    """
    Plot Neural Network actual vs predicted results.

    Args:
        X_test: Test features
        y_test: Actual values
        y_pred: NN predictions
        save_path: Path to save figure
        time_vals: Optional time values for x-axis (if None, uses sample index)
        r2: Optional R² score to display in title
    """
    if X_test is None or y_test is None or y_pred is None:
        print("   No NN results to plot")
        return None

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')

    r2_str = f' (R\u00b2 = {r2:.6f})' if r2 is not None else ''

    # Scatter: Actual vs Predicted
    ax1.scatter(y_test, y_pred.flatten(), c='orange', s=10, alpha=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    ax1.set_xlabel('Actual Magnitude (nT)', color='white')
    ax1.set_ylabel('Neural Net Predicted (nT)', color='white')
    ax1.set_title(f'Neural Network Prediction{r2_str}', color='white', fontsize=12, fontweight='bold')
    ax1.legend(facecolor='#111', labelcolor='white')
    ax1.set_facecolor('black')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='white')

    # Time series
    if time_vals is not None:
        tv = time_vals
    elif X_test.shape[1] >= 4:
        tv = X_test[:, 3]
    else:
        tv = np.arange(len(X_test))
    sort_idx = np.argsort(tv)

    ax2.scatter(tv[sort_idx], y_test[sort_idx], c='white', s=3, alpha=0.3, label='Actual')
    ax2.scatter(tv[sort_idx], y_pred.flatten()[sort_idx], c='orange', s=3, alpha=0.6, label='Neural Net')
    add_launch_marker(ax2)
    add_gap_shading(ax2)
    ax2.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax2, 'x')
    ax2.set_ylabel('Magnitude (nT)', color='white')
    ax2.set_title('Neural Network Prediction Over Time', color='white', fontsize=12, fontweight='bold')
    ax2.legend(facecolor='#111', labelcolor='white', fontsize=9)
    ax2.set_facecolor('black')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


# ==============================================================================
# PER-MAGNETOMETER ANOMALY TIMELINE (matches 06_anomaly_detection style)
# ==============================================================================

def plot_anomaly_per_mag_timeline(mag_data_by_sensor, anomaly_func,
                                  save_path_prefix=None):
    """
    Generate one anomaly-timeline plot per magnetometer (0, 1, 2, 3),
    matching the GHOST dark-theme style of plot_anomaly_detection.

    Each plot shows the magnitude signal with IF and modified Z-Score
    anomalies overlaid — one clean image per sensor.

    Args:
        mag_data_by_sensor: dict {sensor_id: {'time','x','y','z','magnitude','count'}}
        anomaly_func: callable(X, y) → anomaly_results dict
        save_path_prefix: Path prefix (will append _mag0.png, _mag1.png, …)

    Returns:
        List of saved file paths
    """
    if not mag_data_by_sensor:
        print("   No per-magnetometer data available")
        return []

    saved_paths = []
    sensor_colors_map = {sid: SENSOR_COLORS[i % len(SENSOR_COLORS)]
                         for i, sid in enumerate(sorted(mag_data_by_sensor.keys()))}

    for sensor_id, data in sorted(mag_data_by_sensor.items()):
        time_vals = np.asarray(data['time'])
        x_vals = np.asarray(data['x'])
        y_vals = np.asarray(data['y'])
        z_vals = np.asarray(data['z'])
        mag_vals = np.asarray(data['magnitude'])

        # Only analyse post-gap data to avoid boot-up transient flooding Z-score
        post_gap = time_vals > _post_gap_start()
        time_vals = time_vals[post_gap]
        x_vals = x_vals[post_gap]
        y_vals = y_vals[post_gap]
        z_vals = z_vals[post_gap]
        mag_vals = mag_vals[post_gap]
        n = len(time_vals)

        if n < 10:
            print(f"   Magnetometer #{sensor_id}: skipped (only {n} samples)")
            continue

        # Run anomaly detection through the main pipeline
        X_sensor = np.column_stack([x_vals, y_vals, z_vals, time_vals])
        results = anomaly_func(X_sensor, mag_vals, label=_sensor_label(sensor_id))

        iso_mask = results['isolation_forest']['anomalies']
        z_mask = results['z_score']['anomalies']
        lof_mask = results['lof']['anomalies']
        ens_mask = results['ensemble_anomalies']

        n_iso = int(np.sum(iso_mask)) if iso_mask is not None else 0
        n_z = int(np.sum(z_mask)) if z_mask is not None else 0
        n_lof = int(np.sum(lof_mask)) if lof_mask is not None else 0
        n_ens = int(np.sum(ens_mask)) if ens_mask is not None else 0

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')

        accent = sensor_colors_map.get(sensor_id, 'white')

        # Normal points
        ax.scatter(time_vals, mag_vals, c='white', s=1, alpha=0.3, label='Normal')

        # --- Individual methods as faded background context ---
        if iso_mask is not None and np.any(iso_mask):
            ax.scatter(time_vals[iso_mask], mag_vals[iso_mask],
                       c='red', s=8, alpha=0.25, marker='o',
                       label=f'IF ({n_iso})')

        # Suppress Z-Score display if it flags >15% (detecting flight dynamics, not anomalies)
        if z_mask is not None and np.any(z_mask) and n_z <= 0.15 * n:
            ax.scatter(time_vals[z_mask], mag_vals[z_mask],
                       c='yellow', s=20, alpha=0.3, marker='x', linewidths=1,
                       label=f'Z-Score ({n_z})')

        if lof_mask is not None and np.any(lof_mask):
            ax.scatter(time_vals[lof_mask], mag_vals[lof_mask],
                       c='#FF6B6B', s=10, alpha=0.25, marker='D',
                       label=f'LOF ({n_lof})')

        # --- ENSEMBLE: the hero layer ---
        if ens_mask is not None and np.any(ens_mask):
            ax.scatter(time_vals[ens_mask], mag_vals[ens_mask],
                       c='#00FFFF', s=120, alpha=0.95, marker='*',
                       edgecolors='white', linewidths=0.5, zorder=10,
                       label=f'★ ENSEMBLE ({n_ens})')

        add_launch_marker(ax)
        add_gap_shading(ax)

        ax.set_xlim(left=_post_gap_start())
        set_time_axis_format(ax, 'x')
        ax.set_ylabel('Magnitude (nT)', color='white', fontsize=12)
        ax.set_title(f'Anomaly Detection — {_sensor_label(sensor_id)}  '
                     f'({n:,} samples, Ensemble: {n_ens})',
                     color='#00FFFF', fontsize=14, fontweight='bold')
        ax.set_facecolor('black')
        ax.legend(loc='upper right', facecolor='#111', labelcolor='white', fontsize=9)
        ax.grid(True, alpha=0.3, color='gray')
        ax.tick_params(colors='white')

        plt.tight_layout()
        if save_path_prefix:
            save_path = f"{save_path_prefix}_mag{sensor_id}.png"
            plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
            print(f"      Saved: {save_path}")
            saved_paths.append(save_path)
        plt.close()

    return saved_paths


# ==============================================================================
# TEMPORAL PREDICTION-ERROR ANOMALY PLOT
# ==============================================================================

def plot_temporal_prediction_errors(temporal_results, save_path=None, mag_data_by_sensor=None):
    """
    Plot temporal forecasting: predicted vs actual magnitude over time,
    with prediction-error anomalies highlighted.

    Args:
        temporal_results: dict from run_temporal_forecasting()
        save_path: Path to save figure
        mag_data_by_sensor: Per-sensor data dict for multi-sensor display
    """
    t_all = temporal_results['t_all']
    y_all = temporal_results['y_all']
    pred_all = temporal_results['pred_all']
    errors = temporal_results['errors_all']
    anomaly_mask = temporal_results['anomaly_mask']
    threshold = temporal_results['threshold']
    mae = temporal_results['mae']
    r2 = temporal_results['r2']
    n_flagged = anomaly_mask.sum()

    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(14, 7), facecolor='black')

    sort_idx = np.argsort(t_all)
    t_s = t_all[sort_idx]
    y_s = y_all[sort_idx]
    p_s = pred_all[sort_idx]
    a_s = anomaly_mask[sort_idx]

    if mag_data_by_sensor and len(mag_data_by_sensor) > 1:
        # Per-sensor actual magnitude (colored dots)
        _plot_per_sensor(ax1, mag_data_by_sensor, s=6, alpha=0.4)
        # Prediction overlay (single color)
        ax1.scatter(t_s, p_s, color='#4ECDC4', s=3, alpha=0.4,
                    label=f'Predicted (MAE={mae:.1f} nT)')
        # Anomaly highlights on raw combined data
        if n_flagged > 0:
            ax1.scatter(t_s[a_s], y_s[a_s], color='red', s=30, alpha=0.8,
                        zorder=5, marker='o', facecolors='none', linewidths=1.5,
                        label=f'Prediction-Error Anomalies ({n_flagged})')
    else:
        unique_t = np.unique(t_s)
        if len(unique_t) < len(t_s) * 0.8:
            avg_y = np.array([y_s[t_s == t].mean() for t in unique_t])
            avg_p = np.array([p_s[t_s == t].mean() for t in unique_t])
            avg_a = np.array([a_s[t_s == t].any() for t in unique_t])
            t_plot, y_plot, p_plot, a_plot = unique_t, avg_y, avg_p, avg_a
        else:
            t_plot, y_plot, p_plot, a_plot = t_s, y_s, p_s, a_s

        ax1.scatter(t_plot, y_plot, color='white', s=3, alpha=0.3, label='Actual')
        ax1.scatter(t_plot, p_plot, color='#4ECDC4', s=3, alpha=0.5,
                    label=f'Predicted (MAE={mae:.1f} nT)')
        if n_flagged > 0:
            ax1.scatter(t_plot[a_plot], y_plot[a_plot], color='red', s=30, alpha=0.8,
                        zorder=5, marker='o', facecolors='none', linewidths=1.5,
                        label=f'Prediction-Error Anomalies ({n_flagged})')

    add_launch_marker(ax1)
    add_gap_shading(ax1)
    ax1.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax1, 'x')
    ax1.set_ylabel('Magnetic Field Magnitude (nT)', color='white', fontsize=12)
    ax1.set_xlabel('Mission Time', color='white', fontsize=12)
    ax1.set_title(f'Temporal Forecasting — Predicted vs Actual  (R² = {r2:.6f})',
                  color='white', fontsize=14, fontweight='bold')
    ax1.legend(facecolor='#111', labelcolor='white', loc='upper right', fontsize=9,
               markerscale=2)
    ax1.set_facecolor('black')
    ax1.grid(True, alpha=0.3, color='gray')
    ax1.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


# ==============================================================================
# TEMPORAL FORECASTING PLOTS (used by main.py)
# ==============================================================================

def plot_temporal_timeseries(t_test, y_test, pred_rf, pred_naive,
                             r2_rf, mae_rf, mae_naive, save_path=None):
    """
    Temporal RF forecast vs Actual over mission time.

    This is the genuine forecasting plot — uses sliding-window features
    (past values) to predict future magnitude, NOT the formula |B|=sqrt(X²+Y²+Z²).
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')

    sort_idx = np.argsort(t_test)
    t_s = t_test[sort_idx]
    y_s = y_test[sort_idx]
    rf_s = pred_rf[sort_idx]

    # Average per-sensor values at each timestamp to avoid oscillation
    unique_t = np.unique(t_s)
    if len(unique_t) < len(t_s) * 0.8:
        avg_y = np.array([y_s[t_s == t].mean() for t in unique_t])
        avg_rf = np.array([rf_s[t_s == t].mean() for t in unique_t])
        t_s, y_s, rf_s = unique_t, avg_y, avg_rf

    ax.scatter(t_s, y_s, color='white', s=3, alpha=0.3, label='Actual')
    ax.scatter(t_s, rf_s, color='#4ECDC4', s=3, alpha=0.6,
               label=f'Forecast (avg error: {mae_rf:.1f} nT)')

    add_launch_marker(ax)
    add_gap_shading(ax)

    ax.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax, 'x')
    ax.set_ylabel('Magnetic Field Magnitude (nT)', color='white', fontsize=12)
    ax.set_title(f'Temporal Forecast — Predicting Next Reading  (R\u00b2 = {r2_rf:.6f})',
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#111', labelcolor='white', loc='upper right', fontsize=9,
              markerscale=3)
    ax.set_facecolor('black')
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


def plot_temporal_scatter(y_test, pred_rf, r2_rf, save_path=None):
    """
    Scatter of actual vs temporal-forecast predicted.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')

    ax.scatter(y_test, pred_rf, c='cyan', s=10, alpha=0.5, label='Predictions')

    min_val = min(y_test.min(), pred_rf.min())
    max_val = max(y_test.max(), pred_rf.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')

    ax.set_xlabel('Actual Magnitude (nT)', color='white', fontsize=12)
    ax.set_ylabel('Forecasted Magnitude (nT)', color='white', fontsize=12)
    ax.set_title(f'Temporal Forecast — Actual vs Predicted (R\u00b2 = {r2_rf:.6f})',
                 color='white', fontsize=14, fontweight='bold')
    ax.set_facecolor('black')
    ax.legend(loc='upper left', facecolor='#111', labelcolor='white')
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


def plot_gru_results(t_test, y_test, pred_gru, r2_gru, mae_gru, save_path=None):
    """
    GRU neural network temporal forecast results — scatter + timeline.
    """
    if pred_gru is None:
        print("   No GRU results to plot")
        return None

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')

    # Scatter
    ax1.scatter(y_test, pred_gru, c='orange', s=10, alpha=0.5)
    min_val = min(y_test.min(), pred_gru.min())
    max_val = max(y_test.max(), pred_gru.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    ax1.set_xlabel('Actual Magnitude (nT)', color='white')
    ax1.set_ylabel('GRU Forecasted (nT)', color='white')
    ax1.set_title(f'GRU Forecast Scatter (R\u00b2={r2_gru:.6f})',
                  color='white', fontsize=12, fontweight='bold')
    ax1.legend(facecolor='#111', labelcolor='white')
    ax1.set_facecolor('black')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='white')

    # Timeline
    sort_idx = np.argsort(t_test)
    ax2.scatter(t_test[sort_idx], y_test[sort_idx], c='white', s=3, alpha=0.3, label='Actual')
    ax2.scatter(t_test[sort_idx], pred_gru[sort_idx], c='orange', s=3, alpha=0.6,
                label=f'GRU Forecast (MAE={mae_gru:.1f} nT)')
    add_launch_marker(ax2)
    add_gap_shading(ax2)
    ax2.set_xlim(left=_post_gap_start())
    set_time_axis_format(ax2, 'x')
    ax2.set_ylabel('Magnitude (nT)', color='white')
    ax2.set_title('GRU Forecast Over Time',
                  color='white', fontsize=12, fontweight='bold')
    ax2.legend(facecolor='#111', labelcolor='white', fontsize=9, markerscale=3)
    ax2.set_facecolor('black')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


# ==============================================================================
# RAW SENSOR PLOTS (for .dat post-flight analysis)
# ==============================================================================

def _get_mission_events():
    """Build mission event markers from config. Launch at T=0 always included."""
    events = [(0, 'Launch', LAUNCH_COLOR)]
    for t_event, label, color in MISSION_EVENTS:
        if t_event != 0:
            events.append((t_event, label, color))
    return events


def _flight_window(time_array):
    """
    Compute the valid flight data window from actual data timestamps.

    Returns (t_start, t_end, t_artifact_cutoff) where:
      t_start:  first timestamp after T+0 (skip pre-launch pad data)
      t_end:    last timestamp in the data
      t_artifact_cutoff: 95% of mission duration — data beyond this is
                         often corrupted by shutdown transients

    This is data-driven so it works for any flight, regardless of
    duration, gap timing, or reboot count.
    """
    t = np.asarray(time_array, dtype=float)
    t_min = float(t.min())
    t_max = float(t.max())

    # Flight window: start from T+0 (or data start if all post-launch)
    t_start = max(0.0, t_min)

    # Artifact cutoff: last 5% of data is often shutdown noise
    mission_dur = t_max - t_start
    t_artifact = t_start + mission_dur * 0.95 if mission_dur > 60 else t_max

    return t_start, t_max, t_artifact


def _add_mission_events(ax, y_frac=0.95, include_launch=True):
    """Add vertical lines and labels for key mission events."""
    for t_event, label, color in _get_mission_events():
        if not include_launch and t_event == 0:
            continue
        ax.axvline(x=t_event, color=color, linestyle='--', linewidth=1.2,
                   alpha=0.7, zorder=5)
        trans = ax.get_xaxis_transform()
        ax.text(t_event + 2, y_frac, label, color=color, fontsize=8,
                fontweight='bold', rotation=90, va='top', transform=trans,
                alpha=0.9)


def _smooth(arr, window=5):
    """Apply simple moving average smoothing."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    # 'same' keeps array length; edge effects are minor
    return np.convolve(arr, kernel, mode='same')


def plot_raw_magnetometer_overview(mag_data_by_sensor, save_path=None):
    """
    Improved magnetometer overview with 3 meaningful panels instead of 4
    redundant per-sensor subplots:

    Panel A: |B| magnitude overlay — all sensors on one axis for comparison
    Panel B: Bx/By/Bz for representative sensor — shows spin dynamics
    Panel C: Bx vs By phase plot — visualizes rocket rotation as ellipse

    Data is trimmed to valid flight window (computed from data),
    smoothed, and annotated with mission events.
    """
    n_sensors = len(mag_data_by_sensor)
    if n_sensors == 0:
        return None

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 16), facecolor='black')

    # Grid: top row = magnitude overlay (wide), bottom-left = Bx/By/Bz,
    #        bottom-right = phase plot
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25,
                          height_ratios=[1, 1.2])
    ax_mag = fig.add_subplot(gs[0, :])    # Full width — magnitude overlay
    ax_comp = fig.add_subplot(gs[1, 0])   # Components for one sensor
    ax_phase = fig.add_subplot(gs[1, 1])  # Phase plot

    sensor_colors_cycle = SENSOR_COLORS
    n_valid_total = 0

    # Compute flight window from actual data
    all_times = np.concatenate([np.asarray(d['time'], dtype=float)
                                for d in mag_data_by_sensor.values()])
    t_start, t_end, t_cutoff = _flight_window(all_times)

    # ── Panel A: Magnitude Overlay ────────────────────────────────────────
    for i, (sid, data) in enumerate(sorted(mag_data_by_sensor.items())):
        t = np.asarray(data['time'], dtype=float)
        mag = np.asarray(data['magnitude'], dtype=float)

        # Trim to valid flight window
        mask = (t >= t_start) & (t <= t_cutoff)
        t_clean = t[mask]
        mag_clean = mag[mask]

        if len(t_clean) < 10:
            continue
        n_valid_total += len(t_clean)

        mag_smooth = _smooth(mag_clean, window=5)
        clr = sensor_colors_cycle[i % len(sensor_colors_cycle)]
        ax_mag.plot(t_clean, mag_smooth, color=clr, linewidth=1.0,
                    alpha=0.85, label=f'{_sensor_label(sid)} ({len(t_clean):,})')

    ax_mag.set_title('Magnetic Field Magnitude — All Sensors (Smoothed, Flight Window)',
                     color='#00FFFF', fontsize=14, fontweight='bold')
    ax_mag.set_ylabel('|B| (nT)', color='white', fontsize=12)
    ax_mag.legend(facecolor='#111', labelcolor='white', fontsize=9,
                  loc='upper right', framealpha=0.8)
    ax_mag.set_facecolor('#0a0a0a')
    ax_mag.grid(True, alpha=0.15, color='gray')
    ax_mag.tick_params(colors='white')
    ax_mag.set_xlim(t_start, t_cutoff)
    set_time_axis_format(ax_mag, 'x')
    _add_mission_events(ax_mag, y_frac=0.92)
    add_gap_shading(ax_mag)

    # ── Panel B: Bx/By/Bz for one representative sensor ──────────────────
    # Pick the first sensor as representative (they all show same spin)
    rep_sid = sorted(mag_data_by_sensor.keys())[0]
    rep_data = mag_data_by_sensor[rep_sid]
    t_rep = np.asarray(rep_data['time'], dtype=float)
    bx_rep = np.asarray(rep_data['x'], dtype=float)
    by_rep = np.asarray(rep_data['y'], dtype=float)
    bz_rep = np.asarray(rep_data['z'], dtype=float)

    mask_rep = (t_rep >= t_start) & (t_rep <= t_cutoff)
    t_rep = t_rep[mask_rep]
    bx_rep = _smooth(bx_rep[mask_rep], window=5)
    by_rep = _smooth(by_rep[mask_rep], window=5)
    bz_rep = _smooth(bz_rep[mask_rep], window=5)

    ax_comp.plot(t_rep, bx_rep, color='cyan', linewidth=0.7, alpha=0.85, label='Bx')
    ax_comp.plot(t_rep, by_rep, color='magenta', linewidth=0.7, alpha=0.85, label='By')
    ax_comp.plot(t_rep, bz_rep, color='yellow', linewidth=0.7, alpha=0.85, label='Bz')

    ax_comp.set_title(f'Field Components — {_sensor_label(rep_sid)} (Spin Dynamics)',
                      color='white', fontsize=12, fontweight='bold')
    ax_comp.set_ylabel('Field (nT)', color='white', fontsize=11)
    ax_comp.set_xlabel('Mission Time', color='white', fontsize=11)
    ax_comp.legend(facecolor='#111', labelcolor='white', fontsize=9, loc='upper right')
    ax_comp.set_facecolor('#0a0a0a')
    ax_comp.grid(True, alpha=0.15, color='gray')
    ax_comp.tick_params(colors='white')
    ax_comp.set_xlim(t_start, t_cutoff)
    set_time_axis_format(ax_comp, 'x')
    _add_mission_events(ax_comp, y_frac=0.92)

    # ── Panel C: Bx vs By phase plot (rotation visualisation) ─────────────
    # Color by time to show evolution
    scatter = ax_phase.scatter(bx_rep, by_rep, c=t_rep, cmap='plasma',
                               s=2, alpha=0.6, zorder=2)
    cbar = plt.colorbar(scatter, ax=ax_phase, pad=0.02, shrink=0.85)
    cbar.set_label('Mission Time (s)', color='white', fontsize=10)
    cbar.ax.tick_params(colors='white', labelsize=8)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_time_tplus))

    ax_phase.set_title('Bx vs By Phase Plot — Rocket Spin Pattern',
                       color='white', fontsize=12, fontweight='bold')
    ax_phase.set_xlabel('Bx (nT)', color='white', fontsize=11)
    ax_phase.set_ylabel('By (nT)', color='white', fontsize=11)
    ax_phase.set_facecolor('#0a0a0a')
    ax_phase.grid(True, alpha=0.15, color='gray')
    ax_phase.tick_params(colors='white')
    ax_phase.set_aspect('equal', adjustable='datalim')

    # Draw crosshairs at origin
    ax_phase.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
    ax_phase.axvline(x=0, color='white', linewidth=0.5, alpha=0.3)

    # ── Suptitle ──────────────────────────────────────────────────────────
    n_total = sum(d['count'] for d in mag_data_by_sensor.values())
    n_removed = n_total - n_valid_total
    artifact_note = f'  |  {n_removed:,} shutdown-artifact samples removed' if n_removed > 0 else ''
    fig.suptitle(f'{MISSION_NAME} — Magnetometer Array Overview  '
                 f'({n_sensors} sensors, {n_valid_total:,} valid readings{artifact_note})',
                 color='white', fontsize=15, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


def plot_pressure_timeseries(pressure_df, save_path=None):
    """
    Improved pressure plot with two panels:
      Top:    Zoomed ascent phase (T-10 to T+60) — shows the interesting dynamics
      Bottom: Full mission overview with log scale — shows overall profile

    Removes shutdown artifacts, adds altitude correlation annotations.
    """
    if pressure_df is None or len(pressure_df) == 0:
        return None

    t = pressure_df['time_mission'].values.astype(float)
    p = pressure_df['pressure_mbar'].values.astype(float)

    # Remove shutdown artifacts (data-driven cutoff)
    _, _, t_cutoff = _flight_window(t)
    valid = t <= t_cutoff
    t_clean = t[valid]
    p_clean = p[valid]

    plt.style.use('dark_background')
    fig, (ax_zoom, ax_full) = plt.subplots(2, 1, figsize=(16, 10), facecolor='black',
                                            height_ratios=[1, 1])
    fig.subplots_adjust(hspace=0.35)

    # ── Top Panel: Zoomed Ascent (data-driven window) ─────────────────────
    zoom_start = max(t_clean.min(), -10)
    zoom_end = min(70, t_clean.max())
    ascent_mask = (t_clean >= zoom_start) & (t_clean <= zoom_end)
    t_asc = t_clean[ascent_mask]
    p_asc = p_clean[ascent_mask]

    ax_zoom.plot(t_asc, p_asc, color='#4ECDC4', linewidth=2.0, alpha=0.9, zorder=3)
    ax_zoom.scatter(t_asc, p_asc, c='#4ECDC4', s=12, alpha=0.5, zorder=4)
    ax_zoom.fill_between(t_asc, 0, p_asc, color='#4ECDC4', alpha=0.08)

    ax_zoom.set_title(f'Pressure During Ascent (T{int(zoom_start):+d}s to T+{int(zoom_end)}s) — Dynamic Range',
                      color='#4ECDC4', fontsize=13, fontweight='bold')
    ax_zoom.set_ylabel('Pressure (mbar)', color='white', fontsize=11)
    ax_zoom.set_facecolor('#0a0a0a')
    ax_zoom.grid(True, alpha=0.2, color='gray')
    ax_zoom.tick_params(colors='white')
    ax_zoom.set_xlim(zoom_start, zoom_end)
    set_time_axis_format(ax_zoom, 'x')
    add_launch_marker(ax_zoom)

    # Altitude correlation annotations at key pressure levels
    altitude_markers = [
        (1013, '~0 km (sea level)', '#96CEB4'),
        (500,  '~5.5 km', '#FFE66D'),
        (200,  '~12 km (tropopause)', '#FF6B6B'),
        (50,   '~21 km (stratosphere)', '#45B7D1'),
    ]
    for p_level, alt_label, clr in altitude_markers:
        if p_asc.min() <= p_level <= p_asc.max():
            ax_zoom.axhline(y=p_level, color=clr, linestyle=':', linewidth=1,
                           alpha=0.6)
            ax_zoom.text(zoom_end - 5, p_level, alt_label, color=clr, fontsize=8,
                        va='center', fontweight='bold', alpha=0.9)

    # Burnout marker (only if defined in mission timeline)
    burnout_time = MISSION_TIMELINE.get('T_BURN_END')
    if burnout_time is not None and zoom_start <= burnout_time <= zoom_end:
        ax_zoom.axvline(x=burnout_time, color='#FF6B6B', linestyle='--', linewidth=1.2,
                        alpha=0.7)
        ax_zoom.text(burnout_time + 1.5, p_asc.max() * 0.85, 'Burnout', color='#FF6B6B',
                    fontsize=9, fontweight='bold')

    # ── Bottom Panel: Full Mission Log Scale ──────────────────────────────
    ax_full.semilogy(t_clean, p_clean, color='#4ECDC4', linewidth=1.2,
                     alpha=0.9, zorder=3)
    ax_full.scatter(t_clean, p_clean, c='#4ECDC4', s=3, alpha=0.3, zorder=4)

    ax_full.set_title(f'Pressure — Full Mission Overview (Log Scale)  |  '
                      f'{len(t_clean)} valid readings',
                      color='white', fontsize=13, fontweight='bold')
    ax_full.set_ylabel('Pressure (mbar, log)', color='white', fontsize=11)
    ax_full.set_xlabel('Mission Time', color='white', fontsize=11)
    ax_full.set_facecolor('#0a0a0a')
    ax_full.grid(True, alpha=0.2, color='gray', which='both')
    ax_full.tick_params(colors='white')
    set_time_axis_format(ax_full, 'x')
    add_launch_marker(ax_full)
    add_gap_shading(ax_full)
    _add_mission_events(ax_full, y_frac=0.92, include_launch=False)

    # Annotate sensor floor
    p_floor = p_clean[t_clean > 60].min() if np.any(t_clean > 60) else p_clean.min()
    ax_full.axhline(y=p_floor, color='#FF6B6B', linestyle=':', linewidth=1,
                    alpha=0.5)
    ax_full.text(t_clean.max() * 0.7, p_floor * 1.3,
                f'Sensor floor: {p_floor:.1f} mbar (near-vacuum)',
                color='#FF6B6B', fontsize=9, fontweight='bold', alpha=0.8)

    n_removed = np.sum(~valid)
    if n_removed > 0:
        fig.text(0.99, 0.01, f'{n_removed} shutdown-artifact samples removed',
                color='#666', fontsize=8, ha='right', va='bottom')

    fig.suptitle(f'{MISSION_NAME} — Pressure Sensor (MPRLS)',
                 color='white', fontsize=15, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


def plot_temperature_timeseries(thermo_df, save_path=None):
    """
    Improved thermocouple plot:
    - Clips shutdown artifacts (>100°C after T+270)
    - Zooms y-axis to actual data range with padding
    - Adds annotation when artifacts were removed
    - Shows data quality assessment
    """
    if thermo_df is None or len(thermo_df) == 0:
        return None

    t = thermo_df['time_mission'].values.astype(float)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 6), facecolor='black')

    n_artifacts_removed = 0
    all_clean_temps = []
    channels_plotted = []

    # Compute data-driven artifact cutoff
    _, _, t_cutoff = _flight_window(t)

    for col, label, color in [('tc0_C', 'TC0 — Payload', '#FF6B6B'),
                               ('tc1_C', 'TC1 — External', '#45B7D1')]:
        if col not in thermo_df.columns:
            continue
        raw = thermo_df[col].values.astype(float)
        clean = raw.copy()

        # ── Artifact removal ──────────────────────────────────────────
        # 1) Trim data after cutoff (shutdown transients — consistent
        #    with magnetometer and pressure trimming)
        shutdown_mask = t > t_cutoff
        n_artifacts_removed += int(np.sum(shutdown_mask))
        clean[shutdown_mask] = np.nan

        # 2) Reboot zeros: 0°C values are initialization defaults, not
        #    real measurements (payload never reaches 0°C at Wallops)
        zero_mask = np.abs(clean) < 0.5
        n_artifacts_removed += int(np.sum(zero_mask))
        clean[zero_mask] = np.nan

        valid = clean[~np.isnan(clean)]
        if len(valid) == 0:
            continue

        all_clean_temps.extend(valid)
        ax.plot(t, clean, color=color, linewidth=1.5, alpha=0.9, label=label)
        channels_plotted.append((label, valid))

    # Zoom y-axis to actual data range with modest padding
    if all_clean_temps:
        temp_min = np.nanmin(all_clean_temps)
        temp_max = np.nanmax(all_clean_temps)
        padding = max(2, (temp_max - temp_min) * 0.15)
        ax.set_ylim(temp_min - padding, temp_max + padding)

    ax.set_facecolor('#0a0a0a')
    ax.grid(True, alpha=0.15, color='gray')
    ax.tick_params(colors='white')
    add_launch_marker(ax)
    add_gap_shading(ax)
    set_time_axis_format(ax, 'x')
    ax.set_ylabel('Temperature (°C)', color='white', fontsize=12)
    ax.legend(facecolor='#111', labelcolor='white', fontsize=10, loc='upper right')
    _add_mission_events(ax, y_frac=0.92, include_launch=False)

    # Data quality assessment box
    quality_lines = []
    for ch_name, ch_data in channels_plotted:
        rng = ch_data.max() - ch_data.min()
        quality_lines.append(f'{ch_name}: {ch_data.min():.1f}–{ch_data.max():.1f}°C '
                            f'(Δ={rng:.1f}°C)')
    if quality_lines:
        quality_text = '\n'.join(quality_lines)
        ax.text(0.02, 0.95, quality_text, transform=ax.transAxes,
                fontsize=9, color='#aaa', va='top', ha='left',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a',
                         edgecolor='#333', alpha=0.9))

    n_valid = len(thermo_df) - n_artifacts_removed
    artifact_note = ''
    if n_artifacts_removed > 0:
        artifact_note = f'  |  {n_artifacts_removed} artifacts removed (reboot zeros + shutdown)'
        ax.text(0.99, 0.01,
                f'⚠ {n_artifacts_removed} artifacts removed '
                f'(reboot zeros + shutdown spikes)',
                transform=ax.transAxes, fontsize=8, color='#FF6B6B',
                ha='right', va='bottom', alpha=0.8)

    ax.set_title(f'Payload Thermal Environment (MAX31856) — '
                 f'{n_valid} valid readings{artifact_note}',
                 color='white', fontsize=13, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


def plot_adc_timeseries(adc_df, save_path=None):
    """
    Improved ADC plot:
    - Detects flat (constant) channels vs dynamic channels
    - Only plots channels with actual variation
    - If all channels are flat, shows informative summary instead
    - Labels what the channels might be connected to
    """
    if adc_df is None or len(adc_df) == 0:
        return None

    t = adc_df['time_mission'].values.astype(float)
    colors = {'a0': '#FF6B6B', 'a1': '#4ECDC4', 'a2': '#45B7D1', 'a3': '#FFE66D'}

    # Classify channels as dynamic vs flat
    dynamic_channels = []
    flat_channels = []
    FLAT_THRESHOLD = 0.01  # < 10mV variation = flat/unconnected

    for ch in ['a0', 'a1', 'a2', 'a3']:
        if ch not in adc_df.columns:
            continue
        vals = adc_df[ch].values.astype(float)
        val_range = np.nanmax(vals) - np.nanmin(vals)
        mean_val = np.nanmean(vals)
        if val_range > FLAT_THRESHOLD:
            dynamic_channels.append((ch, vals, val_range, mean_val))
        else:
            flat_channels.append((ch, mean_val))

    plt.style.use('dark_background')

    if not dynamic_channels:
        # All channels are flat — show informative summary panel
        fig, ax = plt.subplots(figsize=(16, 5), facecolor='black')
        ax.set_facecolor('#0a0a0a')

        summary_lines = [
            'ADC (ADS1115) — All Channels Constant',
            '',
            'All 4 channels show no variation (ΔV < 10 mV).',
            'This indicates the channels are either:',
            '  • Not connected to sensors',
            '  • Measuring a rail/reference voltage',
            '  • Saturated at a fixed level',
            '',
        ]
        for ch, mean_v in flat_channels:
            summary_lines.append(f'  {ch.upper()}: {mean_v:.3f} V (constant)')

        summary_lines.append(f'\n  {len(adc_df)} readings from T{t.min():+.0f}s to T{t.max():+.0f}s')
        summary_lines.append('  → Graph omitted (no informative data)')

        text = '\n'.join(summary_lines)
        ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=12,
                color='#999', va='center', ha='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1.0', facecolor='#111',
                         edgecolor='#333', alpha=0.95))
        ax.set_title(f'ADC (ADS1115) — {len(adc_df)} readings  |  No dynamic channels detected',
                     color='#666', fontsize=13, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    else:
        # Plot only dynamic channels
        fig, ax = plt.subplots(figsize=(16, 5), facecolor='black')

        for ch, vals, val_range, mean_val in dynamic_channels:
            ax.plot(t, vals, color=colors[ch], linewidth=1.2, alpha=0.9,
                    label=f'{ch.upper()} (range: {val_range:.3f} V)')

        ax.set_facecolor('#0a0a0a')
        ax.grid(True, alpha=0.15, color='gray')
        ax.tick_params(colors='white')
        ax.set_ylabel('Voltage (V)', color='white', fontsize=11)
        add_launch_marker(ax)
        add_gap_shading(ax)
        set_time_axis_format(ax, 'x')
        ax.legend(facecolor='#111', labelcolor='white', fontsize=10, loc='upper right')

        # Note flat channels
        if flat_channels:
            flat_note = ', '.join(f'{ch.upper()}={v:.3f}V' for ch, v in flat_channels)
            ax.text(0.02, 0.05, f'Flat/disconnected: {flat_note}',
                    transform=ax.transAxes, fontsize=9, color='#666',
                    va='bottom', ha='left')

        n_dynamic = len(dynamic_channels)
        n_flat = len(flat_channels)
        ax.set_title(f'ADC (ADS1115) — {len(adc_df)} readings  |  '
                     f'{n_dynamic} dynamic, {n_flat} flat channels',
                     color='white', fontsize=13, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig


def plot_flight_events_timeline(events_df, save_path=None):
    """
    Improved flight events timeline:
    - Filters 1000+ events down to ~30-50 key events
    - Removes repetitive polling messages
    - Color-codes by category (System, Sensor, Motor, Milestone, Warning)
    - Uses clean dot markers instead of overlapping text
    - Adds aggregated summary of filtered events
    """
    if events_df is None or len(events_df) == 0:
        return None

    t_all = events_df['time_mission'].values
    labels_all = events_df['event'].values
    total_events = len(events_df)

    # ── Event filtering: keep only significant events ─────────────────────
    # Patterns that indicate repetitive/routine messages (ALWAYS skip)
    # These take absolute priority — even if event also contains a KEEP word
    SKIP_PATTERNS_LOWER = [
        'time to launch',
        'waiting for',
        'waiting to retract',
        'tdl: measurementoutput',
        'measurementoutput',
        'polling',
        'heartbeat',
        'loop iteration',
        'reading sensor',
        'adc reading',
        'pressure reading',
        'temperature reading',
        'magneto reading',
        'sample ',
        'data point',
        'tdl: none',
        'checking if',
    ]

    filtered_events = []
    skipped_categories = {}  # category → count

    for i in range(total_events):
        event_text = str(labels_all[i])
        event_time = t_all[i]
        event_lower = event_text.lower()

        # SKIP patterns take absolute priority
        is_skip = False
        for pattern in SKIP_PATTERNS_LOWER:
            if pattern in event_lower:
                is_skip = True
                key = pattern.strip()
                skipped_categories[key] = skipped_categories.get(key, 0) + 1
                break

        if is_skip:
            continue

        filtered_events.append((event_time, event_text))

    # Deduplicate events at the same timestamp (keep first)
    seen_times = {}
    deduped_events = []
    for evt_time, evt_text in filtered_events:
        key = (int(evt_time), evt_text[:30])
        if key not in seen_times:
            seen_times[key] = True
            deduped_events.append((evt_time, evt_text))

    # ── Event categorization ──────────────────────────────────────────────
    def _categorize(text):
        text_lower = text.lower()
        if any(w in text_lower for w in ['error', 'fail', 'warning', 'anomal']):
            return ('Warning/Error', '#FF4444', '×')
        if any(w in text_lower for w in ['launch', 'liftoff', 'mission', 'apogee',
                                          'deploy', 'skirt', 'burnout']):
            return ('Milestone', LAUNCH_COLOR, '★')
        if any(w in text_lower for w in ['motor', 'stepper', 'relay', 'gpio',
                                          'camera', 'power']):
            return ('Actuator', '#FF9F43', '▸')
        if any(w in text_lower for w in ['sensor', 'magneto', 'pressure',
                                          'thermo', 'adc', 'tdl', 'particle',
                                          'calibrat']):
            return ('Sensor', '#4ECDC4', '●')
        return ('System', '#45B7D1', '○')

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(18, 8), facecolor='black')
    ax.set_facecolor('#0a0a0a')

    # Plot filtered events
    category_handles = {}
    n_events = len(deduped_events)

    # Assign vertical positions to avoid overlap
    positions = np.linspace(0.15, 0.85, min(n_events, 60))

    for i, (evt_time, evt_text) in enumerate(deduped_events[:60]):  # Cap at 60
        cat_name, cat_color, cat_marker = _categorize(evt_text)

        y_pos = positions[i] if i < len(positions) else 0.5

        # Marker
        ax.plot(evt_time, y_pos, marker='o', markersize=6, color=cat_color,
                alpha=0.9, zorder=5)

        # Label (truncated)
        display_text = evt_text[:55] + '...' if len(evt_text) > 55 else evt_text
        ax.text(evt_time + 2, y_pos, display_text, fontsize=7, color=cat_color,
                va='center', ha='left', alpha=0.85,
                fontfamily='monospace')

        # Track for legend
        if cat_name not in category_handles:
            category_handles[cat_name] = plt.Line2D(
                [0], [0], marker='o', color='w', markerfacecolor=cat_color,
                markersize=8, linestyle='None', label=cat_name)

    # Legend
    if category_handles:
        ax.legend(handles=list(category_handles.values()),
                 facecolor='#111', labelcolor='white', fontsize=9,
                 loc='upper right', framealpha=0.9)

    # Axis config
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(colors='white')
    set_time_axis_format(ax, 'x')
    add_launch_marker(ax)
    add_gap_shading(ax)

    # Aggregated summary of filtered events
    n_filtered = total_events - len(deduped_events)
    if n_filtered > 0:
        # Top 5 skipped categories
        top_skipped = sorted(skipped_categories.items(), key=lambda x: -x[1])[:5]
        skip_summary = ', '.join(f'"{k}" ×{v}' for k, v in top_skipped)
        ax.text(0.01, 0.02,
                f'Filtered: {n_filtered:,} repetitive events  |  Top: {skip_summary}',
                transform=ax.transAxes, fontsize=8, color='#555',
                va='bottom', ha='left')

    ax.set_title(f'Flight Events Timeline — {len(deduped_events)} key events '
                 f'(from {total_events:,} total, {n_filtered:,} repetitive filtered)',
                 color='white', fontsize=13, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.close()
    return fig