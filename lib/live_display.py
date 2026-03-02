#!/usr/bin/env python3
"""
RockSat-X 2026 — Live Anomaly Detection Display
=================================================
Real-time visualization for the ML flight pipeline.
Opens a separate matplotlib window showing sensor data,
ML predictions, and anomaly detection results.

Two modes:
  LiveAnomalyDisplay  — real-time from sensor loop (test_main.py)
  replay_simulation() — animated batch replay (sim_flight.py)

Usage:
    # Real-time (test_main.py / flight):
    display = LiveAnomalyDisplay()
    display.start()
    display.update(mission_time=12.3, magnitude=48000, rf_pred=47990, ...)
    display.stop()

    # Simulation replay (sim_flight.py):
    replay_simulation(t, magnitude, rf_pred, nn_pred, z_scores, roc,
                      is_z, is_roc, is_ensemble, speed=10)
"""

import time
import numpy as np
import multiprocessing as mp
from collections import deque

from config import MISSION_NAME

# ── Display config ──────────────────────────────────────────────
WINDOW_SECONDS = 60       # Rolling window width (seconds of data visible)
MAX_POINTS = 2700         # 45 Hz * 60 s
UPDATE_MS = 100           # Plot refresh at 10 Hz
FIGSIZE = (14, 8)

# GHOST dark theme (matches lib/visualization.py)
BG       = '#0a0a0a'
PANEL_BG = '#111111'
TEXT     = '#ffffff'
GRID     = '#2a2a2a'
C_MAG    = '#4ECDC4'      # Measured magnitude (teal)
C_RF     = '#FF6B6B'      # RF prediction (coral)
C_NN     = '#FFE66D'      # NN prediction (gold)
C_ZSCORE = '#45B7D1'      # Z-score (sky blue)
C_ROC    = '#96CEB4'      # Rate of change (sage)
C_ANOM   = '#FF0000'      # Anomaly markers (red)
C_THRESH = '#FF4444'      # Threshold lines (bright red)
C_LAUNCH = '#00FF00'      # Launch marker (lime)


def _format_tplus(x, pos=None):
    """Format axis tick as T+/T- flight standard."""
    if abs(x) < 0.5:
        return 'T0'
    return f'T{x:+.0f}'


def _setup_axes(fig, ax_list):
    """Apply GHOST dark theme to axes."""
    for ax in ax_list:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.grid(True, color=GRID, alpha=0.5, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID)


# =============================================================================
# 1. LIVE DISPLAY (separate process — for test_main.py on Pi or Mac bench)
# =============================================================================

class LiveAnomalyDisplay:
    """
    Non-blocking real-time anomaly display.
    Runs matplotlib in a child process; data flows via Queue.
    The sensor loop is never blocked by rendering.
    """

    def __init__(self, z_thresh=3.5, roc_thresh=200.0):
        self._queue = mp.Queue(maxsize=5000)
        self._proc = None
        self._running = mp.Value('b', False)
        self._z_thresh = z_thresh
        self._roc_thresh = roc_thresh

    def start(self):
        """Open the visualization window (separate process)."""
        self._running.value = True
        self._proc = mp.Process(
            target=_live_process_main,
            args=(self._queue, self._running, self._z_thresh, self._roc_thresh),
            daemon=True,
        )
        self._proc.start()

    def update(self, *, mission_time, magnitude, rf_pred=None, nn_pred=None,
               z_score=0.0, roc=0.0, is_ensemble=False, cluster_id=None,
               sensor_id=0, reading_number=0):
        """Push one reading to the display (non-blocking, drops if queue full)."""
        try:
            self._queue.put_nowait({
                't': float(mission_time),
                'mag': float(magnitude),
                'rf': float(rf_pred) if rf_pred is not None else float('nan'),
                'nn': float(nn_pred) if nn_pred is not None else float('nan'),
                'z': float(z_score),
                'roc': float(roc),
                'ens': int(bool(is_ensemble)),
                'cl': cluster_id,
                'sid': int(sensor_id),
                'n': int(reading_number),
            })
        except Exception:
            pass  # Queue full — display can't keep up, drop silently

    def stop(self):
        """Close the visualization window."""
        self._running.value = False
        if self._proc is not None:
            self._proc.join(timeout=3)
            if self._proc.is_alive():
                self._proc.terminate()


def _live_process_main(queue, running, z_thresh, roc_thresh):
    """Child process entry: runs matplotlib event loop with FuncAnimation."""
    try:
        import matplotlib
        matplotlib.use('TkAgg')
    except Exception:
        try:
            import matplotlib
            matplotlib.use('Agg')
            print("  [live_display] No display backend available — skipping visualization")
            return
        except Exception:
            return

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.ticker import FuncFormatter

    # Data ring buffers
    buf = {k: deque(maxlen=MAX_POINTS) for k in ['t', 'mag', 'rf', 'nn', 'z', 'roc', 'ens']}
    stats = {'n': 0, 'anomalies': 0, 'start': time.time()}

    # ── Build figure ─────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=FIGSIZE, facecolor=BG,
        gridspec_kw={'height_ratios': [3, 1.5, 1.5], 'hspace': 0.3})

    fig.canvas.manager.set_window_title(f'{MISSION_NAME} — Live Anomaly Detection')
    title_text = fig.suptitle(
        f'{MISSION_NAME} — Live Anomaly Detection\nWaiting for sensor data...',
        color=TEXT, fontsize=13, fontweight='bold', y=0.98)

    _setup_axes(fig, [ax1, ax2, ax3])
    fmt = FuncFormatter(_format_tplus)

    # ── Panel 1: Magnitude + ML predictions ──────────────────────
    ln_mag, = ax1.plot([], [], color=C_MAG, linewidth=1.2, label='Measured |B|')
    ln_rf,  = ax1.plot([], [], color=C_RF, linewidth=1.0, alpha=0.8, label='RF Prediction')
    ln_nn,  = ax1.plot([], [], color=C_NN, linewidth=1.0, alpha=0.8, label='NN Prediction')
    sc_anom = ax1.scatter([], [], c=C_ANOM, s=50, zorder=5, marker='x',
                          linewidths=2, label='Anomaly (Ensemble)')
    ax1.set_ylabel('Magnitude (nT)', fontsize=10)
    ax1.legend(loc='upper right', fontsize=8, facecolor=PANEL_BG,
               edgecolor=GRID, labelcolor=TEXT)
    ax1.xaxis.set_major_formatter(fmt)

    # ── Panel 2: Z-Score ─────────────────────────────────────────
    ln_z, = ax2.plot([], [], color=C_ZSCORE, linewidth=1.0, label='Modified Z-Score')
    ax2.axhline(y=z_thresh, color=C_THRESH, linestyle='--', linewidth=1, alpha=0.7,
                label=f'Threshold ({z_thresh})')
    ax2.set_ylabel('Z-Score', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8, facecolor=PANEL_BG,
               edgecolor=GRID, labelcolor=TEXT)
    ax2.xaxis.set_major_formatter(fmt)

    # ── Panel 3: Rate of Change ──────────────────────────────────
    ln_roc, = ax3.plot([], [], color=C_ROC, linewidth=1.0, label='Rate of Change')
    ax3.axhline(y=roc_thresh, color=C_THRESH, linestyle='--', linewidth=1, alpha=0.7,
                label=f'Threshold ({roc_thresh})')
    ax3.set_ylabel('|ΔB| (nT)', fontsize=10)
    ax3.set_xlabel('Mission Time', fontsize=10)
    ax3.legend(loc='upper right', fontsize=8, facecolor=PANEL_BG,
               edgecolor=GRID, labelcolor=TEXT)
    ax3.xaxis.set_major_formatter(fmt)

    # Launch marker on all panels
    for ax in (ax1, ax2, ax3):
        ax.axvline(x=0, color=C_LAUNCH, linewidth=1, alpha=0.4, linestyle=':')

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # ── Animation callback ───────────────────────────────────────
    def animate(_frame):
        # Drain queue (up to 300 points per frame to stay responsive)
        count = 0
        while not queue.empty() and count < 300:
            try:
                d = queue.get_nowait()
                buf['t'].append(d['t'])
                buf['mag'].append(d['mag'])
                buf['rf'].append(d['rf'])
                buf['nn'].append(d['nn'])
                buf['z'].append(d['z'])
                buf['roc'].append(d['roc'])
                buf['ens'].append(d['ens'])
                stats['n'] = d['n']
                if d['ens']:
                    stats['anomalies'] += 1
                count += 1
            except Exception:
                break

        if len(buf['t']) < 2:
            return

        t_arr = np.array(buf['t'])
        mag_arr = np.array(buf['mag'])
        t_now = t_arr[-1]
        t_win = t_now - WINDOW_SECONDS

        # Panel 1: magnitude + predictions
        ln_mag.set_data(t_arr, mag_arr)

        rf_arr = np.array(buf['rf'], dtype=float)
        nn_arr = np.array(buf['nn'], dtype=float)
        m_rf = ~np.isnan(rf_arr)
        m_nn = ~np.isnan(nn_arr)
        if m_rf.any():
            ln_rf.set_data(t_arr[m_rf], rf_arr[m_rf])
        if m_nn.any():
            ln_nn.set_data(t_arr[m_nn], nn_arr[m_nn])

        ens_arr = np.array(buf['ens'])
        a_mask = ens_arr > 0
        if a_mask.any():
            sc_anom.set_offsets(np.column_stack([t_arr[a_mask], mag_arr[a_mask]]))
        else:
            sc_anom.set_offsets(np.empty((0, 2)))

        # Panel 2: z-score
        z_arr = np.array(buf['z'])
        ln_z.set_data(t_arr, z_arr)

        # Panel 3: rate of change
        roc_arr = np.array(buf['roc'])
        ln_roc.set_data(t_arr, roc_arr)

        # X limits (rolling window)
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(t_win, t_now + 2)

        # Y limits (auto-scale to visible window)
        vis = t_arr >= t_win
        if vis.any():
            vm = mag_arr[vis]
            if len(vm) > 0:
                pad = max(50, (vm.max() - vm.min()) * 0.1)
                ax1.set_ylim(vm.min() - pad, vm.max() + pad)

            vz = z_arr[vis]
            if len(vz) > 0:
                ax2.set_ylim(0, max(z_thresh * 1.5, vz.max() * 1.3))

            vr = roc_arr[vis]
            if len(vr) > 0:
                ax3.set_ylim(0, max(roc_thresh * 1.5, vr.max() * 1.3))

        # Status title
        elapsed = time.time() - stats['start']
        hz = stats['n'] / max(elapsed, 0.01)
        mt_str = f'T{t_now:+.1f}s' if abs(t_now) >= 0.5 else 'T0'
        title_text.set_text(
            f'{MISSION_NAME} — Live Anomaly Detection\n'
            f'{mt_str}  |  {stats["n"]:,} readings  |  '
            f'{hz:.1f} Hz  |  {stats["anomalies"]} anomalies')

        if not running.value:
            plt.close(fig)

    anim = FuncAnimation(fig, animate, interval=UPDATE_MS, cache_frame_data=False)
    plt.show()


# =============================================================================
# 2. SIMULATION REPLAY (main process — for sim_flight.py)
# =============================================================================

def replay_simulation(t, magnitude, rf_pred, nn_pred, z_scores, roc,
                      is_z, is_roc, is_ensemble, clusters=None,
                      speed=10.0, z_thresh=3.5, roc_thresh=200.0):
    """
    Animated replay of batch simulation results.
    Opens a window that progressively reveals data as if running in real-time.

    Args:
        t: mission time array
        magnitude: measured magnitude array
        rf_pred: RF prediction array
        nn_pred: NN prediction array (may contain NaN)
        z_scores: z-score array
        roc: rate-of-change array
        is_z: z-score flag array (0/1)
        is_roc: roc flag array (0/1)
        is_ensemble: ensemble flag array (0/1)
        clusters: cluster assignment array (optional)
        speed: replay speed multiplier (10 = 10x real time)
        z_thresh: z-score threshold for display
        roc_thresh: rate-of-change threshold for display
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.ticker import FuncFormatter

    total = len(t)
    cursor = [0]  # mutable for closure

    # Points to advance per animation frame
    # At speed=10, 45 Hz data, 10 Hz display → 45 points/frame
    points_per_frame = max(1, int(speed * 45.0 * (UPDATE_MS / 1000.0)))

    # ── Build figure ─────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=FIGSIZE, facecolor=BG,
        gridspec_kw={'height_ratios': [3, 1.5, 1.5], 'hspace': 0.3})

    fig.canvas.manager.set_window_title(f'{MISSION_NAME} — Simulation Replay')
    title_text = fig.suptitle(
        f'{MISSION_NAME} — Simulation Replay ({speed:.0f}x speed)\nStarting...',
        color=TEXT, fontsize=13, fontweight='bold', y=0.98)

    _setup_axes(fig, [ax1, ax2, ax3])
    fmt = FuncFormatter(_format_tplus)

    # Panel 1: Magnitude + ML predictions
    ln_mag, = ax1.plot([], [], color=C_MAG, linewidth=1.2, label='Measured |B|')
    ln_rf,  = ax1.plot([], [], color=C_RF, linewidth=1.0, alpha=0.8, label='RF Prediction')
    ln_nn,  = ax1.plot([], [], color=C_NN, linewidth=1.0, alpha=0.8, label='NN Prediction')
    sc_anom = ax1.scatter([], [], c=C_ANOM, s=50, zorder=5, marker='x',
                          linewidths=2, label='Anomaly (Ensemble)')
    ax1.set_ylabel('Magnitude (nT)', fontsize=10)
    ax1.legend(loc='upper right', fontsize=8, facecolor=PANEL_BG,
               edgecolor=GRID, labelcolor=TEXT)
    ax1.xaxis.set_major_formatter(fmt)

    # Panel 2: Z-Score
    ln_z, = ax2.plot([], [], color=C_ZSCORE, linewidth=1.0, label='Modified Z-Score')
    ax2.axhline(y=z_thresh, color=C_THRESH, linestyle='--', linewidth=1, alpha=0.7,
                label=f'Threshold ({z_thresh})')
    ax2.set_ylabel('Z-Score', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8, facecolor=PANEL_BG,
               edgecolor=GRID, labelcolor=TEXT)
    ax2.xaxis.set_major_formatter(fmt)

    # Panel 3: Rate of Change
    ln_roc, = ax3.plot([], [], color=C_ROC, linewidth=1.0, label='Rate of Change')
    ax3.axhline(y=roc_thresh, color=C_THRESH, linestyle='--', linewidth=1, alpha=0.7,
                label=f'Threshold ({roc_thresh})')
    ax3.set_ylabel('|ΔB| (nT)', fontsize=10)
    ax3.set_xlabel('Mission Time', fontsize=10)
    ax3.legend(loc='upper right', fontsize=8, facecolor=PANEL_BG,
               edgecolor=GRID, labelcolor=TEXT)
    ax3.xaxis.set_major_formatter(fmt)

    # Launch marker
    for ax in (ax1, ax2, ax3):
        ax.axvline(x=0, color=C_LAUNCH, linewidth=1, alpha=0.4, linestyle=':')

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # ── Animation callback ───────────────────────────────────────
    done = [False]

    def animate(_frame):
        if done[0]:
            return

        end = min(cursor[0] + points_per_frame, total)
        cursor[0] = end

        if end < 2:
            return

        s = slice(0, end)
        tc = t[s]
        mc = magnitude[s]

        # Panel 1
        ln_mag.set_data(tc, mc)
        ln_rf.set_data(tc, rf_pred[s])

        nnc = nn_pred[s]
        mask_nn = ~np.isnan(nnc)
        if mask_nn.any():
            ln_nn.set_data(tc[mask_nn], nnc[mask_nn])

        ensc = is_ensemble[s]
        am = ensc > 0
        n_anom = int(ensc.sum())
        if am.any():
            sc_anom.set_offsets(np.column_stack([tc[am], mc[am]]))
        else:
            sc_anom.set_offsets(np.empty((0, 2)))

        # Panel 2
        ln_z.set_data(tc, z_scores[s])

        # Panel 3
        ln_roc.set_data(tc, roc[s])

        # Rolling window
        t_now = tc[-1]
        t_win = t_now - WINDOW_SECONDS
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(t_win, t_now + 2)

        vis = tc >= t_win
        if vis.any():
            vm = mc[vis]
            if len(vm) > 0:
                pad = max(50, (vm.max() - vm.min()) * 0.1)
                ax1.set_ylim(vm.min() - pad, vm.max() + pad)

            vz = z_scores[s][vis]
            if len(vz) > 0:
                ax2.set_ylim(0, max(z_thresh * 1.5, vz.max() * 1.3))

            vr = roc[s][vis]
            if len(vr) > 0:
                ax3.set_ylim(0, max(roc_thresh * 1.5, vr.max() * 1.3))

        # Title
        pct = 100 * end / total
        mt_str = f'T{t_now:+.1f}s' if abs(t_now) >= 0.5 else 'T0'
        if end >= total:
            done[0] = True
            title_text.set_text(
                f'{MISSION_NAME} — Simulation Complete\n'
                f'{total:,} readings  |  {n_anom} anomalies  |  '
                f'Close window to continue')
        else:
            title_text.set_text(
                f'{MISSION_NAME} — Simulation Replay ({speed:.0f}x)\n'
                f'{mt_str}  |  {end:,}/{total:,} ({pct:.0f}%)  |  '
                f'{n_anom} anomalies')

    n_frames = (total // max(points_per_frame, 1)) + 20
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=UPDATE_MS,
                         repeat=False, cache_frame_data=False)
    plt.show()
