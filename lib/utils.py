"""
RockSat-X 2026 - Utility Functions
Time formatting, storage setup, and helper functions.
"""

import os
import glob

from config import STORAGE_PATH, MISSION_TIMELINE


# ==============================================================================
# TIME FORMATTING
# ==============================================================================

def format_time(seconds):
    """
    Format seconds into HH:MM:SS string.

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted time string
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fast_format(seconds):
    """
    Fast time formatting (no divmod for speed).

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted time string
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int((seconds % 3600) % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class HoverTimeCache:
    """
    Pre-computed time strings for fast hover display.
    Caches time strings at minute resolution.
    """

    def __init__(self):
        self.minute_cache = [fast_format(t) for t in range(0, 86400, 60)]

    def get_time(self, timestamp):
        """Get cached time string for a timestamp."""
        return self.minute_cache[int(timestamp % 86400 // 60)]


# Global cache instance
hover_cache = HoverTimeCache()


# ==============================================================================
# MISSION PHASE DETECTION
# ==============================================================================

def get_mission_phase(mission_time):
    """
    Determine current mission phase from mission time.

    Args:
        mission_time: Time in seconds relative to launch (T=0)

    Returns:
        str: Phase name
    """
    t_exp_off = MISSION_TIMELINE.get('T_EXPERIMENTS_OFF', float('inf'))
    t_apogee = MISSION_TIMELINE.get('T_APOGEE', float('inf'))
    t_launch = MISSION_TIMELINE.get('T_LAUNCH', 0)
    t_ml = MISSION_TIMELINE.get('T_ML_START', float('-inf'))
    t_sys = MISSION_TIMELINE.get('T_SYSTEMS_START', float('-inf'))

    if mission_time >= t_exp_off:
        return 'POST_EXPERIMENT'
    elif mission_time >= t_apogee:
        return 'DESCENT'
    elif mission_time >= t_launch:
        return 'ASCENT'
    elif mission_time >= t_ml:
        return 'ML_ACTIVE'
    elif mission_time >= t_sys:
        return 'SYSTEMS_ON'
    else:
        return 'STANDBY'


def get_phase_color(phase):
    """
    Get color associated with a mission phase.

    Args:
        phase: Phase name string

    Returns:
        str: Color code
    """
    colors = {
        'STANDBY': '#555555',
        'SYSTEMS_ON': '#4ECDC4',
        'ML_ACTIVE': '#45B7D1',
        'ASCENT': '#FF6B6B',
        'DESCENT': '#FFE66D',
        'POST_EXPERIMENT': '#96CEB4'
    }
    return colors.get(phase, '#FFFFFF')


# ==============================================================================
# STORAGE MANAGEMENT
# ==============================================================================

def setup_model_storage():
    """
    Set up model storage directories.

    Returns:
        str: Path to storage directory
    """
    os.makedirs(STORAGE_PATH, exist_ok=True)
    return STORAGE_PATH


def cleanup_old_models(storage_path, keep_last=10):
    """
    Remove old model files, keeping only the most recent.

    Args:
        storage_path: Path to model storage
        keep_last: Number of recent models to keep
    """
    try:
        for model_type in ['rf', 'nn', 'cluster']:
            files = glob.glob(os.path.join(storage_path, model_type, '*'))
            files.sort(key=os.path.getmtime, reverse=True)
            for old_file in files[keep_last:]:
                os.remove(old_file)
                print(f"Removed old model: {old_file}")
    except Exception as e:
        print(f"Error cleaning up models: {e}")


# ==============================================================================
# DATA VALIDATION
# ==============================================================================

def validate_sensor_reading(x, y, z, max_range=800000):
    """
    Validate magnetometer reading is within expected range.

    Args:
        x, y, z: Magnetic field components (nT)
        max_range: Maximum expected value (default RM3100 range)

    Returns:
        tuple: (is_valid: bool, reason: str or None)
    """
    for axis, val in [('x', x), ('y', y), ('z', z)]:
        if abs(val) > max_range:
            return False, f"{axis}-axis value {val} exceeds range (+/-{max_range})"

    return True, None


def calculate_magnitude(x, y, z):
    """
    Calculate magnetic field magnitude.

    Args:
        x, y, z: Magnetic field components

    Returns:
        float: Magnitude
    """
    import numpy as np
    return np.sqrt(x**2 + y**2 + z**2)


# ==============================================================================
# FILE DETECTION
# ==============================================================================

def find_telemetry_file(directory=None):
    """
    Auto-detect telemetry file in directory.
    Priority: fixed CSV > .txt > .dat

    Args:
        directory: Directory to search (default: current directory)

    Returns:
        tuple: (file_path, file_type) or (None, None)
    """
    if directory is None:
        directory = os.getcwd()

    # Priority 1: CSV files in data/ or root (newest first, no hardcoded preference)
    for subdir in ['data', '.']:
        csv_files = sorted(
            glob.glob(os.path.join(directory, subdir, '*.csv')),
            key=os.path.getmtime, reverse=True
        )
        if csv_files:
            return csv_files[0], 'csv'

    # Priority 3: Telemetry .txt files (skip requirements/config files)
    txt_files = [f for f in glob.glob(os.path.join(directory, '*.txt'))
                 if 'requirements' not in os.path.basename(f).lower()]
    if txt_files:
        return txt_files[0], 'dat'  # Same parser works

    # Priority 4: .dat files (check data/ subdir too)
    for subdir in ['data', '.']:
        dat_files = [f for f in glob.glob(os.path.join(directory, subdir, '*.dat'))
                     if 'venv' not in f]
        if dat_files:
            return dat_files[0], 'dat'

    return None, None
