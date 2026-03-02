"""
RockSat-X 2026 - Data Loading Module
Parsers for .dat, .csv, and Excel telemetry files.
Handles boot gap detection and time correction.
"""

import os
import re
import time
import numpy as np
import pandas as pd

import config


# ==============================================================================
# GAP DETECTION
# ==============================================================================

def detect_time_gaps(time_values, threshold_seconds=5.0):
    """
    Automatically detect gaps in time data where no samples exist.

    Args:
        time_values: Array of time values (seconds)
        threshold_seconds: Minimum gap size to detect (default 5 seconds)

    Returns:
        List of tuples: [(gap_start, gap_end), ...]
    """
    if len(time_values) < 2:
        config.DETECTED_GAPS = []
        return []

    # Sort unique times
    unique_times = np.sort(np.unique(time_values))

    # Find gaps
    diffs = np.diff(unique_times)
    gap_indices = np.where(diffs > threshold_seconds)[0]

    gaps = []
    for idx in gap_indices:
        gap_start = unique_times[idx]
        gap_end = unique_times[idx + 1]
        gaps.append((gap_start, gap_end))
        print(f"   Detected gap: T+{gap_start:.0f}s to T+{gap_end:.0f}s ({gap_end - gap_start:.0f}s)")

    config.DETECTED_GAPS = gaps
    return gaps


# ==============================================================================
# RAW .DAT FILE PARSER
# ==============================================================================

def parse_dat_file_all(file_path):
    """
    Parse raw .dat telemetry file — extracts ALL sensor types.

    The .dat is the raw serial capture from the flight.  Every sensor script
    writes interleaved lines to the same serial port.  This function parses:
      - Magnetometer (Magneto #0..#3)
      - Pressure (MPRLS, mbar)
      - ADC (4 analog channels a0..a3)
      - Thermocouple (Tc 0, Tc 1 — °C and °F)
      - Flight events (status messages)

    Returns:
        dict with keys: 'magneto', 'pressure', 'adc', 'thermo', 'events'
        Each value is a pd.DataFrame (may be empty if sensor not present).
    """
    result = {
        'magneto': pd.DataFrame(),
        'pressure': pd.DataFrame(),
        'adc': pd.DataFrame(),
        'thermo': pd.DataFrame(),
        'events': pd.DataFrame(),
    }

    if not os.path.exists(file_path):
        print(f"   ERROR: File not found: {file_path}")
        return result

    # ---- Regex patterns (match the team's exact Print() formats) ----
    magneto_pat = re.compile(
        r'T([+-]?\d+):\s*\|(\d+)\|:\s*Magneto\s*#(\d+)\s*X:\s*(-?\d+)\s*Y:\s*(-?\d+)\s*Z:\s*(-?\d+)\s*\(\s*(.+?)\s*\)')
    pressure_pat = re.compile(
        r'T([+-]?\d+):\s*Pressure:\s*([\d.]+)\s*mbar\s*\(\s*(.+?)\s*\)')
    # ADC: a0 has timestamp, a1-a3 follow immediately without timestamp
    adc_a0_pat = re.compile(
        r'T([+-]?\d+):\s*a0:\s*([\d.]+)\s*\(\s*(.+?)\s*\)')
    adc_ax_pat = re.compile(
        r'T([+-]?\d+):\s*a([123]):\s*([\d.]+)')
    # Thermocouple: Tc 0 has timestamp, Tc 1 follows without
    tc0_pat = re.compile(
        r'T([+-]?\d+):\s*Tc\s*0:\s*([\d.]+)\s*C\s*([\d.]+)\s*F\s*\(\s*(.+?)\s*\)')
    tc1_pat = re.compile(
        r'T([+-]?\d+):\s*Tc\s*1:\s*([\d.]+)\s*C\s*([\d.]+)\s*F')
    event_pat = re.compile(
        r'T([+-]?\d+):\s*(.+?)\s*\(\s*(.+?)\s*\)')

    magneto_data, pressure_data, adc_data, thermo_data, events_data = [], [], [], [], []
    # ADC and thermo come in groups — track partial readings
    adc_current = {}
    thermo_current = {}

    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
            lines = f.readlines()

    for raw_line in lines:
        line = re.sub(r'\s+', ' ', raw_line.strip())
        if not line:
            continue

        # --- Magnetometer ---
        m = magneto_pat.match(line)
        if m:
            t, sid, mid, x, y, z, ts = m.groups()
            magneto_data.append({
                'time_mission': int(t), 'sensor_id': int(sid),
                'magneto_id': int(mid), 'x': int(x), 'y': int(y), 'z': int(z),
                'timestamp_str': ts.strip()})
            continue

        # --- Pressure ---
        m = pressure_pat.match(line)
        if m:
            t, val, ts = m.groups()
            pressure_data.append({
                'time_mission': int(t), 'pressure_mbar': float(val),
                'timestamp_str': ts.strip()})
            continue

        # --- ADC a0 (starts new group) ---
        m = adc_a0_pat.match(line)
        if m:
            t, val, ts = m.groups()
            adc_current = {
                'time_mission': int(t), 'a0': float(val),
                'a1': np.nan, 'a2': np.nan, 'a3': np.nan,
                'timestamp_str': ts.strip()}
            continue

        # --- ADC a1/a2/a3 ---
        m = adc_ax_pat.match(line)
        if m:
            t, ch, val = m.groups()
            if adc_current:
                adc_current[f'a{ch}'] = float(val)
                if ch == '3':  # last channel → commit
                    adc_data.append(adc_current)
                    adc_current = {}
            continue

        # --- Thermocouple Tc 0 ---
        m = tc0_pat.match(line)
        if m:
            t, tc, tf, ts = m.groups()
            thermo_current = {
                'time_mission': int(t), 'tc0_C': float(tc), 'tc0_F': float(tf),
                'tc1_C': np.nan, 'tc1_F': np.nan,
                'timestamp_str': ts.strip()}
            continue

        # --- Thermocouple Tc 1 ---
        m = tc1_pat.match(line)
        if m:
            t, tc, tf = m.groups()
            if thermo_current:
                thermo_current['tc1_C'] = float(tc)
                thermo_current['tc1_F'] = float(tf)
                thermo_data.append(thermo_current)
                thermo_current = {}
            continue

        # --- Events (catch-all for timestamped lines) ---
        m = event_pat.match(line)
        if m and 'Magneto' not in line:
            t, desc, ts = m.groups()
            # Skip lines already captured above
            if not any(kw in desc for kw in ['Pressure:', 'a0:', 'Tc 0:']):
                events_data.append({
                    'time_mission': int(t), 'event': desc.strip(),
                    'timestamp_str': ts.strip()})

    # ---- Build DataFrames ----
    # Magnetometer
    mag_df = pd.DataFrame(magneto_data)
    if len(mag_df) > 0:
        mag_df['magnitude'] = np.sqrt(
            mag_df['x']**2 + mag_df['y']**2 + mag_df['z']**2)
        mag_df['datetime'] = pd.to_datetime(
            mag_df['timestamp_str'], format='%a %b %d %H:%M:%S %Y',
            errors='coerce')
        if mag_df['datetime'].notna().any():
            mag_df = mag_df.sort_values('datetime').reset_index(drop=True)

    result['magneto'] = mag_df
    result['pressure'] = pd.DataFrame(pressure_data)
    result['adc'] = pd.DataFrame(adc_data)
    result['thermo'] = pd.DataFrame(thermo_data)
    result['events'] = pd.DataFrame(events_data)

    # Print summary
    counts = {k: len(v) for k, v in result.items()}
    print(f"   Parsed .dat: {counts['magneto']} magneto, "
          f"{counts['pressure']} pressure, {counts['adc']} ADC, "
          f"{counts['thermo']} thermocouple, {counts['events']} events")

    return result


def parse_dat_file(file_path):
    """
    Parse raw .dat telemetry file — magnetometer + events only.
    Backwards-compatible wrapper around parse_dat_file_all().

    Returns:
        tuple: (magneto_df, events_df) DataFrames
    """
    result = parse_dat_file_all(file_path)
    mag_df = result['magneto']
    events_df = result['events']
    if len(mag_df) == 0:
        print("   WARNING: No magnetometer data found in .dat file")
        return pd.DataFrame(), pd.DataFrame()
    return mag_df, events_df


def load_from_dat_file(file_path):
    """
    Load and preprocess data from a .dat file (actual flight format).

    Args:
        file_path: Path to .dat file

    Returns:
        tuple: (df, X, y_values, steps, features, x, y, z, t, mag_data_by_sensor)
    """
    df, events_df = parse_dat_file(file_path)

    empty_return = (pd.DataFrame(), np.zeros((0, 3)), np.zeros(0), 0, 3,
                    np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), {})

    if df is None or len(df) == 0:
        return empty_return

    # Rename columns for consistency
    df = df.rename(columns={'time_mission': 'Time'})

    # Separate data by magnetometer ID
    # Apply per-sensor smoothing to reduce quantization staircase from RM3100 ADC
    mag_data_by_sensor = {}
    if 'magneto_id' in df.columns:
        for mag_id in sorted(df['magneto_id'].unique()):
            mag_df = df[df['magneto_id'] == mag_id].copy()
            sx, sy, sz, smag = smooth_sensor_data(
                mag_df['x'].values, mag_df['y'].values, mag_df['z'].values)
            mag_data_by_sensor[int(mag_id)] = {
                'time': mag_df['Time'].values,
                'x': sx, 'y': sy, 'z': sz,
                'magnitude': smag,
                'count': len(mag_df)
            }
            mask = df['magneto_id'] == mag_id
            df.loc[mask, 'x'] = sx
            df.loc[mask, 'y'] = sy
            df.loc[mask, 'z'] = sz
            df.loc[mask, 'magnitude'] = smag

    # Create feature matrix (3 features: Bx, By, Bz — no time)
    X = np.column_stack([
        df['x'].values,
        df['y'].values,
        df['z'].values,
    ]).astype(float)

    y_values = df['magnitude'].values.astype(float)
    steps = len(df)
    features = 3

    return (df, X, y_values, steps, features,
            df['x'].values, df['y'].values, df['z'].values, df['Time'].values,
            mag_data_by_sensor)


# ==============================================================================
# LEGACY (PRE-GHOST) CSV PARSER
# ==============================================================================

def parse_legacy_csv(file_path):
    """
    Parse pre-GHOST UPR flight CSV files (semi-parsed serial capture).

    These files have rows like:
        -118,|0|:,Magneto #1,X:,-1829,Y:,1788,Z:,-737,( Wed Jun 18 16:20:22 2025 ),...

    Columns:  Time, Counter, SensorName, X:, Xval, Y:, Yval, Z:, Zval, (timestamp), ...

    Returns:
        pd.DataFrame with columns: Time, Sensor, X, Y, Z, Magnitude
    """
    raw = pd.read_csv(file_path, header=None)

    rows = []
    for _, r in raw.iterrows():
        sensor_str = str(r.iloc[2]) if len(r) > 2 else ''
        if 'Magneto' not in sensor_str:
            continue
        try:
            # Extract sensor number (e.g. "Magneto #1" → 0, "#2" → 1, "#3" → 2)
            sensor_num = int(sensor_str.split('#')[1].strip()) - 1
            t_val = int(r.iloc[0])
            x_val = int(r.iloc[4])
            y_val = int(r.iloc[6])
            z_val = int(r.iloc[8])
            rows.append({
                'Time': t_val,
                'Sensor': sensor_num,
                'X': x_val,
                'Y': y_val,
                'Z': z_val,
            })
        except (ValueError, IndexError):
            continue

    if not rows:
        print("   WARNING: No magnetometer data found in legacy CSV")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['Magnitude'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    print(f"   Parsed legacy CSV: {len(df)} magnetometer readings, "
          f"sensors {sorted(df['Sensor'].unique())}, "
          f"T{df['Time'].min()} to T+{df['Time'].max()}")
    return df


def is_legacy_csv(file_path):
    """Check if a CSV is the legacy semi-parsed format (no header, pipe-delimited counters)."""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline()
        return '|' in first_line and 'Magneto' in first_line
    except Exception:
        return False


# ==============================================================================
# FIXED TIMELINE CSV LOADER
# ==============================================================================

def load_from_csv(file_path):
    """
    Load magnetometer data from fixed timeline CSV file.
    This file has correct time gaps where the payload was off.

    Args:
        file_path: Path to CSV file

    Returns:
        tuple: (df, X, y_values, steps, features, x, y, z, t, mag_data_by_sensor)
    """
    # Auto-detect legacy (pre-GHOST) CSV format
    if is_legacy_csv(file_path):
        print(f"   Detected legacy (pre-GHOST) CSV format")
        df = parse_legacy_csv(file_path)
        if len(df) == 0:
            return (pd.DataFrame(), np.zeros((0, 3)), np.zeros(0), 0, 3,
                    np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), {})
    else:
        df = pd.read_csv(file_path)

    print(f"   Loaded {len(df)} rows from CSV")
    print(f"   Time range: T{df['Time'].min():.0f} to T+{df['Time'].max():.0f}")
    print(f"   Sensors: {sorted(df['Sensor'].unique())}")

    # Rename columns to match expected format
    df = df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'})

    # Calculate magnitude if not present
    if 'Magnitude' not in df.columns:
        df['Magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    # Convert columns to float before smoothing (RM3100 outputs integers,
    # but Savitzky-Golay produces floats)
    for col in ['x', 'y', 'z']:
        df[col] = df[col].astype(float)
    if 'Magnitude' in df.columns:
        df['Magnitude'] = df['Magnitude'].astype(float)

    # Group by sensor for multi-magnetometer plots
    # Apply per-sensor smoothing to reduce quantization staircase from RM3100 ADC
    mag_data_by_sensor = {}
    for sensor_id in df['Sensor'].unique():
        sensor_df = df[df['Sensor'] == sensor_id].copy()
        sx, sy, sz, smag = smooth_sensor_data(
            sensor_df['x'].values, sensor_df['y'].values, sensor_df['z'].values)
        mag_data_by_sensor[int(sensor_id)] = {
            'time': sensor_df['Time'].values,
            'x': sx, 'y': sy, 'z': sz,
            'magnitude': smag,
            'count': len(sensor_df)
        }
        # Write smoothed values back to DataFrame for consistent X/y construction
        mask = df['Sensor'] == sensor_id
        df.loc[mask, 'x'] = sx
        df.loc[mask, 'y'] = sy
        df.loc[mask, 'z'] = sz
        df.loc[mask, 'Magnitude'] = smag

    # Prepare feature matrix (3 features: Bx, By, Bz — no time)
    X = np.column_stack([
        df['x'].values,
        df['y'].values,
        df['z'].values,
    ])
    y_values = df['Magnitude'].values
    steps = len(df)
    features = 3

    return (df, X, y_values, steps, features,
            df['x'].values, df['y'].values, df['z'].values, df['Time'].values,
            mag_data_by_sensor)


# ==============================================================================
# EXCEL FILE LOADER (Legacy support)
# ==============================================================================

def load_from_excel(file_path, sheet_name='UPR Flight Data Editado'):
    """
    Load magnetometer data from Excel file (legacy format).

    Args:
        file_path: Path to Excel file
        sheet_name: Name of sheet containing data

    Returns:
        tuple: (df, X, y_values, steps, features, x, y, z, t)
    """
    empty_return = (pd.DataFrame(), np.zeros((0, 3)), np.zeros(0), 0, 3,
                    np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))

    if not os.path.exists(file_path):
        print(f"   ERROR: File not found: {file_path}")
        return empty_return

    print(f'   Reading data from: {file_path}')
    raw_data = pd.read_excel(file_path, sheet_name=sheet_name)

    magneto_data = []
    for index, row in raw_data.iterrows():
        if 'Magneto' in str(row.iloc[2]):
            try:
                magneto_id = str(row.iloc[2]).split('#')[1].strip()
                t_value = row.iloc[0]
                x_value = row.iloc[4]
                y_value = row.iloc[6]
                z_value = row.iloc[8]

                # Parse T value
                if isinstance(t_value, str):
                    t_match = re.match(r'T([+-]?\d+)', t_value)
                    if t_match:
                        t_value = int(t_match.group(1))
                    else:
                        continue
                else:
                    t_value = int(t_value)

                magneto_data.append({
                    'magneto_id': magneto_id,
                    'Time': t_value,
                    'x': float(x_value),
                    'y': float(y_value),
                    'z': float(z_value)
                })
            except Exception as e:
                continue

    if not magneto_data:
        print("   No magnetometer data found in Excel file")
        return empty_return

    df = pd.DataFrame(magneto_data)
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    X = np.column_stack([
        df['x'].values,
        df['y'].values,
        df['z'].values
    ])
    y_values = df['magnitude'].values

    return (df, X, y_values, len(df), 3,
            df['x'].values, df['y'].values, df['z'].values, df['Time'].values)


# ==============================================================================
# LIVE SENSOR DATA LOADER
# ==============================================================================

def load_from_sensor(sensor, n_samples=1000, delay=0.01):
    """
    Load data from live magnetometer sensor.

    Args:
        sensor: MultiMagnetometerReader instance
        n_samples: Number of samples to collect
        delay: Delay between samples (seconds)

    Returns:
        tuple: (df, X, y_values, steps, features, x, y, z, t)
    """
    empty_return = (pd.DataFrame(), np.zeros((0, 3)), np.zeros(0), 0, 3,
                    np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))

    live_data = []
    print(f'   Reading {n_samples} samples from sensor...')

    for i in range(n_samples):
        data = sensor.read_data()
        if 'error' not in data:
            live_data.append([
                data['x'],
                data['y'],
                data['z'],
                data['timestamp']
            ])
        else:
            print(f"   Error reading sensor: {data['error']}")
        time.sleep(delay)

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"   Collected {i + 1}/{n_samples} samples...")

    if not live_data:
        print("   No valid sensor data collected")
        return empty_return

    df = pd.DataFrame(live_data, columns=['x', 'y', 'z', 'Time'])
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    X = np.column_stack([
        df['x'].values,
        df['y'].values,
        df['z'].values
    ])
    y_values = df['magnitude'].values

    return (df, X, y_values, len(df), 3,
            df['x'].values, df['y'].values, df['z'].values, df['Time'].values)


# ==============================================================================
# PREPROCESSING
# ==============================================================================

def smooth_sensor_data(x, y, z):
    """
    Apply Savitzky-Golay smoothing to per-axis magnetometer data.
    Reduces integer quantization staircase artifacts from RM3100 ADC.

    Smoothing is applied per-axis independently, then magnitude is
    recomputed from smoothed components (preserving vector consistency).

    Args:
        x, y, z: Bx, By, Bz arrays (1D, integer or float)

    Returns:
        tuple: (x_smooth, y_smooth, z_smooth, magnitude_smooth)
               Returns originals as float if smoothing is disabled or data too small.
    """
    cfg = config.DATA_SMOOTHING
    xf = x.astype(float)
    yf = y.astype(float)
    zf = z.astype(float)

    if not cfg.get('ENABLED', False):
        return xf, yf, zf, np.sqrt(xf**2 + yf**2 + zf**2)

    n = len(x)
    if n < cfg.get('MIN_SAMPLES', 20):
        return xf, yf, zf, np.sqrt(xf**2 + yf**2 + zf**2)

    try:
        from scipy.signal import savgol_filter
        window = cfg.get('SAVGOL_WINDOW', 7)
        polyorder = cfg.get('SAVGOL_POLYORDER', 2)
        # Window must be odd and <= n
        window = min(window, n)
        if window % 2 == 0:
            window -= 1
        if window < polyorder + 1:
            return xf, yf, zf, np.sqrt(xf**2 + yf**2 + zf**2)

        xs = savgol_filter(xf, window, polyorder)
        ys = savgol_filter(yf, window, polyorder)
        zs = savgol_filter(zf, window, polyorder)
    except ImportError:
        # scipy not available — fallback to no smoothing
        return xf, yf, zf, np.sqrt(xf**2 + yf**2 + zf**2)

    return xs, ys, zs, np.sqrt(xs**2 + ys**2 + zs**2)


def load_combined_training_data(dataset_paths):
    """
    Load and combine multiple CSV datasets for universal model training.

    Models trained on combined data generalize across different magnetic
    environments (e.g., Virginia ~2600 nT vs Norway ~4000 nT).

    Args:
        dataset_paths: List of CSV file paths to combine

    Returns:
        tuple: (combined_X, combined_y) — concatenated [Bx,By,Bz] and magnitudes
    """
    all_X = []
    all_y = []

    print("\n" + "=" * 50)
    print("COMBINED TRAINING DATA")
    print("=" * 50)

    for path in dataset_paths:
        if not os.path.exists(path):
            print(f"   SKIP (not found): {path}")
            continue

        basename = os.path.basename(path)
        try:
            result = load_from_csv(path)
            df, X, y_vals = result[0], result[1], result[2]

            if len(X) == 0:
                print(f"   SKIP (empty): {basename}")
                continue

            all_X.append(X)
            all_y.append(y_vals)
            print(f"   {basename}: {len(X):,} samples, "
                  f"|B| range [{y_vals.min():.0f}, {y_vals.max():.0f}] nT")
        except Exception as e:
            print(f"   SKIP (error): {basename} — {e}")
            continue

    if not all_X:
        print("   ERROR: No training data loaded!")
        return np.zeros((0, 3)), np.zeros(0)

    combined_X = np.vstack(all_X)
    combined_y = np.concatenate(all_y)

    print(f"\n   Combined: {len(combined_X):,} total samples, "
          f"|B| range [{combined_y.min():.0f}, {combined_y.max():.0f}] nT")
    print("=" * 50)

    return combined_X, combined_y


def preprocess_for_ml(X, y):
    """
    Preprocess data for ML algorithms.
    X is now 3-col [Bx, By, Bz] — no time column.

    Args:
        X: Feature matrix (Bx, By, Bz) — 3 columns
        y: Target values (magnitude)

    Returns:
        tuple: (X_copy, y)
    """
    X_norm = X.copy()
    return X_norm, y


def filter_outliers(X, y, max_magnitude=5500):
    """
    Filter extreme outlier readings.

    Args:
        X: Feature matrix
        y: Magnitude values
        max_magnitude: Maximum valid magnitude (nT)

    Returns:
        tuple: (X_filtered, y_filtered)
    """
    mask = y < max_magnitude
    removed = np.sum(~mask)
    if removed > 0:
        print(f"   Filtered {removed} outliers (magnitude > {max_magnitude} nT)")
    return X[mask], y[mask]
