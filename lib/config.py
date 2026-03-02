"""
RockSat-X 2026 - Configuration Module
Platform detection, mission timeline, and global constants.
"""

import os
import platform

# ==============================================================================
# PLATFORM DETECTION
# ==============================================================================

IS_RASPBERRY_PI = platform.system() == 'Linux' and os.path.exists('/proc/device-tree/model')

# ==============================================================================
# STORAGE PATHS (Universal - relative to script location)
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # parent of lib/
STORAGE_PATH = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(STORAGE_PATH, exist_ok=True)

# Output subdirectories
for _subdir in ['flight_data', 'post_flight', 'plots']:
    os.makedirs(os.path.join(STORAGE_PATH, _subdir), exist_ok=True)

# ==============================================================================
# MISSION IDENTITY — Change per flight dataset
# ==============================================================================

MISSION_NAME = "RockSat-X"   # Used in all plot titles and reports

# Mission event annotations for plots: list of (time_s, label, color)
# Populate per-mission. Empty = no event markers on plots.
#
# GHOST 2025 (Norway) example:
#   MISSION_EVENTS = [
#       (31.7, 'Burnout', '#FF6B6B'), (79, 'Skirt Deploy', '#FFE66D'),
#       (198.6, 'Apogee', '#4ECDC4'), (307, 'Flag 3', '#96CEB4'),
#   ]
MISSION_EVENTS = []

# ==============================================================================
# MISSION TIMELINE (seconds relative to launch T=0)
# Only T_LAUNCH is required. All other keys are optional — code uses .get()
# with safe defaults so missing keys never crash.
# ==============================================================================

MISSION_TIMELINE = {
    'T_SYSTEMS_START': -150,   # Systems power on
    'T_ML_START': -126,        # ML algorithms start
    'T_CAMERA_START': -30,     # Camera recording starts
    'T_LAUNCH': 0,             # Liftoff
    'T_EXPERIMENTS_OFF': 336,  # Experiments power off
    # Per-mission events (uncomment/set for your flight):
    # 'T_BURN_END': 31.7,      # Motor burnout
    # 'T_SKIRT_DEPLOY': 79,    # Nosecone separation
    # 'T_APOGEE': 198.6,       # Maximum altitude
    # 'T_FLAG_3': 307,         # Experiment flag 3
    # 'T_PARACHUTE': 480.9,    # Parachute deployment
    # 'T_SPLASH_DOWN': 809.6   # Ocean splashdown
}

# ==============================================================================
# BOOT GAP TIMELINE — GHOST 2025 TEST DATA ONLY
# These values are specific to the GHOST 2025 Norway .dat file.
# For new flight data, gaps are auto-detected by data_loader.detect_time_gaps().
# ==============================================================================

BOOT_GAP = {
    'BOOT2_END': 19,           # Boot 2 ends at T+19 (mission time)
    'GAP_DURATION': 26,        # Payload OFF for 26 seconds
    'BOOT3_START': 45,         # Boot 3 starts at T+45 (mission time)
    'BOOT3_CLOCK_START': -124, # Boot 3 onboard clock reads T-124
    'TIME_OFFSET': 169         # Add this to Boot 3 timestamps
}

# Global variable for detected gaps (populated by data_loader)
DETECTED_GAPS = []

# ==============================================================================
# SENSOR CONFIGURATION
# ==============================================================================

SENSOR_CONFIG = {
    'SAMPLING_RATE_HZ': 45,           # Target sampling rate
    'SAMPLE_PERIOD_MS': 22.2,         # 1000/45 = 22.2ms per sample
    'N_MAGNETOMETERS': 3,             # Number of RM3100 sensors (2026: 3, was 4 in 2025)
    'DEFAULT_CHANNELS': [0, 1, 2],    # MUX channels for the 3 magnetometers
    'MAGNETOMETER_RANGE_NT': 800000,  # RM3100 range: +/- 800 uT = 800,000 nT
    'I2C_MULTIPLEXER_ADDR': 0x70,     # TCA9548A address
    'RM3100_DEFAULT_ADDR': 0x21       # RM3100 address (discovered via MUX scan)
}

# ==============================================================================
# GPIO PIN DEFINITIONS (matches RSX_5TEST.py / flight_controller.py)
# ==============================================================================

GPIO_PINS = {
    'LAUNCH': 12,           # Launch flag (active LOW)
    'SKIRT': 13,            # Skirt deployment flag (active LOW)
    'POWEROFF30': 16,       # 30s before power cut (active LOW)
    'FMI': 26,              # Full Mission Inhibit
    'IBF': 20,              # Insert Before Flight
    'TDL_RELAY': 27,        # TDL relay output (pin 27 per RSX_5TEST.py)
    'STEPPER_RELAY': 17,    # Stepper motor relay output
    'ENA_RELAY': 22,        # Enable relay output
    'PROX_SENSOR': 19,      # Proximity sensor input
}

# ==============================================================================
# SERIAL TELEMETRY CONFIG (matches utility2026.py)
# ==============================================================================

SERIAL_CONFIG = {
    'PORT': '/dev/ttyAMA0',
    'BAUDRATE': 153600,     # RockSat-X 2026 baud rate (was 230400 for GHOST)
}

# ==============================================================================
# ML CONFIGURATION
# ==============================================================================

ML_CONFIG = {
    'RF_N_ESTIMATORS': 200,
    'RF_MAX_DEPTH': 20,
    'NN_EPOCHS': 100,
    'NN_BATCH_SIZE': 32,
    'KMEANS_K': 3,
    'GPR_MAX_SAMPLES': 500,           # Limit GPR training samples for speed
    
    # ══════════════════════════════════════════════════════════════════════════
    # ANOMALY DETECTION THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════
    # ORIGINAL VALUES (conservative - fewer anomalies):
    #   'ANOMALY_Z_THRESHOLD': 3.0      → 2 anomalies
    #   'ISOLATION_FOREST_CONTAMINATION': 0.05  → 800 anomalies (5%)
    #   'ENSEMBLE_VOTE_THRESHOLD': 2    → 89 ensemble anomalies
    #   'RATE_OF_CHANGE_THRESHOLD': N/A (method didn't exist)
    #
    # VALIDATED VALUES (post-validation tuning — Feb 2026):
    # See VALIDATION_FIXES_REPORT.md for full analysis
    'ANOMALY_Z_THRESHOLD': 3.5,           # Modified Z-score with MAD (was 2.5 simple z-score)
    'ISOLATION_FOREST_CONTAMINATION': 0.02,  # 2% contamination (was 0.10 → 97% false positives)
    'ENSEMBLE_VOTE_THRESHOLD': 2,         # 2+ methods must agree
    'RATE_OF_CHANGE_THRESHOLD': 200.0,    # Tightened from 700 nT/sample
    'LOF_N_NEIGHBORS': 50,                # Increased from 20 for more stable density estimates
    'LOF_CONTAMINATION': 0.02,            # Matched to IF contamination (was 'auto')
}

# ==============================================================================
# HAILO NPU CONFIGURATION
# ==============================================================================

# ==============================================================================
# COMBINED TRAINING DATASETS
# ==============================================================================
# All clean CSVs used for universal model training.
# Models are trained on ALL listed datasets so they generalize across
# different magnetic environments (Virginia vs Norway).

TRAINING_DATASETS = [
    os.path.join(PROJECT_ROOT, 'data', 'UPR_2025_Flight.csv'),
    os.path.join(PROJECT_ROOT, 'data', 'Magneto_Fixed_Timeline.csv'),
]

# ==============================================================================
# DATA SMOOTHING (Savitzky-Golay filter for sensor quantization)
# ==============================================================================
# The RM3100 outputs integer ADC counts. Smoothing reduces staircase artifacts
# while preserving peaks and curvature (polynomial local fitting).

DATA_SMOOTHING = {
    'ENABLED': True,
    'SAVGOL_WINDOW': 7,       # Must be odd; 7 samples ~ 155ms at 45 Hz
    'SAVGOL_POLYORDER': 2,    # Quadratic fit preserves curvature at apogee
    'MIN_SAMPLES': 20,        # Don't smooth datasets smaller than this
}

# ==============================================================================
# HAILO NPU CONFIGURATION
# ==============================================================================

HAILO_HEF_PATH = os.path.join(PROJECT_ROOT, 'hailo', 'magnetometer.hef')

# ==============================================================================
# VISUALIZATION COLORS
# ==============================================================================

SENSOR_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D', '#96CEB4']
AXIS_COLORS = {'x': 'cyan', 'y': 'magenta', 'z': 'yellow'}
GAP_COLOR = 'magenta'
LAUNCH_COLOR = 'lime'
