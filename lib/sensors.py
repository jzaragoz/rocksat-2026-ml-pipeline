"""
RockSat-X 2026 - Sensor Module
Multi-magnetometer I2C reader for RM3100 sensors via TCA9548A multiplexer.

Uses the same I2C pattern as the team's magnetometerMVprintable.py:
  - smbus for register-level communication
  - Direct TCA9548A channel select via bus.write_byte(TCA_ADDR, 1 << channel)
  - RM3100 discovery via REVID register (0x22 confirms RM3100)
  - Continuous measurement mode (CMM = 0x79) for reliable reads
  - 9-byte block read from MX2 register for X, Y, Z

Falls back to error returns if hardware is unavailable (Mac/desktop).
"""

import time
from config import SENSOR_CONFIG

# ==============================================================================
# CONDITIONAL IMPORTS — only available on Raspberry Pi with I2C hardware
# ==============================================================================

HARDWARE_AVAILABLE = False
try:
    import smbus
    HARDWARE_AVAILABLE = True
except ImportError:
    pass


class MultiMagnetometerReader:
    """
    Multi-channel magnetometer reader for RM3100 sensors via TCA9548A MUX.

    Uses the same I2C pattern as the team's magnetometerMVprintable.py:
      - smbus for register-level communication
      - Direct mux channel select (bus.write_byte)
      - REVID-based RM3100 discovery
      - Continuous measurement mode (CMM)
      - 9-byte block read for X/Y/Z
      - try/except + continue for graceful skip on sensor failures

    Falls back to error returns if hardware is unavailable (Mac/desktop).
    """

    # RM3100 Register addresses (same as magnetometerMVprintable.py)
    CCX = 0x04
    CCY = 0x06
    CCZ = 0x08
    POLL = 0x00
    CMM = 0x01
    MX2 = 0x24
    STATUS = 0x34
    REVID = 0x36

    # Known RM3100 addresses based on SA0/SA1 pins
    RM3100_ADDRESSES = [0x20, 0x21, 0x22, 0x23]

    def __init__(self, multiplex_address=None, channels=None, sensor_address=None):
        """
        Initialize the magnetometer reader.

        Args:
            multiplex_address: I2C address of TCA9548A multiplexer (default 0x70)
            channels: List of multiplexer channels to use (default [0, 1, 2])
            sensor_address: Not used directly — addresses discovered via REVID scan
        """
        self.multiplex_address = multiplex_address or SENSOR_CONFIG['I2C_MULTIPLEXER_ADDR']
        self.channels = channels if channels is not None else [0, 1, 2]
        self.failed_channels = set()
        self.bus = None
        # Cache discovered sensor addresses per channel: {channel: rm3100_addr}
        self._channel_addrs = {}
        # Track which channels have been initialized with CMM
        self._cmm_initialized = set()

        if HARDWARE_AVAILABLE:
            try:
                self.bus = smbus.SMBus(1)
                print(f"  I2C bus initialized (SMBus 1)")
            except Exception as e:
                print(f"  I2C init failed: {e}")
                self.bus = None
        else:
            print(f"  ⚠ I2C hardware not available (smbus not installed)")

    def _select_mux_channel(self, channel):
        """Select a channel on the TCA9548A multiplexer (same as magnetometerMVprintable.py)."""
        try:
            self.bus.write_byte(self.multiplex_address, 1 << channel)
            time.sleep(0.005)
        except Exception as e:
            raise RuntimeError(f"MUX channel select failed: {e}")

    def _discover_rm3100(self, channel):
        """
        Discover RM3100 on a specific mux channel by checking REVID register.
        Same approach as magnetometerMVprintable.py discover_rm3100_on_multiplexer().

        Returns:
            int or None: RM3100 I2C address if found, None otherwise
        """
        self._select_mux_channel(channel)

        # Try known RM3100 addresses first
        for addr in self.RM3100_ADDRESSES:
            try:
                revid = self.bus.read_byte_data(addr, self.REVID)
                if revid == 0x22:
                    return addr
            except Exception:
                pass

        # Scan all addresses as fallback
        for addr in range(0x03, 0x78):
            if addr == self.multiplex_address:
                continue
            try:
                revid = self.bus.read_byte_data(addr, self.REVID)
                if revid == 0x22:
                    return addr
            except Exception:
                pass

        return None

    def _init_cmm(self, channel, addr):
        """
        Initialize RM3100 in continuous measurement mode.
        Same as magnetometerMVprintable.py sensor_thread() initialization.
        """
        self._select_mux_channel(channel)

        # Set cycle counts for each axis (200 = good balance of speed/resolution)
        cycle_count = 200
        cc_high = (cycle_count >> 8) & 0xFF
        cc_low = cycle_count & 0xFF

        self.bus.write_i2c_block_data(addr, self.CCX, [cc_high, cc_low])
        time.sleep(0.01)
        self.bus.write_i2c_block_data(addr, self.CCY, [cc_high, cc_low])
        time.sleep(0.01)
        self.bus.write_i2c_block_data(addr, self.CCZ, [cc_high, cc_low])
        time.sleep(0.01)

        # Start continuous measurement mode
        # 0x79 = 0b01111001: all axes, continuous mode
        self.bus.write_byte_data(addr, self.CMM, 0x79)
        time.sleep(0.05)

        self._cmm_initialized.add(channel)

    def verify_sensors(self):
        """
        Verify which sensors are responding on each MUX channel.
        Uses REVID-based discovery (same as magnetometerMVprintable.py).

        Returns:
            dict: {channel: bool} indicating which channels have working sensors
        """
        working = {}
        if self.bus is None:
            for ch in self.channels:
                working[ch] = False
            return working

        for channel in self.channels:
            try:
                addr = self._discover_rm3100(channel)
                if addr is not None:
                    self._channel_addrs[channel] = addr
                    # Initialize CMM on discovery
                    self._init_cmm(channel, addr)
                    working[channel] = True
                    print(f"  ✓ RM3100 on ch{channel} at 0x{addr:02X} (CMM initialized)")
                else:
                    working[channel] = False
                    print(f"  ✗ No RM3100 on ch{channel}")
            except Exception as e:
                print(f"  Verify ch{channel}: {e}")
                working[channel] = False

        return working

    def read_magnetometer(self, channel=None):
        """
        Read magnetometer data from a specific MUX channel.
        Uses the same I2C pattern as magnetometerMVprintable.py:
          1. Select mux channel
          2. Read 9 bytes from MX2 register (X, Y, Z as 3 bytes each)
          3. Convert 24-bit two's complement to signed integers

        Graceful skip: if anything fails, returns {'error': ...} and continues.

        Args:
            channel: MUX channel number (0-7)

        Returns:
            dict: {'x': int, 'y': int, 'z': int, 'timestamp': float}
                  or {'error': str} on failure
        """
        if channel is None:
            channel = self.channels[0] if self.channels else 0

        if channel in self.failed_channels:
            return {'error': f'channel {channel} permanently failed'}

        if self.bus is None:
            return {'error': 'I2C hardware not available'}

        # Get cached address or discover
        addr = self._channel_addrs.get(channel)
        if addr is None:
            try:
                addr = self._discover_rm3100(channel)
                if addr is None:
                    return {'error': f'channel {channel} no RM3100 found'}
                self._channel_addrs[channel] = addr
            except Exception as e:
                return {'error': f'channel {channel} discovery failed: {e}'}

        # Initialize CMM if not yet done for this channel
        if channel not in self._cmm_initialized:
            try:
                self._init_cmm(channel, addr)
            except Exception as e:
                return {'error': f'channel {channel} CMM init failed: {e}'}

        try:
            # Select mux channel
            self._select_mux_channel(channel)

            # Read 9 bytes: X(3) + Y(3) + Z(3) from MX2 register
            # Same as magnetometerMVprintable.py sensor_thread() read
            data = self.bus.read_i2c_block_data(addr, self.MX2, 9)

            x = (data[0] << 16) | (data[1] << 8) | data[2]
            y = (data[3] << 16) | (data[4] << 8) | data[5]
            z = (data[6] << 16) | (data[7] << 8) | data[8]

            # Convert to signed integers (24-bit two's complement)
            # Same as magnetometerMVprintable.py
            if x & 0x800000:
                x -= 0x1000000
            if y & 0x800000:
                y -= 0x1000000
            if z & 0x800000:
                z -= 0x1000000

            return {
                'x': x,
                'y': y,
                'z': z,
                'timestamp': time.time()
            }

        except Exception as e:
            return {'error': f'channel {channel} read failed: {e}'}

    def read_data(self):
        """Alias for read_magnetometer() for compatibility."""
        return self.read_magnetometer()

    def read_all_channels(self):
        """
        Read magnetometer data from all configured channels.
        Gracefully skips any channel that fails.

        Returns:
            dict: {channel: reading_dict} for all channels
        """
        readings = {}
        for channel in self.channels:
            if channel not in self.failed_channels:
                readings[channel] = self.read_magnetometer(channel)
        return readings

    def sensor_status(self):
        """
        Get current sensor status.

        Returns:
            dict: Status information
        """
        return {
            'failed_channels': list(self.failed_channels),
            'active_channels': [c for c in self.channels if c not in self.failed_channels],
            'hardware_available': HARDWARE_AVAILABLE,
            'bus_ok': self.bus is not None,
            'discovered_sensors': {ch: f'0x{addr:02X}' for ch, addr in self._channel_addrs.items()},
            'cmm_initialized': list(self._cmm_initialized),
        }

    def validate_reading(self, reading):
        """
        Validate a sensor reading is within expected range.

        Args:
            reading: dict with 'x', 'y', 'z' keys

        Returns:
            bool: True if reading is valid
        """
        if 'error' in reading:
            return False

        max_range = SENSOR_CONFIG['MAGNETOMETER_RANGE_NT']
        for axis in ['x', 'y', 'z']:
            if abs(reading.get(axis, 0)) > max_range:
                return False
        return True

    def close(self):
        """Stop CMM on all channels and close the I2C bus connection."""
        if self.bus is not None:
            # Stop continuous measurement on all initialized channels
            for channel, addr in self._channel_addrs.items():
                if channel in self._cmm_initialized:
                    try:
                        self._select_mux_channel(channel)
                        self.bus.write_byte_data(addr, self.CMM, 0x00)
                    except Exception:
                        pass
            try:
                self.bus.close()
            except Exception:
                pass
