"""
TGF Analysis Tool - Photometer Data Handler
===========================================
Handles loading, calibration, and processing of photometer data.
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict

from config import (
    PHOTOMETER_CALIBRATION, 
    PHOTOMETER_SAMPLE_RATE, 
    PHOTOMETER_RECORD_LENGTH
)


class PhotometerHandler:
    """
    Handles loading, calibration, and processing of photometer data.
    
    The photometer system has 3 channels:
    - Channel 0: 337nm (Blue/UV - NII emission)
    - Channel 1: 391nm (Violet - NII emission)  
    - Channel 2: 777nm (Red/NIR - OI emission)
    
    The ratio of blue to red indicates ionization level/plasma temperature.
    
    TIME ALIGNMENT:
    The photometer may trigger at a different second than other instruments.
    We store the NATIVE photometer time and use second_offset to select data
    in EVENT time coordinates, then shift back for plotting.
    """
    
    def __init__(self):
        """Initialize the PhotometerHandler."""
        self.filename: Optional[str] = None
        self.filepath: Optional[str] = None
        
        # Timing information extracted from filename
        self.date_str: Optional[str] = None
        self.time_str: Optional[str] = None
        self.trigger_microseconds: Optional[float] = None
        
        # Second offset for time alignment
        self.second_offset = 0
        
        # Raw and calibrated data
        self.raw_channels = [None, None, None, None]  # ch0, ch1, ch2, ch3 (camera)
        self.calibrated_channels = [None, None, None]  # ch0, ch1, ch2 only
        
        # Native time array
        self.native_time_array: Optional[np.ndarray] = None
        
        # Sampling parameters
        self.sample_rate = PHOTOMETER_SAMPLE_RATE
        self.record_length = PHOTOMETER_RECORD_LENGTH
        
        # Calculate calibration factors
        self.cal_factors = self._calculate_cal_factors()
        
        # Data loaded flag
        self.is_loaded = False
        
    def _calculate_cal_factors(self) -> Dict[int, float]:
        """Calculate calibration factors for each channel."""
        factors = {}
        for ch, params in PHOTOMETER_CALIBRATION.items():
            factor = 1.0 / (params['rsk'] * params['G'] * params['R'] * 
                          params['A'] * params['T']) * 1e6
            factors[ch] = factor
        return factors
    
    def parse_filename(self, filename: str) -> bool:
        """
        Parse photometer filename to extract timing information.
        
        Expected format: YYYYMMDD-HHMMSS.TTTTTTTT_PXI_P19.dat
        Where TTTTTTTT is the trigger time in units of 10ns
        """
        self.filename = os.path.basename(filename)
        
        try:
            parts = self.filename.split('.')
            date_time = parts[0]
            
            self.date_str = date_time.split('-')[0]
            time_part = date_time.split('-')[1]
            self.time_str = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
            
            trigger_part = parts[1].split('_')[0]
            self.trigger_microseconds = int(trigger_part) * 0.01
            
            return True
        except Exception as e:
            print(f"Error parsing filename: {e}")
            return False
    
    def load_binary_data(self, filepath: str, sample_rate: float = None) -> bool:
        """
        Load photometer data from binary .dat file.
        
        The file contains 4 channels of float64 data:
        - ch0: 337nm photometer
        - ch1: 391nm photometer
        - ch2: 777nm photometer
        - ch3: Camera sync signal
        """
        try:
            self.filepath = filepath
            self.parse_filename(filepath)
            
            if sample_rate is not None:
                self.sample_rate = sample_rate
                self.record_length = int(sample_rate)
            
            with open(filepath, 'rb') as fid:
                for ch in range(4):
                    data = np.fromfile(fid, dtype=np.float64, count=self.record_length)
                    if len(data) < self.record_length:
                        print(f"Warning: Channel {ch} has only {len(data)} samples")
                        data = np.pad(data, (0, self.record_length - len(data)))
                    self.raw_channels[ch] = data
            
            # Create native time array
            dt = 1.0 / self.sample_rate
            t = np.arange(1, self.record_length + 1) * dt * 1e6  # Convert to µs
            
            if self.trigger_microseconds is not None:
                t = t + self.trigger_microseconds
            
            self.native_time_array = t
            
            # Calibrate channels
            self._calibrate_channels()
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading photometer data: {e}")
            self.is_loaded = False
            return False
    
    def _calibrate_channels(self):
        """Apply calibration to raw channels."""
        for ch in range(3):
            if self.raw_channels[ch] is not None:
                # Subtract baseline (first few samples) and apply calibration
                baseline = self.raw_channels[ch][5] if len(self.raw_channels[ch]) > 5 else 0
                self.calibrated_channels[ch] = -self.cal_factors[ch] * (
                    self.raw_channels[ch] - baseline
                )
    
    def set_second_offset(self, offset: int):
        """
        Set the second offset for time alignment.
        
        Parameters:
        -----------
        offset : int
            Number of seconds difference between photometer and event time.
            Positive means photometer triggered earlier.
        """
        self.second_offset = offset
    
    def get_data_in_event_time(self, t_start: float, t_stop: float, 
                                downsample: int = 1) -> Dict:
        """
        Get photometer data in event time coordinates.
        
        Parameters:
        -----------
        t_start, t_stop : float
            Time range in event time (µs)
        downsample : int
            Downsampling factor
            
        Returns:
        --------
        dict with keys: 'time', 'ch0', 'ch1', 'ch2'
        """
        if not self.is_loaded:
            return None
        
        # Convert event time to photometer time
        offset_us = self.second_offset * 1e6
        phot_t_start = t_start + offset_us
        phot_t_stop = t_stop + offset_us
        
        # Find indices
        mask = (self.native_time_array >= phot_t_start) & (self.native_time_array <= phot_t_stop)
        
        # Get data and convert time back to event coordinates
        result = {
            'time': self.native_time_array[mask][::downsample] - offset_us,
            'ch0': self.calibrated_channels[0][mask][::downsample] if self.calibrated_channels[0] is not None else None,
            'ch1': self.calibrated_channels[1][mask][::downsample] if self.calibrated_channels[1] is not None else None,
            'ch2': self.calibrated_channels[2][mask][::downsample] if self.calibrated_channels[2] is not None else None,
        }
        
        return result
    
    def get_ratios_in_range(self, t_start: float, t_stop: float) -> Dict:
        """
        Calculate channel ratios in a time range.
        
        Returns:
        --------
        dict with keys: 'ratio_337_777', 'ratio_391_777', 'ratio_337_391'
        """
        data = self.get_data_in_event_time(t_start, t_stop)
        if data is None:
            return {}
        
        result = {}
        if data['ch0'] is not None and data['ch2'] is not None:
            ch0_max = np.max(data['ch0'])
            ch2_max = np.max(data['ch2'])
            if ch2_max > 0:
                result['ratio_337_777'] = ch0_max / ch2_max
        
        if data['ch1'] is not None and data['ch2'] is not None:
            ch1_max = np.max(data['ch1'])
            ch2_max = np.max(data['ch2'])
            if ch2_max > 0:
                result['ratio_391_777'] = ch1_max / ch2_max
        
        if data['ch0'] is not None and data['ch1'] is not None:
            ch0_max = np.max(data['ch0'])
            ch1_max = np.max(data['ch1'])
            if ch1_max > 0:
                result['ratio_337_391'] = ch0_max / ch1_max
        
        return result
    
    def get_time_range(self) -> Tuple[float, float]:
        """Get the full time range of the data in native photometer time."""
        if not self.is_loaded:
            return None, None
        return self.native_time_array[0], self.native_time_array[-1]
    
    def get_info(self) -> str:
        """Get summary information about loaded data."""
        if not self.is_loaded:
            return "No data loaded"
        
        t_min, t_max = self.get_time_range()
        duration_ms = (t_max - t_min) / 1000
        
        return (
            f"File: {self.filename}\n"
            f"Trigger: {self.trigger_microseconds:.2f} µs\n"
            f"Second Offset: {self.second_offset}\n"
            f"Native Time Range: {t_min:.0f} - {t_max:.0f} µs\n"
            f"Duration: {duration_ms:.1f} ms\n"
            f"Sample Rate: {self.sample_rate/1e6:.1f} MHz"
        )
