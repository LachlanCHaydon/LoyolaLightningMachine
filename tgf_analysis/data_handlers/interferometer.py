import os
import numpy as np
from typing import Optional, Tuple, Dict, List

from config import DEFAULT_COS_SHIFT_A, DEFAULT_COS_SHIFT_B, cmap_mjet


class InterferometerHandler:
    """
    Handles Interferometer (INTF) VHF radiation source location data.
    
    The INTF provides azimuth, elevation, and signal strength for VHF sources.
    Data can be in two formats:
    1. Raw (uncalibrated): Contains cosA, cosB that need cos-shift calibration
    2. Calibrated: Already has calibrated azimuth and elevation
    
    Time can be:
    - Relative to T0 (already in event coordinates)
    - Absolute (needs T0 offset added)
    """
    
    def __init__(self):
        """Initialize the InterferometerHandler."""
        self.filename: Optional[str] = None
        self.filepath: Optional[str] = None
        
        # Data type flags
        self.is_calibrated = True  # Whether file has pre-calibrated az/el
        self.time_is_relative = True  # Whether time is relative to T0
        
        # Calibration parameters (for raw data)
        self.cos_shift_a = 0
        self.cos_shift_b = 0
        
        # T0 reference for time conversion
        self.T0_reference = 0.0
        
        # Raw data from file
        self.raw_time: Optional[np.ndarray] = None  # Time as read from file
        self.raw_azimuth: Optional[np.ndarray] = None
        self.raw_elevation: Optional[np.ndarray] = None
        self.cos_a: Optional[np.ndarray] = None
        self.cos_b: Optional[np.ndarray] = None
        self.pk2pk: Optional[np.ndarray] = None  # Signal strength
        
        # Processed data (in event coordinates)
        self.time_array: Optional[np.ndarray] = None
        self.azimuth: Optional[np.ndarray] = None
        self.elevation: Optional[np.ndarray] = None
        
        # Visualization parameters
        self.colors: Optional[np.ndarray] = None
        self.marker_sizes: Optional[np.ndarray] = None
        
        # Data loaded flag
        self.is_loaded = False
    
    def load_data(self, filepath: str, is_calibrated: bool = True, 
                  skip_header: int = 2, T0_reference: float = 0.0,
                  time_is_relative: bool = True) -> bool:
        """
        Load INTF data from file.
        
        Parameters:
        -----------
        filepath : str
            Path to INTF data file
        is_calibrated : bool
            If True, file has calibrated az/el. If False, needs cos-shift.
        skip_header : int
            Number of header lines to skip
        T0_reference : float
            T0 value in µs for time conversion
        time_is_relative : bool
            If True, time in file is relative to trigger (µs)
            If False, time needs T0 added
        """
        try:
            self.filepath = filepath
            self.filename = os.path.basename(filepath)
            self.is_calibrated = is_calibrated
            self.time_is_relative = time_is_relative
            self.T0_reference = T0_reference
            
            # Load data - columns: time, az, el, cosA, cosB, pk2pk
            data = np.genfromtxt(filepath, skip_header=skip_header, 
                                usecols=(0, 1, 2, 3, 4, 5))
            
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            self.raw_time = data[:, 0]
            self.raw_azimuth = data[:, 1]
            self.raw_elevation = data[:, 2]
            self.cos_a = data[:, 3]
            self.cos_b = data[:, 4]
            self.pk2pk = data[:, 5]
            
            # Process data
            self._process_data()
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading INTF data: {e}")
            self.is_loaded = False
            return False
    
    def load_raw_intf(self, filepath: str, skip_header: int = 57,
                      T0_reference: float = 0.0) -> bool:
        """
        Load raw (uncalibrated) INTF data file.
        
        This is for the format with 57 header lines and time in ms.
        """
        try:
            self.filepath = filepath
            self.filename = os.path.basename(filepath)
            self.is_calibrated = False
            self.time_is_relative = False
            self.T0_reference = T0_reference
            
            data = np.genfromtxt(filepath, skip_header=skip_header,
                                usecols=(0, 1, 2, 3, 4, 5))
            
            # Raw file has time in ms
            self.raw_time = data[:, 0]  # Will be converted
            self.raw_azimuth = data[:, 1]
            self.raw_elevation = data[:, 2]
            self.cos_a = data[:, 3]
            self.cos_b = data[:, 4]
            self.pk2pk = data[:, 5]
            
            self._process_data()
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading raw INTF data: {e}")
            self.is_loaded = False
            return False
    
    def set_cos_shifts(self, cos_shift_a: float, cos_shift_b: float):
        """Set cosine shift calibration values and reprocess."""
        self.cos_shift_a = cos_shift_a
        self.cos_shift_b = cos_shift_b
        if self.is_loaded:
            self._process_data()
    
    def set_T0(self, T0_value: float):
        """Set T0 reference and recalculate time array."""
        self.T0_reference = T0_value
        if self.is_loaded:
            self._calculate_time_array()
    
    def _process_data(self):
        """Process loaded data (calibration and time conversion)."""
        if self.is_calibrated:
            # Use azimuth and elevation as-is
            self.azimuth = self.raw_azimuth.copy()
            self.elevation = self.raw_elevation.copy()
        else:
            # Apply cos-shift calibration
            self._apply_cos_shift_calibration()
        
        self._calculate_time_array()
        self._calculate_visualization_params()
    
    def _apply_cos_shift_calibration(self):
        """Apply cosine shift calibration to calculate az/el from cosA/cosB."""
        cosA_shifted = self.cos_a + self.cos_shift_a
        cosB_shifted = self.cos_b + self.cos_shift_b
        
        # Calculate azimuth
        self.azimuth = np.rad2deg(np.arctan2(cosB_shifted, cosA_shifted)) + 360
        self.azimuth = self.azimuth % 360  # Normalize to 0-360
        
        # Calculate elevation
        p = np.sqrt(cosA_shifted**2 + cosB_shifted**2)
        
        self.elevation = np.zeros_like(p)
        pos_mask = p <= 1.0
        neg_mask = p > 1.0
        
        self.elevation[pos_mask] = np.rad2deg(np.arccos(p[pos_mask]))
        self.elevation[neg_mask] = -np.rad2deg(np.arccos(2 - p[neg_mask]))
    
    def _calculate_time_array(self):
        """Calculate event time array."""
        if self.raw_time is None:
            return
        
        if self.time_is_relative:
            # Time is already in µs relative to trigger
            self.time_array = self.raw_time.copy()
        else:
            # Raw time is in ms, convert to µs and add T0
            self.time_array = self.raw_time * 1000 + self.T0_reference
    
    def _calculate_visualization_params(self):
        """Calculate colors and marker sizes based on signal strength."""
        if self.pk2pk is None or len(self.pk2pk) == 0:
            return
        
        N = len(self.pk2pk)
        
        # Normalize pk2pk on log scale
        ss = np.log10(np.clip(self.pk2pk, 1e-10, None))
        aMin = np.min(ss)
        aMax = np.max(ss)
        
        if aMax > aMin:
            ss_norm = (ss - aMin) / (aMax - aMin)
        else:
            ss_norm = np.zeros_like(ss)
        
        ss_norm = np.clip(ss_norm, 0, 1)
        
        # Calculate marker sizes
        s = (1 + 3 * ss_norm**2)**2
        self.marker_sizes = 6 * s
        
        # Calculate colors using mjet colormap
        max_val = np.max(ss_norm) if np.max(ss_norm) > 0 else 1
        self.colors = cmap_mjet(ss_norm / max_val)
    
    def calibrate_and_save(self, output_filepath: str, 
                           time_range: Tuple[float, float] = None) -> bool:
        """
        Apply calibration and save to new file.
        
        Parameters:
        -----------
        output_filepath : str
            Path for output calibrated file
        time_range : tuple, optional
            (t_start, t_stop) to filter data
        """
        if not self.is_loaded:
            return False
        
        try:
            # Apply time filter if specified
            if time_range is not None:
                mask = (self.time_array >= time_range[0]) & (self.time_array <= time_range[1])
            else:
                mask = np.ones(len(self.time_array), dtype=bool)
            
            with open(output_filepath, 'w') as f:
                f.write("########## Time [µs], Azimuth [deg], Elevation [deg], cosA, cosB, Pk2Pk [raw]\n")
                f.write("########## Data format: calibrated with cos shifts A={:.4f}, B={:.4f}\n".format(
                    self.cos_shift_a, self.cos_shift_b))
                
                for i in np.where(mask)[0]:
                    f.write(f"{self.time_array[i]:15.9f} {self.azimuth[i]:12.8f} "
                           f"{self.elevation[i]:13.8f} {self.cos_a[i]:9.2f} "
                           f"{self.cos_b[i]:6.2f} {self.pk2pk[i]:5.1f}\n")
            
            return True
            
        except Exception as e:
            print(f"Error saving calibrated INTF data: {e}")
            return False
    
    def filter_data(self, elv_min: float = None, elv_max: float = None,
                   azi_min: float = None, azi_max: float = None,
                   t_min: float = None, t_max: float = None) -> Dict:
        """
        Get filtered data as a dictionary.
        
        Returns new arrays, doesn't modify internal data.
        """
        if not self.is_loaded:
            return None
        
        mask = np.ones(len(self.time_array), dtype=bool)
        
        if elv_min is not None:
            mask &= (self.elevation >= elv_min)
        if elv_max is not None:
            mask &= (self.elevation <= elv_max)
        if azi_min is not None:
            mask &= (self.azimuth >= azi_min)
        if azi_max is not None:
            mask &= (self.azimuth <= azi_max)
        if t_min is not None:
            mask &= (self.time_array >= t_min)
        if t_max is not None:
            mask &= (self.time_array <= t_max)
        
        return {
            'time': self.time_array[mask],
            'azimuth': self.azimuth[mask],
            'elevation': self.elevation[mask],
            'pk2pk': self.pk2pk[mask],
            'colors': self.colors[mask] if self.colors is not None else None,
            'marker_sizes': self.marker_sizes[mask] if self.marker_sizes is not None else None
        }
    
    def get_data_in_range(self, t_min: float, t_max: float) -> Dict:
        """Get data within a time range."""
        return self.filter_data(t_min=t_min, t_max=t_max)
    
    def get_full_data(self) -> Dict:
        """Get all data."""
        if not self.is_loaded:
            return None
        
        return {
            'time': self.time_array,
            'azimuth': self.azimuth,
            'elevation': self.elevation,
            'pk2pk': self.pk2pk,
            'colors': self.colors,
            'marker_sizes': self.marker_sizes
        }
    
    def get_time_range(self) -> Tuple[float, float]:
        """Get the full time range of the data."""
        if not self.is_loaded:
            return None, None
        return self.time_array[0], self.time_array[-1]
    
    def get_info(self) -> str:
        """Get summary information about loaded data."""
        if not self.is_loaded:
            return "No data loaded"
        
        t_min, t_max = self.get_time_range()
        duration_ms = (t_max - t_min) / 1000
        
        cal_status = "Calibrated" if self.is_calibrated else f"Raw (cos shifts: A={self.cos_shift_a}, B={self.cos_shift_b})"
        
        return (
            f"File: {self.filename}\n"
            f"Status: {cal_status}\n"
            f"T0 Reference: {self.T0_reference:.3f} µs\n"
            f"Time Range: {t_min:.0f} - {t_max:.0f} µs\n"
            f"Duration: {duration_ms:.1f} ms\n"
            f"Points: {len(self.time_array):,}\n"
            f"Elevation: {np.min(self.elevation):.1f}° - {np.max(self.elevation):.1f}°\n"
            f"Azimuth: {np.min(self.azimuth):.1f}° - {np.max(self.azimuth):.1f}°"
        )
