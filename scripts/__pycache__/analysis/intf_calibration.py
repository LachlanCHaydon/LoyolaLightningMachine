"""
TGF Analysis Tool - INTF Calibration
====================================
Handles cosine-shift calibration for raw INTF data.
"""

import numpy as np
from typing import Optional, Tuple
import os

from config import DEFAULT_COS_SHIFT_A, DEFAULT_COS_SHIFT_B


class INTFCalibrator:
    """
    Calibrates raw INTF data by applying cosine shifts to
    convert cosA/cosB to calibrated azimuth and elevation.
    """
    
    def __init__(self, cos_shift_a: float = DEFAULT_COS_SHIFT_A,
                 cos_shift_b: float = DEFAULT_COS_SHIFT_B):
        """
        Initialize the calibrator.
        
        Parameters:
        -----------
        cos_shift_a : float
            Cosine shift for A component
        cos_shift_b : float
            Cosine shift for B component
        """
        self.cos_shift_a = cos_shift_a
        self.cos_shift_b = cos_shift_b
    
    def set_cos_shifts(self, cos_shift_a: float, cos_shift_b: float):
        """Update cosine shift values."""
        self.cos_shift_a = cos_shift_a
        self.cos_shift_b = cos_shift_b
    
    def calibrate_angles(self, cos_a: np.ndarray, 
                        cos_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply cosine shifts and calculate azimuth/elevation.
        
        Parameters:
        -----------
        cos_a : np.ndarray
            Raw cosine A values
        cos_b : np.ndarray
            Raw cosine B values
            
        Returns:
        --------
        tuple : (azimuth, elevation) arrays in degrees
        """
        # Apply shifts
        cos_a_shifted = cos_a + self.cos_shift_a
        cos_b_shifted = cos_b + self.cos_shift_b
        
        # Calculate azimuth
        azimuth = np.rad2deg(np.arctan2(cos_b_shifted, cos_a_shifted)) + 360
        azimuth = azimuth % 360
        
        # Calculate elevation
        p = np.sqrt(cos_a_shifted**2 + cos_b_shifted**2)
        
        elevation = np.zeros_like(p)
        pos_mask = p <= 1.0
        neg_mask = p > 1.0
        
        # Normal case: p <= 1
        elevation[pos_mask] = np.rad2deg(np.arccos(p[pos_mask]))
        
        # Edge case: p > 1 (below horizon)
        elevation[neg_mask] = -np.rad2deg(np.arccos(2 - p[neg_mask]))
        
        return azimuth, elevation
    
    def calibrate_file(self, input_filepath: str, output_filepath: str,
                      T0_reference: float = 0.0,
                      time_range: Tuple[float, float] = None,
                      skip_header: int = 0) -> bool:
        """
        Calibrate a raw INTF file and save the result.
        
        Parameters:
        -----------
        input_filepath : str
            Path to raw INTF data file
        output_filepath : str
            Path for calibrated output file
        T0_reference : float
            T0 trigger time in µs to add to time values
        time_range : tuple, optional
            (t_min, t_max) to filter output
        skip_header : int
            Number of header lines to skip in input
        """
        try:
            # Load raw data
            data = np.genfromtxt(input_filepath, skip_header=skip_header,
                                usecols=(0, 1, 2, 3, 4, 5))
            
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            raw_time = data[:, 0]
            cos_a = data[:, 3]
            cos_b = data[:, 4]
            pk2pk = data[:, 5]
            
            # Apply calibration
            azimuth, elevation = self.calibrate_angles(cos_a, cos_b)
            
            # Convert time (raw is typically in ms or seconds)
            # Multiply by 1000 to get µs, then add T0
            time_us = raw_time * 1000 + T0_reference
            
            # Apply time filter if specified
            if time_range is not None:
                mask = (time_us >= time_range[0]) & (time_us <= time_range[1])
            else:
                mask = np.ones(len(time_us), dtype=bool)
            
            # Write calibrated file
            with open(output_filepath, 'w') as f:
                f.write("########## Time [µs], Azimuth [deg], Elevation [deg], cosA, cosB, Pk2Pk [raw]\n")
                f.write(f"########## Calibrated with cos shifts: A={self.cos_shift_a:.4f}, B={self.cos_shift_b:.4f}\n")
                
                for i in np.where(mask)[0]:
                    f.write(f"{time_us[i]:15.9f} {azimuth[i]:12.8f} "
                           f"{elevation[i]:13.8f} {cos_a[i]:9.4f} "
                           f"{cos_b[i]:9.4f} {pk2pk[i]:8.1f}\n")
            
            return True
            
        except Exception as e:
            print(f"Error calibrating INTF file: {e}")
            return False
    
    def preview_calibration(self, cos_a: np.ndarray, 
                           cos_b: np.ndarray) -> dict:
        """
        Preview calibration results without saving.
        
        Returns statistics about the calibrated data.
        """
        azimuth, elevation = self.calibrate_angles(cos_a, cos_b)
        
        return {
            'azimuth_min': np.min(azimuth),
            'azimuth_max': np.max(azimuth),
            'azimuth_mean': np.mean(azimuth),
            'elevation_min': np.min(elevation),
            'elevation_max': np.max(elevation),
            'elevation_mean': np.mean(elevation),
            'n_below_horizon': np.sum(elevation < 0),
            'n_points': len(azimuth),
        }
    
    @staticmethod
    def suggest_cos_shifts(known_source_az: float, known_source_el: float,
                          measured_cos_a: float, measured_cos_b: float) -> Tuple[float, float]:
        """
        Suggest cosine shifts based on a known source location.
        
        If you know where a source should appear (e.g., from LMA),
        this can help determine the needed cos shifts.
        
        Parameters:
        -----------
        known_source_az : float
            Known azimuth in degrees
        known_source_el : float
            Known elevation in degrees
        measured_cos_a : float
            Measured cosA value
        measured_cos_b : float
            Measured cosB value
            
        Returns:
        --------
        tuple : (suggested_cos_shift_a, suggested_cos_shift_b)
        """
        # Calculate what cosA and cosB should be
        p_target = np.cos(np.radians(known_source_el))
        angle_rad = np.radians(known_source_az - 360 if known_source_az > 180 else known_source_az)
        
        target_cos_a = p_target * np.cos(angle_rad)
        target_cos_b = p_target * np.sin(angle_rad)
        
        # Calculate needed shifts
        shift_a = target_cos_a - measured_cos_a
        shift_b = target_cos_b - measured_cos_b
        
        return shift_a, shift_b


def calibrate_intf_file(input_path: str, output_path: str,
                       T0: float, cos_shift_a: float = DEFAULT_COS_SHIFT_A,
                       cos_shift_b: float = DEFAULT_COS_SHIFT_B,
                       time_range: Tuple[float, float] = None) -> bool:
    """
    Convenience function to calibrate an INTF file.
    
    Parameters:
    -----------
    input_path : str
        Path to raw INTF file
    output_path : str
        Path for calibrated output
    T0 : float
        T0 reference time in µs
    cos_shift_a, cos_shift_b : float
        Cosine shift calibration values
    time_range : tuple, optional
        Time range filter
    """
    calibrator = INTFCalibrator(cos_shift_a, cos_shift_b)
    return calibrator.calibrate_file(input_path, output_path, T0, time_range)
