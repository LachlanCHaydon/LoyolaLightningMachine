
import os
import numpy as np
from typing import Optional, Tuple, Dict

# Try to import pandas for CSV loading
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class FastAntennaHandler:
    """
    Handles Fast Antenna (FA) electric field measurement data.
    
    The FA measures the vertical electric field change during lightning.
    It typically provides the master time reference (T0) for event alignment.
    
    Data Format:
    - CSV file with columns 'Time [ms]' and 'E [V/m]'
    - T0 trigger value is encoded in the filename
    """
    
    def __init__(self):
        """Initialize the FastAntennaHandler."""
        self.filename: Optional[str] = None
        self.filepath: Optional[str] = None
        
        # Timing
        self.T0_trigger: float = 0.0  # Trigger offset in µs (from filename)
        self.T0_manual: Optional[float] = None  # Manual override
        
        # Data arrays
        self.raw_time_ms: Optional[np.ndarray] = None  # Time in ms from file
        self.time_array: Optional[np.ndarray] = None  # Time in µs (event coordinates)
        self.electric_field: Optional[np.ndarray] = None  # E [V/m]
        
        # Data loaded flag
        self.is_loaded = False
    
    def parse_filename(self, filename: str) -> bool:
        """
        Parse FA filename to extract T0 trigger value.
        
        Expected format: TAR_YYYYMMDD_HHMMSS_TTTTTT_T0.csv
        Where TTTTTT is the trigger time in µs within the second.
        
        Alternative format after decimal: ...HHMMSS.TTTTTT...
        """
        self.filename = os.path.basename(filename)
        
        try:
            # Try format: TAR_YYYYMMDD_HHMMSS_TTTTTT_T0.csv
            parts = self.filename.replace('.csv', '').split('_')
            
            # Look for the trigger value (usually a 6-digit number)
            for part in parts:
                if part.isdigit() and len(part) == 6:
                    self.T0_trigger = float(part)
                    return True
            
            # Try alternative: look in CSV header
            return True  # Will try to get from header later
            
        except Exception as e:
            print(f"Error parsing FA filename: {e}")
            return False
    
    def load_csv(self, filepath: str, skip_rows: int = 6) -> bool:
        """
        Load Fast Antenna data from CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        skip_rows : int
            Number of header rows to skip (default: 6)
        """
        try:
            self.filepath = filepath
            self.parse_filename(filepath)
            
            if HAS_PANDAS:
                # Use pandas for robust CSV parsing
                df = pd.read_csv(filepath, skiprows=list(range(skip_rows)))
                
                # Find time and E-field columns
                time_col = None
                e_col = None
                for col in df.columns:
                    if 'time' in col.lower():
                        time_col = col
                    if 'e' in col.lower() and 'v/m' in col.lower():
                        e_col = col
                
                if time_col is None:
                    time_col = df.columns[0]
                if e_col is None:
                    e_col = df.columns[1]
                
                self.raw_time_ms = df[time_col].values
                self.electric_field = df[e_col].values
                
                # Try to extract T0 from header if not in filename
                if self.T0_trigger == 0:
                    try:
                        with open(filepath, 'r') as f:
                            for i, line in enumerate(f):
                                if i >= skip_rows:
                                    break
                                # Look for trigger time in header
                                if 'trigger' in line.lower() or 't0' in line.lower():
                                    # Extract number
                                    import re
                                    numbers = re.findall(r'\d+\.?\d*', line)
                                    if numbers:
                                        self.T0_trigger = float(numbers[-1])
                    except:
                        pass
            else:
                # Fallback to numpy
                data = np.genfromtxt(filepath, delimiter=',', skip_header=skip_rows)
                self.raw_time_ms = data[:, 0]
                self.electric_field = data[:, 1]
            
            # Convert to event time coordinates
            self._calculate_time_array()
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading FA data: {e}")
            self.is_loaded = False
            return False
    
    def set_T0(self, T0_value: float):
        """
        Manually set the T0 reference time.
        
        Parameters:
        -----------
        T0_value : float
            T0 trigger time in microseconds
        """
        self.T0_manual = T0_value
        if self.is_loaded:
            self._calculate_time_array()
    
    def _calculate_time_array(self):
        """Calculate event time array from raw time and T0."""
        if self.raw_time_ms is None:
            return
        
        T0 = self.T0_manual if self.T0_manual is not None else self.T0_trigger
        
        # Convert ms to µs and add T0
        self.time_array = self.raw_time_ms * 1000 + T0
    
    def get_T0(self) -> float:
        """Get the current T0 value (manual override or from filename)."""
        return self.T0_manual if self.T0_manual is not None else self.T0_trigger
    
    def get_data_in_range(self, t_start: float, t_stop: float) -> Dict:
        """
        Get FA data within a time range.
        
        Parameters:
        -----------
        t_start, t_stop : float
            Time range in µs
            
        Returns:
        --------
        dict with keys: 'time', 'e_field'
        """
        if not self.is_loaded:
            return None
        
        mask = (self.time_array >= t_start) & (self.time_array <= t_stop)
        
        return {
            'time': self.time_array[mask],
            'e_field': self.electric_field[mask]
        }
    
    def get_full_data(self) -> Dict:
        """Get all FA data."""
        if not self.is_loaded:
            return None
        
        return {
            'time': self.time_array,
            'e_field': self.electric_field
        }
    
    def get_time_range(self) -> Tuple[float, float]:
        """Get the full time range of the data."""
        if not self.is_loaded:
            return None, None
        return self.time_array[0], self.time_array[-1]
    
    def find_return_stroke(self, threshold: float = -10.0) -> Optional[float]:
        """
        Find the return stroke time based on E-field minimum.
        
        Parameters:
        -----------
        threshold : float
            Minimum E-field value to consider as RS
            
        Returns:
        --------
        float : Time of return stroke in µs, or None
        """
        if not self.is_loaded:
            return None
        
        # Find the minimum (most negative) E-field
        min_idx = np.argmin(self.electric_field)
        if self.electric_field[min_idx] < threshold:
            return self.time_array[min_idx]
        return None
    
    def get_info(self) -> str:
        """Get summary information about loaded data."""
        if not self.is_loaded:
            return "No data loaded"
        
        t_min, t_max = self.get_time_range()
        duration_ms = (t_max - t_min) / 1000
        e_min, e_max = np.min(self.electric_field), np.max(self.electric_field)
        
        return (
            f"File: {self.filename}\n"
            f"T0: {self.get_T0():.3f} µs\n"
            f"Time Range: {t_min:.0f} - {t_max:.0f} µs\n"
            f"Duration: {duration_ms:.1f} ms\n"
            f"E-field Range: {e_min:.1f} to {e_max:.1f} V/m\n"
            f"Points: {len(self.time_array):,}"
        )
