import os
import numpy as np
from typing import Optional, Tuple, Dict, List


class LMAHandler:
    """
    Handles LMA (Lightning Mapping Array) VHF source location data.
    
    LMA provides 3D locations of VHF radiation sources with:
    - Time (seconds of day)
    - Latitude, Longitude
    - Altitude (meters)
    - Reduced chi-squared (fit quality)
    - Power (dBW)
    """
    
    def __init__(self):
        """Initialize the LMAHandler."""
        self.filename: Optional[str] = None
        self.filepath: Optional[str] = None
        
        # Data arrays
        self.time: Optional[np.ndarray] = None  # Seconds of day
        self.latitude: Optional[np.ndarray] = None
        self.longitude: Optional[np.ndarray] = None
        self.altitude: Optional[np.ndarray] = None  # meters
        self.chi_squared: Optional[np.ndarray] = None
        self.power: Optional[np.ndarray] = None
        
        # Data loaded flag
        self.is_loaded = False
    
    def load_data(self, filepath: str, skip_header: int = 57) -> bool:
        """
        Load LMA data from file.
        
        Parameters:
        -----------
        filepath : str
            Path to LMA data file
        skip_header : int
            Number of header lines to skip (default: 57 for standard format)
        """
        try:
            self.filepath = filepath
            self.filename = os.path.basename(filepath)
            
            # Load data - columns: time, lat, lon, alt, chi2, power
            data = np.genfromtxt(filepath, skip_header=skip_header,
                                usecols=(0, 1, 2, 3, 4, 5))
            
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            self.time = data[:, 0]
            self.latitude = data[:, 1]
            self.longitude = data[:, 2]
            self.altitude = data[:, 3]
            self.chi_squared = data[:, 4]
            self.power = data[:, 5]
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading LMA data: {e}")
            self.is_loaded = False
            return False
    
    def load_cut_file(self, filepath: str, skip_header: int = 2) -> bool:
        """
        Load pre-cut LMA data file (fewer header lines).
        """
        return self.load_data(filepath, skip_header=skip_header)
    
    def filter_by_time(self, event_time_sec: float, time_range_us: float = 1000,
                      chi_limit: float = 2.0, 
                      alt_min: float = 0, alt_max: float = 10000) -> Dict:
        """
        Filter LMA data around an event time.
        
        Parameters:
        -----------
        event_time_sec : float
            Event time in seconds of day (e.g., hour*3600 + min*60 + sec)
        time_range_us : float
            Time range in microseconds around event
        chi_limit : float
            Maximum chi-squared value to include
        alt_min, alt_max : float
            Altitude range in meters
            
        Returns:
        --------
        dict with filtered data
        """
        if not self.is_loaded:
            return None
        
        time_range_sec = time_range_us * 1e-6
        
        mask = (
            (np.abs(self.time - event_time_sec) < time_range_sec) &
            (self.chi_squared < chi_limit) &
            (self.altitude >= alt_min) &
            (self.altitude <= alt_max)
        )
        
        return {
            'time': self.time[mask],
            'latitude': self.latitude[mask],
            'longitude': self.longitude[mask],
            'altitude': self.altitude[mask],
            'chi_squared': self.chi_squared[mask],
            'power': self.power[mask],
            'count': np.sum(mask)
        }
    
    def filter_by_location(self, lat_min: float = None, lat_max: float = None,
                          lon_min: float = None, lon_max: float = None) -> Dict:
        """Filter LMA data by geographic location."""
        if not self.is_loaded:
            return None
        
        mask = np.ones(len(self.time), dtype=bool)
        
        if lat_min is not None:
            mask &= (self.latitude >= lat_min)
        if lat_max is not None:
            mask &= (self.latitude <= lat_max)
        if lon_min is not None:
            mask &= (self.longitude >= lon_min)
        if lon_max is not None:
            mask &= (self.longitude <= lon_max)
        
        return {
            'time': self.time[mask],
            'latitude': self.latitude[mask],
            'longitude': self.longitude[mask],
            'altitude': self.altitude[mask],
            'chi_squared': self.chi_squared[mask],
            'power': self.power[mask],
            'count': np.sum(mask)
        }
    
    def get_mean_location(self, event_time_sec: float, 
                         time_range_us: float = 1000,
                         chi_limit: float = 2.0) -> Dict:
        """
        Get mean location of LMA sources near an event.
        
        Returns:
        --------
        dict with mean lat, lon, alt and standard deviations
        """
        filtered = self.filter_by_time(event_time_sec, time_range_us, chi_limit)
        
        if filtered is None or filtered['count'] == 0:
            return None
        
        return {
            'lat_mean': np.mean(filtered['latitude']),
            'lat_std': np.std(filtered['latitude']),
            'lon_mean': np.mean(filtered['longitude']),
            'lon_std': np.std(filtered['longitude']),
            'alt_mean': np.mean(filtered['altitude']),
            'alt_std': np.std(filtered['altitude']),
            'count': filtered['count']
        }
    
    def save_cut_file(self, output_filepath: str, event_time_sec: float,
                     time_range_us: float = 1000, chi_limit: float = 2.0,
                     alt_min: float = 0, alt_max: float = 10000) -> bool:
        """
        Save filtered LMA data to a new file.
        """
        filtered = self.filter_by_time(event_time_sec, time_range_us, 
                                       chi_limit, alt_min, alt_max)
        
        if filtered is None or filtered['count'] == 0:
            print("No data to save after filtering")
            return False
        
        try:
            with open(output_filepath, 'w') as f:
                f.write("########## Data: time (UT sec of day), lat, lon, alt(m), chi^2, P(dBW)\n")
                f.write("########## Filtered LMA data\n")
                
                for i in range(filtered['count']):
                    f.write(f"{filtered['time'][i]:15.9f} "
                           f"{filtered['latitude'][i]:12.8f} "
                           f"{filtered['longitude'][i]:13.8f} "
                           f"{filtered['altitude'][i]:9.2f} "
                           f"{filtered['chi_squared'][i]:6.2f} "
                           f"{filtered['power'][i]:5.1f}\n")
            
            return True
            
        except Exception as e:
            print(f"Error saving LMA cut file: {e}")
            return False
    
    def get_full_data(self) -> Dict:
        """Get all loaded data."""
        if not self.is_loaded:
            return None
        
        return {
            'time': self.time,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'chi_squared': self.chi_squared,
            'power': self.power
        }
    
    def get_time_range(self) -> Tuple[float, float]:
        """Get the time range of the data (in seconds of day)."""
        if not self.is_loaded:
            return None, None
        return self.time[0], self.time[-1]
    
    def get_info(self) -> str:
        """Get summary information about loaded data."""
        if not self.is_loaded:
            return "No data loaded"
        
        t_min, t_max = self.get_time_range()
        
        return (
            f"File: {self.filename}\n"
            f"Points: {len(self.time):,}\n"
            f"Time Range: {t_min:.6f} - {t_max:.6f} sec of day\n"
            f"Latitude: {np.min(self.latitude):.4f}° - {np.max(self.latitude):.4f}°\n"
            f"Longitude: {np.min(self.longitude):.4f}° - {np.max(self.longitude):.4f}°\n"
            f"Altitude: {np.min(self.altitude):.0f} - {np.max(self.altitude):.0f} m\n"
            f"Chi² Range: {np.min(self.chi_squared):.2f} - {np.max(self.chi_squared):.2f}"
        )
