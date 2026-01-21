import os
import numpy as np
from typing import Optional, Tuple, Dict, List
import geopy

from utils.ta_tools import load_tasd_coordinates, calculate_azimuth_to_detector


class TASDHandler:
    """
    Handles TASD (Surface Detector) waveform and burst data.
    
    The TASD detects gamma-ray and particle showers from TGFs.
    Data includes:
    - Waveform data (FADC counts vs time)
    - Burst detection times and VEM (energy) values
    - Detector GPS coordinates
    """
    
    def __init__(self):
        """Initialize the TASDHandler."""
        self.directory: Optional[str] = None
        self.time_str: Optional[str] = None  # HHMMSS format
        
        # Loaded detector data
        self.detectors: List[Dict] = []  # List of detector data dicts
        
        # Global TASD coordinates
        self.detector_coords: Dict = {}  # {det_id: {lat, lon, alt, ...}}
        
        # Time shift for alignment
        self.time_shift = 0.0
        
        # Data loaded flag
        self.is_loaded = False
    
    def load_coordinates(self, filepath: str = 'tasd_gpscoors.txt'):
        """Load TASD detector coordinates."""
        self.detector_coords = load_tasd_coordinates(filepath)
    
    def load_directory(self, directory: str, time_str: str,
                      time_shift: float = 0.0,
                      trig_num: int = 1,
                      manual_start: bool = False,
                      sd_start_time: float = 0.0) -> bool:
        """
        Load all TASD data from a directory.
        
        Parameters:
        -----------
        directory : str
            Path to directory containing TASD .txt or .stitch files
        time_str : str
            Time string (HHMMSS) to match files
        time_shift : float
            Time shift to apply in µs
        trig_num : int
            Which burst/trigger to extract (1, 2, 3, etc.)
        manual_start : bool
            Whether to use manual start time filtering
        sd_start_time : float
            Manual start time for filtering
        """
        try:
            self.directory = directory
            self.time_str = time_str
            self.time_shift = time_shift
            self.detectors = []
            
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    if filename.endswith('.txt') or filename.endswith('.stitch'):
                        filepath = os.path.join(root, filename)
                        
                        # Check if this file matches our time
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            file_time = parts[1] if parts[0].startswith('SD') else None
                            
                            if file_time == time_str or filename.endswith('.stitch'):
                                det_data = self._load_single_file(
                                    filepath, trig_num, manual_start, 
                                    sd_start_time, time_shift
                                )
                                if det_data is not None:
                                    self.detectors.append(det_data)
            
            self.is_loaded = len(self.detectors) > 0
            return self.is_loaded
            
        except Exception as e:
            print(f"Error loading TASD directory: {e}")
            self.is_loaded = False
            return False
    
    def _load_single_file(self, filepath: str, trig_num: int,
                         manual_start: bool, sd_start_time: float,
                         time_shift: float) -> Optional[Dict]:
        """Load a single TASD file."""
        try:
            filename = os.path.basename(filepath)
            
            # Extract detector ID from filename
            if filename.endswith('.stitch'):
                det_id = filename.split('_')[-1][:4]
            else:
                parts = filename.split('_')
                det_id = parts[-1][:4] if len(parts) >= 3 else None
            
            if det_id is None:
                return None
            
            # Read file
            time_data = []
            signal_lower = []
            signal_upper = []
            
            header_data = {
                'burst_times': [],
                'burst_vem': [],
                'pedestal_l': 0,
                'pedestal_u': 0,
            }
            
            header = True
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('###'):
                        continue
                    
                    if header:
                        if line.startswith('burst times'):
                            parts = line.replace(',', '').split()[3:]
                            header_data['burst_times'] = [int(x) for x in parts]
                        elif line.startswith('burst VEM'):
                            parts = line.replace(',', '').split()[3:]
                            header_data['burst_vem'] = [float(x) for x in parts]
                        elif line.startswith('pedestal'):
                            parts = line.split()
                            if len(parts) >= 4:
                                header_data['pedestal_l'] = float(parts[2].strip(','))
                                header_data['pedestal_u'] = float(parts[3])
                            else:
                                header_data['pedestal_l'] = float(parts[2])
                                header_data['pedestal_u'] = header_data['pedestal_l']
                        elif line.startswith('us'):
                            header = False
                        continue
                    
                    # Waveform data
                    if line.strip() == '':
                        continue
                    
                    cols = line.replace(',', ' ').split()
                    if len(cols) < 3:
                        continue
                    
                    t = float(cols[0])
                    sig_l = float(cols[1])
                    sig_u = float(cols[2])
                    
                    # Skip waveform separators
                    if sig_l == 0.0 and sig_u == 0.0:
                        continue
                    
                    # Get trigger time
                    if header_data['burst_times']:
                        trig_time = header_data['burst_times'][0]
                        
                        # Apply time filtering
                        abs_time = trig_time + t
                        
                        if manual_start and abs_time < sd_start_time:
                            continue
                        
                        # Filter by trigger number
                        if trig_num <= len(header_data['burst_times']):
                            burst_start = header_data['burst_times'][trig_num - 1]
                            if abs_time < burst_start:
                                continue
                            if trig_num < len(header_data['burst_times']):
                                burst_end = header_data['burst_times'][trig_num]
                                if abs_time > burst_end:
                                    break
                        
                        time_data.append(abs_time + time_shift)
                        signal_lower.append(sig_l)
                        signal_upper.append(sig_u)
            
            if len(time_data) == 0:
                return None
            
            # Get detector coordinates
            coords = self.detector_coords.get(det_id, {})
            
            # Calculate VEM for this trigger
            vem = 0
            if trig_num <= len(header_data['burst_vem']):
                vem = header_data['burst_vem'][trig_num - 1]
            
            return {
                'detector_id': det_id,
                'time': np.array(time_data),
                'signal_lower': np.array(signal_lower),
                'signal_upper': np.array(signal_upper),
                'vem': vem,
                'burst_times': header_data['burst_times'],
                'burst_vem': header_data['burst_vem'],
                'lat': coords.get('lat'),
                'lon': coords.get('lon'),
                'alt': coords.get('alt'),
            }
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def get_all_waveforms(self) -> List[Dict]:
        """Get waveform data for all loaded detectors."""
        return self.detectors
    
    def get_detector_waveform(self, det_id: str) -> Optional[Dict]:
        """Get waveform data for a specific detector."""
        for det in self.detectors:
            if det['detector_id'] == det_id:
                return det
        return None
    
    def get_detector_ids(self) -> List[str]:
        """Get list of loaded detector IDs."""
        return [d['detector_id'] for d in self.detectors]
    
    def get_total_vem(self) -> float:
        """Get total VEM across all detectors."""
        return sum(d['vem'] for d in self.detectors)
    
    def get_time_range(self) -> Tuple[float, float]:
        """Get overall time range across all detectors."""
        if not self.is_loaded:
            return None, None
        
        t_min = min(d['time'][0] for d in self.detectors if len(d['time']) > 0)
        t_max = max(d['time'][-1] for d in self.detectors if len(d['time']) > 0)
        return t_min, t_max
    
    def get_data_in_range(self, t_min: float, t_max: float) -> List[Dict]:
        """Get detector data within a time range."""
        result = []
        for det in self.detectors:
            mask = (det['time'] >= t_min) & (det['time'] <= t_max)
            if np.any(mask):
                result.append({
                    'detector_id': det['detector_id'],
                    'time': det['time'][mask],
                    'signal_lower': det['signal_lower'][mask],
                    'signal_upper': det['signal_upper'][mask],
                    'vem': det['vem'],
                    'lat': det['lat'],
                    'lon': det['lon'],
                    'alt': det['alt'],
                })
        return result
    
    def get_info(self) -> str:
        """Get summary information about loaded data."""
        if not self.is_loaded:
            return "No data loaded"
        
        t_min, t_max = self.get_time_range()
        duration_us = t_max - t_min
        total_vem = self.get_total_vem()
        
        return (
            f"Directory: {os.path.basename(self.directory)}\n"
            f"Time: {self.time_str}\n"
            f"Detectors: {len(self.detectors)}\n"
            f"Time Range: {t_min:.0f} - {t_max:.0f} µs\n"
            f"Duration: {duration_us:.0f} µs\n"
            f"Total VEM: {total_vem:.2f}\n"
            f"Time Shift: {self.time_shift:.2f} µs"
        )


class TASDStitchReader:
    """
    Standalone reader for .stitch files.
    
    Use this for quick loading of individual files without the full handler.
    """
    
    @staticmethod
    def read_stitch(filepath: str, time_shift: float = 0.0) -> Optional[Dict]:
        """
        Read a .stitch file and return waveform data.
        
        Parameters:
        -----------
        filepath : str
            Path to .stitch file
        time_shift : float
            Time shift to apply
            
        Returns:
        --------
        dict with keys: time, signal_lower, signal_upper, detector_id, vem, burst_times
        """
        try:
            filename = os.path.basename(filepath)
            det_id = filename.split('_')[-1][:4]
            
            time_data = []
            signal_lower = []
            signal_upper = []
            
            burst_times = []
            burst_vem = []
            trig_time = 0
            
            header = True
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('###'):
                        continue
                    
                    if header:
                        if line.startswith('burst times'):
                            parts = line.replace(',', '').split()[3:]
                            burst_times = [int(x) for x in parts]
                            trig_time = burst_times[0] if burst_times else 0
                        elif line.startswith('burst VEM'):
                            parts = line.replace(',', '').split()[3:]
                            burst_vem = [float(x) for x in parts]
                        elif line.startswith('us'):
                            header = False
                        continue
                    
                    if line.strip() == '':
                        continue
                    
                    cols = line.replace(',', ' ').split()
                    if len(cols) < 3:
                        continue
                    
                    t = float(cols[0])
                    sig_l = float(cols[1])
                    sig_u = float(cols[2])
                    
                    if sig_l == 0.0 and sig_u == 0.0:
                        continue
                    
                    time_data.append(trig_time + t + time_shift)
                    signal_lower.append(sig_l)
                    signal_upper.append(sig_u)
            
            return {
                'detector_id': det_id,
                'time': np.array(time_data),
                'signal_lower': np.array(signal_lower),
                'signal_upper': np.array(signal_upper),
                'vem': sum(burst_vem),
                'burst_times': burst_times,
                'burst_vem': burst_vem,
            }
            
        except Exception as e:
            print(f"Error reading stitch file: {e}")
            return None
