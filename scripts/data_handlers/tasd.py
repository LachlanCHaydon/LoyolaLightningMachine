"""
TGF Analysis Tool - TASD (Surface Detector) Data Handler
========================================================
Handles loading and processing of Telescope Array Surface Detector data.

Unified parser that serves both the INTF tab (Az-El overlay) and the
Timeshift tab (calc_iterative). Parsing logic matches the authoritative
READ_TASD function from READ_calc_iterative_2024.py.
"""

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

    Per-detector dict fields:
        detector_id  : str        - 4-char TASD ID
        lat, lon, alt: float      - GPS coords (alt in km, from tasd_gpscoors.txt)
        burst_times  : list[int]  - all burst/trigger times (µs)
        burst_vem    : list[float]- VEM for each burst
        vem          : float      - VEM for the selected trigger
        pedestal_l   : float      - lower pedestal value
        pedestal_u   : float      - upper pedestal value
        time         : np.array   - waveform timestamps (µs, absolute)
        signal_lower : np.array   - lower FADC counts
        signal_upper : np.array   - upper FADC counts
        trig_time    : int        - first burst time (µs, time reference)
        sd_trig      : float      - individual integrated trigger time
                                    (pedestal-subtracted, 10-count threshold)
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

        # Trigger number used for loading
        self.trig_num = 1

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

        Walks the directory for .txt and .stitch files. For .txt files,
        matches filename against time_str (SD_HHMMSS_DDDD.txt format).
        Detectors without a coordinate match in tasd_gpscoors.txt are
        skipped entirely (matching READ_TASD behavior).

        Parameters
        ----------
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
            self.trig_num = trig_num
            self.detectors = []

            # List top-level directory only (no recursion into subdirs like Trig2/)
            # Matches standalone azi-elv-plotter.py which uses sdDir + filename
            for filename in os.listdir(directory):
                if not (filename.endswith('.txt') or filename.endswith('.stitch')):
                    continue

                # Only process SD files (skip Analysis files, images, etc.)
                if not filename.startswith('SD'):
                    continue

                # Extract detector ID from filename (last segment, first 4 chars)
                # Handles both SD_HHMMSS_DDDD.txt and SDYYYYMMDD_HHMMSS_DDDD.txt
                parts = filename.split('_')
                if len(parts) < 2:
                    continue
                det_id = parts[-1][:4]

                # Verify the time string matches
                # Filename formats:
                #   SD_HHMMSS_DDDD.txt       -> parts[1] = HHMMSS
                #   SDYYYYMMDD_HHMMSS_DDDD.txt -> parts[1] = HHMMSS
                if filename.endswith('.txt'):
                    file_time = parts[1] if len(parts) >= 3 else None
                    if file_time != time_str:
                        continue

                # Look up coordinates (may be None if coords file wasn't loaded)
                coords = self.detector_coords.get(det_id, {})

                filepath = os.path.join(directory, filename)
                det_data = self._load_single_file(
                    filepath, det_id, coords,
                    trig_num, manual_start, sd_start_time, time_shift
                )
                if det_data is not None:
                    self.detectors.append(det_data)

            self.is_loaded = len(self.detectors) > 0
            return self.is_loaded

        except Exception as e:
            print(f"Error loading TASD directory: {e}")
            self.is_loaded = False
            return False

    def _load_single_file(self, filepath: str, det_id: str, coords: Dict,
                         trig_num: int, manual_start: bool,
                         sd_start_time: float,
                         time_shift: float) -> Optional[Dict]:
        """
        Load a single TASD file.

        Parsing logic matches READ_TASD from READ_calc_iterative_2024.py:
        - Header: burst times, burst VEM, pedestal, then 'us' ends header
        - Waveform: split on ', ' (comma-space), skip separators (0.0, 0.0)
        - Computes sd_trig: integrated signal above pedestal, threshold = 10
        """
        try:
            # Header data
            burst_times = []
            burst_vem = []
            pedestal_l = 0.0
            pedestal_u = 0.0

            # Waveform data
            sd_time = []
            sd_sig_l = []
            sd_sig_u = []

            # For sd_trig computation
            sig_count = 0.0
            sd_trig = None

            header = True
            trig_time = 0  # first burst time (reference)

            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('###'):
                        continue

                    if header:
                        if line.startswith('burst times'):
                            bt = line.replace(',', '').split(' ')[3:]
                            burst_times = [int(x) for x in bt if x.strip()]
                            trig_time = burst_times[0] if burst_times else 0
                            continue
                        if line.startswith('burst VEM'):
                            bv = line.replace(',', '').split(' ')[3:]
                            burst_vem = [float(x) for x in bv if x.strip()]
                            continue
                        if line.startswith('pedestal'):
                            pparts = line.split(' ')
                            if len(pparts) >= 4:
                                pedestal_l = float(pparts[2].strip(','))
                                pedestal_u = float(pparts[3])
                            else:
                                pedestal_l = float(pparts[2])
                                pedestal_u = pedestal_l
                            continue
                        if line.startswith('us'):
                            header = False
                            continue
                        continue

                    # Waveform data — split on ', ' to match READ_TASD
                    if line == '\n' or line.strip() == '':
                        break

                    columns = line.split(', ')
                    if len(columns) < 3:
                        continue

                    t = float(columns[0])
                    sig_l = float(columns[1])
                    sig_u = float(columns[2])

                    # Skip waveform separators
                    if sig_l == 0.0 and sig_u == 0.0:
                        continue

                    abs_time = trig_time + t

                    # Manual start time filter
                    if manual_start and abs_time < sd_start_time:
                        continue

                    # Trigger number filter
                    if trig_num <= len(burst_times):
                        if abs_time < burst_times[trig_num - 1]:
                            continue
                        if trig_num < len(burst_times) and abs_time > burst_times[trig_num]:
                            break

                    sd_time.append(abs_time + time_shift)
                    sd_sig_l.append(sig_l)
                    sd_sig_u.append(sig_u)

                    # Compute sd_trig: integrated signal above pedestal
                    # (threshold of 10 counts within first ~15 samples)
                    # Matches READ_TASD logic exactly
                    if sd_trig is None and sig_count < 15:
                        int_sig_l = max(0.0, sig_l - pedestal_l)
                        int_sig_u = max(0.0, sig_u - pedestal_u)
                        ave_sig = (int_sig_l + int_sig_u) / 2.0
                        sig_count += ave_sig
                        if sig_count >= 10:
                            sd_trig = abs_time + time_shift

            if len(sd_time) < 2:
                return None

            # VEM for the selected trigger
            vem = 0.0
            if trig_num <= len(burst_vem):
                vem = burst_vem[trig_num - 1]

            return {
                'detector_id': det_id,
                'lat': coords.get('lat'),
                'lon': coords.get('lon'),
                'alt': coords.get('alt'),
                'burst_times': burst_times,
                'burst_vem': burst_vem,
                'vem': vem,
                'pedestal_l': pedestal_l,
                'pedestal_u': pedestal_u,
                'time': np.array(sd_time),
                'signal_lower': np.array(sd_sig_l),
                'signal_upper': np.array(sd_sig_u),
                'trig_time': trig_time,
                'sd_trig': sd_trig if sd_trig is not None else sd_time[0],
            }

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    # =========================================================================
    # Consumer helpers
    # =========================================================================

    def get_triggered_detectors(self, trig_num: Optional[int] = None) -> List[Dict]:
        """
        Get detectors with VEM > 0 for a given trigger number.

        Used by the INTF tab Az-El overlay to plot detector positions
        and compute the VEM-weighted core.

        Parameters
        ----------
        trig_num : int or None
            Trigger number (1-based). If None, uses the trig_num from loading.

        Returns
        -------
        List of detector dicts with VEM > 0 and valid coordinates.
        """
        result = []
        for det in self.detectors:
            if trig_num is not None and trig_num != self.trig_num:
                # Caller wants a different trigger than was loaded —
                # look up VEM from the burst_vem list
                idx = trig_num - 1
                if idx < len(det['burst_vem']):
                    vem = det['burst_vem'][idx]
                else:
                    vem = 0.0
            else:
                vem = det['vem']

            if vem > 0 and det.get('lat') is not None:
                # Return a copy with the correct VEM for this trigger
                d = dict(det)
                d['vem'] = vem
                result.append(d)
        return result

    def get_timing_data(self) -> List[List]:
        """
        Get data in the format expected by calc_iterative / timeshift.

        Returns a list matching the READ_TASD sds structure:
            sds[i] = [sdTime, sdSigL, sdSigU, det_id, lat, lon, alt_km,
                       vem, trig_time, sd_trig]

        Index mapping:
            [0] sdTime    - list of waveform timestamps
            [1] sdSigL    - list of lower FADC counts
            [2] sdSigU    - list of upper FADC counts
            [3] det_id    - 4-char detector ID
            [4] lat       - detector latitude
            [5] lon       - detector longitude
            [6] alt       - detector altitude (km)
            [7] vem       - VEM for selected trigger
            [8] trig_time - first burst time (µs)
            [9] sd_trig   - individual integrated trigger time
        """
        sds = []
        for det in self.detectors:
            sds.append([
                det['time'].tolist(),       # [0]
                det['signal_lower'].tolist(),  # [1]
                det['signal_upper'].tolist(),  # [2]
                det['detector_id'],          # [3]
                det['lat'],                  # [4]
                det['lon'],                  # [5]
                det['alt'],                  # [6]
                det['vem'],                  # [7]
                det['trig_time'],            # [8]
                det['sd_trig'],              # [9]
            ])
        return sds

    # =========================================================================
    # Existing query methods
    # =========================================================================

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
                d = dict(det)
                d['time'] = det['time'][mask]
                d['signal_lower'] = det['signal_lower'][mask]
                d['signal_upper'] = det['signal_upper'][mask]
                result.append(d)
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
