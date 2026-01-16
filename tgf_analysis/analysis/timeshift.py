"""
TGF Analysis Tool - Timeshift Calculation
=========================================
Calculates timing offsets between TASD and other instruments.
"""

import numpy as np
import heapq
from typing import Optional, Tuple, Dict, List
import geopy
from geopy.distance import geodesic
import scipy.constants as sp

from config import CAMERA_LAT, CAMERA_LON, CAMERA_ALT, R_EARTH
from data_handlers import TASDHandler, LMAHandler, InterferometerHandler


class TimeshiftCalculator:
    """
    Calculate timing offsets between TASD detectors and INTF/LMA data.
    
    This module implements the iterative calculation method used to
    determine the time shift needed to align TASD data with camera/INTF data.
    """
    
    def __init__(self):
        """Initialize the TimeshiftCalculator."""
        # Speed of light in km/µs
        self.c = sp.c * 1e-9
        
        # Reference location (INTF/camera)
        self.ref_lat = CAMERA_LAT
        self.ref_lon = CAMERA_LON
        self.ref_alt = CAMERA_ALT
        
        # Calculation parameters
        self.initial_altitude = 3.0  # km
        self.iteration_limit = 10
        self.sd_timing_error = 0.04  # µs
        
        # LMA filtering
        self.lma_range = 1000  # µs
        self.lma_chi_limit = 2.0
        
        # INTF filtering  
        self.intf_range = 4.0  # µs
        self.intf_elev_min = 1.0
        self.intf_elev_max = 11.0
        self.intf_azi_min = 296.0
        self.intf_azi_max = 302.0
        
        # Results
        self.results: List[Dict] = []
        self.summary: Dict = {}
    
    def set_parameters(self, **kwargs):
        """Set calculation parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def calculate_with_intf(self, tasd_handler: TASDHandler,
                           lma_handler: LMAHandler,
                           intf_handler: InterferometerHandler,
                           event_hour: int, event_minute: int, event_second: int,
                           trig_num: int = 1) -> Dict:
        """
        Calculate timeshift using INTF data (full method).
        
        This is the complete iterative calculation that uses INTF
        elevation data to determine source altitude.
        """
        detectors = tasd_handler.detectors
        if not detectors:
            return {'success': False, 'error': 'No TASD detectors loaded'}
        
        event_time_sec = event_hour * 3600 + event_minute * 60 + event_second
        
        # Initialize altitude guesses
        z_old = [self.initial_altitude] * len(detectors)
        t0 = []
        t0_err = [0] * len(detectors)
        
        # First pass - initial estimates
        for i, det in enumerate(detectors):
            sd_trig_sec = event_time_sec + det['time'][0] * 1e-6
            
            # Get LMA points near this detector's trigger
            lma_data = lma_handler.filter_by_time(
                sd_trig_sec, self.lma_range, self.lma_chi_limit
            )
            
            if lma_data is None or lma_data['count'] == 0:
                t0.append(0)
                continue
            
            # Calculate distances
            lma_lat = np.mean(lma_data['latitude'])
            lma_lon = np.mean(lma_data['longitude'])
            
            x1 = geodesic((lma_lat, lma_lon), 
                         (self.ref_lat, self.ref_lon)).km
            x2 = geodesic((lma_lat, lma_lon),
                         (det['lat'], det['lon'])).km
            
            # Path length difference
            diff = (np.sqrt(x1**2 + z_old[i]**2) - 
                   np.sqrt(x2**2 + z_old[i]**2))
            t0.append(diff / self.c)
        
        # Iterative refinement
        success = False
        count = 0
        
        while count < self.iteration_limit:
            count += 1
            results = []
            
            for i, det in enumerate(detectors):
                sd_trig_us = det['time'][0]
                sd_trig_sec = event_time_sec + sd_trig_us * 1e-6
                tc = t0[i] + sd_trig_us  # Shifted trigger time
                
                # Get LMA data
                lma_data = lma_handler.filter_by_time(
                    sd_trig_sec, self.lma_range, self.lma_chi_limit
                )
                
                if lma_data is None or lma_data['count'] == 0:
                    continue
                
                lma_lat = np.mean(lma_data['latitude'])
                lma_lon = np.mean(lma_data['longitude'])
                
                # Get INTF data near tc
                intf_data = intf_handler.filter_data(
                    t_min=tc - self.intf_range,
                    t_max=tc + self.intf_range,
                    elv_min=self.intf_elev_min,
                    elv_max=self.intf_elev_max,
                    azi_min=self.intf_azi_min,
                    azi_max=self.intf_azi_max
                )
                
                if intf_data is None or len(intf_data['elevation']) == 0:
                    continue
                
                # Calculate distances using spherical earth
                lma_x = R_EARTH * (np.radians(lma_lon) - np.radians(self.ref_lon)) * np.cos(np.radians(self.ref_lat))
                lma_y = R_EARTH * (np.radians(lma_lat) - np.radians(self.ref_lat))
                
                sd_x = R_EARTH * (np.radians(det['lon']) - np.radians(self.ref_lon)) * np.cos(np.radians(self.ref_lat))
                sd_y = R_EARTH * (np.radians(det['lat']) - np.radians(self.ref_lat))
                
                x1 = np.sqrt(lma_x**2 + lma_y**2)
                x2 = np.sqrt((sd_x - lma_x)**2 + (sd_y - lma_y)**2)
                
                # Get elevation and azimuth from INTF
                elv = np.mean(intf_data['elevation'])
                azi = np.mean(intf_data['azimuth'])
                
                # Calculate altitude from elevation
                z = x1 * np.tan(np.radians(elv))
                
                # Slant distances
                r1 = np.sqrt(x1**2 + z**2)
                r2 = np.sqrt(x2**2 + (z - det['alt'])**2)
                
                # Time calculations
                dt = (r1 - r2) / self.c
                ta = tc - r1 / self.c
                
                results.append({
                    'detector': det['detector_id'],
                    'dt': dt,
                    'z': z,
                    'elv': elv,
                    'azi': azi,
                    'x1': x1,
                    'x2': x2,
                    'r1': r1,
                    'r2': r2,
                    'ta': ta,
                    'tc': tc,
                    'vem': det['vem'],
                    'lma_lat': lma_lat,
                    'lma_lon': lma_lon,
                })
            
            if not results:
                break
            
            # Check convergence
            new_z = [r['z'] for r in results]
            if len(new_z) == len(z_old):
                if np.allclose(new_z, z_old[:len(new_z)], atol=0.0001):
                    success = True
                    break
            
            # Update for next iteration
            z_old = new_z + [self.initial_altitude] * (len(z_old) - len(new_z))
            t0 = [r['dt'] for r in results] + t0[len(results):]
        
        self.results = results
        self._calculate_summary()
        
        return {
            'success': success,
            'iterations': count,
            'detectors': results,
            'summary': self.summary
        }
    
    def calculate_lma_only(self, tasd_handler: TASDHandler,
                          lma_handler: LMAHandler,
                          event_hour: int, event_minute: int, event_second: int) -> Dict:
        """
        Calculate timeshift using only LMA data (no INTF).
        
        Use this when INTF data is unavailable or uncalibrated.
        Provides timeshift but not full source geometry.
        """
        detectors = tasd_handler.detectors
        if not detectors:
            return {'success': False, 'error': 'No TASD detectors loaded'}
        
        event_time_sec = event_hour * 3600 + event_minute * 60 + event_second
        
        results = []
        
        for det in detectors:
            sd_trig_sec = event_time_sec + det['time'][0] * 1e-6
            
            # Get LMA points
            lma_data = lma_handler.filter_by_time(
                sd_trig_sec, self.lma_range, self.lma_chi_limit
            )
            
            if lma_data is None or lma_data['count'] == 0:
                continue
            
            lma_lat = np.mean(lma_data['latitude'])
            lma_lon = np.mean(lma_data['longitude'])
            lma_alt = np.mean(lma_data['altitude']) / 1000  # Convert to km
            
            # Calculate distances
            x1 = geodesic((lma_lat, lma_lon),
                         (self.ref_lat, self.ref_lon)).km
            x2 = geodesic((lma_lat, lma_lon),
                         (det['lat'], det['lon'])).km
            
            z = lma_alt
            
            # Path difference
            diff = (np.sqrt(x1**2 + z**2) - np.sqrt(x2**2 + z**2))
            dt = diff / self.c
            
            results.append({
                'detector': det['detector_id'],
                'dt': dt,
                'z': z,
                'x1': x1,
                'x2': x2,
                'vem': det['vem'],
                'lma_lat': lma_lat,
                'lma_lon': lma_lon,
                'lma_alt': lma_alt,
            })
        
        self.results = results
        self._calculate_summary_lma_only()
        
        return {
            'success': True,
            'detectors': results,
            'summary': self.summary
        }
    
    def _calculate_summary(self):
        """Calculate summary statistics from full results."""
        if not self.results:
            self.summary = {}
            return
        
        # Extract arrays
        dts = np.array([r['dt'] for r in self.results])
        zs = np.array([r['z'] for r in self.results])
        elvs = np.array([r['elv'] for r in self.results])
        azis = np.array([r['azi'] for r in self.results])
        tas = np.array([r['ta'] for r in self.results])
        tcs = np.array([r['tc'] for r in self.results])
        x1s = np.array([r['x1'] for r in self.results])
        vems = np.array([r['vem'] for r in self.results])
        
        self.summary = {
            'median': {
                'dt': np.median(dts),
                'z': np.median(zs),
                'elv': np.median(elvs),
                'ta': np.median(tas),
                'tc': np.median(tcs),
                'x1': np.median(x1s),
            },
            'mean': {
                'dt': np.mean(dts),
                'z': np.mean(zs),
                'elv': np.mean(elvs),
                'azi': np.mean(azis),
                'ta': np.mean(tas),
                'tc': np.mean(tcs),
                'x1': np.mean(x1s),
            },
            'std': {
                'dt': np.std(dts),
                'z': np.std(zs),
                'elv': np.std(elvs),
            },
            'n_detectors': len(self.results),
            'total_vem': np.sum(vems),
        }
        
        # Weighted mean (by VEM)
        if np.sum(vems) > 0:
            self.summary['weighted'] = {
                'dt': np.sum(dts * vems) / np.sum(vems),
                'z': np.sum(zs * vems) / np.sum(vems),
                'ta': np.sum(tas * vems) / np.sum(vems),
            }
    
    def _calculate_summary_lma_only(self):
        """Calculate summary for LMA-only calculation."""
        if not self.results:
            self.summary = {}
            return
        
        dts = np.array([r['dt'] for r in self.results])
        zs = np.array([r['z'] for r in self.results])
        x1s = np.array([r['x1'] for r in self.results])
        
        self.summary = {
            'mean': {
                'dt': np.mean(dts),
                'z': np.mean(zs),
                'x1': np.mean(x1s),
            },
            'std': {
                'dt': np.std(dts),
            },
            'n_detectors': len(self.results),
        }
    
    def get_timeshift(self) -> float:
        """Get the calculated timeshift value."""
        if 'mean' in self.summary:
            return self.summary['mean']['dt']
        return 0.0
    
    def format_results(self) -> str:
        """Format results as a string report."""
        if not self.results:
            return "No results calculated"
        
        lines = ["=" * 50]
        lines.append("TIMESHIFT CALCULATION RESULTS")
        lines.append("=" * 50)
        
        # Summary
        if 'mean' in self.summary:
            lines.append(f"\nMean timeshift: {self.summary['mean']['dt']:.3f} µs")
            if 'z' in self.summary['mean']:
                lines.append(f"Mean altitude: {self.summary['mean']['z']:.3f} km")
            if 'x1' in self.summary['mean']:
                lines.append(f"Mean distance: {self.summary['mean']['x1']:.3f} km")
        
        if 'median' in self.summary:
            lines.append(f"\nMedian timeshift: {self.summary['median']['dt']:.3f} µs")
        
        lines.append(f"\nDetectors: {len(self.results)}")
        
        # Per-detector results
        lines.append("\n" + "-" * 50)
        lines.append("Per-detector results:")
        for r in self.results:
            lines.append(f"\n{r['detector']}:")
            lines.append(f"  dt = {r['dt']:.3f} µs")
            if 'z' in r:
                lines.append(f"  z = {r['z']:.3f} km")
            if 'elv' in r:
                lines.append(f"  elv = {r['elv']:.2f}°")
        
        return "\n".join(lines)
