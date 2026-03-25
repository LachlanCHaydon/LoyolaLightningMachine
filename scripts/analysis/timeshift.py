"""
TGF Analysis Tool - Timeshift Calculation
=========================================
Calculates timing offsets between TASD and other instruments.

Error propagation follows the general form:
    dq = sqrt((dx1 * dq/dx1)^2 + ... + (dxn * dq/dxn)^2)
matching the standalone calc_iterative scripts.
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
        self.intf_range = 9.4  # µs
        self.intf_elev_min = 1.0
        self.intf_elev_max = 11.0
        self.intf_azi_min = 240.0
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
        Includes full error propagation matching calc_iterative.
        """
        detectors = tasd_handler.detectors
        if not detectors:
            return {'success': False, 'error': 'No TASD detectors loaded'}

        event_time_sec = event_hour * 3600 + event_minute * 60 + event_second

        # Initialize altitude guesses
        z_old = [self.initial_altitude] * len(detectors)
        t0 = []
        t0_err = [0.0] * len(detectors)

        print(f"[TimeshiftCalc] event_time_sec={event_time_sec}, "
              f"n_detectors={len(detectors)}")

        # First pass - initial estimates
        for i, det in enumerate(detectors):
            sd_trig_sec = event_time_sec + det['sd_trig'] * 1e-6

            print(f"[TimeshiftCalc] Det {det['detector_id']}: "
                  f"sd_trig={det['sd_trig']:.1f} us, "
                  f"sd_trig_sec={sd_trig_sec:.6f} sec of day")

            lma_data = lma_handler.filter_by_time(
                sd_trig_sec, self.lma_range, self.lma_chi_limit
            )

            if lma_data is None or lma_data['count'] == 0:
                print(f"  -> No LMA data found near t={sd_trig_sec:.6f}")
                t0.append(0)
                continue

            print(f"  -> LMA: {lma_data['count']} sources found")

            lma_lat = np.mean(lma_data['latitude'])
            lma_lon = np.mean(lma_data['longitude'])

            x1 = geodesic((lma_lat, lma_lon),
                         (self.ref_lat, self.ref_lon)).km
            x2 = geodesic((lma_lat, lma_lon),
                         (det['lat'], det['lon'])).km

            diff = (np.sqrt(x1**2 + z_old[i]**2) -
                   np.sqrt(x2**2 + z_old[i]**2))
            t0.append(diff / self.c)

        # Iterative refinement
        success = False
        count = 0

        while count < self.iteration_limit:
            count += 1
            results = []
            dt_err_list = []

            for i, det in enumerate(detectors):
                sd_trig_us = det['sd_trig']
                sd_trig_sec = event_time_sec + sd_trig_us * 1e-6
                tc = t0[i] + sd_trig_us

                if count == 1:
                    print(f"[Iter {count}] Det {det['detector_id']}: "
                          f"sd_trig_us={sd_trig_us:.1f}, tc={tc:.1f}, "
                          f"t0[{i}]={t0[i]:.4f}")

                # Get LMA data
                lma_data = lma_handler.filter_by_time(
                    sd_trig_sec, self.lma_range, self.lma_chi_limit
                )

                if lma_data is None or lma_data['count'] == 0:
                    if count == 1:
                        print(f"  -> No LMA data")
                    continue

                lma_lat = np.mean(lma_data['latitude'])
                lma_lon = np.mean(lma_data['longitude'])
                n_lma = lma_data['count']

                # LMA position errors (standard error of mean)
                lma_lat_err = np.std(lma_data['latitude']) / np.sqrt(n_lma) if n_lma > 1 else 0.0
                lma_lon_err = np.std(lma_data['longitude']) / np.sqrt(n_lma) if n_lma > 1 else 0.0

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
                    if count == 1:
                        print(f"  -> No INTF data in range "
                              f"[{tc - self.intf_range:.1f}, "
                              f"{tc + self.intf_range:.1f}] us, "
                              f"elv=[{self.intf_elev_min}, {self.intf_elev_max}], "
                              f"azi=[{self.intf_azi_min}, {self.intf_azi_max}]")
                    continue

                if count == 1:
                    print(f"  -> LMA: {lma_data['count']} srcs, "
                          f"INTF: {len(intf_data['elevation'])} pts, "
                          f"elv={np.mean(intf_data['elevation']):.2f}")

                # Calculate distances using spherical earth
                lma_x = R_EARTH * (np.radians(lma_lon) - np.radians(self.ref_lon)) * np.cos(np.radians(self.ref_lat))
                lma_y = R_EARTH * (np.radians(lma_lat) - np.radians(self.ref_lat))

                sd_x = R_EARTH * (np.radians(det['lon']) - np.radians(self.ref_lon)) * np.cos(np.radians(self.ref_lat))
                sd_y = R_EARTH * (np.radians(det['lat']) - np.radians(self.ref_lat))

                # Position errors in km
                lma_x_err = np.radians(lma_lon_err) * np.cos(np.radians(lma_lat)) * R_EARTH
                lma_y_err = np.radians(lma_lat_err) * R_EARTH

                x1 = np.sqrt(lma_x**2 + lma_y**2)
                x2 = np.sqrt((sd_x - lma_x)**2 + (sd_y - lma_y)**2)

                # Distance errors
                x1_err = np.sqrt((lma_x * lma_x_err)**2 + (lma_y * lma_y_err)**2) / x1 if x1 > 0 else 0.0
                x2_err = np.sqrt(((lma_x - sd_x) * lma_x_err)**2 + ((lma_y - sd_y) * lma_y_err)**2) / x2 if x2 > 0 else 0.0

                # INTF elevation and azimuth with errors
                n_intf = len(intf_data['elevation'])
                elv = np.mean(intf_data['elevation'])
                azi = np.mean(intf_data['azimuth'])
                elv_err = np.std(intf_data['elevation']) / np.sqrt(n_intf) if n_intf > 1 else 0.0
                azi_err = np.std(intf_data['azimuth']) / np.sqrt(n_intf) if n_intf > 1 else 0.0

                # Altitude from elevation: z = x1 * tan(elv)
                elv_rad = np.radians(elv)
                z = x1 * np.tan(elv_rad)
                # zerr = sqrt((tan(elv)*x1err)^2 + (x1*(elverr_rad)/cos^2(elv))^2)
                elv_err_rad = np.radians(elv_err)
                z_err = np.sqrt(
                    (np.tan(elv_rad) * x1_err)**2 +
                    (x1 * elv_err_rad / (np.cos(elv_rad)**2))**2
                )

                # Slant distances
                r1 = np.sqrt(x1**2 + z**2)
                r1_err = np.sqrt((x1 * x1_err)**2 + (z * z_err)**2) / r1 if r1 > 0 else 0.0

                det_alt = det['alt'] - self.ref_alt  # altitude above reference
                r2 = np.sqrt(x2**2 + (z - det_alt)**2)

                # r2 error using dot product term (matching original script)
                dotp = ((sd_x - lma_x) * (-lma_x)) + ((sd_y - lma_y) * (-lma_y))
                r2_err = np.sqrt(
                    (x1 * x1_err * (np.tan(elv_rad)**2))**2 +
                    (elv_err_rad * np.tan(elv_rad) * (x1 / np.cos(elv_rad))**2)**2 +
                    (x2 * x2_err)**2 +
                    (2 * dotp * x1_err * x2_err * (np.tan(elv_rad)**2))
                ) / r2 if r2 > 0 else 0.0

                # Time calculations
                dt = (r1 - r2) / self.c
                ta = tc - r1 / self.c

                # dt error using full propagation (matching original script)
                term1 = (r1 / (x1 * self.c)) - ((x1 * (np.tan(elv_rad)**2)) / (r2 * self.c))
                term2 = -x2 / (r2 * self.c)
                term3 = ((x1**2) / self.c) * ((1 / r1) - (1 / r2)) * (np.tan(elv_rad) / (np.cos(elv_rad)**2))
                cos_alpha = dotp / (x1 * x2) if (x1 * x2) > 0 else 0.0
                dt_err = np.sqrt(
                    (term1 * x1_err)**2 +
                    (term3 * elv_err_rad)**2 +
                    (term2 * x2_err)**2 +
                    (2 * term1 * term2 * cos_alpha * x1_err * x2_err)
                )

                # ta error
                ta_err = np.sqrt(
                    (r1_err / self.c)**2 +
                    (t0_err[i])**2 +
                    (self.sd_timing_error)**2
                )

                # tc error
                tc_err = np.sqrt(ta_err**2 + (r2_err / self.c)**2)

                results.append({
                    'detector': det['detector_id'],
                    'dt': dt, 'dt_err': dt_err,
                    'z': z, 'z_err': z_err,
                    'elv': elv, 'elv_err': elv_err,
                    'azi': azi, 'azi_err': azi_err,
                    'x1': x1, 'x1_err': x1_err,
                    'x2': x2, 'x2_err': x2_err,
                    'r1': r1, 'r1_err': r1_err,
                    'r2': r2, 'r2_err': r2_err,
                    'ta': ta, 'ta_err': ta_err,
                    'tc': tc, 'tc_err': tc_err,
                    'vem': det['vem'],
                    'lma_lat': lma_lat,
                    'lma_lon': lma_lon,
                })
                dt_err_list.append(dt_err)

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
            t0_err = dt_err_list + t0_err[len(dt_err_list):]

        self.results = results
        self._calculate_summary()

        print(f"[TimeshiftCalc] INTF+LMA done: success={success}, "
              f"iterations={count}, n_results={len(results)}")

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

        print(f"[LMA-only] event_time_sec={event_time_sec}, "
              f"n_detectors={len(detectors)}")

        for det in detectors:
            sd_trig_sec = event_time_sec + det['sd_trig'] * 1e-6

            print(f"[LMA-only] Det {det['detector_id']}: "
                  f"sd_trig={det['sd_trig']:.1f} us, "
                  f"sd_trig_sec={sd_trig_sec:.6f}")

            lma_data = lma_handler.filter_by_time(
                sd_trig_sec, self.lma_range, self.lma_chi_limit
            )

            if lma_data is None or lma_data['count'] == 0:
                print(f"  -> No LMA data found")
                continue

            print(f"  -> LMA: {lma_data['count']} sources")

            lma_lat = np.mean(lma_data['latitude'])
            lma_lon = np.mean(lma_data['longitude'])
            lma_alt = np.mean(lma_data['altitude']) / 1000  # Convert to km

            x1 = geodesic((lma_lat, lma_lon),
                         (self.ref_lat, self.ref_lon)).km
            x2 = geodesic((lma_lat, lma_lon),
                         (det['lat'], det['lon'])).km

            z = lma_alt

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

        print(f"[LMA-only] Done: n_results={len(results)}")

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
        dt_errs = np.array([r['dt_err'] for r in self.results])
        zs = np.array([r['z'] for r in self.results])
        z_errs = np.array([r['z_err'] for r in self.results])
        elvs = np.array([r['elv'] for r in self.results])
        elv_errs = np.array([r['elv_err'] for r in self.results])
        azis = np.array([r['azi'] for r in self.results])
        azi_errs = np.array([r['azi_err'] for r in self.results])
        tas = np.array([r['ta'] for r in self.results])
        ta_errs = np.array([r['ta_err'] for r in self.results])
        tcs = np.array([r['tc'] for r in self.results])
        tc_errs = np.array([r['tc_err'] for r in self.results])
        x1s = np.array([r['x1'] for r in self.results])
        x1_errs = np.array([r['x1_err'] for r in self.results])
        r1s = np.array([r['r1'] for r in self.results])
        r1_errs = np.array([r['r1_err'] for r in self.results])
        vems = np.array([r['vem'] for r in self.results])

        self.summary = {
            'median': {
                'dt': np.median(dts), 'dt_err': np.median(dt_errs),
                'z': np.median(zs), 'z_err': np.median(z_errs),
                'elv': np.median(elvs), 'elv_err': np.median(elv_errs),
                'ta': np.median(tas), 'ta_err': np.median(ta_errs),
                'tc': np.median(tcs), 'tc_err': np.median(tc_errs),
                'x1': np.median(x1s), 'x1_err': np.median(x1_errs),
                'r1': np.median(r1s), 'r1_err': np.median(r1_errs),
            },
            'mean': {
                'dt': np.mean(dts), 'dt_err': np.mean(dt_errs),
                'z': np.mean(zs), 'z_err': np.mean(z_errs),
                'elv': np.mean(elvs), 'elv_err': np.mean(elv_errs),
                'azi': np.mean(azis), 'azi_err': np.mean(azi_errs),
                'ta': np.mean(tas), 'ta_err': np.mean(ta_errs),
                'tc': np.mean(tcs), 'tc_err': np.mean(tc_errs),
                'x1': np.mean(x1s), 'x1_err': np.mean(x1_errs),
                'r1': np.mean(r1s), 'r1_err': np.mean(r1_errs),
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
                'z': np.sum(zs * vems) / np.sum(vems),
                'z_err': np.sum(z_errs * vems) / np.sum(vems),
                'ta': np.sum(tas * vems) / np.sum(vems),
                'ta_err': np.sum(ta_errs * vems) / np.sum(vems),
                'tc': np.sum(tcs * vems) / np.sum(vems),
                'tc_err': np.sum(tc_errs * vems) / np.sum(vems),
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

        if 'mean' in self.summary:
            lines.append(f"\nMean timeshift: {self.summary['mean']['dt']:.3f} us")
            if 'z' in self.summary['mean']:
                lines.append(f"Mean altitude: {self.summary['mean']['z']:.3f} km")
            if 'x1' in self.summary['mean']:
                lines.append(f"Mean distance: {self.summary['mean']['x1']:.3f} km")

        if 'median' in self.summary:
            lines.append(f"\nMedian timeshift: {self.summary['median']['dt']:.3f} us")

        lines.append(f"\nDetectors: {len(self.results)}")

        lines.append("\n" + "-" * 50)
        lines.append("Per-detector results:")
        for r in self.results:
            lines.append(f"\n{r['detector']}:")
            lines.append(f"  dt = {r['dt']:.3f} us")
            if 'z' in r:
                lines.append(f"  z = {r['z']:.3f} km")
            if 'elv' in r:
                lines.append(f"  elv = {r['elv']:.2f} deg")

        return "\n".join(lines)
