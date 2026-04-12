#!/usr/bin/env python
"""
INTF Cosine Shift Calibration via LMA Cross-Matching
======================================================
Uses LMA 3D source locations as ground truth to determine
cosShiftA and cosShiftB for the INTF.

Method (following Jensen et al. 2021, Stock et al. 2014):
  1. Time-match INTF and LMA sources (both GPS-referenced)
  2. For each match, compute true azimuth & elevation from INTF to LMA 3D position
  3. Convert true angles to true direction cosines
  4. Shift = true_cosine - raw_cosine
  5. Median across all matched pairs = final calibration

This bypasses NLDN entirely and uses the same approach that
Mark Stanley's group used for the original calibration.

Author: Lachlan (Loyola / TA collaboration)
"""

import numpy as np
import geopy
from vincenty import vincenty

# ============================================================
# Constants
# ============================================================
INTF_LAT = 39.339082
INTF_LON = -112.700696
INTF_ALT_M = 1400.0   # meters above sea level

LATCLF = 39.29693
LONCLF = -112.90875
ZCLF = 1.382  # km

C_LIGHT = 2.998e8  # m/s


# ============================================================
# Coordinate functions
# ============================================================
def gps2cart(lat, lon, alt=ZCLF):
    """Convert GPS lat/lon to CLF-centered Cartesian (km)."""
    g0000 = geopy.point.Point(LATCLF, LONCLF, ZCLF)
    gnnnn = geopy.point.Point(lat, lon, alt)
    g0000eq = geopy.point.Point(0.0, g0000.longitude, g0000.altitude)
    gnnnneq = geopy.point.Point(0.0, gnnnn.longitude, gnnnn.altitude)
    r = vincenty(g0000, gnnnn)
    y = vincenty(gnnnn, gnnnneq) - vincenty(g0000, g0000eq)
    if gnnnn.longitude > g0000.longitude:
        x = np.sqrt(r * r - y * y)
    else:
        x = -1.0 * np.sqrt(r * r - y * y)
    return x, y


def compute_true_angles(source_lat, source_lon, source_alt_m):
    """
    Compute true azimuth and elevation from INTF antenna to a source.

    Parameters
    ----------
    source_lat, source_lon : float, degrees
    source_alt_m : float, meters above sea level

    Returns
    -------
    azimuth : float, degrees (0=N, 90=E, CW)
    elevation : float, degrees above horizon
    range_m : float, slant range in meters
    """
    x_intf, y_intf = gps2cart(INTF_LAT, INTF_LON)
    x_src, y_src = gps2cart(source_lat, source_lon)

    dx = x_src - x_intf   # km, East
    dy = y_src - y_intf   # km, North
    dz = (source_alt_m / 1000.0) - (INTF_ALT_M / 1000.0)  # km, Up

    horiz_dist = np.sqrt(dx**2 + dy**2)  # km
    slant_range = np.sqrt(dx**2 + dy**2 + dz**2)  # km

    azimuth = np.degrees(np.arctan2(dx, dy))  # arctan2(E, N)
    if azimuth < 0:
        azimuth += 360.0

    elevation = np.degrees(np.arctan2(dz, horiz_dist))

    return azimuth, elevation, slant_range * 1000.0  # range in meters


def angles_to_cosines(azi_deg, elv_deg):
    """Convert azimuth/elevation to direction cosines."""
    azi_rad = np.radians(azi_deg)
    elv_rad = np.radians(elv_deg)
    cosA = np.cos(elv_rad) * np.cos(azi_rad)
    cosB = np.cos(elv_rad) * np.sin(azi_rad)
    return cosA, cosB


# ============================================================
# Data loaders
# ============================================================
def load_intf_raw(filepath, skip_header=57):
    """Load raw INTF .dat file."""
    data = np.genfromtxt(filepath, skip_header=skip_header,
                         usecols=(0, 1, 2, 3, 4, 5))
    return {
        'time_ms': data[:, 0],
        'azimuth': data[:, 1],
        'elevation': data[:, 2],
        'cosA': data[:, 3],
        'cosB': data[:, 4],
        'pk2pk': data[:, 5]
    }


def load_lma(filepath):
    """
    Load LMA data file.
    Format: time(sod) lat lon alt(m) chi2 power [mask]
    """
    data = np.genfromtxt(filepath, comments='#',
                         usecols=(0, 1, 2, 3, 4, 5))
    return {
        'sod': data[:, 0],
        'lat': data[:, 1],
        'lon': data[:, 2],
        'alt_m': data[:, 3],
        'chi2': data[:, 4],
        'power': data[:, 5]
    }


# ============================================================
# Main calibration
# ============================================================
if __name__ == '__main__':

    # ============= USER PARAMETERS ===========================

    # INTF raw data file
    INTF_FILE = "TAR_2024.08.17_16-18-44_736616_pix200_S256-I64-P4_W3.dat"
    INTF_RAW_SKIP_HEADER = 57

    # LMA data file
    LMA_FILE = "LMA_240817_161845_cut.txt"

    # Timing: convert INTF raw time (ms rel. to trigger) to seconds-of-day
    # INTF sod = INTF_T0_SOD + time_ms / 1000
    # The manualIntfTrig = 730069 us = 730.069 ms after 16:18:44
    # 16:18:44 = 58724 seconds of day
    INTF_T0_SOD = 58724.730069

    # Matching parameters
    MATCH_WINDOW_US = 50.0    # +/- matching window in microseconds
    # Note: this should be tight enough for good correspondence
    # but loose enough to accommodate light travel time differences.
    # At 20 km, light travel time ~ 67 us. LMA gives emission time,
    # INTF gives arrival time. The manualIntfTrig correction should
    # approximately align these, but some residual offset may exist.

    # LMA quality filters
    LMA_CHI2_MAX = 2.0        # max reduced chi-squared
    LMA_ALT_MIN = 1000.0      # min altitude (m) - above ground
    LMA_ALT_MAX = 15000.0     # max altitude (m) - reasonable cloud top
    LMA_POWER_MIN = -20.0     # min power (dBW) - reject very weak

    # Sigma clipping for shift estimates
    SIGMA_CLIP_THRESH = 2.0
    SIGMA_CLIP_N = 3

    # =========================================================

    # --- Load data ---
    print("\n" + "=" * 60)
    print("  LMA-Based INTF Cosine Shift Calibration")
    print("=" * 60)

    print(f"\n  Loading INTF data: {INTF_FILE}")
    intf = load_intf_raw(INTF_FILE, INTF_RAW_SKIP_HEADER)
    n_intf = len(intf['time_ms'])
    print(f"    {n_intf} points loaded")
    print(f"    Time range: [{intf['time_ms'].min():.3f}, "
          f"{intf['time_ms'].max():.3f}] ms")

    print(f"\n  Loading LMA data: {LMA_FILE}")
    lma = load_lma(LMA_FILE)
    n_lma_raw = len(lma['sod'])
    print(f"    {n_lma_raw} points loaded")
    print(f"    Time range: [{lma['sod'].min():.6f}, "
          f"{lma['sod'].max():.6f}] sod")

    # --- Filter LMA ---
    print(f"\n  Filtering LMA data:")
    print(f"    chi2 < {LMA_CHI2_MAX}")
    print(f"    alt: [{LMA_ALT_MIN}, {LMA_ALT_MAX}] m")
    print(f"    power > {LMA_POWER_MIN} dBW")

    lma_mask = ((lma['chi2'] <= LMA_CHI2_MAX) &
                (lma['alt_m'] >= LMA_ALT_MIN) &
                (lma['alt_m'] <= LMA_ALT_MAX) &
                (lma['power'] >= LMA_POWER_MIN))

    n_lma_good = np.sum(lma_mask)
    print(f"    Surviving: {n_lma_good} / {n_lma_raw}")

    if n_lma_good == 0:
        print("  ERROR: No LMA points passed quality filters.")
        print("  Try relaxing LMA_CHI2_MAX or LMA_POWER_MIN.")
        exit(1)

    # Apply filter
    lma_sod = lma['sod'][lma_mask]
    lma_lat = lma['lat'][lma_mask]
    lma_lon = lma['lon'][lma_mask]
    lma_alt = lma['alt_m'][lma_mask]
    lma_chi2 = lma['chi2'][lma_mask]
    lma_power = lma['power'][lma_mask]

    # --- Convert INTF times to seconds-of-day ---
    intf_sod = INTF_T0_SOD + intf['time_ms'] / 1000.0

    print(f"\n  INTF time range in SOD: [{intf_sod.min():.6f}, "
          f"{intf_sod.max():.6f}]")
    print(f"  LMA time range in SOD:  [{lma_sod.min():.6f}, "
          f"{lma_sod.max():.6f}]")

    # Check overlap
    overlap_start = max(intf_sod.min(), lma_sod.min())
    overlap_end = min(intf_sod.max(), lma_sod.max())
    if overlap_start > overlap_end:
        print("\n  ERROR: No time overlap between INTF and LMA!")
        exit(1)
    print(f"  Time overlap: [{overlap_start:.6f}, {overlap_end:.6f}] sod")
    print(f"  Overlap duration: {(overlap_end - overlap_start)*1e6:.1f} us")

    # --- Match sources ---
    match_window_sod = MATCH_WINDOW_US / 1e6  # convert us to seconds
    print(f"\n  Matching INTF to LMA (window = +/- {MATCH_WINDOW_US} us)...")

    matched_indices_intf = []
    matched_indices_lma = []
    matched_dt_us = []

    for i_lma in range(n_lma_good):
        # Find INTF points within the time window of this LMA source
        dt = intf_sod - lma_sod[i_lma]
        within = np.abs(dt) <= match_window_sod

        if np.sum(within) == 0:
            continue

        # Among candidates, pick the one with the smallest time difference
        candidates = np.where(within)[0]
        best = candidates[np.argmin(np.abs(dt[candidates]))]

        matched_indices_lma.append(i_lma)
        matched_indices_intf.append(best)
        matched_dt_us.append(dt[best] * 1e6)

    n_matched = len(matched_indices_intf)
    print(f"  Matched pairs: {n_matched}")

    if n_matched == 0:
        print("  ERROR: No matches found.")
        print("  Try increasing MATCH_WINDOW_US or check time alignment.")
        # Print a diagnostic
        print(f"\n  Diagnostic: closest LMA to first INTF point:")
        dt_first = lma_sod - intf_sod[0]
        closest_lma = np.argmin(np.abs(dt_first))
        print(f"    INTF[0] sod = {intf_sod[0]:.9f}")
        print(f"    Closest LMA sod = {lma_sod[closest_lma]:.9f}")
        print(f"    Difference = {dt_first[closest_lma]*1e6:.1f} us")
        exit(1)

    matched_dt_us = np.array(matched_dt_us)
    print(f"  Time residuals: mean = {np.mean(matched_dt_us):.1f} us, "
          f"std = {np.std(matched_dt_us):.1f} us")

    # --- Compute true angles and shifts for each match ---
    print(f"\n  Computing true angles from INTF to LMA positions...")

    shiftA_all = np.zeros(n_matched)
    shiftB_all = np.zeros(n_matched)
    true_azi_all = np.zeros(n_matched)
    true_elv_all = np.zeros(n_matched)
    raw_cosA_all = np.zeros(n_matched)
    raw_cosB_all = np.zeros(n_matched)
    range_all = np.zeros(n_matched)

    for k in range(n_matched):
        i_intf = matched_indices_intf[k]
        i_lma = matched_indices_lma[k]

        # True angles from INTF antenna to LMA 3D source
        true_azi, true_elv, slant_range = compute_true_angles(
            lma_lat[i_lma], lma_lon[i_lma], lma_alt[i_lma]
        )

        # True direction cosines
        true_cosA, true_cosB = angles_to_cosines(true_azi, true_elv)

        # Raw INTF cosines
        raw_cA = intf['cosA'][i_intf]
        raw_cB = intf['cosB'][i_intf]

        # Shifts
        shiftA_all[k] = true_cosA - raw_cA
        shiftB_all[k] = true_cosB - raw_cB
        true_azi_all[k] = true_azi
        true_elv_all[k] = true_elv
        raw_cosA_all[k] = raw_cA
        raw_cosB_all[k] = raw_cB
        range_all[k] = slant_range

    # --- Show individual matches ---
    print(f"\n  {'#':>3s}  {'dt(us)':>8s}  {'LMA azi':>8s}  {'LMA elv':>8s}  "
          f"{'Range(km)':>9s}  {'shiftA':>10s}  {'shiftB':>10s}  "
          f"{'chi2':>6s}  {'P(dBW)':>7s}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*10}  "
          f"{'-'*10}  {'-'*6}  {'-'*7}")
    for k in range(n_matched):
        i_lma = matched_indices_lma[k]
        print(f"  {k:3d}  {matched_dt_us[k]:8.1f}  {true_azi_all[k]:8.2f}  "
              f"{true_elv_all[k]:8.2f}  {range_all[k]/1000:9.2f}  "
              f"{shiftA_all[k]:+10.6f}  {shiftB_all[k]:+10.6f}  "
              f"{lma_chi2[matched_indices_lma[k]]:6.2f}  "
              f"{lma_power[matched_indices_lma[k]]:7.1f}")

    # --- Sigma clip and compute final shifts ---
    print(f"\n  Sigma clipping ({SIGMA_CLIP_N} iterations, "
          f"{SIGMA_CLIP_THRESH} sigma)...")

    keep = np.ones(n_matched, dtype=bool)

    for iteration in range(SIGMA_CLIP_N):
        sA = shiftA_all[keep]
        sB = shiftB_all[keep]

        medA = np.median(sA)
        medB = np.median(sB)
        stdA = np.std(sA)
        stdB = np.std(sB)

        if stdA == 0 or stdB == 0:
            break

        devA = np.abs(shiftA_all - medA) / stdA
        devB = np.abs(shiftB_all - medB) / stdB
        keep_new = (devA <= SIGMA_CLIP_THRESH) & (devB <= SIGMA_CLIP_THRESH)

        n_rejected = np.sum(keep) - np.sum(keep_new)
        keep = keep_new

        print(f"    Iteration {iteration + 1}: rejected {n_rejected}, "
              f"remaining {np.sum(keep)}")

    n_final = np.sum(keep)
    if n_final == 0:
        print("    WARNING: All points rejected. Using full set.")
        keep = np.ones(n_matched, dtype=bool)
        n_final = n_matched

    # Final shifts: median of surviving matches
    cosShiftA = np.median(shiftA_all[keep])
    cosShiftB = np.median(shiftB_all[keep])

    # Also compute mean and std for comparison
    mean_shiftA = np.mean(shiftA_all[keep])
    mean_shiftB = np.mean(shiftB_all[keep])
    std_shiftA = np.std(shiftA_all[keep])
    std_shiftB = np.std(shiftB_all[keep])

    # Standard error of the median (approx)
    se_shiftA = 1.2533 * std_shiftA / np.sqrt(n_final)
    se_shiftB = 1.2533 * std_shiftB / np.sqrt(n_final)

    print(f"\n  {'='*60}")
    print(f"  RESULTS ({n_final} matched pairs after clipping)")
    print(f"  {'='*60}")

    print(f"\n  Median shifts:")
    print(f"    cosShiftA = {cosShiftA:+.6f}  (std = {std_shiftA:.6f}, "
          f"SE = {se_shiftA:.6f})")
    print(f"    cosShiftB = {cosShiftB:+.6f}  (std = {std_shiftB:.6f}, "
          f"SE = {se_shiftB:.6f})")

    print(f"\n  Mean shifts (for comparison):")
    print(f"    cosShiftA = {mean_shiftA:+.6f}")
    print(f"    cosShiftB = {mean_shiftB:+.6f}")

    # Compare with known 2024 values
    known_A = -0.0051
    known_B = -0.0178
    print(f"\n  Comparison with 2024 values:")
    print(f"    2024:     cosShiftA = {known_A:+.6f},  cosShiftB = {known_B:+.6f}")
    print(f"    LMA-cal:  cosShiftA = {cosShiftA:+.6f},  cosShiftB = {cosShiftB:+.6f}")
    print(f"    Delta:    cosShiftA = {cosShiftA - known_A:+.6f},  "
          f"cosShiftB = {cosShiftB - known_B:+.6f}")

    # Convert delta to approximate angular difference
    # At typical elevation (~5-10 deg), cos(elv) ~ 0.98
    delta_shift = np.sqrt((cosShiftA - known_A)**2 + (cosShiftB - known_B)**2)
    print(f"    Shift magnitude difference: {delta_shift:.6f} "
          f"(~{np.degrees(np.arcsin(min(delta_shift, 1.0))):.3f} deg)")

    # Formal uncertainty of the calibration
    formal_error_deg_A = np.degrees(np.arcsin(min(se_shiftA, 1.0)))
    formal_error_deg_B = np.degrees(np.arcsin(min(se_shiftB, 1.0)))
    combined_error = np.sqrt(formal_error_deg_A**2 + formal_error_deg_B**2)

    print(f"\n  ┌──────────────────────────────────────────────────────┐")
    print(f"  │  cosShiftA = {cosShiftA:+.6f}  ± {se_shiftA:.6f}              │")
    print(f"  │  cosShiftB = {cosShiftB:+.6f}  ± {se_shiftB:.6f}              │")
    print(f"  │  Formal pointing uncertainty: ± {combined_error:.4f} deg          │")
    print(f"  │  Matched pairs: {n_final}                                 │")
    print(f"  └──────────────────────────────────────────────────────┘")

    print(f"\n  {'='*60}")