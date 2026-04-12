#!/usr/bin/env python
"""
Compare Two Cosine Shift Calibrations
========================================
Applies two different (cosShiftA, cosShiftB) pairs to the same raw INTF
data and reports the angular differences between them.

This quantifies how much the choice of calibration method matters for
the final azimuth/elevation values.

Author: Lachlan (Loyola / TA collaboration)
"""

import numpy as np


# ============================================================
# INTF functions
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


def apply_cos_shifts(cosA, cosB, shiftA, shiftB):
    """Apply cosine shifts and compute azimuth/elevation."""
    cosA_cal = cosA + shiftA
    cosB_cal = cosB + shiftB

    azimuth = np.degrees(np.arctan2(cosB_cal, cosA_cal)) + 360.0
    azimuth = azimuth % 360.0

    p = np.sqrt(cosA_cal**2 + cosB_cal**2)
    elevation = np.zeros_like(p)
    pos = p <= 1.0
    neg = p > 1.0
    elevation[pos] = np.degrees(np.arccos(p[pos]))
    elevation[neg] = -np.degrees(np.arccos(2 - p[neg]))

    return azimuth, elevation


def angular_separation(azi1, elv1, azi2, elv2):
    """
    Compute angular separation between two sky positions.
    Uses small-angle spherical approximation (valid for < ~10 deg separation).
    """
    d_azi = azi1 - azi2
    d_elv = elv1 - elv2
    cos_elv = np.cos(np.radians(0.5 * (elv1 + elv2)))
    return np.sqrt((d_azi * cos_elv)**2 + d_elv**2)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':

    # ============= USER PARAMETERS ===========================

    INTF_FILE = "TAR_2024.08.17_16-18-44_736616_pix200_S256-I64-P4_W3.dat"
    INTF_RAW_SKIP_HEADER = 57

    # Calibration A (e.g. our new method)
    LABEL_A = "New Method"
    COS_SHIFT_A_1 =  -0.005859
    COS_SHIFT_B_1 = -0.017102


    # Calibration B (e.g. 2024 values or Utah values)
    LABEL_B = "2024 values"
    COS_SHIFT_A_2 = -0.0051
    COS_SHIFT_B_2 = -0.0178

    # =========================================================

    # Load data
    print("\n  Loading INTF data...")
    intf = load_intf_raw(INTF_FILE, INTF_RAW_SKIP_HEADER)
    n = len(intf['time_ms'])
    print(f"  {n} points loaded")

    # Apply both calibrations
    azi_1, elv_1 = apply_cos_shifts(intf['cosA'], intf['cosB'],
                                     COS_SHIFT_A_1, COS_SHIFT_B_1)
    azi_2, elv_2 = apply_cos_shifts(intf['cosA'], intf['cosB'],
                                     COS_SHIFT_A_2, COS_SHIFT_B_2)

    # Point-by-point differences
    d_azi = azi_1 - azi_2
    d_elv = elv_1 - elv_2
    sep = angular_separation(azi_1, elv_1, azi_2, elv_2)

    # Filter to physical points only (above horizon in BOTH calibrations)
    physical = (elv_1 > 0) & (elv_2 > 0)
    n_phys = np.sum(physical)

    print(f"\n" + "=" * 60)
    print(f"  COSINE SHIFT COMPARISON")
    print(f"=" * 60)

    print(f"\n  Calibration A: {LABEL_A}")
    print(f"    cosShiftA = {COS_SHIFT_A_1:+.6f},  cosShiftB = {COS_SHIFT_B_1:+.6f}")
    print(f"  Calibration B: {LABEL_B}")
    print(f"    cosShiftA = {COS_SHIFT_A_2:+.6f},  cosShiftB = {COS_SHIFT_B_2:+.6f}")

    print(f"\n  Shift differences:")
    print(f"    delta cosShiftA = {COS_SHIFT_A_1 - COS_SHIFT_A_2:+.6f}")
    print(f"    delta cosShiftB = {COS_SHIFT_B_1 - COS_SHIFT_B_2:+.6f}")

    # --- All points ---
    print(f"\n  All {n} points:")
    print(f"    Mean |d_azi|:  {np.mean(np.abs(d_azi)):.3f} deg")
    print(f"    Mean |d_elv|:  {np.mean(np.abs(d_elv)):.3f} deg")
    print(f"    Mean angular separation: {np.mean(sep):.3f} deg")
    print(f"    Max angular separation:  {np.max(sep):.3f} deg")

    # --- Physical points only ---
    if n_phys > 0:
        print(f"\n  Physical points only (elv > 0 in both): {n_phys} / {n}")
        print(f"    Mean |d_azi|:  {np.mean(np.abs(d_azi[physical])):.3f} deg")
        print(f"    Mean |d_elv|:  {np.mean(np.abs(d_elv[physical])):.3f} deg")
        print(f"    Mean angular separation: {np.mean(sep[physical]):.3f} deg")
        print(f"    Median angular separation: {np.median(sep[physical]):.3f} deg")
        print(f"    90th percentile separation: {np.percentile(sep[physical], 90):.3f} deg")
        print(f"    Max angular separation:  {np.max(sep[physical]):.3f} deg")

    # --- Below-horizon comparison ---
    below_1 = np.sum(elv_1 < 0)
    below_2 = np.sum(elv_2 < 0)
    print(f"\n  Below-horizon points:")
    print(f"    {LABEL_A}: {below_1} ({100 * below_1 / n:.1f}%)")
    print(f"    {LABEL_B}: {below_2} ({100 * below_2 / n:.1f}%)")

    # --- Summary box ---
    mean_sep = np.mean(sep[physical]) if n_phys > 0 else np.mean(sep)
    med_sep = np.median(sep[physical]) if n_phys > 0 else np.median(sep)

    print(f"\n  ┌──────────────────────────────────────────────────────┐")
    print(f"  │  Mean difference between calibrations: {mean_sep:.3f} deg       │")
    print(f"  │  Median difference:                    {med_sep:.3f} deg       │")
    print(f"  └──────────────────────────────────────────────────────┘")
    print(f"=" * 60)




