#!/usr/bin/env python
"""
INTF Camera Overlay Calibration Tool (Manual Mode)
=====================================================
Three-stage script for calibrating cosine shifts.

Stage 1 (STAGE = 1):
  - Displays raw camera frame with pixel coordinates
  - User identifies the RS ground contact pixel (x, y)
  - Enter these values in ANCHOR_PIXEL_X, ANCHOR_PIXEL_Y

Stage 2 (STAGE = 2):
  - Displays camera frame in angular coordinates (anchored to NLDN)
  - Overlays INTF points with current cosine shifts
  - Visually compare INTF channel with camera channel
  - Adjust COS_SHIFT_A / COS_SHIFT_B and re-run until aligned

Stage 3 (STAGE = 3):
  - Computes cosine shifts from raw INTF data at RS time
  - RS time given as (INTF_TRIGGER_US - RS_TIME_US) offset
  - Averages or takes median of cosA/cosB in a time window
  - Calculates shifts using: cosShiftA = cos(azi_true) - cosA_raw
                              cosShiftB = sin(azi_true) - cosB_raw
  - Physical constraint: elevation = 0 deg at ground strike (p = 1)

Camera: Phantom V2012 (2023 specs)
  - Resolution: 1280 x 448 pixels (timestamp bar removed)
  - Horizontal FOV: 84 deg
  - Vertical FOV: 35 deg

Author: Lachlan (
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import geopy
from vincenty import vincenty
import sys
import os

# ============================================================
# Camera specifications (Phantom V2012, 2023 paper)
# ============================================================
CAM_HFOV_DEG = 84.0
CAM_VFOV_DEG = 35.0

# INTF antenna location
INTF_LAT = 39.339082
INTF_LON = -112.700696
INTF_ALT_KM = 1.400

# CLF reference (from taTools)
LATCLF = 39.29693
LONCLF = -112.90875
ZCLF = 1.382


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


def compute_true_azimuth(strike_lat, strike_lon):
    """Compute true azimuth from INTF antenna to NLDN strike."""
    x_intf, y_intf = gps2cart(INTF_LAT, INTF_LON)
    x_strike, y_strike = gps2cart(strike_lat, strike_lon)
    dx = x_strike - x_intf
    dy = y_strike - y_intf
    azimuth = np.degrees(np.arctan2(dx, dy))
    if azimuth < 0:
        azimuth += 360.0
    return azimuth


def compute_pixel_to_angle_mapping(anchor_px, anchor_py, true_azi,
                                    img_width, img_height):
    """
    Compute the angular extent of the camera image given an anchor point.

    The anchor pixel is mapped to (true_azi, 0 deg elevation).
    FOV defines the angular scale per pixel.

    Returns: extent [azi_min, azi_max, elv_min, elv_max]
    """
    deg_per_pix_h = CAM_HFOV_DEG / img_width
    deg_per_pix_v = CAM_VFOV_DEG / img_height

    azi_min = true_azi - anchor_px * deg_per_pix_h
    azi_max = azi_min + img_width * deg_per_pix_h

    # Elevation increases upward; pixel y increases downward
    # anchor_py maps to elevation = 0 deg
    elv_max = 0.0 + anchor_py * deg_per_pix_v
    elv_min = elv_max - img_height * deg_per_pix_v

    return [azi_min, azi_max, elv_min, elv_max]


# ============================================================
# INTF data
# ============================================================
def load_intf_raw(filepath, skip_header=57):
    """Load raw INTF .dat file. Returns dict of arrays."""
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


def load_intf_calibrated(filepath, skip_header=2):
    """Load calibrated INTF file. Returns dict of arrays."""
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
    """
    Apply cosine shifts and compute calibrated azimuth/elevation.
    Matches the convention in INT_data_cut_2024.py exactly.
    """
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


# ============================================================
# Visualization helpers
# ============================================================
def compute_intf_sizes(pk2pk):
    """
    Compute marker sizes from pk2pk signal strength.
    Uses log-normalized scaling consistent with existing scripts.
    Small base size for overlay visibility.
    """
    ss = np.log10(np.clip(pk2pk, 1e-10, None))
    aMin, aMax = np.min(ss), np.max(ss)
    if aMax > aMin:
        ss_norm = (ss - aMin) / (aMax - aMin)
    else:
        ss_norm = np.zeros_like(ss)
    ss_norm = np.clip(ss_norm, 0, 1)

    s = (1 + 3 * ss_norm**2)**2
    marker_sizes = 2.5 * s  # Small for camera visibility

    return marker_sizes


# ============================================================
# STAGE 1: Show raw image for pixel identification
# ============================================================
def stage1_find_anchor(image_path, timestamp_bar_height=0):
    """Display raw camera frame with pixel coordinates and crosshair grid."""
    img = Image.open(image_path)
    image_array = np.array(img)
    if timestamp_bar_height > 0:
        image_array = image_array[:-timestamp_bar_height, :]

    h, w = image_array.shape[:2]
    print(f"\n  Image: {w} x {h} pixels")
    print(f"  Hover over the RS ground contact point and read pixel (x, y)")
    print(f"  from the matplotlib toolbar at the bottom of the window.")
    print(f"  Then enter these values as ANCHOR_PIXEL_X / Y.\n")

    fig, ax = plt.subplots(1, 1, figsize=(18, 7))

    # Enhance contrast for dark frames
    vmax = np.percentile(image_array, 100)
    ax.imshow(image_array, cmap='gray', aspect='auto', origin='upper',
              vmin=0, vmax=max(vmax, 1))

    ax.set_xlabel('Pixel X', fontsize=12)
    ax.set_ylabel('Pixel Y', fontsize=12)
    ax.set_title(f'Stage 1: Identify RS Ground Contact Pixel\n'
                 f'{os.path.basename(image_path)}  ({w}x{h})',
                 fontsize=13)

    # Grid for coordinate reading
    ax.grid(True, linestyle=':', alpha=0.3, color='cyan')
    ax.axhline(h / 2, color='cyan', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.axvline(w / 2, color='cyan', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# STAGE 2: Overlay INTF on camera with angular coordinates
# ============================================================
def stage2_overlay(image_path, intf_data, true_azimuth,
                   anchor_px, anchor_py,
                   cosShiftA, cosShiftB,
                   timestamp_bar_height=0,
                   time_range_ms=None,
                   azi_range=None, elv_range=None):
    """
    Display camera frame in angular coordinates with INTF overlay.
    """
    # Load image
    img = Image.open(image_path)
    image_array = np.array(img)
    if timestamp_bar_height > 0:
        image_array = image_array[:-timestamp_bar_height, :]
    img_h, img_w = image_array.shape[:2]

    # Compute angular mapping
    extent = compute_pixel_to_angle_mapping(
        anchor_px, anchor_py, true_azimuth, img_w, img_h
    )

    print(f"\n  Camera angular mapping:")
    print(f"    Anchor pixel: ({anchor_px}, {anchor_py})")
    print(f"    True azimuth: {true_azimuth:.3f} deg")
    print(f"    Azimuth range:   [{extent[0]:.2f} deg, {extent[1]:.2f} deg]")
    print(f"    Elevation range: [{extent[2]:.2f} deg, {extent[3]:.2f} deg]")
    print(f"    Scale: {CAM_HFOV_DEG/img_w:.4f} deg/pix (H), "
          f"{CAM_VFOV_DEG/img_h:.4f} deg/pix (V)")

    # Apply cosine shifts to INTF data
    azi_cal, elv_cal = apply_cos_shifts(
        intf_data['cosA'], intf_data['cosB'], cosShiftA, cosShiftB
    )

    # Build mask from all filters
    mask = np.ones(len(azi_cal), dtype=bool)

    if time_range_ms is not None:
        mask &= (intf_data['time_ms'] >= time_range_ms[0])
        mask &= (intf_data['time_ms'] <= time_range_ms[1])

    if azi_range is not None:
        mask &= (azi_cal >= azi_range[0])
        mask &= (azi_cal <= azi_range[1])

    if elv_range is not None:
        mask &= (elv_cal >= elv_range[0])
        mask &= (elv_cal <= elv_range[1])

    azi_plot = azi_cal[mask]
    elv_plot = elv_cal[mask]
    time_plot = intf_data['time_ms'][mask]
    pk2pk_plot = intf_data['pk2pk'][mask]
    n_pts = len(azi_plot)

    print(f"\n  INTF overlay:")
    print(f"    cosShiftA = {cosShiftA:+.6f}")
    print(f"    cosShiftB = {cosShiftB:+.6f}")
    print(f"    Total INTF points: {len(azi_cal)}")
    print(f"    After filters: {n_pts}")
    if n_pts > 0:
        print(f"    Azimuth:   [{azi_plot.min():.2f} deg, {azi_plot.max():.2f} deg]")
        print(f"    Elevation: [{elv_plot.min():.2f} deg, {elv_plot.max():.2f} deg]")
        below = np.sum(elv_plot < 0)
        print(f"    Below ground: {below} ({100 * below / n_pts:.1f}%)")

    # Compute marker sizes
    if n_pts > 0:
        marker_sizes = compute_intf_sizes(pk2pk_plot)

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))

    # Camera image with angular extent (enhanced contrast)
    vmax = np.percentile(image_array, 99.9)
    ax.imshow(image_array, cmap='gray', aspect='auto',
              extent=extent, origin='upper', vmin=0, vmax=max(vmax, 1),
              zorder=0)

    # # RS ground contact marker
    # ax.plot(true_azimuth, 0.0, 'r+', markersize=20, markeredgewidth=2.5,
    #         zorder=10, label='RS ground (NLDN)')
    #
    # # Horizon line
    # ax.axhline(0, color='r', linestyle='--', linewidth=1.0, alpha=0.6,
    #            zorder=5, label='Elevation = 0 deg')

    # INTF scatter
    if n_pts > 0:
        sc = ax.scatter(azi_plot, elv_plot, c=time_plot, s=marker_sizes,
                        cmap='jet', alpha=0.7, edgecolors='k',
                        linewidths=0.15, zorder=6,
                        label=f'INTF ({n_pts} pts)')
        cbar = plt.colorbar(sc, ax=ax, pad=0.01, shrink=0.85)
        cbar.set_label('Time (ms rel. to INTF trigger)', fontsize=10)

    # Formatting
    ax.set_xlabel('Azimuth (deg)', fontsize=12)
    ax.set_ylabel('Elevation (deg)', fontsize=12)
    ax.set_title(
        f'Camera + INTF Overlay\n'
        f'cosShiftA = {cosShiftA:+.6f}   |   cosShiftB = {cosShiftB:+.6f}   |   '
        f'{n_pts} points',
        fontsize=12
    )
    ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
    ax.grid(True, linestyle=':', alpha=0.25, color='white')
    ax.tick_params(labelsize=10)

    # Zoom to data region if points exist and no manual azi/elv range
    if n_pts > 0 and azi_range is None and elv_range is None:
        pad_azi = 2.0
        pad_elv = 2.0
        ax.set_xlim(azi_plot.min() - pad_azi, azi_plot.max() + pad_azi)
        ax.set_ylim(min(elv_plot.min(), -1) - pad_elv,
                     elv_plot.max() + pad_elv)

    plt.tight_layout()

    # Save
    outname = 'camera_intf_overlay.png'
    fig.savefig(outname, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {outname}")

    plt.show()

    # --- Diagnostic output ---
    if n_pts > 0:
        print("\n" + "=" * 60)
        print("  DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"  Current shifts:  cosShiftA = {cosShiftA:+.6f}")
        print(f"                   cosShiftB = {cosShiftB:+.6f}")
        print(f"  Points below elv=0 deg: {below} / {n_pts} "
              f"({100 * below / n_pts:.1f}%)")

        # Early vs late point statistics
        t_sorted = np.sort(time_plot)
        if len(t_sorted) > 20:
            early_cut = t_sorted[int(0.1 * len(t_sorted))]
            late_cut = t_sorted[int(0.9 * len(t_sorted))]
            early_mask = time_plot <= early_cut
            late_mask = time_plot >= late_cut

            print(f"\n  Earliest 10% of points (cloud-level activity):")
            print(f"    Time <= {early_cut:.3f} ms")
            print(f"    Mean elv = {elv_plot[early_mask].mean():.2f} deg, "
                  f"range = [{elv_plot[early_mask].min():.2f} deg, "
                  f"{elv_plot[early_mask].max():.2f} deg]")
            print(f"    Mean azi = {azi_plot[early_mask].mean():.2f} deg")

            print(f"  Latest 10% of points (near RS / ground):")
            print(f"    Time >= {late_cut:.3f} ms")
            print(f"    Mean elv = {elv_plot[late_mask].mean():.2f} deg, "
                  f"range = [{elv_plot[late_mask].min():.2f} deg, "
                  f"{elv_plot[late_mask].max():.2f} deg]")
            print(f"    Mean azi = {azi_plot[late_mask].mean():.2f} deg")

        print(f"\n  ADJUSTMENT GUIDE:")
        print(f"    Points too LOW   -> make cosShiftB less negative (increase)")
        print(f"    Points too HIGH  -> make cosShiftB more negative (decrease)")
        print(f"    Points too RIGHT -> make cosShiftA more negative (decrease)")
        print(f"    Points too LEFT  -> make cosShiftA more positive (increase)")
        print("=" * 60)

    return extent


# ============================================================
# STAGE 3: Compute cosine shifts from RS time
# ============================================================
def stage3_compute_shifts(intf_data, true_azimuth,
                          rs_time_ms, rs_window_ms=0.05,
                          sigma_clip_thresh=2.0, sigma_clip_n=2):
    """
    Compute cosShiftA and cosShiftB from raw INTF data at the RS time.

    Uses two constraints:
      1. elevation = 0 deg at RS  ->  p = sqrt(Ac^2 + Bc^2) = 1
      2. azimuth = azi_true       ->  arctan2(Bc, Ac) = azi_true

    Solution (from derivation):
      cosShiftA = cos(azi_true) - cosA_raw
      cosShiftB = sin(azi_true) - cosB_raw

    Aggregation method:
      1. Iterative sigma clipping to remove outlier INTF points
      2. Pk2Pk-weighted mean of surviving points

    Parameters
    ----------
    intf_data : dict from load_intf_raw
    true_azimuth : float, NLDN-derived true azimuth (deg)
    rs_time_ms : float, RS time in ms relative to INTF trigger
    rs_window_ms : float, half-width of time window around RS (ms)
    sigma_clip_thresh : float, sigma threshold for clipping (default 2.0)
    sigma_clip_n : int, number of clipping iterations (default 2)
    """
    print("\n" + "=" * 60)
    print("  STAGE 3: Compute Cosine Shifts from RS")
    print("=" * 60)

    # Find INTF points in the RS time window
    t = intf_data['time_ms']
    mask = np.abs(t - rs_time_ms) <= rs_window_ms
    n_pts = np.sum(mask)

    print(f"\n  RS time: {rs_time_ms:.6f} ms (relative to INTF trigger)")
    print(f"  Window:  +/- {rs_window_ms} ms")
    print(f"  Points found: {n_pts}")

    if n_pts == 0:
        print(f"\n  ERROR: No INTF points found in window.")
        print(f"  INTF time range: [{t.min():.3f}, {t.max():.3f}] ms")
        print(f"  Looking at: {rs_time_ms:.3f} ms")
        print(f"  Try increasing rs_window_ms.")
        return None

    cosA_pts = intf_data['cosA'][mask]
    cosB_pts = intf_data['cosB'][mask]
    azi_pts = intf_data['azimuth'][mask]
    elv_pts = intf_data['elevation'][mask]
    pk2pk_pts = intf_data['pk2pk'][mask]
    time_pts = t[mask]

    # Show all points in window
    print(f"\n  Points in RS window:")
    print(f"  {'Time (ms)':>12s}  {'cosA':>10s}  {'cosB':>10s}  "
          f"{'Azi (deg)':>10s}  {'Elv (deg)':>10s}  {'Pk2Pk':>8s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for i in range(min(n_pts, 40)):  # Cap printout at 40 lines
        print(f"  {time_pts[i]:12.6f}  {cosA_pts[i]:10.6f}  {cosB_pts[i]:10.6f}  "
              f"{azi_pts[i]:10.3f}  {elv_pts[i]:10.3f}  {pk2pk_pts[i]:8.1f}")
    if n_pts > 40:
        print(f"  ... ({n_pts - 40} more points not shown)")

    # ---- Robust aggregation: sigma-clip then pk2pk-weighted mean ----
    #
    # Why this method:
    #   The RS produces a dense cluster of coherent INTF solutions at the
    #   true source location. However, some points in the time window are
    #   noise or mislocated sources that scatter randomly in cosA/cosB space.
    #   Simple mean/median treats all points equally and gets pulled by outliers.
    #
    #   Step 1 - Iterative sigma clipping (2 rounds, 2-sigma):
    #     Compute median of cosA and cosB. Reject any point where EITHER
    #     cosA or cosB is more than N sigma from the median. Recompute.
    #     This removes the sparse outliers that scatter below the horizon.
    #
    #   Step 2 - Pk2Pk-weighted mean of survivors:
    #     Stronger signal (higher pk2pk) = more reliable interferometric
    #     solution. Weighting by pk2pk ensures the brightest, most coherent
    #     sources dominate the average. This is physically motivated: the RS
    #     channel produces the strongest VHF emission, so the highest pk2pk
    #     points are most likely to be correctly located.

    print(f"\n  --- Robust aggregation ---")

    # Start with all points
    keep = np.ones(n_pts, dtype=bool)

    for iteration in range(sigma_clip_n):
        cA = cosA_pts[keep]
        cB = cosB_pts[keep]

        medA = np.median(cA)
        medB = np.median(cB)
        stdA = np.std(cA)
        stdB = np.std(cB)

        if stdA == 0 or stdB == 0:
            break

        # Reject points where EITHER cosA or cosB deviates too much
        devA = np.abs(cosA_pts - medA) / stdA
        devB = np.abs(cosB_pts - medB) / stdB
        keep_new = (devA <= sigma_clip_thresh) & (devB <= sigma_clip_thresh)

        n_rejected = np.sum(keep) - np.sum(keep_new)
        keep = keep_new

        print(f"    Clip iteration {iteration + 1}: "
              f"rejected {n_rejected}, remaining {np.sum(keep)}")

    n_surviving = np.sum(keep)
    if n_surviving == 0:
        print(f"    WARNING: All points rejected. Falling back to full median.")
        keep = np.ones(n_pts, dtype=bool)
        n_surviving = n_pts

    cosA_surv = cosA_pts[keep]
    cosB_surv = cosB_pts[keep]
    pk2pk_surv = pk2pk_pts[keep]

    # Pk2Pk-weighted mean
    weights = pk2pk_surv / np.sum(pk2pk_surv)
    cosA_raw = np.sum(cosA_surv * weights)
    cosB_raw = np.sum(cosB_surv * weights)

    # Also compute unweighted for comparison
    cosA_unweighted = np.mean(cosA_surv)
    cosB_unweighted = np.mean(cosB_surv)
    cosA_simple_median = np.median(cosA_pts)
    cosB_simple_median = np.median(cosB_pts)

    print(f"\n  Comparison of methods ({n_pts} initial, {n_surviving} after clip):")
    print(f"    Simple median (all pts):     cosA = {cosA_simple_median:.6f}, "
          f"cosB = {cosB_simple_median:.6f}")
    print(f"    Clipped unweighted mean:     cosA = {cosA_unweighted:.6f}, "
          f"cosB = {cosB_unweighted:.6f}")
    print(f"    Clipped pk2pk-weighted mean: cosA = {cosA_raw:.6f}, "
          f"cosB = {cosB_raw:.6f}  <-- USED")
    print(f"    Clipped std:  cosA = {np.std(cosA_surv):.6f}, "
          f"cosB = {np.std(cosB_surv):.6f}")

    # Raw p and angles for the aggregated point
    p_raw = np.sqrt(cosA_raw**2 + cosB_raw**2)
    azi_raw = np.degrees(np.arctan2(cosB_raw, cosA_raw))
    if azi_raw < 0:
        azi_raw += 360.0
    if p_raw <= 1.0:
        elv_raw = np.degrees(np.arccos(p_raw))
    else:
        elv_raw = -np.degrees(np.arccos(2 - p_raw))

    print(f"\n  Raw aggregated point:")
    print(f"    cosA = {cosA_raw:.6f},  cosB = {cosB_raw:.6f}")
    print(f"    p    = {p_raw:.6f}")
    print(f"    azi  = {azi_raw:.3f} deg,  elv = {elv_raw:.3f} deg")

    # Compute true direction cosines
    # Constraint: elv_true = 0 deg -> cos(elv) = 1
    # So: cosA_true = cos(azi_true), cosB_true = sin(azi_true)
    azi_true_rad = np.radians(true_azimuth)
    cosA_true = np.cos(azi_true_rad)
    cosB_true = np.sin(azi_true_rad)

    print(f"\n  True values (from NLDN, elv = 0 deg):")
    print(f"    azi_true  = {true_azimuth:.3f} deg")
    print(f"    cosA_true = {cosA_true:.6f}")
    print(f"    cosB_true = {cosB_true:.6f}")
    print(f"    p_true    = {np.sqrt(cosA_true**2 + cosB_true**2):.6f} (should be 1.000)")

    # Compute shifts
    cosShiftA = cosA_true - cosA_raw
    cosShiftB = cosB_true - cosB_raw

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  cosShiftA = {cosShiftA:+.6f}                  │")
    print(f"  │  cosShiftB = {cosShiftB:+.6f}                  │")
    print(f"  └─────────────────────────────────────────┘")

    # Verify: apply shifts to raw
    Ac = cosA_raw + cosShiftA
    Bc = cosB_raw + cosShiftB
    p_cal = np.sqrt(Ac**2 + Bc**2)
    azi_cal = np.degrees(np.arctan2(Bc, Ac))
    if azi_cal < 0:
        azi_cal += 360.0
    if p_cal <= 1.0:
        elv_cal = np.degrees(np.arccos(p_cal))
    else:
        elv_cal = -np.degrees(np.arccos(2 - p_cal))

    print(f"\n  Verification (shifts applied to raw):")
    print(f"    Ac  = {Ac:.6f},  Bc  = {Bc:.6f}")
    print(f"    p   = {p_cal:.6f} (should be 1.000)")
    print(f"    azi = {azi_cal:.3f} deg (should be {true_azimuth:.3f} deg)")
    print(f"    elv = {elv_cal:.3f} deg (should be ~0 deg)")

    # Angular changes
    print(f"\n  Angular correction applied:")
    print(f"    Delta azi = {azi_cal - azi_raw:+.3f} deg")
    print(f"    Delta elv = {elv_cal - elv_raw:+.3f} deg")

    # Compare with 2024 values
    print(f"\n  Comparison with 2024 shifts:")
    print(f"    2024:  cosShiftA = -0.0051,  cosShiftB = -0.0178")
    print(f"    New:   cosShiftA = {cosShiftA:+.6f},  cosShiftB = {cosShiftB:+.6f}")

    print("\n" + "=" * 60)

    return {
        'cosShiftA': cosShiftA,
        'cosShiftB': cosShiftB,
        'cosA_raw': cosA_raw,
        'cosB_raw': cosB_raw,
        'cosA_true': cosA_true,
        'cosB_true': cosB_true,
        'azi_raw': azi_raw,
        'elv_raw': elv_raw,
        'azi_true': true_azimuth,
        'p_raw': p_raw,
        'n_points_initial': n_pts,
        'n_points_after_clip': n_surviving,
        'method': 'sigma-clip + pk2pk-weighted mean'
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':

    # ===================== MODE CONTROL ======================
    # 1 = Stage 1: show raw image to find anchor pixel
    # 2 = Stage 2: show INTF overlay on camera
    # 3 = Stage 3: compute cosine shifts from RS time
    STAGE = 2
    # =========================================================

    # ============= USER PARAMETERS ===========================

    # Camera frame
    IMAGE_PATH = "../Img000116.tif"

    # INTF data file
    INTF_FILE = "../INTF/TAR_2025.09.21_02-53-53_335955_pix200_S256-I64-P4_W3.dat"
    INTF_IS_RAW = True
    INTF_RAW_SKIP_HEADER = 57

    # Cosine shifts (iterate on these until overlay aligns)
    COS_SHIFT_A = -0.003575
    COS_SHIFT_B = -0.01328

    # NLDN ground strike
    NLDN_STRIKE_LAT = 39.2968
    NLDN_STRIKE_LON = -112.9084


    # RS ground contact pixel (from Stage 1)
    ANCHOR_PIXEL_X = 832  # UPDATE after running Stage 1
    ANCHOR_PIXEL_Y = 395 # UPDATE after running Stage 1

    # Timestamp bar crop (0 if already removed)
    TIMESTAMP_BAR_HEIGHT = 36

    # Filters for Stage 2 (None = no filter)
    TIME_RANGE_MS = (-100,-1)     # e.g. (-5.0, 1.0)
    AZI_RANGE = (200,300)      # e.g. (250.0, 265.0)
    ELV_RANGE = (-10,20)        # e.g. (-10.0, 20.0)

    # --- Stage 3 parameters ---
    # RS time: INTF_TRIGGER_US - RS_TIME_US, converted to ms
    # For Sep 21: trigger = 329684 us, RS = 327447 us
    # RS offset = (329684 - 327447) = 2237 us = 2.237 ms
    # INTF raw time is negative before trigger, so RS is at:
    RS_TIME_MS = -2.314784      # ms relative to INTF trigger
    RS_WINDOW_MS = 0.025   # +/- window in ms
    SIGMA_CLIP_THRESH = 1  # sigma threshold for outlier rejection
    SIGMA_CLIP_N = 2           # number of clipping iterations

    # =========================================================

    # Load INTF data (needed for stages 2 and 3)
    if STAGE in [2, 3]:
        print("\n  Loading INTF data...")
        if INTF_IS_RAW:
            intf = load_intf_raw(INTF_FILE, INTF_RAW_SKIP_HEADER)
        else:
            intf = load_intf_calibrated(INTF_FILE)
        print(f"    {len(intf['time_ms'])} points loaded")
        print(f"    Time: [{intf['time_ms'].min():.3f}, "
              f"{intf['time_ms'].max():.3f}] ms")

        true_azi = compute_true_azimuth(NLDN_STRIKE_LAT, NLDN_STRIKE_LON)
        print(f"    True azimuth to NLDN strike: {true_azi:.3f} deg")

    if STAGE == 1:
        print("=" * 60)
        print("  STAGE 1: Identify RS ground contact pixel")
        print("=" * 60)
        stage1_find_anchor(IMAGE_PATH, TIMESTAMP_BAR_HEIGHT)

    elif STAGE == 2:
        print("=" * 60)
        print("  STAGE 2: Camera + INTF Overlay")
        print("=" * 60)

        extent = stage2_overlay(
            image_path=IMAGE_PATH,
            intf_data=intf,
            true_azimuth=true_azi,
            anchor_px=ANCHOR_PIXEL_X,
            anchor_py=ANCHOR_PIXEL_Y,
            cosShiftA=COS_SHIFT_A,
            cosShiftB=COS_SHIFT_B,
            timestamp_bar_height=TIMESTAMP_BAR_HEIGHT,
            time_range_ms=TIME_RANGE_MS,
            azi_range=AZI_RANGE,
            elv_range=ELV_RANGE
        )

        print(f"\n  Camera extent for other scripts:")
        print(f"    extent = [{extent[0]:.2f}, {extent[1]:.2f}, "
              f"{extent[2]:.2f}, {extent[3]:.2f}]")

    elif STAGE == 3:
        results = stage3_compute_shifts(
            intf_data=intf,
            true_azimuth=true_azi,
            rs_time_ms=RS_TIME_MS,
            rs_window_ms=RS_WINDOW_MS,
            sigma_clip_thresh=SIGMA_CLIP_THRESH,
            sigma_clip_n=SIGMA_CLIP_N
        )

        if results:
            print(f"\n  To use in Stage 2, update:")
            print(f"    COS_SHIFT_A = {results['cosShiftA']:.6f}")
            print(f"    COS_SHIFT_B = {results['cosShiftB']:.6f}")

    else:
        print(f"  ERROR: STAGE must be 1, 2, or 3. Got: {STAGE}")