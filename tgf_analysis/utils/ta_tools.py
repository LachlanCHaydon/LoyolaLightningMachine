"""
TGF Analysis Tool - TA Tools Utilities
======================================
GPS coordinate conversions and geometric calculations.
Based on the original taTools.py from the research group.
"""

import numpy as np
import geopy
from geopy.distance import geodesic

from config import CAMERA_LAT, CAMERA_LON, CAMERA_ALT, R_EARTH

# CLF (Central Laser Facility) reference point
ZCLF = 1.382
LATCLF = 39.29693
LONCLF = -112.90875
NTASD = 507


def gps2cart(gps_point):
    """
    Convert from GPS lat/lon to CLF-centered Cartesian coordinates.
    
    Parameters:
    -----------
    gps_point : geopy.point.Point
        GPS coordinates (lat, lon, alt)
        
    Returns:
    --------
    tuple : (x, y, z) in kilometers
    """
    g0000 = geopy.point.Point(LATCLF, LONCLF, ZCLF)
    g0000eq = geopy.point.Point(0.0, g0000.longitude, g0000.altitude)
    gnnnneq = geopy.point.Point(0.0, gps_point.longitude, gps_point.altitude)

    # Calculate distances using vincenty/geodesic
    try:
        from vincenty import vincenty as calc_distance
        r = calc_distance(g0000, gps_point)
        y = calc_distance(gps_point, gnnnneq) - calc_distance(g0000, g0000eq)
    except ImportError:
        # Fallback to geopy geodesic
        r = geodesic((g0000.latitude, g0000.longitude), 
                     (gps_point.latitude, gps_point.longitude)).km
        y = (geodesic((gps_point.latitude, gps_point.longitude),
                      (gnnnneq.latitude, gnnnneq.longitude)).km -
             geodesic((g0000.latitude, g0000.longitude),
                      (g0000eq.latitude, g0000eq.longitude)).km)

    if gps_point.longitude > g0000.longitude:
        x = np.sqrt(max(0, r*r - y*y))
    else:
        x = -np.sqrt(max(0, r*r - y*y))
    
    z = gps_point.altitude

    return x, y, z


def cart2gps(x, y):
    """
    Convert CLF-centered Cartesian coordinates to GPS lat/lon.
    
    Parameters:
    -----------
    x : array-like
        X coordinates in km
    y : array-like
        Y coordinates in km
        
    Returns:
    --------
    tuple : (lons, lats) arrays
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    lons = LONCLF + (x / (R_EARTH * np.sin((90.0 - LATCLF) * (np.pi / 180.0)))) * (180.0 / np.pi)
    lats = LATCLF + (y / R_EARTH) * (180.0 / np.pi)
    
    return lons, lats


def load_tasd_coordinates(filepath='tasd_gpscoors.txt'):
    """
    Load TASD detector coordinates from file.
    
    Parameters:
    -----------
    filepath : str
        Path to tasd_gpscoors.txt file
        
    Returns:
    --------
    dict : {detector_id: {'lat': lat, 'lon': lon, 'alt': alt, 'x': x, 'y': y, 'z': z}}
    """
    detectors = {}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                columns = line.split()
                if len(columns) >= 4:
                    det_id = columns[0]
                    lat = float(columns[1])
                    lon = float(columns[2])
                    alt = 0.001 * float(columns[3])  # Convert m to km
                    
                    # Calculate Cartesian coordinates
                    gps = geopy.point.Point(lat, lon, alt)
                    x, y, z = gps2cart(gps)
                    
                    detectors[det_id] = {
                        'lat': lat,
                        'lon': lon, 
                        'alt': alt,
                        'x': x,
                        'y': y,
                        'z': z
                    }
    except FileNotFoundError:
        print(f"Warning: TASD coordinates file not found: {filepath}")
    except Exception as e:
        print(f"Error loading TASD coordinates: {e}")
    
    return detectors


def calculate_azimuth_to_detector(det_lat, det_lon, ref_lat=CAMERA_LAT, ref_lon=CAMERA_LON):
    """
    Calculate azimuth angle from reference point to detector.
    
    Parameters:
    -----------
    det_lat, det_lon : float
        Detector GPS coordinates
    ref_lat, ref_lon : float
        Reference point (default: camera location)
        
    Returns:
    --------
    float : Azimuth in degrees
    """
    ref_gps = geopy.point.Point(ref_lat, ref_lon, CAMERA_ALT)
    det_gps = geopy.point.Point(det_lat, det_lon, CAMERA_ALT)
    
    ref_cart = gps2cart(ref_gps)
    det_cart = gps2cart(det_gps)
    
    dx = abs(det_cart[0] - ref_cart[0])
    dy = det_cart[1] - ref_cart[1]
    
    if dy > 0:
        azimuth = 270 + (180.0 / np.pi) * np.arctan(dy / dx)
    else:
        azimuth = 270 - (180.0 / np.pi) * np.arctan(abs(dy) / dx)
    
    return azimuth


def horizontal_distance(lat1, lon1, lat2, lon2):
    """
    Calculate horizontal distance between two GPS points.
    
    Returns distance in kilometers.
    """
    return geodesic((lat1, lon1), (lat2, lon2)).km


def slant_distance(horizontal_dist, altitude_diff):
    """
    Calculate slant distance given horizontal distance and altitude difference.
    
    Parameters:
    -----------
    horizontal_dist : float
        Horizontal distance in km
    altitude_diff : float
        Altitude difference in km
        
    Returns:
    --------
    float : Slant distance in km
    """
    return np.sqrt(horizontal_dist**2 + altitude_diff**2)


def elevation_to_altitude(elevation_deg, horizontal_distance_km):
    """
    Convert elevation angle to altitude.
    
    Parameters:
    -----------
    elevation_deg : float
        Elevation angle in degrees
    horizontal_distance_km : float
        Horizontal distance to source in km
        
    Returns:
    --------
    float : Altitude in km
    """
    return horizontal_distance_km * np.tan(np.radians(elevation_deg))


def altitude_to_elevation(altitude_km, horizontal_distance_km):
    """
    Convert altitude to elevation angle.
    
    Parameters:
    -----------
    altitude_km : float
        Altitude in km
    horizontal_distance_km : float
        Horizontal distance to source in km
        
    Returns:
    --------
    float : Elevation angle in degrees
    """
    return np.degrees(np.arctan(altitude_km / horizontal_distance_km))


def ivanov2clf(x, y):
    """Convert Ivanov coordinates to CLF coordinates."""
    x = np.asarray(x)
    y = np.asarray(y)
    return (x - 12.2435) * 1.200, (y - 16.4406) * 1.200


def clf2ivanov(x, y):
    """Convert CLF coordinates to Ivanov coordinates."""
    x = np.asarray(x)
    y = np.asarray(y)
    return x / 1.200 + 12.2435, y / 1.200 + 16.4406
