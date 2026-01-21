"""
TGF Analysis Tool - Configuration and Constants
================================================
Contains known spectral lines, colormaps, and global configuration.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# Application Info
# =============================================================================
APP_NAME = "Loyola Lightning Machine (LLM)"
APP_VERSION = "1.0"

# =============================================================================
# Known Spectral Lines for Lightning
# =============================================================================
# Wavelength (nm): (Species, Priority, Is_Anchor)
# Priority: 0 = anchor (must identify), 1 = common, 2 = occasional, 3 = rare
KNOWN_LINES = {
    424: ('NII', 2, False),
    463: ('NII', 1, False),
    500: ('NII', 1, False),
    568: ('NII', 2, False),
    656: ('HI', 0, True),   # Hα - PRIMARY ANCHOR
    715: ('NI', 3, False),
    744: ('NI', 1, False),
    777: ('OI', 0, True),   # OI triplet - PRIMARY ANCHOR
    795: ('OI', 2, False),
    822: ('NI', 2, False),
    824: ('OI', 1, False),
    844: ('OI', 2, False),
    868: ('NI', 2, False),
}

# =============================================================================
# Camera/Detector Locations
# =============================================================================
CAMERA_LAT = 39.339082
CAMERA_LON = -112.700696
CAMERA_ALT = 1.4  # km

# INTF antenna location (same as camera for most setups)
INTF_LAT = 39.339082
INTF_LON = -112.700696

# Earth radius for calculations
R_EARTH = 6371  # km

# =============================================================================
# Default INTF Cosine Shifts (2024 calibration values)
# =============================================================================
DEFAULT_COS_SHIFT_A = -0.0051
DEFAULT_COS_SHIFT_B = -0.0178

# =============================================================================
# Photometer Calibration Constants
# =============================================================================
PHOTOMETER_CALIBRATION = {
    # Channel 0: 337nm
    0: {
        'wavelength': 337,
        'T': 0.325,          # Filter transmittance
        'rsk': 60.000,       # Cathode Radiance Sensitivity
        'A': 907.46e-6,      # Effective area (m²)
        'G': 24484.37309,    # Gain
        'R': 1,              # Load resistance (kΩ)
        'color': 'blue',
        'label': '337 nm'
    },
    # Channel 1: 391nm  
    1: {
        'wavelength': 391,
        'T': 0.3125,
        'rsk': 80.000,
        'A': 907.46e-6,
        'G': 24484.37309,
        'R': 1,
        'color': 'purple',
        'label': '391 nm'
    },
    # Channel 2: 777nm
    2: {
        'wavelength': 777,
        'T': 0.475,
        'rsk': 14.531,
        'A': 586.7e-6,
        'G': 67145.67,
        'R': 1,
        'color': 'red',
        'label': '777 nm'
    }
}

# Default photometer sample rate
PHOTOMETER_SAMPLE_RATE = 20e6  # 20 MHz
PHOTOMETER_RECORD_LENGTH = int(20e6)  # 20M samples = 1 second

# =============================================================================
# Colormaps
# =============================================================================

# Viridis colormap for spectroscopy
VIRIDIS_COLORS = [
    (0.267004, 0.004874, 0.329415),
    (0.282327, 0.140926, 0.457517),
    (0.253935, 0.265254, 0.529983),
    (0.206756, 0.371758, 0.553117),
    (0.163625, 0.471133, 0.558148),
    (0.127568, 0.566949, 0.550556),
    (0.134692, 0.658636, 0.517649),
    (0.266941, 0.748751, 0.440573),
    (0.477504, 0.821444, 0.318195),
    (0.741388, 0.873449, 0.149561),
    (0.993248, 0.906157, 0.143936)
]

def create_viridis_cmap():
    """Create viridis colormap for matplotlib."""
    return LinearSegmentedColormap.from_list('viridis_custom', VIRIDIS_COLORS, N=256)

# MJET colormap for INTF data visualization
MJET_COLOR_DICT = {
    'red': (
        (0.00, 0.3, 0.3),
        (0.15, 0.0, 0.0),
        (0.20, 0.0, 0.0),
        (0.50, 0.0, 0.0),
        (0.70, 0.7, 0.7),
        (0.90, 1.0, 1.0),
        (1.00, 0.9, 0.9)
    ),
    'green': (
        (0.00, 0.0, 0.0),
        (0.15, 0.0, 0.0),
        (0.25, 0.3, 0.3),
        (0.45, 1.0, 1.0),
        (0.70, 0.9, 0.9),
        (0.90, 0.0, 0.0),
        (1.00, 0.0, 0.0)
    ),
    'blue': (
        (0.00, 0.5, 0.5),
        (0.20, 1.0, 1.0),
        (0.45, 1.0, 1.0),
        (0.50, 0.1, 0.1),
        (0.70, 0.0, 0.0),
        (0.90, 0.0, 0.0),
        (1.00, 0.0, 0.0)
    )
}

def create_mjet_cmap():
    """Create the mjet colormap used for INTF data visualization."""
    return LinearSegmentedColormap('mjet', MJET_COLOR_DICT, 256)

def cmap_mjet(values):
    """
    Apply mjet colormap to normalized values (0-1).
    Returns RGBA colors for each value.
    """
    cmap = create_mjet_cmap()
    return cmap(np.clip(values, 0, 1))

# =============================================================================
# Default Plot Settings
# =============================================================================
DEFAULT_FONT_SIZE = 18
DEFAULT_FIGURE_SIZE = (14, 8)

# Default axis limits (can be overridden per-project)
DEFAULT_LIMITS = {
    'fa': {'y_min': -80, 'y_max': 30},
    'sd': {'y_min': 0, 'y_max': 500},
    'intf_elev': {'y_min': 0, 'y_max': 40},
    'intf_azi': {'y_min': 270, 'y_max': 330},
    'luminosity': {'y_min': 0, 'y_max': 1.1},
    'photometer': {'y_min': -2000, 'y_max': 55000},
}

# Default colors for instruments
INSTRUMENT_COLORS = {
    'fa': 'green',
    'sd': 'magenta',
    'intf': 'red',
    'luminosity': 'tab:olive',
    'photometer_337': 'blue',
    'photometer_391': 'purple', 
    'photometer_777': 'red',
}

# =============================================================================
# File Extensions
# =============================================================================
PROJECT_EXTENSION = '.tgf'
SUPPORTED_IMAGE_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
