"""
TGF Lightning Analysis Tool
===========================

A comprehensive tool for analyzing Terrestrial Gamma-ray Flash (TGF) 
observations from the Telescope Array Surface Detector.

Modules:
--------
- config: Configuration constants and known spectral lines
- data_handlers: Load and process various data types
- analysis: Scientific analysis functions
- gui: Graphical user interface
- utils: Utility functions

Usage:
------
    from tgf_analysis import main
    main.run()

Or from command line:
    python -m tgf_analysis

Version: 2.0.0
"""

from config import APP_NAME, APP_VERSION

__version__ = APP_VERSION
__author__ = "TGF Research Group"

# Convenience imports
from data_handlers import (
    FrameData,
    PhotometerHandler,
    FastAntennaHandler,
    InterferometerHandler,
    TASDHandler,
    LMAHandler,
    LuminosityHandler
)

from analysis import (
    SpectroscopyAnalyzer,
    TimeshiftCalculator,
    INTFCalibrator
)

from utils import (
    ProjectState,
    save_project,
    load_project
)

__all__ = [
    # Config
    'APP_NAME', 'APP_VERSION',
    
    # Data handlers
    'FrameData', 'PhotometerHandler', 'FastAntennaHandler',
    'InterferometerHandler', 'TASDHandler', 'LMAHandler', 'LuminosityHandler',
    
    # Analysis
    'SpectroscopyAnalyzer', 'TimeshiftCalculator', 'INTFCalibrator',
    
    # Utils
    'ProjectState', 'save_project', 'load_project',
]
