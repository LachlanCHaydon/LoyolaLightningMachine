"""
TGF Analysis Tool - GUI Tab Modules
===================================
Individual tab implementations.
"""

from gui.tabs.home_plotter import HomePlotterTab
from gui.tabs.intf_tab import INTFTab
from gui.tabs.photometry_tab import PhotometryTab
from gui.tabs.spectroscopy_tab import SpectroscopyTab
from gui.tabs.timeshift_tab import TimeshiftTab
from gui.tabs.luminosity_tab import LuminosityTab
from gui.tabs.figures import FlashOverviewTab, Figure1Tab, Figure2Tab, Figure3Tab, VelocityTab

__all__ = ['HomePlotterTab', 'INTFTab', 'PhotometryTab', 'SpectroscopyTab',
           'TimeshiftTab', 'LuminosityTab',
           'FlashOverviewTab', 'Figure1Tab', 'Figure2Tab', 'Figure3Tab', 'VelocityTab']
