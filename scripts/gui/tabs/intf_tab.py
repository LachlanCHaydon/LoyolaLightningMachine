"""
TGF Analysis Tool - INTF (Interferometer) Tab
==============================================
Handles INTF data calibration and azimuth-elevation visualization.

Features:
- Load raw or pre-calibrated INTF data
- Apply cosine shift calibration (manual or LMA auto-calibration)
- Azimuth-Elevation scatter plots (time-colored)
- Time-Elevation plots
- TASD detector overlay on Az-El plots
- Export calibrated data to new files
- Filter by time, azimuth, elevation ranges
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os

# Matplotlib with Tk backend
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from config import (
    DEFAULT_COS_SHIFT_A, DEFAULT_COS_SHIFT_B,
    DEFAULT_FONT_SIZE, DEFAULT_LIMITS, cmap_mjet,
    INTF_LAT, INTF_LON, TASD_COORDS_FILENAME
)
from data_handlers import InterferometerHandler, TASDHandler
from utils.ta_tools import calculate_azimuth_to_detector


class INTFTab(ttk.Frame):
    """
    Interferometer calibration and visualization tab.

    Provides:
    - Cosine shift calibration for raw INTF data
    - Multiple plot modes (Az-El, Time-El, Time-Az)
    - Data filtering by time/az/el ranges
    - Export calibrated data
    """

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app

        # Data handler
        self.intf_handler = InterferometerHandler()

        # File path
        self.file_path_var = tk.StringVar()

        # Data type selection
        self.is_calibrated_var = tk.BooleanVar(value=False)
        self.skip_header_var = tk.StringVar(value="57")

        # Timing parameters
        self.T0_var = tk.StringVar(value="0")
        self.time_is_relative_var = tk.BooleanVar(value=True)
        self.event_time_var = tk.StringVar(value="HH:MM:SS")

        # Cosine shift calibration
        self.cos_shift_a_var = tk.StringVar(value=str(DEFAULT_COS_SHIFT_A))
        self.cos_shift_b_var = tk.StringVar(value=str(DEFAULT_COS_SHIFT_B))

        # Filter ranges
        self.time_start_var = tk.StringVar(value="")
        self.time_stop_var = tk.StringVar(value="")
        self.azi_min_var = tk.StringVar(value="270")
        self.azi_max_var = tk.StringVar(value="330")
        self.elv_min_var = tk.StringVar(value="0")
        self.elv_max_var = tk.StringVar(value="40")

        # Plot mode
        self.plot_mode_var = tk.StringVar(value="az_el")
        self.show_colorbar_var = tk.BooleanVar(value=True)
        self.show_grid_var = tk.BooleanVar(value=True)
        self.color_by_var = tk.StringVar(value="time")
        self.marker_size_var = tk.StringVar(value="auto")
        self.fixed_marker_size_var = tk.StringVar(value="10")
        self.plot_title_var = tk.StringVar(value="")
        self.source_distance_var = tk.StringVar(value="")

        # LMA auto-calibration
        self.lma_file_path_var = tk.StringVar()
        self.lma_trigger_offset_var = tk.StringVar(value="0.0")
        self.lma_match_window_var = tk.StringVar(value="50.0")
        self.lma_chi2_max_var = tk.StringVar(value="2.0")
        self.lma_alt_min_var = tk.StringVar(value="1000")
        self.lma_alt_max_var = tk.StringVar(value="15000")
        self.lma_power_min_var = tk.StringVar(value="-20.0")
        self.lma_result_text = None

        # TASD overlay
        self.tasd_dir_var = tk.StringVar()
        self.show_tasd_var = tk.BooleanVar(value=False)
        self.source_azi_var = tk.StringVar(value="")
        self.source_elv_var = tk.StringVar(value="")
        self.tasd_handler = None

        # Build UI
        self._build_ui()

    # =========================================================================
    # UI Construction
    # =========================================================================

    def _build_ui(self):
        """Build the tab UI."""
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - Controls
        self.control_frame = ttk.Frame(self.paned, width=700)
        self.control_frame.pack_propagate(False)
        self.paned.add(self.control_frame, weight=0)

        # Right panel - Plot
        self.plot_frame = ttk.Frame(self.paned)
        self.paned.add(self.plot_frame, weight=1)

        self._build_control_panel()
        self._build_plot_area()

    def _build_control_panel(self):
        """Build the left control panel with scrolling."""
        canvas = tk.Canvas(self.control_frame, width=700)
        scrollbar = ttk.Scrollbar(self.control_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Sections in new order
        self._build_file_section()
        self._build_lma_section()
        self._build_timing_section()
        self._build_cosshift_section()
        self._build_filter_section()
        self._build_tasd_section()
        self._build_plot_section()
        self._build_action_buttons()
        self._build_info_section()

    def _build_file_section(self):
        """Build file input section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="INTF Data File", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # File path
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Entry(row, textvariable=self.file_path_var, width=30).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3, command=self._browse_file).pack(side=tk.LEFT)

        # Calibration toggle + header lines on same row
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(row, text="Pre-calibrated",
                        variable=self.is_calibrated_var,
                        command=self._on_calibration_toggle).pack(side=tk.LEFT)
        ttk.Label(row, text="  Header lines:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.skip_header_var, width=4).pack(side=tk.LEFT, padx=2)

        # Load button
        ttk.Button(frame, text="Load INTF Data", command=self._load_data).pack(pady=4)

    def _build_lma_section(self):
        """Build LMA auto-calibration section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="LMA Auto-Calibration", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # LMA file picker
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="LMA File:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_file_path_var, width=22).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3,
                   command=self._browse_lma_file).pack(side=tk.LEFT)

        # Buttons row
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=3)
        ttk.Button(btn_row, text="Advanced Options",
                   command=self._open_lma_advanced).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Run LMA Calibration",
                   command=self._run_lma_calibration).pack(side=tk.LEFT, padx=2)

        # Results text box
        self.lma_result_text = tk.Text(frame, height=8, width=30, wrap=tk.WORD,
                                       font=('Courier', 9))
        self.lma_result_text.pack(fill=tk.X, pady=2)
        self.lma_result_text.insert('1.0', "No calibration run yet")
        self.lma_result_text.config(state='disabled')

    def _build_timing_section(self):
        """Build timing parameters section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Timing", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Event Time
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Event Time:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.event_time_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="(HH:MM:SS)", foreground='gray').pack(side=tk.LEFT, padx=2)

        # T0 Reference
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="T0 (us):").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.T0_var, width=10).pack(side=tk.LEFT, padx=2)

        ttk.Button(frame, text="Sync T0 from Home Tab",
                   command=self._sync_from_home).pack(pady=2)

    def _build_cosshift_section(self):
        """Build cosine shift calibration section."""
        self.cal_frame = ttk.LabelFrame(self.scrollable_frame, text="Cosine Shift Calibration", padding=5)
        self.cal_frame.pack(fill=tk.X, padx=5, pady=5)

        # Cos Shift A
        row = ttk.Frame(self.cal_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Cos Shift A:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.cos_shift_a_var, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="-", width=2,
                   command=lambda: self._adjust_cos_shift('a', -1)).pack(side=tk.LEFT)
        ttk.Button(row, text="+", width=2,
                   command=lambda: self._adjust_cos_shift('a', +1)).pack(side=tk.LEFT)

        # Cos Shift B
        row = ttk.Frame(self.cal_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Cos Shift B:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.cos_shift_b_var, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="-", width=2,
                   command=lambda: self._adjust_cos_shift('b', -1)).pack(side=tk.LEFT)
        ttk.Button(row, text="+", width=2,
                   command=lambda: self._adjust_cos_shift('b', +1)).pack(side=tk.LEFT)

        # Apply / Reset
        btn_row = ttk.Frame(self.cal_frame)
        btn_row.pack(fill=tk.X, pady=3)
        ttk.Button(btn_row, text="Apply Calibration",
                   command=self._apply_calibration).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Reset to Defaults",
                   command=self._reset_calibration).pack(side=tk.LEFT, padx=2)

    def _build_filter_section(self):
        """Build data filter section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Data Filters", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Time range
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Time (us):", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.time_start_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.time_stop_var, width=10).pack(side=tk.LEFT, padx=2)

        # Azimuth range
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Azimuth:", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.azi_min_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.azi_max_var, width=10).pack(side=tk.LEFT, padx=2)

        # Elevation range
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Elevation:", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.elv_min_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.elv_max_var, width=10).pack(side=tk.LEFT, padx=2)

        ttk.Button(frame, text="Clear Time Filter",
                   command=self._clear_time_filter).pack(pady=2)

    def _build_tasd_section(self):
        """Build TASD overlay section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="TASD Overlay", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # TASD directory
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="TASD Dir:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.tasd_dir_var, width=20).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3,
                   command=self._browse_tasd_dir).pack(side=tk.LEFT)

        # Buttons row
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=3)
        ttk.Button(btn_row, text="Load TASD Data",
                   command=self._load_tasd_for_intf).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Advanced Options",
                   command=self._open_tasd_advanced).pack(side=tk.LEFT, padx=2)

        ttk.Checkbutton(frame, text="Show TASD overlay on Az-El plot",
                        variable=self.show_tasd_var).pack(anchor='w', pady=2)

    def _build_plot_section(self):
        """Build plot options section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Plot Options", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Plot title
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Plot Title:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.plot_title_var, width=22).pack(side=tk.LEFT, padx=2)

        ttk.Button(frame, text="Advanced Plot Options",
                   command=self._open_plot_advanced).pack(pady=3)

    def _build_action_buttons(self):
        """Build action buttons."""
        frame = ttk.Frame(self.scrollable_frame, padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(frame, text="Update Plot",
                   command=self._update_plot).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Export PNG",
                   command=self._export_png).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Export Calibrated Data",
                   command=self._export_calibrated).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Save to Project",
                   command=self._save_to_project).pack(fill=tk.X, pady=2)

    def _build_info_section(self):
        """Build data info display section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Data Info", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        self.info_text = tk.Text(frame, height=8, width=35, wrap=tk.WORD,
                                 font=('Courier', 9))
        self.info_text.pack(fill=tk.X)
        self.info_text.insert('1.0', "No data loaded")
        self.info_text.config(state='disabled')

    def _build_plot_area(self):
        """Build the matplotlib plot area."""
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()

        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.coord_var = tk.StringVar(value="")
        coord_label = ttk.Label(self.plot_frame, textvariable=self.coord_var)
        coord_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    # =========================================================================
    # Popup Dialogs
    # =========================================================================

    def _open_lma_advanced(self):
        """Open LMA advanced options popup."""
        # Auto-sync trigger offset from T0
        try:
            t0_us = float(self.T0_var.get())
            self.lma_trigger_offset_var.set(f"{t0_us / 1000.0:.3f}")
        except ValueError:
            pass

        win = tk.Toplevel(self.winfo_toplevel())
        win.title("LMA Calibration - Advanced Options")
        win.geometry("380x280")
        win.transient(self.winfo_toplevel())
        win.grab_set()

        pad = ttk.Frame(win, padding=10)
        pad.pack(fill=tk.BOTH, expand=True)

        # Trigger offset (pre-filled from T0)
        row = ttk.Frame(pad)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Trigger offset (ms):", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_trigger_offset_var, width=10).pack(side=tk.LEFT)

        # Match window
        row = ttk.Frame(pad)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Match window (us):", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_match_window_var, width=10).pack(side=tk.LEFT)

        # Chi2 max
        row = ttk.Frame(pad)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Chi-squared max:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_chi2_max_var, width=10).pack(side=tk.LEFT)

        # Altitude range
        row = ttk.Frame(pad)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Alt range (m):", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_alt_min_var, width=7).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_alt_max_var, width=7).pack(side=tk.LEFT, padx=2)

        # Power min
        row = ttk.Frame(pad)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Power min (dBW):", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_power_min_var, width=10).pack(side=tk.LEFT)

        ttk.Separator(pad, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Button(pad, text="Close", command=win.destroy).pack(pady=4)

    def _open_tasd_advanced(self):
        """Open TASD advanced options popup (source location for shower axis)."""
        # Auto-pull from timeshift results if available
        results = self.main_app.project_state.results
        if results.get('source_azi') is not None and not self.source_azi_var.get():
            self.source_azi_var.set(f"{float(results['source_azi']):.2f}")
        if results.get('source_elv') is not None and not self.source_elv_var.get():
            self.source_elv_var.set(f"{float(results['source_elv']):.2f}")

        win = tk.Toplevel(self.winfo_toplevel())
        win.title("TASD Overlay - Advanced Options")
        win.geometry("380x250")
        win.transient(self.winfo_toplevel())
        win.grab_set()

        pad = ttk.Frame(win, padding=10)
        pad.pack(fill=tk.BOTH, expand=True)

        ttk.Label(pad, text="Source Location (for shower axis):",
                  font=('Helvetica', 9, 'bold')).pack(anchor='w', pady=(0, 4))

        row = ttk.Frame(pad)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Azimuth:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.source_azi_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="  Elevation:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.source_elv_var, width=8).pack(side=tk.LEFT, padx=2)

        row2 = ttk.Frame(pad)
        row2.pack(fill=tk.X, pady=3)
        ttk.Label(row2, text="Horiz. distance (km):").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.source_distance_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row2, text="(for Height axis)").pack(side=tk.LEFT)

        ttk.Button(pad, text="Sync from Timeshift",
                   command=self._sync_source_from_project).pack(pady=4)

        ttk.Separator(pad, orient='horizontal').pack(fill=tk.X, pady=4)
        ttk.Button(pad, text="Close", command=win.destroy).pack(pady=4)

    def _open_plot_advanced(self):
        """Open advanced plot options popup."""
        win = tk.Toplevel(self.winfo_toplevel())
        win.title("Plot - Advanced Options")
        win.geometry("350x280")
        win.transient(self.winfo_toplevel())
        win.grab_set()

        pad = ttk.Frame(win, padding=10)
        pad.pack(fill=tk.BOTH, expand=True)

        # Plot mode
        ttk.Label(pad, text="Plot Mode:").pack(anchor='w')
        modes_frame = ttk.Frame(pad)
        modes_frame.pack(fill=tk.X, pady=3)
        ttk.Radiobutton(modes_frame, text="Az vs El", variable=self.plot_mode_var,
                        value="az_el").pack(side=tk.LEFT)
        ttk.Radiobutton(modes_frame, text="Time vs El", variable=self.plot_mode_var,
                        value="time_el").pack(side=tk.LEFT)
        ttk.Radiobutton(modes_frame, text="Time vs Az", variable=self.plot_mode_var,
                        value="time_az").pack(side=tk.LEFT)

        # Color by
        row = ttk.Frame(pad)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Color by:").pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="Time", variable=self.color_by_var,
                        value="time").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(row, text="Signal", variable=self.color_by_var,
                        value="signal").pack(side=tk.LEFT)

        # Marker size
        row = ttk.Frame(pad)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Marker size:").pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="Auto", variable=self.marker_size_var,
                        value="auto").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(row, text="Fixed:", variable=self.marker_size_var,
                        value="fixed").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.fixed_marker_size_var, width=5).pack(side=tk.LEFT, padx=2)

        # Colorbar
        ttk.Checkbutton(pad, text="Show Colorbar",
                        variable=self.show_colorbar_var).pack(anchor='w', pady=3)

        # Gridlines
        ttk.Checkbutton(pad, text="Show Gridlines",
                        variable=self.show_grid_var).pack(anchor='w', pady=3)

        # Source distance for height axis
        row = ttk.Frame(pad)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Source distance (km):").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.source_distance_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="(for Height axis)").pack(side=tk.LEFT)

        ttk.Separator(pad, orient='horizontal').pack(fill=tk.X, pady=4)
        ttk.Button(pad, text="Close", command=win.destroy).pack(pady=4)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _find_tasd_coords(self):
        """Auto-find TASD coordinates file by searching common locations."""
        possible_paths = []

        # Next to TASD data directory
        tasd_dir = self.tasd_dir_var.get()
        if tasd_dir:
            possible_paths.append(os.path.join(tasd_dir, TASD_COORDS_FILENAME))
            possible_paths.append(os.path.join(os.path.dirname(tasd_dir), TASD_COORDS_FILENAME))

        # Current working directory
        possible_paths.append(TASD_COORDS_FILENAME)
        possible_paths.append(os.path.join(os.getcwd(), TASD_COORDS_FILENAME))

        # Relative to this script file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for up in ['', '..', os.path.join('..', '..'), os.path.join('..', '..', '..')]:
            possible_paths.append(os.path.join(script_dir, up, TASD_COORDS_FILENAME))

        # If in gui/tabs/ structure, go up to main directory
        if 'gui' in script_dir:
            gui_parent = script_dir.split('gui')[0]
            possible_paths.append(os.path.join(gui_parent, TASD_COORDS_FILENAME))

        # Relative to project file
        try:
            if hasattr(self.main_app, 'project_filepath') and self.main_app.project_filepath:
                project_dir = os.path.dirname(self.main_app.project_filepath)
                possible_paths.append(os.path.join(project_dir, TASD_COORDS_FILENAME))
        except Exception:
            pass

        for p in possible_paths:
            if os.path.isfile(p):
                return os.path.abspath(p)

        return None

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _browse_file(self):
        """Browse for INTF data file."""
        filepath = filedialog.askopenfilename(
            title="Select INTF Data File",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if filepath:
            self.file_path_var.set(filepath)

    def _on_calibration_toggle(self):
        """Handle calibration checkbox toggle."""
        if self.is_calibrated_var.get():
            self.skip_header_var.set("2")
        else:
            self.skip_header_var.set("57")

    def _load_data(self):
        """Load INTF data from file."""
        filepath = self.file_path_var.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Error", "Please select a valid INTF file")
            return

        try:
            is_calibrated = self.is_calibrated_var.get()
            skip_header = int(self.skip_header_var.get())
            T0 = float(self.T0_var.get())
            time_is_relative = self.time_is_relative_var.get()

            if is_calibrated:
                success = self.intf_handler.load_data(
                    filepath,
                    is_calibrated=True,
                    skip_header=skip_header,
                    T0_reference=T0,
                    time_is_relative=time_is_relative
                )
            else:
                cos_a = float(self.cos_shift_a_var.get())
                cos_b = float(self.cos_shift_b_var.get())
                self.intf_handler.cos_shift_a = cos_a
                self.intf_handler.cos_shift_b = cos_b

                success = self.intf_handler.load_raw_intf(
                    filepath,
                    skip_header=skip_header,
                    T0_reference=T0
                )

            if success:
                self._update_info()
                self._auto_set_filters()
                self._update_plot()
                self.main_app.status_var.set(f"Loaded INTF: {os.path.basename(filepath)}")
            else:
                messagebox.showerror("Error", "Failed to load INTF data")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading INTF data:\n{e}")

    def _apply_calibration(self):
        """Apply cosine shift calibration."""
        if not self.intf_handler.is_loaded:
            messagebox.showwarning("Warning", "No data loaded")
            return

        try:
            cos_a = float(self.cos_shift_a_var.get())
            cos_b = float(self.cos_shift_b_var.get())

            self.intf_handler.set_cos_shifts(cos_a, cos_b)
            self._update_info()
            self._update_plot()

            self.main_app.status_var.set(f"Applied cos shifts: A={cos_a:.4f}, B={cos_b:.4f}")
            self.main_app.mark_unsaved()

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid cos shift values:\n{e}")

    def _adjust_cos_shift(self, which, direction):
        """Adjust cos shift by fixed step (0.001)."""
        step = 0.001
        try:
            if which == 'a':
                current = float(self.cos_shift_a_var.get())
                self.cos_shift_a_var.set(f"{current + direction * step:.6f}")
            else:
                current = float(self.cos_shift_b_var.get())
                self.cos_shift_b_var.set(f"{current + direction * step:.6f}")

            if self.intf_handler.is_loaded:
                self._apply_calibration()

        except ValueError:
            pass

    def _reset_calibration(self):
        """Reset cos shifts to defaults."""
        self.cos_shift_a_var.set(str(DEFAULT_COS_SHIFT_A))
        self.cos_shift_b_var.set(str(DEFAULT_COS_SHIFT_B))

        if self.intf_handler.is_loaded:
            self._apply_calibration()

    def _sync_from_home(self):
        """Sync T0 from Home tab. Also updates LMA trigger offset."""
        try:
            home_T0 = self.main_app.home_tab.T0_var.get()
            self.T0_var.set(home_T0)

            # Auto-sync trigger offset (T0 in us -> ms)
            t0_us = float(home_T0)
            self.lma_trigger_offset_var.set(f"{t0_us / 1000.0:.3f}")

            if self.intf_handler.is_loaded:
                self.intf_handler.set_T0(float(home_T0))
                self._update_plot()

            self.main_app.status_var.set(f"Synced T0: {home_T0}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to sync T0:\n{e}")

    def _browse_lma_file(self):
        """Browse for LMA data file."""
        filepath = filedialog.askopenfilename(
            title="Select LMA Data File",
            filetypes=[("Text files", "*.txt *.dat"), ("All files", "*.*")]
        )
        if filepath:
            self.lma_file_path_var.set(filepath)

    def _browse_tasd_dir(self):
        """Browse for TASD data directory."""
        directory = filedialog.askdirectory(title="Select TASD Data Directory")
        if directory:
            self.tasd_dir_var.set(directory)

    def _run_lma_calibration(self):
        """Run LMA auto-calibration and populate cos shift fields."""
        if not self.intf_handler.is_loaded:
            messagebox.showerror("Error", "Load a raw INTF file first")
            return

        lma_file = self.lma_file_path_var.get()
        if not lma_file or not os.path.exists(lma_file):
            messagebox.showerror("Error", "Please select a valid LMA file")
            return

        # Parse event time HH:MM:SS -> seconds-of-day
        event_time_str = self.event_time_var.get().strip()
        if not event_time_str or event_time_str == "HH:MM:SS":
            messagebox.showerror("Error", "Please set Event Time (HH:MM:SS) in Timing section")
            return
        try:
            parts = event_time_str.split(":")
            if len(parts) != 3:
                raise ValueError
            hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
            event_sod = hh * 3600 + mm * 60 + ss
        except ValueError:
            messagebox.showerror("Error", "Event time must be in HH:MM:SS format")
            return

        # Get trigger offset - auto-sync from T0 if not yet set by user
        try:
            trigger_offset_s = float(self.lma_trigger_offset_var.get()) / 1000.0
        except ValueError:
            messagebox.showerror("Error", "Invalid trigger offset value")
            return

        intf_t0_sod = event_sod + trigger_offset_s

        # Gather filter parameters
        try:
            params = {
                'match_window_us': float(self.lma_match_window_var.get()),
                'chi2_max': float(self.lma_chi2_max_var.get()),
                'alt_min_m': float(self.lma_alt_min_var.get()),
                'alt_max_m': float(self.lma_alt_max_var.get()),
                'power_min_dbw': float(self.lma_power_min_var.get()),
            }
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid LMA filter parameter:\n{e}")
            return

        self.main_app.status_var.set("Running LMA calibration...")
        self.update_idletasks()

        result = self.intf_handler.run_lma_calibration(lma_file, intf_t0_sod, **params)

        if result is None:
            self.lma_result_text.config(state='normal')
            self.lma_result_text.delete('1.0', tk.END)
            self.lma_result_text.insert('1.0', "Calibration failed.\nCheck LMA file format,\nevent time, and filter parameters.")
            self.lma_result_text.config(state='disabled')
            self.main_app.status_var.set("LMA calibration failed")
            return

        # Handle diagnostic-only return (no matches found)
        if 'error' in result:
            self.lma_result_text.config(state='normal')
            self.lma_result_text.delete('1.0', tk.END)
            diag = (
                f"No matches found!\n"
                f"intf_t0_sod = {intf_t0_sod:.6f}\n"
                f"INTF SOD: [{result['intf_sod_min']:.6f}, "
                f"{result['intf_sod_max']:.6f}]\n"
                f"LMA SOD:  [{result['lma_sod_min']:.6f}, "
                f"{result['lma_sod_max']:.6f}]\n"
                f"INTF pts: {result['n_intf']}, "
                f"LMA pts: {result['n_lma_filtered']}\n"
                f"Check event time & trigger offset."
            )
            self.lma_result_text.insert('1.0', diag)
            self.lma_result_text.config(state='disabled')
            self.main_app.status_var.set("LMA calibration: no matches")
            return

        # Populate cos shift fields and apply
        self.cos_shift_a_var.set(f"{result['cos_shift_a']:.6f}")
        self.cos_shift_b_var.set(f"{result['cos_shift_b']:.6f}")
        self._apply_calibration()

        # Display result summary with diagnostics
        summary = (
            f"cosShiftA = {result['cos_shift_a']:+.5f} +/- {result['se_a']:.5f}\n"
            f"cosShiftB = {result['cos_shift_b']:+.5f} +/- {result['se_b']:.5f}\n"
            f"n_matched = {result['n_matched']}  n_final = {result['n_final']}\n"
            f"Formal err = +/-{result['formal_error_deg']:.4f} deg\n"
            f"---\n"
            f"intf_t0_sod = {result['intf_t0_sod']:.6f}\n"
            f"INTF SOD: [{result['intf_sod_min']:.6f}, "
            f"{result['intf_sod_max']:.6f}]\n"
            f"LMA SOD:  [{result['lma_sod_min']:.6f}, "
            f"{result['lma_sod_max']:.6f}]"
        )
        self.lma_result_text.config(state='normal')
        self.lma_result_text.delete('1.0', tk.END)
        self.lma_result_text.insert('1.0', summary)
        self.lma_result_text.config(state='disabled')

        self.main_app.mark_unsaved()
        self.main_app.status_var.set(
            f"LMA calibration done: A={result['cos_shift_a']:+.5f}, B={result['cos_shift_b']:+.5f}, "
            f"err=+/-{result['formal_error_deg']:.4f} deg"
        )

    def _load_tasd_for_intf(self):
        """Load TASD data for the Az-El overlay."""
        tasd_dir = self.tasd_dir_var.get()
        if not tasd_dir or not os.path.isdir(tasd_dir):
            messagebox.showerror("Error", "Please select a valid TASD directory")
            return

        # Auto-find coordinates file
        coords_path = self._find_tasd_coords()
        if not coords_path:
            messagebox.showerror("Error",
                                 f"Could not find {TASD_COORDS_FILENAME}.\n"
                                 "Searched near TASD directory, working directory, and script directory.")
            return

        # Derive time string (HHMMSS) from event time field
        event_time_str = self.event_time_var.get().strip()
        if event_time_str and event_time_str != "HH:MM:SS" and ":" in event_time_str:
            time_str = event_time_str.replace(":", "")
        else:
            ei = self.main_app.project_state.event_info
            time_str = ei.get('time', '').replace(":", "") if ei else ""

        self.tasd_handler = TASDHandler()
        self.tasd_handler.load_coordinates(coords_path)
        success = self.tasd_handler.load_directory(tasd_dir, time_str)

        if success:
            n = len(self.tasd_handler.detectors)
            self.main_app.status_var.set(f"Loaded TASD: {n} detector(s)")
        else:
            messagebox.showwarning("Warning",
                                   "No TASD waveform files matched the time string.\n"
                                   "Coordinates loaded; check event time or use .stitch files.")
            self.main_app.status_var.set("TASD: no waveforms matched")

        # Auto-read source azi/elv from Analysis files in the TASD directory
        # (matches the standalone azi-elv-plotter.py pattern)
        if not self.source_azi_var.get() or not self.source_elv_var.get():
            self._try_read_analysis_source(tasd_dir)

    def _try_read_analysis_source(self, directory):
        """Try to read source azi/elv from Analysis files in the TASD directory.

        The standalone scripts write Analysis files with lines like:
            elv(deg)=5.8 ...
            azi(deg)=256.7 ...
        """
        for filename in os.listdir(directory):
            if filename.startswith('Analysis'):
                filepath = os.path.join(directory, filename)
                try:
                    src_azi = None
                    src_elv = None
                    with open(filepath, 'r') as f:
                        for line in f:
                            if line.strip().startswith('elv(deg)'):
                                src_elv = float(line.split('=')[1].split()[0])
                            elif line.strip().startswith('azi(deg)'):
                                src_azi = float(line.split('=')[1].split()[0])
                    if src_azi is not None and src_elv is not None:
                        self.source_azi_var.set(f"{src_azi:.2f}")
                        self.source_elv_var.set(f"{src_elv:.2f}")
                        self.main_app.status_var.set(
                            f"Source from Analysis: Az={src_azi:.2f}, El={src_elv:.2f}")
                        return
                except Exception:
                    pass

    def _sync_source_from_project(self):
        """Populate source azi/elv from Timeshift results in ProjectState."""
        results = self.main_app.project_state.results
        azi = results.get('source_azi')
        elv = results.get('source_elv')
        if azi is not None and elv is not None:
            self.source_azi_var.set(f"{float(azi):.2f}")
            self.source_elv_var.set(f"{float(elv):.2f}")
            self.main_app.status_var.set(f"Source synced: Az={azi:.2f}, El={elv:.2f}")
        else:
            messagebox.showinfo("Info",
                                "No source location found in project results.\n"
                                "Run Timeshift analysis first, or enter values manually.")

    def _clear_time_filter(self):
        """Clear time filter."""
        self.time_start_var.set("")
        self.time_stop_var.set("")

    def _auto_set_filters(self, force=False):
        """Auto-set filter ranges from loaded data.

        Only overwrites if filters are still at defaults (270/330/0/40)
        or if force=True. This preserves user-customized or project-loaded values.
        """
        if not self.intf_handler.is_loaded:
            return

        data = self.intf_handler.get_full_data()
        if data is None:
            return

        # Check if filters are at their initial defaults
        at_defaults = (
            self.azi_min_var.get() == "270" and
            self.azi_max_var.get() == "330" and
            self.elv_min_var.get() == "0" and
            self.elv_max_var.get() == "40"
        )

        if at_defaults or force:
            az_min, az_max = np.min(data['azimuth']), np.max(data['azimuth'])
            el_min, el_max = np.min(data['elevation']), np.max(data['elevation'])

            self.azi_min_var.set(f"{int(az_min - 5)}")
            self.azi_max_var.set(f"{int(az_max + 5)}")
            self.elv_min_var.set(f"{max(0, int(el_min - 2))}")
            self.elv_max_var.set(f"{int(el_max + 5)}")

    def _get_filter_params(self):
        """Get current filter parameters."""
        params = {}

        if self.time_start_var.get():
            try:
                params['t_min'] = float(self.time_start_var.get())
            except ValueError:
                pass
        if self.time_stop_var.get():
            try:
                params['t_max'] = float(self.time_stop_var.get())
            except ValueError:
                pass

        try:
            params['azi_min'] = float(self.azi_min_var.get())
        except ValueError:
            pass
        try:
            params['azi_max'] = float(self.azi_max_var.get())
        except ValueError:
            pass

        try:
            params['elv_min'] = float(self.elv_min_var.get())
        except ValueError:
            pass
        try:
            params['elv_max'] = float(self.elv_max_var.get())
        except ValueError:
            pass

        return params

    def _update_info(self):
        """Update data info display."""
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)

        if self.intf_handler.is_loaded:
            info = self.intf_handler.get_info()
            self.info_text.insert('1.0', info)
        else:
            self.info_text.insert('1.0', "No data loaded")

        self.info_text.config(state='disabled')

    # =========================================================================
    # Plotting
    # =========================================================================

    def _update_plot(self):
        """Update the plot with current data and settings."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if not self.intf_handler.is_loaded:
            self.ax.text(0.5, 0.5, "No data loaded", ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        # Get filtered data
        filter_params = self._get_filter_params()
        data = self.intf_handler.filter_data(**filter_params)

        if data is None or len(data['time']) == 0:
            self.ax.text(0.5, 0.5, "No data in selected range", ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        plot_mode = self.plot_mode_var.get()
        color_by = self.color_by_var.get()

        # Get colors
        if color_by == "time":
            colors = data['time']
            cmap = 'jet'
            event_time = self.event_time_var.get()
            if event_time and event_time != "HH:MM:SS":
                clabel = f"Time relative to {event_time} (us)"
            else:
                clabel = "Time (us)"
        else:
            colors = data['colors'] if data['colors'] is not None else 'blue'
            cmap = None
            clabel = "Signal Strength"

        # Get marker sizes
        if self.marker_size_var.get() == "auto" and data['marker_sizes'] is not None:
            sizes = data['marker_sizes']
        else:
            try:
                sizes = float(self.fixed_marker_size_var.get())
            except ValueError:
                sizes = 10

        # For Az-El mode, clip INTF points below elevation 0 for cleanliness
        plot_azi = data['azimuth']
        plot_elv = data['elevation']
        plot_time = data['time']
        if plot_mode == "az_el":
            elv_clip_mask = plot_elv >= 0
            plot_azi = plot_azi[elv_clip_mask]
            plot_elv = plot_elv[elv_clip_mask]
            plot_time = plot_time[elv_clip_mask]
            if isinstance(colors, np.ndarray):
                colors = colors[elv_clip_mask]
            if isinstance(sizes, np.ndarray):
                sizes = sizes[elv_clip_mask]

        sc = None  # Will hold the scatter for colorbar

        # Create scatter plot based on mode
        if plot_mode == "az_el":
            if isinstance(colors, np.ndarray) and cmap:
                sc = self.ax.scatter(plot_azi, plot_elv,
                                     c=colors, cmap=cmap, s=sizes, alpha=0.7)
            else:
                sc = self.ax.scatter(plot_azi, plot_elv,
                                     c=colors, s=sizes, alpha=0.7)
            self.ax.set_xlabel("Azimuth (deg)", fontsize=12)
            self.ax.set_ylabel("Elevation (deg)", fontsize=12)
            title = self.plot_title_var.get() or "INTF: Azimuth vs Elevation"
            self.ax.set_title(title, fontsize=14)

            try:
                self.ax.set_xlim(float(self.azi_min_var.get()) - 1,
                                 float(self.azi_max_var.get()) + 1)
            except ValueError:
                pass
            try:
                elv_lo = float(self.elv_min_var.get())
                elv_hi = float(self.elv_max_var.get())
                # Extend below 0 slightly so detector rings are fully visible
                self.ax.set_ylim(min(elv_lo, 0) - 1, elv_hi + 1)
            except ValueError:
                pass

        elif plot_mode == "time_el":
            if isinstance(colors, np.ndarray) and cmap:
                sc = self.ax.scatter(data['time'], data['elevation'],
                                     c=colors, cmap=cmap, s=sizes, alpha=0.7)
            else:
                sc = self.ax.scatter(data['time'], data['elevation'],
                                     c=colors, s=sizes, alpha=0.7)
            self.ax.set_xlabel("Time (us)", fontsize=12)
            self.ax.set_ylabel("Elevation (deg)", fontsize=12)
            self.ax.set_title("INTF: Time vs Elevation", fontsize=14)

            try:
                self.ax.set_ylim(float(self.elv_min_var.get()), float(self.elv_max_var.get()))
            except ValueError:
                pass

        elif plot_mode == "time_az":
            if isinstance(colors, np.ndarray) and cmap:
                sc = self.ax.scatter(data['time'], data['azimuth'],
                                     c=colors, cmap=cmap, s=sizes, alpha=0.7)
            else:
                sc = self.ax.scatter(data['time'], data['azimuth'],
                                     c=colors, s=sizes, alpha=0.7)
            self.ax.set_xlabel("Time (us)", fontsize=12)
            self.ax.set_ylabel("Azimuth (deg)", fontsize=12)
            self.ax.set_title("INTF: Time vs Azimuth", fontsize=14)

            try:
                self.ax.set_ylim(float(self.azi_min_var.get()), float(self.azi_max_var.get()))
            except ValueError:
                pass

        # TASD overlay (Az-El mode only)
        if plot_mode == "az_el" and self.show_tasd_var.get() and self.tasd_handler:
            # Auto-sync source from project if not already set
            if not self.source_azi_var.get() or not self.source_elv_var.get():
                results = self.main_app.project_state.results
                if results.get('source_azi') is not None:
                    self.source_azi_var.set(f"{float(results['source_azi']):.2f}")
                if results.get('source_elv') is not None:
                    self.source_elv_var.set(f"{float(results['source_elv']):.2f}")

            triggered = self.tasd_handler.get_triggered_detectors()
            if triggered:
                det_azis = []
                vems = []
                for det in triggered:
                    try:
                        det_azi = calculate_azimuth_to_detector(det['lat'], det['lon'])
                        det_azis.append(det_azi)
                        vem = det['vem']
                        vems.append(vem)
                        # Size matches standalone: 150 * log(energy)
                        size = max(50, 150 * np.log(max(vem, 1.0)))
                        self.ax.scatter(det_azi, 0, s=size,
                                        facecolors='none', edgecolors='red',
                                        linewidths=1.5, zorder=4)
                    except Exception:
                        pass

                # VEM-weighted core location (filled red with black edge)
                core_azi = None
                if det_azis and vems:
                    vems_arr = np.array(vems)
                    core_azi = float(np.average(det_azis, weights=vems_arr))
                    self.ax.scatter(core_azi, 0, marker='o', s=250,
                                    facecolors='red', edgecolors='k',
                                    linewidths=1.5, zorder=5, label='TGF shower core')

                # Shower axis from source location
                try:
                    src_azi = float(self.source_azi_var.get())
                    src_elv = float(self.source_elv_var.get())
                    if det_azis and core_azi is not None:
                        self.ax.plot([src_azi, core_azi], [src_elv, 0],
                                     'r-', lw=1.5, label='TGF shower axis', zorder=5)
                        self.ax.plot([src_azi, min(det_azis)], [src_elv, 0],
                                     'r--', lw=1, alpha=0.6, zorder=5)
                        self.ax.plot([src_azi, max(det_azis)], [src_elv, 0],
                                     'r--', lw=1, alpha=0.6, zorder=5)
                    self.ax.legend(fontsize=9, loc='upper right')
                except (ValueError, TypeError):
                    # No source location — still show detectors and core
                    if det_azis:
                        self.ax.legend(fontsize=9, loc='upper right')

        # Secondary y-axis for Height (km) in Az-El mode
        if plot_mode == "az_el":
            try:
                dist_km = float(self.source_distance_var.get())
                if dist_km > 0:
                    def elv_to_height(x):
                        return dist_km * np.tan(np.radians(x))
                    def height_to_elv(x):
                        return np.degrees(np.arctan(x / dist_km))
                    sec_ax = self.ax.secondary_yaxis('right',
                                                      functions=(elv_to_height, height_to_elv))
                    sec_ax.set_ylabel('Height (km)', fontsize=12)
            except (ValueError, TypeError):
                pass

        # Add colorbar
        if self.show_colorbar_var.get() and sc is not None and isinstance(colors, np.ndarray) and cmap:
            cbar = self.fig.colorbar(sc, ax=self.ax, pad=0.12)
            cbar.set_label(clabel, fontsize=10)

        if self.show_grid_var.get():
            self.ax.grid(True, linestyle=':', alpha=0.5)

        self.fig.tight_layout()
        self.canvas.draw()

        self.main_app.status_var.set(f"Plotting {len(data['time']):,} INTF points")

    def _on_mouse_move(self, event):
        """Handle mouse movement for coordinate display."""
        if event.inaxes:
            self.coord_var.set(f"x={event.xdata:.2f}, y={event.ydata:.2f}")
        else:
            self.coord_var.set("")

    # =========================================================================
    # Export
    # =========================================================================

    def _export_png(self):
        """Export plot as PNG."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")]
        )
        if filepath:
            try:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
                self.main_app.status_var.set(f"Exported: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def _export_calibrated(self):
        """Export calibrated INTF data to file."""
        if not self.intf_handler.is_loaded:
            messagebox.showwarning("Warning", "No data loaded")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".dat",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")],
            initialfile="INTF_calibrated.dat"
        )
        if filepath:
            try:
                time_range = None
                if self.time_start_var.get() and self.time_stop_var.get():
                    time_range = (float(self.time_start_var.get()),
                                  float(self.time_stop_var.get()))

                success = self.intf_handler.calibrate_and_save(filepath, time_range)

                if success:
                    self.main_app.status_var.set(f"Exported: {os.path.basename(filepath)}")
                    messagebox.showinfo("Success", f"Calibrated data saved to:\n{filepath}")
                else:
                    messagebox.showerror("Error", "Failed to export data")

            except Exception as e:
                messagebox.showerror("Error", f"Export error:\n{e}")

    # =========================================================================
    # Project Save / Load
    # =========================================================================

    def _save_to_project(self):
        """Save current INTF settings to project state."""
        state = self.main_app.project_state

        # Save file path
        state.files['intf_raw'] = self.file_path_var.get() or None

        # Save calibration settings
        try:
            state.intf['cos_shift_a'] = float(self.cos_shift_a_var.get())
        except ValueError:
            pass
        try:
            state.intf['cos_shift_b'] = float(self.cos_shift_b_var.get())
        except ValueError:
            pass

        state.intf['is_calibrated'] = self.is_calibrated_var.get()

        # Save filter ranges (individual try/except so one bad value doesn't block others)
        for key, var in [('azimuth_min', self.azi_min_var),
                         ('azimuth_max', self.azi_max_var),
                         ('elevation_min', self.elv_min_var),
                         ('elevation_max', self.elv_max_var)]:
            try:
                state.intf[key] = float(var.get())
            except ValueError:
                pass

        if self.time_start_var.get():
            try:
                state.intf['time_min'] = float(self.time_start_var.get())
            except ValueError:
                pass
        if self.time_stop_var.get():
            try:
                state.intf['time_max'] = float(self.time_stop_var.get())
            except ValueError:
                pass

        # Save event time
        event_time = self.event_time_var.get()
        if event_time and event_time != "HH:MM:SS":
            state.intf['event_time'] = event_time

        # LMA calibration parameters
        state.files['lma'] = self.lma_file_path_var.get() or None
        for key, var in [('trigger_offset_ms', self.lma_trigger_offset_var),
                         ('lma_match_window_us', self.lma_match_window_var),
                         ('lma_chi2_max', self.lma_chi2_max_var),
                         ('lma_alt_min_m', self.lma_alt_min_var),
                         ('lma_alt_max_m', self.lma_alt_max_var),
                         ('lma_power_min', self.lma_power_min_var)]:
            try:
                state.intf[key] = float(var.get())
            except ValueError:
                pass

        # Plot settings
        state.intf['show_grid'] = self.show_grid_var.get()
        try:
            if self.source_distance_var.get():
                state.intf['source_distance_km'] = float(self.source_distance_var.get())
        except ValueError:
            pass

        # TASD overlay settings
        state.intf['show_tasd_overlay'] = self.show_tasd_var.get()
        if self.tasd_dir_var.get():
            state.files['sd_directory'] = self.tasd_dir_var.get()

        # Source azi/elv
        try:
            if self.source_azi_var.get():
                state.results['source_azi'] = float(self.source_azi_var.get())
        except ValueError:
            pass
        try:
            if self.source_elv_var.get():
                state.results['source_elv'] = float(self.source_elv_var.get())
        except ValueError:
            pass

        self.main_app.mark_unsaved()
        self.main_app.status_var.set("INTF settings saved to project")

    def load_from_project(self):
        """Load settings from project state."""
        state = self.main_app.project_state

        # Load file path
        if state.files.get('intf_raw'):
            self.file_path_var.set(state.files['intf_raw'])

        # Load calibration settings
        self.is_calibrated_var.set(state.intf.get('is_calibrated', False))
        self.cos_shift_a_var.set(str(state.intf.get('cos_shift_a', DEFAULT_COS_SHIFT_A)))
        self.cos_shift_b_var.set(str(state.intf.get('cos_shift_b', DEFAULT_COS_SHIFT_B)))

        # Load filter ranges
        if state.intf.get('azimuth_min') is not None:
            self.azi_min_var.set(str(state.intf['azimuth_min']))
        if state.intf.get('azimuth_max') is not None:
            self.azi_max_var.set(str(state.intf['azimuth_max']))
        if state.intf.get('elevation_min') is not None:
            self.elv_min_var.set(str(state.intf['elevation_min']))
        if state.intf.get('elevation_max') is not None:
            self.elv_max_var.set(str(state.intf['elevation_max']))
        if state.intf.get('time_min') is not None:
            self.time_start_var.set(str(state.intf['time_min']))
        if state.intf.get('time_max') is not None:
            self.time_stop_var.set(str(state.intf['time_max']))

        # Load timing
        self.T0_var.set(str(state.timing.get('T0', 0)))

        # Load event time
        if state.intf.get('event_time'):
            self.event_time_var.set(state.intf['event_time'])

        # LMA calibration parameters
        if state.files.get('lma'):
            self.lma_file_path_var.set(state.files['lma'])
        if state.intf.get('trigger_offset_ms') is not None:
            self.lma_trigger_offset_var.set(str(state.intf['trigger_offset_ms']))
        if state.intf.get('lma_match_window_us') is not None:
            self.lma_match_window_var.set(str(state.intf['lma_match_window_us']))
        if state.intf.get('lma_chi2_max') is not None:
            self.lma_chi2_max_var.set(str(state.intf['lma_chi2_max']))
        if state.intf.get('lma_alt_min_m') is not None:
            self.lma_alt_min_var.set(str(state.intf['lma_alt_min_m']))
        if state.intf.get('lma_alt_max_m') is not None:
            self.lma_alt_max_var.set(str(state.intf['lma_alt_max_m']))
        if state.intf.get('lma_power_min') is not None:
            self.lma_power_min_var.set(str(state.intf['lma_power_min']))

        # Plot settings
        self.show_grid_var.set(state.intf.get('show_grid', True))
        if state.intf.get('source_distance_km') is not None:
            self.source_distance_var.set(str(state.intf['source_distance_km']))

        # TASD overlay settings
        self.show_tasd_var.set(state.intf.get('show_tasd_overlay', False))
        if state.files.get('sd_directory'):
            self.tasd_dir_var.set(state.files['sd_directory'])

        # Source azi/elv from results
        if state.results.get('source_azi') is not None:
            self.source_azi_var.set(str(state.results['source_azi']))
        if state.results.get('source_elv') is not None:
            self.source_elv_var.set(str(state.results['source_elv']))

        # Update calibration toggle state
        self._on_calibration_toggle()
