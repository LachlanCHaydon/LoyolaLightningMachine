"""
TGF Analysis Tool - INTF (Interferometer) Tab
==============================================
Handles INTF data calibration and azimuth-elevation visualization.

Features:
- Load raw or pre-calibrated INTF data
- Apply cosine shift calibration
- Azimuth-Elevation scatter plots (time-colored)
- Time-Elevation plots
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
    INTF_LAT, INTF_LON
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
        self.skip_header_var = tk.StringVar(value="2")
        
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
        self.plot_mode_var = tk.StringVar(value="az_el")  # az_el, time_el, time_az
        
        # Plot style
        self.show_colorbar_var = tk.BooleanVar(value=True)
        self.color_by_var = tk.StringVar(value="time")  # time, signal
        self.marker_size_var = tk.StringVar(value="auto")  # auto, fixed
        self.fixed_marker_size_var = tk.StringVar(value="10")
        self.plot_title_var = tk.StringVar(value="")

        # LMA auto-calibration
        self.lma_file_path_var      = tk.StringVar()
        self.lma_trigger_offset_var = tk.StringVar(value="0.0")   # ms added to event SOD
        self.lma_match_window_var   = tk.StringVar(value="50.0")  # µs
        self.lma_chi2_max_var       = tk.StringVar(value="2.0")
        self.lma_alt_min_var        = tk.StringVar(value="1000")  # m
        self.lma_alt_max_var        = tk.StringVar(value="15000") # m
        self.lma_power_min_var      = tk.StringVar(value="-20.0") # dBW
        self.lma_result_text        = None  # Text widget populated after calibration run

        # TASD overlay
        self.tasd_dir_var         = tk.StringVar()
        self.tasd_coords_path_var = tk.StringVar(value="tasd_gpscoors.txt")
        self.show_tasd_var        = tk.BooleanVar(value=False)
        self.source_azi_var       = tk.StringVar(value="")
        self.source_elv_var       = tk.StringVar(value="")
        self.tasd_handler         = None   # TASDHandler populated on load

        # Build UI
        self._build_ui()
    
    def _build_ui(self):
        """Build the tab UI."""
        # Main horizontal paned window
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        self.control_frame = ttk.Frame(self.paned, width=670)
        self.control_frame.pack_propagate(False)
        self.paned.add(self.control_frame, weight=0)
        
        # Right panel - Plot
        self.plot_frame = ttk.Frame(self.paned)
        self.paned.add(self.plot_frame, weight=1)
        
        # Build control panel with scrollbar
        self._build_control_panel()
        
        # Build plot area
        self._build_plot_area()
    
    def _build_control_panel(self):
        """Build the left control panel with scrolling."""
        # Canvas for scrolling
        canvas = tk.Canvas(self.control_frame, width=670)
        scrollbar = ttk.Scrollbar(self.control_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add sections
        self._build_file_section()
        self._build_lma_calibration_section()
        self._build_timing_section()
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
        ttk.Entry(row, textvariable=self.file_path_var, width=28).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3, command=self._browse_file).pack(side=tk.LEFT)
        
        # Data type
        ttk.Checkbutton(frame, text="File is pre-calibrated (has Az/El)", 
                       variable=self.is_calibrated_var,
                       command=self._on_calibration_toggle).pack(anchor='w', pady=2)
        
        # Skip header
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Header lines to skip:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.skip_header_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Load button
        ttk.Button(frame, text="Load INTF Data", command=self._load_data).pack(pady=5)
    
    def _build_lma_calibration_section(self):
        """Build cosine shift calibration section with LMA auto-calibration sub-panel."""
        self.cal_frame = ttk.LabelFrame(self.scrollable_frame, text="Cosine Shift Calibration", padding=5)
        self.cal_frame.pack(fill=tk.X, padx=5, pady=5)

        # ── Manual Shifts ──────────────────────────────────────────────────────
        manual_frame = ttk.LabelFrame(self.cal_frame, text="Manual Shifts", padding=4)
        manual_frame.pack(fill=tk.X, pady=3)

        # Cos Shift A
        row = ttk.Frame(manual_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Cos Shift A:", width=12).pack(side=tk.LEFT)
        self.cos_a_entry = ttk.Entry(row, textvariable=self.cos_shift_a_var, width=12)
        self.cos_a_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="-", width=2,
                   command=lambda: self._adjust_cos_shift('a', -1)).pack(side=tk.LEFT)
        ttk.Button(row, text="+", width=2,
                   command=lambda: self._adjust_cos_shift('a', +1)).pack(side=tk.LEFT)

        # Cos Shift B
        row = ttk.Frame(manual_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Cos Shift B:", width=12).pack(side=tk.LEFT)
        self.cos_b_entry = ttk.Entry(row, textvariable=self.cos_shift_b_var, width=12)
        self.cos_b_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="-", width=2,
                   command=lambda: self._adjust_cos_shift('b', -1)).pack(side=tk.LEFT)
        ttk.Button(row, text="+", width=2,
                   command=lambda: self._adjust_cos_shift('b', +1)).pack(side=tk.LEFT)

        # Step size
        row = ttk.Frame(manual_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Step size:").pack(side=tk.LEFT)
        self.step_size_var = tk.StringVar(value="0.001")
        ttk.Combobox(row, textvariable=self.step_size_var,
                     values=["0.0001", "0.001", "0.01"], width=8).pack(side=tk.LEFT, padx=5)

        # Apply / Reset buttons
        btn_row = ttk.Frame(manual_frame)
        btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row, text="Apply Calibration",
                   command=self._apply_calibration).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Reset to Defaults",
                   command=self._reset_calibration).pack(side=tk.LEFT, padx=2)

        # ── LMA Auto-Calibration ───────────────────────────────────────────────
        lma_frame = ttk.LabelFrame(self.cal_frame, text="LMA Auto-Calibration", padding=4)
        lma_frame.pack(fill=tk.X, pady=3)

        # LMA file picker
        row = ttk.Frame(lma_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="LMA File:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_file_path_var, width=22).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3,
                   command=self._browse_lma_file).pack(side=tk.LEFT)

        # Trigger offset
        row = ttk.Frame(lma_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Trigger offset (ms):", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_trigger_offset_var, width=8).pack(side=tk.LEFT, padx=2)

        # Match window + chi² max
        row = ttk.Frame(lma_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Match window (µs):").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_match_window_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="  chi² max:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_chi2_max_var, width=5).pack(side=tk.LEFT, padx=2)

        # Altitude range
        row = ttk.Frame(lma_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Alt range (m):").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_alt_min_var, width=7).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_alt_max_var, width=7).pack(side=tk.LEFT, padx=2)

        # Power min
        row = ttk.Frame(lma_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Power min (dBW):", width=17).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_power_min_var, width=7).pack(side=tk.LEFT, padx=2)

        # Run button
        ttk.Button(lma_frame, text="Run LMA Calibration",
                   command=self._run_lma_calibration).pack(pady=4)

        # Results text box
        res_frame = ttk.LabelFrame(lma_frame, text="Results", padding=3)
        res_frame.pack(fill=tk.X, pady=2)
        self.lma_result_text = tk.Text(res_frame, height=5, width=30, wrap=tk.WORD,
                                       font=('Courier', 9))
        self.lma_result_text.pack(fill=tk.X)
        self.lma_result_text.insert('1.0', "No calibration run yet")
        self.lma_result_text.config(state='disabled')

    def _build_timing_section(self):
        """Build timing parameters section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Timing", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Event Time (for display on colorbar)
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Event Time:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.event_time_var, width=12).pack(side=tk.LEFT)
        ttk.Label(row, text="(HH:MM:SS)", foreground='gray').pack(side=tk.LEFT, padx=5)

        # T0 Reference
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="T0 Reference (µs):", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.T0_var, width=12).pack(side=tk.LEFT)

        # Time format
        ttk.Checkbutton(frame, text="Time is already relative to T0",
                        variable=self.time_is_relative_var).pack(anchor='w', pady=2)

        # Sync from Home tab button
        ttk.Button(frame, text="Sync T0 from Home Tab",
                   command=self._sync_from_home).pack(pady=2)
    
    def _build_tasd_section(self):
        """Build TASD overlay section for the Az-El plot."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="TASD Overlay", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # TASD directory
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="TASD Directory:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.tasd_dir_var, width=20).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3,
                   command=self._browse_tasd_dir).pack(side=tk.LEFT)

        # Coords file
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Coords file:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.tasd_coords_path_var, width=20).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3,
                   command=self._browse_tasd_coords).pack(side=tk.LEFT)

        ttk.Button(frame, text="Load TASD Data",
                   command=self._load_tasd_for_intf).pack(pady=4)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=3)

        # Source location for shower axis
        ttk.Label(frame, text="Source Location (for shower axis):",
                  font=('Helvetica', 9, 'bold')).pack(anchor='w')

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Azimuth (°):", width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.source_azi_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="Elevation (°):").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Entry(row, textvariable=self.source_elv_var, width=8).pack(side=tk.LEFT, padx=2)

        ttk.Button(frame, text="Sync from Timeshift",
                   command=self._sync_source_from_project).pack(pady=2)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=3)

        ttk.Checkbutton(frame, text="Show TASD overlay on Az-El plot",
                        variable=self.show_tasd_var).pack(anchor='w')

    def _build_filter_section(self):
        """Build data filter section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Data Filters", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Time range
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Time (µs):", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.time_start_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.time_stop_var, width=10).pack(side=tk.LEFT, padx=2)
        
        # Azimuth range
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Azimuth (°):", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.azi_min_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.azi_max_var, width=10).pack(side=tk.LEFT, padx=2)
        
        # Elevation range
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Elevation (°):", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.elv_min_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.elv_max_var, width=10).pack(side=tk.LEFT, padx=2)
        
        # Clear filters
        ttk.Button(frame, text="Clear Time Filter", 
                  command=self._clear_time_filter).pack(pady=2)
    
    def _build_plot_section(self):
        """Build plot options section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Plot Options", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Plot title
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Plot Title:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.plot_title_var, width=25).pack(side=tk.LEFT, padx=5)
        
        # Plot mode
        ttk.Label(frame, text="Plot Mode:").pack(anchor='w')
        modes_frame = ttk.Frame(frame)
        modes_frame.pack(fill=tk.X, pady=2)
        
        ttk.Radiobutton(modes_frame, text="Az vs El", variable=self.plot_mode_var,
                       value="az_el").pack(side=tk.LEFT)
        ttk.Radiobutton(modes_frame, text="Time vs El", variable=self.plot_mode_var,
                       value="time_el").pack(side=tk.LEFT)
        ttk.Radiobutton(modes_frame, text="Time vs Az", variable=self.plot_mode_var,
                       value="time_az").pack(side=tk.LEFT)
        
        # Color by
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Color by:").pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="Time", variable=self.color_by_var,
                       value="time").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(row, text="Signal", variable=self.color_by_var,
                       value="signal").pack(side=tk.LEFT)
        
        # Marker size
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Marker size:").pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="Auto", variable=self.marker_size_var,
                       value="auto").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(row, text="Fixed:", variable=self.marker_size_var,
                       value="fixed").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.fixed_marker_size_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # Show colorbar
        ttk.Checkbutton(frame, text="Show Colorbar", 
                       variable=self.show_colorbar_var).pack(anchor='w')
    
    def _build_action_buttons(self):
        """Build action buttons."""
        frame = ttk.Frame(self.scrollable_frame, padding=5)
        frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(frame, text="🔄 Update Plot",
                  command=self._update_plot).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="📷 Export PNG",
                  command=self._export_png).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="💾 Export Calibrated Data",
                  command=self._export_calibrated).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="📋 Save to Project",
                  command=self._save_to_project).pack(fill=tk.X, pady=2)
    
    def _build_info_section(self):
        """Build data info display section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Data Info", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_text = tk.Text(frame, height=10, width=35, wrap=tk.WORD,
                                font=('Courier', 9))
        self.info_text.pack(fill=tk.X)
        self.info_text.insert('1.0', "No data loaded")
        self.info_text.config(state='disabled')
    
    def _build_plot_area(self):
        """Build the matplotlib plot area."""
        # Create figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        
        # Add toolbar
        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Add canvas widget
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Status bar for coordinates
        self.coord_var = tk.StringVar(value="")
        coord_label = ttk.Label(self.plot_frame, textvariable=self.coord_var)
        coord_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Connect mouse motion event
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
    
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
            # Pre-calibrated - disable calibration controls
            self.skip_header_var.set("2")
            for child in self.cal_frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button)):
                    child.configure(state='disabled')
        else:
            # Raw data - enable calibration controls
            self.skip_header_var.set("57")
            for child in self.cal_frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button)):
                    child.configure(state='normal')
    
    def _load_data(self):
        """Load INTF data from file."""
        filepath = self.file_path_var.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Error", "Please select a valid INTF file")
            return
        
        try:
            # Get parameters
            is_calibrated = self.is_calibrated_var.get()
            skip_header = int(self.skip_header_var.get())
            T0 = float(self.T0_var.get())
            time_is_relative = self.time_is_relative_var.get()
            
            # Load data
            if is_calibrated:
                success = self.intf_handler.load_data(
                    filepath, 
                    is_calibrated=True,
                    skip_header=skip_header,
                    T0_reference=T0,
                    time_is_relative=time_is_relative
                )
            else:
                # Set cos shifts before loading raw data
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
        """Adjust cos shift by step size."""
        try:
            step = float(self.step_size_var.get())
            if which == 'a':
                current = float(self.cos_shift_a_var.get())
                self.cos_shift_a_var.set(f"{current + direction * step:.6f}")
            else:
                current = float(self.cos_shift_b_var.get())
                self.cos_shift_b_var.set(f"{current + direction * step:.6f}")
            
            # Auto-apply if data is loaded
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
        """Sync T0 from Home tab."""
        try:
            home_T0 = self.main_app.home_tab.T0_var.get()
            self.T0_var.set(home_T0)
            
            # Update handler
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

    def _browse_tasd_coords(self):
        """Browse for TASD coordinates file."""
        filepath = filedialog.askopenfilename(
            title="Select TASD Coordinates File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            self.tasd_coords_path_var.set(filepath)

    def _run_lma_calibration(self):
        """Run LMA auto-calibration and populate cos shift fields."""
        if not self.intf_handler.is_loaded:
            messagebox.showerror("Error", "Load a raw INTF file first")
            return

        lma_file = self.lma_file_path_var.get()
        if not lma_file or not os.path.exists(lma_file):
            messagebox.showerror("Error", "Please select a valid LMA file")
            return

        # Parse event time HH:MM:SS → seconds-of-day
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
                'chi2_max':        float(self.lma_chi2_max_var.get()),
                'alt_min_m':       float(self.lma_alt_min_var.get()),
                'alt_max_m':       float(self.lma_alt_max_var.get()),
                'power_min_dbw':   float(self.lma_power_min_var.get()),
            }
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid LMA filter parameter:\n{e}")
            return

        self.main_app.status_var.set("Running LMA calibration…")
        self.update_idletasks()

        result = self.intf_handler.run_lma_calibration(lma_file, intf_t0_sod, **params)

        if result is None:
            self.lma_result_text.config(state='normal')
            self.lma_result_text.delete('1.0', tk.END)
            self.lma_result_text.insert('1.0', "Calibration failed.\nCheck LMA file format,\nevent time, and filter parameters.")
            self.lma_result_text.config(state='disabled')
            self.main_app.status_var.set("LMA calibration failed")
            return

        # Populate cos shift fields and apply
        self.cos_shift_a_var.set(f"{result['cos_shift_a']:.6f}")
        self.cos_shift_b_var.set(f"{result['cos_shift_b']:.6f}")
        self._apply_calibration()

        # Display result summary
        summary = (
            f"cosShiftA = {result['cos_shift_a']:+.5f} ± {result['se_a']:.5f}\n"
            f"cosShiftB = {result['cos_shift_b']:+.5f} ± {result['se_b']:.5f}\n"
            f"n_matched = {result['n_matched']}  n_final = {result['n_final']}\n"
            f"Formal err = ±{result['formal_error_deg']:.4f}°"
        )
        self.lma_result_text.config(state='normal')
        self.lma_result_text.delete('1.0', tk.END)
        self.lma_result_text.insert('1.0', summary)
        self.lma_result_text.config(state='disabled')

        self.main_app.mark_unsaved()
        self.main_app.status_var.set(
            f"LMA calibration done: A={result['cos_shift_a']:+.5f}, B={result['cos_shift_b']:+.5f}, "
            f"err=±{result['formal_error_deg']:.4f}°"
        )

    def _load_tasd_for_intf(self):
        """Load TASD data for the Az-El overlay."""
        tasd_dir = self.tasd_dir_var.get()
        if not tasd_dir or not os.path.isdir(tasd_dir):
            messagebox.showerror("Error", "Please select a valid TASD directory")
            return

        coords_path = self.tasd_coords_path_var.get()
        if not coords_path or not os.path.exists(coords_path):
            messagebox.showerror("Error", "Please select a valid TASD coordinates file")
            return

        # Derive time string (HHMMSS) from event time field
        event_time_str = self.event_time_var.get().strip()
        if event_time_str and event_time_str != "HH:MM:SS" and ":" in event_time_str:
            time_str = event_time_str.replace(":", "")
        else:
            # Fall back to project state event_info
            ei = self.main_app.project_state.event_info
            time_str = ei.get('time', '').replace(":", "") if ei else ""

        self.tasd_handler = TASDHandler()
        self.tasd_handler.load_coordinates(coords_path)
        success = self.tasd_handler.load_directory(tasd_dir, time_str)

        if success:
            n = len(self.tasd_handler.detectors)
            self.main_app.status_var.set(f"Loaded TASD: {n} detector(s) for overlay")
        else:
            messagebox.showwarning("Warning",
                                   "No TASD waveform files matched the time string.\n"
                                   "Coordinates loaded; check event time or use .stitch files.")
            self.main_app.status_var.set("TASD: no waveforms matched")

    def _sync_source_from_project(self):
        """Populate source azi/elv from Timeshift results in ProjectState."""
        results = self.main_app.project_state.results
        azi = results.get('source_azi')
        elv = results.get('source_elv')
        if azi is not None and elv is not None:
            self.source_azi_var.set(f"{float(azi):.2f}")
            self.source_elv_var.set(f"{float(elv):.2f}")
            self.main_app.status_var.set(f"Source synced: Az={azi:.2f}°, El={elv:.2f}°")
        else:
            messagebox.showinfo("Info",
                                "No source location found in project results.\n"
                                "Run Timeshift analysis first, or enter values manually.")

    def _clear_time_filter(self):
        """Clear time filter."""
        self.time_start_var.set("")
        self.time_stop_var.set("")
    
    def _auto_set_filters(self):
        """Auto-set filter ranges from loaded data."""
        if not self.intf_handler.is_loaded:
            return
        
        data = self.intf_handler.get_full_data()
        if data is None:
            return
        
        # Auto-set ranges with some margin
        t_min, t_max = np.min(data['time']), np.max(data['time'])
        az_min, az_max = np.min(data['azimuth']), np.max(data['azimuth'])
        el_min, el_max = np.min(data['elevation']), np.max(data['elevation'])
        
        # Round to nice values
        self.azi_min_var.set(f"{int(az_min - 5)}")
        self.azi_max_var.set(f"{int(az_max + 5)}")
        self.elv_min_var.set(f"{max(0, int(el_min - 2))}")
        self.elv_max_var.set(f"{int(el_max + 5)}")
    
    def _get_filter_params(self):
        """Get current filter parameters."""
        params = {}
        
        # Time
        if self.time_start_var.get():
            try:
                params['t_min'] = float(self.time_start_var.get())
            except:
                pass
        if self.time_stop_var.get():
            try:
                params['t_max'] = float(self.time_stop_var.get())
            except:
                pass
        
        # Azimuth
        try:
            params['azi_min'] = float(self.azi_min_var.get())
        except:
            pass
        try:
            params['azi_max'] = float(self.azi_max_var.get())
        except:
            pass
        
        # Elevation
        try:
            params['elv_min'] = float(self.elv_min_var.get())
        except:
            pass
        try:
            params['elv_max'] = float(self.elv_max_var.get())
        except:
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
        
        # Determine what to plot
        plot_mode = self.plot_mode_var.get()
        color_by = self.color_by_var.get()
        
        # Get colors
        if color_by == "time":
            colors = data['time']
            cmap = 'jet'
            # Dynamic label based on event time
            event_time = self.event_time_var.get()
            if event_time and event_time != "HH:MM:SS":
                clabel = f"Time relative to {event_time} (µs)"
            else:
                clabel = "Time (µs)"
        else:  # signal
            colors = data['colors'] if data['colors'] is not None else 'blue'
            cmap = None
            clabel = "Signal Strength"
        
        # Get marker sizes
        if self.marker_size_var.get() == "auto" and data['marker_sizes'] is not None:
            sizes = data['marker_sizes']
        else:
            try:
                sizes = float(self.fixed_marker_size_var.get())
            except:
                sizes = 10
        
        # Create scatter plot based on mode
        if plot_mode == "az_el":
            if isinstance(colors, np.ndarray) and cmap:
                sc = self.ax.scatter(data['azimuth'], data['elevation'], 
                                    c=colors, cmap=cmap, s=sizes, alpha=0.7)
            else:
                sc = self.ax.scatter(data['azimuth'], data['elevation'],
                                    c=colors, s=sizes, alpha=0.7)
            self.ax.set_xlabel("Azimuth (°)", fontsize=12)
            self.ax.set_ylabel("Elevation (°)", fontsize=12)
            title = self.plot_title_var.get() or "INTF: Azimuth vs Elevation"
            self.ax.set_title(title, fontsize=14)
            
            # Set limits
            try:
                self.ax.set_xlim(float(self.azi_min_var.get()), float(self.azi_max_var.get()))
            except:
                pass
            try:
                self.ax.set_ylim(float(self.elv_min_var.get()), float(self.elv_max_var.get()))
            except:
                pass
                
        elif plot_mode == "time_el":
            if isinstance(colors, np.ndarray) and cmap:
                sc = self.ax.scatter(data['time'], data['elevation'],
                                    c=colors, cmap=cmap, s=sizes, alpha=0.7)
            else:
                sc = self.ax.scatter(data['time'], data['elevation'],
                                    c=colors, s=sizes, alpha=0.7)
            self.ax.set_xlabel("Time (µs)", fontsize=12)
            self.ax.set_ylabel("Elevation (°)", fontsize=12)
            self.ax.set_title("INTF: Time vs Elevation", fontsize=14)
            
            try:
                self.ax.set_ylim(float(self.elv_min_var.get()), float(self.elv_max_var.get()))
            except:
                pass
                
        elif plot_mode == "time_az":
            if isinstance(colors, np.ndarray) and cmap:
                sc = self.ax.scatter(data['time'], data['azimuth'],
                                    c=colors, cmap=cmap, s=sizes, alpha=0.7)
            else:
                sc = self.ax.scatter(data['time'], data['azimuth'],
                                    c=colors, s=sizes, alpha=0.7)
            self.ax.set_xlabel("Time (µs)", fontsize=12)
            self.ax.set_ylabel("Azimuth (°)", fontsize=12)
            self.ax.set_title("INTF: Time vs Azimuth", fontsize=14)
            
            try:
                self.ax.set_ylim(float(self.azi_min_var.get()), float(self.azi_max_var.get()))
            except:
                pass
        
        # TASD overlay (Az-El mode only)
        if plot_mode == "az_el" and self.show_tasd_var.get() and self.tasd_handler:
            triggered = [d for d in self.tasd_handler.detectors
                         if d.get('lat') is not None and d.get('vem', 0) > 0]
            if triggered:
                det_azis = []
                vems = []
                for det in triggered:
                    try:
                        det_azi = calculate_azimuth_to_detector(det['lat'], det['lon'])
                        det_azis.append(det_azi)
                        vems.append(det['vem'])
                        size = max(50, 150 * np.log1p(det['vem']))
                        self.ax.scatter(det_azi, 0, s=size,
                                        facecolors='none', edgecolors='gray',
                                        linewidths=1.5, zorder=4)
                    except Exception:
                        pass

                # Shower axis from source location
                try:
                    src_azi = float(self.source_azi_var.get())
                    src_elv = float(self.source_elv_var.get())
                    if det_azis:
                        core_azi = float(np.average(det_azis, weights=vems))
                        self.ax.plot([src_azi, core_azi], [src_elv, 0],
                                     'r-', lw=1.5, label='Shower axis', zorder=5)
                        self.ax.plot([src_azi, min(det_azis)], [src_elv, 0],
                                     'r--', lw=1, alpha=0.6, zorder=5)
                        self.ax.plot([src_azi, max(det_azis)], [src_elv, 0],
                                     'r--', lw=1, alpha=0.6, zorder=5)
                        self.ax.scatter(src_azi, src_elv, marker='*', s=200,
                                        c='red', zorder=6, label='Source')
                        self.ax.legend(fontsize=9, loc='upper right')
                except (ValueError, TypeError):
                    pass

        # Add colorbar
        if self.show_colorbar_var.get() and isinstance(colors, np.ndarray) and cmap:
            cbar = self.fig.colorbar(sc, ax=self.ax, pad=0.02)
            cbar.set_label(clabel, fontsize=10)
        
        # Grid
        self.ax.grid(True, linestyle=':', alpha=0.5)
        
        # Tight layout
        self.fig.tight_layout()
        
        # Draw
        self.canvas.draw()
        
        # Update status with point count
        self.main_app.status_var.set(f"Plotting {len(data['time']):,} INTF points")
    
    def _on_mouse_move(self, event):
        """Handle mouse movement for coordinate display."""
        if event.inaxes:
            self.coord_var.set(f"x={event.xdata:.2f}, y={event.ydata:.2f}")
        else:
            self.coord_var.set("")
    
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
                # Get time range filter
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
    
    def _save_to_project(self):
        """Save current INTF settings to project state."""
        state = self.main_app.project_state
        
        # Save file path
        state.files['intf_raw'] = self.file_path_var.get() or None
        
        # Save calibration settings
        try:
            state.intf['cos_shift_a'] = float(self.cos_shift_a_var.get())
            state.intf['cos_shift_b'] = float(self.cos_shift_b_var.get())
        except:
            pass
        
        state.intf['is_calibrated'] = self.is_calibrated_var.get()
        
        # Save filter ranges
        try:
            state.intf['azimuth_min'] = float(self.azi_min_var.get())
            state.intf['azimuth_max'] = float(self.azi_max_var.get())
            state.intf['elevation_min'] = float(self.elv_min_var.get())
            state.intf['elevation_max'] = float(self.elv_max_var.get())
        except:
            pass
        
        if self.time_start_var.get():
            try:
                state.intf['time_min'] = float(self.time_start_var.get())
            except:
                pass
        if self.time_stop_var.get():
            try:
                state.intf['time_max'] = float(self.time_stop_var.get())
            except:
                pass

        # Save event time
        event_time = self.event_time_var.get()
        if event_time and event_time != "HH:MM:SS":
            state.intf['event_time'] = event_time

        # LMA calibration parameters
        state.files['lma'] = self.lma_file_path_var.get() or None
        try:
            state.intf['trigger_offset_ms']   = float(self.lma_trigger_offset_var.get())
        except ValueError:
            pass
        try:
            state.intf['lma_match_window_us'] = float(self.lma_match_window_var.get())
        except ValueError:
            pass
        try:
            state.intf['lma_chi2_max']        = float(self.lma_chi2_max_var.get())
        except ValueError:
            pass
        try:
            state.intf['lma_alt_min_m']       = float(self.lma_alt_min_var.get())
        except ValueError:
            pass
        try:
            state.intf['lma_alt_max_m']       = float(self.lma_alt_max_var.get())
        except ValueError:
            pass
        try:
            state.intf['lma_power_min']       = float(self.lma_power_min_var.get())
        except ValueError:
            pass

        # TASD overlay settings
        state.intf['show_tasd_overlay'] = self.show_tasd_var.get()
        state.intf['tasd_coords_path']  = self.tasd_coords_path_var.get()
        if self.tasd_dir_var.get():
            state.files['sd_directory'] = self.tasd_dir_var.get()

        # Source azi/elv (also written by Timeshift tab; keep manual override here)
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

        # TASD overlay settings
        self.show_tasd_var.set(state.intf.get('show_tasd_overlay', False))
        if state.intf.get('tasd_coords_path'):
            self.tasd_coords_path_var.set(state.intf['tasd_coords_path'])
        if state.files.get('sd_directory'):
            self.tasd_dir_var.set(state.files['sd_directory'])

        # Source azi/elv from results (written by Timeshift or manually)
        if state.results.get('source_azi') is not None:
            self.source_azi_var.set(str(state.results['source_azi']))
        if state.results.get('source_elv') is not None:
            self.source_elv_var.set(str(state.results['source_elv']))

        # Update calibration toggle state
        self._on_calibration_toggle()