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
from data_handlers import InterferometerHandler


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
        self._build_calibration_section()
        self._build_timing_section()
        self._build_filter_section()
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
    
    def _build_calibration_section(self):
        """Build cosine shift calibration section."""
        self.cal_frame = ttk.LabelFrame(self.scrollable_frame, text="Cosine Shift Calibration", padding=5)
        self.cal_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Info label
        ttk.Label(self.cal_frame, 
                 text="Adjust cos shifts to align INTF sources\nwith expected lightning channel",
                 font=('Helvetica', 9), foreground='gray').pack(anchor='w')
        
        # Cos Shift A
        row = ttk.Frame(self.cal_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Cos Shift A:", width=12).pack(side=tk.LEFT)
        self.cos_a_entry = ttk.Entry(row, textvariable=self.cos_shift_a_var, width=12)
        self.cos_a_entry.pack(side=tk.LEFT, padx=2)
        
        # Cos Shift A adjustment buttons
        ttk.Button(row, text="-", width=2, 
                  command=lambda: self._adjust_cos_shift('a', -0.001)).pack(side=tk.LEFT)
        ttk.Button(row, text="+", width=2,
                  command=lambda: self._adjust_cos_shift('a', 0.001)).pack(side=tk.LEFT)
        
        # Cos Shift B
        row = ttk.Frame(self.cal_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Cos Shift B:", width=12).pack(side=tk.LEFT)
        self.cos_b_entry = ttk.Entry(row, textvariable=self.cos_shift_b_var, width=12)
        self.cos_b_entry.pack(side=tk.LEFT, padx=2)
        
        # Cos Shift B adjustment buttons
        ttk.Button(row, text="-", width=2,
                  command=lambda: self._adjust_cos_shift('b', -0.001)).pack(side=tk.LEFT)
        ttk.Button(row, text="+", width=2,
                  command=lambda: self._adjust_cos_shift('b', 0.001)).pack(side=tk.LEFT)
        
        # Fine/coarse adjustment
        row = ttk.Frame(self.cal_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Step size:").pack(side=tk.LEFT)
        self.step_size_var = tk.StringVar(value="0.001")
        ttk.Combobox(row, textvariable=self.step_size_var, 
                    values=["0.0001", "0.001", "0.01"], width=8).pack(side=tk.LEFT, padx=5)
        
        # Apply button
        ttk.Button(self.cal_frame, text="Apply Calibration", 
                  command=self._apply_calibration).pack(pady=5)
        
        # Reset to defaults
        ttk.Button(self.cal_frame, text="Reset to Defaults",
                  command=self._reset_calibration).pack(pady=2)

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

        if self.time_stop_var.get():
            try:
                state.intf['time_max'] = float(self.time_stop_var.get())
            except:
                pass

            # Save event time
        event_time = self.event_time_var.get()
        if event_time and event_time != "HH:MM:SS":
            state.intf['event_time'] = event_time

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

            # Update calibration toggle state
            self._on_calibration_toggle()