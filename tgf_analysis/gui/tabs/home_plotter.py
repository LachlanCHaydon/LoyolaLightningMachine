"""
TGF Analysis Tool - Home/Master Plotter Tab
===========================================
Multi-instrument plotting with FA, SD, INTF, Photometer, Luminosity.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os

# Matplotlib with Tk backend
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from config import (
    INSTRUMENT_COLORS, DEFAULT_FONT_SIZE, DEFAULT_LIMITS,
    cmap_mjet, PHOTOMETER_CALIBRATION
)
from data_handlers import (
    PhotometerHandler, FastAntennaHandler, InterferometerHandler,
    TASDHandler, LuminosityHandler
)


class HomePlotterTab(ttk.Frame):
    """
    Master Plotter tab for multi-instrument visualization.

    Features:
    - File input panel for all instrument data
    - Interactive matplotlib plot with zoom/pan
    - Multi-axis overlay with toggleable instruments
    - Customizable plot ranges and export
    """

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app

        # Data handlers
        self.fa_handler = FastAntennaHandler()
        self.intf_handler = InterferometerHandler()
        self.tasd_handler = TASDHandler()
        self.photometer_handler = PhotometerHandler()
        self.luminosity_handler = LuminosityHandler()

        # File paths (will sync with project state)
        self.file_paths = {
            'fa': tk.StringVar(),
            'intf': tk.StringVar(),
            'sd_dir': tk.StringVar(),
            'photometer': tk.StringVar(),
            'luminosity': tk.StringVar(),
        }

        # Timing parameters
        self.T0_var = tk.StringVar(value="730069")
        self.timeshift_var = tk.StringVar(value="35.8")
        self.phot_offset_var = tk.StringVar(value="0")

        # Plot range variables
        self.x_start_var = tk.StringVar(value="0")
        self.x_stop_var = tk.StringVar(value="100000")

        # Zoom range variables
        self.zoom_start_var = tk.StringVar(value="18900")
        self.zoom_stop_var = tk.StringVar(value="19300")
        self.show_zoom_var = tk.BooleanVar(value=False)

        # Visibility toggles
        self.show_fa_var = tk.BooleanVar(value=True)
        self.show_intf_var = tk.BooleanVar(value=True)
        self.show_sd_var = tk.BooleanVar(value=True)
        self.show_lum_var = tk.BooleanVar(value=True)
        self.show_phot_337_var = tk.BooleanVar(value=True)
        self.show_phot_391_var = tk.BooleanVar(value=True)
        self.show_phot_777_var = tk.BooleanVar(value=True)

        # Y-axis limits
        self.fa_ymin_var = tk.StringVar(value="-80")
        self.fa_ymax_var = tk.StringVar(value="30")
        self.sd_ymin_var = tk.StringVar(value="0")
        self.sd_ymax_var = tk.StringVar(value="500")
        self.intf_ymin_var = tk.StringVar(value="0")
        self.intf_ymax_var = tk.StringVar(value="40")
        self.lum_ymin_var = tk.StringVar(value="0")
        self.lum_ymax_var = tk.StringVar(value="1.1")
        self.phot_ymin_var = tk.StringVar(value="-2000")
        self.phot_ymax_var = tk.StringVar(value="55000")

        # INTF settings
        self.intf_calibrated_var = tk.BooleanVar(value=True)

        # Plot style
        self.show_grid_var = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=True)
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

        # Add sections to scrollable frame
        self._build_file_section()
        self._build_timing_section()
        self._build_range_section()
        self._build_visibility_section()
        self._build_limits_section()
        self._build_style_section()
        self._build_action_buttons()

    def _build_file_section(self):
        """Build file input section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Data Files", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Fast Antenna
        self._add_file_row(frame, "Fast Antenna:", self.file_paths['fa'],
                         [("CSV files", "*.csv"), ("All files", "*.*")])

        # INTF
        self._add_file_row(frame, "INTF:", self.file_paths['intf'],
                         [("DAT files", "*.dat"), ("All files", "*.*")])

        # INTF calibrated checkbox
        ttk.Checkbutton(frame, text="INTF is pre-calibrated",
                       variable=self.intf_calibrated_var).pack(anchor='w', padx=20)

        # SD Directory
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="SD Directory:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.file_paths['sd_dir'], width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3,
                  command=lambda: self._browse_directory(self.file_paths['sd_dir'])).pack(side=tk.LEFT)

        # Photometer
        self._add_file_row(frame, "Photometer:", self.file_paths['photometer'],
                         [("DAT files", "*.dat"), ("All files", "*.*")])

        # Luminosity
        self._add_file_row(frame, "Luminosity:", self.file_paths['luminosity'],
                         [("Text files", "*.txt"), ("DAT files", "*.dat"), ("All files", "*.*")])

        # Load button
        ttk.Button(frame, text="Load All Data", command=self._load_all_data).pack(pady=5)

    def _add_file_row(self, parent, label, var, filetypes):
        """Add a file selection row."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3,
                  command=lambda: self._browse_file(var, filetypes)).pack(side=tk.LEFT)

    def _build_timing_section(self):
        """Build timing parameters section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Timing Parameters", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # T0
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="T0 (µs):", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.T0_var, width=12).pack(side=tk.LEFT)

        # Timeshift
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="SD Timeshift (µs):", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.timeshift_var, width=12).pack(side=tk.LEFT)

        # Photometer offset
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Phot. Sec Offset:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.phot_offset_var, width=12).pack(side=tk.LEFT)

    def _build_range_section(self):
        """Build plot range section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Plot Range (µs)", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Main range
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="X Start:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.x_start_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="X Stop:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.x_stop_var, width=10).pack(side=tk.LEFT, padx=2)

        # Zoom range
        ttk.Checkbutton(frame, text="Show Zoom Subplot",
                       variable=self.show_zoom_var).pack(anchor='w', pady=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Zoom Start:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.zoom_start_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="Zoom Stop:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.zoom_stop_var, width=10).pack(side=tk.LEFT, padx=2)

    def _build_visibility_section(self):
        """Build instrument visibility toggles."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Show Instruments", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(frame, text="Fast Antenna",
                       variable=self.show_fa_var).pack(anchor='w')
        ttk.Checkbutton(frame, text="INTF Elevation",
                       variable=self.show_intf_var).pack(anchor='w')
        ttk.Checkbutton(frame, text="Surface Detectors",
                       variable=self.show_sd_var).pack(anchor='w')
        ttk.Checkbutton(frame, text="Luminosity",
                       variable=self.show_lum_var).pack(anchor='w')

        # Photometer channels
        phot_frame = ttk.LabelFrame(frame, text="Photometer Channels", padding=2)
        phot_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(phot_frame, text="337 nm (Blue)",
                       variable=self.show_phot_337_var).pack(anchor='w')
        ttk.Checkbutton(phot_frame, text="391 nm (Purple)",
                       variable=self.show_phot_391_var).pack(anchor='w')
        ttk.Checkbutton(phot_frame, text="777 nm (Red)",
                       variable=self.show_phot_777_var).pack(anchor='w')

    def _build_limits_section(self):
        """Build Y-axis limits section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Y-Axis Limits", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # FA limits
        self._add_limit_row(frame, "FA (V/m):", self.fa_ymin_var, self.fa_ymax_var)

        # SD limits
        self._add_limit_row(frame, "SD (FADC):", self.sd_ymin_var, self.sd_ymax_var)

        # INTF limits
        self._add_limit_row(frame, "INTF (deg):", self.intf_ymin_var, self.intf_ymax_var)

        # Luminosity limits
        self._add_limit_row(frame, "Lum (norm):", self.lum_ymin_var, self.lum_ymax_var)

        # Photometer limits
        self._add_limit_row(frame, "Phot:", self.phot_ymin_var, self.phot_ymax_var)

    def _add_limit_row(self, parent, label, min_var, max_var):
        """Add a Y-axis limit row."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=min_var, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=max_var, width=8).pack(side=tk.LEFT, padx=1)

    def _build_style_section(self):
        """Build plot style section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Plot Style", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(frame, text="Show Grid",
                       variable=self.show_grid_var).pack(anchor='w')
        ttk.Checkbutton(frame, text="Show Legend",
                       variable=self.show_legend_var).pack(anchor='w')

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Title:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.plot_title_var, width=25).pack(side=tk.LEFT, padx=2)

    def _build_action_buttons(self):
        """Build action buttons."""
        frame = ttk.Frame(self.scrollable_frame, padding=5)
        frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(frame, text="🔄 Update Plot",
                  command=self._update_plot).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="📷 Export PNG",
                  command=self._export_png).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="💾 Save to Project",
                  command=self._save_to_project).pack(fill=tk.X, pady=2)

    def _build_plot_area(self):
        """Build the matplotlib plot area."""
        # Create figure
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.ax_main = self.fig.add_subplot(111)

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

    def _browse_file(self, var, filetypes):
        """Browse for a file."""
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            var.set(filepath)

    def _browse_directory(self, var):
        """Browse for a directory."""
        dirpath = filedialog.askdirectory()
        if dirpath:
            var.set(dirpath)

    def _load_all_data(self):
        """Load all specified data files."""
        loaded = []
        errors = []

        # Get timing parameters
        try:
            T0 = float(self.T0_var.get())
        except:
            T0 = 0

        try:
            timeshift = float(self.timeshift_var.get())
        except:
            timeshift = 0

        try:
            phot_offset = int(self.phot_offset_var.get())
        except:
            phot_offset = 0

        # Load Fast Antenna
        fa_path = self.file_paths['fa'].get()
        if fa_path and os.path.exists(fa_path):
            try:
                self.fa_handler.load_csv(fa_path)
                self.fa_handler.set_T0(T0)
                loaded.append("Fast Antenna")
            except Exception as e:
                errors.append(f"FA: {e}")

        # Load INTF
        intf_path = self.file_paths['intf'].get()
        if intf_path and os.path.exists(intf_path):
            try:
                is_calibrated = self.intf_calibrated_var.get()
                if is_calibrated:
                    self.intf_handler.load_data(intf_path, is_calibrated=True, T0_reference=T0)
                else:
                    self.intf_handler.load_raw_intf(intf_path, T0_reference=T0)
                loaded.append("INTF")
            except Exception as e:
                errors.append(f"INTF: {e}")

        # Load SD
        sd_path = self.file_paths['sd_dir'].get()
        if sd_path and os.path.isdir(sd_path):
            try:
                # Try to load TASD GPS coordinates
                gps_file = os.path.join(os.path.dirname(sd_path), 'tasd_gpscoors.txt')
                if os.path.exists(gps_file):
                    self.tasd_handler.load_coordinates(gps_file)

                # Extract time from directory name (e.g., SD_20240817_161845)
                dirname = os.path.basename(sd_path)
                parts = dirname.split('_')
                if len(parts) >= 3:
                    time_str = parts[2]
                else:
                    time_str = ""

                self.tasd_handler.load_directory(sd_path, time_str, time_shift=timeshift)
                loaded.append("Surface Detectors")
            except Exception as e:
                errors.append(f"SD: {e}")

        # Load Photometer
        phot_path = self.file_paths['photometer'].get()
        if phot_path and os.path.exists(phot_path):
            try:
                self.photometer_handler.load_binary_data(phot_path)
                self.photometer_handler.set_second_offset(phot_offset)
                loaded.append("Photometer")
            except Exception as e:
                errors.append(f"Photometer: {e}")

        # Load Luminosity
        lum_path = self.file_paths['luminosity'].get()
        if lum_path and os.path.exists(lum_path):
            try:
                self.luminosity_handler.load_text_file(lum_path)
                loaded.append("Luminosity")
            except Exception as e:
                errors.append(f"Luminosity: {e}")

        # Report results
        msg = ""
        if loaded:
            msg += f"Loaded: {', '.join(loaded)}\n"
        if errors:
            msg += f"Errors: {'; '.join(errors)}"

        if msg:
            self.main_app.status_var.set(msg.replace('\n', ' | '))

        # Update plot
        self._update_plot()

    def _plot_intf_binned(self, ax, t_min, t_max):
        """
        Plot INTF data with binned alpha transparency matching original scripts.

        Points are binned by signal strength (s-ratio) into 3 groups:
        - Low signal (s <= 3): alpha = 0.3
        - Medium signal (3 < s <= 7): alpha = 0.7
        - High signal (s > 7): alpha = 1.0

        All points have black edge colors and are colored by mjet colormap.

        Returns a scatter handle for the legend, or None if no data.
        """
        data = self.intf_handler.get_data_in_range(t_min, t_max)
        if data is None or len(data['time']) == 0:
            return None

        # Get arrays
        time = np.array(data['time'])
        elev = np.array(data['elevation'])
        pkpk = np.array(data['pk2pk'])

        if len(time) == 0:
            return None

        # Signal strength levels and alpha values (matching original)
        sLevelsTup = (1.0, 3., 7., 16.)
        alphaTup = (0.3, 0.7, 1.0)

        # Calculate normalized signal strength for marker size
        pkpk_safe = np.clip(pkpk, 1e-10, None)
        ss = np.log10(pkpk_safe)
        aMin = np.min(ss)
        aMax = np.max(ss)

        if aMax > aMin:
            ss_norm = (ss - aMin) / (aMax - aMin)
        else:
            ss_norm = np.zeros_like(ss)

        ss_norm = np.clip(ss_norm, 0, 1)

        # Calculate s values for binning (same formula as original)
        s = (1 + 3 * ss_norm**2)**2
        markerSz = 6 * s

        # Calculate colors using mjet colormap
        ss_color = np.log10(pkpk_safe)
        ss_color = ss_color / aMax if aMax > 0 else ss_color
        ss_color = np.clip(ss_color, 0, 1)
        colors = cmap_mjet(ss_color)

        # Bin points by signal strength
        bins = [
            {'mask': s <= sLevelsTup[1], 'alpha': alphaTup[0]},
            {'mask': (s > sLevelsTup[1]) & (s <= sLevelsTup[2]), 'alpha': alphaTup[1]},
            {'mask': s > sLevelsTup[2], 'alpha': alphaTup[2]},
        ]

        # Plot each bin with appropriate alpha
        scatter_handle = None
        for i, bin_info in enumerate(bins):
            mask = bin_info['mask']
            if np.any(mask):
                sc = ax.scatter(
                    time[mask],
                    elev[mask],
                    s=markerSz[mask],
                    facecolor=colors[mask],
                    alpha=bin_info['alpha'],
                    edgecolors='k',
                    linewidths=0.5,
                    zorder=1
                )
                # Use highest alpha group for legend handle
                if i == 2:
                    scatter_handle = sc

        # If no high-alpha points, use medium or low for legend
        if scatter_handle is None:
            for i in [1, 0]:
                mask = bins[i]['mask']
                if np.any(mask):
                    scatter_handle = sc
                    break

        return scatter_handle

    def _update_plot(self):
        """Update the plot with current data and settings."""
        # Clear figure
        self.fig.clear()

        # Get plot parameters
        try:
            x_start = float(self.x_start_var.get())
            x_stop = float(self.x_stop_var.get())
        except:
            x_start, x_stop = 0, 100000

        show_zoom = self.show_zoom_var.get()

        # Create axes
        if show_zoom:
            # Two subplots
            self.ax_main = self.fig.add_subplot(211)
            self.ax_zoom = self.fig.add_subplot(212)
            axes_list = [self.ax_main, self.ax_zoom]

            try:
                zoom_start = float(self.zoom_start_var.get())
                zoom_stop = float(self.zoom_stop_var.get())
            except:
                zoom_start, zoom_stop = x_start, x_stop
        else:
            self.ax_main = self.fig.add_subplot(111)
            self.ax_zoom = None
            axes_list = [self.ax_main]
            zoom_start, zoom_stop = x_start, x_stop

        # Track which axes we've created for the legend
        legend_handles = []
        legend_labels = []

        # Plot on each axis
        for ax_idx, ax in enumerate(axes_list):
            if ax_idx == 0:
                t_min, t_max = x_start, x_stop
            else:
                t_min, t_max = zoom_start, zoom_stop

            ax.set_xlim(t_min, t_max)

            # Create twin axes for different y-scales
            ax_twins = []
            twin_count = 0

            # Plot INTF (primary axis) - using binned alpha method
            if self.show_intf_var.get() and self.intf_handler.is_loaded:
                scatter_handle = self._plot_intf_binned(ax, t_min, t_max)
                if scatter_handle is not None and ax_idx == 0:
                    legend_handles.append(scatter_handle)
                    legend_labels.append('INTF')

                ax.set_ylabel("INTF Elevation (deg)", color='red')
                ax.tick_params(axis='y', labelcolor='red')
                try:
                    ax.set_ylim(float(self.intf_ymin_var.get()),
                                float(self.intf_ymax_var.get()))
                except:
                    pass
            else:
                # Hide left y-axis ticks/label when INTF is not shown
                ax.yaxis.set_visible(False)

            # Plot Fast Antenna
            if self.show_fa_var.get() and self.fa_handler.is_loaded:
                data = self.fa_handler.get_data_in_range(t_min, t_max)
                if data and len(data['time']) > 0:
                    ax_fa = ax.twinx()
                    ax_twins.append(ax_fa)
                    line, = ax_fa.plot(data['time'], data['e_field'],
                                      color='green', linewidth=1, label='Fast Antenna')
                    ax_fa.set_ylabel("E-field (V/m)", color='green')
                    ax_fa.tick_params(axis='y', labelcolor='green')
                    try:
                        ax_fa.set_ylim(float(self.fa_ymin_var.get()),
                                      float(self.fa_ymax_var.get()))
                    except:
                        pass

                    if ax_idx == 0:
                        legend_handles.append(line)
                        legend_labels.append('Fast Antenna')
                    twin_count += 1

            # Plot Surface Detectors
            if self.show_sd_var.get() and self.tasd_handler.is_loaded:
                detectors = self.tasd_handler.get_data_in_range(t_min, t_max)
                if detectors:
                    ax_sd = ax.twinx()
                    ax_twins.append(ax_sd)
                    if twin_count > 0:
                        ax_sd.spines['right'].set_position(('axes', 1.0 + 0.1 * twin_count))

                    first = True
                    for det in detectors:
                        if len(det['time']) > 0:
                            line, = ax_sd.plot(det['time'], det['signal_upper'],
                                             color='magenta', linewidth=0.8,
                                             label='SD' if first else None)
                            first = False

                    ax_sd.set_ylabel("FADC Count", color='magenta')
                    ax_sd.tick_params(axis='y', labelcolor='magenta')
                    try:
                        ax_sd.set_ylim(float(self.sd_ymin_var.get()),
                                      float(self.sd_ymax_var.get()))
                    except:
                        pass

                    if ax_idx == 0 and not first:
                        legend_handles.append(line)
                        legend_labels.append('SD Waveforms')
                    twin_count += 1

            # Plot Luminosity
            if self.show_lum_var.get() and self.luminosity_handler.is_loaded:
                data = self.luminosity_handler.get_data_in_range(t_min, t_max)
                if data and len(data['time']) > 0:
                    ax_lum = ax.twinx()
                    ax_twins.append(ax_lum)
                    if twin_count > 0:
                        ax_lum.spines['right'].set_position(('axes', 1.0 + 0.1 * twin_count))

                    line, = ax_lum.plot(data['time'], data['luminosity'],
                                       color='olive', linewidth=1, label='Luminosity')
                    ax_lum.set_ylabel("Luminosity (norm)", color='olive')
                    ax_lum.tick_params(axis='y', labelcolor='olive')
                    try:
                        ax_lum.set_ylim(float(self.lum_ymin_var.get()),
                                       float(self.lum_ymax_var.get()))
                    except:
                        pass

                    if ax_idx == 0:
                        legend_handles.append(line)
                        legend_labels.append('Luminosity')
                    twin_count += 1

            # Plot Photometer channels
            any_phot = (self.show_phot_337_var.get() or
                       self.show_phot_391_var.get() or
                       self.show_phot_777_var.get())

            if any_phot and self.photometer_handler.is_loaded:
                # Downsample for performance
                duration = t_max - t_min
                downsample = max(1, int(duration / 10000))

                data = self.photometer_handler.get_data_in_event_time(t_min, t_max, downsample)
                if data and data['time'] is not None and len(data['time']) > 0:
                    ax_phot = ax.twinx()
                    ax_twins.append(ax_phot)
                    if twin_count > 0:
                        ax_phot.spines['right'].set_position(('axes', 1.0 + 0.1 * twin_count))

                    if self.show_phot_337_var.get() and data['ch0'] is not None:
                        line, = ax_phot.plot(data['time'], data['ch0'],
                                           color='blue', linewidth=0.5, alpha=0.8, label='337nm')
                        if ax_idx == 0:
                            legend_handles.append(line)
                            legend_labels.append('337 nm')

                    if self.show_phot_391_var.get() and data['ch1'] is not None:
                        line, = ax_phot.plot(data['time'], data['ch1'],
                                           color='purple', linewidth=0.5, alpha=0.8, label='391nm')
                        if ax_idx == 0:
                            legend_handles.append(line)
                            legend_labels.append('391 nm')

                    if self.show_phot_777_var.get() and data['ch2'] is not None:
                        line, = ax_phot.plot(data['time'], data['ch2'],
                                           color='red', linewidth=0.5, alpha=0.8, label='777nm')
                        if ax_idx == 0:
                            legend_handles.append(line)
                            legend_labels.append('777 nm')

                    ax_phot.set_ylabel("Irradiance", color='darkred')
                    ax_phot.tick_params(axis='y', labelcolor='darkred')
                    try:
                        ax_phot.set_ylim(float(self.phot_ymin_var.get()),
                                        float(self.phot_ymax_var.get()))
                    except:
                        pass

            # Grid and labels
            if self.show_grid_var.get():
                ax.grid(True, linestyle=':', alpha=0.7)

            ax.set_xlabel("Time (µs)")

            # Add zoom box indicator on main plot
            if ax_idx == 0 and show_zoom:
                ax.axvspan(zoom_start, zoom_stop, alpha=0.1, color='blue')

        # Title
        title = self.plot_title_var.get()
        if title:
            self.ax_main.set_title(title, fontsize=14)

        # Legend
        if self.show_legend_var.get() and legend_handles:
            self.ax_main.legend(legend_handles, legend_labels,
                               loc='upper right', fontsize=9)

        # Adjust layout
        self.fig.tight_layout()

        # Redraw
        self.canvas.draw()

    def _on_mouse_move(self, event):
        """Handle mouse movement for coordinate display."""
        if event.inaxes:
            self.coord_var.set(f"x={event.xdata:.1f} µs, y={event.ydata:.3f}")
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

    def _save_to_project(self):
        """Save current settings to project state."""
        state = self.main_app.project_state

        # Save file paths (absolute paths)
        state.files['fa'] = self.file_paths['fa'].get() or None
        state.files['intf_calibrated'] = self.file_paths['intf'].get() or None
        state.files['sd_directory'] = self.file_paths['sd_dir'].get() or None
        state.files['photometer'] = self.file_paths['photometer'].get() or None
        state.files['luminosity'] = self.file_paths['luminosity'].get() or None

        # Save timing
        try:
            state.timing['T0'] = float(self.T0_var.get())
        except:
            pass
        try:
            state.timing['timeshift'] = float(self.timeshift_var.get())
        except:
            pass
        try:
            state.timing['photometer_second_offset'] = int(self.phot_offset_var.get())
        except:
            pass

        # Save plot ranges
        try:
            state.plot_ranges['main']['x_start'] = float(self.x_start_var.get())
            state.plot_ranges['main']['x_stop'] = float(self.x_stop_var.get())
        except:
            pass
        try:
            state.plot_ranges['zoom']['x_start'] = float(self.zoom_start_var.get())
            state.plot_ranges['zoom']['x_stop'] = float(self.zoom_stop_var.get())
        except:
            pass

        # Save limits
        try:
            state.plot_ranges['fa']['y_min'] = float(self.fa_ymin_var.get())
            state.plot_ranges['fa']['y_max'] = float(self.fa_ymax_var.get())
        except:
            pass
        try:
            state.plot_ranges['sd']['y_min'] = float(self.sd_ymin_var.get())
            state.plot_ranges['sd']['y_max'] = float(self.sd_ymax_var.get())
        except:
            pass
        try:
            state.plot_ranges['intf_elev']['y_min'] = float(self.intf_ymin_var.get())
            state.plot_ranges['intf_elev']['y_max'] = float(self.intf_ymax_var.get())
        except:
            pass

        # Save visibility
        state.visibility['fa'] = self.show_fa_var.get()
        state.visibility['intf'] = self.show_intf_var.get()
        state.visibility['sd'] = self.show_sd_var.get()
        state.visibility['luminosity'] = self.show_lum_var.get()
        state.visibility['photometer_337'] = self.show_phot_337_var.get()
        state.visibility['photometer_391'] = self.show_phot_391_var.get()
        state.visibility['photometer_777'] = self.show_phot_777_var.get()

        # Save INTF settings
        state.intf['is_calibrated'] = self.intf_calibrated_var.get()

        # Save plot style
        state.plot_style['show_grid'] = self.show_grid_var.get()
        state.plot_style['show_legend'] = self.show_legend_var.get()
        state.plot_style['title'] = self.plot_title_var.get()

        # Mark as unsaved
        self.main_app.mark_unsaved()
        self.main_app.status_var.set("Settings saved to project")

    def load_from_project(self):
        """Load settings from project state."""
        state = self.main_app.project_state

        # Load file paths
        if state.files.get('fa'):
            self.file_paths['fa'].set(state.files['fa'])
        if state.files.get('intf_calibrated'):
            self.file_paths['intf'].set(state.files['intf_calibrated'])
        if state.files.get('sd_directory'):
            self.file_paths['sd_dir'].set(state.files['sd_directory'])
        if state.files.get('photometer'):
            self.file_paths['photometer'].set(state.files['photometer'])
        if state.files.get('luminosity'):
            self.file_paths['luminosity'].set(state.files['luminosity'])

        # Load timing
        self.T0_var.set(str(state.timing.get('T0', 0)))
        self.timeshift_var.set(str(state.timing.get('timeshift', 0)))
        self.phot_offset_var.set(str(state.timing.get('photometer_second_offset', 0)))

        # Load plot ranges
        self.x_start_var.set(str(state.plot_ranges['main'].get('x_start', 0)))
        self.x_stop_var.set(str(state.plot_ranges['main'].get('x_stop', 100000)))
        self.zoom_start_var.set(str(state.plot_ranges['zoom'].get('x_start', 0)))
        self.zoom_stop_var.set(str(state.plot_ranges['zoom'].get('x_stop', 1000)))

        # Load limits
        self.fa_ymin_var.set(str(state.plot_ranges['fa'].get('y_min', -80)))
        self.fa_ymax_var.set(str(state.plot_ranges['fa'].get('y_max', 30)))
        self.sd_ymin_var.set(str(state.plot_ranges['sd'].get('y_min', 0)))
        self.sd_ymax_var.set(str(state.plot_ranges['sd'].get('y_max', 500)))
        self.intf_ymin_var.set(str(state.plot_ranges['intf_elev'].get('y_min', 0)))
        self.intf_ymax_var.set(str(state.plot_ranges['intf_elev'].get('y_max', 40)))

        # Load visibility
        self.show_fa_var.set(state.visibility.get('fa', True))
        self.show_intf_var.set(state.visibility.get('intf', True))
        self.show_sd_var.set(state.visibility.get('sd', True))
        self.show_lum_var.set(state.visibility.get('luminosity', True))
        self.show_phot_337_var.set(state.visibility.get('photometer_337', True))
        self.show_phot_391_var.set(state.visibility.get('photometer_391', True))
        self.show_phot_777_var.set(state.visibility.get('photometer_777', True))

        # Load INTF settings
        self.intf_calibrated_var.set(state.intf.get('is_calibrated', True))

        # Load plot style
        self.show_grid_var.set(state.plot_style.get('show_grid', True))
        self.show_legend_var.set(state.plot_style.get('show_legend', True))
        self.plot_title_var.set(state.plot_style.get('title', ''))