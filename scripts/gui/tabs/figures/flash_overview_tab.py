"""
Flash Overview Tab — The main plotter with camera frame strip below.

Replicates HomePlotterTab visualization exactly (same INTF binned scatter,
same multi-axis overlay) with the addition of v711 camera frames connected
to the time axis via ConnectionPatch.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

from config import (
    INSTRUMENT_COLORS, DEFAULT_FONT_SIZE, DEFAULT_LIMITS,
    cmap_mjet, PHOTOMETER_CALIBRATION
)
from data_handlers import (
    PhotometerHandler, FastAntennaHandler, InterferometerHandler,
    TASDHandler, LuminosityHandler
)


class FlashOverviewTab(ttk.Frame):
    """
    Flash Overview — main plotter + v711 camera frame strip.

    Identical visualization to HomePlotterTab with an additional
    camera frame panel below the main plot.
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

        # File paths
        self.file_paths = {
            'fa': tk.StringVar(),
            'intf': tk.StringVar(),
            'sd_dir': tk.StringVar(),
            'photometer': tk.StringVar(),
            'luminosity': tk.StringVar(),
        }

        # Timing
        self.T0_var = tk.StringVar(value="0")
        self.timeshift_var = tk.StringVar(value="0")
        self.phot_offset_var = tk.StringVar(value="0")
        self.event_time_var = tk.StringVar(value="")

        # Plot range
        self.x_start_var = tk.StringVar(value="0")
        self.x_stop_var = tk.StringVar(value="100000")

        # Visibility toggles
        self.show_fa_var = tk.BooleanVar(value=True)
        self.show_intf_var = tk.BooleanVar(value=True)
        self.show_sd_var = tk.BooleanVar(value=True)
        self.show_lum_var = tk.BooleanVar(value=True)
        self.show_phot_337_var = tk.BooleanVar(value=True)
        self.show_phot_391_var = tk.BooleanVar(value=True)
        self.show_phot_777_var = tk.BooleanVar(value=True)

        # Y-axis limits
        self.fa_ymin_var = tk.StringVar(value=str(DEFAULT_LIMITS['fa']['y_min']))
        self.fa_ymax_var = tk.StringVar(value=str(DEFAULT_LIMITS['fa']['y_max']))
        self.sd_ymin_var = tk.StringVar(value=str(DEFAULT_LIMITS['sd']['y_min']))
        self.sd_ymax_var = tk.StringVar(value=str(DEFAULT_LIMITS['sd']['y_max']))
        self.intf_ymin_var = tk.StringVar(value=str(DEFAULT_LIMITS['intf_elev']['y_min']))
        self.intf_ymax_var = tk.StringVar(value=str(DEFAULT_LIMITS['intf_elev']['y_max']))
        self.lum_ymin_var = tk.StringVar(value=str(DEFAULT_LIMITS['luminosity']['y_min']))
        self.lum_ymax_var = tk.StringVar(value=str(DEFAULT_LIMITS['luminosity']['y_max']))
        self.phot_ymin_var = tk.StringVar(value=str(DEFAULT_LIMITS['photometer']['y_min']))
        self.phot_ymax_var = tk.StringVar(value=str(DEFAULT_LIMITS['photometer']['y_max']))

        # Style
        self.intf_calibrated_var = tk.BooleanVar(value=True)
        self.show_grid_var = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=True)
        self.plot_title_var = tk.StringVar(value="")

        # Camera frames
        self.frame_dir_var = tk.StringVar()
        self.frame_rows = []

        self._build_ui()

    # =====================================================================
    # UI Construction
    # =====================================================================

    def _build_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        ctrl = ttk.Frame(paned, width=670)
        ctrl.pack_propagate(False)
        paned.add(ctrl, weight=0)

        self.plot_frame = ttk.Frame(paned)
        paned.add(self.plot_frame, weight=1)

        # Scrollable left panel
        canvas = tk.Canvas(ctrl, width=670)
        scrollbar = ttk.Scrollbar(ctrl, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_file_section()
        self._build_timing_section()
        self._build_range_section()
        self._build_visibility_section()
        self._build_limits_section()
        self._build_style_section()
        self._build_camera_frames_section()
        self._build_action_buttons()
        self._build_plot_area()

    def _build_file_section(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Data Files", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        self._add_file_row(frame, "Fast Antenna:", self.file_paths['fa'],
                           [("CSV files", "*.csv"), ("All files", "*.*")])
        self._add_file_row(frame, "INTF:", self.file_paths['intf'],
                           [("DAT files", "*.dat"), ("All files", "*.*")])

        ttk.Checkbutton(frame, text="INTF is pre-calibrated",
                        variable=self.intf_calibrated_var).pack(anchor='w', padx=20)

        # SD Directory
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="SD Directory:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.file_paths['sd_dir'], width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3,
                   command=lambda: self._browse_directory(self.file_paths['sd_dir'])).pack(side=tk.LEFT)

        self._add_file_row(frame, "Photometer:", self.file_paths['photometer'],
                           [("DAT files", "*.dat"), ("All files", "*.*")])
        self._add_file_row(frame, "Luminosity:", self.file_paths['luminosity'],
                           [("Text files", "*.txt"), ("DAT files", "*.dat"), ("All files", "*.*")])

        ttk.Button(frame, text="Load All Data", command=self._load_all_data).pack(pady=5)
        ttk.Button(frame, text="Load from Project",
                   command=self._load_files_from_project).pack(pady=2)

    def _add_file_row(self, parent, label, var, filetypes):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3,
                   command=lambda: self._browse_file(var, filetypes)).pack(side=tk.LEFT)

    def _build_timing_section(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Timing Parameters", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for label, var in [("Event Time (HH:MM:SS):", self.event_time_var),
                           ("T0 (us):", self.T0_var),
                           ("SD Timeshift (us):", self.timeshift_var),
                           ("Phot. Sec Offset:", self.phot_offset_var)]:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT)

    def _build_range_section(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Plot Range (us)", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="X Start:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.x_start_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="X Stop:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.x_stop_var, width=10).pack(side=tk.LEFT, padx=2)

    def _build_visibility_section(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Show Instruments", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(frame, text="Fast Antenna", variable=self.show_fa_var).pack(anchor='w')
        ttk.Checkbutton(frame, text="INTF Elevation", variable=self.show_intf_var).pack(anchor='w')
        ttk.Checkbutton(frame, text="Surface Detectors", variable=self.show_sd_var).pack(anchor='w')
        ttk.Checkbutton(frame, text="Luminosity", variable=self.show_lum_var).pack(anchor='w')

        phot_frame = ttk.LabelFrame(frame, text="Photometer Channels", padding=2)
        phot_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(phot_frame, text="337 nm (Blue)", variable=self.show_phot_337_var).pack(anchor='w')
        ttk.Checkbutton(phot_frame, text="391 nm (Purple)", variable=self.show_phot_391_var).pack(anchor='w')
        ttk.Checkbutton(phot_frame, text="777 nm (Red)", variable=self.show_phot_777_var).pack(anchor='w')

    def _build_limits_section(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Y-Axis Limits", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for label, min_var, max_var in [
            ("FA (V/m):", self.fa_ymin_var, self.fa_ymax_var),
            ("SD (FADC):", self.sd_ymin_var, self.sd_ymax_var),
            ("INTF (deg):", self.intf_ymin_var, self.intf_ymax_var),
            ("Lum (norm):", self.lum_ymin_var, self.lum_ymax_var),
            ("Phot:", self.phot_ymin_var, self.phot_ymax_var),
        ]:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=min_var, width=8).pack(side=tk.LEFT, padx=1)
            ttk.Label(row, text="-").pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=max_var, width=8).pack(side=tk.LEFT, padx=1)

    def _build_style_section(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Plot Style", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(frame, text="Show Grid", variable=self.show_grid_var).pack(anchor='w')
        ttk.Checkbutton(frame, text="Show Legend", variable=self.show_legend_var).pack(anchor='w')

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Title:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.plot_title_var, width=25).pack(side=tk.LEFT, padx=2)

    def _build_camera_frames_section(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Camera Frames (v711)", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Directory:", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.frame_dir_var, width=22).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3,
                   command=lambda: self._browse_directory(self.frame_dir_var)).pack(side=tk.LEFT)

        self.frame_list_frame = ttk.Frame(frame)
        self.frame_list_frame.pack(fill=tk.X, pady=2)

        ttk.Button(frame, text="Add Frame", command=self._browse_and_add_frame).pack(pady=2)

    def _browse_and_add_frame(self):
        """Open file browser in v711 directory, then add a row for the selected image."""
        initial_dir = self.frame_dir_var.get() or None
        filepath = filedialog.askopenfilename(
            title="Select Camera Frame",
            initialdir=initial_dir,
            filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                       ("All files", "*.*")])
        if filepath:
            filename = os.path.basename(filepath)
            # Auto-set the directory if not already set
            if not self.frame_dir_var.get():
                self.frame_dir_var.set(os.path.dirname(filepath))
            self._add_frame_row(filename=filename, timestamp="")

    def _add_frame_row(self, filename="", timestamp=""):
        row_frame = ttk.Frame(self.frame_list_frame)
        row_frame.pack(fill=tk.X, pady=1)

        fn_var = tk.StringVar(value=filename)
        ts_var = tk.StringVar(value=timestamp)

        ttk.Label(row_frame, textvariable=fn_var, width=18, anchor='w').pack(side=tk.LEFT, padx=1)
        ttk.Label(row_frame, text="us:").pack(side=tk.LEFT)
        ttk.Entry(row_frame, textvariable=ts_var, width=10).pack(side=tk.LEFT, padx=1)

        idx = len(self.frame_rows)
        ttk.Button(row_frame, text="X", width=2,
                   command=lambda: self._remove_frame_row(idx, row_frame)).pack(side=tk.LEFT, padx=2)

        self.frame_rows.append((fn_var, ts_var, row_frame))

    def _remove_frame_row(self, idx, widget):
        widget.destroy()
        if idx < len(self.frame_rows):
            self.frame_rows[idx] = None
        self.frame_rows = [r for r in self.frame_rows if r is not None]

    def _build_action_buttons(self):
        frame = ttk.Frame(self.scrollable_frame, padding=5)
        frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(frame, text="Update Plot", command=self._update_plot).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Export Figure", command=self._export).pack(fill=tk.X, pady=2)

    def _build_plot_area(self):
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax_main = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()

        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.coord_var = tk.StringVar(value="")
        ttk.Label(self.plot_frame, textvariable=self.coord_var).pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    # =====================================================================
    # Helpers
    # =====================================================================

    def _browse_file(self, var, filetypes):
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            var.set(filepath)

    def _browse_directory(self, var):
        dirpath = filedialog.askdirectory()
        if dirpath:
            var.set(dirpath)

    def _load_files_from_project(self):
        state = self.main_app.project_state
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
        self.T0_var.set(str(state.timing.get('T0', 0)))
        self.timeshift_var.set(str(state.timing.get('timeshift', 0)))
        self.phot_offset_var.set(str(state.timing.get('photometer_second_offset', 0)))
        self.main_app.status_var.set("Flash Overview: loaded from project")

    def _get_frame_list(self):
        frames = []
        for row in self.frame_rows:
            if row is None:
                continue
            fn = row[0].get().strip()
            ts = row[1].get().strip()
            if fn and ts:
                try:
                    frames.append((fn, float(ts)))
                except ValueError:
                    pass
        return frames

    def _on_mouse_move(self, event):
        if event.inaxes:
            self.coord_var.set(f"x={event.xdata:.1f} us, y={event.ydata:.3f}")
        else:
            self.coord_var.set("")

    # =====================================================================
    # Data Loading (same as HomePlotterTab)
    # =====================================================================

    def _load_all_data(self):
        loaded = []
        errors = []

        try:
            T0 = float(self.T0_var.get())
        except Exception:
            T0 = 0
        try:
            timeshift = float(self.timeshift_var.get())
        except Exception:
            timeshift = 0
        try:
            phot_offset = int(self.phot_offset_var.get())
        except Exception:
            phot_offset = 0

        # FA
        fa_path = self.file_paths['fa'].get()
        if fa_path and os.path.exists(fa_path):
            try:
                self.fa_handler.load_csv(fa_path)
                self.fa_handler.set_T0(T0)
                loaded.append("Fast Antenna")
            except Exception as e:
                errors.append(f"FA: {e}")

        # INTF
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

        # SD
        sd_path = self.file_paths['sd_dir'].get()
        if sd_path and os.path.isdir(sd_path):
            try:
                gps_file = os.path.join(os.path.dirname(sd_path), 'tasd_gpscoors.txt')
                if os.path.exists(gps_file):
                    self.tasd_handler.load_coordinates(gps_file)
                dirname = os.path.basename(sd_path)
                parts = dirname.split('_')
                time_str = parts[2] if len(parts) >= 3 else ""
                self.tasd_handler.load_directory(sd_path, time_str, time_shift=timeshift)
                loaded.append("Surface Detectors")
            except Exception as e:
                errors.append(f"SD: {e}")

        # Photometer
        phot_path = self.file_paths['photometer'].get()
        if phot_path and os.path.exists(phot_path):
            try:
                self.photometer_handler.load_binary_data(phot_path)
                self.photometer_handler.set_second_offset(phot_offset)
                loaded.append("Photometer")
            except Exception as e:
                errors.append(f"Photometer: {e}")

        # Luminosity
        lum_path = self.file_paths['luminosity'].get()
        if lum_path and os.path.exists(lum_path):
            try:
                self.luminosity_handler.load_text_file(lum_path)
                loaded.append("Luminosity")
            except Exception as e:
                errors.append(f"Luminosity: {e}")

        msg = ""
        if loaded:
            msg += f"Loaded: {', '.join(loaded)}"
        if errors:
            msg += f" | Errors: {'; '.join(errors)}"
        if msg:
            self.main_app.status_var.set(msg)

        self._update_plot()

    # =====================================================================
    # INTF Binned Scatter (identical to HomePlotterTab._plot_intf_binned)
    # =====================================================================

    def _plot_intf_binned(self, ax, t_min, t_max):
        """
        Plot INTF data with binned alpha transparency matching original scripts.

        Points are binned by signal strength (s-ratio) into 3 groups:
        - Low signal (s <= 3): alpha = 0.3
        - Medium signal (3 < s <= 7): alpha = 0.7
        - High signal (s > 7): alpha = 1.0

        All points have black edge colors and are colored by mjet colormap.
        """
        data = self.intf_handler.get_data_in_range(t_min, t_max)
        if data is None or len(data['time']) == 0:
            return None

        time = np.array(data['time'])
        elev = np.array(data['elevation'])
        pkpk = np.array(data['pk2pk'])

        if len(time) == 0:
            return None

        sLevelsTup = (1.0, 3., 7., 16.)
        alphaTup = (0.3, 0.7, 1.0)

        pkpk_safe = np.clip(pkpk, 1e-10, None)
        ss = np.log10(pkpk_safe)
        aMin = np.min(ss)
        aMax = np.max(ss)

        if aMax > aMin:
            ss_norm = (ss - aMin) / (aMax - aMin)
        else:
            ss_norm = np.zeros_like(ss)

        ss_norm = np.clip(ss_norm, 0, 1)

        s = (1 + 3 * ss_norm**2)**2
        markerSz = 6 * s

        ss_color = np.log10(pkpk_safe)
        ss_color = ss_color / aMax if aMax > 0 else ss_color
        ss_color = np.clip(ss_color, 0, 1)
        colors = cmap_mjet(ss_color)

        bins = [
            {'mask': s <= sLevelsTup[1], 'alpha': alphaTup[0]},
            {'mask': (s > sLevelsTup[1]) & (s <= sLevelsTup[2]), 'alpha': alphaTup[1]},
            {'mask': s > sLevelsTup[2], 'alpha': alphaTup[2]},
        ]

        scatter_handle = None
        for i, bin_info in enumerate(bins):
            mask = bin_info['mask']
            if np.any(mask):
                sc = ax.scatter(
                    time[mask], elev[mask],
                    s=markerSz[mask],
                    facecolor=colors[mask],
                    alpha=bin_info['alpha'],
                    edgecolors='k',
                    linewidths=0.5,
                    zorder=1
                )
                if i == 2:
                    scatter_handle = sc

        if scatter_handle is None:
            for i in [1, 0]:
                mask = bins[i]['mask']
                if np.any(mask):
                    scatter_handle = sc
                    break

        return scatter_handle

    # =====================================================================
    # Plot (mirrors HomePlotterTab._update_plot + camera frame strip)
    # =====================================================================

    def _update_plot(self):
        # Re-apply photometer second offset from UI before plotting
        if self.photometer_handler.is_loaded:
            try:
                phot_offset = int(self.phot_offset_var.get())
            except (ValueError, tk.TclError):
                phot_offset = 0
            self.photometer_handler.set_second_offset(phot_offset)

        self.fig.clear()

        try:
            x_start = float(self.x_start_var.get())
            x_stop = float(self.x_stop_var.get())
        except Exception:
            x_start, x_stop = 0, 100000

        t_min, t_max = x_start, x_stop

        frame_list = self._get_frame_list()
        has_frames = len(frame_list) > 0

        if has_frames:
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            self.ax_main = self.fig.add_subplot(gs[0])
            ax_frames = self.fig.add_subplot(gs[1])
        else:
            self.ax_main = self.fig.add_subplot(111)
            ax_frames = None

        ax = self.ax_main
        ax.set_xlim(t_min, t_max)

        legend_handles = []
        legend_labels = []
        rhs_twin_count = 0  # tracks right-side twinx offset

        # --- Photometer (PRIMARY left y-axis) ---
        any_phot = (self.show_phot_337_var.get() or
                    self.show_phot_391_var.get() or
                    self.show_phot_777_var.get())

        if any_phot and self.photometer_handler.is_loaded:
            duration = t_max - t_min
            downsample = max(1, int(duration / 10000))

            data = self.photometer_handler.get_data_in_event_time(t_min, t_max, downsample)
            if data and data['time'] is not None and len(data['time']) > 0:
                if self.show_phot_337_var.get() and data['ch0'] is not None:
                    line, = ax.plot(data['time'], data['ch0'],
                                    color='blue', linewidth=0.5, alpha=0.8, label='337nm')
                    legend_handles.append(line)
                    legend_labels.append('337 nm')

                if self.show_phot_391_var.get() and data['ch1'] is not None:
                    line, = ax.plot(data['time'], data['ch1'],
                                    color='purple', linewidth=0.5, alpha=0.8, label='391nm')
                    legend_handles.append(line)
                    legend_labels.append('391 nm')

                if self.show_phot_777_var.get() and data['ch2'] is not None:
                    line, = ax.plot(data['time'], data['ch2'],
                                    color='red', linewidth=0.5, alpha=0.8, label='777nm')
                    legend_handles.append(line)
                    legend_labels.append('777 nm')

                ax.set_ylabel("Irradiance (\u00b5W/m\u00b2)", color='black', fontsize=12)
                ax.tick_params(axis='y', labelcolor='black')
                try:
                    ax.set_ylim(float(self.phot_ymin_var.get()),
                                float(self.phot_ymax_var.get()))
                except Exception:
                    pass
        else:
            ax.yaxis.set_visible(False)

        # --- INTF (first twinx, right side) ---
        if self.show_intf_var.get() and self.intf_handler.is_loaded:
            ax_intf = ax.twinx()
            scatter_handle = self._plot_intf_binned(ax_intf, t_min, t_max)
            if scatter_handle is not None:
                legend_handles.append(scatter_handle)
                legend_labels.append('INTF')

            ax_intf.set_ylabel("INTF Elevation (deg)", color='red')
            ax_intf.tick_params(axis='y', labelcolor='red')
            try:
                ax_intf.set_ylim(float(self.intf_ymin_var.get()),
                                 float(self.intf_ymax_var.get()))
            except Exception:
                pass
            rhs_twin_count += 1

        # --- Fast Antenna (twinx) ---
        if self.show_fa_var.get() and self.fa_handler.is_loaded:
            data = self.fa_handler.get_data_in_range(t_min, t_max)
            if data and len(data['time']) > 0:
                ax_fa = ax.twinx()
                if rhs_twin_count > 0:
                    ax_fa.spines['right'].set_position(('axes', 1.0 + 0.08 * rhs_twin_count))

                line, = ax_fa.plot(data['time'], data['e_field'],
                                   color='green', linewidth=1, label='Fast Antenna')
                ax_fa.set_ylabel("E-field (V/m)", color='green')
                ax_fa.tick_params(axis='y', labelcolor='green')
                try:
                    ax_fa.set_ylim(float(self.fa_ymin_var.get()),
                                   float(self.fa_ymax_var.get()))
                except Exception:
                    pass
                legend_handles.append(line)
                legend_labels.append('Fast Antenna')
                rhs_twin_count += 1

        # --- Surface Detectors (twinx) ---
        if self.show_sd_var.get() and self.tasd_handler.is_loaded:
            detectors = self.tasd_handler.get_data_in_range(t_min, t_max)
            if detectors:
                ax_sd = ax.twinx()
                if rhs_twin_count > 0:
                    ax_sd.spines['right'].set_position(('axes', 1.0 + 0.08 * rhs_twin_count))

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
                except Exception:
                    pass

                if not first:
                    legend_handles.append(line)
                    legend_labels.append('SD Waveforms')
                rhs_twin_count += 1

        # --- Luminosity (twinx) ---
        if self.show_lum_var.get() and self.luminosity_handler.is_loaded:
            data = self.luminosity_handler.get_data_in_range(t_min, t_max)
            if data and len(data['time']) > 0:
                ax_lum = ax.twinx()
                if rhs_twin_count > 0:
                    ax_lum.spines['right'].set_position(('axes', 1.0 + 0.08 * rhs_twin_count))

                line, = ax_lum.plot(data['time'], data['luminosity'],
                                    color='olive', linewidth=1, label='Luminosity')
                ax_lum.set_ylabel("Luminosity (norm)", color='olive')
                ax_lum.tick_params(axis='y', labelcolor='olive')
                try:
                    ax_lum.set_ylim(float(self.lum_ymin_var.get()),
                                    float(self.lum_ymax_var.get()))
                except Exception:
                    pass

                legend_handles.append(line)
                legend_labels.append('Luminosity')
                rhs_twin_count += 1

        # Grid and labels
        if self.show_grid_var.get():
            ax.grid(True, linestyle=':', alpha=0.7)

        event_time = self.event_time_var.get().strip()
        xlabel_text = f"Time after {event_time} (\u00b5s)" if event_time else "Time (\u00b5s)"
        ax.set_xlabel(xlabel_text, fontsize=12)

        title = self.plot_title_var.get()
        if title:
            ax.set_title(title, fontsize=14)

        if self.show_legend_var.get() and legend_handles:
            ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=9)

        # Layout — leave right margin for twinx spines
        # Apply BEFORE drawing frame strip so pixel extents are stable
        right_margin = max(0.72, 0.92 - 0.06 * max(rhs_twin_count - 1, 0))
        if has_frames:
            self.fig.subplots_adjust(left=0.08, right=right_margin, top=0.95,
                                     bottom=0.05)
        else:
            self.fig.subplots_adjust(left=0.08, right=right_margin, top=0.95,
                                     bottom=0.08, hspace=0.2)

        # --- Camera frame strip ---
        if has_frames and ax_frames is not None:
            self._draw_frame_strip(ax, ax_frames, frame_list, t_min, t_max)

        self.canvas.draw()

    def _draw_frame_strip(self, ax_top, ax_bot, frame_list, t_min, t_max):
        """Draw camera frame images below the main plot, matching original script style."""
        ax_bot.set_xlim(ax_top.get_xlim())
        ax_bot.set_ylim(0, 1)
        ax_bot.set_yticks([])
        ax_bot.set_xticks([])
        for spine in ax_bot.spines.values():
            spine.set_visible(False)
            ax_bot.patch.set_visible(False)

        n = len(frame_list)
        if n == 0:
            return

        frame_dir = self.frame_dir_var.get()
        # Evenly space images across time range (matching original script)
        x_positions = np.linspace(t_min, t_max, n)

        # Get the main plot y-limits for connection line endpoint
        y_bot = ax_top.get_ylim()[0]

        # Layout: images centered at y=0.40, time labels just below at y=0.12
        # Connection lines from top of frame strip (y=0.95) to main plot bottom
        img_y = 0.40
        label_y = 0.10

        # First pass: place images and time labels
        placed = []  # (i, ts, ab)
        for i, (fn, ts) in enumerate(frame_list):
            img_path = os.path.join(frame_dir, fn) if frame_dir else fn
            if not os.path.exists(img_path):
                print(f"Warning: frame not found: {img_path}")
                continue
            try:
                img = mpimg.imread(img_path)

                # Crop timestamp bar if present (Phantom TIF: 1280x448 + bar)
                if img.ndim >= 2 and img.shape[0] > 448:
                    img = img[:448, :]

                imagebox = OffsetImage(img, zoom=0.30)
                ab = AnnotationBbox(imagebox, (x_positions[i], img_y),
                                    xycoords='data',
                                    frameon=False,
                                    box_alignment=(0.5, 0.5),
                                    zorder=5)
                ax_bot.add_artist(ab)
                placed.append((i, ts, ab))

                # Time label just below image
                ax_bot.text(x_positions[i], label_y, f'{int(ts)} \u00b5s',
                            ha='center', va='top', fontsize=10, zorder=5)

            except Exception as e:
                print(f"Warning: could not load frame {fn}: {e}")

        # Render so image bounding boxes are computed
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()

        # Second pass: draw lines from main plot bottom to the top edge of each image
        for i, ts, ab in placed:
            try:
                bbox = ab.get_window_extent(renderer)
                # Top-center of the image in display pixels
                top_display = (bbox.x0 + bbox.width / 2, bbox.y1 - 12)
                # Convert to ax_bot data coordinates
                top_data = ax_bot.transData.inverted().transform(top_display)

                con = ConnectionPatch(
                    xyA=(x_positions[i], top_data[1]), coordsA=ax_bot.transData,
                    xyB=(ts, y_bot), coordsB=ax_top.transData,
                    arrowstyle='-', linestyle='-', color='black',
                    linewidth=0.8, clip_on=False, zorder=1)
                ax_bot.add_artist(con)
            except Exception as e:
                print(f"Warning: could not draw line for frame {i}: {e}")

    # =====================================================================
    # Export
    # =====================================================================

    def _export(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All", "*.*")])
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches='tight')
            self.main_app.status_var.set(f"Exported: {os.path.basename(path)}")

    # =====================================================================
    # Project Save / Load
    # =====================================================================

    def _save_to_project(self):
        state = self.main_app.project_state
        fo = state.flash_overview

        try:
            fo['time_min'] = float(self.x_start_var.get())
        except Exception:
            fo['time_min'] = None
        try:
            fo['time_max'] = float(self.x_stop_var.get())
        except Exception:
            fo['time_max'] = None

        # Event time for xlabel
        fo['event_time'] = self.event_time_var.get().strip() or None

        # v711 directory
        fo['v711_directory'] = self.frame_dir_var.get() or None

        # Camera frames
        frames = []
        for row in self.frame_rows:
            if row is None:
                continue
            fn = row[0].get().strip()
            ts = row[1].get().strip()
            if fn and ts:
                try:
                    frames.append([fn, float(ts)])
                except ValueError:
                    pass
        fo['camera_frames'] = frames

    def load_from_project(self):
        state = self.main_app.project_state
        fo = state.flash_overview

        if fo.get('time_min') is not None:
            self.x_start_var.set(str(fo['time_min']))
        if fo.get('time_max') is not None:
            self.x_stop_var.set(str(fo['time_max']))

        # Event time — saved per-tab, fallback to project event_info.time
        evt = fo.get('event_time') or state.event_info.get('time', '')
        if evt:
            self.event_time_var.set(evt)

        if fo.get('v711_directory'):
            self.frame_dir_var.set(fo['v711_directory'])

        for fn, ts in fo.get('camera_frames', []):
            self._add_frame_row(filename=fn, timestamp=str(ts))

        # Load shared project state
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

        self.T0_var.set(str(state.timing.get('T0', 0)))
        self.timeshift_var.set(str(state.timing.get('timeshift', 0)))
        self.phot_offset_var.set(str(state.timing.get('photometer_second_offset', 0)))

        # Load visibility and limits from main plotter state
        self.show_fa_var.set(state.visibility.get('fa', True))
        self.show_intf_var.set(state.visibility.get('intf', True))
        self.show_sd_var.set(state.visibility.get('sd', True))
        self.show_lum_var.set(state.visibility.get('luminosity', True))
        self.show_phot_337_var.set(state.visibility.get('photometer_337', True))
        self.show_phot_391_var.set(state.visibility.get('photometer_391', True))
        self.show_phot_777_var.set(state.visibility.get('photometer_777', True))

        self.intf_calibrated_var.set(state.intf.get('is_calibrated', True))

        self.fa_ymin_var.set(str(state.plot_ranges['fa'].get('y_min', -80)))
        self.fa_ymax_var.set(str(state.plot_ranges['fa'].get('y_max', 30)))
        self.sd_ymin_var.set(str(state.plot_ranges['sd'].get('y_min', 0)))
        self.sd_ymax_var.set(str(state.plot_ranges['sd'].get('y_max', 500)))
        self.intf_ymin_var.set(str(state.plot_ranges['intf_elev'].get('y_min', 0)))
        self.intf_ymax_var.set(str(state.plot_ranges['intf_elev'].get('y_max', 40)))

        self.show_grid_var.set(state.plot_style.get('show_grid', True))
        self.show_legend_var.set(state.plot_style.get('show_legend', True))
        self.plot_title_var.set(state.plot_style.get('title', ''))
