"""
Figure 1 Tab — Multi-instrument overview with photometer ratio subplot.

Top panel: same multi-instrument overlay as HomePlotterTab / FlashOverviewTab.
Bottom panel: photometer irradiance ratios (337/777, 391/777, 337/391).
"""

import os
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
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


class Figure1Tab(ttk.Frame):
    """
    Figure 1 — multi-instrument overlay + photometer ratio subplot.
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

        # Visibility toggles — instruments
        self.show_fa_var = tk.BooleanVar(value=True)
        self.show_intf_var = tk.BooleanVar(value=True)
        self.show_sd_var = tk.BooleanVar(value=True)
        self.show_lum_var = tk.BooleanVar(value=True)
        self.show_phot_337_var = tk.BooleanVar(value=True)
        self.show_phot_391_var = tk.BooleanVar(value=True)
        self.show_phot_777_var = tk.BooleanVar(value=True)

        # Visibility toggles — ratios
        self.show_ratio_337_777_var = tk.BooleanVar(value=True)
        self.show_ratio_391_777_var = tk.BooleanVar(value=True)
        self.show_ratio_337_391_var = tk.BooleanVar(value=True)

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
        self.legend_loc_var = tk.StringVar(value="upper right")
        self.plot_title_var = tk.StringVar(value="")

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

        ratio_frame = ttk.LabelFrame(frame, text="Irradiance Ratios", padding=2)
        ratio_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(ratio_frame, text="337/777 (Blue)", variable=self.show_ratio_337_777_var).pack(anchor='w')
        ttk.Checkbutton(ratio_frame, text="391/777 (Purple)", variable=self.show_ratio_391_777_var).pack(anchor='w')
        ttk.Checkbutton(ratio_frame, text="337/391 (Orange)", variable=self.show_ratio_337_391_var).pack(anchor='w')

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
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Legend:").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.legend_loc_var,
                     values=["Off", "upper right", "upper left",
                             "lower right", "lower left", "best"],
                     width=14, state='readonly').pack(side=tk.LEFT, padx=2)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Title:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.plot_title_var, width=25).pack(side=tk.LEFT, padx=2)

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
        self.main_app.status_var.set("Figure 1: loaded from project")

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
    # INTF Binned Scatter (identical to FlashOverviewTab)
    # =====================================================================

    def _plot_intf_binned(self, ax, t_min, t_max):
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
    # Plot
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
            self._do_plot()
        except Exception as e:
            traceback.print_exc()
            self.main_app.status_var.set(f"Plot error: {e}")
        finally:
            self.canvas.draw_idle()

    def _do_plot(self):

        try:
            x_start = float(self.x_start_var.get())
            x_stop = float(self.x_stop_var.get())
        except Exception:
            x_start, x_stop = 0, 100000

        t_min, t_max = x_start, x_stop

        # Check if any ratios are enabled
        any_ratio = (self.show_ratio_337_777_var.get() or
                     self.show_ratio_391_777_var.get() or
                     self.show_ratio_337_391_var.get())

        # Always use 2-panel layout: main plot on top, ratio plot below
        if any_ratio:
            gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.12)
            self.ax_main = self.fig.add_subplot(gs[0])
            self.ax_ratio = self.fig.add_subplot(gs[1], sharex=self.ax_main)
        else:
            self.ax_main = self.fig.add_subplot(111)
            self.ax_ratio = None

        ax = self.ax_main
        ax.set_xlim(t_min, t_max)

        legend_handles = []
        legend_labels = []
        rhs_twin_count = 0

        # --- Photometer (PRIMARY left y-axis) ---
        any_phot = (self.show_phot_337_var.get() or
                    self.show_phot_391_var.get() or
                    self.show_phot_777_var.get())

        phot_data = None  # Save for ratio subplot
        if any_phot and self.photometer_handler.is_loaded:
            downsample = 1
            phot_data = self.photometer_handler.get_data_in_event_time(t_min, t_max, downsample)
            if phot_data and phot_data['time'] is not None and len(phot_data['time']) > 0:
                if self.show_phot_337_var.get() and phot_data['ch0'] is not None:
                    line, = ax.plot(phot_data['time'], phot_data['ch0'],
                                    color='blue', linewidth=0.5, alpha=0.8, label='337nm')
                    legend_handles.append(line)
                    legend_labels.append('337 nm')

                if self.show_phot_391_var.get() and phot_data['ch1'] is not None:
                    line, = ax.plot(phot_data['time'], phot_data['ch1'],
                                    color='purple', linewidth=0.5, alpha=0.8, label='391nm')
                    legend_handles.append(line)
                    legend_labels.append('391 nm')

                if self.show_phot_777_var.get() and phot_data['ch2'] is not None:
                    line, = ax.plot(phot_data['time'], phot_data['ch2'],
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

        # Put xlabel on ratio axis if present, else on main
        if self.ax_ratio is not None:
            ax.set_xlabel("")
        else:
            ax.set_xlabel(xlabel_text, fontsize=12)

        title = self.plot_title_var.get()
        if title:
            ax.set_title(title, fontsize=14)

        legend_loc = self.legend_loc_var.get()
        if legend_loc != "Off" and legend_handles:
            ax.legend(legend_handles, legend_labels, loc=legend_loc, fontsize=9)

        # --- Ratio subplot ---
        if self.ax_ratio is not None and self.photometer_handler.is_loaded:
            # Load photometer data for ratios if not already loaded
            if phot_data is None:
                phot_data = self.photometer_handler.get_data_in_event_time(t_min, t_max, 1)
            self._plot_ratios(self.ax_ratio, phot_data, xlabel_text)

        # Layout
        right_margin = max(0.72, 0.92 - 0.06 * max(rhs_twin_count - 1, 0))
        self.fig.subplots_adjust(left=0.08, right=right_margin, top=0.95, bottom=0.08)

    # =====================================================================
    # Ratio Plotting
    # =====================================================================

    def _plot_ratios(self, ax, data, xlabel):
        """Plot photometer channel ratios on the lower subplot."""
        if data is None or data['time'] is None or len(data['time']) == 0:
            return

        time = data['time']

        if self.show_ratio_337_777_var.get() and data['ch0'] is not None and data['ch2'] is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = data['ch0'] / data['ch2']
                ratio = np.where(np.isfinite(ratio), ratio, np.nan)
            ax.plot(time, ratio, color='blue', linewidth=0.5, label='337/777', alpha=0.8)

        if self.show_ratio_391_777_var.get() and data['ch1'] is not None and data['ch2'] is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = data['ch1'] / data['ch2']
                ratio = np.where(np.isfinite(ratio), ratio, np.nan)
            ax.plot(time, ratio, color='purple', linewidth=0.5, label='391/777', alpha=0.8)

        if self.show_ratio_337_391_var.get() and data['ch0'] is not None and data['ch1'] is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = data['ch0'] / data['ch1']
                ratio = np.where(np.isfinite(ratio), ratio, np.nan)
            ax.plot(time, ratio, color='orange', linewidth=0.5, label='337/391', alpha=0.8)

        # Bold reference line at y=1
        ax.axhline(y=1, color='black', linestyle='-', linewidth=2.5, zorder=3)

        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Irradiance Ratios", fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        if self.show_grid_var.get():
            ax.grid(True, linestyle=':', alpha=0.5)

    # =====================================================================
    # Export
    # =====================================================================

    def _export(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All", "*.*")])
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches='tight',
                            facecolor=self.fig.get_facecolor(), edgecolor='none')
            self.main_app.status_var.set(f"Exported: {os.path.basename(path)}")

    # =====================================================================
    # Project Save / Load
    # =====================================================================

    def _save_to_project(self):
        state = self.main_app.project_state
        f1 = state.figure1

        try:
            f1['time_min'] = float(self.x_start_var.get())
        except Exception:
            f1['time_min'] = None
        try:
            f1['time_max'] = float(self.x_stop_var.get())
        except Exception:
            f1['time_max'] = None

        f1['event_time'] = self.event_time_var.get().strip() or None
        f1['show_ratio_337_777'] = self.show_ratio_337_777_var.get()
        f1['show_ratio_391_777'] = self.show_ratio_391_777_var.get()
        f1['show_ratio_337_391'] = self.show_ratio_337_391_var.get()

    def load_from_project(self):
        state = self.main_app.project_state
        f1 = state.figure1

        if f1.get('time_min') is not None:
            self.x_start_var.set(str(f1['time_min']))
        if f1.get('time_max') is not None:
            self.x_stop_var.set(str(f1['time_max']))

        evt = f1.get('event_time') or state.event_info.get('time', '')
        if evt:
            self.event_time_var.set(evt)

        self.show_ratio_337_777_var.set(f1.get('show_ratio_337_777', True))
        self.show_ratio_391_777_var.set(f1.get('show_ratio_391_777', True))
        self.show_ratio_337_391_var.set(f1.get('show_ratio_337_391', True))

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
        legend_loc = state.plot_style.get('legend_location')
        if legend_loc is None:
            legend_loc = 'upper right' if state.plot_style.get('show_legend', True) else 'Off'
        self.legend_loc_var.set(legend_loc)
        self.plot_title_var.set(state.plot_style.get('title', ''))

        # Actually load data from the file paths we just set
        self._load_all_data()
