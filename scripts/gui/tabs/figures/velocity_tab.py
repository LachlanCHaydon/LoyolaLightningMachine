"""
Velocity Tab — Two-panel figure with waveform overlay and azi-elv scatter.

Left panel:  INTF elevation + FA + TASD + photometer + luminosity overlays
             with optional dart-leader velocity linear fit.
Right panel: INTF azimuth vs elevation scatter, colored by time.

Velocity = dz/dt where z = x1 * tan(elevation), using scipy linregress.
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
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress

from config import (
    INSTRUMENT_COLORS, DEFAULT_FONT_SIZE, DEFAULT_LIMITS,
    cmap_mjet, PHOTOMETER_CALIBRATION
)
from data_handlers import (
    PhotometerHandler, FastAntennaHandler, InterferometerHandler,
    TASDHandler, LuminosityHandler
)
from utils.ta_tools import calculate_azimuth_to_detector


class VelocityTab(ttk.Frame):
    """Two square panels: waveform overlay (left) + azi-elv scatter (right)."""

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

        # Waveform plot range
        self.x_start_var = tk.StringVar(value="0")
        self.x_stop_var = tk.StringVar(value="100000")

        # Instrument toggles
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

        # Az-El panel limits
        self.ae_azi_min_var = tk.StringVar(value=str(DEFAULT_LIMITS['intf_azi']['y_min']))
        self.ae_azi_max_var = tk.StringVar(value=str(DEFAULT_LIMITS['intf_azi']['y_max']))
        self.ae_elv_min_var = tk.StringVar(value=str(DEFAULT_LIMITS['intf_elev']['y_min']))
        self.ae_elv_max_var = tk.StringVar(value=str(DEFAULT_LIMITS['intf_elev']['y_max']))
        self.ae_time_min_var = tk.StringVar(value="")
        self.ae_time_max_var = tk.StringVar(value="")

        # Style
        self.intf_calibrated_var = tk.BooleanVar(value=True)
        self.show_grid_var = tk.BooleanVar(value=True)
        self.legend_loc_var = tk.StringVar(value="upper right")
        self.plot_title_var = tk.StringVar(value="")

        # Velocity fit parameters
        self.fit_t_start_var = tk.StringVar(value="")
        self.fit_t_stop_var = tk.StringVar(value="")
        self.fit_elv_min_var = tk.StringVar(value="")
        self.fit_elv_max_var = tk.StringVar(value="")
        self.x1_var = tk.StringVar(value="13.9")
        self.show_fit_var = tk.BooleanVar(value=True)
        self.show_velo_text_var = tk.BooleanVar(value=True)
        self.velo_text_x_var = tk.StringVar(value="")
        self.velo_text_y_var = tk.StringVar(value="")

        # TASD overlay on Az-El
        self.show_tasd_overlay_var = tk.BooleanVar(value=True)
        self.show_tasd_legend_var = tk.BooleanVar(value=True)
        self.source_azi_var = tk.StringVar(value="")
        self.source_elv_var = tk.StringVar(value="")

        # Velocity result (displayed in UI)
        self.velocity_result_var = tk.StringVar(value="—")

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
        self._build_ae_section()
        self._build_velocity_section()
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
        frame = ttk.LabelFrame(self.scrollable_frame, text="Waveform Range (\u00b5s)", padding=5)
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

    def _build_ae_section(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Az-El Panel", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for label, min_var, max_var in [
            ("Azi (deg):", self.ae_azi_min_var, self.ae_azi_max_var),
            ("Elv (deg):", self.ae_elv_min_var, self.ae_elv_max_var),
        ]:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=min_var, width=8).pack(side=tk.LEFT, padx=1)
            ttk.Label(row, text="-").pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=max_var, width=8).pack(side=tk.LEFT, padx=1)

        # TASD overlay
        ttk.Checkbutton(frame, text="Show TASD overlay",
                        variable=self.show_tasd_overlay_var).pack(anchor='w', pady=2)
        ttk.Checkbutton(frame, text="Show TASD overlay legend",
                        variable=self.show_tasd_legend_var).pack(anchor='w')

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="Src Azi:", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.source_azi_var, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Label(row, text="Elv:").pack(side=tk.LEFT, padx=2)
        ttk.Entry(row, textvariable=self.source_elv_var, width=8).pack(side=tk.LEFT, padx=1)

    def _build_velocity_section(self):
        frame = ttk.LabelFrame(self.scrollable_frame, text="Velocity Fit", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Fit time window
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="Fit t start:", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.fit_t_start_var, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Label(row, text="t stop:").pack(side=tk.LEFT, padx=2)
        ttk.Entry(row, textvariable=self.fit_t_stop_var, width=10).pack(side=tk.LEFT, padx=1)

        # Elevation filter for fit
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="Elv min:", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.fit_elv_min_var, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Label(row, text="max:").pack(side=tk.LEFT, padx=2)
        ttk.Entry(row, textvariable=self.fit_elv_max_var, width=8).pack(side=tk.LEFT, padx=1)

        # Horizontal distance
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="x1 (km):", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.x1_var, width=10).pack(side=tk.LEFT, padx=1)

        # Display options
        ttk.Checkbutton(frame, text="Show fit line", variable=self.show_fit_var).pack(anchor='w')
        ttk.Checkbutton(frame, text="Show velocity text", variable=self.show_velo_text_var).pack(anchor='w')

        # Text position
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="Text x:", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.velo_text_x_var, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Label(row, text="y:").pack(side=tk.LEFT, padx=2)
        ttk.Entry(row, textvariable=self.velo_text_y_var, width=8).pack(side=tk.LEFT, padx=1)

        # Result display
        result_frame = ttk.LabelFrame(frame, text="Result", padding=3)
        result_frame.pack(fill=tk.X, pady=5)
        ttk.Label(result_frame, textvariable=self.velocity_result_var,
                  wraplength=300, justify=tk.LEFT).pack(fill=tk.X)

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
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.ax_wave = None
        self.ax_ae = None

        # Coord display at very bottom
        self.coord_var = tk.StringVar(value="")
        ttk.Label(self.plot_frame, textvariable=self.coord_var,
                  font=('Courier', 9)).pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas + toolbar: toolbar auto-packs at bottom, then canvas fills rest
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def _on_mouse_move(self, event):
        if event.inaxes:
            self.coord_var.set(f"x={event.xdata:.1f}, y={event.ydata:.3f}")
        else:
            self.coord_var.set("")

    # =====================================================================
    # File Operations
    # =====================================================================

    def _browse_file(self, var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _browse_directory(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _load_files_from_project(self):
        state = self.main_app.project_state
        if state.files.get('fa'):
            self.file_paths['fa'].set(state.files['fa'])
        if state.files.get('intf_calibrated'):
            self.file_paths['intf'].set(state.files['intf_calibrated'])
            self.intf_calibrated_var.set(True)
        elif state.files.get('intf_raw'):
            self.file_paths['intf'].set(state.files['intf_raw'])
            self.intf_calibrated_var.set(False)
        if state.files.get('sd_directory'):
            self.file_paths['sd_dir'].set(state.files['sd_directory'])
        if state.files.get('photometer'):
            self.file_paths['photometer'].set(state.files['photometer'])
        if state.files.get('luminosity'):
            self.file_paths['luminosity'].set(state.files['luminosity'])

        if state.timing.get('T0') is not None:
            self.T0_var.set(str(state.timing['T0']))
        if state.timing.get('timeshift') is not None:
            self.timeshift_var.set(str(state.timing['timeshift']))
        if state.timing.get('photometer_second_offset') is not None:
            self.phot_offset_var.set(str(state.timing['photometer_second_offset']))

        self._load_all_data()

    def _find_tasd_coords(self, sd_dir):
        """Auto-find TASD coordinates file (matching INTF tab search)."""
        possible_paths = []

        if sd_dir:
            possible_paths.append(os.path.join(sd_dir, 'tasd_gpscoors.txt'))
            possible_paths.append(os.path.join(os.path.dirname(sd_dir), 'tasd_gpscoors.txt'))

        possible_paths.append('tasd_gpscoors.txt')
        possible_paths.append(os.path.join(os.getcwd(), 'tasd_gpscoors.txt'))

        script_dir = os.path.dirname(os.path.abspath(__file__))
        for up in ['', '..', os.path.join('..', '..'), os.path.join('..', '..', '..')]:
            possible_paths.append(os.path.join(script_dir, up, 'tasd_gpscoors.txt'))

        if 'gui' in script_dir:
            gui_parent = script_dir.split('gui')[0]
            possible_paths.append(os.path.join(gui_parent, 'tasd_gpscoors.txt'))

        try:
            if hasattr(self.main_app, 'project_filepath') and self.main_app.project_filepath:
                project_dir = os.path.dirname(self.main_app.project_filepath)
                possible_paths.append(os.path.join(project_dir, 'tasd_gpscoors.txt'))
        except Exception:
            pass

        for p in possible_paths:
            if os.path.isfile(p):
                return os.path.abspath(p)

        return None

    def _load_all_data(self):
        loaded = []
        errors = []

        try:
            T0 = float(self.T0_var.get())
        except (ValueError, tk.TclError):
            T0 = 0
        try:
            timeshift = float(self.timeshift_var.get())
        except (ValueError, tk.TclError):
            timeshift = 0
        try:
            phot_offset = int(self.phot_offset_var.get())
        except (ValueError, tk.TclError):
            phot_offset = 0

        # FA
        fa_path = self.file_paths['fa'].get()
        if fa_path and os.path.exists(fa_path):
            try:
                self.fa_handler.load_csv(fa_path)
                self.fa_handler.set_T0(T0)
                loaded.append("FA")
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

        # TASD
        sd_dir = self.file_paths['sd_dir'].get()
        if sd_dir and os.path.isdir(sd_dir):
            try:
                coords_path = self._find_tasd_coords(sd_dir)
                if coords_path:
                    self.tasd_handler.load_coordinates(coords_path)
                dirname = os.path.basename(sd_dir)
                parts = dirname.split('_')
                time_str = parts[2] if len(parts) >= 3 else ""
                self.tasd_handler.load_directory(sd_dir, time_str, time_shift=timeshift)
                loaded.append("TASD")
            except Exception as e:
                errors.append(f"TASD: {e}")

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
    # INTF Binned Scatter (elevation vs time, for waveform panel)
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
    # Velocity Calculation
    # =====================================================================

    def _calc_velocity(self, ax_wave):
        """
        Compute dart leader velocity via linear regression.

        Matches original script approach:
        1. Filter INTF by time window + elevation cut
        2. polyfit(time, elevation, 1) to get fit line
        3. Convert elevation travel to altitude: d = x1 * tan(elv_range)
        4. v = d / dt

        Also uses scipy linregress on altitude for proper error bars.
        Returns (velocity_km_s, stderr_km_s, v_1e7, se_1e7, fit_time, fit_elv) or None.
        """
        # Check required fields
        fit_t_start_str = self.fit_t_start_var.get().strip()
        fit_t_stop_str = self.fit_t_stop_var.get().strip()
        x1_str = self.x1_var.get().strip()

        if not fit_t_start_str or not fit_t_stop_str:
            self.velocity_result_var.set("Set fit t start and t stop to calculate")
            return None
        if not x1_str:
            self.velocity_result_var.set("Set x1 (horizontal distance in km)")
            return None

        try:
            fit_t_start = float(fit_t_start_str)
            fit_t_stop = float(fit_t_stop_str)
            x1 = float(x1_str)
        except (ValueError, tk.TclError):
            self.velocity_result_var.set("Invalid fit parameters")
            return None

        try:
            fit_elv_max = float(self.fit_elv_max_var.get())
        except (ValueError, tk.TclError):
            fit_elv_max = None
        try:
            fit_elv_min = float(self.fit_elv_min_var.get())
        except (ValueError, tk.TclError):
            fit_elv_min = None

        if not self.intf_handler.is_loaded:
            self.velocity_result_var.set("No INTF data loaded")
            return None

        # Filter INTF data by time window and elevation cut
        filtered = self.intf_handler.filter_data(
            elv_min=fit_elv_min, elv_max=fit_elv_max,
            t_min=fit_t_start, t_max=fit_t_stop)

        if filtered is None or len(filtered['time']) < 3:
            n = 0 if filtered is None else len(filtered['time'])
            self.velocity_result_var.set(
                f"Not enough INTF points ({n} found, need >= 3)\n"
                f"Window: {fit_t_start:.0f} - {fit_t_stop:.0f} \u00b5s"
                + (f", elv < {fit_elv_max}" if fit_elv_max is not None else ""))
            return None

        time_us = np.array(filtered['time'])
        elv_deg = np.array(filtered['elevation'])

        # Convert elevation to altitude: z = x1 * tan(elv)
        z_km = x1 * np.tan(np.deg2rad(elv_deg))

        # Linear regression on altitude vs time (proper method)
        result = linregress(time_us, z_km)
        velocity_km_s = result.slope * 1e6  # km/µs -> km/s
        stderr_km_s = result.stderr * 1e6

        # Velocity in 10^7 m/s
        v_1e7 = velocity_km_s * 1e3 / 1e7  # km/s -> m/s -> 10^7 m/s
        se_1e7 = stderr_km_s * 1e3 / 1e7

        # Polyfit on elevation vs time (for fit line display)
        coef = np.polyfit(time_us, elv_deg, 1)
        elev_fit = np.polyval(coef, time_us)

        # Distance and time spans for display
        elv_travel_deg = elev_fit.max() - elev_fit.min()
        d_km = abs(z_km.max() - z_km.min())
        dt_us = time_us.max() - time_us.min()

        n_points = len(time_us)
        r_squared = result.rvalue ** 2

        # Residual std (matching original script)
        residuals = elv_deg - elev_fit
        std_res = np.sqrt(np.mean(residuals**2))

        self.velocity_result_var.set(
            f"v = {v_1e7:.2f} \u00b1 {se_1e7:.2f} \u00d7 10\u2077 m/s\n"
            f"v = {velocity_km_s:.0f} \u00b1 {stderr_km_s:.0f} km/s\n"
            f"R\u00b2 = {r_squared:.4f},  N = {n_points}\n"
            f"\u0394elv = {elv_travel_deg:.2f}\u00b0,  \u0394z = {d_km:.2f} km,  \u0394t = {dt_us:.1f} \u00b5s\n"
            f"Residual \u03c3 = {std_res:.3f}\u00b0"
        )

        # Fit line for plotting (in elevation space)
        fit_time = np.linspace(time_us.min(), time_us.max(), 50)
        fit_elv = np.polyval(coef, fit_time)

        return velocity_km_s, stderr_km_s, v_1e7, se_1e7, fit_time, fit_elv

    # =====================================================================
    # Plot
    # =====================================================================

    def _update_plot(self):
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

        # Two square data panels (colorbar added after layout from ax_ae position)
        gs = GridSpec(1, 2, figure=self.fig,
                      width_ratios=[1, 1], wspace=0.45)
        ax_w = self.fig.add_subplot(gs[0, 0])
        ax_ae = self.fig.add_subplot(gs[0, 1])
        self._cbar_ax = None  # created after tight_layout

        ax_w.set_box_aspect(1)
        ax_ae.set_box_aspect(1)

        self.ax_wave = ax_w
        self.ax_ae = ax_ae

        ax_w.set_xlim(t_min, t_max)

        legend_handles = []
        legend_labels = []
        rhs_twin_count = 0

        # --- Waveform panel (left) ---
        # Layout matches Figure2BCD: base axis = INTF elevation (left y),
        # FA and TASD twins with visible right y-axes.

        rhs_spine_count = 0  # track how many right-side spines we need to offset

        # INTF elevation scatter on base axis (left y-axis)
        if self.show_intf_var.get() and self.intf_handler.is_loaded:
            scatter_handle = self._plot_intf_binned(ax_w, t_min, t_max)
            if scatter_handle:
                legend_handles.append(scatter_handle)
                legend_labels.append('INTF')
            ax_w.set_ylabel("Elevation (deg)", fontsize=12, color='black')
            ax_w.tick_params(axis='y', labelcolor='black')
            try:
                ax_w.set_ylim(float(self.intf_ymin_var.get()),
                              float(self.intf_ymax_var.get()))
            except Exception:
                pass

        # Photometer (twin, y-axis hidden — overlaid on elevation scale)
        any_phot = (self.show_phot_337_var.get() or
                    self.show_phot_391_var.get() or
                    self.show_phot_777_var.get())

        if any_phot and self.photometer_handler.is_loaded:
            downsample = 1
            phot_data = self.photometer_handler.get_data_in_event_time(t_min, t_max, downsample)
            if phot_data and phot_data['time'] is not None and len(phot_data['time']) > 0:
                ax_phot = ax_w.twinx()
                ax_phot.set_box_aspect(1)
                ax_phot.yaxis.set_visible(False)
                phot_time = phot_data['time']
                if self.show_phot_337_var.get() and phot_data['ch0'] is not None:
                    line, = ax_phot.plot(phot_time, phot_data['ch0'], color='blue',
                                         linewidth=1, alpha=0.8)
                    legend_handles.append(line)
                    legend_labels.append('337 nm')
                if self.show_phot_391_var.get() and phot_data['ch1'] is not None:
                    line, = ax_phot.plot(phot_time, phot_data['ch1'], color='purple',
                                         linewidth=1, alpha=0.8)
                    legend_handles.append(line)
                    legend_labels.append('391 nm')
                if self.show_phot_777_var.get() and phot_data['ch2'] is not None:
                    line, = ax_phot.plot(phot_time, phot_data['ch2'], color='red',
                                         linewidth=1, alpha=0.8)
                    legend_handles.append(line)
                    legend_labels.append('777 nm')
                try:
                    ax_phot.set_ylim(float(self.phot_ymin_var.get()),
                                     float(self.phot_ymax_var.get()))
                except Exception:
                    pass

        # FA (visible right y-axis — matches Figure2BCD par1)
        if self.show_fa_var.get() and self.fa_handler.is_loaded:
            fa_data = self.fa_handler.get_data_in_range(t_min, t_max)
            if fa_data and fa_data['time'] is not None and len(fa_data['time']) > 0:
                ax_fa = ax_w.twinx()
                ax_fa.set_box_aspect(1)
                line, = ax_fa.plot(fa_data['time'], fa_data['e_field'],
                                   color=INSTRUMENT_COLORS['fa'], linewidth=1.5, zorder=2)
                legend_handles.append(line)
                legend_labels.append('Fast Antenna')
                ax_fa.set_ylabel("Fast Antenna (V/m)", fontsize=12,
                                 color=INSTRUMENT_COLORS['fa'])
                ax_fa.tick_params(axis='y', labelcolor=INSTRUMENT_COLORS['fa'])
                try:
                    ax_fa.set_ylim(float(self.fa_ymin_var.get()),
                                   float(self.fa_ymax_var.get()))
                except Exception:
                    pass
                rhs_spine_count += 1

        # TASD (visible right y-axis, offset — matches Figure2BCD par2)
        if self.show_sd_var.get() and self.tasd_handler.is_loaded:
            detectors = self.tasd_handler.get_data_in_range(t_min, t_max)
            if detectors:
                ax_sd = ax_w.twinx()
                ax_sd.set_box_aspect(1)
                if rhs_spine_count > 0:
                    ax_sd.spines["right"].set_position(("axes", 1.0 + 0.20 * rhs_spine_count))
                first = True
                for det in detectors:
                    if len(det['time']) > 0:
                        line, = ax_sd.plot(det['time'], det['signal_upper'],
                                           color=INSTRUMENT_COLORS['sd'], linewidth=1)
                        if first:
                            legend_handles.append(line)
                            legend_labels.append('TASD')
                            first = False
                ax_sd.set_ylabel("FADC count", fontsize=12,
                                 color=INSTRUMENT_COLORS['sd'])
                ax_sd.tick_params(axis='y', labelcolor=INSTRUMENT_COLORS['sd'])
                try:
                    ax_sd.set_ylim(float(self.sd_ymin_var.get()),
                                   float(self.sd_ymax_var.get()))
                except Exception:
                    pass

        # Luminosity (twin, y-axis hidden)
        if self.show_lum_var.get() and self.luminosity_handler.is_loaded:
            data = self.luminosity_handler.get_data_in_range(t_min, t_max)
            if data and len(data['time']) > 0:
                ax_lum = ax_w.twinx()
                ax_lum.set_box_aspect(1)
                ax_lum.yaxis.set_visible(False)
                line, = ax_lum.plot(data['time'], data['luminosity'],
                                    color=INSTRUMENT_COLORS['luminosity'], linewidth=1.5, zorder=2)
                legend_handles.append(line)
                legend_labels.append('Luminosity')
                try:
                    ax_lum.set_ylim(float(self.lum_ymin_var.get()),
                                    float(self.lum_ymax_var.get()))
                except Exception:
                    pass

        # Velocity fit (now on base axis since elevation is on ax_w)
        velo_result = self._calc_velocity(ax_w)
        if velo_result is not None:
            velocity_km_s, stderr_km_s, v_1e7, se_1e7, fit_time, fit_elv = velo_result

            if self.show_fit_var.get():
                ax_w.plot(fit_time, fit_elv, color='orange', linestyle='--',
                          linewidth=3, zorder=5)

            if self.show_velo_text_var.get():
                try:
                    text_x = float(self.velo_text_x_var.get())
                    text_y = float(self.velo_text_y_var.get())
                except (ValueError, tk.TclError):
                    # Default: lower-left area of the waveform panel
                    text_x = t_min + 0.05 * (t_max - t_min)
                    try:
                        text_y = float(self.intf_ymin_var.get()) + 2
                    except Exception:
                        text_y = 5
                ax_w.text(text_x, text_y,
                               f"v = {v_1e7:.2f}\u00d710$^7$ m/s",
                               color='magenta', fontsize=11.7, fontweight='bold',
                               zorder=100)

        # Waveform panel decoration
        if self.show_grid_var.get():
            ax_w.grid(True, linestyle=':', alpha=0.5)

        event_time = self.event_time_var.get().strip()
        xlabel_text = f"Microseconds after {event_time}" if event_time else "Time (\u00b5s)"
        ax_w.set_xlabel(xlabel_text, fontsize=12)
        ax_w.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))

        title = self.plot_title_var.get().strip()
        if title:
            ax_w.set_title(title, fontsize=14, fontweight='bold')

        # --- Az-El panel (right) ---
        self._ae_scatter = None
        self._ae_clabel = ""
        self._plot_azi_elv(ax_ae, t_min, t_max)

        # Finalize layout before adding legend/colorbar
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.05, right=0.90)


        # Colorbar matched to ax_ae's actual square (set_box_aspect shrinks
        # the axes inside the allocated rect — must read true pixel extent).
        if self._ae_scatter is not None:
            self.fig.canvas.draw()  # force full render so box_aspect takes effect
            renderer = self.fig.canvas.get_renderer()
            bbox_disp = ax_ae.get_window_extent(renderer)
            bbox_fig = bbox_disp.transformed(self.fig.transFigure.inverted())
            cbar_ax = self.fig.add_axes([bbox_fig.x1 + 0.045, bbox_fig.y0,
                                         0.012, bbox_fig.height])
            cbar = self.fig.colorbar(self._ae_scatter, cax=cbar_ax)
            cbar.set_label(self._ae_clabel, fontsize=10)
            self._cbar_ax = cbar_ax

        # Waveform legend — fig.legend renders above all twin axes
        legend_loc = self.legend_loc_var.get()
        if legend_loc != "Off" and legend_handles:
            ax_w.legend(legend_handles, legend_labels,
                        loc=legend_loc, fontsize=9, framealpha=0.9)

    # =====================================================================
    # Azi-Elv Scatter (right panel) — mirrors INTF tab _update_plot az_el
    # =====================================================================

    def _plot_azi_elv(self, ax, t_min, t_max):
        """Plot azi-elv scatter using the same approach as INTFTab."""

        if not self.intf_handler.is_loaded:
            ax.set_xlabel("Azimuth (deg)", fontsize=12)
            ax.set_ylabel("Elevation (deg)", fontsize=12)
            ax.text(0.5, 0.5, "No INTF data loaded", transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            if self._cbar_ax is not None:
                self._cbar_ax.set_visible(False)
            return

        # Use the same time range as left waveform panel
        data = self.intf_handler.filter_data(t_min=t_min, t_max=t_max)

        if data is None or len(data['time']) == 0:
            ax.set_xlabel("Azimuth (deg)", fontsize=12)
            ax.set_ylabel("Elevation (deg)", fontsize=12)
            ax.text(0.5, 0.5, "No INTF data in range", transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            if self._cbar_ax is not None:
                self._cbar_ax.set_visible(False)
            return

        # Color by time (matching INTF tab color_by="time" mode)
        colors = data['time']
        cmap = 'jet'
        event_time = self.event_time_var.get().strip()
        clabel = f"Time relative to {event_time} (us)" if event_time else "Time (us)"

        # Marker sizes from handler (pre-computed: 6 * (1+3*ss_norm^2)^2)
        sizes = data['marker_sizes'] if data['marker_sizes'] is not None else 10

        # Clip elevation >= 0 for cleanliness (TASD rings at 0 remain visible)
        plot_azi = data['azimuth']
        plot_elv = data['elevation']
        elv_clip_mask = plot_elv >= 0
        plot_azi = plot_azi[elv_clip_mask]
        plot_elv = plot_elv[elv_clip_mask]
        colors = colors[elv_clip_mask]
        if isinstance(sizes, np.ndarray):
            sizes = sizes[elv_clip_mask]

        # Single scatter call — exactly like INTF tab
        sc = ax.scatter(plot_azi, plot_elv,
                        c=colors, cmap=cmap, s=sizes, alpha=0.7)

        ax.set_xlabel("Azimuth (deg)", fontsize=12)
        # Axis limits (matching INTF tab pattern: min(elv_lo, 0) - 1)
        try:
            ax.set_xlim(float(self.ae_azi_min_var.get()) - 1,
                        float(self.ae_azi_max_var.get()) + 1)
        except (ValueError, tk.TclError):
            pass
        try:
            elv_lo = float(self.ae_elv_min_var.get())
            elv_hi = float(self.ae_elv_max_var.get())
            ax.set_ylim(min(elv_lo, 0) - 1, elv_hi + 1)
        except (ValueError, tk.TclError):
            pass

        # TASD overlay (detector rings at elevation 0)
        if self.show_tasd_overlay_var.get() and self.tasd_handler.is_loaded:
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
                        size = max(50, 150 * np.log(max(vem, 1.0)))
                        ax.scatter(det_azi, 0, s=size,
                                   facecolors='none', edgecolors='red',
                                   linewidths=1.5, zorder=4)
                    except Exception:
                        pass

                core_azi = None
                if det_azis and vems:
                    vems_arr = np.array(vems)
                    core_azi = float(np.average(det_azis, weights=vems_arr))
                    ax.scatter(core_azi, 0, marker='o', s=250,
                               facecolors='red', edgecolors='k',
                               linewidths=1.5, zorder=5, label='TGF shower core')

                try:
                    src_azi = float(self.source_azi_var.get())
                    src_elv = float(self.source_elv_var.get())
                    if det_azis and core_azi is not None:
                        ax.plot([src_azi, core_azi], [src_elv, 0],
                                'r-', lw=1.5, label='TGF shower axis', zorder=5)
                        ax.plot([src_azi, min(det_azis)], [src_elv, 0],
                                'r--', lw=1, alpha=0.6, zorder=5)
                        ax.plot([src_azi, max(det_azis)], [src_elv, 0],
                                'r--', lw=1, alpha=0.6, zorder=5)
                except (ValueError, TypeError):
                    pass

                if self.show_tasd_legend_var.get():
                    ax.legend(fontsize=9, loc='upper right')

        # Secondary y-axis for height (km)
        try:
            x1 = float(self.x1_var.get())
            if x1 > 0:
                def elv_to_height(x):
                    return x1 * np.tan(np.radians(x))
                def height_to_elv(x):
                    return np.degrees(np.arctan(x / x1))
                sec_ax = ax.secondary_yaxis('right',
                                            functions=(elv_to_height, height_to_elv))
                sec_ax.set_ylabel('Height (km)', fontsize=11)
                sec_ax.tick_params(labelsize=10)
        except (ValueError, TypeError, tk.TclError):
            pass

        # Store scatter + label for colorbar creation after tight_layout
        self._ae_scatter = sc
        self._ae_clabel = clabel

        if self.show_grid_var.get():
            ax.grid(True, linestyle=':', alpha=0.5)

    # =====================================================================
    # Export
    # =====================================================================

    def _export(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All", "*.*")])
        if path:
            self.fig.savefig(path, dpi=600, bbox_inches='tight',
                            facecolor='white', edgecolor='none',
                            transparent=False)
            self.main_app.status_var.set(f"Exported: {os.path.basename(path)}")

    # =====================================================================
    # Project Save / Load
    # =====================================================================

    def _safe_float(self, var):
        try:
            return float(var.get())
        except (ValueError, tk.TclError):
            return None

    def _save_to_project(self):
        state = self.main_app.project_state
        v = state.velocity

        v['x_start'] = self._safe_float(self.x_start_var)
        v['x_stop'] = self._safe_float(self.x_stop_var)
        v['event_time'] = self.event_time_var.get().strip()
        v['show_grid'] = self.show_grid_var.get()
        v['legend_location'] = self.legend_loc_var.get()
        v['title'] = self.plot_title_var.get()

        v['show_fa'] = self.show_fa_var.get()
        v['show_intf'] = self.show_intf_var.get()
        v['show_sd'] = self.show_sd_var.get()
        v['show_lum'] = self.show_lum_var.get()
        v['show_phot_337'] = self.show_phot_337_var.get()
        v['show_phot_391'] = self.show_phot_391_var.get()
        v['show_phot_777'] = self.show_phot_777_var.get()

        v['fa_ymin'] = self._safe_float(self.fa_ymin_var)
        v['fa_ymax'] = self._safe_float(self.fa_ymax_var)
        v['sd_ymin'] = self._safe_float(self.sd_ymin_var)
        v['sd_ymax'] = self._safe_float(self.sd_ymax_var)
        v['intf_ymin'] = self._safe_float(self.intf_ymin_var)
        v['intf_ymax'] = self._safe_float(self.intf_ymax_var)
        v['lum_ymin'] = self._safe_float(self.lum_ymin_var)
        v['lum_ymax'] = self._safe_float(self.lum_ymax_var)
        v['phot_ymin'] = self._safe_float(self.phot_ymin_var)
        v['phot_ymax'] = self._safe_float(self.phot_ymax_var)

        v['ae_azi_min'] = self._safe_float(self.ae_azi_min_var)
        v['ae_azi_max'] = self._safe_float(self.ae_azi_max_var)
        v['ae_elv_min'] = self._safe_float(self.ae_elv_min_var)
        v['ae_elv_max'] = self._safe_float(self.ae_elv_max_var)
        v['ae_time_min'] = self._safe_float(self.ae_time_min_var)
        v['ae_time_max'] = self._safe_float(self.ae_time_max_var)

        v['fit_t_start'] = self._safe_float(self.fit_t_start_var)
        v['fit_t_stop'] = self._safe_float(self.fit_t_stop_var)
        v['fit_elv_min'] = self._safe_float(self.fit_elv_min_var)
        v['fit_elv_max'] = self._safe_float(self.fit_elv_max_var)
        v['x1'] = self._safe_float(self.x1_var)
        v['show_fit'] = self.show_fit_var.get()
        v['show_velo_text'] = self.show_velo_text_var.get()
        v['velo_text_x'] = self._safe_float(self.velo_text_x_var)
        v['velo_text_y'] = self._safe_float(self.velo_text_y_var)

        v['show_tasd_overlay'] = self.show_tasd_overlay_var.get()
        v['show_tasd_legend'] = self.show_tasd_legend_var.get()
        v['source_azi'] = self.source_azi_var.get().strip()
        v['source_elv'] = self.source_elv_var.get().strip()

    def load_from_project(self):
        state = self.main_app.project_state
        v = state.velocity

        # Load file paths from project
        if state.files.get('fa'):
            self.file_paths['fa'].set(state.files['fa'])
        if state.files.get('intf_calibrated'):
            self.file_paths['intf'].set(state.files['intf_calibrated'])
            self.intf_calibrated_var.set(True)
        elif state.files.get('intf_raw'):
            self.file_paths['intf'].set(state.files['intf_raw'])
            self.intf_calibrated_var.set(False)
        if state.files.get('sd_directory'):
            self.file_paths['sd_dir'].set(state.files['sd_directory'])
        if state.files.get('photometer'):
            self.file_paths['photometer'].set(state.files['photometer'])
        if state.files.get('luminosity'):
            self.file_paths['luminosity'].set(state.files['luminosity'])

        if state.timing.get('T0') is not None:
            self.T0_var.set(str(state.timing['T0']))
        if state.timing.get('timeshift') is not None:
            self.timeshift_var.set(str(state.timing['timeshift']))
        if state.timing.get('photometer_second_offset') is not None:
            self.phot_offset_var.set(str(state.timing['photometer_second_offset']))

        # Velocity-tab-specific state
        def _set(var, key, default=None):
            val = v.get(key, default)
            if val is not None:
                var.set(str(val))

        _set(self.x_start_var, 'x_start')
        _set(self.x_stop_var, 'x_stop')
        _set(self.event_time_var, 'event_time')
        _set(self.plot_title_var, 'title')

        if 'show_grid' in v:
            self.show_grid_var.set(v['show_grid'])
        if 'legend_location' in v:
            self.legend_loc_var.set(v['legend_location'])

        for key, var in [('show_fa', self.show_fa_var), ('show_intf', self.show_intf_var),
                         ('show_sd', self.show_sd_var), ('show_lum', self.show_lum_var),
                         ('show_phot_337', self.show_phot_337_var),
                         ('show_phot_391', self.show_phot_391_var),
                         ('show_phot_777', self.show_phot_777_var)]:
            if key in v:
                var.set(v[key])

        for key, var in [('fa_ymin', self.fa_ymin_var), ('fa_ymax', self.fa_ymax_var),
                         ('sd_ymin', self.sd_ymin_var), ('sd_ymax', self.sd_ymax_var),
                         ('intf_ymin', self.intf_ymin_var), ('intf_ymax', self.intf_ymax_var),
                         ('lum_ymin', self.lum_ymin_var), ('lum_ymax', self.lum_ymax_var),
                         ('phot_ymin', self.phot_ymin_var), ('phot_ymax', self.phot_ymax_var),
                         ('ae_azi_min', self.ae_azi_min_var), ('ae_azi_max', self.ae_azi_max_var),
                         ('ae_elv_min', self.ae_elv_min_var), ('ae_elv_max', self.ae_elv_max_var),
                         ('ae_time_min', self.ae_time_min_var), ('ae_time_max', self.ae_time_max_var),
                         ('fit_t_start', self.fit_t_start_var), ('fit_t_stop', self.fit_t_stop_var),
                         ('fit_elv_min', self.fit_elv_min_var), ('fit_elv_max', self.fit_elv_max_var),
                         ('x1', self.x1_var),
                         ('velo_text_x', self.velo_text_x_var),
                         ('velo_text_y', self.velo_text_y_var)]:
            _set(var, key)

        if 'show_fit' in v:
            self.show_fit_var.set(v['show_fit'])
        if 'show_velo_text' in v:
            self.show_velo_text_var.set(v['show_velo_text'])
        if 'show_tasd_overlay' in v:
            self.show_tasd_overlay_var.set(v['show_tasd_overlay'])
        if 'show_tasd_legend' in v:
            self.show_tasd_legend_var.set(v['show_tasd_legend'])
        if v.get('source_azi'):
            self.source_azi_var.set(v['source_azi'])
        if v.get('source_elv'):
            self.source_elv_var.set(v['source_elv'])

        self._load_all_data()
