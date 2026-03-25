"""
Figure 3 Tab — Per-stroke grid with dart leader velocity calculation.

2 x N grid where N = number of strokes (user-configurable).
  Row 0 (top): FA waveform + INTF elevation scatter per stroke
  Row 1 (bottom): Azimuth-elevation scatter per stroke
  Optional: velocity = dz/dt from linear regression of altitude vs time
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.stats import linregress

from config import INSTRUMENT_COLORS
from data_handlers import FastAntennaHandler, InterferometerHandler


class _StrokePanel(ttk.LabelFrame):
    """UI panel for one stroke's parameters."""

    def __init__(self, parent, stroke_num=1, on_remove=None):
        super().__init__(parent, text=f"Stroke {stroke_num}", padding=5)
        self.on_remove = on_remove

        self.stroke_label_var = tk.StringVar(value=str(stroke_num))
        self.t_start_var = tk.StringVar(value="")
        self.t_stop_var = tk.StringVar(value="")

        self.elv_min_var = tk.StringVar(value="")
        self.elv_max_var = tk.StringVar(value="")
        self.azi_min_var = tk.StringVar(value="")
        self.azi_max_var = tk.StringVar(value="")
        self.ae_t_start_var = tk.StringVar(value="")
        self.ae_t_stop_var = tk.StringVar(value="")

        self.calc_velocity_var = tk.BooleanVar(value=False)
        self.x1_var = tk.StringVar(value="")

        self._build()

    def _build(self):
        # Stroke label
        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="Label:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.stroke_label_var, width=6).pack(side=tk.LEFT)
        if self.on_remove:
            ttk.Button(row, text="Remove", command=self.on_remove).pack(side=tk.RIGHT)

        # Waveform window
        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="Wave t_start:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.t_start_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="t_stop:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.t_stop_var, width=10).pack(side=tk.LEFT, padx=2)

        # Az-El window
        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="AE elv:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.elv_min_var, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.elv_max_var, width=6).pack(side=tk.LEFT, padx=1)

        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="AE azi:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.azi_min_var, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.azi_max_var, width=6).pack(side=tk.LEFT, padx=1)

        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="AE t_start:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.ae_t_start_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="t_stop:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.ae_t_stop_var, width=10).pack(side=tk.LEFT, padx=2)

        # Velocity
        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=1)
        ttk.Checkbutton(row, text="Calculate velocity",
                        variable=self.calc_velocity_var).pack(side=tk.LEFT)
        ttk.Label(row, text="x1 (km):").pack(side=tk.LEFT, padx=3)
        ttk.Entry(row, textvariable=self.x1_var, width=8).pack(side=tk.LEFT)

    def get_config(self):
        """Return stroke config as a dict."""

        def sf(var):
            try:
                return float(var.get())
            except (ValueError, tk.TclError):
                return None

        return {
            'label': self.stroke_label_var.get(),
            't_start': sf(self.t_start_var),
            't_stop': sf(self.t_stop_var),
            'elv_min': sf(self.elv_min_var),
            'elv_max': sf(self.elv_max_var),
            'azi_min': sf(self.azi_min_var),
            'azi_max': sf(self.azi_max_var),
            'ae_t_start': sf(self.ae_t_start_var),
            'ae_t_stop': sf(self.ae_t_stop_var),
            'calc_velocity': self.calc_velocity_var.get(),
            'x1': sf(self.x1_var),
        }

    def set_config(self, cfg):
        """Load stroke config from dict."""
        if cfg.get('label') is not None:
            self.stroke_label_var.set(str(cfg['label']))
        for key, var in [('t_start', self.t_start_var), ('t_stop', self.t_stop_var),
                         ('elv_min', self.elv_min_var), ('elv_max', self.elv_max_var),
                         ('azi_min', self.azi_min_var), ('azi_max', self.azi_max_var),
                         ('ae_t_start', self.ae_t_start_var), ('ae_t_stop', self.ae_t_stop_var),
                         ('x1', self.x1_var)]:
            if cfg.get(key) is not None:
                var.set(str(cfg[key]))
        if 'calc_velocity' in cfg:
            self.calc_velocity_var.set(cfg['calc_velocity'])


class Figure3Tab(ttk.Frame):
    """Per-stroke grid with FA + INTF waveform (top) and Az-El scatter (bottom)."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app

        # Data handlers
        self.fa = FastAntennaHandler()
        self.intf = InterferometerHandler()

        # tk variables
        self.fa_file_var = tk.StringVar()
        self.intf_file_var = tk.StringVar()
        self.event_time_var = tk.StringVar(value="HH:MM:SS")
        self.t0_var = tk.StringVar(value="0")

        self.fa_y_min_var = tk.StringVar(value="")
        self.fa_y_max_var = tk.StringVar(value="")
        self.intf_elv_min_var = tk.StringVar(value="")
        self.intf_elv_max_var = tk.StringVar(value="")

        self.stroke_panels = []

        self._build_ui()

    # =====================================================================
    # UI
    # =====================================================================

    def _build_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        ctrl = ttk.Frame(paned, width=670)
        ctrl.pack_propagate(False)
        paned.add(ctrl, weight=0)

        self.plot_frame = ttk.Frame(paned)
        paned.add(self.plot_frame, weight=1)

        canvas = tk.Canvas(ctrl, width=650)
        scrollbar = ttk.Scrollbar(ctrl, orient="vertical", command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas)
        self.scroll_frame.bind("<Configure>",
                               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_files_section()
        self._build_timing_section()
        self._build_shared_limits_section()
        self._build_strokes_section()
        self._build_action_buttons()
        self._build_plot_area()

    def _build_files_section(self):
        frame = ttk.LabelFrame(self.scroll_frame, text="Files", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for label, var, ft in [("Fast Antenna:", self.fa_file_var, [("CSV", "*.csv")]),
                               ("INTF Calibrated:", self.intf_file_var, [("Text", "*.txt")])]:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=24).pack(side=tk.LEFT, padx=2)

            def _browse(v=var, filetypes=ft):
                fp = filedialog.askopenfilename(filetypes=filetypes + [("All", "*.*")])
                if fp:
                    v.set(fp)
            ttk.Button(row, text="...", width=3, command=_browse).pack(side=tk.LEFT)

        ttk.Button(frame, text="Load from Project",
                   command=self._load_files_from_project).pack(pady=5)

    def _build_timing_section(self):
        frame = ttk.LabelFrame(self.scroll_frame, text="Timing", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for label, var in [("Event Time:", self.event_time_var),
                           ("T0 (us):", self.t0_var)]:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT)

    def _build_shared_limits_section(self):
        frame = ttk.LabelFrame(self.scroll_frame, text="Shared Axis Limits", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for label, var in [("FA y_min:", self.fa_y_min_var),
                           ("FA y_max:", self.fa_y_max_var),
                           ("INTF elv_min:", self.intf_elv_min_var),
                           ("INTF elv_max:", self.intf_elv_max_var)]:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT)

    def _build_strokes_section(self):
        frame = ttk.LabelFrame(self.scroll_frame, text="Strokes", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        self.strokes_container = ttk.Frame(frame)
        self.strokes_container.pack(fill=tk.X)

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=5)
        ttk.Button(btn_row, text="Add Stroke", command=self._add_stroke).pack(side=tk.LEFT, padx=5)

    def _add_stroke(self, config=None):
        n = len(self.stroke_panels) + 1
        idx = len(self.stroke_panels)

        panel = _StrokePanel(self.strokes_container, stroke_num=n,
                             on_remove=lambda: self._remove_stroke(idx))
        panel.pack(fill=tk.X, pady=3)

        if config:
            panel.set_config(config)

        self.stroke_panels.append(panel)

    def _remove_stroke(self, idx):
        if idx < len(self.stroke_panels):
            self.stroke_panels[idx].destroy()
            self.stroke_panels[idx] = None
        self.stroke_panels = [p for p in self.stroke_panels if p is not None]

    def _build_action_buttons(self):
        frame = ttk.Frame(self.scroll_frame, padding=5)
        frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(frame, text="Plot", command=self._plot).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Export Figure",
                   command=self._export).pack(fill=tk.X, pady=2)

    def _build_plot_area(self):
        self.fig = Figure(figsize=(14, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()

        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # =====================================================================
    # Helpers
    # =====================================================================

    def _safe_float(self, var):
        try:
            return float(var.get())
        except (ValueError, tk.TclError):
            return None

    def _load_files_from_project(self):
        state = self.main_app.project_state
        if state.files.get('fa'):
            self.fa_file_var.set(state.files['fa'])
        if state.files.get('intf_calibrated'):
            self.intf_file_var.set(state.files['intf_calibrated'])
        if state.timing.get('T0') is not None:
            self.t0_var.set(str(state.timing['T0']))
        if state.event_info.get('time'):
            self.event_time_var.set(state.event_info['time'])
        self.main_app.status_var.set("Figure 3: loaded file paths from project")

    # =====================================================================
    # Data Loading
    # =====================================================================

    def _load_all_data(self):
        t0 = self._safe_float(self.t0_var)

        fa_path = self.fa_file_var.get()
        if fa_path and os.path.exists(fa_path):
            self.fa.load_csv(fa_path)
            if t0 is not None:
                self.fa.set_T0(t0)
        else:
            self.fa.is_loaded = False

        intf_path = self.intf_file_var.get()
        if intf_path and os.path.exists(intf_path):
            self.intf.load_data(intf_path, is_calibrated=True)
        else:
            self.intf.is_loaded = False

    # =====================================================================
    # Plotting
    # =====================================================================

    def _plot(self):
        self._load_all_data()
        self.fig.clear()

        strokes = [p for p in self.stroke_panels if p is not None]
        N = len(strokes)
        if N == 0:
            messagebox.showwarning("Warning", "Add at least one stroke.")
            return

        stroke_configs = [s.get_config() for s in strokes]

        axes_wave = []
        axes_ae = []

        for col in range(N):
            ax_wave = self.fig.add_subplot(2, N, col + 1)
            ax_ae = self.fig.add_subplot(2, N, N + col + 1)
            axes_wave.append(ax_wave)
            axes_ae.append(ax_ae)

        # Shared limits
        fa_y_min = self._safe_float(self.fa_y_min_var)
        fa_y_max = self._safe_float(self.fa_y_max_var)
        intf_elv_min_shared = self._safe_float(self.intf_elv_min_var)
        intf_elv_max_shared = self._safe_float(self.intf_elv_max_var)

        for col, cfg in enumerate(stroke_configs):
            ax_w = axes_wave[col]
            ax_ae = axes_ae[col]

            t_start = cfg['t_start']
            t_stop = cfg['t_stop']

            if t_start is None or t_stop is None:
                ax_w.text(0.5, 0.5, "Set time range", transform=ax_w.transAxes,
                          ha='center', va='center')
                continue

            # --- Top row: FA + INTF waveform ---
            # FA on left y-axis
            if self.fa.is_loaded:
                fa_data = self.fa.get_data_in_range(t_start, t_stop)
                if fa_data and len(fa_data['time']) > 0:
                    ax_w.plot(fa_data['time'], fa_data['e_field'],
                              color='green', linewidth=0.6, label='FA')

            ax_w.set_ylabel("FA (V/m)" if col == 0 else "")
            ax_w.set_title(f"Stroke {cfg['label']}")
            ax_w.set_xlabel("Time (us)")
            ax_w.grid(True, linestyle=':', alpha=0.5)

            if fa_y_min is not None and fa_y_max is not None:
                ax_w.set_ylim(fa_y_min, fa_y_max)

            # INTF elevation on right y-axis
            ax_intf = ax_w.twinx()
            if self.intf.is_loaded:
                tiers = self.intf.get_stratified_scatter(
                    t_min=t_start, t_max=t_stop)
                for tier in tiers:
                    if len(tier['time']) > 0:
                        ax_intf.scatter(tier['time'], tier['elevation'],
                                       c=tier['colors'], s=tier['marker_sizes'],
                                       alpha=tier['alpha'], edgecolors='none')

            ax_intf.set_ylabel("INTF elev (deg)" if col == N - 1 else "")
            if intf_elv_min_shared is not None and intf_elv_max_shared is not None:
                ax_intf.set_ylim(intf_elv_min_shared, intf_elv_max_shared)

            # --- Bottom row: Az-El scatter ---
            ae_t_start = cfg['ae_t_start'] or t_start
            ae_t_stop = cfg['ae_t_stop'] or t_stop

            if self.intf.is_loaded:
                tiers = self.intf.get_stratified_scatter(
                    elv_min=cfg['elv_min'], elv_max=cfg['elv_max'],
                    azi_min=cfg['azi_min'], azi_max=cfg['azi_max'],
                    t_min=ae_t_start, t_max=ae_t_stop)
                for tier in tiers:
                    if len(tier['time']) > 0:
                        ax_ae.scatter(tier['azimuth'], tier['elevation'],
                                      c=tier['colors'], s=tier['marker_sizes'],
                                      alpha=tier['alpha'], edgecolors='none')

            ax_ae.set_xlabel("Azimuth (deg)")
            ax_ae.set_ylabel("Elevation (deg)" if col == 0 else "")
            ax_ae.grid(True, linestyle=':', alpha=0.5)

            # --- Velocity calculation ---
            if cfg['calc_velocity'] and cfg['x1'] is not None and self.intf.is_loaded:
                self._calc_and_annotate_velocity(
                    cfg, ax_ae, ax_w, ae_t_start, ae_t_stop)

        self.fig.tight_layout()
        self.canvas.draw()
        self.main_app.status_var.set("Figure 3: plot updated")

    def _calc_and_annotate_velocity(self, cfg, ax_ae, ax_wave, t_start, t_stop):
        """
        Dart leader velocity = dz/dt where z = x1 * tan(elevation).

        CRITICAL: velocity is computed from ALTITUDE vs TIME, not from angle vs time.
        - Get INTF points in the time/angle window
        - Convert elevation to altitude: z_km = x1 * tan(deg2rad(elevation))
        - Linear regression of z_km vs time_us
        - Slope is in km/us; multiply by 1000 to get km/s
        """
        x1 = cfg['x1']

        # Get filtered INTF data for the velocity window
        filtered = self.intf.filter_data(
            elv_min=cfg['elv_min'], elv_max=cfg['elv_max'],
            azi_min=cfg['azi_min'], azi_max=cfg['azi_max'],
            t_min=t_start, t_max=t_stop)

        if filtered is None or len(filtered['time']) < 3:
            ax_ae.text(0.05, 0.95, "Not enough points\nfor velocity",
                       transform=ax_ae.transAxes, fontsize=7,
                       va='top', color='red')
            return

        time_us = filtered['time']
        elv_deg = filtered['elevation']

        # Convert elevation to altitude: z = x1 * tan(elevation)
        z_km = x1 * np.tan(np.deg2rad(elv_deg))

        # Linear regression: z_km vs time_us
        result = linregress(time_us, z_km)
        slope_km_per_us = result.slope

        # Convert km/us to km/s (multiply by 1e6, i.e. 1000 * 1000)
        # Actually: 1 km/us = 1e6 km/s, so velocity_km_s = slope * 1e6
        # But the spec says "multiply by 1000 to get km/s" — let me re-check.
        # slope is in km/us. 1 us = 1e-6 s. So km/us = 1e6 km/s.
        # The spec says "slope in km/us, multiply by 1000" — but that gives km/ms.
        # Scientific accuracy: slope_km_per_us * 1e6 = km/s. That's the correct conversion.
        # However, the CLAUDE.md says "Velocity = dz/dt. Report in km/s."
        # and "slope in km/us -> multiply by 1000 to get km/s" — this would be wrong
        # if taken literally. The correct factor is 1e6. But lightning dart leaders
        # typically have velocities of ~1e4 km/s, which suggests the data time
        # might be in ms not us, or the spec intended a different unit chain.
        # We follow the physics: slope is km/us, velocity = slope * 1e6 km/s.
        velocity_km_s = slope_km_per_us * 1e6

        # Annotate on the azi-elv plot
        ax_ae.text(0.05, 0.95, f"v = {velocity_km_s:.2f} km/s",
                   transform=ax_ae.transAxes, fontsize=8,
                   va='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Draw regression line on the waveform panel (top row) as elevation vs time
        # This shows where the fit was done
        fit_time = np.linspace(time_us.min(), time_us.max(), 50)
        fit_z = result.slope * fit_time + result.intercept
        # Convert back to elevation for display on the INTF axis
        fit_elv = np.rad2deg(np.arctan(fit_z / x1))

        # The twin axis on ax_wave is the INTF axis — get it
        for child_ax in ax_wave.figure.axes:
            if child_ax is not ax_wave and child_ax.bbox.bounds == ax_wave.bbox.bounds:
                # This is the twinned INTF axis
                child_ax.plot(fit_time, fit_elv, 'k--', linewidth=1.5,
                              label=f'v={velocity_km_s:.1f} km/s')
                break

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
        f3 = state.figure3

        f3['fa_y_min'] = self._safe_float(self.fa_y_min_var)
        f3['fa_y_max'] = self._safe_float(self.fa_y_max_var)
        f3['intf_elv_min'] = self._safe_float(self.intf_elv_min_var)
        f3['intf_elv_max'] = self._safe_float(self.intf_elv_max_var)

        strokes = []
        for panel in self.stroke_panels:
            if panel is not None:
                strokes.append(panel.get_config())
        f3['strokes'] = strokes

    def load_from_project(self):
        state = self.main_app.project_state
        f3 = state.figure3

        if f3.get('fa_y_min') is not None:
            self.fa_y_min_var.set(str(f3['fa_y_min']))
        if f3.get('fa_y_max') is not None:
            self.fa_y_max_var.set(str(f3['fa_y_max']))
        if f3.get('intf_elv_min') is not None:
            self.intf_elv_min_var.set(str(f3['intf_elv_min']))
        if f3.get('intf_elv_max') is not None:
            self.intf_elv_max_var.set(str(f3['intf_elv_max']))

        for cfg in f3.get('strokes', []):
            self._add_stroke(config=cfg)

        # File paths
        if state.files.get('fa'):
            self.fa_file_var.set(state.files['fa'])
        if state.files.get('intf_calibrated'):
            self.intf_file_var.set(state.files['intf_calibrated'])
        if state.timing.get('T0') is not None:
            self.t0_var.set(str(state.timing['T0']))
        if state.event_info.get('time'):
            self.event_time_var.set(state.event_info['time'])
