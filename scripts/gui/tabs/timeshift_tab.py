"""
TGF Analysis Tool - Timeshift Tab
==================================
Calculates the timing offset (timeshift) between TASD surface detectors
and the INTF/camera reference using LMA and optionally INTF data.

Automates the calc_iterative workflow from the standalone scripts.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import numpy as np
import os
import csv
import heapq

# Matplotlib with Tk backend
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from config import TASD_COORDS_FILENAME
from data_handlers import TASDHandler, LMAHandler, InterferometerHandler
from analysis.timeshift import TimeshiftCalculator


class TimeshiftTab(ttk.Frame):
    """
    Timeshift calculation tab.

    Left panel: controls, parameters, run button, advanced options.
    Right panel: formatted text output matching standalone calc_iterative.
    """

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app

        # File paths
        self.sd_dir_var = tk.StringVar()
        self.lma_file_var = tk.StringVar()
        self.intf_file_var = tk.StringVar()

        # Event timing
        self.event_time_var = tk.StringVar(value="HH:MM:SS")
        self.t0_var = tk.StringVar(value="0")

        # Calculation mode
        self.mode_var = tk.StringVar(value="intf_lma")

        # Parameters
        self.initial_alt_var = tk.StringVar(value="3.0")
        self.iteration_limit_var = tk.StringVar(value="10")
        self.sd_timing_error_var = tk.StringVar(value="0.04")
        self.lma_range_var = tk.StringVar(value="1000")
        self.lma_chi_limit_var = tk.StringVar(value="2.0")
        self.intf_range_var = tk.StringVar(value="9.4")
        self.intf_elev_min_var = tk.StringVar(value="1.0")
        self.intf_elev_max_var = tk.StringVar(value="11.0")
        self.intf_azi_min_var = tk.StringVar(value="240.0")
        self.intf_azi_max_var = tk.StringVar(value="302.0")
        self.trig_num_var = tk.StringVar(value="1")

        # Cross-second offset
        self.cross_second_offset_var = tk.IntVar(value=0)

        # Results storage
        self.calc_results = None
        self._show_individual = False

        # Cached loaded data for diagnostics
        self._loaded_tasd = None
        self._loaded_lma = None
        self._loaded_intf = None

        # INTF-specific widgets (for greying out)
        self._intf_widgets = []

        # Result labels
        self.result_vars = {
            'n_detectors': tk.StringVar(value="--"),
            'mean_dt': tk.StringVar(value="--"),
            'std_dt': tk.StringVar(value="--"),
            'convergence': tk.StringVar(value="--"),
            'iterations': tk.StringVar(value="--"),
        }

        self._build_ui()

    # =========================================================================
    # UI Construction
    # =========================================================================

    def _build_ui(self):
        """Build the tab UI."""
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - Controls (half the screen)
        self.control_frame = ttk.Frame(self.paned)
        self.paned.add(self.control_frame, weight=1)

        # Right panel - Text output
        self.output_frame = ttk.Frame(self.paned)
        self.paned.add(self.output_frame, weight=1)

        self._build_control_panel()
        self._build_output_panel()

    def _build_control_panel(self):
        """Build the left control panel with scrolling."""
        canvas = tk.Canvas(self.control_frame)
        scrollbar = ttk.Scrollbar(self.control_frame, orient="vertical",
                                  command=canvas.yview)
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

        # Add sections
        self._build_file_section()
        self._build_timing_section()
        self._build_mode_section()
        self._build_params_section()
        self._build_run_section()
        self._build_results_section()
        self._build_advanced_section()

    def _build_file_section(self):
        """Build file input section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Files", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # SD Directory
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="SD Dir:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.sd_dir_var, width=20).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse",
                   command=self._browse_sd_dir).pack(side=tk.LEFT)

        # LMA file
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="LMA File:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.lma_file_var, width=20).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse",
                   command=self._browse_lma_file).pack(side=tk.LEFT)

        # INTF calibrated file
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="INTF Cal:").pack(side=tk.LEFT)
        self._intf_file_entry = ttk.Entry(row, textvariable=self.intf_file_var,
                                          width=20)
        self._intf_file_entry.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        self._intf_browse_btn = ttk.Button(row, text="Browse",
                                           command=self._browse_intf_file)
        self._intf_browse_btn.pack(side=tk.LEFT)
        self._intf_widgets.extend([self._intf_file_entry, self._intf_browse_btn])

        # Load from project
        ttk.Button(frame, text="Load from Project",
                   command=self._load_paths_from_project).pack(pady=4)

    def _build_timing_section(self):
        """Build event timing section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Event Timing",
                               padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Event Time:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.event_time_var, width=10).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(row, text="Sync from Home",
                   command=self._sync_event_time).pack(side=tk.LEFT, padx=2)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="T0 (us):").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.t0_var, width=10).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(row, text="Sync T0",
                   command=self._sync_t0).pack(side=tk.LEFT, padx=2)

    def _build_mode_section(self):
        """Build calculation mode selection."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Calculation Mode",
                               padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Radiobutton(frame, text="INTF + LMA (full iterative)",
                        variable=self.mode_var, value="intf_lma",
                        command=self._on_mode_change).pack(anchor='w')
        ttk.Radiobutton(frame, text="LMA Only (no INTF)",
                        variable=self.mode_var, value="lma_only",
                        command=self._on_mode_change).pack(anchor='w')

    def _build_params_section(self):
        """Build parameters section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Parameters",
                               padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        self._add_param_row(frame, "Initial altitude (km):",
                            self.initial_alt_var)
        self._add_param_row(frame, "Iteration limit:",
                            self.iteration_limit_var)
        self._add_param_row(frame, "SD timing error (us):",
                            self.sd_timing_error_var)
        self._add_param_row(frame, "LMA time range (us):",
                            self.lma_range_var)
        self._add_param_row(frame, "LMA chi^2 limit:",
                            self.lma_chi_limit_var)
        self._add_param_row(frame, "Trigger number:",
                            self.trig_num_var)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=4)
        lbl = ttk.Label(frame, text="INTF Parameters (full mode only):",
                        font=('Helvetica', 8, 'italic'))
        lbl.pack(anchor='w')

        # INTF range with density preview button
        intf_range_row = ttk.Frame(frame)
        intf_range_row.pack(fill=tk.X, pady=1)
        ttk.Label(intf_range_row, text="INTF time range (us):", width=22).pack(side=tk.LEFT)
        w1 = ttk.Entry(intf_range_row, textvariable=self.intf_range_var, width=8)
        w1.pack(side=tk.LEFT, padx=2)
        self._intf_density_btn = ttk.Button(intf_range_row, text="Density",
                                            command=self._show_intf_density_preview)
        self._intf_density_btn.pack(side=tk.LEFT, padx=4)
        self._intf_widgets.append(w1)
        self._intf_widgets.append(self._intf_density_btn)

        w2 = self._add_param_row(frame, "INTF elev min (deg):",
                                 self.intf_elev_min_var)
        w3 = self._add_param_row(frame, "INTF elev max (deg):",
                                 self.intf_elev_max_var)
        w4 = self._add_param_row(frame, "INTF azi min (deg):",
                                 self.intf_azi_min_var)
        w5 = self._add_param_row(frame, "INTF azi max (deg):",
                                 self.intf_azi_max_var)
        self._intf_widgets.extend([w2, w3, w4, w5])

    def _add_param_row(self, parent, label_text, var):
        """Add a label + entry row. Returns the entry widget."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label_text, width=22).pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=var, width=8)
        entry.pack(side=tk.LEFT, padx=2)
        return entry

    def _build_run_section(self):
        """Build the run button."""
        frame = ttk.Frame(self.scrollable_frame)
        frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(frame, text="Calculate Timeshift",
                   command=self._run_calculation).pack(fill=tk.X)

    def _build_results_section(self):
        """Build results summary section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Results Summary",
                               padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for key, label in [('n_detectors', 'Detectors:'),
                           ('mean_dt', 'Mean dt (us):'),
                           ('std_dt', 'Std dt (us):'),
                           ('convergence', 'Convergence:'),
                           ('iterations', 'Iterations:')]:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
            ttk.Label(row, textvariable=self.result_vars[key]).pack(
                side=tk.LEFT)

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="Export Results",
                   command=self._export_txt).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Save to Project",
                   command=self._save_timeshift_to_project).pack(
                       side=tk.LEFT, padx=2)

    def _build_advanced_section(self):
        """Build collapsible advanced options section."""
        # Toggle button
        self._adv_visible = False
        self._adv_toggle_btn = ttk.Button(
            self.scrollable_frame, text="+ Advanced Options",
            command=self._toggle_advanced)
        self._adv_toggle_btn.pack(fill=tk.X, padx=5, pady=(5, 0))

        # Container frame (hidden by default)
        self._adv_frame = ttk.LabelFrame(self.scrollable_frame,
                                         text="Advanced Options", padding=5)
        # Not packed initially — toggled on/off

        # Cross-second offset
        cs_frame = ttk.LabelFrame(self._adv_frame,
                                  text="Cross-Second Offset", padding=5)
        cs_frame.pack(fill=tk.X, padx=2, pady=4)

        ttk.Label(cs_frame,
                  text="Shift SD trigger times for flashes spanning "
                       "clock-second boundaries:",
                  wraplength=400, font=('Helvetica', 8, 'italic')).pack(
                      anchor='w', pady=(0, 4))

        btn_row = ttk.Frame(cs_frame)
        btn_row.pack(fill=tk.X)
        for offset_sec, label in [(-2, "-2s"), (-1, "-1s"),
                                   (0, "0"), (1, "+1s"), (2, "+2s")]:
            offset_us = offset_sec * 1_000_000
            btn = ttk.Button(
                btn_row, text=label, width=5,
                command=lambda o=offset_us: self._set_cross_second_offset(o))
            btn.pack(side=tk.LEFT, padx=2, expand=True)

        self.offset_status_var = tk.StringVar(value="Offset: 0 us (0 s)")
        ttk.Label(cs_frame, textvariable=self.offset_status_var,
                  font=('Courier', 9)).pack(anchor='w', pady=(4, 0))

        # Data span diagnostic
        diag_frame = ttk.LabelFrame(self._adv_frame,
                                    text="Data Span Diagnostic", padding=5)
        diag_frame.pack(fill=tk.X, padx=2, pady=4)

        self.diag_fig = Figure(figsize=(5, 1.2), dpi=100)
        self.diag_ax = self.diag_fig.add_subplot(111)
        self.diag_canvas = FigureCanvasTkAgg(self.diag_fig, master=diag_frame)
        self.diag_canvas.get_tk_widget().pack(fill=tk.X)

        self.diag_warning_var = tk.StringVar(value="")
        ttk.Label(diag_frame, textvariable=self.diag_warning_var,
                  foreground='red', font=('Helvetica', 8)).pack(anchor='w')

        self.diag_ax.set_yticks([])
        self.diag_ax.set_xlabel("Time (us)", fontsize=7)
        self.diag_ax.tick_params(labelsize=7)
        self.diag_ax.text(0.5, 0.5, "Load data to see spans",
                          ha='center', va='center',
                          transform=self.diag_ax.transAxes,
                          fontsize=8, color='gray')
        self.diag_fig.tight_layout(pad=0.5)
        self.diag_canvas.draw()

        # Detector scatter plots
        plot_frame = ttk.LabelFrame(self._adv_frame,
                                    text="Detector Plots", padding=5)
        plot_frame.pack(fill=tk.X, padx=2, pady=4)

        self.adv_fig = Figure(figsize=(5, 4), dpi=100)
        self.ax_dt = self.adv_fig.add_subplot(211)
        self.ax_z = self.adv_fig.add_subplot(212)
        self.adv_canvas = FigureCanvasTkAgg(self.adv_fig, master=plot_frame)
        self.adv_canvas.get_tk_widget().pack(fill=tk.X)

        self.ax_dt.set_title("Per-Detector Timeshift", fontsize=9)
        self.ax_dt.set_xlabel("Detector", fontsize=8)
        self.ax_dt.set_ylabel("dt (us)", fontsize=8)
        self.ax_z.set_title("Source Altitude", fontsize=9)
        self.ax_z.set_xlabel("Detector", fontsize=8)
        self.ax_z.set_ylabel("z (km)", fontsize=8)
        self.adv_fig.tight_layout()
        self.adv_canvas.draw()

    def _build_output_panel(self):
        """Build the right panel with formatted text output."""
        # Header
        header = ttk.Frame(self.output_frame)
        header.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(header, text="Calculation Output",
                  font=('Helvetica', 11, 'bold')).pack(side=tk.LEFT)

        btn_frame = ttk.Frame(header)
        btn_frame.pack(side=tk.RIGHT)

        self._show_indiv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_frame, text="Show Individual Detectors",
                        variable=self._show_indiv_var,
                        command=self._refresh_output_text).pack(
                            side=tk.LEFT, padx=4)

        ttk.Button(btn_frame, text="Copy",
                   command=self._copy_output).pack(side=tk.LEFT, padx=2)

        # Text widget for output (using tk.Text for tag support)
        text_frame = ttk.Frame(self.output_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.output_text = tk.Text(text_frame, font=('Consolas', 10),
                                   wrap=tk.NONE, state=tk.DISABLED,
                                   bg='#1e1e1e', fg='#d4d4d4',
                                   insertbackground='white',
                                   selectbackground='#264f78',
                                   padx=10, pady=10)

        # Scrollbars
        y_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL,
                                 command=self.output_text.yview)
        x_scroll = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL,
                                 command=self.output_text.xview)
        self.output_text.configure(yscrollcommand=y_scroll.set,
                                   xscrollcommand=x_scroll.set)

        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure text tags for styled output
        self.output_text.tag_configure('title',
                                       font=('Consolas', 12, 'bold'),
                                       foreground='#569cd6')
        self.output_text.tag_configure('section_header',
                                       font=('Consolas', 10, 'bold'),
                                       foreground='#4ec9b0')
        self.output_text.tag_configure('success',
                                       foreground='#6a9955',
                                       font=('Consolas', 10, 'bold'))
        self.output_text.tag_configure('warning',
                                       foreground='#ce9178',
                                       font=('Consolas', 10, 'bold'))
        self.output_text.tag_configure('label',
                                       foreground='#9cdcfe')
        self.output_text.tag_configure('value',
                                       foreground='#ce9178')
        self.output_text.tag_configure('error_val',
                                       foreground='#808080')
        self.output_text.tag_configure('percent',
                                       foreground='#646464')
        self.output_text.tag_configure('det_name',
                                       foreground='#dcdcaa',
                                       font=('Consolas', 10, 'bold'))
        self.output_text.tag_configure('separator',
                                       foreground='#404040')
        self.output_text.tag_configure('dim',
                                       foreground='#606060')

        # Initial message
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END,
                                "Run a calculation to see results here.\n",
                                'dim')
        self.output_text.configure(state=tk.DISABLED)

    # =========================================================================
    # UI Callbacks
    # =========================================================================

    def _on_mode_change(self):
        """Enable/disable INTF-specific widgets based on mode."""
        state = 'normal' if self.mode_var.get() == 'intf_lma' else 'disabled'
        for widget in self._intf_widgets:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def _toggle_advanced(self):
        """Toggle advanced options visibility."""
        if self._adv_visible:
            self._adv_frame.pack_forget()
            self._adv_toggle_btn.configure(text="+ Advanced Options")
            self._adv_visible = False
        else:
            self._adv_frame.pack(fill=tk.X, padx=5, pady=(0, 5),
                                 after=self._adv_toggle_btn)
            self._adv_toggle_btn.configure(text="- Advanced Options")
            self._adv_visible = True

    def _browse_sd_dir(self):
        directory = filedialog.askdirectory(title="Select TASD Data Directory")
        if directory:
            self.sd_dir_var.set(directory)

    def _browse_lma_file(self):
        filepath = filedialog.askopenfilename(
            title="Select LMA Data File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filepath:
            self.lma_file_var.set(filepath)

    def _browse_intf_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Calibrated INTF File",
            filetypes=[("Data files", "*.dat *.txt"), ("All files", "*.*")])
        if filepath:
            self.intf_file_var.set(filepath)

    def _load_paths_from_project(self):
        """Pull file paths from project state."""
        state = self.main_app.project_state
        if state.files.get('sd_directory'):
            self.sd_dir_var.set(state.files['sd_directory'])
        if state.files.get('lma'):
            self.lma_file_var.set(state.files['lma'])
        if state.files.get('intf_calibrated'):
            self.intf_file_var.set(state.files['intf_calibrated'])
        ei = state.event_info
        if ei.get('time'):
            self.event_time_var.set(ei['time'])
        self.main_app.status_var.set("Timeshift: loaded paths from project")

    def _sync_event_time(self):
        """Sync event time from the home tab."""
        try:
            home = self.main_app.home_tab
            if hasattr(home, 'event_time_var'):
                val = home.event_time_var.get()
                if val and val != "HH:MM:SS":
                    self.event_time_var.set(val)
        except Exception:
            pass

    def _sync_t0(self):
        """Sync T0 from project state timing."""
        try:
            t0 = self.main_app.project_state.timing.get('T0', 0)
            self.t0_var.set(str(t0))
        except Exception:
            pass

    def _set_cross_second_offset(self, offset_us):
        """Set the cross-second offset and refresh diagnostic."""
        self.cross_second_offset_var.set(offset_us)
        sec = offset_us / 1_000_000
        self.offset_status_var.set(
            f"Offset: {offset_us:+,} us ({sec:+.0f} s)")
        self._refresh_diagnostic()

    # =========================================================================
    # Auto-find TASD coordinates file
    # =========================================================================

    def _find_tasd_coords(self):
        """Auto-find TASD coordinates file by searching common locations."""
        possible_paths = []

        sd_dir = self.sd_dir_var.get()
        if sd_dir:
            possible_paths.append(os.path.join(sd_dir, TASD_COORDS_FILENAME))
            possible_paths.append(os.path.join(os.path.dirname(sd_dir),
                                               TASD_COORDS_FILENAME))

        possible_paths.append(TASD_COORDS_FILENAME)
        possible_paths.append(os.path.join(os.getcwd(), TASD_COORDS_FILENAME))

        script_dir = os.path.dirname(os.path.abspath(__file__))
        for up in ['', '..', os.path.join('..', '..'),
                    os.path.join('..', '..', '..')]:
            possible_paths.append(os.path.join(script_dir, up,
                                               TASD_COORDS_FILENAME))

        if 'gui' in script_dir:
            gui_parent = script_dir.split('gui')[0]
            possible_paths.append(os.path.join(gui_parent,
                                               TASD_COORDS_FILENAME))

        for p in possible_paths:
            if os.path.isfile(p):
                return os.path.abspath(p)
        return None

    # =========================================================================
    # Data Span Diagnostic
    # =========================================================================

    def _refresh_diagnostic(self):
        """Refresh the data span timeline figure."""
        self.diag_ax.clear()
        self.diag_warning_var.set("")

        offset = self.cross_second_offset_var.get()
        bars = []

        if self._loaded_intf is not None:
            t = self._loaded_intf.time_array
            if len(t) > 0:
                bars.append(("INTF", t[0], t[-1], '#1f77b4'))

        if self._loaded_lma is not None and len(self._loaded_lma.time) > 0:
            time_str = self.event_time_var.get().strip()
            try:
                parts = time_str.split(':')
                evt_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            except (ValueError, IndexError):
                evt_sec = 0
            lma_t = self._loaded_lma.time
            lma_start_us = (lma_t[0] - evt_sec) * 1e6
            lma_end_us = (lma_t[-1] - evt_sec) * 1e6
            bars.append(("LMA", lma_start_us, lma_end_us, '#2ca02c'))

        if self._loaded_tasd is not None and self._loaded_tasd.detectors:
            sd_trigs = [d['sd_trig'] + offset
                        for d in self._loaded_tasd.detectors]
            sd_min = min(sd_trigs)
            sd_max = max(sd_trigs)
            bars.append(("SD", sd_min, sd_max, '#d62728'))

        if not bars:
            self.diag_ax.text(0.5, 0.5, "No data loaded",
                              ha='center', va='center',
                              transform=self.diag_ax.transAxes,
                              fontsize=8, color='gray')
            self.diag_ax.set_yticks([])
            self.diag_fig.tight_layout(pad=0.5)
            self.diag_canvas.draw()
            return

        for i, (label, t_start, t_end, color) in enumerate(bars):
            width = max(t_end - t_start, 0.1)
            self.diag_ax.barh(i, width, left=t_start, height=0.6,
                              color=color, alpha=0.7, edgecolor='black',
                              linewidth=0.5)

        self.diag_ax.set_yticks(range(len(bars)))
        self.diag_ax.set_yticklabels([b[0] for b in bars], fontsize=7)
        self.diag_ax.set_xlabel("Time (us)", fontsize=7)
        self.diag_ax.tick_params(labelsize=7)

        intf_bar = next((b for b in bars if b[0] == "INTF"), None)
        sd_bar = next((b for b in bars if b[0] == "SD"), None)
        if intf_bar and sd_bar:
            sd_mid = (sd_bar[1] + sd_bar[2]) / 2
            if sd_mid < intf_bar[1] or sd_mid > intf_bar[2]:
                self.diag_warning_var.set(
                    "WARNING: SD triggers outside INTF time range! "
                    "Try adjusting cross-second offset.")

        self.diag_fig.tight_layout(pad=0.5)
        self.diag_canvas.draw()

    # =========================================================================
    # INTF Source Density Preview
    # =========================================================================

    def _show_intf_density_preview(self):
        """Show popup histogram of INTF point density near computed tc values."""
        if self._loaded_intf is None:
            messagebox.showinfo("Info",
                                "Load INTF data first (run a calculation).")
            return

        if self._loaded_tasd is None or not self._loaded_tasd.detectors:
            messagebox.showinfo("Info",
                                "Load TASD data first (run a calculation).")
            return

        intf_times = self._loaded_intf.time_array
        if len(intf_times) == 0:
            messagebox.showinfo("Info", "INTF data is empty.")
            return

        offset = self.cross_second_offset_var.get()
        tc_values = [d['sd_trig'] + offset
                     for d in self._loaded_tasd.detectors]

        try:
            intf_range = float(self.intf_range_var.get())
        except ValueError:
            intf_range = 9.4

        median_tc = np.median(tc_values)
        search_half = 500.0
        mask = (intf_times >= median_tc - search_half) & \
               (intf_times <= median_tc + search_half)
        nearby_times = intf_times[mask]

        if len(nearby_times) == 0:
            messagebox.showinfo("Info",
                                f"No INTF points within +/-{search_half} us "
                                f"of median tc={median_tc:.1f} us.")
            return

        fig_popup, ax = plt.subplots(figsize=(8, 5))
        ax.hist(nearby_times, bins=100, color='#1f77b4', alpha=0.7,
                edgecolor='black', linewidth=0.3)

        for i, tc in enumerate(tc_values):
            ax.axvline(x=tc, color='red', linestyle='-', alpha=0.5,
                       linewidth=0.8)

        ax.axvline(x=median_tc - intf_range, color='green', linestyle='--',
                   linewidth=1.5, label=f'-intfRange ({intf_range} us)')
        ax.axvline(x=median_tc + intf_range, color='green', linestyle='--',
                   linewidth=1.5, label=f'+intfRange ({intf_range} us)')
        ax.axvline(x=median_tc, color='orange', linestyle='-', linewidth=1.5,
                   label=f'Median tc = {median_tc:.1f} us')

        ax.set_xlabel("INTF Time (us)")
        ax.set_ylabel("Point Count")
        ax.set_title("INTF Source Density near SD Trigger Times")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.4)
        fig_popup.tight_layout()
        plt.show(block=False)

    # =========================================================================
    # Calculation
    # =========================================================================

    def _run_calculation(self):
        """Run the timeshift calculation."""
        # 1. Parse event time
        time_str = self.event_time_var.get().strip()
        try:
            parts = time_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2])
        except (ValueError, IndexError):
            messagebox.showerror("Error",
                                 "Invalid event time. Use HH:MM:SS format.")
            return

        mode = self.mode_var.get()
        offset = self.cross_second_offset_var.get()
        print(f"\n[Timeshift] Starting calculation: mode={mode}, "
              f"time={hour:02d}:{minute:02d}:{second:02d}, "
              f"cross_second_offset={offset} us")

        # 2. Validate paths
        sd_dir = self.sd_dir_var.get()
        if not sd_dir or not os.path.isdir(sd_dir):
            messagebox.showerror("Error", "Please select a valid SD directory.")
            return

        lma_path = self.lma_file_var.get()
        if not lma_path or not os.path.isfile(lma_path):
            messagebox.showerror("Error", "Please select a valid LMA file.")
            return

        if mode == 'intf_lma':
            intf_path = self.intf_file_var.get()
            if not intf_path or not os.path.isfile(intf_path):
                messagebox.showerror("Error",
                                     "Please select a valid calibrated INTF file.")
                return

        # 3. Load TASD data
        self.main_app.status_var.set("Loading TASD data...")
        self.update_idletasks()

        tasd_handler = TASDHandler()
        coords_path = self._find_tasd_coords()
        if coords_path:
            print(f"[Timeshift] TASD coords file: {coords_path}")
            tasd_handler.load_coordinates(coords_path)
        else:
            messagebox.showerror("Error",
                                 f"Could not find {TASD_COORDS_FILENAME}.")
            return

        dirname = os.path.basename(sd_dir)
        dir_parts = dirname.split('_')
        if len(dir_parts) >= 3:
            sd_time_str = dir_parts[2]
        else:
            sd_time_str = f"{hour:02d}{minute:02d}{second:02d}"
        print(f"[Timeshift] SD time string: {sd_time_str}, dir: {dirname}")

        trig_num = int(self.trig_num_var.get())
        success = tasd_handler.load_directory(sd_dir, sd_time_str,
                                              trig_num=trig_num)
        if not success:
            messagebox.showerror("Error",
                                 "No TASD detectors loaded. Check SD directory "
                                 "and event time.")
            return

        print(f"[Timeshift] TASD loaded: {len(tasd_handler.detectors)} detectors")
        for det in tasd_handler.detectors:
            print(f"  {det['detector_id']}: lat={det.get('lat')}, "
                  f"lon={det.get('lon')}, alt={det.get('alt')}, "
                  f"vem={det['vem']:.2f}, trig_time={det['trig_time']}, "
                  f"sd_trig={det['sd_trig']}, "
                  f"time[0]={det['time'][0]:.1f}, "
                  f"time[-1]={det['time'][-1]:.1f}")

        # Apply cross-second offset
        if offset != 0:
            print(f"[Timeshift] Applying cross-second offset: {offset} us")
            for det in tasd_handler.detectors:
                det['sd_trig'] += offset

        # 4. Load LMA data
        self.main_app.status_var.set("Loading LMA data...")
        self.update_idletasks()

        lma_handler = LMAHandler()
        skip = 2 if 'cut' in os.path.basename(lma_path).lower() else 57
        print(f"[Timeshift] Loading LMA: {lma_path}, skip_header={skip}")
        if not lma_handler.load_data(lma_path, skip_header=skip):
            messagebox.showerror("Error", "Failed to load LMA data.")
            return

        print(f"[Timeshift] LMA loaded: {len(lma_handler.time)} sources, "
              f"time range: {lma_handler.time[0]:.6f} - "
              f"{lma_handler.time[-1]:.6f} sec of day")

        event_time_sec = hour * 3600 + minute * 60 + second
        print(f"[Timeshift] Event time in sec of day: {event_time_sec}")
        lma_test = lma_handler.filter_by_time(event_time_sec, 1000, 2.0)
        print(f"[Timeshift] LMA sources within 1000us of event time: "
              f"{lma_test['count'] if lma_test else 'None'}")

        # 5. Load INTF data (if needed)
        intf_handler = None
        if mode == 'intf_lma':
            self.main_app.status_var.set("Loading INTF data...")
            self.update_idletasks()

            intf_handler = InterferometerHandler()
            print(f"[Timeshift] Loading INTF: {intf_path}")
            if not intf_handler.load_data(intf_path, is_calibrated=True,
                                          skip_header=2):
                messagebox.showerror("Error",
                                     "Failed to load calibrated INTF data.")
                return
            print(f"[Timeshift] INTF loaded: {len(intf_handler.time_array)} points, "
                  f"time range: {intf_handler.time_array[0]:.1f} - "
                  f"{intf_handler.time_array[-1]:.1f} us")

        # Cache loaded data for diagnostics
        self._loaded_tasd = tasd_handler
        self._loaded_lma = lma_handler
        self._loaded_intf = intf_handler
        self._refresh_diagnostic()

        # 6. Set up calculator
        self.main_app.status_var.set("Calculating timeshift...")
        self.update_idletasks()

        calc = TimeshiftCalculator()
        calc.set_parameters(
            initial_altitude=float(self.initial_alt_var.get()),
            iteration_limit=int(self.iteration_limit_var.get()),
            sd_timing_error=float(self.sd_timing_error_var.get()),
            lma_range=float(self.lma_range_var.get()),
            lma_chi_limit=float(self.lma_chi_limit_var.get()),
            intf_range=float(self.intf_range_var.get()),
            intf_elev_min=float(self.intf_elev_min_var.get()),
            intf_elev_max=float(self.intf_elev_max_var.get()),
            intf_azi_min=float(self.intf_azi_min_var.get()),
            intf_azi_max=float(self.intf_azi_max_var.get()),
        )
        print(f"[Timeshift] Parameters: alt={calc.initial_altitude}, "
              f"lma_range={calc.lma_range}, chi={calc.lma_chi_limit}, "
              f"intf_range={calc.intf_range}, "
              f"azi=[{calc.intf_azi_min}, {calc.intf_azi_max}], "
              f"elv=[{calc.intf_elev_min}, {calc.intf_elev_max}]")

        # 7. Run calculation
        try:
            if mode == 'intf_lma':
                result = calc.calculate_with_intf(
                    tasd_handler, lma_handler, intf_handler,
                    hour, minute, second, trig_num=trig_num)
            else:
                result = calc.calculate_lma_only(
                    tasd_handler, lma_handler, hour, minute, second)
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Calculation Error", str(e))
            self.main_app.status_var.set("Timeshift calculation failed")
            return

        # 8. Handle results
        print(f"[Timeshift] Result: success={result.get('success')}, "
              f"detectors={len(result.get('detectors', []))}, "
              f"iterations={result.get('iterations', 'N/A')}")
        if result.get('detectors'):
            for d in result['detectors']:
                print(f"  {d['detector']}: dt={d['dt']:.4f}, z={d.get('z', 'N/A')}")
        else:
            print("[Timeshift] WARNING: No detector results returned!")

        if not result.get('success', False) and 'error' in result:
            messagebox.showerror("Calculation Error", result['error'])
            self.main_app.status_var.set("Timeshift calculation failed")
            return

        # Store extra context for output formatting
        result['_mode'] = mode
        result['_event_time'] = f"{hour:02d}:{minute:02d}:{second:02d}"
        result['_cross_second_offset'] = offset
        result['_lma_handler'] = lma_handler
        result['_intf_handler'] = intf_handler
        result['_tasd_handler'] = tasd_handler
        result['_calc'] = calc

        self.calc_results = result
        self._populate_results(result)
        self._update_advanced_plots(result, mode)
        self._refresh_output_text()

        n = result.get('summary', {}).get('n_detectors', 0)
        mean_dt = result.get('summary', {}).get('mean', {}).get('dt', 0)
        self.main_app.status_var.set(
            f"Timeshift complete: {n} detectors, mean dt = {mean_dt:.3f} us")

    def _populate_results(self, result):
        """Fill in the results summary labels."""
        summary = result.get('summary', {})
        detectors = result.get('detectors', [])

        self.result_vars['n_detectors'].set(str(len(detectors)))

        mean_info = summary.get('mean', {})
        std_info = summary.get('std', {})

        if 'dt' in mean_info:
            self.result_vars['mean_dt'].set(f"{mean_info['dt']:.4f}")
        if 'dt' in std_info:
            self.result_vars['std_dt'].set(f"{std_info['dt']:.4f}")

        if 'iterations' in result:
            self.result_vars['iterations'].set(str(result['iterations']))
            converged = result.get('success', False)
            self.result_vars['convergence'].set(
                "Yes" if converged else "No")
        else:
            self.result_vars['iterations'].set("N/A (LMA only)")
            self.result_vars['convergence'].set("N/A (LMA only)")

    def _update_advanced_plots(self, result, mode):
        """Update the detector scatter plots in Advanced Options."""
        self.ax_dt.clear()
        self.ax_z.clear()

        detectors = result.get('detectors', [])
        if not detectors:
            self.adv_canvas.draw()
            return

        det_ids = [d['detector'] for d in detectors]
        dts = [d['dt'] for d in detectors]
        zs = [d.get('z', 0) for d in detectors]
        x = np.arange(len(det_ids))

        mean_dt = result.get('summary', {}).get('mean', {}).get('dt', 0)
        sd_err = float(self.sd_timing_error_var.get())

        self.ax_dt.errorbar(x, dts, yerr=sd_err, fmt='o', color='magenta',
                            capsize=4, markersize=5, label='dt per detector')
        self.ax_dt.axhline(y=mean_dt, color='k', linestyle='--', linewidth=1,
                           label=f'Mean = {mean_dt:.3f} us')
        self.ax_dt.set_xticks(x)
        self.ax_dt.set_xticklabels(det_ids, rotation=45, ha='right',
                                    fontsize=7)
        self.ax_dt.set_xlabel("Detector", fontsize=8)
        self.ax_dt.set_ylabel("dt (us)", fontsize=8)
        self.ax_dt.set_title("Per-Detector Timeshift", fontsize=9)
        self.ax_dt.legend(fontsize=7)
        self.ax_dt.grid(True, linestyle=':', alpha=0.5)

        if mode == 'intf_lma' and any(d.get('z') is not None for d in detectors):
            mean_z = result.get('summary', {}).get('mean', {}).get('z', 0)
            self.ax_z.scatter(x, zs, color='blue', s=30, zorder=3)
            self.ax_z.axhline(y=mean_z, color='k', linestyle='--',
                              linewidth=1, label=f'Mean = {mean_z:.3f} km')
            self.ax_z.set_xticks(x)
            self.ax_z.set_xticklabels(det_ids, rotation=45, ha='right',
                                      fontsize=7)
            self.ax_z.set_xlabel("Detector", fontsize=8)
            self.ax_z.set_ylabel("z (km)", fontsize=8)
            self.ax_z.set_title("Source Altitude", fontsize=9)
            self.ax_z.legend(fontsize=7)
            self.ax_z.grid(True, linestyle=':', alpha=0.5)
        else:
            self.ax_z.text(0.5, 0.5, "N/A in LMA-only mode",
                           ha='center', va='center',
                           transform=self.ax_z.transAxes, fontsize=9,
                           color='gray')
            self.ax_z.set_title("Source Altitude", fontsize=9)

        self.adv_fig.tight_layout()
        self.adv_canvas.draw()

    # =========================================================================
    # Right Panel Output Formatting
    # =========================================================================

    def _refresh_output_text(self):
        """Refresh the right-panel text output with styled formatting."""
        if not self.calc_results or not self.calc_results.get('detectors'):
            return

        show_indiv = self._show_indiv_var.get()
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self._write_styled_output(show_individual=show_indiv)
        self.output_text.configure(state=tk.DISABLED)

    def _ins(self, text, tag=None):
        """Helper: insert text with optional tag into output_text."""
        if tag:
            self.output_text.insert(tk.END, text, tag)
        else:
            self.output_text.insert(tk.END, text)

    def _write_val_line(self, label, val, err=None, pct=None):
        """Write a formatted value line:  label=value +/- err  or  pct%"""
        self._ins(f" {label:>9s}=", 'label')
        self._ins(f"{val}", 'value')
        if err is not None:
            self._ins(f" +/- ", 'dim')
            self._ins(f"{err}", 'error_val')
        if pct is not None:
            self._ins(f" or ", 'dim')
            self._ins(f"{pct:.2f} %", 'percent')
        self._ins("\n")

    def _pct(self, val, err):
        """Calculate percentage error, safe for zero values."""
        if val == 0:
            return None
        return abs(100 * err / val)

    def _write_styled_output(self, show_individual=False):
        """Write styled output to the text widget using tags."""
        result = self.calc_results
        detectors = result.get('detectors', [])
        summary = result.get('summary', {})
        mode = result.get('_mode', 'intf_lma')
        calc = result.get('_calc')
        lma_handler = result.get('_lma_handler')
        intf_handler = result.get('_intf_handler')
        has_errors = 'dt_err' in detectors[0] if detectors else False

        # Title
        trig_num = int(self.trig_num_var.get())
        self._ins(f"TRIGGER  {trig_num}\n", 'title')
        self._ins("=" * 50 + "\n", 'separator')

        # Convergence
        if 'iterations' in result:
            if result.get('success'):
                self._ins("convergence!\n", 'success')
                if result['iterations'] == 1:
                    self._ins("  Warning: onset time may be misreported "
                              "on first iteration!\n", 'warning')
            else:
                self._ins(f"consistency failed "
                          f"({result['iterations']} attempts)\n", 'warning')
            self._ins("iterations = ", 'label')
            self._ins(f"{result['iterations']}\n", 'value')

        self._ins("\n")

        # LMA/INTF data counts
        if lma_handler and calc:
            self._ins("LMA points ", 'label')
            self._ins(f"(+/- {calc.lma_range} us): ", 'dim')
            self._ins(f"{len(lma_handler.time)}\n", 'value')
        if intf_handler and calc:
            self._ins("INTF points ", 'label')
            self._ins(f"(+/- {calc.intf_range} us): ", 'dim')
            self._ins(f"{len(intf_handler.time_array)}\n", 'value')

        offset = result.get('_cross_second_offset', 0)
        if offset != 0:
            self._ins("Cross-second offset: ", 'label')
            self._ins(f"{offset:+,} us ({offset / 1e6:+.0f} s)\n", 'warning')

        # ---- Summary statistics ----
        if mode == 'intf_lma' and detectors:
            median = summary.get('median', {})
            mean = summary.get('mean', {})
            std = summary.get('std', {})

            self._ins("\n")
            self._ins("-" * 50 + "\n", 'separator')
            self._ins("  MEDIANS\n", 'section_header')
            self._ins("-" * 50 + "\n", 'separator')
            self._write_val_line("z(km)", f"{median.get('z', 0):.6f}",
                                 f"{median.get('z_err', 0):.6f}")
            self._write_val_line("ta(us)", f"{median.get('ta', 0):.6f}",
                                 f"{median.get('ta_err', 0):.6f}")
            self._write_val_line("tc(us)", f"{median.get('tc', 0):.6f}",
                                 f"{median.get('tc_err', 0):.6f}")
            self._write_val_line("elv(deg)", f"{median.get('elv', 0):.6f}",
                                 f"{median.get('elv_err', 0):.6f}")

            self._ins("\n")
            self._ins("-" * 50 + "\n", 'separator')
            self._ins("  MEANS\n", 'section_header')
            self._ins("-" * 50 + "\n", 'separator')
            self._write_val_line("z(km)", f"{mean.get('z', 0):.6f}",
                                 f"{mean.get('z_err', 0):.6f}")
            self._write_val_line("ta(us)", f"{mean.get('ta', 0):.6f}",
                                 f"{mean.get('ta_err', 0):.6f}")
            self._write_val_line("tc(us)", f"{mean.get('tc', 0):.6f}",
                                 f"{mean.get('tc_err', 0):.6f}")

            # 2 highest VEM, 2 closest, 2 earliest
            if len(detectors) >= 2:
                self._ins("\n")
                self._ins("-" * 50 + "\n", 'separator')
                self._ins("  SUBSETS\n", 'section_header')
                self._ins("-" * 50 + "\n", 'separator')

                sorted_by_vem = sorted(detectors, key=lambda d: d['vem'],
                                       reverse=True)
                top2 = sorted_by_vem[:2]
                self._ins("2 highest VEM", 'label')
                self._ins(f" ({top2[0]['detector']}, {top2[1]['detector']})\n", 'dim')
                z_2h = np.mean([d['z'] for d in top2])
                z_2h_e = np.mean([d.get('z_err', 0) for d in top2])
                self._write_val_line("z(km)", f"{z_2h:.6f}", f"{z_2h_e:.6f}")
                ta_2h = np.mean([d['ta'] for d in top2])
                ta_2h_e = np.mean([d.get('ta_err', 0) for d in top2])
                self._write_val_line("ta(us)", f"{ta_2h:.6f}", f"{ta_2h_e:.6f}")
                tc_2h = np.mean([d['tc'] for d in top2])
                tc_2h_e = np.mean([d.get('tc_err', 0) for d in top2])
                self._write_val_line("tc(us)", f"{tc_2h:.6f}", f"{tc_2h_e:.6f}")

                self._ins("\n")
                sorted_by_x2 = sorted(detectors, key=lambda d: d['x2'])
                close2 = sorted_by_x2[:2]
                self._ins("2 closest", 'label')
                self._ins(f" ({close2[0]['detector']}, {close2[1]['detector']})\n", 'dim')
                z_2c = np.mean([d['z'] for d in close2])
                z_2c_e = np.mean([d.get('z_err', 0) for d in close2])
                self._write_val_line("z(km)", f"{z_2c:.6f}", f"{z_2c_e:.6f}")
                ta_2c = np.mean([d['ta'] for d in close2])
                ta_2c_e = np.mean([d.get('ta_err', 0) for d in close2])
                self._write_val_line("ta(us)", f"{ta_2c:.6f}", f"{ta_2c_e:.6f}")
                tc_2c = np.mean([d['tc'] for d in close2])
                tc_2c_e = np.mean([d.get('tc_err', 0) for d in close2])
                self._write_val_line("tc(us)", f"{tc_2c:.6f}", f"{tc_2c_e:.6f}")

                self._ins("\n")
                sorted_by_tc = sorted(detectors, key=lambda d: d['tc'])
                early2 = sorted_by_tc[:2]
                self._ins("2 earliest", 'label')
                self._ins(f" ({early2[0]['detector']}, {early2[1]['detector']})\n", 'dim')
                z_2e = np.mean([d['z'] for d in early2])
                z_2e_e = np.mean([d.get('z_err', 0) for d in early2])
                self._write_val_line("z(km)", f"{z_2e:.6f}", f"{z_2e_e:.6f}")
                ta_2e = np.mean([d['ta'] for d in early2])
                ta_2e_e = np.mean([d.get('ta_err', 0) for d in early2])
                self._write_val_line("ta(us)", f"{ta_2e:.6f}", f"{ta_2e_e:.6f}")
                tc_2e = np.mean([d['tc'] for d in early2])
                tc_2e_e = np.mean([d.get('tc_err', 0) for d in early2])
                self._write_val_line("tc(us)", f"{tc_2e:.6f}", f"{tc_2e_e:.6f}")

            # Weighted averages
            weighted = summary.get('weighted', {})
            if weighted:
                self._ins("\n")
                self._ins("VEM-weighted", 'label')
                self._ins(f" (total VEM = {summary.get('total_vem', 0):.1f})\n", 'dim')
                self._write_val_line("z(km)", f"{weighted.get('z', 0):.6f}",
                                     f"{weighted.get('z_err', 0):.6f}")
                self._write_val_line("ta(us)", f"{weighted.get('ta', 0):.6f}",
                                     f"{weighted.get('ta_err', 0):.6f}")
                self._write_val_line("tc(us)", f"{weighted.get('tc', 0):.6f}",
                                     f"{weighted.get('tc_err', 0):.6f}")

        # ---- Individual detectors (toggled) ----
        if show_individual and detectors:
            self._ins("\n")
            self._ins("=" * 50 + "\n", 'separator')
            self._ins("  PER-DETECTOR RESULTS\n", 'section_header')
            self._ins("=" * 50 + "\n", 'separator')

            for d in detectors:
                self._ins(f"\n{d['detector']}:\n", 'det_name')
                dt_e = d.get('dt_err')
                self._write_val_line("dt(us)", f"{d['dt']}", dt_e,
                                     self._pct(d['dt'], dt_e) if dt_e else None)
                if 'z' in d:
                    z_e = d.get('z_err')
                    self._write_val_line("z(km)", f"{d['z']}", z_e,
                                         self._pct(d['z'], z_e) if z_e else None)
                if 'elv' in d:
                    e_e = d.get('elv_err')
                    self._write_val_line("elv(deg)", f"{d['elv']}", e_e,
                                         self._pct(d['elv'], e_e) if e_e else None)
                if 'x1' in d:
                    x1_e = d.get('x1_err')
                    self._write_val_line("x1(km)", f"{d['x1']}", x1_e,
                                         self._pct(d['x1'], x1_e) if x1_e else None)
                if 'r1' in d:
                    r1_e = d.get('r1_err')
                    self._write_val_line("r1(km)", f"{d['r1']}", r1_e,
                                         self._pct(d['r1'], r1_e) if r1_e else None)
                if 'ta' in d:
                    ta_e = d.get('ta_err')
                    self._write_val_line("ta(us)", f"{d['ta']}", ta_e,
                                         self._pct(d['ta'], ta_e) if ta_e else None)
                if 'tc' in d:
                    tc_e = d.get('tc_err')
                    self._write_val_line("tc(us)", f"{d['tc']}", tc_e,
                                         self._pct(d['tc'], tc_e) if tc_e else None)

        # ---- Footer: full medians and means tables ----
        if mode == 'intf_lma' and detectors and has_errors:
            dts = np.array([d['dt'] for d in detectors])
            dt_es = np.array([d['dt_err'] for d in detectors])
            zs = np.array([d['z'] for d in detectors])
            z_es = np.array([d['z_err'] for d in detectors])
            elvs = np.array([d['elv'] for d in detectors])
            elv_es = np.array([d['elv_err'] for d in detectors])
            azis = np.array([d['azi'] for d in detectors])
            azi_es = np.array([d['azi_err'] for d in detectors])
            x1s = np.array([d['x1'] for d in detectors])
            x1_es = np.array([d['x1_err'] for d in detectors])
            r1s = np.array([d['r1'] for d in detectors])
            r1_es = np.array([d['r1_err'] for d in detectors])
            tas = np.array([d['ta'] for d in detectors])
            ta_es = np.array([d['ta_err'] for d in detectors])
            tcs = np.array([d['tc'] for d in detectors])
            tc_es = np.array([d['tc_err'] for d in detectors])

            self._ins("\n")
            self._ins("=" * 50 + "\n", 'separator')
            self._ins("  MEDIANS\n", 'section_header')
            self._ins("=" * 50 + "\n", 'separator')

            md, mde = np.median(dts), np.median(dt_es)
            self._write_val_line("dt(us)", f"{md}", f"{mde}",
                                 self._pct(md, mde))
            mz, mze = np.median(zs), np.median(z_es)
            self._write_val_line("z(km)", f"{mz}", f"{mze}",
                                 self._pct(mz, mze))
            me, mee = np.median(elvs), np.median(elv_es)
            self._write_val_line("elv(deg)", f"{me}", f"{mee}",
                                 self._pct(me, mee))
            mx1, mx1e = np.median(x1s), np.median(x1_es)
            self._write_val_line("x1(km)", f"{mx1}", f"{mx1e}",
                                 self._pct(mx1, mx1e))
            mr1, mr1e = np.median(r1s), np.median(r1_es)
            self._write_val_line("r1(km)", f"{mr1}", f"{mr1e}",
                                 self._pct(mr1, mr1e))
            mta, mtae = np.median(tas), np.median(ta_es)
            self._write_val_line("ta(us)", f"{mta}", f"{mtae}",
                                 self._pct(mta, mtae))
            mtc, mtce = np.median(tcs), np.median(tc_es)
            self._write_val_line("tc(us)", f"{mtc}", f"{mtce}",
                                 self._pct(mtc, mtce))

            self._ins("\n")
            self._ins("=" * 50 + "\n", 'separator')
            self._ins("  MEANS\n", 'section_header')
            self._ins("=" * 50 + "\n", 'separator')

            md, mde = np.mean(dts), np.mean(dt_es)
            self._write_val_line("dt(us)", f"{md}", f"{mde}",
                                 self._pct(md, mde))
            mz, mze = np.mean(zs), np.mean(z_es)
            self._write_val_line("z(km)", f"{mz}", f"{mze}",
                                 self._pct(mz, mze))
            me, mee = np.mean(elvs), np.mean(elv_es)
            self._write_val_line("elv(deg)", f"{me}", f"{mee}",
                                 self._pct(me, mee))
            ma, mae = np.mean(azis), np.mean(azi_es)
            self._write_val_line("azi(deg)", f"{ma}", f"{mae}",
                                 self._pct(ma, mae))
            mx1, mx1e = np.mean(x1s), np.mean(x1_es)
            self._write_val_line("x1(km)", f"{mx1}", f"{mx1e}",
                                 self._pct(mx1, mx1e))
            mr1, mr1e = np.mean(r1s), np.mean(r1_es)
            self._write_val_line("r1(km)", f"{mr1}", f"{mr1e}",
                                 self._pct(mr1, mr1e))
            mta, mtae = np.mean(tas), np.mean(ta_es)
            self._write_val_line("ta(us)", f"{mta}", f"{mtae}",
                                 self._pct(mta, mtae))
            mtc, mtce = np.mean(tcs), np.mean(tc_es)
            self._write_val_line("tc(us)", f"{mtc}", f"{mtce}",
                                 self._pct(mtc, mtce))

            lma_lats = [d['lma_lat'] for d in detectors]
            lma_lons = [d['lma_lon'] for d in detectors]
            self._ins("\n")
            self._ins("   source @ ", 'label')
            self._ins(f"{np.mean(lma_lats):.6f}, {np.mean(lma_lons):.6f}",
                      'value')
            self._ins(" (deg)\n", 'dim')

        elif mode == 'intf_lma' and detectors:
            # Fallback if no error data (shouldn't happen but safe)
            self._ins("\n")
            self._ins("=" * 50 + "\n", 'separator')
            self._ins("  MEDIANS\n", 'section_header')
            self._ins("=" * 50 + "\n", 'separator')
            dts = np.array([d['dt'] for d in detectors])
            zs = np.array([d['z'] for d in detectors])
            self._write_val_line("dt(us)", f"{np.median(dts)}")
            self._write_val_line("z(km)", f"{np.median(zs)}")

            self._ins("\n")
            self._ins("=" * 50 + "\n", 'separator')
            self._ins("  MEANS\n", 'section_header')
            self._ins("=" * 50 + "\n", 'separator')
            self._write_val_line("dt(us)", f"{np.mean(dts)}")
            self._write_val_line("z(km)", f"{np.mean(zs)}")

        elif detectors:
            # LMA-only
            dts = np.array([d['dt'] for d in detectors])
            zs = np.array([d['z'] for d in detectors])
            x1s = np.array([d['x1'] for d in detectors])
            self._ins("\n")
            self._ins("=" * 50 + "\n", 'separator')
            self._ins("  MEANS (LMA-only)\n", 'section_header')
            self._ins("=" * 50 + "\n", 'separator')
            self._write_val_line("dt(us)", f"{np.mean(dts):.6f}")
            self._write_val_line("z(km)", f"{np.mean(zs):.6f}")
            self._write_val_line("x1(km)", f"{np.mean(x1s):.6f}")

    def _copy_output(self):
        """Copy right-panel output to clipboard."""
        text = self.output_text.get("1.0", tk.END).strip()
        if text and text != "Run a calculation to see results here.":
            self.clipboard_clear()
            self.clipboard_append(text)
            self.main_app.status_var.set("Output copied to clipboard")

    # =========================================================================
    # Export
    # =========================================================================

    def _format_plain_output(self):
        """Format results as plain text with +/- errors for export."""
        result = self.calc_results
        detectors = result.get('detectors', [])
        summary = result.get('summary', {})
        mode = result.get('_mode', 'intf_lma')
        calc = result.get('_calc')
        lma_handler = result.get('_lma_handler')
        intf_handler = result.get('_intf_handler')
        has_errors = 'dt_err' in detectors[0] if detectors else False

        lines = []
        trig_num = int(self.trig_num_var.get())
        lines.append(f"TRIGGER  {trig_num}")

        if 'iterations' in result:
            if result.get('success'):
                lines.append("convergence!")
                if result['iterations'] == 1:
                    lines.append("  Warning: onset time may be misreported "
                                 "on first iteration!")
            else:
                lines.append(f"consistency failed "
                             f"({result['iterations']} attempts)")
            lines.append(f"iterations = {result['iterations']}")

        if lma_handler and calc:
            lines.append(f"\nnumber of LMA points within +/- "
                         f"{calc.lma_range} us: {len(lma_handler.time)}")
        if intf_handler and calc:
            lines.append(f"number of INTF points within +/- "
                         f"{calc.intf_range} us: {len(intf_handler.time_array)}")

        offset = result.get('_cross_second_offset', 0)
        if offset != 0:
            lines.append(f"Cross-second offset: {offset:+,} us "
                         f"({offset / 1e6:+.0f} s)")

        def _vl(label, val, err=None, pct=None):
            s = f" {label:>9s}={val}"
            if err is not None:
                s += f" +/- {err}"
            if pct is not None:
                s += f" or {pct:.2f} %"
            return s

        def _p(val, err):
            if val == 0:
                return None
            return abs(100 * err / val)

        if mode == 'intf_lma' and detectors:
            median = summary.get('median', {})
            mean = summary.get('mean', {})

            lines.append(f"\nz median = {median.get('z', 0)} +/- {median.get('z_err', 0)}")
            lines.append(f"ta median= {median.get('ta', 0)} +/- {median.get('ta_err', 0)}")
            lines.append(f"tc median= {median.get('tc', 0)} +/- {median.get('tc_err', 0)}")
            lines.append(f"elv median= {median.get('elv', 0)} +/- {median.get('elv_err', 0)}")

            lines.append(f"\nz mean = {mean.get('z', 0)} +/- {mean.get('z_err', 0)}")
            lines.append(f"ta mean= {mean.get('ta', 0)} +/- {mean.get('ta_err', 0)}")
            lines.append(f"tc mean= {mean.get('tc', 0)} +/- {mean.get('tc_err', 0)}")

            if len(detectors) >= 2:
                top2 = sorted(detectors, key=lambda d: d['vem'], reverse=True)[:2]
                lines.append(f"\nz 2high = {np.mean([d['z'] for d in top2])} +/- {np.mean([d.get('z_err', 0) for d in top2])}")
                lines.append(f"ta 2high= {np.mean([d['ta'] for d in top2])} +/- {np.mean([d.get('ta_err', 0) for d in top2])}")
                lines.append(f"tc 2high= {np.mean([d['tc'] for d in top2])} +/- {np.mean([d.get('tc_err', 0) for d in top2])}")

                close2 = sorted(detectors, key=lambda d: d['x2'])[:2]
                lines.append(f"\nz 2close = {np.mean([d['z'] for d in close2])} +/- {np.mean([d.get('z_err', 0) for d in close2])}")
                lines.append(f"ta 2close= {np.mean([d['ta'] for d in close2])} +/- {np.mean([d.get('ta_err', 0) for d in close2])}")
                lines.append(f"tc 2close= {np.mean([d['tc'] for d in close2])} +/- {np.mean([d.get('tc_err', 0) for d in close2])}")

                early2 = sorted(detectors, key=lambda d: d['tc'])[:2]
                lines.append(f"\nz earliest = {np.mean([d['z'] for d in early2])} +/- {np.mean([d.get('z_err', 0) for d in early2])}")
                lines.append(f"ta earliest= {np.mean([d['ta'] for d in early2])} +/- {np.mean([d.get('ta_err', 0) for d in early2])}")
                lines.append(f"tc earliest= {np.mean([d['tc'] for d in early2])} +/- {np.mean([d.get('tc_err', 0) for d in early2])}")

            weighted = summary.get('weighted', {})
            if weighted:
                lines.append(f"\nz weighted = {weighted.get('z', 0)} +/- {weighted.get('z_err', 0)}")
                lines.append(f"ta weighted= {weighted.get('ta', 0)} +/- {weighted.get('ta_err', 0)}")
                lines.append(f"tc weighted= {weighted.get('tc', 0)} +/- {weighted.get('tc_err', 0)}")

        # Per-detector
        lines.append("\n")
        for d in detectors:
            lines.append(f"{d['detector']}:")
            dt_e = d.get('dt_err')
            lines.append(_vl("dt(us)", d['dt'], dt_e, _p(d['dt'], dt_e) if dt_e else None))
            if 'z' in d:
                z_e = d.get('z_err')
                lines.append(_vl("z(km)", d['z'], z_e, _p(d['z'], z_e) if z_e else None))
            if 'elv' in d:
                e_e = d.get('elv_err')
                lines.append(_vl("elv(deg)", d['elv'], e_e, _p(d['elv'], e_e) if e_e else None))
            if 'x1' in d:
                x1_e = d.get('x1_err')
                lines.append(_vl("x1(km)", d['x1'], x1_e, _p(d['x1'], x1_e) if x1_e else None))
            if 'r1' in d:
                r1_e = d.get('r1_err')
                lines.append(_vl("r1(km)", d['r1'], r1_e, _p(d['r1'], r1_e) if r1_e else None))
            if 'ta' in d:
                ta_e = d.get('ta_err')
                lines.append(_vl("ta(us)", d['ta'], ta_e, _p(d['ta'], ta_e) if ta_e else None))
            if 'tc' in d:
                tc_e = d.get('tc_err')
                lines.append(_vl("tc(us)", d['tc'], tc_e, _p(d['tc'], tc_e) if tc_e else None))

        # Footer medians/means
        if mode == 'intf_lma' and detectors and has_errors:
            dts = np.array([d['dt'] for d in detectors])
            dt_es = np.array([d['dt_err'] for d in detectors])
            zs = np.array([d['z'] for d in detectors])
            z_es = np.array([d['z_err'] for d in detectors])
            elvs = np.array([d['elv'] for d in detectors])
            elv_es = np.array([d['elv_err'] for d in detectors])
            azis = np.array([d['azi'] for d in detectors])
            azi_es = np.array([d['azi_err'] for d in detectors])
            x1s = np.array([d['x1'] for d in detectors])
            x1_es = np.array([d['x1_err'] for d in detectors])
            r1s = np.array([d['r1'] for d in detectors])
            r1_es = np.array([d['r1_err'] for d in detectors])
            tas = np.array([d['ta'] for d in detectors])
            ta_es = np.array([d['ta_err'] for d in detectors])
            tcs = np.array([d['tc'] for d in detectors])
            tc_es = np.array([d['tc_err'] for d in detectors])

            lines.append("\nmedians:")
            for arr, earr, lbl in [(dts, dt_es, "dt(us)"), (zs, z_es, "z(km)"),
                                   (elvs, elv_es, "elv(deg)"), (x1s, x1_es, "x1(km)"),
                                   (r1s, r1_es, "r1(km)"), (tas, ta_es, "ta(us)"),
                                   (tcs, tc_es, "tc(us)")]:
                mv, me = np.median(arr), np.median(earr)
                lines.append(_vl(lbl, mv, me, _p(mv, me)))

            lines.append("\nmeans:")
            for arr, earr, lbl in [(dts, dt_es, "dt(us)"), (zs, z_es, "z(km)"),
                                   (elvs, elv_es, "elv(deg)"), (azis, azi_es, "azi(deg)"),
                                   (x1s, x1_es, "x1(km)"), (r1s, r1_es, "r1(km)"),
                                   (tas, ta_es, "ta(us)"), (tcs, tc_es, "tc(us)")]:
                mv, me = np.mean(arr), np.mean(earr)
                lines.append(_vl(lbl, mv, me, _p(mv, me)))

            lma_lats = [d['lma_lat'] for d in detectors]
            lma_lons = [d['lma_lon'] for d in detectors]
            lines.append(f"   source@{np.mean(lma_lats)}, "
                         f"{np.mean(lma_lons)} (deg)")

        return "\n".join(lines)

    def _export_txt(self):
        """Export results as formatted text file."""
        if not self.calc_results or not self.calc_results.get('detectors'):
            messagebox.showinfo("Info", "No results to export.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not filepath:
            return

        text = self._format_plain_output()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

        self.main_app.status_var.set(f"Exported: {os.path.basename(filepath)}")

    def _save_timeshift_to_project(self):
        """Save the mean timeshift to project state."""
        if not self.calc_results:
            messagebox.showinfo("Info", "No results to save.")
            return

        summary = self.calc_results.get('summary', {})
        mean_dt = summary.get('mean', {}).get('dt', 0)

        self.main_app.project_state.timing['timeshift'] = mean_dt
        self.main_app.status_var.set(
            f"Timeshift saved to project: {mean_dt:.3f} us")
        self.main_app.mark_unsaved()

    # =========================================================================
    # Project Save / Load
    # =========================================================================

    def _save_to_project(self):
        """Save tab state to project."""
        state = self.main_app.project_state

        state.timeshift['mode'] = self.mode_var.get()
        state.timeshift['initial_altitude'] = float(self.initial_alt_var.get())
        state.timeshift['iteration_limit'] = int(self.iteration_limit_var.get())
        state.timeshift['sd_timing_error'] = float(
            self.sd_timing_error_var.get())
        state.timeshift['lma_range'] = float(self.lma_range_var.get())
        state.timeshift['lma_chi_limit'] = float(self.lma_chi_limit_var.get())
        state.timeshift['intf_range'] = float(self.intf_range_var.get())
        state.timeshift['intf_elev_min'] = float(self.intf_elev_min_var.get())
        state.timeshift['intf_elev_max'] = float(self.intf_elev_max_var.get())
        state.timeshift['intf_azi_min'] = float(self.intf_azi_min_var.get())
        state.timeshift['intf_azi_max'] = float(self.intf_azi_max_var.get())
        state.timeshift['cross_second_offset_us'] = \
            self.cross_second_offset_var.get()

        if self.calc_results:
            summary = self.calc_results.get('summary', {})
            mean_info = summary.get('mean', {})
            std_info = summary.get('std', {})
            state.timeshift['mean_dt'] = mean_info.get('dt')
            state.timeshift['std_dt'] = std_info.get('dt')
            state.timeshift['n_detectors'] = summary.get('n_detectors')
            state.timeshift['converged'] = self.calc_results.get('success')
            state.timeshift['iterations'] = self.calc_results.get('iterations')

        if self.lma_file_var.get():
            state.files['lma'] = self.lma_file_var.get()
        if self.sd_dir_var.get():
            state.files['sd_directory'] = self.sd_dir_var.get()
        if self.intf_file_var.get():
            state.files['intf_calibrated'] = self.intf_file_var.get()

    def load_from_project(self):
        """Load tab state from project."""
        state = self.main_app.project_state

        if state.files.get('sd_directory'):
            self.sd_dir_var.set(state.files['sd_directory'])
        if state.files.get('lma'):
            self.lma_file_var.set(state.files['lma'])
        if state.files.get('intf_calibrated'):
            self.intf_file_var.set(state.files['intf_calibrated'])

        if state.event_info.get('time'):
            self.event_time_var.set(state.event_info['time'])

        ts = state.timeshift
        self.mode_var.set(ts.get('mode', 'intf_lma'))
        if ts.get('mean_dt') is not None:
            self.initial_alt_var.set(str(ts.get('initial_altitude', 3.0)))
            self.iteration_limit_var.set(str(ts.get('iteration_limit', 10)))
            self.sd_timing_error_var.set(str(ts.get('sd_timing_error', 0.04)))
            self.lma_range_var.set(str(ts.get('lma_range', 1000)))
            self.lma_chi_limit_var.set(str(ts.get('lma_chi_limit', 2.0)))
            self.intf_range_var.set(str(ts.get('intf_range', 9.4)))
            self.intf_elev_min_var.set(str(ts.get('intf_elev_min', 1.0)))
            self.intf_elev_max_var.set(str(ts.get('intf_elev_max', 11.0)))
            self.intf_azi_min_var.set(str(ts.get('intf_azi_min', 240.0)))
            self.intf_azi_max_var.set(str(ts.get('intf_azi_max', 302.0)))

        offset = ts.get('cross_second_offset_us', 0)
        self.cross_second_offset_var.set(offset)
        sec = offset / 1_000_000 if offset else 0
        self.offset_status_var.set(
            f"Offset: {offset:+,} us ({sec:+.0f} s)" if offset else
            "Offset: 0 us (0 s)")

        if ts.get('mean_dt') is not None:
            self.result_vars['mean_dt'].set(f"{ts['mean_dt']:.4f}")
        if ts.get('std_dt') is not None:
            self.result_vars['std_dt'].set(f"{ts['std_dt']:.4f}")
        if ts.get('n_detectors') is not None:
            self.result_vars['n_detectors'].set(str(ts['n_detectors']))
        if ts.get('converged') is not None:
            self.result_vars['convergence'].set(
                "Yes" if ts['converged'] else "No")
        if ts.get('iterations') is not None:
            self.result_vars['iterations'].set(str(ts['iterations']))

        self._on_mode_change()
