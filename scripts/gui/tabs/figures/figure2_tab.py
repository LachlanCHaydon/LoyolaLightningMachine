"""
Figure 2 Tab — Multi-instrument overview with camera frames.

Three vertically stacked panels sharing x-axis:
  Panel 1: INTF elevation scatter (stratified s-ratio)
  Panel 2: FA waveform (green)
  Panel 3: TASD waveforms (per-detector, offset vertically)
Bottom strip: Camera frames connected to time axis
Stroke labels: vertical dashed lines spanning all panels
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
import matplotlib.cm as cm

from config import INSTRUMENT_COLORS, TASD_COORDS_FILENAME
from data_handlers import FastAntennaHandler, InterferometerHandler, TASDHandler


class Figure2Tab(ttk.Frame):
    """Publication-quality multi-instrument figure with INTF, FA, TASD, and camera frames."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app

        # Data handlers
        self.fa = FastAntennaHandler()
        self.intf = InterferometerHandler()
        self.tasd = TASDHandler()

        # --- tk variables ---
        self.fa_file_var = tk.StringVar()
        self.intf_file_var = tk.StringVar()
        self.sd_dir_var = tk.StringVar()

        self.event_time_var = tk.StringVar(value="HH:MM:SS")
        self.t0_var = tk.StringVar(value="0")
        self.timeshift_var = tk.StringVar(value="0")

        self.t_min_var = tk.StringVar(value="")
        self.t_max_var = tk.StringVar(value="")

        self.intf_elv_min_var = tk.StringVar(value="")
        self.intf_elv_max_var = tk.StringVar(value="")
        self.intf_azi_min_var = tk.StringVar(value="")
        self.intf_azi_max_var = tk.StringVar(value="")

        self.vem_threshold_var = tk.StringVar(value="0")
        self.max_detectors_var = tk.StringVar(value="20")

        self.frame_dir_var = tk.StringVar()
        self.frame_rows = []

        self.stroke_rows = []

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
        self._build_time_range_section()
        self._build_intf_filters_section()
        self._build_tasd_section()
        self._build_camera_frames_section()
        self._build_stroke_labels_section()
        self._build_action_buttons()
        self._build_plot_area()

    def _build_files_section(self):
        frame = ttk.LabelFrame(self.scroll_frame, text="Files", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for label, var, cmd in [
            ("Fast Antenna:", self.fa_file_var, lambda: self._browse_file("fa")),
            ("INTF Calibrated:", self.intf_file_var, lambda: self._browse_file("intf")),
        ]:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=24).pack(side=tk.LEFT, padx=2)
            ttk.Button(row, text="...", width=3, command=cmd).pack(side=tk.LEFT)

        # SD directory
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="TASD Directory:", width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.sd_dir_var, width=24).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3,
                   command=self._browse_sd_dir).pack(side=tk.LEFT)

        ttk.Button(frame, text="Load from Project",
                   command=self._load_files_from_project).pack(pady=5)

    def _build_timing_section(self):
        frame = ttk.LabelFrame(self.scroll_frame, text="Timing", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for label, var in [("Event Time:", self.event_time_var),
                           ("T0 (us):", self.t0_var),
                           ("Timeshift (us):", self.timeshift_var)]:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT)

    def _build_time_range_section(self):
        frame = ttk.LabelFrame(self.scroll_frame, text="Time Range", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for label, var in [("t_min (us):", self.t_min_var),
                           ("t_max (us):", self.t_max_var)]:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT)

    def _build_intf_filters_section(self):
        frame = ttk.LabelFrame(self.scroll_frame, text="INTF Filters", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for label, var in [("Elev min:", self.intf_elv_min_var),
                           ("Elev max:", self.intf_elv_max_var),
                           ("Azi min:", self.intf_azi_min_var),
                           ("Azi max:", self.intf_azi_max_var)]:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT)

    def _build_tasd_section(self):
        frame = ttk.LabelFrame(self.scroll_frame, text="TASD", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for label, var in [("VEM Threshold:", self.vem_threshold_var),
                           ("Max Detectors:", self.max_detectors_var)]:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT)

    def _build_camera_frames_section(self):
        frame = ttk.LabelFrame(self.scroll_frame, text="Camera Frames (v711)", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Directory:", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.frame_dir_var, width=22).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3,
                   command=self._browse_frame_dir).pack(side=tk.LEFT)

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
            if not self.frame_dir_var.get():
                self.frame_dir_var.set(os.path.dirname(filepath))
            self._add_frame_row(filename=filename, timestamp="")

    def _build_stroke_labels_section(self):
        frame = ttk.LabelFrame(self.scroll_frame, text="Stroke Labels", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        self.stroke_list_frame = ttk.Frame(frame)
        self.stroke_list_frame.pack(fill=tk.X, pady=2)

        ttk.Button(frame, text="Add Stroke Label",
                   command=self._add_stroke_row).pack(pady=2)

    def _add_stroke_row(self, number="", time_us="", label=""):
        row_frame = ttk.Frame(self.stroke_list_frame)
        row_frame.pack(fill=tk.X, pady=1)

        num_var = tk.StringVar(value=number)
        time_var = tk.StringVar(value=time_us)
        label_var = tk.StringVar(value=label)

        ttk.Label(row_frame, text="#").pack(side=tk.LEFT)
        ttk.Entry(row_frame, textvariable=num_var, width=4).pack(side=tk.LEFT, padx=1)
        ttk.Label(row_frame, text="us:").pack(side=tk.LEFT)
        ttk.Entry(row_frame, textvariable=time_var, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Label(row_frame, text="Label:").pack(side=tk.LEFT)
        ttk.Entry(row_frame, textvariable=label_var, width=8).pack(side=tk.LEFT, padx=1)

        idx = len(self.stroke_rows)
        ttk.Button(row_frame, text="X", width=2,
                   command=lambda: self._remove_stroke_row(idx, row_frame)).pack(side=tk.LEFT, padx=2)

        self.stroke_rows.append((num_var, time_var, label_var, row_frame))

    def _remove_stroke_row(self, idx, widget):
        widget.destroy()
        if idx < len(self.stroke_rows):
            self.stroke_rows[idx] = None
        self.stroke_rows = [r for r in self.stroke_rows if r is not None]

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
        frame = ttk.Frame(self.scroll_frame, padding=5)
        frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(frame, text="Plot", command=self._plot).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Export Figure",
                   command=self._export).pack(fill=tk.X, pady=2)

    def _build_plot_area(self):
        self.fig = Figure(figsize=(12, 10), dpi=100)
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

    def _browse_file(self, which):
        if which == "fa":
            fp = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
            if fp:
                self.fa_file_var.set(fp)
        elif which == "intf":
            fp = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All", "*.*")])
            if fp:
                self.intf_file_var.set(fp)

    def _browse_sd_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.sd_dir_var.set(d)

    def _browse_frame_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.frame_dir_var.set(d)

    def _load_files_from_project(self):
        state = self.main_app.project_state
        if state.files.get('fa'):
            self.fa_file_var.set(state.files['fa'])
        if state.files.get('intf_calibrated'):
            self.intf_file_var.set(state.files['intf_calibrated'])
        if state.files.get('sd_directory'):
            self.sd_dir_var.set(state.files['sd_directory'])
        if state.timing.get('T0') is not None:
            self.t0_var.set(str(state.timing['T0']))
        if state.timing.get('timeshift') is not None:
            self.timeshift_var.set(str(state.timing['timeshift']))
        if state.event_info.get('time'):
            self.event_time_var.set(state.event_info['time'])
        self.main_app.status_var.set("Figure 2: loaded file paths from project")

    def _safe_float(self, var):
        try:
            return float(var.get())
        except (ValueError, tk.TclError):
            return None

    def _safe_int(self, var):
        try:
            return int(var.get())
        except (ValueError, tk.TclError):
            return None

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

    def _get_stroke_labels(self):
        labels = []
        for row in self.stroke_rows:
            if row is None:
                continue
            num = row[0].get().strip()
            t = row[1].get().strip()
            lbl = row[2].get().strip()
            if t:
                try:
                    labels.append((num, float(t), lbl))
                except ValueError:
                    pass
        return labels

    # =====================================================================
    # Data Loading
    # =====================================================================

    def _load_all_data(self):
        t0 = self._safe_float(self.t0_var)
        timeshift = self._safe_float(self.timeshift_var) or 0.0

        # FA
        fa_path = self.fa_file_var.get()
        if fa_path and os.path.exists(fa_path):
            self.fa.load_csv(fa_path)
            if t0 is not None:
                self.fa.set_T0(t0)
        else:
            self.fa.is_loaded = False

        # INTF
        intf_path = self.intf_file_var.get()
        if intf_path and os.path.exists(intf_path):
            self.intf.load_data(intf_path, is_calibrated=True)
        else:
            self.intf.is_loaded = False

        # TASD
        sd_dir = self.sd_dir_var.get()
        if sd_dir and os.path.isdir(sd_dir):
            event_time = self.event_time_var.get().replace(":", "")
            if event_time and event_time != "HHMMSS":
                self.tasd.load_coordinates(TASD_COORDS_FILENAME)
                self.tasd.load_directory(sd_dir, event_time, time_shift=timeshift)
        else:
            self.tasd.is_loaded = False

    # =====================================================================
    # Plotting
    # =====================================================================

    def _plot(self):
        self._load_all_data()
        self.fig.clear()

        t_min = self._safe_float(self.t_min_var)
        t_max = self._safe_float(self.t_max_var)
        if t_min is None or t_max is None:
            messagebox.showwarning("Warning", "Please set time range.")
            return

        event_time = self.event_time_var.get()
        time_label = f"Time after {event_time} (us)" if event_time != "HHMMSS" else "Time (us)"

        frame_list = self._get_frame_list()
        has_frames = len(frame_list) > 0

        if has_frames:
            gs = self.fig.add_gridspec(4, 1, height_ratios=[2, 2, 2, 1], hspace=0.08)
        else:
            gs = self.fig.add_gridspec(3, 1, height_ratios=[2, 2, 2], hspace=0.08)

        ax_intf = self.fig.add_subplot(gs[0])
        ax_fa = self.fig.add_subplot(gs[1], sharex=ax_intf)
        ax_tasd = self.fig.add_subplot(gs[2], sharex=ax_intf)
        ax_frames = self.fig.add_subplot(gs[3]) if has_frames else None

        instrument_axes = [ax_intf, ax_fa, ax_tasd]

        # --- Panel 1: INTF elevation ---
        if self.intf.is_loaded:
            elv_min = self._safe_float(self.intf_elv_min_var)
            elv_max = self._safe_float(self.intf_elv_max_var)
            azi_min = self._safe_float(self.intf_azi_min_var)
            azi_max = self._safe_float(self.intf_azi_max_var)
            tiers = self.intf.get_stratified_scatter(
                elv_min=elv_min, elv_max=elv_max,
                azi_min=azi_min, azi_max=azi_max,
                t_min=t_min, t_max=t_max)
            for tier in tiers:
                if len(tier['time']) > 0:
                    ax_intf.scatter(tier['time'], tier['elevation'],
                                   c=tier['colors'], s=tier['marker_sizes'],
                                   alpha=tier['alpha'], edgecolors='none')
        ax_intf.set_ylabel("INTF Elevation (deg)")
        ax_intf.grid(True, linestyle=':', alpha=0.5)

        # --- Panel 2: FA waveform ---
        if self.fa.is_loaded:
            fa_data = self.fa.get_data_in_range(t_min, t_max)
            if fa_data and len(fa_data['time']) > 0:
                ax_fa.plot(fa_data['time'], fa_data['e_field'],
                           color='green', linewidth=0.6)
        ax_fa.set_ylabel("Fast Antenna (V/m)")
        ax_fa.grid(True, linestyle=':', alpha=0.5)

        # --- Panel 3: TASD waveforms ---
        if self.tasd.is_loaded:
            vem_thresh = self._safe_float(self.vem_threshold_var) or 0
            max_det = self._safe_int(self.max_detectors_var) or 20

            # Filter by VEM and sort
            dets = [d for d in self.tasd.detectors if d['vem'] >= vem_thresh]
            dets.sort(key=lambda d: d['vem'], reverse=True)
            dets = dets[:max_det]

            if dets:
                # Color by arrival time
                trig_times = [d['sd_trig'] for d in dets]
                t_arr_min = min(trig_times)
                t_arr_max = max(trig_times)
                cmap_tasd = cm.get_cmap('viridis')

                for i, det in enumerate(dets):
                    mask = (det['time'] >= t_min) & (det['time'] <= t_max)
                    if not np.any(mask):
                        continue
                    # Normalize arrival time for color
                    if t_arr_max > t_arr_min:
                        c_val = (det['sd_trig'] - t_arr_min) / (t_arr_max - t_arr_min)
                    else:
                        c_val = 0.5
                    # Average signal, offset vertically by detector index
                    sig = (det['signal_lower'][mask] + det['signal_upper'][mask]) / 2.0
                    # Subtract pedestal
                    ped = (det['pedestal_l'] + det['pedestal_u']) / 2.0
                    sig = sig - ped
                    ax_tasd.plot(det['time'][mask], sig + i * 50,
                                color=cmap_tasd(c_val), linewidth=0.5,
                                label=det['detector_id'] if i < 5 else None)

                if len(dets) <= 10:
                    ax_tasd.legend(fontsize=6, loc='upper right')

        ax_tasd.set_ylabel("TASD Signal (VEM)")
        ax_tasd.grid(True, linestyle=':', alpha=0.5)
        ax_tasd.set_xlabel(time_label if not has_frames else "")

        # x limits
        ax_intf.set_xlim(t_min, t_max)

        # --- Stroke labels ---
        stroke_labels = self._get_stroke_labels()
        for num, t_stroke, lbl in stroke_labels:
            for ax in instrument_axes:
                ax.axvline(t_stroke, color='gray', linestyle='--', linewidth=0.8)
            # Annotate at top of INTF panel
            display_text = f"#{num}" if num else lbl
            ax_intf.annotate(display_text, xy=(t_stroke, 1.0),
                             xycoords=('data', 'axes fraction'),
                             fontsize=7, ha='center', va='bottom')

        # Hide x-tick labels on upper panels
        for ax in [ax_intf, ax_fa]:
            ax.tick_params(labelbottom=False)

        # --- Camera frame strip ---
        if has_frames and ax_frames is not None:
            self._draw_frame_strip(ax_tasd, ax_frames, frame_list, t_min, t_max)
            ax_frames.set_xlabel(time_label)
            ax_tasd.set_xlabel("")

        self.fig.tight_layout()
        self.canvas.draw()
        self.main_app.status_var.set("Figure 2: plot updated")

    def _draw_frame_strip(self, ax_top, ax_bot, frame_list, t_min, t_max):
        ax_bot.axis('off')
        ax_bot.set_xlim(t_min, t_max)
        ax_bot.set_ylim(0, 1)

        n = len(frame_list)
        if n == 0:
            return

        frame_dir = self.frame_dir_var.get()
        x_positions = np.linspace(t_min + (t_max - t_min) * 0.05,
                                  t_max - (t_max - t_min) * 0.05, n)

        for i, (fn, ts) in enumerate(frame_list):
            img_path = os.path.join(frame_dir, fn) if frame_dir else fn
            if not os.path.exists(img_path):
                continue
            try:
                img = mpimg.imread(img_path)
                im = OffsetImage(img, zoom=0.15)
                ab = AnnotationBbox(im, (x_positions[i], 0.5),
                                    frameon=False, box_alignment=(0.5, 0.5))
                ax_bot.add_artist(ab)

                ax_bot.text(x_positions[i], 0.02, f"{ts:.0f} us",
                            ha='center', va='bottom', fontsize=6)

                con = ConnectionPatch(
                    xyA=(x_positions[i], 1.0), coordsA=ax_bot.transData,
                    xyB=(ts, 0), coordsB=ax_top.get_xaxis_transform(),
                    axesA=ax_bot, axesB=ax_top,
                    linestyle='--', linewidth=0.8, color='gray')
                self.fig.add_artist(con)
            except Exception as e:
                print(f"Warning: could not load frame {fn}: {e}")

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
        f2 = state.figure2

        f2['time_min'] = self._safe_float(self.t_min_var)
        f2['time_max'] = self._safe_float(self.t_max_var)
        f2['intf_elv_min'] = self._safe_float(self.intf_elv_min_var)
        f2['intf_elv_max'] = self._safe_float(self.intf_elv_max_var)
        f2['intf_azi_min'] = self._safe_float(self.intf_azi_min_var)
        f2['intf_azi_max'] = self._safe_float(self.intf_azi_max_var)
        f2['vem_threshold'] = self._safe_float(self.vem_threshold_var) or 0
        f2['max_detectors'] = self._safe_int(self.max_detectors_var) or 20

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
        f2['camera_frames'] = frames

        # Stroke labels
        strokes = []
        for row in self.stroke_rows:
            if row is None:
                continue
            num = row[0].get().strip()
            t = row[1].get().strip()
            lbl = row[2].get().strip()
            if t:
                try:
                    strokes.append([num, float(t), lbl])
                except ValueError:
                    pass
        f2['stroke_labels'] = strokes

    def load_from_project(self):
        state = self.main_app.project_state
        f2 = state.figure2

        if f2.get('time_min') is not None:
            self.t_min_var.set(str(f2['time_min']))
        if f2.get('time_max') is not None:
            self.t_max_var.set(str(f2['time_max']))
        if f2.get('intf_elv_min') is not None:
            self.intf_elv_min_var.set(str(f2['intf_elv_min']))
        if f2.get('intf_elv_max') is not None:
            self.intf_elv_max_var.set(str(f2['intf_elv_max']))
        if f2.get('intf_azi_min') is not None:
            self.intf_azi_min_var.set(str(f2['intf_azi_min']))
        if f2.get('intf_azi_max') is not None:
            self.intf_azi_max_var.set(str(f2['intf_azi_max']))
        if f2.get('vem_threshold') is not None:
            self.vem_threshold_var.set(str(f2['vem_threshold']))
        if f2.get('max_detectors') is not None:
            self.max_detectors_var.set(str(f2['max_detectors']))

        for fn, ts in f2.get('camera_frames', []):
            self._add_frame_row(filename=fn, timestamp=str(ts))

        for entry in f2.get('stroke_labels', []):
            if len(entry) >= 3:
                self._add_stroke_row(number=entry[0], time_us=str(entry[1]), label=entry[2])

        # File paths
        if state.files.get('fa'):
            self.fa_file_var.set(state.files['fa'])
        if state.files.get('intf_calibrated'):
            self.intf_file_var.set(state.files['intf_calibrated'])
        if state.files.get('sd_directory'):
            self.sd_dir_var.set(state.files['sd_directory'])
        if state.event_info.get('time'):
            self.event_time_var.set(state.event_info['time'])
        if state.timing.get('T0') is not None:
            self.t0_var.set(str(state.timing['T0']))
        if state.timing.get('timeshift') is not None:
            self.timeshift_var.set(str(state.timing['timeshift']))
