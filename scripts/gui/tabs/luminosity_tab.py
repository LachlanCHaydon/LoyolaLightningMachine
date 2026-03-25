"""
TGF Analysis Tool - Luminosity Tab
===================================
Calculates luminosity timeseries from v2012 high-speed camera images.
User defines an ROI (drag-to-draw or manual entry), background is subtracted,
and pixel values within the ROI are summed per frame to produce a timeseries.
Output text file is compatible with the Home Plotter luminosity overlay.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from PIL import Image

from config import INSTRUMENT_COLORS, DEFAULT_LIMITS
from data_handlers import LuminosityHandler


class LuminosityTab(ttk.Frame):
    """
    Luminosity calculation tab.

    Left panel: controls for image directory, timing, ROI, background, run.
    Right panel: image preview with interactive ROI + luminosity timeseries.
    """

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.handler = LuminosityHandler()

        # Image state
        self.image_files = []
        self.image_directory = None
        self.current_preview_image = None
        self.output_filepath = None

        # ROI drawing state
        self.drawing_roi = False
        self.roi_start = None
        self.roi = {'x': 800, 'y': 280, 'width': 100, 'height': 140}

        # Computed results
        self.times = None
        self.luminosities = None

        self._init_variables()
        self._build_ui()

    def _init_variables(self):
        """Initialize tkinter variables."""
        self.dir_var = tk.StringVar()

        # Timing
        self.start_time_var = tk.StringVar(value="0.0")
        self.interval_var = tk.StringVar(value="25.0")
        self.exposure_var = tk.StringVar(value="2.41")

        # ROI
        self.roi_x_var = tk.StringVar(value=str(self.roi['x']))
        self.roi_y_var = tk.StringVar(value=str(self.roi['y']))
        self.roi_w_var = tk.StringVar(value=str(self.roi['width']))
        self.roi_h_var = tk.StringVar(value=str(self.roi['height']))

        # Background
        self.bg_start_var = tk.StringVar(value="0")
        self.bg_count_var = tk.StringVar(value="10")

        # Preview
        self.preview_frame_var = tk.StringVar(value="100")

        # Status
        self.status_var = tk.StringVar(value="No images loaded")
        self.progress_var = tk.DoubleVar(value=0.0)

    def _build_ui(self):
        """Build the tab UI."""
        # Main horizontal paned window
        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=5, pady=5)

        # --- Left sidebar ---
        left_frame = ttk.Frame(paned, width=670)
        paned.add(left_frame, weight=0)

        # Scrollable sidebar
        sidebar_canvas = tk.Canvas(left_frame, width=650)
        scrollbar = ttk.Scrollbar(left_frame, orient='vertical', command=sidebar_canvas.yview)
        self.sidebar = ttk.Frame(sidebar_canvas)

        self.sidebar.bind('<Configure>',
                          lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox('all')))
        sidebar_canvas.create_window((0, 0), window=self.sidebar, anchor='nw')
        sidebar_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side='right', fill='y')
        sidebar_canvas.pack(side='left', fill='both', expand=True)

        self._build_sidebar()

        # --- Right panel ---
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)

        self._build_canvas(right_frame)

    def _build_sidebar(self):
        """Build the left sidebar controls."""
        parent = self.sidebar

        # ---- Image Directory ----
        dir_frame = ttk.LabelFrame(parent, text="Image Directory (v2012)")
        dir_frame.pack(fill='x', padx=5, pady=5)

        dir_row = ttk.Frame(dir_frame)
        dir_row.pack(fill='x', padx=5, pady=3)
        ttk.Entry(dir_row, textvariable=self.dir_var, width=50).pack(side='left', fill='x', expand=True)
        ttk.Button(dir_row, text="Browse", command=self._browse_directory).pack(side='left', padx=5)

        self.dir_info_label = ttk.Label(dir_frame, text="No directory selected")
        self.dir_info_label.pack(anchor='w', padx=5, pady=2)

        # ---- Timing Parameters ----
        timing_frame = ttk.LabelFrame(parent, text="Timing Parameters")
        timing_frame.pack(fill='x', padx=5, pady=5)

        for label, var, unit in [
            ("Start Time:", self.start_time_var, "us"),
            ("Frame Interval:", self.interval_var, "us"),
            ("Exposure:", self.exposure_var, "us"),
        ]:
            row = ttk.Frame(timing_frame)
            row.pack(fill='x', padx=5, pady=2)
            ttk.Label(row, text=label, width=16).pack(side='left')
            ttk.Entry(row, textvariable=var, width=12).pack(side='left', padx=5)
            ttk.Label(row, text=unit).pack(side='left')

        # ---- ROI Parameters ----
        roi_frame = ttk.LabelFrame(parent, text="Region of Interest (ROI)")
        roi_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(roi_frame, text="Draw on image or enter manually:",
                  font=('', 9, 'italic')).pack(anchor='w', padx=5)

        roi_grid = ttk.Frame(roi_frame)
        roi_grid.pack(fill='x', padx=5, pady=3)

        for col, (label, var) in enumerate([
            ("X:", self.roi_x_var), ("Y:", self.roi_y_var),
            ("W:", self.roi_w_var), ("H:", self.roi_h_var),
        ]):
            ttk.Label(roi_grid, text=label).grid(row=0, column=col*2, padx=2)
            ttk.Entry(roi_grid, textvariable=var, width=6).grid(row=0, column=col*2+1, padx=2)

        ttk.Button(roi_frame, text="Apply Manual ROI",
                   command=self._apply_manual_roi).pack(fill='x', padx=5, pady=3)

        # ---- Background Settings ----
        bg_frame = ttk.LabelFrame(parent, text="Background Subtraction")
        bg_frame.pack(fill='x', padx=5, pady=5)

        for label, var in [
            ("Start Frame:", self.bg_start_var),
            ("Number of Frames:", self.bg_count_var),
        ]:
            row = ttk.Frame(bg_frame)
            row.pack(fill='x', padx=5, pady=2)
            ttk.Label(row, text=label, width=18).pack(side='left')
            ttk.Entry(row, textvariable=var, width=8).pack(side='left', padx=5)

        # ---- Preview Controls ----
        preview_frame = ttk.LabelFrame(parent, text="Preview")
        preview_frame.pack(fill='x', padx=5, pady=5)

        prev_row = ttk.Frame(preview_frame)
        prev_row.pack(fill='x', padx=5, pady=3)
        ttk.Label(prev_row, text="Frame #:").pack(side='left')
        ttk.Entry(prev_row, textvariable=self.preview_frame_var, width=8).pack(side='left', padx=5)
        ttk.Button(prev_row, text="Preview", command=self._preview_frame).pack(side='left', padx=5)

        # ---- Run Calculation ----
        run_frame = ttk.LabelFrame(parent, text="Calculate Luminosity")
        run_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(run_frame, text="Run Luminosity Calculation",
                   command=self._run_calculation).pack(fill='x', padx=5, pady=5)

        self.progress_bar = ttk.Progressbar(run_frame, variable=self.progress_var,
                                            maximum=100, mode='determinate')
        self.progress_bar.pack(fill='x', padx=5, pady=2)

        self.status_label = ttk.Label(run_frame, textvariable=self.status_var,
                                      font=('', 9))
        self.status_label.pack(anchor='w', padx=5, pady=2)

        # ---- Export ----
        export_frame = ttk.LabelFrame(parent, text="Export")
        export_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(export_frame, text="Save Luminosity File",
                   command=self._save_luminosity_file).pack(fill='x', padx=5, pady=3)
        ttk.Button(export_frame, text="Send to Home Plotter",
                   command=self._send_to_home_plotter).pack(fill='x', padx=5, pady=3)

    def _build_canvas(self, parent):
        """Build the right-side matplotlib canvas."""
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()

        # Mouse events for ROI drawing
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)

    # =========================================================================
    # Directory / File Loading
    # =========================================================================

    def _browse_directory(self):
        """Browse for v2012 image directory."""
        folder = filedialog.askdirectory(title="Select v2012 Image Directory")
        if folder:
            self.dir_var.set(folder)
            self._load_directory(folder)

    def _load_directory(self, folder):
        """Load image file list from directory."""
        self.image_directory = folder
        self.image_files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))
        ])

        if self.image_files:
            self.dir_info_label.config(
                text=f"{len(self.image_files)} images found")
            self.status_var.set(f"Loaded {len(self.image_files)} images from directory")
            # Auto-preview first available frame
            self._preview_frame()
        else:
            self.dir_info_label.config(text="No image files found!")
            self.status_var.set("No image files found in directory")

    # =========================================================================
    # ROI Interaction
    # =========================================================================

    def _on_mouse_press(self, event):
        """Handle mouse press - start drawing ROI."""
        if event.inaxes and event.inaxes == self.ax_image and event.button == 1:
            self.drawing_roi = True
            self.roi_start = (event.xdata, event.ydata)

    def _on_mouse_move(self, event):
        """Handle mouse move - update ROI rectangle."""
        if self.drawing_roi and event.inaxes and self.roi_start:
            x1, y1 = self.roi_start
            x2, y2 = event.xdata, event.ydata

            self.roi = {
                'x': int(min(x1, x2)),
                'y': int(min(y1, y2)),
                'width': int(abs(x2 - x1)),
                'height': int(abs(y2 - y1))
            }
            self._update_display()

    def _on_mouse_release(self, event):
        """Handle mouse release - finalize ROI."""
        if self.drawing_roi:
            self.drawing_roi = False
            self.roi_start = None
            # Sync to entry fields
            self.roi_x_var.set(str(self.roi['x']))
            self.roi_y_var.set(str(self.roi['y']))
            self.roi_w_var.set(str(self.roi['width']))
            self.roi_h_var.set(str(self.roi['height']))

    def _apply_manual_roi(self):
        """Apply manually entered ROI values."""
        try:
            self.roi = {
                'x': int(self.roi_x_var.get()),
                'y': int(self.roi_y_var.get()),
                'width': int(self.roi_w_var.get()),
                'height': int(self.roi_h_var.get()),
            }
            self._update_display()
        except ValueError:
            messagebox.showerror("Error", "ROI values must be integers")

    # =========================================================================
    # Preview
    # =========================================================================

    def _preview_frame(self):
        """Preview a specific frame with ROI overlay."""
        if not self.image_files or not self.image_directory:
            messagebox.showwarning("Warning", "Load an image directory first")
            return

        try:
            frame_num = int(self.preview_frame_var.get())
        except ValueError:
            messagebox.showerror("Error", "Frame number must be an integer")
            return

        # Find the image file matching this frame number
        target_filename = f"IMG{frame_num:06d}.tif"
        if target_filename not in self.image_files:
            # Try to find closest match or use index
            if frame_num < len(self.image_files):
                target_filename = self.image_files[frame_num]
            else:
                messagebox.showerror("Error",
                                     f"Frame {frame_num} not found. "
                                     f"Available: 0-{len(self.image_files)-1}")
                return

        img_path = os.path.join(self.image_directory, target_filename)
        try:
            img = np.array(Image.open(img_path), dtype=np.float64)
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            self.current_preview_image = img
            self._update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _update_display(self):
        """Update the matplotlib display with image + ROI + timeseries."""
        self.fig.clear()

        has_image = self.current_preview_image is not None
        has_results = self.times is not None

        if has_image and has_results:
            self.ax_image = self.fig.add_subplot(2, 1, 1)
            ax_lum = self.fig.add_subplot(2, 1, 2)
        elif has_image:
            self.ax_image = self.fig.add_subplot(1, 1, 1)
            ax_lum = None
        elif has_results:
            self.ax_image = None
            ax_lum = self.fig.add_subplot(1, 1, 1)
        else:
            self.ax_image = None
            ax_lum = None

        # Draw image with ROI
        if has_image and self.ax_image is not None:
            self.ax_image.imshow(self.current_preview_image, cmap='gray', aspect='auto')
            rect = Rectangle(
                (self.roi['x'], self.roi['y']),
                self.roi['width'], self.roi['height'],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            self.ax_image.add_patch(rect)
            self.ax_image.set_title(
                f"v2012 Preview - ROI: ({self.roi['x']}, {self.roi['y']}) "
                f"{self.roi['width']}x{self.roi['height']}")
            self.ax_image.set_xlabel("Pixels")
            self.ax_image.set_ylabel("Pixels")

        # Draw luminosity timeseries
        if has_results and ax_lum is not None:
            max_val = np.max(self.luminosities)
            if max_val > 0:
                lum_norm = self.luminosities / max_val
            else:
                lum_norm = self.luminosities
            ax_lum.plot(self.times, lum_norm, linewidth=1.5,
                        color=INSTRUMENT_COLORS.get('luminosity', 'tab:olive'),
                        label="Luminosity")
            ax_lum.set_xlabel("Time (us)")
            ax_lum.set_ylabel("Luminosity (normalized)")
            ax_lum.set_title("Luminosity Timeseries")
            ax_lum.legend()
            ax_lum.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # =========================================================================
    # Calculation
    # =========================================================================

    def _run_calculation(self):
        """Run the luminosity calculation on all frames."""
        if not self.image_files or not self.image_directory:
            messagebox.showwarning("Warning", "Load an image directory first")
            return

        try:
            start_time = float(self.start_time_var.get())
            interval = float(self.interval_var.get())
            bg_start = int(self.bg_start_var.get())
            bg_count = int(self.bg_count_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values")
            return

        self.status_var.set("Calculating background...")
        self.progress_var.set(0)
        self.update_idletasks()

        try:
            # Load a reference image to get dimensions
            ref_path = os.path.join(self.image_directory, self.image_files[0])
            ref_img = np.array(Image.open(ref_path), dtype=np.float64)
            if len(ref_img.shape) == 3:
                ref_img = np.mean(ref_img, axis=2)
            img_h, img_w = ref_img.shape

            # Calculate background from specified frames
            bg_images = []
            for i in range(bg_count):
                frame_idx = bg_start + i
                if frame_idx >= len(self.image_files):
                    break
                bg_path = os.path.join(self.image_directory, self.image_files[frame_idx])
                img = np.array(Image.open(bg_path), dtype=np.float64)
                if len(img.shape) == 3:
                    img = np.mean(img, axis=2)
                bg_images.append(img)

            if bg_images:
                bg_mean = np.median(bg_images, axis=0)
            else:
                bg_mean = np.zeros((img_h, img_w))

            self.status_var.set("Processing frames...")
            self.update_idletasks()

            # Process all frames
            times = []
            luminosities = []
            total = len(self.image_files)

            x, y = self.roi['x'], self.roi['y']
            w, h = self.roi['width'], self.roi['height']

            for i, filename in enumerate(self.image_files):
                img_path = os.path.join(self.image_directory, filename)
                img = np.array(Image.open(img_path), dtype=np.float64)
                if len(img.shape) == 3:
                    img = np.mean(img, axis=2)

                # Subtract background
                img = img - bg_mean

                # Extract frame number from filename (IMG######.tif)
                try:
                    # Try 6-digit format: IMG000100.tif
                    basename = os.path.splitext(filename)[0]
                    n = float(basename[3:])  # Strip "IMG" prefix
                except (ValueError, IndexError):
                    n = float(i)

                n_time = start_time + n * interval

                # Sum pixel values in ROI
                roi_data = img[y:y+h, x:x+w]
                lum = np.sum(roi_data)

                times.append(n_time)
                luminosities.append(lum)

                # Update progress
                if i % 10 == 0:
                    pct = (i + 1) / total * 100
                    self.progress_var.set(pct)
                    self.status_var.set(f"Processing frame {i+1}/{total}...")
                    self.update_idletasks()

            self.times = np.array(times)
            self.luminosities = np.array(luminosities)

            # Sort by time (in case files weren't in order)
            sort_idx = np.argsort(self.times)
            self.times = self.times[sort_idx]
            self.luminosities = self.luminosities[sort_idx]

            # Update handler with results
            self.handler.time = self.times
            self.handler.luminosity = self.luminosities
            max_val = np.max(self.luminosities)
            if max_val > 0:
                self.handler.luminosity_normalized = self.luminosities / max_val
            else:
                self.handler.luminosity_normalized = self.luminosities.copy()
            self.handler.is_loaded = True
            self.handler.roi = self.roi.copy()

            self.progress_var.set(100)
            self.status_var.set(
                f"Done! {len(times)} frames processed. "
                f"Time range: {self.times[0]:.1f} - {self.times[-1]:.1f} us")

            self._update_display()
            self.main_app.mark_unsaved()

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Calculation failed: {e}")
            self.status_var.set(f"Error: {e}")

    # =========================================================================
    # Export
    # =========================================================================

    def _save_luminosity_file(self):
        """Save luminosity data to a text file."""
        if self.times is None:
            messagebox.showwarning("Warning", "Run the calculation first")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Luminosity File",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            initialfile="Luminosity_v2012.txt"
        )
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write("#Time Lum\n")
                    for t, lum in zip(self.times, self.luminosities):
                        f.write(f"{t:.6f} {lum:.6f}\n")
                self.output_filepath = filepath
                self.status_var.set(f"Saved to {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

    def _send_to_home_plotter(self):
        """Save file and set it as the luminosity source for Home Plotter."""
        if self.times is None:
            messagebox.showwarning("Warning", "Run the calculation first")
            return

        # If not saved yet, prompt to save
        if not self.output_filepath:
            self._save_luminosity_file()

        if self.output_filepath:
            # Set in project state so home plotter picks it up
            self.main_app.project_state.files['luminosity'] = self.output_filepath

            # Also update home plotter directly if it exists
            if hasattr(self.main_app, 'home_tab'):
                self.main_app.home_tab.file_paths['luminosity'].set(self.output_filepath)

            self.main_app.mark_unsaved()
            self.status_var.set(
                f"Sent to Home Plotter: {os.path.basename(self.output_filepath)}")
            messagebox.showinfo("Success",
                                "Luminosity file set for Home Plotter.\n"
                                "Click 'Load & Plot' on the Home tab to see it.")

    # =========================================================================
    # Project Save / Load
    # =========================================================================

    def _save_to_project(self):
        """Save tab state to project."""
        state = self.main_app.project_state

        state.luminosity_settings = {
            'source_directory': self.image_directory,
            'start_time': float(self.start_time_var.get()) if self.start_time_var.get() else 0.0,
            'frame_interval': float(self.interval_var.get()) if self.interval_var.get() else 25.0,
            'exposure': float(self.exposure_var.get()) if self.exposure_var.get() else 2.41,
            'roi': {
                'x': int(self.roi['x']),
                'y': int(self.roi['y']),
                'width': int(self.roi['width']),
                'height': int(self.roi['height']),
            },
            'bg_start': int(self.bg_start_var.get()) if self.bg_start_var.get() else 0,
            'bg_count': int(self.bg_count_var.get()) if self.bg_count_var.get() else 10,
            'output_file': self.output_filepath,
            'preview_frame': self.preview_frame_var.get(),
        }

    def load_from_project(self):
        """Load tab state from project."""
        state = self.main_app.project_state

        if not hasattr(state, 'luminosity_settings'):
            return

        settings = state.luminosity_settings
        if not settings:
            return

        # Restore timing
        if 'start_time' in settings:
            self.start_time_var.set(str(settings['start_time']))
        if 'frame_interval' in settings:
            self.interval_var.set(str(settings['frame_interval']))
        if 'exposure' in settings:
            self.exposure_var.set(str(settings['exposure']))

        # Restore ROI
        if settings.get('roi') and isinstance(settings['roi'], dict):
            self.roi = settings['roi']
            self.roi_x_var.set(str(self.roi.get('x', 800)))
            self.roi_y_var.set(str(self.roi.get('y', 280)))
            self.roi_w_var.set(str(self.roi.get('width', 100)))
            self.roi_h_var.set(str(self.roi.get('height', 140)))

        # Restore background settings
        if 'bg_start' in settings:
            self.bg_start_var.set(str(settings['bg_start']))
        if 'bg_count' in settings:
            self.bg_count_var.set(str(settings['bg_count']))

        # Restore preview frame
        if 'preview_frame' in settings:
            self.preview_frame_var.set(str(settings['preview_frame']))

        # Restore output file
        if settings.get('output_file'):
            self.output_filepath = settings['output_file']

        # Reload image directory if it exists
        source_dir = settings.get('source_directory')
        if source_dir and os.path.exists(source_dir):
            self.dir_var.set(source_dir)
            self._load_directory(source_dir)
