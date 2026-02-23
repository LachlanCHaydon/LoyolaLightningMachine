"""
TGF Analysis Tool - Spectroscopy Tab
====================================
Multi-frame spectroscopy analysis with keyframe calibration interpolation.

Features:
- Load and navigate frame sequences (TIF/PNG)
- Define ROI on reference frame, track across sequence
- Full wavelength calibration on 3-5 keyframes
- Coefficient interpolation between keyframes
- Batch process all frames
- Export results and animations
"""

import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import rotate, minimum_filter1d, uniform_filter1d

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

# Optional imports
try:
    import peakutils

    HAS_PEAKUTILS = True
except ImportError:
    HAS_PEAKUTILS = False

# Local imports
from config import KNOWN_LINES


# =============================================================================
# Data Classes
# =============================================================================

class FrameData:
    """Container for a single frame's data and analysis results."""

    def __init__(self, index, filename, image_path):
        self.index = index
        self.filename = filename
        self.image_path = image_path
        self.image_data = None  # Raw grayscale numpy array
        self.timestamp = None  # Microseconds

        # ROI offset for this frame (for tracking)
        self.roi_offset_y = 0

        # Keyframe calibration data
        self.is_keyframe = False
        self.calibration_points = []  # [(pixel, wavelength), ...]
        self.calibration_coeffs = None  # Polynomial coefficients for this keyframe
        self.calibration_r_squared = 0.0

        # Interpolated calibration (for non-keyframes)
        self.interpolated_coeffs = None

        # Analysis results
        self.spectrum = None
        self.spectrum_baseline_removed = None
        self.peaks = None
        self.wavelengths = None

    def load_image(self):
        """Load image data from file."""
        img = Image.open(self.image_path)
        self.image_data = np.array(img, dtype=np.float64)
        if len(self.image_data.shape) == 3:
            self.image_data = np.mean(self.image_data, axis=2)
        return self.image_data

    def get_effective_coeffs(self):
        """Get the calibration coefficients to use for this frame."""
        if self.is_keyframe and self.calibration_coeffs is not None:
            return self.calibration_coeffs
        elif self.interpolated_coeffs is not None:
            return self.interpolated_coeffs
        return None

    def to_dict(self):
        """Convert to dictionary for saving."""
        return {
            'index': self.index,
            'filename': self.filename,
            'timestamp': self.timestamp,
            'roi_offset_y': self.roi_offset_y,
            'is_keyframe': self.is_keyframe,
            'calibration_points': self.calibration_points,
            'calibration_coeffs': self.calibration_coeffs.tolist() if self.calibration_coeffs is not None else None,
            'calibration_r_squared': self.calibration_r_squared,
        }

    def from_dict(self, data):
        """Load from dictionary."""
        self.timestamp = data.get('timestamp')
        self.roi_offset_y = data.get('roi_offset_y', 0)
        self.is_keyframe = data.get('is_keyframe', False)
        self.calibration_points = data.get('calibration_points', [])
        coeffs = data.get('calibration_coeffs')
        self.calibration_coeffs = np.array(coeffs) if coeffs else None
        self.calibration_r_squared = data.get('calibration_r_squared', 0.0)


# =============================================================================
# Spectrum Analysis Methods (Static)
# =============================================================================

class SpectrumAnalyzer:
    """Static methods for spectrum extraction and analysis."""

    @staticmethod
    def extract_spectrum(image_data, roi, tilt_angle=0, flip_horizontal=False):
        """Extract 1D spectrum from ROI."""
        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']

        # Bounds checking
        y_start = max(0, y)
        y_end = min(image_data.shape[0], y + h)
        x_start = max(0, x)
        x_end = min(image_data.shape[1], x + w)

        roi_data = image_data[y_start:y_end, x_start:x_end].copy()

        if abs(tilt_angle) > 0.1:
            roi_data = rotate(roi_data, tilt_angle, reshape=False, mode='nearest')

        spectrum = np.mean(roi_data, axis=0)

        if flip_horizontal:
            spectrum = spectrum[::-1]

        return spectrum

    @staticmethod
    def extract_roi_image(image_data, roi, tilt_angle=0, flip_horizontal=False, expand_y=0):
        """Extract ROI image with optional expansion."""
        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']

        y_start = max(0, y - expand_y)
        y_end = min(image_data.shape[0], y + h + expand_y)
        x_start = max(0, x)
        x_end = min(image_data.shape[1], x + w)

        roi_data = image_data[y_start:y_end, x_start:x_end].copy()

        roi_top_in_expanded = y - y_start
        roi_bottom_in_expanded = roi_top_in_expanded + h

        if abs(tilt_angle) > 0.1:
            roi_data = rotate(roi_data, tilt_angle, reshape=False, mode='nearest')

        if flip_horizontal:
            roi_data = roi_data[:, ::-1]

        return roi_data, (roi_top_in_expanded, roi_bottom_in_expanded)

    @staticmethod
    def remove_baseline(spectrum, degree=8):
        """Remove baseline using iterative polynomial fit."""
        if HAS_PEAKUTILS:
            try:
                baseline = peakutils.baseline(spectrum, deg=degree)
                return spectrum - baseline, baseline
            except:
                pass

        # Fallback method
        x = np.arange(len(spectrum))
        baseline = np.copy(spectrum)
        for _ in range(3):
            coeffs = np.polyfit(x, baseline, degree)
            baseline = np.polyval(coeffs, x)
            baseline = np.minimum(baseline, spectrum)

        flattened = spectrum - baseline
        flattened = flattened - np.min(flattened)
        return flattened, baseline

    @staticmethod
    def find_peaks_in_spectrum(spectrum, prominence=0.15, distance=10, height_fraction=0.1,
                               use_baseline_removal=True, baseline_degree=8):
        """Find peaks in spectrum."""
        if use_baseline_removal:
            spec_processed, _ = SpectrumAnalyzer.remove_baseline(spectrum, baseline_degree)
        else:
            spec_processed = spectrum.copy()

        spec_min = np.min(spec_processed)
        spec_max = np.max(spec_processed)
        if spec_max - spec_min < 1e-10:
            return [], spec_processed

        spec_norm = (spec_processed - spec_min) / (spec_max - spec_min)

        peak_indices, properties = find_peaks(
            spec_norm,
            prominence=prominence,
            distance=distance,
            height=height_fraction
        )

        peaks = []
        for i, idx in enumerate(peak_indices):
            peaks.append({
                'pixel': float(idx),
                'intensity': float(spec_processed[idx]),
                'intensity_original': float(spectrum[idx]),
                'prominence': float(properties['prominences'][i]) if 'prominences' in properties else 0
            })

        peaks.sort(key=lambda p: p['intensity'], reverse=True)
        return peaks, spec_processed

    @staticmethod
    def refine_peak_position(spectrum, peak_pixel, window=3):
        """Refine peak position using quadratic fit."""
        idx = int(peak_pixel)
        if idx < window or idx >= len(spectrum) - window:
            return float(idx)

        x = np.arange(idx - window, idx + window + 1)
        y = spectrum[idx - window:idx + window + 1]

        try:
            coeffs = np.polyfit(x, y, 2)
            if abs(coeffs[0]) > 1e-10:
                refined = -coeffs[1] / (2 * coeffs[0])
                if abs(refined - idx) <= window:
                    return refined
        except:
            pass

        return float(idx)

    @staticmethod
    def polynomial_fit(calibration_points, order=4):
        """Fit polynomial to calibration points."""
        if len(calibration_points) < 2:
            return None, 0

        pixels = np.array([p[0] for p in calibration_points])
        wavelengths = np.array([p[1] for p in calibration_points])

        actual_order = min(order, len(calibration_points) - 1)
        coeffs = np.polyfit(pixels, wavelengths, actual_order)

        predicted = np.polyval(coeffs, pixels)
        ss_res = np.sum((wavelengths - predicted) ** 2)
        ss_tot = np.sum((wavelengths - np.mean(wavelengths)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return coeffs, r_squared

    @staticmethod
    def pixel_to_wavelength(pixel, coeffs):
        """Convert pixel to wavelength using coefficients."""
        if coeffs is None:
            return None
        return np.polyval(coeffs, pixel)

    @staticmethod
    def interpolate_coefficients(frame_idx, keyframes_data):
        """
        Interpolate calibration coefficients between keyframes.

        Parameters:
        -----------
        frame_idx : int
            Frame index to interpolate for
        keyframes_data : dict
            {frame_idx: coeffs_array} for each keyframe

        Returns:
        --------
        np.ndarray : Interpolated coefficients
        """
        if not keyframes_data:
            return None

        indices = sorted(keyframes_data.keys())

        if len(indices) == 1:
            return keyframes_data[indices[0]].copy()

        # Before first keyframe
        if frame_idx <= indices[0]:
            return keyframes_data[indices[0]].copy()

        # After last keyframe
        if frame_idx >= indices[-1]:
            return keyframes_data[indices[-1]].copy()

        # Find bracketing keyframes
        for i, idx in enumerate(indices[:-1]):
            if idx <= frame_idx < indices[i + 1]:
                idx1, idx2 = idx, indices[i + 1]
                break

        # Linear interpolation of coefficients
        coeffs1 = keyframes_data[idx1]
        coeffs2 = keyframes_data[idx2]

        # Ensure same length (pad shorter with zeros if needed)
        max_len = max(len(coeffs1), len(coeffs2))
        c1 = np.zeros(max_len)
        c2 = np.zeros(max_len)
        c1[-len(coeffs1):] = coeffs1
        c2[-len(coeffs2):] = coeffs2

        t = (frame_idx - idx1) / (idx2 - idx1)
        interpolated = c1 * (1 - t) + c2 * t

        return interpolated

    @staticmethod
    def assign_known_lines(peaks, coeffs, tolerance=15):
        """Assign known spectral lines to peaks (no duplicates)."""
        assigned_lines = set()
        peaks_sorted = sorted(peaks, key=lambda p: p.get('intensity', 0), reverse=True)

        for peak in peaks_sorted:
            wl = SpectrumAnalyzer.pixel_to_wavelength(peak['pixel'], coeffs)
            peak['wavelength'] = wl
            peak['assigned_line'] = None

            if wl is None:
                continue

            best_match = None
            best_diff = tolerance
            best_wl = None

            for known_wl, (species, priority, is_anchor) in KNOWN_LINES.items():
                if known_wl in assigned_lines:
                    continue

                diff = abs(wl - known_wl)
                if diff < best_diff:
                    best_diff = diff
                    best_wl = known_wl
                    best_match = {
                        'wavelength': known_wl,
                        'species': species,
                        'priority': priority,
                        'is_anchor': is_anchor
                    }

            if best_match is not None:
                peak['assigned_line'] = best_match
                assigned_lines.add(best_wl)

        return peaks

    # Add these methods to the SpectrumAnalyzer class:

    @staticmethod
    def auto_track_roi(image_data, current_roi, search_range=50, direction="both"):
        """
        Automatically find the best Y position for ROI by finding brightest horizontal band.

        Parameters:
        -----------
        image_data : 2D numpy array
        current_roi : dict
            {'x': int, 'y': int, 'width': int, 'height': int}
        search_range : int
            Pixels to search above and below current position
        direction : str
            'down' - only search below current position
            'up' - only search above current position
            'both' - search both directions

        Returns:
        --------
        suggested_offset : int
            Suggested Y offset from current position
        confidence : float
            Confidence in the suggestion (0-1)
        """
        x, y, w, h = current_roi['x'], current_roi['y'], current_roi['width'], current_roi['height']

        # Define search region based on direction
        if direction == "down":
            y_min = y
            y_max = min(image_data.shape[0] - h, y + search_range)
        elif direction == "up":
            y_min = max(0, y - search_range)
            y_max = y + h
        else:  # both
            y_min = max(0, y - search_range)
            y_max = min(image_data.shape[0] - h, y + search_range)

        if y_max <= y_min:
            return 0, 0

        # Calculate brightness for each possible Y position
        brightness = []
        positions = []
        for test_y in range(y_min, y_max):
            x_start = max(0, x)
            x_end = min(image_data.shape[1], x + w)
            if x_end <= x_start:
                continue
            roi_slice = image_data[test_y:test_y + h, x_start:x_end]
            if roi_slice.size > 0:
                brightness.append(np.mean(roi_slice))
                positions.append(test_y)

        if len(brightness) == 0:
            return 0, 0

        brightness = np.array(brightness)
        positions = np.array(positions)

        # Find the brightest position
        best_idx = np.argmax(brightness)
        best_y = positions[best_idx]
        suggested_offset = best_y - y

        # Calculate confidence based on how distinct the peak is
        if len(brightness) > 1:
            peak_val = brightness[best_idx]
            mean_val = np.mean(brightness)
            std_val = np.std(brightness)
            if std_val > 0:
                confidence = min(1.0, (peak_val - mean_val) / (std_val * 2 + 1e-10))
            else:
                confidence = 0
        else:
            confidence = 0

        return suggested_offset, max(0, confidence)

# =============================================================================
# Main Spectroscopy Tab Class
# =============================================================================

class SpectroscopyTab(ttk.Frame):
    """Spectroscopy analysis tab with keyframe calibration interpolation."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.project_state = main_app.project_state

        # Frame data
        self.frames = []
        self.current_frame_idx = 0

        # ROI settings
        self.roi = {'x': 50, 'y': 100, 'width': 300, 'height': 30}
        self.tilt_angle = 0.0
        self.flip_horizontal = False

        # Timing
        self.start_timestamp = 0
        self.frame_interval = 40  # microseconds

        # Peak detection settings
        self.peak_prominence = 0.15
        self.peak_distance = 10
        self.baseline_degree = 8
        self.use_baseline_removal = True

        # Current calibration state (for calibration sub-tab)
        self.detected_peaks = []
        self.selected_peak_idx = None
        self.current_spectrum_display = None
        self.temp_calibration_points = []  # Working calibration points

        # Drawing state
        self.drawing_roi = False
        self.roi_start = None

        # Analysis state
        self.analysis_complete = False

        # Initialize variables
        self._init_variables()

        # Build UI
        self._build_ui()

    def _init_variables(self):
        """Initialize tkinter variables."""
        # ROI variables
        self.roi_x_var = tk.StringVar(value="50")
        self.roi_y_var = tk.StringVar(value="100")
        self.roi_w_var = tk.StringVar(value="300")
        self.roi_h_var = tk.StringVar(value="30")
        self.tilt_var = tk.DoubleVar(value=0.0)
        self.flip_var = tk.BooleanVar(value=False)

        # Timing variables
        self.start_time_var = tk.StringVar(value="0")
        self.interval_var = tk.StringVar(value="40")

        # Peak detection variables
        self.prominence_var = tk.DoubleVar(value=0.15)
        self.distance_var = tk.IntVar(value=10)
        self.baseline_var = tk.BooleanVar(value=True)
        self.baseline_degree_var = tk.IntVar(value=8)

        # Calibration variables
        self.poly_order_var = tk.IntVar(value=4)
        self.wavelength_var = tk.StringVar()

        # Navigation variables
        self.goto_frame_var = tk.StringVar(value="0")

        # ROI tracking variables
        self.search_range_var = tk.IntVar(value=50)
        self.track_direction_var = tk.StringVar(value="down")

        # Manual peak entry variables
        self.manual_pixel_var = tk.StringVar()
        self.manual_wavelength_var = tk.StringVar()
        self.click_mode_var = tk.BooleanVar(value=False)

        # Direct offset entry variable
        self.direct_offset_var = tk.StringVar(value="0")

    def _build_ui(self):
        """Build the main UI with sub-notebook."""
        self.sub_notebook = ttk.Notebook(self)
        self.sub_notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Create sub-tabs
        self.setup_frame = ttk.Frame(self.sub_notebook)
        self.tracking_review_frame = ttk.Frame(self.sub_notebook)
        self.calibrate_frame = ttk.Frame(self.sub_notebook)
        self.review_frame = ttk.Frame(self.sub_notebook)
        self.export_frame = ttk.Frame(self.sub_notebook)

        self.sub_notebook.add(self.setup_frame, text='Setup')
        self.sub_notebook.add(self.tracking_review_frame, text='ROI Tracking')
        self.sub_notebook.add(self.calibrate_frame, text='Calibrate')
        self.sub_notebook.add(self.review_frame, text='Review')
        self.sub_notebook.add(self.export_frame, text='Export')

        # Build each sub-tab
        self._build_setup_tab()
        self._build_tracking_review_tab()
        self._build_calibrate_tab()
        self._build_review_tab()
        self._build_export_tab()

    # =========================================================================
    # Setup Sub-Tab
    # =========================================================================

    def _build_setup_tab(self):
        """Build the Setup sub-tab."""
        # Left panel - controls
        left_container = ttk.Frame(self.setup_frame, width=670)
        left_container.pack(side='left', fill='y', padx=5, pady=5)
        left_container.pack_propagate(False)

        # Scrollable frame
        canvas = tk.Canvas(left_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        left_frame = ttk.Frame(canvas)

        left_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=left_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- Load Frames Section ---
        load_frame = ttk.LabelFrame(left_frame, text="1. Load Frames")
        load_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(load_frame, text="Select Frame Folder...",
                   command=self._load_frames).pack(fill='x', padx=5, pady=5)

        self.frames_label = ttk.Label(load_frame, text="No frames loaded")
        self.frames_label.pack(padx=5, pady=2)

        # --- Timing Section ---
        timing_frame = ttk.LabelFrame(left_frame, text="2. Timing Configuration")
        timing_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(timing_frame, text="Start Timestamp (μs):").pack(anchor='w', padx=5)
        ttk.Entry(timing_frame, textvariable=self.start_time_var).pack(fill='x', padx=5, pady=2)

        ttk.Label(timing_frame, text="Frame Interval (μs):").pack(anchor='w', padx=5)
        ttk.Entry(timing_frame, textvariable=self.interval_var).pack(fill='x', padx=5, pady=2)

        ttk.Button(timing_frame, text="Apply Timing",
                   command=self._apply_timing).pack(fill='x', padx=5, pady=5)

        # --- ROI Section ---
        roi_frame = ttk.LabelFrame(left_frame, text="3. Define ROI")
        roi_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(roi_frame, text="Draw ROI on image or enter manually:").pack(anchor='w', padx=5)

        roi_entries = ttk.Frame(roi_frame)
        roi_entries.pack(fill='x', padx=5, pady=2)

        ttk.Label(roi_entries, text="X:").grid(row=0, column=0)
        ttk.Entry(roi_entries, textvariable=self.roi_x_var, width=6).grid(row=0, column=1)
        ttk.Label(roi_entries, text="Y:").grid(row=0, column=2)
        ttk.Entry(roi_entries, textvariable=self.roi_y_var, width=6).grid(row=0, column=3)
        ttk.Label(roi_entries, text="W:").grid(row=1, column=0)
        ttk.Entry(roi_entries, textvariable=self.roi_w_var, width=6).grid(row=1, column=1)
        ttk.Label(roi_entries, text="H:").grid(row=1, column=2)
        ttk.Entry(roi_entries, textvariable=self.roi_h_var, width=6).grid(row=1, column=3)

        ttk.Button(roi_frame, text="Apply ROI Values",
                   command=self._apply_roi_values).pack(fill='x', padx=5, pady=2)

        # Tilt adjustment
        ttk.Label(roi_frame, text="Tilt Angle (°):").pack(anchor='w', padx=5, pady=(5, 0))
        tilt_frame = ttk.Frame(roi_frame)
        tilt_frame.pack(fill='x', padx=10)
        ttk.Scale(tilt_frame, from_=-6, to=6, variable=self.tilt_var,
                  orient='horizontal', command=lambda _: self._update_setup_display()).pack(side='left', fill='x',
                                                                                            expand=True)
        self.tilt_label = ttk.Label(tilt_frame, text="0.0°", width=8)
        self.tilt_label.pack(side='left')

        # Flip checkbox
        ttk.Checkbutton(roi_frame, text="Flip Spectrum Horizontally",
                        variable=self.flip_var,
                        command=self._update_setup_display).pack(anchor='w', padx=5, pady=5)

        # Frame Y-offset
        ttk.Label(roi_frame, text="Current Frame Y-Offset:").pack(anchor='w', padx=5, pady=(5, 0))
        offset_buttons = ttk.Frame(roi_frame)
        offset_buttons.pack(fill='x', padx=5, pady=2)
        ttk.Button(offset_buttons, text="↑ Up 5",
                   command=lambda: self._adjust_frame_offset(-5)).pack(side='left', expand=True, fill='x')
        ttk.Button(offset_buttons, text="↓ Down 5",
                   command=lambda: self._adjust_frame_offset(5)).pack(side='left', expand=True, fill='x')
        self.offset_label = ttk.Label(roi_frame, text="Offset: 0 px")
        self.offset_label.pack(padx=5)

        # --- Navigation Section ---
        nav_frame = ttk.LabelFrame(left_frame, text="Frame Navigation")
        nav_frame.pack(fill='x', padx=5, pady=5)

        # Direct frame entry
        direct_frame = ttk.Frame(nav_frame)
        direct_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(direct_frame, text="Go to Frame:").pack(side='left')
        ttk.Entry(direct_frame, textvariable=self.goto_frame_var, width=6).pack(side='left', padx=5)
        ttk.Button(direct_frame, text="Go", command=self._goto_frame).pack(side='left')

        # Prev/Next buttons
        nav_buttons = ttk.Frame(nav_frame)
        nav_buttons.pack(fill='x', padx=5, pady=2)
        ttk.Button(nav_buttons, text="← Prev",
                   command=self._prev_frame).pack(side='left', expand=True, fill='x')
        ttk.Button(nav_buttons, text="Next →",
                   command=self._next_frame).pack(side='left', expand=True, fill='x')

        # Slider
        self.frame_slider = ttk.Scale(nav_frame, from_=0, to=0, orient='horizontal',
                                      command=self._on_slider_change)
        self.frame_slider.pack(fill='x', padx=5, pady=2)

        # Frame info label
        self.nav_label = ttk.Label(nav_frame, text="Frame 0 / 0", font=('', 10, 'bold'))
        self.nav_label.pack(padx=5, pady=2)

        # Proceed button
        ttk.Button(left_frame, text="Proceed to Calibration →",
                   command=self._go_to_calibration).pack(fill='x', padx=5, pady=10)

        # --- Right panel - matplotlib figure ---
        right_frame = ttk.Frame(self.setup_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        self.setup_fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.setup_canvas = FigureCanvasTkAgg(self.setup_fig, master=right_frame)
        self.setup_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Mouse events for ROI drawing
        self.setup_canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.setup_canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.setup_canvas.mpl_connect('button_release_event', self._on_mouse_release)

        # Toolbar
        toolbar = NavigationToolbar2Tk(self.setup_canvas, right_frame)
        toolbar.update()

        # Add this after the offset_label in the ROI section:

        # --- ROI Tracking Section ---
        track_frame = ttk.LabelFrame(left_frame, text="ROI Tracking")
        track_frame.pack(fill='x', padx=5, pady=5)

        # Search range
        ttk.Label(track_frame, text="Search Range (px):").pack(anchor='w', padx=5)
        self.search_range_var = tk.IntVar(value=50)
        search_row = ttk.Frame(track_frame)
        search_row.pack(fill='x', padx=5)
        ttk.Scale(search_row, from_=10, to=100, variable=self.search_range_var,
                  orient='horizontal').pack(side='left', fill='x', expand=True)
        self.search_label = ttk.Label(search_row, text="50", width=4)
        self.search_label.pack(side='left')

        # Tracking direction
        ttk.Label(track_frame, text="Track Direction:").pack(anchor='w', padx=5)
        self.track_direction_var = tk.StringVar(value="down")
        dir_row = ttk.Frame(track_frame)
        dir_row.pack(fill='x', padx=5)
        ttk.Radiobutton(dir_row, text="Down", value="down",
                        variable=self.track_direction_var).pack(side='left')
        ttk.Radiobutton(dir_row, text="Up", value="up",
                        variable=self.track_direction_var).pack(side='left')
        ttk.Radiobutton(dir_row, text="Both", value="both",
                        variable=self.track_direction_var).pack(side='left')

        # Tracking buttons
        ttk.Button(track_frame, text="Auto-Track Current Frame",
                   command=self._auto_track_current_frame).pack(fill='x', padx=5, pady=2)

        ttk.Button(track_frame, text="Auto-Track ALL Frames",
                   command=self._auto_track_all_frames).pack(fill='x', padx=5, pady=2)

        ttk.Button(track_frame, text="Reset All Offsets",
                   command=self._reset_all_offsets).pack(fill='x', padx=5, pady=2)

        # Tracking status
        self.tracking_status_label = ttk.Label(track_frame, text="No tracking applied")
        self.tracking_status_label.pack(padx=5, pady=2)

    # =========================================================================
    # ROI Tracking Methods
    # =========================================================================

    def _auto_track_current_frame(self):
        """Auto-track ROI for current frame."""
        if not self.frames:
            return

        frame = self.frames[self.current_frame_idx]
        effective_roi = self.roi.copy()
        effective_roi['y'] += frame.roi_offset_y

        search_range = self.search_range_var.get()
        direction = self.track_direction_var.get()

        suggested_offset, confidence = SpectrumAnalyzer.auto_track_roi(
            frame.image_data, effective_roi, search_range=search_range, direction=direction
        )

        if confidence > 0.1:
            frame.roi_offset_y += suggested_offset
            self.offset_label.config(text=f"Offset: {frame.roi_offset_y} px (conf: {confidence:.2f})")
            self._update_tracking_status()
            self._update_setup_display()
        else:
            messagebox.showinfo("Auto-Track", "Could not find a clear bright region")

    def _auto_track_all_frames(self):
        """Auto-track ROI for all frames sequentially."""
        if not self.frames:
            return

        if not messagebox.askyesno("Auto-Track All",
                                   "This will auto-track all frames starting from the current frame's ROI position.\n\n"
                                   "For best results, manually set the ROI on the first frame before running.\n\n"
                                   "Continue?"):
            return

        search_range = self.search_range_var.get()
        direction = self.track_direction_var.get()

        # Create progress window
        progress = tk.Toplevel(self.winfo_toplevel())
        progress.title("Auto-Tracking...")
        progress.geometry("400x150")
        progress_label = ttk.Label(progress, text="Tracking frames...")
        progress_label.pack(pady=10)
        progress_bar = ttk.Progressbar(progress, length=350, mode='determinate')
        progress_bar.pack(pady=10)
        status_label = ttk.Label(progress, text="")
        status_label.pack(pady=5)

        tracked_count = 0
        failed_frames = []

        # Track forward from current frame
        for i in range(self.current_frame_idx, len(self.frames)):
            progress_label.config(text=f"Tracking frame {i + 1}/{len(self.frames)}")
            progress_bar['value'] = (i / len(self.frames)) * 100
            progress.update()

            frame = self.frames[i]

            if i == self.current_frame_idx:
                # Use current frame's offset as starting point
                continue

            # Use previous frame's position as reference
            prev_frame = self.frames[i - 1]
            effective_roi = self.roi.copy()
            effective_roi['y'] += prev_frame.roi_offset_y

            suggested_offset, confidence = SpectrumAnalyzer.auto_track_roi(
                frame.image_data, effective_roi, search_range=search_range, direction=direction
            )

            if confidence > 0.05:  # Lower threshold for batch processing
                frame.roi_offset_y = prev_frame.roi_offset_y + suggested_offset
                tracked_count += 1
                status_label.config(text=f"Frame {i}: offset={frame.roi_offset_y}, conf={confidence:.2f}")
            else:
                # Keep previous frame's offset if tracking fails
                frame.roi_offset_y = prev_frame.roi_offset_y
                failed_frames.append(i)
                status_label.config(text=f"Frame {i}: FAILED - using previous offset")

        # Track backward from current frame if needed
        if self.current_frame_idx > 0:
            for i in range(self.current_frame_idx - 1, -1, -1):
                progress_label.config(text=f"Tracking frame {i + 1}/{len(self.frames)} (backward)")
                progress_bar['value'] = ((len(self.frames) - i) / len(self.frames)) * 100
                progress.update()

                frame = self.frames[i]
                next_frame = self.frames[i + 1]
                effective_roi = self.roi.copy()
                effective_roi['y'] += next_frame.roi_offset_y

                # For backward tracking, flip direction
                back_direction = "up" if direction == "down" else ("down" if direction == "up" else "both")

                suggested_offset, confidence = SpectrumAnalyzer.auto_track_roi(
                    frame.image_data, effective_roi, search_range=search_range, direction=back_direction
                )

                if confidence > 0.05:
                    frame.roi_offset_y = next_frame.roi_offset_y + suggested_offset
                    tracked_count += 1
                else:
                    frame.roi_offset_y = next_frame.roi_offset_y
                    failed_frames.append(i)

        progress.destroy()

        # Update display
        self._update_tracking_status()
        self._update_setup_display()

        # Show summary
        msg = f"Auto-tracking complete!\n\nTracked: {tracked_count} frames"
        if failed_frames:
            msg += f"\nFailed (using neighbor's offset): {len(failed_frames)} frames"
            if len(failed_frames) <= 10:
                msg += f"\n\nFailed frames: {failed_frames}"
            else:
                msg += f"\n\nFailed frames: {failed_frames[:10]}..."

        messagebox.showinfo("Auto-Track Complete", msg)

    def _reset_all_offsets(self):
        """Reset all frame offsets to zero."""
        if not self.frames:
            return

        if not messagebox.askyesno("Reset Offsets", "Reset all frame Y-offsets to zero?"):
            return

        for frame in self.frames:
            frame.roi_offset_y = 0

        self._update_tracking_status()
        self._update_setup_display()
        messagebox.showinfo("Reset", "All offsets reset to zero")

    def _update_tracking_status(self):
        """Update the tracking status label."""
        if not self.frames:
            self.tracking_status_label.config(text="No frames loaded")
            return

        offsets = [f.roi_offset_y for f in self.frames]
        non_zero = sum(1 for o in offsets if o != 0)
        min_off, max_off = min(offsets), max(offsets)

        if non_zero == 0:
            self.tracking_status_label.config(text="No tracking applied")
        else:
            self.tracking_status_label.config(
                text=f"Tracked: {non_zero}/{len(self.frames)} frames\nRange: {min_off} to {max_off} px"
            )

        # Update search range label
        self.search_label.config(text=str(self.search_range_var.get()))
    # =========================================================================
    # ROI Tracking Review Sub-Tab
    # =========================================================================
    def _build_tracking_review_tab(self):
        """Build the ROI Tracking Review sub-tab."""
        # Left panel (width=670)
        left_container = ttk.Frame(self.tracking_review_frame, width=670)
        left_container.pack(side='left', fill='y', padx=5, pady=5)
        left_container.pack_propagate(False)

        left_frame = ttk.Frame(left_container)
        left_frame.pack(fill='both', expand=True)

        # --- Navigation ---
        nav_frame = ttk.LabelFrame(left_frame, text="Frame Navigation")
        nav_frame.pack(fill='x', padx=5, pady=5)

        nav_buttons = ttk.Frame(nav_frame)
        nav_buttons.pack(fill='x', padx=5, pady=2)
        ttk.Button(nav_buttons, text="← Prev",
                   command=self._tracking_prev_frame).pack(side='left', expand=True, fill='x')
        ttk.Button(nav_buttons, text="Next →",
                   command=self._tracking_next_frame).pack(side='left', expand=True, fill='x')

        self.tracking_slider = ttk.Scale(nav_frame, from_=0, to=0, orient='horizontal',
                                         command=self._on_tracking_slider)
        self.tracking_slider.pack(fill='x', padx=5, pady=2)

        self.tracking_nav_label = ttk.Label(nav_frame, text="Frame 0 / 0", font=('', 10, 'bold'))
        self.tracking_nav_label.pack(padx=5, pady=2)

        # --- Current Frame Offset ---
        offset_frame = ttk.LabelFrame(left_frame, text="Current Frame Offset")
        offset_frame.pack(fill='x', padx=5, pady=5)

        self.tracking_offset_label = ttk.Label(offset_frame, text="Y-Offset: 0 px", font=('', 12))
        self.tracking_offset_label.pack(padx=5, pady=5)

        # Manual offset adjustment
        ttk.Label(offset_frame, text="Adjust offset:").pack(anchor='w', padx=5)

        adjust_row1 = ttk.Frame(offset_frame)
        adjust_row1.pack(fill='x', padx=5, pady=2)
        ttk.Button(adjust_row1, text="↑↑ -10",
                   command=lambda: self._adjust_tracking_offset(-10)).pack(side='left', expand=True, fill='x')
        ttk.Button(adjust_row1, text="↑ -5",
                   command=lambda: self._adjust_tracking_offset(-5)).pack(side='left', expand=True, fill='x')
        ttk.Button(adjust_row1, text="↑ -1",
                   command=lambda: self._adjust_tracking_offset(-1)).pack(side='left', expand=True, fill='x')

        adjust_row2 = ttk.Frame(offset_frame)
        adjust_row2.pack(fill='x', padx=5, pady=2)
        ttk.Button(adjust_row2, text="↓↓ +10",
                   command=lambda: self._adjust_tracking_offset(10)).pack(side='left', expand=True, fill='x')
        ttk.Button(adjust_row2, text="↓ +5",
                   command=lambda: self._adjust_tracking_offset(5)).pack(side='left', expand=True, fill='x')
        ttk.Button(adjust_row2, text="↓ +1",
                   command=lambda: self._adjust_tracking_offset(1)).pack(side='left', expand=True, fill='x')

        # Direct entry
        direct_row = ttk.Frame(offset_frame)
        direct_row.pack(fill='x', padx=5, pady=5)
        ttk.Label(direct_row, text="Set offset:").pack(side='left')
        self.direct_offset_var = tk.StringVar(value="0")
        ttk.Entry(direct_row, textvariable=self.direct_offset_var, width=8).pack(side='left', padx=5)
        ttk.Button(direct_row, text="Apply",
                   command=self._apply_direct_offset).pack(side='left')

        # --- Batch Operations ---
        batch_frame = ttk.LabelFrame(left_frame, text="Batch Operations")
        batch_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(batch_frame, text="Copy Offset to All Following Frames",
                   command=self._copy_offset_forward).pack(fill='x', padx=5, pady=2)

        ttk.Button(batch_frame, text="Copy Offset to All Previous Frames",
                   command=self._copy_offset_backward).pack(fill='x', padx=5, pady=2)

        ttk.Button(batch_frame, text="Interpolate Offsets Between Keyframes",
                   command=self._interpolate_offsets).pack(fill='x', padx=5, pady=2)

        # --- Offset Summary ---
        summary_frame = ttk.LabelFrame(left_frame, text="Offset Summary")
        summary_frame.pack(fill='x', padx=5, pady=5)

        self.offset_summary_label = ttk.Label(summary_frame, text="No frames loaded", justify='left')
        self.offset_summary_label.pack(padx=5, pady=5)

        # Offset listbox (shows frames with non-zero offsets)
        ttk.Label(summary_frame, text="Frames with modified offsets:").pack(anchor='w', padx=5)

        list_frame = ttk.Frame(summary_frame)
        list_frame.pack(fill='x', padx=5, pady=2)

        self.offset_listbox = tk.Listbox(list_frame, height=8, exportselection=False)
        offset_scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.offset_listbox.yview)
        self.offset_listbox.configure(yscrollcommand=offset_scrollbar.set)
        self.offset_listbox.pack(side='left', fill='x', expand=True)
        offset_scrollbar.pack(side='right', fill='y')
        self.offset_listbox.bind('<<ListboxSelect>>', self._on_offset_listbox_select)

        # --- Right panel - matplotlib figure ---
        right_frame = ttk.Frame(self.tracking_review_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        self.tracking_fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.tracking_canvas = FigureCanvasTkAgg(self.tracking_fig, master=right_frame)
        self.tracking_canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self.tracking_canvas, right_frame)
        toolbar.update()


    # =========================================================================
    # ROI Tracking Review Methods
    # =========================================================================

    def _tracking_prev_frame(self):
        """Previous frame in tracking review."""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.tracking_slider.set(self.current_frame_idx)
            self._update_tracking_review_display()

    def _tracking_next_frame(self):
        """Next frame in tracking review."""
        if self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
            self.tracking_slider.set(self.current_frame_idx)
            self._update_tracking_review_display()

    def _on_tracking_slider(self, value):
        """Handle tracking slider change."""
        self.current_frame_idx = int(float(value))
        self._update_tracking_review_display()

    def _adjust_tracking_offset(self, delta):
        """Adjust offset for current frame."""
        if not self.frames:
            return
        self.frames[self.current_frame_idx].roi_offset_y += delta
        self._update_tracking_review_display()

    def _apply_direct_offset(self):
        """Apply directly entered offset value."""
        if not self.frames:
            return
        try:
            offset = int(self.direct_offset_var.get())
            self.frames[self.current_frame_idx].roi_offset_y = offset
            self._update_tracking_review_display()
        except ValueError:
            messagebox.showerror("Error", "Invalid offset value")

    def _copy_offset_forward(self):
        """Copy current frame's offset to all following frames."""
        if not self.frames:
            return
        current_offset = self.frames[self.current_frame_idx].roi_offset_y
        count = 0
        for i in range(self.current_frame_idx + 1, len(self.frames)):
            self.frames[i].roi_offset_y = current_offset
            count += 1
        self._update_tracking_review_display()
        messagebox.showinfo("Done", f"Applied offset {current_offset} to {count} frames")

    def _copy_offset_backward(self):
        """Copy current frame's offset to all previous frames."""
        if not self.frames:
            return
        current_offset = self.frames[self.current_frame_idx].roi_offset_y
        count = 0
        for i in range(self.current_frame_idx):
            self.frames[i].roi_offset_y = current_offset
            count += 1
        self._update_tracking_review_display()
        messagebox.showinfo("Done", f"Applied offset {current_offset} to {count} frames")

    def _interpolate_offsets(self):
        """Interpolate offsets between frames with manually set offsets."""
        if not self.frames:
            return

        # Find frames where user has modified offset (non-zero or explicitly set)
        # For now, use frames with non-zero offsets as anchor points
        anchors = [(i, f.roi_offset_y) for i, f in enumerate(self.frames) if f.roi_offset_y != 0]

        if len(anchors) < 2:
            messagebox.showwarning("Warning",
                                   "Need at least 2 frames with non-zero offsets to interpolate.\n\n"
                                   "Set offsets on at least 2 frames first.")
            return

        # Add first and last frame if not already anchors
        if anchors[0][0] != 0:
            anchors.insert(0, (0, self.frames[0].roi_offset_y))
        if anchors[-1][0] != len(self.frames) - 1:
            anchors.append((len(self.frames) - 1, self.frames[-1].roi_offset_y))

        # Interpolate between anchor points
        for i in range(len(anchors) - 1):
            idx1, off1 = anchors[i]
            idx2, off2 = anchors[i + 1]

            for j in range(idx1 + 1, idx2):
                t = (j - idx1) / (idx2 - idx1)
                self.frames[j].roi_offset_y = int(off1 + t * (off2 - off1))

        self._update_tracking_review_display()
        messagebox.showinfo("Done", f"Interpolated offsets between {len(anchors)} anchor frames")

    def _on_offset_listbox_select(self, event):
        """Jump to selected frame from offset listbox."""
        selection = self.offset_listbox.curselection()
        if not selection:
            return

        text = self.offset_listbox.get(selection[0])
        import re
        match = re.search(r'Frame (\d+):', text)
        if match:
            idx = int(match.group(1))
            self.current_frame_idx = idx
            self.tracking_slider.set(idx)
            self._update_tracking_review_display()

    def _update_tracking_review_display(self):
        """Update the tracking review display."""
        if not self.frames:
            return

        frame = self.frames[self.current_frame_idx]
        effective_roi = self.roi.copy()
        effective_roi['y'] += frame.roi_offset_y

        # Update labels
        self.tracking_nav_label.config(
            text=f"Frame {self.current_frame_idx} / {len(self.frames) - 1} - {frame.timestamp} μs"
        )
        self.tracking_offset_label.config(text=f"Y-Offset: {frame.roi_offset_y} px")
        self.direct_offset_var.set(str(frame.roi_offset_y))

        # Update summary
        offsets = [f.roi_offset_y for f in self.frames]
        non_zero = sum(1 for o in offsets if o != 0)
        min_off, max_off = min(offsets), max(offsets)
        mean_off = np.mean(offsets)

        summary = f"Total frames: {len(self.frames)}\n"
        summary += f"Frames with offset: {non_zero}\n"
        summary += f"Offset range: {min_off} to {max_off} px\n"
        summary += f"Mean offset: {mean_off:.1f} px"
        self.offset_summary_label.config(text=summary)

        # Update offset listbox
        self.offset_listbox.delete(0, tk.END)
        for i, f in enumerate(self.frames):
            if f.roi_offset_y != 0:
                marker = "→ " if i == self.current_frame_idx else ""
                self.offset_listbox.insert(tk.END, f"{marker}Frame {i}: offset={f.roi_offset_y} px")

        # Update plot
        self.tracking_fig.clear()
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

        # Main image with ROI
        ax_img = self.tracking_fig.add_subplot(gs[0, 0])
        ax_img.imshow(frame.image_data, cmap='viridis', aspect='auto')
        rect = Rectangle((effective_roi['x'], effective_roi['y']),
                         effective_roi['width'], effective_roi['height'],
                         linewidth=2, edgecolor='white', facecolor='none')
        ax_img.add_patch(rect)
        # Also show original ROI position in red
        original_rect = Rectangle((self.roi['x'], self.roi['y']),
                                  self.roi['width'], self.roi['height'],
                                  linewidth=1, edgecolor='red', facecolor='none',
                                  linestyle='--', alpha=0.5)
        ax_img.add_patch(original_rect)
        ax_img.set_title(f"Frame {self.current_frame_idx} - White=Current, Red=Original")

        # ROI zoom with context
        ax_roi = self.tracking_fig.add_subplot(gs[0, 1])
        roi_image, roi_bounds = SpectrumAnalyzer.extract_roi_image(
            frame.image_data, effective_roi, self.tilt_angle, self.flip_horizontal, expand_y=40
        )
        if roi_image.size > 0:
            ax_roi.imshow(roi_image, cmap='viridis', aspect='auto')
            ax_roi.axhline(roi_bounds[0], color='white', linestyle='-', linewidth=2, alpha=0.8)
            ax_roi.axhline(roi_bounds[1], color='white', linestyle='-', linewidth=2, alpha=0.8)
        ax_roi.set_title(f"ROI Context")

        # Spectrum comparison
        ax_spec = self.tracking_fig.add_subplot(gs[1, 0])
        spectrum = SpectrumAnalyzer.extract_spectrum(
            frame.image_data, effective_roi, self.tilt_angle, self.flip_horizontal
        )
        ax_spec.plot(spectrum, 'g-', linewidth=1, label='Current ROI')
        ax_spec.set_title("Extracted Spectrum")
        ax_spec.set_xlabel("Pixel")
        ax_spec.set_ylabel("Intensity")
        ax_spec.grid(True, alpha=0.3)
        ax_spec.legend()

        # Offset timeline
        ax_timeline = self.tracking_fig.add_subplot(gs[1, 1])
        frame_indices = np.arange(len(self.frames))
        offsets = [f.roi_offset_y for f in self.frames]
        ax_timeline.plot(frame_indices, offsets, 'b-', linewidth=1)
        ax_timeline.axvline(self.current_frame_idx, color='red', linestyle='--', linewidth=2)
        ax_timeline.scatter([self.current_frame_idx], [frame.roi_offset_y], color='red', s=50, zorder=5)
        ax_timeline.set_xlabel("Frame")
        ax_timeline.set_ylabel("Y-Offset (px)")
        ax_timeline.set_title("Offset Timeline")
        ax_timeline.grid(True, alpha=0.3)

        self.tracking_fig.tight_layout()
        self.tracking_canvas.draw()

    # =========================================================================
    # Calibrate Sub-Tab
    # =========================================================================

    def _build_calibrate_tab(self):
        """Build the Calibration sub-tab."""
        # Left panel (width=670)
        left_container = ttk.Frame(self.calibrate_frame, width=670)
        left_container.pack(side='left', fill='y', padx=5, pady=5)
        left_container.pack_propagate(False)

        # Scrollable frame
        canvas = tk.Canvas(left_container, highlightthickness=0, width=650)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        left_frame = ttk.Frame(canvas, width=640)

        def _configure_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Ensure the frame width matches the canvas
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())

        left_frame.bind("<Configure>", _configure_scroll)
        canvas_window = canvas.create_window((0, 0), window=left_frame, anchor="nw", width=640)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Bind canvas resize to update frame width
        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)

        # --- Frame Navigation ---
        nav_frame = ttk.LabelFrame(left_frame, text="Frame Navigation")
        nav_frame.pack(fill='x', padx=5, pady=5)

        slider_row = ttk.Frame(nav_frame)
        slider_row.pack(fill='x', padx=5, pady=2)

        ttk.Button(slider_row, text="◀", width=3,
                   command=lambda: self._cal_change_frame(-1)).pack(side='left')

        self.cal_frame_slider = ttk.Scale(slider_row, from_=0, to=0, orient='horizontal',
                                          command=self._on_cal_frame_slider)
        self.cal_frame_slider.pack(side='left', fill='x', expand=True, padx=5)

        ttk.Button(slider_row, text="▶", width=3,
                   command=lambda: self._cal_change_frame(1)).pack(side='left')

        self.cal_frame_label = ttk.Label(nav_frame, text="Frame: 0 / 0")
        self.cal_frame_label.pack(padx=5, pady=2)

        # Keyframe indicator
        self.keyframe_indicator = ttk.Label(nav_frame, text="", foreground='green')
        self.keyframe_indicator.pack(padx=5)

        # --- Peak Detection ---
        peak_frame = ttk.LabelFrame(left_frame, text="Peak Detection")
        peak_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(peak_frame, text="Prominence:").pack(anchor='w', padx=5)
        prom_row = ttk.Frame(peak_frame)
        prom_row.pack(fill='x', padx=5)
        ttk.Scale(prom_row, from_=0.05, to=0.5, variable=self.prominence_var,
                  orient='horizontal').pack(side='left', fill='x', expand=True)
        self.prom_label = ttk.Label(prom_row, text="0.15", width=6)
        self.prom_label.pack(side='left')

        ttk.Label(peak_frame, text="Min Distance (px):").pack(anchor='w', padx=5)
        dist_row = ttk.Frame(peak_frame)
        dist_row.pack(fill='x', padx=5)
        ttk.Scale(dist_row, from_=5, to=30, variable=self.distance_var,
                  orient='horizontal').pack(side='left', fill='x', expand=True)
        self.dist_label = ttk.Label(dist_row, text="10", width=6)
        self.dist_label.pack(side='left')

        ttk.Checkbutton(peak_frame, text="Remove Baseline",
                        variable=self.baseline_var).pack(anchor='w', padx=5, pady=2)

        ttk.Label(peak_frame, text="Baseline Degree:").pack(anchor='w', padx=5)
        degree_row = ttk.Frame(peak_frame)
        degree_row.pack(fill='x', padx=5)
        ttk.Scale(degree_row, from_=1, to=10, variable=self.baseline_degree_var,
                  orient='horizontal').pack(side='left', fill='x', expand=True)
        self.degree_label = ttk.Label(degree_row, text="8", width=6)
        self.degree_label.pack(side='left')

        ttk.Button(peak_frame, text="Detect Peaks",
                   command=self._detect_peaks).pack(fill='x', padx=5, pady=5)

        # --- Peak Assignment ---
        assign_frame = ttk.LabelFrame(left_frame, text="Peak Assignment")
        assign_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(assign_frame, text="Select peak, then assign wavelength:").pack(anchor='w', padx=5)

        # Peak listbox
        peak_list_frame = ttk.Frame(assign_frame)
        peak_list_frame.pack(fill='x', padx=5, pady=2)

        self.peak_listbox = tk.Listbox(peak_list_frame, height=8, exportselection=False)
        peak_scrollbar = ttk.Scrollbar(peak_list_frame, orient='vertical', command=self.peak_listbox.yview)
        self.peak_listbox.configure(yscrollcommand=peak_scrollbar.set)
        self.peak_listbox.pack(side='left', fill='x', expand=True)
        peak_scrollbar.pack(side='right', fill='y')
        self.peak_listbox.bind('<<ListboxSelect>>', self._on_peak_select)

        # Wavelength assignment
        assign_row = ttk.Frame(assign_frame)
        assign_row.pack(fill='x', padx=5, pady=2)

        ttk.Label(assign_row, text="λ (nm):").pack(side='left')
        self.wavelength_combo = ttk.Combobox(assign_row, textvariable=self.wavelength_var, width=18)
        self.wavelength_combo['values'] = [f"{wl} ({info[0]}){' ★' if info[2] else ''}"
                                           for wl, info in sorted(KNOWN_LINES.items())]
        self.wavelength_combo.pack(side='left', padx=5)

        ttk.Button(assign_row, text="Assign",
                   command=self._assign_wavelength).pack(side='left')

        # Quick actions
        quick_row = ttk.Frame(assign_frame)
        quick_row.pack(fill='x', padx=5, pady=5)
        ttk.Button(quick_row, text="Clear All",
                   command=self._clear_temp_calibration).pack(side='left', expand=True, fill='x', padx=1)

        # Add after the quick_row in assign_frame:

        ttk.Separator(assign_frame, orient='horizontal').pack(fill='x', padx=5, pady=10)

        # Manual peak entry section
        ttk.Label(assign_frame, text="Manual Peak Entry:", font=('', 9, 'bold')).pack(anchor='w', padx=5)

        manual_peak_row = ttk.Frame(assign_frame)
        manual_peak_row.pack(fill='x', padx=5, pady=2)

        ttk.Label(manual_peak_row, text="Pixel:").pack(side='left')
        self.manual_pixel_var = tk.StringVar()
        ttk.Entry(manual_peak_row, textvariable=self.manual_pixel_var, width=8).pack(side='left', padx=5)

        ttk.Label(manual_peak_row, text="λ (nm):").pack(side='left')
        self.manual_wavelength_var = tk.StringVar()
        self.manual_wavelength_combo = ttk.Combobox(manual_peak_row, textvariable=self.manual_wavelength_var, width=12)
        self.manual_wavelength_combo['values'] = [str(wl) for wl in sorted(KNOWN_LINES.keys())]
        self.manual_wavelength_combo.pack(side='left', padx=5)

        ttk.Button(manual_peak_row, text="Add",
                   command=self._add_manual_peak).pack(side='left', padx=5)

        # Quick pixel selection from click
        self.click_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(assign_frame, text="Click spectrum to set pixel position",
                        variable=self.click_mode_var,
                        command=self._toggle_click_mode).pack(anchor='w', padx=5, pady=2)

        self.click_status_label = ttk.Label(assign_frame, text="", foreground='blue')
        self.click_status_label.pack(anchor='w', padx=5)

        # Calibration points list
        ttk.Label(assign_frame, text="Calibration Points:").pack(anchor='w', padx=5, pady=(10, 0))
        self.cal_listbox = tk.Listbox(assign_frame, height=6)
        self.cal_listbox.pack(fill='x', padx=5, pady=2)

        ttk.Button(assign_frame, text="Remove Selected Point",
                   command=self._remove_cal_point).pack(fill='x', padx=5, pady=2)

        # --- Calibration Calculation ---
        calc_frame = ttk.LabelFrame(left_frame, text="Calibration")
        calc_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(calc_frame, text="Polynomial Order:").pack(anchor='w', padx=5)
        order_row = ttk.Frame(calc_frame)
        order_row.pack(fill='x', padx=5)
        for i in range(1, 5):
            ttk.Radiobutton(order_row, text=str(i), value=i,
                            variable=self.poly_order_var).pack(side='left')

        ttk.Button(calc_frame, text="Set as Keyframe & Calculate",
                   command=self._set_as_keyframe).pack(fill='x', padx=5, pady=5)

        self.cal_result_label = ttk.Label(calc_frame, text="No calibration yet")
        self.cal_result_label.pack(padx=5, pady=2)

        # --- Keyframe Summary ---
        keyframe_summary = ttk.LabelFrame(left_frame, text="Keyframe Summary")
        keyframe_summary.pack(fill='x', padx=5, pady=5)

        self.keyframe_listbox = tk.Listbox(keyframe_summary, height=5)
        self.keyframe_listbox.pack(fill='x', padx=5, pady=2)
        self.keyframe_listbox.bind('<<ListboxSelect>>', self._on_keyframe_select)

        self.keyframe_count_label = ttk.Label(keyframe_summary, text="0 keyframes (need 3-5)")
        self.keyframe_count_label.pack(padx=5, pady=2)

        ttk.Button(keyframe_summary, text="Remove Selected Keyframe",
                   command=self._remove_keyframe).pack(fill='x', padx=5, pady=2)

        ttk.Separator(calc_frame, orient='horizontal').pack(fill='x', padx=5, pady=10)

        ttk.Label(calc_frame, text="Single Frame Mode:", font=('', 9, 'bold')).pack(anchor='w', padx=5)
        ttk.Label(calc_frame, text="(Calibrate only this frame, no interpolation)",
                  font=('', 8)).pack(anchor='w', padx=5)

        ttk.Button(calc_frame, text="Calibrate This Frame Only",
                   command=self._calibrate_single_frame).pack(fill='x', padx=5, pady=5)

        self.single_frame_status = ttk.Label(calc_frame, text="", foreground='blue')
        self.single_frame_status.pack(padx=5, pady=2)

        # --- Process All ---
        ttk.Button(left_frame, text="Interpolate & Process All Frames →",
                   command=self._process_all_frames).pack(fill='x', padx=5, pady=10)

        # --- Right panel - matplotlib figure ---
        right_frame = ttk.Frame(self.calibrate_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        self.cal_fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.cal_canvas = FigureCanvasTkAgg(self.cal_fig, master=right_frame)
        self.cal_canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self.cal_canvas, right_frame)
        toolbar.update()


    # =========================================================================
    # Review Sub-Tab
    # =========================================================================
    def _calibrate_single_frame(self):
        """Calibrate only the current frame and go directly to review."""
        if len(self.temp_calibration_points) < 2:
            messagebox.showerror("Error", "Need at least 2 calibration points")
            return

        order = self.poly_order_var.get()
        coeffs, r_squared = SpectrumAnalyzer.polynomial_fit(self.temp_calibration_points, order)

        if coeffs is None:
            messagebox.showerror("Error", "Calibration fitting failed")
            return

        frame = self.frames[self.current_frame_idx]

        # Store calibration (not as keyframe for interpolation)
        frame.calibration_points = list(self.temp_calibration_points)
        frame.calibration_coeffs = coeffs
        frame.calibration_r_squared = r_squared
        # is_keyframe stays False - won't be used for interpolation

        # Extract and process spectrum
        effective_roi = self.roi.copy()
        effective_roi['y'] += frame.roi_offset_y

        frame.spectrum = SpectrumAnalyzer.extract_spectrum(
            frame.image_data, effective_roi, self.tilt_angle, self.flip_horizontal
        )

        peaks, frame.spectrum_baseline_removed = SpectrumAnalyzer.find_peaks_in_spectrum(
            frame.spectrum,
            prominence=self.prominence_var.get(),
            distance=self.distance_var.get(),
            use_baseline_removal=self.baseline_var.get(),
            baseline_degree=self.baseline_degree_var.get()
        )

        # Assign wavelengths
        for peak in peaks:
            peak['pixel'] = SpectrumAnalyzer.refine_peak_position(
                frame.spectrum_baseline_removed if frame.spectrum_baseline_removed is not None else frame.spectrum,
                peak['pixel']
            )
            peak['wavelength'] = SpectrumAnalyzer.pixel_to_wavelength(peak['pixel'], coeffs)

        frame.peaks = SpectrumAnalyzer.assign_known_lines(peaks, coeffs)

        # Update status
        self.single_frame_status.config(
            text=f"✓ Frame {self.current_frame_idx} calibrated (R²={r_squared:.6f})"
        )
        self.cal_result_label.config(text=f"Single frame R² = {r_squared:.6f}")

        # Show brief confirmation then go to review
        if r_squared < 0.99:
            if not messagebox.askyesno("Warning",
                                       f"Low R² ({r_squared:.4f}).\n\n"
                                       f"Found {len(frame.peaks)} peaks.\n\n"
                                       f"Continue to review anyway?"):
                return

        # Go to review tab for this frame
        self._go_to_single_frame_review()

    def _go_to_single_frame_review(self):
        """Switch to review tab and show current single-calibrated frame."""
        # We don't set analysis_complete=True since this is single frame mode
        # But we allow viewing this specific frame

        # Switch to review tab
        self.sub_notebook.select(self.review_frame)

        # Update the review display for this frame
        self._update_single_frame_review()

    def _update_single_frame_review(self):
        """Update review display for a single calibrated frame (not full analysis)."""
        if not self.frames:
            return

        frame = self.frames[self.current_frame_idx]

        # Check if this frame has calibration
        coeffs = frame.calibration_coeffs
        if coeffs is None:
            self.review_label.config(text="Frame not calibrated")
            return

        effective_roi = self.roi.copy()
        effective_roi['y'] += frame.roi_offset_y

        # Update label
        self.review_label.config(
            text=f"Frame {self.current_frame_idx} / {len(self.frames) - 1} - {frame.timestamp} μs [SINGLE FRAME MODE]"
        )

        # Update info
        info_text = f"Timestamp: {frame.timestamp} μs\n"
        info_text += f"ROI Offset: {frame.roi_offset_y} px\n"
        info_text += f"Mode: Single Frame Calibration\n"
        info_text += f"R²: {frame.calibration_r_squared:.6f}\n"
        info_text += f"Peaks found: {len(frame.peaks) if frame.peaks else 0}"
        self.frame_info_label.config(text=info_text)

        # Update peak list
        self.review_peak_listbox.delete(0, tk.END)
        if frame.peaks:
            for peak in frame.peaks[:15]:
                wl = peak.get('wavelength')
                line = peak.get('assigned_line')
                if wl:
                    if line:
                        self.review_peak_listbox.insert(
                            tk.END,
                            f"{wl:.1f}nm → {line['wavelength']}nm ({line['species']})"
                        )
                    else:
                        self.review_peak_listbox.insert(tk.END, f"{wl:.1f}nm (unassigned)")

        # Plot
        self.review_fig.clear()
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)

        # Image
        ax_img = self.review_fig.add_subplot(gs[0, 0])
        ax_img.imshow(frame.image_data, cmap='viridis', aspect='auto')
        rect = Rectangle((effective_roi['x'], effective_roi['y']),
                         effective_roi['width'], effective_roi['height'],
                         linewidth=2, edgecolor='white', facecolor='none')
        ax_img.add_patch(rect)
        ax_img.set_title(f"Frame {self.current_frame_idx} - {frame.timestamp} μs")

        # ROI zoom
        ax_roi = self.review_fig.add_subplot(gs[0, 1])
        roi_image, roi_bounds = SpectrumAnalyzer.extract_roi_image(
            frame.image_data, effective_roi, self.tilt_angle, self.flip_horizontal, expand_y=25
        )
        if roi_image.size > 0:
            ax_roi.imshow(roi_image, cmap='viridis', aspect='auto')
            ax_roi.axhline(roi_bounds[0], color='white', linestyle='--', linewidth=1, alpha=0.7)
            ax_roi.axhline(roi_bounds[1], color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax_roi.set_title("ROI Context")

        # Calibrated spectrum
        ax_spec = self.review_fig.add_subplot(gs[1, :])

        if frame.spectrum is not None:
            spectrum_to_plot = frame.spectrum_baseline_removed if frame.spectrum_baseline_removed is not None else frame.spectrum
            wavelengths = np.array([SpectrumAnalyzer.pixel_to_wavelength(i, coeffs)
                                    for i in range(len(spectrum_to_plot))])

            ax_spec.plot(wavelengths, spectrum_to_plot, 'g-', linewidth=1)

            # Mark peaks
            if frame.peaks:
                for peak in frame.peaks[:10]:
                    wl = peak.get('wavelength')
                    if wl:
                        intensity = peak.get('intensity', 0)
                        if peak.get('assigned_line'):
                            label = f"{peak['assigned_line']['wavelength']} ({peak['assigned_line']['species']})"
                            color = 'orange'
                        else:
                            label = f"{wl:.0f}"
                            color = 'gray'
                        ax_spec.plot(wl, intensity, 'o', color=color, markersize=5)
                        ax_spec.annotate(label, (wl, intensity), textcoords="offset points",
                                         xytext=(0, 10), ha='center', fontsize=8, rotation=45, color=color)

            ax_spec.set_xlim(350, 900)

        ax_spec.set_title(f"Calibrated Spectrum - Single Frame Mode (R²={frame.calibration_r_squared:.4f})")
        ax_spec.set_xlabel("Wavelength (nm)")
        ax_spec.set_ylabel("Intensity")
        ax_spec.grid(True, alpha=0.3)

        self.review_fig.tight_layout()
        self.review_canvas.draw()

    def _build_review_tab(self):
        """Build the Review sub-tab."""
        # Left panel (width=670)
        left_container = ttk.Frame(self.review_frame, width=670)
        left_container.pack(side='left', fill='y', padx=5, pady=5)
        left_container.pack_propagate(False)

        left_frame = ttk.Frame(left_container)
        left_frame.pack(fill='both', expand=True)

        # Navigation
        nav_frame = ttk.LabelFrame(left_frame, text="Frame Navigation")
        nav_frame.pack(fill='x', padx=5, pady=5)

        nav_buttons = ttk.Frame(nav_frame)
        nav_buttons.pack(fill='x', padx=5, pady=2)
        ttk.Button(nav_buttons, text="← Prev",
                   command=self._prev_review_frame).pack(side='left', expand=True, fill='x')
        ttk.Button(nav_buttons, text="Next →",
                   command=self._next_review_frame).pack(side='left', expand=True, fill='x')

        self.review_slider = ttk.Scale(nav_frame, from_=0, to=0, orient='horizontal',
                                       command=self._on_review_slider)
        self.review_slider.pack(fill='x', padx=5, pady=2)

        self.review_label = ttk.Label(nav_frame, text="Frame 0 / 0", font=('', 10, 'bold'))
        self.review_label.pack(padx=5, pady=2)

        # Frame info
        info_frame = ttk.LabelFrame(left_frame, text="Frame Information")
        info_frame.pack(fill='x', padx=5, pady=5)

        self.frame_info_label = ttk.Label(info_frame, text="", justify='left')
        self.frame_info_label.pack(padx=5, pady=5)

        # Peak summary
        peak_frame = ttk.LabelFrame(left_frame, text="Detected Peaks")
        peak_frame.pack(fill='x', padx=5, pady=5)

        self.review_peak_listbox = tk.Listbox(peak_frame, height=10)
        self.review_peak_listbox.pack(fill='x', padx=5, pady=2)

        # Right panel
        right_frame = ttk.Frame(self.review_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        self.review_fig = plt.Figure(figsize=(12, 8), dpi=100)
        self.review_canvas = FigureCanvasTkAgg(self.review_fig, master=right_frame)
        self.review_canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self.review_canvas, right_frame)
        toolbar.update()

    def _save_and_notify(self):
        """Save spectroscopy data and trigger project save."""
        if not self.frames:
            messagebox.showwarning("Warning", "No frames loaded to save")
            return

        # First update state
        self._save_to_project()


        print("=== DEBUG: Spectroscopy state after _save_to_project ===")
        print(f"spectroscopy dict exists: {hasattr(self.project_state, 'spectroscopy')}")
        if hasattr(self.project_state, 'spectroscopy'):
            print(
                f"spectroscopy keys: {self.project_state.spectroscopy.keys() if self.project_state.spectroscopy else 'None'}")
            print(f"source_directory: {self.project_state.spectroscopy.get('source_directory', 'NOT SET')}")
            print(f"frames_data count: {len(self.project_state.spectroscopy.get('frames_data', []))}")
        print("=" * 50)

        # Then save project file
        try:
            from utils.project_io import save_project

            # Check both possible path locations
            project_path = getattr(self.project_state, 'project_path', None) or self.main_app.project_filepath

            if project_path:
                save_project(self.project_state, project_path)
                messagebox.showinfo("Saved", f"Spectroscopy data saved!\n\n"
                                             f"Frames: {len(self.frames)}\n"
                                             f"Keyframes: {sum(1 for f in self.frames if f.is_keyframe)}")
            else:
                # No project path yet - prompt to save
                from tkinter import filedialog
                filepath = filedialog.asksaveasfilename(
                    defaultextension=".tgf",
                    filetypes=[("TGF Project", "*.tgf"), ("All Files", "*.*")],
                    title="Save Project As"
                )
                if filepath:
                    save_project(self.project_state, filepath)
                    self.main_app.project_filepath = filepath
                    messagebox.showinfo("Saved", f"Project saved to:\n{filepath}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save: {e}")
    # =========================================================================
    # Export Sub-Tab
    # =========================================================================

    def _build_export_tab(self):
        """Build the Export sub-tab."""
        main_frame = ttk.Frame(self.export_frame)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Summary
        summary_frame = ttk.LabelFrame(main_frame, text="Analysis Summary")
        summary_frame.pack(fill='x', pady=10)

        self.summary_label = ttk.Label(summary_frame, text="No analysis completed yet")
        self.summary_label.pack(padx=10, pady=10)

        # Export options
        export_frame = ttk.LabelFrame(main_frame, text="Export Options")
        export_frame.pack(fill='x', pady=10)

        ttk.Button(export_frame, text="Export Peak Data (CSV)",
                   command=self._export_csv).pack(fill='x', padx=10, pady=5)

        ttk.Button(export_frame, text="Export Calibration (JSON)",
                   command=self._export_calibration).pack(fill='x', padx=10, pady=5)

        ttk.Button(export_frame, text="Export All Spectra Plots (PNG)",
                   command=self._export_plots).pack(fill='x', padx=10, pady=5)

        ttk.Button(export_frame, text="Save to Project",
                   command=self._save_and_notify).pack(fill='x', padx=10, pady=5)


    # =========================================================================
    # Frame Loading and Navigation
    # =========================================================================

    def _load_frames(self):
        """Load frames from folder."""
        folder = filedialog.askdirectory(title="Select folder containing frame images")
        if not folder:
            return

        extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(extensions)])


        if not files:
            messagebox.showerror("Error", "No image files found in folder")
            return

        self.frames = []
        for i, filename in enumerate(files):
            frame = FrameData(i, filename, os.path.join(folder, filename))
            frame.load_image()
            self.frames.append(frame)

        # Update UI
        self.frames_label.config(text=f"Loaded {len(self.frames)} frames from {os.path.basename(folder)}")
        self.frame_slider.config(to=len(self.frames) - 1)
        self.cal_frame_slider.config(to=len(self.frames) - 1)
        self.review_slider.config(to=len(self.frames) - 1)
        self.current_frame_idx = 0
        self.tracking_slider.config(to=len(self.frames) - 1)

        # Store folder path in project
        self.project_state.files['spectra_directory'] = folder

        self._apply_timing()
        self._update_setup_display()

    def _apply_timing(self):
        """Apply timing to frames."""
        try:
            self.start_timestamp = int(self.start_time_var.get())
            self.frame_interval = int(self.interval_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid timing values")
            return

        for i, frame in enumerate(self.frames):
            frame.timestamp = self.start_timestamp + (i * self.frame_interval)

        self._update_nav_labels()

    def _apply_roi_values(self):
        """Apply ROI values from entries."""
        try:
            self.roi = {
                'x': int(self.roi_x_var.get()),
                'y': int(self.roi_y_var.get()),
                'width': int(self.roi_w_var.get()),
                'height': int(self.roi_h_var.get())
            }
            self._update_setup_display()
        except ValueError:
            messagebox.showerror("Error", "Invalid ROI values")

    def _adjust_frame_offset(self, delta):
        """Adjust Y offset for current frame."""
        if not self.frames:
            return
        self.frames[self.current_frame_idx].roi_offset_y += delta
        self.offset_label.config(text=f"Offset: {self.frames[self.current_frame_idx].roi_offset_y} px")
        self._update_setup_display()

    def _prev_frame(self):
        """Go to previous frame."""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.frame_slider.set(self.current_frame_idx)
            self._update_setup_display()

    def _next_frame(self):
        """Go to next frame."""
        if self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
            self.frame_slider.set(self.current_frame_idx)
            self._update_setup_display()

    def _on_slider_change(self, value):
        """Handle slider change."""
        self.current_frame_idx = int(float(value))
        self._update_setup_display()

    def _goto_frame(self):
        """Go to specific frame."""
        try:
            idx = int(self.goto_frame_var.get())
            if 0 <= idx < len(self.frames):
                self.current_frame_idx = idx
                self.frame_slider.set(idx)
                self._update_setup_display()
        except ValueError:
            pass

    def _update_nav_labels(self):
        """Update navigation labels."""
        if not self.frames:
            self.nav_label.config(text="Frame 0 / 0")
            return

        frame = self.frames[self.current_frame_idx]
        self.nav_label.config(text=f"Frame {self.current_frame_idx} / {len(self.frames) - 1} - {frame.timestamp} μs")
        self.offset_label.config(text=f"Offset: {frame.roi_offset_y} px")

    def _go_to_calibration(self):
        """Switch to calibration tab."""
        if not self.frames:
            messagebox.showerror("Error", "Please load frames first")
            return
        self.sub_notebook.select(self.calibrate_frame)
        self._update_cal_display()

    # =========================================================================
    # Mouse Events for ROI Drawing
    # =========================================================================

    def _on_mouse_press(self, event):
        """Handle mouse press."""
        if event.inaxes and event.button == 1:
            self.drawing_roi = True
            self.roi_start = (event.xdata, event.ydata)

    def _on_mouse_move(self, event):
        """Handle mouse move."""
        if self.drawing_roi and event.inaxes and self.roi_start:
            x1, y1 = self.roi_start
            x2, y2 = event.xdata, event.ydata

            self.roi = {
                'x': int(min(x1, x2)),
                'y': int(min(y1, y2)),
                'width': int(abs(x2 - x1)),
                'height': int(abs(y2 - y1))
            }
            self._update_setup_display()

    def _on_mouse_release(self, event):
        """Handle mouse release."""
        if self.drawing_roi:
            self.drawing_roi = False
            self.roi_start = None
            self.roi_x_var.set(str(self.roi['x']))
            self.roi_y_var.set(str(self.roi['y']))
            self.roi_w_var.set(str(self.roi['width']))
            self.roi_h_var.set(str(self.roi['height']))

    # =========================================================================
    # Display Updates
    # =========================================================================

    def _update_setup_display(self):
        """Update setup tab display."""
        if not self.frames:
            return

        if self.roi is None:
            self.roi = {'x': 0, 'y': 0, 'width': 100, 'height': 50}

        self.tilt_angle = self.tilt_var.get()
        self.flip_horizontal = self.flip_var.get()
        self.tilt_label.config(text=f"{self.tilt_angle:.1f}°")
        self._update_nav_labels()

        frame = self.frames[self.current_frame_idx]
        effective_roi = self.roi.copy()
        effective_roi['y'] += frame.roi_offset_y

        self.setup_fig.clear()
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1], width_ratios=[1.5, 1],
                               hspace=0.3, wspace=0.3)

        # Main image
        ax_main = self.setup_fig.add_subplot(gs[0, :])
        ax_main.imshow(frame.image_data, cmap='viridis', aspect='auto')
        rect = Rectangle((effective_roi['x'], effective_roi['y']),
                         effective_roi['width'], effective_roi['height'],
                         linewidth=2, edgecolor='white', facecolor='none')
        ax_main.add_patch(rect)
        ax_main.set_title(f"Frame {self.current_frame_idx} - {frame.timestamp} μs - {frame.filename}")
        ax_main.set_xlabel("Pixels")
        ax_main.set_ylabel("Pixels")

        # ROI zoom
        ax_roi = self.setup_fig.add_subplot(gs[1, 0])
        roi_image, roi_bounds = SpectrumAnalyzer.extract_roi_image(
            frame.image_data, effective_roi, self.tilt_angle, self.flip_horizontal, expand_y=25
        )
        if roi_image.size > 0:
            ax_roi.imshow(roi_image, cmap='viridis', aspect='auto')
            ax_roi.axhline(roi_bounds[0], color='white', linestyle='--', linewidth=1, alpha=0.7)
            ax_roi.axhline(roi_bounds[1], color='white', linestyle='--', linewidth=1, alpha=0.7)

        title_suffix = ""
        if self.tilt_angle != 0:
            title_suffix += f" (Tilt: {self.tilt_angle:.1f}°)"
        if self.flip_horizontal:
            title_suffix += " [FLIPPED]"
        ax_roi.set_title(f"ROI Context (±25px){title_suffix}")

        # Spectrum preview
        ax_spec = self.setup_fig.add_subplot(gs[1, 1])
        spectrum = SpectrumAnalyzer.extract_spectrum(
            frame.image_data, effective_roi, self.tilt_angle, self.flip_horizontal
        )
        ax_spec.plot(spectrum, 'b-', linewidth=1)
        ax_spec.set_title("Spectrum Preview")
        ax_spec.set_xlabel("Pixel")
        ax_spec.set_ylabel("Intensity")
        ax_spec.grid(True, alpha=0.3)

        self.setup_fig.tight_layout()
        self.setup_canvas.draw()

    # =========================================================================
    # Calibration Methods
    # =========================================================================

    def _cal_change_frame(self, delta):
        """Change frame in calibration tab."""
        if not self.frames:
            return

        new_idx = max(0, min(self.current_frame_idx + delta, len(self.frames) - 1))
        if new_idx != self.current_frame_idx:
            self.current_frame_idx = new_idx
            self.cal_frame_slider.set(new_idx)
            self.frame_slider.set(new_idx)
            self._load_frame_calibration()
            self._update_cal_display()

    def _on_cal_frame_slider(self, value):
        """Handle calibration slider."""
        new_idx = int(float(value))
        if new_idx != self.current_frame_idx:
            self.current_frame_idx = new_idx
            self.frame_slider.set(new_idx)
            self._load_frame_calibration()
            self._update_cal_display()

    def _load_frame_calibration(self):
        """Load calibration points from current frame if it's a keyframe."""
        if not self.frames:
            return

        frame = self.frames[self.current_frame_idx]
        if frame.is_keyframe:
            self.temp_calibration_points = list(frame.calibration_points)
        else:
            self.temp_calibration_points = []

        self._update_cal_listbox()

    def _detect_peaks(self):
        """Detect peaks in current frame."""
        if not self.frames:
            return

        frame = self.frames[self.current_frame_idx]
        effective_roi = self.roi.copy()
        effective_roi['y'] += frame.roi_offset_y

        spectrum = SpectrumAnalyzer.extract_spectrum(
            frame.image_data, effective_roi, self.tilt_angle, self.flip_horizontal
        )

        self.detected_peaks, self.current_spectrum_display = SpectrumAnalyzer.find_peaks_in_spectrum(
            spectrum,
            prominence=self.prominence_var.get(),
            distance=self.distance_var.get(),
            use_baseline_removal=self.baseline_var.get(),
            baseline_degree=self.baseline_degree_var.get()
        )

        # Refine peaks
        for peak in self.detected_peaks:
            peak['pixel'] = SpectrumAnalyzer.refine_peak_position(
                self.current_spectrum_display, peak['pixel']
            )

        # Update listbox
        self.peak_listbox.delete(0, tk.END)
        for i, peak in enumerate(self.detected_peaks[:15]):
            self.peak_listbox.insert(tk.END, f"Peak {i + 1}: px={peak['pixel']:.1f}, I={peak['intensity']:.1f}")

        # Update labels
        self.prom_label.config(text=f"{self.prominence_var.get():.2f}")
        self.dist_label.config(text=f"{self.distance_var.get()}")
        self.degree_label.config(text=f"{self.baseline_degree_var.get()}")

        self._update_cal_display()

    def _on_peak_select(self, event):
        """Handle peak selection."""
        selection = self.peak_listbox.curselection()
        if selection:
            self.selected_peak_idx = selection[0]
            self._update_cal_display()

    def _assign_wavelength(self):
        """Assign wavelength to selected peak."""
        if self.selected_peak_idx is None:
            messagebox.showwarning("Warning", "Please select a peak first")
            return

        wl_str = self.wavelength_var.get()
        if not wl_str:
            messagebox.showwarning("Warning", "Please select a wavelength")
            return

        try:
            wavelength = int(wl_str.split()[0])
        except:
            messagebox.showerror("Error", "Invalid wavelength")
            return

        pixel = self.detected_peaks[self.selected_peak_idx]['pixel']

        # Remove existing point at same pixel
        self.temp_calibration_points = [(p, w) for p, w in self.temp_calibration_points if abs(p - pixel) > 1]
        self.temp_calibration_points.append((pixel, wavelength))
        self.temp_calibration_points.sort(key=lambda x: x[0])

        self._update_cal_listbox()
        self._update_cal_display()

    def _update_cal_listbox(self):
        """Update calibration points listbox."""
        self.cal_listbox.delete(0, tk.END)
        for px, wl in self.temp_calibration_points:
            species = KNOWN_LINES.get(wl, ('?', 0, False))[0]
            self.cal_listbox.insert(tk.END, f"px={px:.1f} → {wl}nm ({species})")

    def _remove_cal_point(self):
        """Remove selected calibration point."""
        selection = self.cal_listbox.curselection()
        if selection:
            del self.temp_calibration_points[selection[0]]
            self._update_cal_listbox()
            self._update_cal_display()

    def _clear_temp_calibration(self):
        """Clear temporary calibration points."""
        self.temp_calibration_points = []
        self._update_cal_listbox()
        self._update_cal_display()

    def _set_as_keyframe(self):
        """Set current frame as keyframe with calibration."""
        if len(self.temp_calibration_points) < 2:
            messagebox.showerror("Error", "Need at least 2 calibration points")
            return

        order = self.poly_order_var.get()
        coeffs, r_squared = SpectrumAnalyzer.polynomial_fit(self.temp_calibration_points, order)

        if coeffs is None:
            messagebox.showerror("Error", "Calibration fitting failed")
            return

        # Store in frame
        frame = self.frames[self.current_frame_idx]
        frame.is_keyframe = True
        frame.calibration_points = list(self.temp_calibration_points)
        frame.calibration_coeffs = coeffs
        frame.calibration_r_squared = r_squared

        # Update UI
        self.cal_result_label.config(text=f"R² = {r_squared:.6f}")
        self._update_keyframe_summary()
        self._update_cal_display()

        if r_squared < 0.99:
            messagebox.showwarning("Warning", f"Low R² ({r_squared:.4f}). Consider adjusting points.")
        else:
            messagebox.showinfo("Success", f"Keyframe set! R² = {r_squared:.6f}")

    def _update_keyframe_summary(self):
        """Update keyframe summary listbox."""
        self.keyframe_listbox.delete(0, tk.END)

        keyframe_count = 0
        for i, frame in enumerate(self.frames):
            if frame.is_keyframe:
                marker = "→ " if i == self.current_frame_idx else ""
                self.keyframe_listbox.insert(
                    tk.END,
                    f"{marker}Frame {i}: R²={frame.calibration_r_squared:.4f}, {len(frame.calibration_points)} pts"
                )
                keyframe_count += 1

        color = 'green' if keyframe_count >= 3 else ('orange' if keyframe_count > 0 else 'red')
        self.keyframe_count_label.config(
            text=f"{keyframe_count} keyframes (need 3-5)",
            foreground=color
        )

    def _on_keyframe_select(self, event):
        """Jump to selected keyframe."""
        selection = self.keyframe_listbox.curselection()
        if not selection:
            return

        text = self.keyframe_listbox.get(selection[0])
        import re
        match = re.search(r'Frame (\d+):', text)
        if match:
            idx = int(match.group(1))
            self.current_frame_idx = idx
            self.cal_frame_slider.set(idx)
            self.frame_slider.set(idx)
            self._load_frame_calibration()
            self._update_cal_display()

    def _remove_keyframe(self):
        """Remove selected keyframe."""
        selection = self.keyframe_listbox.curselection()
        if not selection:
            return

        text = self.keyframe_listbox.get(selection[0])
        import re
        match = re.search(r'Frame (\d+):', text)
        if match:
            idx = int(match.group(1))
            frame = self.frames[idx]
            frame.is_keyframe = False
            frame.calibration_points = []
            frame.calibration_coeffs = None
            frame.calibration_r_squared = 0.0

            self._update_keyframe_summary()
            if idx == self.current_frame_idx:
                self.temp_calibration_points = []
                self._update_cal_listbox()
            self._update_cal_display()

    def _update_cal_display(self):
        """Update calibration display."""
        if not self.frames:
            return

        if self.roi is None:
            self.roi = {'x': 0, 'y': 0, 'width': 100, 'height': 50}

        frame = self.frames[self.current_frame_idx]
        effective_roi = self.roi.copy()
        effective_roi['y'] += frame.roi_offset_y

        # Update frame label
        self.cal_frame_label.config(text=f"Frame: {self.current_frame_idx} / {len(self.frames) - 1}")

        # Update keyframe indicator
        if frame.is_keyframe:
            self.keyframe_indicator.config(text=f"★ KEYFRAME (R²={frame.calibration_r_squared:.4f})",
                                           foreground='green')
        else:
            self.keyframe_indicator.config(text="Not a keyframe", foreground='gray')

        # Get spectrum
        if self.current_spectrum_display is not None:
            spectrum_display = self.current_spectrum_display
        else:
            spectrum = SpectrumAnalyzer.extract_spectrum(
                frame.image_data, effective_roi, self.tilt_angle, self.flip_horizontal
            )
            spectrum_display, _ = SpectrumAnalyzer.remove_baseline(spectrum, self.baseline_degree_var.get())

        self.cal_fig.clear()
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

        # Image with ROI
        ax_img = self.cal_fig.add_subplot(gs[0, 0])
        ax_img.imshow(frame.image_data, cmap='viridis', aspect='auto')
        rect = Rectangle((effective_roi['x'], effective_roi['y']),
                         effective_roi['width'], effective_roi['height'],
                         linewidth=2, edgecolor='white', facecolor='none')
        ax_img.add_patch(rect)
        title = f"Frame {self.current_frame_idx}"
        if frame.is_keyframe:
            title += " ★"
        ax_img.set_title(title)

        # Spectrum with peaks
        ax_spec = self.cal_fig.add_subplot(gs[0, 1])
        ax_spec.plot(spectrum_display, 'b-', linewidth=1)
        ax_spec.set_xlabel("Pixel")
        ax_spec.set_ylabel("Intensity")
        ax_spec.set_title("Spectrum with Peaks")
        ax_spec.grid(True, alpha=0.3)

        # Mark peaks
        assigned_pixels = {px for px, wl in self.temp_calibration_points}

        for i, peak in enumerate(self.detected_peaks[:15]):
            x = peak['pixel']
            y = peak['intensity']

            is_assigned = any(abs(px - x) < 2 for px in assigned_pixels)

            if is_assigned:
                ax_spec.plot(x, y, 'o', color='orange', markersize=8)
                for px, wl in self.temp_calibration_points:
                    if abs(px - x) < 2:
                        ax_spec.annotate(f"{wl}nm", (x, y), textcoords="offset points",
                                         xytext=(0, 8), ha='center', fontsize=9, color='orange')
                        break
            elif i == self.selected_peak_idx:
                ax_spec.plot(x, y, 'o', color='red', markersize=8)
            else:
                ax_spec.plot(x, y, 'o', color='green', markersize=6)

        # Calibration fit plot
        ax_cal = self.cal_fig.add_subplot(gs[1, :])
        if self.temp_calibration_points:
            pixels = [p[0] for p in self.temp_calibration_points]
            wavelengths = [p[1] for p in self.temp_calibration_points]
            ax_cal.scatter(pixels, wavelengths, s=100, c='red', zorder=5, label='Calibration Points')

            # If we have enough points, show fit
            if len(self.temp_calibration_points) >= 2:
                coeffs, r_sq = SpectrumAnalyzer.polynomial_fit(
                    self.temp_calibration_points, self.poly_order_var.get()
                )
                if coeffs is not None:
                    px_range = np.linspace(min(pixels) - 20, max(pixels) + 20, 100)
                    wl_fit = np.polyval(coeffs, px_range)
                    ax_cal.plot(px_range, wl_fit, 'b-', linewidth=2, label=f'Fit (R²={r_sq:.6f})')
                    ax_cal.legend()

        ax_cal.set_xlabel("Pixel")
        ax_cal.set_ylabel("Wavelength (nm)")
        ax_cal.set_title("Wavelength Calibration")
        ax_cal.grid(True, alpha=0.3)

        self.cal_fig.tight_layout()
        self.cal_canvas.draw()

        # Update keyframe summary
        self._update_keyframe_summary()

    # =========================================================================
    # Processing
    # =========================================================================

    def _process_all_frames(self):
        """Interpolate coefficients and process all frames."""
        # Get keyframes
        keyframes_data = {}
        for i, frame in enumerate(self.frames):
            if frame.is_keyframe and frame.calibration_coeffs is not None:
                keyframes_data[i] = frame.calibration_coeffs

        if len(keyframes_data) < 1:
            messagebox.showerror("Error", "Need at least 1 keyframe")
            return

        if len(keyframes_data) < 3:
            if not messagebox.askyesno("Warning",
                                       f"Only {len(keyframes_data)} keyframe(s). 3-5 recommended.\nContinue?"):
                return

        # Progress window
        progress = tk.Toplevel(self.winfo_toplevel())
        progress.title("Processing...")
        progress.geometry("350x120")
        progress_label = ttk.Label(progress, text="Interpolating and processing...")
        progress_label.pack(pady=10)
        progress_bar = ttk.Progressbar(progress, length=300, mode='determinate')
        progress_bar.pack(pady=10)

        # Process each frame
        for i, frame in enumerate(self.frames):
            progress_label.config(text=f"Processing frame {i + 1}/{len(self.frames)}")
            progress_bar['value'] = (i / len(self.frames)) * 100
            progress.update()

            # Get coefficients (from keyframe or interpolated)
            if frame.is_keyframe:
                coeffs = frame.calibration_coeffs
            else:
                coeffs = SpectrumAnalyzer.interpolate_coefficients(i, keyframes_data)
                frame.interpolated_coeffs = coeffs

            if coeffs is None:
                continue

            # Extract spectrum
            effective_roi = self.roi.copy()
            effective_roi['y'] += frame.roi_offset_y

            frame.spectrum = SpectrumAnalyzer.extract_spectrum(
                frame.image_data, effective_roi, self.tilt_angle, self.flip_horizontal
            )

            # Find peaks
            peaks, frame.spectrum_baseline_removed = SpectrumAnalyzer.find_peaks_in_spectrum(
                frame.spectrum,
                prominence=self.prominence_var.get(),
                distance=self.distance_var.get(),
                use_baseline_removal=self.baseline_var.get(),
                baseline_degree=self.baseline_degree_var.get()
            )

            # Assign wavelengths
            for peak in peaks:
                peak['pixel'] = SpectrumAnalyzer.refine_peak_position(
                    frame.spectrum_baseline_removed if frame.spectrum_baseline_removed is not None else frame.spectrum,
                    peak['pixel']
                )
                peak['wavelength'] = SpectrumAnalyzer.pixel_to_wavelength(peak['pixel'], coeffs)

            frame.peaks = SpectrumAnalyzer.assign_known_lines(peaks, coeffs)

        progress.destroy()
        self.analysis_complete = True

        # Update summary
        self._update_export_summary()

        messagebox.showinfo("Complete", f"Processed {len(self.frames)} frames")

        # Switch to review
        self.sub_notebook.select(self.review_frame)
        self._update_review_display()

    # =========================================================================
    # Review Methods
    # =========================================================================

    def _prev_review_frame(self):
        """Previous review frame."""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.review_slider.set(self.current_frame_idx)
            self._update_review_display()

    def _next_review_frame(self):
        """Next review frame."""
        if self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
            self.review_slider.set(self.current_frame_idx)
            self._update_review_display()

    def _on_review_slider(self, value):
        """Handle review slider."""
        self.current_frame_idx = int(float(value))
        self._update_review_display()

    def _update_review_display(self):
        """Update review display."""
        if not self.frames or not self.analysis_complete:
            return

        frame = self.frames[self.current_frame_idx]

        if not self.analysis_complete:
            if frame.calibration_coeffs is not None:
                self._update_single_frame_review()
            return

        effective_roi = self.roi.copy()
        effective_roi['y'] += frame.roi_offset_y

        # Update label
        keyframe_str = " ★KEYFRAME" if frame.is_keyframe else ""
        self.review_label.config(
            text=f"Frame {self.current_frame_idx} / {len(self.frames) - 1} - {frame.timestamp} μs{keyframe_str}"
        )

        # Update info
        coeffs = frame.get_effective_coeffs()
        info_text = f"Timestamp: {frame.timestamp} μs\n"
        info_text += f"ROI Offset: {frame.roi_offset_y} px\n"
        info_text += f"Keyframe: {'Yes' if frame.is_keyframe else 'No'}\n"
        if frame.is_keyframe:
            info_text += f"R²: {frame.calibration_r_squared:.6f}\n"
        info_text += f"Peaks found: {len(frame.peaks) if frame.peaks else 0}"
        self.frame_info_label.config(text=info_text)

        # Update peak list
        self.review_peak_listbox.delete(0, tk.END)
        if frame.peaks:
            for peak in frame.peaks[:15]:
                wl = peak.get('wavelength')
                line = peak.get('assigned_line')
                if wl:
                    if line:
                        self.review_peak_listbox.insert(
                            tk.END,
                            f"{wl:.1f}nm → {line['wavelength']}nm ({line['species']})"
                        )
                    else:
                        self.review_peak_listbox.insert(tk.END, f"{wl:.1f}nm (unassigned)")

        # Plot
        self.review_fig.clear()
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)

        # Image
        ax_img = self.review_fig.add_subplot(gs[0, 0])
        ax_img.imshow(frame.image_data, cmap='viridis', aspect='auto')
        rect = Rectangle((effective_roi['x'], effective_roi['y']),
                         effective_roi['width'], effective_roi['height'],
                         linewidth=2, edgecolor='white', facecolor='none')
        ax_img.add_patch(rect)
        ax_img.set_title(f"Frame {self.current_frame_idx} - {frame.timestamp} μs")

        # ROI zoom
        ax_roi = self.review_fig.add_subplot(gs[0, 1])
        roi_image, roi_bounds = SpectrumAnalyzer.extract_roi_image(
            frame.image_data, effective_roi, self.tilt_angle, self.flip_horizontal, expand_y=25
        )
        if roi_image.size > 0:
            ax_roi.imshow(roi_image, cmap='viridis', aspect='auto')
            ax_roi.axhline(roi_bounds[0], color='white', linestyle='--', linewidth=1, alpha=0.7)
            ax_roi.axhline(roi_bounds[1], color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax_roi.set_title("ROI Context")


        # Calibrated spectrum
        ax_spec = self.review_fig.add_subplot(gs[1, :])

        if frame.spectrum is not None and coeffs is not None:
            spectrum_to_plot = frame.spectrum_baseline_removed if frame.spectrum_baseline_removed is not None else frame.spectrum
            wavelengths = np.array([SpectrumAnalyzer.pixel_to_wavelength(i, coeffs)
                                    for i in range(len(spectrum_to_plot))])

            ax_spec.plot(wavelengths, spectrum_to_plot, 'g-', linewidth=1)

            # Mark peaks
            if frame.peaks:
                for peak in frame.peaks[:10]:
                    wl = peak.get('wavelength')
                    if wl:
                        intensity = peak.get('intensity', 0)
                        if peak.get('assigned_line'):
                            label = f"{peak['assigned_line']['wavelength']} ({peak['assigned_line']['species']})"
                            color = 'orange'
                        else:
                            label = f"{wl:.0f}"
                            color = 'gray'
                        ax_spec.plot(wl, intensity, 'o', color=color, markersize=5)
                        ax_spec.annotate(label, (wl, intensity), textcoords="offset points",
                                         xytext=(0, 10), ha='center', fontsize=8, rotation=45, color=color)

            ax_spec.set_xlim(350, 900)

        title = f"Calibrated Spectrum"
        if frame.is_keyframe:
            title += " ★KEYFRAME"
        ax_spec.set_title(title)
        ax_spec.set_xlabel("Wavelength (nm)")
        ax_spec.set_ylabel("Intensity")
        ax_spec.grid(True, alpha=0.3)

        self.review_fig.tight_layout()
        self.review_canvas.draw()

    def _add_manual_peak(self):
        """Add a manually specified peak to calibration."""
        try:
            pixel = float(self.manual_pixel_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid pixel value")
            return

        wl_str = self.manual_wavelength_var.get()
        if not wl_str:
            messagebox.showerror("Error", "Please specify a wavelength")
            return

        try:
            wavelength = int(wl_str.split()[0])
        except ValueError:
            messagebox.showerror("Error", "Invalid wavelength value")
            return

        # Remove existing point at similar pixel position
        self.temp_calibration_points = [(p, w) for p, w in self.temp_calibration_points if abs(p - pixel) > 1]
        self.temp_calibration_points.append((pixel, wavelength))
        self.temp_calibration_points.sort(key=lambda x: x[0])

        # Clear inputs
        self.manual_pixel_var.set("")

        self._update_cal_listbox()
        self._update_cal_display()

        messagebox.showinfo("Added", f"Added calibration point: pixel {pixel:.1f} → {wavelength} nm")

    def _toggle_click_mode(self):
        """Toggle click-to-select-pixel mode."""
        if self.click_mode_var.get():
            self.click_status_label.config(text="Click on spectrum plot to select pixel position")
            self._cal_click_cid = self.cal_canvas.mpl_connect('button_press_event', self._on_spectrum_click)
        else:
            self.click_status_label.config(text="")
            if hasattr(self, '_cal_click_cid'):
                self.cal_canvas.mpl_disconnect(self._cal_click_cid)

    def _on_spectrum_click(self, event):
        """Handle click on spectrum plot to get pixel position."""
        if not self.click_mode_var.get():
            return
        if event.inaxes is None:
            return
        pixel = event.xdata
        if pixel is None or pixel < 0:
            return
        self.manual_pixel_var.set(f"{pixel:.1f}")
        self.click_status_label.config(text=f"Selected pixel: {pixel:.1f} - now assign wavelength")
    # =========================================================================
    # Export Methods
    # =========================================================================

    def _update_export_summary(self):
        """Update export summary."""
        if not self.analysis_complete:
            return

        keyframe_count = sum(1 for f in self.frames if f.is_keyframe)
        total_peaks = sum(len(f.peaks) if f.peaks else 0 for f in self.frames)

        summary = f"Frames: {len(self.frames)}\n"
        summary += f"Keyframes: {keyframe_count}\n"
        summary += f"Total Peaks: {total_peaks}\n"
        if self.frames:
            summary += f"Time Range: {self.frames[0].timestamp} - {self.frames[-1].timestamp} μs"

        self.summary_label.config(text=summary)

    def _export_csv(self):
        """Export peak data to CSV."""
        if not self.analysis_complete:
            messagebox.showerror("Error", "No analysis results")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not filepath:
            return

        with open(filepath, 'w') as f:
            f.write("Frame,Timestamp_us,Pixel,Wavelength_nm,Intensity,Assigned_Line,Species\n")
            for frame in self.frames:
                if frame.peaks:
                    for peak in frame.peaks:
                        wl = peak.get('wavelength', 'N/A')
                        line = peak.get('assigned_line')
                        f.write(f"{frame.index},{frame.timestamp},{peak['pixel']:.2f},")
                        f.write(f"{wl:.2f if isinstance(wl, float) else wl},{peak['intensity']:.2f},")
                        f.write(f"{line['wavelength'] if line else 'N/A'},{line['species'] if line else 'N/A'}\n")

        messagebox.showinfo("Success", f"Exported to {filepath}")

    def _export_calibration(self):
        """Export calibration to JSON."""
        import json

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not filepath:
            return

        data = {
            'roi': self.roi,
            'tilt_angle': self.tilt_angle,
            'flip_horizontal': self.flip_horizontal,
            'keyframes': []
        }

        for frame in self.frames:
            if frame.is_keyframe:
                data['keyframes'].append({
                    'frame_index': frame.index,
                    'calibration_points': frame.calibration_points,
                    'coefficients': frame.calibration_coeffs.tolist() if frame.calibration_coeffs is not None else None,
                    'r_squared': frame.calibration_r_squared
                })

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        messagebox.showinfo("Success", f"Exported to {filepath}")

    def _export_plots(self):
        """Export all spectrum plots."""
        if not self.analysis_complete:
            messagebox.showerror("Error", "No analysis results")
            return

        folder = filedialog.askdirectory(title="Select output folder")
        if not folder:
            return

        progress = tk.Toplevel(self.winfo_toplevel())
        progress.title("Exporting...")
        progress_bar = ttk.Progressbar(progress, length=250, mode='determinate')
        progress_bar.pack(pady=20, padx=20)

        for i, frame in enumerate(self.frames):
            progress_bar['value'] = (i / len(self.frames)) * 100
            progress.update()

            fig = self._create_frame_figure(frame)
            fig.savefig(os.path.join(folder, f"spectrum_{i:04d}_{frame.timestamp}us.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        progress.destroy()
        messagebox.showinfo("Success", f"Exported {len(self.frames)} plots")

    def _create_frame_figure(self, frame):
        """Create figure for a single frame."""
        effective_roi = self.roi.copy()
        effective_roi['y'] += frame.roi_offset_y
        coeffs = frame.get_effective_coeffs()

        fig = plt.figure(figsize=(10, 2))
        ax = fig.add_subplot(111)

        if frame.spectrum is not None and coeffs is not None:
            spectrum = frame.spectrum_baseline_removed if frame.spectrum_baseline_removed is not None else frame.spectrum
            wavelengths = np.array([SpectrumAnalyzer.pixel_to_wavelength(i, coeffs)
                                    for i in range(len(spectrum))])
            ax.plot(wavelengths, spectrum, 'g-', linewidth=1)
            ax.set_xlim(350, 900)

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Frame {frame.index} - {frame.timestamp} μs")

        fig.tight_layout()
        return fig

    # =========================================================================
    # Project Save/Load
    # =========================================================================
    def _save_to_project(self):
        """Save all spectroscopy settings and data to project state (no file write)."""
        print("\n" + "=" * 60)
        print("SPECTROSCOPY _save_to_project() CALLED")
        print("=" * 60)

        print(f"  self.frames exists: {self.frames is not None}")
        print(f"  self.frames count: {len(self.frames) if self.frames else 0}")
        print(f"  self.roi: {self.roi}")

        state = self.main_app.project_state

        print(f"  project_state exists: {state is not None}")
        print(f"  project_state.spectroscopy type: {type(state.spectroscopy)}")

        # Ensure spectroscopy dict exists
        if not hasattr(state, 'spectroscopy') or state.spectroscopy is None:
            print("  Creating new spectroscopy dict")
            state.spectroscopy = {}

        # Core settings - ALWAYS save these
        state.spectroscopy['roi'] = {
            'x': int(self.roi['x']),
            'y': int(self.roi['y']),
            'width': int(self.roi['width']),
            'height': int(self.roi['height']),
        } if self.roi else None
        print(f"  Saved ROI: {state.spectroscopy['roi']}")

        state.spectroscopy['tilt_angle'] = float(self.tilt_angle)
        state.spectroscopy['flip_horizontal'] = bool(self.flip_horizontal)
        state.spectroscopy['start_timestamp'] = int(self.start_timestamp)
        state.spectroscopy['frame_interval'] = int(self.frame_interval)

        # Peak detection settings - ALWAYS save
        state.spectroscopy['peak_prominence'] = float(self.prominence_var.get())
        state.spectroscopy['peak_distance'] = int(self.distance_var.get())
        state.spectroscopy['baseline_removal'] = bool(self.baseline_var.get())
        state.spectroscopy['baseline_degree'] = int(self.baseline_degree_var.get())
        state.spectroscopy['poly_order'] = int(self.poly_order_var.get())

        # Save frame data only if frames exist
        if self.frames:
            frames_data = []
            for frame in self.frames:
                frame_dict = {
                    'index': int(frame.index),
                    'filename': str(frame.filename),
                    'timestamp': int(frame.timestamp) if frame.timestamp is not None else 0,
                    'roi_offset_y': int(frame.roi_offset_y),
                    'is_keyframe': bool(frame.is_keyframe),
                    'calibration_points': [[float(p), int(w)] for p, w in
                                           frame.calibration_points] if frame.calibration_points else [],
                    'calibration_coeffs': [float(c) for c in
                                           frame.calibration_coeffs.tolist()] if frame.calibration_coeffs is not None else None,
                    'calibration_r_squared': float(frame.calibration_r_squared),
                    'interpolated_coeffs': [float(c) for c in
                                            frame.interpolated_coeffs.tolist()] if frame.interpolated_coeffs is not None else None,
                }

                # Save peaks if they exist
                if frame.peaks:
                    peaks_data = []
                    for peak in frame.peaks:
                        peak_data = {
                            'pixel': float(peak.get('pixel', 0)),
                            'intensity': float(peak.get('intensity', 0)),
                            'wavelength': float(peak.get('wavelength')) if peak.get('wavelength') is not None else None,
                        }
                        if peak.get('assigned_line'):
                            line = peak['assigned_line']
                            peak_data['assigned_line'] = {
                                'wavelength': int(line.get('wavelength', 0)),
                                'species': str(line.get('species', '')),
                                'priority': int(line.get('priority', 0)),
                                'is_anchor': bool(line.get('is_anchor', False)),
                            }
                        peaks_data.append(peak_data)
                    frame_dict['peaks'] = peaks_data

                frames_data.append(frame_dict)

            state.spectroscopy['frames_data'] = frames_data
            print(f"  Saved {len(frames_data)} frames")

            # Count keyframes
            keyframe_count = sum(1 for f in self.frames if f.is_keyframe)
            print(f"  Keyframes: {keyframe_count}")

            # Store source directory
            if self.frames[0].image_path:
                state.spectroscopy['source_directory'] = str(os.path.dirname(self.frames[0].image_path))
                print(f"  Source directory: {state.spectroscopy['source_directory']}")
        else:
            state.spectroscopy['frames_data'] = []
            state.spectroscopy['source_directory'] = None
            print("  No frames to save")

        state.spectroscopy['analysis_complete'] = bool(self.analysis_complete)
        state.spectroscopy['current_frame_idx'] = int(self.current_frame_idx)

        print(f"  Final spectroscopy keys: {list(state.spectroscopy.keys())}")
        print("=" * 60 + "\n")

    def load_from_project(self):
        """Load all spectroscopy settings and data from project."""
        print("\n" + "=" * 60)
        print("SPECTROSCOPY load_from_project() CALLED")
        print("=" * 60)

        state = self.main_app.project_state
        spec = state.spectroscopy

        print(f"  state.spectroscopy type: {type(spec)}")
        print(f"  state.spectroscopy keys: {list(spec.keys()) if spec else 'None/Empty'}")

        if not spec:
            print("  spec is empty/None, returning early")
            print("=" * 60 + "\n")
            return

        # Core settings - only load ROI if it exists and is valid
        if spec.get('roi') and isinstance(spec['roi'], dict):
            self.roi = spec['roi']
            self.roi_x_var.set(str(self.roi.get('x', 0)))
            self.roi_y_var.set(str(self.roi.get('y', 0)))
            self.roi_w_var.set(str(self.roi.get('width', 100)))
            self.roi_h_var.set(str(self.roi.get('height', 50)))
            print(f"  Loaded ROI: {self.roi}")
        else:
            print(f"  No ROI to load (spec['roi'] = {spec.get('roi')})")

        if 'tilt_angle' in spec:
            self.tilt_angle = spec['tilt_angle']
            self.tilt_var.set(self.tilt_angle)
            print(f"  Loaded tilt_angle: {self.tilt_angle}")

        if 'flip_horizontal' in spec:
            self.flip_horizontal = spec['flip_horizontal']
            self.flip_var.set(self.flip_horizontal)

        if 'start_timestamp' in spec:
            self.start_timestamp = spec['start_timestamp']
            self.start_time_var.set(str(self.start_timestamp))

        if 'frame_interval' in spec:
            self.frame_interval = spec['frame_interval']
            self.interval_var.set(str(self.frame_interval))

        # Peak detection settings
        if 'peak_prominence' in spec:
            self.prominence_var.set(spec['peak_prominence'])
        if 'peak_distance' in spec:
            self.distance_var.set(spec['peak_distance'])
        if 'baseline_removal' in spec:
            self.baseline_var.set(spec['baseline_removal'])
        if 'baseline_degree' in spec:
            self.baseline_degree_var.set(spec['baseline_degree'])
        if 'poly_order' in spec:
            self.poly_order_var.set(spec['poly_order'])

        # Load frames if source directory exists
        source_dir = spec.get('source_directory')
        print(f"  source_directory from spec: {source_dir}")

        if source_dir and os.path.exists(source_dir):
            print(f"  Source directory exists, loading frames...")
            folder = source_dir
            extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
            files = sorted([f for f in os.listdir(folder) if f.lower().endswith(extensions)])
            print(f"  Found {len(files)} image files")

            if files:
                self.frames = []
                for i, filename in enumerate(files):
                    frame = FrameData(i, filename, os.path.join(folder, filename))
                    frame.load_image()
                    self.frames.append(frame)

                # Update sliders
                self.frame_slider.config(to=len(self.frames) - 1)
                self.cal_frame_slider.config(to=len(self.frames) - 1)
                self.review_slider.config(to=len(self.frames) - 1)
                self.tracking_slider.config(to=len(self.frames) - 1)
                self.frames_label.config(text=f"Loaded {len(self.frames)} frames from {os.path.basename(folder)}")

                # Restore frame-specific data
                frames_data = spec.get('frames_data', [])
                print(f"  frames_data count: {len(frames_data)}")

                for frame_data in frames_data:
                    idx = frame_data.get('index', 0)
                    if idx < len(self.frames):
                        frame = self.frames[idx]
                        frame.timestamp = frame_data.get('timestamp')
                        frame.roi_offset_y = frame_data.get('roi_offset_y', 0)
                        frame.is_keyframe = frame_data.get('is_keyframe', False)
                        frame.calibration_points = frame_data.get('calibration_points', [])

                        coeffs = frame_data.get('calibration_coeffs')
                        frame.calibration_coeffs = np.array(coeffs) if coeffs else None

                        frame.calibration_r_squared = frame_data.get('calibration_r_squared', 0.0)

                        interp_coeffs = frame_data.get('interpolated_coeffs')
                        frame.interpolated_coeffs = np.array(interp_coeffs) if interp_coeffs else None

                        # Restore peaks
                        if 'peaks' in frame_data:
                            frame.peaks = frame_data['peaks']

                # Count restored keyframes
                keyframe_count = sum(1 for f in self.frames if f.is_keyframe)
                print(f"  Restored {keyframe_count} keyframes")
        else:
            if source_dir:
                print(f"  WARNING: Source directory does not exist: {source_dir}")
            else:
                print("  No source_directory specified")

        # Restore state
        self.analysis_complete = spec.get('analysis_complete', False)
        self.current_frame_idx = spec.get('current_frame_idx', 0)
        print(f"  analysis_complete: {self.analysis_complete}")
        print(f"  current_frame_idx: {self.current_frame_idx}")

        # Update displays only if we have valid data
        if self.frames:
            print("  Updating displays...")
            self._apply_timing()
            if self.roi:
                self._update_setup_display()
            self._update_tracking_status()
            self._update_keyframe_summary()
            if self.analysis_complete:
                self._update_export_summary()

        print("=" * 60 + "\n")