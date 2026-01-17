"""
TGF Analysis Tool - Photometry Tab
==================================
Handles photometer data visualization and ratio analysis.

Features:
- Load photometer binary data (337nm, 391nm, 777nm channels)
- Plot waveforms with optional FA/INTF overlay
- Compute and plot channel ratios (337/777, 391/777, 337/391)
- Time alignment with second offset
- Irradiance calibration display
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

from config import PHOTOMETER_CALIBRATION, INSTRUMENT_COLORS
from data_handlers import PhotometerHandler


class PhotometryTab(ttk.Frame):
    """
    Photometer data visualization and ratio analysis tab.

    Provides:
    - Multi-channel waveform plotting (337nm, 391nm, 777nm)
    - Channel ratio calculations and plots
    - Time alignment controls
    - Integration with FA and INTF data from Home tab
    """

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app

        # Data handler
        self.photometer = PhotometerHandler()


        # File path
        self.file_path_var = tk.StringVar()

        # Timing parameters
        self.event_time_var = tk.StringVar(value="HH:MM:SS")
        self.second_offset_var = tk.StringVar(value="0")

        # Time range for plotting
        self.time_start_var = tk.StringVar(value="")
        self.time_stop_var = tk.StringVar(value="")

        # Channel visibility
        self.show_337_var = tk.BooleanVar(value=True)
        self.show_391_var = tk.BooleanVar(value=True)
        self.show_777_var = tk.BooleanVar(value=True)

        # Plot options
        self.show_raw_var = tk.BooleanVar(value=False)
        self.downsample_var = tk.StringVar(value="100")
        self.show_ratios_var = tk.BooleanVar(value=False)
        self.plot_title_var = tk.StringVar(value="")

        # Y-axis limits
        self.y_min_var = tk.StringVar(value="")
        self.y_max_var = tk.StringVar(value="")

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
        canvas = tk.Canvas(self.control_frame, width=330)
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
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add sections
        self._build_file_section()
        self._build_timing_section()
        self._build_channel_section()
        self._build_plot_options_section()
        self._build_ratio_section()
        self._build_action_buttons()
        self._build_info_section()

    def _build_file_section(self):
        """Build file input section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Photometer Data File", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # File path
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Entry(row, textvariable=self.file_path_var, width=28).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="...", width=3, command=self._browse_file).pack(side=tk.LEFT)

        # Expected format info
        ttk.Label(frame, text="Format: YYYYMMDD-HHMMSS.TTTT_PXI_P19.dat",
                  font=('Helvetica', 8), foreground='gray').pack(anchor='w')

        # Load button
        ttk.Button(frame, text="Load Photometer Data", command=self._load_data).pack(pady=5)

    def _build_timing_section(self):
        """Build timing parameters section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Timing", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Event time (for display)
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Event Time:", width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.event_time_var, width=10).pack(side=tk.LEFT)
        ttk.Label(row, text="(HH:MM:SS)", foreground='gray').pack(side=tk.LEFT, padx=3)

        # Second offset
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Second Offset:", width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.second_offset_var, width=6).pack(side=tk.LEFT)
        ttk.Label(row, text="sec", foreground='gray').pack(side=tk.LEFT, padx=3)

        # Help text
        ttk.Label(frame,
                  text="Offset = (photometer second) - (event second)\nPositive if photometer triggered earlier",
                  font=('Helvetica', 8), foreground='gray').pack(anchor='w')

        # Apply offset button
        ttk.Button(frame, text="Apply Second Offset", command=self._apply_offset).pack(pady=2)

    def _build_channel_section(self):
        """Build channel visibility section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Channels", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Channel checkboxes with colors
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)

        ttk.Checkbutton(row, text="337nm (Blue)", variable=self.show_337_var).pack(side=tk.LEFT)
        ttk.Checkbutton(row, text="391nm (Purple)", variable=self.show_391_var).pack(side=tk.LEFT)
        ttk.Checkbutton(row, text="777nm (Red)", variable=self.show_777_var).pack(side=tk.LEFT)

        # Raw vs calibrated
        ttk.Checkbutton(frame, text="Show raw voltage (uncalibrated)",
                        variable=self.show_raw_var).pack(anchor='w', pady=2)

    def _build_plot_options_section(self):
        """Build plot options section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Plot Options", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Plot title
        self.plot_title_var = tk.StringVar(value="")

        # Time range
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Time Range (µs):", width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.time_start_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.time_stop_var, width=10).pack(side=tk.LEFT, padx=2)

        # Downsample
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Downsample:", width=14).pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.downsample_var,
                     values=["1", "10", "50", "100", "200", "500", "1000"],
                     width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="(1=full resolution)", foreground='gray').pack(side=tk.LEFT, padx=3)

        # Y-axis limits
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Y Limits:", width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.y_min_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="-").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.y_max_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="(blank=auto)", foreground='gray').pack(side=tk.LEFT, padx=3)


    def _build_ratio_section(self):
        """Build ratio analysis section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Ratio Analysis", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Show ratios checkbox
        ttk.Checkbutton(frame, text="Show ratio plot below waveforms",
                        variable=self.show_ratios_var).pack(anchor='w', pady=2)

        # Calculated ratios display
        ttk.Label(frame, text="Peak Ratios in Time Range:").pack(anchor='w', pady=2)

        self.ratio_337_777_var = tk.StringVar(value="--")
        self.ratio_391_777_var = tk.StringVar(value="--")
        self.ratio_337_391_var = tk.StringVar(value="--")

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="337/777:", width=10).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.ratio_337_777_var, width=10).pack(side=tk.LEFT)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="391/777:", width=10).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.ratio_391_777_var, width=10).pack(side=tk.LEFT)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="337/391:", width=10).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.ratio_337_391_var, width=10).pack(side=tk.LEFT)

        # Calculate button
        ttk.Button(frame, text="Calculate Ratios", command=self._calculate_ratios).pack(pady=5)

    def _build_action_buttons(self):
        """Build action buttons."""
        frame = ttk.Frame(self.scrollable_frame, padding=5)
        frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(frame, text="🔄 Update Plot",
                   command=self._update_plot).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="📷 Export PNG",
                   command=self._export_png).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="💾 Export Data (CSV)",
                   command=self._export_csv).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="💾 Save to Project",
                   command=self._save_and_prompt).pack(fill=tk.X, pady=2)

    def _save_and_prompt(self):
        """Save settings to project state and prompt to save file."""
        self._save_to_project()
        # Trigger the main app's save functionality
        if hasattr(self.main_app, 'save_project'):
            self.main_app.save_project()

    def _build_info_section(self):
        """Build data info display section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Data Info", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        self.info_text = tk.Text(frame, height=8, width=35, wrap=tk.WORD,
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
        """Browse for photometer data file."""
        filepath = filedialog.askopenfilename(
            title="Select Photometer Data File",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if filepath:
            self.file_path_var.set(filepath)

    def _load_data(self):
        """Load photometer data from file."""
        print(
            f"DEBUG _load_data: BEFORE - time_start={self.time_start_var.get()}, time_stop={self.time_stop_var.get()}")
        """Load photometer data from file."""
        filepath = self.file_path_var.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Error", "Please select a valid photometer file")
            return

        try:
            success = self.photometer.load_binary_data(filepath)

            if success:
                self._update_info()
                # Only auto-set time range if not already set (e.g., from project load)
                # if not self.time_start_var.get() or not self.time_stop_var.get():
                #     self._auto_set_time_range()
                self._update_plot()
                self.main_app.status_var.set(f"Loaded photometer: {os.path.basename(filepath)}")
            else:
                messagebox.showerror("Error", "Failed to load photometer data")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading photometer data:\n{e}")

    def _apply_offset(self):
        """Apply second offset to photometer data."""
        try:
            offset = int(self.second_offset_var.get())
            self.photometer.set_second_offset(offset)
            self._update_info()
            self._update_plot()
            self.main_app.status_var.set(f"Applied second offset: {offset}")
        except ValueError:
            messagebox.showerror("Error", "Second offset must be an integer")

    def _get_time_range(self):
        """Get current time range settings."""
        try:
            t_start = float(self.time_start_var.get())
        except:
            t_start = None

        try:
            t_stop = float(self.time_stop_var.get())
        except:
            t_stop = None

        return t_start, t_stop

    def _calculate_ratios(self):
        """Calculate and display channel ratios."""
        if not self.photometer.is_loaded:
            messagebox.showwarning("Warning", "No data loaded")
            return

        t_start, t_stop = self._get_time_range()
        if t_start is None or t_stop is None:
            messagebox.showwarning("Warning", "Please set time range")
            return

        ratios = self.photometer.get_ratios_in_range(t_start, t_stop)

        if 'ratio_337_777' in ratios:
            self.ratio_337_777_var.set(f"{ratios['ratio_337_777']:.4f}")
        else:
            self.ratio_337_777_var.set("--")

        if 'ratio_391_777' in ratios:
            self.ratio_391_777_var.set(f"{ratios['ratio_391_777']:.4f}")
        else:
            self.ratio_391_777_var.set("--")

        if 'ratio_337_391' in ratios:
            self.ratio_337_391_var.set(f"{ratios['ratio_337_391']:.4f}")
        else:
            self.ratio_337_391_var.set("--")

        self.main_app.status_var.set("Ratios calculated")

    def _update_info(self):
        """Update data info display."""
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)

        if self.photometer.is_loaded:
            info = self.photometer.get_info()
            self.info_text.insert('1.0', info)
        else:
            self.info_text.insert('1.0', "No data loaded")

        self.info_text.config(state='disabled')

    def _update_plot(self):
        """Update the plot with current data and settings."""
        self.fig.clear()

        if not self.photometer.is_loaded:
            self.ax = self.fig.add_subplot(111)
            self.ax.text(0.5, 0.5, "No data loaded", ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        # Get time range
        t_start, t_stop = self._get_time_range()
        if t_start is None or t_stop is None:
            t_start, t_stop = self.photometer.get_time_range()

        # Get downsample factor
        try:
            downsample = int(self.downsample_var.get())
        except:
            downsample = 100

        # Get data
        data = self.photometer.get_data_in_event_time(t_start, t_stop, downsample)
        if data is None or len(data['time']) == 0:
            self.ax = self.fig.add_subplot(111)
            self.ax.text(0.5, 0.5, "No data in selected range", ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        # Determine subplot layout
        show_ratios = self.show_ratios_var.get()
        if show_ratios:
            self.ax = self.fig.add_subplot(211)
            self.ax_ratio = self.fig.add_subplot(212, sharex=self.ax)
        else:
            self.ax = self.fig.add_subplot(111)

        # Use raw or calibrated data
        use_raw = self.show_raw_var.get()

      # For raw data, we need to get the same time-masked indices
        if use_raw:
            # Get the mask for the time range (matching what get_data_in_event_time does)
            offset_us = self.photometer.second_offset * 1e6
            phot_t_start = t_start + offset_us
            phot_t_stop = t_stop + offset_us
            mask = (self.photometer.native_time_array >= phot_t_start) & \
                   (self.photometer.native_time_array <= phot_t_stop)

        # Plot channels
        if self.show_337_var.get() and data['ch0'] is not None:
            if use_raw:
                y_data = self.photometer.raw_channels[0][mask][::downsample]
            else:
                y_data = data['ch0']
            self.ax.plot(data['time'], y_data, color='blue', linewidth=0.5,
                         label='337nm', alpha=0.8)

        if self.show_391_var.get() and data['ch1'] is not None:
            if use_raw:
                y_data = self.photometer.raw_channels[1][mask][::downsample]
            else:
                y_data = data['ch1']
            self.ax.plot(data['time'], y_data, color='purple', linewidth=0.5,
                         label='391nm', alpha=0.8)

        if self.show_777_var.get() and data['ch2'] is not None:
            if use_raw:
                y_data = self.photometer.raw_channels[2][mask][::downsample]
            else:
                y_data = data['ch2']
            self.ax.plot(data['time'], y_data, color='red', linewidth=0.5,
                         label='777nm', alpha=0.8)

        # Labels and title
        event_time = self.event_time_var.get()
        if event_time and event_time != "HH:MM:SS":
            xlabel = f"Time relative to {event_time} (µs)"
        else:
            xlabel = "Time (µs)"

        self.ax.set_xlabel(xlabel if not show_ratios else "", fontsize=11)

        if use_raw:
            self.ax.set_ylabel("Voltage (V)", fontsize=11)
        else:
            self.ax.set_ylabel("Irradiance (W/m²/nm)", fontsize=11)

        title = self.plot_title_var.get() or "Photometer Waveforms"
        self.ax.set_title(title, fontsize=12)
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.grid(True, linestyle=':', alpha=0.5)

        # Set Y limits if specified
        try:
            y_min = float(self.y_min_var.get())
            y_max = float(self.y_max_var.get())
            self.ax.set_ylim(y_min, y_max)
        except:
            pass

        # Plot ratios if enabled
        if show_ratios:
            self._plot_ratios(data, self.ax_ratio, xlabel)



        self.fig.tight_layout()
        self.canvas.draw()

        self.main_app.status_var.set(f"Plotted {len(data['time']):,} points")

    def _plot_ratios(self, data, ax, xlabel):
        """Plot channel ratios on secondary axis."""
        time = data['time']

        # Calculate point-by-point ratios
        if data['ch0'] is not None and data['ch2'] is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_337_777 = data['ch0'] / data['ch2']
                # Replace inf/nan with nan for log scale compatibility
                ratio_337_777 = np.where(np.isfinite(ratio_337_777), ratio_337_777, np.nan)
            ax.plot(time, ratio_337_777, color='blue', linewidth=0.5,
                    label='337/777', alpha=0.8)

        if data['ch1'] is not None and data['ch2'] is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_391_777 = data['ch1'] / data['ch2']
                ratio_391_777 = np.where(np.isfinite(ratio_391_777), ratio_391_777, np.nan)
            ax.plot(time, ratio_391_777, color='purple', linewidth=0.5,
                    label='391/777', alpha=0.8)

        if data['ch0'] is not None and data['ch1'] is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_337_391 = data['ch0'] / data['ch1']
                ratio_337_391 = np.where(np.isfinite(ratio_337_391), ratio_337_391, np.nan)
            ax.plot(time, ratio_337_391, color='orange', linewidth=0.5,
                    label='337/391', alpha=0.8)

        # Use log scale like the 2023 script
        ax.set_yscale('log')

        # Bold reference line at y=1
        ax.axhline(y=1, color='black', linestyle='-', linewidth=2.5, zorder=3)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Irradiance Ratios", fontsize=11)
        ax.set_title("Channel Ratios", fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)
        # Remove fixed y-limits to allow auto-scaling with log

    def _on_mouse_move(self, event):
        """Handle mouse movement for coordinate display."""
        if event.inaxes:
            self.coord_var.set(f"x={event.xdata:.2f}, y={event.ydata:.4f}")
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

    def _export_csv(self):
        """Export photometer data to CSV."""
        if not self.photometer.is_loaded:
            messagebox.showwarning("Warning", "No data loaded")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="photometer_data.csv"
        )
        if filepath:
            try:
                t_start, t_stop = self._get_time_range()
                if t_start is None or t_stop is None:
                    t_start, t_stop = self.photometer.get_time_range()

                downsample = int(self.downsample_var.get())
                data = self.photometer.get_data_in_event_time(t_start, t_stop, downsample)

                with open(filepath, 'w') as f:
                    f.write("Time_us,337nm_Wm2nm,391nm_Wm2nm,777nm_Wm2nm\n")
                    for i in range(len(data['time'])):
                        ch0 = data['ch0'][i] if data['ch0'] is not None else 0
                        ch1 = data['ch1'][i] if data['ch1'] is not None else 0
                        ch2 = data['ch2'][i] if data['ch2'] is not None else 0
                        f.write(f"{data['time'][i]:.3f},{ch0:.6e},{ch1:.6e},{ch2:.6e}\n")

                self.main_app.status_var.set(f"Exported: {os.path.basename(filepath)}")
                messagebox.showinfo("Success", f"Data exported to:\n{filepath}")

            except Exception as e:
                messagebox.showerror("Error", f"Export error:\n{e}")

    def _save_to_project(self):
        """Save current settings to project state."""
        state = self.main_app.project_state

        # Save file path
        state.files['photometer'] = self.file_path_var.get() or None

        # Save timing
        event_time = self.event_time_var.get()
        if event_time and event_time != "HH:MM:SS":
            state.photometer['event_time'] = event_time

        try:
            state.photometer['second_offset'] = int(self.second_offset_var.get())
        except:
            pass

        # Save time range (always save to allow clearing)
        time_start_str = self.time_start_var.get()
        time_stop_str = self.time_stop_var.get()

        state.photometer['time_start'] = float(time_start_str) if time_start_str else None
        state.photometer['time_stop'] = float(time_stop_str) if time_stop_str else None

        # Save display options
        state.photometer['show_337'] = self.show_337_var.get()
        state.photometer['show_391'] = self.show_391_var.get()
        state.photometer['show_777'] = self.show_777_var.get()
        state.photometer['show_raw'] = self.show_raw_var.get()
        state.photometer['show_ratios'] = self.show_ratios_var.get()

        self.main_app.mark_unsaved()
        self.main_app.status_var.set("Photometry settings saved to project")

    def load_from_project(self):
        """Load settings from project state."""
        state = self.main_app.project_state

        # Load file path
        if state.files.get('photometer'):
            self.file_path_var.set(state.files['photometer'])

        # Load timing
        if state.photometer.get('event_time'):
            self.event_time_var.set(state.photometer['event_time'])

        if state.photometer.get('second_offset') is not None:
            self.second_offset_var.set(str(state.photometer['second_offset']))

        # Load time range
        if state.photometer.get('time_start') is not None:
            self.time_start_var.set(str(state.photometer['time_start']))
        else:
            self.time_start_var.set("")

        if state.photometer.get('time_stop') is not None:
            self.time_stop_var.set(str(state.photometer['time_stop']))
        else:
            self.time_stop_var.set("")

        # Load display options
        if 'show_337' in state.photometer:
            self.show_337_var.set(state.photometer['show_337'])
        if 'show_391' in state.photometer:
            self.show_391_var.set(state.photometer['show_391'])
        if 'show_777' in state.photometer:
            self.show_777_var.set(state.photometer['show_777'])
        if 'show_raw' in state.photometer:
            self.show_raw_var.set(state.photometer['show_raw'])
        if 'show_ratios' in state.photometer:
            self.show_ratios_var.set(state.photometer['show_ratios'])
