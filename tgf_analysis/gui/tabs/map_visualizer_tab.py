import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Matplotlib with Tk backend
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import geopy.distance


class MapVisualizerTab(ttk.Frame):
    """
    Map visualization tab for NLDN lightning strike data.

    Displays strikes on TASD detector array map with interactive
    selection and filtering capabilities.
    """

    # Camera/INTF location (fixed reference point)
    CAMERA_LAT = 39.339082
    CAMERA_LON = -112.700696

    # Boundary expansion beyond detector array (km)
    BOUNDARY_BUFFER_KM = 2.0

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app

        # Configure custom treeview style with proper row height
        style = ttk.Style()
        style.configure('NLDN.Treeview', rowheight=50)
        style.configure('NLDN.Treeview.Heading', font=('Helvetica', 9, 'bold'))


        # Data storage
        self.nldn_data = None  # Full loaded DataFrame
        self.filtered_data = None  # Time-filtered DataFrame
        self.detector_coords = None  # TASD detector coordinates
        self.boundary_box = None  # (min_lat, max_lat, min_lon, max_lon)

        # Selection state
        self.selected_index = None
        self.show_all_points = True  # Toggle for showing all vs selected only

        # UI variables
        self.file_path_var = tk.StringVar()
        self.time_start_var = tk.StringVar(value="")
        self.time_stop_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="No data loaded")

        # Load detector coordinates
        self._load_detector_coords()

        # Build UI
        self._build_ui()

        # Draw initial map with detectors
        self.after(100, self._draw_base_map)

    def _load_detector_coords(self):
        """Load TASD detector coordinates from file."""
        try:
            # Build comprehensive list of possible file locations
            possible_paths = []

            # Current working directory
            possible_paths.append('tasd_gpscoors.txt')
            possible_paths.append(os.path.join(os.getcwd(), 'tasd_gpscoors.txt'))

            # Relative to this script file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths.append(os.path.join(script_dir, 'tasd_gpscoors.txt'))
            possible_paths.append(os.path.join(script_dir, '..', 'tasd_gpscoors.txt'))
            possible_paths.append(os.path.join(script_dir, '..', '..', 'tasd_gpscoors.txt'))
            possible_paths.append(os.path.join(script_dir, '..', '..', '..', 'tasd_gpscoors.txt'))

            # If in gui/tabs/ structure, go up to main directory
            if 'gui' in script_dir:
                gui_parent = script_dir.split('gui')[0]
                possible_paths.append(os.path.join(gui_parent, 'tasd_gpscoors.txt'))

            # Try relative to main app directory if available
            if hasattr(self, 'main_app'):
                if hasattr(self.main_app, 'project_filepath') and self.main_app.project_filepath:
                    project_dir = os.path.dirname(self.main_app.project_filepath)
                    possible_paths.append(os.path.join(project_dir, 'tasd_gpscoors.txt'))

            # Search for the file
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    self.detector_coords = np.loadtxt(abs_path)
                    print(f"✓ Loaded {len(self.detector_coords)} TASD detector coordinates from:\n  {abs_path}")
                    self._calculate_boundary()
                    return

            # File not found - print helpful debug info
            print("=" * 60)
            print("WARNING: tasd_gpscoors.txt not found!")
            print("Searched these locations:")
            for p in possible_paths:
                print(f"  ✗ {os.path.abspath(p)}")
            print("\nPlease ensure tasd_gpscoors.txt is in your working directory")
            print("or the same folder as main.py")
            print("=" * 60)
            self.detector_coords = None

        except Exception as e:
            print(f"Error loading detector coordinates: {e}")
            import traceback
            traceback.print_exc()
            self.detector_coords = None

    def _calculate_boundary(self):
        """Calculate the boundary box for filtering points."""
        if self.detector_coords is None:
            return

        # Get lat/lon bounds from detector array
        lats = self.detector_coords[:, 1]
        lons = self.detector_coords[:, 2]

        min_lat, max_lat = np.min(lats), np.max(lats)
        min_lon, max_lon = np.min(lons), np.max(lons)

        # Convert buffer from km to approximate degrees
        # At ~39° latitude: 1° lat ≈ 111 km, 1° lon ≈ 86 km
        lat_buffer = self.BOUNDARY_BUFFER_KM / 111.0
        lon_buffer = self.BOUNDARY_BUFFER_KM / 86.0

        self.boundary_box = (
            min_lat - lat_buffer,
            max_lat + lat_buffer,
            min_lon - lon_buffer,
            max_lon + lon_buffer
        )
        print(f"  Boundary box: lat [{self.boundary_box[0]:.4f}, {self.boundary_box[1]:.4f}], "
              f"lon [{self.boundary_box[2]:.4f}, {self.boundary_box[3]:.4f}]")

    def _build_ui(self):
        """Build the tab UI."""
        # Main horizontal paned window
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - Controls and Table (WIDER - 500 pixels)
        self.control_frame = ttk.Frame(self.paned, width=1000)
        self.control_frame.pack_propagate(False)
        self.paned.add(self.control_frame, weight=0)

        # Right panel - Map
        self.map_frame = ttk.Frame(self.paned)
        self.paned.add(self.map_frame, weight=1)

        # Build sections
        self._build_control_panel()
        self._build_map_area()

    def _build_control_panel(self):
        """Build the left control panel."""
        # File section
        self._build_file_section()

        # Time range section
        self._build_time_section()

        # Action buttons
        self._build_action_buttons()

        # Data table
        self._build_data_table()

        # Selection info and buttons
        self._build_selection_section()

        # Status
        status_frame = ttk.Frame(self.control_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(status_frame, textvariable=self.status_var,
                  font=('Helvetica', 9), foreground='gray').pack(anchor='w')

    def _build_file_section(self):
        """Build file input section."""
        frame = ttk.LabelFrame(self.control_frame, text="NLDN Data File", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # File path row
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Entry(row, textvariable=self.file_path_var, width=50).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(row, text="...", width=3, command=self._browse_file).pack(side=tk.LEFT)

        # Format info
        ttk.Label(frame, text="Format: CSV with time, longitude, latitude, signalStrengthKA, cloud",
                  font=('Helvetica', 8), foreground='gray').pack(anchor='w')

        # Load button
        ttk.Button(frame, text="Load NLDN Data", command=self._load_data).pack(pady=5)

    def _build_time_section(self):
        """Build time range filter section."""
        frame = ttk.LabelFrame(self.control_frame, text="Time Range Filter", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Start time
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Start Time:", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.time_start_var, width=20).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="HH:MM:SS.xxxxxx", foreground='gray',
                  font=('Helvetica', 8)).pack(side=tk.LEFT)

        # Stop time
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Stop Time:", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.time_stop_var, width=20).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="HH:MM:SS.xxxxxx", foreground='gray',
                  font=('Helvetica', 8)).pack(side=tk.LEFT)

        # Filter info
        ttk.Label(frame, text="Leave blank to show all data",
                  font=('Helvetica', 8), foreground='gray').pack(anchor='w')

    def _build_action_buttons(self):
        """Build main action buttons."""
        frame = ttk.Frame(self.control_frame, padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(frame, text="🗺️ Load Map",
                   command=self._update_map).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="📷 Export PNG",
                   command=self._export_png).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="💾 Save to Project",
                   command=self._save_and_prompt).pack(fill=tk.X, pady=2)

    def _build_data_table(self):
        """Build the interactive data table."""
        frame = ttk.LabelFrame(self.control_frame, text="Strike Data", padding=5)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        y_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        x_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)

        # Treeview
        columns = ('time', 'current', 'distance', 'type', 'lat', 'lon')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings',
                                 yscrollcommand=y_scroll.set,
                                 xscrollcommand=x_scroll.set,
                                 height=8,
                                 style='NLDN.Treeview')


        # Configure scrollbars
        y_scroll.config(command=self.tree.yview)
        x_scroll.config(command=self.tree.xview)

        # Column headings and widths - WIDER for readability
        self.tree.heading('time', text='Time', command=lambda: self._sort_table('time'))
        self.tree.heading('current', text='kA', command=lambda: self._sort_table('current'))
        self.tree.heading('distance', text='Dist(km)', command=lambda: self._sort_table('distance'))
        self.tree.heading('type', text='Type', command=lambda: self._sort_table('type'))
        self.tree.heading('lat', text='Latitude', command=lambda: self._sort_table('lat'))
        self.tree.heading('lon', text='Longitude', command=lambda: self._sort_table('lon'))

        self.tree.column('time', width=120, minwidth=100, anchor='w')
        self.tree.column('current', width=50, minwidth=45, anchor='center')
        self.tree.column('distance', width=60, minwidth=50, anchor='center')
        self.tree.column('type', width=55, minwidth=50, anchor='center')
        self.tree.column('lat', width=80, minwidth=70, anchor='e')
        self.tree.column('lon', width=85, minwidth=75, anchor='e')

        # Pack
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self._on_row_select)

        # Sort state
        self.sort_column = None
        self.sort_reverse = False

        # Configure row height using style
        style = ttk.Style()
        style.configure('Treeview', rowheight=25)


    def _build_selection_section(self):
        """Build selection info and action buttons."""
        frame = ttk.LabelFrame(self.control_frame, text="Selected Strike", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Selection info display
        self.selection_info_var = tk.StringVar(value="No strike selected")
        ttk.Label(frame, textvariable=self.selection_info_var,
                  font=('Courier', 9), wraplength=450).pack(anchor='w', pady=2)

        # Button row
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=5)

        ttk.Button(btn_row, text="Plot Selected Only",
                   command=self._plot_selected_only).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Show All Points",
                   command=self._show_all_points).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="📋 Copy Info",
                   command=self._copy_to_clipboard).pack(side=tk.LEFT, padx=2)

    def _build_map_area(self):
        """Build the matplotlib map area."""
        # Create figure
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_frame)
        self.canvas.draw()

        # Add toolbar
        toolbar_frame = ttk.Frame(self.map_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Add canvas widget
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Draw initial empty map
        self._draw_base_map()

    # =========================================================================
    # Data Loading and Processing
    # =========================================================================

    def _browse_file(self):
        """Browse for NLDN CSV file."""
        filepath = filedialog.askopenfilename(
            title="Select NLDN Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.file_path_var.set(filepath)

    def _load_data(self):
        """Load NLDN data from CSV file."""
        filepath = self.file_path_var.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Error", "Please select a valid NLDN CSV file")
            return

        try:
            # Load CSV
            self.nldn_data = pd.read_csv(filepath, sep=None, engine='python')

            # Standardize column names (handle variations)
            col_mapping = {}
            for col in self.nldn_data.columns:
                col_lower = col.lower().strip()
                if 'time' in col_lower:
                    col_mapping[col] = 'time'
                elif 'lon' in col_lower:
                    col_mapping[col] = 'longitude'
                elif 'lat' in col_lower:
                    col_mapping[col] = 'latitude'
                elif 'signal' in col_lower or 'current' in col_lower or 'ka' in col_lower:
                    col_mapping[col] = 'signalStrengthKA'
                elif 'cloud' in col_lower:
                    col_mapping[col] = 'cloud'
                elif 'distance' in col_lower:
                    col_mapping[col] = 'distanceM'

            self.nldn_data.rename(columns=col_mapping, inplace=True)

            # Parse time column
            self._parse_times()

            # Calculate distances from camera
            self._calculate_distances()

            # Apply boundary filter
            self._apply_boundary_filter()

            # Update status
            n_total = len(self.nldn_data)
            self.status_var.set(f"Loaded {n_total} strikes from {os.path.basename(filepath)}")
            self.main_app.status_var.set(f"Loaded NLDN data: {n_total} strikes")

            # Initialize filtered data as full dataset
            self.filtered_data = self.nldn_data.copy()

            # Populate table
            self._populate_table()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load NLDN data:\n{e}")
            import traceback
            traceback.print_exc()

    def _parse_times(self):
        """Parse time column and extract time-of-day for filtering."""
        if 'time' not in self.nldn_data.columns:
            return

        # Parse ISO format times
        self.nldn_data['datetime'] = pd.to_datetime(self.nldn_data['time'])

        # Extract time of day as string (HH:MM:SS.ffffff)
        def format_time(dt):
            return dt.strftime('%H:%M:%S.%f')

        self.nldn_data['time_str'] = self.nldn_data['datetime'].apply(format_time)

        # Extract time as total seconds for sorting/filtering
        def time_to_seconds(dt):
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

        self.nldn_data['time_seconds'] = self.nldn_data['datetime'].apply(time_to_seconds)

    def _calculate_distances(self):
        """Calculate distance from camera for each strike."""
        if self.nldn_data is None:
            return

        camera_point = (self.CAMERA_LAT, self.CAMERA_LON)

        distances = []
        for _, row in self.nldn_data.iterrows():
            try:
                strike_point = (row['latitude'], row['longitude'])
                dist = geopy.distance.distance(camera_point, strike_point).km
                distances.append(round(dist, 2))
            except:
                distances.append(np.nan)

        self.nldn_data['distance_km'] = distances

    def _apply_boundary_filter(self):
        """Filter out strikes far outside the detector boundary + buffer."""
        if self.nldn_data is None or self.boundary_box is None:
            return

        min_lat, max_lat, min_lon, max_lon = self.boundary_box

        # Use a more generous boundary (10km buffer instead of 2km for filtering)
        # This keeps strikes visible even if slightly outside the array
        extra_buffer_lat = 10.0 / 111.0  # ~10km in latitude degrees
        extra_buffer_lon = 10.0 / 86.0   # ~10km in longitude degrees

        mask = (
            (self.nldn_data['latitude'] >= min_lat - extra_buffer_lat) &
            (self.nldn_data['latitude'] <= max_lat + extra_buffer_lat) &
            (self.nldn_data['longitude'] >= min_lon - extra_buffer_lon) &
            (self.nldn_data['longitude'] <= max_lon + extra_buffer_lon)
        )

        n_before = len(self.nldn_data)
        self.nldn_data = self.nldn_data[mask].reset_index(drop=True)
        n_after = len(self.nldn_data)

        if n_before != n_after:
            print(f"Filtered {n_before - n_after} strikes outside extended boundary (kept {n_after})")

    def _parse_time_string(self, time_str):
        """Parse time string (HH:MM:SS.xxxxxx) to seconds."""
        if not time_str or not time_str.strip():
            return None

        try:
            parts = time_str.strip().split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except:
            return None

    def _filter_by_time(self):
        """Filter data by time range."""
        if self.nldn_data is None:
            return

        t_start = self._parse_time_string(self.time_start_var.get())
        t_stop = self._parse_time_string(self.time_stop_var.get())

        print(f"DEBUG: t_start={t_start}, t_stop={t_stop}")
        print(f"DEBUG: nldn_data has {len(self.nldn_data)} rows")

        if t_start is None and t_stop is None:
            # No filter - use all data
            self.filtered_data = self.nldn_data.copy()
        else:
            mask = pd.Series([True] * len(self.nldn_data))

            if t_start is not None:
                mask &= self.nldn_data['time_seconds'] >= t_start
            if t_stop is not None:
                mask &= self.nldn_data['time_seconds'] <= t_stop

            self.filtered_data = self.nldn_data[mask].reset_index(drop=True)
            print(f"DEBUG: After filter, {len(self.filtered_data)} rows remain")

        # Update table
        self._populate_table()

        # Update status
        self.status_var.set(f"Showing {len(self.filtered_data)} strikes in time range")

    # =========================================================================
    # Table Management
    # =========================================================================

    def _populate_table(self):
        """Populate the treeview table with filtered data."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        if self.filtered_data is None or len(self.filtered_data) == 0:
            return

        # Add rows - use enumerate for sequential iids (0, 1, 2...) that match iloc
        for idx, (_, row) in enumerate(self.filtered_data.iterrows()):
            # Format time as HH:MM:SS.fff (truncate microseconds for display)
            time_str = row.get('time_str', '')
            if len(time_str) > 12:
                time_str = time_str[:12]  # Show HH:MM:SS.fff

            current = f"{row.get('signalStrengthKA', 0):.1f}"
            distance = f"{row.get('distance_km', 0):.1f}"

            # Determine type (cloud or ground)
            cloud_val = row.get('cloud', False)
            if isinstance(cloud_val, str):
                strike_type = 'Cloud' if cloud_val.upper() == 'TRUE' else 'Ground'
            else:
                strike_type = 'Cloud' if cloud_val else 'Ground'

            lat = f"{row.get('latitude', 0):.4f}"
            lon = f"{row.get('longitude', 0):.4f}"

            self.tree.insert('', tk.END, iid=str(idx),
                             values=(time_str, current, distance, strike_type, lat, lon))

    def _sort_table(self, column):
        """Sort table by column."""
        if self.filtered_data is None:
            return

        # Toggle sort direction if same column
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False

        # Map column to dataframe column
        col_map = {
            'time': 'time_seconds',
            'current': 'signalStrengthKA',
            'distance': 'distance_km',
            'type': 'cloud',
            'lat': 'latitude',
            'lon': 'longitude'
        }

        df_col = col_map.get(column, column)

        if df_col in self.filtered_data.columns:
            self.filtered_data = self.filtered_data.sort_values(
                by=df_col, ascending=not self.sort_reverse
            ).reset_index(drop=True)
            self._populate_table()

    def _on_row_select(self, event):
        """Handle row selection in table."""
        selection = self.tree.selection()

        # Ignore empty selections (don't clear existing selection)
        if not selection:
            return

        # Get selected row index
        item_id = selection[0]
        self.selected_index = int(item_id)

        # Get row data
        if self.filtered_data is not None and self.selected_index < len(self.filtered_data):
            row = self.filtered_data.iloc[self.selected_index]

            # Format info string
            time_str = row.get('time_str', 'N/A')
            current = row.get('signalStrengthKA', 0)
            distance = row.get('distance_km', 0)
            lat = row.get('latitude', 0)
            lon = row.get('longitude', 0)

            cloud_val = row.get('cloud', False)
            if isinstance(cloud_val, str):
                strike_type = 'Cloud' if cloud_val.upper() == 'TRUE' else 'Ground'
            else:
                strike_type = 'Cloud' if cloud_val else 'Ground'

            info = (f"Time: {time_str}\n"
                    f"Current: {current:.1f} kA\n"
                    f"Distance: {distance:.2f} km\n"
                    f"Type: {strike_type}\n"
                    f"Lat: {lat:.6f}, Lon: {lon:.6f}")

            self.selection_info_var.set(info)

    # =========================================================================
    # Map Drawing
    # =========================================================================

    def _draw_base_map(self):
        """Draw the base map with detector positions."""
        self.ax.clear()

        # Plot detector positions
        if self.detector_coords is not None and len(self.detector_coords) > 0:
            lons = self.detector_coords[:, 2]
            lats = self.detector_coords[:, 1]
            self.ax.scatter(lons, lats, s=8, color='black', label='TASD Detectors', zorder=1)

            # Set axis limits based on detector array (with some padding)
            lon_pad = 0.02
            lat_pad = 0.02
            self.ax.set_xlim(np.min(lons) - lon_pad, np.max(lons) + lon_pad)
            self.ax.set_ylim(np.min(lats) - lat_pad, np.max(lats) + lat_pad)
        else:
            # Default view centered on camera if no detectors loaded
            self.ax.set_xlim(-113.1, -112.65)
            self.ax.set_ylim(39.1, 39.45)

        # Plot camera position
        self.ax.plot(self.CAMERA_LON, self.CAMERA_LAT, 's', markersize=12,
                    color='blue', label='Camera', zorder=5)

        # Labels
        self.ax.set_xlabel('Longitude', fontsize=11)
        self.ax.set_ylabel('Latitude', fontsize=11)
        self.ax.set_title('NLDN Lightning Strikes on TASD', fontsize=12)
        self.ax.grid(True, linestyle=':', alpha=0.5)
        self.ax.legend(loc='upper right', fontsize=9)

        # Make aspect ratio roughly equal (account for latitude)
        self.ax.set_aspect('equal', adjustable='box')

        self.fig.tight_layout()
        self.canvas.draw()

    def _update_map(self):
        """Update map with filtered strike data."""
        # First filter by time
        self._filter_by_time()

        # Reset to show all points
        self.show_all_points = True
        self.selected_index = None

        # Clear and redraw
        self.ax.clear()

        # Plot detector positions first (background)
        if self.detector_coords is not None and len(self.detector_coords) > 0:
            lons = self.detector_coords[:, 2]
            lats = self.detector_coords[:, 1]
            self.ax.scatter(lons, lats, s=8, color='black', label='TASD', zorder=1)

            # Set axis limits based on detector array
            lon_pad = 0.02
            lat_pad = 0.02
            self.ax.set_xlim(np.min(lons) - lon_pad, np.max(lons) + lon_pad)
            self.ax.set_ylim(np.min(lats) - lat_pad, np.max(lats) + lat_pad)
        else:
            self.ax.set_xlim(-113.1, -112.65)
            self.ax.set_ylim(39.1, 39.45)

        # Plot camera position
        self.ax.plot(self.CAMERA_LON, self.CAMERA_LAT, 's', markersize=12,
                    color='blue', label='Camera', zorder=5)

        if self.filtered_data is None or len(self.filtered_data) == 0:
            self.ax.set_title('No strikes in selected time range', fontsize=12)
            self.ax.set_xlabel('Longitude', fontsize=11)
            self.ax.set_ylabel('Latitude', fontsize=11)
            self.ax.grid(True, linestyle=':', alpha=0.5)
            self.ax.legend(loc='upper right', fontsize=9)
            self.ax.set_aspect('equal', adjustable='box')
            self.fig.tight_layout()
            self.canvas.draw()
            return

        # Separate cloud and ground strikes
        cloud_mask = self.filtered_data['cloud'].apply(
            lambda x: str(x).upper() == 'TRUE' if isinstance(x, str) else bool(x)
        )

        cloud_strikes = self.filtered_data[cloud_mask]
        ground_strikes = self.filtered_data[~cloud_mask]

        # Plot cloud strikes (green triangles pointing up)
        if len(cloud_strikes) > 0:
            self.ax.scatter(cloud_strikes['longitude'], cloud_strikes['latitude'],
                          s=80, color='green', marker='^', label=f'Cloud ({len(cloud_strikes)})',
                          alpha=0.8, edgecolors='darkgreen', linewidths=1, zorder=3)

        # Plot ground strikes (red triangles pointing down)
        if len(ground_strikes) > 0:
            self.ax.scatter(ground_strikes['longitude'], ground_strikes['latitude'],
                          s=80, color='red', marker='v', label=f'Ground ({len(ground_strikes)})',
                          alpha=0.8, edgecolors='darkred', linewidths=1, zorder=3)

        # Update title with time range
        t_start = self.time_start_var.get() or "start"
        t_stop = self.time_stop_var.get() or "end"
        self.ax.set_title(f'NLDN Strikes: {t_start} - {t_stop}\n({len(self.filtered_data)} total)',
                         fontsize=12)

        self.ax.set_xlabel('Longitude', fontsize=11)
        self.ax.set_ylabel('Latitude', fontsize=11)
        self.ax.grid(True, linestyle=':', alpha=0.5)
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.set_aspect('equal', adjustable='box')
        self.fig.tight_layout()
        self.canvas.draw()

        self.status_var.set(f"Plotted {len(self.filtered_data)} strikes")

    def _highlight_selected(self):
        """Highlight the selected strike on the map."""
        if self.selected_index is None or self.filtered_data is None:
            return

        if self.selected_index >= len(self.filtered_data):
            return

        # Save the index before _update_map resets it
        saved_index = self.selected_index

        # Redraw map with all points
        self._update_map()

        # Restore the index
        self.selected_index = saved_index

        # Highlight selected point with a larger yellow star
        row = self.filtered_data.iloc[self.selected_index]
        cloud_val = row.get('cloud', False)
        if isinstance(cloud_val, str):
            is_cloud = cloud_val.upper() == 'TRUE'
        else:
            is_cloud = bool(cloud_val)

        color = 'green' if is_cloud else 'red'
        marker = '^' if is_cloud else 'v'

        # Plot larger version of the same marker
        self.ax.scatter(row['longitude'], row['latitude'],
                        s=250, color=color, marker=marker,
                        edgecolors='yellow', linewidths=3, zorder=10,
                        label='Selected')

        self.ax.legend(loc='upper right', fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_selected_only(self):
        """Plot only the selected strike."""
        if self.selected_index is None or self.filtered_data is None:
            messagebox.showwarning("Warning", "No strike selected")
            return

        if self.selected_index >= len(self.filtered_data):
            return

        self.show_all_points = False
        row = self.filtered_data.iloc[self.selected_index]

        # Clear and redraw base elements
        self.ax.clear()

        # Plot detector positions
        if self.detector_coords is not None and len(self.detector_coords) > 0:
            lons = self.detector_coords[:, 2]
            lats = self.detector_coords[:, 1]
            self.ax.scatter(lons, lats, s=8, color='black', label='TASD', zorder=1)

            # Set axis limits
            lon_pad = 0.02
            lat_pad = 0.02
            self.ax.set_xlim(np.min(lons) - lon_pad, np.max(lons) + lon_pad)
            self.ax.set_ylim(np.min(lats) - lat_pad, np.max(lats) + lat_pad)

        # Plot camera
        self.ax.plot(self.CAMERA_LON, self.CAMERA_LAT, 's', markersize=12,
                    color='blue', label='Camera', zorder=5)

        # Determine strike type
        cloud_val = row.get('cloud', False)
        if isinstance(cloud_val, str):
            is_cloud = cloud_val.upper() == 'TRUE'
        else:
            is_cloud = bool(cloud_val)

        # Plot single strike
        color = 'green' if is_cloud else 'red'
        marker = '^' if is_cloud else 'v'
        strike_type = 'Cloud' if is_cloud else 'Ground'

        self.ax.scatter(row['longitude'], row['latitude'],
                       s=80, color=color, marker=marker,

                       label=f'{strike_type} Strike')

        # Add info text box
        info_text = (f"Time: {row.get('time_str', 'N/A')[:15]}\n"
                    f"Current: {row.get('signalStrengthKA', 0):.1f} kA\n"
                    f"Distance: {row.get('distance_km', 0):.2f} km\n"
                    f"Lat: {row.get('latitude', 0):.4f}\n"
                    f"Lon: {row.get('longitude', 0):.4f}")

        # Position text in upper left
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    family='monospace')

        self.ax.set_title(f'Selected Strike - {strike_type}', fontsize=12)
        self.ax.set_xlabel('Longitude', fontsize=11)
        self.ax.set_ylabel('Latitude', fontsize=11)
        self.ax.grid(True, linestyle=':', alpha=0.5)
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.set_aspect('equal', adjustable='box')
        self.fig.tight_layout()
        self.canvas.draw()

        self.status_var.set(f"Showing selected strike only")

    def _show_all_points(self):
        """Show all points in the filtered data."""
        self.show_all_points = True
        self._update_map()

    # =========================================================================
    # Export Functions
    # =========================================================================

    def _copy_to_clipboard(self):
        """Copy selected strike info to clipboard."""
        if self.selected_index is None or self.filtered_data is None:
            messagebox.showwarning("Warning", "No strike selected")
            return

        if self.selected_index >= len(self.filtered_data):
            return

        row = self.filtered_data.iloc[self.selected_index]

        # Format for clipboard
        cloud_val = row.get('cloud', False)
        if isinstance(cloud_val, str):
            strike_type = 'Cloud' if cloud_val.upper() == 'TRUE' else 'Ground'
        else:
            strike_type = 'Cloud' if cloud_val else 'Ground'

        text = (f"Time: {row.get('time_str', 'N/A')}\n"
               f"Latitude: {row.get('latitude', 0):.6f}\n"
               f"Longitude: {row.get('longitude', 0):.6f}\n"
               f"Peak Current: {row.get('signalStrengthKA', 0):.1f} kA\n"
               f"Distance from Camera: {row.get('distance_km', 0):.2f} km\n"
               f"Type: {strike_type}")

        # Copy to clipboard
        self.clipboard_clear()
        self.clipboard_append(text)

        self.status_var.set("Strike info copied to clipboard")
        self.main_app.status_var.set("Strike info copied to clipboard")

    def _export_png(self):
        """Export map as PNG with metadata."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            initialfile="nldn_map.png"
        )

        if not filepath:
            return

        try:
            # Add metadata text box if we have data
            if self.filtered_data is not None and len(self.filtered_data) > 0:
                # Build metadata string
                t_start = self.time_start_var.get() or "N/A"
                t_stop = self.time_stop_var.get() or "N/A"

                cloud_count = self.filtered_data['cloud'].apply(
                    lambda x: str(x).upper() == 'TRUE' if isinstance(x, str) else bool(x)
                ).sum()
                ground_count = len(self.filtered_data) - cloud_count

                metadata = (f"Time Range: {t_start} - {t_stop}\n"
                           f"Total Strikes: {len(self.filtered_data)}\n"
                           f"Cloud: {cloud_count}, Ground: {ground_count}\n"
                           f"Camera: ({self.CAMERA_LAT:.4f}, {self.CAMERA_LON:.4f})")

                # Add text box in corner (won't affect data area)
                self.ax.text(0.02, 0.02, metadata, transform=self.ax.transAxes,
                            fontsize=8, verticalalignment='bottom',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                            family='monospace')

            # Save
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')

            # Redraw to remove metadata box from interactive view
            self._update_map()

            self.status_var.set(f"Exported: {os.path.basename(filepath)}")
            self.main_app.status_var.set(f"Map exported: {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    # =========================================================================
    # Project Save/Load
    # =========================================================================

    def _save_and_prompt(self):
        """Save settings and prompt to save project."""
        self._save_to_project()
        if hasattr(self.main_app, 'save_project'):
            self.main_app.save_project()

    def _save_to_project(self):
        """Save current settings to project state."""
        state = self.main_app.project_state

        # Ensure map_visualizer dict exists
        if not hasattr(state, 'map_visualizer'):
            state.map_visualizer = {}

        # Save file path
        state.files['nldn'] = self.file_path_var.get() or None

        # Save time range
        state.map_visualizer['time_start'] = self.time_start_var.get() or None
        state.map_visualizer['time_stop'] = self.time_stop_var.get() or None

        self.main_app.mark_unsaved()
        self.status_var.set("Settings saved to project")

    def load_from_project(self):
        """Load settings from project state."""
        state = self.main_app.project_state

        # Load file path
        if state.files.get('nldn'):
            self.file_path_var.set(state.files['nldn'])

        # Load time range
        if hasattr(state, 'map_visualizer'):
            if state.map_visualizer.get('time_start'):
                self.time_start_var.set(state.map_visualizer['time_start'])
            if state.map_visualizer.get('time_stop'):
                self.time_stop_var.set(state.map_visualizer['time_stop'])