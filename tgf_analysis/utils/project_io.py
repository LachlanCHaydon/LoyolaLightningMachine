"""
TGF Analysis Tool - Project I/O
===============================
Handles saving/loading project state to JSON files.
"""

import json
import os
from datetime import datetime


class ProjectState:
    """
    Container for all project settings and data paths.
    """

    def __init__(self):
        """Initialize with default values."""
        # Project metadata
        self.project_name = "Untitled"
        self.project_path = None
        self.created = datetime.now().isoformat()
        self.modified = datetime.now().isoformat()

        # Event information
        self.event_info = {
            'date': '',
            'time': '',
            'stroke_number': 1,
            'flash_number': 1,
            'description': '',
        }

        # File paths (stored as ABSOLUTE paths)
        self.files = {
            'fa': None,
            'intf_raw': None,
            'intf_calibrated': None,
            'sd_directory': None,
            'lma': None,
            'photometer': None,
            'luminosity': None,
            'hsv_directory': None,
            'spectra_directory': None,
        }

        # Timing parameters
        self.timing = {
            'T0': 0,
            'timeshift': 0,
            'photometer_second_offset': 0,
        }

        # Photometer settings
        self.photometer = {
            'event_time': None,
            'second_offset': 0,
            'time_start': None,
            'time_stop': None,
            'show_337': True,
            'show_391': True,
            'show_777': True,
            'show_raw': False,
            'show_ratios': False,
        }

        # INTF calibration settings
        self.intf = {
            'cos_shift_a': -0.0051,
            'cos_shift_b': -0.0178,
            'is_calibrated': False,
        }

        # Plot ranges
        self.plot_ranges = {
            'main': {'x_start': 0, 'x_stop': 100000},
            'zoom': {'x_start': 0, 'x_stop': 1000},
            'intf_elev': {'y_min': 0, 'y_max': 40},
            'intf_azi': {'y_min': 270, 'y_max': 330},
            'fa': {'y_min': -80, 'y_max': 30},
            'sd': {'y_min': 0, 'y_max': 500},
        }

        # Visibility settings
        self.visibility = {
            'fa': True,
            'intf': True,
            'sd': True,
            'lma': True,
            'luminosity': True,
            'photometer_337': True,
            'photometer_391': True,
            'photometer_777': True,
        }

        # Plot style settings
        self.plot_style = {
            'show_grid': True,
            'show_legend': True,
            'title': '',
        }

        # Spectroscopy settings
        self.spectroscopy = {
            'roi': None,
            'tilt_angle': 0.0,
            'flip_horizontal': False,
            'start_timestamp': 0,
            'frame_interval': 40,
            'keyframes': [],
            'frames_data': [],
            'analysis_complete': False,
            'current_frame_idx': 0,
            'source_directory': None,
            'peak_prominence': 0.15,
            'peak_distance': 10,
            'baseline_removal': True,
            'baseline_degree': 8,
            'poly_order': 4,
        }

        # Analysis results
        self.results = {
            'timeshift_analysis': None,
            'source_location': None,
            'spectroscopy_ratios': [],
        }

    def to_dict(self):
        """Convert state to dictionary for JSON serialization."""
        self.modified = datetime.now().isoformat()

        return {
            'project_name': self.project_name,
            'created': self.created,
            'modified': self.modified,
            'event_info': self.event_info,
            'files': self.files,
            'timing': self.timing,
            'photometer': self.photometer,
            'intf': self.intf,
            'plot_ranges': self.plot_ranges,
            'visibility': self.visibility,
            'plot_style': self.plot_style,
            'spectroscopy': self.spectroscopy,
            'results': self.results,
        }

    def from_dict(self, data):
        """Load state from dictionary."""
        self.project_name = data.get('project_name', 'Untitled')
        self.created = data.get('created', datetime.now().isoformat())
        self.modified = data.get('modified', datetime.now().isoformat())

        if 'event_info' in data:
            self.event_info.update(data['event_info'])

        if 'files' in data:
            self.files.update(data['files'])

        if 'timing' in data:
            self.timing.update(data['timing'])

        if 'photometer' in data:
            self.photometer.update(data['photometer'])

        if 'spectroscopy' in data:
            self.spectroscopy.update(data['spectroscopy'])

        if 'intf' in data:
            self.intf.update(data['intf'])

        if 'plot_ranges' in data:
            for key in data['plot_ranges']:
                if key in self.plot_ranges:
                    self.plot_ranges[key].update(data['plot_ranges'][key])
                else:
                    self.plot_ranges[key] = data['plot_ranges'][key]

        if 'visibility' in data:
            self.visibility.update(data['visibility'])

        if 'plot_style' in data:
            self.plot_style.update(data['plot_style'])

        if 'spectroscopy' in data:
            self.spectroscopy.update(data['spectroscopy'])

        if 'results' in data:
            self.results.update(data['results'])


def save_project(state, filepath):
    """
    Save project state to a JSON file.

    Args:
        state: ProjectState instance
        filepath: Path to save the .tgf file

    Returns:
        True if successful, raises exception otherwise
    """
    # Update state metadata
    state.project_path = filepath
    state.project_name = os.path.splitext(os.path.basename(filepath))[0]

    # Convert to dictionary
    data = state.to_dict()

    # DEBUG: Print what's being written
    print("\n" + "="*60)
    print("DEBUG: save_project() - About to write JSON")
    print("="*60)
    print(f"  filepath: {filepath}")
    if 'spectroscopy' in data:
        spec = data['spectroscopy']
        print(f"  spectroscopy.roi: {spec.get('roi')}")
        print(f"  spectroscopy.source_directory: {spec.get('source_directory')}")
        print(f"  spectroscopy.frames_data count: {len(spec.get('frames_data', []))}")
    else:
        print("  NO SPECTROSCOPY KEY IN DATA!")
    print("="*60 + "\n")

    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Update recent projects
    _add_to_recent_projects(filepath)

    return True


def load_project(filepath):
    """
    Load project state from a JSON file.

    Args:
        filepath: Path to the .tgf file

    Returns:
        ProjectState instance
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    state = ProjectState()
    state.from_dict(data)
    state.project_path = filepath

    _add_to_recent_projects(filepath)

    return state


def get_recent_projects(max_count=10):
    """
    Get list of recently opened projects.

    Returns:
        List of (name, path) tuples
    """
    config_dir = _get_config_dir()
    recent_file = os.path.join(config_dir, 'recent_projects.json')

    if not os.path.exists(recent_file):
        return []

    try:
        with open(recent_file, 'r', encoding='utf-8') as f:
            recent = json.load(f)

        # Filter out non-existent files
        valid = [(r['name'], r['path']) for r in recent
                 if os.path.exists(r['path'])]

        return valid[:max_count]
    except Exception:
        return []


def _add_to_recent_projects(filepath):
    """Add a project to the recent projects list."""
    config_dir = _get_config_dir()
    recent_file = os.path.join(config_dir, 'recent_projects.json')

    # Load existing
    recent = []
    if os.path.exists(recent_file):
        try:
            with open(recent_file, 'r', encoding='utf-8') as f:
                recent = json.load(f)
        except Exception:
            recent = []

    # Remove if already present
    recent = [r for r in recent if r['path'] != filepath]

    # Add to front
    name = os.path.splitext(os.path.basename(filepath))[0]
    recent.insert(0, {'name': name, 'path': filepath})

    # Keep only last 20
    recent = recent[:20]

    # Save
    try:
        with open(recent_file, 'w', encoding='utf-8') as f:
            json.dump(recent, f, indent=2)
    except Exception:
        pass


def _get_config_dir():
    """Get the configuration directory, creating if needed."""
    home = os.path.expanduser('~')
    config_dir = os.path.join(home, '.tgf_analysis')

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    return config_dir