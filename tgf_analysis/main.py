import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Fix DPI scaling on Windows
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except:
    pass

from config import APP_NAME, APP_VERSION
from utils.project_io import ProjectState, save_project, load_project, get_recent_projects
from gui.tabs.home_plotter import HomePlotterTab
from gui.tabs.intf_tab import INTFTab
from gui.tabs.photometry_tab import PhotometryTab
from gui.tabs.spectroscopy_tab import SpectroscopyTab


class StartupDialog(tk.Toplevel):
    """
    Startup dialog for project selection.

    Options:
    - Open existing project
    - Create new project
    - Find event with NLDN (future)
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.result = None  # Will be set to action taken
        self.project_path = None
        # Style the notebook tabs
        style = ttk.Style()
        style.configure('TNotebook.Tab',
                        padding=[12, 6],  # Horizontal, vertical padding
                        font=('Helvetica', 10))
        style.map('TNotebook.Tab',
                  background=[('selected', '#4a90d9'), ('!selected', '#d9d9d9')],
                  padding=[('selected', [14, 8])])  # Slightly larger when selected

        self.title(f"{APP_NAME} - Welcome")
        self.geometry("500x400")
        self.resizable(False, False)

        # Center on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 500) // 2
        y = (self.winfo_screenheight() - 400) // 2
        self.geometry(f"+{x}+{y}")

        # Make modal
        self.transient(parent)
        self.grab_set()

        self._build_ui()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _build_ui(self):
        """Build the startup dialog UI."""
        # Header
        header_frame = ttk.Frame(self, padding=20)
        header_frame.pack(fill=tk.X)

        ttk.Label(
            header_frame,
            text=APP_NAME,
            font=('Helvetica', 16, 'bold')
        ).pack()

        ttk.Label(
            header_frame,
            text=f"Version {APP_VERSION}",
            font=('Helvetica', 10)
        ).pack()

        ttk.Separator(self, orient='horizontal').pack(fill=tk.X, padx=20)

        # Main content
        content_frame = ttk.Frame(self, padding=20)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Buttons
        btn_frame = ttk.Frame(content_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            btn_frame,
            text="📂 Open Existing Project",
            command=self._open_project,
            width=30
        ).pack(pady=5)

        ttk.Button(
            btn_frame,
            text="✨ Create New Project",
            command=self._new_project,
            width=30
        ).pack(pady=5)

        ttk.Button(
            btn_frame,
            text="🔍 Find Event with NLDN",
            command=self._find_event,
            width=30
        ).pack(pady=5)

        # Recent projects
        recent = get_recent_projects(5)
        if recent:
            ttk.Separator(content_frame, orient='horizontal').pack(fill=tk.X, pady=10)

            ttk.Label(
                content_frame,
                text="Recent Projects:",
                font=('Helvetica', 10, 'bold')
            ).pack(anchor='w')

            for project_name, project_path in recent:
                if os.path.exists(project_path):
                    name = os.path.basename(project_path)
                    btn = ttk.Button(
                        content_frame,
                        text=f"  {name}",
                        command=lambda p=project_path: self._open_recent(p),
                        width=40
                    )
                    btn.pack(anchor='w', pady=2)

    def _open_project(self):
        """Open an existing project file."""
        filepath = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("TGF Project", "*.tgf"), ("All Files", "*.*")]
        )
        if filepath:
            self.result = 'open'
            self.project_path = filepath
            self.destroy()

    def _open_recent(self, filepath):
        """Open a recent project."""
        if os.path.exists(filepath):
            self.result = 'open'
            self.project_path = filepath
            self.destroy()
        else:
            messagebox.showerror("Error", f"Project file not found:\n{filepath}")

    def _new_project(self):
        """Create a new project."""
        self.result = 'new'
        self.destroy()

    def _find_event(self):
        """Open NLDN event finder (future feature)."""
        self.result = 'nldn'
        self.destroy()

    def _on_cancel(self):
        """Handle dialog cancellation."""
        self.result = 'cancel'
        self.destroy()


class MainApplication(tk.Tk):
    """
    Main application window.

    Contains the notebook with all analysis tabs and manages
    the global project state.
    """

    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("1400x900")

        # Initialize project state
        self.project_state = ProjectState()
        self.project_filepath = None
        self.unsaved_changes = False

        # Data handlers (shared across tabs)
        self.data_handlers = {
            'photometer': None,
            'fast_antenna': None,
            'intf': None,
            'tasd': None,
            'lma': None,
            'luminosity': None,
        }

        # Build UI
        self._build_menu()
        self._build_ui()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Show startup dialog
        self.after(100, self._show_startup)

    def _build_menu(self):
        """Build the menu bar."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="New Project", command=self._new_project, accelerator="Ctrl+N")
        file_menu.add_command(label="Open Project...", command=self._open_project, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Project", command=self._save_project, accelerator="Ctrl+S")
        file_menu.add_command(label="Save Project As...", command=self._save_project_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)

        # Bind keyboard shortcuts
        self.bind('<Control-n>', lambda e: self._new_project())
        self.bind('<Control-o>', lambda e: self._open_project())
        self.bind('<Control-s>', lambda e: self._save_project())

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _build_ui(self):
        """Build the main UI."""
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create placeholder tabs
        # These will be replaced with full implementations

        # Home/Master Plotter tab
        self.home_tab = HomePlotterTab(self.notebook, self)
        self.notebook.add(self.home_tab, text="Home (Master Plotter)")

        # INTF tab
        self.intf_tab = INTFTab(self.notebook, self)
        self.notebook.add(self.intf_tab, text="INTF")

        # Timeshift tab
        self.timeshift_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.timeshift_frame, text="Timeshift")
        self._build_placeholder(self.timeshift_frame, "Timeshift Calculator",
                               "Calculate timing offsets using INTF, SD, and LMA data")

        # Luminosity tab
        self.luminosity_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.luminosity_frame, text="Luminosity")
        self._build_placeholder(self.luminosity_frame, "Luminosity & LMA",
                               "Luminosity calculation from high-speed camera and LMA data processing")

        # Spectroscopy tab
        self.spectroscopy_tab = SpectroscopyTab(self.notebook, self)
        self.notebook.add(self.spectroscopy_tab, text='Spectroscopy')

        # Photometry tab
        self.photometry_tab = PhotometryTab(self.notebook, self)
        self.notebook.add(self.photometry_tab, text="Photometry")

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w')
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_placeholder(self, parent, title, description):
        """Build a placeholder for tabs not yet implemented."""
        frame = ttk.Frame(parent, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=title, font=('Helvetica', 18, 'bold')).pack(pady=10)
        ttk.Label(frame, text=description, font=('Helvetica', 12)).pack(pady=5)

        ttk.Label(
            frame,
            text="🚧 Tab under construction 🚧",
            font=('Helvetica', 14),
            foreground='gray'
        ).pack(pady=30)

        # Add file input section as proof of concept
        file_frame = ttk.LabelFrame(frame, text="Data Files", padding=10)
        file_frame.pack(fill=tk.X, pady=20)

        ttk.Label(file_frame, text="Global file paths will be loaded here").pack()

    def _show_startup(self):
        """Show the startup dialog."""
        dialog = StartupDialog(self)
        self.wait_window(dialog)

        if dialog.result == 'open' and dialog.project_path:
            self._load_project(dialog.project_path)
        elif dialog.result == 'new':
            self._new_project()
        elif dialog.result == 'nldn':
            self._show_nldn_finder()
        elif dialog.result == 'cancel':
            pass  # Just continue with empty project

    def _new_project(self):
        """Create a new project."""
        if self.unsaved_changes:
            if not messagebox.askyesno("Unsaved Changes",
                                       "You have unsaved changes. Create new project anyway?"):
                return

        self.project_state = ProjectState()
        self.project_filepath = None
        self.unsaved_changes = False
        self._update_title()
        self.status_var.set("New project created")

    def _open_project(self):
        """Open a project file."""
        if self.unsaved_changes:
            if not messagebox.askyesno("Unsaved Changes",
                                       "You have unsaved changes. Open another project?"):
                return

        filepath = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("TGF Project", "*.tgf"), ("All Files", "*.*")]
        )
        if filepath:
            self._load_project(filepath)

    def _load_project(self, filepath):
        """Load a project from file."""
        state = load_project(filepath)
        if state:
            self.project_state = state
            self.project_filepath = filepath
            self.unsaved_changes = False
            self._update_title()
            self.status_var.set(f"Loaded: {os.path.basename(filepath)}")

            # Update tabs with loaded data - each in try/except so one failure doesn't block others
            try:
                self.home_tab.load_from_project()
            except Exception as e:
                print(f"Warning: Failed to load home_tab: {e}")

            try:
                self.intf_tab.load_from_project()
            except Exception as e:
                print(f"Warning: Failed to load intf_tab: {e}")

            try:
                self.photometry_tab.load_from_project()
            except Exception as e:
                print(f"Warning: Failed to load photometry_tab: {e}")

            try:
                self.spectroscopy_tab.load_from_project()
            except Exception as e:
                print(f"Warning: Failed to load spectroscopy_tab: {e}")
                import traceback
                traceback.print_exc()

        else:
            messagebox.showerror("Error", f"Failed to load project:\n{filepath}")

    def _save_project(self):
        """Save the current project."""
        if self.project_filepath:
            try:
                # Sync current tab state to project_state before saving
                if hasattr(self, 'home_tab'):
                    self.home_tab._save_to_project()
                if hasattr(self, 'intf_tab'):
                    self.intf_tab._save_to_project()
                if hasattr(self, 'photometry_tab'):
                    self.photometry_tab._save_to_project()
                if hasattr(self, 'spectroscopy_tab'):
                    self.spectroscopy_tab._save_to_project()


                save_project(self.project_state, self.project_filepath)
                self.unsaved_changes = False
                self._update_title()
                self.status_var.set(f"Saved: {os.path.basename(self.project_filepath)}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", f"Failed to save project: {e}")
        else:
            self._save_project_as()

    def _save_project_as(self):
        """Save project with a new name."""
        filepath = filedialog.asksaveasfilename(
            title="Save Project As",
            defaultextension=".tgf",
            filetypes=[("TGF Project", "*.tgf"), ("All Files", "*.*")]
        )
        if filepath:
            try:
                # Sync current tab state to project_state before saving
                if hasattr(self, 'home_tab'):
                    self.home_tab._save_to_project()
                if hasattr(self, 'intf_tab'):
                    self.intf_tab._save_to_project()
                if hasattr(self, 'photometry_tab'):
                    self.photometry_tab._save_to_project()
                if hasattr(self, 'spectroscopy_tab'):
                    self.spectroscopy_tab._save_to_project()

                self.project_state.project_name = os.path.splitext(os.path.basename(filepath))[0]
                save_project(self.project_state, filepath)
                self.project_filepath = filepath
                self.unsaved_changes = False
                self._update_title()
                self.status_var.set(f"Saved: {os.path.basename(filepath)}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", f"Failed to save project: {e}")

    def _show_nldn_finder(self):
        """Show the NLDN event finder (placeholder)."""
        messagebox.showinfo("NLDN Event Finder",
                           "NLDN event finder will be implemented here.\n\n"
                           "This feature will allow you to:\n"
                           "• Load NLDN data files\n"
                           "• Plot source locations on TASD map\n"
                           "• Match with rtuple data to find TGF events")

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            f"{APP_NAME}\n\n"
            f"Version: {APP_VERSION}\n\n"
            "A tool for analyzing Terrestrial Gamma-ray Flash\n"
            "observations from the Telescope Array Surface Detector.\n\n"
            "Developed by Lachlan Haydon, 2026. "
            "Built with code developed by Davide Muzzucco, Ny Kieu, "
            "Rasha Abbasi, and the Telescope Array Project "
        )

    def _update_title(self):
        """Update window title with project name and save status."""
        name = self.project_state.project_name or "Untitled"
        modified = " *" if self.unsaved_changes else ""
        self.title(f"{name}{modified} - {APP_NAME} v{APP_VERSION}")

    def _on_close(self):
        """Handle window close event."""
        if self.unsaved_changes:
            result = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Save before closing?"
            )
            if result is True:  # Yes
                self._save_project()
                if not self.unsaved_changes:  # Save succeeded
                    self.destroy()
            elif result is False:  # No
                self.destroy()
            # Cancel - do nothing
        else:
            self.destroy()

    def mark_unsaved(self):
        """Mark the project as having unsaved changes."""
        self.unsaved_changes = True
        self._update_title()


def run():
    """Run the TGF Analysis Tool."""
    app = MainApplication()
    app.mainloop()


if __name__ == "__main__":
    run()
