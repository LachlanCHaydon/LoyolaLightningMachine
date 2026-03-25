# Loyola Lightning Machine (LLM) — Claude Code Project Context

## Project Overview
This is **TGF Tool v2**, a Python/tkinter GUI application for analyzing Terrestrial Gamma-ray Flash (TGF) events observed by the Telescope Array Surface Detector (TASD) at the Central Laser Facility (CLF) in Utah. Developed by Lachlan Haydon under Dr. Rasha Abbasi at Loyola University Chicago, as part of the Mulcahy Fellowship / Telescope Array collaboration.

The tool integrates multi-instrument data: VHF interferometry (INTF), fast antenna electric field (FA), surface detector array (TASD/SD), photometry (3-channel optical), high-speed camera spectroscopy, and Lightning Mapping Array (LMA).

---

## Directory Structure

```
tgf_analysis/
├── data_handlers/
│   ├── __init__.py
│   ├── fast_antenna.py       # FA electric field data loader
│   ├── frame_data.py         # High-speed camera frame handling
│   ├── interferometer.py     # INTF VHF direction cosine data
│   ├── lma.py                # Lightning Mapping Array data
│   ├── luminosity.py         # Luminosity from camera frames
│   ├── photometer.py         # 3-channel optical photometer
│   └── tasd.py               # Surface detector array
├── gui/
│   ├── __init__.py
│   └── tabs/
│       ├── __init__.py
│       ├── home_plotter.py       # Master multi-instrument overlay plot
│       ├── intf_tab.py           # INTF calibration and Az-El plotting
│       ├── map_visualizer_tab.py # NLDN map visualizer
│       ├── photometry_tab.py     # Photometer waveforms and ratios
│       └── spectroscopy_tab.py   # High-speed camera spectroscopy
├── utils/
│   ├── __init__.py
│   ├── project_io.py         # Save/load .tgf JSON project files
│   └── ta_tools.py           # Telescope Array utility functions
├── __init__.py
├── __main__.py
├── config.py                 # Constants, colormaps, calibration values
└── main.py                   # Entry point, MainApplication, StartupDialog
```

---

## Tab Inventory (in notebook order)

| Tab | Status | Class | Key methods |
|-----|--------|-------|-------------|
| Home (Master Plotter) | ✅ Built | `HomePlotterTab` | `load_from_project()`, `_save_to_project()` |
| INTF | ✅ Built | `INTFTab` | `load_from_project()`, `_save_to_project()` |
| Timeshift | 🚧 Placeholder | — | — |
| Luminosity | 🚧 Placeholder | — | — |
| Spectroscopy | ✅ Built | `SpectroscopyTab` | `load_from_project()`, `_save_to_project()` |
| Photometry | ✅ Built | `PhotometryTab` | `load_from_project()`, `_save_to_project()` |
| Map Visualizer (NLDN) | ✅ Built | `MapVisualizerTab` | `load_from_project()`, `_save_to_project()` |

---

## Project Save/Load Pattern

- Project files use `.tgf` extension (JSON format)
- `main.py` calls `tab._save_to_project()` on **every** tab before writing to disk
- `project_io.py` uses **absolute paths** for all file references
- `ProjectState` is the central data container — all tabs read/write to it
- `map_visualizer` dict exists in `ProjectState` but is **not yet included in `to_dict()`** — known gap, needs fixing

### ProjectState Dict Keys
```python
state.event_info      # date, time, stroke_number, flash_number, description
state.files           # fa, intf_raw, intf_calibrated, sd_directory, lma,
                      # photometer, luminosity, hsv_directory, spectra_directory
state.timing          # T0, timeshift, photometer_second_offset
state.photometer      # event_time, second_offset, time_start/stop, show_337/391/777, show_raw, show_ratios
state.intf            # cos_shift_a, cos_shift_b, is_calibrated
state.plot_ranges     # main, zoom, intf_elev, intf_azi, fa, sd
state.visibility      # fa, intf, sd, lma, luminosity, photometer_337/391/777
state.plot_style      # show_grid, show_legend, title
state.spectroscopy    # roi, tilt_angle, flip_horizontal, start_timestamp, frame_interval,
                      # keyframes, frames_data, analysis_complete, current_frame_idx,
                      # source_directory, peak_prominence, peak_distance,
                      # baseline_removal, baseline_degree, poly_order
state.results         # timeshift_analysis, source_location, spectroscopy_ratios
state.map_visualizer  # (dict, currently not persisted — known bug)
```

---

## Key Scientific / Technical Context

### Instruments
- **INTF (VHF Interferometer):** Measures direction of arrival of VHF radiation using direction cosines (cosA, cosB). Requires calibration constants `cosShiftA` and `cosShiftB` to correct systematic offsets. Default 2024 values: `cosShiftA = -0.0051`, `cosShiftB = -0.0178`.
- **FA (Fast Antenna):** Electric field waveform data. Used to identify return stroke timing.
- **TASD:** 507-detector surface particle detector array. VEM (Vertical Equivalent Muon) weighted.
- **Photometer:** 3 channels — 337nm (NII, blue), 391nm (NII, purple), 777nm (OI triplet, red). Sampled at 20 MHz.
- **Spectroscopy:** Phantom V711 / V2012 high-speed cameras capturing lightning spectra. ROI tracked across frames, keyframe calibration used.
- **LMA:** GPS-referenced 3D lightning source positions. Used for INTF calibration cross-matching.

### INTF Calibration
- LMA cross-matching method achieves **0.059° angular accuracy** (superior to NLDN ~0.5°)
- Quality filters: chi-squared < 2.0, altitude 1–15 km, power > -20 dBW
- Return strokes must reach ground (elevation = 0°) — key physical constraint
- Sigma clipping used for robust outlier rejection
- Systematic tilt in azimuth-elevation correlation indicates need for azimuth correction

### INTF Plotting Specs
- Colormap: **mjet** (custom, defined in `config.py`)
- Alpha binning: **0.3 / 0.7 / 1.0** (low / mid / high confidence)
- Az-El scatter plot with time-colored points

### Spectroscopy
- Primary anchor lines: **656nm (Hα)** and **777nm (OI triplet)**
- Known lines: 424(NII), 463(NII), 500(NII), 568(NII), 656(HI), 715(NI), 744(NI), 777(OI), 795(OI), 822(NI), 824(OI), 844(OI), 868(NI)
- Keyframe calibration: user manually calibrates 3–5 frames, program interpolates polynomial coefficients between them
- Spectral drift between frames is ~±25nm — keyframe interpolation addresses this
- Ion-to-neutral ratios (e.g. 337/777) indicate lightning energy state; high ratios suggest energetic breakdown

### Photometer Physics
- 337nm and 391nm: nitrogen ion emission (in-cloud, high energy)
- 777nm: oxygen neutral emission (lower altitude, return stroke)
- Large 777nm spike at cloud-break > return stroke spike is a key observed phenomenon in TGF events

### Camera / Geometry
- Camera location: lat=39.339082, lon=-112.700696, alt=1.4 km
- INTF antenna co-located with camera
- Image dimensions for Phantom TIF files: **1280×448** (not 1280×484 — the extra pixels are a timestamp bar)
- Distance to event from `calc_iterative` output (not assumed) — past error used 31.1 km vs actual 17.83 km

---

## Code Preferences (Critical)

1. **No relative imports** — never use dots before module names (e.g. use `from config import X`, not `from .config import X`)
2. **Output only changed files** — do not output full packages or zip archives
3. **Specify line changes** — when possible, state exact line numbers being changed
4. **Scientific integrity is non-negotiable** — do not simplify or alter physics/math for convenience
5. **Consistency with published work** — follow established 2023 analysis workflows; maintain matplotlib formatting consistent with prior publications

---

## GUI Layout Specs
- Sidebar width: **670px**
- Home tab hides y-axis when INTF is toggled off
- Event time labels on colorbars

---

## Key Collaborators / Data Sources
- **Dr. Rasha Abbasi** — PI, Loyola University Chicago
- **Mark Stanley** — VHF calibration (New Mexico Tech), provided 2024 cosShift values
- **Bill Rison** — LMA data provider
- **Utah researchers** — updated cosShiftA/cosShiftB values pending for 2025 season
- **NLDN** — lightning strike database
- **ICRR rtuple files** — TASD event data format

---

## Known Issues / TODOs
- `map_visualizer` dict not included in `ProjectState.to_dict()` — won't persist on save
- Timeshift and Luminosity tabs are placeholder stubs
- 2025 INTF calibration values (cosShiftA/cosShiftB) from Utah still pending
- Debug print statements remain in `project_io.py` `save_project()` function


## Publication Figure Tabs (v2 Addition)

Three new figure tabs produce paper-ready plots. All share data already
loaded in the project (FA, INTF, photometer, TASD, LMA). Tabs live in
`gui/tabs/figures/` as:
  - `flash_overview_tab.py`   (Flash Overview)
  - `figure2_tab.py`          (Figure 2 — Multi-instrument overview with v711 frames)
  - `figure3_tab.py`          (Figure 3 — Per-stroke grid with dart leader velocity)

### Handler Changes Required
- `data_handlers/photometer.py`: expose `sample_rate` (20 or 30 MHz) as a
  user-settable parameter at load time; add it to ProjectState.photometer dict.
  The 2023 event used 30 MHz; do not hardcode 20 MHz.
- `data_handlers/interferometer.py`: add `get_stratified_scatter()` method that
  returns the three s-ratio tiers (sLevelsTup = [1.0, 3.0, 7.0, 16.0],
  alphaTup = [0.3, 0.7, 1.0]) as a list of dicts, each with keys: time, elv,
  azi, colors, marker_sizes, alpha. This is used by all figure tabs for INTF
  scatter. Do not break existing `filter_data()` or `get_full_data()`.

### INTF Scatter Convention
All INTF scatter plots use the s-ratio stratified scatter (3 calls, lowest-s 
drawn first/under). Colors from cmap_mjet, marker sizes from (1+3*ss²)². 
This is the scientific standard for this data type.

### Frame Panel Convention (v711 frames)
Camera frame panels below main plots: images loaded from a user-specified
directory, displayed via OffsetImage + AnnotationBbox, connected to the time
axis via ConnectionPatch dashed lines. Users provide image paths + timestamps.

### Velocity Calculation (Figure 3)
Dart leader velocity = linear regression of INTF elevation vs time in a 
user-specified window. Convert elevation to altitude via z = x1 * tan(elv),
where x1 = horizontal distance camera to source (km, from LMA or user input).
Velocity = dz/dt. Report in km/s. Use scipy.stats.linregress.
