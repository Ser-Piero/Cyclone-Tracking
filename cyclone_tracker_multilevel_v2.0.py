#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cyclone_tracker_multilevel.py
=============================
Multi-level cyclone tracking tool for ICON and WRF mesoscale model output.

The algorithm tracks a cyclone center by locating the minimum and centroid
of geopotential height (or sea-level pressure) within a search radius at
multiple pressure levels.  A weighted-mean track is then derived from all
levels and optionally enriched with diagnostic variables (SLP, SST, winds,
heat fluxes, precipitation, potential vorticity, ...).

Supported model back-ends
--------------------------
  MODEL = "ICON"  –  ICON unstructured-grid NetCDF output (via xarray + MetPy)
  MODEL = "WRF"   –  WRF structured-grid output          (via netCDF4 + wrf-python)

Quick-start
-----------
  1. Set MODEL to "ICON" or "WRF" in the USER CONFIGURATION section below.
  2. Fill in infolders, sims, outfolder and the initial cyclone position/date.
  3. Toggle export_variables flags as needed.
  4. Run:  python cyclone_tracker_multilevel.py

Output
------
  * <cyclone>_<sim>_track_multivarz_wm.csv  – weighted-mean track + diagnostics
  * (optional) per-level CSV files           – when save_all_tracks = True
  * (optional) PNG plots                     – when plot = True

Authors
-------
  Piero Serafini
    PhD student in Atmospheric Physics
      University of L'Aquila (UNIVAQ)
      Center of Excellence in Telesensing of Environment and Model Prediction of Severe Events (CETEMPS)
        Via Vetoio, Edificio Renato Ricamo, L'Aquila (AQ), Italy, 67100
          piero.serafini@graduate.univaq.it

License
-------
  MIT License – see LICENSE file.

Citation / DOI
--------------
  This code is archived on Zenodo.
  If you use this tool in a publication, please cite it as:
    Serafini P. et al. (2025). cyclone_tracker_multilevel – Multi-level
    cyclone tracking for ICON and WRF. Zenodo.
    https://doi.org/10.5281/zenodo.XXXXXXX  <-- replace after upload

How to get a DOI for this code
--------------------------------
  1. Push the code to a GitHub repository (https://github.com).
  2. Go to https://zenodo.org and log in with your GitHub account.
  3. In Zenodo → "GitHub" tab, enable the repository.
  4. Create a GitHub Release (tag, e.g. v1.0.0).
  5. Zenodo automatically archives the release and mints a DOI.
  Alternative: upload the .zip directly at https://zenodo.org/deposit.
  Zenodo is free and its DOIs are permanent and citable.
"""

# =============================================================================
# SECTION 0 – STANDARD LIBRARY / CROSS-MODEL IMPORTS
# =============================================================================

import os           # path and permission checks
import sys          # clean exit
import time         # wall-clock timing
import logging      # structured, levelled debug output
import traceback    # full tracebacks on unexpected errors
import warnings     # issue non-fatal warnings with source location
import inspect      # introspection to report line numbers in messages
import locale       # set locale so month names are always in English

import numpy as np                  # numerical arrays
import pandas as pd                 # DataFrames, CSV I/O, datetime parsing
import matplotlib                   # backend must be set before pyplot import
import matplotlib.pyplot as plt
import cartopy.crs as ccrs                          # map projections
from cartopy.feature import NaturalEarthFeature     # land/ocean shading
from cartopy.feature import COLORS                  # predefined map colors
from shapely.geometry import Polygon                # convex-hull centroid
from scipy.spatial import ConvexHull                # convex-hull computation

# Force English month names in timestamps (works on Linux/macOS)
try:
    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
except locale.Error:
    warnings.warn(
        "Could not set locale to 'en_US.UTF-8'. "
        "Month names in timestamps may be in the system language.",
        UserWarning, stacklevel=2
    )

# Use a non-interactive Matplotlib backend (safe for servers without a display)
matplotlib.use('Agg')

# =============================================================================
# SECTION 1 – LOGGING CONFIGURATION
# =============================================================================
# All runtime messages go through the standard logging module.
# Change the level to logging.DEBUG to see every internal detail,
# or to logging.WARNING to suppress informational messages.

logging.basicConfig(
    level=logging.INFO,   # <-- change to logging.DEBUG for maximum verbosity
    format="[%(levelname)s | line %(lineno)d | %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),   # console output
        # logging.FileHandler("tracker.log"), # optionally also log to a file
    ]
)
log = logging.getLogger(__name__)

# =============================================================================
# SECTION 2 – USER CONFIGURATION  ← EDIT THIS SECTION BEFORE RUNNING
# =============================================================================

# --- 2.0  Model selection (the ONLY thing that changes the code path) --------
#     Set MODEL to "ICON" or "WRF".
MODEL = "WRF"   # <-- "ICON" or "WRF"

# --- 2.1  Experiment metadata ------------------------------------------------
CYCLONE = "IANOS"    # name of the cyclone (used in output file names)

# --- 2.2  Input/output folders -----------------------------------------------
# List one entry per simulation.
# ICON: separate model-level (_ML_) and pressure-level (_PL_) sub-folders
#       can live in the SAME parent folder; the script filters by filename tag.
# WRF:  one folder per simulation containing wrfout_* files.
INFOLDERS = [
    # ICON example
    # "/mnt/TMP_DANIEL/ICON/V2_EXP",
    # "/mnt/TMP_DANIEL/ICON/V2_SH",

    # WRF example  (uncomment / edit as needed)
    "/home/HDD20TBB/UMBERTO/TAY10/ouputWRF",
    "/home/HDD20TBB/UMBERTO/NEW00/ouputWRF",
    "/home/HDD20TBB/UMBERTO/NEW01/ouputWRF",
    "/home/HDD20TBB/UMBERTO/NEW10/ouputWRF",
    "/home/HDD20TBB/UMBERTO/NEW20/ouputWRF",
]

# Short label for each simulation (must match INFOLDERS order)
SIMS = [
    # ICON example
    # "ICON_EXP",
    # "ICON_SH",

    # WRF example
    "TAY10",
    "NEW00",
    "NEW01",
    "NEW10",
    "NEW20",
]

# Output folder where all CSV and PNG files will be written
OUTFOLDER = "/home/HDD20TBB/UMBERTO/Postprocess"

# WRF only: prefix of WRF output files (one entry per simulation)
# ICON does not use this; you can leave it as-is when MODEL = "ICON"
WRF_PREFIXES = ["wrfout_"] * len(SIMS)   # default: same prefix for all sims

# --- 2.3  Tracking parameters ------------------------------------------------
# Date and time of the FIRST timestep you want to analyse.
# Files before this date are silently skipped.
# Format: 'DD-Mon-YYYY HH:MM UTC'  (e.g. '15-Sep-2020 00:00 UTC')
START_DATE = '15-Sep-2020 00:00 UTC'

# Approximate cyclone center at START_DATE (degrees North / East)
# Use a weather-map, reanalysis, or best-track dataset to estimate this.
S0LAT = 32.9   # initial latitude  [°N]
S0LON = 15.7   # initial longitude [°E]

# Search radius for the tracking algorithm [km]
# Recommended: 150 km (Mediterranean), 300 km (Atlantic/large systems)
SEARCH_RADIUS_KM = 150

# Pressure levels at which geopotential height is tracked [hPa]
# Level 0 is a special code meaning "sea-level pressure"
INTERP_LEVELS_HPA = np.array(
    [800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 925, 950, 0],
    dtype=int
)

# --- 2.4  Diagnostic variables to export ------------------------------------
# Set each flag to True to include that variable in the output CSV.
# Variables that are unavailable for a given model are silently skipped
# (a warning is issued instead of crashing the program).
EXPORT_VARIABLES = {
    "min_slp":      True,   # minimum sea-level pressure inside search circle [hPa]
    "max_sst":      True,   # maximum sea-surface temperature                  [K]
    "mean_sst":     True,   # mean    sea-surface temperature                  [K]
    "max_wind10m":  True,   # maximum 10-m wind speed                          [m/s]
    "max_lhf":      True,   # maximum latent heat flux                         [W/m²]
    "mean_lhf":     True,   # mean    latent heat flux                         [W/m²]
    "max_shf":      True,   # maximum sensible heat flux                       [W/m²]
    "mean_shf":     True,   # mean    sensible heat flux                       [W/m²]
    "max_qvf":      True,   # maximum water-vapour flux                        [kg/m²/s]
    "mean_qvf":     True,   # mean    water-vapour flux                        [kg/m²/s]
    "mean_pw":      True,   # mean    precipitable water                       [mm]
    "max_pvo":      True,   # maximum potential vorticity at 300 hPa           [PVU]
    "max_rh":       True,   # maximum 2-m relative humidity                    [%]
    "max_rain":     True,   # maximum hourly accumulated rainfall               [mm/h]
    "mean_rain":    True,   # mean    hourly accumulated rainfall               [mm/h]
}

# --- 2.5  Output options -----------------------------------------------------
SAVE_ALL_TRACKS = False   # True → also save per-level CSV files (larger output)
PLOT            = True    # True → generate PNG maps (multi-level + SLP track)
CHECK_PLOTS     = False   # True → generate per-timestep SLP check plots
                          #         (slow – only for debugging the tracker)

# =============================================================================
# SECTION 3 – CONDITIONAL IMPORTS  (loaded only for the selected model)
# =============================================================================

if MODEL == "ICON":
    # ICON: xarray for NetCDF I/O, MetPy for unit-aware CF metadata
    try:
        import xarray as xr
        import metpy  # noqa – metpy.parse_cf() accessed via xr accessor
    except ImportError as exc:
        log.error(
            "Cannot import ICON dependencies (xarray / metpy). "
            "Install them with:  pip install xarray metpy\n"
            "Original error: %s", exc
        )
        sys.exit(1)

elif MODEL == "WRF":
    # WRF: netCDF4 for file I/O, wrf-python for variable extraction
    try:
        from netCDF4 import Dataset
        from wrf import getvar, to_np, vinterp
    except ImportError as exc:
        log.error(
            "Cannot import WRF dependencies (netCDF4 / wrf-python). "
            "Install them with:  pip install netCDF4 wrf-python\n"
            "Original error: %s", exc
        )
        sys.exit(1)

else:
    # Catch typos in MODEL immediately so the user gets a clear message
    log.error(
        "Unknown MODEL value: '%s'. Please set MODEL to 'ICON' or 'WRF' "
        "in the USER CONFIGURATION section (line ~90 of this file).",
        MODEL
    )
    sys.exit(1)

log.info("Model back-end selected: %s", MODEL)

# =============================================================================
# SECTION 4 – HELPER FUNCTIONS
# =============================================================================

def check_and_create_folder(folder_path):
    """
    Ensure a directory exists, creating it (including parents) if necessary.

    Parameters
    ----------
    folder_path : str
        Absolute or relative path to the directory.

    Raises
    ------
    OSError
        If the directory cannot be created (e.g. insufficient permissions).
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)   # makedirs creates all intermediate dirs
            log.info("Created output folder: %s", folder_path)
        except OSError as exc:
            log.error(
                "Could not create folder '%s'. "
                "Check that the parent directory exists and is writable.\n"
                "Error type : %s\n"
                "Error detail: %s",
                folder_path, type(exc).__name__, exc
            )
            raise   # re-raise so the calling code can decide whether to abort


def haversine(lat_array, lon_array, lat_center, lon_center):
    """
    Compute the great-circle distance [km] from a single center point to every
    point in a 1-D or 2-D lat/lon array.

    Uses the Haversine formula instead of geodesic distance because it is
    ~100× faster and gives indistinguishable results at the scales used here.

    Parameters
    ----------
    lat_array, lon_array : np.ndarray
        Arrays of latitude and longitude (degrees) for all grid points.
        Shapes must match.
    lat_center, lon_center : float
        Center point (degrees).

    Returns
    -------
    np.ndarray
        Distance in kilometres, same shape as lat_array / lon_array.
    """
    EARTH_RADIUS_KM = 6371.0   # mean radius of a spherical Earth

    lat1 = np.radians(lat_center)
    lon1 = np.radians(lon_center)
    lat2 = np.radians(lat_array)
    lon2 = np.radians(lon_array)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def locate_center(var, mask, lats, lons):
    """
    Find the centroid and minimum of *var* inside the search circle defined by
    *mask*.

    The centroid is computed as the geometric centroid of the convex hull
    formed by the points that fall below the 5th percentile of *var* within
    the mask.  This makes the algorithm robust to isolated noisy pixels.

    Parameters
    ----------
    var : array-like
        2-D field to analyse (e.g. geopotential height or SLP).
        Must be convertible to a NumPy array.
    mask : np.ndarray of bool
        True where the grid point lies inside the search circle.
    lats, lons : np.ndarray
        Latitude and longitude arrays with the same shape as *var*.

    Returns
    -------
    lat_centroid, lon_centroid : float
        Centroid of the convex hull built from the lowest 5% of values.
    lat_minimum, lon_minimum : float
        Location of the absolute minimum of *var* inside the mask.

    Raises
    ------
    ValueError
        If fewer than 3 valid points are found (ConvexHull needs ≥ 3 points).
    """
    # Convert to a plain NumPy array regardless of whether var is an
    # xarray.DataArray (ICON) or a wrf-python masked array (WRF)
    var_np = np.asarray(var)

    # Mask points outside the search circle with NaN
    var_masked = np.where(mask, var_np, np.nan)

    # Find the 5th percentile threshold to isolate the low-value core
    threshold = np.nanpercentile(var_masked, 5)

    # Build a secondary mask for the core region (below threshold)
    core_mask = var_masked < threshold

    # Extract latitudes and longitudes of core points
    lats_core = np.where(core_mask, lats, np.nan)
    lons_core = np.where(core_mask, lons, np.nan)

    valid = ~np.isnan(lons_core) & ~np.isnan(lats_core)
    lon_valid = lons_core[valid].flatten()
    lat_valid = lats_core[valid].flatten()

    # ConvexHull requires at least 3 non-collinear points
    if len(lon_valid) < 3:
        warnings.warn(
            f"locate_center: only {len(lon_valid)} valid core points found "
            f"(need ≥ 3 for ConvexHull). Returning NaN for centroid. "
            f"Consider increasing SEARCH_RADIUS_KM. "
            f"[called from line {inspect.currentframe().f_back.f_lineno}]",
            UserWarning, stacklevel=2
        )
        min_idx = np.unravel_index(np.nanargmin(var_masked), var_masked.shape)
        return np.nan, np.nan, float(lats[min_idx]), float(lons[min_idx])

    points = np.column_stack((lon_valid, lat_valid))   # shape (N, 2)

    try:
        hull = ConvexHull(points)
        poly = Polygon(points[hull.vertices])
        lat_centroid = poly.centroid.y
        lon_centroid = poly.centroid.x
    except Exception as exc:
        # ConvexHull can fail (e.g. all points collinear); fall back to mean
        warnings.warn(
            f"locate_center: ConvexHull failed ({type(exc).__name__}: {exc}). "
            f"Falling back to arithmetic mean of core points. "
            f"[called from line {inspect.currentframe().f_back.f_lineno}]",
            UserWarning, stacklevel=2
        )
        lat_centroid = float(np.nanmean(lat_valid))
        lon_centroid = float(np.nanmean(lon_valid))

    # Location of the absolute minimum inside the full search circle
    min_idx = np.unravel_index(np.nanargmin(var_masked), var_masked.shape)
    lat_minimum = float(lats[min_idx])
    lon_minimum = float(lons[min_idx])

    return lat_centroid, lon_centroid, lat_minimum, lon_minimum


# =============================================================================
# SECTION 5 – INPUT VALIDATION
# =============================================================================

def validate_inputs(infolders, sims, outfolder, wrf_prefixes=None):
    """
    Check that all user-supplied paths and list lengths are consistent before
    the main loop starts.  Raises clear errors with actionable messages.

    Parameters
    ----------
    infolders : list of str
    sims      : list of str
    outfolder : str
    wrf_prefixes : list of str or None (only checked when MODEL == "WRF")
    """
    if len(infolders) != len(sims):
        raise ValueError(
            f"INFOLDERS has {len(infolders)} entries but SIMS has {len(sims)}. "
            "They must have the same length (one entry per simulation)."
        )

    if MODEL == "WRF" and wrf_prefixes is not None:
        if len(wrf_prefixes) != len(sims):
            raise ValueError(
                f"WRF_PREFIXES has {len(wrf_prefixes)} entries but SIMS has "
                f"{len(sims)}. They must match."
            )

    for folder in infolders:
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Input folder does not exist: '{folder}'\n"
                "Please update INFOLDERS in the USER CONFIGURATION section."
            )
        if not os.access(folder, os.R_OK):
            raise PermissionError(
                f"No read permission on input folder: '{folder}'"
            )

    log.info("Input validation passed. %d simulation(s) queued.", len(sims))


# =============================================================================
# SECTION 6 – MODEL-SPECIFIC FILE DISCOVERY
# =============================================================================

def get_file_lists_icon(infolder):
    """
    Return sorted lists of ICON model-level (_ML_) and pressure-level (_PL_)
    NetCDF files found in *infolder*.

    Parameters
    ----------
    infolder : str

    Returns
    -------
    ml_filelist, pl_filelist : list of str
        Absolute paths, sorted alphabetically (which usually = chronologically
        for standard ICON naming conventions).

    Raises
    ------
    FileNotFoundError
        If no pressure-level files are found (ML files are optional).
    """
    all_files = os.listdir(infolder)

    ml_filelist = sorted(
        [os.path.join(infolder, f) for f in all_files if "_ML_" in f]
    )
    pl_filelist = sorted(
        [os.path.join(infolder, f) for f in all_files if "_PL_" in f]
    )

    log.info("  ICON ML files found: %d", len(ml_filelist))
    log.info("  ICON PL files found: %d", len(pl_filelist))

    if not pl_filelist:
        raise FileNotFoundError(
            f"No pressure-level files (containing '_PL_') found in: '{infolder}'.\n"
            "Check that INFOLDERS points to the correct directory."
        )

    return ml_filelist, pl_filelist


def get_file_list_wrf(infolder, prefix):
    """
    Return a sorted list of WRF output files starting with *prefix* in *infolder*.

    Parameters
    ----------
    infolder : str
    prefix   : str  (e.g. "wrfout_")

    Returns
    -------
    list of str

    Raises
    ------
    FileNotFoundError
        If no matching files are found.
    """
    wrf_filelist = sorted(
        [os.path.join(infolder, f)
         for f in os.listdir(infolder) if f.startswith(prefix)]
    )

    log.info("  WRF files found (prefix='%s'): %d", prefix, len(wrf_filelist))

    if not wrf_filelist:
        raise FileNotFoundError(
            f"No WRF files starting with '{prefix}' found in: '{infolder}'.\n"
            "Check that WRF_PREFIXES and INFOLDERS are correct."
        )

    return wrf_filelist


# =============================================================================
# SECTION 7 – MAIN PROCESSING LOOP
# =============================================================================

def run_tracking(
    model, cyclone, sims, infolders, outfolder,
    start_date, s0lat, s0lon, search_radius_km,
    interp_levels_hpa, export_variables,
    save_all_tracks, plot, check_plots,
    wrf_prefixes=None
):
    """
    Iterate over all simulations and run the multi-level cyclone tracking
    algorithm.  Results are written to CSV files in *outfolder*.

    Parameters
    ----------
    model            : str          "ICON" or "WRF"
    cyclone          : str          cyclone name (used in output filenames)
    sims             : list of str  simulation labels
    infolders        : list of str  input data folders
    outfolder        : str          output folder (created if missing)
    start_date       : str          first date to process
    s0lat, s0lon     : float        initial cyclone position [°N, °E]
    search_radius_km : float        search radius [km]
    interp_levels_hpa: np.ndarray   pressure levels to track [hPa]; 0 = SLP
    export_variables : dict         {variable_name: bool}
    save_all_tracks  : bool
    plot             : bool
    check_plots      : bool
    wrf_prefixes     : list of str or None
    """

    check_and_create_folder(outfolder)   # ensure the output folder exists

    # ------------------------------------------------------------------
    # Outer loop: one iteration per simulation
    # ------------------------------------------------------------------
    for sim_idx, (sim, infolder) in enumerate(zip(sims, infolders)):

        log.info(
            "=" * 60 + "\nStarting simulation %d/%d : %s\n  Input folder: %s",
            sim_idx + 1, len(sims), sim, infolder
        )

        # ---- 7.1  Discover input files --------------------------------
        try:
            if model == "ICON":
                ml_filelist, pl_filelist = get_file_lists_icon(infolder)
                # Number of time steps driven by pressure-level files
                n_timesteps = len(pl_filelist)
            else:  # WRF
                prefix = wrf_prefixes[sim_idx]
                wrf_filelist = get_file_list_wrf(infolder, prefix)
                n_timesteps = len(wrf_filelist)
        except (FileNotFoundError, OSError) as exc:
            log.error(
                "Simulation '%s' skipped – could not build file list.\n"
                "Error type   : %s\n"
                "Error detail : %s\n%s",
                sim, type(exc).__name__, exc, traceback.format_exc()
            )
            continue   # skip to the next simulation instead of crashing

        # ---- 7.2  Resolve pressure levels for ICON --------------------
        # ICON pressure-level files contain discrete levels; we find the
        # closest available level to each requested level.
        if model == "ICON":
            try:
                ncfile_pl_probe = xr.open_dataset(pl_filelist[1])
                ncfile_pl_probe = ncfile_pl_probe.metpy.parse_cf().squeeze()
                plev_pa = ncfile_pl_probe["plev"].values   # levels in Pa

                # Map each requested hPa level to the nearest available level
                p_indices = np.unique(
                    np.array([
                        np.argmin(np.abs(plev_pa - lv * 100))
                        for lv in interp_levels_hpa
                        if lv != 0          # skip the SLP placeholder
                    ])
                ).astype(int)

                # The actual levels that will be used (back in Pa)
                icon_levels_pa = plev_pa[p_indices]
                # Append 0 as the SLP placeholder
                interp_levels = np.append(icon_levels_pa, 0)
                log.info(
                    "  ICON pressure levels mapped (Pa): %s",
                    interp_levels[:-1].astype(int)
                )
            except Exception as exc:
                log.error(
                    "Could not read ICON pressure levels from '%s'.\n"
                    "Error type   : %s\n"
                    "Error detail : %s",
                    pl_filelist[1], type(exc).__name__, exc
                )
                continue
        else:
            # WRF: wrf-python's vinterp handles level interpolation internally
            interp_levels = interp_levels_hpa.copy()

        n_levels = len(interp_levels)

        # ---- 7.3  Initialise result arrays ----------------------------
        # Arrays are pre-filled with NaN so that skipped time steps
        # (before start_date) appear as NaN rather than zero
        lat_centroid = np.full((n_timesteps, n_levels), np.nan)
        lon_centroid = np.full((n_timesteps, n_levels), np.nan)
        lat_minimum  = np.full((n_timesteps, n_levels), np.nan)
        lon_minimum  = np.full((n_timesteps, n_levels), np.nan)
        timestep_str = np.empty(n_timesteps, dtype="U21")   # 21-char string

        # One scalar per time step for each activated diagnostic variable
        diag_vars = {
            key: np.full(n_timesteps, np.nan)
            for key, enabled in export_variables.items() if enabled
        }

        # Running cyclone center estimate (updated each time step)
        slat, slon = s0lat, s0lon

        # ---- 7.4  Inner loop: one iteration per time step ------------
        tic = time.perf_counter()   # start the wall-clock timer

        for t in range(n_timesteps):

            # -- 7.4.1  Read the current timestamp and skip if too early --
            try:
                if model == "ICON":
                    ds_time = xr.open_dataset(ml_filelist[t])
                    step_date = (
                        pd.to_datetime(ds_time["time"].values)
                        .strftime("%d-%b-%Y %H:%M UTC")
                    )
                    # ICON returns an array; pick the single element
                    if isinstance(step_date, (list, np.ndarray)):
                        step_date = step_date[-1]
                else:  # WRF
                    step_date = (
                        pd.to_datetime(
                            getvar(Dataset(wrf_filelist[t]), "times").values
                        ).strftime("%d-%b-%Y %H:%M UTC")
                    )
            except Exception as exc:
                log.warning(
                    "Could not read timestamp from file %d/%d ('%s'). "
                    "Skipping this step.\n"
                    "Error type   : %s\n"
                    "Error detail : %s",
                    t + 1, n_timesteps,
                    wrf_filelist[t] if model == "WRF" else ml_filelist[t],
                    type(exc).__name__, exc
                )
                continue

            if pd.to_datetime(step_date) < pd.to_datetime(start_date):
                # Still before the requested start date – skip silently
                continue

            elapsed = time.perf_counter() - tic
            log.info(
                "  Sim %d/%d | Step %d/%d | %s | elapsed %.1f s",
                sim_idx + 1, len(sims), t + 1, n_timesteps,
                step_date, elapsed
            )

            # -- 7.4.2  Open model output files for this time step -------
            try:
                if model == "ICON":
                    ncfile_ml = xr.open_dataset(ml_filelist[t])
                    ncfile_ml = ncfile_ml.metpy.parse_cf().squeeze()
                    ncfile_pl = xr.open_dataset(pl_filelist[t])
                    ncfile_pl = ncfile_pl.metpy.parse_cf().squeeze()
                else:  # WRF
                    ncfile = Dataset(wrf_filelist[t])
            except Exception as exc:
                log.error(
                    "Could not open file for time step %d. Skipping.\n"
                    "Error type   : %s\n"
                    "Error detail : %s",
                    t + 1, type(exc).__name__, exc
                )
                continue

            # -- 7.4.3  Extract grid, SLP and geopotential ---------------
            try:
                if model == "ICON":
                    # ICON: unstructured grid, coordinates in radians
                    lons_grid = np.rad2deg(ncfile_pl["clon"].values)
                    lats_grid = np.rad2deg(ncfile_pl["clat"].values)
                    slp       = ncfile_ml["pres_msl"]        # SLP [Pa]
                    gph       = ncfile_pl["geopot"]          # geopotential [m²/s²]
                    gph_levels = np.array(gph[p_indices, :]) # shape (n_lev, n_pts)
                else:  # WRF
                    # WRF: structured 2-D grid
                    lons_grid = to_np(getvar(ncfile, "XLONG"))
                    lats_grid = to_np(getvar(ncfile, "XLAT"))
                    slp       = getvar(ncfile, "slp")        # SLP [hPa]
                    gph       = getvar(ncfile, "geopotential")
                    # Interpolate GPH onto the requested pressure levels
                    gph_levels = vinterp(
                        ncfile, gph, "pressure",
                        interp_levels[interp_levels != 0],   # skip SLP placeholder
                        extrapolate=True
                    )
            except Exception as exc:
                log.error(
                    "Variable extraction failed at time step %d. Skipping.\n"
                    "Error type   : %s\n"
                    "Error detail : %s\n%s",
                    t + 1, type(exc).__name__, exc, traceback.format_exc()
                )
                continue

            # Store the formatted timestamp string
            timestep_str[t] = step_date

            # -- 7.4.4  Compute haversine distances and build search mask -
            distances = haversine(lats_grid, lons_grid, slat, slon)
            mask = distances <= search_radius_km   # True inside the circle

            # -- 7.4.5  Locate the cyclone center at each pressure level --
            for lv in interp_levels:
                idx = int(np.argwhere(interp_levels == lv)[0, 0])

                if lv == 0:
                    # SLP level: use sea-level pressure field
                    z_field = slp
                else:
                    # Geopotential level
                    if model == "ICON":
                        z_field = gph_levels[idx, :].squeeze()
                    else:  # WRF
                        z_field = gph_levels.sel(interp_level=lv)

                try:
                    (lat_centroid[t, idx],
                     lon_centroid[t, idx],
                     lat_minimum[t, idx],
                     lon_minimum[t, idx]) = locate_center(
                         z_field, mask, lats_grid, lons_grid
                     )
                except Exception as exc:
                    log.warning(
                        "locate_center failed at t=%d, level=%s. "
                        "Filling with NaN.\n"
                        "Error type   : %s\n"
                        "Error detail : %s",
                        t + 1,
                        "SLP" if lv == 0 else f"{lv} Pa/hPa",
                        type(exc).__name__, exc
                    )
                    # Result arrays remain NaN from initialisation

            # -- 7.4.6  Update the cyclone center estimate ----------------
            # The new center is the mean of all centroid positions across
            # levels.  This makes the tracker more robust to outliers at
            # individual levels.
            slat = float(np.nanmean(lat_centroid[t, :]))
            slon = float(np.nanmean(lon_centroid[t, :]))

            if np.isnan(slat) or np.isnan(slon):
                # Fall back to the previous step's center if all levels failed
                log.warning(
                    "All levels returned NaN at t=%d. "
                    "Keeping previous center (%.2f°N, %.2f°E).",
                    t + 1, s0lat if t == 0 else slat, s0lon if t == 0 else slon
                )
                slat = lat_centroid[max(t - 1, 0), 0] if t > 0 else s0lat
                slon = lon_centroid[max(t - 1, 0), 0] if t > 0 else s0lon

            # Recalculate the mask centered on the updated position
            distances = haversine(lats_grid, lons_grid, slat, slon)
            mask = distances <= search_radius_km

            # -- 7.4.7  Extract diagnostic variables ---------------------
            try:
                _extract_diagnostics(
                    t, model, ncfile_ml if model == "ICON" else ncfile,
                    ncfile_pl if model == "ICON" else None,
                    ml_filelist if model == "ICON" else wrf_filelist,
                    mask, lats_grid, lons_grid,
                    export_variables, diag_vars,
                    plev_pa if model == "ICON" else None
                )
            except Exception as exc:
                log.warning(
                    "Diagnostic extraction partially failed at t=%d.\n"
                    "Error type   : %s\n"
                    "Error detail : %s",
                    t + 1, type(exc).__name__, exc
                )

        # ---- End of inner loop ----------------------------------------
        tac = time.perf_counter()
        total_time = tac - tic
        log.info(
            "Simulation '%s' completed in %.1f s (%.1f min).",
            sim, total_time, total_time / 60
        )

        # ---- 7.5  Save per-level CSV files (optional) -----------------
        if save_all_tracks:
            _save_per_level_tracks(
                outfolder, cyclone, sim,
                interp_levels, timestep_str,
                lat_centroid, lon_centroid,
                lat_minimum, lon_minimum
            )

        # ---- 7.6  Compute and save the weighted-mean track ------------
        mtrack_df = _compute_weighted_mean_track(
            outfolder, cyclone, sim,
            interp_levels, timestep_str,
            lat_centroid, lon_centroid,
            lat_minimum, lon_minimum,
            export_variables, diag_vars
        )

        # ---- 7.7  Generate plots (optional) ---------------------------
        if plot:
            _plot_tracks(
                outfolder, cyclone, model, sim,
                interp_levels, mtrack_df,
                lat_centroid, lon_centroid,
                lat_minimum, lon_minimum
            )

        if check_plots and plot:
            # Detailed per-timestep SLP check plots (slow – for debugging)
            _plot_slp_checks(
                outfolder, cyclone, model, sim,
                mtrack_df, interp_levels,
                lat_centroid, lon_centroid,
                lat_minimum, lon_minimum,
                ml_filelist if model == "ICON" else wrf_filelist,
                lats_grid, lons_grid,
                n_levels, search_radius_km,
                model
            )

    log.info("=" * 60)
    log.info("ALL SIMULATIONS COMPLETED.  That's all, folks!")


# =============================================================================
# SECTION 8 – DIAGNOSTIC VARIABLE EXTRACTION (model-specific)
# =============================================================================

def _extract_diagnostics(
    t, model, ncfile_ml_or_wrf, ncfile_pl,
    filelist, mask, lats, lons,
    export_variables, diag_vars,
    plev_pa=None
):
    """
    Extract all activated diagnostic variables at time step *t* and store them
    in *diag_vars*.

    Parameters
    ----------
    t                : int         current time index
    model            : str         "ICON" or "WRF"
    ncfile_ml_or_wrf : dataset     ICON model-level xarray.Dataset or WRF Dataset
    ncfile_pl        : dataset     ICON pressure-level xarray.Dataset (None for WRF)
    filelist         : list        file paths (to access previous time step for rain)
    mask             : np.ndarray  boolean search-circle mask
    lats, lons       : np.ndarray  grid coordinates
    export_variables : dict        {name: bool}
    diag_vars        : dict        {name: np.ndarray}  (modified in-place)
    plev_pa          : np.ndarray  ICON pressure levels in Pa (None for WRF)
    """

    # --- Minimum SLP --------------------------------------------------------
    if export_variables.get("min_slp"):
        if model == "ICON":
            slp_hpa = np.asarray(ncfile_ml_or_wrf["pres_msl"]) / 100.0   # Pa → hPa
        else:
            slp_hpa = to_np(getvar(ncfile_ml_or_wrf, "slp"))              # already hPa
        diag_vars["min_slp"][t] = np.nanmin(np.where(mask, slp_hpa, np.nan))

    # --- Sea-Surface Temperature --------------------------------------------
    sst_needed = (
        export_variables.get("max_sst") or export_variables.get("mean_sst")
    )
    if sst_needed:
        if model == "ICON":
            # SST variable name may differ – warn if not found
            if "sst" in ncfile_ml_or_wrf:
                sst = np.asarray(ncfile_ml_or_wrf["sst"])
            else:
                warnings.warn(
                    "SST variable ('sst') not found in ICON model-level file. "
                    "Skipping SST diagnostics. "
                    f"[line {inspect.currentframe().f_back.f_lineno}]",
                    UserWarning, stacklevel=3
                )
                sst = None
        else:  # WRF
            sst = to_np(getvar(ncfile_ml_or_wrf, "SST"))

        if sst is not None:
            sst_masked = np.where(mask, sst, np.nan)
            if export_variables.get("max_sst"):
                diag_vars["max_sst"][t]  = np.nanmax(sst_masked)
            if export_variables.get("mean_sst"):
                diag_vars["mean_sst"][t] = np.nanmean(sst_masked)

    # --- 10-m Wind Speed ----------------------------------------------------
    if export_variables.get("max_wind10m"):
        if model == "ICON":
            u10 = np.asarray(ncfile_ml_or_wrf["u_10m"])
            v10 = np.asarray(ncfile_ml_or_wrf["v_10m"])
            wind10m = np.sqrt(u10**2 + v10**2)
        else:  # WRF
            wind10m = to_np(getvar(ncfile_ml_or_wrf, "uvmet10_wspd"))
        diag_vars["max_wind10m"][t] = np.nanmax(np.where(mask, wind10m, np.nan))

    # --- Latent Heat Flux ---------------------------------------------------
    lhf_needed = (
        export_variables.get("max_lhf") or export_variables.get("mean_lhf")
    )
    if lhf_needed:
        if model == "ICON":
            lhf = np.asarray(ncfile_ml_or_wrf["lhfl_s"])
        else:  # WRF
            lhf = to_np(getvar(ncfile_ml_or_wrf, "LH"))
        lhf_masked = np.where(mask, lhf, np.nan)
        if export_variables.get("max_lhf"):
            diag_vars["max_lhf"][t]  = np.nanmax(lhf_masked)
        if export_variables.get("mean_lhf"):
            diag_vars["mean_lhf"][t] = np.nanmean(lhf_masked)

    # --- Sensible Heat Flux -------------------------------------------------
    shf_needed = (
        export_variables.get("max_shf") or export_variables.get("mean_shf")
    )
    if shf_needed:
        if model == "ICON":
            shf = np.asarray(ncfile_ml_or_wrf["shfl_s"])
        else:  # WRF
            shf = to_np(getvar(ncfile_ml_or_wrf, "HFX"))
        shf_masked = np.where(mask, shf, np.nan)
        if export_variables.get("max_shf"):
            diag_vars["max_shf"][t]  = np.nanmax(shf_masked)
        if export_variables.get("mean_shf"):
            diag_vars["mean_shf"][t] = np.nanmean(shf_masked)

    # --- Water Vapour Flux --------------------------------------------------
    qvf_needed = (
        export_variables.get("max_qvf") or export_variables.get("mean_qvf")
    )
    if qvf_needed:
        if model == "ICON":
            if "qvf" in ncfile_ml_or_wrf:
                qvf = np.asarray(ncfile_ml_or_wrf["qvf"])
            else:
                warnings.warn(
                    "QVF variable not found in ICON output. "
                    "Skipping QVF diagnostics. "
                    f"[line {inspect.currentframe().f_back.f_lineno}]",
                    UserWarning, stacklevel=3
                )
                qvf = None
        else:  # WRF
            qvf = to_np(getvar(ncfile_ml_or_wrf, "QFX"))

        if qvf is not None:
            qvf_masked = np.where(mask, qvf, np.nan)
            if export_variables.get("max_qvf"):
                diag_vars["max_qvf"][t]  = np.nanmax(qvf_masked)
            if export_variables.get("mean_qvf"):
                diag_vars["mean_qvf"][t] = np.nanmean(qvf_masked)

    # --- Precipitable Water -------------------------------------------------
    if export_variables.get("mean_pw"):
        if model == "ICON":
            warnings.warn(
                "Precipitable water (mean_pw) for ICON requires expensive "
                "vertical integration and is disabled by default. "
                "Set export_variables['mean_pw'] = False to suppress this warning.",
                UserWarning, stacklevel=3
            )
        else:  # WRF
            pw = to_np(getvar(ncfile_ml_or_wrf, "pw"))
            diag_vars["mean_pw"][t] = np.nanmean(np.where(mask, pw, np.nan))

    # --- Potential Vorticity at 300 hPa ------------------------------------
    if export_variables.get("max_pvo"):
        if model == "ICON":
            # Find the index of the level closest to 300 hPa
            idx300 = int(np.argmin(np.abs(plev_pa - 300 * 100)))
            pvo = np.asarray(ncfile_pl["pv"][idx300, :]).squeeze()
        else:  # WRF
            pvo_3d   = getvar(ncfile_ml_or_wrf, "pvo")
            pvo_interp = vinterp(
                ncfile_ml_or_wrf, pvo_3d, "pressure", [300], extrapolate=True
            )
            pvo = to_np(pvo_interp.squeeze())
        diag_vars["max_pvo"][t] = np.nanmax(np.where(mask, pvo, np.nan))

    # --- 2-m Relative Humidity ----------------------------------------------
    if export_variables.get("max_rh"):
        if model == "ICON":
            rh = np.asarray(ncfile_ml_or_wrf["rh_2m"])
        else:  # WRF
            rh = to_np(getvar(ncfile_ml_or_wrf, "rh2"))
        diag_vars["max_rh"][t] = np.nanmax(np.where(mask, rh, np.nan))

    # --- Hourly Accumulated Rainfall (current – previous step) -------------
    rain_needed = (
        export_variables.get("max_rain") or export_variables.get("mean_rain")
    )
    if rain_needed:
        if t == 0:
            log.warning(
                "Rainfall accumulation cannot be computed for t=0 "
                "(no previous time step). Filling with NaN."
            )
        else:
            try:
                if model == "ICON":
                    nc_prev    = xr.open_dataset(filelist[t - 1])
                    nc_prev    = nc_prev.metpy.parse_cf().squeeze()
                    nc_curr    = ncfile_ml_or_wrf
                    rain = (
                        np.asarray(nc_curr["rain_gsp"])
                        - np.asarray(nc_prev["rain_gsp"])
                    )
                else:  # WRF
                    nc_prev = Dataset(filelist[t - 1])
                    rain = (
                        to_np(getvar(ncfile_ml_or_wrf, "RAINNC"))
                        - to_np(getvar(nc_prev, "RAINNC"))
                    )
                rain_masked = np.where(mask, rain, np.nan)
                if export_variables.get("max_rain"):
                    diag_vars["max_rain"][t]  = np.nanmax(rain_masked)
                if export_variables.get("mean_rain"):
                    diag_vars["mean_rain"][t] = np.nanmean(rain_masked)
            except Exception as exc:
                log.warning(
                    "Rainfall computation failed at t=%d. Filling with NaN.\n"
                    "Error type   : %s\n"
                    "Error detail : %s",
                    t + 1, type(exc).__name__, exc
                )


# =============================================================================
# SECTION 9 – TRACK AVERAGING AND OUTPUT
# =============================================================================

def _save_per_level_tracks(
    outfolder, cyclone, sim, interp_levels, timestep_str,
    lat_centroid, lon_centroid, lat_minimum, lon_minimum
):
    """Save separate CSV files for each pressure level (centroid and minimum)."""
    log.info("Saving per-level track files ...")

    for lv in interp_levels:
        idx = int(np.argwhere(interp_levels == lv)[0, 0])
        level_label = "slp" if lv == 0 else f"z{int(lv)}"

        for track_type, lat_arr, lon_arr in [
            ("centroid", lat_centroid, lon_centroid),
            ("minimum",  lat_minimum,  lon_minimum),
        ]:
            fname = os.path.join(
                outfolder, f"{cyclone}_{sim}_track_{level_label}_{track_type}.csv"
            )
            df = pd.DataFrame({
                "date": timestep_str,
                "lat":  lat_arr[:, idx],
                "lon":  lon_arr[:, idx],
            })
            df.to_csv(fname, sep=",", index=False)
            log.info("    Saved: %s", fname)

    # Also save the full multi-level array in a single wide CSV
    fname_multi = os.path.join(
        outfolder, f"{cyclone}_{sim}_track_multilevel_all.csv"
    )
    col_parts = (
        [("centroid", "lat", lat_centroid),
         ("centroid", "lon", lon_centroid),
         ("minimum",  "lat", lat_minimum),
         ("minimum",  "lon", lon_minimum)]
    )
    multi_dict = {"date": timestep_str}
    for track_type, coord, arr in col_parts:
        for lv in interp_levels:
            idx = int(np.argwhere(interp_levels == lv)[0, 0])
            level_label = "slp" if lv == 0 else f"z{int(lv)}"
            col_name = f"{coord}_{track_type}_{level_label}"
            multi_dict[col_name] = arr[:, idx]
    pd.DataFrame(multi_dict).to_csv(fname_multi, sep=",", index=False)
    log.info("  Multi-level CSV saved: %s", fname_multi)


def _compute_weighted_mean_track(
    outfolder, cyclone, sim, interp_levels, timestep_str,
    lat_centroid, lon_centroid, lat_minimum, lon_minimum,
    export_variables, diag_vars
):
    """
    Compute a weighted-mean track by averaging centroid and minimum positions
    across all pressure levels, then save to CSV.

    Currently all levels/methods receive equal weight.  To assign higher
    weight to specific levels (e.g. the SLP minimum), edit the *weightss*
    array below.

    Returns
    -------
    pd.DataFrame
        The weighted-mean track dataframe (also saved to CSV).
    """
    log.info("Computing weighted-mean track ...")

    n_levels = len(interp_levels)

    # Stack centroid and minimum latitude/longitude arrays side by side
    # Shape: (n_timesteps, 2 * n_levels)
    lats_all = np.concatenate((lat_centroid, lat_minimum), axis=1)
    lons_all = np.concatenate((lon_centroid, lon_minimum), axis=1)

    # Define weights (all equal by default; customise here if needed)
    weightss = np.ones(2 * n_levels)
    # Example: give double weight to centroid levels:
    #   weightss[:n_levels] = 2
    # Example: give triple weight to the SLP minimum:
    #   idx_slp = int(np.argwhere(interp_levels == 0)[0, 0])
    #   weightss[n_levels + idx_slp] = 3

    weights = weightss / np.sum(weightss)   # normalise so weights sum to 1

    # Weighted mean latitude and longitude at each time step
    mlat = np.dot(lats_all, weights.reshape(-1, 1)).flatten()   # shape (n_t,)
    mlon = np.dot(lons_all, weights.reshape(-1, 1)).flatten()

    # Build the output dictionary
    mtrack_dict = {
        "date": timestep_str,
        "lat":  mlat,
        "lon":  mlon,
    }
    for key, enabled in export_variables.items():
        if enabled and key in diag_vars:
            mtrack_dict[key] = diag_vars[key]

    mtrack_df = pd.DataFrame(mtrack_dict)
    # Drop rows where lat or lon is NaN (time steps that were fully skipped)
    mtrack_df = mtrack_df.dropna(subset=["lat", "lon"])
    mtrack_df = mtrack_df.reset_index(drop=True)

    fname = os.path.join(
        outfolder, f"{cyclone}_{sim}_track_multivarz_wm.csv"
    )
    mtrack_df.to_csv(fname, sep=",", index=False)
    log.info("  Weighted-mean track saved: %s", fname)
    log.info("  Track spans %d time steps.", len(mtrack_df))

    return mtrack_df


# =============================================================================
# SECTION 10 – PLOTTING
# =============================================================================

def _plot_tracks(
    outfolder, cyclone, model, sim,
    interp_levels, mtrack_df,
    lat_centroid, lon_centroid,
    lat_minimum, lon_minimum
):
    """
    Generate two PNG plots:
      1. All-levels scatter plot showing centroid and minimum positions.
      2. Weighted-mean track coloured by minimum SLP.
    """
    log.info("Generating track plots for simulation '%s' ...", sim)

    n_levels = len(interp_levels)

    # Rolling-window smoothing (window=3) for the mean track line
    rm_lon = pd.Series(mtrack_df["lon"]).rolling(window=3, min_periods=1).mean().to_numpy()
    rm_lat = pd.Series(mtrack_df["lat"]).rolling(window=3, min_periods=1).mean().to_numpy()

    # Domain extent with a 1° margin
    all_lats = np.concatenate([lat_centroid.ravel(), lat_minimum.ravel()])
    all_lons = np.concatenate([lon_centroid.ravel(), lon_minimum.ravel()])
    min_lon = np.round(np.nanmin(all_lons)) - 1
    max_lon = np.round(np.nanmax(all_lons)) + 1
    min_lat = np.round(np.nanmin(all_lats)) - 1
    max_lat = np.round(np.nanmax(all_lats)) + 1

    # Spectral colour map – one colour per pressure level
    cmap       = plt.get_cmap("nipy_spectral")
    levcolors  = np.flipud(
        [(*cmap(i / max(n_levels - 1, 1))[:3], 0.75) for i in range(n_levels)]
    )
    proj = ccrs.PlateCarree()

    # ----- Plot 1: all levels ----------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 10), subplot_kw={"projection": proj})
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=proj)
    ax.coastlines(resolution="10m", color="black", linewidth=2)

    for lv in interp_levels:
        idx = int(np.argwhere(interp_levels == lv)[0, 0])
        label = "slp" if lv == 0 else (
            f"z{lv / 100:.0f} hPa" if model == "ICON" else f"z{lv} hPa"
        )
        lcolor = levcolors[idx]
        ax.scatter(
            lon_centroid[:, idx], lat_centroid[:, idx],
            transform=proj, color=lcolor, marker="o", s=25,
            label=f"centroid {label}", zorder=20
        )
        ax.scatter(
            lon_minimum[:, idx], lat_minimum[:, idx],
            transform=proj, color=lcolor, marker="x", s=25,
            label=f"minimum {label}", zorder=21
        )

    ax.plot(rm_lon, rm_lat, transform=proj, color="#ee4400ff",
            linewidth=2, label="Mean track", zorder=30)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(
        f"Multi-level cyclone tracking – {cyclone} | {model} {sim}",
        fontsize=14, fontweight="bold"
    )
    figname = os.path.join(outfolder, f"{cyclone}_{model}_{sim}_track_all_levels.png")
    plt.savefig(figname, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    log.info("  Saved: %s", figname)

    # ----- Plot 2: weighted-mean track + SLP colour coding -----------------
    if "min_slp" not in mtrack_df.columns or mtrack_df["min_slp"].isna().all():
        log.warning(
            "min_slp column not available or all NaN – skipping SLP track plot."
        )
        return

    fig, ax = plt.subplots(figsize=(18, 10), subplot_kw={"projection": proj})
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=proj)
    ax.gridlines(
        draw_labels=True, linewidth=0.5, color="#777777",
        linestyle="--", x_inline=False, y_inline=False
    )
    ax.add_feature(
        NaturalEarthFeature("physical", "ocean", "10m", facecolor=COLORS["water"])
    )
    ax.add_feature(
        NaturalEarthFeature("physical", "land",  "10m", facecolor=COLORS["land"])
    )

    scatter = ax.scatter(
        rm_lon, rm_lat, transform=proj,
        c=mtrack_df["min_slp"], s=80,
        cmap="hot", edgecolor="black", linewidth=0.1,
        label="Track (MSLP)", zorder=30
    )
    ax.plot(rm_lon, rm_lat, transform=proj, color="#ee4400ff",
            linewidth=2, label="Track", zorder=25)

    cbar_min = int(np.floor(mtrack_df["min_slp"].min()))
    cbar_max = int(np.ceil(mtrack_df["min_slp"].max()))
    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical")
    cbar.set_ticks(np.arange(cbar_min, cbar_max + 1, dtype=np.int32))
    cbar.set_label("Min SLP [hPa]", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(
        f"{cyclone} – {model} {sim} – Weighted-mean track and min SLP",
        fontsize=14, fontweight="bold"
    )
    figname = os.path.join(outfolder, f"{cyclone}_{model}_{sim}_track_slp.png")
    plt.savefig(figname, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    log.info("  Saved: %s", figname)


def _plot_slp_checks(
    outfolder, cyclone, model, sim,
    mtrack_df, interp_levels,
    lat_centroid, lon_centroid,
    lat_minimum, lon_minimum,
    filelist, lats_grid, lons_grid,
    n_levels, search_radius_km,
    model_str
):
    """
    Generate per-timestep SLP contour maps to visually verify that the tracker
    has found the correct cyclone center.  Output goes to outfolder/slp_check/.
    This function is slow and should only be used for debugging.
    """
    check_folder = os.path.join(outfolder, "slp_check")
    check_and_create_folder(check_folder)

    n_steps = len(mtrack_df)
    cmap = plt.get_cmap("nipy_spectral")
    levcolors = np.flipud(
        [(*cmap(i / max(n_levels - 1, 1))[:3], 0.9) for i in range(n_levels)]
    )
    proj = ccrs.PlateCarree()

    log.info("Generating %d SLP check plots ...", n_steps)

    for row_idx, row in mtrack_df.iterrows():
        t = row_idx   # time index in the original arrays

        log.info("  Check plot %d/%d", t + 1, n_steps)

        # Compute the search mask centered on the weighted-mean position
        distances = haversine(lats_grid, lons_grid, row["lat"], row["lon"])
        mask = distances <= search_radius_km

        # Read SLP for this time step
        try:
            if model_str == "ICON":
                nc_ml = xr.open_dataset(filelist[t])
                slp_hpa = np.asarray(nc_ml["pres_msl"]).squeeze() / 100.0
            else:  # WRF
                ncfile = Dataset(filelist[t])
                slp_hpa = to_np(getvar(ncfile, "slp"))
        except Exception as exc:
            log.warning("Could not read SLP for check plot %d: %s", t + 1, exc)
            continue

        slp_masked = np.where(mask, slp_hpa, np.nan)

        # Map extent from the mask bounds
        lon_min_local = np.nanmin(np.where(mask, lons_grid, np.nan))
        lon_max_local = np.nanmax(np.where(mask, lons_grid, np.nan))
        lat_min_local = np.nanmin(np.where(mask, lats_grid, np.nan))
        lat_max_local = np.nanmax(np.where(mask, lats_grid, np.nan))

        fig, ax = plt.subplots(figsize=(18, 10), subplot_kw={"projection": proj})
        ax.set_extent(
            [lon_min_local, lon_max_local, lat_min_local, lat_max_local],
            crs=proj
        )

        # Contour fill – use tricontourf for ICON (unstructured) and
        # pcolormesh for WRF (structured)
        if model_str == "ICON":
            contour = ax.tricontourf(
                lons_grid, lats_grid, slp_masked, 20,
                transform=proj, cmap="pink", zorder=10
            )
        else:
            contour = ax.contourf(
                lons_grid, lats_grid, slp_masked, 20,
                transform=proj, cmap="pink", zorder=10
            )

        cbar = plt.colorbar(
            contour, ax=ax, orientation="horizontal",
            pad=0.05, aspect=20, fraction=0.05
        )
        cbar.set_label("SLP [hPa]", fontsize=12)
        ax.coastlines(resolution="10m", color="black", linewidth=2)

        # Scatter: centroid (circle) and minimum (cross) at each level
        for lv in interp_levels:
            idx = int(np.argwhere(interp_levels == lv)[0, 0])
            label = "slp" if lv == 0 else f"z{lv}"
            lcolor = levcolors[idx]
            ax.scatter(
                lon_centroid[t, idx], lat_centroid[t, idx],
                transform=proj, color=lcolor, marker="o", s=60,
                label=f"centroid {label}", zorder=20
            )
            ax.scatter(
                lon_minimum[t, idx], lat_minimum[t, idx],
                transform=proj, color=lcolor, marker="x", s=60,
                label=f"minimum {label}", zorder=21
            )
        ax.scatter(
            row["lon"], row["lat"],
            transform=proj, color="#ee4400ee", marker="*", s=80,
            label="Weighted-mean track", zorder=22
        )
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
        ax.set_title(
            f"SLP check – {row['date']} – {cyclone} {model_str} {sim}",
            fontsize=14, fontweight="bold"
        )

        figname = os.path.join(
            check_folder, f"{cyclone}_{sim}_check_slp_{t:04d}.png"
        )
        plt.savefig(figname, dpi=200, bbox_inches="tight", format="png")
        plt.close(fig)

    log.info("SLP check plots saved to: %s", check_folder)


# =============================================================================
# SECTION 11 – ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    log.info("cyclone_tracker_multilevel.py – starting")
    log.info("Python version : %s", sys.version)
    log.info("NumPy version  : %s", np.__version__)
    log.info("Pandas version : %s", pd.__version__)

    # Validate user inputs before doing any heavy I/O
    try:
        validate_inputs(INFOLDERS, SIMS, OUTFOLDER, WRF_PREFIXES)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        log.error(
            "Input validation failed – fix the USER CONFIGURATION section "
            "and re-run.\nError type   : %s\nError detail : %s",
            type(exc).__name__, exc
        )
        sys.exit(1)

    # Run the main tracking loop
    # Any unhandled exception here will print a full traceback before exiting
    try:
        run_tracking(
            model            = MODEL,
            cyclone          = CYCLONE,
            sims             = SIMS,
            infolders        = INFOLDERS,
            outfolder        = OUTFOLDER,
            start_date       = START_DATE,
            s0lat            = S0LAT,
            s0lon            = S0LON,
            search_radius_km = SEARCH_RADIUS_KM,
            interp_levels_hpa= INTERP_LEVELS_HPA,
            export_variables = EXPORT_VARIABLES,
            save_all_tracks  = SAVE_ALL_TRACKS,
            plot             = PLOT,
            check_plots      = CHECK_PLOTS,
            wrf_prefixes     = WRF_PREFIXES,
        )
    except KeyboardInterrupt:
        log.warning("Execution interrupted by the user (Ctrl+C).")
        sys.exit(0)
    except Exception as exc:
        log.error(
            "Unexpected fatal error.\n"
            "Error type   : %s\n"
            "Error detail : %s\n"
            "Full traceback:\n%s",
            type(exc).__name__, exc, traceback.format_exc()
        )
        sys.exit(1)
