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
    https://doi.org/10.5281/zenodo.19695732  <-- replace after upload

"""

