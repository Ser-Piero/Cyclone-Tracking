# cyclone_tracker_multilevel

**Multi-level cyclone tracking tool for ICON and WRF mesoscale model output**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1217993590.svg)](https://doi.org/10.5281/zenodo.19695732)

---

## What it does

`cyclone_tracker_multilevel.py` tracks a Mediterranean (or other basin) cyclone
through a sequence of model output files.  At each time step it:

1. Searches within a configurable radius around the previous center.
2. Finds the **centroid** (convex-hull centre of the lowest 5th-percentile
   points) and the **absolute minimum** of geopotential height (or SLP) at
   multiple pressure levels simultaneously.
3. Combines all level-estimates into a single **weighted-mean track**.
4. Optionally extracts diagnostic variables along the track (SLP, SST, winds,
   heat fluxes, precipitation, PV, …).
5. Saves results as CSV files and optionally plots track maps.

Supported model back-ends: **ICON** (xarray + MetPy) and **WRF** (netCDF4 +
wrf-python).  Switch between them by changing a single line.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/cyclone_tracker_multilevel.git
cd cyclone_tracker_multilevel

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
#    For ICON:
pip install xarray metpy cartopy shapely scipy pandas matplotlib

#    For WRF (add to the above):
pip install netCDF4 wrf-python
```

---

## Quick start

1. Open `cyclone_tracker_multilevel.py`.
2. In **SECTION 2 – USER CONFIGURATION** (around line 90), set:
   - `MODEL = "ICON"` or `"WRF"`
   - `INFOLDERS`, `SIMS`, `OUTFOLDER`
   - `START_DATE`, `S0LAT`, `S0LON` (initial cyclone position)
   - `SEARCH_RADIUS_KM` (150 km for Mediterranean, 300 km for Atlantic)
   - Toggle `EXPORT_VARIABLES` flags as needed
3. Run: `python cyclone_tracker_multilevel.py`

---

## Output files

| File | Description |
|---|---|
| `<CYCLONE>_<SIM>_track_multivarz_wm.csv` | Weighted-mean track + diagnostics |
| `<CYCLONE>_<SIM>_track_all_levels.png`   | All-level scatter plot (if `PLOT=True`) |
| `<CYCLONE>_<SIM>_track_slp.png`          | Mean track coloured by min SLP |
| `<CYCLONE>_<SIM>_track_<level>_centroid.csv` | Per-level centroid (if `SAVE_ALL_TRACKS=True`) |
| `slp_check/<CYCLONE>_<SIM>_check_slp_NNNN.png` | Per-timestep debug maps (if `CHECK_PLOTS=True`) |

---

## Citation

If you use this code in a publication, please cite it as:

> Serafini P. (2026). *cyclone_tracker_multilevel – Multi-level cyclone
> tracking for ICON and WRF*. Zenodo.
> https://doi.org/10.5281/zenodo.19695732

---

## License

MIT – see [LICENSE](LICENSE).

---

## Authors

- Piero Serafini - PhD student in Atmospheric Physics - University of L'Aquila (UNIVAQ) - Center of Excellence in Telesensing of Environment and Model Prediction of Severe Events (CETEMPS) - Via Vetoio, Edificio Renato Ricamo, L'Aquila (AQ), Italy, 67100 - piero.serafini@graduate.univaq.it
