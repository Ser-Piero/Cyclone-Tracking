# Cyclone-Tracking

Multi-level cyclone tracking tool for ICON and WRF NWP model output.

The repository includes a Python implementation (`cyclone_tracking.py`) that tracks a cyclone center by locating minima/centroids from geopotential height and sea-level pressure inside a search radius at multiple pressure levels, then combines levels into a weighted-mean track.

## Citation and DOI

This repository includes `CITATION.cff` and `.zenodo.json` metadata so it can be cited in scientific articles and archived with a DOI through Zenodo/GitHub integration.

After DOI publication, cite the software using the DOI assigned to the release.
