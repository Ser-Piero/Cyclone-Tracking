# Cyclone-Tracking
Multi-level cyclone tracking tool for ICON and WRF nwp model output.  The algorithm tracks a cyclone center by locating the minimum and centroid of geopotential height and sea-level pressure within a search radius at multiple pressure levels.  A weighted-mean track is then derived from all levels and optionally enriched with diagnostic variables.
