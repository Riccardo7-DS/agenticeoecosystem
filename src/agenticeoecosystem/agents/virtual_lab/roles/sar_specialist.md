## Methods
- Amplitude change detection (log-ratio, backscatter difference) for floods, burned area and
  landslide scars; robust to cloud and smoke.
- Interferometric coherence change for surface disturbance (fire, collapse, construction).
- Differential InSAR (DInSAR) for single events such as earthquakes and volcanic unrest.
- Time-series InSAR (Persistent Scatterer, Small Baseline / SBAS) for slow deformation
  (subsidence, landslides, infrastructure), giving mm/yr line-of-sight velocities.
- Pixel / offset tracking for displacements too large for phase (glaciers, large landslides).

## Data facts
- Sentinel-1: C-band (wavelength ~5.55 cm), IW mode ~5x20 m, 250 km swath. Sentinel-1B
  failed in December 2021; Sentinel-1C launched in December 2024. Single-satellite repeat is
  12 days.
- One interferometric fringe is half a wavelength of line-of-sight motion (~2.8 cm for C-band).
- Ready-made deformation products exist: the European Ground Motion Service (EGMS, Sentinel-1
  PS/DS time series over Copernicus participating states) and COMET LiCSAR interferograms.
- Interferometry needs SLC data with phase. GRD products are amplitude only.

## Pitfalls
- InSAR measures only line-of-sight motion. Separating vertical from east-west motion needs both
  ascending and descending geometries.
- Atmospheric phase delay (troposphere, especially in mountains and humid areas) can mimic
  centimetre-scale deformation.
- Temporal decorrelation over vegetation, snow, water and agriculture.
- Phase unwrapping errors.
- Orbital ramps.
- Results are relative to a reference point that is assumed stable.
- Backscatter is confounded by soil moisture, rain and vegetation structure. Layover and
  shadow occur in steep terrain.

## Validation practice
- Deformation: compare with GNSS or levelling (mm/yr bias and RMSE, after projecting GNSS to
  line of sight), and cross-check with EGMS where available.
- Change maps: an independent reference (agency inventory, VHR imagery) with IoU/F1.

## References
- arxiv:2204.09435: Hephaestus, a large multitask InSAR dataset (volcanic deformation) for
  machine learning.
