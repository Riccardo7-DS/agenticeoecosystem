## Methods
- Spectral indices: NDVI and EVI for vegetation, NDWI and MNDWI for water, NBR for burned
  area.
- Bi-temporal change: dNBR / RdNBR for burn severity, index differencing and change vector
  analysis.
- Classification and segmentation of land cover, burned area, landslide scars and flood
  extent.
- Time-series analysis (e.g. break detection) for gradual change.

## Data facts
- Sentinel-2 MSI: 10/20/60 m, 13 bands, 5-day revisit with two satellites. Level-2A is surface
  reflectance.
- Landsat 8/9 OLI/TIRS: 30 m reflective, 100 m thermal, 8-day combined revisit. Its archive
  goes back to 1972 for long baselines.
- Harmonized Landsat-Sentinel (HLS) gives a common 30 m surface-reflectance product.
- All reflective sensors observe in daytime only and are blocked by cloud and thick smoke.

## Pitfalls
- Cloud, cloud shadow, haze and smoke contamination. Cloud masks such as SCL or Fmask miss
  thin cloud and smoke.
- Phenology and illumination differences between pre- and post-event scenes. Match the season
  and the sun angle.
- Topographic shading in mountains.
- Geometric misregistration between dates creates false change at edges.
- Fixed thresholds (e.g. dNBR classes) are calibrated on specific ecosystems and do not
  transfer without checks.
- Optical imagery does not measure ground deformation directly. It sees only its surface
  expression (scarps, cracks, vegetation loss).

## Validation practice
- An independent reference: Copernicus EMS or agency delineations, VHR imagery, field data.
- Report IoU/F1 and omission/commission errors, and a confusion matrix with area-adjusted
  accuracy where possible.
