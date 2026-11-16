## Methods
- Reanalysis-based context: ERA5 (hourly, ~31 km) for wind, temperature, humidity,
  precipitation and soil moisture, and ERA5-Land (~9 km) for land variables.
- Satellite precipitation: GPM IMERG (0.1°, half-hourly; NASA Earthdata) and EUMETSAT H SAF /
  geostationary (MSG SEVIRI, MTG FCI) rain-rate products for near-real-time, high-revisit
  estimates. Use ERA5 precipitation for long, consistent records.
- Fire weather indices (e.g. the Canadian FWI) and drought indicators.
- Trace gases and aerosols: Sentinel-5P TROPOMI (CO, NO2, aerosol index) for smoke and
  pollution plumes. Use transport and trajectory reasoning for source attribution.
- Atmospheric corrections for other sensors, e.g. the tropospheric delay in InSAR from ERA5
  or GNSS zenith delay.

## Pitfalls
- Reanalysis is a model product: it smooths extremes and local terrain effects at its grid
  scale.
- Satellite precipitation is indirect (microwave and infrared retrievals): it misses or
  misplaces orographic and short convective rain, and errors grow over snow, coasts and
  complex terrain. Infrared-only rain rates infer rain from cloud-top temperature.
- Column-integrated trace gases do not give surface concentrations or plume height.
- Weather explains conditions, not causes. A favourable fire weather index is not evidence
  of ignition.

## Validation practice
- Compare reanalysis against station observations where available.
- Validate satellite precipitation against rain gauges or ground radar (bias, RMSE,
  detection scores such as POD/FAR), at matched space and time scales.
- Cross-check satellite trace gases with ground monitors or a second sensor.
