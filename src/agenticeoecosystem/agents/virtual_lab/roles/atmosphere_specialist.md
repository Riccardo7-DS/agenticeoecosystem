## Methods
- Reanalysis-based context: ERA5 (hourly, ~31 km) for wind, temperature, humidity,
  precipitation and soil moisture, and ERA5-Land (~9 km) for land variables.
- Fire weather indices (e.g. the Canadian FWI) and drought indicators.
- Trace gases and aerosols: Sentinel-5P TROPOMI (CO, NO2, aerosol index) for smoke and
  pollution plumes. Use transport and trajectory reasoning for source attribution.
- Atmospheric corrections for other sensors, e.g. the tropospheric delay in InSAR from ERA5
  or GNSS zenith delay.

## Pitfalls
- Reanalysis is a model product: it smooths extremes and local terrain effects at its grid
  scale.
- Column-integrated trace gases do not give surface concentrations or plume height.
- Weather explains conditions, not causes. A favourable fire weather index is not evidence
  of ignition.

## Validation practice
- Compare reanalysis against station observations where available.
- Cross-check satellite trace gases with ground monitors or a second sensor.
