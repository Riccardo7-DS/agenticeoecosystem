## Methods
- Contextual active-fire detection with mid-infrared (~3.9 um) and thermal (~11 um) channels.
- Fire radiative power (FRP) retrieval.
- Land and sea surface temperature, and thermal anomalies (volcanic, industrial, urban heat).
- Timing analysis: first detection versus true ignition, bounded by satellite overpass times.

## Data facts
- Polar orbiters:
  - VIIRS: 375 m I-bands, about 4 overpasses a day across S-NPP, NOAA-20 and NOAA-21.
  - MODIS: 1 km fire product on Terra and Aqua, satellites drifting at end of life.
  - Sentinel-3 SLSTR: 1 km, day and night.
- Geostationary imagers (MTG FCI, MSG SEVIRI, GOES ABI, Himawari AHI) observe every 10-15
  minutes at 2-3 km. They only work well within roughly 60-65 degrees of the sub-satellite
  point. Check the viewing zenith angle for the study area.
- Thermal sensors observe day and night, but not through thick cloud.

## Pitfalls
- Overpass gaps: a first detection is only an upper bound on ignition time.
- Small or cool fires fall below the detection limit, especially at geostationary
  resolution.
- False alarms from sun glint, hot bare soil and industrial sources.
- Smoke and cloud obscure the fire. Detection probability depends on viewing geometry.
- FRP is underestimated at large viewing angles and for partly obscured fires.

## Validation practice
- Compare against higher-resolution detections (VIIRS 375 m, Landsat or Sentinel-2 SWIR)
  or ground reports.
- Report detection latency against a reference time, and omission and commission rates.
