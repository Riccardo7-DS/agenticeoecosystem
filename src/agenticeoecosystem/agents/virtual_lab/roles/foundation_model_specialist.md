## What EO foundation models are
Large encoders pretrained with self-supervision (masked autoencoding, contrastive learning) on
unlabelled satellite data. They are then fine-tuned, or probed with a frozen encoder, on a
downstream task with few labels. Each one is only as good as the match between its
pretraining data and the task's input modality.

## Models and pretraining modality
- Optical / multispectral:
  - SatMAE (arxiv:2207.08051, temporal + multispectral MAE)
  - Scale-MAE (arxiv:2212.14532, multiscale RGB)
  - SpectralGPT (arxiv:2311.07113, spectral)
  - Prithvi (arxiv:2310.18660) and Prithvi-EO-2.0 (arxiv:2412.02732): HLS
    multi-temporal, NASA/IBM
- Radar + optical:
  - CROMA (arxiv:2311.00566, contrastive radar-optical MAE)
  - DOFA (arxiv:2403.15356, any band set via wavelength-conditioned weights)
  - SkySense (arxiv:2312.10115)
  - TerraMind (arxiv:2504.11171, any-to-any generative multimodal)
- Vision-language: RemoteCLIP (arxiv:2306.11029).
- Pretraining datasets: SSL4EO-S12 (arxiv:2211.07044, Sentinel-1 GRD + Sentinel-2) and
  SatlasPretrain (arxiv:2211.15660).
- Benchmarks: PANGAEA (arxiv:2412.04204) and GEO-Bench (arxiv:2306.03831). Use their
  protocols when comparing models.

## Pitfalls
- Modality mismatch. The SAR-aware models above were pretrained on Sentinel-1 GRD amplitude,
  not on interferometric phase, coherence or InSAR time series. Applying them to deformation
  means feeding wrapped interferograms or velocity maps as images. That is a domain shift the
  model was never pretrained on, and has to be tested rather than assumed.
- Unfair comparisons: different fine-tuning budgets, frozen vs fine-tuned encoders, different
  input sizes or band subsets. Fix one protocol for all models.
- Spatial leakage: random train/test splits over autocorrelated imagery inflate scores. Split
  by region or by event.
- Rare positives: deformation and other events are a small fraction of pixels. Report F1 or IoU
  for the positive class, not accuracy.
- Baselines: always include a task-specific model trained from scratch (e.g. a U-Net) and a
  classical non-learned method. A foundation model must beat both to be worth its cost.
- "Best" depends on the task (detection vs velocity regression), the inputs available, the
  label budget and compute. State which one is meant.

## Validation practice
- Same splits, same fine-tuning protocol and several seeds for every model.
- Report mean ± spread, and use a paired test across seeds or regions.
- For deformation, validate against GNSS or InSAR-derived reference products, not against
  another model.
