# VQVAE Architecture Notes

This document summarizes the evolution of the seismic VQVAE code and the current design intent.

## Scientific objective

The current objective is not only reconstruction quality. The main goal is to discover discrete coherent waveform modes so that averaging within each discovered state reveals physically meaningful structure (especially around expected arrivals).

Core implications:
- focus quantization on expected-arrival latent times (vmin/vmax window)
- avoid forcing the model to explain all waveform variance equally
- use coarse codebook assignments for subgroup discovery and waveform stacking

---

## v1 — VQVAE_architecture_v1.jl

Initial working baseline.

Key points:
- EMA vector quantizer with straight-through estimator and dead-code reset
- accepted 2D and 3D input layouts
- training used random unpaired pooled waveforms

Main limitation for discovery:
- representation was still effectively per-waveform; no explicit mechanism to prioritize coherent shared structure

---

## v2 / v2.1

Stabilization and indexing upgrades.

Key points:
- cleaner batch handling and combination-index utilities for multi-slot analysis
- improved analysis helpers (cluster combinations, labels, cross-analysis)
- slot-attention/sequence variants explored in v2.1

Main limitation:
- still largely reconstruction-driven, which can spread code usage across nuisance variation

---

## v3 family

Two important branches were used as references:

1. VQVAE_architecture.jl
- RVQ-style staged quantization and time-resolved options

2. VQVAE_architecture_v3.1.jl and VQVAE_architecture_v3.2.jl
- latent-time arrival window support from vmin/vmax and interstation distance
- per-position/per-slot experiments in v3.2
- v3.2 also introduced arrival-window weighted MSE

Design conclusion reached in later iterations:
- arrival focus should come primarily from selecting latent times in the expected arrival window (v3.1 style), not from weighted MSE terms

---

## v4 — VQVAE_architecture_v4.jl

Current direction: multiscale quantization for coherent-mode discovery.

### Main idea

Use a two-level quantization strategy:
- Stage 1 coarse codebook quantizes smoothed latent representation (shared/coherent features)
- Stage 2+ detail codebooks quantize residuals (fine-scale variation)

This is implemented via a multiscale RVQ quantizer with cosine-neighborhood smoothing.

### Smoothing strategy used before coarse quantization

Hybrid neighborhood rule (across waveforms in a batch):
- cosine similarity threshold
- max-neighbor cap
- min-neighbor fallback
- soft weighting with temperature

This creates a smoothed coarse latent while avoiding full global averaging.

### v4 parameters of interest

Multiscale settings:
- use_multiscale_rvq
- Ksmall (coarse codebook size)
- Klarge (detail codebook size)
- detail_stages
- ema_decay_small, ema_decay_large

Smoothing settings:
- smooth_cos_threshold
- smooth_max_neighbors
- smooth_min_neighbors
- smooth_temperature
- smooth_blend

Arrival focus settings:
- interstation_distance
- velocity_range = (vmin, vmax)
- dt

### Arrival focus policy in v4

Current v4 uses v3.1-style arrival focus:
- latent-time window selection based on vmin/vmax and distance
- plain reconstruction MSE in loss (no arrival-weighted MSE path)

arrival_mse_weight remains only as a compatibility field and is effectively deprecated for v4 workflows.

### Dense versus per-position in v4

For the current objective:
- prefer dense whole-waveform quantization over the selected latent arrival window
- do not use per-position/per-slot quantization as the primary mode-discovery path

### What to use for clustering and stacking

For subgroup discovery and waveform averaging:
- use coarse (Stage 1) code assignments
- treat detail stages as residual refinements, not primary class labels

v4 encode outputs include coarse and detail index diagnostics so downstream notebooks can focus on coarse states.

---

## Training notebook linkage

Current notebook for this direction:
- Training_VQVAE_v4.jl

Intended usage in that notebook:
- train v4 multiscale model with arrival-window latent selection
- run cluster analysis using coarse codebook assignments
- average waveforms by coarse state
- run MFT diagnostics on those averaged states

---

## Practical tuning guidance

If coarse clusters are noisy:
- increase smooth_cos_threshold
- reduce Ksmall
- increase smooth_blend toward 1.0

If coarse clusters are over-smoothed/merged:
- decrease smooth_blend
- reduce smooth_max_neighbors
- increase Ksmall moderately

If detail stages absorb too much:
- reduce detail_stages
- lower Klarge
- increase ema_decay_small relative to ema_decay_large

---

## Current recommendation

For coherent source-mode discovery, prioritize:
1. arrival-window latent selection
2. coarse multiscale codebook assignments
3. subgroup averaging and physics-based validation (symmetry, SNR, MFT)

Reconstruction quality remains useful, but is secondary to obtaining stable, physically interpretable coarse states.
