# SeismicAutoencoders.jl

Seismic waveform autoencoders for ambient noise cross-correlation analysis.

The primary model is a **VQ-VAE (Vector Quantized Variational Autoencoder)** trained on
ambient seismic noise cross-correlations between station pairs. The goal is not reconstruction
quality alone — it is to discover discrete, physically meaningful waveform modes that reveal
coherent seismic arrivals when averaged within each quantized state.

> **Other architectures** (SymAE, CatVAE, PhaseAligner) are in `experimental/` and are
> research prototypes, not production-ready.

---

## Repository structure

```
SeismicAutoencoders/
├── vqvae/                    Primary VQ-VAE v9 code
│   ├── VQVAE_architecture_v9.jl   Model, training loop, XLA compilation
│   ├── train_vqvae.jl             CLI training entry point
│   ├── train_vqvae.sh             Launcher (single or multi-GPU, background or foreground)
│   ├── data_generators.jl         Data loading and preprocessing
│   ├── Prepare_Tomography_v9.jl   Post-training tomography preparation
│   ├── TomographySelection_v9.jl  Station-pair selection for tomography
│   ├── VQVAE_readme.md            Detailed architecture evolution notes (v1→v9)
│   └── history/                   Older versions (v1–v8) and experimental variants
│
├── vqvae_analysis/           Post-training analysis notebooks
│   ├── Trained_VQVAE_MFT_v11.jl           Current analysis script
│   ├── Trained_VQVAE_MFT_v10.jl           Previous analysis script
│   ├── Trained_VQVAE_MFT_v10_codebook_superatom_dump.jl
│   ├── Trained_VQVAE_Best_Mix_v9.jl
│   └── Analysis_VQVAE_v5.jl
│
├── dsurftomo/                Surface wave tomography (DSurfTomo / MFT)
│   ├── DSurfTomo_v1.jl, pDSurfTomo_v1.jl
│   ├── MFT.jl, MFT_v1.jl, MFT_v2.jl, MFT.pluto.jl
│   ├── Global_Average_MFT_Tomography.jl
│   └── Trained_VQVAE_MFT_Tomography_v8.jl, _v9.jl
│
├── experimental/             Research prototypes — not production-ready
│   ├── symae/                Symmetric Autoencoder
│   ├── catvae/               Categorical VAE
│   └── phase_aligner/        Phase alignment network
│
├── Notebooks/                Pluto/Jupyter notebooks
├── Project.toml              Julia package manifest
├── Manifest.toml             Pinned dependency versions
└── .gitignore
```

---

## Installation

Requires **Julia 1.10+** and an **NVIDIA GPU** with CUDA support.

```bash
git clone https://github.com/pawbz/SeismicAutoencoders.jl
cd SeismicAutoencoders.jl
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

This installs all dependencies (Lux, Reactant, Enzyme, CUDA, etc.).

---

## Quick start

```bash
# List available station pairs in your data directory
./vqvae/train_vqvae.sh --list-pairs --data-dir /path/to/jld2/files

# Inspect a random pair (prints stats + terminal plots of waveforms and PSD)
./vqvae/train_vqvae.sh --sample-pair --data-dir /path/to/jld2/files

# Train all pairs on GPU 0, background (default)
./vqvae/train_vqvae.sh --data-dir /path/to/jld2/files --nepoch 100

# Train all pairs on GPU 0, print to terminal
./vqvae/train_vqvae.sh --foreground --data-dir /path/to/jld2/files --nepoch 100

# Train across two GPUs in parallel
./vqvae/train_vqvae.sh --gpus 0,1 --data-dir /path/to/jld2/files --nepoch 100

# Train specific pairs
./vqvae/train_vqvae.sh AP-BK,AP-CL --data-dir /path/to/jld2/files
```

---

## VQ-VAE v9 architecture

The v9 model (internally called "v10 Split-Decoder") is a **Split-Decoder Interferometric
Mixture VQ-VAE** built with Lux.jl and compiled to XLA via Reactant.jl.

**Design:**
- Single shared SEANet-style encoder → shared latent features
- Two independent encoder heads → z_e1, z_e2 (each d÷2 dimensional)
- Two independent codebooks (K₁, K₂) — quantized independently via EMA
- Two independent decoders → x̂₁, x̂₂
- Additive reconstruction: x̂ = x̂₁ + x̂₂ — forces each decoder to specialize

**Training:**
- XLA compiled once per session, reused across all pairs and seeds
- Ensemble nearest-neighbor targets for stabilized codebook learning
- Per-pair, per-seed training; models saved independently

Key parameters (defaults):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--K` | `5,3` | Codebook sizes for the two stages |
| `--d` | `40` | Latent dimension (split equally between two heads) |
| `--nepoch` | `100` | Training epochs |
| `--seeds` | `1234,1235` | Random seeds (one model trained per seed) |
| `--batchsize` | `4096` | Minibatch size |
| `--nwindows` | `20000` | Waveforms per pair |
| `--period-min/max` | `10/75 s` | Bandpass filter period range |

See `vqvae/VQVAE_readme.md` for the full architecture evolution from v1 to v9.

---

## Multi-GPU training

`train_vqvae.sh` splits station pairs round-robin across N GPUs via `--gpus`.
Each GPU gets its own Julia process, compiles XLA once, and trains its assigned pairs
and seeds sequentially. Use `--foreground` to stream all GPU logs to the terminal live.

```bash
# Two GPUs, background
./vqvae/train_vqvae.sh --gpus 0,1 --data-dir /path/to/data --nepoch 100

# Two GPUs, stream to terminal
./vqvae/train_vqvae.sh --gpus 0,1 --foreground --data-dir /path/to/data --nepoch 100

# Four GPUs
./vqvae/train_vqvae.sh --gpus 0,1,2,3 --data-dir /path/to/data --nepoch 100

# Monitor logs manually
tail -f train_vqvae_gpu*_<timestamp>.out
```

---

## Data format

Input data is expected as **JLD2 files**, one per station pair, in the directory passed
to `--data-dir`. File names must follow the pattern `NET1_STA1_NET2_STA2*.jld2`.

Each file contains ambient noise cross-correlations (causal and acausal sides) for that
station pair. The `data_generators.jl` module handles loading, normalisation, tapering,
bandpass filtering, and train/test splitting.

---

## Output

Trained models are saved under `--save-dir` (default: `<data-dir>/SavedModels/vqvae_v9_K=[...]`).
Each run creates a timestamped directory per pair and seed containing:

- `run_summary.jld2` — hyperparameters and loss history
- `loss_history.jld2` — per-epoch metrics
- `source_state_averages.jld2` — codebook assignments and waveform averages

---

## Citation / contact

Pawan Bharadwaj — bharadwaj.pawan@gmail.com
