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
│   ├── symvqvae.jl                CLI entry point (train, inspect)
│   ├── symvqvae.sh                Launcher (background/foreground)
│   ├── train_vqvae.jl             Legacy CLI (deprecated)
│   ├── train_vqvae.sh             Legacy launcher (deprecated)
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

**Requirements:** Julia 1.12.6 (tested), an NVIDIA GPU with CUDA support, bash.

### 1. Clone the repository

```bash
git clone https://github.com/pawbz/SeismicAutoencoders.jl
cd SeismicAutoencoders.jl
```

### 2. Install Julia dependencies

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

This resolves and precompiles all dependencies (Lux, Reactant, Enzyme, CUDA, UnicodePlots, etc.)
from the pinned `Manifest.toml`. Takes a few minutes on first run.

### 3. Add the CLI to your shell

Add the following line to your `~/.bashrc` (or `~/.zshrc`), replacing the path with wherever
you cloned the repo:

```bash
export SEISMIC_AE_DIR="/path/to/SeismicAutoencoders"
alias symvqvae="$SEISMIC_AE_DIR/vqvae/symvqvae.sh"
alias train_vqvae="symvqvae train"  # backward compatibility
```

Then reload your shell:

```bash
source ~/.bashrc
```

### 4. Verify

```bash
symvqvae --help
```

---

## Quick start

Once the alias is set up, `symvqvae` works from any directory:

```bash
# Show all commands and options
symvqvae --help

# Inspect a specific station pair (unified waveform axis, raw+whitened PSD comparison)
symvqvae inspect AP_BK --data-dir /path/to/jld2/files

# Customize period range and whitening kernel for inspection
symvqvae inspect AP_BK --data-dir /path/to/jld2/files --period-min 3 --period-max 100 --whitening-kernel-length 256

# Train all pairs, background (default) — logs to file
symvqvae train --data-dir /path/to/jld2/files --nepoch 100

# Train specific pairs only
symvqvae train AP-BK,AP-CL --data-dir /path/to/jld2/files --nepoch 50

# Train with custom parameters
symvqvae train --data-dir /path/to/jld2/files --nepoch 200 --lr 0.0005 --K 8,5 --d 64

# Backward compatibility: old alias still works
train_vqvae --data-dir /path/to/jld2/files --nepoch 100
```

---

## Versioning

**SeismicAutoencoders uses calendar-based versioning (YYYY.MM format).**

Version numbers show the month and year of release:
- `2026.06` = June 2026 release
- `2026.06.1` = Second release in June 2026 (if needed)
- `2026.07` = July 2026 release

This scheme makes it immediately clear when a version was released, without ambiguity about major/minor/patch semantics. Common in scientific and research software.

**Current Version**: v2026.06 (June 2026)

Check your version:
```bash
symvqvae --help
# Shows "SeismicAutoencoders v2026.06" in the header

julia --project=. -e 'include("vqvae/version.jl"); println(version_string())'
# Prints: SeismicAutoencoders v2026.06
```

See [releases](https://github.com/pawbz/SeismicAutoencoders/releases) for release notes and changelog.

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

## Training

By default, `symvqvae train` runs in **background** with automatic logging to a dated file:

```bash
symvqvae train --data-dir /path/to/data --nepoch 100
# Logs to: symvqvae_20260605_143022.out in the current directory
```

To run in the **foreground** (blocking, output to terminal):

```bash
# Run the Julia script directly for foreground output
julia --project=. vqvae/symvqvae.jl train --data-dir /path/to/data --nepoch 100 --foreground
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
