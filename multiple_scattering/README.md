# `multiple_scattering/` — synthetic blind-deconvolution data generator

Generates synthetic seismic gathers for **multichannel blind deconvolution**: a
single earthquake source recorded by many receivers, each through its own
Green's function.

## Forward model

A point earthquake source radiates into a 2-D acoustic medium filled with
randomly placed circular scatterers. The multiple-scattering wavefield is solved
once per frequency (via [`MultipleScattering.jl`](https://github.com/JuliaWaveScattering/MultipleScattering.jl)).
The complex per-receiver frequency response **is** that receiver's Green's
function `g_r`, and the recorded seismogram is

```
d_r = s * g_r          (source convolved with the receiver's response)
```

- **One common source** `s` (the earthquake source-time function) is shared by all receivers.
- **Each receiver has its own** `g_r` (a full multi-sample scattering coda, not a delay).
- **Traveltime moveout** between receivers arises *naturally* from the receiver-line
  geometry (unequal source–receiver distances). No jitter is injected.

Recovering the bare `s` from the many `d_r` is the blind-deconvolution problem
solved in [`../experimental/coherent_n2n/BlindDeconv_main.jl`](../experimental/coherent_n2n/BlindDeconv_main.jl).

## Multiple sources = multiple independent events

Each "source" `k` is an *independent event*: its own random scatterer field, its
own source-time function `s_k`, its own `(Nt × R)` gather. Set `n_sources` in the
config to produce several. Each event is written as **one JLD2 file plus a
matching PNG overview**.

## Running

```bash
cd multiple_scattering
julia -p 8 generate_mscatter_data.jl      # 8 workers for the per-frequency solves
```

The per-frequency multiple-scattering solves are distributed with `pmap`; the
event loop is serial (each event already saturates the pool). Without `-p N` the
script calls `addprocs()` itself.

To customize, `include` and call with a config:

```julia
include("generate_mscatter_data.jl")
generate_dataset(MScatterConfig(Nt=512, n_sources=5, num_particles=70, Mw=6.5))
```

First run installs/precompiles `MultipleScattering` (it is in the General
registry). PGFPlotsX PNG export needs a LaTeX toolchain (`lualatex`/`pdflatex`
+ `pdftoppm`); if it is unavailable the JLD2 is still written and a warning is
logged.

## Output — `data/<basename>.jld2` (+ `data/<basename>.png`)

`<basename>` is `event_<timestamp>_k<index>`. Each `.jld2` is fully
self-describing:

| key                 | type / shape        | meaning                                        |
|---------------------|---------------------|------------------------------------------------|
| `d`                 | `Nt × R` Float      | data `s * g` (+ noise), one column per receiver |
| `d_clean`           | `Nt × R` Float      | noiseless data                                 |
| `g`                 | `Nt × R` Float      | per-receiver Green's / impulse response        |
| `s`                 | `Nt` Float          | source-time function (the recovery target)     |
| `tvec`              | `Nt` Float          | time axis (s)                                  |
| `dt`, `Nt`          | scalars             | sample interval, number of samples             |
| `omega`             | `Nt÷2+1` Float      | `rfftfreq` frequency axis                      |
| `rX`, `rZ`          | `R` Float           | receiver coordinates                           |
| `R`                 | Int                 | number of receivers                            |
| `source_xy`         | 2-vector            | point-source location                          |
| `particle_origins`  | `Vector` of 2-vec   | scatterer centers                              |
| `particle_radii`    | `Vector` Float      | scatterer radii                                |
| `source_index`      | Int                 | event index `k`                                |
| `n_sources`         | Int                 | events in this dataset                         |
| `seed`, `noise_amp` | scalars             | RNG seed, additive-noise fraction              |
| `config`            | `Dict`              | full `MScatterConfig` (media, Mw, geometry, …) |
| `created`           | String              | timestamp                                      |
| `git_commit`        | String              | repo commit for provenance                     |

Each `.png` is a PGFPlotsX overview — **a)** wavefield with scatterers,
receivers, and source overlaid; **b)** impulse-response wiggles `g`; **c)** data
wiggles `d`; **d)** the source wavelet — so you can understand a `.jld2` without
loading it.

## Loading

```julia
using JLD2, FileIO
dat = load("data/event_20260722_120000_k1.jld2")
D = Float32.(dat["d"]); s_true = dat["s"]; tvec = dat["tvec"]
```
