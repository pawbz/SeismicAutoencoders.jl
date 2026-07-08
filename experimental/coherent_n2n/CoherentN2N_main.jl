### A Pluto.jl notebook ###
# v0.1.0

using Markdown
using InteractiveUtils

# ╔═╡ c0000001-0000-0000-0000-000000000001
begin
    using CUDA, cuDNN, Flux, Zygote, FFTW
    using Optimisers, Random, Statistics, LinearAlgebra
    using JLD2
    CUDA.device!(0)  # NVIDIA RTX 6000 Ada Generation
end

# ╔═╡ 20c996f5-d056-4e10-b6fb-0995b444732a
using PlutoPlotly

# ╔═╡ b3f6f1ae-066b-4427-a5f6-5512a0919865
using PlutoUI

# ╔═╡ c0000032-0000-0000-0000-000000000032
begin
    using PlutoHooks, PlutoLinks
    using PlutoLinks: @ingredients
end

# ╔═╡ c000000c-0000-0000-0000-00000000000c
include(joinpath(@__DIR__, "test", "synthetic_data.jl"))

# ╔═╡ c0000002-0000-0000-0000-000000000002
md"""# CoherentN2N — Main Notebook

Alternating block-coordinate scheme for aligning and denoising a single
station's multi-earthquake gather: cross-spectrum phase-slope shift
estimation (Block B) alternated with Noise2Noise complex-spectrum
denoising (Block A), with an optional per-earthquake polarity/gain
correction.

Unlike `experimental/phase_aligner/` (a learned siamese scalar-phase
network, no denoising), this method uses classical frequency-domain
cross-correlation for alignment and a trained Noise2Noise network for
denoising, alternating the two.

## Files loaded by this notebook

- `CoherentN2N_shift.jl` — Fourier shift theorem (complex-domain), coarse
  xcorr + phase-slope sub-sample shift estimator (`estimate_shift_two_stage`)
- `CoherentN2N_gauge.jl` — zero-sum gauge fixing for per-earthquake shifts
- `CoherentN2N_polarity.jl` — per-earthquake complex gain (polarity/amplitude)
  estimation (used only if `use_polarity_gain=true`)
- `CoherentN2N_denoiser.jl` — complex-valued conv encoder/decoder network
- `CoherentN2N_n2n_pairs.jl` — Noise2Noise pair sampling (random earthquake
  pairs at the same station)
- `CoherentN2N_train_denoiser.jl` — denoiser training loop
- `CoherentN2N_outer_loop.jl` — the alternating loop itself (`run_coherent_n2n`)

## Workflow
1. Load all architecture/algorithm files (below)
2. Synthetic validation (known ground truth — sanity check before real data)
3. Real single-station receiver-function data (same JLD2 convention as
   `experimental/phase_aligner/Training_PhaseAligner.jl`)
4. Diagnostics: shift recovery, source stack, convergence history
"""

# ╔═╡ c0000003-0000-0000-0000-000000000003
md"## Load architecture / algorithm files"

# ╔═╡ c0000004-0000-0000-0000-000000000004
# Single @ingredients call over CoherentN2N_lib.jl, which include()s all 7
# architecture files (shift/gauge/polarity/denoiser/n2n_pairs/train_denoiser/
# outer_loop). @ingredients uses Revise under the hood, so edits to any of
# those files (or CoherentN2N_lib.jl's include list) are picked up
# automatically — no manual cell rerun or kernel restart needed for
# function-body changes. All functions/structs are accessed as cn.name(...).
cn = @ingredients(joinpath(@__DIR__, "CoherentN2N_lib.jl"))

# ╔═╡ c000000b-0000-0000-0000-00000000000b
md"---
## Synthetic Validation

Known ground-truth shifts, source wavelet, and (optionally) per-earthquake
polarity/gain — sanity-checks the pipeline before touching real data.
Mirrors the test scenarios in `test/test_outer_loop.jl` and
`test/test_outer_loop_polarity.jl`.
"

# ╔═╡ c000000d-0000-0000-0000-00000000000d
use_polarity_gain_syn = false  # toggle to true to exercise the polarity/gain path

# ╔═╡ c000000e-0000-0000-0000-00000000000e
begin
    rng_syn = MersenneTwister(1)
    nt_syn = 128
    R_syn = 200
    τ_true_syn = Float32.(range(-8, 8, length=R_syn))
    g_true_syn = use_polarity_gain_syn ?
        ComplexF32.(rand(rng_syn, (-1.0, 1.0), R_syn) .* rand(rng_syn, 0.7:0.1:1.3, R_syn)) :
        ones(ComplexF32, R_syn)

    s_true_syn, D_syn, freqs_syn, _, _ = make_synthetic_gather(
        nt=nt_syn, R=R_syn, f0=0.05, source_kind=:broadband,
        true_shifts=τ_true_syn, true_gains=g_true_syn,
        noise_std=0.2, rng=rng_syn)
end

# ╔═╡ c000001a-0000-0000-0000-00000000001a
md"### Raw Training Data (before alignment/denoising)"

# ╔═╡ c000001b-0000-0000-0000-00000000001b
let
    tr = PlutoPlotly.heatmap(z=D_syn, colorscale="RdBu", zmid=0)
    layout = Layout(
        title=attr(text="Raw synthetic gather (unaligned, noisy) — $(R_syn) earthquakes"),
        xaxis=attr(title="Earthquake index"), yaxis=attr(title="Sample"),
        height=400, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot([tr], layout))
end

# ╔═╡ c000001c-0000-0000-0000-00000000001c
let
    ts = 1:nt_syn
    n_show = min(8, R_syn)
    traces = [
        PlutoPlotly.scatter(x=collect(ts), y=D_syn[:, r] .+ 3 * (r - 1), mode="lines",
            name="eq $r (τ=$(round(τ_true_syn[r], digits=1)))", line=attr(width=1.2))
        for r in 1:n_show
    ]
    layout = Layout(
        title=attr(text="Raw synthetic traces (first $(n_show), offset for visibility)"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Amplitude (offset)", showticklabels=false),
        height=450, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ c000000f-0000-0000-0000-00000000000f
para_syn = cn.CoherentN2N_Para(nt=nt_syn, enc_kernels=[16, 8], enc_filters=[8, 16],
                                dec_kernels=[8, 16], dec_filters=[8, 2], use_gpu=true)

# ╔═╡ c0000010-0000-0000-0000-000000000010
outer_para_syn = cn.CoherentN2N_Outer_Para(
    n_outer_iters=10, use_polarity_gain=use_polarity_gain_syn,
    denoiser_training=cn.CoherentN2N_Denoiser_Training_Para(
        n_samples_per_epoch=256, batchsize=32, nepoch=80,
        initial_lr=0.003, restart_period=40, nprint=20))

# ╔═╡ c0000011-0000-0000-0000-000000000011
result_syn = cn.run_coherent_n2n(D_syn, para_syn, outer_para_syn; rng=rng_syn)

# ╔═╡ c0000012-0000-0000-0000-000000000012
let
    ŝ_time = real(ifft(result_syn.ŝ))
    c = cor(ŝ_time, s_true_syn)
    @info "Synthetic source recovery" correlation=c
    (; correlation=c, delta_s=result_syn.history.delta_s, delta_tau=result_syn.history.delta_tau)
end

# ╔═╡ c000001d-0000-0000-0000-00000000001d
md"### Aligned Data (final recovered shifts applied)"

# ╔═╡ c000001e-0000-0000-0000-00000000001e
D_aligned_syn = let
    grid_syn = ComplexF32.(-im .* 2f0 .* Float32(π) .* freqs_syn)
    X̂_syn = ComplexF32.(fft(D_syn, 1))
    # result_syn.τ is absolute (anchor reapplied); re-center to the same
    # gauge used internally so the alignment below matches what the
    # outer loop itself used to build its final coherent stack.
    τ_gauge = result_syn.τ .- mean(result_syn.τ)
    X̂_aligned_syn = cn.shift_spectrum(X̂_syn, reshape(-τ_gauge, 1, R_syn), grid_syn)
    real(ifft(X̂_aligned_syn, 1))
end

# ╔═╡ c000001f-0000-0000-0000-00000000001f
let
    tr = PlutoPlotly.heatmap(z=D_aligned_syn, colorscale="RdBu", zmid=0)
    layout = Layout(
        title=attr(text="Aligned synthetic gather (τ from run_coherent_n2n) — $(R_syn) earthquakes"),
        xaxis=attr(title="Earthquake index"), yaxis=attr(title="Sample"),
        height=400, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot([tr], layout))
end

# ╔═╡ c0000020-0000-0000-0000-000000000020
let
    ts = 1:nt_syn
    n_show = min(8, R_syn)
    traces = [
        PlutoPlotly.scatter(x=collect(ts), y=D_aligned_syn[:, r] .+ 3 * (r - 1), mode="lines",
            name="eq $r", line=attr(width=1.2))
        for r in 1:n_show
    ]
    layout = Layout(
        title=attr(text="Aligned synthetic traces (first $(n_show), offset for visibility)"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Amplitude (offset)", showticklabels=false),
        height=450, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ fd5a5b29-7c3f-4fa6-959c-bb34c22c035b
md"### Synthetic Diagnostics"

# ╔═╡ c0000016-0000-0000-0000-000000000016
md"---
## Diagnostics

`result.ŝ` — final complex source-spectrum estimate (`real(ifft(result.ŝ))`
for the time-domain stack). `result.τ` — recovered per-earthquake shifts
(absolute, anchor reapplied). `result.gains` — per-earthquake complex gain
(all-ones if `use_polarity_gain=false`). `result.outliers` — energy-outlier
mask (excluded from the stack/gauge fit).

Two distinct \"loss\"-like signals are plotted below — don't conflate them:
- **`result.history.delta_s` / `delta_tau`**: the **outer alternating loop's
  own convergence** — how much the recovered source estimate / shift vector
  changed between successive outer iterations. Not a trained loss; a
  block-coordinate convergence check (should trend toward zero if the loop
  is settling).
- **`result.history.denoiser_loss`**: the **Noise2Noise network's actual
  training loss** — MSE between the denoiser's prediction on one earthquake
  and a different, randomly-paired earthquake's aligned spectrum, one curve
  per outer iteration, one point per training epoch. Measures whether the
  network is learning within a given outer iteration's training run,
  independent of whether the outer loop itself is converging.
"

# ╔═╡ c0000017-0000-0000-0000-000000000017
let
    ŝ_time = real(ifft(result_syn.ŝ))
    raw_mean = vec(mean(D_syn, dims=2))
    ts = 1:nt_syn
    # Normalize each trace by its own max-abs so amplitude-scale differences
    # (raw mean is typically much smaller than ŝ before denoising) don't
    # visually dominate the comparison.
    norm1(x) = x ./ (maximum(abs, x) + eps(Float32))
    traces = [
        PlutoPlotly.scatter(x=collect(ts), y=norm1(s_true_syn), mode="lines",
            name="True source", line=attr(color="grey", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=collect(ts), y=norm1(ŝ_time), mode="lines",
            name="Recovered ŝ", line=attr(color="#d62728", width=2)),
        PlutoPlotly.scatter(x=collect(ts), y=norm1(raw_mean), mode="lines",
            name="Raw Mean", line=attr(color="blue", width=2)),
    ]
    layout = Layout(
        title=attr(text="Synthetic: recovered vs. true source (normalized)"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Normalized amplitude"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ c0000018-0000-0000-0000-000000000018
let
    τ_est_centered = result_syn.τ .- mean(result_syn.τ)
    τ_true_centered = τ_true_syn .- mean(τ_true_syn)
    idx = collect(1:length(τ_true_syn))
    traces = [
        PlutoPlotly.scatter(x=idx, y=τ_true_centered, mode="markers",
            name="True τ (centered)", marker=attr(color="grey", size=8, symbol="circle-open")),
        PlutoPlotly.scatter(x=idx, y=τ_est_centered, mode="markers",
            name="Recovered τ (centered)", marker=attr(color="#d62728", size=6)),
    ]
    layout = Layout(
        title=attr(text="Synthetic: recovered vs. true shifts (gauge-centered)"),
        xaxis=attr(title="Earthquake index"), yaxis=attr(title="τ (samples)"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ c0000019-0000-0000-0000-000000000019
let
    iters = collect(1:length(result_syn.history.delta_s))
    tr1 = PlutoPlotly.scatter(x=iters, y=result_syn.history.delta_s, mode="lines+markers",
        name="Δŝ", line=attr(color="#1f77b4", width=2))
    tr2 = PlutoPlotly.scatter(x=iters, y=result_syn.history.delta_tau, mode="lines+markers",
        name="Δτ", line=attr(color="#d62728", width=2), yaxis="y2")
    layout = Layout(
        title=attr(text="Synthetic: OUTER-LOOP convergence (Δŝ, Δτ between iterations — not a trained loss)"),
        xaxis=attr(title="Outer iteration"),
        yaxis=attr(title="‖Δŝ‖", side="left"),
        yaxis2=attr(title="‖Δτ‖", side="right", overlaying="y"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot([tr1, tr2], layout))
end

# ╔═╡ c0000030-0000-0000-0000-000000000030
let
    traces = [
        PlutoPlotly.scatter(x=collect(1:length(loss_curve)), y=loss_curve, mode="lines",
            name="Outer iter $i", line=attr(width=1.5))
        for (i, loss_curve) in enumerate(result_syn.history.denoiser_loss)
    ]
    layout = Layout(
        title=attr(text="Synthetic: Noise2Noise DENOISER training loss (per outer iteration)"),
        xaxis=attr(title="Epoch (within outer iteration)"),
        yaxis=attr(title="MSE(pred, other earthquake)", type="log"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ c0000013-0000-0000-0000-000000000013
md"---
## Real Data Section

Same single-station, multi-earthquake JLD2 convention as
`experimental/phase_aligner/Training_PhaseAligner.jl`: `Data[station_index]`
-> `(nt, n_events)` matrix, parallel `Sta`/`EventLoc`/`EventMag`/`EventDep`
arrays, companion SNR file for quality filtering. Replace `fldir`/`dn`/`StaN`
below with your real path and station of interest.
"

# ╔═╡ c0000014-0000-0000-0000-000000000014
fldir_real = "/mnt/NAS/EQData/RFData"

# ╔═╡ c0000015-0000-0000-0000-000000000015
dn_real = "GSN_150ZTR_Bandpass_0.05_0.8_29nov_rf_iter_f1.jld2"

# ╔═╡ c0000021-0000-0000-0000-000000000021
snrf_real = "GSN_150ZTR_Bandpass_0.05_0.8_29nov_snr.jld2"

# ╔═╡ c0000022-0000-0000-0000-000000000022
StaN_real = "RAYN"  # station of interest — change to select a different station

# ╔═╡ c0000023-0000-0000-0000-000000000023
begin
    dfile_real = "$(fldir_real)/$(dn_real)"
    StaName_real = load(dfile_real)["Sta"][1]
    ses_snr_real = load("$(fldir_real)/$(snrf_real)", "SNR")
    ix_real = findall(x -> x == StaN_real, StaName_real)[1]
    R_real_raw = load(dfile_real)["Data"][ix_real]
    snr_tres_real = 0.0
    sel_snr_real = findall(x -> x > snr_tres_real, ses_snr_real[ix_real])
    R_real_sel = R_real_raw[:, sel_snr_real]
    raw_data_real = R_real_sel[501:800, :]   # trim RF window, matches Training_PhaseAligner.jl convention

    # Station coordinate ([lat, lon]) and per-earthquake event coordinates,
    # filtered by the same sel_snr_real mask as raw_data_real so indices
    # line up 1:1 with D_real's columns.
    StaLoc_real = load(dfile_real)["Sta"][2][ix_real]
    EvtLoc_real = load(dfile_real)["EventLoc"][ix_real][sel_snr_real]

    (n_traces=size(raw_data_real, 2), nt=size(raw_data_real, 1))
end

# ╔═╡ c0000024-0000-0000-0000-000000000024
"""Per-trace normalize + taper (same convention as Training_PhaseAligner.jl:388-391)."""
function taper_sin_real(x)
    nt = size(x, 1)
    w = sin.(range(0, π, length=nt)) .^ 1
    return x .* reshape(w, nt, ntuple(_ -> 1, ndims(x) - 1)...)
end

# ╔═╡ c0000025-0000-0000-0000-000000000025
D_real = let
    mr = mean(raw_data_real; dims=1)
    sr = std(raw_data_real; dims=1)
    Float32.(taper_sin_real((raw_data_real .- mr) ./ max.(sr, 1f-8)))
end

# ╔═╡ c0000026-0000-0000-0000-000000000026
md"### Raw Real Data (before alignment/denoising)"

# ╔═╡ c0000027-0000-0000-0000-000000000027
let
    tr = PlutoPlotly.heatmap(z=D_real, colorscale="RdBu", zmid=0)
    layout = Layout(
        title=attr(text="Raw real gather — station $(StaN_real), $(size(D_real, 2)) earthquakes"),
        xaxis=attr(title="Earthquake index"), yaxis=attr(title="Sample"),
        height=400, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot([tr], layout))
end

# ╔═╡ c0000028-0000-0000-0000-000000000028
para_real = cn.CoherentN2N_Para(nt=size(D_real, 1), use_gpu=true)

# ╔═╡ c0000029-0000-0000-0000-000000000029
outer_para_real = cn.CoherentN2N_Outer_Para(n_outer_iters=10, 
											use_polarity_gain=false)

# ╔═╡ c000002a-0000-0000-0000-00000000002a
result_real = cn.run_coherent_n2n(D_real, para_real, outer_para_real)

# ╔═╡ c000002b-0000-0000-0000-00000000002b
let
    ŝ_time = real(ifft(result_real.ŝ))
    raw_mean = vec(mean(D_real; dims=2))
    ts = 1:size(D_real, 1)
    # Normalize each trace by its own max-abs — the recovered ŝ and the raw
    # mean can differ by orders of magnitude in scale (ŝ isn't amplitude-
    # calibrated against the raw stack), which otherwise dominates the plot.
    norm1(x) = x #./ (maximum(abs, x) + eps(Float32))
    traces = [
        PlutoPlotly.scatter(x=collect(ts), y=norm1(raw_mean), mode="lines",
            name="Raw mean", line=attr(color="grey", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=collect(ts), y=norm1(ŝ_time), mode="lines",
            name="Recovered ŝ", line=attr(color="#d62728", width=2)),
    ]
    layout = Layout(
        title=attr(text="Real data: recovered coherent stack vs. raw mean ($(StaN_real), normalized)"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Normalized amplitude"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ c000002c-0000-0000-0000-00000000002c
D_aligned_real = let
    freqs_real = Float32.(fftfreq(size(D_real, 1)))
    grid_real = ComplexF32.(-im .* 2f0 .* Float32(π) .* freqs_real)
    X̂_real = ComplexF32.(fft(D_real, 1))
    τ_gauge_real = result_real.τ .- mean(result_real.τ)
    X̂_aligned_real = cn.shift_spectrum(X̂_real, reshape(-τ_gauge_real, 1, length(τ_gauge_real)), grid_real)
    real(ifft(X̂_aligned_real, 1))
end

# ╔═╡ c000002d-0000-0000-0000-00000000002d
md"### Aligned Real Data (final recovered shifts applied)"

# ╔═╡ c000002e-0000-0000-0000-00000000002e
let
    tr = PlutoPlotly.heatmap(z=D_aligned_real, colorscale="RdBu", zmid=0)
    layout = Layout(
        title=attr(text="Aligned real gather — station $(StaN_real)"),
        xaxis=attr(title="Earthquake index"), yaxis=attr(title="Sample"),
        height=400, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot([tr], layout))
end

# ╔═╡ c000002f-0000-0000-0000-00000000002f
let
    iters = collect(1:length(result_real.history.delta_s))
    tr1 = PlutoPlotly.scatter(x=iters, y=result_real.history.delta_s, mode="lines+markers",
        name="Δŝ", line=attr(color="#1f77b4", width=2))
    tr2 = PlutoPlotly.scatter(x=iters, y=result_real.history.delta_tau, mode="lines+markers",
        name="Δτ", line=attr(color="#d62728", width=2), yaxis="y2")
    layout = Layout(
        title=attr(text="Real data: OUTER-LOOP convergence ($(StaN_real)) — Δŝ, Δτ between iterations, not a trained loss"),
        xaxis=attr(title="Outer iteration"),
        yaxis=attr(title="‖Δŝ‖", side="left"),
        yaxis2=attr(title="‖Δτ‖", side="right", overlaying="y"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot([tr1, tr2], layout))
end

# ╔═╡ c0000031-0000-0000-0000-000000000031
let
    traces = [
        PlutoPlotly.scatter(x=collect(1:length(loss_curve)), y=loss_curve, mode="lines",
            name="Outer iter $i", line=attr(width=1.5))
        for (i, loss_curve) in enumerate(result_real.history.denoiser_loss)
    ]
    layout = Layout(
        title=attr(text="Real data: Noise2Noise DENOISER training loss ($(StaN_real), per outer iteration)"),
        xaxis=attr(title="Epoch (within outer iteration)"),
        yaxis=attr(title="MSE(pred, other earthquake)", type="log"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ c0000033-0000-0000-0000-000000000033
md"### Shift Map (epicentral distance vs. backazimuth)"

# ╔═╡ c0000034-0000-0000-0000-000000000034
begin
    """Great-circle epicentral distance in degrees (spherical Earth)."""
    function epicentral_distance_deg(sta_lat, sta_lon, evt_lat, evt_lon)
        φ1, φ2 = deg2rad(sta_lat), deg2rad(evt_lat)
        dφ = deg2rad(evt_lat - sta_lat)
        dλ = deg2rad(evt_lon - sta_lon)
        h = sin(dφ / 2)^2 + cos(φ1) * cos(φ2) * sin(dλ / 2)^2
        rad2deg(2 * asin(sqrt(h)))
    end

    """Backazimuth in degrees (0-360, from N, clockwise): direction FROM the
    station back TO the source, as seen at the station — the standard
    seismological convention (event -> station azimuth, i.e. azimuth(evt, sta))."""
    function backazimuth_deg(sta_lat, sta_lon, evt_lat, evt_lon)
        φ1, φ2 = deg2rad(evt_lat), deg2rad(sta_lat)
        dλ = deg2rad(sta_lon - evt_lon)
        y = sin(dλ) * cos(φ2)
        x = cos(φ1) * sin(φ2) - sin(φ1) * cos(φ2) * cos(dλ)
        mod(rad2deg(atan(y, x)) + 360.0, 360.0)
    end
end

# ╔═╡ c0000035-0000-0000-0000-000000000035
let
    sta_lat, sta_lon = StaLoc_real[1], StaLoc_real[2]
    epi_dist = [epicentral_distance_deg(sta_lat, sta_lon, ev[1], ev[2]) for ev in EvtLoc_real]
    baz = [backazimuth_deg(sta_lat, sta_lon, ev[1], ev[2]) for ev in EvtLoc_real]
    τ_gauge_map = result_real.τ .- mean(result_real.τ)

    tr = PlutoPlotly.scatterpolar(
        r=epi_dist, theta=baz, mode="markers",
        marker=attr(size=6, color=τ_gauge_map, colorscale="RdBu", cmid=0,
                    colorbar=attr(title="τ (samples)"), showscale=true),
    )
    layout = Layout(
        title=attr(text="Real data: recovered shifts by epicentral distance / backazimuth ($(StaN_real))"),
        polar=attr(
            radialaxis=attr(title="Epicentral distance (°)"),
            angularaxis=attr(direction="clockwise", rotation=90),  # 0=N at top, clockwise like a compass
        ),
        height=600, width=700, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot([tr], layout))
end

# ╔═╡ c0000036-0000-0000-0000-000000000036
let
    τ_gauge_hist = result_real.τ .- mean(result_real.τ)
    tr = PlutoPlotly.histogram(x=τ_gauge_hist, marker=attr(color="#d62728", opacity=0.75))
    layout = Layout(
        title=attr(text="Real data: recovered shift distribution ($(StaN_real), gauge-centered)"),
        xaxis=attr(title="τ (samples)"), yaxis=attr(title="Count"),
        height=350, width=700, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot([tr], layout))
end

# ╔═╡ a550a648-7a94-11f1-8a00-f3225eb4ac28
md"---"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[compat]
CUDA = "~6.2.0"
FFTW = "~1.10.0"
Flux = "~0.16.10"
JLD2 = "~0.6.5"
Optimisers = "~0.4.7"
PlutoHooks = "~0.1.0"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.6"
PlutoUI = "~0.7.83"
Zygote = "~0.7.11"
cuDNN = "~6.2.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "2b1fe4c1bbdc631ffcde1d297327a5b5334717a6"

[[deps.ADTypes]]
git-tree-sha1 = "d9aaef7c63466eee4de23b4d9dad03629df54bea"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.22.1"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "daa72978cd7a624246e894a4f4f067706d4e17e2"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "b8651b2eb5796a386b0398a20b519a6a6150f75c"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.3"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "e386db8b4753b42caac75ac81d0a4fe161a68a97"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.6.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CUDA]]
deps = ["CUDACore", "CUDATools", "Reexport", "cuBLAS", "cuFFT", "cuRAND", "cuSOLVER", "cuSPARSE"]
git-tree-sha1 = "6fd394787b8b0aa63abaf28490a5e4b65b4cdfc2"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "6.2.0"

[[deps.CUDACore]]
deps = ["Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVPTX_LLVM_Backend_jll", "PrecompileTools", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "StaticArrays"]
git-tree-sha1 = "6e4602b5aed1ba1e6aa867fca91ee5d9618dce90"
uuid = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
version = "6.2.0"
weakdeps = ["CUDA", "ChainRulesCore", "EnzymeCore", "SpecialFunctions"]

    [deps.CUDACore.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.CUDATools]]
deps = ["CUDACore", "CUDA_Compiler_jll", "CUPTI", "Crayons", "GPUCompiler", "LLVM", "NVML", "NVTX", "PrecompileTools", "Preferences", "PrettyTables", "Printf", "Statistics", "demumble_jll"]
git-tree-sha1 = "d40ef634fed4aeb96884baf9fbf35eeb3401c24a"
uuid = "9ec180c6-1c07-47c7-9e6e-ebefa4d1f6d0"
version = "6.2.0"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "c32d22f2f563ce192c88a44b09c2b569f1e7a980"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.4.4+1"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "2d6222474d868469a72de5bd47c5c25c0e1fe518"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.3.0+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "79312abe5261a660f94e746e449d2cb2fe3284d9"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "2.1.0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "2e0352eb2a8321e46e1de54059bed9be8fd9391c"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.23.0+1"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "70dea6a7133d2100a143b515a00d6d887e208500"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "9.20.0+0"

[[deps.CUPTI]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox"]
git-tree-sha1 = "0dfc25f792ca2728c1e4e1f3e268570b3f9489bd"
uuid = "9e67e8f6-ba02-4b6c-a7db-3b11ae1e7ab7"
version = "6.2.0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3c190c570fb3108c09f838607386d10c71701789"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.73.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "1a3ad7e16a321667698a19e77362b35a1e94c544"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.1"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "d4101e848e8d3f585d61d244c2fe0c80a70e6b3b"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.1.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "REPL", "UUIDs"]
git-tree-sha1 = "cfb7a2e89e245a9d5016b70323db412b3a7438d5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6fb53a69613a0b2b68a0d12671717d307ab8b24e"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.5"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EnzymeCore]]
git-tree-sha1 = "971d7831cc85f43bc9f51d615a3f7f21270c2f1d"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.21"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6866aec60ef98e3164cd8d6855225684207e9dff"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.12+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "8e9c059d6857607253e837730dbf780b6b151acd"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.19.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Flux]]
deps = ["ADTypes", "Adapt", "ChainRulesCore", "Compat", "EnzymeCore", "Functors", "GPUArrays", "LinearAlgebra", "MLCore", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "cb318a415a089c337d0c15000d1608cee8434ebf"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.16.10"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxEnzymeExt = "Enzyme"
    FluxFiniteDifferencesExt = "FiniteDifferences"
    FluxMPIExt = "MPI"
    FluxMPINCCLExt = ["CUDA", "MPI", "NCCL"]
    FluxMooncakeExt = "Mooncake"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "2c5d0b0e12088cde2cf84afb2784415b1ea3dfee"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.1"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "SparseArrays", "Statistics"]
git-tree-sha1 = "4ea5e2aecfd52e595ff6c343410e32f83f5b9bbf"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.5.8"
weakdeps = ["JLD2"]

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "REPL", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "a39f85e004573d4951fa4a094b0be2944ab0b47d"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.22.7"

    [deps.GPUCompiler.weakdeps]
    LLVMDowngrader_jll = "f52de702-fb25-5922-94ba-81dd59b07444"
    NVPTX_LLVM_Backend_jll = "ef6e0fe3-e6ef-59c0-bde6-4989574699e0"

[[deps.GPUToolbox]]
deps = ["LLVM"]
git-tree-sha1 = "a589b6c1a0eff953571f5d8b0474f5020831114d"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "1.1.1"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "ae3dae80f39426a5e598374e929522285e6ba8d0"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.17"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "9ebadf3f8f0de07031359917549bbdadc23f5dc3"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.5"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c89d196f5ffb64bfbf80985b699ea913b0d2c211"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "58927c485919bf17ea308d9d82156de1adf4b006"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.12"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "a5b87110fa95d711355af44832497745aa93fb52"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.42"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "f74a9668f02e33399baa5ed3a092b3f7a93f192e"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.10.0"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "70c96f133c78c3cdc06234157144fab3744c6b38"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.43+1"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d4e20500d210247322901841d4eafc7a0c52642d"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.13.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "3733419e9a71156b389f3e331672d2e95436783f"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.6.2"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "c4ab44fe709638fda6f2c0cbfea2c114932d6c2f"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.1.0"

    [deps.MLCore.extensions]
    MLCorePythonCallExt = "PythonCall"

    [deps.MLCore.weakdeps]
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "29b00f22be6fd821a214760f0224329f21998a05"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.17.10"

    [deps.MLDataDevices.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"
    ChainRulesCoreExt = "ChainRulesCore"
    ChainRulesExt = "ChainRules"
    ComponentArraysExt = "ComponentArrays"
    FillArraysExt = "FillArrays"
    GPUArraysSparseArraysExt = ["GPUArrays", "SparseArrays"]
    MLUtilsExt = "MLUtils"
    MetalExt = ["GPUArrays", "Metal"]
    OneHotArraysExt = "OneHotArrays"
    OpenCLExt = ["GPUArrays", "OpenCL"]
    ReactantExt = "Reactant"
    RecursiveArrayToolsExt = "RecursiveArrayTools"
    ReverseDiffExt = "ReverseDiff"
    SparseArraysExt = "SparseArrays"
    TrackerExt = "Tracker"
    ZygoteExt = "Zygote"
    cuDNNExt = ["CUDA", "cuDNN"]
    oneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "CodeTracking", "Compat", "DataAPI", "DelimitedFiles", "Distributed", "InteractiveUtils", "MLCore", "Mmap", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "cbaae75c0473c1650f472ca6ed1ec7fc09153b75"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.12"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "446a44652d12ea0a70cbb6f7a9a00ca314ad784a"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.38"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreCUDNNExt = ["EnzymeCore", "CUDA", "cuDNN"]
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibMooncakeCUDAExt = ["Mooncake", "CUDA"]
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NVML]]
deps = ["CEnum", "CUDACore", "GPUToolbox", "Libdl"]
git-tree-sha1 = "1cc497d2e8fc62bdee2809a5b70c5f915f99597f"
uuid = "611af6d1-644e-4c5d-bd58-854d7d1254b9"
version = "6.2.0"

[[deps.NVPTX_LLVM_Backend_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "6a1ec02404e66d69b1b6bc884299a4c80f932c6d"
uuid = "ef6e0fe3-e6ef-59c0-bde6-4989574699e0"
version = "22.1.7+1"

[[deps.NVTX]]
deps = ["JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "a9083c3e469e63cca454d1fc3b19472d9d92c14a"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.3"
weakdeps = ["Colors"]

    [deps.NVTX.extensions]
    NVTXColorsExt = "Colors"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "af2232f69447494514c25742ba1503ec7e9877fe"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.2.2+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "9510d7008275fc5b33fc72a73f8fddef0b5430c6"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.11"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "36b5d2b9dd06290cd65fcf5bdbc3a551ed133af5"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.7"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "32a4e09c5f29402573d673901778a0e03b0807b9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.6"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.23"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "844a829c8dc9fd0fe62eced22bc2d0dfd66a3f51"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.1.0"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "aea4eede5ab3ee188906d0cf3bbfa36eb543dccc"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.8"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "2b9e3d771adfe535a4fdda855f4741fdaacd3f7f"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.6"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "ebf455bb866ee6737030e3d3816bb6a0683c4325"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.4.0"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CRC32c", "CodeTracking", "FileWatching", "InteractiveUtils", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "27e3ee13fc8739a59b380d6163d6a82f52c03bd7"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.15.1"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLPublic]]
git-tree-sha1 = "2b1b64add566435a768abdb3b053cac17d19ff3c"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.2.1"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "6547cbdd8ce32efba0d21c5a40fa96d1a3548f9f"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.8.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "e4d7a1a0edc20af42689ea6f4f3587a2175d50ee"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.12"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "ad8002667372439f2e3611cfd14097e03fa4bccd"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.3"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "0f38a06c83f0007bbab3cf911262841c9a0f07e0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.13.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "0f30765c32d66d58e41f4cb5624d4fc8a82ec13b"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.1"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "615fae83fbb607abc43cf2f84f51bcf3be1a706b"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.11"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteCUDAExt = "CUDA"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.cuBLAS]]
deps = ["Adapt", "BFloat16s", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUArrays", "GPUToolbox", "LLVM", "LinearAlgebra"]
git-tree-sha1 = "4d0e194bf452a3014f2991a29737c6ab158c6f5a"
uuid = "182d3088-87b7-4494-8cad-fc6afaa545bc"
version = "6.2.0"
weakdeps = ["EnzymeCore"]

    [deps.cuBLAS.extensions]
    EnzymeCoreExt = "EnzymeCore"

[[deps.cuDNN]]
deps = ["BFloat16s", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDNN_jll"]
git-tree-sha1 = "d1132e06f9e57ba249078304792d012839f4f3db"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "6.2.0"

[[deps.cuFFT]]
deps = ["AbstractFFTs", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "2bb68317cf1b7e65b8e3cd7240e12089e2604a5c"
uuid = "533571aa-0936-420e-b4be-9c66f5f626ca"
version = "6.2.0"

[[deps.cuRAND]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "Random", "Random123", "RandomNumbers"]
git-tree-sha1 = "4ab8549cd582dd59cb75835ba78b36599cd18606"
uuid = "20fd9a0b-12d5-4c2f-a8af-7c34e9e60431"
version = "6.2.0"

[[deps.cuSOLVER]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "LinearAlgebra", "SparseArrays", "cuBLAS", "cuSPARSE"]
git-tree-sha1 = "297c9da8bc6948db381387ebaf395f28cf011ef0"
uuid = "887afef0-6a32-4de5-add4-7827692ba8fc"
version = "6.2.0"

[[deps.cuSPARSE]]
deps = ["Adapt", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUArrays", "GPUToolbox", "KernelAbstractions", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4e72d4bd131581b83ef0d08a669ab9ae8309c176"
uuid = "b26da814-b3bc-49ef-b0ee-c816305aa060"
version = "6.2.0"

    [deps.cuSPARSE.extensions]
    SparseMatricesCSRExt = "SparseMatricesCSR"

    [deps.cuSPARSE.weakdeps]
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "da8c1f6eee04831f14edcfa5dae611d309807e57"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.3.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╠═c0000001-0000-0000-0000-000000000001
# ╠═20c996f5-d056-4e10-b6fb-0995b444732a
# ╠═b3f6f1ae-066b-4427-a5f6-5512a0919865
# ╠═c0000032-0000-0000-0000-000000000032
# ╠═c0000002-0000-0000-0000-000000000002
# ╠═c0000003-0000-0000-0000-000000000003
# ╠═c0000004-0000-0000-0000-000000000004
# ╠═c000000b-0000-0000-0000-00000000000b
# ╠═c000000c-0000-0000-0000-00000000000c
# ╠═c000000d-0000-0000-0000-00000000000d
# ╠═c000000e-0000-0000-0000-00000000000e
# ╠═c000001a-0000-0000-0000-00000000001a
# ╠═c000001b-0000-0000-0000-00000000001b
# ╠═c000001c-0000-0000-0000-00000000001c
# ╠═c000000f-0000-0000-0000-00000000000f
# ╠═c0000010-0000-0000-0000-000000000010
# ╠═c0000011-0000-0000-0000-000000000011
# ╠═c0000012-0000-0000-0000-000000000012
# ╠═c000001d-0000-0000-0000-00000000001d
# ╠═c000001e-0000-0000-0000-00000000001e
# ╠═c000001f-0000-0000-0000-00000000001f
# ╠═c0000020-0000-0000-0000-000000000020
# ╠═fd5a5b29-7c3f-4fa6-959c-bb34c22c035b
# ╠═c0000016-0000-0000-0000-000000000016
# ╠═c0000017-0000-0000-0000-000000000017
# ╠═c0000018-0000-0000-0000-000000000018
# ╠═c0000019-0000-0000-0000-000000000019
# ╠═c0000030-0000-0000-0000-000000000030
# ╠═c0000013-0000-0000-0000-000000000013
# ╠═c0000014-0000-0000-0000-000000000014
# ╠═c0000015-0000-0000-0000-000000000015
# ╠═c0000021-0000-0000-0000-000000000021
# ╠═c0000022-0000-0000-0000-000000000022
# ╠═c0000023-0000-0000-0000-000000000023
# ╠═c0000024-0000-0000-0000-000000000024
# ╠═c0000025-0000-0000-0000-000000000025
# ╠═c0000026-0000-0000-0000-000000000026
# ╠═c0000027-0000-0000-0000-000000000027
# ╠═c0000028-0000-0000-0000-000000000028
# ╠═c0000029-0000-0000-0000-000000000029
# ╠═c000002a-0000-0000-0000-00000000002a
# ╠═c000002b-0000-0000-0000-00000000002b
# ╠═c000002c-0000-0000-0000-00000000002c
# ╠═c000002d-0000-0000-0000-00000000002d
# ╠═c000002e-0000-0000-0000-00000000002e
# ╠═c000002f-0000-0000-0000-00000000002f
# ╠═c0000031-0000-0000-0000-000000000031
# ╟─c0000033-0000-0000-0000-000000000033
# ╠═c0000034-0000-0000-0000-000000000034
# ╠═c0000035-0000-0000-0000-000000000035
# ╠═c0000036-0000-0000-0000-000000000036
# ╠═a550a648-7a94-11f1-8a00-f3225eb4ac28
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
