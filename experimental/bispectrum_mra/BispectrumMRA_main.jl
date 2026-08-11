### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ c1000001-0000-0000-0000-000000000001
using Revise

# ╔═╡ c1000002-0000-0000-0000-000000000002
begin
    using FFTW, Random, Statistics, LinearAlgebra
    using JLD2
end

# ╔═╡ c1000003-0000-0000-0000-000000000003
using DSP

# ╔═╡ c1000004-0000-0000-0000-000000000004
using PlutoPlotly

# ╔═╡ c1000005-0000-0000-0000-000000000005
using PlutoUI

# ╔═╡ c1000006-0000-0000-0000-000000000006
begin
    using PlutoHooks, PlutoLinks
    using PlutoLinks: @ingredients
end

# ╔═╡ c1000007-0000-0000-0000-000000000007
include(joinpath(@__DIR__, "..", "coherent_n2n", "test", "synthetic_data.jl"))

# ╔═╡ c1000008-0000-0000-0000-000000000008
md"""
# Bispectrum-inversion MRA — alignment-free source recovery

Same problem as the CoherentN2N notebooks: each trace is a jittered, noisy copy
of one unknown source, `yᵢ = T_{ℓᵢ} x + ξᵢ`. **The method is the opposite.**
CoherentN2N estimates the shifts `ℓᵢ` and stacks; that is exactly the step that
breaks down at low SNR, because a mis-estimated lag cannot be stacked away.

Bispectrum MRA never estimates a shift. It reads the source off two
**shift-invariant** moments of the gather:

| moment | degree | recovers |
|---|---|---|
| power spectrum `E|X̂(k)|²` | 2 | Fourier magnitudes `a[k]` |
| bispectrum `E X̂(k₁)X̂(k₂)conj(X̂(k₁+k₂))` | 3 | Fourier phases `φ[k]` |

Recovery is up to a **global circular shift** (unavoidable — every shifted copy
has identical moments), but **not** up to reflection: time reversal maps
`B → conj(B)`, which is measurably different, so orientation is determined.

Grouping here is **one group per station** — every SNR-passing earthquake at
that station, with no distance/backazimuth sub-binning. That maximises the trace
count, which is what this estimator is hungry for (see the sample-complexity note
below). The trade: events spanning all azimuths and distances share a common
source wavelet less cleanly than events within one bin, so per-station event
counts are kept on screen throughout.

⚠️ **Polarity.** The bispectrum is degree 3, so a per-trace sign enters as
`g³ = g` and does *not* cancel. Mixed polarity collapses `B` (measured to
1.7–9.5% of its clean magnitude, worsening with more traces) and makes the
recovered phase untrustworthy — while the power spectrum, being degree 2, still
looks perfectly healthy. Resolve polarity upstream if it is present.
"""

# ╔═╡ c1000009-0000-0000-0000-000000000009
md"## Load algorithm files"

# ╔═╡ c100000a-0000-0000-0000-00000000000a
# The MRA implementation (Revise-tracked). Re-run after ADDING a new top-level
# name to BispectrumMRA.jl.
mra = @ingredients(joinpath(@__DIR__, "BispectrumMRA_lib.jl"))

# ╔═╡ c100000b-0000-0000-0000-00000000000b
# Alignment-free stack baseline — the τ ≡ 0 comparison curve.
#
# Implemented locally on purpose: pulling in CoherentN2N_lib.jl for this would
# drag Flux/CUDA into an otherwise FFTW-only notebook (and fails outright in this
# environment, which has no Flux). What that library's `noalign_baseline` does
# with `stack_type=:l1` is an IRLS-reweighted mean of the raw traces with no
# shift estimation anywhere, which is these few lines.
#
# `:l2` is exactly the plain column mean, so with that setting this curve and the
# "raw mean" curve below MUST coincide — a free consistency check on the plot.
function robust_stack(D::AbstractMatrix{Float32}; stack_type::Symbol=:l1, niter::Int=5)
    s = vec(mean(D, dims=2))
    stack_type === :l2 && return Float32.(s)
    for _ in 1:niter
        # Huber/IRLS: weight each trace by the inverse of its residual norm, so a
        # few wild traces stop dominating the mean.
        r = [sqrt(sum(abs2, @view(D[:, i]) .- s)) for i in 1:size(D, 2)]
        w = 1f0 ./ max.(Float32.(r), 1f-8)
        w ./= sum(w)
        s = vec(D * w)
    end
    return Float32.(s)
end

# ╔═╡ c100000c-0000-0000-0000-00000000000c
norm1(x) = x ./ (maximum(abs, x) + eps(Float32))

# ╔═╡ c100000d-0000-0000-0000-00000000000d
# Best-circular-lag correlation: blind recovery only determines the source up to
# a shift, so every waveform comparison in this notebook goes through this.
function aligned_cor(x̂::AbstractVector, s::AbstractVector)
    a = x̂ ./ (sqrt(sum(abs2, x̂)) + eps(Float32))
    b = s ./ (sqrt(sum(abs2, s)) + eps(Float32))
    cc = real(ifft(conj(fft(Float32.(a))) .* fft(Float32.(b))))
    lag = argmax(abs.(cc)) - 1
    return cor(circshift(x̂, lag), s)
end

# ╔═╡ c100000e-0000-0000-0000-00000000000e
md"""---
## 1. Synthetic control — known source, known jitter

Ground truth is available here, so this section is the sanity check: if MRA
cannot beat a raw mean on synthetic data with a planted wavelet, nothing in the
real-data section below is worth reading.
"""

# ╔═╡ c100000f-0000-0000-0000-00000000000f
md"""Synthetic settings —
noise std: $(@bind noise_syn PlutoUI.Slider(0.0:0.1:2.0; default=0.4, show_value=true)) •
traces R: $(@bind R_syn PlutoUI.Slider(50:50:2000; default=800, show_value=true)) •
nt: $(@bind nt_syn PlutoUI.Slider(64:64:256; default=128, show_value=true)) •
method: $(@bind method_syn PlutoUI.Select([:marching => "frequency marching", :sync => "marching + LS refinement"]; default=:marching))"""

# ╔═╡ c1000010-0000-0000-0000-000000000010
# Unit gains — a random ±1 polarity would break the bispectrum (see the warning
# at the top), so the synthetic control deliberately isolates the shift problem.
begin
    s_true_syn, D_syn, _, τ_syn, _ = make_synthetic_gather(
        nt=nt_syn, R=R_syn, f0=0.05, noise_std=noise_syn,
        rng=MersenneTwister(1), source_kind=:broadband,
        true_gains=ones(ComplexF32, R_syn))
    (nt=nt_syn, R=R_syn, shift_range=extrema(τ_syn))
end

# ╔═╡ c1000011-0000-0000-0000-000000000011
# MRA is two moment sums plus a small phase solve — fast enough to recompute on
# every parameter change, so there is no training button and no stale-result
# failure mode to guard against.
result_syn = mra.recover_mra(D_syn; method=method_syn, sigma2=Float32(noise_syn^2))

# ╔═╡ c1000012-0000-0000-0000-000000000012
let
    ts = 0:nt_syn-1
    raw = vec(mean(D_syn, dims=2))
    c_mra = aligned_cor(result_syn.x, s_true_syn)
    c_raw = aligned_cor(raw, s_true_syn)

    # Align each estimate to the truth for display: recovery is only defined up
    # to a circular shift, so plotting the raw output would show an arbitrary
    # (and distracting) time offset.
    bestlag(x) = argmax([cor(circshift(x, l), s_true_syn) for l in 0:nt_syn-1]) - 1

    traces = [
        PlutoPlotly.scatter(x=collect(ts), y=norm1(s_true_syn), mode="lines",
            name="True wavelet", line=attr(color="black", width=2.5)),
        PlutoPlotly.scatter(x=collect(ts), y=norm1(circshift(raw, bestlag(raw))), mode="lines",
            name="Raw mean (cor $(round(c_raw, digits=4)))",
            line=attr(color="grey", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=collect(ts), y=norm1(circshift(result_syn.x, 0.0)),
            mode="lines", name="Bispectrum MRA (cor $(round(c_mra, digits=4)))",
            line=attr(color="#d62728", width=2)),
    ]
    layout = Layout(
        title=attr(text="Synthetic: MRA vs raw mean vs truth — MRA better? $(c_mra > c_raw ? "✅ yes" : "❌ no")"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Normalized amplitude"),
        height=380, width=900, plot_bgcolor="white", paper_bgcolor="white")
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ c1000013-0000-0000-0000-000000000013
let
    # Phase is the scientific target, so score it directly rather than inferring
    # it from the waveform. The reliable band is where the magnitude actually
    # carries energy; outside it the recovered phase is meaningless by design.
    K = result_syn.kmax
    a = result_syn.a[1:K+1]
    φt = Float32.(angle.(fft(s_true_syn)))[1:K+1]
    keep = findall(k -> a[k+1] >= 0.1 * maximum(a), 0:K)
    wrap(x) = mod(x + π, 2π) - π
    d = [wrap((result_syn.φ[k+1] - result_syn.φ[k]) - (φt[k+1] - φt[k])) for k in 1:K]
    Δ = angle(sum(a[2:K+1] .* cis.(d)))      # circular mean: stable at ±π
    res = [wrap(result_syn.φ[k+1] - φt[k+1] - k * Δ) for k in keep .- 1]
    rms = sqrt(sum(a[keep] .* res .^ 2) / sum(a[keep]))

    ks = collect(0:K)
    traces = [
        PlutoPlotly.scatter(x=ks, y=a ./ maximum(a), mode="lines", name="|a[k]| (normalized)",
            line=attr(color="#1f77b4", width=2)),
        PlutoPlotly.scatter(x=ks[keep], y=abs.(res), mode="markers",
            name="|phase residual| (rad)", marker=attr(color="#d62728", size=5)),
    ]
    layout = Layout(
        title=attr(text="Recovered magnitude and phase error — weighted RMS = $(round(rms, digits=4)) rad (kmax=$K)"),
        xaxis=attr(title="Frequency index k"), yaxis=attr(title="normalized |a| / radians"),
        height=340, width=900, plot_bgcolor="white", paper_bgcolor="white")
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ c1000014-0000-0000-0000-000000000014
md"""---
## 2. Real data — one group per station

Loading is identical to `CoherentN2N_binned_main.jl` (same files, same SNR mask,
same receiver-function window), but **the binning layer is gone**: every
SNR-passing event at a station forms that station's single group.
"""

# ╔═╡ c1000015-0000-0000-0000-000000000015
fldir_real = "/mnt/rengneichuong/rengneichuong/KGrouping/RFData/"

# ╔═╡ c1000016-0000-0000-0000-000000000016
# dn_real = "GSN_150ZTR_Bandpass_0.05_1.5_24jun_rf_wlevel0.01_f2.0_24jun.jld2"
dn_real = "GSN_150ZTR_TaperBandpass_0.01_0.3_8aug_taperbp0.01-0.3_rf_iter300_f0.8.jld2"

# ╔═╡ c1000017-0000-0000-0000-000000000017
snrf_real = "GSN_150ZTR_TaperBandpass_0.01_0.3_8aug_snr.jld2"

# ╔═╡ c1000018-0000-0000-0000-000000000018
# Load file handles / metadata ONCE, independent of the station selection, so the
# picker below can list every available station.
begin
    dfile_real = "$(fldir_real)/$(dn_real)"
    StaName_real = load(dfile_real)["Sta"][1]        # station codes (parallel to Data)
    StaAll_real = load(dfile_real)["Sta"][2]         # station [lat, lon]
    ses_snr_real = load("$(fldir_real)/$(snrf_real)", "SNR")
    Data_real = load(dfile_real)["Data"]
    EventLoc_real = load(dfile_real)["EventLoc"]
    StaName_all_real = unique(StaName_real)
    snr_tres_real = 0.0
    (n_stations_available=length(StaName_all_real),)
end

# ╔═╡ c1000019-0000-0000-0000-000000000019
md"""#### Stations
Each selected station contributes **one group** — all of its SNR-passing events."""

# ╔═╡ c100001a-0000-0000-0000-00000000001a
@bind StaN_list_real PlutoUI.MultiCheckBox(select_all=true, StaName_all_real;
    default=StaName_all_real[1:min(4, length(StaName_all_real))])

# ╔═╡ 509e030b-620a-46b3-8f64-5c4abbba3645
StaN_list_real

# ╔═╡ c100001b-0000-0000-0000-00000000001b
# Per-station gather. Ragged: each station keeps its own SNR-passing events, so
# R_g varies. Same SNR mask and RF-window trim as the CoherentN2N notebooks —
# this is where that notebook would start carving out a distance/azimuth bin, and
# where we simply stop.
begin
    raw_data_list_full = Vector{Matrix{Float64}}()
    StaLoc_list = Vector{Any}()
    EvtLoc_list = Vector{Any}()
    StaN_used = String[]
    for sta in StaN_list_real
        hits = findall(x -> x == sta, StaName_real)
        isempty(hits) && (@warn "Station $sta not found in $(dn_real); skipping"; continue)
        ix = hits[1]
        sel = findall(x -> x > snr_tres_real, ses_snr_real[ix])
        isempty(sel) && (@warn "Station $sta has no SNR-passing events; skipping"; continue)
f1 = 0.01/10.              # lower corner, Hz
f2 = 0.3/10.           # upper corner, Hz

bp = digitalfilter(
    Bandpass(f1, f2),
    Butterworth(2)
)

         # push!(raw_data_list_full, filtfilt(bp, taper_sin_real(randn(1000, 1000)))) 
        push!(raw_data_list_full, Data_real[ix][:, sel][550:750, :])   # trim RF window
        push!(StaLoc_list, StaAll_real[ix])
        push!(EvtLoc_list, EventLoc_real[ix][sel])
        push!(StaN_used, sta)
    end
    @assert !isempty(raw_data_list_full) "No requested station produced any data"
    (n_groups=length(raw_data_list_full),
     events_per_station=[size(d, 2) for d in raw_data_list_full])
end

# ╔═╡ 450e8a79-20ba-4827-a60d-5144932bf4cf
raw_data_list_full

# ╔═╡ c100001c-0000-0000-0000-00000000001c
md"""#### Preprocessing

Taper: $(@bind use_taper PlutoUI.CheckBox(default=false)) apply a Tukey window per trace

⚠️ **Off by default, unlike the CoherentN2N notebooks.** A per-trace taper is
applied *after* the (unknown) shift, so it is **not** circularly shift-equivariant
and it breaks the MRA model. Measured cost on synthetics: correlation
0.954 → 0.868 at σ=0.3, and 0.690 → 0.604 at σ=0.8. Demeaning and dividing by the
per-trace standard deviation are shift-equivariant and are always applied.
"""

# ╔═╡ c100001d-0000-0000-0000-00000000001d
taper_sin_real(x) = cat(DSP.tukey(size(x, 1), 0.3), dims=ndims(x)) .* x

# ╔═╡ c100001e-0000-0000-0000-00000000001e
# One group per station. StaN_used stays 1:1 with groups_sta, so a result index
# maps straight back to a station name with no keep-list indirection.
groups_sta = map(raw_data_list_full) do rd
    mr = mean(rd; dims=1)
    sr = std(rd; dims=1)
    z = (rd .- mr) ./ max.(sr, 1f-8)
    Float32.(use_taper ? taper_sin_real(z) : z)
end

# ╔═╡ c100001f-0000-0000-0000-00000000001f
md"""#### MRA settings

Method: $(@bind method_real PlutoUI.Select([:marching => "frequency marching", :sync => "marching + LS refinement"]; default=:marching)) •
min events/station: $(@bind min_events_real PlutoUI.Slider(2:2:200; default=20, show_value=true))

Band fraction (kmax auto-cut): $(@bind band_frac_real PlutoUI.Select([0.001, 0.01, 0.05, 0.1]; default=0.01)) •
noise σ²: $(@bind sigma2_mode PlutoUI.Select([:auto => "auto — from HF tail", :zero => "none (σ²=0)"]; default=:auto))

Time origin: $(@bind origin_real PlutoUI.Select([
    :center => "centre — energy at sample nt÷2",
    :data => "data frame — match the raw stack",
    :gauge => "raw gauge (φ₀=φ₁=0)"]; default=:center))

The moments are shift-invariant, so the absolute origin is **not recoverable** —
this setting *declares* one rather than estimating it, and never changes the
recovered shape. `centre` depends only on the result, so it adds no error;
`data frame` borrows the gather's own timing but degrades if the raw stack is
incoherent; `raw gauge` is the unshifted inversion output.

Sample size is the dominant accuracy driver: measured mean correlation rises
0.44 → 0.70 going from 100 to 1600 traces at σ=0.8. `min_events` is a floor, not
a recommendation — prefer stations with hundreds of events.
"""

# ╔═╡ c1000020-0000-0000-0000-000000000020
# Computed eagerly — no button. MRA costs O(kmax²·R) in a couple of moment sums,
# so it recomputes on every control change and can never go stale against the data.
result_mra = mra.recover_mra_grouped(groups_sta;
    method=method_real, min_events=min_events_real,
    band_fraction=Float32(band_frac_real),
    origin=origin_real,
    sigma2=(sigma2_mode === :zero ? 0f0 : nothing))

# ╔═╡ c1000021-0000-0000-0000-000000000021
# The comparison curve, one entry per station (1:1 with groups_sta): the
# alignment-free robust stack. Cheap, so computed eagerly like everything else.
result_baseline = [robust_stack(g; stack_type=:l1) for g in groups_sta]

# ╔═╡ c1000022-0000-0000-0000-000000000022
md"#### Station to inspect"

# ╔═╡ c1000029-0000-0000-0000-000000000029
# The dropdown's option list, in its own plain cell.
#
# This indirection is load-bearing. Writing `@bind sta_sel PlutoUI.Select(StaN_used)`
# hides `StaN_used` inside a macro argument, and Pluto then does not always
# register it as a reactive dependency: adding stations left the bond rendering
# its ORIGINAL option list while `groups_sta` had already grown, so the dropdown
# showed 2 entries for an 11-station run. Computing the options here — a normal
# cell with a normal variable reference — makes the dependency explicit, so the
# selector re-renders whenever the station set changes.
sta_options = collect(StaN_used)

# ╔═╡ c1000023-0000-0000-0000-000000000023
# Bind the station NAME, not its position: PlutoUI.Select hands back an opaque
# per-option key, so a previously-held selection does NOT re-map when the option
# list changes. Binding the name makes the selection self-describing, and g_sel
# below resolves it against the CURRENT station list.
begin
	StaN_list_real
@bind sta_sel PlutoUI.Select(sta_options)
end

# ╔═╡ c1000028-0000-0000-0000-000000000028
# Resolve the bound station name against the CURRENT station list. `something`
# handles the transient where the bond still holds a station you just
# unchecked: we fall back to the first one rather than indexing out of bounds.
g_sel = something(findfirst(==(sta_sel), StaN_used), 1)

# ╔═╡ 8430eecb-0e7f-45c1-addf-17d4cb5afc96
plot(heatmap(z=groups_sta[g_sel]))

# ╔═╡ c1000024-0000-0000-0000-000000000024
let
    if !result_mra.valid[g_sel]
        md"⚠️ **$(StaN_used[g_sel])** has only $(size(groups_sta[g_sel], 2)) event(s), below `min_events=$(min_events_real)` — skipped. Lower the threshold or pick another station."
    else
        r = result_mra.groups[g_sel]
        nt = size(groups_sta[g_sel], 1)
        ts = 0:nt-1
        raw = vec(mean(groups_sta[g_sel], dims=2))
        base = result_baseline[g_sel]

        # Every curve is plotted in the data's own sample coordinates — no display
        # circshift. The MRA estimate is put on a declared origin upstream by the
        # `origin` setting, so it is directly comparable to the stacks instead of
        # floating in the inversion's arbitrary phase gauge. The correlations
        # quoted below are still best-lag, hence shift-blind either way.
        traces = [
            PlutoPlotly.scatter(x=collect(ts), y=norm1(r.x), mode="lines",
                name="Bispectrum MRA", line=attr(color="#d62728", width=2)),
            PlutoPlotly.scatter(x=collect(ts), y=norm1(base), mode="lines",
                name="Robust stack, no alignment (cor $(round(aligned_cor(base, r.x), digits=3)))",
                line=attr(color="#1f77b4", width=1.6)),
            PlutoPlotly.scatter(x=collect(ts), y=norm1(raw), mode="lines",
                name="Raw mean (cor $(round(aligned_cor(raw, r.x), digits=3)))",
                line=attr(color="grey", width=1.4, dash="dash")),
        ]
        layout = Layout(
            title=attr(text="$(StaN_used[g_sel]) — $(size(groups_sta[g_sel], 2)) events, kmax=$(r.kmax)"),
            xaxis=attr(title="Sample"), yaxis=attr(title="Normalized amplitude"),
            height=380, width=900, plot_bgcolor="white", paper_bgcolor="white")
        WideCell(PlutoPlotly.plot(traces, layout))
    end
end

# ╔═╡ c1000025-0000-0000-0000-000000000025
let
    if !result_mra.valid[g_sel]
        md"—"
    else
        r = result_mra.groups[g_sel]
        K = r.kmax
        a = r.a[1:K+1]
        ks = collect(0:K)
        thr = 0.1 * maximum(a)
        # Phase is the target, so show it directly alongside the magnitude that
        # determines where it is trustworthy.
        traces = [
            PlutoPlotly.scatter(x=ks, y=a ./ maximum(a), mode="lines",
                name="|a[k]| (normalized)", line=attr(color="#1f77b4", width=2)),
            PlutoPlotly.scatter(x=ks, y=fill(thr / maximum(a), K + 1), mode="lines",
                name="reliable-band threshold", line=attr(color="grey", width=1, dash="dot")),
            PlutoPlotly.scatter(x=ks, y=r.φ ./ π, mode="lines",
                name="φ[k] / π", line=attr(color="#d62728", width=1.6), yaxis="y2"),
        ]
        layout = Layout(
            title=attr(text="$(StaN_used[g_sel]) — recovered magnitude and phase spectrum"),
            xaxis=attr(title="Frequency index k"),
            yaxis=attr(title="normalized |a[k]|"),
            yaxis2=attr(title="φ[k] / π", overlaying="y", side="right"),
            height=340, width=900, plot_bgcolor="white", paper_bgcolor="white")
        PlutoPlotly.plot(traces, layout)
    end
end

# ╔═╡ c1000026-0000-0000-0000-000000000026
md"""---
## 3. Diagnostics
"""

# ╔═╡ c1000027-0000-0000-0000-000000000027
let
    rows = ["| station | events | valid | kmax | σ̂² |", "|---|---|---|---|---|"]
    for i in 1:length(StaN_used)
        r = result_mra.groups[i]
        push!(rows, "| $(StaN_used[i]) | $(size(groups_sta[i], 2)) | $(result_mra.valid[i] ? "✅" : "⛔ skipped") | $(r.kmax) | $(round(r.sigma2, digits=5)) |")
    end
    Markdown.parse(join(rows, "\n"))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DSP = "~0.8.6"
FFTW = "~1.10.0"
JLD2 = "~0.6.5"
PlutoHooks = "~0.1.0"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.6"
PlutoUI = "~0.7.83"
Revise = "~3.16.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "a2ca64af367082e6f344172c21b324aa845c7f13"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

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

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "a65cfc2999988f5ba09fc4bd8049e3ed914e5a04"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.6"

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

    [deps.DSP.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

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
git-tree-sha1 = "6621fef488e496356c9c9625d0562c12a6070819"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.20.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

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

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

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
git-tree-sha1 = "c3d401f110454b4ea24a76be33f6ee0d7d385103"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.11.4"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

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
git-tree-sha1 = "1d4c737ab26f51ceed52ab2019c09b7660eb7440"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.8.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

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

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "Setfield", "SparseArrays"]
git-tree-sha1 = "2d99b4c8a7845ab1342921733fa29366dae28b24"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.1"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"
    PolynomialsRecipesBaseExt = "RecipesBase"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

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
deps = ["CRC32c", "CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "838f84266bf2e9ca4b7d0b8965807c7a72745501"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.16.1"

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

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

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "6547cbdd8ce32efba0d21c5a40fa96d1a3548f9f"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.8.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

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

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

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
# ╟─c1000008-0000-0000-0000-000000000008
# ╟─c1000009-0000-0000-0000-000000000009
# ╠═c1000001-0000-0000-0000-000000000001
# ╠═c1000002-0000-0000-0000-000000000002
# ╠═c1000003-0000-0000-0000-000000000003
# ╠═c1000004-0000-0000-0000-000000000004
# ╠═c1000005-0000-0000-0000-000000000005
# ╠═c1000006-0000-0000-0000-000000000006
# ╠═c1000007-0000-0000-0000-000000000007
# ╠═c100000a-0000-0000-0000-00000000000a
# ╠═c100000b-0000-0000-0000-00000000000b
# ╠═c100000c-0000-0000-0000-00000000000c
# ╠═c100000d-0000-0000-0000-00000000000d
# ╟─c100000e-0000-0000-0000-00000000000e
# ╟─c100000f-0000-0000-0000-00000000000f
# ╠═c1000010-0000-0000-0000-000000000010
# ╠═c1000011-0000-0000-0000-000000000011
# ╠═c1000012-0000-0000-0000-000000000012
# ╟─c1000013-0000-0000-0000-000000000013
# ╟─c1000014-0000-0000-0000-000000000014
# ╠═c1000015-0000-0000-0000-000000000015
# ╠═c1000016-0000-0000-0000-000000000016
# ╠═c1000017-0000-0000-0000-000000000017
# ╠═c1000018-0000-0000-0000-000000000018
# ╟─c1000019-0000-0000-0000-000000000019
# ╟─c100001a-0000-0000-0000-00000000001a
# ╠═509e030b-620a-46b3-8f64-5c4abbba3645
# ╠═c100001b-0000-0000-0000-00000000001b
# ╠═450e8a79-20ba-4827-a60d-5144932bf4cf
# ╠═8430eecb-0e7f-45c1-addf-17d4cb5afc96
# ╟─c100001c-0000-0000-0000-00000000001c
# ╠═c100001d-0000-0000-0000-00000000001d
# ╠═c100001e-0000-0000-0000-00000000001e
# ╟─c100001f-0000-0000-0000-00000000001f
# ╠═c1000020-0000-0000-0000-000000000020
# ╠═c1000021-0000-0000-0000-000000000021
# ╟─c1000022-0000-0000-0000-000000000022
# ╟─c1000029-0000-0000-0000-000000000029
# ╠═c1000023-0000-0000-0000-000000000023
# ╟─c1000028-0000-0000-0000-000000000028
# ╠═c1000024-0000-0000-0000-000000000024
# ╟─c1000025-0000-0000-0000-000000000025
# ╟─c1000026-0000-0000-0000-000000000026
# ╟─c1000027-0000-0000-0000-000000000027
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
