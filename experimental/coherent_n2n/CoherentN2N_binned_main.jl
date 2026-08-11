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

# ╔═╡ b1000001-0000-0000-0000-000000000001
using Revise

# ╔═╡ b1000002-0000-0000-0000-000000000002
begin
    using CUDA, cuDNN, Flux, Zygote, FFTW
    using Optimisers, Random, Statistics, LinearAlgebra
    using JLD2
    CUDA.device!(0)  # NVIDIA RTX 6000 Ada Generation
end

# ╔═╡ b1000003-0000-0000-0000-000000000003
using DSP

# ╔═╡ b1000004-0000-0000-0000-000000000004
begin
    # Silence cuDNN 9.x's verbose engine-selection tracebacks (harmless
    # "Dilation not supported" info logs it emits when it skips a preferred
    # backward-conv engine for our dilated 1-D convs). Must be set BEFORE
    # `using cuDNN` loads the C library.
    using Base: C_NULL
    cuDNN.cudnnSetCallback(0, C_NULL, C_NULL)
    "cuDNN callback disabled"
end

# ╔═╡ b1000005-0000-0000-0000-000000000005
using PlutoPlotly

# ╔═╡ b1000006-0000-0000-0000-000000000006
using PlutoUI

# ╔═╡ b1000007-0000-0000-0000-000000000007
begin
    using PlutoHooks, PlutoLinks
    using PlutoLinks: @ingredients
end

# ╔═╡ b1000008-0000-0000-0000-000000000008
using ProgressLogging

# ╔═╡ b1000009-0000-0000-0000-000000000009
include(joinpath(@__DIR__, "test", "synthetic_data.jl"))

# ╔═╡ b100000a-0000-0000-0000-00000000000a
md"""# CoherentN2N — Binned, Alignment-Free

A variant of `CoherentN2N_main.jl` that removes the **alignment degree of
freedom** entirely.

In the main notebook, each station's group holds *all* of its SNR-passing
earthquakes — thousands of receiver functions spanning the whole
epicentral-distance / backazimuth range. Those events genuinely arrive at
different times, so the alternating loop has to solve for a per-event shift τ
(Block B) at the same time as it trains the denoiser. That works, but it
confounds two things: *did the denoiser learn something*, or *did the aligner
converge*?

Here we remove the ambiguity by construction. Each station is restricted to a
single **10° × 10° epicentral-distance × backazimuth bin**, chosen automatically
as that station's densest cell. Events inside one such cell share essentially the
same ray parameter and incidence geometry, so they are already coherent — no
shift needed. With `τ ≡ 0` there is no Block B, no gauge, and no shift prior:
whatever improvement shows up in ŝ is attributable to the denoiser alone.

## Workflow
1. Load the algorithm files (below)
2. **Synthetic control** — a zero-jitter gather with a known planted wavelet.
   The no-align loop must recover it and beat the plain stack. Run this first.
3. **Real data, binned** — pick stations, auto-select each one's densest bin
   (inspect it on the polar plot), then train one shared denoiser across bins.

## Files loaded
- `CoherentN2N_binning.jl` — epicentral distance / backazimuth geometry and the
  `densest_bin` / `bin_member_indices` cell selection **(new for this notebook)**
- `CoherentN2N_denoiser.jl`, `CoherentN2N_n2n_pairs.jl`,
  `CoherentN2N_train_denoiser.jl` — the denoiser and its N2N training (shared
  with the main notebook, unchanged)
- `CoherentN2N_outer_loop.jl` — `run_coherent_n2n_grouped_noalign` and
  `run_coherent_n2n_grouped_noalign_baseline` **(new)**

`CoherentN2N_shift.jl` / `CoherentN2N_gauge.jl` are loaded by the library but
deliberately **unused** here — they are the alignment machinery this notebook
exists to do without.
"""

# ╔═╡ b100000b-0000-0000-0000-00000000000b
md"## Load architecture / algorithm files"

# ╔═╡ b100000c-0000-0000-0000-00000000000c
# Single @ingredients call over CoherentN2N_lib.jl (Revise-tracked, so edits to
# any included file are picked up automatically). Re-run this cell after ADDING a
# new top-level name to an included file.
cn = @ingredients(joinpath(@__DIR__, "CoherentN2N_lib.jl"))  # reload5 (bilinear upsample)

# ╔═╡ b100000d-0000-0000-0000-00000000000d
md"""---
## 0. Architecture probe — white noise in, where does energy land?

Before any training, feed the **untrained** network white Gaussian noise (whose
spectrum is flat, so no frequency or time position is special) and measure the
per-sample RMS of its output in the time domain.

A position-neutral operator must give a **flat** profile. Any structure here is
pure architecture bias — it has nothing to do with the data or the loss, and it
will contaminate every ŝ the network ever produces.

This matters especially for the half-spectrum denoiser: the conv stack runs over
frequency bins `1 … nt÷2+1`, where bin 1 is DC and the last bin is Nyquist.
Convolving across that axis has to do *something* at those two boundaries, and an
error in the lowest frequencies appears in the time domain as a slow trend across
the window — i.e. as apparent energy at the trace edges.
"""

# ╔═╡ b100000e-0000-0000-0000-00000000000e
md"""Probe settings — noise realizations: $(@bind probe_B PlutoUI.Slider(16:16:256; default=64, show_value=true)) •
nt: $(@bind probe_nt PlutoUI.Slider(100:100:400; default=300, show_value=true)) •
edge width: $(@bind probe_edge PlutoUI.Slider(5:5:40; default=10, show_value=true)) samples"""

# ╔═╡ b100000f-0000-0000-0000-00000000000f
let
    Random.seed!(11)
    m_probe = cn.build_complex_denoiser(probe_nt; kernels=[32, 16, 8], filters=[16, 32, 64])
    D = randn(Float32, probe_nt, probe_B)
    Y = m_probe(ComplexF32.(fft(D, 1)))
    yt = real(ifft(Y, 1))
    rms = vec(sqrt.(mean(abs2, yt; dims=2)))
    rms ./= mean(rms)                      # 1.0 == perfectly flat

    e = probe_edge
    ctr = mean(rms[(e + 1):(end - e)])
    edge = (mean(rms[1:e]) + mean(rms[(end - e + 1):end])) / 2
    ratio = edge / ctr

    tr = PlutoPlotly.scatter(x=collect(1:probe_nt), y=rms, mode="lines",
                             name="per-sample RMS", line=attr(color="#d62728", width=1.5))
    flat = PlutoPlotly.scatter(x=[1, probe_nt], y=[1.0, 1.0], mode="lines",
                               name="flat (ideal)", line=attr(color="black", width=1.5, dash="dash"))
    layout = Layout(
        title=attr(text="Untrained network, white-noise in — edge/centre RMS ratio $(round(ratio, digits=2)) $(ratio > 1.5 ? "⚠️ biased" : "✅ flat")"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Normalized RMS", type="log"),
        height=340, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot([tr, flat], layout))
end

# ╔═╡ b1000010-0000-0000-0000-000000000010
md"""---
## 1. Synthetic control — no time jitter

Ground truth: one broadband wavelet, planted identically in every column
(`true_shifts = 0`), plus independent noise. This is exactly the situation a
distance/azimuth bin is supposed to produce, so it is the honest test of the
alignment-free loop: **ŝ must beat the plain stack** of the same gather.
"""

# ╔═╡ b1000011-0000-0000-0000-000000000011
md"""#### ⚙️ Synthetic controls

| control | |
|---|---|
| **Noise std** | $(@bind noise_std_syn PlutoUI.Slider(0.1:0.1:1.5; default=0.4, show_value=true)) (wavelet RMS is only ~0.1, so 0.4 is already ~4:1 noise-to-signal per trace) |
| **Traces** R | $(@bind R_syn PlutoUI.Slider(20:20:400; default=200, show_value=true)) |
| **Outer iterations** | $(@bind ni_syn PlutoUI.Slider(1:20; default=4, show_value=true)) |
| **Epochs / iteration** | $(@bind nepoch_syn PlutoUI.Slider(10:10:200; default=80, show_value=true)) |
| **Coherent stack** (`stack_type`) | $(@bind stack_type_syn PlutoUI.Select([:l2 => "L2 — mean stack", :l1 => "L1 — robust Huber/IRLS"]; default=:l2)) |
| **Denoiser N2N loss** | $(@bind denoiser_loss_syn PlutoUI.Select([:l2 => "L2 / MSE", :l1 => "L1 / mean-abs"]; default=:l2)) |

Every shift-related control from the main notebook (`max_shift`, shift prior,
polarity gain, stochastic references) is **gone** — none of it applies when
τ ≡ 0.
"""

# ╔═╡ b1000013-0000-0000-0000-000000000013
md"### Raw synthetic gather (coherent by construction, noisy)"

# ╔═╡ b1000017-0000-0000-0000-000000000017
# Only the fields the alignment-free loop actually reads: n_outer_iters,
# stack_type, denoiser_training. Setting any shift field here would just trigger
# the loop's "ignored" warning.
outer_para_syn = cn.CoherentN2N_Outer_Para(
    n_outer_iters=ni_syn,
    stack_type=stack_type_syn,
    denoiser_training=cn.CoherentN2N_Denoiser_Training_Para(
        n_samples_per_epoch=256, batchsize=200, nepoch=nepoch_syn,
        initial_lr=0.003, restart_period=40, nprint=20,
        denoiser_loss_type=denoiser_loss_syn))

# ╔═╡ b1000018-0000-0000-0000-000000000018
train_syn_button = @bind train_syn_click PlutoUI.CounterButton("Train synthetic (no-align N2N)")

# ╔═╡ b100001a-0000-0000-0000-00000000001a
md"### Ground-truth comparison — the pass/fail criterion"

# ╔═╡ b1000020-0000-0000-0000-000000000020
md"""---
## 2. Real data — one distance/azimuth bin per station

Same JLD2 convention as `CoherentN2N_main.jl`: `Data[station_index]` →
`(nt, n_events)`, with parallel `Sta` / `EventLoc` arrays and a companion SNR
file. The difference is what happens after loading: instead of handing a
station's whole earthquake set to the loop, we keep only the events inside that
station's densest 10°×10° distance/backazimuth cell.
"""

# ╔═╡ b1000021-0000-0000-0000-000000000021
fldir_real = "/mnt/rengneichuong/rengneichuong/KGrouping/RFData"

# ╔═╡ b1000022-0000-0000-0000-000000000022
dn_real = "GSN_150ZTR_Bandpass_0.05_1.5_24jun_rf_wlevel0.01_f2.0_24jun.jld2"

# ╔═╡ b1000023-0000-0000-0000-000000000023
snrf_real = "GSN_150ZTR_Bandpass_0.01_0.3_24july_snr.jld2"

# ╔═╡ b1000024-0000-0000-0000-000000000024
# Load the JLD2 file handles / metadata ONCE, independent of which receivers are
# selected, so the receiver picker below can list every available station.
begin
    dfile_real = "$(fldir_real)/$(dn_real)"
    StaName_real = load(dfile_real)["Sta"][1]        # station codes (parallel to Data)
    StaAll_real = load(dfile_real)["Sta"][2]         # station [lat, lon] coords
    ses_snr_real = load("$(fldir_real)/$(snrf_real)", "SNR")
    Data_real = load(dfile_real)["Data"]
    EventLoc_real = load(dfile_real)["EventLoc"]
    StaName_all_real = unique(StaName_real)          # picker options
    snr_tres_real = 0.0
    (n_stations_available=length(StaName_all_real),)
end

# ╔═╡ b1000025-0000-0000-0000-000000000025
md"""#### Stations to consider
Each selected station contributes **one bin** (its densest cell) as one group."""

# ╔═╡ b1000026-0000-0000-0000-000000000026
@bind StaN_list_real PlutoUI.MultiCheckBox(StaName_all_real;
                                           default=StaName_all_real, select_all=true)

# ╔═╡ b1000036-0000-0000-0000-000000000036
train_binned_button = @bind train_binned_click PlutoUI.CounterButton("Train bins (no-align N2N)")

# ╔═╡ b1000027-0000-0000-0000-000000000027
md"""#### 📍 Binning controls

| control | |
|---|---|
| **Bin size** | $(@bind bin_size_deg PlutoUI.Slider(5:5:30; default=10, show_value=true))° × the same in backazimuth |
| **Min events per bin** | $(@bind min_bin_events PlutoUI.Slider(2:2:200; default=20, show_value=true)) — stations whose densest bin is sparser than this are dropped |

**Override the bin for the station selected below** (leave off to use the
automatically chosen densest cell):

| | |
|---|---|
| **Override** | $(@bind override_bin PlutoUI.CheckBox(default=false)) enable |
| **Distance start** | $(@bind override_dist_start PlutoUI.Slider(0:5:175; default=60, show_value=true))° |
| **Backazimuth start** | $(@bind override_baz_start PlutoUI.Slider(0:5:355; default=140, show_value=true))° |

The override applies only to the station you have selected on the map, so you can
inspect one station's alternative bin without disturbing the others.
"""

# ╔═╡ b1000028-0000-0000-0000-000000000028
# Per-station gather + metadata, BEFORE binning. Ragged: each station keeps its
# own SNR-passing events (R_g varies). Same SNR mask and RF-window trim as the
# main notebook. This is the pool the bin is then carved out of.
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
        push!(raw_data_list_full, Data_real[ix][:, sel][451:850,:])  # trim RF window
        push!(StaLoc_list, StaAll_real[ix])
        push!(EvtLoc_list, EventLoc_real[ix][sel])                    # 1:1 with columns
        push!(StaN_used, sta)
    end
    @assert !isempty(raw_data_list_full) "No requested station produced any data"

    (n_stations=length(raw_data_list_full), R_per_station=size.(raw_data_list_full, 2),
     nt=size(raw_data_list_full[1], 1))
end

# ╔═╡ b1000029-0000-0000-0000-000000000029
md"""### Receiver map — click a station to inspect its bin
All per-station plots below (including the polar bin plot) follow this selection."""

# ╔═╡ b100002a-0000-0000-0000-00000000002a
# Clickable receiver map: a plotly_click listener writes the clicked point's
# 1-based index into the cell's bound value, so g_sel updates reactively.
@bind g_sel_click let
    lats = [loc[1] for loc in StaLoc_list]
    lons = [loc[2] for loc in StaLoc_list]
    tr = PlutoPlotly.scattergeo(
        lat=lats, lon=lons, text=StaN_used, mode="markers+text",
        textposition="top center",
        marker=attr(size=10, color="#d62728", line=attr(width=1, color="white")),
    )
    layout = Layout(
        title=attr(text="Receivers — click a marker to select its station"),
        geo=attr(projection=attr(type="natural earth"), showland=true,
                 landcolor="rgb(240,240,240)", showcountries=true,
                 countrycolor="rgb(200,200,200)", showcoastlines=true),
        height=450, width=900, margin=attr(l=0, r=0, t=40, b=0),
    )
    p = PlutoPlotly.plot([tr], layout)
    PlutoPlotly.add_plotly_listener!(p, "plotly_click", PlutoPlotly.htl_js("""
    function(e) {
        // 1-based Julia index of the clicked receiver marker.
        PLOT.value = e.points[0].pointIndex + 1;
        PLOT.dispatchEvent(new CustomEvent("input"));
    }
    """))
    p
end

# ╔═╡ b100002b-0000-0000-0000-00000000002b
g_sel = (g_sel_click === missing || g_sel_click === nothing) ? 1 :
        clamp(Int(g_sel_click), 1, length(raw_data_list_full))

# ╔═╡ b100002c-0000-0000-0000-00000000002c
# Bin assignment: for each station pick the densest bin_size° x bin_size° cell,
# except the selected station when the override is on. Stations whose best cell
# is sparser than min_bin_events are dropped from training (binned_keep records
# which survived, so result.groups[k] traces back to station binned_keep[k]).
begin
    bin_cell_list = Vector{Union{Nothing,Tuple{Int,Int}}}(undef, length(raw_data_list_full))
    bin_idx_list = [Int[] for _ in 1:length(raw_data_list_full)]
    for g in 1:length(raw_data_list_full)
        cell, idx = if override_bin && g == g_sel
            c = (floor(Int, override_dist_start / bin_size_deg),
                 floor(Int, mod(override_baz_start, 360) / bin_size_deg))
            (c, cn.bin_member_indices(StaLoc_list[g], EvtLoc_list[g], c;
                                      bin_size=Float64(bin_size_deg)))
        else
            best = cn.densest_bin(StaLoc_list[g], EvtLoc_list[g];
                                  bin_size=Float64(bin_size_deg), min_events=min_bin_events)
            best === nothing ? (nothing, Int[]) : (best.cell, best.idx)
        end
        bin_cell_list[g] = cell
        bin_idx_list[g] = idx
    end

    # Groups actually trained on: a bin needs >= min_bin_events (and >= 2 to form
    # an N2N pair at all).
    binned_keep = [g for g in 1:length(raw_data_list_full)
                   if bin_cell_list[g] !== nothing && length(bin_idx_list[g]) >= max(min_bin_events, 2)]
    for g in 1:length(raw_data_list_full)
        g in binned_keep && continue
        @warn "Station $(StaN_used[g]): densest $(bin_size_deg)°×$(bin_size_deg)° bin has only $(length(bin_idx_list[g])) event(s) (< min_bin_events=$min_bin_events); excluded from training"
    end
    @assert !isempty(binned_keep) "No station has a bin with >= $min_bin_events events — lower min_bin_events or raise bin_size"

    (n_stations_binned=length(binned_keep),
     R_per_bin=[length(bin_idx_list[g]) for g in binned_keep],
     bins=[(StaN_used[g], cn.bin_ranges(bin_cell_list[g], Float64(bin_size_deg)))
           for g in binned_keep])
end

# ╔═╡ b100002d-0000-0000-0000-00000000002d
md"""### 🎯 Selected bin — epicentral distance × backazimuth

Grey = all of this station's events; coloured = the events inside its bin; the
shaded wedge is the bin itself. **Click a different station on the map above** to
move the wedge. This renders as soon as the bins are computed — no training
required."""

# ╔═╡ b100002e-0000-0000-0000-00000000002e
let
    b = cn.event_bins(StaLoc_list[g_sel], EvtLoc_list[g_sel]; bin_size=Float64(bin_size_deg))
    idx = bin_idx_list[g_sel]
    cell = bin_cell_list[g_sel]
    in_bin = falses(length(b.dist))
    in_bin[idx] .= true

    traces = PlutoPlotly.GenericTrace[
        PlutoPlotly.scatterpolar(
            r=b.dist[.!in_bin], theta=b.baz[.!in_bin], mode="markers",
            name="outside bin ($(count(.!in_bin)))",
            marker=attr(size=4, color="rgba(140,140,140,0.45)")),
    ]

    if cell !== nothing
        r = cn.bin_ranges(cell, Float64(bin_size_deg))
        d0, d1 = r.dist_range
        b0, b1 = r.baz_range
        # Draw the cell as a closed wedge: the two radial edges are straight, the
        # two azimuthal edges are arcs, so sample θ along them (a 2-point edge
        # would render as a chord and misrepresent the bin).
        θs = collect(range(b0, b1; length=24))
        # inner arc b0->b1 at d0, radial edge out to d1, outer arc b1->b0, close.
        wedge_r = vcat(fill(d0, length(θs)), fill(d1, length(θs)), [d0])
        wedge_θ = vcat(θs, reverse(θs), [b0])
        push!(traces, PlutoPlotly.scatterpolar(
            r=wedge_r, theta=wedge_θ, mode="lines", fill="toself",
            name="bin $(Int(d0))–$(Int(d1))° / $(Int(b0))–$(Int(b1))°",
            line=attr(color="#1f77b4", width=2),
            fillcolor="rgba(31,119,180,0.15)"))
        push!(traces, PlutoPlotly.scatterpolar(
            r=b.dist[in_bin], theta=b.baz[in_bin], mode="markers",
            name="in bin ($(length(idx)))",
            marker=attr(size=7, color="#d62728",
                        line=attr(width=0.5, color="white"))))
    end

    src = cell === nothing ? "no bin met min_bin_events" :
          (override_bin ? "manual override" : "densest cell")
    layout = Layout(
        title=attr(text="$(StaN_used[g_sel]) — $(bin_size_deg)°×$(bin_size_deg)° bin ($(src)): $(length(idx)) of $(length(b.dist)) events"),
        polar=attr(
            radialaxis=attr(title="Epicentral distance (°)"),
            angularaxis=attr(direction="clockwise", rotation=90),  # 0=N at top, compass
        ),
        height=600, width=700, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ b100002f-0000-0000-0000-00000000002f
"""Per-trace normalize + taper (same convention as the main notebook)."""
function taper_sin_real(x)
    # return x
    w = cat(DSP.tukey(size(x, 1), 0.2), dims=ndims(x))
    return w .* x
end

# ╔═╡ b1000012-0000-0000-0000-000000000012
begin
    rng_syn = MersenneTwister(1)
    nt_syn = 128
    # NO time jitter — the entire premise of this notebook. Every column carries
    # the SAME wavelet at the SAME time; only the noise differs.
    τ_true_syn = zeros(Float32, R_syn)
    g_true_syn = ones(ComplexF32, R_syn)   # no polarity flips either

    s_true_syn, D_syn, freqs_syn, _, _ = make_synthetic_gather(
        nt=nt_syn, R=R_syn, f0=0.05, source_kind=:broadband,
        true_shifts=τ_true_syn, true_gains=g_true_syn,
        noise_std=noise_std_syn, rng=rng_syn)
    D_syn = Float32.(taper_sin_real(D_syn))
end

# ╔═╡ b1000014-0000-0000-0000-000000000014
let
    tr = PlutoPlotly.heatmap(z=D_syn, colorscale="RdBu", zmid=0)
    layout = Layout(
        title=attr(text="Raw synthetic gather — $(R_syn) traces, zero jitter, noise std $(noise_std_syn)"),
        xaxis=attr(title="Trace index"), yaxis=attr(title="Sample"),
        height=400, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot([tr], layout))
end

# ╔═╡ b1000015-0000-0000-0000-000000000015
let
    ts = 1:nt_syn
    n_show = min(8, R_syn)
    traces = [
        PlutoPlotly.scatter(x=collect(ts), y=D_syn[:, r] .+ 3 * (r - 1), mode="lines",
            name="trace $r", line=attr(width=1.2))
        for r in 1:n_show
    ]
    layout = Layout(
        title=attr(text="Raw synthetic traces (first $(n_show), offset for visibility) — all at the same time origin"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Amplitude (offset)", showticklabels=false),
        height=450, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ b1000016-0000-0000-0000-000000000016
para_syn = cn.CoherentN2N_Para(nt=nt_syn, kernels=[16, 8], filters=[8, 16], use_gpu=true)

# ╔═╡ b1000019-0000-0000-0000-000000000019
# Button-gated: CounterButton starts at 0 so nothing runs on load. A single
# one-element group vector exercises the grouped code path on one gather.
result_syn = @use_memo([train_syn_click]) do
    train_syn_click == 0 ? nothing :
    cn.run_coherent_n2n_grouped_noalign([D_syn], para_syn, outer_para_syn)
end

# ╔═╡ b100001c-0000-0000-0000-00000000001c
result_syn === nothing ? md"⏳ Click **Train synthetic** to populate this." : let
    # τ is structurally zero here, so there is no Δτ trace to plot — only the
    # denoiser's own loss and how much ŝ still moves between outer iterations.
    curves = result_syn.history.denoiser_loss
    traces = PlutoPlotly.GenericTrace[]
    for (i, c) in enumerate(curves)
        push!(traces, PlutoPlotly.scatter(x=collect(1:length(c)), y=c, mode="lines",
                                          name="outer iter $i", line=attr(width=1.5)))
    end
    layout = Layout(
        title=attr(text="Synthetic denoiser N2N loss (one curve per outer iteration)"),
        xaxis=attr(title="Epoch"), yaxis=attr(title="Loss", type="log"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ b100001d-0000-0000-0000-00000000001d
result_syn === nothing ? md"⏳ Click **Train synthetic** to populate this." : let
    ds = result_syn.history.delta_s[1]
    tr = PlutoPlotly.scatter(x=collect(1:length(ds)), y=ds, mode="lines+markers",
                             name="Δŝ", line=attr(color="#1f77b4", width=2))
    layout = Layout(
        title=attr(text="Synthetic outer-loop convergence — ‖Δŝ‖ per iteration (Δτ is identically 0)"),
        xaxis=attr(title="Outer iteration"), yaxis=attr(title="‖Δŝ‖"),
        height=320, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot([tr], layout))
end

# ╔═╡ b100001b-0000-0000-0000-00000000001b
result_syn === nothing ? md"⏳ Click **Train synthetic** to populate this." : let
    ts = 1:nt_syn
    norm1(x) = x ./ (maximum(abs, x) + eps(Float32))
    ŝ_time = real(ifft(result_syn.groups[1].ŝ))
    raw_time = vec(mean(D_syn; dims=2))
    # No best-lag alignment anywhere: with τ ≡ 0 the recovered ŝ lives in the
    # SAME time frame as the planted wavelet, so correlations are directly
    # comparable (unlike the aligned loops, where ŝ is only defined up to a shift).
    c_n2n = cor(ŝ_time, s_true_syn)
    c_raw = cor(raw_time, s_true_syn)

    traces = [
        PlutoPlotly.scatter(x=collect(ts), y=norm1(s_true_syn), mode="lines",
            name="True wavelet", line=attr(color="black", width=2.5)),
        PlutoPlotly.scatter(x=collect(ts), y=norm1(raw_time), mode="lines",
            name="Raw stack (cor $(round(c_raw, digits=4)))",
            line=attr(color="grey", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=collect(ts), y=norm1(ŝ_time), mode="lines",
            name="N2N ŝ (cor $(round(c_n2n, digits=4)))",
            line=attr(color="#d62728", width=2)),
    ]
    layout = Layout(
        title=attr(text="Synthetic: N2N ŝ vs raw stack vs truth — N2N better? $(c_n2n > c_raw ? "✅ yes" : "❌ no")"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Normalized amplitude"),
        height=380, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ b1000030-0000-0000-0000-000000000030
# The training set: one group per SURVIVING station, holding only that station's
# in-bin columns. Same demean / divide-by-std / taper recipe as the main notebook.
groups_binned = map(binned_keep) do g
    rd = raw_data_list_full[g][:, bin_idx_list[g]]
    mr = mean(rd; dims=1)
    sr = std(rd; dims=1)
    Float32.(taper_sin_real((rd .- mr) ./ max.(sr, 1f-8)))
end

# ╔═╡ b1000032-0000-0000-0000-000000000032
let
    k = findfirst(==(g_sel), binned_keep)
    k === nothing ? md"⚠️ Station **$(StaN_used[g_sel])** has no bin meeting `min_bin_events` — it is excluded from training. Pick another station, lower the threshold, or raise the bin size." : let
        tr = PlutoPlotly.heatmap(z=groups_binned[k], colorscale="RdBu", zmid=0)
        layout = Layout(
            title=attr(text="Raw binned gather — $(StaN_used[g_sel]), $(size(groups_binned[k], 2)) events in bin"),
            xaxis=attr(title="Event index (within bin)"), yaxis=attr(title="Sample"),
            height=400, width=900, plot_bgcolor="white", paper_bgcolor="white",
        )
        WideCell(PlutoPlotly.plot([tr], layout))
    end
end

# ╔═╡ b1000031-0000-0000-0000-000000000031
md"### Raw binned gather (selected station)"

# ╔═╡ b1000033-0000-0000-0000-000000000033
md"""#### Training controls
Outer iters: $(@bind ni_real PlutoUI.Slider(1:20; default=1, show_value=true))  •
Epochs/iter: $(@bind nepoch_real PlutoUI.Slider(1:10:400; default=10, show_value=true))

Batch size: $(@bind bs_real PlutoUI.Slider(16:16:2048; default=512, show_value=true))  •
Samples/epoch (per group): $(@bind nspe_real PlutoUI.Slider(64:64:1024; default=512, show_value=true))

Restart period: $(@bind rp_real PlutoUI.Slider(10:10:200; default=50, show_value=true))  •
Initial LR: $(@bind lr_real PlutoUI.Select([0.0003, 0.001, 0.003, 0.01]; default=0.001))

Coherent stack (`stack_type`): $(@bind stack_type_real PlutoUI.Select([:l2 => "L2 — mean", :l1 => "L1 — robust Huber/IRLS"]; default=:l2))  •
Denoiser loss: $(@bind denoiser_loss_real PlutoUI.Select([:l2 => "L2 / MSE", :l1 => "L1 / mean-abs"]; default=:l2))

---
**Denoiser domain**: $(@bind denoiser_domain PlutoUI.Select([:time => "Time — Wave-U-Net on waveforms", :freq => "Frequency — complex mask"]; default=:time))

*Time* denoises the waveforms directly (no FFT anywhere) with the Wave-U-Net
architecture ported from the attached PyTorch notebook; CoherentN2N still uses
within-bin Noise2Noise pairs, not receiver-neighbour masks. *Frequency* is the
original complex mask denoiser.

Wave-U-Net layers: $(@bind unet_depth PlutoUI.Slider(1:9; default=9, show_value=true))  •
initial filters: $(@bind unet_width PlutoUI.Slider(8:8:64; default=24, show_value=true))  •
filter: $(@bind unet_kernel PlutoUI.Slider(3:2:21; default=15, show_value=true))  •
merge filter: $(@bind unet_merge_kernel PlutoUI.Slider(3:2:15; default=5, show_value=true))"""

# ╔═╡ b1000034-0000-0000-0000-000000000034
para_binned = cn.CoherentN2N_Para(nt=size(groups_binned[1], 1), use_gpu=true)

# ╔═╡ b1000035-0000-0000-0000-000000000035
# Only the fields the alignment-free loop reads — no max_shift, no shift prior,
# no polarity gain, no stochastic references.
outer_para_binned = cn.CoherentN2N_Outer_Para(
    n_outer_iters=ni_real,
    stack_type=stack_type_real,
    denoiser_training=cn.CoherentN2N_Denoiser_Training_Para(
        n_samples_per_epoch=nspe_real, batchsize=bs_real, nepoch=nepoch_real,
        initial_lr=lr_real, restart_period=rp_real,
        denoiser_loss_type=denoiser_loss_real))

# ╔═╡ b1000037-0000-0000-0000-000000000037
# Button-gated, but the dep list ALSO carries the data/params identity: without
# that, changing the station list / bin size / stack_type would leave a stale
# result cached from the previous data, and the plots below would silently
# compare a new gather against an old ŝ. objectid is enough — groups_binned is
# rebuilt (new object) whenever anything upstream of it changes.
#
# NOTE the one thing the deps CANNOT see: edits to the CoherentN2N_*.jl library.
# Revise hot-reloads those into `cn`, but neither the button counter nor any
# objectid changes, so a cached result (including a diverged all-NaN one) will
# happily survive a denoiser fix. Click the button again after changing the
# algorithm — that is what the counter is for.
result_binned_n2n = @use_memo([train_binned_click]) do
    train_binned_click == 0 ? nothing :
    denoiser_domain === :time ?
        cn.run_coherent_n2n_grouped_time(groups_binned, para_binned, outer_para_binned;
                                         depth=unet_depth, width=unet_width,
                                         kernel_size=unet_kernel,
                                         merge_filter_size=unet_merge_kernel) :
        cn.run_coherent_n2n_grouped_noalign(groups_binned, para_binned, outer_para_binned)
end

# ╔═╡ b1000039-0000-0000-0000-000000000039
# With τ ≡ 0 and no network this is literally the plain stack of each bin — one
# FFT + one stack per group, so it needs no button: computing it eagerly means it
# can never go stale against groups_binned, and it is always available as the
# reference the denoiser has to beat. (For stack_type=:l2 it is exactly the raw
# column mean, so the "Baseline ŝ" and "Raw mean" curves below MUST coincide —
# if they ever diverge, something upstream is inconsistent.)
# Must follow denoiser_domain: the two baselines return ŝ in DIFFERENT domains
# (real waveform vs complex spectrum), and ŝ_time_of below dispatches on the same
# flag. Numerically they agree — for stack_type=:l2 both are the plain trace mean.
result_binned_baseline =
    denoiser_domain === :time ?
        cn.run_coherent_n2n_grouped_time_baseline(groups_binned, para_binned, outer_para_binned) :
        cn.run_coherent_n2n_grouped_noalign_baseline(groups_binned, para_binned, outer_para_binned)

# ╔═╡ b100003c-0000-0000-0000-00000000003c
let
    k = findfirst(==(g_sel), binned_keep)
    k === nothing ? md"⚠️ Selected station is not in the training set." : let
        D_g = groups_binned[k]
        ts = 1:size(D_g, 1)
        norm1(x) = x ./ (maximum(abs, x) + eps(Float32))
        # Guard against comparing a result against data it was not computed
        # from: any result whose group count disagrees with the current
        # groups_binned is stale (the station list or binning changed since it
        # ran) and is dropped rather than plotted misleadingly.
        fresh(res) = res !== nothing && length(res.groups) == length(groups_binned) ? res : nothing
        # Domain-aware accessor — the ONE place that knows where ŝ lives. The
        # time-domain loops return ŝ as a real waveform already; applying ifft to
        # it would be silently wrong (it would transform a waveform as if it were
        # a spectrum), so this must follow denoiser_domain, not the value's type.
        ŝ_of(res) = res === nothing ? nothing :
                    denoiser_domain === :time ? Float32.(res.groups[k].ŝ) :
                                                real(ifft(res.groups[k].ŝ))
        ŝ_base = ŝ_of(fresh(result_binned_baseline))
        ŝ_n2n = ŝ_of(fresh(result_binned_n2n))
        stale_n2n = result_binned_n2n !== nothing && fresh(result_binned_n2n) === nothing
        # A diverged run (NaN ŝ) would otherwise plot as an INVISIBLE trace and
        # look exactly like "the N2N result is missing" with no explanation —
        # catch it explicitly and say so.
        nan_n2n = ŝ_n2n !== nothing && !all(isfinite, ŝ_n2n)
        nan_n2n && (ŝ_n2n = nothing)

        # No best-lag alignment: τ ≡ 0 means every estimate already shares the
        # data's time origin, so any offset between these curves would be a real
        # difference, not a gauge artefact.
        traces = PlutoPlotly.GenericTrace[
            PlutoPlotly.scatter(x=collect(ts), y=norm1(vec(mean(D_g; dims=2))), mode="lines",
                name="Raw mean", line=attr(color="grey", width=1.5, dash="dash")),
        ]
        ŝ_base !== nothing && push!(traces,
            PlutoPlotly.scatter(x=collect(ts), y=norm1(ŝ_base), mode="lines",
                name="Baseline ŝ (plain bin stack)", line=attr(color="#2ca02c", width=2)))
        ŝ_n2n !== nothing && push!(traces,
            PlutoPlotly.scatter(x=collect(ts), y=norm1(ŝ_n2n), mode="lines",
                name="N2N ŝ", line=attr(color="#d62728", width=2)))
        layout = Layout(
            title=attr(text="Binned coherent stack — $(nan_n2n ? "❌ N2N DIVERGED (ŝ is NaN) — retrain; a NaN trace draws as nothing" : stale_n2n ? "⚠️ N2N result is STALE (data changed since training) — click Train bins again" : ŝ_n2n === nothing ? "baseline only — click Train bins for N2N" : "baseline vs N2N vs raw mean") ($(StaN_used[g_sel]), $(size(D_g, 2)) events)"),
            xaxis=attr(title="Sample"), yaxis=attr(title="Normalized amplitude"),
            height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
        )
        WideCell(PlutoPlotly.plot(traces, layout))
    end
end

# ╔═╡ b100003a-0000-0000-0000-00000000003a
# The baseline always exists now, so this is non-nothing from the start; it only
# upgrades to the N2N result once training has been run.
result_binned = result_binned_n2n !== nothing ? result_binned_n2n : result_binned_baseline

# ╔═╡ b100003b-0000-0000-0000-00000000003b
md"---
## 3. Diagnostics

`result.groups[k].ŝ` — the coherent stack for bin `k` (time domain via
`real(ifft(ŝ))`). `result.groups[k].τ` is identically zero by construction, so
there is no shift histogram, no gauge, and no aligned-gather plot here — those
only exist in the aligned notebook."

# ╔═╡ b100003d-0000-0000-0000-00000000003d
result_binned_n2n === nothing ? md"⏳ Click **Train bins** to populate this." : let
    curves = result_binned_n2n.history.denoiser_loss
    traces = PlutoPlotly.GenericTrace[]
    for (i, c) in enumerate(curves)
        push!(traces, PlutoPlotly.scatter(x=collect(1:length(c)), y=c, mode="lines",
                                          name="outer iter $i", line=attr(width=1.5)))
    end
    layout = Layout(
        title=attr(text="Denoiser N2N loss — shared across all bins (one curve per outer iteration)"),
        xaxis=attr(title="Epoch"), yaxis=attr(title="Loss", type="log"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ b100003e-0000-0000-0000-00000000003e
result_binned_n2n === nothing ? md"⏳ Click **Train bins** to populate this." :
length(result_binned_n2n.groups) != length(groups_binned) ? md"⚠️ The N2N result is **stale** — the station list or binning changed since it was trained. Click **Train bins** again." : let
    k = findfirst(==(g_sel), binned_keep)
    k === nothing ? md"⚠️ Selected station is not in the training set." : let
        ds = result_binned_n2n.history.delta_s[k]
        tr = PlutoPlotly.scatter(x=collect(1:length(ds)), y=ds, mode="lines+markers",
                                 name="Δŝ", line=attr(color="#1f77b4", width=2))
        layout = Layout(
            title=attr(text="Outer-loop convergence ‖Δŝ‖ — $(StaN_used[g_sel]) (Δτ is identically 0)"),
            xaxis=attr(title="Outer iteration"), yaxis=attr(title="‖Δŝ‖"),
            height=320, width=900, plot_bgcolor="white", paper_bgcolor="white",
        )
        WideCell(PlutoPlotly.plot([tr], layout))
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[compat]
CUDA = "~6.2.0"
DSP = "~0.8.6"
FFTW = "~1.10.0"
Flux = "~0.16.10"
JLD2 = "~0.6.5"
Optimisers = "~0.4.7"
PlutoHooks = "~0.1.0"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.6"
PlutoUI = "~0.7.83"
ProgressLogging = "~0.1.6"
Revise = "~3.16.1"
Zygote = "~0.7.11"
cuDNN = "~6.2.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "7323af74958f0588c353470eeae5bcb81b08e549"

[[deps.ADTypes]]
git-tree-sha1 = "ec6be48a85c93d995563b84bff8a86bc98df45ce"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.22.2"
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

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

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
git-tree-sha1 = "ff4a0abf98be36d9fa15ca7e968df562a210ab4b"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.3.0+1"

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
git-tree-sha1 = "77b169898d4cdc234b1cd9afc1d0a8cac1017a24"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "9.24.0+0"

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
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "a65cfc2999988f5ba09fc4bd8049e3ed914e5a04"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.6"

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

    [deps.DSP.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

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

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "5bad39456d9f0166184fce2248783dd9862645c1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.17.0"

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
git-tree-sha1 = "5e54ec63c34bcc878558b173c411b8efe6b08344"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.23.0"

    [deps.GPUCompiler.weakdeps]
    AMDGPU_LLVM_Backend_jll = "cc5c0156-bd05-5a77-8a68-bb0aafb29019"
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

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

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
git-tree-sha1 = "c3d401f110454b4ea24a76be33f6ee0d7d385103"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.11.4"

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
git-tree-sha1 = "24af42ea221a14a04283e362d83d2cdcb115cd12"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.10.1"
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

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "7cf039cf79bb41afda7336edf2f3ca2115c44f76"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.4.2"

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
deps = ["CRC32c", "CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "838f84266bf2e9ca4b7d0b8965807c7a72745501"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.16.1"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLPublic]]
git-tree-sha1 = "24ff31136f3f991b74fbef71d5c638e2881d29d2"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.2.3"

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
git-tree-sha1 = "6a73aec31c56a0c2833e8efa637d10b532cb2f0c"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.5"

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
# ╠═b1000002-0000-0000-0000-000000000002
# ╠═b1000001-0000-0000-0000-000000000001
# ╠═b1000003-0000-0000-0000-000000000003
# ╠═b1000004-0000-0000-0000-000000000004
# ╠═b1000005-0000-0000-0000-000000000005
# ╠═b1000006-0000-0000-0000-000000000006
# ╠═b1000007-0000-0000-0000-000000000007
# ╠═b1000008-0000-0000-0000-000000000008
# ╟─b100000a-0000-0000-0000-00000000000a
# ╟─b100000b-0000-0000-0000-00000000000b
# ╠═b1000009-0000-0000-0000-000000000009
# ╠═b100000c-0000-0000-0000-00000000000c
# ╟─b100000d-0000-0000-0000-00000000000d
# ╟─b100000e-0000-0000-0000-00000000000e
# ╟─b100000f-0000-0000-0000-00000000000f
# ╟─b1000010-0000-0000-0000-000000000010
# ╟─b1000011-0000-0000-0000-000000000011
# ╠═b1000012-0000-0000-0000-000000000012
# ╟─b1000013-0000-0000-0000-000000000013
# ╟─b1000014-0000-0000-0000-000000000014
# ╟─b1000015-0000-0000-0000-000000000015
# ╠═b1000016-0000-0000-0000-000000000016
# ╠═b1000017-0000-0000-0000-000000000017
# ╠═b1000018-0000-0000-0000-000000000018
# ╠═b1000019-0000-0000-0000-000000000019
# ╟─b100001a-0000-0000-0000-00000000001a
# ╟─b100001b-0000-0000-0000-00000000001b
# ╟─b100001c-0000-0000-0000-00000000001c
# ╟─b100001d-0000-0000-0000-00000000001d
# ╟─b1000020-0000-0000-0000-000000000020
# ╠═b1000021-0000-0000-0000-000000000021
# ╠═b1000022-0000-0000-0000-000000000022
# ╠═b1000023-0000-0000-0000-000000000023
# ╠═b1000024-0000-0000-0000-000000000024
# ╟─b1000025-0000-0000-0000-000000000025
# ╟─b1000026-0000-0000-0000-000000000026
# ╠═b1000036-0000-0000-0000-000000000036
# ╟─b1000027-0000-0000-0000-000000000027
# ╠═b1000028-0000-0000-0000-000000000028
# ╟─b1000029-0000-0000-0000-000000000029
# ╟─b100002a-0000-0000-0000-00000000002a
# ╟─b1000032-0000-0000-0000-000000000032
# ╟─b100003c-0000-0000-0000-00000000003c
# ╠═b100002b-0000-0000-0000-00000000002b
# ╠═b100002c-0000-0000-0000-00000000002c
# ╟─b100002d-0000-0000-0000-00000000002d
# ╟─b100002e-0000-0000-0000-00000000002e
# ╠═b100002f-0000-0000-0000-00000000002f
# ╠═b1000030-0000-0000-0000-000000000030
# ╟─b1000031-0000-0000-0000-000000000031
# ╟─b1000033-0000-0000-0000-000000000033
# ╠═b1000034-0000-0000-0000-000000000034
# ╠═b1000035-0000-0000-0000-000000000035
# ╠═b1000037-0000-0000-0000-000000000037
# ╠═b1000039-0000-0000-0000-000000000039
# ╠═b100003a-0000-0000-0000-00000000003a
# ╟─b100003b-0000-0000-0000-00000000003b
# ╟─b100003d-0000-0000-0000-00000000003d
# ╟─b100003e-0000-0000-0000-00000000003e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
