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
cn = @ingredients(joinpath(@__DIR__, "CoherentN2N_lib.jl"))

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
end

# ╔═╡ b1000013-0000-0000-0000-000000000013
md"### Raw synthetic gather (coherent by construction, noisy)"

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

# ╔═╡ b1000019-0000-0000-0000-000000000019
# Button-gated: CounterButton starts at 0 so nothing runs on load. A single
# one-element group vector exercises the grouped code path on one gather.
result_syn = @use_memo([train_syn_click]) do
    train_syn_click == 0 ? nothing :
    cn.run_coherent_n2n_grouped_noalign([D_syn], para_syn, outer_para_syn)
end

# ╔═╡ b100001a-0000-0000-0000-00000000001a
md"### Ground-truth comparison — the pass/fail criterion"

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
fldir_real = "/mnt/NAS/EQData/RFData"

# ╔═╡ b1000022-0000-0000-0000-000000000022
dn_real = "GSN_150ZTR_Bandpass_0.05_0.8_29nov_rf_iter_f1.jld2"

# ╔═╡ b1000023-0000-0000-0000-000000000023
snrf_real = "GSN_150ZTR_Bandpass_0.05_0.8_29nov_snr.jld2"

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
        push!(raw_data_list_full, Data_real[ix][:, sel][501:800, :])  # trim RF window
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
    w = cat(DSP.tukey(size(x, 1), 0.3), dims=ndims(x))
    return w .* x
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

# ╔═╡ b1000031-0000-0000-0000-000000000031
md"### Raw binned gather (selected station)"

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

# ╔═╡ b1000033-0000-0000-0000-000000000033
md"""#### Training controls
Outer iters: $(@bind ni_real PlutoUI.Slider(1:20; default=10, show_value=true))  •
Epochs/iter: $(@bind nepoch_real PlutoUI.Slider(1:20:400; default=1, show_value=true))

Batch size: $(@bind bs_real PlutoUI.Slider(16:16:2048; default=512, show_value=true))  •
Samples/epoch (per group): $(@bind nspe_real PlutoUI.Slider(64:64:1024; default=512, show_value=true))

Restart period: $(@bind rp_real PlutoUI.Slider(10:10:200; default=50, show_value=true))  •
Initial LR: $(@bind lr_real PlutoUI.Select([0.0003, 0.001, 0.003, 0.01]; default=0.001))

Coherent stack (`stack_type`): $(@bind stack_type_real PlutoUI.Select([:l2 => "L2 — mean", :l1 => "L1 — robust Huber/IRLS"]; default=:l2))  •
Denoiser loss: $(@bind denoiser_loss_real PlutoUI.Select([:l2 => "L2 / MSE", :l1 => "L1 / mean-abs"]; default=:l2))"""

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

# ╔═╡ b1000036-0000-0000-0000-000000000036
train_binned_button = @bind train_binned_click PlutoUI.CounterButton("Train bins (no-align N2N)")

# ╔═╡ b1000037-0000-0000-0000-000000000037
result_binned_n2n = @use_memo([train_binned_click]) do
    train_binned_click == 0 ? nothing :
    cn.run_coherent_n2n_grouped_noalign(groups_binned, para_binned, outer_para_binned)
end

# ╔═╡ b1000038-0000-0000-0000-000000000038
baseline_binned_button = @bind baseline_binned_click PlutoUI.CounterButton("Run baseline (plain bin stack)")

# ╔═╡ b1000039-0000-0000-0000-000000000039
# With τ ≡ 0 and no network this is literally the plain stack of each bin — the
# thing the denoiser has to beat. Cheap: one FFT + one stack per group.
result_binned_baseline = @use_memo([baseline_binned_click]) do
    baseline_binned_click == 0 ? nothing :
    cn.run_coherent_n2n_grouped_noalign_baseline(groups_binned, para_binned, outer_para_binned)
end

# ╔═╡ b100003a-0000-0000-0000-00000000003a
result_binned = result_binned_n2n !== nothing ? result_binned_n2n : result_binned_baseline

# ╔═╡ b100003b-0000-0000-0000-00000000003b
md"---
## 3. Diagnostics

`result.groups[k].ŝ` — the coherent stack for bin `k` (time domain via
`real(ifft(ŝ))`). `result.groups[k].τ` is identically zero by construction, so
there is no shift histogram, no gauge, and no aligned-gather plot here — those
only exist in the aligned notebook."

# ╔═╡ b100003c-0000-0000-0000-00000000003c
result_binned === nothing ? md"⏳ Run the **baseline** and/or **N2N** to populate this." : let
    k = findfirst(==(g_sel), binned_keep)
    k === nothing ? md"⚠️ Selected station is not in the training set." : let
        D_g = groups_binned[k]
        ts = 1:size(D_g, 1)
        norm1(x) = x ./ (maximum(abs, x) + eps(Float32))
        ŝ_of(res) = res === nothing ? nothing : real(ifft(res.groups[k].ŝ))
        ŝ_base = ŝ_of(result_binned_baseline)
        ŝ_n2n = ŝ_of(result_binned_n2n)

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
            title=attr(text="Binned coherent stack — baseline vs N2N vs raw mean ($(StaN_used[g_sel]), $(size(D_g, 2)) events)"),
            xaxis=attr(title="Sample"), yaxis=attr(title="Normalized amplitude"),
            height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
        )
        WideCell(PlutoPlotly.plot(traces, layout))
    end
end

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
result_binned_n2n === nothing ? md"⏳ Click **Train bins** to populate this." : let
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
# ╟─b1000027-0000-0000-0000-000000000027
# ╠═b1000028-0000-0000-0000-000000000028
# ╟─b1000029-0000-0000-0000-000000000029
# ╟─b100002a-0000-0000-0000-00000000002a
# ╠═b100002b-0000-0000-0000-00000000002b
# ╠═b100002c-0000-0000-0000-00000000002c
# ╟─b100002d-0000-0000-0000-00000000002d
# ╟─b100002e-0000-0000-0000-00000000002e
# ╠═b100002f-0000-0000-0000-00000000002f
# ╠═b1000030-0000-0000-0000-000000000030
# ╟─b1000031-0000-0000-0000-000000000031
# ╟─b1000032-0000-0000-0000-000000000032
# ╟─b1000033-0000-0000-0000-000000000033
# ╠═b1000034-0000-0000-0000-000000000034
# ╠═b1000035-0000-0000-0000-000000000035
# ╠═b1000036-0000-0000-0000-000000000036
# ╠═b1000037-0000-0000-0000-000000000037
# ╠═b1000038-0000-0000-0000-000000000038
# ╠═b1000039-0000-0000-0000-000000000039
# ╠═b100003a-0000-0000-0000-00000000003a
# ╟─b100003b-0000-0000-0000-00000000003b
# ╟─b100003c-0000-0000-0000-00000000003c
# ╟─b100003d-0000-0000-0000-00000000003d
# ╟─b100003e-0000-0000-0000-00000000003e
# ╟─00000000-0000-0000-0000-000000000001
