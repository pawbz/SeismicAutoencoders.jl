### A Pluto.jl notebook ###
# v0.20.23

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

# ╔═╡ a0f8a2b4-8fb5-4f06-bda6-c362a61065a1
begin
    using Base.Threads
    using ColorSchemes
    using Colors
    using JLD2
    using LinearAlgebra
    using PlutoLinks
    using PlutoPlotly
    using PlutoUI
    using Printf
    using ProgressLogging
    using Statistics
end

# ╔═╡ 2f66e040-8d6d-4376-b97d-0169cbdc1efe
using FFTW, StatsBase, Peaks

# ╔═╡ 75e66ac1-57bf-468d-ab55-ada1d5e9ef91
using DSP

# ╔═╡ aa000050-0000-0000-0000-000000000001
using CSV, DataFrames

# ╔═╡ b00fd94f-291e-46d8-84ff-48f8606c2a1e
md"""
# Trained VQ-VAE MFT v9

CPU-only notebook for saved source-state artifacts. It runs the MFT analysis for
all selected receiver pairs upfront, keeps pair selection mostly plot-only, scores
geometry-aware tomography candidate mixes, and writes DSurfTomo-ready dispersion
tables from the consensus group-velocity picks.
"""

# ╔═╡ dcbf026e-957a-4b9b-9757-bd0638a25b26
begin
    # saved_root = "/mnt/NAS2/Sanket_data/California_TO_with_latlong/SavedModels/vqvae_v9_K=[5, 3]"
saved_root =
	"/mnt/NAS2/Sanket_data/California_XJ_13032026/SavedModels/vqvae_v10_K=[5, 3]"
#
	# saved_root =
	# 	"/mnt/NAS2/Sanket_data/California_2013_BK_CI_20032026/SavedModels/vqvae_v10"
end

# ╔═╡ aa000051-0000-0000-0000-000000000001
xj_latlong = let
    csv_paths = [
        "/mnt/NAS2/Sanket_data/California_09032026/data/stationlists/Stations_California_XJ.csv",
        "/mnt/NAS2/Sanket_data/California_09032026/data/stationlists/Stations_California_XJ_new.csv",
    ]
    dfs = [CSV.read(p, DataFrame) for p in csv_paths if isfile(p)]
    isempty(dfs) ? DataFrame(; Network=String[], var"Station Code"=String[],
                               Latitude=Float64[], Longitude=Float64[]) :
        unique(vcat(dfs...))
end

# ╔═╡ f842f93e-16e8-4ec9-9c7f-c63d1f18c9f9
@bind reload_saved_artifacts_button CounterButton("Reload saved source-state artifacts")

# ╔═╡ e4c00001-0000-0000-0000-000000000001
@bind ui_mft_mode Select(["Joint states (K1×K2)", "Marginal stages (K1+K2)"]; default="Joint states (K1×K2)")

# ╔═╡ bb000010-0000-0000-0000-000000000001
md"## Global Average Dispersion Picks — All Pairs"

# ╔═╡ bb000011-0000-0000-0000-000000000001
# Outlier rejection threshold: picks more than this many σ from the per-period mean are removed.
@bind ui_globalavg_outlier_nsigma NumberField(0.5:0.1:5.0; default=0.8)

# ╔═╡ d6be1ffc-5468-11f1-a6b3-4db6669eeadd
md"## Single-Peak Global Average Dispersion Picks"

# ╔═╡ d6be2036-5468-11f1-a727-b909116351a8
# Outlier rejection threshold for single causal/acausal maximum-amplitude picks.
@bind ui_single_peak_globalavg_outlier_nsigma NumberField(0.5:0.1:5.0; default=0.8)

# ╔═╡ d6be20a4-5468-11f1-b705-a3f511139964
md"### Write Single-Peak Global Average DSurfTomo File"

# ╔═╡ d6be20cc-5468-11f1-b014-99325710d623
begin
    @bind _single_peak_globalavg_export PlutoUI.combine() do Child
        md"""
        | Single-peak global-average export | Value |
        |:---|:---|
        | DSurfTomo path | $(Child("path", TextField(default=joinpath(@__DIR__, "DSurfTomo_runs", "global_avg_single_peak_dispersion.txt")))) |
        | Include count header | $(Child("header", CheckBox(default=false))) |
        | Write | $(Child("write", CounterButton("Write single-peak global avg DSurfTomo file"))) |
        """
    end
    dsurftomo_single_peak_globalavg_path = _single_peak_globalavg_export.path
    dsurftomo_single_peak_globalavg_include_header = _single_peak_globalavg_export.header
    write_single_peak_globalavg_dsurftomo_button = _single_peak_globalavg_export.write
end

# ╔═╡ d6be213a-5468-11f1-be47-d922a2dae738
md""

# ╔═╡ d6be216c-5468-11f1-b0af-6bf8fd2f274f
md""

# ╔═╡ bb000017-0000-0000-0000-000000000001
md"### Write Global Average DSurfTomo File"

# ╔═╡ bb000018-0000-0000-0000-000000000001
begin
    @bind _globalavg_export PlutoUI.combine() do Child
        md"""
        | Global-average export | Value |
        |:---|:---|
        | DSurfTomo path | $(Child("path", TextField(default=joinpath(@__DIR__, "DSurfTomo_runs", "global_avg_dispersion.txt")))) |
        | Include count header | $(Child("header", CheckBox(default=false))) |
        | Write | $(Child("write", CounterButton("Write global avg DSurfTomo file"))) |
        """
    end
    dsurftomo_globalavg_path = _globalavg_export.path
    dsurftomo_globalavg_include_header = _globalavg_export.header
    write_globalavg_dsurftomo_button = _globalavg_export.write
end

# ╔═╡ bb000019-0000-0000-0000-000000000001
md""

# ╔═╡ bb00001a-0000-0000-0000-000000000001
md""

# ╔═╡ b584dc56-8127-402c-98d0-1da3135f8f9a
md"""
## Appendix
"""

# ╔═╡ c7d47e38-e24f-4b40-b3a4-bc894188a750
mft = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/MFT_v2.jl")

# ╔═╡ d50d63be-d58b-4704-8211-ed7875e04857
md"## Saved Runs"

# ╔═╡ c7ec82e0-d5b6-4f31-8600-c8b1d276dc92
function _parse_seed_timestamp(run_dir::String)
    name = basename(run_dir)
    m = match(r"^seed([0-9]+)_(.+)$", name)
    m === nothing && return (; seed=missing, timestamp=name)
    return (; seed=parse(Int, m.captures[1]), timestamp=m.captures[2])
end

# ╔═╡ e4499887-1a64-4eaa-a599-4ed4941a7b2d
md"## Load Saved Source-State Averages"

# ╔═╡ e4f7b3cf-7f26-4ea9-b9bb-95f3df9e7790
function _source_state_artifact_path(run)
    return joinpath(run.run_dir, "source_state_averages.jld2")
end

# ╔═╡ 02d134b5-7ce3-47a9-86ef-a43e6c52287a
function _load_saved_source_state_averages(run)
    path = _source_state_artifact_path(run)
    isfile(path) || error("Missing source-state artifact: $(path)")
    d = load(path)
    return (;
        acausal=d["acausal"],
        causal=d["causal"],
        counts_ac=d["counts_ac"],
        counts_c=d["counts_c"],
        combo_labels=String.(d["combo_labels"]),
        global_avg_ac=d["global_avg_ac"],
        global_avg_c=d["global_avg_c"],
        window_headers=haskey(d, "window_headers") ? String.(d["window_headers"]) : String[],
        window_time_labels=haskey(d, "window_time_labels") ? String.(d["window_time_labels"]) : String[],
        source_state_ac=haskey(d, "source_state_ac") ? Int.(d["source_state_ac"]) : Int[],
        source_state_c=haskey(d, "source_state_c") ? Int.(d["source_state_c"]) : Int[],
        stage_assignments_ac=haskey(d, "stage_assignments_ac") ? Int.(d["stage_assignments_ac"]) : zeros(Int, 0, 0),
        stage_assignments_c=haskey(d, "stage_assignments_c") ? Int.(d["stage_assignments_c"]) : zeros(Int, 0, 0),
        assignment_table=haskey(d, "assignment_table") ? String.(d["assignment_table"]) : Matrix{String}(undef, 0, 0),
        assignment_table_columns=haskey(d, "assignment_table_columns") ? String.(d["assignment_table_columns"]) : String[],
        assignment_table_ac=haskey(d, "assignment_table_ac") ? String.(d["assignment_table_ac"]) : Matrix{String}(undef, 0, 0),
        assignment_table_c=haskey(d, "assignment_table_c") ? String.(d["assignment_table_c"]) : Matrix{String}(undef, 0, 0),
        assignment_table_ac_columns=haskey(d, "assignment_table_ac_columns") ? String.(d["assignment_table_ac_columns"]) : String[],
        assignment_table_c_columns=haskey(d, "assignment_table_c_columns") ? String.(d["assignment_table_c_columns"]) : String[],
        analysis_settings=d["analysis_settings"],
        global_avg_raw_ac=haskey(d, "global_avg_raw_ac") ? d["global_avg_raw_ac"] : Float32[],
        global_avg_raw_c=haskey(d, "global_avg_raw_c") ? d["global_avg_raw_c"] : Float32[],
        global_avg_whitened_ac=haskey(d, "global_avg_whitened_ac") ? d["global_avg_whitened_ac"] : Float32[],
        global_avg_whitened_c=haskey(d, "global_avg_whitened_c") ? d["global_avg_whitened_c"] : Float32[],
        marginal_stage1_ac=haskey(d, "marginal_stage1_ac") ? Float32.(d["marginal_stage1_ac"]) : Float32[;;],
        marginal_stage1_c=haskey(d, "marginal_stage1_c") ? Float32.(d["marginal_stage1_c"]) : Float32[;;],
        marginal_stage2_ac=haskey(d, "marginal_stage2_ac") ? Float32.(d["marginal_stage2_ac"]) : Float32[;;],
        marginal_stage2_c=haskey(d, "marginal_stage2_c") ? Float32.(d["marginal_stage2_c"]) : Float32[;;],
        marginal_stage1_labels=haskey(d, "marginal_stage1_labels") ? String.(d["marginal_stage1_labels"]) : String[],
        marginal_stage2_labels=haskey(d, "marginal_stage2_labels") ? String.(d["marginal_stage2_labels"]) : String[],
        marginal_stage1_counts_ac=haskey(d, "marginal_stage1_counts_ac") ? Int.(d["marginal_stage1_counts_ac"]) : Int[],
        marginal_stage2_counts_ac=haskey(d, "marginal_stage2_counts_ac") ? Int.(d["marginal_stage2_counts_ac"]) : Int[],
        marginal_stage1_counts_c=haskey(d, "marginal_stage1_counts_c") ? Int.(d["marginal_stage1_counts_c"]) : Int[],
        marginal_stage2_counts_c=haskey(d, "marginal_stage2_counts_c") ? Int.(d["marginal_stage2_counts_c"]) : Int[],
        codebook_stage1_waves=haskey(d, "codebook_stage1_waves") ? Float32.(d["codebook_stage1_waves"]) : Float32[;;],
        codebook_stage1_labels=haskey(d, "codebook_stage1_labels") ? String.(d["codebook_stage1_labels"]) : String[],
        codebook_stage2_waves=haskey(d, "codebook_stage2_waves") ? Float32.(d["codebook_stage2_waves"]) : Float32[;;],
        codebook_stage2_labels=haskey(d, "codebook_stage2_labels") ? String.(d["codebook_stage2_labels"]) : String[],
        codebook_joint_waves=haskey(d, "codebook_joint_waves") ? Float32.(d["codebook_joint_waves"]) : Float32[;;],
        codebook_joint_labels=haskey(d, "codebook_joint_labels") ? String.(d["codebook_joint_labels"]) : String[],
        whitening_fir=d["whitening_fir"],
        spike_bins=d["spike_bins"],
        distance=d["distance"],
        latitudes=d["latitudes"],
        longitudes=d["longitudes"],
        pair=run.pair,
        pair_label=run.pair_label,
        run_dir=run.run_dir,
        seed=run.seed,
    )
end

# ╔═╡ eec83733-193d-4f52-9a75-e6f1d03c7aa5
md"## MFT By Receiver Pair"

# ╔═╡ 2c887c9a-4a41-4d07-a295-f5c14cfdd110
function _mean_global_branch(items, branch::Symbol)
    vectors = [Float64.(vec(getproperty(item, branch))) for item in items]
    isempty(vectors) && return Float64[]
    n = minimum(length.(vectors))
    n == 0 && return Float64[]
    return vec(mean(hcat((v[1:n] for v in vectors)...); dims=2))
end

# ╔═╡ fb000002-0000-0000-0000-000000000001
mft_filter_banks = let
    # `@ingredients` rebuilds the MFT module after edits; cached banks from an
    # older module look similar in Pluto errors but have incompatible Julia types.
    mft
    Dict{Any,Any}()
end

# ╔═╡ b350e7a5-ae7e-46ca-a246-d60b66a68e17
md"## Stationary-Zone Codebook Picks"

# ╔═╡ e89d81cb-8596-4364-8241-01578fb81c6b
md"## Quick Plots"

# ╔═╡ d1d675f4-ebea-4432-8d0e-ddeada2f5fa3
mft_analysis_test = md"Interactive selected-pair test analysis removed. All source-state MFT is computed in the all-pairs batch dictionaries above."


# ╔═╡ c3000001-0000-0000-0000-000000000001
md"## Codebook Waveform MFT"

# ╔═╡ c3000007-0000-0000-0000-000000000001
begin
    @bind _codebook_stationary_controls PlutoUI.combine() do Child
        md"""
        | Codebook stationary-zone control | Value |
        |:---|:---|
        | Codebook NCC threshold | $(Child("ncc_threshold", Slider(0.50:0.05:0.99; default=0.80, show_value=true))) |
        | Candidate codebook family | $(Child("codebook_family", Select(["joint (K1×K2)", "K1+K2", "stage2", "stage1"]; default="joint (K1×K2)"))) |
        | High-SNR quality threshold | $(Child("quality_threshold", Slider(1.0:0.25:8.0; default=3.0, show_value=true))) |
        | Minimum high-SNR periods | $(Child("min_periods", NumberField(2:1:12; default=3))) |
        | Stationary relative tolerance | $(Child("relative_tolerance", Slider(0.02:0.01:0.30; default=0.10, show_value=true))) |
        """
    end
   
end

# ╔═╡ 7f3b6cae-a0f6-463e-8375-05a86d200d3a
begin
	 ui_codebook_ncc_threshold = _codebook_stationary_controls.ncc_threshold
	    ui_codebook_stationary_family = _codebook_stationary_controls.codebook_family
	    ui_codebook_stationary_quality_threshold = _codebook_stationary_controls.quality_threshold
	    ui_codebook_stationary_min_periods = _codebook_stationary_controls.min_periods
	    ui_codebook_stationary_relative_tolerance = _codebook_stationary_controls.relative_tolerance
end

# ╔═╡ 18793657-2479-4260-927d-8a39088fb814
function _codebook_family_waves(item, mode::AbstractString)
    if mode == "K1+K2"
        waves_parts = Matrix{Float32}[]
        label_parts = String[]
        if !isempty(item.codebook_stage1_waves)
            push!(waves_parts, item.codebook_stage1_waves)
            append!(label_parts, ["S1-$(k <= length(item.codebook_stage1_labels) ? item.codebook_stage1_labels[k] : string(k))"
                for k in axes(item.codebook_stage1_waves, 2)])
        end
        if !isempty(item.codebook_stage2_waves)
            push!(waves_parts, item.codebook_stage2_waves)
            append!(label_parts, ["S2-$(k <= length(item.codebook_stage2_labels) ? item.codebook_stage2_labels[k] : string(k))"
                for k in axes(item.codebook_stage2_waves, 2)])
        end
        if isempty(waves_parts)
            return Float32[;;], String[], "K1+K2"
        else
            n = minimum(size(w, 1) for w in waves_parts)
            return reduce(hcat, [w[1:n, :] for w in waves_parts]), label_parts, "K1+K2"
        end
    elseif mode == "joint (K1×K2)" && !isempty(item.codebook_joint_waves)
        return item.codebook_joint_waves, item.codebook_joint_labels, "J"
    elseif mode == "stage2" && !isempty(item.codebook_stage2_waves)
        return item.codebook_stage2_waves, item.codebook_stage2_labels, "S2"
    else
        return item.codebook_stage1_waves, item.codebook_stage1_labels, "S1"
    end
end

# ╔═╡ 657949fd-620a-43e2-916f-f9172573c50f
function _codebook_columns_for_pair(pair_items, mode::AbstractString)
    columns = Vector{Float64}[]
    labels = String[]
    seeds = Int[]
    families = String[]
    for item in pair_items
        family_specs = if mode == "K1+K2"
            [
                (item.codebook_stage1_waves, item.codebook_stage1_labels, "S1"),
                (item.codebook_stage2_waves, item.codebook_stage2_labels, "S2"),
            ]
        else
            (waves, local_labels, family) = _codebook_family_waves(item, mode)
            [(waves, local_labels, family)]
        end
        for (waves, local_labels, family) in family_specs
            isempty(waves) && continue
            for k in axes(waves, 2)
                label = k <= length(local_labels) ? local_labels[k] : string(k)
                push!(columns, Float64.(vec(waves[:, k])))
                push!(labels, "seed$(item.seed)-$(family)-$(label)")
                push!(seeds, Int(item.seed))
                push!(families, family)
            end
        end
    end
    return columns, labels, seeds, families
end

# ╔═╡ 2c18a0ad-0963-4e63-8c20-548674e42c09
function _stationary_uc_rows(res, label::String, pair_label::String;
        quality_threshold::Float64=3.0,
        relative_tolerance::Float64=0.10)
    u_pred = any(isfinite, res.u_predicted_from_phase) ?
        res.u_predicted_from_phase : mft.compute_group_velocity_from_phase(res)
    rows = NamedTuple[]
    for ip in eachindex(res.periods)
        period = Float64(res.periods[ip])
        u_meas = Float64(res.group_velocities[ip])
        u_hat = Float64(u_pred[ip])
        quality = Float64(res.quality_factors[ip])
        phase_velocity = Float64(res.phase_velocities[ip])
        isfinite(period) && period > 0 || continue
        isfinite(u_meas) && u_meas > 0 || continue
        isfinite(u_hat) && u_hat > 0 || continue
        isfinite(quality) && quality >= quality_threshold || continue
        relerr = abs(u_meas - u_hat) / max(abs(u_hat), eps(Float64))
        push!(rows, (; pair_label, label, period,
            group_velocity=u_meas, predicted_group_velocity=u_hat,
            phase_velocity, relative_error=relerr,
            quality, pass=relerr <= relative_tolerance,
            phase_suspect=ip <= length(res.phase_suspect) ? Bool(res.phase_suspect[ip]) : false))
    end
    return rows
end

# ╔═╡ c3000010-0000-0000-0000-000000000001
function _stationary_summary_for_result(res, label::String, pair_label::String;
        quality_threshold::Float64=3.0,
        relative_tolerance::Float64=0.10)
    rows = _stationary_uc_rows(res, label, pair_label;
        quality_threshold=quality_threshold,
        relative_tolerance=relative_tolerance)
    if isempty(rows)
        return (; label, pair_label, rows,
            n_periods=0, pass_fraction=0.0,
            median_relative_error=Inf, mean_relative_error=Inf,
            mean_quality=NaN, result=res)
    end
    rel = [r.relative_error for r in rows]
    return (; label, pair_label, rows,
        n_periods=length(rows),
        pass_fraction=count(r.pass for r in rows) / length(rows),
        median_relative_error=median(rel),
        mean_relative_error=mean(rel),
        mean_quality=mean(r.quality for r in rows),
        result=res)
end

# ╔═╡ c300000a-0000-0000-0000-000000000001
md""

# ╔═╡ c300000f-0000-0000-0000-000000000001
md""

# ╔═╡ c300000b-0000-0000-0000-000000000001
md"Codebook selection uses the deterministic stationary-zone U-c agreement cells above."

# ╔═╡ ff000010-0000-0000-0000-000000000001
md"""
## DSurfTomo Output

Writes one whitespace-separated row per valid receiver-pair/period pick:

`period_s  lat1  lon1  lat2  lon2  group_velocity_km_s`

When the all-pairs stationary-zone codebook cell has run successfully, export
uses those U-c-consistent picks for every available selected pair. Pairs without
a stationary codebook pick fall back to the all-pairs source-state `pair_consensus`.
"""

# ╔═╡ ff000011-0000-0000-0000-000000000001
begin
    @bind _dsurftomo_export PlutoUI.combine() do Child
        md"""
        | VQ-VAE DSurfTomo export | Value |
        |:---|:---|
        | Dispersion path | $(Child("path", TextField(default=joinpath(@__DIR__, "DSurfTomo_runs", "mft_v9_dispersion.txt")))) |
        | Include count header | $(Child("header", CheckBox(default=false))) |
        | Write | $(Child("write", CounterButton("Write DSurfTomo dispersion file"))) |
        """
    end
    dsurftomo_output_path = _dsurftomo_export.path
    dsurftomo_include_count_header = _dsurftomo_export.header
    write_dsurftomo_button = _dsurftomo_export.write
end

# ╔═╡ ff000012-0000-0000-0000-000000000001
md""

# ╔═╡ ff000013-0000-0000-0000-000000000001
md""

# ╔═╡ aa000030-0000-0000-0000-000000000001
# MFT compute cutoff: only keep periods where reference wavelength is shorter
# than the requested fraction of interstation distance.
begin
    @bind _wavelength_controls PlutoUI.combine() do Child
        md"""
        | MFT wavelength cutoff | Value |
        |:---|:---|
        | Reference velocity (km/s) | $(Child("ref_velocity", NumberField(0.5:0.5:5.0; default=2.0))) |
        | Fraction of interstation distance | $(Child("fraction", NumberField(0.05:0.01:1.0; default=0.33))) |
        """
    end
    ui_wavelength_ref_velocity = _wavelength_controls.ref_velocity
    ui_wavelength_fraction = _wavelength_controls.fraction
end

# ╔═╡ aa000031-0000-0000-0000-000000000001
md""

# ╔═╡ b4c9a332-7e9f-4933-b12f-7c13f6d5f112
function _read_saved_analysis_settings(run_dir::String)
    path = joinpath(run_dir, "source_state_averages.jld2")
    d = load(path)
    haskey(d, "analysis_settings") ||
        error("Saved artifact is missing analysis_settings: $(path). Re-save this v9 run.")
    return d["analysis_settings"]
end

# ╔═╡ e63099d0-5fb5-43b0-967c-b7c468dc4f83
function discover_vqvae_runs(saved_root::String)
    isdir(saved_root) || return NamedTuple[]
    runs = NamedTuple[]
    for pair_dir in sort(filter(isdir, readdir(saved_root, join=true)))
        pair_label = basename(pair_dir)
        parts = split(pair_label, "_")
        length(parts) == 2 || continue
        for run_dir in sort(filter(isdir, readdir(pair_dir, join=true)))
            isfile(joinpath(run_dir, "source_state_averages.jld2")) || continue
            parsed = _parse_seed_timestamp(run_dir)
            analysis_settings = _read_saved_analysis_settings(run_dir)
            push!(runs, (; pair=(String(parts[1]), String(parts[2])),
                         pair_label=replace(pair_label, "_" => "-"),
                         run_dir, seed=parsed.seed, timestamp=parsed.timestamp,
                         analysis_settings))
        end
    end
    return runs
end

# ╔═╡ ed704922-12f0-475b-b628-19a9c37bca7a
begin
	reload_saved_artifacts_button
	all_saved_runs = discover_vqvae_runs(saved_root)
end

# ╔═╡ e216a473-6433-4658-b2b7-a4eaa670cc5e
begin
    pair_options = sort(unique([run.pair_label for run in all_saved_runs]))
    default_pair_names = pair_options[1:min(end, 8)]
end;

# ╔═╡ 9d75dc60-d654-4089-a297-abdcf0163493
@bind selected_pair_names confirm(MultiCheckBox(pair_options; default=[], select_all=true))

# ╔═╡ aa000040-0000-0000-0000-000000000001
# One run per pair (any seed) — global averages are seed-independent so we only need
# one artifact per pair to get global_avg_c / global_avg_ac.
all_runs_for_globalavg = let
    keep = Dict{String,Any}()
    for run in all_saved_runs
        # Keep the latest-timestamped run per pair (seed doesn't matter)
        if !haskey(keep, run.pair_label) ||
                string(run.timestamp) > string(keep[run.pair_label].timestamp)
            keep[run.pair_label] = run
        end
    end
    sort(collect(values(keep)), by=r -> r.pair_label)
end;

# ╔═╡ aa000041-0000-0000-0000-000000000001
all_pairs_source_state_averages = let
    reload_key = reload_saved_artifacts_button
    out = Vector{Any}(undef, length(all_runs_for_globalavg))
    @progress name="Loading all-pair artifacts" for i in eachindex(all_runs_for_globalavg)
        out[i] = _load_saved_source_state_averages(all_runs_for_globalavg[i])
    end
    out
end;

# ╔═╡ aa000042-0000-0000-0000-000000000001
all_pair_labels = sort(unique([item.pair_label for item in all_pairs_source_state_averages]));

# ╔═╡ d8d742f8-b6d4-44e0-bf11-29ff4d22e117
_setting(settings, name::Symbol) =
    hasproperty(settings, name) ? getproperty(settings, name) :
    error("Saved analysis_settings is missing $(name). Re-save this v9 run.")

# ╔═╡ b5a1cf7d-d464-4409-b43a-074c8aa22108
selected_runs = begin
    raw = [run for run in all_saved_runs if run.pair_label in selected_pair_names]
    keep_only_latest_rerun_per_seed = isempty(raw) ? false :
        Bool(_setting(first(raw).analysis_settings, :use_latest_run_per_seed))
    if keep_only_latest_rerun_per_seed
        keep = Dict{Tuple{String,Any},Any}()
        for run in raw
            key = (run.pair_label, run.seed)
            if !haskey(keep, key) || string(run.timestamp) > string(keep[key].timestamp)
                keep[key] = run
            end
        end
        sort(collect(values(keep)), by=run -> (run.pair_label, string(run.seed), run.timestamp))
    else
        sort(raw, by=run -> (run.pair_label, string(run.seed), run.timestamp))
    end
end;

# ╔═╡ a4c73d31-cada-44dd-81c8-fbb0f5e84f6a
md"Selected **$(length(selected_runs))** trained runs across **$(length(unique([r.pair_label for r in selected_runs])))** receiver pairs."

# ╔═╡ ddfd42a8-4ae7-408a-8b8f-2d335746798b
run_source_state_averages = let
    reload_key = reload_saved_artifacts_button
    out = Vector{Any}(undef, length(selected_runs))
    @progress name="Loading artifacts" for i in eachindex(selected_runs)
        out[i] = _load_saved_source_state_averages(selected_runs[i])
    end
    out
end

# ╔═╡ b7c7a358-71c8-4797-a259-68bc75ab6e65
md"Loaded source-state artifacts for **$(length(run_source_state_averages))** trained runs using **$(Threads.nthreads())** Julia threads."

# ╔═╡ b01af348-c4b0-4ad1-81cd-116e9f2ed765
pair_labels = sort(unique([item.pair_label for item in run_source_state_averages]))

# ╔═╡ d7a8effd-1bc2-4f21-a947-7e8bbf82349a
@bind selected_plot_pair Select(pair_labels)

# ╔═╡ 50bcfca1-b35c-4af5-8da7-26e6a9aa7914
source_state_plot_runs = begin
    items = [item for item in run_source_state_averages if item.pair_label == selected_plot_pair]
    labels = String[]
    for (i, item) in enumerate(items)
        run_label = basename(item.run_dir)
        push!(labels, "$(item.pair_label) seed $(item.seed) | $(run_label)")
    end
    (; items, labels)
end;

# ╔═╡ f30a7fe9-67f2-438e-9b9c-57e41b583c0e
if isempty(source_state_plot_runs.labels)
    md"No saved runs available for $(selected_plot_pair)."
else
    @bind selected_source_state_run Select(source_state_plot_runs.labels)
end

# ╔═╡ dccacb14-c2de-43a2-943c-0e52a5e1276f
selected_source_state_item = begin
    if isempty(source_state_plot_runs.items) || !@isdefined(selected_source_state_run)
        nothing
    else
        idx = selected_source_state_run isa Integer ?
            Int(selected_source_state_run) :
            something(findfirst(==(String(selected_source_state_run)), source_state_plot_runs.labels), 1)
        source_state_plot_runs.items[idx]
    end
end;

# ╔═╡ c3000006-0000-0000-0000-000000000001
let
    item = selected_source_state_item
    if isnothing(item)
        md""
    else
        has_joint  = !isempty(item.codebook_joint_waves)
        has_stage1 = !isempty(item.codebook_stage1_waves)
        has_stage2 = !isempty(item.codebook_stage2_waves)
        opts = ["stage1"]
        has_stage2  && push!(opts, "stage2")
        (has_stage1 || has_stage2) && push!(opts, "K1+K2")
        has_joint   && push!(opts, "joint (K1×K2)")
        @bind ui_codebook_mode Select(opts; default=last(opts))
    end
end

# ╔═╡ c3000008-0000-0000-0000-000000000001
let
    items = [item for item in run_source_state_averages
             if item.pair_label == selected_plot_pair]
    if isempty(items)
        md"No runs for $selected_plot_pair."
    else

    mode = @isdefined(ui_codebook_mode) ? String(ui_codebook_mode) : "joint (K1×K2)"
    threshold = @isdefined(ui_codebook_ncc_threshold) ? Float64(ui_codebook_ncc_threshold) : 0.80

    ncc_local(a, b) = begin
        a_ = a .- mean(a); b_ = b .- mean(b)
        dot(a_, b_) / (norm(a_) * norm(b_) + 1e-8)
    end

    all_waves  = Vector{Float32}[]
    all_labels = String[]
    all_seeds  = Int[]

    for item in items
        waves, labels, _ = _codebook_family_waves(item, mode)
        isempty(waves) && continue
        K = size(waves, 2)
        for k in 1:K
            lbl = k <= length(labels) ? labels[k] : string(k)
            push!(all_waves, waves[:, k])
            push!(all_labels, "s$(item.seed)-$lbl")
            push!(all_seeds, item.seed)
        end
    end

    N = length(all_waves)
    if N == 0
        md"No codebook waveforms found for $selected_plot_pair ($mode)."
    else

    ncc_mat = ones(Float64, N, N)
    for i in 1:N, j in i+1:N
        v = ncc_local(Float64.(all_waves[i]), Float64.(all_waves[j]))
        ncc_mat[i, j] = v
        ncc_mat[j, i] = v
    end

    n_repeated = sum(
        any(ncc_mat[i, j] >= threshold && all_seeds[i] != all_seeds[j] for j in 1:N if j != i)
        for i in 1:N)

    hm = PlutoPlotly.heatmap(
        z=ncc_mat,
        x=all_labels, y=all_labels,
        colorscale="RdBu", zmid=0.0, zmin=-1.0, zmax=1.0,
        colorbar=PlutoPlotly.attr(title="NCC"),
    )
    layout = PlutoPlotly.Layout(
        title="Cross-Seed Codebook NCC — $selected_plot_pair ($mode) | threshold=$(threshold) | $n_repeated/$N codes repeated",
        xaxis=PlutoPlotly.attr(title="Code", tickangle=-45),
        yaxis=PlutoPlotly.attr(title="Code", autorange="reversed"),
        width=max(600, 60 + 55 * N),
        height=max(550, 80 + 55 * N),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot([hm], layout))
    end  # N == 0
    end  # isempty(items)
end

# ╔═╡ c3000009-0000-0000-0000-000000000001
let
    items = [item for item in run_source_state_averages
             if item.pair_label == selected_plot_pair]
    if isempty(items)
        md""
    else
        mode = @isdefined(ui_codebook_mode) ? String(ui_codebook_mode) : "joint (K1×K2)"
        threshold = @isdefined(ui_codebook_ncc_threshold) ? Float64(ui_codebook_ncc_threshold) : 0.80

        ncc_local(a, b) = begin
            a_ = a .- mean(a); b_ = b .- mean(b)
            dot(a_, b_) / (norm(a_) * norm(b_) + 1e-8)
        end

        all_waves  = Vector{Float32}[]
        all_labels = String[]
        all_seeds  = Int[]

        for item in items
            waves, labels, _ = _codebook_family_waves(item, mode)
            isempty(waves) && continue
            K = size(waves, 2)
            for k in 1:K
                lbl = k <= length(labels) ? labels[k] : string(k)
                push!(all_waves, waves[:, k])
                push!(all_labels, "s$(item.seed)-$lbl")
                push!(all_seeds, item.seed)
            end
        end

        N = length(all_waves)
        if N == 0
            md""
        else
            parent = collect(1:N)
            find(x) = parent[x] == x ? x : (parent[x] = find(parent[x]); parent[x])
            function unite!(x, y)
                px, py = find(x), find(y)
                px != py && (parent[px] = py)
            end
            for i in 1:N, j in i+1:N
                all_seeds[i] == all_seeds[j] && continue
                v = ncc_local(Float64.(all_waves[i]), Float64.(all_waves[j]))
                v >= threshold && unite!(i, j)
            end

            clusters = Dict{Int,Vector{Int}}()
            for i in 1:N
                push!(get!(clusters, find(i), Int[]), i)
            end
            repeated = sort(
                filter(kv -> length(unique(all_seeds[kv.second])) > 1, collect(clusters)),
                by=kv -> -length(kv.second),
            )

            if isempty(repeated)
                md"No cross-seed NCC ≥ $(threshold) matches found for **$(selected_plot_pair)** ($(mode))."
            else
                rows = ["| Cluster | Codes | Seeds | Min cross-seed NCC |",
                        "|---:|---|---|---:|"]
                for (ci, (_, idxs)) in enumerate(repeated)
                    codes_str = join(all_labels[idxs], ", ")
                    seeds_str = join(sort(unique(all_seeds[idxs])), ", ")
                    min_ncc = minimum(
                        ncc_local(Float64.(all_waves[i]), Float64.(all_waves[j]))
                        for i in idxs, j in idxs
                        if i < j && all_seeds[i] != all_seeds[j];
                        init=1.0,
                    )
                    push!(rows, "| $ci | $codes_str | $seeds_str | $(round(min_ncc; digits=3)) |")
                end
                Markdown.parse(join(rows, "\n"))
            end
        end
    end
end

# ╔═╡ eb76dfcf-b8ce-445d-a152-52eb8a6f94a7
analysis_settings = begin
    isempty(all_saved_runs) && error("No saved source-state artifacts are available.")
    settings_run = first(all_saved_runs)
    settings = settings_run.analysis_settings
    mismatched = [run.run_dir for run in all_saved_runs if run.analysis_settings != settings]
    isempty(mismatched) ||
        @warn "Saved artifacts have different analysis_settings; using the first available artifact." first_run=settings_run.run_dir mismatched_count=length(mismatched)
    settings
end

# ╔═╡ f01dc5e7-ae8a-4c31-a8df-64cabd2abe35
begin
    dt = Float64(_setting(analysis_settings, :dt))
    period_min = Float64(_setting(analysis_settings, :period_min))
    period_max = Float64(_setting(analysis_settings, :period_max))
    mft_nperiods = Int(_setting(analysis_settings, :mft_nperiods))
    mft_max_modes = hasproperty(analysis_settings, :mft_max_modes) ?
        Int(getproperty(analysis_settings, :mft_max_modes)) : 6
    velocity_range_saved = _setting(analysis_settings, :velocity_range)
    velocity_range = (Float64(velocity_range_saved[1]), Float64(velocity_range_saved[2]))
    bandwidth_factor = Float64(_setting(analysis_settings, :bandwidth_factor))
    zero_pad_factor = Int(_setting(analysis_settings, :zero_pad_factor))

    function mft_title_context(pair_label, distance=nothing; seed=nothing, detail=nothing)
        labels = String[String(pair_label)]
        !isnothing(seed) && push!(labels, "seed=$(seed)")
        if isnothing(distance) || !isfinite(Float64(distance))
            push!(labels, "distance unavailable")
        else
            push!(labels, "$(round(Int, Float64(distance)))km")
        end
        push!(labels, "$(period_min)-$(period_max)s")
        !isnothing(detail) && !isempty(String(detail)) && push!(labels, String(detail))
        return join(labels, " ")
    end

    function mft_title_context(pair_label, batch::mft.BranchBatchAnalysisResult;
                               seed=nothing, detail=nothing)
        distance = isempty(batch.state_results) ? nothing : first(batch.state_results).distance
        return mft_title_context(pair_label, distance; seed=seed, detail=detail)
    end
end

# ╔═╡ d4c01002-0000-0000-0000-000000000001
function _normalized_codebook_psd(waves::AbstractMatrix, dt::Real)
    nt, ncols = size(waves)
    freqs = fftfreq(nt, inv(Float64(dt)))
    pos = findall(freqs .> 0)
    periods = 1.0 ./ Float64.(freqs[pos])
    order = sortperm(periods)
    periods = periods[order]
    psd = zeros(Float64, length(pos), ncols)
    for k in 1:ncols
        p = abs2.(fft(Float64.(waves[:, k])))
        pk = Float64.(p[pos][order])
        mx = maximum(pk)
        psd[:, k] .= mx > 0.0 && isfinite(mx) ? pk ./ mx : pk
    end
    (; periods, psd)
end

# ╔═╡ d4c01003-0000-0000-0000-000000000001
function _spectral_overlap_matrix(psd1::AbstractMatrix, psd2::AbstractMatrix)
    K1 = size(psd1, 2)
    K2 = size(psd2, 2)
    overlap = zeros(Float64, K1, K2)
    for i in 1:K1, j in 1:K2
        a = Float64.(psd1[:, i])
        b = Float64.(psd2[:, j])
        den = sqrt(sum(abs2, a) * sum(abs2, b))
        overlap[i, j] = den > 0.0 && isfinite(den) ? sum(a .* b) / den : NaN
    end
    overlap
end

# ╔═╡ d4c00001-0000-0000-0000-000000000001
if isnothing(selected_source_state_item)
    md""
else
    let item = selected_source_state_item
        has_s1 = !isempty(item.codebook_stage1_waves)
        has_s2 = !isempty(item.codebook_stage2_waves)
        (has_s1 || has_s2) || md"No codebook waves in artifact."
        nt = has_s1 ? size(item.codebook_stage1_waves, 1) : size(item.codebook_stage2_waves, 1)
        t = collect(0:nt-1) .* dt
        K1 = has_s1 ? size(item.codebook_stage1_waves, 2) : 0
        K2 = has_s2 ? size(item.codebook_stage2_waves, 2) : 0
        spacing = 2.5
        traces = AbstractTrace[]
        # Stage-1 codes: decoder1 waveforms
        for k in 1:K1
            w = Float64.(item.codebook_stage1_waves[:, k])
            nrm = max(maximum(abs.(w)), 1e-8)
            push!(traces, PlutoPlotly.scatter(
                x=t, y=w ./ nrm .+ (k - 1) * spacing,
                mode="lines", name="s1-$k",
                line=PlutoPlotly.attr(color="royalblue", width=1.5),
                showlegend=k == 1,
                legendgroup="stage1",
                legendgrouptitle_text=k == 1 ? "decoder1 (K1=$K1)" : nothing,
            ))
        end
        # Stage-2 codes: decoder2 waveforms, offset above stage-1
        s2_offset = (K1 + 1) * spacing
        for k in 1:K2
            w = Float64.(item.codebook_stage2_waves[:, k])
            nrm = max(maximum(abs.(w)), 1e-8)
            push!(traces, PlutoPlotly.scatter(
                x=t, y=w ./ nrm .+ (s2_offset + (k - 1) * spacing),
                mode="lines", name="s2-$k",
                line=PlutoPlotly.attr(color="tomato", width=1.5),
                showlegend=k == 1,
                legendgroup="stage2",
                legendgrouptitle_text=k == 1 ? "decoder2 (K2=$K2)" : nothing,
            ))
        end
        # Tick labels on y-axis
        tickvals = vcat(
            [(k - 1) * spacing for k in 1:K1],
            [s2_offset + (k - 1) * spacing for k in 1:K2]
        )
        ticktext = vcat(
            ["s1-$k" for k in 1:K1],
            ["s2-$k" for k in 1:K2]
        )
        layout = PlutoPlotly.Layout(
            title="All Codebook Waveforms ($(mft_title_context(item.pair_label, item.distance; seed=item.seed))) (blue=decoder1, red=decoder2)",
            xaxis_title="Time (s)",
            yaxis=PlutoPlotly.attr(
                title="Code", tickmode="array",
                tickvals=tickvals, ticktext=ticktext, showgrid=true
            ),
            height=max(350, 60 * (K1 + K2 + 1)),
            width=950,
            plot_bgcolor="white", paper_bgcolor="white",
            legend=PlutoPlotly.attr(groupclick="toggleitem"),
        )
        WideCell(PlutoPlotly.plot(traces, layout))
    end
end

# ╔═╡ d4c01004-0000-0000-0000-000000000001
if isnothing(selected_source_state_item)
    md""
else
    let item = selected_source_state_item
        has_s1 = !isempty(item.codebook_stage1_waves)
        has_s2 = !isempty(item.codebook_stage2_waves)
        if !(has_s1 && has_s2)
            md"Both stage-1 and stage-2 decoded codebook waveforms are required for PSD exclusivity."
        else
            s1 = _normalized_codebook_psd(item.codebook_stage1_waves, dt)
            s2 = _normalized_codebook_psd(item.codebook_stage2_waves, dt)
            traces = AbstractTrace[]
            K1 = size(item.codebook_stage1_waves, 2)
            K2 = size(item.codebook_stage2_waves, 2)
            for k in 1:K1
                label = k <= length(item.codebook_stage1_labels) ? item.codebook_stage1_labels[k] : "s1=$k"
                push!(traces, PlutoPlotly.scatter(
                    x=s1.periods, y=s1.psd[:, k],
                    mode="lines",
                    name="S1 $(label)",
                    legendgroup="stage1-psd",
                    line=PlutoPlotly.attr(color="rgba(65,105,225,0.72)", width=1.7),
                    text=["S1 $(label)<br>T=$(round(T; digits=3)) s<br>normalized PSD=$(round(P; digits=4))" for (T, P) in zip(s1.periods, s1.psd[:, k])],
                    hoverinfo="text",
                    showlegend=k == 1,
                ))
            end
            for k in 1:K2
                label = k <= length(item.codebook_stage2_labels) ? item.codebook_stage2_labels[k] : "s2=$k"
                push!(traces, PlutoPlotly.scatter(
                    x=s2.periods, y=s2.psd[:, k],
                    mode="lines",
                    name="S2 $(label)",
                    legendgroup="stage2-psd",
                    line=PlutoPlotly.attr(color="rgba(255,99,71,0.78)", width=1.7),
                    text=["S2 $(label)<br>T=$(round(T; digits=3)) s<br>normalized PSD=$(round(P; digits=4))" for (T, P) in zip(s2.periods, s2.psd[:, k])],
                    hoverinfo="text",
                    showlegend=k == 1,
                ))
            end
            shapes = [
                PlutoPlotly.attr(
                    type="rect",
                    xref="x", yref="paper",
                    x0=period_min, x1=period_max,
                    y0=0.0, y1=1.0,
                    fillcolor="rgba(120,120,120,0.10)",
                    line=PlutoPlotly.attr(width=0),
                    layer="below",
                )
            ]
            WideCell(PlutoPlotly.plot(traces, PlutoPlotly.Layout(
                title="Normalized Codebook PSD Overlay ($(mft_title_context(item.pair_label, item.distance; seed=item.seed)))",
                xaxis=PlutoPlotly.attr(title="Period (s)", type="log", showgrid=true),
                yaxis=PlutoPlotly.attr(title="Normalized PSD", type="log", range=[-6, 0.05], showgrid=true),
                shapes=shapes,
                height=430, width=980,
                plot_bgcolor="white", paper_bgcolor="white",
                legend=PlutoPlotly.attr(groupclick="toggleitem"),
                margin=PlutoPlotly.attr(l=70, r=25, t=65, b=60),
            )))
        end
    end
end

# ╔═╡ d4c01005-0000-0000-0000-000000000001
if isnothing(selected_source_state_item)
    md""
else
    let item = selected_source_state_item
        has_s1 = !isempty(item.codebook_stage1_waves)
        has_s2 = !isempty(item.codebook_stage2_waves)
        if !(has_s1 && has_s2)
            md""
        else
            s1 = _normalized_codebook_psd(item.codebook_stage1_waves, dt)
            s2 = _normalized_codebook_psd(item.codebook_stage2_waves, dt)
            overlap = _spectral_overlap_matrix(s1.psd, s2.psd)
            labels1 = [k <= length(item.codebook_stage1_labels) ? item.codebook_stage1_labels[k] : "s1=$k"
                for k in axes(item.codebook_stage1_waves, 2)]
            labels2 = [k <= length(item.codebook_stage2_labels) ? item.codebook_stage2_labels[k] : "s2=$k"
                for k in axes(item.codebook_stage2_waves, 2)]
            text = ["S1 $(labels1[i]) vs S2 $(labels2[j])<br>spectral overlap=$(round(overlap[i, j]; digits=4))"
                for i in axes(overlap, 1), j in axes(overlap, 2)]
            WideCell(PlutoPlotly.plot([
                PlutoPlotly.heatmap(
                    z=overlap,
                    x=labels2,
                    y=labels1,
                    colorscale="Viridis",
                    zmin=0.0,
                    zmax=1.0,
                    text=text,
                    hoverinfo="text",
                    colorbar=PlutoPlotly.attr(title="overlap"),
                )
            ], PlutoPlotly.Layout(
                title="Stage-1 vs Stage-2 Normalized PSD Overlap ($(mft_title_context(item.pair_label, item.distance; seed=item.seed)))",
                xaxis_title="Stage 2 code",
                yaxis_title="Stage 1 code",
                height=max(360, 55 * length(labels1)),
                width=650,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=PlutoPlotly.attr(l=80, r=90, t=65, b=65),
            )))
        end
    end
end

# ╔═╡ d4c00002-0000-0000-0000-000000000001
if isnothing(selected_source_state_item)
    md""
else
    let item = selected_source_state_item
        has_s1 = !isempty(item.marginal_stage1_ac) && !isempty(item.marginal_stage1_c)
        has_s2 = !isempty(item.marginal_stage2_ac) && !isempty(item.marginal_stage2_c)
        (has_s1 || has_s2) || md"No per-stage marginal averages in artifact."
        nt = has_s1 ? size(item.marginal_stage1_ac, 1) : size(item.marginal_stage2_ac, 1)
        t_neg = [-(nt - i + 1) * dt for i in 1:nt]
        t_pos = [i * dt for i in 1:nt]
        t_full = [t_neg; t_pos]
        K1 = has_s1 ? size(item.marginal_stage1_ac, 2) : 0
        K2 = has_s2 ? size(item.marginal_stage2_ac, 2) : 0
        ncodes = K1 + K2

        # Normalise each column (same as plot_source_state_waveforms)
        col_norm(X) = mapslices(c -> begin nrm = max(maximum(abs.(c)), 1e-8); c ./ nrm end, X; dims=1)
        ncc_cols(a, b) = begin
            a_ = a .- mean(a); b_ = b .- mean(b)
            dot(a_, b_) / (norm(a_) * norm(b_) + 1e-8)
        end
        s1_ac_n = has_s1 ? col_norm(Float64.(item.marginal_stage1_ac)) : zeros(nt, 0)
        s1_c_n  = has_s1 ? col_norm(Float64.(item.marginal_stage1_c))  : zeros(nt, 0)
        s2_ac_n = has_s2 ? col_norm(Float64.(item.marginal_stage2_ac)) : zeros(nt, 0)
        s2_c_n  = has_s2 ? col_norm(Float64.(item.marginal_stage2_c))  : zeros(nt, 0)

        # Marginal counts: prefer saved values, else derive from joint counts_ac/c (always present)
        derive_marginals(counts_flat, k1, k2) = begin
            W = reshape(Int.(round.(counts_flat)), k1, k2)
            vec(sum(W; dims=2)), vec(sum(W; dims=1))  # (K1,), (K2,)
        end
        cnt1_ac, cnt2_ac = if !isempty(item.marginal_stage1_counts_ac) && K1 > 0 && K2 > 0
            item.marginal_stage1_counts_ac, item.marginal_stage2_counts_ac
        elseif K1 > 0 && K2 > 0 && length(item.counts_ac) == K1 * K2
            derive_marginals(item.counts_ac, K1, K2)
        else
            ones(Int, K1), ones(Int, K2)
        end
        cnt1_c, cnt2_c = if !isempty(item.marginal_stage1_counts_c) && K1 > 0 && K2 > 0
            item.marginal_stage1_counts_c, item.marginal_stage2_counts_c
        elseif K1 > 0 && K2 > 0 && length(item.counts_c) == K1 * K2
            derive_marginals(item.counts_c, K1, K2)
        else
            ones(Int, K1), ones(Int, K2)
        end
        tot1_ac = max(sum(cnt1_ac), 1); tot1_c = max(sum(cnt1_c), 1)
        tot2_ac = max(sum(cnt2_ac), 1); tot2_c = max(sum(cnt2_c), 1)

        amp_peak = maximum(abs.(vcat(
            vec(s1_ac_n), vec(s1_c_n), vec(s2_ac_n), vec(s2_c_n), [1e-3]
        )))
        vertical_spacing = amp_peak * 2.5 + 1e-3

        labels1 = has_s1 ? item.marginal_stage1_labels : String[]
        labels2 = has_s2 ? item.marginal_stage2_labels : String[]

        # Global average — each branch normalised separately, then mirrored
        global_ac_raw = Float64.(vec(item.global_avg_ac))
        global_c_raw  = Float64.(vec(item.global_avg_c))
        global_ac_n = global_ac_raw ./ max(maximum(abs.(global_ac_raw)), 1e-8)
        global_c_n  = global_c_raw  ./ max(maximum(abs.(global_c_raw)),  1e-8)
        global_full = [reverse(global_ac_n); global_c_n]
        global_ncc = round(ncc_cols(global_ac_raw, global_c_raw); digits=3)

        traces = AbstractTrace[]

        # Stage-1 codes (blue shades)
        for k in 1:K1
            ac_k = s1_ac_n[:, k]
            c_k  = s1_c_n[:, k]
            full_k = [reverse(ac_k); c_k]
            offset = (k - 1) * vertical_spacing
            label_k = k <= length(labels1) ? labels1[k] : "s1-$k"
            pct_ac = round(100 * cnt1_ac[k] / tot1_ac; digits=1)
            pct_c  = round(100 * cnt1_c[k]  / tot1_c;  digits=1)
            ncc    = round(ncc_cols(ac_k, c_k); digits=3)
            legend_text = "S1-$label_k  ac:$(pct_ac)%  c:$(pct_c)%  corr:$ncc"
            push!(traces, PlutoPlotly.scatter(
                x=t_full, y=global_full .+ offset, mode="lines",
                name=k == 1 ? "Global mean (corr:$global_ncc)" : "Global mean",
                showlegend=k == 1,
                line=PlutoPlotly.attr(color="rgba(0,0,0,0.18)", width=3),
                legendgroup="global",
            ))
            push!(traces, PlutoPlotly.scatter(
                x=t_full, y=full_k .+ offset, mode="lines",
                name=legend_text,
                line=PlutoPlotly.attr(color="royalblue", width=2),
                legendgroup="stage1",
                showlegend=true,
            ))
        end

        # Stage-2 codes (red shades), offset above stage-1
        s2_base = K1 * vertical_spacing
        for k in 1:K2
            ac_k = s2_ac_n[:, k]
            c_k  = s2_c_n[:, k]
            full_k = [reverse(ac_k); c_k]
            offset = s2_base + (k - 1) * vertical_spacing
            label_k = k <= length(labels2) ? labels2[k] : "s2-$k"
            pct_ac = round(100 * cnt2_ac[k] / tot2_ac; digits=1)
            pct_c  = round(100 * cnt2_c[k]  / tot2_c;  digits=1)
            ncc    = round(ncc_cols(ac_k, c_k); digits=3)
            legend_text = "S2-$label_k  ac:$(pct_ac)%  c:$(pct_c)%  corr:$ncc"
            push!(traces, PlutoPlotly.scatter(
                x=t_full, y=global_full .+ offset, mode="lines",
                name="Global mean",
                showlegend=false,
                line=PlutoPlotly.attr(color="rgba(0,0,0,0.18)", width=3),
                legendgroup="global",
            ))
            push!(traces, PlutoPlotly.scatter(
                x=t_full, y=full_k .+ offset, mode="lines",
                name=legend_text,
                line=PlutoPlotly.attr(color="tomato", width=2),
                legendgroup="stage2",
                showlegend=true,
            ))
        end

        # Velocity marker lines
        shapes = if isnothing(item.distance)
            []
        else
            vmin, vmax = velocity_range
            t_fast = item.distance / vmax
            t_slow = item.distance / vmin
            [PlutoPlotly.attr(type="line", x0=t, x1=t, y0=0, y1=1, yref="paper",
                line=PlutoPlotly.attr(color="rgba(0,0,0,0.25)", width=1, dash="dash"))
             for t in (-t_slow, -t_fast, t_fast, t_slow)]
        end

        layout = PlutoPlotly.Layout(
            title="Per-Stage Marginal Averaged Waveforms ($(mft_title_context(item.pair_label, item.distance; seed=item.seed))) (blue=decoder1/Stage1, red=decoder2/Stage2)",
            xaxis=PlutoPlotly.attr(title="Lag (s)", zeroline=true, zerolinecolor="rgba(0,0,0,0.3)"),
            yaxis=PlutoPlotly.attr(title="Code (amplitude offset)"),
            height=max(400, 80 * (ncodes + 1)),
            width=1000,
            plot_bgcolor="white", paper_bgcolor="white",
            shapes=shapes,
            legend=PlutoPlotly.attr(
                x=0.5, xanchor="center", y=-0.15, orientation="h",
                groupclick="toggleitem",
                font=PlutoPlotly.attr(size=12),
            ),
        )
        WideCell(PlutoPlotly.plot(traces, layout))
    end
end

# ╔═╡ c753f15f-12ab-4f54-92f2-59d96197f85c
if isnothing(selected_source_state_item)
    md""
else
    WideCell(mft.plot_source_state_waveforms(
        selected_source_state_item;
        dt=dt,
        velocity_range=velocity_range,
        period_min=period_min,
        period_max=period_max,
    ))
end

# ╔═╡ 067d2587-8eb1-41c3-95f8-9f785171f2ce
mft_periods = exp10.(range(log10(Float64(period_min)), log10(Float64(period_max)); length=mft_nperiods))

# ╔═╡ b2000001-0000-0000-0000-000000000001
if isnothing(selected_source_state_item)
    md""
else
    WideCell(mft.plot_cluster_histogram(
        selected_source_state_item.counts_ac,
        selected_source_state_item.counts_c;
        labels=selected_source_state_item.combo_labels,
        title="Source State Usage ($(mft_title_context(selected_source_state_item.pair_label, selected_source_state_item.distance; seed=selected_source_state_item.seed)))",
    ))
end

# ╔═╡ b2000002-0000-0000-0000-000000000001
if isnothing(selected_source_state_item)
    md""
else
    WideCell(mft.plot_state_ncc_heatmap(
        selected_source_state_item.acausal,
        selected_source_state_item.causal;
        labels=selected_source_state_item.combo_labels,
        title="State-State NCC ($(mft_title_context(selected_source_state_item.pair_label, selected_source_state_item.distance; seed=selected_source_state_item.seed)))",
    ))
end

# ╔═╡ c3000016-0000-0000-0000-000000000001
begin
	function _plot_codebook_uc_agreement(pair_label::String, distance::Real, summary;
	        title_prefix::String="Codebook U-c Agreement",
	        relative_tolerance::Float64=0.10,
	        global_avg_ac=nothing,
	        global_avg_c=nothing,
	        global_avg_raw_ac=nothing,
	        global_avg_raw_c=nothing,
	        periods_ref=nothing)
	    rows = summary.rows
	    rows_are_high_snr = !isempty(rows)
	    if isempty(rows)
	        res = summary.result
	        u_pred = any(isfinite, res.u_predicted_from_phase) ?
	            res.u_predicted_from_phase : mft.compute_group_velocity_from_phase(res)
	        fallback_rows = NamedTuple[]
	        for ip in eachindex(res.periods)
	            period = Float64(res.periods[ip])
	            u_meas = Float64(res.group_velocities[ip])
	            u_hat = Float64(u_pred[ip])
	            quality = Float64(res.quality_factors[ip])
	            phase_velocity = Float64(res.phase_velocities[ip])
	            isfinite(period) && period > 0 || continue
	            isfinite(u_meas) && u_meas > 0 || continue
	            isfinite(u_hat) && u_hat > 0 || continue
	            relerr = abs(u_meas - u_hat) / max(abs(u_hat), eps(Float64))
	            push!(fallback_rows, (; pair_label, label=summary.label, period,
	                group_velocity=u_meas, predicted_group_velocity=u_hat,
	                phase_velocity, relative_error=relerr, quality,
	                pass=relerr <= relative_tolerance,
	                phase_suspect=ip <= length(res.phase_suspect) ? Bool(res.phase_suspect[ip]) : false))
	        end
	        rows = fallback_rows
	    end
	    if isempty(rows)
	        PlutoPlotly.plot(PlutoPlotly.scatter(
	            x=[0.0], y=[0.0], text=["No finite U-c agreement periods for $(summary.label)"]))
	    else
	        quality_note = rows_are_high_snr ? "high-SNR periods" : "all finite periods"
	        traces = AbstractTrace[]
	        push!(traces, PlutoPlotly.scatter(
	            x=[r.period for r in rows],
	            y=[r.group_velocity for r in rows],
	            mode="lines+markers",
	            name="U_meas",
	            marker=PlutoPlotly.attr(size=8, color=[r.relative_error for r in rows],
	                colorscale="Viridis", colorbar=PlutoPlotly.attr(title="rel err")),
	            line=PlutoPlotly.attr(color="black", width=2),
	            text=["$(quality_note)<br>quality=$(round(r.quality; digits=2))<br>rel err=$(round(r.relative_error; digits=3))<br>pass=$(r.pass)" for r in rows],
	        ))
	        push!(traces, PlutoPlotly.scatter(
	            x=[r.period for r in rows],
	            y=[r.predicted_group_velocity for r in rows],
	            mode="lines+markers",
	            name="U_pred from c(T)",
	            marker=PlutoPlotly.attr(size=8, color="firebrick", symbol="diamond"),
	            line=PlutoPlotly.attr(color="firebrick", width=2, dash="dash"),
	        ))
	        push!(traces, PlutoPlotly.scatter(
	            x=[r.period for r in rows],
	            y=[r.relative_error for r in rows],
	            yaxis="y2",
	            mode="lines+markers",
	            name="relative agreement error",
	            marker=PlutoPlotly.attr(size=7, color="royalblue", symbol="circle-open"),
	            line=PlutoPlotly.attr(color="royalblue", width=1),
	            text=["U=$(round(r.group_velocity; digits=3)) km/s<br>U_pred=$(round(r.predicted_group_velocity; digits=3)) km/s<br>pass=$(r.pass)" for r in rows],
	        ))
	        
	        # Add global average reference picks if available
	        if !isnothing(global_avg_raw_ac) && !isempty(global_avg_raw_ac) && !isnothing(periods_ref)
	            try
	                # Reshape to column vector for MFT analysis
	                W_ac = Float32.(reshape(global_avg_raw_ac, :, 1))
	                bank = mft._mft_filter_bank_for(periods_ref, size(W_ac, 1); 
	                    storage_mode=:picks_only, n_waveforms=1)
	                if !isnothing(bank)
	                    mres_ac = mft.perform_mft_analysis_batch!(bank, W_ac, distance; compute_phase=true)
	                    if !isempty(mres_ac)
	                        res_ac = mres_ac[1]
	                        if any(isfinite, res_ac.group_velocities)
	                            u_pred_ac = any(isfinite, res_ac.u_predicted_from_phase) ?
	                                res_ac.u_predicted_from_phase : mft.compute_group_velocity_from_phase(res_ac)
	                            periods_ac = Float64.(res_ac.periods)
	                            u_meas_ac = Float64.(res_ac.group_velocities)
	                            valid_ac = isfinite.(periods_ac) .& (periods_ac .> 0) .& isfinite.(u_meas_ac) .& (u_meas_ac .> 0)
	                            if any(valid_ac)
	                                push!(traces, PlutoPlotly.scatter(
	                                    x=periods_ac[valid_ac],
	                                    y=u_meas_ac[valid_ac],
	                                    mode="lines+markers",
	                                    name="Global avg acausal picks",
	                                    marker=PlutoPlotly.attr(size=6, color="green", symbol="cross"),
	                                    line=PlutoPlotly.attr(color="green", width=2, dash="dot"),
	                                    opacity=0.7,
	                                ))
	                            end
	                        end
	                    end
	                end
	            catch e
	            end
	        end
	        if !isnothing(global_avg_raw_c) && !isempty(global_avg_raw_c) && !isnothing(periods_ref)
	            try
	                # Reshape to column vector for MFT analysis
	                W_c = Float32.(reshape(global_avg_raw_c, :, 1))
	                bank = mft._mft_filter_bank_for(periods_ref, size(W_c, 1); 
	                    storage_mode=:picks_only, n_waveforms=1)
	                if !isnothing(bank)
	                    mres_c = mft.perform_mft_analysis_batch!(bank, W_c, distance; compute_phase=true)
	                    if !isempty(mres_c)
	                        res_c = mres_c[1]
	                        if any(isfinite, res_c.group_velocities)
	                            u_pred_c = any(isfinite, res_c.u_predicted_from_phase) ?
	                                res_c.u_predicted_from_phase : mft.compute_group_velocity_from_phase(res_c)
	                            periods_c = Float64.(res_c.periods)
	                            u_meas_c = Float64.(res_c.group_velocities)
	                            valid_c = isfinite.(periods_c) .& (periods_c .> 0) .& isfinite.(u_meas_c) .& (u_meas_c .> 0)
	                            if any(valid_c)
	                                push!(traces, PlutoPlotly.scatter(
	                                    x=periods_c[valid_c],
	                                    y=u_meas_c[valid_c],
	                                    mode="lines+markers",
	                                    name="Global avg causal picks",
	                                    marker=PlutoPlotly.attr(size=6, color="purple", symbol="cross"),
	                                    line=PlutoPlotly.attr(color="purple", width=2, dash="dot"),
	                                    opacity=0.7,
	                                ))
	                            end
	                        end
	                    end
	                end
	            catch e
	            end
	        end
	        
	        PlutoPlotly.plot(traces, PlutoPlotly.Layout(
	            title="$(title_prefix): $(quality_note) ($(mft_title_context(pair_label, distance; detail=summary.label)))",
	            xaxis=PlutoPlotly.attr(title="Period (s)", type="log"),
	            yaxis=PlutoPlotly.attr(title="Group velocity (km/s)"),
	            yaxis2=PlutoPlotly.attr(
	                title="relative |U_meas - U_pred| / U_pred",
	                overlaying="y",
	                side="right",
	                rangemode="tozero",
	            ),
	            width=1000, height=540,
	            plot_bgcolor="white", paper_bgcolor="white",
	        ))
	    end
	end
	
	function _all_finite_uc_rows(summary; relative_tolerance::Float64=0.10)
	    res = summary.result
	    u_pred = any(isfinite, res.u_predicted_from_phase) ?
	        res.u_predicted_from_phase : mft.compute_group_velocity_from_phase(res)
	    rows = NamedTuple[]
	    for ip in eachindex(res.periods)
	        period = Float64(res.periods[ip])
	        u_meas = Float64(res.group_velocities[ip])
	        u_hat = Float64(u_pred[ip])
	        quality = Float64(res.quality_factors[ip])
	        phase_velocity = Float64(res.phase_velocities[ip])
	        isfinite(period) && period > 0 || continue
	        isfinite(u_meas) && u_meas > 0 || continue
	        isfinite(u_hat) && u_hat > 0 || continue
	        relerr = abs(u_meas - u_hat) / max(abs(u_hat), eps(Float64))
	        push!(rows, (; pair_label=summary.pair_label, label=summary.label, period,
	            group_velocity=u_meas, predicted_group_velocity=u_hat,
	            phase_velocity, relative_error=relerr,
	            relative_agreement=1.0 / (1.0 + relerr),
	            quality, pass=relerr <= relative_tolerance,
	            high_snr=any(r -> r.period == period, summary.rows),
	            phase_suspect=ip <= length(res.phase_suspect) ? Bool(res.phase_suspect[ip]) : false))
	    end
	    rows
	end
	
	function _plot_all_codebook_uc_agreements(result;
	        title_prefix::String="All Codebook U-c Agreement")
	    all_rows = NamedTuple[]
	    for summary in result.summaries
	        append!(all_rows, _all_finite_uc_rows(summary;
	            relative_tolerance=Float64(result.relative_tolerance)))
	    end
	    if isempty(all_rows)
	        PlutoPlotly.plot(PlutoPlotly.scatter(
	            x=[0.0], y=[0.0],
	            text=["No finite U-c agreement periods for $(result.pair_label)"]))
	    else
	        by_label = Dict{String,Vector{NamedTuple}}()
	        for row in all_rows
	            push!(get!(by_label, row.label, NamedTuple[]), row)
	        end
	        traces = AbstractTrace[]
	        for label in sort(collect(keys(by_label)))
	            rows = sort(by_label[label], by=r -> r.period)
	            push!(traces, PlutoPlotly.scatter(
	                x=[r.period for r in rows],
	                y=[r.relative_agreement for r in rows],
	                mode="lines+markers",
	                name=label,
	                opacity=0.28,
	                line=PlutoPlotly.attr(width=1),
	                marker=PlutoPlotly.attr(
	                    size=6,
	                    color=[r.quality for r in rows],
	                    colorscale="Viridis",
	                    showscale=false,
	                ),
	                text=["$(r.label)<br>T=$(round(r.period; digits=3)) s<br>agreement=$(round(r.relative_agreement; digits=4))<br>rel err=$(round(r.relative_error; digits=4))<br>U=$(round(r.group_velocity; digits=3)) km/s<br>U_pred=$(round(r.predicted_group_velocity; digits=3)) km/s<br>quality=$(round(r.quality; digits=2))<br>high-SNR=$(r.high_snr)" for r in rows],
	                hovertemplate="%{text}<extra></extra>",
	                showlegend=false,
	            ))
	        end
	
	        best_rows = NamedTuple[]
	        for period in sort(unique(r.period for r in all_rows))
	            period_rows = [r for r in all_rows if r.period == period]
	            isempty(period_rows) && continue
	            best = sort(period_rows, by=r -> (-r.relative_agreement, -r.quality, r.label))[1]
	            push!(best_rows, best)
	        end
	        push!(traces, PlutoPlotly.scatter(
	            x=[r.period for r in best_rows],
	            y=[r.relative_agreement for r in best_rows],
	            mode="markers+lines",
	            name="best per period",
	            marker=PlutoPlotly.attr(
	                size=13,
	                color=[r.relative_error for r in best_rows],
	                colorscale="Plasma",
	                reversescale=true,
	                colorbar=PlutoPlotly.attr(title="best rel err"),
	                symbol="star",
	                line=PlutoPlotly.attr(color="black", width=1),
	            ),
	            line=PlutoPlotly.attr(color="black", width=2, dash="dash"),
	            text=["BEST<br>$(r.label)<br>T=$(round(r.period; digits=3)) s<br>agreement=$(round(r.relative_agreement; digits=4))<br>rel err=$(round(r.relative_error; digits=4))<br>U=$(round(r.group_velocity; digits=3)) km/s<br>U_pred=$(round(r.predicted_group_velocity; digits=3)) km/s<br>quality=$(round(r.quality; digits=2))" for r in best_rows],
	            hovertemplate="%{text}<extra></extra>",
	        ))
	        PlutoPlotly.plot(traces, PlutoPlotly.Layout(
	            title="$(title_prefix) ($(mft_title_context(result.pair_label, result.distance; detail=result.mode)))",
	            xaxis=PlutoPlotly.attr(title="Period (s)", type="log"),
	            yaxis=PlutoPlotly.attr(title="Relative agreement score, 1 / (1 + relative error)", range=[0, 1.02]),
	            width=1050,
	            height=560,
	            plot_bgcolor="white",
	            paper_bgcolor="white",
	            showlegend=true,
	        ))
	    end
	end
end

# ╔═╡ cc15f5d5-764d-4ee8-a8f2-862c5207c630
md"""
Using saved MFT settings:
`dt=$(dt)`, period band `$(period_min)-$(period_max)` s, `$(mft_nperiods)` periods,
velocity range `$(velocity_range)`, max modes `$(mft_max_modes)`, bandwidth factor `$(bandwidth_factor)`,
zero padding `$(zero_pad_factor)`.

Wavelength compute cutoff: `$(ui_wavelength_ref_velocity) km/s × period < $(ui_wavelength_fraction) × distance`.
"""

# ╔═╡ ed7943c5-d57e-4592-8fa6-e402edffd627
function _subset_branch_batch(batch, inds::AbstractVector{<:Integer})
        return mft.BranchBatchAnalysisResult(
            batch.state_results[inds],
            batch.branch_correlation[:, inds],
            batch.periods,
            batch.state_labels[inds],
        )
    end

# ╔═╡ 04212dbb-a028-4f26-9a96-f90bc36c00b2
function _split_batch_by_pair(batch, pair_keys::AbstractVector{<:AbstractString})
        grouped = Dict{String,Vector{Int}}()
        for (i, pair_label) in enumerate(pair_keys)
            push!(get!(grouped, String(pair_label), Int[]), i)
        end
        return Dict(pair_label => _subset_branch_batch(batch, inds)
                    for (pair_label, inds) in grouped)
    end

# ╔═╡ 0e351ef2-554f-4697-802f-74b785273c93
function _mft_compute_periods(distance::Real)
        return Float64[period for period in mft_periods
                       if mft.wavelength_valid_period(period, distance;
                           wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
                           wavelength_fraction=Float64(ui_wavelength_fraction))]
    end

# ╔═╡ 683da2a1-d2b9-4f21-880c-f93e945805fb
function _empty_branch_batch()
        return mft.BranchBatchAnalysisResult(
            mft.BranchAnalysisResult[],
            zeros(Float64, 0, 0),
            Float64[],
            String[],
        )
    end

# ╔═╡ fb000001-0000-0000-0000-000000000001
begin
    @bind _mft_runtime_controls PlutoUI.combine() do Child
        md"""
        | MFT runtime control | Value |
        |:---|:---|
        | Upsample factor | $(Child("upsample_factor", confirm(NumberField(1:1:10; default=2)))) |
        | Numeric precision | $(Child("precision", Select(["Float32" => "Float32 (default)", "Float64" => "Float64"]))) |
        """
    end
    ui_mft_upsample_factor = _mft_runtime_controls.upsample_factor
    ui_mft_precision = _mft_runtime_controls.precision
end

# ╔═╡ e9904c62-eb2f-4fd5-9dd9-4cfdf071b6e3
_mft_precision_type() = String(ui_mft_precision) == "Float64" ? Float64 : Float32

# ╔═╡ c98b88da-1ce4-4d8f-845d-7434e6c5f4ab
function _mft_filter_bank_for(periods::Vector{Float64}, npts_raw::Int;
                                  storage_mode::Symbol=:picks_only,
                                  n_waveforms::Int=1)
        isempty(periods) && return nothing
        key = (mft.MFTFilterBank, npts_raw, Tuple(periods), bandwidth_factor, zero_pad_factor,
               Float64(ui_mft_upsample_factor), velocity_range,
               _mft_precision_type(), storage_mode, n_waveforms)
        return get!(mft_filter_banks, key) do
            mft.MFTFilterBank(dt, npts_raw, periods;
                bandwidth_factor=bandwidth_factor,
                zero_pad_factor=zero_pad_factor,
                upsample_factor=Float64(ui_mft_upsample_factor),
                velocity_range=velocity_range,
                precision=_mft_precision_type(),
                storage_mode=storage_mode,
                N_initial=n_waveforms,
            )
        end
    end

# ╔═╡ c3000002-0000-0000-0000-000000000001
codebook_mft_analysis = let
    item = selected_source_state_item
    isnothing(item) && error("Select a saved run above.")

    has_joint  = !isempty(item.codebook_joint_waves)
    has_stage1 = !isempty(item.codebook_stage1_waves)
    has_stage2 = !isempty(item.codebook_stage2_waves)
    (has_stage1 || has_joint) || error("No codebook waves in artifact. Re-save this run.")

    mode = @isdefined(ui_codebook_mode) ? String(ui_codebook_mode) : (has_joint ? "joint (K1×K2)" : "stage1")

    waves, labels, _ = _codebook_family_waves(item, mode)

    dist    = Float64(item.distance)
    W_codes = Float64.(waves)  # (nt × K)

    global_ac = Float64.(vec(item.global_avg_ac))
    global_c  = Float64.(vec(item.global_avg_c))
    n = min(length(global_ac), length(global_c), size(W_codes, 1))
    periods = _mft_compute_periods(dist)
    bank = n > 0 ? _mft_filter_bank_for(periods, n;
        storage_mode=:full,
        n_waveforms=max(2, size(W_codes, 2))) : nothing

    reference_result, batch_result = if isnothing(bank)
        nothing, _empty_branch_batch()
    else
        ref_bank = _mft_filter_bank_for(periods, n; storage_mode=:full, n_waveforms=2)
        ref = mft.analyze_causal_acausal_branches(
            global_c[1:n], global_ac[1:n], dist, ref_bank;
            max_modes=mft_max_modes,
            compute_phase=true,
        )
        batch = mft.analyze_self_branches(
            W_codes[1:n, :], dist, bank;
            state_labels=String.(labels),
            max_modes=mft_max_modes,
            compute_phase=true,
        )
        ref, batch
    end

    (; batch=batch_result, reference=reference_result,
       pair_label=item.pair_label, seed=item.seed, mode)
end

# ╔═╡ c3000005-0000-0000-0000-000000000001
if isempty(codebook_mft_analysis.batch.state_results)
    md"No wavelength-valid codebook MFT periods for $(codebook_mft_analysis.pair_label)."
else
    WideCell(mft.plot_all_highcorr_groupvelocity_picks(
        codebook_mft_analysis.batch;
        correlation_threshold=0.0,
        pair_and_average=true,
        title="Codebook Group Velocity Picks ($(mft_title_context(codebook_mft_analysis.pair_label, codebook_mft_analysis.batch; seed=codebook_mft_analysis.seed, detail=codebook_mft_analysis.mode)))",
        velocity_tolerance_fraction=0.1,
        reference_results=isnothing(codebook_mft_analysis.reference) ? mft.BranchAnalysisResult[] : [codebook_mft_analysis.reference],
        reference_labels=["Global avg"],
        wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
        wavelength_fraction=Float64(ui_wavelength_fraction),
    ))
end

# ╔═╡ c3000003-0000-0000-0000-000000000001
let
    if isempty(codebook_mft_analysis.batch.state_results)
        md"No codebook filtered-trace source states are available."
    elseif !mft.mft_has_full_storage(first(codebook_mft_analysis.batch.state_results).causal_result)
        md"Codebook MFT runs in picks-only mode; filtered traces are not retained."
    else
        codebook_periods = [period for period in codebook_mft_analysis.batch.periods
            if mft.wavelength_valid_period(period, first(codebook_mft_analysis.batch.state_results).distance;
                wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
                wavelength_fraction=Float64(ui_wavelength_fraction))]
        if isempty(codebook_periods)
            md"No codebook filtered-trace periods remain after the wavelength filter."
        else
            @bind ui_period_codebook Slider(codebook_periods;
                default=mean(codebook_periods), show_value=true)
        end
    end
end

# ╔═╡ c3000004-0000-0000-0000-000000000001
let
    if isempty(codebook_mft_analysis.batch.state_results)
        md"No codebook filtered-trace source states are available."
    elseif !mft.mft_has_full_storage(first(codebook_mft_analysis.batch.state_results).causal_result)
        md"Codebook MFT runs in picks-only mode; filtered traces are not retained."
    else
        codebook_periods = [period for period in codebook_mft_analysis.batch.periods
            if mft.wavelength_valid_period(period, first(codebook_mft_analysis.batch.state_results).distance;
                wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
                wavelength_fraction=Float64(ui_wavelength_fraction))]
        if isempty(codebook_periods) || !@isdefined(ui_period_codebook)
            md"No codebook filtered-trace plot remains after the wavelength filter."
        else
            WideCell(mft.plot_filtered_traces_by_period(
                codebook_mft_analysis.batch;
                period=ui_period_codebook,
                correlation_threshold=nothing,
                normalize_each=true,
                scale=0.7,
                spacing=2.2,
                title="Codebook MFT Filtered Traces ($(mft_title_context(codebook_mft_analysis.pair_label, codebook_mft_analysis.batch; seed=codebook_mft_analysis.seed, detail=codebook_mft_analysis.mode)); period=$(round(ui_period_codebook; digits=2))s)",
            ))
        end
    end
end

# ╔═╡ c3000011-0000-0000-0000-000000000001
codebook_stationary_results = let
    mode = @isdefined(ui_codebook_stationary_family) ?
        String(ui_codebook_stationary_family) : "joint (K1×K2)"
    qmin = @isdefined(ui_codebook_stationary_quality_threshold) ?
        Float64(ui_codebook_stationary_quality_threshold) : 3.0
    rtol = @isdefined(ui_codebook_stationary_relative_tolerance) ?
        Float64(ui_codebook_stationary_relative_tolerance) : 0.10
    min_periods = @isdefined(ui_codebook_stationary_min_periods) ?
        Int(ui_codebook_stationary_min_periods) : 3

    specs = NamedTuple[]
    for pair_label in pair_labels
        items = [item for item in run_source_state_averages if item.pair_label == pair_label]
        isempty(items) && continue
        columns, labels, seeds, families = _codebook_columns_for_pair(items, mode)
        isempty(columns) && continue
        n = minimum(length.(columns))
        n > 0 || continue
        dist = Float64(first(items).distance)
        periods = _mft_compute_periods(dist)
        isempty(periods) && continue
        W = reduce(hcat, [col[1:n] for col in columns])
        push!(specs, (; pair_label=String(pair_label), dist, periods, W,
            labels, seeds, families, n))
    end

    bank_key(spec) = (Tuple(spec.periods), spec.n, size(spec.W, 2))
    banks = Dict{Any,Any}()
    for spec in specs
        get!(banks, bank_key(spec)) do
            _mft_filter_bank_for(spec.periods, spec.n;
                storage_mode=:picks_only,
                n_waveforms=size(spec.W, 2))
        end
    end

    results = NamedTuple[]
    @progress name="Codebook U-c stationary picks" for spec in specs
        bank = banks[bank_key(spec)]
        isnothing(bank) && continue
        mres = mft.perform_mft_analysis_batch!(bank, spec.W, spec.dist;
            compute_phase=true)
        summaries = [_stationary_summary_for_result(
                mres[j], spec.labels[j], spec.pair_label;
                quality_threshold=qmin,
                relative_tolerance=rtol)
            for j in eachindex(mres)]
        eligible = [s for s in summaries if s.n_periods >= min_periods]
        best = isempty(eligible) ? nothing :
            sort(eligible, by=s -> (s.median_relative_error, -s.pass_fraction, -s.n_periods))[1]
        push!(results, (; pair_label=spec.pair_label, distance=spec.dist,
            mode, labels=spec.labels, seeds=spec.seeds, families=spec.families,
            summaries, best, min_periods, quality_threshold=qmin,
            relative_tolerance=rtol))
    end
    results
end

# ╔═╡ c3000012-0000-0000-0000-000000000001
codebook_stationary_pick_rows = let
    rows = NamedTuple[]
    for result in codebook_stationary_results
        isnothing(result.best) && continue
        for row in result.best.rows
            push!(rows, merge(row, (;
                distance=result.distance,
                selected_label=result.best.label,
                median_relative_error=result.best.median_relative_error,
                pass_fraction=result.best.pass_fraction,
                n_stationary_periods=result.best.n_periods)))
        end
    end
    sort!(rows, by=r -> (r.pair_label, r.period))
    rows
end

# ╔═╡ c3000013-0000-0000-0000-000000000001
codebook_stationary_consensus_by_pair = let
    out = Dict{String,Any}()
    for result in codebook_stationary_results
        isnothing(result.best) && continue
        res = result.best.result
        row_by_period = Dict(Float64(row.period) => row for row in result.best.rows if row.pass)
        periods = Float64.(res.periods)
        group_velocities = fill(NaN, length(periods))
        arrival_times = fill(NaN, length(periods))
        confidence = zeros(Float64, length(periods))
        support = zeros(Int, length(periods))
        for ip in eachindex(periods)
            row = get(row_by_period, Float64(periods[ip]), nothing)
            isnothing(row) && continue
            group_velocities[ip] = row.group_velocity
            arrival_times[ip] = res.arrival_times[ip]
            confidence[ip] = clamp(1.0 - row.relative_error / result.relative_tolerance, 0.0, 1.0)
            support[ip] = 1
        end
        out[result.pair_label] = (; periods, group_velocities, arrival_times,
            confidence, support, source_label=result.best.label,
            median_relative_error=result.best.median_relative_error,
            pass_fraction=result.best.pass_fraction)
    end
    out
end

# ╔═╡ c3000014-0000-0000-0000-000000000001
let
    selected = [r for r in codebook_stationary_results if r.pair_label == selected_plot_pair]
    if isempty(selected)
        md"No all-pairs codebook stationary-zone result for $(selected_plot_pair)."
    else
        result = first(selected)
        if isempty(result.summaries)
            md"No codebook U-c summaries are available for **$(selected_plot_pair)**."
        else
            ranked = sort(result.summaries,
                by=s -> (s.n_periods >= result.min_periods ? 0 : 1,
                         s.median_relative_error, -s.pass_fraction, -s.n_periods, s.label))
            top = first(ranked, min(10, length(ranked)))
            rows = String[]
            push!(rows, "| rank | status | label | high-SNR periods | median rel U-c error | pass fraction |")
            push!(rows, "|---:|:---|:---|---:|---:|---:|")
            for (i, s) in enumerate(top)
                status = !isnothing(result.best) && s.label == result.best.label ? "best" :
                    (s.n_periods >= result.min_periods ? "eligible" : "below min")
                med_label = isfinite(s.median_relative_error) ?
                    string(round(s.median_relative_error; digits=3)) : "NA"
                pass_label = string(round(s.pass_fraction; digits=3))
                push!(rows, "| $(i) | $(status) | $(s.label) | $(s.n_periods) | $(med_label) | $(pass_label) |")
            end
            headline = isnothing(result.best) ?
                "No codebook atom for **$(selected_plot_pair)** has at least **$(result.min_periods)** high-SNR U-c periods." :
                "Codebook stationary-zone top candidates for **$(selected_plot_pair)**:"
            Markdown.parse("""
            $(headline)

            $(join(rows, "\n"))
            """)
        end
    end
end

# ╔═╡ c3000017-0000-0000-0000-000000000001
let
    selected = [r for r in codebook_stationary_results if r.pair_label == selected_plot_pair]
    if isempty(selected)
        md"No codebook U-c result is available for **$(selected_plot_pair)** yet. Run the all-pairs codebook MFT cell above, or choose a pair that has codebook results."
    else
        result = first(selected)
        if isempty(result.summaries)
            md"No codebook U-c summaries are available for **$(selected_plot_pair)**."
        else
            ranked = sort(result.summaries,
                by=s -> (s.n_periods >= result.min_periods ? 0 : 1,
                         s.median_relative_error, -s.pass_fraction, -s.n_periods))
            labels = [s.label for s in ranked]
            display_labels = String[]
            for s in ranked
                med_label = isfinite(s.median_relative_error) ? string(round(s.median_relative_error; digits=3)) : "NA"
                best_label = !isnothing(result.best) && s.label == result.best.label ? " [best]" : ""
                push!(display_labels, "$(s.label) | n=$(s.n_periods), med=$(med_label), pass=$(round(s.pass_fraction; digits=2))$(best_label)")
            end
            options = Pair{String,String}.(display_labels, labels)
            default = isnothing(result.best) ? first(labels) : result.best.label
            @bind selected_codebook_uc_label Select(options; default=default)
        end
    end
end

# ╔═╡ 9f5473e3-fa64-4030-9e78-aba7da828231
let
	selected = [r for r in codebook_stationary_results if r.pair_label == selected_plot_pair]
	if isempty(selected)
		md"No codebook U-c result is available for **$(selected_plot_pair)** yet."
	else
		result = first(selected)
		if isempty(result.summaries)
			md"No codebook U-c summaries are available for **$(selected_plot_pair)**."
		else
			WideCell(_plot_all_codebook_uc_agreements(result;
				title_prefix="All Seeds/Codes U-c Agreement"))
		end
	end
end

# ╔═╡ c3000015-0000-0000-0000-000000000001
let
    selected = [r for r in codebook_stationary_results if r.pair_label == selected_plot_pair]
    if isempty(selected)
        md"No codebook U-c result is available for **$(selected_plot_pair)** yet. Run the all-pairs codebook MFT cell above, or choose a pair that has codebook results."
    else
        result = first(selected)
        if isempty(result.summaries)
            md"No codebook U-c summaries are available for **$(selected_plot_pair)**."
        else
            # Try selected label first, fall back to best or first
            selected_label = @isdefined(selected_codebook_uc_label) ? String(selected_codebook_uc_label) : ""
            available_labels = [s.label for s in result.summaries]
            
            # Check if selected label exists in current summaries
            idx = if !isempty(selected_label) && selected_label in available_labels
                findfirst(s -> s.label == selected_label, result.summaries)
            else
                # Fall back to best or first
                fallback_label = isnothing(result.best) ? first(result.summaries).label : result.best.label
                findfirst(s -> s.label == fallback_label, result.summaries)
            end
            
            if isnothing(idx)
                md"""
                No U-c summary found.
                
                Pair: **$(selected_plot_pair)**
                
                Selected label: **$(selected_label)** (defined: $(@isdefined(selected_codebook_uc_label)))
                
                Available labels: $(available_labels)
                """
            else
                # Get global averages from source state data
                source_state_item = first([item for item in run_source_state_averages if item.pair_label == selected_plot_pair])
                global_avg_ac = isempty(source_state_item.global_avg_ac) ? nothing : Float64.(source_state_item.global_avg_ac)
                global_avg_c = isempty(source_state_item.global_avg_c) ? nothing : Float64.(source_state_item.global_avg_c)
                global_avg_raw_ac = isempty(source_state_item.global_avg_raw_ac) ? nothing : Float32.(source_state_item.global_avg_raw_ac)
                global_avg_raw_c = isempty(source_state_item.global_avg_raw_c) ? nothing : Float32.(source_state_item.global_avg_raw_c)
                periods_ref = _mft_compute_periods(result.distance)
                
                summary = result.summaries[idx]
                WideCell(_plot_codebook_uc_agreement(
                    selected_plot_pair, result.distance, summary;
                    relative_tolerance=Float64(result.relative_tolerance),
                    title_prefix=(!isnothing(result.best) && summary.label == result.best.label) ?
                        "Best Stationary-Zone U-c Check" : "Selected Codebook U-c Check",
                    global_avg_ac=global_avg_ac,
                    global_avg_c=global_avg_c,
                    global_avg_raw_ac=global_avg_raw_ac,
                    global_avg_raw_c=global_avg_raw_c,
                    periods_ref=periods_ref))
            end
        end
    end
end

# ╔═╡ 88841b07-7dbe-409f-8dc0-dc274373b91c
function _analyze_pair_branch_specs(specs; storage_mode_for_spec=spec -> :picks_only,
                                    compute_phase::Bool=false)
        isempty(specs) && return Dict{String,Any}()
        grouped = Dict{Tuple{Tuple{Vararg{Float64}},Symbol},Vector{Int}}()
        for (i, spec) in enumerate(specs)
            periods = Tuple(_mft_compute_periods(spec.distance))
            isempty(periods) && continue
            storage_mode = storage_mode_for_spec(spec)
            push!(get!(grouped, (periods, storage_mode), Int[]), i)
        end

        analyses = Dict{String,Any}()
        for ((period_key, storage_mode), inds) in grouped
            cols_c = [Float64.(vec(specs[i].causal)) for i in inds]
            cols_ac = [Float64.(vec(specs[i].acausal)) for i in inds]
            n = min(minimum(length.(cols_c)), minimum(length.(cols_ac)))
            n == 0 && continue
            bank = _mft_filter_bank_for(collect(period_key), n;
                storage_mode=storage_mode, n_waveforms=2 * length(inds))
            W_c = reduce(hcat, [col[1:n] for col in cols_c])
            W_ac = reduce(hcat, [col[1:n] for col in cols_ac])
            dists = [Float64(specs[i].distance) for i in inds]
            labels = [String(specs[i].label) for i in inds]
            pair_keys = [String(specs[i].pair_label) for i in inds]
            batch = mft.analyze_causal_acausal_branches(
                W_c, W_ac, dists, bank;
                state_labels=labels,
                max_modes=mft_max_modes,
                compute_phase=compute_phase,
            )
            merge!(analyses, _split_batch_by_pair(batch, pair_keys))
        end
        return analyses
    end

# ╔═╡ 61b6bba8-38c5-43a4-9d32-a95b1b7ccfd8
global_average_mft_analyses = let
    specs = NamedTuple[]
    for item in all_pairs_source_state_averages
        # global_avg_c/ac are seed-independent — use the artifact directly (one per pair)
        gc  = Float64.(vec(item.global_avg_c))
        gac = Float64.(vec(item.global_avg_ac))
        (isempty(gc) || isempty(gac)) && continue
        n = min(length(gc), length(gac))
        pair_label = String(item.pair_label)
        push!(specs, (; pair_label, label=pair_label,
                      causal=gc[1:n], acausal=gac[1:n],
                      distance=Float64(item.distance)))
    end
    pair_batches = _analyze_pair_branch_specs(specs)
    Dict(pair_label => first(batch.state_results)
         for (pair_label, batch) in pair_batches
         if !isempty(batch.state_results))
end;

# ╔═╡ a55f4597-feda-4c04-ae53-75a082be08ed
get(global_average_mft_analyses, selected_plot_pair,
    md"No wavelength-valid global-average MFT periods for $(selected_plot_pair).")

# ╔═╡ bb000012-0000-0000-0000-000000000001
# Extract all matched causal-acausal picks from the global average MFT for every pair.
# Returns a vector of (pair_label, distance, period, velocity) NamedTuples.
global_avg_all_picks = let
    vtol = 0.10
    wref = @isdefined(ui_wavelength_ref_velocity) ? Float64(ui_wavelength_ref_velocity) : 2.0
    wfrac = @isdefined(ui_wavelength_fraction) ? Float64(ui_wavelength_fraction) : 0.33
    rows = NamedTuple[]
    for (pair_label, res) in global_average_mft_analyses
        dist = res.distance
        hi = mft.high_correlation_indices(res; correlation_threshold=0.0)
        for ip in hi
            period = res.periods[ip]
            # Wavelength filter
            wref * period >= wfrac * dist && continue
            vavg = mft._matched_peak_average_velocities(
                res.causal_result.all_peaks[ip],
                res.acausal_result.all_peaks[ip],
                dist; velocity_tolerance_fraction=vtol)
            for v in vavg
                isfinite(v) && v > 0 || continue
                push!(rows, (; pair_label=String(pair_label), distance=dist, period, velocity=v))
            end
        end
    end
    sort(rows, by=r -> (r.period, r.pair_label))
end

# ╔═╡ bb000013-0000-0000-0000-000000000001
# For each (pair, period): keep at most one pick — the one closest to the cross-pair mean.
# Algorithm (two passes):
#   Pass 1 — provisional cross-pair mean from all raw picks per period.
#   Pass 2 — for each pair, keep the single pick closest to that provisional mean;
#             then recompute final mean/std from the one-per-pair picks and flag
#             outliers (|v - mean| > nsigma*std).
# Result fields: pair_label, period, velocity, distance, status ("kept"/"outlier")
global_avg_filtered_picks = let
    nsigma = @isdefined(ui_globalavg_outlier_nsigma) ? Float64(ui_globalavg_outlier_nsigma) : 2.0
    all_picks = global_avg_all_picks

    # Group all raw picks by period
    by_period = Dict{Float64, Vector{NamedTuple}}()
    for row in all_picks
        push!(get!(by_period, row.period, NamedTuple[]), row)
    end

    period_stats = Dict{Float64,NamedTuple}()
    filtered = NamedTuple[]

    for period in sort(collect(keys(by_period)))
        rows = get(by_period, period, NamedTuple[])
        isempty(rows) && continue

        # Pass 1: provisional mean from all raw picks
        μ_prov = mean(r.velocity for r in rows)

        # Pass 2: one pick per pair — closest to provisional mean
        by_pair = Dict{String, Vector{NamedTuple}}()
        for row in rows
            push!(get!(by_pair, row.pair_label, NamedTuple[]), row)
        end
        one_per_pair = NamedTuple[]
        for (_, pair_rows) in by_pair
            best = pair_rows[argmin(abs(r.velocity - μ_prov) for r in pair_rows)]
            push!(one_per_pair, best)
        end

        # Final mean/std from the reduced one-per-pair set
        vs = [r.velocity for r in one_per_pair]
        μ = mean(vs)
        σ = length(vs) > 1 ? std(vs) : 0.0
        period_stats[period] = (; mean=μ, std=σ, n_raw=length(rows), n_pairs=length(one_per_pair))

        # Outlier rejection on the one-per-pair picks
        for row in one_per_pair
            status = abs(row.velocity - μ) <= nsigma * σ ? "kept" : "outlier"
            push!(filtered, merge(row, (; status)))
        end
    end

    (; picks=filtered, period_stats=Dict(k => v for (k, v) in period_stats))
end;

# ╔═╡ bb000015-0000-0000-0000-000000000001
# Plot 2: After outlier filtering — kept picks (filled) vs rejected (hollow X),
# one kept pick per (pair × period).  Shows the algorithm's selection visually.
let
    if isempty(global_avg_filtered_picks.picks)
        md"No filtered picks."
    else
        picks = global_avg_filtered_picks.picks
        period_stats = global_avg_filtered_picks.period_stats
        pair_names = sort(unique([r.pair_label for r in picks]))
        colors_list = [
            "royalblue","tomato","forestgreen","darkorange","purple",
            "deepskyblue","crimson","limegreen","goldenrod","mediumorchid",
            "steelblue","salmon","mediumseagreen","peru","mediumpurple",
        ]
        traces = AbstractTrace[]
        # Kept picks — one per (pair, period)
        for (ci, pair_label) in enumerate(pair_names)
            kept = [r for r in picks if r.pair_label == pair_label && r.status == "kept"]
            isempty(kept) && continue
            color = colors_list[mod1(ci, length(colors_list))]
            push!(traces, PlutoPlotly.scatter(
                x=[r.period for r in kept],
                y=[r.velocity for r in kept],
                mode="markers",
                name=pair_label,
                marker=PlutoPlotly.attr(color=color, size=8, symbol="circle"),
                legendgroup=pair_label,
                showlegend=true,
            ))
            # Outlier picks for same pair — hollow X
            rejected = [r for r in picks if r.pair_label == pair_label && r.status == "outlier"]
            isempty(rejected) && continue
            push!(traces, PlutoPlotly.scatter(
                x=[r.period for r in rejected],
                y=[r.velocity for r in rejected],
                mode="markers",
                name="$(pair_label) (rejected)",
                marker=PlutoPlotly.attr(color=color, size=9, symbol="x-open", opacity=0.45,
                    line=PlutoPlotly.attr(width=2, color=color)),
                legendgroup=pair_label,
                showlegend=false,
            ))
        end
        # Mean curve
        ps = sort(collect(keys(period_stats)))
        μs = [period_stats[p].mean for p in ps]
        push!(traces, PlutoPlotly.scatter(
            x=ps, y=μs, mode="lines+markers",
            name="Per-period mean",
            line=PlutoPlotly.attr(color="black", width=2.5),
            marker=PlutoPlotly.attr(color="black", size=5, symbol="diamond"),
        ))
        n_kept = count(r.status == "kept" for r in picks)
        n_rej  = count(r.status == "outlier" for r in picks)
        nsig = @isdefined(ui_globalavg_outlier_nsigma) ? Float64(ui_globalavg_outlier_nsigma) : 2.0
        layout = PlutoPlotly.Layout(
            title="Filtered Global Avg Picks (kept=$(n_kept), rejected=$(n_rej), nsigma=$(nsig)) — filled=kept, ×=outlier",
            xaxis=PlutoPlotly.attr(title="Period (s)", type="log"),
            yaxis=PlutoPlotly.attr(title="Group velocity (km/s)"),
            width=1050, height=540,
            plot_bgcolor="white", paper_bgcolor="white",
            legend=PlutoPlotly.attr(orientation="v", x=1.01, xanchor="left"),
        )
        WideCell(PlutoPlotly.plot(traces, layout))
    end
end

# ╔═╡ bb000016-0000-0000-0000-000000000001
# Plot 3: Per-pair dispersion curves using only kept picks — one panel per pair
# overlaid on the same axes, so you can judge consistency across pairs.
let
    kept = [r for r in global_avg_filtered_picks.picks if r.status == "kept"]
    if isempty(kept)
        md"No kept picks to display."
    else
        pair_names = sort(unique([r.pair_label for r in kept]))
        colors_list = [
            "royalblue","tomato","forestgreen","darkorange","purple",
            "deepskyblue","crimson","limegreen","goldenrod","mediumorchid",
            "steelblue","salmon","mediumseagreen","peru","mediumpurple",
        ]
        traces = AbstractTrace[]
        for (ci, pair_label) in enumerate(pair_names)
            rows = sort([r for r in kept if r.pair_label == pair_label], by=r -> r.period)
            isempty(rows) && continue
            color = colors_list[mod1(ci, length(colors_list))]
            dist_km = round(Int, first(rows).distance)
            push!(traces, PlutoPlotly.scatter(
                x=[r.period for r in rows],
                y=[r.velocity for r in rows],
                mode="lines+markers",
                name="$(pair_label) ($(dist_km)km)",
                line=PlutoPlotly.attr(color=color, width=1.8),
                marker=PlutoPlotly.attr(color=color, size=6),
            ))
        end
        layout = PlutoPlotly.Layout(
            title="Global Avg Dispersion Curves — Kept Picks ($(length(pair_names)) pairs)",
            xaxis=PlutoPlotly.attr(title="Period (s)", type="log"),
            yaxis=PlutoPlotly.attr(title="Group velocity (km/s)"),
            width=1050, height=540,
            plot_bgcolor="white", paper_bgcolor="white",
            legend=PlutoPlotly.attr(orientation="v", x=1.01, xanchor="left"),
        )
        WideCell(PlutoPlotly.plot(traces, layout))
    end
end

# ╔═╡ bb000014-0000-0000-0000-000000000001
# Plot 1: All raw global-average picks for every pair, plus per-period mean curve.
let
    if isempty(global_avg_all_picks)
        md"No global average picks available."
    else
        pair_names = sort(unique([r.pair_label for r in global_avg_all_picks]))
        colors_list = [
            "royalblue","tomato","forestgreen","darkorange","purple",
            "deepskyblue","crimson","limegreen","goldenrod","mediumorchid",
            "steelblue","salmon","mediumseagreen","peru","mediumpurple",
        ]
        traces = AbstractTrace[]
        for (ci, pair_label) in enumerate(pair_names)
            rows = [r for r in global_avg_all_picks if r.pair_label == pair_label]
            isempty(rows) && continue
            color = colors_list[mod1(ci, length(colors_list))]
            push!(traces, PlutoPlotly.scatter(
                x=[r.period for r in rows],
                y=[r.velocity for r in rows],
                mode="markers",
                name=pair_label,
                marker=PlutoPlotly.attr(color=color, size=6, opacity=0.7),
            ))
        end
        # Per-period mean
        period_stats = global_avg_filtered_picks.period_stats
        ps = sort(collect(keys(period_stats)))
        μs = [period_stats[p].mean for p in ps]
        push!(traces, PlutoPlotly.scatter(
            x=ps, y=μs, mode="lines+markers",
            name="Per-period mean",
            line=PlutoPlotly.attr(color="black", width=2.5),
            marker=PlutoPlotly.attr(color="black", size=5, symbol="diamond"),
        ))
        nsig = @isdefined(ui_globalavg_outlier_nsigma) ? Float64(ui_globalavg_outlier_nsigma) : 2.0
        σs = [period_stats[p].std for p in ps]
        push!(traces, PlutoPlotly.scatter(
            x=vcat(ps, reverse(ps)),
            y=vcat(μs .+ nsig .* σs, reverse(μs .- nsig .* σs)),
            fill="toself", fillcolor="rgba(0,0,0,0.08)",
            line=PlutoPlotly.attr(color="transparent"),
            name="±$(nsig)σ band",
            showlegend=true,
        ))
        layout = PlutoPlotly.Layout(
            title="Global Average MFT Picks — All Pairs ($(length(pair_names)) pairs, $(length(global_avg_all_picks)) raw picks) | λ≥$(round(@isdefined(ui_wavelength_fraction) ? Float64(ui_wavelength_fraction) : 0.33; digits=2))×dist excluded from MFT",
            xaxis=PlutoPlotly.attr(title="Period (s)", type="log"),
            yaxis=PlutoPlotly.attr(title="Group velocity (km/s)"),
            width=1050, height=540,
            plot_bgcolor="white", paper_bgcolor="white",
            legend=PlutoPlotly.attr(orientation="v", x=1.01, xanchor="left"),
        )
        WideCell(PlutoPlotly.plot(traces, layout))
    end
end

# ╔═╡ fb5d92a0-5468-11f1-8329-3d47b6993594
# Extract one maximum-amplitude causal peak and one maximum-amplitude acausal
# peak per pair × period, then average the two velocity estimates directly.
single_peak_globalavg_raw_picks = let
    wref = @isdefined(ui_wavelength_ref_velocity) ? Float64(ui_wavelength_ref_velocity) : 2.0
    wfrac = @isdefined(ui_wavelength_fraction) ? Float64(ui_wavelength_fraction) : 0.33
    rows = NamedTuple[]
    for (pair_label, res) in global_average_mft_analyses
        dist = Float64(res.distance)
        for ip in eachindex(res.periods)
            period = Float64(res.periods[ip])
            wref * period >= wfrac * dist && continue

    ;        causal_peaks = res.causal_result.all_peaks[ip]
            acausal_peaks = res.acausal_result.all_peaks[ip]
            (isempty(causal_peaks) || isempty(acausal_peaks)) && continue

            t_causal, amp_causal = first(causal_peaks)
            t_acausal, amp_acausal = first(acausal_peaks)
            isfinite(t_causal) && t_causal > 0.0 || continue
            isfinite(t_acausal) && t_acausal > 0.0 || continue

            v_causal = dist / t_causal
            v_acausal = dist / t_acausal
            isfinite(v_causal) && v_causal > 0.0 || continue
            isfinite(v_acausal) && v_acausal > 0.0 || continue

            velocity = 0.5 * (v_causal + v_acausal)
            rel_mismatch = abs(v_causal - v_acausal) /
                max((abs(v_causal) + abs(v_acausal)) / 2.0, eps(Float64))
            push!(rows, (; pair_label=String(pair_label), distance=dist, period,
                          velocity, v_causal, v_acausal,
                          amp_causal=Float64(amp_causal),
                          amp_acausal=Float64(amp_acausal),
                          rel_mismatch))
        end
    end
    sort(rows, by=r -> (r.period, r.pair_label))
end;

# ╔═╡ fb5d9778-5468-11f1-8f5b-e9bad3a4c868
# Per-period σ rejection across receiver pairs. Initial mean/std define the
# rejection band; final mean/std are recomputed once from kept picks.
single_peak_globalavg_filtered_picks = let
    nsigma = @isdefined(ui_single_peak_globalavg_outlier_nsigma) ?
        Float64(ui_single_peak_globalavg_outlier_nsigma) : 0.8
    all_picks = single_peak_globalavg_raw_picks

    by_period = Dict{Float64, Vector{NamedTuple}}()
    for row in all_picks
        push!(get!(by_period, row.period, NamedTuple[]), row)
    end

    period_stats = Dict{Float64,NamedTuple}()
    filtered = NamedTuple[]

    for period in sort(collect(keys(by_period)))
        rows = get(by_period, period, NamedTuple[])
        isempty(rows) && continue

        vs = [r.velocity for r in rows]
        μ_initial = mean(vs)
        σ_initial = length(vs) > 1 ? std(vs) : 0.0

        statuses = [abs(row.velocity - μ_initial) <= nsigma * σ_initial ?
                    "kept" : "outlier" for row in rows]
        kept_vs = [row.velocity for (row, status) in zip(rows, statuses)
                   if status == "kept"]
        μ_final = isempty(kept_vs) ? NaN : mean(kept_vs)
        σ_final = length(kept_vs) > 1 ? std(kept_vs) : 0.0

        period_stats[period] = (; mean_initial=μ_initial,
                                std_initial=σ_initial,
                                mean_final=μ_final,
                                std_final=σ_final,
                                n_raw=length(rows),
                                n_kept=length(kept_vs),
                                nsigma)

        for (row, status) in zip(rows, statuses)
            push!(filtered, merge(row, (; status,
                                        mean_initial=μ_initial,
                                        std_initial=σ_initial,
                                        mean_final=μ_final,
                                        std_final=σ_final)))
        end
    end

    (; picks=filtered, period_stats=Dict(k => v for (k, v) in period_stats))
end;

# ╔═╡ fb5d9d18-5468-11f1-ab70-af6b201e12b6
# Single-peak raw/filtered plot — filled markers are kept, hollow x markers are
# cross-pair σ outliers, and the black curve is the final kept-pick mean.
let
    if isempty(single_peak_globalavg_filtered_picks.picks)
        md"No single-peak global average picks available."
    else
        picks = single_peak_globalavg_filtered_picks.picks
        period_stats = single_peak_globalavg_filtered_picks.period_stats
        pair_names = sort(unique([r.pair_label for r in picks]))
        colors_list = [
            "royalblue","tomato","forestgreen","darkorange","purple",
            "deepskyblue","crimson","limegreen","goldenrod","mediumorchid",
            "steelblue","salmon","mediumseagreen","peru","mediumpurple",
        ]
        traces = AbstractTrace[]
        for (ci, pair_label) in enumerate(pair_names)
            color = colors_list[mod1(ci, length(colors_list))]
            kept = [r for r in picks if r.pair_label == pair_label && r.status == "kept"]
            if !isempty(kept)
                push!(traces, PlutoPlotly.scatter(
                    x=[r.period for r in kept],
                    y=[r.velocity for r in kept],
                    mode="markers",
                    name=pair_label,
                    marker=PlutoPlotly.attr(color=color, size=8, symbol="circle"),
                    customdata=[[r.v_causal, r.v_acausal, r.amp_causal,
                                 r.amp_acausal, r.rel_mismatch] for r in kept],
                    hovertemplate="Pair: $(pair_label)<br>Period: %{x:.3f} s<br>Avg v: %{y:.3f} km/s<br>Causal v: %{customdata[0]:.3f}<br>Acausal v: %{customdata[1]:.3f}<br>C amp: %{customdata[2]:.3g}<br>AC amp: %{customdata[3]:.3g}<br>Rel mismatch: %{customdata[4]:.3f}<extra></extra>",
                    legendgroup=pair_label,
                    showlegend=true,
                ))
            end
            rejected = [r for r in picks if r.pair_label == pair_label && r.status == "outlier"]
            if !isempty(rejected)
                push!(traces, PlutoPlotly.scatter(
                    x=[r.period for r in rejected],
                    y=[r.velocity for r in rejected],
                    mode="markers",
                    name="$(pair_label) (outlier)",
                    marker=PlutoPlotly.attr(color=color, size=10, symbol="x-open",
                        opacity=0.55, line=PlutoPlotly.attr(width=2, color=color)),
                    customdata=[[r.v_causal, r.v_acausal, r.amp_causal,
                                 r.amp_acausal, r.rel_mismatch] for r in rejected],
                    hovertemplate="OUTLIER<br>Pair: $(pair_label)<br>Period: %{x:.3f} s<br>Avg v: %{y:.3f} km/s<br>Causal v: %{customdata[0]:.3f}<br>Acausal v: %{customdata[1]:.3f}<br>C amp: %{customdata[2]:.3g}<br>AC amp: %{customdata[3]:.3g}<br>Rel mismatch: %{customdata[4]:.3f}<extra></extra>",
                    legendgroup=pair_label,
                    showlegend=false,
                ))
            end
        end

        ps = sort(collect(keys(period_stats)))
        μ_initial = [period_stats[p].mean_initial for p in ps]
        σ_initial = [period_stats[p].std_initial for p in ps]
        μ_final = [period_stats[p].mean_final for p in ps]
        nsig = @isdefined(ui_single_peak_globalavg_outlier_nsigma) ?
            Float64(ui_single_peak_globalavg_outlier_nsigma) : 0.8
        push!(traces, PlutoPlotly.scatter(
            x=vcat(ps, reverse(ps)),
            y=vcat(μ_initial .+ nsig .* σ_initial,
                   reverse(μ_initial .- nsig .* σ_initial)),
            fill="toself", fillcolor="rgba(0,0,0,0.08)",
            line=PlutoPlotly.attr(color="transparent"),
            name="initial ±$(nsig)σ band",
            showlegend=true,
        ))
        push!(traces, PlutoPlotly.scatter(
            x=ps, y=μ_final, mode="lines+markers",
            name="Final kept-pick mean",
            line=PlutoPlotly.attr(color="black", width=2.7),
            marker=PlutoPlotly.attr(color="black", size=5, symbol="diamond"),
        ))

        n_kept = count(r.status == "kept" for r in picks)
        n_rej = count(r.status == "outlier" for r in picks)
        layout = PlutoPlotly.Layout(
            title="Single-Peak Global Avg Picks (kept=$(n_kept), outliers=$(n_rej), nsigma=$(nsig))",
            xaxis=PlutoPlotly.attr(title="Period (s)", type="log"),
            yaxis=PlutoPlotly.attr(title="Group velocity (km/s)"),
            width=1050, height=540,
            plot_bgcolor="white", paper_bgcolor="white",
            legend=PlutoPlotly.attr(orientation="v", x=1.01, xanchor="left"),
        )
        WideCell(PlutoPlotly.plot(traces, layout))
    end
end

# ╔═╡ fb5da8b2-5468-11f1-8e8a-05be83306bca
# Per-pair curves from kept single-peak picks only.
let
    kept = [r for r in single_peak_globalavg_filtered_picks.picks if r.status == "kept"]
    if isempty(kept)
        md"No kept single-peak picks to display."
    else
        pair_names = sort(unique([r.pair_label for r in kept]))
        colors_list = [
            "royalblue","tomato","forestgreen","darkorange","purple",
            "deepskyblue","crimson","limegreen","goldenrod","mediumorchid",
            "steelblue","salmon","mediumseagreen","peru","mediumpurple",
        ]
        traces = AbstractTrace[]
        for (ci, pair_label) in enumerate(pair_names)
            rows = sort([r for r in kept if r.pair_label == pair_label], by=r -> r.period)
            isempty(rows) && continue
            color = colors_list[mod1(ci, length(colors_list))]
            dist_km = round(Int, first(rows).distance)
            push!(traces, PlutoPlotly.scatter(
                x=[r.period for r in rows],
                y=[r.velocity for r in rows],
                mode="lines+markers",
                name="$(pair_label) ($(dist_km)km)",
                line=PlutoPlotly.attr(color=color, width=1.8),
                marker=PlutoPlotly.attr(color=color, size=6),
            ))
        end
        layout = PlutoPlotly.Layout(
            title="Single-Peak Global Avg Dispersion Curves — Kept Picks ($(length(pair_names)) pairs)",
            xaxis=PlutoPlotly.attr(title="Period (s)", type="log"),
            yaxis=PlutoPlotly.attr(title="Group velocity (km/s)"),
            width=1050, height=540,
            plot_bgcolor="white", paper_bgcolor="white",
            legend=PlutoPlotly.attr(orientation="v", x=1.01, xanchor="left"),
        )
        WideCell(PlutoPlotly.plot(traces, layout))
    end
end

# ╔═╡ 2a24e610-546c-11f1-9b53-57166bc3e159
# Lightweight invariants for the single-peak workflow and DSurfTomo export rows.
single_peak_globalavg_validation = let
    raw = single_peak_globalavg_raw_picks
    filtered = single_peak_globalavg_filtered_picks.picks
    stats = single_peak_globalavg_filtered_picks.period_stats
    kept = [r for r in filtered if r.status == "kept"]

    seen = Set{Tuple{String,Float64}}()
    duplicate_pair_period_rows = 0
    for r in raw
        key = (r.pair_label, r.period)
        if key in seen
            duplicate_pair_period_rows += 1
        else
            push!(seen, key)
        end
    end

    wref = @isdefined(ui_wavelength_ref_velocity) ? Float64(ui_wavelength_ref_velocity) : 2.0
    wfrac = @isdefined(ui_wavelength_fraction) ? Float64(ui_wavelength_fraction) : 0.33
    finite_positive_kept = all(isfinite(r.period) && r.period > 0.0 &&
                               isfinite(r.velocity) && r.velocity > 0.0 for r in kept)
    kept_pass_wavelength_filter = all(wref * r.period < wfrac * r.distance for r in kept)
    statuses_ok = all(r.status == "kept" || r.status == "outlier" for r in filtered)

    final_stats_recomputed_from_kept = all(collect(keys(stats))) do period
        kept_vs = [r.velocity for r in kept if r.period == period]
        st = stats[period]
        if isempty(kept_vs)
            isnan(st.mean_final)
        else
            μ = mean(kept_vs)
            σ = length(kept_vs) > 1 ? std(kept_vs) : 0.0
            isapprox(st.mean_final, μ; rtol=1e-10, atol=1e-12) &&
                isapprox(st.std_final, σ; rtol=1e-10, atol=1e-12)
        end
    end

    (; n_raw=length(raw),
       n_kept=length(kept),
       duplicate_pair_period_rows,
       finite_positive_kept,
       kept_pass_wavelength_filter,
       statuses_ok,
       final_stats_recomputed_from_kept)
end;

# ╔═╡ c61a6cfe-b14e-4aa9-a711-450e35a3a9bd
_pair_mft_analyses_joint = let
    specs = NamedTuple[]
    for item in run_source_state_averages
        nstates = size(item.acausal, 2)
        for i in 1:nstates
            label = i <= length(item.combo_labels) ? item.combo_labels[i] : string(i)
            push!(specs, (;
                pair_label=String(item.pair_label),
                label="$(item.pair_label) seed $(item.seed) | $(label)",
                causal=Float64.(vec(item.causal[:, i])),
                acausal=Float64.(vec(item.acausal[:, i])),
                distance=Float64(item.distance),
            ))
        end
    end
    _analyze_pair_branch_specs(specs;
        storage_mode_for_spec=spec ->
            spec.pair_label == selected_plot_pair ? :full : :picks_only)
end

# ╔═╡ a7c6d7af-cec7-4c4c-adb6-e2b370a49042
_pair_consensus_joint = Dict(pair_label => mft.consensus_group_velocity_picks(
    analysis;
    correlation_threshold=0.0,
    velocity_tolerance_fraction=0.10,
    cluster_tolerance_fraction=nothing,
    max_candidates=5,
    selection_mode=:low_velocity,
    min_candidate_periods=3,
    max_smooth_jump_fraction=0.08,
    max_gap_periods=1,
) for (pair_label, analysis) in _pair_mft_analyses_joint);

# ╔═╡ e4c00002-0000-0000-0000-000000000001
_pair_mft_analyses_marginal = let
    specs = NamedTuple[]
    for item in run_source_state_averages
        K1 = isempty(item.marginal_stage1_ac) ? 0 : size(item.marginal_stage1_ac, 2)
        K2 = isempty(item.marginal_stage2_ac) ? 0 : size(item.marginal_stage2_ac, 2)
        for k in 1:K1
            lbl = k <= length(item.marginal_stage1_labels) ? item.marginal_stage1_labels[k] : "s1=$k"
            push!(specs, (;
                pair_label=String(item.pair_label),
                label="$(item.pair_label) seed $(item.seed) | S1 $lbl",
                causal=Float64.(vec(item.marginal_stage1_c[:, k])),
                acausal=Float64.(vec(item.marginal_stage1_ac[:, k])),
                distance=Float64(item.distance),
            ))
        end
        for k in 1:K2
            lbl = k <= length(item.marginal_stage2_labels) ? item.marginal_stage2_labels[k] : "s2=$k"
            push!(specs, (;
                pair_label=String(item.pair_label),
                label="$(item.pair_label) seed $(item.seed) | S2 $lbl",
                causal=Float64.(vec(item.marginal_stage2_c[:, k])),
                acausal=Float64.(vec(item.marginal_stage2_ac[:, k])),
                distance=Float64(item.distance),
            ))
        end
    end
    _analyze_pair_branch_specs(specs;
        storage_mode_for_spec=spec ->
            spec.pair_label == selected_plot_pair ? :full : :picks_only)
end;

# ╔═╡ e4c00004-0000-0000-0000-000000000001
pair_mft_analyses = ui_mft_mode == "Marginal stages (K1+K2)" ? _pair_mft_analyses_marginal : _pair_mft_analyses_joint

# ╔═╡ 5064e2e2-1272-48ad-a417-02772071be86
if haskey(pair_mft_analyses, selected_plot_pair)
    WideCell(mft.plot_all_highcorr_groupvelocity_picks(
        pair_mft_analyses[selected_plot_pair];
        correlation_threshold=0.0,
        pair_and_average=true,
        title="Group Velocity Picks ($(mft_title_context(selected_plot_pair, pair_mft_analyses[selected_plot_pair]; detail=ui_mft_mode)))",
        velocity_tolerance_fraction=0.1,
        reference_results=haskey(global_average_mft_analyses, selected_plot_pair) ? [global_average_mft_analyses[selected_plot_pair]] : mft.BranchAnalysisResult[],
        reference_labels=["Global avg"],
        wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
        wavelength_fraction=Float64(ui_wavelength_fraction),
    ))
else
    md"No wavelength-valid source-state MFT periods for $(selected_plot_pair)."
end

# ╔═╡ c04cb032-ff13-4dfe-9338-70c5ce785db2
if haskey(pair_mft_analyses, selected_plot_pair)
    WideCell(mft.plot_branch_correlation(
        pair_mft_analyses[selected_plot_pair];
        title="MFT Branch Correlation ($(mft_title_context(selected_plot_pair, pair_mft_analyses[selected_plot_pair]; detail=ui_mft_mode)))",
        reference_results=haskey(global_average_mft_analyses, selected_plot_pair) ? [global_average_mft_analyses[selected_plot_pair]] : mft.BranchAnalysisResult[],
        reference_labels=["Global average"],
        wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
        wavelength_fraction=Float64(ui_wavelength_fraction)))
else
    md"No wavelength-valid branch-correlation periods for $(selected_plot_pair)."
end

# ╔═╡ 4ee99c0a-0c4e-4fd4-875e-dc42adef0d8e
mft_compute_wavelength_validation = let
    valid_result_periods(result, distance) = all(period -> period in mft_periods &&
        mft.wavelength_valid_period(period, distance;
            wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
            wavelength_fraction=Float64(ui_wavelength_fraction)),
        result.periods)

    global_results_valid = all(valid_result_periods(result, result.distance)
                               for result in values(global_average_mft_analyses))
    pair_results_valid = all(valid_result_periods(state, state.distance)
                             for batch in values(pair_mft_analyses)
                             for state in batch.state_results)
    cached_nfreqs = sort(unique([bank.nfreq for bank in values(mft_filter_banks)]))

    (; global_results_valid,
       pair_results_valid,
       full_period_count=length(mft_periods),
       cached_period_counts=cached_nfreqs,
       n_cached_banks=length(mft_filter_banks))
end;

# ╔═╡ b2000003-0000-0000-0000-000000000001
let
    if haskey(pair_mft_analyses, selected_plot_pair)
        trained_analysis = pair_mft_analyses[selected_plot_pair]
        if isempty(trained_analysis.state_results)
            md"No filtered-trace source states are available for $(selected_plot_pair)."
        else
            valid_trained_periods = [period for period in trained_analysis.periods
                if mft.wavelength_valid_period(period, first(trained_analysis.state_results).distance;
                    wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
                    wavelength_fraction=Float64(ui_wavelength_fraction))]
            if isempty(valid_trained_periods)
                md"No filtered-trace periods remain after the wavelength filter for $(selected_plot_pair)."
            else
                @bind ui_period_trained Slider(valid_trained_periods;
                    default=mean(valid_trained_periods), show_value=true)
            end
        end
    else
        md""
    end
end

# ╔═╡ b2000004-0000-0000-0000-000000000001
let
    if haskey(pair_mft_analyses, selected_plot_pair)
        trained_analysis = pair_mft_analyses[selected_plot_pair]
        if isempty(trained_analysis.state_results)
            md"No filtered-trace source states are available for $(selected_plot_pair)."
        else
            valid_trained_periods = [period for period in trained_analysis.periods
                if mft.wavelength_valid_period(period, first(trained_analysis.state_results).distance;
                    wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
                    wavelength_fraction=Float64(ui_wavelength_fraction))]
            if isempty(valid_trained_periods) || !@isdefined(ui_period_trained)
                md"No filtered-trace plot remains after the wavelength filter for $(selected_plot_pair)."
            else
                WideCell(mft.plot_filtered_traces_by_period(
                    trained_analysis;
                    period=ui_period_trained,
                    correlation_threshold=nothing,
                    normalize_each=true,
                    scale=0.7,
                    spacing=2.2,
                    title="MFT Filtered Source-State Traces ($(mft_title_context(selected_plot_pair, trained_analysis; detail=ui_mft_mode)); period=$(round(ui_period_trained; digits=2))s)",
                ))
            end
        end
    else
        md""
    end
end

# ╔═╡ e4c00003-0000-0000-0000-000000000001
_pair_consensus_marginal = Dict(pair_label => mft.consensus_group_velocity_picks(
    analysis;
    correlation_threshold=0.0,
    velocity_tolerance_fraction=0.10,
    cluster_tolerance_fraction=nothing,
    max_candidates=5,
    selection_mode=:low_velocity,
    min_candidate_periods=3,
    max_smooth_jump_fraction=0.08,
    max_gap_periods=1,
) for (pair_label, analysis) in _pair_mft_analyses_marginal);

# ╔═╡ e4c00005-0000-0000-0000-000000000001
pair_consensus = ui_mft_mode == "Marginal stages (K1+K2)" ? _pair_consensus_marginal : _pair_consensus_joint;

# ╔═╡ e3767110-37f7-4e37-a01f-93f72dcda465
if isempty(pair_labels) || !(selected_plot_pair in keys(pair_mft_analyses))
    md""
else
    WideCell(mft.plot_consensus_groupvelocity_picks(
        pair_mft_analyses[selected_plot_pair],
        pair_consensus[selected_plot_pair];
        correlation_threshold=0.0,
        velocity_tolerance_fraction=0.1,
        title="Trained VQ-VAE Consensus Picks ($(mft_title_context(selected_plot_pair, pair_mft_analyses[selected_plot_pair]; detail=ui_mft_mode)))",
        wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
        wavelength_fraction=Float64(ui_wavelength_fraction),
    ))
end

# ╔═╡ ec938992-7a8a-45e0-b38e-4ba40bc7dfdc
md"Computed MFT consensus candidates for **$(length(pair_consensus))** receiver pairs."

# ╔═╡ ff000015-0000-0000-0000-000000000001
begin
    function _latlong_from_csv(station_code::String)
        tbl = xj_latlong
        isempty(tbl) && return nothing
        col = tbl[!, "Station Code"]
        i = findfirst(==(station_code), col)
        i === nothing && return nothing
        return (lat=Float64(tbl[i, :Latitude]), lon=Float64(tbl[i, :Longitude]))
    end

    function _pair_geometry_for_export(pair_label::String)
        # Try artifact coords first
        pool = vcat(run_source_state_averages, all_pairs_source_state_averages)
        idx = findfirst(item -> item.pair_label == pair_label, pool)
        if !isnothing(idx)
            item = pool[idx]
            if !isnothing(item.latitudes) && !isnothing(item.longitudes) &&
                    length(item.latitudes) >= 2 && length(item.longitudes) >= 2
                return (;
                    pair_label,
                    lat1=Float64(item.latitudes[1]),
                    lon1=Float64(item.longitudes[1]),
                    lat2=Float64(item.latitudes[2]),
                    lon2=Float64(item.longitudes[2]),
                    distance=Float64(item.distance),
                )
            end
            # Artifact has no coords — fall back to CSV lookup by station code
            # pair_label format: "STA1-STA2"
            parts = split(pair_label, "-")
            if length(parts) == 2
                g1 = _latlong_from_csv(String(parts[1]))
                g2 = _latlong_from_csv(String(parts[2]))
                if !isnothing(g1) && !isnothing(g2)
                    dist = Float64(item.distance)
                    return (; pair_label, lat1=g1.lat, lon1=g1.lon,
                              lat2=g2.lat, lon2=g2.lon, distance=dist)
                end
            end
        end
        return nothing
    end

    function _consensus_for_export(pair_label::String)
        if haskey(codebook_stationary_consensus_by_pair, pair_label)
            return codebook_stationary_consensus_by_pair[pair_label], "stationary_codebook"
        elseif haskey(pair_consensus, pair_label)
            return pair_consensus[pair_label], "source_state"
        else
            return nothing, "missing"
        end
    end

    function dsurftomo_dispersion_rows(pair_labels;
            wavelength_ref_velocity=2.0, wavelength_fraction=0.33)
        rows = NamedTuple[]
        skipped = NamedTuple[]
        for pair_label in pair_labels
            geom = _pair_geometry_for_export(pair_label)
            if isnothing(geom)
                push!(skipped, (; pair_label, reason="missing_geometry"))
                continue
            end
            consensus, source = _consensus_for_export(pair_label)
            if isnothing(consensus)
                push!(skipped, (; pair_label, reason="missing_consensus"))
                continue
            end
            for i in eachindex(consensus.periods)
                period = Float64(consensus.periods[i])
                gv = Float64(consensus.group_velocities[i])
                isfinite(period) && period > 0 && isfinite(gv) && gv > 0 || continue
                # Wavelength filter: skip if λ = ref_velocity × period ≥ fraction × distance
                wavelength = wavelength_ref_velocity * period
                wavelength >= wavelength_fraction * geom.distance && continue
                confidence = i <= length(consensus.confidence) ? Float64(consensus.confidence[i]) : NaN
                support = i <= length(consensus.support) ? Int(consensus.support[i]) : 0
                push!(rows, (; geom.pair_label, period,
                             geom.lat1, geom.lon1, geom.lat2, geom.lon2,
                             group_velocity=gv, source, confidence, support))
            end
        end
        sort!(rows, by=row -> (row.period, row.pair_label))
        return rows, skipped
    end

    function write_dsurftomo_dispersion(path::AbstractString, rows; include_count_header::Bool=false)
        isempty(strip(path)) && error("Set dsurftomo_output_path before writing.")
        mkpath(dirname(path))
        open(path, "w") do io
            include_count_header && println(io, length(rows))
            for row in rows
                @printf(io, "%.8g %.8f %.8f %.8f %.8f %.8f\n",
                        row.period, row.lat1, row.lon1, row.lat2, row.lon2,
                        row.group_velocity)
            end
        end
        return path
    end
end

# ╔═╡ bb00001b-0000-0000-0000-000000000001
# Preview table: kept picks with geometry, ready for DSurfTomo
globalavg_dsurftomo_preview = let
    kept = [r for r in global_avg_filtered_picks.picks if r.status == "kept"]
    rows = NamedTuple[]
    for r in kept
        geom = _pair_geometry_for_export(r.pair_label)
        isnothing(geom) && continue
        push!(rows, (; r.pair_label, r.period, r.velocity,
                     geom.lat1, geom.lon1, geom.lat2, geom.lon2))
    end
    sort!(rows, by=row -> (row.period, row.pair_label))
    n_pairs = length(unique([row.pair_label for row in rows]))
    md"Global avg DSurfTomo preview: **$(length(rows))** measurements from **$(n_pairs)** pairs."
end

# ╔═╡ bb00001c-0000-0000-0000-000000000001
let
    clicks = try Int(write_globalavg_dsurftomo_button) catch; 0 end
    if clicks == 0
        md"Press **Write global avg DSurfTomo file** to write."
    else
        kept = [r for r in global_avg_filtered_picks.picks if r.status == "kept"]
        rows = NamedTuple[]
        for r in kept
            geom = _pair_geometry_for_export(r.pair_label)
            isnothing(geom) && continue
            push!(rows, (; r.pair_label, r.period, r.velocity,
                         geom.lat1, geom.lon1, geom.lat2, geom.lon2))
        end
        sort!(rows, by=row -> (row.period, row.pair_label))
        out_path = String(dsurftomo_globalavg_path)
        isempty(strip(out_path)) && error("Set dsurftomo_globalavg_path before writing.")
        mkpath(dirname(out_path))
        open(out_path, "w") do io
            Bool(dsurftomo_globalavg_include_header) && println(io, length(rows))
            for row in rows
                @printf(io, "%.8g %.8f %.8f %.8f %.8f %.8f\n",
                        row.period, row.lat1, row.lon1, row.lat2, row.lon2,
                        row.velocity)
            end
        end
        n_pairs = length(unique([row.pair_label for row in rows]))
        Markdown.parse("Wrote **$(length(rows))** global avg measurements from **$(n_pairs)** pairs to:\n\n`$(out_path)`")
    end
end

# ╔═╡ 0740383e-5469-11f1-9cd6-5926b074e7c9
# Preview table: kept single-peak picks with geometry, ready for DSurfTomo.
single_peak_globalavg_dsurftomo_preview = let
    kept = [r for r in single_peak_globalavg_filtered_picks.picks if r.status == "kept"]
    rows = NamedTuple[]
    for r in kept
        geom = _pair_geometry_for_export(r.pair_label)
        isnothing(geom) && continue
        push!(rows, (; r.pair_label, r.period, r.velocity,
                     geom.lat1, geom.lon1, geom.lat2, geom.lon2,
                     r.v_causal, r.v_acausal, r.amp_causal, r.amp_acausal,
                     r.mean_final, r.std_final))
    end
    sort!(rows, by=row -> (row.period, row.pair_label))
    n_pairs = length(unique([row.pair_label for row in rows]))
    md"Single-peak global avg DSurfTomo preview: **$(length(rows))** kept measurements from **$(n_pairs)** pairs."
end

# ╔═╡ 07403a78-5469-11f1-972c-477491544696
let
    clicks = try Int(write_single_peak_globalavg_dsurftomo_button) catch; 0 end
    if clicks == 0
        md"Press **Write single-peak global avg DSurfTomo file** to write."
    else
        kept = [r for r in single_peak_globalavg_filtered_picks.picks if r.status == "kept"]
        rows = NamedTuple[]
        for r in kept
            geom = _pair_geometry_for_export(r.pair_label)
            isnothing(geom) && continue
            push!(rows, (; r.pair_label, r.period, r.velocity,
                         geom.lat1, geom.lon1, geom.lat2, geom.lon2))
        end
        sort!(rows, by=row -> (row.period, row.pair_label))
        out_path = String(dsurftomo_single_peak_globalavg_path)
        isempty(strip(out_path)) && error("Set dsurftomo_single_peak_globalavg_path before writing.")
        mkpath(dirname(out_path))
        open(out_path, "w") do io
            Bool(dsurftomo_single_peak_globalavg_include_header) && println(io, length(rows))
            for row in rows
                @printf(io, "%.8g %.8f %.8f %.8f %.8f %.8f\n",
                        row.period, row.lat1, row.lon1, row.lat2, row.lon2,
                        row.velocity)
            end
        end
        n_pairs = length(unique([row.pair_label for row in rows]))
        Markdown.parse("Wrote **$(length(rows))** single-peak global avg measurements from **$(n_pairs)** pairs to:\n\n`$(out_path)`")
    end
end

# ╔═╡ ff000016-0000-0000-0000-000000000001
dsurftomo_export_preview = let
    rows, skipped = dsurftomo_dispersion_rows(pair_labels;
        wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
        wavelength_fraction=Float64(ui_wavelength_fraction))
    source_counts = Dict{String,Int}()
    for row in rows
        source_counts[row.source] = get(source_counts, row.source, 0) + 1
    end
    (; n_measurements=length(rows),
      n_pairs=length(unique([row.pair_label for row in rows])),
      source_counts,
      skipped)
end

# ╔═╡ ff000017-0000-0000-0000-000000000001
begin
    clicks = try
        Int(write_dsurftomo_button)
    catch
        0
    end

    if clicks == 0
        md"Press **Write DSurfTomo dispersion file** to write the current export preview."
    else
        rows, skipped = dsurftomo_dispersion_rows(pair_labels;
            wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
            wavelength_fraction=Float64(ui_wavelength_fraction))
        out_path = write_dsurftomo_dispersion(
            String(dsurftomo_output_path),
            rows;
            include_count_header=Bool(dsurftomo_include_count_header),
        )
        rows_by_source = Dict{String,Int}()
        for row in rows
            rows_by_source[row.source] = get(rows_by_source, row.source, 0) + 1
        end
        Markdown.parse("""
        Wrote **$(length(rows))** DSurfTomo dispersion measurements from **$(length(unique([row.pair_label for row in rows])))** receiver pairs to:

        `$(out_path)`

        Source rows: `$(rows_by_source)`.

        Skipped pairs: `$(length(skipped))`.
        """)
    end
end

# ╔═╡ ada28a7c-5597-11f1-8d40-b77369b13aaa
codebook_stationary_pick_rows

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Peaks = "18e31ff7-3703-566c-8e60-38913d67486b"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
CSV = "~0.10.16"
ColorSchemes = "~3.31.0"
Colors = "~0.13.1"
DSP = "~0.8.5"
DataFrames = "~1.8.2"
FFTW = "~1.10.0"
JLD2 = "~0.6.4"
Peaks = "~0.6.2"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.80"
ProgressLogging = "~0.1.6"
StatsBase = "~0.34.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "0ee272d1ef46dec11199e0c2fd66e2f3965359a9"

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
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

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

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "8d8e0b0f350b8e1c91420b5e64e5de774c2f0f4d"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.16"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "1a3ad7e16a321667698a19e77362b35a1e94c544"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.1"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "cee8104904c53d39eb94fd06cbe60cb5acde7177"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.0.0"

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

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

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

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "d335b2929e1b6067951a1250df247cc5fab7d40e"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.5"

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

    [deps.DSP.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

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
git-tree-sha1 = "8e9c059d6857607253e837730dbf780b6b151acd"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.19.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

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
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

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
git-tree-sha1 = "941f87a0ae1b14d1ac2fa57245425b23a9d7a516"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.4"
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
git-tree-sha1 = "fe23330af47b8ab4e135b2ff65f7398c3a2bfc65"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.5.2"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "58927c485919bf17ea308d9d82156de1adf4b006"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.12"

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
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

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
git-tree-sha1 = "5d4278f755440f70648d80cc6225f51e78e94094"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.5.1"

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
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "5d5e0a78e971354b1c7bff0655d11fdc1b0e12c8"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.4"

[[deps.Peaks]]
deps = ["SIMD"]
git-tree-sha1 = "a9b6680fb7fb097fb6eb1210c35549218d73da84"
uuid = "18e31ff7-3703-566c-8e60-38913d67486b"
version = "0.6.2"

    [deps.Peaks.extensions]
    MakieExt = "Makie"
    PlotsExt = "RecipesBase"

    [deps.Peaks.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

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
git-tree-sha1 = "8acd04abc9a636ef57004f4c2e6f3f6ed4611099"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.5"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "fbc875044d82c113a9dee6fc14e16cf01fd48872"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.80"

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

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "624de6279ab7d94fc9f672f0068107eb6619732c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.3.2"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

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
deps = ["CodeTracking", "FileWatching", "InteractiveUtils", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "5f4f629c085b87e71125eec6773f5f872c74a47a"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.14.2"

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

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

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ebe7e59b37c400f694f52b58c93d26201387da70"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.9"

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

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"

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

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "dd974aefe288ef2898733aecf40858dc86742d74"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.1"

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

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

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

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

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

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "0716e01c3b40413de5dedbc9c5c69f27cddfddfc"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.3"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

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
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─b00fd94f-291e-46d8-84ff-48f8606c2a1e
# ╠═dcbf026e-957a-4b9b-9757-bd0638a25b26
# ╠═aa000051-0000-0000-0000-000000000001
# ╟─f842f93e-16e8-4ec9-9c7f-c63d1f18c9f9
# ╠═e216a473-6433-4658-b2b7-a4eaa670cc5e
# ╠═9d75dc60-d654-4089-a297-abdcf0163493
# ╟─a4c73d31-cada-44dd-81c8-fbb0f5e84f6a
# ╟─e4c00001-0000-0000-0000-000000000001
# ╟─d7a8effd-1bc2-4f21-a947-7e8bbf82349a
# ╠═e3767110-37f7-4e37-a01f-93f72dcda465
# ╠═5064e2e2-1272-48ad-a417-02772071be86
# ╠═a55f4597-feda-4c04-ae53-75a082be08ed
# ╠═bb000010-0000-0000-0000-000000000001
# ╠═bb000011-0000-0000-0000-000000000001
# ╠═d6be1ffc-5468-11f1-a6b3-4db6669eeadd
# ╠═d6be2036-5468-11f1-a727-b909116351a8
# ╠═d6be20a4-5468-11f1-b705-a3f511139964
# ╠═d6be20cc-5468-11f1-b014-99325710d623
# ╠═d6be213a-5468-11f1-be47-d922a2dae738
# ╠═d6be216c-5468-11f1-b0af-6bf8fd2f274f
# ╠═bb000012-0000-0000-0000-000000000001
# ╠═bb000013-0000-0000-0000-000000000001
# ╠═fb5d92a0-5468-11f1-8329-3d47b6993594
# ╠═fb5d9778-5468-11f1-8f5b-e9bad3a4c868
# ╠═fb5d9d18-5468-11f1-ab70-af6b201e12b6
# ╠═fb5da8b2-5468-11f1-8e8a-05be83306bca
# ╠═2a24e610-546c-11f1-9b53-57166bc3e159
# ╠═bb000014-0000-0000-0000-000000000001
# ╠═bb000015-0000-0000-0000-000000000001
# ╠═bb000016-0000-0000-0000-000000000001
# ╠═bb000017-0000-0000-0000-000000000001
# ╠═bb000018-0000-0000-0000-000000000001
# ╠═bb000019-0000-0000-0000-000000000001
# ╠═bb00001a-0000-0000-0000-000000000001
# ╠═bb00001b-0000-0000-0000-000000000001
# ╠═bb00001c-0000-0000-0000-000000000001
# ╠═0740383e-5469-11f1-9cd6-5926b074e7c9
# ╠═07403a78-5469-11f1-972c-477491544696
# ╟─c04cb032-ff13-4dfe-9338-70c5ce785db2
# ╟─f30a7fe9-67f2-438e-9b9c-57e41b583c0e
# ╠═c3000006-0000-0000-0000-000000000001
# ╠═c3000005-0000-0000-0000-000000000001
# ╠═d4c01002-0000-0000-0000-000000000001
# ╠═d4c01003-0000-0000-0000-000000000001
# ╠═d4c00001-0000-0000-0000-000000000001
# ╠═d4c01004-0000-0000-0000-000000000001
# ╠═d4c01005-0000-0000-0000-000000000001
# ╠═d4c00002-0000-0000-0000-000000000001
# ╟─c753f15f-12ab-4f54-92f2-59d96197f85c
# ╟─b584dc56-8127-402c-98d0-1da3135f8f9a
# ╠═a0f8a2b4-8fb5-4f06-bda6-c362a61065a1
# ╠═2f66e040-8d6d-4376-b97d-0169cbdc1efe
# ╠═75e66ac1-57bf-468d-ab55-ada1d5e9ef91
# ╠═aa000050-0000-0000-0000-000000000001
# ╠═c7d47e38-e24f-4b40-b3a4-bc894188a750
# ╟─d50d63be-d58b-4704-8211-ed7875e04857
# ╠═c7ec82e0-d5b6-4f31-8600-c8b1d276dc92
# ╠═e63099d0-5fb5-43b0-967c-b7c468dc4f83
# ╠═ed704922-12f0-475b-b628-19a9c37bca7a
# ╠═aa000040-0000-0000-0000-000000000001
# ╠═aa000041-0000-0000-0000-000000000001
# ╠═aa000042-0000-0000-0000-000000000001
# ╠═b5a1cf7d-d464-4409-b43a-074c8aa22108
# ╟─e4499887-1a64-4eaa-a599-4ed4941a7b2d
# ╠═e4f7b3cf-7f26-4ea9-b9bb-95f3df9e7790
# ╠═02d134b5-7ce3-47a9-86ef-a43e6c52287a
# ╠═ddfd42a8-4ae7-408a-8b8f-2d335746798b
# ╟─b7c7a358-71c8-4797-a259-68bc75ab6e65
# ╟─eec83733-193d-4f52-9a75-e6f1d03c7aa5
# ╠═2c887c9a-4a41-4d07-a295-f5c14cfdd110
# ╠═61b6bba8-38c5-43a4-9d32-a95b1b7ccfd8
# ╠═067d2587-8eb1-41c3-95f8-9f785171f2ce
# ╠═fb000002-0000-0000-0000-000000000001
# ╠═b01af348-c4b0-4ad1-81cd-116e9f2ed765
# ╠═c61a6cfe-b14e-4aa9-a711-450e35a3a9bd
# ╠═e4c00002-0000-0000-0000-000000000001
# ╠═e4c00004-0000-0000-0000-000000000001
# ╠═4ee99c0a-0c4e-4fd4-875e-dc42adef0d8e
# ╠═a7c6d7af-cec7-4c4c-adb6-e2b370a49042
# ╠═e4c00003-0000-0000-0000-000000000001
# ╠═e4c00005-0000-0000-0000-000000000001
# ╟─ec938992-7a8a-45e0-b38e-4ba40bc7dfdc
# ╟─b350e7a5-ae7e-46ca-a246-d60b66a68e17
# ╟─e89d81cb-8596-4364-8241-01578fb81c6b
# ╠═50bcfca1-b35c-4af5-8da7-26e6a9aa7914
# ╠═dccacb14-c2de-43a2-943c-0e52a5e1276f
# ╠═b2000001-0000-0000-0000-000000000001
# ╠═b2000002-0000-0000-0000-000000000001
# ╟─b2000003-0000-0000-0000-000000000001
# ╠═b2000004-0000-0000-0000-000000000001
# ╠═d1d675f4-ebea-4432-8d0e-ddeada2f5fa3
# ╟─c3000001-0000-0000-0000-000000000001
# ╟─c3000007-0000-0000-0000-000000000001
# ╠═7f3b6cae-a0f6-463e-8375-05a86d200d3a
# ╠═18793657-2479-4260-927d-8a39088fb814
# ╠═657949fd-620a-43e2-916f-f9172573c50f
# ╠═2c18a0ad-0963-4e63-8c20-548674e42c09
# ╠═c3000002-0000-0000-0000-000000000001
# ╟─c3000003-0000-0000-0000-000000000001
# ╟─c3000004-0000-0000-0000-000000000001
# ╠═c3000010-0000-0000-0000-000000000001
# ╠═c3000011-0000-0000-0000-000000000001
# ╠═c3000012-0000-0000-0000-000000000001
# ╠═c3000013-0000-0000-0000-000000000001
# ╠═c3000016-0000-0000-0000-000000000001
# ╠═c3000014-0000-0000-0000-000000000001
# ╠═c3000017-0000-0000-0000-000000000001
# ╠═9f5473e3-fa64-4030-9e78-aba7da828231
# ╠═c3000015-0000-0000-0000-000000000001
# ╠═c3000008-0000-0000-0000-000000000001
# ╠═c3000009-0000-0000-0000-000000000001
# ╠═c300000a-0000-0000-0000-000000000001
# ╠═c300000f-0000-0000-0000-000000000001
# ╠═c300000b-0000-0000-0000-000000000001
# ╟─ff000010-0000-0000-0000-000000000001
# ╠═ff000011-0000-0000-0000-000000000001
# ╠═ff000012-0000-0000-0000-000000000001
# ╠═ff000013-0000-0000-0000-000000000001
# ╠═ff000015-0000-0000-0000-000000000001
# ╠═aa000030-0000-0000-0000-000000000001
# ╠═aa000031-0000-0000-0000-000000000001
# ╠═ff000016-0000-0000-0000-000000000001
# ╠═ff000017-0000-0000-0000-000000000001
# ╠═b4c9a332-7e9f-4933-b12f-7c13f6d5f112
# ╠═d8d742f8-b6d4-44e0-bf11-29ff4d22e117
# ╠═eb76dfcf-b8ce-445d-a152-52eb8a6f94a7
# ╠═f01dc5e7-ae8a-4c31-a8df-64cabd2abe35
# ╠═cc15f5d5-764d-4ee8-a8f2-862c5207c630
# ╠═ed7943c5-d57e-4592-8fa6-e402edffd627
# ╠═04212dbb-a028-4f26-9a96-f90bc36c00b2
# ╠═0e351ef2-554f-4697-802f-74b785273c93
# ╠═683da2a1-d2b9-4f21-880c-f93e945805fb
# ╠═e9904c62-eb2f-4fd5-9dd9-4cfdf071b6e3
# ╠═c98b88da-1ce4-4d8f-845d-7434e6c5f4ab
# ╠═88841b07-7dbe-409f-8dc0-dc274373b91c
# ╠═fb000001-0000-0000-0000-000000000001
# ╠═ada28a7c-5597-11f1-8d40-b77369b13aaa
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
