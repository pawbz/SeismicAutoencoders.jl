### A Pluto.jl notebook ###
# v0.20.27

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

# ╔═╡ 1f2f5740-5bd0-11f1-0010-4b3f11e7a001
begin
    using Base.Threads
    using JLD2
    using LinearAlgebra
    using PlutoLinks
    using PlutoPlotly
    using PlutoUI
    using ProgressLogging
    using Statistics
end

# ╔═╡ 721a47ce-86ca-493f-ba3c-b18047cc8395
using FFTW, StatsBase, Peaks, CSV, DataFrames

# ╔═╡ 0bfa7475-8c46-4cac-87a7-6741a3a12aae
using ColorSchemes

# ╔═╡ 83975653-0038-4e91-b282-463dd5eb70c7
using Colors

# ╔═╡ a7b1a698-46d0-4a34-a2e8-f02cbafe26ff
using DSP

# ╔═╡ 1f2f5740-5bd0-11f1-0001-4b3f11e7a001
md"""
# Trained VQ-VAE MFT v11

Slim global-average U-c consistency workbench. This notebook only loads saved source-state artifacts, computes global-average/source-state MFT rows, and shows the selected source-state overlay plot.
"""

# ╔═╡ 1f2f5740-5bd0-11f1-0002-4b3f11e7a001
begin
    _saved_root_default = "/mnt/NAS2/Sanket_data/California_XJ_13032026/SavedModels/vqvae_v10_K=[5, 3]"
    @bind _data_source_controls PlutoUI.combine() do Child
        md"""
        | Data source | Value |
        |:---|:---|
        | Saved root | $(Child("saved_root", TextField(default=_saved_root_default))) |
        | Reload artifacts | $(Child("reload", CounterButton("Reload"))) |
        """
    end
end

# ╔═╡ 1f2f5740-5bd0-11f1-000c-4b3f11e7a001
begin
    @bind _uc_score_controls PlutoUI.combine() do Child
        md"""
        | U-c consistency score | Value |
        |:---|:---|
        | U-c score | $(Child("method", Select(["geomean" => "geomean (default)", "median" => "median", "mean" => "mean", "p95" => "p95", "max" => "max", "huber" => "Huber mean"]; default="geomean"))) |
        | Huber delta | $(Child("huber_delta", NumberField(0.01:0.01:1.0; default=0.10))) |
        """
    end
end

# ╔═╡ 1f2f5740-5bd0-11f1-000d-4b3f11e7a001
begin
    ui_uc_score_method = String(_uc_score_controls.method)
    ui_uc_huber_delta = Float64(_uc_score_controls.huber_delta)
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0003-4b3f11e7a001
saved_root = String(_data_source_controls.saved_root)

# ╔═╡ b074735f-2faf-4cc3-9497-f33b8fe211c6
reload_saved_artifacts_button = _data_source_controls.reload

# ╔═╡ 1f2f5740-5bd0-11f1-0009-4b3f11e7a001
md"## Appendix"

# ╔═╡ 1f2f5740-5bd0-11f1-0011-4b3f11e7a001
mft = (@ingredients("/mnt/NAS/EQData/FTAN.jl/src/FTAN.jl")).FTAN

# ╔═╡ 1f2f5740-5bd0-11f1-0012-4b3f11e7a001
function _parse_seed_timestamp(run_dir::String)
    name = basename(run_dir)
    m = match(r"^seed([0-9]+)_(.+)$", name)
    m === nothing && return (; seed=missing, timestamp=name)
    (; seed=parse(Int, m.captures[1]), timestamp=m.captures[2])
end

# ╔═╡ 1f2f5740-5bd0-11f1-0013-4b3f11e7a001
function _read_saved_analysis_settings(run_dir::String)
    path = joinpath(run_dir, "source_state_averages.jld2")
    d = load(path)
    haskey(d, "analysis_settings") || error("Saved artifact is missing analysis_settings: $(path).")
    d["analysis_settings"]
end

# ╔═╡ 1f2f5740-5bd0-11f1-0014-4b3f11e7a001
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
            push!(runs, (; pair=(String(parts[1]), String(parts[2])),
                pair_label=replace(pair_label, "_" => "-"),
                run_dir, seed=parsed.seed, timestamp=parsed.timestamp,
                analysis_settings=_read_saved_analysis_settings(run_dir)))
        end
    end
    runs
end

# ╔═╡ 1f2f5740-5bd0-11f1-0015-4b3f11e7a001
function _source_state_artifact_path(run)
    joinpath(run.run_dir, "source_state_averages.jld2")
end

# ╔═╡ 1f2f5740-5bd0-11f1-0016-4b3f11e7a001
function _load_saved_source_state_averages(run)
    path = _source_state_artifact_path(run)
    isfile(path) || error("Missing source-state artifact: $(path)")
    d = load(path)
    (;
        acausal=Float32.(d["acausal"]),
        causal=Float32.(d["causal"]),
        combo_labels=String.(d["combo_labels"]),
        global_avg_ac=Float32.(d["global_avg_ac"]),
        global_avg_c=Float32.(d["global_avg_c"]),
        marginal_stage1_ac=haskey(d, "marginal_stage1_ac") ? Float32.(d["marginal_stage1_ac"]) : Float32[;;],
        marginal_stage1_c=haskey(d, "marginal_stage1_c") ? Float32.(d["marginal_stage1_c"]) : Float32[;;],
        marginal_stage2_ac=haskey(d, "marginal_stage2_ac") ? Float32.(d["marginal_stage2_ac"]) : Float32[;;],
        marginal_stage2_c=haskey(d, "marginal_stage2_c") ? Float32.(d["marginal_stage2_c"]) : Float32[;;],
        marginal_stage1_labels=haskey(d, "marginal_stage1_labels") ? String.(d["marginal_stage1_labels"]) : String[],
        marginal_stage2_labels=haskey(d, "marginal_stage2_labels") ? String.(d["marginal_stage2_labels"]) : String[],
        analysis_settings=d["analysis_settings"],
        distance=Float64(d["distance"]),
        pair=run.pair,
        pair_label=run.pair_label,
        run_dir=run.run_dir,
        seed=run.seed,
    )
end

# ╔═╡ 1f2f5740-5bd0-11f1-0017-4b3f11e7a001
_setting(settings, name::Symbol, default=nothing) =
    settings !== nothing && hasproperty(settings, name) ? getproperty(settings, name) :
    default === nothing ? error("Saved analysis_settings is missing $(name).") : default

# ╔═╡ 1f2f5740-5bd0-11f1-0018-4b3f11e7a001
begin
    reload_saved_artifacts_button
    all_saved_runs = discover_vqvae_runs(saved_root)
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0019-4b3f11e7a001
begin
    pair_options = sort(unique([run.pair_label for run in all_saved_runs]))
end;

# ╔═╡ f3feca4e-36f9-4d82-a0c3-39234d240872
begin
        _pair_options_ui = isempty(pair_options) ? ["None"] : pair_options
    _selected_pairs_default_ui = isempty(pair_options) ? String[] : pair_options
    @bind _pair_mode_controls PlutoUI.combine() do Child
        md"""
        | Computation selection | Value |
        |:---|:---|
        | Selected pairs | $(Child("pairs", MultiCheckBox(_pair_options_ui; default=_selected_pairs_default_ui))) |
        """
    end
end

# ╔═╡ 4947c5cf-e1db-402e-8816-4e1cb0426802
begin
    @bind _pair_mode_controls_plot PlutoUI.combine() do Child
        md"""
        | Computation selection | Value |
        |:---|:---|
        | Plot pair | $(Child("plot_pair", Select(_pair_options_ui; default=first(_pair_options_ui)))) |
        | MFT mode | $(Child("mode", Select(["Joint states (K1×K2)", "Marginal stages (K1+K2)"]; default="Joint states (K1×K2)"))) |
        """
    end
end

# ╔═╡ 1f2f5740-5bd0-11f1-000b-4b3f11e7a001
begin
    selected_pair_names = String.(collect(_pair_mode_controls.pairs))
    selected_plot_pair = String(_pair_mode_controls_plot.plot_pair)
    ui_mft_mode = String(_pair_mode_controls_plot.mode)
end;

# ╔═╡ 1f2f5740-5bd0-11f1-001a-4b3f11e7a001
_selected_pair_set = isempty(selected_pair_names) || selected_pair_names == ["None"] ?
    Set(pair_options) : Set(selected_pair_names);

# ╔═╡ 1f2f5740-5bd0-11f1-0020-4b3f11e7a001
analysis_settings = isempty(all_saved_runs) ? nothing : first(all_saved_runs).analysis_settings;

# ╔═╡ 1f2f5740-5bd0-11f1-0021-4b3f11e7a001
begin
    dt = Float64(_setting(analysis_settings, :dt, 1.0))
    period_min = Float64(_setting(analysis_settings, :period_min, 3.0))
    period_max = Float64(_setting(analysis_settings, :period_max, 10.0))
    mft_nperiods = Int(_setting(analysis_settings, :mft_nperiods, 100))
    mft_periods = collect(exp.(range(log(period_min), log(period_max); length=mft_nperiods)))
    mft_max_modes_default = Int(_setting(analysis_settings, :mft_max_modes, 6))
    velocity_range_saved = _setting(analysis_settings, :velocity_range, (1.0, 8.0))
    velocity_range_default = (Float64(velocity_range_saved[1]), Float64(velocity_range_saved[2]))
    bandwidth_factor_default = Float64(_setting(analysis_settings, :bandwidth_factor, 1.0))
    zero_pad_factor_default = Int(_setting(analysis_settings, :zero_pad_factor, 8))
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0004-4b3f11e7a001
begin
    _vel_default = velocity_range_default
    @bind _mft_plot_controls PlutoUI.combine() do Child
        md"""
        | MFT / wavelength / plot settings | Value |
        |:---|:---|
        | Wavelength reference velocity (km/s) | $(Child("wref", NumberField(0.5:0.5:5.0; default=2.0))) |
        | Wavelength fraction of distance | $(Child("wfrac", NumberField(0.05:0.01:1.0; default=0.33))) |
        | Velocity min (km/s) | $(Child("vmin", NumberField(0.1:0.1:20.0; default=_vel_default[1]))) |
        | Velocity max (km/s) | $(Child("vmax", NumberField(0.1:0.1:20.0; default=_vel_default[2]))) |
        | Phase velocity min (km/s) | $(Child("phase_vmin", NumberField(1.0:0.1:8.0; default=2.0))) |
        | Phase velocity max (km/s) | $(Child("phase_vmax", NumberField(1.0:0.1:8.0; default=5.0))) |
        | Phase method | $(Child("phase_method", Select(["branch resolver", "phtovel", "compare both"]; default="phtovel"))) |
        | Max modes | $(Child("max_modes", NumberField(1:1:20; default=mft_max_modes_default))) |
        | Bandwidth factor | $(Child("bandwidth", NumberField(0.1:0.05:5.0; default=bandwidth_factor_default))) |
        | Zero-pad factor | $(Child("zero_pad", NumberField(1:1:32; default=zero_pad_factor_default))) |
        | Upsample factor | $(Child("upsample", NumberField(1:1:10; default=2))) |
        | Numeric precision | $(Child("precision", Select(["Float32", "Float64"]; default="Float32"))) |
        """
    end
end

# ╔═╡ 1f2f5740-5bd0-11f1-0005-4b3f11e7a001
begin
    ui_wavelength_ref_velocity = Float64(_mft_plot_controls.wref)
    ui_wavelength_fraction = Float64(_mft_plot_controls.wfrac)
    velocity_range = (Float64(_mft_plot_controls.vmin), Float64(_mft_plot_controls.vmax))
    phase_velocity_range = sort((Float64(_mft_plot_controls.phase_vmin), Float64(_mft_plot_controls.phase_vmax)))
    ui_phase_velocity_method = String(_mft_plot_controls.phase_method)
    ui_mft_use_phtovel = ui_phase_velocity_method == "phtovel"
    mft_max_modes = Int(_mft_plot_controls.max_modes)
    bandwidth_factor = Float64(_mft_plot_controls.bandwidth)
    zero_pad_factor = Int(_mft_plot_controls.zero_pad)
    ui_mft_upsample_factor = Float64(_mft_plot_controls.upsample)
    ui_mft_precision = String(_mft_plot_controls.precision)
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0022-4b3f11e7a001
selected_runs = begin
    raw = [run for run in all_saved_runs if run.pair_label in _selected_pair_set]
    keep_only_latest_rerun_per_seed = isempty(raw) ? false :
        Bool(_setting(first(raw).analysis_settings, :use_latest_run_per_seed, false))
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

# ╔═╡ 1f2f5740-5bd0-11f1-0023-4b3f11e7a001
run_source_state_averages = let
    reload_saved_artifacts_button
    out = Vector{Any}(undef, length(selected_runs))
    @progress name="Loading selected source-state artifacts" for i in eachindex(selected_runs)
        out[i] = _load_saved_source_state_averages(selected_runs[i])
    end
    out
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0024-4b3f11e7a001
pair_labels = sort(unique([item.pair_label for item in run_source_state_averages]));

# ╔═╡ 1f2f5740-5bd0-11f1-0025-4b3f11e7a001
source_state_plot_runs = begin
    items = [item for item in run_source_state_averages if item.pair_label == selected_plot_pair]
    labels = String[]
    for item in items
        push!(labels, "$(item.pair_label) seed $(item.seed) | $(basename(item.run_dir))")
    end
    (; items, labels)
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0027-4b3f11e7a001
selected_globalavg_runs = let
    keep = Dict{String,Any}()
    for run in all_saved_runs
        run.pair_label in _selected_pair_set || continue
        if !haskey(keep, run.pair_label) || string(run.timestamp) > string(keep[run.pair_label].timestamp)
            keep[run.pair_label] = run
        end
    end
    sort(collect(values(keep)), by=r -> r.pair_label)
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0028-4b3f11e7a001
globalavg_source_state_averages = let
    reload_saved_artifacts_button
    out = Vector{Any}(undef, length(selected_globalavg_runs))
    @progress name="Loading global-average artifacts" for i in eachindex(selected_globalavg_runs)
        out[i] = _load_saved_source_state_averages(selected_globalavg_runs[i])
    end
    out
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0029-4b3f11e7a001
begin
    mft_filter_banks = Dict{Any,Any}()
    _mft_precision_type() = ui_mft_precision == "Float64" ? Float64 : Float32
end

# ╔═╡ 1f2f5740-5bd0-11f1-0030-4b3f11e7a001
function _mft_compute_periods(distance::Real)
    Float64[period for period in mft_periods
        if mft.wavelength_valid_period(period, distance;
            wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
            wavelength_fraction=Float64(ui_wavelength_fraction))]
end

# ╔═╡ 1f2f5740-5bd0-11f1-0031-4b3f11e7a001
function _mft_filter_bank_for(periods::Vector{Float64}, npts_raw::Int;
        storage_mode::Symbol=:picks_only,
        n_waveforms::Int=1)
    isempty(periods) && return nothing
    key = (mft.MFTFilterBank, npts_raw, Tuple(periods), bandwidth_factor,
        zero_pad_factor, Float64(ui_mft_upsample_factor), velocity_range,
        _mft_precision_type(), storage_mode, n_waveforms)
    get!(mft_filter_banks, key) do
        mft.MFTFilterBank(dt, npts_raw, periods;
            bandwidth_factor=bandwidth_factor,
            zero_pad_factor=zero_pad_factor,
            upsample_factor=Float64(ui_mft_upsample_factor),
            velocity_range=velocity_range,
            precision=_mft_precision_type(),
            storage_mode=storage_mode,
            N_initial=n_waveforms)
    end
end

function _mft_shared_periods_for_specs(specs, inds)
    Float64.(mft_periods)
end

function _mask_multimodal_dispersion(mode, valid::AbstractVector{Bool})
    inds = findall(valid[1:length(mode.periods)])
    mft.MultimodalDispersion(
        Float64.(mode.periods[inds]),
        Float64.(mode.arrival_times[inds]),
        Float64.(mode.group_velocities[inds]),
        Float64.(mode.peak_amplitudes[inds]),
        mode.mode_index)
end

function _mask_mft_result(res, distance::Real)
    valid_periods = Set(_mft_compute_periods(distance))
    valid = [period in valid_periods for period in res.periods]
    invalid = .!valid

    group_velocities = copy(res.group_velocities)
    phase_velocities = copy(res.phase_velocities)
    phase_velocity_branches = copy(res.phase_velocity_branches)
    measured_phases = copy(res.measured_phases)
    selected_phase_branches = copy(res.selected_phase_branches)
    phase_suspect = copy(res.phase_suspect)
    u_predicted_from_phase = copy(res.u_predicted_from_phase)
    amplitudes = copy(res.amplitudes)
    arrival_times = copy(res.arrival_times)
    quality_factors = copy(res.quality_factors)
    all_peaks = [copy(peaks) for peaks in res.all_peaks]

    group_velocities[invalid] .= NaN
    phase_velocities[invalid] .= NaN
    phase_velocity_branches[invalid, :] .= NaN
    measured_phases[invalid] .= NaN
    selected_phase_branches[invalid] .= 0
    phase_suspect[invalid] .= false
    u_predicted_from_phase[invalid] .= NaN
    amplitudes[invalid] .= NaN
    arrival_times[invalid] .= NaN
    quality_factors[invalid] .= NaN
    for ip in findall(invalid)
        all_peaks[ip] = Tuple{Float64,Float64}[]
    end

    filtered_traces = copy(res.filtered_traces)
    envelopes = copy(res.envelopes)
    if ndims(filtered_traces) == 2 && size(filtered_traces, 2) == length(valid)
        filtered_traces[:, invalid] .= NaN
    end
    if ndims(envelopes) == 2 && size(envelopes, 2) == length(valid)
        envelopes[:, invalid] .= NaN
    end

    mft.MFTResult(
        res.periods, res.frequencies,
        group_velocities, phase_velocities,
        res.phase_branch_numbers, phase_velocity_branches,
        measured_phases, selected_phase_branches, phase_suspect,
        u_predicted_from_phase,
        amplitudes, filtered_traces, envelopes, arrival_times,
        res.distance, quality_factors, all_peaks, res.time, res.storage_mode)
end

function _mask_branch_result(res, distance::Real)
    valid_periods = Set(_mft_compute_periods(distance))
    valid = [period in valid_periods for period in res.periods]
    branch_correlation = copy(res.branch_correlation)
    branch_correlation[.!valid] .= NaN
    mft.BranchAnalysisResult(
        _mask_mft_result(res.causal_result, distance),
        _mask_mft_result(res.acausal_result, distance),
        [_mask_multimodal_dispersion(mode, valid) for mode in res.causal_modes],
        [_mask_multimodal_dispersion(mode, valid) for mode in res.acausal_modes],
        branch_correlation,
        res.periods,
        res.distance)
end

function _mask_branch_batch(batch, distances::AbstractVector{<:Real})
    masked_results = [_mask_branch_result(batch.state_results[i], Float64(distances[i]))
        for i in eachindex(batch.state_results)]
    branch_correlation = copy(batch.branch_correlation)
    for i in eachindex(masked_results)
        branch_correlation[:, i] .= masked_results[i].branch_correlation
    end
    mft.BranchBatchAnalysisResult(masked_results, branch_correlation,
        batch.periods, batch.state_labels)
end

# ╔═╡ 1f2f5740-5bd0-11f1-0032-4b3f11e7a001
function _subset_branch_batch(batch, inds::AbstractVector{<:Integer})
    mft.BranchBatchAnalysisResult(
        batch.state_results[inds],
        batch.branch_correlation[:, inds],
        batch.periods,
        batch.state_labels[inds])
end

# ╔═╡ 1f2f5740-5bd0-11f1-0033-4b3f11e7a001
function _split_batch_by_pair(batch, pair_keys::AbstractVector{<:AbstractString})
    grouped = Dict{String,Vector{Int}}()
    for (i, pair_label) in enumerate(pair_keys)
        push!(get!(grouped, String(pair_label), Int[]), i)
    end
    Dict(pair_label => _subset_branch_batch(batch, inds) for (pair_label, inds) in grouped)
end

function _append_branch_batch!(analyses::Dict{String,Any}, pair_label::String, batch)
    if haskey(analyses, pair_label)
        existing = analyses[pair_label]
        if existing.periods != batch.periods
            error("Cannot append MFT batches for $(pair_label): period grids differ")
        end
        analyses[pair_label] = mft.BranchBatchAnalysisResult(
            vcat(existing.state_results, batch.state_results),
            hcat(existing.branch_correlation, batch.branch_correlation),
            existing.periods,
            vcat(existing.state_labels, batch.state_labels))
    else
        analyses[pair_label] = batch
    end
    analyses
end

function _append_split_batch_by_pair!(analyses::Dict{String,Any}, batch,
        pair_keys::AbstractVector{<:AbstractString})
    for (pair_label, pair_batch) in _split_batch_by_pair(batch, pair_keys)
        _append_branch_batch!(analyses, pair_label, pair_batch)
    end
    analyses
end

# ╔═╡ 1f2f5740-5bd0-11f1-0034-4b3f11e7a001
function _analyze_pair_branch_specs(specs; storage_mode_for_spec=spec -> :picks_only,
        compute_phase::Bool=false,
        phase_velocity_range::Tuple{Float64,Float64}=(2.0, 5.0),
        use_phtovel::Bool=false)
    isempty(specs) && return Dict{String,Any}()
    grouped = Dict{Tuple{Symbol,Int},Vector{Int}}()
    for (i, spec) in enumerate(specs)
        n = min(length(vec(spec.causal)), length(vec(spec.acausal)))
        n == 0 && continue
        isempty(_mft_compute_periods(spec.distance)) && continue
        push!(get!(grouped, (storage_mode_for_spec(spec), n), Int[]), i)
    end
    analyses = Dict{String,Any}()
    for ((storage_mode, n), inds) in grouped
        periods = _mft_shared_periods_for_specs(specs, inds)
        isempty(periods) && continue
        cols_c = [Float64.(vec(specs[i].causal)) for i in inds]
        cols_ac = [Float64.(vec(specs[i].acausal)) for i in inds]
        bank = _mft_filter_bank_for(periods, n;
            storage_mode=storage_mode, n_waveforms=2 * length(inds))
        isnothing(bank) && continue
        W_c = reduce(hcat, [col[1:n] for col in cols_c])
        W_ac = reduce(hcat, [col[1:n] for col in cols_ac])
        distances = [Float64(specs[i].distance) for i in inds]
        batch = mft.analyze_causal_acausal_branches(
            W_c, W_ac, distances, bank;
            state_labels=[String(specs[i].label) for i in inds],
            max_modes=mft_max_modes,
            compute_phase=compute_phase,
            use_phtovel=use_phtovel,
            phase_velocity_range=phase_velocity_range)
        masked_batch = _mask_branch_batch(batch, distances)
        _append_split_batch_by_pair!(analyses, masked_batch, [String(specs[i].pair_label) for i in inds])
    end
    analyses
end

# ╔═╡ 1f2f5740-5bd0-11f1-0035-4b3f11e7a001
global_average_mft_analyses = let
    specs = NamedTuple[]
    for item in globalavg_source_state_averages
        gc = Float64.(vec(item.global_avg_c))
        gac = Float64.(vec(item.global_avg_ac))
        (isempty(gc) || isempty(gac)) && continue
        n = min(length(gc), length(gac))
        push!(specs, (; pair_label=String(item.pair_label), label=String(item.pair_label),
            causal=gc[1:n], acausal=gac[1:n], distance=Float64(item.distance)))
    end
    pair_batches = _analyze_pair_branch_specs(specs; compute_phase=true,
        phase_velocity_range=phase_velocity_range,
        use_phtovel=ui_mft_use_phtovel)
    Dict(pair_label => first(batch.state_results)
        for (pair_label, batch) in pair_batches if !isempty(batch.state_results))
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0036-4b3f11e7a001
function _global_average_uc_rows(res, pair_label::String;
        wavelength_ref_velocity::Float64=2.0,
        wavelength_fraction::Float64=0.33)
    rows = NamedTuple[]
    dist = Float64(res.distance)
    for (branch, branch_res) in (("causal", res.causal_result), ("acausal", res.acausal_result))
        u_pred = any(isfinite, branch_res.u_predicted_from_phase) ?
            branch_res.u_predicted_from_phase : mft.compute_group_velocity_from_phase(branch_res)
        for ip in eachindex(branch_res.periods)
            period = Float64(branch_res.periods[ip])
            isfinite(period) && period > 0.0 || continue
            wavelength_ref_velocity * period < wavelength_fraction * dist || continue
            u_meas = Float64(branch_res.group_velocities[ip])
            u_hat = Float64(u_pred[ip])
            c = Float64(branch_res.phase_velocities[ip])
            quality = Float64(branch_res.quality_factors[ip])
            isfinite(u_meas) && u_meas > 0.0 || continue
            isfinite(u_hat) && u_hat > 0.0 || continue
            isfinite(c) && c > 0.0 || continue
            relerr = abs(u_meas - u_hat) / max(abs(u_hat), eps(Float64))
            selected_branch = ip <= length(branch_res.selected_phase_branches) ?
                Int(branch_res.selected_phase_branches[ip]) : 0
            push!(rows, (; pair_label, branch, distance=dist, period,
                group_velocity=u_meas,
                phase_velocity=c,
                predicted_group_velocity=u_hat,
                relative_error=relerr,
                relative_agreement=1.0 / (1.0 + relerr),
                quality,
                selected_phase_branch=selected_branch))
        end
    end
    sort(rows, by=r -> (r.pair_label, r.branch, r.period))
end

# ╔═╡ 1f2f5740-5bd0-11f1-0037-4b3f11e7a001
function _uc_rows_from_mft_result(res, pair_label::String, label::String)
    u_pred = any(isfinite, res.u_predicted_from_phase) ?
        res.u_predicted_from_phase : mft.compute_group_velocity_from_phase(res)
    rows = NamedTuple[]
    for ip in eachindex(res.periods)
        period = Float64(res.periods[ip])
        u_meas = Float64(res.group_velocities[ip])
        u_hat = Float64(u_pred[ip])
        c = Float64(res.phase_velocities[ip])
        quality = Float64(res.quality_factors[ip])
        isfinite(period) && period > 0.0 || continue
        isfinite(u_meas) && u_meas > 0.0 || continue
        isfinite(u_hat) && u_hat > 0.0 || continue
        isfinite(c) && c > 0.0 || continue
        relerr = abs(u_meas - u_hat) / max(abs(u_hat), eps(Float64))
        push!(rows, (; pair_label, label, period,
            group_velocity=u_meas,
            phase_velocity=c,
            predicted_group_velocity=u_hat,
            relative_error=relerr,
            relative_agreement=1.0 / (1.0 + relerr),
            quality))
    end
    sort(rows, by=r -> r.period)
end

# ╔═╡ 1f2f5740-5bd0-11f1-0038-4b3f11e7a001
function _uc_rows_from_branch_analysis_result(res::mft.BranchAnalysisResult,
        pair_label::String, label::String)
    rows = NamedTuple[]
    for (branch, branch_res) in (("causal", res.causal_result), ("acausal", res.acausal_result))
        append!(rows, [merge(row, (; branch)) for row in _uc_rows_from_mft_result(branch_res, pair_label, label)])
    end
    sort(rows, by=r -> (r.branch, r.period))
end

# ╔═╡ 1f2f5740-5bd0-11f1-0039-4b3f11e7a001
global_average_uc_rows = let
    rows = NamedTuple[]
    for (pair_label, res) in global_average_mft_analyses
        append!(rows, _global_average_uc_rows(res, String(pair_label);
            wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
            wavelength_fraction=Float64(ui_wavelength_fraction)))
    end
    sort(rows, by=r -> (r.pair_label, r.branch, r.period))
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0040-4b3f11e7a001
_pair_mft_analyses_joint = let
    specs = NamedTuple[]
    for item in run_source_state_averages
        nstates = min(size(item.causal, 2), size(item.acausal, 2))
        for i in 1:nstates
            label = i <= length(item.combo_labels) ? item.combo_labels[i] : string(i)
            push!(specs, (; pair_label=String(item.pair_label),
                label="$(item.pair_label) seed $(item.seed) | $(label)",
                causal=Float64.(vec(item.causal[:, i])),
                acausal=Float64.(vec(item.acausal[:, i])),
                distance=Float64(item.distance)))
        end
    end
    _analyze_pair_branch_specs(specs; compute_phase=true,
        phase_velocity_range=phase_velocity_range,
        use_phtovel=ui_mft_use_phtovel)
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0041-4b3f11e7a001
_pair_mft_analyses_marginal = let
    specs = NamedTuple[]
    for item in run_source_state_averages
        K1 = isempty(item.marginal_stage1_ac) ? 0 : size(item.marginal_stage1_ac, 2)
        K2 = isempty(item.marginal_stage2_ac) ? 0 : size(item.marginal_stage2_ac, 2)
        for k in 1:K1
            lbl = k <= length(item.marginal_stage1_labels) ? item.marginal_stage1_labels[k] : "s1=$k"
            push!(specs, (; pair_label=String(item.pair_label),
                label="$(item.pair_label) seed $(item.seed) | S1 $(lbl)",
                causal=Float64.(vec(item.marginal_stage1_c[:, k])),
                acausal=Float64.(vec(item.marginal_stage1_ac[:, k])),
                distance=Float64(item.distance)))
        end
        for k in 1:K2
            lbl = k <= length(item.marginal_stage2_labels) ? item.marginal_stage2_labels[k] : "s2=$k"
            push!(specs, (; pair_label=String(item.pair_label),
                label="$(item.pair_label) seed $(item.seed) | S2 $(lbl)",
                causal=Float64.(vec(item.marginal_stage2_c[:, k])),
                acausal=Float64.(vec(item.marginal_stage2_ac[:, k])),
                distance=Float64(item.distance)))
        end
    end
    _analyze_pair_branch_specs(specs; compute_phase=true,
        phase_velocity_range=phase_velocity_range,
        use_phtovel=ui_mft_use_phtovel)
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0042-4b3f11e7a001
pair_mft_analyses = ui_mft_mode == "Marginal stages (K1+K2)" ?
    _pair_mft_analyses_marginal : _pair_mft_analyses_joint;

# ╔═╡ 13d7824e-5ddc-11f1-a7cc-3f88810004cb
begin
    _run_options_ui = isempty(source_state_plot_runs.labels) ? ["None"] : source_state_plot_runs.labels
    @bind _inspection_run_controls PlutoUI.combine() do Child
        md"""
        | Inspection run | Value |
        |:---|:---|
        | Selected run/seed | $(Child("run", Select(_run_options_ui; default=first(_run_options_ui)))) |
        """
    end
end

# ╔═╡ 13d78550-5ddc-11f1-a8f8-cf4f9a494ec6
selected_source_state_run = String(_inspection_run_controls.run);

# ╔═╡ 1f2f5740-5bd0-11f1-0026-4b3f11e7a001
selected_source_state_item = begin
    if isempty(source_state_plot_runs.items) || selected_source_state_run == "None"
        nothing
    else
        idx = something(findfirst(==(selected_source_state_run), source_state_plot_runs.labels), 1)
        source_state_plot_runs.items[idx]
    end
end;

# ╔═╡ 1f2f5740-5bd0-11f1-003a-4b3f11e7a001
function _uc_error_score(errs; method::AbstractString="geomean", huber_delta::Real=0.10)
    vals = Float64[e for e in errs if isfinite(e) && e >= 0.0]
    isempty(vals) && return Inf
    method0 = lowercase(String(method))
    if method0 == "geomean"
        return exp(mean(log.(max.(vals, eps(Float64)))))
    elseif method0 == "median"
        return median(vals)
    elseif method0 == "mean"
        return mean(vals)
    elseif method0 == "p95"
        return quantile(vals, 0.95)
    elseif method0 == "max"
        return maximum(vals)
    elseif method0 == "huber"
        δ = max(Float64(huber_delta), eps(Float64))
        return mean(v <= δ ? 0.5 * v^2 : δ * (v - 0.5δ) for v in vals)
    else
        error("Unknown U-c score method: $(method)")
    end
end

# ╔═╡ 1f2f5740-5bd0-11f1-003b-4b3f11e7a001
function _uc_error_stats(errs; method::AbstractString="geomean", huber_delta::Real=0.10)
    vals = Float64[e for e in errs if isfinite(e) && e >= 0.0]
    if isempty(vals)
        return (; uc_score=Inf, median_relative_error=Inf,
            geomean_relative_error=Inf, mean_relative_error=Inf,
            p95_relative_error=Inf, max_relative_error=Inf,
            huber_relative_error=Inf, n_valid=0)
    end
    (; uc_score=_uc_error_score(vals; method, huber_delta),
        median_relative_error=median(vals),
        geomean_relative_error=_uc_error_score(vals; method="geomean", huber_delta),
        mean_relative_error=mean(vals),
        p95_relative_error=quantile(vals, 0.95),
        max_relative_error=maximum(vals),
        huber_relative_error=_uc_error_score(vals; method="huber", huber_delta),
        n_valid=length(vals))
end


# ╔═╡ 1f2f5740-5bd0-11f1-003c-4b3f11e7a001
_format_uc_score_value(x; digits::Int=3) = isfinite(x) ? string(round(x; digits)) : "NA"

# ╔═╡ 1f2f5740-5bd0-11f1-0048-4b3f11e7a001
function _plot_global_average_uc_consistency(rows, pair_label::String;
        codebook_rows=NamedTuple[],
        velocity_range=nothing,
        title::String="Global Average U-c Consistency",
        uc_score_method::AbstractString="geomean",
        uc_huber_delta::Real=0.10)
    pair_rows_for_score = [r for r in rows if r.pair_label == pair_label]
    overlay_rows_for_score = [r for r in codebook_rows if r.pair_label == pair_label]
    global_errs = Float64[r.relative_error for r in pair_rows_for_score
        if isfinite(r.relative_error) && r.relative_error >= 0.0]
    overlay_errs = Float64[r.relative_error for r in overlay_rows_for_score
        if isfinite(r.relative_error) && r.relative_error >= 0.0]
    global_score = _uc_error_score(global_errs; method=uc_score_method,
        huber_delta=uc_huber_delta)
    overlay_score = _uc_error_score(overlay_errs; method=uc_score_method,
        huber_delta=uc_huber_delta)
    score_suffix = isempty(overlay_errs) ?
        "score $(uc_score_method): global=$(_format_uc_score_value(global_score))" :
        "score $(uc_score_method): global=$(_format_uc_score_value(global_score)), overlay=$(_format_uc_score_value(overlay_score))"
    plot_title = "$(title) | $(score_suffix)"
    return mft.plot_uc_consistency_comparison(rows;
        pair_label=pair_label,
        overlay_rows=codebook_rows,
        velocity_range=velocity_range,
        title=plot_title)
    pair_rows = [r for r in rows if r.pair_label == pair_label]
    isempty(pair_rows) && return PlutoPlotly.plot(PlutoPlotly.scatter(
        x=[0.0], y=[0.0], text=["No finite global-average U-c rows for $(pair_label)"]))

    branches = ["causal", "acausal"]
    traces = AbstractTrace[]
    colors = Dict("causal" => "#1f77b4", "acausal" => "#d62728")
    symbols = Dict("causal" => "circle", "acausal" => "diamond")
    error_scales = Dict("causal" => "Blues", "acausal" => "Reds")
    for branch in branches
        b = sort([r for r in pair_rows if r.branch == branch], by=r -> r.period)
        isempty(b) && continue
        err_text = ["$(branch)<br>T=$(round(r.period; digits=3)) s<br>U=$(round(r.group_velocity; digits=3)) km/s<br>c=$(round(r.phase_velocity; digits=3)) km/s<br>U_pred=$(round(r.predicted_group_velocity; digits=3)) km/s<br>rel err=$(round(r.relative_error; digits=3))<br>phase branch=$(r.selected_phase_branch)<br>quality=$(round(r.quality; digits=2))" for r in b]
        push!(traces, PlutoPlotly.scatter(x=[r.period for r in b], y=[r.group_velocity for r in b],
            yaxis="y", mode="markers+lines", name="$(branch) U", legendgroup=branch,
            marker=PlutoPlotly.attr(size=8, color=[r.relative_error for r in b],
                cmin=0.0, cmax=0.5, colorscale=error_scales[branch], showscale=false,
                symbol=symbols[branch], line=PlutoPlotly.attr(color="black", width=0.8)),
            line=PlutoPlotly.attr(color=colors[branch], width=1.5), text=err_text, hoverinfo="text"))
        push!(traces, PlutoPlotly.scatter(x=[r.period for r in b], y=[r.predicted_group_velocity for r in b],
            yaxis="y", mode="lines+markers", name="$(branch) U from c(T)", legendgroup=branch,
            marker=PlutoPlotly.attr(size=5, color=colors[branch], symbol="circle-open"),
            line=PlutoPlotly.attr(color=colors[branch], width=1.8, dash="dash"),
            hovertemplate="$(branch) U_pred<br>T=%{x:.3f} s<br>U_pred=%{y:.3f} km/s<extra></extra>"))
        push!(traces, PlutoPlotly.scatter(x=[r.period for r in b], y=[r.phase_velocity for r in b],
            yaxis="y", mode="lines+markers", name="$(branch) c", legendgroup=branch,
            marker=PlutoPlotly.attr(size=4.5, color=colors[branch], symbol="square-open"),
            line=PlutoPlotly.attr(color=colors[branch], width=1.3, dash="dot"),
            hovertemplate="$(branch) phase velocity<br>T=%{x:.3f} s<br>c=%{y:.3f} km/s<extra></extra>"))
        push!(traces, PlutoPlotly.scatter(x=[r.period for r in b], y=[min(r.relative_error, 0.5) for r in b],
            xaxis="x2", yaxis="y2", mode="lines+markers", name="$(branch) relative error",
            legendgroup=branch, showlegend=false,
            marker=PlutoPlotly.attr(size=7, color=[min(r.relative_error, 0.5) for r in b],
                cmin=0.0, cmax=0.5, colorscale=error_scales[branch], showscale=false,
                symbol=symbols[branch], line=PlutoPlotly.attr(color="black", width=0.5)),
            line=PlutoPlotly.attr(color=colors[branch], width=1.4),
            text=["$(branch)<br>T=$(round(r.period; digits=3)) s<br>relative error=$(round(r.relative_error; digits=3))$(r.relative_error > 0.5 ? "<br>plotted at 0.5 cap" : "")" for r in b],
            hoverinfo="text"))
    end

    overlay_rows_all = [r for r in codebook_rows if r.pair_label == pair_label]
    overlay_labels = unique(String(r.label) for r in overlay_rows_all)
    overlay_title = isempty(overlay_labels) ? "" :
        " | overlay: $(join(first(overlay_labels, min(2, length(overlay_labels))), " + "))$(length(overlay_labels) > 2 ? " + ..." : "")"
    overlay_palette = ["#111111", "#7b3294", "#008837", "#e66101", "#5e3c99", "#4d4d4d"]
    overlay_groups = NamedTuple[]
    for label in overlay_labels
        label_rows = [r for r in overlay_rows_all if String(r.label) == label]
        if any(r -> hasproperty(r, :branch), label_rows)
            for branch in branches
                rows_branch = sort([r for r in label_rows if hasproperty(r, :branch) && String(r.branch) == branch], by=r -> r.period)
                isempty(rows_branch) || push!(overlay_groups, (; label, branch, rows=rows_branch))
            end
        else
            push!(overlay_groups, (; label, branch="", rows=sort(label_rows, by=r -> r.period)))
        end
    end
    for (igroup, group) in enumerate(overlay_groups)
        overlay_rows = group.rows
        isempty(overlay_rows) && continue
        label = String(group.label)
        branch = String(group.branch)
        branch_suffix = isempty(branch) ? "" : " $(branch)"
        codebook_color = overlay_palette[mod1(igroup, length(overlay_palette))]
        accent_color = branch == "acausal" ? "#7b3294" : "#b8860b"
        hover = ["$(label)$(isempty(branch) ? "" : "<br>$(branch)")<br>T=$(round(r.period; digits=3)) s<br>U=$(round(r.group_velocity; digits=3)) km/s<br>c=$(round(r.phase_velocity; digits=3)) km/s<br>U_pred=$(round(r.predicted_group_velocity; digits=3)) km/s<br>rel err=$(round(r.relative_error; digits=3))<br>quality=$(round(r.quality; digits=2))" for r in overlay_rows]
        push!(traces, PlutoPlotly.scatter(x=[r.period for r in overlay_rows], y=[r.group_velocity for r in overlay_rows],
            yaxis="y", mode="markers+lines", name="overlay$(branch_suffix) U", legendgroup="overlay $(igroup)",
            marker=PlutoPlotly.attr(size=8.5, color=accent_color, symbol="star",
                line=PlutoPlotly.attr(color=codebook_color, width=0.8)),
            line=PlutoPlotly.attr(color=codebook_color, width=2.0), text=hover, hoverinfo="text"))
        push!(traces, PlutoPlotly.scatter(x=[r.period for r in overlay_rows], y=[r.predicted_group_velocity for r in overlay_rows],
            yaxis="y", mode="lines+markers", name="overlay$(branch_suffix) U from c(T)", legendgroup="overlay $(igroup)",
            marker=PlutoPlotly.attr(size=5, color=codebook_color, symbol="star-open"),
            line=PlutoPlotly.attr(color=codebook_color, width=1.8, dash="dash"),
            hovertemplate="overlay$(branch_suffix) U_pred<br>T=%{x:.3f} s<br>U_pred=%{y:.3f} km/s<extra></extra>"))
        push!(traces, PlutoPlotly.scatter(x=[r.period for r in overlay_rows], y=[r.phase_velocity for r in overlay_rows],
            yaxis="y", mode="lines+markers", name="overlay$(branch_suffix) c", legendgroup="overlay $(igroup)",
            marker=PlutoPlotly.attr(size=4.8, color=codebook_color, symbol="square-open"),
            line=PlutoPlotly.attr(color=codebook_color, width=1.3, dash="dot"),
            hovertemplate="overlay$(branch_suffix) phase velocity<br>T=%{x:.3f} s<br>c=%{y:.3f} km/s<extra></extra>"))
        push!(traces, PlutoPlotly.scatter(x=[r.period for r in overlay_rows], y=[min(r.relative_error, 0.5) for r in overlay_rows],
            xaxis="x2", yaxis="y2", mode="lines+markers", name="overlay$(branch_suffix) relative error",
            legendgroup="overlay $(igroup)", showlegend=false,
            marker=PlutoPlotly.attr(size=7.5, color=accent_color, symbol="star",
                line=PlutoPlotly.attr(color=codebook_color, width=0.6)),
            line=PlutoPlotly.attr(color=codebook_color, width=1.5),
            text=["$(label)$(isempty(branch) ? "" : "<br>$(branch)")<br>T=$(round(r.period; digits=3)) s<br>relative error=$(round(r.relative_error; digits=3))$(r.relative_error > 0.5 ? "<br>plotted at 0.5 cap" : "")" for r in overlay_rows],
            hoverinfo="text"))
    end

    all_vels = Float64[]
    for r in pair_rows
        append!(all_vels, [r.group_velocity, r.phase_velocity, r.predicted_group_velocity])
    end
    for r in overlay_rows_all
        append!(all_vels, [r.group_velocity, r.phase_velocity, r.predicted_group_velocity])
    end
    all_vels = [v for v in all_vels if isfinite(v) && v > 0.0]
    y_range = isnothing(velocity_range) ?
        (isempty(all_vels) ? nothing : [0.9 * minimum(all_vels), 1.1 * maximum(all_vels)]) :
        [Float64(velocity_range[1]), Float64(velocity_range[2])]
    PlutoPlotly.plot(traces, PlutoPlotly.Layout(
        title="$(title): $(pair_label)$(overlay_title)",
        xaxis=PlutoPlotly.attr(type="log", domain=[0.0, 0.86], anchor="y",
            showticklabels=false, showgrid=true, gridcolor="rgba(0,0,0,0.10)", zeroline=false),
        xaxis2=PlutoPlotly.attr(title="Period (s)", type="log", domain=[0.0, 0.86],
            anchor="y2", matches="x", showgrid=true, gridcolor="rgba(0,0,0,0.10)", zeroline=false),
        yaxis=PlutoPlotly.attr(title="Velocity (km/s)", range=y_range, domain=[0.33, 1.0],
            showgrid=true, gridcolor="rgba(0,0,0,0.10)", zeroline=false),
        yaxis2=PlutoPlotly.attr(title="relative |U - U(c)| / U(c) (capped at 0.5)",
            range=[0.0, 0.5], domain=[0.0, 0.23], showgrid=true,
            gridcolor="rgba(0,0,0,0.10)", zeroline=false),
        width=1100, height=680,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=PlutoPlotly.attr(l=78, r=145, t=78, b=72),
        legend=PlutoPlotly.attr(orientation="h", x=0.0, y=1.10,
            bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1)))
end

# ╔═╡ 1f2f5740-5bd0-11f1-e001-4b3f11e7a001
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

# ╔═╡ 1f2f5740-5bd0-11f1-e002-4b3f11e7a001
function _latlong_from_csv(station_code::String)
    tbl = xj_latlong
    isempty(tbl) && return nothing
    col = tbl[!, "Station Code"]
    i = findfirst(==(station_code), col)
    i === nothing && return nothing
    (lat=Float64(tbl[i, :Latitude]), lon=Float64(tbl[i, :Longitude]))
end

# ╔═╡ 1f2f5740-5bd0-11f1-e003-4b3f11e7a001
function _pair_geometry_for_export(pair_label::String)
    parts = split(pair_label, "-")
    length(parts) == 2 || return nothing
    g1 = _latlong_from_csv(String(parts[1]))
    g2 = _latlong_from_csv(String(parts[2]))
    (isnothing(g1) || isnothing(g2)) && return nothing
    idx = findfirst(x -> x.pair_label == pair_label, run_source_state_averages)
    dist = isnothing(idx) ? NaN : Float64(run_source_state_averages[idx].distance)
    (; pair_label, lat1=g1.lat, lon1=g1.lon, lat2=g2.lat, lon2=g2.lon, distance=dist)
end

# ╔═╡ 1f2f5740-5bd0-11f1-e004-4b3f11e7a001
# Score every (state, branch) independently across all pairs in one batched MFT pass.
# Returns specs plus metadata keyed by a stable label so result mapping is not order-fragile.
function _build_scoring_branch_specs()
    analysis_specs = NamedTuple[]
    meta = NamedTuple[]
    for item in run_source_state_averages
        pair_label = String(item.pair_label)
        dist = Float64(item.distance)
        seed = item.seed
        nstates = min(size(item.causal, 2), size(item.acausal, 2))
        for i in 1:nstates
            lbl = i <= length(item.combo_labels) ? item.combo_labels[i] : string(i)
            display_str = "seed$(seed) | joint $(lbl)"
            for (br, sig) in (("causal", Float64.(vec(item.causal[:, i]))),
                               ("acausal", Float64.(vec(item.acausal[:, i]))))
                label = "$(pair_label) | score $(length(analysis_specs) + 1) | $(display_str) [$(br)]"
                push!(analysis_specs, (; pair_label, label, causal=sig, acausal=sig, distance=dist))
                push!(meta, (; label, pair_label, display=display_str, kind="joint", branch=br, seed))
            end
        end
        K1 = isempty(item.marginal_stage1_c) ? 0 : size(item.marginal_stage1_c, 2)
        K2 = isempty(item.marginal_stage2_c) ? 0 : size(item.marginal_stage2_c, 2)
        for k in 1:K1
            lbl = k <= length(item.marginal_stage1_labels) ? item.marginal_stage1_labels[k] : "s1=$k"
            display_str = "seed$(seed) | S1 $(lbl)"
            n = min(size(item.marginal_stage1_c, 1), size(item.marginal_stage1_ac, 1))
            for (br, sig) in (("causal", Float64.(item.marginal_stage1_c[1:n, k])),
                               ("acausal", Float64.(item.marginal_stage1_ac[1:n, k])))
                label = "$(pair_label) | score $(length(analysis_specs) + 1) | $(display_str) [$(br)]"
                push!(analysis_specs, (; pair_label, label, causal=sig, acausal=sig, distance=dist))
                push!(meta, (; label, pair_label, display=display_str, kind="S1", branch=br, seed))
            end
        end
        for k in 1:K2
            lbl = k <= length(item.marginal_stage2_labels) ? item.marginal_stage2_labels[k] : "s2=$k"
            display_str = "seed$(seed) | S2 $(lbl)"
            n = min(size(item.marginal_stage2_c, 1), size(item.marginal_stage2_ac, 1))
            for (br, sig) in (("causal", Float64.(item.marginal_stage2_c[1:n, k])),
                               ("acausal", Float64.(item.marginal_stage2_ac[1:n, k])))
                label = "$(pair_label) | score $(length(analysis_specs) + 1) | $(display_str) [$(br)]"
                push!(analysis_specs, (; pair_label, label, causal=sig, acausal=sig, distance=dist))
                push!(meta, (; label, pair_label, display=display_str, kind="S2", branch=br, seed))
            end
        end
        for k1 in 1:K1, k2 in 1:K2
            l1 = k1 <= length(item.marginal_stage1_labels) ? item.marginal_stage1_labels[k1] : "s1=$k1"
            l2 = k2 <= length(item.marginal_stage2_labels) ? item.marginal_stage2_labels[k2] : "s2=$k2"
            display_str = "seed$(seed) | S1 $(l1) + S2 $(l2)"
            n = minimum((size(item.marginal_stage1_c, 1), size(item.marginal_stage1_ac, 1),
                size(item.marginal_stage2_c, 1), size(item.marginal_stage2_ac, 1)))
            c_sum  = Float64.(item.marginal_stage1_c[1:n, k1])  .+ Float64.(item.marginal_stage2_c[1:n, k2])
            ac_sum = Float64.(item.marginal_stage1_ac[1:n, k1]) .+ Float64.(item.marginal_stage2_ac[1:n, k2])
            for (br, sig) in (("causal", c_sum), ("acausal", ac_sum))
                label = "$(pair_label) | score $(length(analysis_specs) + 1) | $(display_str) [$(br)]"
                push!(analysis_specs, (; pair_label, label, causal=sig, acausal=sig, distance=dist))
                push!(meta, (; label, pair_label, display=display_str, kind="S1+S2", branch=br, seed))
            end
        end
    end
    analysis_specs, meta
end

function _score_all_branches_batched()
    result = Dict{String,Vector{NamedTuple}}(pl => NamedTuple[] for pl in pair_labels)
    analysis_specs, meta = _build_scoring_branch_specs()
    isempty(analysis_specs) && return result

    meta_by_label = Dict(String(m.label) => m for m in meta)
    analyses = _analyze_pair_branch_specs(analysis_specs;
        compute_phase=true, phase_velocity_range=phase_velocity_range,
        use_phtovel=ui_mft_use_phtovel)

    for (pair_label, batch) in analyses
        scored = NamedTuple[]
        for (res, label) in zip(batch.state_results, batch.state_labels)
            m = get(meta_by_label, String(label), nothing)
            isnothing(m) && continue
            pair_label_m = String(m.pair_label)
            pair_label_m == String(pair_label) || continue
            branch_res = String(m.branch) == "causal" ? res.causal_result : res.acausal_result
            uc_rows = _uc_rows_from_mft_result(branch_res, pair_label_m, String(label))
            errs = Float64[r.relative_error for r in uc_rows if isfinite(r.relative_error) && r.relative_error >= 0.0]
            stats = _uc_error_stats(errs; method=ui_uc_score_method, huber_delta=ui_uc_huber_delta)
            push!(scored, merge(m, stats,
                (; periods=Float64.(branch_res.periods),
                   group_velocities=Float64.(branch_res.group_velocities))))
        end
        sort!(scored, by=s -> (s.uc_score, s.max_relative_error, -s.n_valid, s.display, s.branch))
        result[String(pair_label)] = scored
    end
    result
end

# ╔═╡ 1f2f5740-5bd0-11f1-e005-4b3f11e7a001
_all_scored_specs_by_pair = _score_all_branches_batched();

# ╔═╡ 1f2f5740-5bd0-11f1-0006-4b3f11e7a001
begin
    _scored_overlay_options = let
        specs = get(_all_scored_specs_by_pair, selected_plot_pair, NamedTuple[])
        ["None"; String[
            "$(s.display) | $(s.branch) | score=$(_format_uc_score_value(s.uc_score)) ($(ui_uc_score_method)), med=$(_format_uc_score_value(s.median_relative_error)), max=$(_format_uc_score_value(s.max_relative_error))"
            for s in specs]]
    end
    @bind _selected_source_state_overlay_controls PlutoUI.combine() do Child
        md"""
        | Selected source-state overlay | Value |
        |:---|:---|
        | Ranked state (U-c score) | $(Child("combo", Select(_scored_overlay_options; default="None"))) |
        """
    end
end

# ╔═╡ 1f2f5740-5bd0-11f1-0007-4b3f11e7a001
selected_marginal_combo_overlay = String(_selected_source_state_overlay_controls.combo);

# ╔═╡ 1f2f5740-5bd0-11f1-0046-4b3f11e7a001
# Overlay rows for the plot: look up the selected ranked (state, branch) spec
# and re-run MFT for just that spec to produce uc rows.
selected_marginal_combo_overlay_rows = let
    if selected_marginal_combo_overlay == "None"
        NamedTuple[]
    else
        specs = get(_all_scored_specs_by_pair, selected_plot_pair, NamedTuple[])
        idx = findfirst(s ->
            startswith(selected_marginal_combo_overlay,
                "$(s.display) | $(s.branch)"),
            specs)
        if isnothing(idx)
            NamedTuple[]
        else
            spec = specs[idx]
            label = "$(spec.display) [$(spec.branch)] | score=$(_format_uc_score_value(spec.uc_score))"
            # periods and group_velocities already computed on spec — build uc rows directly
            rows = NamedTuple[]
            for ip in eachindex(spec.periods)
                period = Float64(spec.periods[ip])
                gv     = Float64(spec.group_velocities[ip])
                isfinite(period) && period > 0 || continue
                isfinite(gv)    && gv    > 0 || continue
                # relative_error not re-derived here; use placeholder for plot colour
                push!(rows, (; pair_label=selected_plot_pair, label, period,
                    group_velocity=gv,
                    phase_velocity=NaN, predicted_group_velocity=NaN,
                    relative_error=spec.uc_score,
                    relative_agreement=1.0 / (1.0 + spec.uc_score),
                    quality=1.0, branch=spec.branch))
            end
            rows
        end
    end
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0047-4b3f11e7a001
selected_mft_waveform_combined_overlay_rows = selected_marginal_combo_overlay_rows;

# ╔═╡ 1f2f5740-5bd0-11f1-0008-4b3f11e7a001
if @isdefined(selected_plot_pair)
    WideCell(_plot_global_average_uc_consistency(
        global_average_uc_rows, selected_plot_pair;
        codebook_rows=selected_mft_waveform_combined_overlay_rows,
        velocity_range=velocity_range,
        uc_score_method=ui_uc_score_method,
        uc_huber_delta=ui_uc_huber_delta,
        title="Global Average Canonical-Pick U-c Consistency with Selected $(ui_mft_mode) Overlay"))
else
    md""
end

# ╔═╡ 1f2f5740-5bd0-11f1-e006-4b3f11e7a001
best_branch_per_pair = let
    result = Dict{String,NamedTuple}()
    for (pl, specs) in _all_scored_specs_by_pair
        isempty(specs) || (result[pl] = first(specs))
    end
    result
end;

# ╔═╡ fd0e06f4-4e57-4982-a4cf-92ddbc67abcc
function _period_band_stats(rows)
    grouped = Dict{Float64,Vector{Float64}}()
    for r in rows
        push!(get!(grouped, Float64(r.period), Float64[]), Float64(r.group_velocity))
    end
    Dict(p => (; mean=mean(vs), std=std(vs; corrected=false), n=length(vs))
         for (p, vs) in grouped if length(vs) >= 2)
end

# ╔═╡ 36ce5571-c057-4a2b-a69a-c2393e36056d
function _sigma_filter_rows(rows; nsigma::Real=0.8)
    kept = rows
    for _ in 1:12
        stats = _period_band_stats(kept)
        next = filter(kept) do r
            st = get(stats, Float64(r.period), nothing)
            isnothing(st) && return true
            abs(Float64(r.group_velocity) - st.mean) <= nsigma * max(st.std, 1e-9)
        end
        next == kept && break
        kept = next
    end
    kept
end

# ╔═╡ 1f2f5740-5bd0-11f1-e008-4b3f11e7a001
function write_pdsurftomo_dispersion(path::AbstractString, rows; include_count_header::Bool=false)
    isempty(strip(path)) && error("Set output path before writing.")
    mkpath(dirname(path))
    open(path, "w") do io
        include_count_header && println(io, length(rows))
        for row in rows
            @printf(io, "%.8g %.8f %.8f %.8f %.8f %.8f\n",
                Float64(row.period), Float64(row.lat1), Float64(row.lon1),
                Float64(row.lat2), Float64(row.lon2), Float64(row.group_velocity))
        end
    end
    path
end

# ╔═╡ 1f2f5740-5bd0-11f1-e009-4b3f11e7a001
function _globalavg_candidate_rows(wref::Float64, wfrac::Float64)
    rows = NamedTuple[]
    for pl in pair_labels
        geom = _pair_geometry_for_export(pl)
        isnothing(geom) && continue
        for r in global_average_uc_rows
            r.pair_label != pl && continue
            wref * Float64(r.period) < wfrac * geom.distance || continue
            isfinite(Float64(r.group_velocity)) && Float64(r.group_velocity) > 0 || continue
            push!(rows, (; period=Float64(r.period), geom.lat1, geom.lon1, geom.lat2, geom.lon2,
                group_velocity=Float64(r.group_velocity)))
        end
    end
    rows
end

# ╔═╡ 1f2f5740-5bd0-11f1-e00a-4b3f11e7a001
function _vqvae_candidate_rows(wref::Float64, wfrac::Float64, selection::Dict)
    rows = NamedTuple[]
    for (pl, spec) in selection
        geom = _pair_geometry_for_export(pl)
        isnothing(geom) && continue
        for ip in eachindex(spec.periods)
            period = Float64(spec.periods[ip])
            gv     = Float64(spec.group_velocities[ip])
            isfinite(period) && period > 0 || continue
            isfinite(gv)     && gv     > 0 || continue
            wref * period < wfrac * geom.distance || continue
            push!(rows, (; period, geom.lat1, geom.lon1, geom.lat2, geom.lon2, group_velocity=gv))
        end
    end
    sort(rows, by=r -> r.period)
end

# ╔═╡ 1f2f5740-5bd0-11f1-e00b-4b3f11e7a001
begin
    @bind _export_controls PlutoUI.combine() do Child
        md"""
        ### pDSurfTomo Export

        | | Value |
        |:---|:---|
        | **Outlier σ threshold** | $(Child("nsigma", NumberField(0.5:0.1:5.0; default=0.8))) |
        | **Global-average path** | $(Child("globalavg_path", TextField(80; default=joinpath(@__DIR__, "DSurfTomo_runs", "v11_global_avg_dispersion.txt")))) |
        | **VQVAE best-branch path** | $(Child("vqvae_path", TextField(80; default=joinpath(@__DIR__, "DSurfTomo_runs", "v11_vqvae_best_branch_dispersion.txt")))) |
        | Include count header | $(Child("header", CheckBox(default=false))) |
        | Write global avg | $(Child("write_globalavg", CounterButton("Write global-avg"))) |
        | Write VQVAE best-branch | $(Child("write_vqvae", CounterButton("Write VQVAE best-branch"))) |
        """
    end
end

# ╔═╡ 1f2f5740-5bd0-11f1-e00c-4b3f11e7a001
let
    if isempty(best_branch_per_pair)
        md"*(No scored states yet — load artifacts and select pairs.)*"
    else
        rows_sorted = sort(collect(best_branch_per_pair), by=x -> x[2].uc_score)
        header = "| Pair | Best State | Branch | U-c ($ui_uc_score_method) | Median err | Max err | N valid |\n|:-----|:-----------|:-------|:--------------------------|:-----------|:--------|:--------|\n"
        body = join(["| $(k) | $(v.display) | $(v.branch) | $(_format_uc_score_value(v.uc_score)) | $(_format_uc_score_value(v.median_relative_error)) | $(_format_uc_score_value(v.max_relative_error)) | $(v.n_valid) |"
            for (k, v) in rows_sorted], "\n")
        Markdown.parse("### Best-Branch U-c Score Summary\n*(causal & acausal scored independently — lowest score auto-selected for export)*\n\n" * header * body)
    end
end

# ╔═╡ 1f2f5740-5bd0-11f1-e00d-4b3f11e7a001
begin
    _override_pair_keys = sort(collect(keys(best_branch_per_pair)))
    @bind _override_controls PlutoUI.combine() do Child
        if isempty(_override_pair_keys)
            md"*(No pairs loaded yet.)*"
        else
            rows_md = join([let
                all_opts = get(_all_scored_specs_by_pair, pair, NamedTuple[])
                opts = vcat(["auto (best U-c)"], ["$(s.display) | $(s.branch) | score=$(_format_uc_score_value(s.uc_score))" for s in all_opts])
                safe = replace(pair, r"[^A-Za-z0-9]" => "_")
                "| $(pair) | $(Child(Symbol("ov_$(safe)"), Select(opts; default="auto (best U-c)"))) |"
            end for pair in _override_pair_keys], "\n")
            Markdown.parse("### Manual Branch Override *(optional — overrides auto-selection per pair)*\n\n| Pair | Selection |\n|:-----|:----------|\n" * rows_md)
        end
    end
end

# ╔═╡ 1f2f5740-5bd0-11f1-e00e-4b3f11e7a001
resolved_export_selection = let
    result = Dict{String,NamedTuple}()
    for pair in _override_pair_keys
        safe = replace(pair, r"[^A-Za-z0-9]" => "_")
        key  = Symbol("ov_$(safe)")
        ov   = hasproperty(_override_controls, key) ? String(getproperty(_override_controls, key)) : "auto (best U-c)"
        if ov == "auto (best U-c)" || !haskey(best_branch_per_pair, pair)
            haskey(best_branch_per_pair, pair) && (result[pair] = best_branch_per_pair[pair])
        else
            all_opts = get(_all_scored_specs_by_pair, pair, NamedTuple[])
            idx = findfirst(s -> startswith(ov, "$(s.display) | $(s.branch)"), all_opts)
            result[pair] = isnothing(idx) ? best_branch_per_pair[pair] : all_opts[idx]
        end
    end
    result
end;

# ╔═╡ 1f2f5740-5bd0-11f1-e00f-4b3f11e7a001
begin
    Int(_export_controls.write_globalavg) == 0 &&
        return md"Press **Write global-avg** to export the global-average dispersion to pDSurfTomo format."
    let
    raw  = _globalavg_candidate_rows(ui_wavelength_ref_velocity, ui_wavelength_fraction)
    rows = _sigma_filter_rows(raw; nsigma=Float64(_export_controls.nsigma))
    out  = write_pdsurftomo_dispersion(String(_export_controls.globalavg_path), rows;
        include_count_header=Bool(_export_controls.header))
    md"Wrote **$(length(rows))** rows ($(length(raw)-length(rows)) removed as σ-outliers) to `$(out)`"
    end
end

# ╔═╡ 1f2f5740-5bd0-11f1-e010-4b3f11e7a001
let
    Int(_export_controls.write_vqvae) == 0 &&
        return md"Press **Write VQVAE best-branch** to export the U-c–selected VQVAE dispersion to pDSurfTomo format."
    raw  = _vqvae_candidate_rows(ui_wavelength_ref_velocity, ui_wavelength_fraction,
        resolved_export_selection)
    rows = _sigma_filter_rows(raw; nsigma=Float64(_export_controls.nsigma))
    out  = write_pdsurftomo_dispersion(String(_export_controls.vqvae_path), rows;
        include_count_header=Bool(_export_controls.header))
    n_pairs   = length(unique(zip([r.lat1 for r in rows], [r.lon1 for r in rows],
                                   [r.lat2 for r in rows], [r.lon2 for r in rows])))
    md"Wrote **$(length(rows))** rows from **$(n_pairs)** pairs ($(length(raw)-length(rows)) removed as σ-outliers) to `$(out)`"
end

# ╔═╡ 1f2f5740-5bd0-11f1-f001-4b3f11e7a001
begin
    _plot_pair_options_ui = isempty(pair_labels) ? ["None"] : sort(pair_labels)
    @bind _plot_controls PlutoUI.combine() do Child
        md"""
        | Plot controls | Value |
        |:---|:---|
        | Plot pair | $(Child("plot_pair", Select(_plot_pair_options_ui; default=first(_plot_pair_options_ui)))) |
        | Velocity display min (km/s) | $(Child("vmin", NumberField(0.1:0.1:20.0; default=velocity_range_default[1]))) |
        | Velocity display max (km/s) | $(Child("vmax", NumberField(0.1:0.1:20.0; default=velocity_range_default[2]))) |
        """
    end
end

# ╔═╡ Cell order:
# ╟─1f2f5740-5bd0-11f1-0001-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-0002-4b3f11e7a001
# ╟─f3feca4e-36f9-4d82-a0c3-39234d240872
# ╟─1f2f5740-5bd0-11f1-000c-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-0004-4b3f11e7a001
# ╠═4947c5cf-e1db-402e-8816-4e1cb0426802
# ╟─1f2f5740-5bd0-11f1-0006-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-0008-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-000b-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-000d-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0003-4b3f11e7a001
# ╠═b074735f-2faf-4cc3-9497-f33b8fe211c6
# ╠═1f2f5740-5bd0-11f1-0005-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0007-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-0009-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0010-4b3f11e7a001
# ╠═721a47ce-86ca-493f-ba3c-b18047cc8395
# ╠═0bfa7475-8c46-4cac-87a7-6741a3a12aae
# ╠═83975653-0038-4e91-b282-463dd5eb70c7
# ╠═a7b1a698-46d0-4a34-a2e8-f02cbafe26ff
# ╠═1f2f5740-5bd0-11f1-0011-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0012-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0013-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0014-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0015-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0016-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0017-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0018-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0019-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-001a-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0020-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0021-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0022-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0023-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0028-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0024-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0025-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0026-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0027-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0029-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0030-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0031-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0032-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0033-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0034-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0035-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0036-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0037-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0038-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0039-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0040-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0041-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0042-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0046-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0047-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0048-4b3f11e7a001
# ╠═13d7824e-5ddc-11f1-a7cc-3f88810004cb
# ╠═13d78550-5ddc-11f1-a8f8-cf4f9a494ec6
# ╠═1f2f5740-5bd0-11f1-003a-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-003b-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-003c-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e001-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e002-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e003-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e004-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e005-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e006-4b3f11e7a001
# ╠═fd0e06f4-4e57-4982-a4cf-92ddbc67abcc
# ╠═36ce5571-c057-4a2b-a69a-c2393e36056d
# ╠═1f2f5740-5bd0-11f1-e008-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e009-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e00a-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e00b-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e00c-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e00d-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e00e-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e00f-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e010-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-f001-4b3f11e7a001
