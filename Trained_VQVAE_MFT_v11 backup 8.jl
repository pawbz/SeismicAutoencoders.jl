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

# ╔═╡ b8173cf5-5a3e-4078-8344-86f9d7a468eb
using Printf

# ╔═╡ 1f2f5740-5bd0-11f1-0010-4b3f11e7a001
begin
    using Base.Threads
    using JLD2
    using LinearAlgebra
    using PlutoLinks
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

# ╔═╡ 8f91a8aa-e2a8-4b35-a3b3-6e81b850d0c1
begin
    import Makie, CairoMakie
    const _glmakie_available = try
        @eval import GLMakie
        true
    catch err
        @warn "GLMakie is unavailable; falling back to CairoMakie for inline plots." exception=(err, catch_backtrace())
        false
    end
end

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

# ╔═╡ 1f2f5740-5bd0-11f1-0004-4b3f11e7a001
begin
    @bind _mft_plot_controls PlutoUI.combine() do Child
        md"""
        | MFT / wavelength / plot settings | Value |
        |:---|:---|
        | Sampling interval dt (s) | $(Child("dt", NumberField(0.001:0.001:10.0; default=0.8))) |
        | Number of MFT periods | $(Child("nperiods", NumberField(2:1:500; default=100))) |
        | Wavelength reference velocity (km/s) | $(Child("wref", NumberField(0.5:0.5:5.0; default=2.0))) |
        | Wavelength fraction of distance | $(Child("wfrac", NumberField(0.05:0.01:1.0; default=0.33))) |
        | Velocity min (km/s) | $(Child("vmin", NumberField(0.1:0.1:20.0; default=1.0))) |
        | Velocity max (km/s) | $(Child("vmax", NumberField(0.1:0.1:20.0; default=8.0))) |
        | Phase velocity min (km/s) | $(Child("phase_vmin", NumberField(1.0:0.1:8.0; default=2.0))) |
        | Phase velocity max (km/s) | $(Child("phase_vmax", NumberField(1.0:0.1:8.0; default=5.0))) |
        | Phase method | $(Child("phase_method", Select(["branch resolver", "phtovel", "compare both"]; default="phtovel"))) |
        | Max modes | $(Child("max_modes", NumberField(1:1:20; default=6))) |
        | Bandwidth factor | $(Child("bandwidth", NumberField(0.1:0.05:1.0; default=0.15))) |
        | Zero-pad factor | $(Child("zero_pad", NumberField(1:1:32; default=2))) |
        | Upsample factor | $(Child("upsample", NumberField(1:1:10; default=2))) |
        | Numeric precision | $(Child("precision", Select(["Float32", "Float64"]; default="Float32"))) |
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

# ╔═╡ 1f2f5740-5bd0-11f1-0005-4b3f11e7a001
begin
    dt = Float64(_mft_plot_controls.dt)
    analysis_settings = isempty(all_saved_runs) ? nothing : first(all_saved_runs).analysis_settings
    period_min = Float64(_setting(analysis_settings, :period_min, 3.0))
    period_max = Float64(_setting(analysis_settings, :period_max, 10.0))
    period_lo, period_hi = sort((period_min, period_max))
    mft_nperiods = Int(_mft_plot_controls.nperiods)
    mft_periods = collect(exp.(range(log(period_lo), log(period_hi); length=mft_nperiods)))
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
        """
    end
end

# ╔═╡ 2330530c-de1a-4a2c-8f5e-396f0b7f3f8a
selected_plot_pair = String(_pair_mode_controls_plot.plot_pair)

# ╔═╡ 1f2f5740-5bd0-11f1-000b-4b3f11e7a001
selected_pair_names = String.(collect(_pair_mode_controls.pairs))

# ╔═╡ 1f2f5740-5bd0-11f1-001a-4b3f11e7a001
_selected_pair_set = isempty(selected_pair_names) || selected_pair_names == ["None"] ?
    Set(pair_options) : Set(selected_pair_names);

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
    mft_batch_diagnostics = Ref(NamedTuple[])
end

# ╔═╡ 1f2f5740-5bd0-11f1-0030-4b3f11e7a001
_mft_precision_type(precision::AbstractString) = precision == "Float64" ? Float64 : Float32

# ╔═╡ e2e32a41-c83d-4b74-84aa-b2b32f142984
function _mft_filter_bank_for(periods::Vector{Float64}, npts_raw::Int;
        storage_mode::Symbol=:picks_only,
        n_waveforms::Int=1,
        cfg)
    isempty(periods) && return nothing
    key = (mft.MFTFilterBank, npts_raw, Tuple(periods), cfg.bandwidth_factor,
        cfg.zero_pad_factor, cfg.upsample_factor, cfg.velocity_range,
        cfg.precision_type, storage_mode, n_waveforms)
    get!(cfg.cache, key) do
        mft.MFTFilterBank(cfg.dt, npts_raw, periods;
            bandwidth_factor=cfg.bandwidth_factor,
            zero_pad_factor=cfg.zero_pad_factor,
            upsample_factor=cfg.upsample_factor,
            velocity_range=cfg.velocity_range,
            precision=cfg.precision_type,
            storage_mode=storage_mode,
            N_initial=n_waveforms)
    end
end

# ╔═╡ 7ea8fe72-e5ff-499b-91bc-ae29aa4b6f11
function _mft_shared_periods_for_distances(distances::AbstractVector{<:Real}, cfg)
    periods = Float64[]
    for period in cfg.periods
        if any(distance -> mft.wavelength_valid_period(period, Float64(distance);
                wavelength_ref_velocity=cfg.wavelength_ref_velocity,
                wavelength_fraction=cfg.wavelength_fraction), distances)
            push!(periods, Float64(period))
        end
    end
    periods
end

# ╔═╡ 101d1048-8a10-4a96-b51a-bd007f007a68
function _mask_multimodal_dispersion(mode, valid::AbstractVector{Bool})
    inds = findall(valid[1:length(mode.periods)])
    mft.MultimodalDispersion(
        Float64.(mode.periods[inds]),
        Float64.(mode.arrival_times[inds]),
        Float64.(mode.group_velocities[inds]),
        Float64.(mode.peak_amplitudes[inds]),
        mode.mode_index)
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

# ╔═╡ 1f2f5740-5bd0-11f1-0034-4b3f11e7a001
function _has_nonzero_signal(x)
    any(v -> isfinite(v) && !iszero(v), x)
end

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

# ╔═╡ 95ccc2a8-315e-468f-817b-081626355394
function _safe_filename(s::AbstractString)
    cleaned = replace(strip(String(s)), r"[^A-Za-z0-9_.=-]+" => "_")
    isempty(cleaned) ? "none" : cleaned
end

# ╔═╡ 67ab9973-c221-44f9-8e3f-caa818efd692
function _pair_distance_for_title(pair_label::String)
    for collection in (run_source_state_averages, globalavg_source_state_averages)
        idx = findfirst(item -> String(item.pair_label) == pair_label, collection)
        isnothing(idx) || return Float64(collection[idx].distance)
    end
    NaN
end

# ╔═╡ 5029d9e6-d8c2-4991-aca9-3f5a177d2448
function _selected_score_spec(scored_by_pair, pair_label::String, selection::String)
    selection == "None" && return nothing
    specs = get(scored_by_pair, pair_label, NamedTuple[])
    idx = findfirst(s -> startswith(selection, "$(s.display) | $(s.branch)"), specs)
    isnothing(idx) ? nothing : specs[idx]
end

# ╔═╡ cd4099f0-b104-4ea5-abc6-0786cf0d448d
function _default_uc_pdf_path(pair_label::String, selected_spec)
    selected_label = isnothing(selected_spec) ? "none" :
        "$(selected_spec.display)_$(selected_spec.branch)"
    joinpath(@__DIR__, "figures",
        "v11_uc_consistency_$(_safe_filename(pair_label))_$(_safe_filename(selected_label)).pdf")
end

# ╔═╡ e0583b3a-f6f9-4a48-9f15-e286cca88a56
function _rows_for_pair_and_branch(rows, pair_label::String, branch::String)
    sort([r for r in rows
        if hasproperty(r, :pair_label) &&
           String(r.pair_label) == pair_label &&
           hasproperty(r, :branch) &&
           String(r.branch) == branch], by=r -> Float64(r.period))
end

# ╔═╡ c9d47fbb-89d3-4370-9b4e-4fe8ae22e25e
function _series_xy(rows, field::Symbol)
    xs = Float64[]
    ys = Float64[]
    for row in rows
        x = Float64(row.period)
        y = Float64(getproperty(row, field))
        if isfinite(x) && x > 0.0 && isfinite(y) && y > 0.0
            push!(xs, x)
            push!(ys, y)
        end
    end
    xs, ys
end

# ╔═╡ 3115ff7f-ae1c-403a-aec8-64c46b0c1a42
function _series_err_xy(rows; error_cap::Real=0.5)
    xs = Float64[]
    ys = Float64[]
    for row in rows
        x = Float64(row.period)
        y = Float64(row.relative_error)
        if isfinite(x) && x > 0.0 && isfinite(y) && y >= 0.0
            push!(xs, x)
            push!(ys, min(y, Float64(error_cap)))
        end
    end
    xs, ys
end

# ╔═╡ a79d5919-5dda-47ae-a7d1-73a38f210afb
function _add_makie_series!(ax, rows, field::Symbol;
        label::String, color, linestyle=:solid, marker=:circle,
        linewidth::Real=1.8, markersize::Real=7.0,
        legend_plots=nothing, legend_labels=nothing)
    xs, ys = _series_xy(rows, field)
    isempty(xs) && return nothing
    line = Makie.lines!(ax, xs, ys; color, linewidth, linestyle)
    Makie.scatter!(ax, xs, ys; color, marker, markersize,
        strokecolor=:black, strokewidth=0.5)
    if !isnothing(legend_plots) && !isnothing(legend_labels)
        push!(legend_plots, line)
        push!(legend_labels, label)
    end
    line
end

# ╔═╡ 9af1e52f-38fb-4fcc-a5fc-95bf47954140
function _add_makie_error_series!(ax, rows;
        color, linestyle=:solid, marker=:circle,
        linewidth::Real=1.3, markersize::Real=5.5,
        error_cap::Real=0.5)
    xs, ys = _series_err_xy(rows; error_cap)
    isempty(xs) && return nothing
    Makie.lines!(ax, xs, ys; color, linewidth, linestyle)
    Makie.scatter!(ax, xs, ys; color, marker, markersize,
        strokecolor=:black, strokewidth=0.35)
end

# ╔═╡ de3a25df-c397-4a93-89e0-a19459332725
function _make_empty_uc_figure(pair_label::String, message::String; for_export::Bool=false)
    fig = Makie.Figure(size=for_export ? (1050, 520) : (1100, 560),
        backgroundcolor=:white)
    Makie.Label(fig[1, 1], "U-c Consistency | $(pair_label)";
        fontsize=18, font=:bold, tellwidth=false)
    Makie.Label(fig[2, 1], message;
        fontsize=12, color=:gray25, tellwidth=false)
    fig
end

# ╔═╡ b074c3a9-6b04-4d67-b9c4-9840ec65ee65
function _activate_interactive_makie!()
    if _glmakie_available
        try
            GLMakie.activate!(render_on_demand=true, fxaa=true)
            return :GLMakie
        catch err
            @warn "GLMakie activation failed; using CairoMakie for inline plots." exception=(err, catch_backtrace())
        end
    end
    CairoMakie.activate!()
    :CairoMakie
end

# ╔═╡ a3c92cba-edcb-4b4f-bc55-60e260e5d3ce
function _write_uc_consistency_pdf(path::AbstractString, fig)
    isempty(strip(path)) && error("Set a PDF output path before writing.")
    mkpath(dirname(path))
    CairoMakie.activate!()
    Makie.save(path, fig; pdf_version="1.4")
    _activate_interactive_makie!()
    path
end

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


# ╔═╡ 642a8975-e2b6-4843-b8f2-a8d7f017f84b
function _branch_score_summary(rows, pair_label::String;
        method::AbstractString="geomean", huber_delta::Real=0.10)
    out = Dict{String,NamedTuple}()
    for branch in ("causal", "acausal")
        errs = Float64[
            Float64(r.relative_error) for r in rows
            if hasproperty(r, :pair_label) &&
               String(r.pair_label) == pair_label &&
               hasproperty(r, :branch) &&
               String(r.branch) == branch &&
               isfinite(Float64(r.relative_error)) &&
               Float64(r.relative_error) >= 0.0
        ]
        out[branch] = _uc_error_stats(errs; method, huber_delta)
    end
    out
end

# ╔═╡ 86480184-b295-4453-980a-a0a6ed8ecd16
function _selected_state_score_summary(specs, selected_spec)
    out = Dict{String,NamedTuple}(
        "causal" => _uc_error_stats(Float64[]),
        "acausal" => _uc_error_stats(Float64[]))
    isnothing(selected_spec) && return out
    for spec in specs
        same_state =
            String(spec.display) == String(selected_spec.display) &&
            String(spec.kind) == String(selected_spec.kind) &&
            string(spec.seed) == string(selected_spec.seed)
        same_state || continue
        haskey(out, String(spec.branch)) && (out[String(spec.branch)] = spec)
    end
    out
end

# ╔═╡ 1f2f5740-5bd0-11f1-003c-4b3f11e7a001
_format_uc_score_value(x; digits::Int=3) = isfinite(x) ? string(round(x; digits)) : "NA"

# ╔═╡ 2feae7c1-826d-405d-b98e-90a820186b4e
function _format_branch_scores(stats)
    "causal=$(_format_uc_score_value(stats["causal"].uc_score)), " *
    "acausal=$(_format_uc_score_value(stats["acausal"].uc_score))"
end

# ╔═╡ 8031b109-82ea-4ca6-aca7-99cc2b181b29
function _make_uc_consistency_figure(global_rows, overlay_rows, pair_label::String;
        selected_spec=nothing,
        selected_specs=NamedTuple[],
        velocity_range=nothing,
        score_method::AbstractString="geomean",
        huber_delta::Real=0.10,
        error_cap::Real=0.5,
        for_export::Bool=false)
    pair_global_rows = [r for r in global_rows
        if hasproperty(r, :pair_label) && String(r.pair_label) == pair_label]
    isempty(pair_global_rows) && return _make_empty_uc_figure(pair_label,
        "No finite global-average U-c rows for this pair. Include the pair in Selected pairs and rerun global_average_mft_analyses.";
        for_export)

    pair_overlay_rows = [r for r in overlay_rows
        if hasproperty(r, :pair_label) && String(r.pair_label) == pair_label]
    global_scores = _branch_score_summary(global_rows, pair_label;
        method=score_method, huber_delta)
    selected_scores = _selected_state_score_summary(selected_specs, selected_spec)
    distance = _pair_distance_for_title(pair_label)
    distance_label = isfinite(distance) ? "distance = $(round(distance; digits=1)) km" :
        "distance = unknown"
    selected_label = isnothing(selected_spec) ? "selected state = none" :
        "selected = $(selected_spec.display) [$(selected_spec.branch)]"
    title_text = "$(pair_label) | $(distance_label)"
    subtitle_text =
        "$(selected_label)\n" *
        "U-c $(score_method): global $(_format_branch_scores(global_scores)); " *
        "selected $(_format_branch_scores(selected_scores))"

    fig = Makie.Figure(size=for_export ? (1120, 760) : (1160, 780),
        backgroundcolor=:white)
    Makie.Label(fig[1, 1], title_text;
        fontsize=20, font=:bold, tellwidth=false)
    Makie.Label(fig[2, 1], subtitle_text;
        fontsize=12, color=:gray20, tellwidth=false)

    ax_vel = Makie.Axis(fig[3, 1];
        ylabel="Velocity (km/s)",
        xscale=log10,
        xticklabelsvisible=false,
        xlabelvisible=false,
        backgroundcolor=:white,
        xgridcolor=Makie.RGBAf(0, 0, 0, 0.10),
        ygridcolor=Makie.RGBAf(0, 0, 0, 0.10),
        spinewidth=1.0)
    ax_err = Makie.Axis(fig[4, 1];
        xlabel="Period (s)",
        ylabel="relative |U - U(c)| / U(c)",
        xscale=log10,
        backgroundcolor=:white,
        xgridcolor=Makie.RGBAf(0, 0, 0, 0.10),
        ygridcolor=Makie.RGBAf(0, 0, 0, 0.10),
        spinewidth=1.0)
    Makie.linkxaxes!(ax_vel, ax_err)
    Makie.ylims!(ax_err, 0.0, Float64(error_cap))
    isnothing(velocity_range) || Makie.ylims!(ax_vel,
        Float64(velocity_range[1]), Float64(velocity_range[2]))

    legend_plots = Any[]
    legend_labels = String[]
    global_style = Dict(
        "causal" => (; color=Makie.RGBAf(0.05, 0.32, 0.68, 1.0), marker=:circle),
        "acausal" => (; color=Makie.RGBAf(0.78, 0.18, 0.16, 1.0), marker=:diamond))
    selected_style = Dict(
        "causal" => (; color=Makie.RGBAf(0.08, 0.45, 0.25, 1.0), marker=:star5),
        "acausal" => (; color=Makie.RGBAf(0.45, 0.18, 0.62, 1.0), marker=:star5))

    for branch in ("causal", "acausal")
        rows_b = _rows_for_pair_and_branch(global_rows, pair_label, branch)
        style = global_style[branch]
        _add_makie_series!(ax_vel, rows_b, :group_velocity;
            label="global $(branch) U", color=style.color,
            marker=style.marker, linewidth=2.2, markersize=7.5,
            legend_plots, legend_labels)
        _add_makie_series!(ax_vel, rows_b, :predicted_group_velocity;
            label="global $(branch) U(c)", color=style.color,
            linestyle=:dash, marker=:circle, linewidth=1.8, markersize=4.6,
            legend_plots, legend_labels)
        _add_makie_series!(ax_vel, rows_b, :phase_velocity;
            label="global $(branch) c", color=style.color,
            linestyle=:dot, marker=:rect, linewidth=1.4, markersize=4.2,
            legend_plots, legend_labels)
        _add_makie_error_series!(ax_err, rows_b;
            color=style.color, marker=style.marker, linewidth=1.4,
            markersize=5.8, error_cap)
    end

    for branch in ("causal", "acausal")
        rows_b = _rows_for_pair_and_branch(pair_overlay_rows, pair_label, branch)
        style = selected_style[branch]
        _add_makie_series!(ax_vel, rows_b, :group_velocity;
            label="selected $(branch) U", color=style.color,
            marker=style.marker, linewidth=2.5, markersize=8.5,
            legend_plots, legend_labels)
        _add_makie_series!(ax_vel, rows_b, :predicted_group_velocity;
            label="selected $(branch) U(c)", color=style.color,
            linestyle=:dash, marker=:circle, linewidth=1.8, markersize=4.8,
            legend_plots, legend_labels)
        _add_makie_series!(ax_vel, rows_b, :phase_velocity;
            label="selected $(branch) c", color=style.color,
            linestyle=:dot, marker=:rect, linewidth=1.4, markersize=4.2,
            legend_plots, legend_labels)
        _add_makie_error_series!(ax_err, rows_b;
            color=style.color, linestyle=:dash, marker=style.marker,
            linewidth=1.3, markersize=5.8, error_cap)
    end

    if !isempty(legend_plots)
        Makie.Legend(fig[5, 1], legend_plots, legend_labels;
            orientation=:horizontal, nbanks=3, framevisible=false,
            labelsize=10, patchsize=(24, 10))
    end
    Makie.rowgap!(fig.layout, 8)
    fig
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

# ╔═╡ c45d58e7-ca37-4607-92cd-19bc806c359f
function _score_display_from_state_label(state_label::AbstractString, kind::AbstractString)
    m = match(r"^(.+) seed (.+) \| (.+)$", String(state_label))
    isnothing(m) && return (; display=String(state_label), seed=missing)
    seed = m.captures[2]
    suffix = m.captures[3]
    display_suffix = kind == "joint" ? "joint $(suffix)" : suffix
    (; display="seed$(seed) | $(display_suffix)", seed)
end

# ╔═╡ 2bd5208b-3d7b-4138-a11e-3612eb26fc9c
function _score_branch_analysis_result(res::mft.BranchAnalysisResult,
        pair_label::String, state_label::String, kind::String)
    parsed = _score_display_from_state_label(state_label, kind)
    scored = NamedTuple[]
    for (branch, branch_res) in (("causal", res.causal_result), ("acausal", res.acausal_result))
        score_label = "$(state_label) [$(branch)]"
        uc_rows = [merge(row, (; branch)) for row in _uc_rows_from_mft_result(branch_res, pair_label, score_label)]
        errs = Float64[r.relative_error for r in uc_rows if isfinite(r.relative_error) && r.relative_error >= 0.0]
        stats = _uc_error_stats(errs; method=ui_uc_score_method, huber_delta=ui_uc_huber_delta)
        push!(scored, merge(parsed, stats,
            (; label=score_label,
               pair_label,
               kind,
               branch,
               periods=Float64[r.period for r in uc_rows],
               group_velocities=Float64[r.group_velocity for r in uc_rows],
               uc_rows)))
    end
    scored
end

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

# ╔═╡ 1f2f5740-5bd0-11f1-e00a-4b3f11e7a001
function _vqvae_candidate_rows(wref::Float64, wfrac::Float64, selection::Dict)
    rows = NamedTuple[]
    for (pl, spec) in selection
        geom = _pair_geometry_for_export(pl)
        isnothing(geom) && continue
        spec_rows = hasproperty(spec, :uc_rows) ? spec.uc_rows :
            [(; period=Float64(spec.periods[ip]), group_velocity=Float64(spec.group_velocities[ip]))
                for ip in eachindex(spec.periods)]
        for row in spec_rows
            period = Float64(row.period)
            gv     = Float64(row.group_velocity)
            isfinite(period) && period > 0 || continue
            isfinite(gv)     && gv     > 0 || continue
            wref * period < wfrac * geom.distance || continue
            push!(rows, (; period, geom.lat1, geom.lon1, geom.lat2, geom.lon2, group_velocity=gv))
        end
    end
    sort(rows, by=r -> r.period)
end

# ╔═╡ 1f2f5740-5bd0-11f1-0030-4b3f11e7a002
function _mft_config(; dt::Real, periods, wavelength_ref_velocity::Real,
        wavelength_fraction::Real, velocity_range, bandwidth_factor::Real,
        zero_pad_factor::Integer, upsample_factor::Real, precision::AbstractString,
        max_modes::Integer, phase_velocity_range, use_phtovel::Bool,
        cache::Dict, diagnostics::Base.RefValue)
    (;
        dt=Float64(dt),
        periods=Float64.(periods),
        wavelength_ref_velocity=Float64(wavelength_ref_velocity),
        wavelength_fraction=Float64(wavelength_fraction),
        velocity_range=(Float64(velocity_range[1]), Float64(velocity_range[2])),
        bandwidth_factor=Float64(bandwidth_factor),
        zero_pad_factor=Int(zero_pad_factor),
        upsample_factor=Float64(upsample_factor),
        precision_type=_mft_precision_type(String(precision)),
        max_modes=Int(max_modes),
        phase_velocity_range=(Float64(phase_velocity_range[1]), Float64(phase_velocity_range[2])),
        use_phtovel,
        cache,
        diagnostics)
end

# ╔═╡ 1f2f5740-5bd0-11f1-0030-4b3f11e7a003
mft_config = _mft_config(; dt, periods=mft_periods,
    wavelength_ref_velocity=ui_wavelength_ref_velocity,
    wavelength_fraction=ui_wavelength_fraction,
    velocity_range, bandwidth_factor, zero_pad_factor,
    upsample_factor=ui_mft_upsample_factor,
    precision=ui_mft_precision,
    max_modes=mft_max_modes,
    phase_velocity_range,
    use_phtovel=ui_mft_use_phtovel,
    cache=mft_filter_banks,
    diagnostics=mft_batch_diagnostics);

# ╔═╡ 1f2f5740-5bd0-11f1-0030-4b3f11e7a004
function _mft_compute_periods(distance::Real, cfg)
    Float64[period for period in cfg.periods
        if mft.wavelength_valid_period(period, distance;
            wavelength_ref_velocity=cfg.wavelength_ref_velocity,
            wavelength_fraction=cfg.wavelength_fraction)]
end

# ╔═╡ 799a4631-139a-4629-84fa-92597c1606c6
PlutoUI.TableOfContents(include_definitions=true)

# ╔═╡ 69e9bb8b-9164-4de1-8da7-1e405dfed96d
function _valid_period_mask(periods, distance::Real, cfg)
    [mft.wavelength_valid_period(Float64(period), Float64(distance);
        wavelength_ref_velocity=cfg.wavelength_ref_velocity,
        wavelength_fraction=cfg.wavelength_fraction) for period in periods]
end

# ╔═╡ 951f2c00-5b30-41f3-aa77-98403ce2fdb2
function _mask_mft_result(res, distance::Real, cfg)
    valid = _valid_period_mask(res.periods, distance, cfg)
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

# ╔═╡ 7e3ae938-5d3f-423f-8acd-0ab664b6bb44
function _mask_branch_result(res, distance::Real, cfg)
    valid = _valid_period_mask(res.periods, distance, cfg)
    branch_correlation = copy(res.branch_correlation)
    branch_correlation[.!valid] .= NaN
    mft.BranchAnalysisResult(
        _mask_mft_result(res.causal_result, distance, cfg),
        _mask_mft_result(res.acausal_result, distance, cfg),
        [_mask_multimodal_dispersion(mode, valid) for mode in res.causal_modes],
        [_mask_multimodal_dispersion(mode, valid) for mode in res.acausal_modes],
        branch_correlation,
        res.periods,
        res.distance)
end

# ╔═╡ d51a4ab4-6ba1-4703-a212-eeed8c0170f2
function _mask_branch_batch(batch, distances::AbstractVector{<:Real}, cfg)
    masked_results = [_mask_branch_result(batch.state_results[i], Float64(distances[i]), cfg)
        for i in eachindex(batch.state_results)]
    branch_correlation = copy(batch.branch_correlation)
    for i in eachindex(masked_results)
        branch_correlation[:, i] .= masked_results[i].branch_correlation
    end
    mft.BranchBatchAnalysisResult(masked_results, branch_correlation,
        batch.periods, batch.state_labels)
end

# ╔═╡ 1f2f5740-5bd0-11f1-0034-4b3f11e7a002
function _analyze_branch_arrays(W_c::AbstractArray{<:Real}, W_ac::AbstractArray{<:Real},
        distances::AbstractVector{<:Real}, pair_keys::AbstractVector{<:AbstractString},
        state_labels::AbstractVector{<:AbstractString}, cfg;
        storage_mode::Symbol=:picks_only,
        compute_phase::Bool=false)
    ndims(W_c) >= 2 || throw(ArgumentError("W_c must have shape (nt × states...)"))
    size(W_c)[2:end] == size(W_ac)[2:end] ||
        throw(ArgumentError("W_c and W_ac trailing dimensions must match"))
    nstates = prod(size(W_c)[2:end])
    nstates == length(distances) == length(pair_keys) == length(state_labels) ||
        throw(ArgumentError("distances, pair_keys, and state_labels must match number of waveform states"))

    W_c_flat = reshape(W_c, size(W_c, 1), nstates)
    W_ac_flat = reshape(W_ac, size(W_ac, 1), nstates)
    n = min(size(W_c_flat, 1), size(W_ac_flat, 1))
    valid_period = [!isempty(_mft_compute_periods(distances[i], cfg)) for i in 1:nstates]
    nonzero_state = [_has_nonzero_signal(@view W_c_flat[1:n, i]) &&
        _has_nonzero_signal(@view W_ac_flat[1:n, i]) for i in 1:nstates]
    keep = [i for i in 1:nstates if valid_period[i] && nonzero_state[i]]
    analyses = Dict{String,Any}()
    if isempty(keep)
        cfg.diagnostics[] = [(;
            n_batches=0,
            n_input_states=nstates,
            n_specs=0,
            n_waveforms=0,
            n_samples=n,
            n_periods=0,
            storage_mode,
            n_zero_skipped=count(!, nonzero_state),
            n_invalid_period_skipped=count(!, valid_period),
            input_shape_c=size(W_c),
            input_shape_ac=size(W_ac))]
        return analyses
    end
    W_c_batch = @view W_c_flat[1:n, keep]
    W_ac_batch = @view W_ac_flat[1:n, keep]
    distances_batch = Float64.(distances[keep])
    pair_keys_batch = String.(pair_keys[keep])
    state_labels_batch = String.(state_labels[keep])
    periods = _mft_shared_periods_for_distances(distances_batch, cfg)
    isempty(periods) && return analyses

    bank = _mft_filter_bank_for(periods, n;
        storage_mode=storage_mode, n_waveforms=2 * length(keep), cfg)
    isnothing(bank) && return analyses
    batch = mft.analyze_causal_acausal_branches(
        W_c_batch, W_ac_batch, distances_batch, bank;
        state_labels=state_labels_batch,
        max_modes=cfg.max_modes,
        compute_phase=compute_phase,
        use_phtovel=cfg.use_phtovel,
        phase_velocity_range=cfg.phase_velocity_range)
    masked_batch = _mask_branch_batch(batch, distances_batch, cfg)
    cfg.diagnostics[] = [(;
        n_batches=1,
        n_input_states=nstates,
        n_specs=length(keep),
        n_waveforms=2 * length(keep),
        n_samples=n,
        n_periods=length(periods),
        storage_mode,
        n_zero_skipped=count(!, nonzero_state),
        n_invalid_period_skipped=count(!, valid_period),
        input_shape_c=size(W_c),
        input_shape_ac=size(W_ac))]
    merge!(analyses, _split_batch_by_pair(masked_batch, pair_keys_batch))
    analyses
end

# ╔═╡ 1f2f5740-5bd0-11f1-0035-4b3f11e7a001
global_average_mft_analyses = let
    cols_c = Vector{AbstractVector}()
    cols_ac = Vector{AbstractVector}()
    distances = Float64[]
    pair_keys = String[]
    state_labels = String[]
    for item in globalavg_source_state_averages
        gc = vec(item.global_avg_c)
        gac = vec(item.global_avg_ac)
        (isempty(gc) || isempty(gac)) && continue
        push!(cols_c, gc)
        push!(cols_ac, gac)
        push!(distances, Float64(item.distance))
        push!(pair_keys, String(item.pair_label))
        push!(state_labels, String(item.pair_label))
    end
    n = isempty(cols_c) ? 0 : minimum(min(length(cols_c[i]), length(cols_ac[i])) for i in eachindex(cols_c))
    pair_batches = n == 0 ? Dict{String,Any}() :
        _analyze_branch_arrays(
            reduce(hcat, [view(col, 1:n) for col in cols_c]),
            reduce(hcat, [view(col, 1:n) for col in cols_ac]),
            distances, pair_keys, state_labels, mft_config;
            compute_phase=true)
    Dict(pair_label => first(batch.state_results)
        for (pair_label, batch) in pair_batches if !isempty(batch.state_results))
end;

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

# ╔═╡ 1f2f5740-5bd0-11f1-e00f-4b3f11e7a001
let
    if Int(_export_controls.write_globalavg) == 0
        md"Press **Write global-avg** to export the global-average dispersion to pDSurfTomo format."
    else
        raw  = _globalavg_candidate_rows(ui_wavelength_ref_velocity, ui_wavelength_fraction)
        rows = _sigma_filter_rows(raw; nsigma=Float64(_export_controls.nsigma))
        out  = write_pdsurftomo_dispersion(String(_export_controls.globalavg_path), rows;
            include_count_header=Bool(_export_controls.header))
        md"Wrote **$(length(rows))** rows ($(length(raw)-length(rows)) removed as σ-outliers) to `$(out)`"
    end
end

# ╔═╡ 1f2f5740-5bd0-11f1-0040-4b3f11e7a001
_pair_mft_analyses_joint = let
    chunks_c = AbstractMatrix[]
    chunks_ac = AbstractMatrix[]
    distances = Float64[]
    pair_keys = String[]
    state_labels = String[]
    for item in run_source_state_averages
        nstates = min(size(item.causal, 2), size(item.acausal, 2))
        nstates == 0 && continue
        push!(chunks_c, @view item.causal[:, 1:nstates])
        push!(chunks_ac, @view item.acausal[:, 1:nstates])
        for i in 1:nstates
            label = i <= length(item.combo_labels) ? item.combo_labels[i] : string(i)
            push!(distances, Float64(item.distance))
            push!(pair_keys, String(item.pair_label))
            push!(state_labels, "$(item.pair_label) seed $(item.seed) | $(label)")
        end
    end
    n = isempty(chunks_c) ? 0 : minimum(min(size(chunks_c[i], 1), size(chunks_ac[i], 1)) for i in eachindex(chunks_c))
    n == 0 ? Dict{String,Any}() :
        _analyze_branch_arrays(
            reduce(hcat, [view(chunk, 1:n, :) for chunk in chunks_c]),
            reduce(hcat, [view(chunk, 1:n, :) for chunk in chunks_ac]),
            distances, pair_keys, state_labels, mft_config;
            compute_phase=true)
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0041-4b3f11e7a001
_pair_mft_analyses_marginal = let
    chunks_c = AbstractMatrix[]
    chunks_ac = AbstractMatrix[]
    distances = Float64[]
    pair_keys = String[]
    state_labels = String[]
    for item in run_source_state_averages
        K1 = isempty(item.marginal_stage1_ac) ? 0 : size(item.marginal_stage1_ac, 2)
        K2 = isempty(item.marginal_stage2_ac) ? 0 : size(item.marginal_stage2_ac, 2)
        if K1 > 0
            push!(chunks_c, @view item.marginal_stage1_c[:, 1:K1])
            push!(chunks_ac, @view item.marginal_stage1_ac[:, 1:K1])
        end
        for k in 1:K1
            lbl = k <= length(item.marginal_stage1_labels) ? item.marginal_stage1_labels[k] : "s1=$k"
            push!(distances, Float64(item.distance))
            push!(pair_keys, String(item.pair_label))
            push!(state_labels, "$(item.pair_label) seed $(item.seed) | S1 $(lbl)")
        end
        if K2 > 0
            push!(chunks_c, @view item.marginal_stage2_c[:, 1:K2])
            push!(chunks_ac, @view item.marginal_stage2_ac[:, 1:K2])
        end
        for k in 1:K2
            lbl = k <= length(item.marginal_stage2_labels) ? item.marginal_stage2_labels[k] : "s2=$k"
            push!(distances, Float64(item.distance))
            push!(pair_keys, String(item.pair_label))
            push!(state_labels, "$(item.pair_label) seed $(item.seed) | S2 $(lbl)")
        end
    end
    n = isempty(chunks_c) ? 0 : minimum(min(size(chunks_c[i], 1), size(chunks_ac[i], 1)) for i in eachindex(chunks_c))
    n == 0 ? Dict{String,Any}() :
        _analyze_branch_arrays(
            reduce(hcat, [view(chunk, 1:n, :) for chunk in chunks_c]),
            reduce(hcat, [view(chunk, 1:n, :) for chunk in chunks_ac]),
            distances, pair_keys, state_labels, mft_config;
            compute_phase=true)
end;

# ╔═╡ 5f365173-fec9-4758-aaf7-c184fe81a048
function _score_all_branches_from_mft_results()
    result = Dict{String,Vector{NamedTuple}}(pl => NamedTuple[] for pl in pair_labels)
    for (kind, analyses) in (("joint", _pair_mft_analyses_joint), ("marginal", _pair_mft_analyses_marginal))
        for (pair_label, batch) in analyses
            scored = get!(result, String(pair_label), NamedTuple[])
            for (res, label) in zip(batch.state_results, batch.state_labels)
                append!(scored, _score_branch_analysis_result(res, String(pair_label), String(label), kind))
            end
        end
    end
    for scored in values(result)
        sort!(scored, by=s -> (s.uc_score, s.max_relative_error, -s.n_valid, s.display, s.branch))
    end
    result
end

# ╔═╡ 1f2f5740-5bd0-11f1-e005-4b3f11e7a001
_all_scored_specs_by_pair = _score_all_branches_from_mft_results();

# ╔═╡ 1f2f5740-5bd0-11f1-e006-4b3f11e7a001
best_branch_per_pair = let
    result = Dict{String,NamedTuple}()
    for (pl, specs) in _all_scored_specs_by_pair
        isempty(specs) || (result[pl] = first(specs))
    end
    result
end;

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

# ╔═╡ 1f2f5740-5bd0-11f1-e010-4b3f11e7a001
let
    if Int(_export_controls.write_vqvae) == 0
        md"Press **Write VQVAE best-branch** to export the U-c–selected VQVAE dispersion to pDSurfTomo format."
    else
        raw  = _vqvae_candidate_rows(ui_wavelength_ref_velocity, ui_wavelength_fraction,
            resolved_export_selection)
        rows = _sigma_filter_rows(raw; nsigma=Float64(_export_controls.nsigma))
        out  = write_pdsurftomo_dispersion(String(_export_controls.vqvae_path), rows;
            include_count_header=Bool(_export_controls.header))
        n_pairs   = length(unique(zip([r.lat1 for r in rows], [r.lon1 for r in rows],
                                       [r.lat2 for r in rows], [r.lon2 for r in rows])))
        md"Wrote **$(length(rows))** rows from **$(n_pairs)** pairs ($(length(raw)-length(rows)) removed as σ-outliers) to `$(out)`"
    end
end

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

# ╔═╡ 3cb1b9b6-dc91-45fc-8fb1-7d0e0cde114e
selected_mft_overlay_spec = _selected_score_spec(
    _all_scored_specs_by_pair, selected_plot_pair, selected_marginal_combo_overlay);

# ╔═╡ 1f2f5740-5bd0-11f1-0046-4b3f11e7a001
# Overlay rows for the plot: look up the selected ranked (state, branch) spec
# and use the already-computed MFT summary stored in the score row.
selected_marginal_combo_overlay_rows = let
    if isnothing(selected_mft_overlay_spec)
        NamedTuple[]
    else
        spec = selected_mft_overlay_spec
        label = "$(spec.display) [$(spec.branch)] | score=$(_format_uc_score_value(spec.uc_score))"
        hasproperty(spec, :uc_rows) ?
            [merge(row, (; label)) for row in spec.uc_rows] :
            NamedTuple[]
    end
end;

# ╔═╡ 1f2f5740-5bd0-11f1-0047-4b3f11e7a001
selected_mft_waveform_combined_overlay_rows = selected_marginal_combo_overlay_rows;

# ╔═╡ ed13cf22-e978-4d1c-9ad7-ccbd77f13c34
begin
    _uc_pdf_default_path = _default_uc_pdf_path(selected_plot_pair, selected_mft_overlay_spec)
    @bind _uc_pdf_controls PlutoUI.combine() do Child
        md"""
        | Figure PDF export | Value |
        |:---|:---|
        | PDF path | $(Child("path", TextField(90; default=_uc_pdf_default_path))) |
        | Write PDF | $(Child("write_pdf", CounterButton("Write PDF"))) |
        """
    end
end

# ╔═╡ 1f2f5740-5bd0-11f1-0008-4b3f11e7a001
if @isdefined(selected_plot_pair)
    _activate_interactive_makie!()
    WideCell(_make_uc_consistency_figure(
        global_average_uc_rows,
        selected_mft_waveform_combined_overlay_rows,
        selected_plot_pair;
        selected_spec=selected_mft_overlay_spec,
        selected_specs=get(_all_scored_specs_by_pair, selected_plot_pair, NamedTuple[]),
        velocity_range=velocity_range,
        score_method=ui_uc_score_method,
        huber_delta=ui_uc_huber_delta))
else
    md""
end

# ╔═╡ bb996180-ff88-44c9-bd25-45d3d11d3500
let
    if Int(_uc_pdf_controls.write_pdf) == 0
        md"Press **Write PDF** to export the current Makie U-c consistency figure."
    else
        fig = _make_uc_consistency_figure(
            global_average_uc_rows,
            selected_mft_waveform_combined_overlay_rows,
            selected_plot_pair;
            selected_spec=selected_mft_overlay_spec,
            selected_specs=get(_all_scored_specs_by_pair, selected_plot_pair, NamedTuple[]),
            velocity_range=velocity_range,
            score_method=ui_uc_score_method,
            huber_delta=ui_uc_huber_delta,
            for_export=true)
        out = _write_uc_consistency_pdf(String(_uc_pdf_controls.path), fig)
        md"Wrote publication PDF to `$(out)`"
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
Peaks = "18e31ff7-3703-566c-8e60-38913d67486b"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
CSV = "~0.10.16"
CairoMakie = "~0.15.11"
ColorSchemes = "~3.31.0"
Colors = "~0.13.1"
DSP = "~0.8.5"
DataFrames = "~1.8.2"
FFTW = "~1.10.0"
GLMakie = "~0.13.11"
JLD2 = "~0.6.4"
Makie = "~0.24.11"
Peaks = "~0.6.2"
PlutoLinks = "~0.1.8"
PlutoUI = "~0.7.83"
ProgressLogging = "~0.1.6"
StatsBase = "~0.34.11"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "3deb016bb714183514631c82de5f96345ea09661"

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

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "28e1637322d4019ed2577cbec9268fab9b7da117"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.6.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "bca794632b8a9bbe159d56bf9e31c422671b35e0"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.3.2"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "8d8e0b0f350b8e1c91420b5e64e5de774c2f0f4d"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.16"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "1a063740329b7ee9ec602505c41cccb8500b637d"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.11"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

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

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "da54a6cd93c54950c15adf1d336cfd7d71f51a56"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.8.7"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

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

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "7bc84b769c1d384315e7b5c4ac03a6c303e6cf35"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.8"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoreMath]]
deps = ["CoreMath_jll"]
git-tree-sha1 = "8c0480f92b1b1796239156a1b9b1bfb1b39499b4"
uuid = "b7a15901-be09-4a0e-87d2-2e66b0e09b5a"
version = "0.1.0"

[[deps.CoreMath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a692a4c1dc59a4b8bc0b6403876eb3250fde2bc3"
uuid = "a38c48d9-6df1-5ac9-9223-b6ada3b5572b"
version = "0.1.0+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "d335b2929e1b6067951a1250df247cc5fab7d40e"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.5"
weakdeps = ["OffsetArrays"]

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

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

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "c55f5a9fd67bdbc8e089b5a3111fe4292986a8e8"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.6"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "e421c1938fafab0165b04dc1a9dbe2a26272952c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.125"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c307cd83373868391f3ac30b41530bc5d5d05d08"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.1+0"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "cac41ca6b2d399adfc95e51240566f8a60a80806"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.0+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

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

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

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

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW]]
deps = ["GLFW_jll"]
git-tree-sha1 = "af06f66cca2b698ab9c482de55977ff8178d025e"
uuid = "f7f18e0c-5ee9-5ccd-a5bf-e8befd85ed98"
version = "3.4.6"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GLMakie]]
deps = ["ColorTypes", "Colors", "FileIO", "FixedPointNumbers", "FreeTypeAbstraction", "GLFW", "GeometryBasics", "LinearAlgebra", "Makie", "Markdown", "MeshIO", "ModernGL", "Observables", "PrecompileTools", "Printf", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "a9964f9ef0b244352e04d30e8c215ec232eee699"
uuid = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
version = "0.13.11"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "1f5a80f4ed9f5a4aada88fc2db456e637676414b"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.10"

    [deps.GeometryBasics.extensions]
    GeometryBasicsGeoInterfaceExt = "GeoInterface"

    [deps.GeometryBasics.weakdeps]
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "93d5c27c8de51687a2c70ec0716e6e76f298416f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

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

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

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

[[deps.IntegerMathUtils]]
git-tree-sha1 = "4c1acff2dc6b6967e7e750633c50bc3b8d83e617"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.3"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "65d505fa4c0d7072990d659ef3fc086eb6da8208"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.2"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "CoreMath", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "921d7e91687e15a2c7c269c226960491fc041832"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.9"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "79d6bd28c8d9bccc2229784f1bd637689b256377"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.14"

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

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

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

    [deps.JLD2.weakdeps]
    UnPack = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "f76f7560267b840e492180f9899b472f30b88450"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "58927c485919bf17ea308d9d82156de1adf4b006"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.12"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

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
git-tree-sha1 = "86492c15dee3182863d7452dd2b2635242dedaca"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.5.2"

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

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "1cdb9a8ca42229d1ff880ce85e9b31ef2291bce3"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.11"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "aa1078778be5a8e5259ff04fbc3d258b3e78d464"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.9"

[[deps.MeshIO]]
deps = ["ColorTypes", "FileIO", "GeometryBasics", "Printf"]
git-tree-sha1 = "c009236e222df68e554c7ce5c720e4a33cc0c23f"
uuid = "7269a6da-0436-5bbc-96c2-40638cbb6118"
version = "0.5.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.ModernGL]]
deps = ["Libdl"]
git-tree-sha1 = "ac6cb1d8807a05cf1acc9680e09d2294f9d33956"
uuid = "66fc600b-dfda-50eb-8b99-91cfa97b1301"
version = "1.1.8"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3287ec88df50429a934ebc6cf14606215e27b987"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.33+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "9ac7c730c53b3b5d9a73fb900ac4b4fc263774db"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.4.9+0"

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

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "32b657a0d57c310a1a172bfc8c8cf68c5e674323"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.5"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

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

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

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

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

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
git-tree-sha1 = "624de6279ab7d94fc9f672f0068107eb6619732c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.3.2"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "InteractiveUtils", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "2f722c6581f297014be42c95d36f862e9bc2d668"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.14.5"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

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
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

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
git-tree-sha1 = "6547cbdd8ce32efba0d21c5a40fa96d1a3548f9f"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.8.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

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
git-tree-sha1 = "c6f18e5a52a176a383f6f6c635e0f81feed1d6d4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.11"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

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

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.TiffImages]]
deps = ["CodecZstd", "ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "9ca5f1f2d42f80df4b8c9f6ab5a64f438bbd9976"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.9"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "57e1b2c9de4bd6f40ecb9de4ac1797b81970d008"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.28.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "0716e01c3b40413de5dedbc9c5c69f27cddfddfc"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.3"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "ed349d26affcacafbc7fc2941ace1fb98f71e715"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+1"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

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

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╠═b8173cf5-5a3e-4078-8344-86f9d7a468eb
# ╠═1f2f5740-5bd0-11f1-0010-4b3f11e7a001
# ╠═8f91a8aa-e2a8-4b35-a3b3-6e81b850d0c1
# ╠═721a47ce-86ca-493f-ba3c-b18047cc8395
# ╠═0bfa7475-8c46-4cac-87a7-6741a3a12aae
# ╠═83975653-0038-4e91-b282-463dd5eb70c7
# ╠═a7b1a698-46d0-4a34-a2e8-f02cbafe26ff
# ╟─1f2f5740-5bd0-11f1-0001-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-0002-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-000c-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-0004-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-000d-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0003-4b3f11e7a001
# ╠═b074735f-2faf-4cc3-9497-f33b8fe211c6
# ╟─1f2f5740-5bd0-11f1-e00b-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-0009-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0011-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0012-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0013-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0014-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0015-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0016-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0017-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0018-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0005-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0019-4b3f11e7a001
# ╟─f3feca4e-36f9-4d82-a0c3-39234d240872
# ╠═4947c5cf-e1db-402e-8816-4e1cb0426802
# ╠═2330530c-de1a-4a2c-8f5e-396f0b7f3f8a
# ╠═1f2f5740-5bd0-11f1-000b-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-001a-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0022-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0023-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0024-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0025-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0027-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0028-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0029-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0030-4b3f11e7a001
# ╠═e2e32a41-c83d-4b74-84aa-b2b32f142984
# ╠═7ea8fe72-e5ff-499b-91bc-ae29aa4b6f11
# ╠═101d1048-8a10-4a96-b51a-bd007f007a68
# ╠═951f2c00-5b30-41f3-aa77-98403ce2fdb2
# ╠═7e3ae938-5d3f-423f-8acd-0ab664b6bb44
# ╠═d51a4ab4-6ba1-4703-a212-eeed8c0170f2
# ╠═1f2f5740-5bd0-11f1-0032-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0033-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0034-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0036-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0037-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0038-4b3f11e7a001
# ╠═95ccc2a8-315e-468f-817b-081626355394
# ╠═67ab9973-c221-44f9-8e3f-caa818efd692
# ╠═642a8975-e2b6-4843-b8f2-a8d7f017f84b
# ╠═86480184-b295-4453-980a-a0a6ed8ecd16
# ╠═5029d9e6-d8c2-4991-aca9-3f5a177d2448
# ╠═cd4099f0-b104-4ea5-abc6-0786cf0d448d
# ╠═e0583b3a-f6f9-4a48-9f15-e286cca88a56
# ╠═c9d47fbb-89d3-4370-9b4e-4fe8ae22e25e
# ╠═3115ff7f-ae1c-403a-aec8-64c46b0c1a42
# ╠═a79d5919-5dda-47ae-a7d1-73a38f210afb
# ╠═9af1e52f-38fb-4fcc-a5fc-95bf47954140
# ╠═2feae7c1-826d-405d-b98e-90a820186b4e
# ╠═de3a25df-c397-4a93-89e0-a19459332725
# ╠═8031b109-82ea-4ca6-aca7-99cc2b181b29
# ╠═a3c92cba-edcb-4b4f-bc55-60e260e5d3ce
# ╠═b074c3a9-6b04-4d67-b9c4-9840ec65ee65
# ╠═13d7824e-5ddc-11f1-a7cc-3f88810004cb
# ╠═13d78550-5ddc-11f1-a8f8-cf4f9a494ec6
# ╠═1f2f5740-5bd0-11f1-0026-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-003a-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-003b-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-003c-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e001-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e002-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e003-4b3f11e7a001
# ╠═c45d58e7-ca37-4607-92cd-19bc806c359f
# ╠═2bd5208b-3d7b-4138-a11e-3612eb26fc9c
# ╠═5f365173-fec9-4758-aaf7-c184fe81a048
# ╠═1f2f5740-5bd0-11f1-e005-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e006-4b3f11e7a001
# ╠═fd0e06f4-4e57-4982-a4cf-92ddbc67abcc
# ╠═36ce5571-c057-4a2b-a69a-c2393e36056d
# ╠═1f2f5740-5bd0-11f1-e008-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e009-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e00a-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e00c-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e00d-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e00e-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e00f-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-e010-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0030-4b3f11e7a002
# ╠═1f2f5740-5bd0-11f1-0030-4b3f11e7a003
# ╠═1f2f5740-5bd0-11f1-0030-4b3f11e7a004
# ╠═1f2f5740-5bd0-11f1-0034-4b3f11e7a002
# ╠═1f2f5740-5bd0-11f1-0035-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0039-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0040-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0041-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-0006-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0007-4b3f11e7a001
# ╠═3cb1b9b6-dc91-45fc-8fb1-7d0e0cde114e
# ╠═1f2f5740-5bd0-11f1-0046-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0047-4b3f11e7a001
# ╟─ed13cf22-e978-4d1c-9ad7-ccbd77f13c34
# ╠═1f2f5740-5bd0-11f1-0008-4b3f11e7a001
# ╠═bb996180-ff88-44c9-bd25-45d3d11d3500
# ╠═799a4631-139a-4629-84fa-92597c1606c6
# ╠═69e9bb8b-9164-4de1-8da7-1e405dfed96d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
