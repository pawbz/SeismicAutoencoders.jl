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

# ╔═╡ 1f2f5740-5bd0-11f1-0048-4b3f11e7a001
function _safe_filename(s::AbstractString)
    cleaned = replace(strip(String(s)), r"[^A-Za-z0-9_.=-]+" => "_")
    isempty(cleaned) ? "none" : cleaned
end

function _pair_distance_for_title(pair_label::String)
    for collection in (run_source_state_averages, globalavg_source_state_averages)
        idx = findfirst(item -> String(item.pair_label) == pair_label, collection)
        isnothing(idx) || return Float64(collection[idx].distance)
    end
    NaN
end

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

function _selected_score_spec(scored_by_pair, pair_label::String, selection::String)
    selection == "None" && return nothing
    specs = get(scored_by_pair, pair_label, NamedTuple[])
    idx = findfirst(s -> startswith(selection, "$(s.display) | $(s.branch)"), specs)
    isnothing(idx) ? nothing : specs[idx]
end

function _default_uc_pdf_path(pair_label::String, selected_spec)
    selected_label = isnothing(selected_spec) ? "none" :
        "$(selected_spec.display)_$(selected_spec.branch)"
    joinpath(@__DIR__, "figures",
        "v11_uc_consistency_$(_safe_filename(pair_label))_$(_safe_filename(selected_label)).pdf")
end

function _rows_for_pair_and_branch(rows, pair_label::String, branch::String)
    sort([r for r in rows
        if hasproperty(r, :pair_label) &&
           String(r.pair_label) == pair_label &&
           hasproperty(r, :branch) &&
           String(r.branch) == branch], by=r -> Float64(r.period))
end

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

function _format_branch_scores(stats)
    "causal=$(_format_uc_score_value(stats["causal"].uc_score)), " *
    "acausal=$(_format_uc_score_value(stats["acausal"].uc_score))"
end

function _make_empty_uc_figure(pair_label::String, message::String; for_export::Bool=false)
    fig = Makie.Figure(size=for_export ? (1050, 520) : (1100, 560),
        backgroundcolor=:white)
    Makie.Label(fig[1, 1], "U-c Consistency | $(pair_label)";
        fontsize=18, font=:bold, tellwidth=false)
    Makie.Label(fig[2, 1], message;
        fontsize=12, color=:gray25, tellwidth=false)
    fig
end

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

function _write_uc_consistency_pdf(path::AbstractString, fig)
    isempty(strip(path)) && error("Set a PDF output path before writing.")
    mkpath(dirname(path))
    CairoMakie.activate!()
    Makie.save(path, fig; pdf_version="1.4")
    GLMakie.activate!(render_on_demand=true, fxaa=true)
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


# ╔═╡ 1f2f5740-5bd0-11f1-003c-4b3f11e7a001
_format_uc_score_value(x; digits::Int=3) = isfinite(x) ? string(round(x; digits)) : "NA"

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
# and use the already-computed MFT summary stored in the score row.
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
            hasproperty(spec, :uc_rows) ?
                [merge(row, (; label)) for row in spec.uc_rows] :
                NamedTuple[]
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
        primary_score_label="global",
        uc_score_method=ui_uc_score_method,
        uc_huber_delta=ui_uc_huber_delta,
        title="Global Average U-c Consistency with Selected Source-State Overlay"))
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

# ╔═╡ 8f91a8aa-e2a8-4b35-a3b3-6e81b850d0c1
import Makie, GLMakie, CairoMakie

# ╔═╡ Cell order:
# ╟─1f2f5740-5bd0-11f1-0001-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-0002-4b3f11e7a001
# ╟─f3feca4e-36f9-4d82-a0c3-39234d240872
# ╟─1f2f5740-5bd0-11f1-000c-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-0004-4b3f11e7a001
# ╠═4947c5cf-e1db-402e-8816-4e1cb0426802
# ╟─1f2f5740-5bd0-11f1-0006-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0008-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-000b-4b3f11e7a001
# ╠═2330530c-de1a-4a2c-8f5e-396f0b7f3f8a
# ╠═1f2f5740-5bd0-11f1-000d-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0003-4b3f11e7a001
# ╠═b074735f-2faf-4cc3-9497-f33b8fe211c6
# ╠═1f2f5740-5bd0-11f1-0005-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0007-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-e00b-4b3f11e7a001
# ╟─1f2f5740-5bd0-11f1-0009-4b3f11e7a001
# ╠═b8173cf5-5a3e-4078-8344-86f9d7a468eb
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
# ╠═1f2f5740-5bd0-11f1-0022-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0023-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0028-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0024-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0025-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0026-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0027-4b3f11e7a001
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
# ╠═1f2f5740-5bd0-11f1-0035-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0036-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0037-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0038-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0039-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0040-4b3f11e7a001
# ╠═1f2f5740-5bd0-11f1-0041-4b3f11e7a001
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
# ╠═799a4631-139a-4629-84fa-92597c1606c6
# ╠═69e9bb8b-9164-4de1-8da7-1e405dfed96d
# ╠═8f91a8aa-e2a8-4b35-a3b3-6e81b850d0c1
