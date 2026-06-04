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
    using Statistics
end

# ╔═╡ 2f66e040-8d6d-4376-b97d-0169cbdc1efe
using FFTW, StatsBase, Peaks

# ╔═╡ 75e66ac1-57bf-468d-ab55-ada1d5e9ef91
using DSP

# ╔═╡ b00fd94f-291e-46d8-84ff-48f8606c2a1e
md"""
# Trained VQ-VAE Source-State MFT Tomography

CPU-only notebook for saved v8 source-state artifacts. It reads
`source_state_averages.jld2` files written by the training notebook, runs MFT for
each receiver pair, and scores geometry-aware tomography candidate mixes. It does
not load model weights, raw CCF data, or use GPU/Reactant.
"""

# ╔═╡ dcbf026e-957a-4b9b-9757-bd0638a25b26
begin
    saved_root = "/mnt/NAS2/Sanket_data/California_TO_with_latlong/SavedModels/vqvae_v8_K=[5, 3]"
# saved_root = 
	# "/mnt/NAS2/Sanket_data/California_XJ_13032026/SavedModels/vqvae_v8_K=[5, 3]"
# 
	# saved_root = 
	# 	"/mnt/NAS2/Sanket_data/California_2013_BK_CI_20032026/SavedModels/vqvae_v8"
end

# ╔═╡ f842f93e-16e8-4ec9-9c7f-c63d1f18c9f9
@bind reload_saved_artifacts_button CounterButton("Reload saved source-state artifacts")

# ╔═╡ b584dc56-8127-402c-98d0-1da3135f8f9a
md"""
## Appendix
"""

# ╔═╡ c7d47e38-e24f-4b40-b3a4-bc894188a750
mft = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/MFT.jl")

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

# ╔═╡ b350e7a5-ae7e-46ca-a246-d60b66a68e17
md"## Geometry-Aware Tomography Candidate Mixes"

# ╔═╡ cfcb4a13-2a2e-4f4d-a008-f864292d059e
function candidate_mix_table(mixes; n::Int=25)
    isempty(mixes) && return md"No tomography candidate mixes available."
    rows = ["| Rank | Pair / mix | Periods | Mean conf | Neighbor agreement | Score |",
            "|---:|---|---:|---:|---:|---:|"]
    for (rank, mix) in enumerate(mixes[1:min(n, length(mixes))])
        push!(rows, @sprintf("| %d | %s | %d | %.3f | %.3f | %.3f |",
                             rank, mix.label, mix.coverage_count,
                             mix.mean_confidence, mix.neighbor_agreement,
                             mix.total_score))
    end
    return Markdown.parse(join(rows, "\n"))
end

# ╔═╡ e89d81cb-8596-4364-8241-01578fb81c6b
md"## Quick Plots"

# ╔═╡ c3000001-0000-0000-0000-000000000001
md"## Codebook Waveform MFT"

# ╔═╡ b4c9a332-7e9f-4933-b12f-7c13f6d5f112
function _read_saved_analysis_settings(run_dir::String)
    path = joinpath(run_dir, "source_state_averages.jld2")
    d = load(path)
    haskey(d, "analysis_settings") ||
        error("Saved artifact is missing analysis_settings: $(path). Re-save this v8 run.")
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
    @bind selected_pair_names confirm(MultiCheckBox(pair_options; default=[]))
end

# ╔═╡ d8d742f8-b6d4-44e0-bf11-29ff4d22e117
_setting(settings, name::Symbol) =
    hasproperty(settings, name) ? getproperty(settings, name) :
    error("Saved analysis_settings is missing $(name). Re-save this v8 run.")

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
end

# ╔═╡ a4c73d31-cada-44dd-81c8-fbb0f5e84f6a
md"Selected **$(length(selected_runs))** trained runs across **$(length(unique([r.pair_label for r in selected_runs])))** receiver pairs."

# ╔═╡ ddfd42a8-4ae7-408a-8b8f-2d335746798b
run_source_state_averages = let
    reload_key = reload_saved_artifacts_button
    out = Vector{Any}(undef, length(selected_runs))
    Threads.@threads for i in eachindex(selected_runs)
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
end

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
end

# ╔═╡ c3000006-0000-0000-0000-000000000001
let
    item = selected_source_state_item
    if isnothing(item)
        md""
    else
        has_joint  = !isempty(item.codebook_joint_waves)
        has_stage2 = !isempty(item.codebook_stage2_waves)
        opts = ["stage1"]
        has_stage2  && push!(opts, "stage2")
        has_joint   && push!(opts, "joint (K1×K2)")
        @bind ui_codebook_mode Select(opts; default=last(opts))
    end
end

# ╔═╡ b2000001-0000-0000-0000-000000000001
if isnothing(selected_source_state_item)
    md""
else
    WideCell(mft.plot_cluster_histogram(
        selected_source_state_item.counts_ac,
        selected_source_state_item.counts_c;
        labels=selected_source_state_item.combo_labels,
        title="Source State Usage ($(selected_source_state_item.pair_label) seed=$(selected_source_state_item.seed))",
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
        title="State-State NCC ($(selected_source_state_item.pair_label) seed=$(selected_source_state_item.seed))",
    ))
end

# ╔═╡ eb76dfcf-b8ce-445d-a152-52eb8a6f94a7
analysis_settings = begin
    isempty(selected_runs) && error("Select at least one saved run.")
    settings = first(selected_runs).analysis_settings
    mismatched = [run.run_dir for run in selected_runs if run.analysis_settings != settings]
    isempty(mismatched) ||
        @warn "Selected runs have different saved analysis_settings; using the first selected run." first_run=first(selected_runs).run_dir mismatched_count=length(mismatched)
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
end

# ╔═╡ 067d2587-8eb1-41c3-95f8-9f785171f2ce
mft_periods = exp10.(range(log10(Float64(period_min)), log10(Float64(period_max)); length=mft_nperiods))

# ╔═╡ 61b6bba8-38c5-43a4-9d32-a95b1b7ccfd8
global_average_mft_analyses = let
    analyses = Dict{String,Any}()
    for pair_label in pair_labels
        items = [item for item in run_source_state_averages if item.pair_label == pair_label]
        isempty(items) && continue
        ref_item = first(items)
        global_c = _mean_global_branch(items, :global_avg_c)
        global_ac = _mean_global_branch(items, :global_avg_ac)
        (isempty(global_c) || isempty(global_ac)) && continue
        n = min(length(global_c), length(global_ac))
        c_trace = mft.SeismicTrace(data=global_c[1:n], dt=dt, distance=ref_item.distance)
        ac_trace = mft.SeismicTrace(data=global_ac[1:n], dt=dt, distance=ref_item.distance)
        analyses[pair_label] = mft.analyze_causal_acausal_branches(
            c_trace, ac_trace, mft_periods;
            max_modes=mft_max_modes,
            velocity_range=velocity_range,
            bandwidth_factor=bandwidth_factor,
            zero_pad_factor=zero_pad_factor,
        )
    end
    analyses
end

# ╔═╡ a55f4597-feda-4c04-ae53-75a082be08ed
global_average_mft_analyses[selected_plot_pair]

# ╔═╡ c61a6cfe-b14e-4aa9-a711-450e35a3a9bd
pair_mft_analyses = let
    analyses = Dict{String,Any}()
    for pair_label in pair_labels
        items = [item for item in run_source_state_averages if item.pair_label == pair_label]
        ac_traces = mft.SeismicTrace[]
        c_traces = mft.SeismicTrace[]
        labels = String[]
        for item in items
            nstates = size(item.acausal, 2)
            for i in 1:nstates
                push!(ac_traces, mft.SeismicTrace(data=vec(item.acausal[:, i]), dt=dt, distance=item.distance))
                push!(c_traces,  mft.SeismicTrace(data=vec(item.causal[:, i]),  dt=dt, distance=item.distance))
                label = i <= length(item.combo_labels) ? item.combo_labels[i] : string(i)
                push!(labels, "$(pair_label) seed $(item.seed) | $(label)")
            end
        end
        analyses[pair_label] = mft.analyze_causal_acausal_branches(
            c_traces, ac_traces, mft_periods;
            state_labels=labels,
            max_modes=mft_max_modes,
            velocity_range=velocity_range,
            bandwidth_factor=bandwidth_factor,
            zero_pad_factor=zero_pad_factor,
        )
    end
    analyses
end

# ╔═╡ 5064e2e2-1272-48ad-a417-02772071be86
WideCell(mft.plot_all_highcorr_groupvelocity_picks(
    pair_mft_analyses[selected_plot_pair];
    correlation_threshold=0.0,
    pair_and_average=true,
    title=string("Group Velocity Picks "),
    velocity_tolerance_fraction=0.1,
    reference_results=haskey(global_average_mft_analyses, selected_plot_pair) ? [global_average_mft_analyses[selected_plot_pair]] : mft.BranchAnalysisResult[],
    reference_labels=["Global avg"],
))

# ╔═╡ c04cb032-ff13-4dfe-9338-70c5ce785db2
 WideCell(mft.plot_branch_correlation(
        pair_mft_analyses[selected_plot_pair];
        title="MFT Branch Correlation",
        reference_results=haskey(global_average_mft_analyses, selected_plot_pair) ? [global_average_mft_analyses[selected_plot_pair]] : mft.BranchAnalysisResult[],
        reference_labels=["Global average"]))

# ╔═╡ a7c6d7af-cec7-4c4c-adb6-e2b370a49042
pair_consensus = Dict(pair_label => mft.consensus_group_velocity_picks(
    analysis;
    correlation_threshold=0.0,
    velocity_tolerance_fraction=0.10,
    cluster_tolerance_fraction=nothing,
    max_candidates=5,
    selection_mode=:low_velocity,
    min_candidate_periods=3,
    max_smooth_jump_fraction=0.08,
    max_gap_periods=1,
) for (pair_label, analysis) in pair_mft_analyses)

# ╔═╡ e3767110-37f7-4e37-a01f-93f72dcda465
if isempty(pair_labels) || !(selected_plot_pair in keys(pair_mft_analyses))
    md""
else
    WideCell(mft.plot_consensus_groupvelocity_picks(
        pair_mft_analyses[selected_plot_pair],
        pair_consensus[selected_plot_pair];
        correlation_threshold=0.0,
        velocity_tolerance_fraction=0.10,
        title="Trained VQ-VAE source-state consensus $(selected_plot_pair)",
    ))
end

# ╔═╡ ec938992-7a8a-45e0-b38e-4ba40bc7dfdc
md"Computed MFT consensus candidates for **$(length(pair_consensus))** receiver pairs."

# ╔═╡ b68a7252-510c-40c3-825a-d004a51a4cc4
tomography_pair_inputs = begin
    inputs = mft.PairConsensusForTomography[]
    for pair_label in sort(collect(keys(pair_consensus)))
        item = first([x for x in run_source_state_averages if x.pair_label == pair_label])
        if !isnothing(item.latitudes) && !isnothing(item.longitudes)
            push!(inputs, mft.tomography_pair_consensus(
                item.pair,
                pair_consensus[pair_label];
                latitudes=item.latitudes,
                longitudes=item.longitudes,
                distance=item.distance,
                label=pair_label,
            ))
        end
    end
    inputs
end

# ╔═╡ a38d32af-7c17-4f4e-ac6f-2bfcc5eb737e
tomography_candidate_mixes = mft.tomography_candidate_mixes(
    tomography_pair_inputs;
    max_mix_parts=3,
    min_candidate_periods=3,
    midpoint_radius_km=75.0,
    azimuth_tolerance_deg=25.0,
    distance_tolerance_fraction=0.35,
    velocity_tolerance_fraction=0.10,
)

# ╔═╡ d8609990-6852-40c4-81e5-59474f5ebd7c
candidate_mix_table(tomography_candidate_mixes; n=30)

# ╔═╡ b2000003-0000-0000-0000-000000000001
if haskey(pair_mft_analyses, selected_plot_pair)
    @bind ui_period_trained Slider(pair_mft_analyses[selected_plot_pair].periods;
        default=mean(pair_mft_analyses[selected_plot_pair].periods), show_value=true)
else
    md""
end

# ╔═╡ b2000004-0000-0000-0000-000000000001
if haskey(pair_mft_analyses, selected_plot_pair)
    WideCell(mft.plot_filtered_traces_by_period(
        pair_mft_analyses[selected_plot_pair];
        period=ui_period_trained,
        correlation_threshold=nothing,
        normalize_each=true,
        scale=0.7,
        spacing=2.2,
        title="MFT Filtered Traces ($(selected_plot_pair); period=$(round(ui_period_trained; digits=2))s)",
    ))
else
    md""
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

# ╔═╡ d1d675f4-ebea-4432-8d0e-ddeada2f5fa3
mft_analysis_test = let
	
ac = selected_source_state_item.acausal
c = selected_source_state_item.causal
dist=  selected_source_state_item.distance

    ac_traces = mft.SeismicTrace[]
    c_traces = mft.SeismicTrace[]
    labels = String[]



            nstates = size(ac, 2)
       
            for i in 1:nstates
                push!(ac_traces, mft.SeismicTrace(data=vec(ac[:, i]), dt=dt, distance=dist))
                push!(c_traces,  mft.SeismicTrace(data=vec(c[:, i]),  dt=dt, distance=dist))
                push!(labels, "seed $(i)")
            end
    


    mft_periods = exp10.(range(log10(Float64(period_min)), log10(Float64(period_max)); length=mft_nperiods))
    mft.analyze_causal_acausal_branches(
        c_traces, ac_traces, mft_periods;
        state_labels=labels,
		max_modes=6,
        velocity_range=velocity_range,
        bandwidth_factor=bandwidth_factor,
        zero_pad_factor=zero_pad_factor,
    )
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

    waves, labels = if mode == "joint (K1×K2)" && has_joint
        item.codebook_joint_waves, item.codebook_joint_labels
    elseif mode == "stage2" && has_stage2
        item.codebook_stage2_waves, item.codebook_stage2_labels
    else
        item.codebook_stage1_waves, item.codebook_stage1_labels
    end

    dist = item.distance
    traces = [mft.SeismicTrace(data=Float64.(waves[:, k]), dt=dt, distance=dist)
              for k in 1:length(labels)]

    global_ac = Float64.(vec(item.global_avg_ac))
    global_c  = Float64.(vec(item.global_avg_c))
    n_ref = min(length(global_ac), length(global_c))
    reference_result = mft.analyze_causal_acausal_branches(
        mft.SeismicTrace(data=global_c[1:n_ref], dt=dt, distance=dist),
        mft.SeismicTrace(data=global_ac[1:n_ref], dt=dt, distance=dist),
        mft_periods;
        max_modes=mft_max_modes,
        velocity_range=velocity_range,
        bandwidth_factor=bandwidth_factor,
        zero_pad_factor=zero_pad_factor,
    )

    batch_result = mft.analyze_causal_acausal_branches(
        traces, traces, mft_periods;
        state_labels=String.(labels),
        max_modes=mft_max_modes,
        velocity_range=velocity_range,
        bandwidth_factor=bandwidth_factor,
        zero_pad_factor=zero_pad_factor,
    )

    (; batch=batch_result, reference=reference_result,
       pair_label=item.pair_label, seed=item.seed, mode)
end

# ╔═╡ c3000005-0000-0000-0000-000000000001
WideCell(mft.plot_all_highcorr_groupvelocity_picks(
    codebook_mft_analysis.batch;
    correlation_threshold=0.0,
    pair_and_average=true,
    title="Codebook GV Picks ($(codebook_mft_analysis.pair_label) seed=$(codebook_mft_analysis.seed))",
    velocity_tolerance_fraction=0.1,
    reference_results=[codebook_mft_analysis.reference],
    reference_labels=["Global avg"],
))

# ╔═╡ c3000003-0000-0000-0000-000000000001
@bind ui_period_codebook Slider(codebook_mft_analysis.batch.periods;
    default=mean(codebook_mft_analysis.batch.periods), show_value=true)

# ╔═╡ c3000004-0000-0000-0000-000000000001
WideCell(mft.plot_filtered_traces_by_period(
    codebook_mft_analysis.batch;
    period=ui_period_codebook,
    correlation_threshold=nothing,
    normalize_each=true,
    scale=0.7,
    spacing=2.2,
    title="Codebook MFT Filtered Traces ($(codebook_mft_analysis.pair_label) seed=$(codebook_mft_analysis.seed); period=$(round(ui_period_codebook; digits=2))s)",
))

# ╔═╡ f06d6be1-7986-4d46-bb9a-dd634a6a44c5
WideCell(mft.plot_top_tomography_mixes(tomography_candidate_mixes, tomography_pair_inputs;
    n=20, period_min=period_min, period_max=period_max))

# ╔═╡ cc15f5d5-764d-4ee8-a8f2-862c5207c630
md"""
Using saved MFT settings:
`dt=$(dt)`, period band `$(period_min)-$(period_max)` s, `$(mft_nperiods)` periods,
velocity range `$(velocity_range)`, max modes `$(mft_max_modes)`, bandwidth factor `$(bandwidth_factor)`,
zero padding `$(zero_pad_factor)`.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Peaks = "18e31ff7-3703-566c-8e60-38913d67486b"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
ColorSchemes = "~3.31.0"
Colors = "~0.13.1"
DSP = "~0.8.5"
FFTW = "~1.10.0"
JLD2 = "~0.6.4"
Peaks = "~0.6.2"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.80"
StatsBase = "~0.34.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "0367d30bd0c338ea0caec6c494aeed63e505aaa8"

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

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

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
# ╟─f842f93e-16e8-4ec9-9c7f-c63d1f18c9f9
# ╟─e216a473-6433-4658-b2b7-a4eaa670cc5e
# ╟─a4c73d31-cada-44dd-81c8-fbb0f5e84f6a
# ╟─c3000006-0000-0000-0000-000000000001
# ╟─d7a8effd-1bc2-4f21-a947-7e8bbf82349a
# ╟─e3767110-37f7-4e37-a01f-93f72dcda465
# ╠═5064e2e2-1272-48ad-a417-02772071be86
# ╠═a55f4597-feda-4c04-ae53-75a082be08ed
# ╟─c04cb032-ff13-4dfe-9338-70c5ce785db2
# ╟─f30a7fe9-67f2-438e-9b9c-57e41b583c0e
# ╟─c3000005-0000-0000-0000-000000000001
# ╟─b584dc56-8127-402c-98d0-1da3135f8f9a
# ╠═a0f8a2b4-8fb5-4f06-bda6-c362a61065a1
# ╠═2f66e040-8d6d-4376-b97d-0169cbdc1efe
# ╠═75e66ac1-57bf-468d-ab55-ada1d5e9ef91
# ╠═c7d47e38-e24f-4b40-b3a4-bc894188a750
# ╟─d50d63be-d58b-4704-8211-ed7875e04857
# ╠═c7ec82e0-d5b6-4f31-8600-c8b1d276dc92
# ╠═e63099d0-5fb5-43b0-967c-b7c468dc4f83
# ╠═ed704922-12f0-475b-b628-19a9c37bca7a
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
# ╠═b01af348-c4b0-4ad1-81cd-116e9f2ed765
# ╠═c61a6cfe-b14e-4aa9-a711-450e35a3a9bd
# ╠═a7c6d7af-cec7-4c4c-adb6-e2b370a49042
# ╟─ec938992-7a8a-45e0-b38e-4ba40bc7dfdc
# ╟─b350e7a5-ae7e-46ca-a246-d60b66a68e17
# ╠═b68a7252-510c-40c3-825a-d004a51a4cc4
# ╠═a38d32af-7c17-4f4e-ac6f-2bfcc5eb737e
# ╠═cfcb4a13-2a2e-4f4d-a008-f864292d059e
# ╠═d8609990-6852-40c4-81e5-59474f5ebd7c
# ╟─e89d81cb-8596-4364-8241-01578fb81c6b
# ╠═50bcfca1-b35c-4af5-8da7-26e6a9aa7914
# ╠═dccacb14-c2de-43a2-943c-0e52a5e1276f
# ╠═c753f15f-12ab-4f54-92f2-59d96197f85c
# ╠═b2000001-0000-0000-0000-000000000001
# ╠═b2000002-0000-0000-0000-000000000001
# ╠═b2000003-0000-0000-0000-000000000001
# ╠═b2000004-0000-0000-0000-000000000001
# ╠═d1d675f4-ebea-4432-8d0e-ddeada2f5fa3
# ╟─c3000001-0000-0000-0000-000000000001
# ╠═c3000002-0000-0000-0000-000000000001
# ╟─c3000003-0000-0000-0000-000000000001
# ╠═c3000004-0000-0000-0000-000000000001
# ╠═f06d6be1-7986-4d46-bb9a-dd634a6a44c5
# ╠═b4c9a332-7e9f-4933-b12f-7c13f6d5f112
# ╠═d8d742f8-b6d4-44e0-bf11-29ff4d22e117
# ╠═eb76dfcf-b8ce-445d-a152-52eb8a6f94a7
# ╠═f01dc5e7-ae8a-4c31-a8df-64cabd2abe35
# ╠═cc15f5d5-764d-4ee8-a8f2-862c5207c630
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
