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

# ╔═╡ 10000001-0000-0000-0000-000000000001
begin
    using CUDA, cuDNN,
        Flux,
        Distances,
        JLD2,
        Random,
        MLUtils,
        DSP,
        ProgressLogging,
        Statistics,
        LinearAlgebra,
        PlutoLinks,
        PlutoUI,
        PlutoHooks,
        PlutoPlotly,
        FFTW,
        StatsBase,
        Optimisers,
        ParameterSchedulers,
        Functors
    CUDA.device!(0)
end

# ╔═╡ 02556dd0-4cb7-4251-a969-6bea09a41358
using Clustering

# ╔═╡ 9db29532-6d82-495c-bc3f-0daa882f5064
using ColorSchemes, Colors

# ╔═╡ 341dbed8-1d09-4b46-8434-eb332c332f75
using Peaks

# ╔═╡ c7f70869-8f84-4c33-a455-d79f78ac02ec
using Printf

# ╔═╡ da62431a-7cc6-4253-986d-5ba7d39e9f90
using Zygote

# ╔═╡ 53f17afb-91fb-4881-a9f4-9fa87a24fee6
using Enzyme

# ╔═╡ 418c15e5-8116-4d86-8c3e-aeac13cc3ef1
using BenchmarkTools

# ╔═╡ 53204e4f-16e5-4960-a451-b5660ea0f182
using NearestNeighbors

# ╔═╡ e85ac38a-e243-4233-8b75-1bcbc3884cb1
using NNlib

# ╔═╡ 10000004-0000-0000-0000-000000000001
md"""# VQ-VAE Training Notebook

Train a VQ-VAE model on a single station pair for clustering causal/acausal cross-correlation branches.
"""

# ╔═╡ 10000002-0000-0000-0000-000000000001
TableOfContents(include_definitions=true)

# ╔═╡ 10000003-0000-0000-0000-000000000001
xpu = gpu

# ╔═╡ 10000017-0000-0000-0000-000000000001
md"## Load VQ-VAE Architecture"

# ╔═╡ 10000018-0000-0000-0000-000000000001
vqvae = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/VQVAE_architecture_v6.jl")

# ╔═╡ 6d7cebf7-fb3e-4134-a428-91dea9f272b4
mft = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/MFT.jl")

# ╔═╡ 10000005-0000-0000-0000-000000000001
md"## Data Loading"

# ╔═╡ 10000007-0000-0000-0000-000000000001
dt = 1.0 # sanket

# ╔═╡ 10000006-0000-0000-0000-000000000001
begin
    period_min = 10
    period_max = 50
    responsetype = Bandpass(inv(period_max), inv(period_min))
    designmethod = Butterworth(2)
    digfilter = digitalfilter(responsetype, designmethod; fs=inv(dt))
end

# ╔═╡ 10000008-0000-0000-0000-000000000001
function taper(x)
    w = cat(tukey(size(x, 1), 0.1), dims=ndims(x))
    return w .* x
end

# ╔═╡ 10000009-0000-0000-0000-000000000001
# function get_acausal_causal(pair::String, filepath::String)
# 	jldfile = load(filter(x -> occursin(pair, x), readdir(filepath, join=true))[1])
# 		correlations = DSP.resample(jldfile["D"][1], 0.25, dims=1)
# 	headers = jldfile["headers"][1]
# 	distance = jldfile["Distances"][1]
# 	return (; correlations, headers, distance)
# end
function get_acausal_causal(pair::String, filepath::String)
    jldfile = load(filter(x -> occursin(pair, x), readdir(filepath, join=true))[1])
    correlations = jldfile["correlations"]
	# correlations = randn(size(correlations)...)
    headers = jldfile["headers"]
    distance = jldfile["dist"] # sanket
    latitudes = haskey(jldfile, "latitudes") ? Float64.(jldfile["latitudes"]) : nothing
    longitudes = haskey(jldfile, "longitudes") ? Float64.(jldfile["longitudes"]) : nothing
    station_pair = haskey(jldfile, "pairs") ? collect(jldfile["pairs"]) : nothing
    return (; correlations, headers, distance, latitudes, longitudes, station_pair)
end

# ╔═╡ 1000000a-0000-0000-0000-000000000001
function split_causal_acausal(X::AbstractMatrix, zero_lag::Bool, max_lag=nothing)
    nt, ntr = size(X)
    !isodd(nt) && error("nt should be odd")
    center = div(nt + 1, 2)
    half = div(nt - 1, 2)
    N = isnothing(max_lag) ? half : max(0, min(half, max_lag))
    if N == 0
        return similar(X, 0, ntr), similar(X, 0, ntr)
    end
    X_acausal = reverse(X[center-N:center-1, :], dims=1)
    X_causal = X[center+1:center+N, :]
    if zero_lag
        return vcat(zeros(1, size(X)[2:end]...), Array(X_acausal)),
        vcat(zeros(1, size(X)[2:end]...), Array(X_causal))
    else
        return Array(X_acausal), Array(X_causal)
    end
end

# ╔═╡ bbf3679e-9fbc-47b4-a1eb-03e48ee94a59
function build_training_bundle(pair;
    filepath="/mnt/NAS/Sanket_DRDO/station_pairs_12112025_30mins/")
    pair_name = join(pair, "_")
    data_pair_local = get_acausal_causal(pair_name, filepath)
    D1 = data_pair_local.correlations
    D1 = normalise(D1, dims=1)
    D1ac, D1c = split_causal_acausal(D1, true)
    D1ac = taper(D1ac)
    D1c = taper(D1c)
    D1fac = filtfilt(digfilter, D1ac)
    D1fc = filtfilt(digfilter, D1c)
    D1fac = Float32.(normalise(D1fac[2:end, :], dims=1))
    D1fc = Float32.(normalise(D1fc[2:end, :], dims=1))
    return (pair=pair, D1=Float32.(D1), D1fac=D1fac, D1fc=D1fc,
        distance=data_pair_local.distance,
        latitudes=data_pair_local.latitudes,
        longitudes=data_pair_local.longitudes,
        station_pair=data_pair_local.station_pair)
end

# ╔═╡ 8f596d58-2051-40f8-a52e-00afd3fe975d
function order_invariant_pair_geometry(bundle)
    isnothing(bundle.latitudes) && error("Missing latitudes for pair $(bundle.pair).")
    isnothing(bundle.longitudes) && error("Missing longitudes for pair $(bundle.pair).")
    length(bundle.latitudes) == 2 || error("Expected two latitudes for pair $(bundle.pair), got $(length(bundle.latitudes)).")
    length(bundle.longitudes) == 2 || error("Expected two longitudes for pair $(bundle.pair), got $(length(bundle.longitudes)).")

    lat1, lat2 = bundle.latitudes
    lon1, lon2 = bundle.longitudes
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2
    dx_km = 111.32 * cosd(mid_lat) * (lon2 - lon1)
    dy_km = 111.32 * (lat2 - lat1)
    axis_norm = hypot(dx_km, dy_km)
    if axis_norm <= eps(Float64)
        ux, uy = 1.0, 0.0
    else
        ux, uy = dx_km / axis_norm, dy_km / axis_norm
    end
    axis_cos2 = ux^2 - uy^2
    axis_sin2 = 2 * ux * uy
    return Float32[mid_lat, mid_lon, Float32(bundle.distance), axis_cos2, axis_sin2]
end

# ╔═╡ a7d54261-fa3b-4a07-bc32-ecebf682d0bf
function standardize_geometry_features(features::AbstractMatrix{<:Real})
    features32 = Float32.(features)
    μ = mean(features32; dims=2)
    σ = std(features32; dims=2, corrected=false)
    σ = max.(σ, 1f-6)
    standardized = (features32 .- μ) ./ σ
    return (; standardized=Float32.(standardized), mean=Float32.(μ), std=Float32.(σ))
end

# ╔═╡ 1000000d-0000-0000-0000-000000000001
md"### Select Station Pairs"

# ╔═╡ 1000000e-0000-0000-0000-000000000001
# data_filepath = "/mnt/NAS2/Sanket_data/California_2013_BK_CI_20032026/"
data_filepath = "/mnt/NAS2/Sanket_data/California_TO_with_latlong/"
# data_filepath = "/mnt/NAS2/Sanket_data/California_XJ_13032026/"

# ╔═╡ 10000010-0000-0000-0000-000000000001
md"### Train/Test Split (Pooled)"

# ╔═╡ 116cc111-1e02-44bb-bf5f-1c52e58e75ee
function make_pair_split(D1fac, D1fc, pair_id::Int; at=0.9, shuffle=true)
    D_all = hcat(D1fac, D1fc)
    nw = size(D_all, 2)
    idx = collect(1:nw)
    shuffle && Random.shuffle!(idx)
    ntrain = round(Int, at * nw)
    train_idx = idx[1:ntrain]
    test_idx = idx[ntrain+1:end]
    pair_ids_all = fill(Int32(pair_id), nw)
    return (
        D_train=xpu(D_all[:, train_idx]),
        D_test=xpu(D_all[:, test_idx]),
        pair_ids_train=pair_ids_all[train_idx],
        pair_ids_test=pair_ids_all[test_idx],
        D_all=xpu(D_all),
        pair_ids_all=pair_ids_all,
        D_ac_all=xpu(D1fac),
        D_c_all=xpu(D1fc),
        pair_ids_ac=fill(Int32(pair_id), size(D1fac, 2)),
        pair_ids_c=fill(Int32(pair_id), size(D1fc, 2)),
    )
end

# ╔═╡ a05d0dce-3c06-4b38-88bc-5202dfd851d9
function sample_columns(X, ids, ncols)
    size(X, 2) <= ncols && return X, ids
    idx = sort(Random.randperm(size(X, 2))[1:ncols])
    return X[:, idx], ids[idx]
end

# ╔═╡ c1c3dea0-4590-4c82-88b4-3b79ddcea7f1
function combine_pair_splits(pair_splits)
    train_counts = [size(ps.D_train, 2) for ps in pair_splits]
    test_counts = [size(ps.D_test, 2) for ps in pair_splits]
    ntrain_target = minimum(train_counts)
    ntest_target = minimum(test_counts)

    train_x = Any[]
    train_pair_ids = Int32[]
    test_x = Any[]
    test_pair_ids = Int32[]

    for ps in pair_splits
        xt, pit = sample_columns(ps.D_train, ps.pair_ids_train, ntrain_target)
        xv, piv = sample_columns(ps.D_test, ps.pair_ids_test, ntest_target)
        push!(train_x, xt)
        append!(train_pair_ids, pit)
        push!(test_x, xv)
        append!(test_pair_ids, piv)
    end

    return (
        train=(x=hcat(train_x...), pair_ids=train_pair_ids),
        test=(x=hcat(test_x...), pair_ids=test_pair_ids),
        ntrain_per_pair=ntrain_target,
        ntest_per_pair=ntest_target,
    )
end

# ╔═╡ 10000019-0000-0000-0000-000000000001
md"## Model Setup"

# ╔═╡ 1000001a-0000-0000-0000-000000000001
reload_network_button = @bind reload_network CounterButton("Reload Network")

# ╔═╡ 946fe8d3-9b9a-4d3e-9bb1-4e65f10bb3f0
conditioning_mode = :pair_id  # choose :geometry, :pair_id, or :none

# ╔═╡ 1000001e-0000-0000-0000-000000000001
md"""
## Training
"""

# ╔═╡ 10000020-0000-0000-0000-000000000002
md"""
### Multi-pair batch controls

For `v6`, the important training knobs are:

- `pairs_per_batch`: GPU-memory knob. Increase or decrease this depending on available memory.

The effective waveforms per optimizer step are:

`pairs_per_batch * per_pair_batchsize`

So the intended workflow is:
3. choose the largest `pairs_per_batch` that fits on the GPU

Inference should also be done pair-by-pair, or at least with enough same-pair waveforms in the batch for the pair-specific codebooks to stay training-like.
"""

# ╔═╡ 10000020-0000-0000-0000-000000000001
CUDA.pool_status()

# ╔═╡ 10000020-0000-0000-0000-000000000003
md"""
### Post-training cache controls

After training, the notebook can build encoded caches for **all selected pairs** and then switch analysis between pairs without re-encoding.

- `cache_pairs_per_batch`: GPU-memory knob for cache building
- larger values encode more pairs at once
- smaller values reduce memory and process the selected pairs in chunks
"""

# ╔═╡ 6558ecfe-42df-11f1-b755-35afb4d87b8a
md"""
### Portable analysis cache

`all_pair_encoded_cache` stores only encoded/source-state information per pair:

- coarse indices
- detail indices
- final code indices

It does **not** store raw waveforms.

The raw arrays used for plotting in this notebook live separately in `data`, `pair_splits`, and `data_cpu_cache`.
For saving to disk and loading in a separate CPU-only analysis notebook, use `analysis_cache_for_disk`, which keeps training metadata and encoded results but excludes raw data.
"""

# ╔═╡ 10000022-0000-0000-0000-000000000001
md"## Loss Curves"

# ╔═╡ a6000001-0000-0000-0000-000000000001
md"## V6 Ensemble Diagnostics"

# ╔═╡ 10000024-0000-0000-0000-000000000001
md"## Source State Analysis"

# ╔═╡ 10000025-0000-0000-0000-000000000001
md"""
## Analysis Moved

This training notebook now focuses on:

- selecting pairs and training the model
- building the portable encoded cache
- saving the run directory next to the raw data
"""

# ╔═╡ 10000036-0000-0000-0000-000000000001
md"### Encoded cache and analysis helpers"

# ╔═╡ eb127aaf-8f69-4d14-b132-811e60350e89
function cluster_averages_from_codes(x_cpu, ci; K::Int, per_position::Bool=false, time_index=nothing)
        nt = size(x_cpu, 1)
        if !isnothing(time_index)
            indices = vec(ci[time_index, :])
            out = zeros(Float32, nt, K)
            counts = zeros(Int, K)
            for j in 1:length(indices)
                k = indices[j]
                out[:, k] .+= x_cpu[:, j]
                counts[k] += 1
            end
            for k in 1:K
                counts[k] > 0 && (out[:, k] ./= counts[k])
            end
            return out
        elseif per_position
            out = zeros(Float32, nt, K)
            counts = zeros(Int, K)
            for j in 1:size(ci, 2), l in 1:size(ci, 1)
                k = ci[l, j]
                out[:, k] .+= x_cpu[:, j]
                counts[k] += 1
            end
            for k in 1:K
                counts[k] > 0 && (out[:, k] ./= counts[k])
            end
            return out
        else
            ncomb = K^size(ci, 1)
            out = zeros(Float32, nt, ncomb)
            counts = zeros(Int, ncomb)
            for j in 1:size(ci, 2)
                combo_idx = 1
                for t in 1:size(ci, 1)
                    combo_idx += (ci[t, j] - 1) * (K^(t - 1))
                end
                out[:, combo_idx] .+= x_cpu[:, j]
                counts[combo_idx] += 1
            end
            for k in 1:ncomb
                counts[k] > 0 && (out[:, k] ./= counts[k])
            end
            return out
        end
    end

# ╔═╡ 8d249a8d-4124-4c9d-bea4-64c940f88a32
function plot_cluster_average_matrix(avg; title::String, dt::Real=1.0, reverse_time::Bool=false)
        nt, nstates = size(avg)
        t = collect(1:nt) .* dt
        traces = AbstractTrace[]
        cs = ColorSchemes.rainbow
        for k in 1:nstates
            y = reverse_time ? reverse(avg[:, k]) : avg[:, k]
            color = Colors.hex(get(cs, (k - 1) / max(1, nstates - 1)))
            push!(traces, PlutoPlotly.scatter(
                x=t,
                y=y,
                mode="lines",
                name="coherent $(k)",
                line=attr(color=color, width=2),
            ))
        end
        layout = Layout(
            title=attr(text=title, font=attr(size=16, family="Computer Modern, serif")),
            xaxis=attr(title="Time lag (s)"),
            yaxis=attr(title="Mean amplitude"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=850,
            height=420,
            legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.2),
        )
        return WideCell(PlutoPlotly.plot(traces, layout))
    end

# ╔═╡ 1000002b-0000-0000-0000-000000000001
md"### Source State Averages"

# ╔═╡ c618590d-ca23-4901-b143-2f6482f32249
function safe_corr(a::AbstractVector, b::AbstractVector)
    a0 = a .- mean(a)
    b0 = b .- mean(b)
    dot(a0, b0) / (norm(a0) * norm(b0) + 1f-8)
end

# ╔═╡ e6e3dd04-924d-4688-a9a9-c411764719f3
corr_pool_threshold=0.9

# ╔═╡ 10000028-0000-0000-0000-000000000001
md"### Confusion Matrix"

# ╔═╡ 1000002a-0000-0000-0000-000000000001
md"The diagonal shows window-pairs where causal and acausal branches share the same code. Off-diagonal entries reveal branch-specific clustering."

# ╔═╡ 1000002f-0000-0000-0000-000000000001
md"""## Gather Plots

Select a source state to view all waveforms assigned to it.
"""

# ╔═╡ ed09ec19-f1b0-4819-9c9f-7ee796bd8f09
islot = 1

# ╔═╡ 60f24a45-507d-44cf-af59-00058418617c
combo_labels_plot = (12,12)

# ╔═╡ 10000032-0000-0000-0000-000000000001
md"""## Reconstruction Quality

Visualize a few reconstructions vs originals.
"""

# ╔═╡ 3da3c466-6d7f-4d8b-8ba2-1f7a66ff9a46
md"### Filtered reconstruction for selected state"

# ╔═╡ b8dfc2d6-43cb-4ed7-a32e-9df8a73d8a91
md"### Sub-cluster averages for selected state"

# ╔═╡ ab3f68f7-6179-4243-84d5-df8ea504c055
select_combo_button

# ╔═╡ 3d9ee2f7-a887-4e37-9cf0-552e72705791
md"### Correlation matrix of sub-cluster averages (selected state)"

# ╔═╡ 10b69e70-8c7f-44ce-85ab-44a48c4eac5e
md"### H(nuisance|coherent) / log(Knuisance) vs coherent code"

# ╔═╡ c1f2e3d4-a5b6-4c7d-8e9f-0a1b2c3d4e5f
md"### Mutual information disentanglement check (linear classifier)"

# ╔═╡ d8477392-2546-411e-baf0-302c03de50b0
md"## MFT"

# ╔═╡ 10000034-0000-0000-0000-000000000001
md"## Saving"

# ╔═╡ 10000035-0000-0000-0000-000000000001
# ╠═╡ disabled = true
#=╠═╡
begin
	using Dates
	timestamp = now()
	run_tag = Dates.format(timestamp, "yyyymmdd_HHMMSS")
	run_name = "vqvae_v6_run_" * run_tag
	run_dir = joinpath(data_filepath, run_name)
    mkpath(run_dir)
	jldsave(joinpath(run_dir, "model_state.jld2"),
		model_state = Flux.state(cpu(model)))
	jldsave(joinpath(run_dir, "vqvae_parameters.jld2");
		vqvae_parameters=vqvae_parameters,
        selected_pairs=selected_pairs,
        selected_pair_id=selected_pair_id)
	jldsave(joinpath(run_dir, "loss_history.jld2");
		loss_history)
    jldsave(joinpath(run_dir, "analysis_cache.jld2");
        analysis_cache=analysis_cache_for_disk)
end
  ╠═╡ =#

# ╔═╡ db4ddb38-2938-11f1-b8e3-e5227df9322c
md"## Appendix"

# ╔═╡ e928b8ab-e159-427a-b525-c6d60e6d6015
resample_button = @bind resample CounterButton("Resample Waveforms")

# ╔═╡ 3bf7a033-db24-4e6c-93a9-8706d2ea57c3
resample_button

# ╔═╡ 84c4a49c-c6fb-45b4-ac33-a5510cd6618e
resample_button

# ╔═╡ ed8e51a4-2eb7-447d-a1c4-c0b346005e79
resample_button

# ╔═╡ 6a3e970f-a7e7-4c67-a9e6-dc79be22c547
resample_button

# ╔═╡ b1f70fd0-1a7a-4f8f-9e7c-8e3e8cd3f5e1
function list_station_pairs(filepath::String)
    files = readdir(filepath)
    pairs = Set{Tuple{String,String}}()
    for f in files
        m = match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)", basename(f))
        m === nothing && continue
        push!(pairs, (m.captures[1], m.captures[2]))
    end
    return sort!(collect(pairs), by=x -> (x[1], x[2]))
end

# ╔═╡ f3b8c867-f8f9-45e0-a41f-4b7d7a1b17f0
available_pairs = list_station_pairs(data_filepath)

# ╔═╡ 0f84f2f6-6403-4a2e-9c42-6f8a84a2bc3f
md"Found **$(length(available_pairs))** station pairs in $(data_filepath)"

# ╔═╡ 2a8a4d12-c96b-4d64-8f58-98f54f81a77b
available_pairs

# ╔═╡ eaa7d770-d2d0-42f0-a65b-99447f11649a
begin
    if isempty(available_pairs)
        error("No station pairs found in $(data_filepath)")
    end
    pair_options = ["$(p[1])-$(p[2])" for p in available_pairs]
    default_pairs = pair_options[1:min(end, 4)]
    @bind selected_training_pair_names confirm(MultiCheckBox(pair_options, default=default_pairs))
end

# ╔═╡ 74d0f419-79d0-4a6a-ae65-18f6be9d64c8
selected_pairs = let
    isempty(selected_training_pair_names) && error("Select at least one receiver pair for training.")
    [begin
        parts = split(name, "-", limit=2)
        length(parts) == 2 || error("Invalid selected pair format: $(name)")
        (parts[1], parts[2])
    end for name in selected_training_pair_names]
end

# ╔═╡ c669f1d3-26c0-4028-b3d4-5a87bd696924
pair_bundles = [build_training_bundle(pair; filepath=data_filepath) for pair in selected_pairs]

# ╔═╡ 10000012-0000-0000-0000-000000000001
pair_splits = [make_pair_split(bundle.D1fac, bundle.D1fc, i) for (i, bundle) in enumerate(pair_bundles)]

# ╔═╡ 10000012-0000-0000-0000-000000000002
combined_data = combine_pair_splits(pair_splits)

# ╔═╡ 10000020-0000-0000-0000-000000000004
@bind cache_pairs_per_batch Slider(1:max(1, length(selected_pairs)), default=min(length(selected_pairs), 4), show_value=true)

# ╔═╡ 6558d732-42df-11f1-a447-1bdaa63b389c
begin
    selected_pair_name_ui = @bind selected_pair_name confirm(Select(selected_training_pair_names, default=first(selected_training_pair_names)))
end

# ╔═╡ 6558d7e6-42df-11f1-accb-c305628b5b77
selected_pair = let
    parts = split(selected_pair_name, "-", limit=2)
    length(parts) == 2 || error("Invalid selected pair format: $(selected_pair_name)")
    (parts[1], parts[2])
end

# ╔═╡ f86fec7f-f467-4411-80aa-c1621e3de063
data_bundle_pushkar = try
    build_training_bundle(selected_pair; filepath="/mnt/NAS2/Pushkar_Data/uttaranchal_data/jldfiles/30mins_dt_0p25_band_0p01_2p00_500maxlag/Z/")
catch
    nothing
end

# ╔═╡ 1000000f-0000-0000-0000-000000000001
data_bundle_cc = try
    build_training_bundle(selected_pair; filepath="/mnt/NAS2/Sanket_data/California_TO_with_latlong/")
catch
    nothing
end

# ╔═╡ 10000012-0000-0000-0000-000000000003
selected_pair_id = findfirst(==(selected_pair), selected_pairs)

# ╔═╡ 10000012-0000-0000-0000-000000000004
data_bundle = pair_bundles[selected_pair_id]

# ╔═╡ 10000012-0000-0000-0000-000000000005
data = pair_splits[selected_pair_id]

# ╔═╡ 10000013-0000-0000-0000-000000000001
nth = size(data.D_train, 1)

# ╔═╡ 10000014-0000-0000-0000-000000000001
tgrid = collect(-nth:nth) .* dt

# ╔═╡ 10000021-0000-0000-0000-000000000001
training_para = let

    vqvae.VQVAE_Training_Para(
        nepoch=20,
        initial_learning_rate=0.001,  # ↑ from 0.001,
        # stop_on_recon_loss = 0.98,
        lr_decay=0.99,
        pairs_per_batch=1,
        per_pair_batchsize=512,
        sample_with_replacement=true,
        Mnn_schedule=[(1, 10, :median), (6, 10, :mean), (26, 10, :mean)],
        warmup_epochs=5,
        smoothing_window=max(1, nth ÷ 16),
        envelope_floor=0.1f0,
        index_refresh_every=1,
        latent_index_batch_size=256,
        latent_index_space=:z_metric_flat,
    )
end

# ╔═╡ a6000002-0000-0000-0000-000000000001
WideCell(vqvae.plot_envelope_comparison(combined_data.train.x;
    floor=training_para.envelope_floor))

# ╔═╡ 24ceed68-00a3-4d29-9025-89bcf2f9251c
selected_pair_name_ui

# ╔═╡ 66b2b56b-cb5e-4620-866f-903f774fdbe5
selected_pair_name_ui

# ╔═╡ d5639a0d-9b87-45f3-b5a2-4c3132482394
selected_pair_name_ui

# ╔═╡ 83946706-1d00-4794-af60-8c65979236f8
data_bundle.distance / 1.2

# ╔═╡ e8ae6df6-fa04-42b7-a6e5-b2ade4322995
selected_pairs |> typeof

# ╔═╡ 83946706-1d00-4794-af60-8c65979236f9
selected_pair_ids(n::Int) = fill(Int32(selected_pair_id), n)

# ╔═╡ 83946706-1d00-4794-af60-8c65979236fc
function slice_encoded_result(res, col_range)
    detail = isnothing(res.detail_codebook_indices) ? nothing : res.detail_codebook_indices[:, col_range]
    (; ci=res.codebook_indices[:, col_range],
        coarse=res.coarse_codebook_indices[:, col_range],
        detail=detail)
end

# ╔═╡ 83946706-1d00-4794-af60-8c65979236fd
function build_all_pair_encoded_cache(model, pair_splits; pairs_per_batch::Int=1)
    npairs = length(pair_splits)
    npairs == 0 && return NamedTuple[]
    pairs_per_batch = clamp(pairs_per_batch, 1, npairs)
    caches = Vector{Any}(undef, npairs)

    for start_idx in 1:pairs_per_batch:npairs
        active = start_idx:min(start_idx + pairs_per_batch - 1, npairs)

        ac_blocks = [pair_splits[pid].D_ac_all for pid in active]
        c_blocks = [pair_splits[pid].D_c_all for pid in active]
        ac_pair_ids = vcat([pair_splits[pid].pair_ids_ac for pid in active]...)
        c_pair_ids = vcat([pair_splits[pid].pair_ids_c for pid in active]...)

        res_ac = vqvae.encode(model, hcat(ac_blocks...), ac_pair_ids; training=false)
        res_c = vqvae.encode(model, hcat(c_blocks...), c_pair_ids; training=false)

        ac_start = 1
        c_start = 1
        for pid in active
            nac = size(pair_splits[pid].D_ac_all, 2)
            nc = size(pair_splits[pid].D_c_all, 2)
            ac_range = ac_start:(ac_start + nac - 1)
            c_range = c_start:(c_start + nc - 1)
            cache_ac = slice_encoded_result(res_ac, ac_range)
            cache_c = slice_encoded_result(res_c, c_range)
            caches[pid] = (;
                ci_ac=cache_ac.ci,
                ci_c=cache_c.ci,
                coarse_ac=cache_ac.coarse,
                coarse_c=cache_c.coarse,
                detail_ac=cache_ac.detail,
                detail_c=cache_c.detail,
            )
            ac_start += nac
            c_start += nc
        end
    end
    return caches
end

# ╔═╡ d6590132-7190-4759-8739-3c6f5e7d9969


# ╔═╡ 10000015-0000-0000-0000-000000000001
md"""
## Data Visualization
"""

# ╔═╡ 10000037-0000-0000-0000-000000000001
md"### Single-code diagnostics"

# ╔═╡ a2000003-0000-0000-0000-000000000001
begin
    selected_latent_pos = 1
    nothing
end

# ╔═╡ a2000010-0000-0000-0000-000000000001
# ╠═╡ disabled = true
#=╠═╡
slot_entropy_pooled = let
    trained
    vqvae.per_latent_slot_entropy(model, data.D_all, data.pair_ids_all)
end
  ╠═╡ =#

# ╔═╡ a2000011-0000-0000-0000-000000000001
# ╠═╡ disabled = true
#=╠═╡
slot_entropy_ac = let
    trained
    vqvae.per_latent_slot_entropy(model, data.D_ac_all, data.pair_ids_ac)
end
  ╠═╡ =#

# ╔═╡ a2000012-0000-0000-0000-000000000001
# ╠═╡ disabled = true
#=╠═╡
slot_entropy_c = let
    trained
    vqvae.per_latent_slot_entropy(model, data.D_c_all, data.pair_ids_c)
end
  ╠═╡ =#

# ╔═╡ a2000013-0000-0000-0000-000000000001
# ╠═╡ disabled = true
#=╠═╡
begin
    trained
    WideCell(vqvae.plot_per_latent_slot_entropy(slot_entropy_pooled.entropy_norm;
        title="Per-slot entropy (pooled) $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km"))
end
  ╠═╡ =#

# ╔═╡ a2000014-0000-0000-0000-000000000001
# ╠═╡ disabled = true
#=╠═╡
let
    trained
    slots = collect(1:length(slot_entropy_ac.entropy_norm))
    tr_ac = PlutoPlotly.scatter(
        x=slots, y=slot_entropy_ac.entropy_norm,
        mode="lines+markers", name="Acausal",
        line=attr(color="#1f77b4", width=2), marker=attr(size=6)
    )
    tr_c = PlutoPlotly.scatter(
        x=slots, y=slot_entropy_c.entropy_norm,
        mode="lines+markers", name="Causal",
        line=attr(color="#d62728", width=2), marker=attr(size=6)
    )
    layout = Layout(
        title=attr(text="Per-slot entropy by branch $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km"),
        width=900,
        height=360,
        xaxis=attr(title="Latent slot"),
        yaxis=attr(title="Entropy / log(K)", range=[0, 1.05]),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    WideCell(PlutoPlotly.plot([tr_ac, tr_c], layout))
end
  ╠═╡ =#

# ╔═╡ 0d061de8-14f2-49c9-8d43-1e6b87e9d785


# ╔═╡ 0d061de8-14f2-49c9-8d43-1e6b87e9d786


# ╔═╡ 0d061de8-14f2-49c9-8d43-1e6b87e9d787
using_geometry_conditioning = conditioning_mode === :geometry

# ╔═╡ 0d061de8-14f2-49c9-8d43-1e6b87e9d788
using_pair_id_conditioning = conditioning_mode === :pair_id

# ╔═╡ c669f1d3-26c0-4028-b3d4-5a87bd696925
geometry_raw = using_geometry_conditioning ?
    hcat([order_invariant_pair_geometry(bundle) for bundle in pair_bundles]...) :
    zeros(Float32, 5, length(pair_bundles))

# ╔═╡ c669f1d3-26c0-4028-b3d4-5a87bd696926
geometry_standardization = standardize_geometry_features(geometry_raw)

# ╔═╡ c669f1d3-26c0-4028-b3d4-5a87bd696927
geometry_features = geometry_standardization.standardized

# ╔═╡ 1000001b-0000-0000-0000-000000000001
vqvae_parameters = (;
    nt=nth,
    d=24,
    beta_commit=0.25f0,
    # enc_strides=[1, 1, 1, 1],
    # enc_kernels=[5, 5, 3, 3],
    dead_threshold=50,
    entropy_weight=0.1,
    reconstruction_loss=:l2,
    interstation_distance=nothing,
    dt=dt,
    velocity_range=(2, 6),
    arrival_mse_weight=10.0,

    # multiscale RVQ: pair-specific coherent codebooks + shared nuisance residuals
    use_multiscale_rvq=true,
    Kcoherent=5,
    Knuisance=256,
    detail_stages=0,
    ema_decay_small=0.999f0,
    ema_decay_large=0.99f0,
    num_pairs=length(selected_pairs),
    pair_codebook_mode=:pair_specific_coarse_shared_detail,
    pair_names=collect(selected_training_pair_names),

    # order-invariant receiver-geometry conditioning
    use_geometry_conditioning=using_geometry_conditioning,
    use_pair_id_conditioning=using_pair_id_conditioning,
    geometry_dim=using_geometry_conditioning ? size(geometry_features, 1) : 0,
    condition_dim=16,
    geometry_features=using_geometry_conditioning ? geometry_features : nothing,
)

# ╔═╡ 1000001c-0000-0000-0000-000000000001
vqvae_para = vqvae.VQVAE_Para(; vqvae_parameters...)

# ╔═╡ a0000101-0000-0000-0000-000000000001
# Velocity-range travel times for vertical lines on all lag-time plots
begin
    _vmin, _vmax = vqvae_para.velocity_range
    _dist = data_bundle.distance
    t_vmin = _dist / _vmin   # slowest velocity → latest arrival (s)
    t_vmax = _dist / _vmax   # fastest velocity → earliest arrival (s)

    """Return PlutoPlotly shape attrs for vertical lines at vmin/vmax arrivals.
    Pass `symmetric=true` for joined acausal+causal plots (adds negative-lag lines)."""
    function velocity_vlines(t_vmin, t_vmax; symmetric=false)
        lines = [
            attr(type="line", x0=t_vmax, x1=t_vmax, y0=0, y1=1, yref="paper",
                 line=attr(color="steelblue", dash="dash", width=1.5)),
            attr(type="line", x0=t_vmin, x1=t_vmin, y0=0, y1=1, yref="paper",
                 line=attr(color="tomato", dash="dash", width=1.5)),
        ]
        if symmetric
            append!(lines, [
                attr(type="line", x0=-t_vmax, x1=-t_vmax, y0=0, y1=1, yref="paper",
                     line=attr(color="steelblue", dash="dash", width=1.5)),
                attr(type="line", x0=-t_vmin, x1=-t_vmin, y0=0, y1=1, yref="paper",
                     line=attr(color="tomato", dash="dash", width=1.5)),
            ])
        end
        lines
    end
end

# ╔═╡ a2000004-0000-0000-0000-000000000001
begin
    @bind selected_code_for_decode Slider(1:vqvae_para.Kcoherent, show_value=true)
end

# ╔═╡ f2369548-2d88-11f1-a737-85bad04c89cb
model, loss_history = @use_memo([reload_network, vqvae_parameters]) do
    reload_network
    model, loss_history = vqvae.get_vqvae(vqvae_para)
    model, loss_history
end

# ╔═╡ 1a623368-a2fc-4b7f-8d01-68e36d04a891
model.pre_vq

# ╔═╡ 10000021-0000-0000-0000-000000000002
trained = @use_memo([]) do

    vqvae.update(model, loss_history,
        combined_data.train, combined_data.test,
        vqvae_para, training_para)

	   # vqvae.update(model, loss_history,
    #     data.D_train[:, :], data.D_test,
    #     vqvae_para, training_para)

    randn(), loss_history
end

# ╔═╡ 1000001d-0000-0000-0000-000000000001
 combo_count = let
    trained
    combo_labels = string.(1:vqvae_para.Kcoherent)
   length(combo_labels)
end;

# ╔═╡ 699d2c90-4a52-44de-a58e-215f07fa6028
begin
    trained
    WideCell(vqvae.plot_cluster_histogram(cross.pct_ac, cross.pct_c;
        title="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s Source State Usage",
        labels=cross.labels))
end

# ╔═╡ a2000001-0000-0000-0000-000000000001
per_position_diag = let
    trained
    nothing
end

# ╔═╡ a2000002-0000-0000-0000-000000000001
let
    trained
    if isnothing(per_position_diag)
        md"Per-position diagnostics are disabled for the single-code v6 architecture."
    else
        WideCell(vqvae.plot_per_position_snr_heatmap(per_position_diag.snr_db;
            title="Per-position Prototype SNR (dB) $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km"))
    end
end

# ╔═╡ 1ddfb196-7d6a-44bf-82fc-793a9a30fb9c
all_pair_encoded_cache = @use_memo([trained, cache_pairs_per_batch, selected_training_pair_names]) do
    trained

    build_all_pair_encoded_cache(model, pair_splits;
        pairs_per_batch=cache_pairs_per_batch)
end

# ╔═╡ 1ddfb196-7d6a-44bf-82fc-793a9a30fb9d
begin
    trained
    encoded_cache = all_pair_encoded_cache[selected_pair_id]

    data_cpu_cache = (;
        D_ac=cpu(data.D_ac_all),
        D_c=cpu(data.D_c_all),
        D_all=cpu(data.D_all),
    )

    coherent_codes_ac = isnothing(encoded_cache.coarse_ac) ? encoded_cache.ci_ac : encoded_cache.coarse_ac
    coherent_codes_c = isnothing(encoded_cache.coarse_c) ? encoded_cache.ci_c : encoded_cache.coarse_c


    cluster_avg_c = cluster_averages_from_codes(data_cpu_cache.D_c, coherent_codes_c;
        K=vqvae_para.Kcoherent,
        per_position=false)
    cluster_avg_ac = cluster_averages_from_codes(data_cpu_cache.D_ac, coherent_codes_ac;
        K=vqvae_para.Kcoherent,
        per_position=false)
end

# ╔═╡ eba0784e-4ab9-4774-9851-56538b579fa6
let
    trained
	combo_labels = string.(1:size(cluster_avg_ac, 2))
    labels = string.(combo_labels)
    n = length(labels)

    function norm_corr_matrix(A)
        # A: (nt, n); returns (n, n) normalized correlation matrix
        C = Matrix{Float32}(undef, n, n)
        cols = [begin v = vec(A[:, i]); v .- mean(v) end for i in 1:n]
        norms = [norm(c) + 1f-8 for c in cols]
        for i in 1:n, j in 1:n
            C[i, j] = dot(cols[i], cols[j]) / (norms[i] * norms[j])
        end
        C
    end

    C_ac = norm_corr_matrix(cluster_avg_ac)
    C_c  = norm_corr_matrix(cluster_avg_c)
    trace_ac = PlutoPlotly.heatmap(
        z=C_ac, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=attr(title="Corr", len=0.9, x=0.46),
        xaxis="x1", yaxis="y1",
    )
    trace_c = PlutoPlotly.heatmap(
        z=C_c, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=attr(title="Corr", len=0.9, x=1.01),
        xaxis="x2", yaxis="y2",
    )

    sz = max(350, n * 40)
    layout = Layout(
        title=attr(text="State–State Normalised Correlation — $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s",
            font=attr(size=16)),
        grid=attr(rows=1, columns=2, pattern="independent"),
        annotations=[
            attr(text="Acausal", x=0.22, xref="paper", y=1.05, yref="paper",
                 showarrow=false, font=attr(size=14)),
            attr(text="Causal",  x=0.78, xref="paper", y=1.05, yref="paper",
                 showarrow=false, font=attr(size=14)),
        ],
        xaxis=attr(title="State", tickangle=-45),
        yaxis=attr(title="State"),
        xaxis2=attr(title="State", tickangle=-45),
        yaxis2=attr(title="State"),
        width=900, height=sz + 80,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=attr(t=80, b=80, l=80, r=80),
    )
    WideCell(PlutoPlotly.plot([trace_ac, trace_c], layout))
end

# ╔═╡ e45e700d-4780-437f-8300-78398c10b927
encoded_cache

# ╔═╡ a2000006-0000-0000-0000-000000000001
begin
	state_tuple_for_model = combo_idx -> (combo_idx,)
	cluster_filter_kwargs = () -> (;)

    function select_state_indices_from_codes(ci::AbstractMatrix{<:Integer}, ks_tuple::Tuple;
        per_position::Bool=false, time_index::Int=1)
        if per_position
            k = ks_tuple[1]
            pos = clamp(time_index, 1, size(ci, 1))
            return findall(vec(ci[pos, :]) .== k)
        end
        mask = trues(size(ci, 2))
        Tlocal = min(length(ks_tuple), size(ci, 1))
        for t in 1:Tlocal
            mask .&= vec(ci[t, :]) .== ks_tuple[t]
        end
        return findall(mask)
    end

	    selected_indices_ac = combo_idx -> select_state_indices_from_codes(
	        isnothing(encoded_cache.coarse_ac) ? encoded_cache.ci_ac : encoded_cache.coarse_ac,
	        state_tuple_for_model(combo_idx);
	        per_position=false,
	        time_index=1,
	    )

	    selected_indices_c = combo_idx -> select_state_indices_from_codes(
	        isnothing(encoded_cache.coarse_c) ? encoded_cache.ci_c : encoded_cache.coarse_c,
	        state_tuple_for_model(combo_idx);
	        per_position=false,
	        time_index=1,
    )

    detail_keys_for_indices = function (detail_mat, selected_ids)
        if isnothing(detail_mat) || isempty(selected_ids)
            return String[]
        end
        [join(detail_mat[:, j], "-") for j in selected_ids]
    end

    group_detail_positions = function(detail_mat, selected_ids)
        grouped = Dict{String,Vector{Int}}()
        isnothing(detail_mat) && return grouped
        for (local_idx, data_idx) in enumerate(selected_ids)
            key = join(detail_mat[:, data_idx], "-")
            push!(get!(grouped, key, Int[]), local_idx)
        end
        return grouped
    end

    branch_mean_by_positions = function(data_cpu, selected_ids, local_positions)
        isempty(local_positions) && return zeros(Float32, nth)
        cols = selected_ids[local_positions]
        return Float32.(vec(mean(@view(data_cpu[:, cols]); dims=2)))
    end
end

# ╔═╡ 1000002e-0000-0000-0000-000000000001
let
	trained
    K = vqvae_para.Kcoherent
    # Time axes: acausal → negative lags (reversed), causal → positive lags
    t_neg = [-(nth - i + 1) * dt for i in 1:nth]  # -nth*dt … -dt
    t_pos = [i * dt for i in 1:nth]                # dt … nth*dt
    t_full = [t_neg; t_pos]

    # Global averages across all waveforms
    global_avg_ac = vec(mean(cpu(data.D_ac_all); dims=2))
    global_avg_c = vec(mean(cpu(data.D_c_all); dims=2))
    global_full = [reverse(global_avg_ac); global_avg_c]
    global_ac0 = global_avg_ac .- mean(global_avg_ac)
    global_c0 = global_avg_c .- mean(global_avg_c)
    global_ncc = dot(global_ac0, global_c0) / ((norm(global_ac0) * norm(global_c0)) + 1e-8)

    combo_labels_local = string.(1:size(cluster_avg_ac, 2))
    ncomb = length(combo_labels_local)
    traces = AbstractTrace[]
    begin
        nc = max(ncomb, 1)
        cs = ColorSchemes.rainbow
        colors = [Colors.hex(get(cs, (i - 1) / max(1, nc - 1))) for i in 1:nc]
    end

    # Per-cluster joined CCF: acausal (negative lags) + causal (positive lags)
    total_ac = size(data.D_ac_all, 2)
    total_c = size(data.D_c_all, 2)
    # compute vertical spacing from typical amplitude
    mean_ac = vec(mean(cluster_avg_ac; dims=1))
    mean_c = vec(mean(cluster_avg_c; dims=1))
    amp_peak = maximum(abs.(vcat(mean_ac, mean_c)))
    vertical_spacing = amp_peak * 2.5 + 1e-3

    for combo_idx in 1:ncomb
        c = colors[mod1(combo_idx, length(colors))]
        # Build per-state joined CCF (acausal reversed to align with causal)
        a = cluster_avg_ac[:, combo_idx]
        a_rev = reverse(cluster_avg_ac[:, combo_idx])
        b = cluster_avg_c[:, combo_idx]
        full_k = [a_rev; b]
        # normalized cross-correlation (zero-mean cosine-like similarity)
        a0 = a .- mean(a)
        b0 = b .- mean(b)
        ncc = dot(a0, b0) / ((norm(a0) * norm(b0)) + 1e-8)
        # Get percentage of windows used for averaging in each cluster
        sel_ac = selected_indices_ac(combo_idx)
        sel_c = selected_indices_c(combo_idx)
        pct_ac = 100 * length(sel_ac) / max(total_ac, 1)
        pct_c = 100 * length(sel_c) / max(total_c, 1)
        legend_label = "State $(combo_labels_local[combo_idx]) (ac: $(round(pct_ac; digits=1))%, c: $(round(pct_c; digits=1))%, corr=$(round(ncc; digits=3)))"
        offset = (combo_idx - 1) * vertical_spacing
        push!(traces, PlutoPlotly.scatter(x=t_full, y=full_k .+ offset, mode="lines",
            name=legend_label,
            line=attr(color=c, width=2)))
    end

    # Global mean overlay
    push!(traces, PlutoPlotly.scatter(x=t_full, y=global_full, mode="lines",
        name="Global mean (corr=$(round(global_ncc; digits=3)))",
        line=attr(color="black", width=2, dash="dot")))

    layout = Layout(
        title=attr(text="Source State Average Waveforms ($(selected_pair[1])-$(selected_pair[2])) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s",
            font=attr(size=18, family="Computer Modern, serif")),
        height=500, width=900,
        xaxis=attr(title="Lag (s)", zeroline=true, zerolinecolor="rgba(0,0,0,0.3)"),
        yaxis=attr(title="Amplitude"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=attr(x=0.5, xanchor="center", y=-0.2, orientation="h",
            font=attr(size=12, family="Computer Modern, serif")),
        shapes=velocity_vlines(t_vmin, t_vmax; symmetric=true),
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ e1f9f9d6-8d55-41a7-9f84-7b49ebf69f2e
let
    trained
    resample

    combo_idx = selected_combo isa Integer ? selected_combo : findfirst(x -> x == selected_combo, combo_labels)
    if combo_idx === nothing
        md"Could not resolve selected source state."
    # elseif !(model.quantizer isa vqvae.MultiscaleRVQQuantizer)
        # md"Sub-cluster view requires `use_multiscale_rvq=true`."
    else
        selected_ac = selected_indices_ac(combo_idx)
        selected_c = selected_indices_c(combo_idx)

        if isempty(selected_ac) && isempty(selected_c)
            md"No windows found for selected state $(combo_labels[combo_idx])."
        else
            counts = Dict{String, Int}()
            ac_groups = group_detail_positions(encoded_cache.detail_ac, selected_ac)
            c_groups = group_detail_positions(encoded_cache.detail_c, selected_c)
            for (k, ids) in ac_groups
                counts[k] = get(counts, k, 0) + length(ids)
            end
            for (k, ids) in c_groups
                counts[k] = get(counts, k, 0) + length(ids)
            end

            if isempty(counts)
                md"No detail-code sub-clusters found for selected state $(combo_labels[combo_idx])."
            else
                keys_sorted = sort(collect(keys(counts)); by=k -> counts[k], rev=true)
                top_n = min(8, length(keys_sorted))
                keys_top = keys_sorted[1:top_n]

                t_neg = [-(nth - i + 1) * dt for i in 1:nth]
                t_pos = [i * dt for i in 1:nth]
                t_full = [t_neg; t_pos]

                cs = ColorSchemes.rainbow
                colors = [Colors.hex(get(cs, (i - 1) / max(1, top_n - 1))) for i in 1:top_n]

                amp_seed = Float32[]
                for k in keys_top
                    ac_sel = get(ac_groups, k, Int[])
                    c_sel = get(c_groups, k, Int[])
                    if !isempty(ac_sel)
                        ac_avg = branch_mean_by_positions(data_cpu_cache.D_ac, selected_ac, ac_sel)
                        append!(amp_seed, abs.(ac_avg))
                    end
                    if !isempty(c_sel)
                        c_avg = branch_mean_by_positions(data_cpu_cache.D_c, selected_c, c_sel)
                        append!(amp_seed, abs.(c_avg))
                    end
                end
                spacing = (isempty(amp_seed) ? 1.0 : maximum(amp_seed) * 2.5) + 1e-3

                traces = AbstractTrace[]
                for (i, k) in enumerate(keys_top)
                    ac_sel = get(ac_groups, k, Int[])
                    c_sel = get(c_groups, k, Int[])

                    ac_avg = isempty(ac_sel) ? zeros(Float32, nth) : branch_mean_by_positions(data_cpu_cache.D_ac, selected_ac, ac_sel)
                    c_avg = isempty(c_sel) ? zeros(Float32, nth) : branch_mean_by_positions(data_cpu_cache.D_c, selected_c, c_sel)
                    joined = [reverse(ac_avg); c_avg]

                    label = "detail=$(k) | ac=$(length(ac_sel)), c=$(length(c_sel))"
                    offset = (i - 1) * spacing
                    push!(traces, PlutoPlotly.scatter(
                        x=t_full,
                        y=joined .+ offset,
                        mode="lines",
                        line=attr(color=colors[i], width=2),
                        name=label,
                        showlegend=true,
                    ))
                end

                layout = Layout(
                    title=attr(text="Top detail sub-cluster averages in state $(combo_labels[combo_idx]) — $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km",
                        font=attr(size=16, family="Computer Modern, serif")),
                    xaxis=attr(title="Lag (s)"),
                    yaxis=attr(title="Amplitude + offset"),
                    height=max(500, top_n * 80),
                    width=950,
                    legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.2, font=attr(size=10)),
                    plot_bgcolor="white", paper_bgcolor="white",
                    shapes=velocity_vlines(t_vmin, t_vmax; symmetric=true),
                )

                WideCell(PlutoPlotly.plot(traces, layout))
            end
        end
    end
end

# ╔═╡ 2faf3842-106c-43d8-96fd-0a7154b1a338
encoded_cache.detail_ac

# ╔═╡ 0cdffa4e-1a61-4dae-9d72-fb4a89e0c246
encoded_cache.detail_ac

# ╔═╡ f4f14bc5-e69d-4e98-b6ec-17efd4f80ea2
let
    trained
    resample

    combo_idx = selected_combo isa Integer ? selected_combo : findfirst(x -> x == selected_combo, combo_labels)
    if combo_idx === nothing
        md"Could not resolve selected source state."
    # elseif !(model.quantizer isa vqvae.MultiscaleRVQQuantizer)
        # md"Sub-cluster correlation view requires `use_multiscale_rvq=true`."
    else
        selected_ac = selected_indices_ac(combo_idx)
        selected_c = selected_indices_c(combo_idx)

        if isempty(selected_ac) && isempty(selected_c)
            md"No windows found for selected state $(combo_labels[combo_idx])."
        else
            counts = Dict{String, Int}()
            ac_groups = group_detail_positions(encoded_cache.detail_ac, selected_ac)
            c_groups = group_detail_positions(encoded_cache.detail_c, selected_c)
            for (k, ids) in ac_groups
                counts[k] = get(counts, k, 0) + length(ids)
            end
            for (k, ids) in c_groups
                counts[k] = get(counts, k, 0) + length(ids)
            end

            if isempty(counts)
                md"No detail-code sub-clusters found for selected state $(combo_labels[combo_idx])."
            else
                keys_sorted = sort(collect(keys(counts)); by=k -> counts[k], rev=true)
                top_n = min(8, length(keys_sorted))
                keys_top = keys_sorted[1:top_n]

                function build_avg_matrix(groups, selected_branch, data_cpu, keys_top)
                    traces = Matrix{Float32}(undef, nth, length(keys_top))
                    valid = falses(length(keys_top))
                    for (i, k) in enumerate(keys_top)
                        sel = get(groups, k, Int[])
                        if isempty(sel)
                            traces[:, i] .= 0f0
                        else
                            traces[:, i] .= branch_mean_by_positions(data_cpu, selected_branch, sel)
                            valid[i] = true
                        end
                    end
                    return traces, valid
                end

                function norm_corr_matrix(A)
                    n = size(A, 2)
                    C = Matrix{Float32}(undef, n, n)
                    cols = [begin
                        v = vec(A[:, i])
                        v .- mean(v)
                    end for i in 1:n]
                    norms = [norm(c) + 1f-8 for c in cols]
                    for i in 1:n, j in 1:n
                        C[i, j] = dot(cols[i], cols[j]) / (norms[i] * norms[j])
                    end
                    return C
                end

                A_ac, valid_ac = build_avg_matrix(ac_groups, selected_ac, data_cpu_cache.D_ac, keys_top)
                A_c, valid_c = build_avg_matrix(c_groups, selected_c, data_cpu_cache.D_c, keys_top)

                C_ac = norm_corr_matrix(A_ac)
                C_c = norm_corr_matrix(A_c)

                labels = ["$(k)" for k in keys_top]
                ann_ac = [valid_ac[i] ? "" : " (no ac)" for i in 1:length(keys_top)]
                ann_c = [valid_c[i] ? "" : " (no c)" for i in 1:length(keys_top)]
                labels_ac = [labels[i] * ann_ac[i] for i in 1:length(labels)]
                labels_c = [labels[i] * ann_c[i] for i in 1:length(labels)]

                trace_ac = PlutoPlotly.heatmap(
                    z=C_ac,
                    x=labels_ac,
                    y=labels_ac,
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                    zmid=0,
                    colorbar=attr(title="Corr", len=0.9, x=0.46),
                    xaxis="x1",
                    yaxis="y1",
                )

                trace_c = PlutoPlotly.heatmap(
                    z=C_c,
                    x=labels_c,
                    y=labels_c,
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                    zmid=0,
                    colorbar=attr(title="Corr", len=0.9, x=1.01),
                    xaxis="x2",
                    yaxis="y2",
                )

                sz = max(420, top_n * 70)
                layout = Layout(
                    title=attr(text="Sub-cluster average correlation in state $(combo_labels[combo_idx]) — $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km",
                        font=attr(size=16, family="Computer Modern, serif")),
                    grid=attr(rows=1, columns=2, pattern="independent"),
                    annotations=[
                        attr(text="Acausal", x=0.22, xref="paper", y=1.07, yref="paper", showarrow=false, font=attr(size=14)),
                        attr(text="Causal", x=0.78, xref="paper", y=1.07, yref="paper", showarrow=false, font=attr(size=14)),
                    ],
                    xaxis=attr(title="Sub-cluster", tickangle=-45),
                    yaxis=attr(title="Sub-cluster"),
                    xaxis2=attr(title="Sub-cluster", tickangle=-45),
                    yaxis2=attr(title="Sub-cluster"),
                    width=980,
                    height=sz,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=attr(t=95, b=110, l=80, r=80),
                )

                WideCell(PlutoPlotly.plot([trace_ac, trace_c], layout))
            end
        end
    end
end

# ╔═╡ 7d3aa945-56f2-4444-aeea-dac1734da1a0
let
    trained

    if isnothing(encoded_cache.detail_ac) || isnothing(encoded_cache.detail_c)
        md"Detail indices are unavailable. This plot requires multiscale RVQ."
    else
        Kcoherent = vqvae_para.Kcoherent
        Knuisance = vqvae_para.Knuisance
        denom = max(log(Float32(Knuisance)), 1f-8)

        coarse_mat_ac = isnothing(encoded_cache.coarse_ac) ? encoded_cache.ci_ac : encoded_cache.coarse_ac
        coarse_mat_c  = isnothing(encoded_cache.coarse_c)  ? encoded_cache.ci_c  : encoded_cache.coarse_c

        # Whole-waveform mode uses row 1; per-position users can switch to selected_latent_pos.
        row_ac = 1
        row_c = 1

        coarse_vec_ac = vec(coarse_mat_ac[row_ac, :])
        coarse_vec_c  = vec(coarse_mat_c[row_c, :])

        detail_ac = encoded_cache.detail_ac
        detail_c  = encoded_cache.detail_c

        function cond_entropy_ratio_by_coarse(coarse_vec, detail_mat, Kcoherent, denom)
            y = fill(NaN32, Kcoherent)
            for k in 1:Kcoherent
                sel = findall(coarse_vec .== k)
                if isempty(sel)
                    continue
                end

                counts = Dict{String, Int}()
                for j in sel
                    key = join(detail_mat[:, j], "-")
                    counts[key] = get(counts, key, 0) + 1
                end

                probs = Float32.(collect(values(counts)))
                probs ./= max(sum(probs), 1f-8)
                H = -sum(probs .* log.(probs .+ 1f-8))
                y[k] = H / denom
            end
            y
        end

        y_ac = cond_entropy_ratio_by_coarse(coarse_vec_ac, detail_ac, Kcoherent, denom)
        y_c  = cond_entropy_ratio_by_coarse(coarse_vec_c,  detail_c,  Kcoherent, denom)

        coarse_vec_pool = vcat(coarse_vec_ac, coarse_vec_c)
        detail_pool = hcat(detail_ac, detail_c)
        y_pool = cond_entropy_ratio_by_coarse(coarse_vec_pool, detail_pool, Kcoherent, denom)

        xk = collect(1:Kcoherent)

        tr_ac = PlutoPlotly.scatter(
            x=xk, y=y_ac,
            mode="lines+markers", name="Acausal",
            line=attr(color="#1f77b4", width=2),
            marker=attr(size=7)
        )
        tr_c = PlutoPlotly.scatter(
            x=xk, y=y_c,
            mode="lines+markers", name="Causal",
            line=attr(color="#d62728", width=2),
            marker=attr(size=7)
        )
        tr_pool = PlutoPlotly.scatter(
            x=xk, y=y_pool,
            mode="lines+markers", name="Pooled",
            line=attr(color="#2ca02c", width=2, dash="dot"),
            marker=attr(size=7)
        )

        layout = Layout(
            title=attr(text="H(nuisance | coherent=k) / log(Knuisance) vs coherent code — $(selected_pair[1])-$(selected_pair[2])"),
            xaxis=attr(title="k_coherent", dtick=1),
            yaxis=attr(title="Entropy ratio", rangemode="tozero"),
            width=950,
            height=430,
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.2)
        )

        WideCell(PlutoPlotly.plot([tr_ac, tr_c, tr_pool], layout))
    end
end

# ╔═╡ 976a3916-41e8-11f1-a0d8-61b7d62f95de
let
    trained

    if isnothing(encoded_cache.detail_ac) || isnothing(encoded_cache.detail_c)
        md"Detail indices unavailable. This check requires multiscale RVQ."
    else
        # Extract coarse and detail codes (using row 1 for whole-waveform mode)
        coarse_ac = vec(encoded_cache.coarse_ac[1, :])
        coarse_c  = vec(encoded_cache.coarse_c[1, :])
        
        detail_ac = encoded_cache.detail_ac
        detail_c  = encoded_cache.detail_c
        
        # Pool acausal and causal for pooled classifier
        coarse_pool = vcat(coarse_ac, coarse_c)
        detail_pool = hcat(detail_ac, detail_c)
        
        # Train linear classifiers for each branch
        @info "Training coarse→detail classifiers..."
        result_ac = vqvae.train_coarse_to_detail_classifier(coarse_ac, detail_ac; train_ratio=0.8, nepochs=100)
        result_c  = vqvae.train_coarse_to_detail_classifier(coarse_c, detail_c; train_ratio=0.8, nepochs=100)
        result_pool = vqvae.train_coarse_to_detail_classifier(coarse_pool, detail_pool; train_ratio=0.8, nepochs=100)
        
        # Pre-compute formatted values for markdown
        acc_ac = @sprintf "%.3f" result_ac.test_acc
        null_ac = @sprintf "%.3f" result_ac.null_baseline
        acc_c = @sprintf "%.3f" result_c.test_acc
        null_c = @sprintf "%.3f" result_c.null_baseline
        acc_pool = @sprintf "%.3f" result_pool.test_acc
        null_pool = @sprintf "%.3f" result_pool.null_baseline
        
        # Compute assessment
        margin_ac = result_ac.test_acc - result_ac.null_baseline
        margin_c = result_c.test_acc - result_c.null_baseline
        margin_pool = result_pool.test_acc - result_pool.null_baseline
        
        assessment_str = if all([margin_ac < 0.05, margin_c < 0.05, margin_pool < 0.05])
            "✓ **Good disentanglement** — coarse provides minimal info about detail"
        elseif all([margin_ac < 0.15, margin_c < 0.15, margin_pool < 0.15])
            "~ **Moderate disentanglement** — some coupling between coarse and detail"
        else
            "✗ **Entanglement detected** — coarse predicts detail well; consider hyperparameter tuning"
        end
        
        # Summary text
        summary = Markdown.parse("""
        ## Mutual Information Disentanglement Check

        **Interpretation:**
        - **Null baseline:** always guess most common detail combo
        - **Accuracy >> baseline:** coarse and detail are entangled (redundant information)
        - **Accuracy ~= baseline:** coarse and detail are well-disentangled

        **Results:**
        - **Acausal:**  test_acc = $acc_ac, null = $null_ac
        - **Causal:**   test_acc = $acc_c, null = $null_c
        - **Pooled:**   test_acc = $acc_pool, null = $null_pool

        **Assessment:** $assessment_str
        """)
        
        # Bar plot: accuracy vs null baseline
        branches = ["Acausal", "Causal", "Pooled"]
        test_accs = Float32[result_ac.test_acc, result_c.test_acc, result_pool.test_acc]
        null_bases = Float32[result_ac.null_baseline, result_c.null_baseline, result_pool.null_baseline]
        
        tr_test = PlutoPlotly.bar(
            x=branches, y=test_accs,
            name="Linear classifier (test)",
            marker=attr(color="#1f77b4"),
            text=string.(round.(test_accs; digits=3)),
            textposition="outside"
        )
        
        tr_null = PlutoPlotly.bar(
            x=branches, y=null_bases,
            name="Null baseline",
            marker=attr(color="#ff7f0e", opacity=0.6),
            text=string.(round.(null_bases; digits=3)),
            textposition="outside"
        )
        
        layout = Layout(
            title=attr(text="Coarse→Detail Prediction Accuracy (lower = better disentanglement)"),
            barmode="group",
            xaxis=attr(title="Branch"),
            yaxis=attr(title="Accuracy", range=[0, 1]),
            width=700,
            height=450,
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.2),
            margin=attr(b=100)
        )
        
        plot_mi = PlutoPlotly.plot([tr_test, tr_null], layout)
        
        vcat(summary, WideCell(plot_mi))
    end
end

# ╔═╡ 9b3dd9f5-15fa-4785-b650-85ecf5c5c6f7
let
    trained

    # Config
    use_coarse_codes = true
    min_support = 5

    # Build one label per waveform from code indices
    function combo_index_local(digits::AbstractVector{<:Integer}, K::Int)
        idx = 1
        for t in 1:length(digits)
            idx += (digits[t] - 1) * (K^(t - 1))
        end
        return idx
    end

    function labels_from_ci(ci::AbstractMatrix{<:Integer}, K::Int)
        N = size(ci, 2)
        y = Vector{Int}(undef, N)
        for j in 1:N
            y[j] = combo_index_local(vec(ci[:, j]), K)
        end
        return y
    end

    # Hungarian algorithm for minimum-cost assignment on square matrix
    function hungarian_min_cost(C::AbstractMatrix{<:Real})
        n, m = size(C)
        n == m || error("Hungarian matching requires a square cost matrix")
        N = n

        u = zeros(Float64, N + 1)
        v = zeros(Float64, N + 1)
        p = zeros(Int, N + 1)
        way = zeros(Int, N + 1)

        for i in 1:N
            p[1] = i
            j0 = 1
            minv = fill(Inf, N + 1)
            used = falses(N + 1)

            while true
                used[j0] = true
                i0 = p[j0]
                delta = Inf
                j1 = 1

                for j in 2:N+1
                    if !used[j]
                        cur = C[i0, j - 1] - u[i0 + 1] - v[j]
                        if cur < minv[j]
                            minv[j] = cur
                            way[j] = j0
                        end
                        if minv[j] < delta
                            delta = minv[j]
                            j1 = j
                        end
                    end
                end

                for j in 1:N+1
                    if used[j]
                        u[p[j] + 1] += delta
                        v[j] -= delta
                    else
                        minv[j] -= delta
                    end
                end

                j0 = j1
                p[j0] == 0 && break
            end

            while true
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                j0 == 1 && break
            end
        end

        assign = zeros(Int, N) # row i -> col assign[i]
        for j in 2:N+1
            assign[p[j]] = j - 1
        end
        return assign
    end

    # Safe waveform correlation
    function safe_corr(a::AbstractVector, b::AbstractVector)
        a0 = a .- mean(a)
        b0 = b .- mean(b)
        return dot(a0, b0) / (norm(a0) * norm(b0) + 1f-8)
    end

    # Choose which labels to evaluate
    ci_ac = if use_coarse_codes && !isnothing(encoded_cache.coarse_ac)
        encoded_cache.coarse_ac
    else
        encoded_cache.ci_ac
    end
    ci_c = if use_coarse_codes && !isnothing(encoded_cache.coarse_c)
        encoded_cache.coarse_c
    else
        encoded_cache.ci_c
    end

    Kbase = vqvae_para.Kcoherent
    y_ac = labels_from_ci(ci_ac, Kbase)
    y_c = labels_from_ci(ci_c, Kbase)

    # Pooled dataset and pooled labels (same ordering as y_ac then y_c)
    X = cpu(hcat(data.D_ac_all, data.D_c_all))
    y = vcat(y_ac, y_c)

    N = length(y)
    idx_odd = collect(1:2:N)
    idx_even = collect(2:2:N)

    Xo = X[:, idx_odd]
    Xe = X[:, idx_even]
    yo = y[idx_odd]
    ye = y[idx_even]

    # Active labels (support threshold)
    labels_all = sort(unique(vcat(yo, ye)))
    labels_keep = Int[]
    for k in labels_all
        co = count(==(k), yo)
        ce = count(==(k), ye)
        if max(co, ce) >= min_support
            push!(labels_keep, k)
        end
    end

    if isempty(labels_keep)
        md"No labels pass min_support=$(min_support)."
    else
        L = length(labels_keep)

        # Prototypes per split on kept labels
        P_odd = Matrix{Float32}(undef, size(X, 1), L)
        P_even = Matrix{Float32}(undef, size(X, 1), L)
        n_odd = zeros(Int, L)
        n_even = zeros(Int, L)

        for (i, k) in enumerate(labels_keep)
            sel_o = findall(==(k), yo)
            sel_e = findall(==(k), ye)
            n_odd[i] = length(sel_o)
            n_even[i] = length(sel_e)

            P_odd[:, i] .= n_odd[i] > 0 ? Float32.(vec(mean(Xo[:, sel_o]; dims=2))) : 0f0
            P_even[:, i] .= n_even[i] > 0 ? Float32.(vec(mean(Xe[:, sel_e]; dims=2))) : 0f0
        end

        # Similarity matrix S(i,j) = corr(proto_odd_i, proto_even_j)
        S = fill(-1f0, L, L)
        for i in 1:L, j in 1:L
            if n_odd[i] > 0 && n_even[j] > 0
                S[i, j] = safe_corr(view(P_odd, :, i), view(P_even, :, j))
            end
        end

        # Hungarian on cost = max(S) - S  (maximize similarity)
        smax = maximum(S)
        C = smax .- S
        match = hungarian_min_cost(C)  # odd index i -> even index match[i]

        # Agreement after relabeling even-half labels through Hungarian map
        map_even_to_odd = Dict{Int, Int}()
        for i in 1:L
            k_odd = labels_keep[i]
            k_even = labels_keep[match[i]]
            map_even_to_odd[k_even] = k_odd
        end

        ye_mapped = [get(map_even_to_odd, k, k) for k in ye]

        # Distribution agreement (1 - total variation distance)
        p_o = zeros(Float64, L)
        p_e = zeros(Float64, L)
        p_em = zeros(Float64, L)
        for (i, k) in enumerate(labels_keep)
            p_o[i] = count(==(k), yo) / max(length(yo), 1)
            p_e[i] = count(==(k), ye) / max(length(ye), 1)
            p_em[i] = count(==(k), ye_mapped) / max(length(ye_mapped), 1)
        end
        agreement_tv = 1.0 - 0.5 * sum(abs.(p_o .- p_em))

        matched_corrs = [S[i, match[i]] for i in 1:L if n_odd[i] > 0 && n_even[match[i]] > 0]
        mean_matched_corr = isempty(matched_corrs) ? NaN : mean(matched_corrs)

        labels_txt = string.(labels_keep)

        tr1 = PlutoPlotly.heatmap(
            z=S,
            x=labels_txt,
            y=labels_txt,
            colorscale="RdBu",
            zmin=-1, zmax=1, zmid=0,
            colorbar=attr(title="Corr"),
            xaxis="x1", yaxis="y1"
        )

        Sdiag = zeros(Float32, L, L)
        for i in 1:L
            Sdiag[i, i] = S[i, match[i]]
        end

        tr2 = PlutoPlotly.heatmap(
            z=Sdiag,
            x=labels_txt,
            y=labels_txt,
            colorscale="RdBu",
            zmin=-1, zmax=1, zmid=0,
            colorbar=attr(title="Corr"),
            xaxis="x2", yaxis="y2"
        )

        layout = Layout(
            title=attr(
                text="Split-half reproducibility (odd/even): agreement=$(round(agreement_tv; digits=3)), matched corr=$(round(mean_matched_corr; digits=3))"
            ),
            grid=attr(rows=1, columns=2, pattern="independent"),
            annotations=[
                attr(text="Prototype corr: odd vs even", x=0.22, xref="paper", y=1.08, yref="paper", showarrow=false),
                attr(text="Hungarian-matched diagonal", x=0.78, xref="paper", y=1.08, yref="paper", showarrow=false),
            ],
            xaxis=attr(title="Even-half labels"),
            yaxis=attr(title="Odd-half labels"),
            xaxis2=attr(title="Even-half labels (matched)"),
            yaxis2=attr(title="Odd-half labels"),
            width=1050,
            height=560,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=attr(t=90, b=100, l=80, r=80)
        )

        summary = md"""
        **Split-half summary**
        - Label space: $(use_coarse_codes ? "coarse" : "full")
        - Kept labels: $(L)
        - Agreement after Hungarian relabeling (TV-based): $(round(agreement_tv; digits=4))
        - Mean matched prototype correlation: $(round(mean_matched_corr; digits=4))
        """

        WideCell(PlutoPlotly.plot([tr1, tr2], layout))
    end
end

# ╔═╡ 3214034a-c6db-445d-b466-9ea422b7a94e
mft_analysis_all_states = let
    nstates = size(cluster_avg_ac, 2)

	global_avg_ac = vec(mean(cpu(data.D_ac_all); dims=2))
    global_avg_c = vec(mean(cpu(data.D_c_all); dims=2))
	
    ac_traces = [
        mft.SeismicTrace(
            data=vec(cluster_avg_ac[:, i]),
            dt=dt,
            distance=data_bundle.distance
        )
        for i in 1:nstates
    ]
	push!(ac_traces,  mft.SeismicTrace(
            data=global_avg_ac,
            dt=dt,
            distance=data_bundle.distance
        ))

    c_traces = [
        mft.SeismicTrace(
            data=vec(cluster_avg_c[:, i]),
            dt=dt,
            distance=data_bundle.distance
        )
        for i in 1:nstates
    ]

	push!(c_traces,  mft.SeismicTrace(
            data=global_avg_c,
            dt=dt,
            distance=data_bundle.distance
        ))

    labels = vcat(string.(1:vqvae_para.Kcoherent), "Full")

    mft.analyze_causal_acausal_branches(
        ac_traces,
        c_traces,
        state_labels=labels,
		period_max=80.0,
		velocity_range=(1.0, 8.0),
        bandwidth_factor=0.15,
	 zero_pad_factor=4,
    )
end

# ╔═╡ f0f2d04a-95e7-4126-adc4-f6f0124b0113
@bind ui_period Slider(mft_analysis_all_states.periods, default=10, show_value=true)

# ╔═╡ cff400dc-49c8-4d9a-ac7f-5178ab097dce
WideCell(
    mft.plot_filtered_traces_by_period(
        mft_analysis_all_states;
        period=ui_period,                   # or period_index=...
        correlation_threshold=nothing,      # e.g. 0.9 to keep only high-symmetry states
        normalize_each=true,
        scale=0.7,
        spacing=2.2,
        title="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s Filtered Traces by Source State"
    )
)

# ╔═╡ 48bc91de-f66c-4443-b61f-ff5eb45d23c5
WideCell(mft.plot_branch_correlation(mft_analysis_all_states; title="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s Branch Correlation Across Source States"))

# ╔═╡ d7efd06d-8f55-4717-a228-cccf2acfe30d
WideCell(
    mft.plot_filtered_traces_by_period(
        mft_analysis_all_states;
        period=ui_period,                   # or period_index=...
        correlation_threshold=nothing,      # e.g. 0.9 to keep only high-symmetry states
        normalize_each=true,
        scale=0.7,
        spacing=2.2,
        title="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s Filtered Traces by Source State"
    )
)

# ╔═╡ 3ab7d986-0597-4fac-b1b1-4c1a56a181a9
WideCell(mft.plot_all_highcorr_groupvelocity_picks(mft_analysis_all_states; correlation_threshold=0.9, title="Group Velocity Picks $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s"))

# ╔═╡ 240393ec-e1d5-4086-b8d8-a2b9e72078d8
WideCell(mft.plot_all_highcorr_groupvelocity_picks(mft_analysis_all_states; correlation_threshold=0.85, pair_and_average=true, title="Group Velocity Picks $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s"))

# ╔═╡ 964da977-5394-4369-9185-60797ad764e3
begin
    trained
    WideCell(vqvae.plot_training_dashboard(loss_history;
        title="VQ-VAE: $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s"))
end

# ╔═╡ a6000003-0000-0000-0000-000000000001
v6_diagnostic_index = @use_memo([trained, reload_network, training_para.Mnn_schedule, selected_training_pair_names]) do
    trained
    diagnostic_Mnn = vqvae.max_Mnn(training_para)
    idx = vqvae.LatentIndex(diagnostic_Mnn)
    vqvae.rebuild_latent_index!(idx, model, combined_data.train.x, combined_data.train.pair_ids;
        Mnn=diagnostic_Mnn,
        batch_size=training_para.latent_index_batch_size,
        latent_index_space=training_para.latent_index_space)
    idx
end

# ╔═╡ a6000004-0000-0000-0000-000000000001
WideCell(vqvae.plot_ensemble_target_examples(combined_data.train.x, v6_diagnostic_index;
    nsamples=5))

# ╔═╡ a6000005-0000-0000-0000-000000000001
WideCell(vqvae.plot_neighbor_examples(combined_data.train.x, v6_diagnostic_index;
    nsamples=5, nneighbors=min(5, vqvae.max_Mnn(training_para))))

# ╔═╡ c7a6235f-c023-4149-92a1-837902763836
let
    trained
    # Get codebook matrix
    E = vqvae.get_codebook(model)
    if ndims(E) == 3
        E = E[:, :, 1]
    end

    E = Array(E)  # ensure CPU Array

    K = size(E, 2)
    kmax = min(K, 20)
    Esel = E[:, 1:kmax]

    # axis labels
    xlabels = string.(1:kmax)
    ylabels = string.(1:size(Esel, 1))

    # Choose colormap; `Viridis` is readable
    cm = ColorSchemes.viridis

    # Build heatmap
    trace = PlutoPlotly.heatmap(
        z=Esel,
        x=xlabels,
        y=ylabels,
        # colorscale = [ [i, colorant"$(RGB(cm[i]))"] for i in range(0, stop=1, length=256) ],
        colorbar=attr(title="Embedding\nvalue", titleside="right"),
        zmid=0
    )

    layout = Layout(
        title=attr(text="Codebook Embedding Heatmap (first $kmax codes) — $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s", font=attr(size=16)),
        xaxis=attr(title="Code index"),
        yaxis=attr(title="Embedding dimension"),
        width=900,
        height=600,
        margin=attr(t=70, b=60, l=80, r=140)
	)


    WideCell(PlutoPlotly.plot([trace], layout))
end

# ╔═╡ b6000001-0000-0000-0000-000000000001
let
    trained
    resample
    nshow = 8
    target_Mnn = vqvae.max_Mnn(training_para)
    X = data.D_all
    idx = vqvae.LatentIndex(target_Mnn)
    vqvae.rebuild_latent_index!(idx, model, X, data.pair_ids_all;
        Mnn=target_Mnn,
        batch_size=training_para.latent_index_batch_size,
        latent_index_space=training_para.latent_index_space)

    sample_ids = sort(Random.randperm(size(X, 2))[1:min(nshow, size(X, 2))])
    targets = vqvae.build_ensemble_targets(X, idx, sample_ids;
        Mnn=target_Mnn,
        aggregation=:mean)
    x_orig = cpu(X[:, sample_ids])
    t = collect(1:nth) .* dt

    traces = AbstractTrace[]
    for (j, sample_id) in enumerate(sample_ids)
        offset = (j - 1) * 4
        push!(traces, PlutoPlotly.scatter(x=t, y=x_orig[:, j] .+ offset,
            mode="lines", line=attr(color="black", width=1), opacity=0.45,
            showlegend=j == 1, name="Input waveform"))
        push!(traces, PlutoPlotly.scatter(x=t, y=targets[:, j] .+ offset,
            mode="lines", line=attr(color="#1f77b4", width=2),
            showlegend=j == 1, name="Ensemble target = mean(neighbors)"))
    end

    layout = Layout(
        title=attr(text="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km Direct Ensemble Targets (Mnn=$target_Mnn, mean) $(period_min)-$(period_max)s",
            font=attr(size=18, family="Computer Modern, serif")),
        height=700, width=900,
        xaxis=attr(title="Time Lag (s)"),
        yaxis=attr(title="Amplitude + offset"),
        legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.15, font=attr(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        shapes=velocity_vlines(t_vmin, t_vmax),
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 83946706-1d00-4794-af60-8c65979236fa
selected_pair_forward(x; training=false, beta_commit=0.25f0) =
    model(x, selected_pair_ids(size(x, 2)); training=training, beta_commit=beta_commit)

# ╔═╡ 1ce2fb4a-ff49-4873-9f93-6decbf9b8e80
let
    trained
    resample
    nshow = 10
    combo_idx = selected_combo isa Integer ? selected_combo : findfirst(x -> x == selected_combo, combo_labels)
    if combo_idx === nothing
        md"### Could not resolve the selected source state."
    else
        selected_ac = selected_indices_ac(combo_idx)
        selected_c = selected_indices_c(combo_idx)
        ac_ids = collect(selected_ac)
        c_ids = collect(selected_c)
        ac_count = min(nshow, length(ac_ids))
        c_count = min(nshow, length(c_ids))
        if ac_count == 0 || c_count == 0
            missing = String[]
            if ac_count == 0
                push!(missing, "acausal")
            end
            if c_count == 0
                push!(missing, "causal")
            end
            md"""### Selected state $(combo_labels[combo_idx]) has no $(join(missing, " and ")) windows to display."""
        else
            function random_sample(ids, count)
                if count >= length(ids)
                    collect(ids)
                else
                    perm = randperm(length(ids))
                    ids[perm[1:count]]
                end
            end
            ac_indices = random_sample(ac_ids, ac_count)
            c_indices = random_sample(c_ids, c_count)
            plot_n = min(length(ac_indices), length(c_indices))
            ac_indices = ac_indices[1:plot_n]
            c_indices = c_indices[1:plot_n]
            if plot_n == 0
                md"### Not enough windows to pair causal and acausal samples for plotting."
            else
                x_ac_input = xpu(data.D_ac_all[:, ac_indices])
                x_c_input = xpu(data.D_c_all[:, c_indices])
                r_ac = selected_pair_forward(x_ac_input; training=false)
                r_c = selected_pair_forward(x_c_input; training=false)
                x_ac_recon = cpu(r_ac.xhat)
                x_c_recon = cpu(r_c.xhat)
                x_ac_raw = cpu(data.D_ac_all[:, ac_indices])
                x_c_raw = cpu(data.D_c_all[:, c_indices])
                t_neg = [-(nth - i + 1) * dt for i in 1:nth]
                t_pos = [i * dt for i in 1:nth]
                t_full = [t_neg; t_pos]
                amplitude_pool = vcat(vec(mean(x_ac_raw; dims=2)), vec(mean(x_c_raw; dims=2)),
                    vec(mean(x_ac_recon[:, :, 1]; dims=2)), vec(mean(x_c_recon[:, :, 1]; dims=2)))
                vertical_spacing = maximum(abs.(amplitude_pool)) * 2.5 + 1e-3
                cs = ColorSchemes.rainbow
                colors = [Colors.hex(get(cs, (i - 1) / max(1, plot_n - 1))) for i in 1:plot_n]
                traces = AbstractTrace[]
                for i in 1:plot_n
                    raw_ac = x_ac_raw[:, i]
                    raw_c = x_c_raw[:, i]
                    recon_ac = x_ac_recon[:, i, 1]
                    recon_c = x_c_recon[:, i, 1]
                    raw_combo = [reverse(raw_ac); raw_c]
                    recon_combo = [reverse(recon_ac); recon_c]
                    offset = (i - 1) * vertical_spacing
                    push!(traces, PlutoPlotly.scatter(x=t_full, y=raw_combo .+ offset,
                        mode="lines", opacity=0.5, line=attr(color="black", width=1.),
                        name="Raw", showlegend=i == 1))
                    push!(traces, PlutoPlotly.scatter(x=t_full, y=recon_combo * 2 .+ offset,
                        mode="lines", line=attr(color="red", width=2,),
                        name="Recon", showlegend=i == 1))
                end
                layout = Layout(
                    title=attr(text="($(period_min)-$(period_max)s) Filtered $(combo_labels[combo_idx]): joined acausal+causal reconstructions $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km",
                        font=attr(size=18, family="Computer Modern, serif")),
                    height=max(750, plot_n * 40), width=950,
                    xaxis=attr(title="Time Lag (s)"),
                    yaxis=attr(title="Trace + offset"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.2, font=attr(size=10)),
                    shapes=velocity_vlines(t_vmin, t_vmax; symmetric=true))
                WideCell(PlutoPlotly.plot(traces, layout))
            end
        end
    end
end

# ╔═╡ 10000033-0000-0000-0000-000000000001
let
    trained
    resample
    nshow = 10
    x_sample = randobs(data.D_ac_all, nshow)
    result = selected_pair_forward(x_sample; training=false)
    x_orig = cpu(x_sample)

    x_recon = cpu(result.xhat) * 2.5

    t = collect(1:nth) .* dt

    traces = AbstractTrace[]
    for i in 1:nshow
        offset = (i - 1) * 4
        push!(traces, PlutoPlotly.scatter(x=t, y=x_orig[:, i] .+ offset,
            mode="lines", line=attr(color="black", width=1), opacity=0.5,
            showlegend=i == 1, name="Original"))
        push!(traces, PlutoPlotly.scatter(x=t, y=x_recon[:, i, 1] .+ offset,
            mode="lines", line=attr(color="red", width=2),
            showlegend=i == 1, name="Reconstructed"))
    end

    layout = Layout(
        title=attr(text="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km Reconstruction Examples (Acausal) $(period_min)-$(period_max)s",
            font=attr(size=18, family="Computer Modern, serif")),
        height=750, width=900,
        xaxis=attr(title="Time Lag (s)"),
        legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.15, font=attr(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        shapes=velocity_vlines(t_vmin, t_vmax),
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 41391781-fa8e-4479-a1a8-7132053026bf
let
    trained
    resample
    nshow = 10
    x_sample = randobs(data.D_c_all, nshow)
    result = selected_pair_forward(x_sample; training=false)
    x_orig = cpu(x_sample)

    x_recon = cpu(result.xhat) * 2.5

    t = collect(1:nth) .* dt

    traces = AbstractTrace[]
    for i in 1:nshow
        offset = (i - 1) * 4
        push!(traces, PlutoPlotly.scatter(x=t, y=x_orig[:, i] .+ offset,
            mode="lines", line=attr(color="black", width=1), opacity=0.5,
            showlegend=i == 1, name="Original"))
        push!(traces, PlutoPlotly.scatter(x=t, y=x_recon[:, i, 1] .+ offset,
            mode="lines", line=attr(color="red", width=2),
            showlegend=i == 1, name="Reconstructed"))
    end

    layout = Layout(
        title=attr(text="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km Reconstruction Examples (Causal) $(period_min)-$(period_max)s",
            font=attr(size=18, family="Computer Modern, serif")),
        height=750, width=900,
        xaxis=attr(title="Time Lag (s)"),
        legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.15, font=attr(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        shapes=velocity_vlines(t_vmin, t_vmax),
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 83946706-1d00-4794-af60-8c65979236fb
selected_pair_encode(x; training=false, beta_commit=0.25f0) =
    vqvae.encode(model, x, selected_pair_ids(size(x, 2)); training=training, beta_commit=beta_commit)

# ╔═╡ 10000031-0000-0000-0000-000000000001
# ╠═╡ disabled = true
#=╠═╡
let
    trained
    t_neg = [-(nth - i + 1) * dt for i in 1:nth]
    t_pos = [i * dt for i in 1:nth]
    t = [t_neg; t_pos]


    averaged_combinations = map(combo_labels) do selected_combo

        ks_tuple = state_tuple_for_model(findall(x -> x == selected_combo, combo_labels)[1])

        # Get indices for selected cluster (acausal)
        _, selected_ac = vqvae.filter_cluster(model, data.D_ac_all, data.pair_ids_ac, ks_tuple; cluster_filter_kwargs()...)
        # Get indices for selected cluster (causal)
        _, selected_c = vqvae.filter_cluster(model, data.D_c_all, data.pair_ids_c, ks_tuple; cluster_filter_kwargs()...)

        if (selected_ac == [] || selected_c == [])
            @info "No waveforms assigned to state $(selected_combo)"
            return (; raw=nothing, recon=nothing)
        else
            # Get the actual waveforms for these indices
            x_ac_sel = xpu(data.D_ac_all[:, selected_ac])
            x_c_sel = xpu(data.D_c_all[:, selected_c])

            # Reconstruct all selected waveforms and get per-component contributions
            r_ac = selected_pair_forward(x_ac_sel; training=false)
            r_c = selected_pair_forward(x_c_sel; training=false)

            # x_ac_recon = cpu(r_ac.xhat_per_slot[:, 2, :])
            # x_c_recon = cpu(r_c.xhat_per_slot[:, 2, :])

            r_ac = normalise(cpu(vec(mean(r_ac.xhat, dims=2))), dims=1)
            r_c = normalise(cpu(vec(mean(r_c.xhat, dims=2))), dims=1)

            ac = normalise(cpu(vec(mean(x_ac_sel; dims=2))), dims=1)
            c = normalise(cpu(vec(mean(x_c_sel; dims=2))), dims=1)
            return (; raw=[reverse(ac); c], recon=[reverse(r_ac); r_c])
        end


    end


    traces = AbstractTrace[]

    cs = ColorSchemes.rainbow
    nc = length(combo_labels)
    colors = [Colors.hex(get(cs, (i - 1) / max(1, nc - 1))) for i in 1:nc]

    j = 1
    for i in 1:length(combo_labels)
        if(isnothing(averaged_combinations[i].raw))
             @info "Skipping state $(combo_labels[i]) due to no assigned waveforms."
            continue
        end
        j += 1
        offset = (j - 1) * 3.0
        c = colors[i]
        push!(traces, PlutoPlotly.scatter(x=t, y=averaged_combinations[i].raw .* 0.25 .+ offset, mode="lines", line=attr(color="black", width=2), opacity=0.5,
            showlegend=(i == 1), name="Raw $(combo_labels[i])"))

        push!(traces, PlutoPlotly.scatter(x=t, y=averaged_combinations[i].recon .* 0.25 .+ offset,
            mode="lines", line=attr(color=c, width=1),
            showlegend=(i == 1), name="Recon. $(combo_labels[i])"))
    end


    layout = Layout(
        title=attr(text="Source States ($(selected_pair[1])-$(selected_pair[2])) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s",
            font=attr(size=18, family="Computer Modern, serif")),
        height=900, width=900,
        xaxis=attr(title="Lag Time (s)"),
        yaxis=attr(title="Source State"),
        legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.15, font=attr(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        shapes=velocity_vlines(t_vmin, t_vmax; symmetric=true),
    )
    WideCell(PlutoPlotly.plot(traces, layout))

end
  ╠═╡ =#

# ╔═╡ a2000005-0000-0000-0000-000000000001
let
    trained
    E = vqvae.get_codebook(model)
    if ndims(E) == 3
        E = E[:, :, 1]
    end
    z = Float32.(E[:, selected_code_for_decode:selected_code_for_decode])
    xhat = cpu(model.decoder(xpu(z)))
    wave = vec(xhat[:, 1])
    t = collect(1:length(wave)) .* dt
    trace = PlutoPlotly.scatter(x=t, y=wave, mode="lines",
        line=attr(width=2, color="firebrick"),
        name="Decoded prototype")
    layout = Layout(
        title=attr(text="Decoded Prototype: code=$(selected_code_for_decode)"),
        xaxis=attr(title="Time (s)"),
        yaxis=attr(title="Amplitude"),
        shapes=velocity_vlines(t_vmin, t_vmax; symmetric=false),
        plot_bgcolor="white", paper_bgcolor="white",
        width=850, height=350
    )
    WideCell(PlutoPlotly.plot([trace], layout))
end

# ╔═╡ 6558ee70-42df-11f1-bd0f-091af6e4ca0c
analysis_cache_for_disk = @use_memo([trained, cache_pairs_per_batch, selected_training_pair_names, reload_network]) do
    trained

    struct_namedtuple(x) = NamedTuple{fieldnames(typeof(x))}(
        map(name -> getfield(x, name), fieldnames(typeof(x))))

    pair_metadata = [
        (;
            pair_id=pid,
            pair=pair_bundles[pid].pair,
            pair_name=join(pair_bundles[pid].pair, "-"),
            distance=pair_bundles[pid].distance,
            latitudes=pair_bundles[pid].latitudes,
            longitudes=pair_bundles[pid].longitudes,
            geometry_raw=geometry_raw[:, pid],
            geometry_standardized=geometry_features[:, pid],
            n_ac=size(pair_splits[pid].D_ac_all, 2),
            n_c=size(pair_splits[pid].D_c_all, 2),
        )
        for pid in eachindex(pair_splits)
    ]

    (;
        cache_version="vqvae_v6_analysis_cache",
        created_at_unix=time(),
        selected_pairs=selected_pairs,
        selected_training_pair_names=collect(selected_training_pair_names),
        pair_metadata,
        geometry_feature_names=["mid_lat", "mid_lon", "distance_km", "axis_cos2", "axis_sin2"],
        geometry_standardization=(mean=geometry_standardization.mean, std=geometry_standardization.std),
        vqvae_parameters=vqvae_parameters,
        vqvae_para=struct_namedtuple(vqvae_para),
        training_para=struct_namedtuple(training_para),
        loss_history,
        model_state=Flux.state(cpu(model)),
        coarse_codebooks=[vqvae.get_pair_coarse_codebook(model, pid) for pid in 1:length(pair_splits)],
        shared_detail_codebooks=vqvae.get_shared_detail_codebooks(model),
        all_pair_encoded_cache,
    )
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
ParameterSchedulers = "d7d3b36b-41b8-4d0d-a2bf-768c6151755e"
Peaks = "18e31ff7-3703-566c-8e60-38913d67486b"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[compat]
BenchmarkTools = "~1.8.0"
CUDA = "~6.0.0"
Clustering = "~0.15.8"
ColorSchemes = "~3.31.0"
Colors = "~0.13.1"
DSP = "~0.8.4"
Distances = "~0.10.12"
Enzyme = "~0.13.138"
FFTW = "~1.10.0"
Flux = "~0.16.10"
Functors = "~0.5.2"
JLD2 = "~0.6.4"
MLUtils = "~0.4.8"
NNlib = "~0.9.34"
NearestNeighbors = "~0.4.27"
Optimisers = "~0.4.7"
ParameterSchedulers = "~0.4.3"
Peaks = "~0.6.2"
PlutoHooks = "~0.1.0"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.80"
ProgressLogging = "~0.1.6"
StatsBase = "~0.34.10"
Zygote = "~0.7.10"
cuDNN = "~6.0.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "ea6342982141bc992f765fb351843e356d9370f3"

[[deps.ADTypes]]
git-tree-sha1 = "f7304359109c768cf32dc5fa2d371565bb63b68a"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.21.0"
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
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "2eeb2c9bef11013efc6f8f97f32ee59b146b09fb"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.44"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0761717147821d696c9470a7a86364b2fbd22fd8"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.2"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "e0b47732a192dd59b9d079a06d04235e2f833963"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.12.2"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

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

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "cceb62468025be98d42a5dc581b163c20896b040"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.9"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "PrecompileTools", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "9670d3febc2b6da60a0ae57846ba74670290653f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.8.0"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CUDA]]
deps = ["CUDACore", "CUDATools", "Reexport", "cuBLAS", "cuFFT", "cuRAND", "cuSOLVER", "cuSPARSE"]
git-tree-sha1 = "bcbaecc92b4b8b0fb25997f4d84451b198344d4d"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "6.0.0"

[[deps.CUDACore]]
deps = ["Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "PrecompileTools", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "StaticArrays"]
git-tree-sha1 = "dc5b6ea53fa3b3bedd2fe1c6037687dd4ee85e70"
uuid = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
version = "6.0.0"
weakdeps = ["ChainRulesCore", "EnzymeCore", "SpecialFunctions"]

    [deps.CUDACore.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.CUDATools]]
deps = ["CUDACore", "CUDA_Compiler_jll", "CUPTI", "Crayons", "GPUCompiler", "LLVM", "NVML", "NVTX", "PrecompileTools", "Preferences", "PrettyTables", "Printf", "Statistics", "demumble_jll"]
git-tree-sha1 = "38ee815c0b8b1423035d10f657f9f756e39c5205"
uuid = "9ec180c6-1c07-47c7-9e6e-ebefa4d1f6d0"
version = "6.0.0"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "b977706846cb0a75d3842a1fed810ab2e6ab2f94"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.4.3+0"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "3b759ec65ac87ad192c2925114fa5c126657a5bd"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.2.1+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "f9a521f52d236fe49f1028d69e549e7f2644bb72"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "1.0.0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "c0314d9fb0ebd00e404feba4c3fbc04c9975abc1"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.21.0+1"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "70dea6a7133d2100a143b515a00d6d887e208500"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "9.20.0+0"

[[deps.CUPTI]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox"]
git-tree-sha1 = "b37790736de8e067a26ade5cbcd6bf240ddd20ec"
uuid = "9e67e8f6-ba02-4b6c-a7db-3b11ae1e7ab7"
version = "6.0.0"

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
git-tree-sha1 = "cee8104904c53d39eb94fd06cbe60cb5acde7177"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.0.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

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

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

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

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5989debfc3b38f736e69724818210c67ffee4352"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.4"

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

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

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
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

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

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "InteractiveUtils", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "PrecompileTools", "Preferences", "Printf", "Random", "SparseArrays"]
git-tree-sha1 = "d6dd65421104fa9f7d5cc37283a998937f359a39"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.13.138"
weakdeps = ["ADTypes", "BFloat16s", "ChainRulesCore", "GPUArraysCore", "LogExpFunctions", "SpecialFunctions", "StaticArrays"]

    [deps.Enzyme.extensions]
    EnzymeBFloat16sExt = "BFloat16s"
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeGPUArraysCoreExt = "GPUArraysCore"
    EnzymeLogExpFunctionsExt = "LogExpFunctions"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

[[deps.EnzymeCore]]
git-tree-sha1 = "24bbb6fc8fb87eb71c1f8d00184a60fc22c63903"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.19"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "4c22000e08aaa862526d9a41cfb7003e4002e653"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.256+0"

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

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6522cfb3b8fe97bec632252263057996cbd3de20"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.18.0"

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
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

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
git-tree-sha1 = "cddeab6487248a39dae1a960fff0ac17b2a28888"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.3"
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
git-tree-sha1 = "34fd745547978beb471f029f447290ef4dbc7bbd"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.5.3"
weakdeps = ["JLD2"]

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "fedfe5e7db7035271c3f58359007f971da1dde87"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.9.1"

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
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "57e9ce6cf68d0abf5cb6b3b4abf9bedf05c939c0"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.15"

[[deps.InfiniteArrays]]
deps = ["ArrayLayouts", "FillArrays", "Infinities", "LazyArrays", "LinearAlgebra"]
git-tree-sha1 = "e61675cbcf3ce57ea3566b0abcfe46e4a521bf6f"
uuid = "4858937d-0d70-526a-a4dd-2d5cb5dd786c"
version = "0.12.15"
weakdeps = ["Statistics"]

    [deps.InfiniteArrays.extensions]
    InfiniteArraysStatisticsExt = "Statistics"

[[deps.Infinities]]
git-tree-sha1 = "4495006c20b2fd27b8c453a1dd31d423654f3772"
uuid = "e1ba4f0e-776d-440f-acd9-e1d2e9742647"
version = "0.1.12"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

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
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "67c6f1f085cb2671c93fe34244c9cccde30f7a26"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.5.0"

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

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "f2e76d3ced51a2a9e185abc0b97494c7273f649f"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.41"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "f1b04cbf4be550fabad4bbc38c3b18ba5bdf53a6"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.7.0"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "f1d1adfff151fd02b4062d1af82df02052dc4a0c"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.42+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "MatrixFactorizations", "SparseArrays"]
git-tree-sha1 = "35079a6a869eecace778bcda8641f9a54ca3a828"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "1.10.0"
weakdeps = ["StaticArrays"]

    [deps.LazyArrays.extensions]
    LazyArraysStaticArraysExt = "StaticArrays"

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

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "2dfe3b4b96c6ecbea7c798dfbe96d493fd7a1848"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.17.8"

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

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "6731e0574fa5ee21c02733e397beb133df90de35"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "2.2.0"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

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
git-tree-sha1 = "78cd28dbd5f03f99ccaba45c987107adcb61c115"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.34"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NVML]]
deps = ["CEnum", "CUDACore", "GPUToolbox", "Libdl"]
git-tree-sha1 = "d041854ab4c16d1b1b6d8ba1092183745a7fe26a"
uuid = "611af6d1-644e-4c5d-bd58-854d7d1254b9"
version = "6.0.0"

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
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NearestNeighbors]]
deps = ["AbstractTrees", "Distances", "StaticArrays"]
git-tree-sha1 = "e2c3bba08dd6dedfe17a17889131b885b8c082f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.27"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "22faba70c22d2f03e60fbc61da99c4ebfc3eb9ba"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.5.0"

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
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.ParameterSchedulers]]
deps = ["InfiniteArrays", "Optimisers"]
git-tree-sha1 = "c62f0da0663704d0472ae578c9bb802c44e70a4c"
uuid = "d7d3b36b-41b8-4d0d-a2bf-768c6151755e"
version = "0.4.3"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

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

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

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

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
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
deps = ["CodeTracking", "FileWatching", "InteractiveUtils", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "5f4f629c085b87e71125eec6773f5f872c74a47a"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.14.2"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.SciMLPublic]]
git-tree-sha1 = "0ba076dbdce87ba230fff48ca9bca62e1f345c9b"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.1"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "ac4b837d89a58c848e85e698e2a2514e9d59d8f6"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.0"

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
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

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

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

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
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

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

[[deps.StructIO]]
git-tree-sha1 = "c581be48ae1cbf83e899b14c07a807e1787512cc"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.1"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "86f5831495301b2a1387476cb30f86af7ab99194"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.0"

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

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "SplittablesBase", "Tables"]
git-tree-sha1 = "4aa1fdf6c1da74661f6f5d3edfd96648321dade9"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.85"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

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
git-tree-sha1 = "a29cbf3968d36022198bcc6f23fdfd70f7caf737"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.10"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
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
git-tree-sha1 = "5df9edbdfff9fed8b818535e7b86e92a85fc7709"
uuid = "182d3088-87b7-4494-8cad-fc6afaa545bc"
version = "6.0.0"
weakdeps = ["EnzymeCore"]

    [deps.cuBLAS.extensions]
    EnzymeCoreExt = "EnzymeCore"

[[deps.cuDNN]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDNN_jll"]
git-tree-sha1 = "6af96a746f385200baec6398c71c19f2efb4bf7e"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "6.0.0"

[[deps.cuFFT]]
deps = ["AbstractFFTs", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "c5de5ab272aae86658d3b05999b9ea7bc60503d0"
uuid = "533571aa-0936-420e-b4be-9c66f5f626ca"
version = "6.0.0"

[[deps.cuRAND]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "Random", "Random123", "RandomNumbers"]
git-tree-sha1 = "43d84e8d12e75c401d69d88475d304ca7a038afd"
uuid = "20fd9a0b-12d5-4c2f-a8af-7c34e9e60431"
version = "6.0.0"

[[deps.cuSOLVER]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "LinearAlgebra", "SparseArrays", "cuBLAS", "cuSPARSE"]
git-tree-sha1 = "4b15758b0667ba4b715252fe0dfae9dafae1b739"
uuid = "887afef0-6a32-4de5-add4-7827692ba8fc"
version = "6.0.0"

[[deps.cuSPARSE]]
deps = ["Adapt", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUArrays", "GPUToolbox", "KernelAbstractions", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f5d1fdae1053286374c80e5f6608a913aedad7ef"
uuid = "b26da814-b3bc-49ef-b0ee-c816305aa060"
version = "6.0.0"

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
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─10000004-0000-0000-0000-000000000001
# ╠═10000002-0000-0000-0000-000000000001
# ╠═10000003-0000-0000-0000-000000000001
# ╠═10000001-0000-0000-0000-000000000001
# ╠═02556dd0-4cb7-4251-a969-6bea09a41358
# ╠═9db29532-6d82-495c-bc3f-0daa882f5064
# ╠═341dbed8-1d09-4b46-8434-eb332c332f75
# ╠═c7f70869-8f84-4c33-a455-d79f78ac02ec
# ╠═da62431a-7cc6-4253-986d-5ba7d39e9f90
# ╠═53f17afb-91fb-4881-a9f4-9fa87a24fee6
# ╠═418c15e5-8116-4d86-8c3e-aeac13cc3ef1
# ╠═53204e4f-16e5-4960-a451-b5660ea0f182
# ╠═e85ac38a-e243-4233-8b75-1bcbc3884cb1
# ╟─10000017-0000-0000-0000-000000000001
# ╠═10000018-0000-0000-0000-000000000001
# ╠═6d7cebf7-fb3e-4134-a428-91dea9f272b4
# ╟─10000005-0000-0000-0000-000000000001
# ╠═10000007-0000-0000-0000-000000000001
# ╠═10000006-0000-0000-0000-000000000001
# ╠═10000008-0000-0000-0000-000000000001
# ╠═10000009-0000-0000-0000-000000000001
# ╠═1000000a-0000-0000-0000-000000000001
# ╠═bbf3679e-9fbc-47b4-a1eb-03e48ee94a59
# ╠═8f596d58-2051-40f8-a52e-00afd3fe975d
# ╠═a7d54261-fa3b-4a07-bc32-ecebf682d0bf
# ╟─1000000d-0000-0000-0000-000000000001
# ╠═1000000e-0000-0000-0000-000000000001
# ╠═f3b8c867-f8f9-45e0-a41f-4b7d7a1b17f0
# ╠═0f84f2f6-6403-4a2e-9c42-6f8a84a2bc3f
# ╠═2a8a4d12-c96b-4d64-8f58-98f54f81a77b
# ╠═eaa7d770-d2d0-42f0-a65b-99447f11649a
# ╠═74d0f419-79d0-4a6a-ae65-18f6be9d64c8
# ╠═c669f1d3-26c0-4028-b3d4-5a87bd696924
# ╠═6558d732-42df-11f1-a447-1bdaa63b389c
# ╠═6558d7e6-42df-11f1-accb-c305628b5b77
# ╠═f86fec7f-f467-4411-80aa-c1621e3de063
# ╠═1000000f-0000-0000-0000-000000000001
# ╟─10000010-0000-0000-0000-000000000001
# ╠═116cc111-1e02-44bb-bf5f-1c52e58e75ee
# ╠═a05d0dce-3c06-4b38-88bc-5202dfd851d9
# ╠═c1c3dea0-4590-4c82-88b4-3b79ddcea7f1
# ╠═10000012-0000-0000-0000-000000000001
# ╠═10000012-0000-0000-0000-000000000002
# ╠═10000012-0000-0000-0000-000000000003
# ╠═10000012-0000-0000-0000-000000000004
# ╠═10000012-0000-0000-0000-000000000005
# ╠═10000013-0000-0000-0000-000000000001
# ╠═10000014-0000-0000-0000-000000000001
# ╟─10000019-0000-0000-0000-000000000001
# ╠═1000001a-0000-0000-0000-000000000001
# ╠═1000001b-0000-0000-0000-000000000001
# ╠═946fe8d3-9b9a-4d3e-9bb1-4e65f10bb3f0
# ╠═1000001c-0000-0000-0000-000000000001
# ╠═f2369548-2d88-11f1-a737-85bad04c89cb
# ╠═1a623368-a2fc-4b7f-8d01-68e36d04a891
# ╠═a0000101-0000-0000-0000-000000000001
# ╟─1000001e-0000-0000-0000-000000000001
# ╟─10000020-0000-0000-0000-000000000002
# ╠═10000021-0000-0000-0000-000000000001
# ╠═10000020-0000-0000-0000-000000000001
# ╠═10000021-0000-0000-0000-000000000002
# ╠═1000001d-0000-0000-0000-000000000001
# ╟─10000020-0000-0000-0000-000000000003
# ╠═10000020-0000-0000-0000-000000000004
# ╠═1ddfb196-7d6a-44bf-82fc-793a9a30fb9c
# ╟─6558ecfe-42df-11f1-b755-35afb4d87b8a
# ╠═6558ee70-42df-11f1-bd0f-091af6e4ca0c
# ╟─10000022-0000-0000-0000-000000000001
# ╠═964da977-5394-4369-9185-60797ad764e3
# ╟─a6000001-0000-0000-0000-000000000001
# ╠═a6000002-0000-0000-0000-000000000001
# ╠═a6000003-0000-0000-0000-000000000001
# ╠═a6000004-0000-0000-0000-000000000001
# ╠═a6000005-0000-0000-0000-000000000001
# ╟─10000024-0000-0000-0000-000000000001
# ╟─24ceed68-00a3-4d29-9025-89bcf2f9251c
# ╟─eba0784e-4ab9-4774-9851-56538b579fa6
# ╠═699d2c90-4a52-44de-a58e-215f07fa6028
# ╟─10000025-0000-0000-0000-000000000001
# ╟─10000036-0000-0000-0000-000000000001
# ╠═e45e700d-4780-437f-8300-78398c10b927
# ╠═1ddfb196-7d6a-44bf-82fc-793a9a30fb9d
# ╠═eb127aaf-8f69-4d14-b132-811e60350e89
# ╠═8d249a8d-4124-4c9d-bea4-64c940f88a32
# ╠═a2000006-0000-0000-0000-000000000001
# ╟─1000002b-0000-0000-0000-000000000001
# ╠═c618590d-ca23-4901-b143-2f6482f32249
# ╠═e6e3dd04-924d-4688-a9a9-c411764719f3
# ╟─66b2b56b-cb5e-4620-866f-903f774fdbe5
# ╠═1000002e-0000-0000-0000-000000000001
# ╠═c7a6235f-c023-4149-92a1-837902763836
# ╟─10000028-0000-0000-0000-000000000001
# ╟─1000002a-0000-0000-0000-000000000001
# ╟─1000002f-0000-0000-0000-000000000001
# ╟─3bf7a033-db24-4e6c-93a9-8706d2ea57c3
# ╟─ed09ec19-f1b0-4819-9c9f-7ee796bd8f09
# ╠═60f24a45-507d-44cf-af59-00058418617c
# ╠═1ce2fb4a-ff49-4873-9f93-6decbf9b8e80
# ╟─10000032-0000-0000-0000-000000000001
# ╟─84c4a49c-c6fb-45b4-ac33-a5510cd6618e
# ╟─10000033-0000-0000-0000-000000000001
# ╟─ed8e51a4-2eb7-447d-a1c4-c0b346005e79
# ╠═b6000001-0000-0000-0000-000000000001
# ╟─6a3e970f-a7e7-4c67-a9e6-dc79be22c547
# ╟─41391781-fa8e-4479-a1a8-7132053026bf
# ╟─3da3c466-6d7f-4d8b-8ba2-1f7a66ff9a46
# ╠═cff400dc-49c8-4d9a-ac7f-5178ab097dce
# ╟─b8dfc2d6-43cb-4ed7-a32e-9df8a73d8a91
# ╠═ab3f68f7-6179-4243-84d5-df8ea504c055
# ╠═e1f9f9d6-8d55-41a7-9f84-7b49ebf69f2e
# ╠═2faf3842-106c-43d8-96fd-0a7154b1a338
# ╟─3d9ee2f7-a887-4e37-9cf0-552e72705791
# ╠═0cdffa4e-1a61-4dae-9d72-fb4a89e0c246
# ╠═f4f14bc5-e69d-4e98-b6ec-17efd4f80ea2
# ╟─10b69e70-8c7f-44ce-85ab-44a48c4eac5e
# ╠═7d3aa945-56f2-4444-aeea-dac1734da1a0
# ╟─c1f2e3d4-a5b6-4c7d-8e9f-0a1b2c3d4e5f
# ╠═976a3916-41e8-11f1-a0d8-61b7d62f95de
# ╠═9b3dd9f5-15fa-4785-b650-85ecf5c5c6f7
# ╟─d8477392-2546-411e-baf0-302c03de50b0
# ╠═3214034a-c6db-445d-b466-9ea422b7a94e
# ╠═f0f2d04a-95e7-4126-adc4-f6f0124b0113
# ╠═48bc91de-f66c-4443-b61f-ff5eb45d23c5
# ╠═d7efd06d-8f55-4717-a228-cccf2acfe30d
# ╠═3ab7d986-0597-4fac-b1b1-4c1a56a181a9
# ╟─d5639a0d-9b87-45f3-b5a2-4c3132482394
# ╠═240393ec-e1d5-4086-b8d8-a2b9e72078d8
# ╟─10000034-0000-0000-0000-000000000001
# ╠═10000035-0000-0000-0000-000000000001
# ╟─db4ddb38-2938-11f1-b8e3-e5227df9322c
# ╠═e928b8ab-e159-427a-b525-c6d60e6d6015
# ╠═b1f70fd0-1a7a-4f8f-9e7c-8e3e8cd3f5e1
# ╠═83946706-1d00-4794-af60-8c65979236f8
# ╠═e8ae6df6-fa04-42b7-a6e5-b2ade4322995
# ╠═83946706-1d00-4794-af60-8c65979236f9
# ╠═83946706-1d00-4794-af60-8c65979236fa
# ╠═83946706-1d00-4794-af60-8c65979236fb
# ╠═83946706-1d00-4794-af60-8c65979236fc
# ╠═83946706-1d00-4794-af60-8c65979236fd
# ╠═10000031-0000-0000-0000-000000000001
# ╠═d6590132-7190-4759-8739-3c6f5e7d9969
# ╟─10000015-0000-0000-0000-000000000001
# ╟─10000037-0000-0000-0000-000000000001
# ╠═a2000003-0000-0000-0000-000000000001
# ╠═a2000004-0000-0000-0000-000000000001
# ╠═a2000001-0000-0000-0000-000000000001
# ╠═a2000002-0000-0000-0000-000000000001
# ╠═a2000005-0000-0000-0000-000000000001
# ╠═a2000010-0000-0000-0000-000000000001
# ╠═a2000011-0000-0000-0000-000000000001
# ╠═a2000012-0000-0000-0000-000000000001
# ╠═a2000013-0000-0000-0000-000000000001
# ╠═a2000014-0000-0000-0000-000000000001
# ╠═0d061de8-14f2-49c9-8d43-1e6b87e9d785
# ╠═0d061de8-14f2-49c9-8d43-1e6b87e9d786
# ╠═0d061de8-14f2-49c9-8d43-1e6b87e9d787
# ╠═0d061de8-14f2-49c9-8d43-1e6b87e9d788
# ╠═c669f1d3-26c0-4028-b3d4-5a87bd696925
# ╠═c669f1d3-26c0-4028-b3d4-5a87bd696926
# ╠═c669f1d3-26c0-4028-b3d4-5a87bd696927
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
