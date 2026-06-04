### A Pluto.jl notebook ###
# v0.20.21

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

# ╔═╡ ee3a210b-2e72-4658-82dc-de6fb6179e60
using Dates,DrWatson

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
        Peaks,
		StatsBase,
	    Optimisers,
		ParameterSchedulers,
		Functors
	CUDA.device!(1)
end

# ╔═╡ 66eeaea8-8d59-4dc8-98bf-afbc2ad5c42e
using CSV,DataFrames

# ╔═╡ da62431a-7cc6-4253-986d-5ba7d39e9f90
using Zygote

# ╔═╡ 53f17afb-91fb-4881-a9f4-9fa87a24fee6
using Enzyme

# ╔═╡ 418c15e5-8116-4d86-8c3e-aeac13cc3ef1
using BenchmarkTools

# ╔═╡ d2a370a2-841d-4014-b419-33076fe051ab
using Printf

# ╔═╡ 6197fbaa-60cc-43ab-8db4-8b4060fa4610
using GMT

# ╔═╡ e00d151e-64aa-4677-8dfb-d543065cd0e0
XJ_latlong=unique(vcat(CSV.read("/mnt/NASdata2/Sanket_data/California_09032026/data/stationlists/Stations_California_XJ.csv",DataFrame),CSV.read("/mnt/NASdata2/Sanket_data/California_09032026/data/stationlists/Stations_California_XJ_new.csv",DataFrame)))

# ╔═╡ 10000002-0000-0000-0000-000000000001
TableOfContents(include_definitions=true)

# ╔═╡ 10000003-0000-0000-0000-000000000001
xpu = gpu

# ╔═╡ 10000004-0000-0000-0000-000000000001
md"""# VQ-VAE Analysis Notebook

Analyze the trained data for all station pairs
"""

# ╔═╡ 10000005-0000-0000-0000-000000000001
md"## Data Loading"

# ╔═╡ b9acedbd-2e86-44ab-9d21-61a09d67ec51
ntrainings=1

# ╔═╡ d87a898c-7d93-4b37-8dad-4184575631e1
# basedr(training_idx)="/mnt/NASdata2/Sanket_data/California_results_13032026_XJ_$(training_idx)_v28"

# ╔═╡ 0c6af546-a71f-411b-8bb4-b52fb507390a
# [sortperm(distances_above_30)]

# ╔═╡ 6bcd1038-25e2-4562-ad78-6100e1e08c9a
@bind select_BAND Select(["3_7","8_50","8_80"])

# ╔═╡ 7bae698c-4223-4eb0-b11f-7588d11c96fe
begin
    dt = 0.8
    # Define main folder containing the training results (first training run)
	if select_BAND=="8_50"
mainfolder = "/mnt/NASdata2/Sanket_data/California_results_13032026_XJ_1_v28/" 
elseif select_BAND=="8_80"
mainfolder = "/mnt/NASdata2/Sanket_data/California_results_13032026_XJ_80sec_1_v28/"
elseif select_BAND=="3_7"
mainfolder = "/mnt/NASdata2/Sanket_data/California_results_25032026_XJ_3_7sec_new_vqvae_1_v28/"
end
    # mainfolder = "/mnt/NASdata2/Sanket_data/California_results_13032026_XJ_1_v28/"  
		# "/Data1/California_results_05022026_tapered_[(8, 50)]_v28/"
    # Extract pair names from filenames
end

# ╔═╡ 9354d1b2-a6d7-4d2c-abee-43949070a726
mainfiles1 = filter(x -> occursin("k=10-", x), readdir(mainfolder, join=true))

# ╔═╡ be4a46d9-241f-4ee9-a010-d6665ad69877
rec_names = map(mainfiles1) do rds
        split(split(rds, "28/")[2], "-k=")[1]
    end

# ╔═╡ 05a75438-74f8-453f-b72b-dcd146d22e4b
if select_BAND=="8_50"
basedr(training_idx)="/mnt/NASdata2/Sanket_data/California_results_13032026_XJ_$(training_idx)_v28"
elseif select_BAND=="8_80"
basedr(training_idx)="/mnt/NASdata2/Sanket_data/California_results_13032026_XJ_80sec_$(training_idx)_v28"
elseif select_BAND=="3_7"
basedr(training_idx)="/mnt/NASdata2/Sanket_data/California_results_25032026_XJ_3_7sec_new_vqvae_$(training_idx)_v28"
end

# ╔═╡ 654fa7a5-af59-4a5e-a56f-c72c45239057
#ARC2_TWR2

# ╔═╡ 162350c6-9fb7-45ac-9368-c0c4a6d71eef
let
# Map plot for the selected station pair
region = (-120, -117, 35.5, 37.5)
topo = grdcut("@earth_relief_01m", region=region, figsize=(6, 8))
grdimage(topo, region=region, proj=:Mercator, color=:oleron)

receivers = XJ_latlong[!, "Station Code"]
	# [st_id]
lon = XJ_latlong[!, "Longitude"]
	# [st_id]
lat = XJ_latlong[!, "Latitude"]
	# [st_id]
text!(String.(receivers), x=lon, y=lat .+ 0.05, font=(12, "Helvetica-Bold", :black), justify=:RB)
GMT.plot!(lon, lat, region=region, marker=:circle, ms=0.3, fill=:red, markerline=:cyan, show=true)
end

# ╔═╡ 586e46ab-7a24-4a09-82b1-86a674060fed
selected_training=1

# ╔═╡ 0ff4cfa0-a08d-4695-84c8-729b025e2851
@bind period_to_plot Slider(3:0.5:8,show_value=true)

# ╔═╡ 120d5d95-6254-4b61-9928-8a7e01d34594
md"""
#### Velocity filters for `disp_rank1`

Upper velocity cutoff: $(@bind velocity_threshold_rank1 Slider(2.0:0.5:6.0, default=4.5, show_value=true)) km/s &nbsp;&nbsp; Mean-window fraction ±: $(@bind rank1_frac Slider(0.05:0.05:0.50, default=0.25, show_value=true))
"""

# ╔═╡ cc06cac4-5759-4eaa-892d-1550966475b5
periodss = 
	# 5:1:30
	3:0.5:8

# ╔═╡ b361db3e-7c6f-4d88-b1f1-532e7b49be19
# save_periodwise_separate_csv(
#     records,
#     periodss,
# 	"/mnt/NASData2/Sanket_data/California_XJ_DC_VQVAE_$(select_BAND)_28032026_3tr_9e-1_corr_filtered_bw_$(bandwidth_percent)_above_first_continuous/"
#     # "/Data1/California_DC_CSS_27022026_v2/"
# )

# ╔═╡ a01ed292-1f8e-479e-a335-08fd34bdc319
struct DispersionRecord
    sta1::String
    sta2::String
    lat1::Float64
    lon1::Float64
    lat2::Float64
    lon2::Float64
    mode::Union{Int, Symbol}
    period::Vector{Float64}
    groupvel::Vector{Float64}
end

# ╔═╡ bbd1ec61-fe83-466d-a022-a1cb869fb690


# ╔═╡ fc52a726-a744-4de1-8231-096ed224afb5
function save_periodwise_separate_csv(records, periodss, outfolder; tol=1e-3)

    isdir(outfolder) || mkdir(outfolder)

    for target_period in periodss

        @info "Processing period = $target_period s"

        df = DataFrame(
            period = Float64[],
            St1 = String[],
            St2 = String[],
            lat1 = Float64[],
            lon1 = Float64[],
            lat2 = Float64[],
            lon2 = Float64[],
            group_velocity = Float64[]
        )

        for r in records

            # locate this period in that station pair
            idx = findfirst(p -> isapprox(p, target_period; atol=tol), r.period)
            idx === nothing && continue

            gv = r.groupvel[idx]

            push!(df, (
                target_period,
                r.sta1,
                r.sta2,
                r.lat1,
                r.lon1,
                r.lat2,
                r.lon2,
                gv
            ))
        end

        # ---- filename ----
        fname = joinpath(outfolder,@sprintf("dispersion_%04.1fs.csv", target_period))
						 # %04.1fs
                         # @sprintf("dispersion_%02ds.csv", Int(round(target_period))))

        CSV.write(fname, df)

        println("Saved $(nrow(df)) rays → $fname")
    end
end

# ╔═╡ 22202b34-9585-49cc-9b6f-c7a3c35f90b2
function attach_geolocation_df(avg_results, california_latlongdat)

    station_col = california_latlongdat[!, "Station Code"]
    lat_col     = california_latlongdat[!, "Latitude"]
    lon_col     = california_latlongdat[!, "Longitude"]

    # convert station column to clean strings
    # station_clean = strip.(string.(station_col))

    records = DispersionRecord[]

    # ---- main loop ----
    for ((pairname, mode), (periods, gvel)) in avg_results

        sta1, sta2 = split(String(pairname), "_")

        # find station rows
        id1 = findfirst(==(sta1), station_col)
        id2 = findfirst(==(sta2), station_col)

        if id1 === nothing || id2 === nothing
            @warn "Skipping $pairname (station not found in CSV)"
            continue
        end

        lat1 = Float64(lat_col[id1])
        lon1 = Float64(lon_col[id1])
        lat2 = Float64(lat_col[id2])
        lon2 = Float64(lon_col[id2])

        push!(records,
            DispersionRecord(
                sta1, sta2,
                lat1, lon1,
                lat2, lon2,
                mode,
                periods,
                gvel
            )
        )
    end

    return records
end

# ╔═╡ fa452b46-d89b-4a29-969a-77116f1c1c23
function load_mode_from_training(pair, training_idx::Int)
    basedir = basedr(training_idx)
    files = readdir(basedir)
    fil_files = filter(x -> occursin("k=10", x), files)
    matching = filter(f -> occursin(pair, f), fil_files)
    if !isempty(matching)
        filepath = joinpath(basedir, matching[1])
        return jldopen(filepath)["full_modes"]
    end
    return nothing
end

# ╔═╡ 468730f9-0619-4ecd-b580-13082f450b4b
"""
    average_dispersion_same_periods(all_pairs_2lambda; min_count=1, period_digits=6)

Build a single average dispersion curve from `all_pairs_2lambda` by averaging
velocities only at the same period values.

# Arguments
- `all_pairs_2lambda`: Vector of entries like
  `(pair, combo_idx) => (periods, velocities)`
- `min_count`: keep only periods that appear in at least this many curves
- `period_digits`: rounding precision used to match period values robustly

# Returns
Named tuple with:
- `periods`: sorted common period values
- `avg_velocities`: mean velocity for each period
- `counts`: number of contributing curves at each period
"""
function average_dispersion_same_periods(
    all_pairs_2lambda;
    min_count::Int=1,
    period_digits::Int=6
)
    period_to_vels = Dict{Float64, Vector{Float64}}()

    for entry in all_pairs_2lambda
        periods, velocities = entry.second
        for (p, v) in zip(periods, velocities)
            p_key = round(Float64(p), digits=period_digits)
            push!(get!(period_to_vels, p_key, Float64[]), Float64(v))
        end
    end

    if isempty(period_to_vels)
        return (periods=Float64[], avg_velocities=Float64[], counts=Int[])
    end

    sorted_periods = sort(collect(keys(period_to_vels)))
    out_periods = Float64[]
    out_velocities = Float64[]
    out_counts = Int[]

    for p in sorted_periods
        vals = period_to_vels[p]
        c = length(vals)
        c < min_count && continue

        push!(out_periods, p)
        push!(out_velocities, mean(vals))
        push!(out_counts, c)
    end

    return (periods=out_periods, avg_velocities=out_velocities, counts=out_counts)
end

# ╔═╡ cc01ab54-7913-4e9a-aff9-2ed745abf700
"""
    filter_by_mean_velocity_window(all_pairs_2lambda; frac=0.25, min_count=1, period_digits=6)

Filter dispersion data using a period-wise velocity window around
`average_dispersion_same_periods(all_pairs_2lambda).avg_velocities`.

For each period p, kept range is:
avg_vel(p) * (1 - frac) to avg_vel(p) * (1 + frac).

# Returns
Named tuple with:
- `periods`, `avg_velocities`, `lower_bounds`, `upper_bounds`
- `avg_all`: unfiltered averaged dispersion (periods, avg_velocities, counts)
- `avg_filtered`: filtered averaged dispersion (periods, avg_velocities, counts)
- `filtered_pairs`: filtered `all_pairs_2lambda` entries
"""
function filter_by_mean_velocity_window(
    all_pairs_2lambda;
    frac::Float64=0.25,
    min_count::Int=1,
    period_digits::Int=6
)
    avg_all = average_dispersion_same_periods(
        all_pairs_2lambda;
        min_count=min_count,
        period_digits=period_digits
    )

    isempty(avg_all.avg_velocities) && return (
        periods=Float64[],
        avg_velocities=Float64[],
        lower_bounds=Float64[],
        upper_bounds=Float64[],
        avg_all=avg_all,
        avg_filtered=(periods=Float64[], avg_velocities=Float64[], counts=Int[]),
        filtered_pairs=[]
    )

    period_to_bounds = Dict{Float64, Tuple{Float64, Float64, Float64}}()
    for (p, v) in zip(avg_all.periods, avg_all.avg_velocities)
        period_to_bounds[p] = (v, (1 - frac) * v, (1 + frac) * v)
    end

    avg_filtered = (
        periods=copy(avg_all.periods),
        avg_velocities=copy(avg_all.avg_velocities),
        counts=copy(avg_all.counts)
    )

    filtered_pairs = []
    for entry in all_pairs_2lambda
        key = entry.first
        periods, velocities = entry.second

        periods_f = Float64[]
        velocities_f = Float64[]

        for (p, v) in zip(periods, velocities)
            p_key = round(Float64(p), digits=period_digits)
            haskey(period_to_bounds, p_key) || continue
            _, lb, ub = period_to_bounds[p_key]
            if lb <= v <= ub
                push!(periods_f, Float64(p))
                push!(velocities_f, Float64(v))
            end
        end

        isempty(periods_f) || push!(filtered_pairs, key => (periods_f, velocities_f))
    end

    return (
        periods=copy(avg_all.periods),
        avg_velocities=copy(avg_all.avg_velocities),
        lower_bounds=(1 - frac) .* avg_all.avg_velocities,
        upper_bounds=(1 + frac) .* avg_all.avg_velocities,
        avg_all=avg_all,
        avg_filtered=avg_filtered,
        filtered_pairs=filtered_pairs
    )
end

# ╔═╡ 85f35f48-ea77-4c3a-8a34-a8818a41d85d
function load_full_stack_from_training(pair, training_idx::Int)
    basedir = basedr(training_idx)
	# "/mnt/NASdata2/Sanket_data/California_results_13032026_XJ_80sec_$(training_idx)_v28"
	# "/Data1/California_results_11022026_adaptive_filtering_$(training_idx)_v28"
    files = readdir(basedir)
    fil_files = filter(x -> occursin("k=10", x), files)
    matching = filter(f -> occursin(pair, f), fil_files)
    if !isempty(matching)
        filepath = joinpath(basedir, matching[1])
        return jldopen(filepath)["full_stack"]
    end
    return nothing
end

# ╔═╡ 8508dbd2-3910-4c52-8ac3-797811c5905d
function ac_c_per_from_training(pair, training_idx::Int)
    basedir = basedr(training_idx)
	# "/mnt/NASdata2/Sanket_data/California_results_13032026_XJ_80sec_$(training_idx)_v28/"
    files = readdir(basedir)
    fil_files = filter(x -> occursin("k=10", x), files)
    matching = filter(f -> occursin(pair, f), fil_files)
    if !isempty(matching)
        filepath = joinpath(basedir, matching[1])
        acausal_per=jldopen(filepath)["per_allo_acausal"]
		causal_per=jldopen(filepath)["per_allo_causal"]
		return acausal_per,causal_per
    end
    return nothing
end

# ╔═╡ 29266f59-439b-4d34-b207-d1b3c7b9191f
function get_acausal_causal_size(pair::String, filepath::String)
	jldfile = load(filter(x -> occursin(pair, x), readdir(filepath, join=true))[1])
	correlations = jldfile["correlations"]
	# headers = jldfile["headers"]
	# distance = jldfile["dist"]
	return size(correlations,2)
end

# ╔═╡ 10000009-0000-0000-0000-000000000001
function get_acausal_causal(pair::String, filepath::String)
	jldfile = load(filter(x -> occursin(pair, x), readdir(filepath, join=true))[1])
	correlations = jldfile["correlations"]
	headers = jldfile["headers"]
	distance = jldfile["dist"]
	return (; correlations, headers, distance)
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

# ╔═╡ 1000000c-0000-0000-0000-000000000001
function build_training_bundle(pair::Tuple{String,String},Tmax,Tmin;
        filepath="/mnt/NASData/Sanket_DRDO/station_pairs_12112025_30mins/")
    pair_name = join(pair, "_")
    data_pair_local = get_acausal_causal(pair_name, filepath)
	responsetype = Bandpass(inv(Tmax), inv(Tmin))
	designmethod = Butterworth(1)
	digfilter = digitalfilter(responsetype, designmethod; fs=inv(dt))
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
            distance=data_pair_local.distance)
end

# ╔═╡ 6897e2bd-68a7-4473-a8cc-c20320a58fbe
function build_training_bundle_new(pair_name,Tmax,Tmin;
        filepath="/mnt/NASData/Sanket_DRDO/station_pairs_12112025_30mins/")
    # pair_name = join(pair, "_")
    data_pair_local = get_acausal_causal(pair_name, filepath)
	responsetype = Bandpass(inv(Tmax), inv(Tmin))
	designmethod = Butterworth(1)
	digfilter = digitalfilter(responsetype, designmethod; fs=inv(dt))
    D1 = data_pair_local.correlations
    D1 = normalise(D1, dims=1)
    D1ac, D1c = split_causal_acausal(D1, true)
    D1ac = taper(D1ac)
    D1c = taper(D1c)
    D1fac = filtfilt(digfilter, D1ac)
    D1fc = filtfilt(digfilter, D1c)
    D1fac = Float32.(normalise(D1fac[2:end, :], dims=1))
    D1fc = Float32.(normalise(D1fc[2:end, :], dims=1))
    return (pair=pair_name, D1=Float32.(D1), D1fac=D1fac, D1fc=D1fc,
            distance=data_pair_local.distance)
end

# ╔═╡ cacf4048-cdc3-4d1f-b230-535830d34729
maindir = "/mnt/NASdata2/Sanket_data/California_XJ_13032026/"

# ╔═╡ 1000000d-0000-0000-0000-000000000001
md"### Select Station Pair"

# ╔═╡ 10000010-0000-0000-0000-000000000001
md"### Train/Test Split (Pooled)"

# ╔═╡ 10000011-0000-0000-0000-000000000001
function make_pooled_split(D1fac, D1fc; at=0.9, shuffle=true)
    # Pool causal and acausal into one matrix
    D_all = hcat(D1fac, D1fc)
    nw = size(D_all, 2)
    idx = collect(1:nw)
    shuffle && Random.shuffle!(idx)
    ntrain = round(Int, at * nw)
    train_idx = idx[1:ntrain]
    test_idx = idx[ntrain+1:end]
    return (
        D_train  = xpu(D_all[:, train_idx]),
        D_test   = xpu(D_all[:, test_idx]),
        D_all    = xpu(D_all),
        # Keep separate branch references for post-hoc analysis
        D_ac_all = xpu(D1fac),
        D_c_all  = xpu(D1fc),
    )
end

# ╔═╡ 10000013-0000-0000-0000-000000000001
nth =500

# ╔═╡ 10000014-0000-0000-0000-000000000001
tgrid = collect(-nth:nth) .* dt

# ╔═╡ 10000015-0000-0000-0000-000000000001
md"""
## Data Visualization
"""

# ╔═╡ 10000017-0000-0000-0000-000000000001
md"## Load VQ-VAE Architecture"

# ╔═╡ 10000018-0000-0000-0000-000000000001
vqvae = @ingredients("/mnt/NASData/EQData/SeismicAutoencoders/VQVAE_architecture.jl")

# ╔═╡ 10000024-0000-0000-0000-000000000001
md"## Cluster Analysis"

# ╔═╡ 10000028-0000-0000-0000-000000000001
md"### Confusion Matrix"

# ╔═╡ 1000002a-0000-0000-0000-000000000001
md"The diagonal shows window-pairs where causal and acausal branches share the same code. Off-diagonal entries reveal branch-specific clustering."

# ╔═╡ 1000002b-0000-0000-0000-000000000001
md"### Cluster Averages"

# ╔═╡ 1000002f-0000-0000-0000-000000000001
md"""## Gather Plot

Select a cluster to view all waveforms assigned to it.
"""

# ╔═╡ f0e0c5d0-0f80-4ee4-b5c7-49889adb2c6d
nmodes=vqvae_para.K*vqvae_para.T

# ╔═╡ 10000030-0000-0000-0000-000000000001
@bind selected_k Slider(1:nmodes, default=1, show_value=true)

# ╔═╡ 29bc277f-a757-4e0f-aa84-034b2f3f92b1
md"""
### MFT Dispersion Settings

Period range (s): $(@bind period_min Slider(1.0:1.0:20.0, default=3, show_value=true)) to $(@bind period_max Slider(10.0:5.0:200.0, default=15, show_value=true))

Number of periods: $(@bind n_periods Slider(10:5:120, default=90, show_value=true))

Filter bandwidth (%): $(@bind bandwidth_percent Slider(5:1:50, default=20, show_value=true))
"""

# ╔═╡ 67f58bd0-b83b-49c3-acb7-cb9e26089226
begin
period_step = 0.5
    period_min_adjusted = floor(3 / period_step) * period_step
    period_max_adjusted = ceil(8 / period_step) * period_step
    
    # Calculate number of periods with fixed 0.5 step
    n_periods_auto = Int((period_max_adjusted - period_min_adjusted) / period_step) + 1
velocity_range = (0.5, 6.0)
v_min=2.5
end

# ╔═╡ 10000032-0000-0000-0000-000000000001
md"""## Reconstruction Quality

Visualize a few reconstructions vs originals.
"""

# ╔═╡ 446f0a9b-a225-4bdd-9c1c-47f6e31c994f
md""" ## Multiple pairs training"""

# ╔═╡ 904e2d6f-590b-4067-b586-e33872dbc2b0
begin
	# pairs_used=unique(map(readdir(maindir)) do sts
	# split(sts,"-full")[1]
	# end)
	# ind_st_above_30=findall(x->x>=30,distances)
	# state_names=pairs_used[ind_st_above_30]
end

# ╔═╡ 2d5d5e97-5b31-4cfb-aecc-b738f8aae2d4
# ╠═╡ disabled = true
#=╠═╡
ntrainings=1
  ╠═╡ =#

# ╔═╡ 8e81d11e-83d2-4311-a1f8-07415c394581
results_dirs=map(1:ntrainings) do i
	"/mnt/NASdata2/Sanket_data/California_results_25032026_XJ_3_7sec_new_vqvae_$(i)_v28/"
end

# ╔═╡ 31783e01-f9ad-4b40-8565-50b8bad0cc17
# mkdir("/mnt/NASdata2/Sanket_data/Tmodels_California_vqvae/")

# ╔═╡ 24e15056-30cd-4278-9460-1529a6f3e9c0
function acausal_causal_combined(acau,cau)
acau=reverse(mean(cpu(acau),dims=2),dims=1)
cau=mean(cpu(cau),dims=2)
full=
	Flux.normalise(
	cat(acau,zeros32(1,1),cau,dims=1)
	,dims=1)
return full
end

# ╔═╡ 10000034-0000-0000-0000-000000000001
md"## Saving"

# ╔═╡ 10000035-0000-0000-0000-000000000001
# ╠═╡ disabled = true
#=╠═╡
begin
	using Dates
	timestamp = now()
	pair_str = join(selected_pair, "_")
	jldsave("SavedModels/vqvae_model-$(pair_str)-$(timestamp).jld2",
		model_state = Flux.state(cpu(model)))
	jldsave("SavedModels/vqvae_para-$(pair_str)-$(timestamp).jld2";
		vqvae_parameters)
	jldsave("SavedModels/vqvae_loss-$(pair_str)-$(timestamp).jld2";
		loss_history)
end
  ╠═╡ =#

# ╔═╡ b932e2a5-ea58-4432-8d4d-2391f36608f2
mft=@ingredients("/home/sanket/Desktop/ambient_noise_codes/MFT.jl")

# ╔═╡ db4ddb38-2938-11f1-b8e3-e5227df9322c
md"## Appendix"

# ╔═╡ 381c3a52-8a56-474b-9ca1-59efd79d5aec
function dist_using_haversine(lat1,lon1,lat2,lon2)
sta1=(lon1,lat1)
sta2=(lon2,lat2)
# d=haversine_distance(sta1[2],sta1[1],sta2[2],sta2[1])
rad_sta1=sta1.*(pi/180)
rad_sta2=sta2.*(pi/180)
Δφ=rad_sta2[2].-rad_sta1[2]
Δλ=rad_sta2[1].-rad_sta1[1]
a = (sin(Δφ/2)).^2 + cos(rad_sta1[2])*cos(rad_sta2[2])*(sin(Δλ/2))^2
d = 2 * 6371 * asin(sqrt(a))
return d
end

# ╔═╡ dc1644c2-47d1-493e-acbb-7313f01b3120
function distance_for_stations(station_pair,station_latlong)
st1,st2=split(station_pair,"_")
st1ind=findall(x->x==st1,station_latlong[!,"Station Code"])[1]
st2ind=findall(x->x==st2,station_latlong[!,"Station Code"])[1]
lat1=station_latlong[st1ind,"Latitude"]
lat2=station_latlong[st2ind,"Latitude"]
lon1=station_latlong[st1ind,"Longitude"]
lon2=station_latlong[st2ind,"Longitude"]
# lon1=station_latlong[st1ind,"Longitude "]
# lon2=station_latlong[st2ind,"Longitude "]
dist_using_haversine(lat1,lon1,lat2,lon2)
end

# ╔═╡ 9df0b89e-5c5a-49f2-8eef-099d1f1aa6e1
# Calculate distances for all pairs
distances = map(rec_names) do rec_name
    distance_for_stations(rec_name, XJ_latlong)
end

# ╔═╡ f270c30c-a119-4990-b2b5-6c0fcfa4b18b
rec_names_above_30=rec_names[distances.>30]

# ╔═╡ fcc7c1ab-6649-4d8d-9784-64efb1c11964
@bind selrecpair Select(rec_names_above_30, default="BRR_SFT")

# ╔═╡ 72c1b63b-c456-484b-a7a7-fbae8d253b7c
begin
    pairs_used = split(selrecpair, "_")
end

# ╔═╡ 8e8eb12e-d3cc-442e-adc2-b3c48e718031
begin
# Map plot for the selected station pair
region = (-125, -116.5, 34, 39)
topo = grdcut("@earth_relief_01m", region=region, figsize=(6, 8))
grdimage(topo, region=region, proj=:Mercator, color=:oleron)

st_id = sort([findall(x -> x == ps, XJ_latlong[!, "Station Code"])[1] for ps in pairs_used])
receivers = XJ_latlong[!, "Station Code"][st_id]
lon = XJ_latlong[!, "Longitude"][st_id]
lat = XJ_latlong[!, "Latitude"][st_id]
text!(String.(receivers), x=lon, y=lat .+ 0.05, font=(12, "Helvetica-Bold", :black), justify=:RB)
GMT.plot!(lon, lat, region=region, marker=:circle, ms=0.3, fill=:red, markerline=:cyan, show=true)
end

# ╔═╡ 1000000e-0000-0000-0000-000000000001
# selected_pair = ("BLA", "PKME")
# selected_pair = ("GOGA", "PKME")
# selected_pair = ("CNNC", "GOGA")
selected_pair = selrecpair
	# ("BRR","CCC")
	# ("ARC2","BGR")
	# ("BGR","BRR") 
	# ("BRR","SFT")
	# ("BINY", "GOGA")

# ╔═╡ 7dbcee8c-583b-4250-9895-759585dac882
selected_pair_=selrecpair
	# join(selected_pair, "_")

# ╔═╡ 4761cee0-09f9-4aec-8518-96e0b127216b
distances_above_30=distances[distances.>30]

# ╔═╡ 66bfe53e-6264-4d2f-90f4-261f44dd7816
distance=distance_for_stations(selrecpair,XJ_latlong)

# ╔═╡ ee4e2ae3-faf7-4b84-93e5-105cc419a57c
"""
Plot linear_average (nt×1) and subsampled (nt×2×1) where 2 = number of substates.

- linear_average: Array of shape (nt, 1) or (nt,)
- subsampled: Array of shape (nt, 2, 1) or (nt, 2)
- labels: tuple of legend labels (linear, sub1, sub2)
- show_center: draw vertical line at the center sample if nt is odd

Returns a PlutoPlotly plot.
"""
function plot_linear_and_substates(
    linear_average,
	linear_average_many, 
    subsampled;
    labels=vcat(["No subsampling"], ["λ=$i" for i in 1:size(subsampled, 2)]),
    title="",
    colors=("black","#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"),
    show_center::Bool=true,dist_km=distance,trange=nothing
)
    @assert size(linear_average, 1) == size(subsampled, 1) "nt mismatch between inputs"

    nt = size(subsampled, 1)
    x = collect(-div(nt - 1, 2):div(nt - 1, 2)) * dt
	  # Arrival times
    t2 = dist_km / 2.0   # 2 km/s
    t5 = dist_km / 5.0   # 5 km/s
    # Extract vectors
    lin = ndims(linear_average) == 1 ? linear_average : @view(linear_average[:, 1])
    lin = vec(cpu(lin))
	if trange!=nothing
	xmin,xmax=trange
	else
	xmin=x[1]
	xmax=x[end]
	end
    # Linear average trace with thicker line
    traces = [
        PlutoPlotly.scatter(
            x=x, y=lin, mode="lines", name=labels[1],
            line=attr(dash="dash", color=colors[1], width=1.5),
            opacity=1
        ),
		PlutoPlotly.scatter(
            x=x, y=cpu(vec(linear_average_many)), mode="lines", name=string("True"),
            line=attr( color=colors[1], width=4),
            opacity=0.5
        )
    ]

    # Substate traces
    for i in 1:size(subsampled, 2)
        s1 = ndims(subsampled) == 3 ? @view(subsampled[:, i, 1]) : @view(subsampled[:, i])
        y1 = vec(cpu(s1))
        push!(traces,
            PlutoPlotly.scatter(
                x=x, y=y1, mode="lines", name=labels[i+1],
                line=attr(color=colors[i+1], width=1.5),
                opacity=0.7
            )
        )
    end

    # Optional center line
    shapes = Any[]
    if show_center
        push!(shapes, attr(
            type="line", xref="x", yref="paper",
            x0=0, x1=0, y0=0, y1=1,
            line=attr(color="rgba(128,128,128,0.3)", width=1.5, dash="dash")
        ))
    end
	
    # ±2 km/s
    for sgn in (-1, 1)
        push!(shapes, attr(
            type="line", xref="x", yref="paper",
            x0=sgn*t2, x1=sgn*t2, y0=0, y1=1,
            line=attr(color="rgba(31,119,180,0.6)", width=2, dash="dot")
        ))
    end

    # ±5 km/s
    for sgn in (-1, 1)
        push!(shapes, attr(
            type="line", xref="x", yref="paper",
            x0=sgn*t5, x1=sgn*t5, y0=0, y1=1,
            line=attr(color="rgba(214,39,40,0.6)", width=2, dash="dot")
        ))
    end
	
    layout = Layout(
        title=attr(text=title, font=attr(size=27, family="Computer Modern, Latin Modern Math, serif")),
        height=700,
        width=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=attr(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.25, yanchor="top",
            font=attr(size=20, family="Computer Modern, Latin Modern Math, serif"),
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        xaxis=attr(
            title=attr(text="Time (s)", font=attr(size=23, family="Computer Modern, Latin Modern Math, serif")),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            gridwidth=1,
            showline=true,
            linewidth=1.5,
            linecolor="black",
			range=(xmin,xmax),
            mirror=true,
            tickfont=attr(size=20, family="Computer Modern, Latin Modern Math, serif"),
            zeroline=false
        ),
        yaxis=attr(
            title=attr(text="Amplitude", font=attr(size=23, family="Computer Modern, Latin Modern Math, serif")),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            gridwidth=1,
            showline=true,
            linewidth=1.5,
            linecolor="black",
            mirror=true,
            tickfont=attr(size=20, family="Computer Modern, Latin Modern Math, serif"),
            zeroline=true,
            zerolinewidth=1,
            zerolinecolor="rgba(0,0,0,0.3)"
        ),
        shapes=shapes,
        margin=attr(l=80, r=40, t=80, b=100)
    )
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 8ca289ac-d12c-4383-9538-d83c26b4ec24
begin
    # Plot all modes for the selected training and station pair
    full_modes_sel = load_mode_from_training(selrecpair, selected_training)
    full_stack_sel = load_full_stack_from_training(selrecpair, selected_training)
	ac_per,c_per=ac_c_per_from_training(selrecpair, selected_training)
	@info ac_per,c_per
	labels=vcat(["No subsampling"], ["λ=$i (ac: $(round(100*ac_per[i]; digits=2))%, c: $(round(100*c_per[i]; digits=2))%)" for i in 1:size(full_modes_sel, 2)])
    if full_modes_sel !== nothing && full_stack_sel !== nothing
        mode_labels = ["mode$(i)" for i in 1:size(full_modes_sel, 2)]
        pair_distance_idx = findfirst(x -> x == selrecpair, rec_names)
        if pair_distance_idx !== nothing
            WideCell(plot_linear_and_substates(
                full_stack_sel,
                [],
                full_modes_sel;labels=labels,
                dist_km=distances[pair_distance_idx],
                title="All Modes - Training $selected_training - Pair: $selrecpair",
                trange=[-320, 320]
            ))
        end
    end
end

# ╔═╡ 1c59dd1a-f93c-4cb4-9ca7-f5638e0f1dcb
function zero_lag_corr(C)
	nt = size(C, 1)
	# @show div(nt,2)
	C_neg = reverse(C[1:div(nt, 2), :, :], dims=1)
	# C_neg = C[1:div(nt, 2), :, :]
		# , dims=1)
	C_pos = C[div(nt, 2)+2:end, :, :]
	# C_pos = C[div(nt, 2)+1:end, :, :]
	return zero_lag_corr(C_neg, C_pos)
end

# ╔═╡ 27b79706-6029-475b-9648-91b8ac308dbe
function zero_lag_corr(A::AbstractArray, B::AbstractArray)
    size(A) == size(B) || throw(ArgumentError("A and B must have the same size"))

    # subtract mean along first dim
    A₀ = A .- mean(A, dims=1)
    B₀ = B .- mean(B, dims=1)

    # numerator = dot products along first dim
    num = sum(A₀ .* B₀, dims=1)

    # denominator = product of norms along first dim
    denom = sqrt.(sum(A₀.^2, dims=1) .* sum(B₀.^2, dims=1))

    # normalized correlation
    corr = num ./ denom

    return dropdims(corr, dims=1)
end

# ╔═╡ 45c568c0-a4c7-4c4c-b3f2-4a7c96a23230
begin
	
    n_modes_to_plot = size(full_modes_sel,2)
    n_periods_local = n_periods_auto
    mode_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    traces = PlutoPlotly.PlotlyBase.AbstractTrace[]
res_a_all=[]
res_c_all=[]
    for mode_idx in 1:n_modes_to_plot
        sig = full_modes_sel[:, mode_idx]
        # acausal=full_modes_sel[:,mode_idx]
		# causal =cluster_avg_c[:,mode_idx] 
		acausal,causal=split_causal_acausal(reshape(sig, :, 1), true)

        tr_a = mft.SeismicTrace(acausal[:,1], dt, distance)
        res_a = mft.perform_mft_analysis(
            tr_a,
            (period_min_adjusted, period_max_adjusted),
            n_periods_local,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )
		push!(res_a_all,res_a)
        tr_c = mft.SeismicTrace(causal[:, 1], dt, distance)
        res_c = mft.perform_mft_analysis(
            tr_c,
            (period_min_adjusted, period_max_adjusted),
            n_periods_local,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )
		push!(res_c_all,res_c)
		        Tmax = distance / (2.0 * v_min)
        # @info "Pair $selected_pair_: D=$(round(distance,digits=1)) km, v_min=$v_min km/s → Tmax=$(round(Tmax,digits=1)) s"

        mask      = res_c.periods .<= Tmax
        period_indices = findall(mask)
        periods_vec = Float64[]
        corrs_mode = Float64[]
        for period_idx in period_indices
            corr_val = zero_lag_corr(res_a.filtered_traces[:, period_idx], res_c.filtered_traces[:, period_idx])
            corr_scalar = corr_val isa Number ? Float64(corr_val) : Float64(only(corr_val))
            # if corr_scalar >= 0.9
                push!(periods_vec, res_c.periods[period_idx])
                push!(corrs_mode, corr_scalar)
            # end
        end


        push!(traces,
            PlutoPlotly.scatter(
                x=periods_vec,
                y=corrs_mode,
                mode="lines+markers",
                name="Mode $(mode_idx)",
                line=attr(color=mode_colors[mode_idx], width=2.5),
                marker=attr(color=mode_colors[mode_idx], size=8, line=attr(width=0.4)),
                opacity=0.8
            )
        )
    end

	for mode_idx in 1:1
        sig = full_stack_sel[:, mode_idx]
        # acausal = vec(mean(cpu(data.D_ac_all); dims=2))
    	# causal  = vec(mean(cpu(data.D_c_all);  dims=2))
		acausal,causal = split_causal_acausal(reshape(sig, :, 1), true)

        tr_a = mft.SeismicTrace(acausal[:, 1], dt, distance)
        res_a = mft.perform_mft_analysis(
            tr_a,
            (period_min_adjusted, period_max_adjusted),
            n_periods_local,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )

        tr_c = mft.SeismicTrace(causal[:, 1], dt, distance)
        res_c = mft.perform_mft_analysis(
            tr_c,
            (period_min_adjusted, period_max_adjusted),
            n_periods_local,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )
		    		        Tmax = distance / (2.0 * v_min)
        @info "Pair $selected_pair_: D=$(round(distance,digits=1)) km, v_min=$v_min km/s → Tmax=$(round(Tmax,digits=1)) s"

        mask      = res_c.periods .<= Tmax
        period_indices = findall(mask)
        periods_vec = Float64[]
        corrs_mode = Float64[]
        for period_idx in period_indices
            corr_val = zero_lag_corr(res_a.filtered_traces[:, period_idx], res_c.filtered_traces[:, period_idx])
            corr_scalar = corr_val isa Number ? Float64(corr_val) : Float64(only(corr_val))
            # if corr_scalar >= 0.9
                push!(periods_vec, res_c.periods[period_idx])
                push!(corrs_mode, corr_scalar)
            # end
        end


        push!(traces,
            PlutoPlotly.scatter(
                x=periods_vec,
                y=corrs_mode,
                mode="lines+markers",
                name="LS",
                line=attr(color="black", width=2.5),
                marker=attr(color="black", size=8, line=attr(width=0.4)),
                opacity=0.8
            )
        )
    end

    layout = Layout(
        title=attr(text="All Modes - Training - Pair: $selected_pair_ - bandwidth -$bandwidth_percent % - band -$(8-50) sec - D=$(round(distance,digits=1)) km", font=attr(size=16)),
        # width=1300,
        height=700,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=attr(
            x=1.0, xanchor="right",
            y=0.02, yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        xaxis=attr(
            title=attr(text="Period (s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        yaxis=attr(
            title=attr(text="Zero-Lag Correlation (Acausal vs Causal)", font=attr(size=18)),
            range=[0.5, 1],
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
			tickmode="linear", tick0=0.8, dtick=0.05,
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14),
            zeroline=true,
            zerolinecolor="rgba(0,0,0,0.25)"
        ),
        margin=attr(l=95, r=30, t=70, b=75)
    )

    WideCell(PlutoPlotly.plot(traces, layout))
end
	
# 	p
# end

# ╔═╡ b1896dc8-2721-421b-afc6-a6a4f55157ce
let
	# period_to_plot=period_to_plot
    n_modes_to_plot = size(full_modes_sel,2)
    n_periods_local = n_periods_auto
    mode_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    traces = PlutoPlotly.PlotlyBase.AbstractTrace[]
# res_a_all=[]
# res_c_all=[]
    for mode_idx in 1:n_modes_to_plot
  #       sig = full_modes_sel[:, mode_idx]
  #       # acausal=full_modes_sel[:,mode_idx]
		# # causal =cluster_avg_c[:,mode_idx] 
		# acausal,causal=split_causal_acausal(reshape(sig, :, 1), true)

  #       tr_a = mft.SeismicTrace(acausal[:,1], dt, distance)
  #       res_a = mft.perform_mft_analysis(
  #           tr_a,
  #           (period_min_adjusted, period_max_adjusted),
  #           n_periods_local,
  #           bandwidth_factor=bandwidth_percent / 100.0,
  #           velocity_range=velocity_range
  #       )
		# push!(res_a_all,res_a)
  #       tr_c = mft.SeismicTrace(causal[:, 1], dt, distance)
  #       res_c = mft.perform_mft_analysis(
  #           tr_c,
  #           (period_min_adjusted, period_max_adjusted),
  #           n_periods_local,
  #           bandwidth_factor=bandwidth_percent / 100.0,
  #           velocity_range=velocity_range
  #       )
		# push!(res_c_all,res_c)
		#         Tmax = distance / (2.0 * v_min)
        # @info "Pair $selected_pair_: D=$(round(distance,digits=1)) km, v_min=$v_min km/s → Tmax=$(round(Tmax,digits=1)) s"

        # mask      = res_c.periods .<= Tmax
        # period_indices = findall(mask)
        # periods_vec = Float64[]
        # corrs_mode = Float64[]
        period_idx = findfirst(x->x==period_to_plot,res_c_all[mode_idx].periods)
        #     corr_val = zero_lag_corr(res_a.filtered_traces[:, period_idx], res_c.filtered_traces[:, period_idx])
        #     corr_scalar = corr_val isa Number ? Float64(corr_val) : Float64(only(corr_val))
        #     # if corr_scalar >= 0.9
        #         push!(periods_vec, res_c.periods[period_idx])
        #         push!(corrs_mode, corr_scalar)
        #     # end
        # end
		# findfirs

		filtered_trace_full=cat(reverse(res_a_all[mode_idx].filtered_traces[:,period_idx]),res_c_all[mode_idx].filtered_traces[:,period_idx],dims=1)
		# @info size(filtered_trace_full)
        push!(traces,
            PlutoPlotly.scatter(
                x=tgrid,
                y=filtered_trace_full,
                mode="lines",
                name="Mode $(mode_idx)",
                line=attr(color=mode_colors[mode_idx], width=2.5),
                marker=attr(color=mode_colors[mode_idx], size=8, line=attr(width=0.4)),
                opacity=0.8
            )
        )
    end

	for mode_idx in 1:1
        sig = full_stack_sel[:, mode_idx]
        # acausal = vec(mean(cpu(data.D_ac_all); dims=2))
    	# causal  = vec(mean(cpu(data.D_c_all);  dims=2))
		acausal,causal = split_causal_acausal(reshape(sig, :, 1), true)

        tr_a = mft.SeismicTrace(acausal[:, 1], dt, distance)
        res_a = mft.perform_mft_analysis(
            tr_a,
            (period_min_adjusted, period_max_adjusted),
            n_periods_local,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )

        tr_c = mft.SeismicTrace(causal[:, 1], dt, distance)
        res_c = mft.perform_mft_analysis(
            tr_c,
            (period_min_adjusted, period_max_adjusted),
            n_periods_local,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )
		    		        Tmax = distance / (2.0 * v_min)
        @info "Pair $selected_pair_: D=$(round(distance,digits=1)) km, v_min=$v_min km/s → Tmax=$(round(Tmax,digits=1)) s"

       period_idx = findfirst(x->x==period_to_plot,res_c.periods)
		# @info period_idx
        #     corr_val = zero_lag_corr(res_a.filtered_traces[:, period_idx], res_c.filtered_traces[:, period_idx])
        #     corr_scalar = corr_val isa Number ? Float64(corr_val) : Float64(only(corr_val))
        #     # if corr_scalar >= 0.9
        #         push!(periods_vec, res_c.periods[period_idx])
        #         push!(corrs_mode, corr_scalar)
        #     # end
        # end
		# findfirs

		filtered_trace_full=cat(reverse(res_a.filtered_traces[:,period_idx]),res_c.filtered_traces[:,period_idx],dims=1)
		# @info size(filtered_trace_full)
        push!(traces,
            PlutoPlotly.scatter(
                x=tgrid,
                y=filtered_trace_full,
                mode="lines",
                name="No subsampling",
                line=attr(color="black", width=2.5),
                marker=attr(color="black", size=8, line=attr(width=0.4)),
                opacity=0.8
            )
        )
    end

    layout = Layout(
        title=attr(text="All Modes - Training - Pair: $selected_pair_ - bandwidth -$bandwidth_percent % - band -$(8-50) sec - D=$(round(distance,digits=1)) km", font=attr(size=16)),
        # width=1300,
        height=700,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=attr(
            x=1.0, xanchor="right",
            y=0.02, yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        xaxis=attr(
            title=attr(text="Period (s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        yaxis=attr(
            title=attr(text="Zero-Lag Correlation (Acausal vs Causal)", font=attr(size=18)),
            # range=[0.5, 1],
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
			tickmode="linear",
			# tick0=0.8, dtick=0.05,
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14),
            zeroline=true,
            zerolinecolor="rgba(0,0,0,0.25)"
        ),
        margin=attr(l=95, r=30, t=70, b=75)
    )

    WideCell(PlutoPlotly.plot(traces, layout))
end
	


# ╔═╡ 02e72481-8128-4372-b806-c77e2e23a23b
let
    mode_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    traces = PlutoPlotly.PlotlyBase.AbstractTrace[]
    avg_trace_table = DataFrame(
        mode = Int[],
        period = Float64[],
        time = Float64[],
        avg_filtered_amplitude = Float64[]
    )

    for mode_idx in 1:length(res_a_all)
        period_idx = findfirst(p -> isapprox(p, period_to_plot; atol=1e-6), res_c_all[mode_idx].periods)
        period_idx === nothing && continue

        avg_trace = 0.5 .* (
           normalise(res_a_all[mode_idx].filtered_traces[:, period_idx],dims=1) .+
            normalise(res_c_all[mode_idx].filtered_traces[:, period_idx],dims=1)
        )
        t_half = res_a_all[mode_idx].time

        for i in eachindex(t_half)
            push!(avg_trace_table, (mode_idx, res_c_all[mode_idx].periods[period_idx], t_half[i], avg_trace[i]))
        end

        col = mode_colors[mod(mode_idx - 1, length(mode_colors)) + 1]
        push!(traces,
            PlutoPlotly.scatter(
                x=t_half,
                y=avg_trace,
                mode="lines",
                name="Mode $(mode_idx)",
                line=attr(color=col, width=2.0),
                opacity=0.85
            )
        )
    end

    # LS overlay: average acausal and causal filtered traces at selected period.
    if full_stack_sel !== nothing && size(full_stack_sel, 2) >= 1
        sig_ls = full_stack_sel[:, 1]
        acausal_ls, causal_ls = split_causal_acausal(reshape(sig_ls, :, 1), true)

        tr_a_ls = mft.SeismicTrace(acausal_ls[:, 1], dt, distance)
        tr_c_ls = mft.SeismicTrace(causal_ls[:, 1], dt, distance)

        res_a_ls = mft.perform_mft_analysis(
            tr_a_ls,
            (period_min_adjusted, period_max_adjusted),
            n_periods_auto,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )
        res_c_ls = mft.perform_mft_analysis(
            tr_c_ls,
            (period_min_adjusted, period_max_adjusted),
            n_periods_auto,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )

        period_idx_ls = findfirst(p -> isapprox(p, period_to_plot; atol=1e-6), res_c_ls.periods)
        if period_idx_ls !== nothing
            avg_trace_ls = 0.5 .* (
                normalise(res_a_ls.filtered_traces[:, period_idx_ls],dims=1) .+
                normalise(res_c_ls.filtered_traces[:, period_idx_ls],dims=1)
            )

            push!(traces,
                PlutoPlotly.scatter(
                    x=res_a_ls.time,
                    y=avg_trace_ls,
                    mode="lines",
                    name="No subsampling (LS)",
                    line=attr(color="black", width=2.8, dash="dash"),
                    opacity=0.95
                )
            )
        end
    end

    layout = Layout(
        title=attr(
            text="Averaged (Acausal+Causal)/2 Filtered Traces at Period $(period_to_plot) s",
            font=attr(size=16)
        ),
        height=700,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=attr(
            title=attr(text="Time (s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        yaxis=attr(
            title=attr(text="Averaged Filtered Amplitude", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        legend=attr(
            x=1.0,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=attr(l=95, r=30, t=70, b=75)
    )

    global averaged_filtered_traces_selected_period_table = avg_trace_table

    WideCell(PlutoPlotly.plot(traces, layout))
end


# ╔═╡ 7f4f2d6e-0e1a-4683-b8f1-1b3f0f7dcf3a
let
    mode_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    traces = PlutoPlotly.PlotlyBase.AbstractTrace[]
    avg_env_table = DataFrame(
        mode = Int[],
        period = Float64[],
        time = Float64[],
        avg_envelope = Float64[]
    )

    for mode_idx in 1:length(res_a_all)
        period_idx = findfirst(p -> isapprox(p, period_to_plot; atol=1e-6), res_c_all[mode_idx].periods)
        period_idx === nothing && continue

        avg_trace = 0.5 .* (
           normalise(res_a_all[mode_idx].filtered_traces[:, period_idx],dims=1) .+
            normalise(res_c_all[mode_idx].filtered_traces[:, period_idx],dims=1)
        )
        avg_env = abs.(hilbert(avg_trace))
        t_half = res_a_all[mode_idx].time

        for i in eachindex(t_half)
            push!(avg_env_table, (mode_idx, res_c_all[mode_idx].periods[period_idx], t_half[i], avg_env[i]))
        end

        col = mode_colors[mod(mode_idx - 1, length(mode_colors)) + 1]
        push!(traces,
            PlutoPlotly.scatter(
                x=t_half,
                y=avg_env,
                mode="lines",
                name="Mode $(mode_idx)",
                line=attr(color=col, width=2.0),
                opacity=0.85
            )
        )
    end

    # LS overlay envelope at selected period.
    if full_stack_sel !== nothing && size(full_stack_sel, 2) >= 1
        sig_ls = full_stack_sel[:, 1]
        acausal_ls, causal_ls = split_causal_acausal(reshape(sig_ls, :, 1), true)

        tr_a_ls = mft.SeismicTrace(acausal_ls[:, 1], dt, distance)
        tr_c_ls = mft.SeismicTrace(causal_ls[:, 1], dt, distance)

        res_a_ls = mft.perform_mft_analysis(
            tr_a_ls,
            (period_min_adjusted, period_max_adjusted),
            n_periods_auto,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )
        res_c_ls = mft.perform_mft_analysis(
            tr_c_ls,
            (period_min_adjusted, period_max_adjusted),
            n_periods_auto,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )

        period_idx_ls = findfirst(p -> isapprox(p, period_to_plot; atol=1e-6), res_c_ls.periods)
        if period_idx_ls !== nothing
            avg_trace_ls = 0.5 .* (
                normalise(res_a_ls.filtered_traces[:, period_idx_ls],dims=1) .+
                 normalise(res_c_ls.filtered_traces[:, period_idx_ls],dims=1)
            )
            avg_env_ls = abs.(hilbert(avg_trace_ls))

            push!(traces,
                PlutoPlotly.scatter(
                    x=res_a_ls.time,
                    y=avg_env_ls,
                    mode="lines",
                    name="No subsampling (LS)",
                    line=attr(color="black", width=2.8, dash="dash"),
                    opacity=0.95
                )
            )
        end
    end

    layout = Layout(
        title=attr(
            text="Envelopes of Averaged (Acausal+Causal)/2 Filtered Traces at Period $(period_to_plot) s",
            font=attr(size=16)
        ),
        height=700,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=attr(
            title=attr(text="Time (s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        yaxis=attr(
            title=attr(text="Envelope Amplitude", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        legend=attr(
            x=1.0,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=attr(l=95, r=30, t=70, b=75)
    )

    global averaged_filtered_envelopes_selected_period_table = avg_env_table

    WideCell(PlutoPlotly.plot(traces, layout))
end


# ╔═╡ 671308fa-7136-437a-9b25-e14790407c39
let
	# period_to_plot=period_to_plot
    n_modes_to_plot = size(full_modes_sel,2)
    n_periods_local = n_periods_auto
    mode_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    traces = PlutoPlotly.PlotlyBase.AbstractTrace[]
# res_a_all=[]
# res_c_all=[]
    for mode_idx in 1:n_modes_to_plot
  #       sig = full_modes_sel[:, mode_idx]
  #       # acausal=full_modes_sel[:,mode_idx]
		# # causal =cluster_avg_c[:,mode_idx] 
		# acausal,causal=split_causal_acausal(reshape(sig, :, 1), true)

  #       tr_a = mft.SeismicTrace(acausal[:,1], dt, distance)
  #       res_a = mft.perform_mft_analysis(
  #           tr_a,
  #           (period_min_adjusted, period_max_adjusted),
  #           n_periods_local,
  #           bandwidth_factor=bandwidth_percent / 100.0,
  #           velocity_range=velocity_range
  #       )
		# push!(res_a_all,res_a)
  #       tr_c = mft.SeismicTrace(causal[:, 1], dt, distance)
  #       res_c = mft.perform_mft_analysis(
  #           tr_c,
  #           (period_min_adjusted, period_max_adjusted),
  #           n_periods_local,
  #           bandwidth_factor=bandwidth_percent / 100.0,
  #           velocity_range=velocity_range
  #       )
		# push!(res_c_all,res_c)
		#         Tmax = distance / (2.0 * v_min)
        # @info "Pair $selected_pair_: D=$(round(distance,digits=1)) km, v_min=$v_min km/s → Tmax=$(round(Tmax,digits=1)) s"

        # mask      = res_c.periods .<= Tmax
        # period_indices = findall(mask)
        # periods_vec = Float64[]
        # corrs_mode = Float64[]
        period_idx = findfirst(x->x==period_to_plot,res_c_all[mode_idx].periods)
        #     corr_val = zero_lag_corr(res_a.filtered_traces[:, period_idx], res_c.filtered_traces[:, period_idx])
        #     corr_scalar = corr_val isa Number ? Float64(corr_val) : Float64(only(corr_val))
        #     # if corr_scalar >= 0.9
        #         push!(periods_vec, res_c.periods[period_idx])
        #         push!(corrs_mode, corr_scalar)
        #     # end
        # end
		# findfirs

		filtered_trace_full=cat(reverse(res_a_all[mode_idx].envelopes[:,period_idx]),res_c_all[mode_idx].envelopes[:,period_idx],dims=1)
		@info size(filtered_trace_full)
        push!(traces,
            PlutoPlotly.scatter(
                x=tgrid,
                y=filtered_trace_full,
                mode="lines",
                name="Mode $(mode_idx)",
                line=attr(color=mode_colors[mode_idx], width=2.5),
                marker=attr(color=mode_colors[mode_idx], size=8, line=attr(width=0.4)),
                opacity=0.8
            )
        )
    end

	for mode_idx in 1:1
        sig = full_stack_sel[:, mode_idx]
        # acausal = vec(mean(cpu(data.D_ac_all); dims=2))
    	# causal  = vec(mean(cpu(data.D_c_all);  dims=2))
		acausal,causal = split_causal_acausal(reshape(sig, :, 1), true)

        tr_a = mft.SeismicTrace(acausal[:, 1], dt, distance)
        res_a = mft.perform_mft_analysis(
            tr_a,
            (period_min_adjusted, period_max_adjusted),
            n_periods_local,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )

        tr_c = mft.SeismicTrace(causal[:, 1], dt, distance)
        res_c = mft.perform_mft_analysis(
            tr_c,
            (period_min_adjusted, period_max_adjusted),
            n_periods_local,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )
		    		        Tmax = distance / (2.0 * v_min)
        @info "Pair $selected_pair_: D=$(round(distance,digits=1)) km, v_min=$v_min km/s → Tmax=$(round(Tmax,digits=1)) s"

       period_idx = findfirst(x->x==period_to_plot,res_c.periods)
		@info period_idx
        #     corr_val = zero_lag_corr(res_a.filtered_traces[:, period_idx], res_c.filtered_traces[:, period_idx])
        #     corr_scalar = corr_val isa Number ? Float64(corr_val) : Float64(only(corr_val))
        #     # if corr_scalar >= 0.9
        #         push!(periods_vec, res_c.periods[period_idx])
        #         push!(corrs_mode, corr_scalar)
        #     # end
        # end
		# findfirs

		filtered_trace_full=cat(reverse(res_a.envelopes[:,period_idx]),res_c.envelopes[:,period_idx],dims=1)
		@info size(filtered_trace_full)
        push!(traces,
            PlutoPlotly.scatter(
                x=tgrid,
                y=filtered_trace_full,
                mode="lines",
                name="No subsampling",
                line=attr(color="black", width=2.5),
                marker=attr(color="black", size=8, line=attr(width=0.4)),
                opacity=0.8
            )
        )
    end

    layout = Layout(
        title=attr(text="All Modes - Training - Pair: $selected_pair_ - bandwidth -$bandwidth_percent % - band -$(8-50) sec - D=$(round(distance,digits=1)) km", font=attr(size=16)),
        # width=1300,
        height=700,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=attr(
            x=1.0, xanchor="right",
            y=0.02, yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        xaxis=attr(
            title=attr(text="Period (s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        yaxis=attr(
            title=attr(text="Zero-Lag Correlation (Acausal vs Causal)", font=attr(size=18)),
            # range=[0.5, 1],
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
			tickmode="linear",
			# tick0=0.8, dtick=0.05,
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14),
            zeroline=true,
            zerolinecolor="rgba(0,0,0,0.25)"
        ),
        margin=attr(l=95, r=30, t=70, b=75)
    )

    WideCell(PlutoPlotly.plot(traces, layout))
end
	

# ╔═╡ 7dbb51c2-bcdc-41af-b37b-898392b7c3cd
let
traces = PlutoPlotly.PlotlyBase.AbstractTrace[]
# res_a_all=[]
# res_c_all=[]
    for mode_idx in 1:n_modes_to_plot
		# full_k = [reverse(cluster_avg_ac[:, k]); cluster_avg_c[:, k]]
		gp=mean(vcat(res_a_all[mode_idx].group_velocities,res_c_all[mode_idx].group_velocities),dims=2)
		periods=res_a_all[mode_idx].periods
  #       sig = full_modes_sel[:, mode_idx]
  #       acausal, causal = split_causal_acausal(reshape(sig, :, 1), true)

  #       tr_a = mft.SeismicTrace(acausal[:,1], dt, dist_subset[1])
  #       res_a = mft.perform_mft_analysis(
  #           tr_a,
  #           (period_min_adjusted, period_max_adjusted),
  #           n_periods_local,
  #           bandwidth_factor=bandwidth_percent / 100.0,
  #           velocity_range=velocity_range
  #       )
		# push!(res_a_all,res_a)
  #       tr_c = mft.SeismicTrace(causal[:, 1], dt, dist_subset[1])
  #       res_c = mft.perform_mft_analysis(
  #           tr_c,
  #           (period_min_adjusted, period_max_adjusted),
  #           n_periods_local,
  #           bandwidth_factor=bandwidth_percent / 100.0,
  #           velocity_range=velocity_range
  #       )
		# push!(res_c_all,res_c)
		#         Tmax = dist_subset[1] / (2.0 * v_min)
  #       @info "Pair $selrecpair: D=$(round(dist_subset[1],digits=1)) km, v_min=$v_min km/s → Tmax=$(round(Tmax,digits=1)) s"

  #       mask      = res_c.periods .<= Tmax
  #       period_indices = findall(mask)
  #       periods_vec = Float64[]
  #       corrs_mode = Float64[]
  #       for period_idx in period_indices
  #           corr_val = zero_lag_corr(res_a.filtered_traces[:, period_idx], res_c.filtered_traces[:, period_idx])
  #           corr_scalar = corr_val isa Number ? Float64(corr_val) : Float64(only(corr_val))
  #           # if corr_scalar >= 0.9
  #               push!(periods_vec, res_c.periods[period_idx])
  #               push!(corrs_mode, corr_scalar)
  #           # end
  #       end


        push!(traces,
            PlutoPlotly.scatter(
                x=periods,
                y=gp,
                mode="lines+markers",
                name="Mode $(mode_idx)",
                line=attr(color=mode_colors[mode_idx], width=2.5),
                marker=attr(color=mode_colors[mode_idx], size=8, line=attr(width=0.4)),
                opacity=0.8
            )
        )
    end

	for mode_idx in 1:1
        sig = full_stack_sel[:, mode_idx]
        acausal, causal = 
			split_causal_acausal(reshape(sig, :, 1), true)
 		 # global_avg_ac = vec(mean(cpu(data.D_ac_all); dims=2))
    		# global_avg_c  = vec(mean(cpu(data.D_c_all);  dims=2))
    	# global_mean_full=mean(vcat(acausal,causal),dims=2)[:,1]
        tr = mft.SeismicTrace(mean(cat(acausal[:,1],causal[:, 1],dims=2),dims=2)[:,1], dt, distance)
        res = mft.perform_mft_analysis(
            tr,
            (period_min_adjusted, period_max_adjusted),
            n_periods_local,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )
		map(argmax(res.envelopes,dims=1)) do arg
		@info res.periods[arg[2]],tgrid[div(size(tgrid,1),2)+1:end][arg[1]]
		end
        # tr_c = mft.SeismicTrace(causal[:, 1], dt, dist_subset[1])
        # res_c = mft.perform_mft_analysis(
        #     tr_c,
        #     (period_min_adjusted, period_max_adjusted),
        #     n_periods_local,
        #     bandwidth_factor=bandwidth_percent / 100.0,
        #     velocity_range=velocity_range
        # )
		      #   Tmax = dist_subset[1] / (2.0 * v_min)
        # @info "Pair $selrecpair: D=$(round(dist_subset[1],digits=1)) km, v_min=$v_min km/s → Tmax=$(round(Tmax,digits=1)) s"

        # mask      = res_c.periods .<= Tmax
        # period_indices = findall(mask)
        # periods_vec = Float64[]
        # corrs_mode = Float64[]
        # for period_idx in period_indices
        #     corr_val = zero_lag_corr(res_a.filtered_traces[:, period_idx], res_c.filtered_traces[:, period_idx])
        #     corr_scalar = corr_val isa Number ? Float64(corr_val) : Float64(only(corr_val))
            # if corr_scalar >= 0.9
                # push!(periods_vec, res_c.periods[period_idx])
                # push!(corrs_mode, corr_scalar)
            # end
        # end


        push!(traces,
            PlutoPlotly.scatter(
                x=res.periods,
                y=res.group_velocities,
                mode="lines+markers",
                name="LS",
                line=attr(color="black", width=2.5),
                marker=attr(color="black", size=8, line=attr(width=0.4)),
                opacity=0.8
            )
        )
    end
	
layout = Layout(
        title=attr(text="All Modes - Training  - Pair: $selected_pair_ - bandwidth -$bandwidth_percent % - band  sec - D=$(round(distance,digits=1)) km", font=attr(size=16)),
        # width=1300,
        height=700,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=attr(
            x=1.0, xanchor="right",
            y=0.02, yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        xaxis=attr(
            title=attr(text="Period (s)", font=attr(size=18)),
            # showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        yaxis=attr(
            title=attr(text="Group velocity (km/s)", font=attr(size=18)),
            range=[0.5, 6],
            # showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
			# tickmode="linear", tick0=0.8, dtick=0.05,
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14),
            zeroline=true,
            zerolinecolor="rgba(0,0,0,0.25)"
        ),
        margin=attr(l=95, r=30, t=70, b=75)
    )

    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 96102e0b-2f7c-497f-ba7e-762b546ada17
let
    function dispersion_from_avg_filtered(res_a, res_c, dist_km; velocity_rng=velocity_range)
        t = res_a.time
        periods = res_a.periods
        vmin, vmax = velocity_rng
        tmin = dist_km / vmax
        tmax = dist_km / vmin

        candidates = DataFrame(
            period = Float64[],
            pick_time = Float64[]
        )

        for pi in eachindex(periods)
            avg_trace = 0.5 .* (res_a.filtered_traces[:, pi] .+ res_c.filtered_traces[:, pi])
            env = abs.(hilbert(avg_trace))

            win_idx = findall(tt -> tmin <= tt <= tmax, t)
            isempty(win_idx) && continue

            env_win = env[win_idx]
            pks = Peaks.findmaxima(env_win)
            lm_idx = Int.(pks.indices)
            isempty(lm_idx) && continue

            # Keep top amplitude local maxima per period as continuity candidates.
            ord = sortperm(env_win[lm_idx], rev=true)
            keep = lm_idx[ord[1:min(3, length(ord))]]
            for k in keep
                t_pick = Float64(t[win_idx[k]])
                t_pick <= 0 && continue
                push!(candidates, (periods[pi], t_pick))
            end
        end

        out_period = Float64[]
        out_time = Float64[]
        out_vel = Float64[]

        if nrow(candidates) > 0
            periods_sorted = sort(unique(candidates.period))
            prev_t = nothing

            for p in periods_sorted
                idxp = findall(candidates.period .== p)
                isempty(idxp) && continue
                times_p = candidates.pick_time[idxp]

                chosen = if isnothing(prev_t)
                    idxp[argmax(times_p)]
                else
                    idxp[argmin(abs.(times_p .- prev_t))]
                end

                t_pick = candidates.pick_time[chosen]
                push!(out_period, p)
                push!(out_time, t_pick)
                push!(out_vel, dist_km / t_pick)
                prev_t = t_pick
            end
        end

        return (period=out_period, time=out_time, velocity=out_vel)
    end

    mode_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    traces = PlutoPlotly.PlotlyBase.AbstractTrace[]
    avg_filtered_dispersion_table = DataFrame(
        mode = Int[],
        period = Float64[],
        pick_time = Float64[],
        group_velocity = Float64[]
    )

    for mode_idx in 1:length(res_a_all)
        disp = dispersion_from_avg_filtered(res_a_all[mode_idx], res_c_all[mode_idx], distance)
        isempty(disp.period) && continue

        for i in eachindex(disp.period)
            push!(avg_filtered_dispersion_table, (mode_idx, disp.period[i], disp.time[i], disp.velocity[i]))
        end

        col = mode_colors[mod(mode_idx - 1, length(mode_colors)) + 1]
        push!(traces,
            PlutoPlotly.scatter(
                x=disp.period,
                y=disp.velocity,
                mode="markers+lines",
                name="Mode $(mode_idx)",
                marker=attr(size=8, color=col, opacity=0.85),
                line=attr(color=col, width=2.0),
                text=string.("M", mode_idx, " V=", round.(disp.velocity; digits=2), " km/s", " T=", round.(disp.time; digits=2), " s"),
                hoverinfo="text+x+y"
            )
        )
    end

    # LS overlay: average acausal/causal first, then run per-period averaged-filter pick.
    if full_stack_sel !== nothing && size(full_stack_sel, 2) >= 1
        sig_ls = full_stack_sel[:, 1]
        acausal_ls, causal_ls = split_causal_acausal(reshape(sig_ls, :, 1), true)

        tr_a_ls = mft.SeismicTrace(acausal_ls[:, 1], dt, distance)
        tr_c_ls = mft.SeismicTrace(causal_ls[:, 1], dt, distance)

        res_a_ls = mft.perform_mft_analysis(
            tr_a_ls,
            (period_min_adjusted, period_max_adjusted),
            n_periods_auto,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )
        res_c_ls = mft.perform_mft_analysis(
            tr_c_ls,
            (period_min_adjusted, period_max_adjusted),
            n_periods_auto,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )

        disp_ls = dispersion_from_avg_filtered(res_a_ls, res_c_ls, distance)
        if !isempty(disp_ls.period)
            for i in eachindex(disp_ls.period)
                push!(avg_filtered_dispersion_table, (0, disp_ls.period[i], disp_ls.time[i], disp_ls.velocity[i]))
            end

            push!(traces,
                PlutoPlotly.scatter(
                    x=disp_ls.period,
                    y=disp_ls.velocity,
                    mode="markers+lines",
                    name="No subsampling (LS)",
                    marker=attr(size=9, color="black", opacity=0.95),
                    line=attr(color="black", width=2.8, dash="dash"),
                    text=string.("LS V=", round.(disp_ls.velocity; digits=2), " km/s", " T=", round.(disp_ls.time; digits=2), " s"),
                    hoverinfo="text+x+y"
                )
            )
        end
    end

    layout = Layout(
        title=attr(
            text="Dispersion from Period-wise Averaged Filtered Traces (All Modes)",
            font=attr(size=16)
        ),
        height=680,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=attr(
            title=attr(text="Period (s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        yaxis=attr(
            title=attr(text="Group Velocity (km/s)", font=attr(size=18)),
            showgrid=true,
			range=[1,6],
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        legend=attr(
            x=1.0,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=attr(l=95, r=30, t=70, b=75)
    )

    global avg_filtered_dispersion_table_all_modes = avg_filtered_dispersion_table

    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 799c0703-d9e7-4f4e-a205-9fd9aee6ad62
function envelope_per_period_local_maxima(res_envelope; peaks_to_output::Int=3)    
    pks = Peaks.findmaxima(res_envelope)
    idx = Int.(pks.indices)[1:min(peaks_to_output, length(pks.indices))]
    return idx
end

# ╔═╡ c041b902-17e2-4e4b-9bb5-f924cc129993
function select_continuous_period_picks(df::DataFrame; period_col::Symbol=:period, time_col::Symbol=:time_mean)
    nrow(df) == 0 && return df

    periods_sorted = sort(unique(df[!, period_col]))
    selected_idx = Int[]
    prev_t = nothing

    for p in periods_sorted
        idx_period = findall(df[!, period_col] .== p)
        isempty(idx_period) && continue

        if isnothing(prev_t)
            local_times = df[idx_period, time_col]
            chosen_local = idx_period[argmax(local_times)]
        else
            local_times = df[idx_period, time_col]
            d = abs.(local_times .- prev_t)
            chosen_local = idx_period[argmin(d)]
        end

        push!(selected_idx, chosen_local)
        prev_t = df[chosen_local, time_col]
    end

    return df[selected_idx, :]
end

# ╔═╡ c906f5e8-76d2-4632-a623-6d4dac36b4d7
function local_maxima_per_mode(res_a, res_c, mode; peaks_to_output::Int=3, time_frac_tol::Float64=0.01)
    periods = res_a[mode].periods
    res_a_env = res_a[mode].envelopes
    res_c_env = res_c[mode].envelopes
    time_vec = res_a[mode].time
    
    results = DataFrame(
        period = Float64[],
        time_acausal = Float64[],
        time_causal = Float64[],
        time_mean = Float64[]
    )
    
    for period_idx in 1:length(periods)
        res_ac_p_lm = envelope_per_period_local_maxima(res_a_env[:, period_idx]; peaks_to_output=peaks_to_output)
        res_c_p_lm = envelope_per_period_local_maxima(res_c_env[:, period_idx]; peaks_to_output=peaks_to_output)
        
        # Convert peak indices to times
        times_ac = Float64.(time_vec[res_ac_p_lm])
        times_c = Float64.(time_vec[res_c_p_lm])
        
        # Apply 1% criterion to all combinations
        for t_ac in times_ac, t_c in times_c
			# @info t_ac,t_c
            t_mean = 0.5 * (t_ac + t_c)
            t_mean <= 0 && continue
            if abs(t_ac - t_c) <= time_frac_tol * t_mean
                push!(results, (
                    period = periods[period_idx],
                    time_acausal = t_ac,
                    time_causal = t_c,
                    time_mean = t_mean
                ))
            end
        end
    end
    
    return results
end

# ╔═╡ 238ab25d-6d5e-454b-a002-268c02c6889c
let
    all_mode_results = DataFrame(
        mode = Int[],
        period = Float64[],
        time_acausal = Float64[],
        time_causal = Float64[],
        time_mean = Float64[]
    )
    
    for k in 1:n_modes_to_plot
        mode_result = local_maxima_per_mode(res_a_all, res_c_all, k, time_frac_tol=0.05)
        for row in eachrow(mode_result)
            push!(all_mode_results, (k, row.period, row.time_acausal, row.time_causal, row.time_mean))
        end
    end
    
    mode_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    traces = PlutoPlotly.PlotlyBase.AbstractTrace[]
    selected_results = DataFrame(
        mode = Int[],
        period = Float64[],
        time_acausal = Float64[],
        time_causal = Float64[],
        time_mean = Float64[]
    )

    for mode_idx in unique(all_mode_results.mode)
        mask = all_mode_results.mode .== mode_idx
        subset = all_mode_results[mask, :]
        subset = select_continuous_period_picks(subset; period_col=:period, time_col=:time_mean)

        for i in 1:nrow(subset)
            push!(selected_results, (
                mode_idx,
                subset.period[i],
                subset.time_acausal[i],
                subset.time_causal[i],
                subset.time_mean[i]
            ))
        end
        
        col = mode_colors[mod(mode_idx-1, length(mode_colors)) + 1]
        @info mode_idx,subset.period,distance./subset.time_mean
        push!(traces,
            PlutoPlotly.scatter(
                x=subset.period,
                y=subset.time_mean,
                mode="markers+lines",
                name="Mode $(mode_idx)",
                marker=attr(size=8, color=col, opacity=0.8),
                text=string.("M", mode_idx," ", round.(distance./subset.time_mean; digits=2)," km/s", " Tac:",subset.time_acausal," Tc:",subset.time_causal),
                hoverinfo="text+x+y"
            )
        )
    end
    
    layout = Layout(
        title=attr(
            text="Valid Peak Times (1% combined acausal/causal): Period vs Time - All Modes",
            font=attr(size=16)
        ),
        height=680,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=attr(
            title=attr(text="Period (s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        yaxis=attr(
            title=attr(text="Peak Time (s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        legend=attr(
            x=1.0,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=attr(l=95, r=30, t=70, b=75)
    )
    
    global all_valid_peaks_table = selected_results
    
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 82f69b08-103a-466b-88b8-c978179c7d61
let
    all_mode_results = DataFrame(
        mode = Int[],
        period = Float64[],
        time_acausal = Float64[],
        time_causal = Float64[],
        time_mean = Float64[]
    )

    for k in 1:n_modes_to_plot
        mode_result = local_maxima_per_mode(res_a_all, res_c_all, k, time_frac_tol=0.05)
        for row in eachrow(mode_result)
            push!(all_mode_results, (k, row.period, row.time_acausal, row.time_causal, row.time_mean))
        end
    end

    mode_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    traces = PlutoPlotly.PlotlyBase.AbstractTrace[]
    reduced_results = DataFrame(
        mode = Int[],
        period = Float64[],
        time_acausal = Float64[],
        time_causal = Float64[],
        time_mean = Float64[],
        group_velocity = Float64[]
    )

    for mode_idx in unique(all_mode_results.mode)
        subset = all_mode_results[all_mode_results.mode .== mode_idx, :]
        subset = select_continuous_period_picks(subset; period_col=:period, time_col=:time_mean)
        gv = distance ./ subset.time_mean

        for i in 1:nrow(subset)
            push!(reduced_results, (
                mode_idx,
                subset.period[i],
                subset.time_acausal[i],
                subset.time_causal[i],
                subset.time_mean[i],
                gv[i]
            ))
        end

        col = mode_colors[mod(mode_idx - 1, length(mode_colors)) + 1]
        push!(traces,
            PlutoPlotly.scatter(
                x=subset.period,
                y=gv,
                mode="markers+lines",
                name="Mode $(mode_idx)",
                marker=attr(size=8, color=col, opacity=0.85),
                line=attr(color=col, width=2.0),
                text=string.(
                    "M", mode_idx,
                    " V=", round.(gv; digits=2), " km/s",
                    " Tac=", round.(subset.time_acausal; digits=2),
                    " Tc=", round.(subset.time_causal; digits=2)
                ),
                hoverinfo="text+x+y"
            )
        )
    end

    # Linear-average (no subsampling) reference picks on the same plot.
    if full_stack_sel !== nothing && size(full_stack_sel, 2) >= 1
        sig_ls = full_stack_sel[:, 1]
        acausal_ls, causal_ls = split_causal_acausal(reshape(sig_ls, :, 1), true)

        # For LS: average acausal and causal first, then run a single MFT.
        sig_ls_avg = 0.5 .* (acausal_ls[:, 1] .+ causal_ls[:, 1])
        tr_ls = mft.SeismicTrace(sig_ls_avg,
            dt,
            distance)

        res_ls = mft.perform_mft_analysis(
            tr_ls,
            (period_min_adjusted, period_max_adjusted),
            n_periods_auto,
            bandwidth_factor=bandwidth_percent / 100.0,
            velocity_range=velocity_range
        )

        ls_candidates = DataFrame(
            period = Float64[],
            time_pick = Float64[]
        )

        for period_idx in eachindex(res_ls.periods)
            lm_idx = envelope_per_period_local_maxima(
                res_ls.envelopes[:, period_idx];
                peaks_to_output=3
            )
            isempty(lm_idx) && continue

            for idx in lm_idx
                t_pick = Float64(res_ls.time[idx])
                push!(ls_candidates, (res_ls.periods[period_idx], t_pick))
            end
        end

        ls_selected = select_continuous_period_picks(ls_candidates; period_col=:period, time_col=:time_pick)
        ls_result = DataFrame(
            period = Float64[],
            time_acausal = Float64[],
            time_causal = Float64[],
            time_mean = Float64[]
        )

        for row in eachrow(ls_selected)
            push!(ls_result, (
                row.period,
                row.time_pick,
                row.time_pick,
                row.time_pick
            ))
        end

        if nrow(ls_result) > 0
            gv_ls = distance ./ ls_result.time_mean

            for i in 1:nrow(ls_result)
                push!(reduced_results, (
                    0,
                    ls_result.period[i],
                    ls_result.time_acausal[i],
                    ls_result.time_causal[i],
                    ls_result.time_mean[i],
                    gv_ls[i]
                ))
            end

            push!(traces,
                PlutoPlotly.scatter(
                    x=ls_result.period,
                    y=gv_ls,
                    mode="markers+lines",
                    name="No subsampling (LS)",
                    marker=attr(size=9, color="black", opacity=0.95),
                    line=attr(color="black", width=2.8, dash="dash"),
                    text=string.(
                        "LS V=", round.(gv_ls; digits=2), " km/s",
                        " Tavg=", round.(ls_result.time_mean; digits=2)
                    ),
                    hoverinfo="text+x+y"
                )
            )
        end
    end

    layout = Layout(
        title=attr(
            text="Group Velocity from Valid Picks (All Modes)",
            font=attr(size=16)
        ),
        height=680,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=attr(
            title=attr(text="Period (s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        yaxis=attr(
            title=attr(text="Group Velocity (km/s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
			range=[1,6],
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        legend=attr(
            x=1.0,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=attr(l=95, r=30, t=70, b=75)
    )

    global all_valid_picks_velocity_table = reduced_results

    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 68e1c79d-b2f1-474d-8c29-1cec5aef8a13
function extract_dispersion_for_pair_rank_by_corrfiltered_min_one(
    pair::String,
    # ranking_df::DataFrame,
    distance::Float64,
    # matched_df::DataFrame,
	pmin,pmax;
    ntrains::Int         = ntrainings,
    dt_val::Float64      = dt,
    period_step::Float64 = 0.5,
    bandwidth_frac::Float64= bandwidth_percent/100,
    velocity_rng::Tuple{Float64,Float64} = (0.5, 6.0),
    v_min_cutoff::Float64 = v_min,
    rel_dev_tol::Float64  = 0.15,
    corr_threshold::Float64 = 0.85,
    mft_module            = mft
)
    pm_adj = floor(pmin / period_step) * period_step
    px_adj = ceil(pmax / period_step) * period_step
	
    n_p = Int((px_adj - pm_adj) / period_step) + 1
    Tmax = distance / (2.0 * v_min_cutoff)

    per_training = []
    for ti in 1:ntrains
        # ti > length(mtuple) && continue
        m_all = load_mode_from_training(pair, ti)
        m_all === nothing && continue

        # midx = Int(mtuple[ti])
        # midx <= size(m_all, 2) || continue
		for midx in 1:size(m_all,2)
        sig = m_all[:, midx]
        acausal, causal = split_causal_acausal(reshape(sig, :, 1), true)
        tr_a = mft_module.SeismicTrace(acausal[:, 1], dt_val, distance)
        tr_c = mft_module.SeismicTrace(causal[:, 1], dt_val, distance)

        res_a = mft_module.perform_mft_analysis(
            tr_a, (pm_adj, px_adj), n_p;
            bandwidth_factor = bandwidth_frac,
            velocity_range = velocity_rng
        )
        res_c = mft_module.perform_mft_analysis(
            tr_c, (pm_adj, px_adj), n_p;
            bandwidth_factor = bandwidth_frac,
            velocity_range = velocity_rng
        )

        corrs = map(eachindex(res_c.periods)) do pi
            cv = zero_lag_corr(
                res_a.filtered_traces[:, pi],
                res_c.filtered_traces[:, pi]
            )
            cv isa Number ? Float64(cv) : Float64(only(cv))
        end

        valid = res_c.periods .<= Tmax
        push!(per_training, (
            periods = res_c.periods[valid],
            group_velocities = res_c.group_velocities[valid],
            corrs = corrs[valid]
        ))
    end
	end
	per_training
    isempty(per_training) && return nothing

	avg_periods = Float64[]
avg_velocities = Float64[]
ref_periods = per_training[1].periods
n_modes = length(per_training)

minlen = minimum(length.(getfield.(per_training, :corrs)))
periods_common = ref_periods[1:minlen]

# for pi in 1:minlen
min_group_velocities = Float64[]
selected_periods = Float64[]
min_mode_indices = Int[]

for pi in 1:minlen
    # Collect (group_velocity, mode_index) for modes above threshold
    vals = [(r.group_velocities[pi], mi) for (mi, r) in enumerate(per_training)
            if pi <= length(r.group_velocities) &&
               isfinite(r.group_velocities[pi]) &&
               r.corrs[pi] >= corr_threshold]
    if !isempty(vals)
        gv, mi = findmin(first.(vals))  # findmin returns (min_value, index_in_vals)
        push!(selected_periods, periods_common[pi])
        push!(min_group_velocities, gv)
        push!(min_mode_indices, vals[mi][2])  # mode index in per_training
    end
end
    return (selected_periods, min_group_velocities,min_mode_indices)
# end
end

# ╔═╡ 06668d2a-d5ae-4a20-a46f-cb48130ace54
extract_dispersion_for_pair_rank_by_corrfiltered_min_one(String(selrecpair),distance,3,8)

# ╔═╡ 8a7d0899-93c9-4b14-8cb4-664ae24a478b
"""
    build_all_pairs_dispersion_from_rankings(
        all_pair_rankings, pair_names, distances, matched_df;
        kwargs...
    ) -> Vector of (pair, 0) => (periods, velocities)

Sweeps every pair that appears in `all_pair_rankings`, calls
`extract_dispersion_for_pair_rank1`, and collects results in the standard
`(pair, combo_idx) => (periods, velocities)` format used by tomography helpers.
"""
function build_all_pairs_dispersion_from_rankings(
    pair_names::Vector{String},
    distances::Vector{Float64},pmin,pmax;
    ntrains::Int         = ntrainings,
    dt_val::Float64      = dt,
    period_step::Float64 = 0.5,
    bandwidth_frac::Float64              = bandwidth_percent/100,
    velocity_rng::Tuple{Float64,Float64} = (0.5, 6.0),
    v_min_cutoff::Float64  = v_min,
    tol::Float64           = 0.15,
	corr_threshold=0.85,
    mft_module             = mft
)
    results = []

    for (i, pair) in enumerate(pair_names)
        # haskey(all_pair_rankings, pair) || continue
        # ranking_df = all_pair_rankings[pair]
        # isempty(ranking_df) && continue

        D = distances[i]
		out=extract_dispersion_for_pair_rank_by_corrfiltered_min_one(
            pair, D,pmin,pmax;
            ntrains        = ntrains,
            dt_val         = dt_val,
            period_step    = period_step,
            bandwidth_frac = bandwidth_frac,
            velocity_rng   = velocity_rng,
            v_min_cutoff   = v_min_cutoff,
            corr_threshold=corr_threshold,
            mft_module     = mft_module
        )

        if out !== nothing && !isempty(out[1])
            push!(results, (pair, 0) => out)
            @info "$(lpad(i,3)) $pair → $(length(out[1])) periods | T=$(round(out[1][1],digits=1))–$(round(out[1][end],digits=1)) s"
        else
            @warn "$(lpad(i,3)) $pair → no dispersion curve"
        end
    end

    @info "build_all_pairs_dispersion_from_rankings: $(length(results))/$(length(pair_names)) pairs returned curves"
    return results
end

# ╔═╡ 0ab22a84-ce39-4986-8030-a150cd80c895
disp_rank1=build_all_pairs_dispersion_from_rankings(String.(rec_names_above_30[sortperm(distances_above_30)]),
    distances_above_30[sortperm(distances_above_30)],3,8;v_min_cutoff=v_min, bandwidth_frac=bandwidth_percent/100,corr_threshold=0.9)

# ╔═╡ dc42ebe1-2b60-485d-99cb-a42c6cc288ab
WideCell(mft.plot_avg_dispersion_curves_all(
    filter(x->occursin(selrecpair,x.first[1]),disp_rank1),
    trange=[3,8],
    # colors="redblue",
	visibility=true,
    velocity_range=[1,5]
))

# ╔═╡ 5cfafb23-0904-45af-8004-8607ab17e338
WideCell(mft.plot_avg_dispersion_curves_all(
    disp_rank1,
    trange=[3, 8],
    colors="redblue",
	visibility=true,
    velocity_range=[0.5,6]
))

# ╔═╡ 374e8b44-56c0-4620-919f-325d631e5c2d
# Step 1: drop periods where group velocity > velocity_threshold_rank1
disp_rank1_vthr = let
    out = []
    for (key, (periods, velocities)) in disp_rank1
        mask = velocities .<= velocity_threshold_rank1
        any(mask) || continue
        push!(out, key => (periods[mask], velocities[mask]))
    end
    @info "disp_rank1_vthr: $(length(out)) pairs after v <= $(velocity_threshold_rank1) km/s"
    out
end

# ╔═╡ d7a3ceef-fea6-4230-9788-0c44726a3904
# Step 2: mean-window filter -+rank1_frac of the period-wise mean velocity
disp_rank1_filtered = filter_by_mean_velocity_window(disp_rank1_vthr; frac=rank1_frac)

# ╔═╡ a71cd612-18fe-45cd-a3f5-c873cc09e096
let
    prs1 = disp_rank1_filtered.filtered_pairs
    pavg = disp_rank1_filtered.periods
    vavg = disp_rank1_filtered.avg_velocities
    vlb = disp_rank1_filtered.lower_bounds
    vub = disp_rank1_filtered.upper_bounds

    if isempty(pavg)
        md"No points remain after mean ±$(frac)% filtering."
    else
        traces = Vector{typeof(PlutoPlotly.scatter(x=pavg, y=vavg, mode="lines"))}()

        for entry in prs1
            periods_i, velocities_i = entry.second
            push!(traces, PlutoPlotly.scatter(
                x=periods_i,
                y=velocities_i,
                mode="lines",
                name="Filtered pairs",
                showlegend=false,
                line=attr(color="rgba(100,100,100,0.20)", width=1)
            ))
        end

        push!(traces, PlutoPlotly.scatter(
            x=pavg,
            y=vavg,
            mode="lines+markers",
            name="Average velocity",
            line=attr(color="#1f77b4", width=3),
            marker=attr(size=5, color="#1f77b4")
        ))

        push!(traces, PlutoPlotly.scatter(
            x=pavg,
            y=vlb,
            mode="lines",
            name="Lower bound (-25%)",
            line=attr(color="#d62728", width=2, dash="dash")
        ))

        push!(traces, PlutoPlotly.scatter(
            x=pavg,
            y=vub,
            mode="lines",
            name="Upper bound (+25%)",
            line=attr(color="#d62728", width=2, dash="dash")
        ))

        layout = Layout(
            title=attr(text="Filtered Curves + Period-wise Mean ±$(rank1_frac)% Bounds"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            # width=980,
            # height=620,
            xaxis=attr(
                title="Period (s)",
                showgrid=true,
                gridcolor="rgba(128,128,128,0.2)",
                showline=true,
				range=
				# [5,30],
				[3,10],
                linecolor="black",
                mirror=true
            ),
            yaxis=attr(
                title="Velocity (km/s)",
                range=[0.5, 5.0],
                showgrid=true,
                gridcolor="rgba(128,128,128,0.2)",
                showline=true,
                linecolor="black",
                mirror=true
            ),
            legend=attr(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.2,
                yanchor="top"
            ),
            margin=attr(l=80, r=30, t=70, b=90)
        )

        WideCell(PlutoPlotly.plot(traces, layout))
    end
end

# ╔═╡ 4f94500d-8b28-4c7b-81a3-6012fca2a5e6
WideCell(mft.plot_avg_dispersion_curves_all(
    disp_rank1_filtered.filtered_pairs,
    trange=[3, 8],
    colors="redblue",
	# visibility=true,
    velocity_range=[0.5,6]
))

# ╔═╡ 4f1de2d4-6240-4f26-8cf1-0cbfb9cd527c
function score_curve_coverage_consistency(
    periods::AbstractVector,
    velocities::AbstractVector,
    corrs::AbstractVector;
    corr_threshold::Float64=0.85,
    w_cov::Float64=0.45,
    w_cons::Float64=0.30,
    w_smooth::Float64=0.15,
    w_corr::Float64=0.10
)
    n_total = length(periods)
    n_total == 0 && return (
        score=-Inf,
        periods=Float64[],
        velocities=Float64[],
        corrs=Float64[],
        coverage_frac=0.0,
        consistency=0.0,
        smoothness=0.0,
        corr_mean=0.0
    )

    mask = map(1:n_total) do i
        isfinite(Float64(velocities[i])) &&
        isfinite(Float64(corrs[i])) &&
        Float64(corrs[i]) >= corr_threshold
    end

    any(mask) || return (
        score=-Inf,
        periods=Float64[],
        velocities=Float64[],
        corrs=Float64[],
        coverage_frac=0.0,
        consistency=0.0,
        smoothness=0.0,
        corr_mean=0.0
    )

    p = Float64.(periods[mask])
    v = Float64.(velocities[mask])
    c = Float64.(corrs[mask])

    coverage_frac = length(v) / n_total
    vmean = mean(v)
    vstd = length(v) > 1 ? std(v) : 0.0
    consistency = 1.0 / (1.0 + vstd / (abs(vmean) + eps()))

    if length(v) > 2
        dvdp = diff(v) ./ diff(p)
        rough = std(dvdp) / (abs(vmean) + eps())
        smoothness = 1.0 / (1.0 + rough)
    else
        smoothness = 1.0
    end

    corr_mean = mean(c)
    score = w_cov * coverage_frac + w_cons * consistency + w_smooth * smoothness + w_corr * corr_mean

    return (
        score=score,
        periods=p,
        velocities=v,
        corrs=c,
        coverage_frac=coverage_frac,
        consistency=consistency,
        smoothness=smoothness,
        corr_mean=corr_mean
    )
end

# ╔═╡ 5a6662ed-7a40-465e-a7eb-f6736f5b592b
function extract_dispersion_for_pair_top1_ranked(
    pair::String,
    distance::Float64,
    pmin,
    pmax;
    ntrains::Int=ntrainings,
    dt_val::Float64=dt,
    period_step::Float64=0.5,
    bandwidth_frac::Float64=bandwidth_percent / 100,
    velocity_rng::Tuple{Float64, Float64}=(0.5, 6.0),
    v_min_cutoff::Float64=v_min,
    corr_threshold::Float64=0.85,
    mft_module=mft
)
    pm_adj = floor(pmin / period_step) * period_step
    px_adj = ceil(pmax / period_step) * period_step
    n_p = Int((px_adj - pm_adj) / period_step) + 1
    Tmax = distance / (2.0 * v_min_cutoff)

    ranking_rows = DataFrame(
        pair=String[],
        training=Int[],
        mode=Int[],
        score=Float64[],
        coverage_frac=Float64[],
        consistency=Float64[],
        smoothness=Float64[],
        corr_mean=Float64[],
        n_kept=Int[]
    )

    best = nothing
    best_meta = nothing

    for ti in 1:ntrains
        m_all = load_mode_from_training(pair, ti)
        m_all === nothing && continue

        for midx in 1:size(m_all, 2)
            sig = m_all[:, midx]
            acausal, causal = split_causal_acausal(reshape(sig, :, 1), true)
            tr_a = mft_module.SeismicTrace(acausal[:, 1], dt_val, distance)
            tr_c = mft_module.SeismicTrace(causal[:, 1], dt_val, distance)

            res_a = mft_module.perform_mft_analysis(
                tr_a, (pm_adj, px_adj), n_p;
                bandwidth_factor=bandwidth_frac,
                velocity_range=velocity_rng
            )
            res_c = mft_module.perform_mft_analysis(
                tr_c, (pm_adj, px_adj), n_p;
                bandwidth_factor=bandwidth_frac,
                velocity_range=velocity_rng
            )

            corrs = map(eachindex(res_c.periods)) do pi
                cv = zero_lag_corr(res_a.filtered_traces[:, pi], res_c.filtered_traces[:, pi])
                cv isa Number ? Float64(cv) : Float64(only(cv))
            end

            valid = res_c.periods .<= Tmax
            sc = score_curve_coverage_consistency(
                res_c.periods[valid],
                res_c.group_velocities[valid],
                corrs[valid];
                corr_threshold=corr_threshold
            )

            push!(ranking_rows, (
                pair,
                ti,
                midx,
                sc.score,
                sc.coverage_frac,
                sc.consistency,
                sc.smoothness,
                sc.corr_mean,
                length(sc.periods)
            ))

            if isfinite(sc.score) && (best === nothing || sc.score > best.score)
                best = sc
                best_meta = (training=ti, mode=midx)
            end
        end
    end

    best === nothing && return nothing

    return (
        periods=best.periods,
        velocities=best.velocities,
        meta=best_meta,
        ranking=ranking_rows
    )
end

# ╔═╡ 657c32f6-3cef-40a6-a04f-3116cb25291d
function build_all_pairs_dispersion_top1_ranked(
    pair_names::Vector{String},
    distances::Vector{Float64},
    pmin,
    pmax;
    ntrains::Int=ntrainings,
    dt_val::Float64=dt,
    period_step::Float64=0.5,
    bandwidth_frac::Float64=bandwidth_percent / 100,
    velocity_rng::Tuple{Float64, Float64}=(0.5, 6.0),
    v_min_cutoff::Float64=v_min,
    corr_threshold::Float64=0.85,
    mft_module=mft
)
    results = []
    ranking_all = DataFrame(
        pair=String[],
        training=Int[],
        mode=Int[],
        score=Float64[],
        coverage_frac=Float64[],
        consistency=Float64[],
        smoothness=Float64[],
        corr_mean=Float64[],
        n_kept=Int[],
        selected=Bool[]
    )

    for (i, pair) in enumerate(pair_names)
        out = extract_dispersion_for_pair_top1_ranked(
            pair,
            distances[i],
            pmin,
            pmax;
            ntrains=ntrains,
            dt_val=dt_val,
            period_step=period_step,
            bandwidth_frac=bandwidth_frac,
            velocity_rng=velocity_rng,
            v_min_cutoff=v_min_cutoff,
            corr_threshold=corr_threshold,
            mft_module=mft_module
        )

        if out === nothing || isempty(out.periods)
            @warn "$(lpad(i,3)) $pair -> no ranked curve"
            continue
        end

        push!(results, (pair, 0) => (out.periods, out.velocities))

        rank_df = out.ranking
        selected_mask = (rank_df.training .== out.meta.training) .& (rank_df.mode .== out.meta.mode)
        rank_df2 = DataFrame(rank_df)
        rank_df2.selected = selected_mask
        append!(ranking_all, rank_df2)

        @info "$(lpad(i,3)) $pair -> top1(training=$(out.meta.training), mode=$(out.meta.mode)) | kept=$(length(out.periods)) | score=$(round(maximum(rank_df.score), digits=4))"
    end

    @info "build_all_pairs_dispersion_top1_ranked: $(length(results))/$(length(pair_names)) pairs returned"
    return (results=results, ranking=ranking_all)
end

# ╔═╡ 71eea2a2-7878-4812-a41d-52b63d52ca5c
function build_all_pairs_dispersion_top1_ranked_threshold(
    pair_names::Vector{String},
    distances::Vector{Float64},
    pmin,
    pmax;
    score_threshold::Float64=0.60,
    ntrains::Int=ntrainings,
    dt_val::Float64=dt,
    period_step::Float64=0.5,
    bandwidth_frac::Float64=bandwidth_percent / 100,
    velocity_rng::Tuple{Float64, Float64}=(0.5, 6.0),
    v_min_cutoff::Float64=v_min,
    corr_threshold::Float64=0.85,
    mft_module=mft
)
    out = build_all_pairs_dispersion_top1_ranked(
        pair_names,
        distances,
        pmin,
        pmax;
        ntrains=ntrains,
        dt_val=dt_val,
        period_step=period_step,
        bandwidth_frac=bandwidth_frac,
        velocity_rng=velocity_rng,
        v_min_cutoff=v_min_cutoff,
        corr_threshold=corr_threshold,
        mft_module=mft_module
    )

    ranking = out.ranking
    selected = ranking[ranking.selected .== true, :]
    selected_keep = selected[selected.score .>= score_threshold, :]
    keep_pairs = Set(String.(selected_keep.pair))

    filtered_results = [x for x in out.results if String(x.first[1]) in keep_pairs]
    filtered_ranking = ranking[[String(ranking.pair[i]) in keep_pairs for i in 1:nrow(ranking)], :]

    @info "Top1 threshold filter: kept $(length(filtered_results))/$(length(out.results)) pairs with selected score >= $(score_threshold)"

    return (
        results=filtered_results,
        ranking=filtered_ranking,
        selected=selected_keep,
        score_threshold=score_threshold
    )
end

# ╔═╡ a8c6aa8d-bf3f-4d09-8a4d-9fcb676171bd
ranked_top1_all_pairs = build_all_pairs_dispersion_top1_ranked_threshold(
    String.(rec_names_above_30[sortperm(distances_above_30)]),
    distances_above_30[sortperm(distances_above_30)],
    3,
    8;
    v_min_cutoff=v_min,
    bandwidth_frac=bandwidth_percent / 100,
    corr_threshold=0.85,score_threshold=0.6
)

# ╔═╡ 9729858c-6cf7-4502-9447-67cf53ba1f35
disp_rank1_top1 = ranked_top1_all_pairs.results

# ╔═╡ b4d22adf-e0f4-40cb-ac8f-0299cfcccf34
WideCell(mft.plot_avg_dispersion_curves_all(
    disp_rank1_top1,
    trange=[3, 8],
    colors="redblue",
    visibility=true,
    velocity_range=[0.5, 6]
))

# ╔═╡ c5ef03a5-08c8-4f31-a915-f31dc1d860c1
ranked_top1_table = ranked_top1_all_pairs.ranking

# ╔═╡ 14d83bc3-ff02-47ec-a3e5-b143121a5040
sel = ranked_top1_table[ranked_top1_table.selected .== true, :]

# ╔═╡ 86f19653-7c18-4678-b39c-212e8940c7b2
minimum(sel.score), maximum(sel.score)

# ╔═╡ f07576a0-dac8-4791-9758-6823025faa1b
sel[sel.score.>=0.6,:]

# ╔═╡ 2dfbe495-3f31-4d80-8f10-5cbfcb8976d9
function dispersion_from_added_filtered_traces(
    res_a,
    res_c,
    distance::Float64,
    Tmax::Float64;
    velocity_rng::Tuple{Float64, Float64}=(0.5, 6.0)
)
    min_vel, max_vel = velocity_rng
    tmin = distance / max_vel
    tmax = distance / min_vel

    candidates = DataFrame(
        period = Float64[],
        pick_time = Float64[],
        corr = Float64[]
    )

    for pi in eachindex(res_c.periods)
        res_c.periods[pi] <= Tmax || continue

        corr_val = zero_lag_corr(res_a.filtered_traces[:, pi], res_c.filtered_traces[:, pi])
        corr_scalar = corr_val isa Number ? Float64(corr_val) : Float64(only(corr_val))

        add_trace = res_a.filtered_traces[:, pi] .+ res_c.filtered_traces[:, pi]
        env = abs.(hilbert(add_trace))
        win_idx = findall(tt -> tmin <= tt <= tmax, res_c.time)
        isempty(win_idx) && continue

        env_win = env[win_idx]
        pks = Peaks.findmaxima(env_win)
        lm_idx = Int.(pks.indices)
        isempty(lm_idx) && continue

        ord = sortperm(env_win[lm_idx], rev=true)
        keep = lm_idx[ord[1:min(3, length(ord))]]
        for k in keep
            t_pick = Float64(res_c.time[win_idx[k]])
            t_pick > 0 || continue
            push!(candidates, (Float64(res_c.periods[pi]), t_pick, corr_scalar))
        end
    end

    periods_out = Float64[]
    velocities_out = Float64[]
    corrs_out = Float64[]

    if nrow(candidates) > 0
        periods_sorted = sort(unique(candidates.period))
        prev_t = nothing

        for p in periods_sorted
            idxp = findall(candidates.period .== p)
            isempty(idxp) && continue
            times_p = candidates.pick_time[idxp]

            chosen = if isnothing(prev_t)
                idxp[argmax(times_p)]
            else
                idxp[argmin(abs.(times_p .- prev_t))]
            end

            t_pick = candidates.pick_time[chosen]
            gv = distance / t_pick
            isfinite(gv) || continue

            push!(periods_out, p)
            push!(velocities_out, gv)
            push!(corrs_out, candidates.corr[chosen])
            prev_t = t_pick
        end
    end

    return (periods=periods_out, velocities=velocities_out, corrs=corrs_out)
end

# ╔═╡ a6765026-7fcb-44c7-bf18-77e9f6dc0fce
function extract_dispersion_for_pair_top1_ranked_added(
    pair::String,
    distance::Float64,
    pmin,
    pmax;
    ntrains::Int=ntrainings,
    dt_val::Float64=dt,
    period_step::Float64=0.5,
    bandwidth_frac::Float64=bandwidth_percent / 100,
    velocity_rng::Tuple{Float64, Float64}=(0.5, 6.0),
    v_min_cutoff::Float64=v_min,
    corr_threshold::Float64=0.85,
    mft_module=mft
)
    pm_adj = floor(pmin / period_step) * period_step
    px_adj = ceil(pmax / period_step) * period_step
    n_p = Int((px_adj - pm_adj) / period_step) + 1
    Tmax = distance / (2.0 * v_min_cutoff)

    ranking_rows = DataFrame(
        pair=String[],
        training=Int[],
        mode=Int[],
        score=Float64[],
        coverage_frac=Float64[],
        consistency=Float64[],
        smoothness=Float64[],
        corr_mean=Float64[],
        n_kept=Int[]
    )

    best = nothing
    best_meta = nothing

    for ti in 1:ntrains
        m_all = load_mode_from_training(pair, ti)
        m_all === nothing && continue

        for midx in 1:size(m_all, 2)
            sig = m_all[:, midx]
            acausal, causal = split_causal_acausal(reshape(sig, :, 1), true)
            tr_a = mft_module.SeismicTrace(acausal[:, 1], dt_val, distance)
            tr_c = mft_module.SeismicTrace(causal[:, 1], dt_val, distance)

            res_a = mft_module.perform_mft_analysis(
                tr_a, (pm_adj, px_adj), n_p;
                bandwidth_factor=bandwidth_frac,
                velocity_range=velocity_rng
            )
            res_c = mft_module.perform_mft_analysis(
                tr_c, (pm_adj, px_adj), n_p;
                bandwidth_factor=bandwidth_frac,
                velocity_range=velocity_rng
            )

            cand = dispersion_from_added_filtered_traces(
                res_a,
                res_c,
                distance,
                Tmax;
                velocity_rng=velocity_rng
            )

            sc = score_curve_coverage_consistency(
                cand.periods,
                cand.velocities,
                cand.corrs;
                corr_threshold=corr_threshold
            )

            push!(ranking_rows, (
                pair,
                ti,
                midx,
                sc.score,
                sc.coverage_frac,
                sc.consistency,
                sc.smoothness,
                sc.corr_mean,
                length(sc.periods)
            ))

            if isfinite(sc.score) && (best === nothing || sc.score > best.score)
                best = sc
                best_meta = (training=ti, mode=midx)
            end
        end
    end

    best === nothing && return nothing

    return (
        periods=best.periods,
        velocities=best.velocities,
        meta=best_meta,
        ranking=ranking_rows
    )
end

# ╔═╡ ba0f0ba1-e9c3-4f22-af62-5f0f87346ee3
function build_all_pairs_dispersion_top1_ranked_added(
    pair_names::Vector{String},
    distances::Vector{Float64},
    pmin,
    pmax;
    ntrains::Int=ntrainings,
    dt_val::Float64=dt,
    period_step::Float64=0.5,
    bandwidth_frac::Float64=bandwidth_percent / 100,
    velocity_rng::Tuple{Float64, Float64}=(0.5, 6.0),
    v_min_cutoff::Float64=v_min,
    corr_threshold::Float64=0.85,
    mft_module=mft
)
    results = []
    ranking_all = DataFrame(
        pair=String[],
        training=Int[],
        mode=Int[],
        score=Float64[],
        coverage_frac=Float64[],
        consistency=Float64[],
        smoothness=Float64[],
        corr_mean=Float64[],
        n_kept=Int[],
        selected=Bool[]
    )

    for (i, pair) in enumerate(pair_names)
        out = extract_dispersion_for_pair_top1_ranked_added(
            pair,
            distances[i],
            pmin,
            pmax;
            ntrains=ntrains,
            dt_val=dt_val,
            period_step=period_step,
            bandwidth_frac=bandwidth_frac,
            velocity_rng=velocity_rng,
            v_min_cutoff=v_min_cutoff,
            corr_threshold=corr_threshold,
            mft_module=mft_module
        )

        if out === nothing || isempty(out.periods)
            @warn "$(lpad(i,3)) $pair -> no ranked added-trace curve"
            continue
        end

        push!(results, (pair, 0) => (out.periods, out.velocities))

        rank_df = out.ranking
        selected_mask = (rank_df.training .== out.meta.training) .& (rank_df.mode .== out.meta.mode)
        rank_df2 = DataFrame(rank_df)
        rank_df2.selected = selected_mask
        append!(ranking_all, rank_df2)

        @info "$(lpad(i,3)) $pair -> top1 added(training=$(out.meta.training), mode=$(out.meta.mode)) | kept=$(length(out.periods)) | score=$(round(maximum(rank_df.score), digits=4))"
    end

    @info "build_all_pairs_dispersion_top1_ranked_added: $(length(results))/$(length(pair_names)) pairs returned"
    return (results=results, ranking=ranking_all)
end

# ╔═╡ 90d8dd45-ea1f-46e1-a3b9-0e5cbc2ee7ea
function build_all_pairs_dispersion_top1_ranked_added_threshold(
    pair_names::Vector{String},
    distances::Vector{Float64},
    pmin,
    pmax;
    score_threshold::Float64=0.60,
    ntrains::Int=ntrainings,
    dt_val::Float64=dt,
    period_step::Float64=0.5,
    bandwidth_frac::Float64=bandwidth_percent / 100,
    velocity_rng::Tuple{Float64, Float64}=(0.5, 6.0),
    v_min_cutoff::Float64=v_min,
    corr_threshold::Float64=0.85,
    mft_module=mft
)
    out = build_all_pairs_dispersion_top1_ranked_added(
        pair_names,
        distances,
        pmin,
        pmax;
        ntrains=ntrains,
        dt_val=dt_val,
        period_step=period_step,
        bandwidth_frac=bandwidth_frac,
        velocity_rng=velocity_rng,
        v_min_cutoff=v_min_cutoff,
        corr_threshold=corr_threshold,
        mft_module=mft_module
    )

    ranking = out.ranking
    selected = ranking[ranking.selected .== true, :]
    selected_keep = selected[selected.score .>= score_threshold, :]
    keep_pairs = Set(String.(selected_keep.pair))

    filtered_results = [x for x in out.results if String(x.first[1]) in keep_pairs]
    filtered_ranking = ranking[[String(ranking.pair[i]) in keep_pairs for i in 1:nrow(ranking)], :]

    @info "Added-trace top1 threshold filter: kept $(length(filtered_results))/$(length(out.results)) pairs with selected score >= $(score_threshold)"

    return (
        results=filtered_results,
        ranking=filtered_ranking,
        selected=selected_keep,
        score_threshold=score_threshold
    )
end

# ╔═╡ 74b713d0-266f-4f1f-b5dc-5b4c8f5176cf
ranked_top1_added_all_pairs = build_all_pairs_dispersion_top1_ranked_added_threshold(
    String.(rec_names_above_30[sortperm(distances_above_30)]),
    distances_above_30[sortperm(distances_above_30)],
    3,
    8;
    v_min_cutoff=v_min,
    bandwidth_frac=bandwidth_percent / 100,
    corr_threshold=0.85,
    score_threshold=0.5
)

# ╔═╡ 2b86cdb9-badf-4f59-ba10-c19b54f5a646
disp_rank1_top1_added = ranked_top1_added_all_pairs.results

# ╔═╡ 6fd3168e-6d8d-4de8-9aa0-f7aecdc127af
ranked_top1_added_table = ranked_top1_added_all_pairs.ranking

# ╔═╡ 453d85d2-b9e6-426d-88b1-853402ecf176
WideCell(mft.plot_avg_dispersion_curves_all(
    disp_rank1_top1_added,
    trange=[3, 8],
    colors="redblue",
    visibility=true,
    velocity_range=[0.5, 6]
))

# ╔═╡ cb7707d6-50ed-4157-8563-897a39582f50
# Step 1: drop periods where group velocity > velocity_threshold_rank1
disp_rank1_vthr_top1 = let
    out = []
    for (key, (periods, velocities)) in disp_rank1_top1_added
        mask = velocities .<= velocity_threshold_rank1
        any(mask) || continue
        push!(out, key => (periods[mask], velocities[mask]))
    end
    @info "disp_rank1_vthr: $(length(out)) pairs after v <= $(velocity_threshold_rank1) km/s"
    out
end

# ╔═╡ 23717119-86fc-4745-9af4-286ec0212633
disp_rank1_filtered_top1 = filter_by_mean_velocity_window(disp_rank1_vthr_top1; frac=rank1_frac)

# ╔═╡ 7b3cbebe-5c0d-4508-9707-283b0d735e44
records=
	# attach_geolocation_df(avg_results_sel_pairs_8_50,california_latlongdat)
	attach_geolocation_df(disp_rank1_filtered_top1.filtered_pairs,XJ_latlong)

# ╔═╡ 8eb30f68-da67-4b29-8011-9a024decbc0a
let
    prs1 = disp_rank1_filtered_top1.filtered_pairs
    pavg = disp_rank1_filtered_top1.periods
    vavg = disp_rank1_filtered_top1.avg_velocities
    vlb = disp_rank1_filtered_top1.lower_bounds
    vub = disp_rank1_filtered_top1.upper_bounds

    if isempty(pavg)
        md"No points remain after mean ±$(frac)% filtering."
    else
        traces = Vector{typeof(PlutoPlotly.scatter(x=pavg, y=vavg, mode="lines"))}()

        for entry in prs1
            periods_i, velocities_i = entry.second
            push!(traces, PlutoPlotly.scatter(
                x=periods_i,
                y=velocities_i,
                mode="lines",
                name="Filtered pairs",
                showlegend=false,
                line=attr(color="rgba(100,100,100,0.20)", width=1)
            ))
        end

        push!(traces, PlutoPlotly.scatter(
            x=pavg,
            y=vavg,
            mode="lines+markers",
            name="Average velocity",
            line=attr(color="#1f77b4", width=3),
            marker=attr(size=5, color="#1f77b4")
        ))

        push!(traces, PlutoPlotly.scatter(
            x=pavg,
            y=vlb,
            mode="lines",
            name="Lower bound (-25%)",
            line=attr(color="#d62728", width=2, dash="dash")
        ))

        push!(traces, PlutoPlotly.scatter(
            x=pavg,
            y=vub,
            mode="lines",
            name="Upper bound (+25%)",
            line=attr(color="#d62728", width=2, dash="dash")
        ))

        layout = Layout(
            title=attr(text="Filtered Curves + Period-wise Mean ±$(rank1_frac)% Bounds"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            # width=980,
            # height=620,
            xaxis=attr(
                title="Period (s)",
                showgrid=true,
                gridcolor="rgba(128,128,128,0.2)",
                showline=true,
				range=
				# [5,30],
				[3,10],
                linecolor="black",
                mirror=true
            ),
            yaxis=attr(
                title="Velocity (km/s)",
                range=[0.5, 5.0],
                showgrid=true,
                gridcolor="rgba(128,128,128,0.2)",
                showline=true,
                linecolor="black",
                mirror=true
            ),
            legend=attr(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.2,
                yanchor="top"
            ),
            margin=attr(l=80, r=30, t=70, b=90)
        )

        WideCell(PlutoPlotly.plot(traces, layout))
    end
end

# ╔═╡ 402d44d9-ec83-440f-8616-7efa091ae59b
WideCell(mft.plot_avg_dispersion_curves_all(
    disp_rank1_filtered_top1.filtered_pairs,
    trange=[3, 8],
    colors="redblue",
	visibility=true,
    velocity_range=[0.5,6]
))

# ╔═╡ 1f5710ff-e36a-4906-8bc1-209d1eefeb71
function extract_dispersion_for_pair_top1_ranked_linearavg(
    pair::String,
    distance::Float64,
    pmin,
    pmax;
    ntrains::Int=ntrainings,
    dt_val::Float64=dt,
    period_step::Float64=0.5,
    bandwidth_frac::Float64=bandwidth_percent / 100,
    velocity_rng::Tuple{Float64, Float64}=(0.5, 6.0),
    velocity_ceiling::Float64=4.5,
    velocity_floor::Float64=1.0,
    v_min_cutoff::Float64=v_min,
    corr_threshold::Float64=0.85,
    mft_module=mft
)
    pm_adj = floor(pmin / period_step) * period_step
    px_adj = ceil(pmax / period_step) * period_step
    n_p = Int((px_adj - pm_adj) / period_step) + 1
    Tmax = distance / (2.0 * v_min_cutoff)

    ranking_rows = DataFrame(
        pair=String[],
        training=Int[],
        mode=Int[],
        score=Float64[],
        coverage_frac=Float64[],
        consistency=Float64[],
        smoothness=Float64[],
        corr_mean=Float64[],
        n_kept=Int[]
    )

    best = nothing
    best_meta = nothing

    for ti in 1:ntrains
        stack = load_full_stack_from_training(pair, ti)
        stack === nothing && continue
        size(stack, 2) >= 1 || continue

        begin
            midx = 0
            sig = stack[:, 1]
            acausal, causal = split_causal_acausal(reshape(sig, :, 1), true)

            # Linear-average method: average branches first, then run one MFT.
            sig_avg = 0.5 .* (acausal[:, 1] .+ causal[:, 1])
            tr_avg = mft_module.SeismicTrace(sig_avg, dt_val, distance)
            res_avg = mft_module.perform_mft_analysis(
                tr_avg,
                (pm_adj, px_adj),
                n_p;
                bandwidth_factor=bandwidth_frac,
                velocity_range=velocity_rng
            )

                valid = (res_avg.periods .<= Tmax) .&
                    isfinite.(res_avg.group_velocities) .&
                    (res_avg.group_velocities .>= velocity_floor) .&
                    (res_avg.group_velocities .<= velocity_ceiling)
            p = Float64.(res_avg.periods[valid])
            v = Float64.(res_avg.group_velocities[valid])
            # No acausal-causal branch correlation for single-branch linear-average curve.
            c = ones(Float64, length(p))

            sc = score_curve_coverage_consistency(
                p,
                v,
                c;
                corr_threshold=corr_threshold
            )

            push!(ranking_rows, (
                pair,
                ti,
                midx,
                sc.score,
                sc.coverage_frac,
                sc.consistency,
                sc.smoothness,
                sc.corr_mean,
                length(sc.periods)
            ))

            if isfinite(sc.score) && (best === nothing || sc.score > best.score)
                best = sc
                best_meta = (training=ti, mode=midx)
            end
        end
    end

    best === nothing && return nothing

    return (
        periods=best.periods,
        velocities=best.velocities,
        meta=best_meta,
        ranking=ranking_rows
    )
end

# ╔═╡ 909b15e4-f8eb-4eff-a4d4-f737e94a8a26
function build_all_pairs_dispersion_top1_ranked_linearavg(
    pair_names::Vector{String},
    distances::Vector{Float64},
    pmin,
    pmax;
    ntrains::Int=ntrainings,
    dt_val::Float64=dt,
    period_step::Float64=0.5,
    bandwidth_frac::Float64=bandwidth_percent / 100,
    velocity_rng::Tuple{Float64, Float64}=(0.5, 6.0),
    velocity_ceiling::Float64=4.5,
    velocity_floor::Float64=1.0,
    v_min_cutoff::Float64=v_min,
    corr_threshold::Float64=0.85,
    mft_module=mft
)
    results = []
    ranking_all = DataFrame(
        pair=String[],
        training=Int[],
        mode=Int[],
        score=Float64[],
        coverage_frac=Float64[],
        consistency=Float64[],
        smoothness=Float64[],
        corr_mean=Float64[],
        n_kept=Int[],
        selected=Bool[]
    )

    for (i, pair) in enumerate(pair_names)
        out = extract_dispersion_for_pair_top1_ranked_linearavg(
            pair,
            distances[i],
            pmin,
            pmax;
            ntrains=ntrains,
            dt_val=dt_val,
            period_step=period_step,
            bandwidth_frac=bandwidth_frac,
            velocity_rng=velocity_rng,
            velocity_ceiling=velocity_ceiling,
            velocity_floor=velocity_floor,
            v_min_cutoff=v_min_cutoff,
            corr_threshold=corr_threshold,
            mft_module=mft_module
        )

        if out === nothing || isempty(out.periods)
            @warn "$(lpad(i,3)) $pair -> no ranked linear-avg curve"
            continue
        end

        push!(results, (pair, 0) => (out.periods, out.velocities))

        rank_df = out.ranking
        selected_mask = (rank_df.training .== out.meta.training) .& (rank_df.mode .== out.meta.mode)
        rank_df2 = DataFrame(rank_df)
        rank_df2.selected = selected_mask
        append!(ranking_all, rank_df2)

        @info "$(lpad(i,3)) $pair -> top1 linearavg(training=$(out.meta.training), mode=$(out.meta.mode)) | kept=$(length(out.periods)) | score=$(round(maximum(rank_df.score), digits=4))"
    end

    @info "build_all_pairs_dispersion_top1_ranked_linearavg: $(length(results))/$(length(pair_names)) pairs returned"
    return (results=results, ranking=ranking_all)
end

# ╔═╡ 3d526cd0-c6cd-4637-8132-5c5920484de3
function build_all_pairs_dispersion_top1_ranked_linearavg_threshold(
    pair_names::Vector{String},
    distances::Vector{Float64},
    pmin,
    pmax;
    score_threshold::Float64=0.60,
    ntrains::Int=ntrainings,
    dt_val::Float64=dt,
    period_step::Float64=0.5,
    bandwidth_frac::Float64=bandwidth_percent / 100,
    velocity_rng::Tuple{Float64, Float64}=(0.5, 6.0),
    velocity_ceiling::Float64=4.5,
    velocity_floor::Float64=1.0,
    v_min_cutoff::Float64=v_min,
    corr_threshold::Float64=0.85,
    mft_module=mft
)
    out = build_all_pairs_dispersion_top1_ranked_linearavg(
        pair_names,
        distances,
        pmin,
        pmax;
        ntrains=ntrains,
        dt_val=dt_val,
        period_step=period_step,
        bandwidth_frac=bandwidth_frac,
        velocity_rng=velocity_rng,
        velocity_ceiling=velocity_ceiling,
        velocity_floor=velocity_floor,
        v_min_cutoff=v_min_cutoff,
        corr_threshold=corr_threshold,
        mft_module=mft_module
    )

    ranking = out.ranking
    selected = ranking[ranking.selected .== true, :]
    selected_keep = selected[selected.score .>= score_threshold, :]
    keep_pairs = Set(String.(selected_keep.pair))

    filtered_results = [x for x in out.results if String(x.first[1]) in keep_pairs]
    filtered_ranking = ranking[[String(ranking.pair[i]) in keep_pairs for i in 1:nrow(ranking)], :]

    @info "Linear-avg top1 threshold filter: kept $(length(filtered_results))/$(length(out.results)) pairs with selected score >= $(score_threshold)"

    return (
        results=filtered_results,
        ranking=filtered_ranking,
        selected=selected_keep,
        score_threshold=score_threshold
    )
end

# ╔═╡ 28f6f4bc-47c4-4fd5-8f2b-a326d9f8e45a
ranked_top1_linearavg_all_pairs = build_all_pairs_dispersion_top1_ranked_linearavg_threshold(
    String.(rec_names_above_30[sortperm(distances_above_30)]),
    distances_above_30[sortperm(distances_above_30)],
    3,
    8;
    v_min_cutoff=v_min,
    bandwidth_frac=bandwidth_percent / 100,
    corr_threshold=0.85,
    velocity_ceiling=velocity_threshold_rank1,
    velocity_floor=1.0,
    score_threshold=0.5
)

# ╔═╡ 2a553a6f-8901-4fdf-9f34-39e0f6354427
disp_rank1_top1_linearavg = ranked_top1_linearavg_all_pairs.results

# ╔═╡ 3a7106f8-72a4-4397-b66e-cbe33ad5469c
ranked_top1_linearavg_table = ranked_top1_linearavg_all_pairs.ranking

# ╔═╡ c1f5f3e4-3f2c-4681-b4f7-84f3b2c257a7
function compare_with_filtered_plot(
    disp_curves;
    velocity_threshold::Float64=velocity_threshold_rank1,
    frac::Float64=rank1_frac,
    title_prefix::String="Linear-average Top1"
)
    disp_vthr = let
        out = []
        for (key, (periods, velocities)) in disp_curves
            mask = velocities .<= velocity_threshold
            any(mask) || continue
            push!(out, key => (periods[mask], velocities[mask]))
        end
        out
    end

    filt = filter_by_mean_velocity_window(disp_vthr; frac=frac)
    avg_unf = average_dispersion_same_periods(disp_vthr)
    avg_fil = average_dispersion_same_periods(filt.filtered_pairs)

    traces = PlutoPlotly.PlotlyBase.AbstractTrace[]

    if !isempty(avg_unf.periods)
        push!(traces,
            PlutoPlotly.scatter(
                x=avg_unf.periods,
                y=avg_unf.avg_velocities,
                mode="lines+markers",
                name="Unfiltered average",
                line=attr(color="#d62728", width=2.8),
                marker=attr(size=8, color="#d62728")
            )
        )
    end

    if !isempty(avg_fil.periods)
        push!(traces,
            PlutoPlotly.scatter(
                x=avg_fil.periods,
                y=avg_fil.avg_velocities,
                mode="lines+markers",
                name="Filtered average",
                line=attr(color="#1f77b4", width=2.8, dash="dash"),
                marker=attr(size=8, color="#1f77b4")
            )
        )
    end

    layout = Layout(
        title=attr(
            text="$(title_prefix): Unfiltered vs Filtered (v<=$(round(velocity_threshold, digits=2)), frac=$(round(frac, digits=2)))",
            font=attr(size=16)
        ),
        height=680,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=attr(
            title=attr(text="Period (s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        yaxis=attr(
            title=attr(text="Group Velocity (km/s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        legend=attr(
            x=1.0,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=attr(l=95, r=30, t=70, b=75)
    )

    return (
        plot=PlutoPlotly.plot(traces, layout),
        disp_vthr=disp_vthr,
        filtered=filt,
        avg_unfiltered=avg_unf,
        avg_filtered=avg_fil
    )
end

# ╔═╡ ebd0c89b-9f95-45e3-a68a-17fd2bfcda10
linearavg_filtered_compare = compare_with_filtered_plot(
    disp_rank1_top1_linearavg;
    velocity_threshold=velocity_threshold_rank1,
    frac=rank1_frac,
    title_prefix="Linear-average Top1"
)

# ╔═╡ 61ca1735-af39-4400-8931-bf4fb2f3f3f8
WideCell(linearavg_filtered_compare.plot)

# ╔═╡ 1dc90fe7-a789-4378-abf7-a4f179f9dc26
function filter_and_average_dispersion(
    disp_curves;
    velocity_threshold::Float64=velocity_threshold_rank1,
    frac::Float64=rank1_frac
)
    disp_vthr = let
        out = []
        for (key, (periods, velocities)) in disp_curves
            mask = velocities .<= velocity_threshold
            any(mask) || continue
            push!(out, key => (periods[mask], velocities[mask]))
        end
        out
    end

    filt = filter_by_mean_velocity_window(disp_vthr; frac=frac)
    avg_fil = average_dispersion_same_periods(filt.filtered_pairs)

    return (disp_vthr=disp_vthr, filtered=filt, avg_filtered=avg_fil)
end

# ╔═╡ 91d7c8e2-bad3-4b79-bb7e-973b59e59011
    linearavg = filter_and_average_dispersion(
        disp_rank1_top1_linearavg;
        velocity_threshold=velocity_threshold_rank1,
        frac=rank1_frac
    ).filtered

# ╔═╡ 13d0dbf1-1f5a-4e2e-83a8-b5a7f11e6d2d
WideCell(mft.plot_avg_dispersion_curves_all(
    linearavg.filtered_pairs,
    trange=[3, 8],
    colors="redblue",
    visibility=true,
    velocity_range=[0.5, 6]
))

# ╔═╡ 168f0229-38ad-4365-8803-6b426735de1a
compare_all_methods_filtered = let
    base = filter_and_average_dispersion(
        disp_rank1_top1;
        velocity_threshold=velocity_threshold_rank1,
        frac=rank1_frac
    )
    added = filter_and_average_dispersion(
        disp_rank1_top1_added;
        velocity_threshold=velocity_threshold_rank1,
        frac=rank1_frac
    )
    linearavg = filter_and_average_dispersion(
        disp_rank1_top1_linearavg;
        velocity_threshold=velocity_threshold_rank1,
        frac=rank1_frac
    )

    traces = PlutoPlotly.PlotlyBase.AbstractTrace[]

    if !isempty(base.avg_filtered.periods)
        push!(traces,
            PlutoPlotly.scatter(
                x=base.avg_filtered.periods,
                y=base.avg_filtered.avg_velocities,
                mode="lines+markers",
                name="Filtered: branch-correlation top1",
                line=attr(color="#1f77b4", width=2.6),
                marker=attr(size=8, color="#1f77b4")
            )
        )
    end

    if !isempty(added.avg_filtered.periods)
        push!(traces,
            PlutoPlotly.scatter(
                x=added.avg_filtered.periods,
                y=added.avg_filtered.avg_velocities,
                mode="lines+markers",
                name="Filtered: added-trace top1",
                line=attr(color="#d62728", width=2.6, dash="dash"),
                marker=attr(size=8, color="#d62728")
            )
        )
    end

    if !isempty(linearavg.avg_filtered.periods)
        push!(traces,
            PlutoPlotly.scatter(
                x=linearavg.avg_filtered.periods,
                y=linearavg.avg_filtered.avg_velocities,
                mode="lines+markers",
                name="Filtered: linear-average top1",
                line=attr(color="#2ca02c", width=2.6, dash="dot"),
                marker=attr(size=8, color="#2ca02c")
            )
        )
    end

    layout = Layout(
        title=attr(
            text="Filtered Comparison of All Methods (v<=$(round(velocity_threshold_rank1, digits=2)), frac=$(round(rank1_frac, digits=2)))",
            font=attr(size=16)
        ),
        height=700,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=attr(
            title=attr(text="Period (s)", font=attr(size=18)),
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        yaxis=attr(
            title=attr(text="Group Velocity (km/s)", font=attr(size=18)),
            showgrid=true,
			range=[1,6],
            gridcolor="rgba(128,128,128,0.2)",
            showline=true,
            linecolor="black",
            linewidth=1.5,
            mirror=true,
            tickfont=attr(size=14)
        ),
        legend=attr(
            x=1.0,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=attr(l=95, r=30, t=70, b=75)
    )

    # (
        plot=PlutoPlotly.plot(traces, layout)
        # base=base,
        # added=added,
        # linearavg=linearavg
    # )
end

# ╔═╡ 6eb5fd0f-cf28-4866-a6c0-56a08a314bfd
WideCell(compare_all_methods_filtered.plot)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
DrWatson = "634d3b9d-ee7a-5ddf-bec9-22491ea816e1"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
GMT = "5752ebe1-31b9-557e-87aa-f909b540aa54"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
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
BenchmarkTools = "~1.6.3"
CSV = "~0.10.16"
CUDA = "~5.9.6"
DSP = "~0.8.4"
DataFrames = "~1.8.1"
Distances = "~0.10.12"
DrWatson = "~2.19.1"
Enzyme = "~0.13.134"
FFTW = "~1.10.0"
Flux = "~0.16.9"
Functors = "~0.5.2"
GMT = "~1.36.0"
JLD2 = "~0.5.15"
MLUtils = "~0.4.8"
Optimisers = "~0.4.7"
ParameterSchedulers = "~0.4.3"
Peaks = "~0.5.3"
PlutoHooks = "~0.1.0"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.80"
ProgressLogging = "~0.1.6"
StatsBase = "~0.34.10"
Zygote = "~0.7.10"
cuDNN = "~1.4.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "f494a1f09f1fca4f77dfa47cf57d93534b25ba63"

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
git-tree-sha1 = "35ea197a51ce46fcd01c4a44befce0578a1aaeca"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.0"
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

[[deps.Arrow_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Lz4_jll", "Thrift_jll", "Zlib_jll", "Zstd_jll", "boost_jll", "brotli_jll", "snappy_jll"]
git-tree-sha1 = "55ecf3d16295c26e96d2f0b65386d1a8414e2283"
uuid = "8ce61222-c28f-5041-a97a-c2198fb817bf"
version = "19.0.1+0"

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
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "535c80f1c0847a4c967ea945fca21becc9de1522"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.7+0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "8d8e0b0f350b8e1c91420b5e64e5de774c2f0f4d"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.16"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "demumble_jll"]
git-tree-sha1 = "3fe1fb600b6ec029697416d5851ef0661c538f20"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.9.6"

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SparseMatricesCSRExt = "SparseMatricesCSR"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8c19e97de5b7574672e4a7a3abd55714ad66d59a"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.4.2+0"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "061f39cc84e99928830aa1005d79f7e99097ba28"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.2.0+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "f9a521f52d236fe49f1028d69e549e7f2644bb72"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "1.0.0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "92cd84e2b760e471d647153ea5efc5789fc5e8b2"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.19.2+0"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "70dea6a7133d2100a143b515a00d6d887e208500"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "9.20.0+0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3c190c570fb3108c09f838607386d10c71701789"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.73.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "b7231a755812695b8046e8471ddc34c8268cbad5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.0"

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

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

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
version = "1.6.0"

[[deps.DrWatson]]
deps = ["Dates", "FileIO", "JLD2", "LibGit2", "MacroTools", "Pkg", "Random", "Requires", "Scratch", "UnPack"]
git-tree-sha1 = "5b6632df14cf24fc2cdb805aab24147001463336"
uuid = "634d3b9d-ee7a-5ddf-bec9-22491ea816e1"
version = "2.19.1"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "InteractiveUtils", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "PrecompileTools", "Preferences", "Printf", "Random", "SparseArrays"]
git-tree-sha1 = "3b8ff73a9885aaea237bc262bd5fc798b5fd866f"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.13.134"
weakdeps = ["ADTypes", "BFloat16s", "ChainRulesCore", "GPUArraysCore", "LogExpFunctions", "SpecialFunctions", "StaticArrays"]

    [deps.Enzyme.extensions]
    EnzymeBFloat16sExt = "BFloat16s"
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeGPUArraysCoreExt = "GPUArraysCore"
    EnzymeLogExpFunctionsExt = "LogExpFunctions"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

[[deps.EnzymeCore]]
git-tree-sha1 = "990991b8aa76d17693a98e3a915ac7aa49f08d1a"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.18"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "4c22000e08aaa862526d9a41cfb7003e4002e653"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.256+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

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
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

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
deps = ["ADTypes", "Adapt", "ChainRulesCore", "Compat", "EnzymeCore", "Functors", "LinearAlgebra", "MLCore", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "ea6715b3d7a95a07a62109df1c9ede2641a50706"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.16.9"

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

[[deps.GDAL_jll]]
deps = ["Arrow_jll", "Artifacts", "Blosc_jll", "Expat_jll", "GEOS_jll", "HDF4_jll", "HDF5_jll", "JLLWrappers", "LERC_jll", "LibCURL_jll", "LibPQ_jll", "Libdl", "Libtiff_jll", "Lz4_jll", "NetCDF_jll", "OpenJpeg_jll", "PCRE2_jll", "PROJ_jll", "Qhull_jll", "SQLite_jll", "XML2_jll", "XZ_jll", "Zlib_jll", "Zstd_jll", "libgeotiff_jll", "libpng_jll", "libwebp_jll", "muparser_jll"]
git-tree-sha1 = "0e4385131431afe4cadb02f2e8b70156c23ac8f0"
uuid = "a7073274-a066-55f0-b90d-d619367d196c"
version = "303.1100.500+0"

[[deps.GEOS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fdaf62d2354bb398652ee612d487eb19d74468a6"
uuid = "d604d12d-fa86-5845-992e-78dc15976526"
version = "3.14.1+0"

[[deps.GMT]]
deps = ["Dates", "Downloads", "GDAL_jll", "GMT_jll", "Ghostscript_jll", "InteractiveUtils", "LASzip_jll", "Leptonica_jll", "LinearAlgebra", "PROJ_jll", "PrecompileTools", "Printf", "SparseArrays", "Statistics", "Tables"]
git-tree-sha1 = "e117799e805db61e685ae835f73620f49ec7e5da"
uuid = "5752ebe1-31b9-557e-87aa-f909b540aa54"
version = "1.36.0"

    [deps.GMT.extensions]
    GMTDataFramesExt = "DataFrames"
    GMTExcelExt = "XLSX"
    GMTParkerFFTExt = "FFTW"

    [deps.GMT.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.GMT_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "FFTW_jll", "GDAL_jll", "Ghostscript_jll", "Glib_jll", "JLLWrappers", "LAPACK32_jll", "LLVMOpenMP_jll", "LibCURL_jll", "Libdl", "NetCDF_jll", "OpenBLAS32_jll", "PCRE_jll", "PROJ_jll"]
git-tree-sha1 = "a63357b5b46c5fd6f48d343b95b245cee4eb2317"
uuid = "b68b8c3f-ed99-5bef-9675-4739d9426b26"
version = "6.6.0+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "SparseArrays", "Statistics"]
git-tree-sha1 = "6487601563e4a1d1dab796e88b4548bf5544209e"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.4.1"
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
git-tree-sha1 = "966946d226e8b676ca6409454718accb18c34c54"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.8.2"

[[deps.GPUToolbox]]
deps = ["LLVM"]
git-tree-sha1 = "e5cc871cac863a14706d745dcf73c91de948eca5"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "1.1.0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

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

[[deps.HDF4_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "ea9eff9cfef5f45b771096e5c2de3de0eab937c3"
uuid = "818ab7a1-5177-5f44-ba99-6e845030c6cb"
version = "4.3.2+0"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "e94f84da9af7ce9c6be049e9067e511e17ff89ec"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.6+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "157e2e5838984449e44af851a52fe374d56b9ada"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.13.0+0"

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

[[deps.ICU_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b3d8be712fbf9237935bde0ce9b5a736ae38fc34"
uuid = "a51ab1cf-af8e-5615-a023-bc2c838bba6b"
version = "76.2.0+0"

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

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues", "TranscodingStreams"]
git-tree-sha1 = "d97791feefda45729613fafeccc4fbef3f539151"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.15"
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
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "3d3b79166e2a0afcf875df20db110af91ad3ab61"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.11"

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

[[deps.Kerberos_krb5_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0f2899fdadaab4b8f57db558ba21bdb4fb52f1f0"
uuid = "b39eb1a6-c29a-53d7-8c32-632cd16f18da"
version = "1.21.3+0"

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

[[deps.LAPACK32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "libblastrampoline_jll"]
git-tree-sha1 = "ff8dd29d35e5cdb26128a590487cad31b829cae3"
uuid = "17f450c3-bd24-55df-bb84-8c51b4b939e3"
version = "3.12.1+1"

[[deps.LASzip_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be79377cdff896d9e19f5c23795b05b056e8d7cd"
uuid = "8372b9c3-1e34-5cc3-bfab-1a98e101de11"
version = "3.4.4001+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "69e4739502b7ab5176117e97e1664ed181c35036"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.4.6"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8e76807afb59ebb833e9b131ebf1a8c006510f33"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.38+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

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

[[deps.Leptonica_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Zlib_jll", "libpng_jll", "libwebp_jll"]
git-tree-sha1 = "0c37e62c28b9402f82b332d3828ed341268c3d00"
uuid = "6a1430e4-294a-53a5-a485-ec66ef6b843c"
version = "1.85.0+0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.11.1+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibPQ_jll]]
deps = ["Artifacts", "ICU_jll", "JLLWrappers", "Kerberos_krb5_jll", "Libdl", "OpenSSL_jll", "Zstd_jll"]
git-tree-sha1 = "7757f54f007cc0eb516a5000fb9a6fc19a49da7e"
uuid = "08be9ffa-1c94-5ee5-a977-46a84ec9b350"
version = "16.8.0+0"

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
git-tree-sha1 = "97bbca976196f2a1eb9607131cb108c69ec3f8a6"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "8e6a74641caf3b84800f2ccd55dc7ab83893c10b"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.17.0+0"

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

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "191686b1ac1ea9c89fc52e996ad15d1d241d1e33"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.10.1+0"

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
git-tree-sha1 = "39a69ca451c3e78b9a6a2e42ef894fdf7505e629"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.17.5"

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

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "9341048b9f723f2ae2a72a5269ac2f15f80534dc"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.3.2+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "8e98d5d80b87403c311fd51e8455d4546ba7a5f8"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.12"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "36c2d142e7d45fb98b5f83925213feb3292ca348"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.5+0"

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

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

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
version = "2025.5.20"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "6dc9ffc3a9931e6b988f913b49630d0fb986d0a8"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.33"

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

[[deps.NetCDF_jll]]
deps = ["Artifacts", "Blosc_jll", "Bzip2_jll", "HDF5_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML", "XML2_jll", "Zlib_jll", "Zstd_jll", "libaec_jll", "libzip_jll"]
git-tree-sha1 = "d574803b6055116af212434460adf654ce98e345"
uuid = "7243133f-43d8-5620-bbf4-c2c921802cf3"
version = "401.900.300+0"

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
git-tree-sha1 = "bfe8e84c71972f77e775f75e6d8048ad3fdbe8bc"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.10"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "46cce8b42186882811da4ce1f4c7208b02deb716"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.30+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "libpng_jll"]
git-tree-sha1 = "215a6666fee6d6b3a6e75f2cc22cb767e2dd393a"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.5.5+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "2f3d05e419b6125ffe06e55784102e99325bdbe2"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.10+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

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

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ccf0e9339e1f3e66e241ce01bbcbf57a0a9c15a1"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.45.0+0"

[[deps.PROJ_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "Libtiff_jll", "SQLite_jll"]
git-tree-sha1 = "af57004c3b686097d563f9c394d7886431a38c75"
uuid = "58948b4f-47e0-5654-a9ad-f609743f8632"
version = "902.700.100+0"

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
deps = ["RecipesBase", "SIMD"]
git-tree-sha1 = "75d0ce1c30696d77bc60840222d7fc5d549ebf5f"
uuid = "18e31ff7-3703-566c-8e60-38913d67486b"
version = "0.5.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.0"
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

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "211530a7dc76ab59087f4d4d1fc3f086fbe87594"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.2.3"

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

[[deps.Qhull_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c69da20496799bbdd56c15ecf5d80a5e6cbcc904"
uuid = "784f63db-0788-585a-bace-daefebcd302b"
version = "10008.0.1004+0"

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

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

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
git-tree-sha1 = "d97d78d4fc5f858d8ce44f6b88bc972f2023f51d"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.14.0"
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

[[deps.SQLite_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll", "dlfcn_win32_jll"]
git-tree-sha1 = "0b5f220f90642566b65ba86549d1ee4118ab2579"
uuid = "76ed43ae-9a5d-5a62-8c75-30186b810ce8"
version = "3.51.2+0"

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
git-tree-sha1 = "5acc6a41b3082920f79ca3c759acbcecf18a8d78"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.1"
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
git-tree-sha1 = "a2c37d815bf00575332b7bd0389f771cb7987214"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.2"
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
git-tree-sha1 = "fa95b3b097bcef5845c142ea2e085f1b2591e92c"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.7.1"

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

[[deps.Thrift_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "boost_jll"]
git-tree-sha1 = "4d16a4b4eab80099c19342b10d0bdb252c39bea6"
uuid = "e0b8ae26-5307-5830-91fd-398402328850"
version = "0.21.1+0"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

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

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9cce64c0fdd1960b597ba7ecda2950b5ed957438"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.2+0"

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

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4909eb8f1cbf6bd4b1c30dd18b2ead9019ef2fad"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.18.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

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

[[deps.boost_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "25fb6ecbb784a45f8ea74584fa631a9e85393dd0"
uuid = "28df3c45-c428-5900-9ff8-a3135698ca75"
version = "1.87.0+0"

[[deps.brotli_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "46fda47f4215c957bc92fd5fbb5ad04fee1e3743"
uuid = "4611771a-a7d2-5e23-8d00-b1becdba1aae"
version = "1.2.0+0"

[[deps.cuDNN]]
deps = ["CEnum", "CUDA", "CUDA_Runtime_Discovery", "CUDNN_jll"]
git-tree-sha1 = "c1e756c5b075d06f19595ac0bc6388ab2973237a"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "1.4.6"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

[[deps.dlfcn_win32_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e141d67ffe550eadfb5af1bdbdaf138031e4805f"
uuid = "c4b69c83-5512-53e3-94e6-de98773c479f"
version = "1.4.2+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "13b760f97c6e753b47df30cb438d4dc3b50df282"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libgeotiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LibCURL_jll", "Libdl", "Libtiff_jll", "PROJ_jll", "Zlib_jll"]
git-tree-sha1 = "cbdbc9ae1127f81cb653a4f7545d89f8db2a17a7"
uuid = "06c338fa-64ff-565b-ac2f-249532af990e"
version = "100.702.400+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e015f211ebb898c8180887012b938f3851e719ac"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.55+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.libzip_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "86addc139bca85fdf9e7741e10977c45785727b7"
uuid = "337d8026-41b4-5cde-a456-74a10e5b31d1"
version = "1.11.3+0"

[[deps.muparser_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "70ee0f42a44ef6e16298e5bfc8b6e311d08e49bb"
uuid = "888e69b1-873b-5047-a2fc-24c07cbe9dc8"
version = "2.3.5+0"

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
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.5.0+2"

[[deps.snappy_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ca88363dd41d2547f52118287dd34dbbc14f3eb7"
uuid = "fe1e1685-f7be-5f59-ac9f-4ca204017dfd"
version = "1.2.3+0"
"""

# ╔═╡ Cell order:
# ╠═10000001-0000-0000-0000-000000000001
# ╠═66eeaea8-8d59-4dc8-98bf-afbc2ad5c42e
# ╠═e00d151e-64aa-4677-8dfb-d543065cd0e0
# ╠═da62431a-7cc6-4253-986d-5ba7d39e9f90
# ╠═53f17afb-91fb-4881-a9f4-9fa87a24fee6
# ╠═418c15e5-8116-4d86-8c3e-aeac13cc3ef1
# ╠═d2a370a2-841d-4014-b419-33076fe051ab
# ╠═10000002-0000-0000-0000-000000000001
# ╠═10000003-0000-0000-0000-000000000001
# ╟─10000004-0000-0000-0000-000000000001
# ╟─10000005-0000-0000-0000-000000000001
# ╠═7bae698c-4223-4eb0-b11f-7588d11c96fe
# ╠═b9acedbd-2e86-44ab-9d21-61a09d67ec51
# ╠═d87a898c-7d93-4b37-8dad-4184575631e1
# ╠═05a75438-74f8-453f-b72b-dcd146d22e4b
# ╠═9354d1b2-a6d7-4d2c-abee-43949070a726
# ╠═be4a46d9-241f-4ee9-a010-d6665ad69877
# ╠═9df0b89e-5c5a-49f2-8eef-099d1f1aa6e1
# ╠═f270c30c-a119-4990-b2b5-6c0fcfa4b18b
# ╠═4761cee0-09f9-4aec-8518-96e0b127216b
# ╠═0c6af546-a71f-411b-8bb4-b52fb507390a
# ╠═fcc7c1ab-6649-4d8d-9784-64efb1c11964
# ╠═6bcd1038-25e2-4562-ad78-6100e1e08c9a
# ╠═6197fbaa-60cc-43ab-8db4-8b4060fa4610
# ╠═72c1b63b-c456-484b-a7a7-fbae8d253b7c
# ╠═8e8eb12e-d3cc-442e-adc2-b3c48e718031
# ╠═654fa7a5-af59-4a5e-a56f-c72c45239057
# ╠═162350c6-9fb7-45ac-9368-c0c4a6d71eef
# ╠═586e46ab-7a24-4a09-82b1-86a674060fed
# ╠═8ca289ac-d12c-4383-9538-d83c26b4ec24
# ╠═0ff4cfa0-a08d-4695-84c8-729b025e2851
# ╠═b1896dc8-2721-421b-afc6-a6a4f55157ce
# ╠═02e72481-8128-4372-b806-c77e2e23a23b
# ╟─7f4f2d6e-0e1a-4683-b8f1-1b3f0f7dcf3a
# ╠═671308fa-7136-437a-9b25-e14790407c39
# ╠═66bfe53e-6264-4d2f-90f4-261f44dd7816
# ╟─45c568c0-a4c7-4c4c-b3f2-4a7c96a23230
# ╟─7dbb51c2-bcdc-41af-b37b-898392b7c3cd
# ╠═238ab25d-6d5e-454b-a002-268c02c6889c
# ╟─82f69b08-103a-466b-88b8-c978179c7d61
# ╟─96102e0b-2f7c-497f-ba7e-762b546ada17
# ╠═0ab22a84-ce39-4986-8030-a150cd80c895
# ╠═dc42ebe1-2b60-485d-99cb-a42c6cc288ab
# ╠═5cfafb23-0904-45af-8004-8607ab17e338
# ╠═120d5d95-6254-4b61-9928-8a7e01d34594
# ╠═374e8b44-56c0-4620-919f-325d631e5c2d
# ╠═d7a3ceef-fea6-4230-9788-0c44726a3904
# ╟─a71cd612-18fe-45cd-a3f5-c873cc09e096
# ╠═4f94500d-8b28-4c7b-81a3-6012fca2a5e6
# ╠═7b3cbebe-5c0d-4508-9707-283b0d735e44
# ╠═cc06cac4-5759-4eaa-892d-1550966475b5
# ╠═b361db3e-7c6f-4d88-b1f1-532e7b49be19
# ╠═a01ed292-1f8e-479e-a335-08fd34bdc319
# ╠═bbd1ec61-fe83-466d-a022-a1cb869fb690
# ╠═fc52a726-a744-4de1-8231-096ed224afb5
# ╠═22202b34-9585-49cc-9b6f-c7a3c35f90b2
# ╠═fa452b46-d89b-4a29-969a-77116f1c1c23
# ╠═cc01ab54-7913-4e9a-aff9-2ed745abf700
# ╠═468730f9-0619-4ecd-b580-13082f450b4b
# ╠═85f35f48-ea77-4c3a-8a34-a8818a41d85d
# ╠═8508dbd2-3910-4c52-8ac3-797811c5905d
# ╠═29266f59-439b-4d34-b207-d1b3c7b9191f
# ╠═10000009-0000-0000-0000-000000000001
# ╠═1000000a-0000-0000-0000-000000000001
# ╠═1000000c-0000-0000-0000-000000000001
# ╠═6897e2bd-68a7-4473-a8cc-c20320a58fbe
# ╠═cacf4048-cdc3-4d1f-b230-535830d34729
# ╟─1000000d-0000-0000-0000-000000000001
# ╠═1000000e-0000-0000-0000-000000000001
# ╠═7dbcee8c-583b-4250-9895-759585dac882
# ╟─10000010-0000-0000-0000-000000000001
# ╠═10000011-0000-0000-0000-000000000001
# ╠═10000013-0000-0000-0000-000000000001
# ╠═10000014-0000-0000-0000-000000000001
# ╟─10000015-0000-0000-0000-000000000001
# ╟─10000017-0000-0000-0000-000000000001
# ╠═10000018-0000-0000-0000-000000000001
# ╟─10000024-0000-0000-0000-000000000001
# ╟─10000028-0000-0000-0000-000000000001
# ╟─1000002a-0000-0000-0000-000000000001
# ╟─1000002b-0000-0000-0000-000000000001
# ╠═ee4e2ae3-faf7-4b84-93e5-105cc419a57c
# ╟─1000002f-0000-0000-0000-000000000001
# ╠═f0e0c5d0-0f80-4ee4-b5c7-49889adb2c6d
# ╠═10000030-0000-0000-0000-000000000001
# ╟─29bc277f-a757-4e0f-aa84-034b2f3f92b1
# ╠═67f58bd0-b83b-49c3-acb7-cb9e26089226
# ╟─10000032-0000-0000-0000-000000000001
# ╠═446f0a9b-a225-4bdd-9c1c-47f6e31c994f
# ╠═904e2d6f-590b-4067-b586-e33872dbc2b0
# ╠═2d5d5e97-5b31-4cfb-aecc-b738f8aae2d4
# ╠═8e81d11e-83d2-4311-a1f8-07415c394581
# ╠═31783e01-f9ad-4b40-8565-50b8bad0cc17
# ╠═ee3a210b-2e72-4658-82dc-de6fb6179e60
# ╠═24e15056-30cd-4278-9460-1529a6f3e9c0
# ╟─10000034-0000-0000-0000-000000000001
# ╠═10000035-0000-0000-0000-000000000001
# ╠═b932e2a5-ea58-4432-8d4d-2391f36608f2
# ╠═db4ddb38-2938-11f1-b8e3-e5227df9322c
# ╠═dc1644c2-47d1-493e-acbb-7313f01b3120
# ╠═381c3a52-8a56-474b-9ca1-59efd79d5aec
# ╠═1c59dd1a-f93c-4cb4-9ca7-f5638e0f1dcb
# ╠═27b79706-6029-475b-9648-91b8ac308dbe
# ╠═06668d2a-d5ae-4a20-a46f-cb48130ace54
# ╠═799c0703-d9e7-4f4e-a205-9fd9aee6ad62
# ╠═c041b902-17e2-4e4b-9bb5-f924cc129993
# ╠═8a7d0899-93c9-4b14-8cb4-664ae24a478b
# ╠═c906f5e8-76d2-4632-a623-6d4dac36b4d7
# ╠═68e1c79d-b2f1-474d-8c29-1cec5aef8a13
# ╠═4f1de2d4-6240-4f26-8cf1-0cbfb9cd527c
# ╠═5a6662ed-7a40-465e-a7eb-f6736f5b592b
# ╠═657c32f6-3cef-40a6-a04f-3116cb25291d
# ╠═a8c6aa8d-bf3f-4d09-8a4d-9fcb676171bd
# ╠═9729858c-6cf7-4502-9447-67cf53ba1f35
# ╠═c5ef03a5-08c8-4f31-a915-f31dc1d860c1
# ╠═b4d22adf-e0f4-40cb-ac8f-0299cfcccf34
# ╠═14d83bc3-ff02-47ec-a3e5-b143121a5040
# ╠═86f19653-7c18-4678-b39c-212e8940c7b2
# ╠═f07576a0-dac8-4791-9758-6823025faa1b
# ╠═71eea2a2-7878-4812-a41d-52b63d52ca5c
# ╠═2dfbe495-3f31-4d80-8f10-5cbfcb8976d9
# ╠═a6765026-7fcb-44c7-bf18-77e9f6dc0fce
# ╠═ba0f0ba1-e9c3-4f22-af62-5f0f87346ee3
# ╠═90d8dd45-ea1f-46e1-a3b9-0e5cbc2ee7ea
# ╠═74b713d0-266f-4f1f-b5dc-5b4c8f5176cf
# ╠═2b86cdb9-badf-4f59-ba10-c19b54f5a646
# ╠═6fd3168e-6d8d-4de8-9aa0-f7aecdc127af
# ╠═453d85d2-b9e6-426d-88b1-853402ecf176
# ╠═cb7707d6-50ed-4157-8563-897a39582f50
# ╠═23717119-86fc-4745-9af4-286ec0212633
# ╠═8eb30f68-da67-4b29-8011-9a024decbc0a
# ╠═402d44d9-ec83-440f-8616-7efa091ae59b
# ╠═1f5710ff-e36a-4906-8bc1-209d1eefeb71
# ╠═909b15e4-f8eb-4eff-a4d4-f737e94a8a26
# ╠═3d526cd0-c6cd-4637-8132-5c5920484de3
# ╠═28f6f4bc-47c4-4fd5-8f2b-a326d9f8e45a
# ╠═2a553a6f-8901-4fdf-9f34-39e0f6354427
# ╠═3a7106f8-72a4-4397-b66e-cbe33ad5469c
# ╠═13d0dbf1-1f5a-4e2e-83a8-b5a7f11e6d2d
# ╠═c1f5f3e4-3f2c-4681-b4f7-84f3b2c257a7
# ╠═ebd0c89b-9f95-45e3-a68a-17fd2bfcda10
# ╠═61ca1735-af39-4400-8931-bf4fb2f3f3f8
# ╠═1dc90fe7-a789-4378-abf7-a4f179f9dc26
# ╠═91d7c8e2-bad3-4b79-bb7e-973b59e59011
# ╠═168f0229-38ad-4365-8803-6b426735de1a
# ╠═6eb5fd0f-cf28-4866-a6c0-56a08a314bfd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
