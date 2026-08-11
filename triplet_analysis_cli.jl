#!/usr/bin/env julia

using CSV
using DataFrames
using DSP
using JLD2
using Pkg
using PlutoLinks
using Printf
using Statistics
using TOML

const CLI_FTAN_PATH = "/home/sanket/Desktop/FTAN.jl/src/FTAN.jl"
const CLI_MFT_CACHE = Ref{Any}(nothing)
const DEFAULT_STATION_CSV = "/home/sanket/Desktop/Stations_XI_2011_13_SN.csv"

function _mft()
    if isnothing(CLI_MFT_CACHE[])
        CLI_MFT_CACHE[] = (@ingredients(CLI_FTAN_PATH)).FTAN
    end
    CLI_MFT_CACHE[]
end

station_latlon_dataframe(path::AbstractString=DEFAULT_STATION_CSV) =
    isfile(path) ? CSV.read(path, DataFrame) : DataFrame()

function station_latlon_map(station_csv::AbstractString=DEFAULT_STATION_CSV)
    df = station_latlon_dataframe(station_csv)
    isempty(df) && return Dict{String,NamedTuple}()
    Dict(String(row[Symbol("Station Code")]) => (; lat=Float64(row.Latitude), lon=Float64(row.Longitude))
        for row in eachrow(df))
end

function _parse_pair_label(s::AbstractString)
    parts = occursin("-", String(s)) ? split(String(s), "-"; limit=2) : split(String(s), "_"; limit=2)
    length(parts) == 2 || error("Pair must look like SN43-SN63 or SN43_SN63; got $(s)")
    String(strip(parts[1])), String(strip(parts[2]))
end

_pair_display_label(pair::Tuple{String,String}) = "$(pair[1])-$(pair[2])"

deg2radf(x) = Float64(x) * pi / 180

function _haversine_km(lat1, lon1, lat2, lon2)
    r = 6371.0
    phi1, phi2 = deg2radf(lat1), deg2radf(lat2)
    dphi = deg2radf(Float64(lat2) - Float64(lat1))
    dlambda = deg2radf(Float64(lon2) - Float64(lon1))
    a = sin(dphi / 2)^2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)^2
    2r * atan(sqrt(a), sqrt(max(0.0, 1.0 - a)))
end

function _azimuth_deg(lat1::Real, lon1::Real, lat2::Real, lon2::Real)
    phi1, phi2 = deg2rad(Float64(lat1)), deg2rad(Float64(lat2))
    dlambda = deg2rad(Float64(lon2) - Float64(lon1))
    y = sin(dlambda) * cos(phi2)
    x = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dlambda)
    mod(rad2deg(atan(y, x)) + 360.0, 360.0)
end

function _azimuth_spread_deg(azimuths)
    vals = Float64[a for a in azimuths if isfinite(Float64(a))]
    length(vals) < 2 && return Inf
    maximum(min(abs(a - b), 360.0 - abs(a - b)) for a in vals for b in vals)
end

function _pair_lookup_by_station(pair_labels)
    lookup = Dict{Tuple{String,String},String}()
    for pair_label in pair_labels
        a, b = _parse_pair_label(pair_label)
        haskey(lookup, (a, b)) || (lookup[(a, b)] = String(pair_label))
        haskey(lookup, (b, a)) || (lookup[(b, a)] = String(pair_label))
    end
    lookup
end

function station_triple_geometry(a::String, b::String, c::String, pair_labels;
        station_csv::AbstractString=DEFAULT_STATION_CSV, pair_distances=Dict{String,Float64}())
    latlon = station_latlon_map(station_csv)
    lookup = _pair_lookup_by_station(pair_labels)
    pair_ab = get(lookup, (a, b), missing)
    pair_bc = get(lookup, (b, c), missing)
    pair_ac = get(lookup, (a, c), missing)
    missing_pairs = String[]
    ismissing(pair_ab) && push!(missing_pairs, "$(a)-$(b)")
    ismissing(pair_bc) && push!(missing_pairs, "$(b)-$(c)")
    ismissing(pair_ac) && push!(missing_pairs, "$(a)-$(c)")
    distinct = length(unique([a, b, c])) == 3
    coords_ok = all(haskey(latlon, sta) for sta in (a, b, c))
    dist_for(pair, x, y) = !ismissing(pair) && haskey(pair_distances, String(pair)) ?
        Float64(pair_distances[String(pair)]) :
        (coords_ok ? _haversine_km(latlon[x].lat, latlon[x].lon, latlon[y].lat, latlon[y].lon) : Inf)
    dab = dist_for(pair_ab, a, b)
    dbc = dist_for(pair_bc, b, c)
    dac = dist_for(pair_ac, a, c)
    distance_closure = all(isfinite, (dab, dbc, dac)) ? dab + dbc - dac : Inf
    az_ab = coords_ok ? _azimuth_deg(latlon[a].lat, latlon[a].lon, latlon[b].lat, latlon[b].lon) : Inf
    az_bc = coords_ok ? _azimuth_deg(latlon[b].lat, latlon[b].lon, latlon[c].lat, latlon[c].lon) : Inf
    az_ac = coords_ok ? _azimuth_deg(latlon[a].lat, latlon[a].lon, latlon[c].lat, latlon[c].lon) : Inf
    azimuth_spread = _azimuth_spread_deg((az_ab, az_bc, az_ac))
    ratio = isfinite(dab) && isfinite(dbc) && dbc > 0.0 ? dab / dbc : Inf
    (; a, b, c, pair_ab, pair_bc, pair_ac, missing_pairs, distinct, coords_ok,
        dab, dbc, dac, distance_closure, az_ab, az_bc, az_ac, azimuth_spread, ratio)
end

function _period_velocity_map(rows, pair_label; velocity_field::Symbol=:group_velocity)
    out = Dict{Float64,Float64}()
    for r in rows
        hasproperty(r, :pair_label) && String(r.pair_label) == String(pair_label) || continue
        hasproperty(r, velocity_field) || continue
        period = round(Float64(getproperty(r, :period)); digits=8)
        velocity = Float64(getproperty(r, velocity_field))
        isfinite(period) && period > 0.0 && isfinite(velocity) && velocity > 0.0 || continue
        haskey(out, period) || (out[period] = velocity)
    end
    out
end

function _station_triple_consistency_rows(source_label::String, geom,
        ab_map::Dict{Float64,Float64}, bc_map::Dict{Float64,Float64}, ac_map::Dict{Float64,Float64};
        velocity_kind::String="group", data_source::String="VQVAE artifacts", month=missing,
        ab_source_map::Dict{Float64,String}=Dict{Float64,String}(),
        bc_source_map::Dict{Float64,String}=Dict{Float64,String}(),
        ac_source_map::Dict{Float64,String}=Dict{Float64,String}())
    rows = NamedTuple[]
    common_periods = sort(collect(intersect(intersect(keys(ab_map), keys(bc_map)), keys(ac_map))))
    for period in common_periods
        vab, vbc, vac = ab_map[period], bc_map[period], ac_map[period]
        all(isfinite, (vab, vbc, vac, geom.dab, geom.dbc, geom.dac)) || continue
        minimum((vab, vbc, vac, geom.dab, geom.dbc, geom.dac)) > 0.0 || continue
        vabc = (geom.dab + geom.dbc) / (geom.dab / vab + geom.dbc / vbc)
        tac = geom.dac / vac
        tdif = (geom.dab / vab + geom.dbc / vbc) - tac
        tpdif = tdif / tac * 100.0
        push!(rows, (; source=source_label, data_source, velocity_kind, month,
            period, pair_ab=String(geom.pair_ab), pair_bc=String(geom.pair_bc), pair_ac=String(geom.pair_ac),
            source_ab=get(ab_source_map, period, ""), source_bc=get(bc_source_map, period, ""),
            source_ac=get(ac_source_map, period, ""), dab=geom.dab, dbc=geom.dbc, dac=geom.dac,
            Delta_az_deg=Float64(geom.azimuth_spread), Delta_D_km=Float64(geom.distance_closure),
            ratio=Float64(geom.ratio),
            passes_strict_geometry=geom.azimuth_spread < 1.0 && abs(geom.distance_closure) < 0.1,
            vab, vbc, vac, vabc, vdif=vabc - vac, tac, tdif, tpdif))
    end
    rows
end

function station_triple_dataframe(rows)
    isempty(rows) && return DataFrame()
    sorted_rows = sort(collect(rows), by=r -> (String(r.data_source), String(r.source),
        ismissing(r.month) ? "" : String(r.month), Float64(r.period)))
    DataFrame(
        triplet=String[hasproperty(r, :triplet) ? String(r.triplet) : "" for r in sorted_rows],
        curve_family=String[hasproperty(r, :curve_family) ? String(r.curve_family) : "" for r in sorted_rows],
        selected_rank=Int[hasproperty(r, :selected_rank) ? Int(r.selected_rank) : 0 for r in sorted_rows],
        selected_state_label=String[hasproperty(r, :selected_state_label) ? String(r.selected_state_label) : "" for r in sorted_rows],
        source=String[String(r.source) for r in sorted_rows],
        data_source=String[String(r.data_source) for r in sorted_rows],
        velocity_kind=String[String(r.velocity_kind) for r in sorted_rows],
        period_s=Float64[Float64(r.period) for r in sorted_rows],
        abs_Vabc_minus_Vac_km_s=Float64[abs(Float64(r.vdif)) for r in sorted_rows],
        Vab_km_s=Float64[Float64(r.vab) for r in sorted_rows],
        Vbc_km_s=Float64[Float64(r.vbc) for r in sorted_rows],
        Vac_km_s=Float64[Float64(r.vac) for r in sorted_rows],
        Vabc_km_s=Float64[Float64(r.vabc) for r in sorted_rows],
        Vabc_minus_Vac_km_s=Float64[Float64(r.vdif) for r in sorted_rows],
        Tdiff_s=Float64[Float64(r.tdif) for r in sorted_rows],
        Tdiff_percent=Float64[Float64(r.tpdif) for r in sorted_rows],
        pair_ab=String[String(r.pair_ab) for r in sorted_rows],
        pair_bc=String[String(r.pair_bc) for r in sorted_rows],
        pair_ac=String[String(r.pair_ac) for r in sorted_rows],
        Delta_az_deg=Float64[Float64(r.Delta_az_deg) for r in sorted_rows],
        Delta_D_km=Float64[Float64(r.Delta_D_km) for r in sorted_rows],
        ratio=Float64[Float64(r.ratio) for r in sorted_rows],
        passes_strict_geometry=Bool[Bool(r.passes_strict_geometry) for r in sorted_rows],
        source_ab=String[String(r.source_ab) for r in sorted_rows],
        source_bc=String[String(r.source_bc) for r in sorted_rows],
        source_ac=String[String(r.source_ac) for r in sorted_rows],
        month=[r.month for r in sorted_rows])
end

function _parse_seed_timestamp(run_dir::String)
    name = basename(run_dir)
    m = match(r"^seed([0-9]+)_(.+)$", name)
    m === nothing && return (; seed=missing, timestamp=name)
    (; seed=parse(Int, m.captures[1]), timestamp=m.captures[2])
end

function discover_vqvae_runs(saved_root::AbstractString)
    saved_root = String(saved_root)
    isdir(saved_root) || return NamedTuple[]
    runs = NamedTuple[]
    for pair_dir in sort(filter(isdir, readdir(saved_root, join=true)))
        pair_label = basename(pair_dir)
        parts = split(pair_label, "_")
        length(parts) == 2 || continue
        for run_dir in sort(filter(isdir, readdir(pair_dir, join=true)))
            artifact = joinpath(run_dir, "source_state_averages.jld2")
            isfile(artifact) || continue
            parsed = _parse_seed_timestamp(run_dir)
            push!(runs, (; pair=(String(parts[1]), String(parts[2])),
                pair_label=replace(pair_label, "_" => "-"), run_dir, artifact_path=artifact,
                seed=parsed.seed, timestamp=parsed.timestamp,
                artifact_kind="trained", reference_pair_label=""))
        end
    end
    runs
end

function discover_selected_state_transfer_runs(output_root::AbstractString)
    output_root = String(output_root)
    isdir(output_root) || return NamedTuple[]
    runs = NamedTuple[]
    for pair_dir in sort(filter(isdir, readdir(output_root, join=true)))
        pair_label_raw = basename(pair_dir)
        parts = split(pair_label_raw, "_")
        length(parts) == 2 || continue
        for run_dir in sort(filter(isdir, readdir(pair_dir, join=true)))
            for state_dir in sort(filter(isdir, readdir(run_dir, join=true)))
                artifact = joinpath(state_dir, "selected_state_transfer.jld2")
                isfile(artifact) || continue
                d = load(artifact)
                ref_pair = haskey(d, "selected_from_pair") ?
                    _pair_display_label(Tuple(String.(d["selected_from_pair"]))) : "SN43-SN63"
                ref_run = haskey(d, "reference_run_tag") ? String(d["reference_run_tag"]) :
                    (haskey(d, "reference_run_dir") ? basename(String(d["reference_run_dir"])) : basename(run_dir))
                parsed = _parse_seed_timestamp(ref_run)
                push!(runs, (; pair=(String(parts[1]), String(parts[2])),
                    pair_label=replace(pair_label_raw, "_" => "-"), run_dir=state_dir,
                    artifact_path=artifact, seed=parsed.seed, timestamp=parsed.timestamp,
                    artifact_kind="selected_state_transfer", reference_pair_label=ref_pair))
            end
        end
    end
    runs
end

function discover_transferred_vqvae_runs(output_root::AbstractString)
    output_root = String(output_root)
    isdir(output_root) || return NamedTuple[]
    runs = NamedTuple[]
    for pair_dir in sort(filter(isdir, readdir(output_root, join=true)))
        pair_label_raw = basename(pair_dir)
        parts = split(pair_label_raw, "_")
        length(parts) == 2 || continue
        for run_dir in sort(filter(isdir, readdir(pair_dir, join=true)))
            artifact = joinpath(run_dir, "transferred_source_state_averages.jld2")
            isfile(artifact) || continue
            parsed = _parse_seed_timestamp(run_dir)
            push!(runs, (; pair=(String(parts[1]), String(parts[2])),
                pair_label=replace(pair_label_raw, "_" => "-"), run_dir,
                artifact_path=artifact, seed=parsed.seed, timestamp=parsed.timestamp,
                artifact_kind="transferred", reference_pair_label="SN43-SN63"))
        end
    end
    runs
end

function discover_all_vqvae_runs(saved_root::AbstractString; transfer_root::AbstractString=saved_root)
    runs = vcat(discover_vqvae_runs(saved_root),
        discover_transferred_vqvae_runs(transfer_root),
        discover_selected_state_transfer_runs(transfer_root))
    seen = Set{String}()
    unique_runs = NamedTuple[]
    for run in runs
        path = String(run.artifact_path)
        path in seen && continue
        push!(seen, path)
        push!(unique_runs, run)
    end
    sort(unique_runs; by=r -> (r.pair_label, r.artifact_kind, string(r.seed), string(r.timestamp)))
end

function _as_state_matrix(x)
    X = Float32.(x)
    ndims(X) == 1 ? reshape(X, :, 1) : X
end

function _as_count_vector(d, key::AbstractString, n::Integer)
    vals = haskey(d, key) ? Int.(vec(d[key])) : fill(0, Int(n))
    length(vals) < n && append!(vals, fill(0, Int(n) - length(vals)))
    vals[1:Int(n)]
end

function _distance_for_pair(pair_label::String; station_csv::AbstractString=DEFAULT_STATION_CSV)
    a, b = _parse_pair_label(pair_label)
    latlon = station_latlon_map(station_csv)
    haskey(latlon, a) && haskey(latlon, b) || return NaN
    _haversine_km(latlon[a].lat, latlon[a].lon, latlon[b].lat, latlon[b].lon)
end

function load_source_state_artifact(run; station_csv::AbstractString=DEFAULT_STATION_CSV)
    isfile(String(run.artifact_path)) || error("Missing source-state artifact: $(run.artifact_path)")
    d = load(String(run.artifact_path))
    artifact_kind = hasproperty(run, :artifact_kind) ? String(run.artifact_kind) : "trained"
    acausal = haskey(d, "acausal") ? _as_state_matrix(d["acausal"]) : Float32[;;]
    causal = haskey(d, "causal") ? _as_state_matrix(d["causal"]) : Float32[;;]
    transfer_branch = haskey(d, "selected_transfer_branch") ? String(d["selected_transfer_branch"]) : "both"
    if transfer_branch == "causal" && size(causal, 2) > 0 && size(acausal, 2) == 0
        acausal = zeros(Float32, size(causal, 1), size(causal, 2))
    elseif transfer_branch == "acausal" && size(acausal, 2) > 0 && size(causal, 2) == 0
        causal = zeros(Float32, size(acausal, 1), size(acausal, 2))
    end
    selected_label = haskey(d, "selected_state_label") ? String(d["selected_state_label"]) : "selected state"
    combo_labels = haskey(d, "combo_labels") ? String.(d["combo_labels"]) : [selected_label]
    njoint = min(size(causal, 2), size(acausal, 2))
    length(combo_labels) < njoint && append!(combo_labels, ["state $(k)" for k in length(combo_labels)+1:njoint])
    causal = causal[:, 1:njoint]
    acausal = acausal[:, 1:njoint]
    global_avg_ac = haskey(d, "global_avg_ac") ? Float32.(d["global_avg_ac"]) : (njoint > 0 ? vec(acausal[:, 1]) : Float32[])
    global_avg_c = haskey(d, "global_avg_c") ? Float32.(d["global_avg_c"]) : (njoint > 0 ? vec(causal[:, 1]) : Float32[])
    marginal_stage1_ac = haskey(d, "marginal_stage1_ac") ? _as_state_matrix(d["marginal_stage1_ac"]) : Float32[;;]
    marginal_stage1_c = haskey(d, "marginal_stage1_c") ? _as_state_matrix(d["marginal_stage1_c"]) : Float32[;;]
    marginal_stage2_ac = haskey(d, "marginal_stage2_ac") ? _as_state_matrix(d["marginal_stage2_ac"]) : Float32[;;]
    marginal_stage2_c = haskey(d, "marginal_stage2_c") ? _as_state_matrix(d["marginal_stage2_c"]) : Float32[;;]
    labels1 = haskey(d, "marginal_stage1_labels") ? String.(d["marginal_stage1_labels"]) : String[]
    labels2 = haskey(d, "marginal_stage2_labels") ? String.(d["marginal_stage2_labels"]) : String[]
    K1 = min(size(marginal_stage1_c, 2), size(marginal_stage1_ac, 2))
    K2 = min(size(marginal_stage2_c, 2), size(marginal_stage2_ac, 2))
    length(labels1) < K1 && append!(labels1, ["s1=$(k)" for k in length(labels1)+1:K1])
    length(labels2) < K2 && append!(labels2, ["s2=$(k)" for k in length(labels2)+1:K2])
    counts_c = _as_count_vector(d, "counts_c", njoint)
    counts_ac = _as_count_vector(d, "counts_ac", njoint)
    marginal_stage1_counts_c = _as_count_vector(d, "marginal_stage1_counts_c", K1)
    marginal_stage1_counts_ac = _as_count_vector(d, "marginal_stage1_counts_ac", K1)
    marginal_stage2_counts_c = _as_count_vector(d, "marginal_stage2_counts_c", K2)
    marginal_stage2_counts_ac = _as_count_vector(d, "marginal_stage2_counts_ac", K2)
    (; acausal, causal, combo_labels=combo_labels[1:njoint],
        counts_c, counts_ac,
        global_avg_ac, global_avg_c,
        marginal_stage1_ac=marginal_stage1_ac[:, 1:K1], marginal_stage1_c=marginal_stage1_c[:, 1:K1],
        marginal_stage2_ac=marginal_stage2_ac[:, 1:K2], marginal_stage2_c=marginal_stage2_c[:, 1:K2],
        marginal_stage1_labels=labels1[1:K1], marginal_stage2_labels=labels2[1:K2],
        marginal_stage1_counts_c, marginal_stage1_counts_ac,
        marginal_stage2_counts_c, marginal_stage2_counts_ac,
        distance=haskey(d, "distance") ? Float64(d["distance"]) : _distance_for_pair(String(run.pair_label); station_csv),
        pair=run.pair, pair_label=String(run.pair_label), run_dir=String(run.run_dir),
        seed=run.seed, artifact_kind, reference_pair_label=run.reference_pair_label)
end

_mft_precision_type(precision::AbstractString) = precision == "Float64" ? Float64 : Float32

function _mft_config(; dt::Real=1.0, period_min::Real=10.0, period_max::Real=60.0,
        nperiods::Integer=100, wavelength_ref_velocity::Real=2.0, wavelength_fraction::Real=0.33,
        velocity_range=(1.0, 8.0), bandwidth_factor::Real=0.15, zero_pad_factor::Integer=2,
        upsample_factor::Real=2.0, precision::AbstractString="Float32", max_modes::Integer=6,
        phase_velocity_range=(1.0, 6.0), use_phtovel::Bool=true,
        phvel_correction::Real=-π/4)
    lo, hi = sort((Float64(period_min), Float64(period_max)))
    (; dt=Float64(dt),
        periods=collect(exp.(range(log(lo), log(hi); length=Int(nperiods)))),
        wavelength_ref_velocity=Float64(wavelength_ref_velocity),
        wavelength_fraction=Float64(wavelength_fraction),
        velocity_range=(Float64(velocity_range[1]), Float64(velocity_range[2])),
        bandwidth_factor=Float64(bandwidth_factor),
        zero_pad_factor=Int(zero_pad_factor),
        upsample_factor=Float64(upsample_factor),
        precision_type=_mft_precision_type(precision),
        max_modes=Int(max_modes),
        phase_velocity_range=(Float64(phase_velocity_range[1]), Float64(phase_velocity_range[2])),
        use_phtovel=Bool(use_phtovel),
        phvel_correction=Float64(phvel_correction),
        cache=Dict{Any,Any}())
end

function _mft_filter_bank_for(periods::Vector{Float64}, npts_raw::Int; storage_mode::Symbol=:picks_only, n_waveforms::Int=1, cfg)
    isempty(periods) && return nothing
    ftan = _mft()
    key = (ftan.MFTFilterBank, npts_raw, Tuple(periods), cfg.bandwidth_factor,
        cfg.zero_pad_factor, cfg.upsample_factor, cfg.velocity_range,
        cfg.precision_type, storage_mode, n_waveforms)
    get!(cfg.cache, key) do
        Base.invokelatest(ftan.MFTFilterBank, cfg.dt, npts_raw, periods;
            bandwidth_factor=cfg.bandwidth_factor,
            zero_pad_factor=cfg.zero_pad_factor,
            upsample_factor=cfg.upsample_factor,
            velocity_range=cfg.velocity_range,
            precision=cfg.precision_type,
            storage_mode=storage_mode,
            N_initial=n_waveforms)
    end
end

function _mft_wavelength_valid_period(period::Real, distance::Real, cfg)
    ftan = _mft()
    Base.invokelatest(ftan.wavelength_valid_period, period, Float64(distance);
        wavelength_ref_velocity=cfg.wavelength_ref_velocity,
        wavelength_fraction=cfg.wavelength_fraction)
end

function _mft_compute_periods(distance::Real, cfg)
    Float64[period for period in cfg.periods if _mft_wavelength_valid_period(period, distance, cfg)]
end

function _mft_shared_periods_for_distances(distances::AbstractVector{<:Real}, cfg)
    Float64[period for period in cfg.periods
        if any(distance -> _mft_wavelength_valid_period(period, distance, cfg), distances)]
end

_has_nonzero_signal(x) = any(v -> isfinite(v) && !iszero(v), x)

function _analyze_branch_arrays(W_c::AbstractArray{<:Real}, W_ac::AbstractArray{<:Real},
        distances::AbstractVector{<:Real}, pair_keys::AbstractVector{<:AbstractString},
        state_labels::AbstractVector{<:AbstractString}, cfg; storage_mode::Symbol=:picks_only,
        compute_phase::Bool=true)
    W_c_flat = reshape(W_c, size(W_c, 1), :)
    W_ac_flat = reshape(W_ac, size(W_ac, 1), :)
    nstates = size(W_c_flat, 2)
    nstates == size(W_ac_flat, 2) == length(distances) == length(pair_keys) == length(state_labels) ||
        throw(ArgumentError("waveform states, distances, pair keys, and labels must have matching lengths"))
    n = min(size(W_c_flat, 1), size(W_ac_flat, 1))
    valid_period = [!isempty(_mft_compute_periods(distances[i], cfg)) for i in 1:nstates]
    nonzero_state = [_has_nonzero_signal(@view W_c_flat[1:n, i]) ||
        _has_nonzero_signal(@view W_ac_flat[1:n, i]) for i in 1:nstates]
    keep = [i for i in 1:nstates if valid_period[i] && nonzero_state[i]]
    isempty(keep) && return Dict{String,Any}()
    W_batch = hcat(@view(W_c_flat[1:n, keep]), @view(W_ac_flat[1:n, keep]))
    distances_batch = Float64.(distances[keep])
    periods = _mft_shared_periods_for_distances(distances_batch, cfg)
    isempty(periods) && return Dict{String,Any}()
    bank = _mft_filter_bank_for(periods, n; storage_mode, n_waveforms=size(W_batch, 2), cfg)
    isnothing(bank) && return Dict{String,Any}()
    ftan = _mft()
    results = Base.invokelatest(ftan.perform_mft_analysis_batch!, bank, W_batch, vcat(distances_batch, distances_batch);
        compute_phase, use_phtovel=cfg.use_phtovel, phase_velocity_range=cfg.phase_velocity_range,
        wavelength_ref_velocity=cfg.wavelength_ref_velocity, wavelength_fraction=cfg.wavelength_fraction,
        phvel_correction=get(cfg, :phvel_correction, -π/4))
    pair_keys_batch = String.(pair_keys[keep])
    labels_batch = String.(state_labels[keep])
    waveform_pair_keys = vcat(pair_keys_batch, pair_keys_batch)
    waveform_labels = vcat(labels_batch .* ":causal", labels_batch .* ":acausal")
    analyses = Dict{String,Any}()
    for (i, res) in enumerate(results)
        push!(get!(analyses, waveform_pair_keys[i], Dict{String,Any}()), waveform_labels[i] => res)
    end
    analyses
end

function _uc_rows_from_mft_result(res, pair_label::String, label::String)
    u_pred = any(isfinite, res.u_predicted_from_phase) ?
        res.u_predicted_from_phase : Base.invokelatest(_mft().compute_group_velocity_from_phase, res)
    rows = NamedTuple[]
    for ip in eachindex(res.periods)
        period = Float64(res.periods[ip])
        u_meas = Float64(res.group_velocities[ip])
        u_hat = Float64(u_pred[ip])
        c = Float64(res.phase_velocities[ip])
        quality = Float64(res.quality_factors[ip])
        isfinite(period) && period > 0.0 && isfinite(u_meas) && u_meas > 0.0 &&
            isfinite(u_hat) && u_hat > 0.0 && isfinite(c) && c > 0.0 || continue
        relerr = abs(u_meas - u_hat) / max(abs(u_hat), eps(Float64))
        push!(rows, (; pair_label, label, period, group_velocity=u_meas, phase_velocity=c,
            predicted_group_velocity=u_hat, relative_error=relerr,
            relative_agreement=1.0 / (1.0 + relerr), quality))
    end
    sort(rows; by=r -> r.period)
end

function _uc_error_score(errs; method::AbstractString="geomean", huber_delta::Real=0.10)
    vals = Float64[e for e in errs if isfinite(e) && e >= 0.0]
    isempty(vals) && return Inf
    method0 = lowercase(String(method))
    method0 == "geomean" && return exp(mean(log.(max.(vals, eps(Float64)))))
    method0 == "median" && return median(vals)
    method0 == "mean" && return mean(vals)
    method0 == "p95" && return quantile(vals, 0.95)
    method0 == "max" && return maximum(vals)
    if method0 == "huber"
        delta = max(Float64(huber_delta), eps(Float64))
        return mean(v <= delta ? 0.5 * v^2 : delta * (v - 0.5delta) for v in vals)
    end
    error("Unknown U-c score method: $(method)")
end

function _uc_error_stats(errs; method::AbstractString="geomean", huber_delta::Real=0.10)
    vals = Float64[e for e in errs if isfinite(e) && e >= 0.0]
    isempty(vals) && return (; uc_score=Inf, median_relative_error=Inf,
        geomean_relative_error=Inf, mean_relative_error=Inf, p95_relative_error=Inf,
        max_relative_error=Inf, huber_relative_error=Inf, n_valid=0)
    (; uc_score=_uc_error_score(vals; method, huber_delta),
        median_relative_error=median(vals),
        geomean_relative_error=_uc_error_score(vals; method="geomean", huber_delta),
        mean_relative_error=mean(vals),
        p95_relative_error=quantile(vals, 0.95),
        max_relative_error=maximum(vals),
        huber_relative_error=_uc_error_score(vals; method="huber", huber_delta),
        n_valid=length(vals))
end

function _score_display_from_state_label(state_label::AbstractString, kind::AbstractString)
    m = match(r"^(.+) seed (.+) \| (.+)$", String(state_label))
    isnothing(m) && return (; display=String(state_label), seed=missing)
    suffix = replace(m.captures[3], r" \[(causal|acausal) transfer\]" => "")
    display_suffix = kind == "joint" ? "joint $(suffix)" : suffix
    (; display="seed$(m.captures[2]) | $(display_suffix)", seed=m.captures[2])
end

function _logical_state_display(display::AbstractString)
    s = String(display)
    s = replace(s, r" \[(causal|acausal)\]$" => "")
    s = replace(s, r" \| joint stage1 " => " | S1 ")
    s = replace(s, r" \| joint stage2 " => " | S2 ")
    s = replace(s, r" \| joint S1 " => " | S1 ")
    s = replace(s, r" \| joint S2 " => " | S2 ")
    s = replace(s, " | joint " => " | ")
    s
end

function _split_mft_waveform_label(state_label::AbstractString)
    s = String(state_label)
    endswith(s, ":causal") && return s[1:end - length(":causal")], "causal"
    endswith(s, ":acausal") && return s[1:end - length(":acausal")], "acausal"
    s, "unknown"
end

function _score_mft_result(res, pair_label::String, state_label::String, kind::String;
        score_method::AbstractString="geomean", huber_delta::Real=0.10)
    base_label, branch = _split_mft_waveform_label(state_label)
    branch in ("causal", "acausal") || return NamedTuple[]
    parsed = _score_display_from_state_label(base_label, kind)
    score_label = "$(base_label) [$(branch)]"
    uc_rows = [merge(row, (; branch)) for row in _uc_rows_from_mft_result(res, pair_label, score_label)]
    errs = Float64[r.relative_error for r in uc_rows if isfinite(r.relative_error) && r.relative_error >= 0.0]
    stats = _uc_error_stats(errs; method=score_method, huber_delta)
    [merge(parsed, stats, (; label=score_label, pair_label, kind, branch,
        periods=Float64[r.period for r in uc_rows],
        group_velocities=Float64[r.group_velocity for r in uc_rows],
        uc_rows))]
end

function _run_state_mft(items, cfg)
    chunks_c = AbstractMatrix[]
    chunks_ac = AbstractMatrix[]
    distances = Float64[]
    pair_keys = String[]
    state_labels = String[]
    for item in items
        state_prefix = String(item.artifact_kind) == "selected_state_transfer" ?
            "$(item.pair_label) selected transfer from $(item.reference_pair_label) seed $(item.seed)" :
            "$(item.pair_label) seed $(item.seed)"
        nstates = min(size(item.causal, 2), size(item.acausal, 2))
        if nstates > 0 && isfinite(item.distance) && item.distance > 0.0
            push!(chunks_c, @view item.causal[:, 1:nstates])
            push!(chunks_ac, @view item.acausal[:, 1:nstates])
            for i in 1:nstates
                label = i <= length(item.combo_labels) ? item.combo_labels[i] : string(i)
                push!(distances, item.distance); push!(pair_keys, item.pair_label)
                push!(state_labels, "$(state_prefix) | $(label)")
            end
        end
        K1 = size(item.marginal_stage1_c, 2)
        if K1 > 0 && isfinite(item.distance) && item.distance > 0.0
            push!(chunks_c, item.marginal_stage1_c); push!(chunks_ac, item.marginal_stage1_ac)
            for k in 1:K1
                label = k <= length(item.marginal_stage1_labels) ? item.marginal_stage1_labels[k] : "s1=$k"
                push!(distances, item.distance); push!(pair_keys, item.pair_label)
                push!(state_labels, "$(state_prefix) | S1 $(label)")
            end
        end
        K2 = size(item.marginal_stage2_c, 2)
        if K2 > 0 && isfinite(item.distance) && item.distance > 0.0
            push!(chunks_c, item.marginal_stage2_c); push!(chunks_ac, item.marginal_stage2_ac)
            for k in 1:K2
                label = k <= length(item.marginal_stage2_labels) ? item.marginal_stage2_labels[k] : "s2=$k"
                push!(distances, item.distance); push!(pair_keys, item.pair_label)
                push!(state_labels, "$(state_prefix) | S2 $(label)")
            end
        end
    end
    n = isempty(chunks_c) ? 0 : minimum(min(size(chunks_c[i], 1), size(chunks_ac[i], 1)) for i in eachindex(chunks_c))
    n == 0 ? Dict{String,Any}() :
        _analyze_branch_arrays(reduce(hcat, [view(chunk, 1:n, :) for chunk in chunks_c]),
            reduce(hcat, [view(chunk, 1:n, :) for chunk in chunks_ac]),
            distances, pair_keys, state_labels, cfg)
end

function _triple_candidate_specs(scored_by_pair, pair_label; max_candidates::Integer=25)
    specs = get(scored_by_pair, String(pair_label), NamedTuple[])
    finite = [s for s in specs if hasproperty(s, :uc_score) &&
        isfinite(Float64(s.uc_score)) && hasproperty(s, :uc_rows)]
    finite[1:min(length(finite), max_candidates)]
end

function _triple_spec_state_branch_key(spec)
    display = hasproperty(spec, :display) ? String(spec.display) : "state"
    branch = hasproperty(spec, :branch) ? String(spec.branch) : "branch"
    (; display=_logical_state_display(display), branch)
end

function _station_triple_combination_rows(label::String, geom, spec_ab, spec_bc, spec_ac;
        velocity_field::Symbol=:phase_velocity)
    velocity_kind = velocity_field == :phase_velocity ? "phase" : "group"
    _station_triple_consistency_rows(label, geom,
        _period_velocity_map(spec_ab.uc_rows, geom.pair_ab; velocity_field),
        _period_velocity_map(spec_bc.uc_rows, geom.pair_bc; velocity_field),
        _period_velocity_map(spec_ac.uc_rows, geom.pair_ac; velocity_field);
        velocity_kind, data_source="VQVAE artifacts")
end

function _station_triple_combination_score(rows)
    vals = Float64[Float64(r.vdif) for r in rows if hasproperty(r, :vdif) && isfinite(Float64(r.vdif))]
    isempty(vals) && return Inf
    sqrt(mean(vals .^ 2))
end

function _rank_station_triple_state_combinations(geom, scored_by_pair; max_candidates::Integer=25,
        velocity_field::Symbol=:phase_velocity)
    (!geom.distinct || !isempty(geom.missing_pairs)) && return NamedTuple[]
    ab_specs = _triple_candidate_specs(scored_by_pair, geom.pair_ab; max_candidates)
    bc_specs = _triple_candidate_specs(scored_by_pair, geom.pair_bc; max_candidates)
    ac_specs = _triple_candidate_specs(scored_by_pair, geom.pair_ac; max_candidates)
    (isempty(ab_specs) || isempty(bc_specs) || isempty(ac_specs)) && return NamedTuple[]
    ranked = NamedTuple[]
    for spec_ab in ab_specs, spec_bc in bc_specs, spec_ac in ac_specs
        key_ab = _triple_spec_state_branch_key(spec_ab)
        key_ab == _triple_spec_state_branch_key(spec_bc) == _triple_spec_state_branch_key(spec_ac) || continue
        rows = _station_triple_combination_rows("ranked triple selected", geom, spec_ab, spec_bc, spec_ac; velocity_field)
        isempty(rows) && continue
        rms_vdiff = _station_triple_combination_score(rows)
        isfinite(rms_vdiff) || continue
        score_sum = Float64(spec_ab.uc_score) + Float64(spec_bc.uc_score) + Float64(spec_ac.uc_score)
        push!(ranked, (; rms_vdiff, n_periods=length(rows), score_sum,
            spec_ab, spec_bc, spec_ac,
            label_ab="$(spec_ab.display) | $(spec_ab.branch)",
            label_bc="$(spec_bc.display) | $(spec_bc.branch)",
            label_ac="$(spec_ac.display) | $(spec_ac.branch)",
            rows))
    end
    sort!(ranked; by=r -> (r.rms_vdiff, -r.n_periods, r.score_sum, r.label_ab, r.label_bc, r.label_ac))
    ranked
end

const DEFAULT_TRIPLETS_CSV = joinpath(@__DIR__, "station_triplet_csvs", "all_station_triplets_unfiltered.csv")
const DEFAULT_DENOMINATORS = Float64[0.0; collect(2.0:0.2:12.0)]

const DEFAULT_OPTIONS = Dict{String,Any}(
    "config" => "",
    "triplets-csv" => DEFAULT_TRIPLETS_CSV,
    "max-delta-az" => 0.7,
    "max-delta-d" => 0.1,
    "min-segment-distance" => 45.0,
    "bandpass" => "10:30",
    "dt" => 1.0,
    "period-min" => 10.0,
    "period-max" => 30.0,
    "nperiods" => 100,
    "velocity-min" => 1.0,
    "velocity-max" => 8.0,
    "phase-velocity-min" => 1.0,
    "phase-velocity-max" => 6.0,
    "wavelength-ref-velocity" => 2.0,
    "wavelength-fraction" => 0.33,
    "bandwidth-factor" => 0.15,
    "zero-pad-factor" => 2,
    "upsample-factor" => 2.0,
    "precision" => "Float32",
    "max-modes" => 6,
    "use-phtovel" => true,
    "selected-phvel-correction-denominator" => 5.5,
    "causal-phvel-correction-denominator" => 5.5,
    "selected-mean-phvel-correction-denominator" => 5.5,
    "top-ranks" => 3,
    "max-triplets" => 0,
    "max-candidates" => 25,
    "sweep-denominators" => "",
    "overwrite" => false,
    "dry-run" => false,
    "no-interactive" => false,
)

const FLOAT_OPTION_KEYS = Set([
    "max-delta-az", "max-delta-d", "min-segment-distance", "dt",
    "period-min", "period-max", "velocity-min", "velocity-max",
    "phase-velocity-min", "phase-velocity-max", "wavelength-ref-velocity",
    "wavelength-fraction", "bandwidth-factor", "upsample-factor",
    "selected-phvel-correction-denominator", "causal-phvel-correction-denominator",
    "selected-mean-phvel-correction-denominator",
])

const INT_OPTION_KEYS = Set([
    "nperiods", "top-ranks", "max-triplets", "max-candidates",
    "zero-pad-factor", "max-modes",
])

const BOOL_OPTION_KEYS = Set(["use-phtovel"])

const TOML_TO_OPTION_KEY = Dict{String,String}(
    "dt" => "dt",
    "period_min" => "period-min",
    "period_max" => "period-max",
    "nperiods" => "nperiods",
    "velocity_min" => "velocity-min",
    "velocity_max" => "velocity-max",
    "phase_velocity_min" => "phase-velocity-min",
    "phase_velocity_max" => "phase-velocity-max",
    "wavelength_ref_velocity" => "wavelength-ref-velocity",
    "wavelength_fraction" => "wavelength-fraction",
    "bandwidth_factor" => "bandwidth-factor",
    "zero_pad_factor" => "zero-pad-factor",
    "upsample_factor" => "upsample-factor",
    "precision" => "precision",
    "max_modes" => "max-modes",
    "use_phtovel" => "use-phtovel",
    "bandpass" => "bandpass",
    "selected_phvel_correction_denominator" => "selected-phvel-correction-denominator",
    "causal_phvel_correction_denominator" => "causal-phvel-correction-denominator",
    "selected_mean_phvel_correction_denominator" => "selected-mean-phvel-correction-denominator",
    "sweep_denominators" => "sweep-denominators",
)

function usage()
    println("""
    Usage:
      julia triplet_analysis_cli.jl --saved-root DIR --reference-pair SN43-SN63 \\
          --raw-data-dir DIR --triplets-csv FILE --output-dir DIR [options]

    Required:
      --saved-root DIR              Root containing trained/transferred source-state artifacts
      --reference-pair PAIR         Reference pair label, e.g. SN43-SN63
      --raw-data-dir DIR            Raw pair directory used for target-pair discovery/reporting
      --output-dir DIR              Destination for CSVs, plots, and selected waveform CSVs

    Options:
      --config FILE                TOML file with MFT/triplet-analysis defaults
      --triplets-csv FILE           Unfiltered triplet CSV. Default: $(DEFAULT_TRIPLETS_CSV)
      --max-delta-az X              Maximum azimuth spread in degrees. Default: 0.7
      --max-delta-d X               Maximum abs distance-closure error in km. Default: 0.1
      --min-segment-distance X      Minimum Dab and Dbc segment distance in km. Default: 45
      --bandpass none|TMIN:TMAX     Zero-phase period bandpass before MFT. Default: 10:30
      --dt X                        Waveform sample interval in seconds. Default: 1
      --period-min X                Minimum MFT period in seconds. Default: 10
      --period-max X                Maximum MFT period in seconds. Default: 30
      --nperiods N                  Number of log-spaced MFT periods. Default: 100
      --velocity-min X              Minimum MFT group/filter velocity. Default: 1
      --velocity-max X              Maximum MFT group/filter velocity. Default: 8
      --phase-velocity-min X        Minimum phase velocity search value. Default: 1
      --phase-velocity-max X        Maximum phase velocity search value. Default: 6
      --wavelength-ref-velocity X   Reference velocity for wavelength period filtering. Default: 2
      --wavelength-fraction X       Distance fraction for wavelength period filtering. Default: 0.33
      --bandwidth-factor X          MFT bandwidth factor. Default: 0.15
      --zero-pad-factor N           MFT zero-pad factor. Default: 2
      --upsample-factor X           MFT upsample factor. Default: 2
      --precision Float32|Float64   MFT numeric precision. Default: Float32
      --max-modes N                 Maximum FTAN modes. Default: 6
      --use-phtovel true|false      Use phase-to-velocity branch. Default: true
      --selected-phvel-correction-denominator X       Selected-state correction denominator. Default: 5.5
      --causal-phvel-correction-denominator X         Global causal correction denominator. Default: 5.5
      --selected-mean-phvel-correction-denominator X  Selected mean correction denominator. Default: 5.5
      --top-ranks N                 Selected ranks per source family. Default: 3
      --max-triplets N              Optional cap for quick checks. Default: all passing triplets
      --max-candidates N            Candidate states per pair before triplet ranking. Default: 25
      --sweep-denominators LIST     Optional comma list for phase sweep. Default: 0,2:0.2:12
      --overwrite                   Replace existing output files
      --dry-run                     Print discovered work without running MFT or writing outputs
      --no-interactive              Do not prompt for waveform export
      --help                        Show this message
    """)
end

function parse_args(argv)
    opts = copy(DEFAULT_OPTIONS)
    explicit = Set{String}()
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg in ("--help", "-h")
            opts["help"] = true
            push!(explicit, "help")
            i += 1
        elseif arg in ("--overwrite", "--dry-run", "--no-interactive")
            opts[arg[3:end]] = true
            push!(explicit, arg[3:end])
            i += 1
        elseif startswith(arg, "--")
            key = arg[3:end]
            i == length(argv) && error("Missing value for $(arg)")
            val = argv[i + 1]
            if key in FLOAT_OPTION_KEYS
                parsed = tryparse(Float64, val)
                isnothing(parsed) && error("$(arg) must be numeric; got $(val)")
                opts[key] = parsed
            elseif key in INT_OPTION_KEYS
                parsed = tryparse(Int, val)
                isnothing(parsed) && error("$(arg) must be an integer; got $(val)")
                opts[key] = parsed
            elseif key in BOOL_OPTION_KEYS
                parsed = parse_bool_option(val, arg)
                opts[key] = parsed
            else
                opts[key] = val
            end
            push!(explicit, key)
            i += 2
        else
            error("Unexpected argument: $(arg)")
        end
    end
    opts["__explicit_keys__"] = explicit
    opts
end

function parse_bool_option(value, flag)
    raw = lowercase(strip(String(value)))
    raw in ("true", "t", "yes", "y", "1") && return true
    raw in ("false", "f", "no", "n", "0") && return false
    error("$(flag) must be true or false; got $(value)")
end

function coerce_config_value(key::String, value)
    if key in FLOAT_OPTION_KEYS
        x = tryparse(Float64, string(value))
        isnothing(x) && error("Config value $(key) must be numeric; got $(value)")
        return x
    elseif key in INT_OPTION_KEYS
        x = value isa Integer ? Int(value) : tryparse(Int, string(value))
        isnothing(x) && error("Config value $(key) must be an integer; got $(value)")
        return x
    elseif key in BOOL_OPTION_KEYS
        value isa Bool && return value
        return parse_bool_option(value, "config $(key)")
    else
        return String(value)
    end
end

function apply_toml_config!(opts)
    config_path = strip(String(get(opts, "config", "")))
    isempty(config_path) && return opts
    isfile(config_path) || error("Config file not found: $(config_path)")
    data = TOML.parsefile(config_path)
    explicit = get(opts, "__explicit_keys__", Set{String}())
    for (toml_key, value) in data
        haskey(TOML_TO_OPTION_KEY, String(toml_key)) ||
            error("Unknown config key $(toml_key) in $(config_path)")
        opt_key = TOML_TO_OPTION_KEY[String(toml_key)]
        opt_key in explicit && continue
        opts[opt_key] = coerce_config_value(opt_key, value)
    end
    opts
end

function parse_denominators(text::AbstractString)
    raw = strip(String(text))
    isempty(raw) && return DEFAULT_DENOMINATORS
    vals = Float64[]
    for part in split(raw, ",")
        p = strip(part)
        isempty(p) && continue
        pieces = split(p, ":")
        if length(pieces) == 1
            v = tryparse(Float64, p)
            isnothing(v) && error("Bad --sweep-denominators value: $(p)")
            push!(vals, v)
        elseif length(pieces) == 3
            a, step, b = tryparse.(Float64, pieces)
            any(isnothing, (a, step, b)) && error("Bad --sweep-denominators range: $(p)")
            step == 0 && error("Bad --sweep-denominators range with zero step: $(p)")
            append!(vals, collect(Float64(a):Float64(step):Float64(b)))
        else
            error("Bad --sweep-denominators entry: $(p). Use values or start:step:stop.")
        end
    end
    isempty(vals) && error("--sweep-denominators produced no values")
    sort(unique(Float64.(vals)))
end

function require_arg(opts, key)
    val = strip(String(get(opts, key, "")))
    isempty(val) && error("Missing required --$(key)")
    val
end

function validate_resolved_options!(opts)
    Float64(opts["dt"]) > 0 || error("--dt must be > 0")
    Float64(opts["period-min"]) > 0 || error("--period-min must be > 0")
    Float64(opts["period-max"]) > Float64(opts["period-min"]) ||
        error("--period-max must be greater than --period-min")
    Int(opts["nperiods"]) >= 2 || error("--nperiods must be >= 2")
    Float64(opts["velocity-min"]) < Float64(opts["velocity-max"]) ||
        error("--velocity-min must be less than --velocity-max")
    Float64(opts["phase-velocity-min"]) < Float64(opts["phase-velocity-max"]) ||
        error("--phase-velocity-min must be less than --phase-velocity-max")
    Float64(opts["wavelength-ref-velocity"]) > 0 || error("--wavelength-ref-velocity must be > 0")
    Float64(opts["wavelength-fraction"]) > 0 || error("--wavelength-fraction must be > 0")
    Float64(opts["bandwidth-factor"]) > 0 || error("--bandwidth-factor must be > 0")
    Int(opts["zero-pad-factor"]) >= 1 || error("--zero-pad-factor must be >= 1")
    Float64(opts["upsample-factor"]) > 0 || error("--upsample-factor must be > 0")
    String(opts["precision"]) in ("Float32", "Float64") || error("--precision must be Float32 or Float64")
    Int(opts["max-modes"]) >= 1 || error("--max-modes must be >= 1")
    for key in ("selected-phvel-correction-denominator", "causal-phvel-correction-denominator",
            "selected-mean-phvel-correction-denominator")
        Float64(opts[key]) > 0 || error("--$(key) must be > 0")
    end
    parse_bandpass(String(opts["bandpass"]))
    parse_denominators(String(opts["sweep-denominators"]))
    opts
end

function correction_from_config_denominator(opts, key::AbstractString)
    -pi / Float64(opts[String(key)])
end

safe_name(s::AbstractString) = (x = replace(strip(String(s)), r"[^A-Za-z0-9_.=+-]+" => "_"); isempty(x) ? "none" : x)

function parse_pair_label_cli(s::AbstractString)
    parts = occursin("-", String(s)) ? split(String(s), "-"; limit=2) : split(String(s), "_"; limit=2)
    length(parts) == 2 || error("Pair must look like SN43-SN63 or SN43_SN63; got $(s)")
    String(strip(parts[1])), String(strip(parts[2]))
end

function parse_bandpass(text::AbstractString)
    raw = lowercase(strip(String(text)))
    raw in ("none", "off", "false", "0") && return nothing
    parts = split(raw, ":")
    length(parts) == 2 || error("--bandpass must be none or TMIN:TMAX, e.g. 10:30")
    vals = tryparse.(Float64, parts)
    any(isnothing, vals) && error("--bandpass has nonnumeric periods: $(text)")
    lo, hi = sort(Float64.(vals))
    lo > 0 && hi > lo || error("--bandpass periods must satisfy 0 < TMIN < TMAX")
    (; period_min=lo, period_max=hi, label="$(lo):$(hi)")
end

function maybe_bandpass_vector(x::AbstractVector, bp, dt::Real)
    y = Float32.(vec(x))
    isnothing(bp) && return y
    fs = 1.0 / Float64(dt)
    f1 = 1.0 / Float64(bp.period_max)
    f2 = 1.0 / Float64(bp.period_min)
    nyq = fs / 2
    f2 < nyq || error("Bandpass high frequency $(f2) Hz exceeds Nyquist $(nyq) Hz for dt=$(dt)")
    filt = DSP.digitalfilter(DSP.Bandpass(f1, f2; fs), DSP.Butterworth(4))
    Float32.(DSP.filtfilt(filt, Float64.(y)))
end

function maybe_bandpass_matrix(X::AbstractMatrix, bp, dt::Real)
    Y = Float32.(X)
    isnothing(bp) && return Y
    out = similar(Y)
    for j in axes(Y, 2)
        out[:, j] .= maybe_bandpass_vector(@view(Y[:, j]), bp, dt)
    end
    out
end

function bandpass_item(item, bp, dt::Real)
    isnothing(bp) && return item
    merge(item, (;
        acausal=maybe_bandpass_matrix(item.acausal, bp, dt),
        causal=maybe_bandpass_matrix(item.causal, bp, dt),
        global_avg_ac=maybe_bandpass_vector(item.global_avg_ac, bp, dt),
        global_avg_c=maybe_bandpass_vector(item.global_avg_c, bp, dt),
        marginal_stage1_ac=maybe_bandpass_matrix(item.marginal_stage1_ac, bp, dt),
        marginal_stage1_c=maybe_bandpass_matrix(item.marginal_stage1_c, bp, dt),
        marginal_stage2_ac=maybe_bandpass_matrix(item.marginal_stage2_ac, bp, dt),
        marginal_stage2_c=maybe_bandpass_matrix(item.marginal_stage2_c, bp, dt)))
end

mean_cols(A::AbstractArray, B::AbstractArray) = begin
    n = min(size(A, 1), size(B, 1))
    m = min(size(A, 2), size(B, 2))
    n == 0 || m == 0 ? Float32[;;] : Float32.(0.5 .* (Float32.(A[1:n, 1:m]) .+ Float32.(B[1:n, 1:m])))
end

mean_vec(a::AbstractVector, b::AbstractVector) = begin
    n = min(length(a), length(b))
    n == 0 ? Float32[] : Float32.(0.5 .* (Float32.(a[1:n]) .+ Float32.(b[1:n])))
end

function mean_item(item)
    c = mean_cols(item.causal, item.acausal)
    s1 = mean_cols(item.marginal_stage1_c, item.marginal_stage1_ac)
    s2 = mean_cols(item.marginal_stage2_c, item.marginal_stage2_ac)
    g = mean_vec(item.global_avg_c, item.global_avg_ac)
    merge(item, (;
        causal=c, acausal=c,
        marginal_stage1_c=s1, marginal_stage1_ac=s1,
        marginal_stage2_c=s2, marginal_stage2_ac=s2,
        global_avg_c=g, global_avg_ac=g))
end

function list_raw_pairs(data_dir::AbstractString)
    isdir(data_dir) || return String[]
    pairs = Set{String}()
    for name in readdir(data_dir)
        m = match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)-.*\.jld2$", name)
        isnothing(m) || push!(pairs, "$(m.captures[1])-$(m.captures[2])")
    end
    sort(collect(pairs))
end

function load_triplets(path::AbstractString; max_delta_az, max_delta_d, min_segment_distance, max_triplets::Int=0)
    isfile(path) || error("Triplet CSV not found: $(path)")
    df = CSV.read(path, DataFrame)
    required = [:triplet, :station_a, :station_b, :station_c, :pair_ab, :pair_bc, :pair_ac,
        :dab_km, :dbc_km, :dac_km, :delta_az_deg, :abs_delta_d_km]
    missing_cols = setdiff(required, Symbol.(names(df)))
    isempty(missing_cols) || error("Triplet CSV is missing required columns: $(missing_cols)")
    mask = (Float64.(df.delta_az_deg) .<= Float64(max_delta_az)) .&
        (Float64.(df.abs_delta_d_km) .<= Float64(max_delta_d)) .&
        (Float64.(df.dab_km) .>= Float64(min_segment_distance)) .&
        (Float64.(df.dbc_km) .>= Float64(min_segment_distance))
    out = df[mask, :]
    max_triplets > 0 && nrow(out) > max_triplets && (out = out[1:max_triplets, :])
    out
end

function geom_from_triplet_row(row, pair_labels)
    station_triple_geometry(String(row.station_a), String(row.station_b), String(row.station_c), pair_labels;
        pair_distances=Dict(
            String(row.pair_ab) => Float64(row.dab_km),
            String(row.pair_bc) => Float64(row.dbc_km),
            String(row.pair_ac) => Float64(row.dac_km)))
end

function latest_trained_items_by_pair(items)
    by_pair = Dict{String,Any}()
    for item in items
        String(item.artifact_kind) == "selected_state_transfer" && continue
        key = String(item.pair_label)
        item_tag = basename(String(item.run_dir))
        old_tag = haskey(by_pair, key) ? basename(String(by_pair[key].run_dir)) : ""
        if !haskey(by_pair, key) || item_tag > old_tag
            by_pair[key] = item
        end
    end
    by_pair
end

function items_for_pairs(items, pairs)
    wanted = Set(String.(pairs))
    [item for item in items if String(item.pair_label) in wanted]
end

function score_state_items(items, cfg; branch_filter::Union{Nothing,String}=nothing,
        branch_label::Union{Nothing,String}=nothing, score_method::AbstractString="geomean",
        huber_delta::Real=0.10)
    analyses = _run_state_mft(items, cfg)
    pair_labels = sort(unique(String.(getproperty.(items, :pair_label))))
    out = Dict{String,Vector{NamedTuple}}(pl => NamedTuple[] for pl in pair_labels)
    for (pair_label, batch) in analyses
        scored = get!(out, String(pair_label), NamedTuple[])
        for (label, res) in batch
            for spec in _score_mft_result(res, String(pair_label), String(label), "state";
                    score_method, huber_delta)
                if isnothing(branch_filter) || String(spec.branch) == branch_filter
                    pushed = isnothing(branch_label) ? spec :
                        merge(spec, (; branch=branch_label,
                            label=replace(String(spec.label), "[$(spec.branch)]" => "[$(branch_label)]")))
                    push!(scored, pushed)
                end
            end
        end
    end
    for scored in values(out)
        sort!(scored; by=s -> (s.uc_score, s.max_relative_error, -s.n_valid,
            _logical_state_display(String(s.display)), String(s.branch)))
    end
    out
end

function analyze_global_rows(items, cfg, family::AbstractString)
    cols = Vector{Vector{Float32}}()
    distances = Float64[]
    pair_keys = String[]
    labels = String[]
    for item in items
        wave = if family == "global_average_causal"
            Float32.(vec(item.global_avg_c))
        elseif family == "global_average_acausal"
            Float32.(vec(item.global_avg_ac))
        elseif family == "global_average_mean"
            mean_vec(item.global_avg_c, item.global_avg_ac)
        else
            error("Unknown global family: $(family)")
        end
        isempty(wave) && continue
        isfinite(item.distance) && item.distance > 0 || continue
        push!(cols, wave)
        push!(distances, Float64(item.distance))
        push!(pair_keys, String(item.pair_label))
        push!(labels, "$(item.pair_label) $(family)")
    end
    n = isempty(cols) ? 0 : minimum(length, cols)
    n == 0 && return NamedTuple[]
    W = reduce(hcat, [view(col, 1:n) for col in cols])
    analyses = _analyze_branch_arrays(W, W, distances, pair_keys, labels, cfg)
    rows = NamedTuple[]
    for (pair_label, batch) in analyses
        isempty(batch) && continue
        label, res = first(collect(batch))
        branch_rows = _uc_rows_from_mft_result(res, String(pair_label), "$(pair_label) $(family)")
        append!(rows, [merge(r, (; branch=family, source_branch=family,
            source_waveform=family)) for r in branch_rows])
    end
    sort(rows; by=r -> (r.pair_label, r.period))
end

function selected_scores_for_family(items, cfg, family::AbstractString; score_method="geomean", huber_delta=0.10)
    if family == "selected_state_causal"
        return score_state_items(items, cfg; branch_filter="causal", branch_label="causal", score_method, huber_delta)
    elseif family == "selected_state_acausal"
        return score_state_items(items, cfg; branch_filter="acausal", branch_label="acausal", score_method, huber_delta)
    elseif family == "selected_state_mean"
        return score_state_items(mean_item.(items), cfg; branch_filter="causal", branch_label="mean", score_method, huber_delta)
    else
        error("Unknown selected family: $(family)")
    end
end

function matching_spec(scored_by_pair, pair_label, template)
    specs = get(scored_by_pair, String(pair_label), NamedTuple[])
    target_display = _logical_state_display(String(template.display))
    target_branch = String(template.branch)
    for s in specs
        _logical_state_display(String(s.display)) == target_display && String(s.branch) == target_branch && return s
    end
    nothing
end

function correction_from_denominator(d::Real)
    d <= 0 ? 0.0 : -pi / Float64(d)
end

function summarize_triplet_rows(rows)
    vals = Float64[abs(Float64(r.vdif)) for r in rows if isfinite(Float64(r.vdif))]
    isempty(vals) && return (; rms=Inf, median_error=Inf, max_error=Inf, n_valid_periods=0)
    (; rms=sqrt(mean(vals .^ 2)), median_error=median(vals), max_error=maximum(vals), n_valid_periods=length(vals))
end

function rows_for_global_triplet(geom, rows; velocity_field::Symbol)
    _station_triple_consistency_rows("global", geom,
        _period_velocity_map(rows, geom.pair_ab; velocity_field),
        _period_velocity_map(rows, geom.pair_bc; velocity_field),
        _period_velocity_map(rows, geom.pair_ac; velocity_field);
        velocity_kind=velocity_field == :phase_velocity ? "phase" : "group",
        data_source="triplet_analysis_cli")
end

function rows_for_selected_combo(geom, combo; velocity_field::Symbol, source_label::String="selected")
    _station_triple_combination_rows(source_label, geom, combo.spec_ab, combo.spec_bc, combo.spec_ac; velocity_field)
end

function global_sweep(items, geom, family, cfg_base, denominators)
    rows = NamedTuple[]
    for d in denominators
        cfg = merge(cfg_base, (; phvel_correction=correction_from_denominator(d), cache=Dict{Any,Any}()))
        global_rows = analyze_global_rows(items, cfg, family)
        triple_rows = rows_for_global_triplet(geom, global_rows; velocity_field=:phase_velocity)
        stats = summarize_triplet_rows(triple_rows)
        isfinite(stats.rms) || continue
        push!(rows, (; denominator=Float64(d), correction=correction_from_denominator(d), stats...))
    end
    rows
end

function selected_sweep(items, geom, combo, family, cfg_base, denominators; score_method="geomean", huber_delta=0.10)
    rows = NamedTuple[]
    triplet_items = items_for_pairs(items, (geom.pair_ab, geom.pair_bc, geom.pair_ac))
    for d in denominators
        cfg = merge(cfg_base, (; phvel_correction=correction_from_denominator(d), cache=Dict{Any,Any}()))
        scored = selected_scores_for_family(triplet_items, cfg, family; score_method, huber_delta)
        spec_ab = matching_spec(scored, geom.pair_ab, combo.spec_ab)
        spec_bc = matching_spec(scored, geom.pair_bc, combo.spec_bc)
        spec_ac = matching_spec(scored, geom.pair_ac, combo.spec_ac)
        any(isnothing, (spec_ab, spec_bc, spec_ac)) && continue
        phase_rows = _station_triple_combination_rows("selected sweep", geom, spec_ab, spec_bc, spec_ac;
            velocity_field=:phase_velocity)
        stats = summarize_triplet_rows(phase_rows)
        isfinite(stats.rms) || continue
        push!(rows, (; denominator=Float64(d), correction=correction_from_denominator(d), stats...))
    end
    rows
end

function argmin_row(sweep_rows)
    isempty(sweep_rows) && return nothing
    best = first(sort(sweep_rows; by=r -> (r.rms, r.denominator)))
    best
end

function recommendation_row(; triplet_row, family, rank, combo=nothing, best, rows_kind)
    state_label = isnothing(combo) ? "" : _logical_state_display(String(combo.label_ab))
    (; triplet=String(triplet_row.triplet),
        pair_ab=String(triplet_row.pair_ab), pair_bc=String(triplet_row.pair_bc), pair_ac=String(triplet_row.pair_ac),
        curve_family=String(family), selected_rank=rank,
        selected_state_label=state_label,
        argmin_denominator=Float64(best.denominator),
        correction_label=best.denominator <= 0 ? "0" : @sprintf("-pi/%.3g", best.denominator),
        correction_value=Float64(best.correction),
        min_rms_abs_vabc_minus_vac=Float64(best.rms),
        median_closure_error=Float64(best.median_error),
        max_closure_error=Float64(best.max_error),
        n_valid_periods=Int(best.n_valid_periods),
        row_type=rows_kind)
end

function write_csv(path::AbstractString, table; overwrite::Bool)
    mkpath(dirname(path))
    isfile(path) && !overwrite && error("Refusing to overwrite $(path); pass --overwrite")
    CSV.write(path, table)
    path
end

function activate_ftan_project!()
    ftan_root = abspath(joinpath(dirname(CLI_FTAN_PATH), ".."))
    project = joinpath(ftan_root, "Project.toml")
    isfile(project) || return false
    try
        Pkg.activate(ftan_root; io=devnull)
        return true
    catch err
        @warn "Could not activate FTAN project; MFT may fail if FTAN dependencies are not in the active environment" ftan_root error=err
        return false
    end
end

function html_escape(s::AbstractString)
    replace(String(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")
end

function write_argmin_html(path::AbstractString, df::DataFrame; overwrite::Bool)
    mkpath(dirname(path))
    isfile(path) && !overwrite && error("Refusing to overwrite $(path); pass --overwrite")
    traces = String[]
    for family in sort(unique(String.(df.curve_family)))
        sub = df[String.(df.curve_family) .== family, :]
        xs = join(string.(Float64.(sub.argmin_denominator)), ",")
        ys = join(string.(Float64.(sub.min_rms_abs_vabc_minus_vac)), ",")
        labels = join(["\"" * html_escape(String(t)) * "\"" for t in sub.triplet], ",")
        push!(traces, "{x:[$xs], y:[$ys], text:[$labels], mode:'markers', type:'scatter', name:'$(html_escape(family))', marker:{size:9}}")
    end
    body = """
    <!doctype html>
    <html><head><meta charset="utf-8"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
    <body><div id="plot" style="width:1200px;height:760px;"></div>
    <script>
    const data = [$(join(traces, ","))];
    const layout = {
      title: 'Combined phase-correction argmin across triplets',
      xaxis: {title: 'Argmin denominator d in -pi/d (0 = no correction)'},
      yaxis: {title: 'Minimum RMS abs(Vabc - Vac) (km/s)'},
      hovermode: 'closest'
    };
    Plotly.newPlot('plot', data, layout, {responsive: true});
    </script></body></html>
    """
    open(path, "w") do io
        write(io, body)
    end
    path
end

function write_argmin_png(path::AbstractString, df::DataFrame; overwrite::Bool)
    mkpath(dirname(path))
    isfile(path) && !overwrite && error("Refusing to overwrite $(path); pass --overwrite")
    tmp = tempname() * ".csv"
    CSV.write(tmp, df)
    py = raw"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path, out_path = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_path)
fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
for family, sub in df.groupby("curve_family", sort=True):
    ax.scatter(sub["argmin_denominator"], sub["min_rms_abs_vabc_minus_vac"], s=28, label=family, alpha=0.85)
ax.set_title("Combined phase-correction argmin across triplets")
ax.set_xlabel("Argmin denominator d in -pi/d (0 = no correction)")
ax.set_ylabel("Minimum RMS abs(Vabc - Vac) (km/s)")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(out_path)
"""
    try
        run(`python3 -c $py $tmp $path`)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    path
end

function selected_closure_summary(triplet_rows_df::DataFrame, selected_recs::DataFrame)
    isempty(triplet_rows_df) && return DataFrame()
    isempty(selected_recs) && return DataFrame()
    wanted = Set(String.(selected_recs.selected_state_label))
    rows = triplet_rows_df[in.(String.(triplet_rows_df.selected_state_label), Ref(wanted)), :]
    isempty(rows) && return DataFrame()
    out = NamedTuple[]
    for sub in groupby(rows, [:selected_state_label, :curve_family, :velocity_kind, :period_s])
        vals = Float64.(sub.Vabc_minus_Vac_km_s)
        absvals = abs.(vals)
        push!(out, (; selected_state_label=String(first(sub.selected_state_label)),
            curve_family=String(first(sub.curve_family)),
            velocity_kind=String(first(sub.velocity_kind)),
            period_s=Float64(first(sub.period_s)),
            mean_Vabc_minus_Vac_km_s=mean(vals),
            mean_abs_Vabc_minus_Vac_km_s=mean(absvals),
            median_abs_Vabc_minus_Vac_km_s=median(absvals),
            max_abs_Vabc_minus_Vac_km_s=maximum(absvals),
            n_triplet_periods=nrow(sub)))
    end
    sort!(DataFrame(out), [:selected_state_label, :curve_family, :velocity_kind, :period_s])
end

function write_selected_closure_plots(output_dir::AbstractString, summary_df::DataFrame; overwrite::Bool)
    isempty(summary_df) && return String[]
    plots_dir = joinpath(output_dir, "plots", "selected_state_closure")
    mkpath(plots_dir)
    csv_path = joinpath(plots_dir, "selected_state_phase_group_mean_closure.csv")
    write_csv(csv_path, summary_df; overwrite)
    html_path = joinpath(plots_dir, "selected_state_phase_group_mean_closure.html")
    png_path = joinpath(plots_dir, "selected_state_phase_group_mean_closure.png")
    isfile(html_path) && !overwrite && error("Refusing to overwrite $(html_path); pass --overwrite")
    traces = String[]
    for state in sort(unique(String.(summary_df.selected_state_label)))
        for family in sort(unique(String.(summary_df.curve_family)))
            for velocity_kind in ("phase", "group")
                sub = summary_df[(String.(summary_df.selected_state_label) .== state) .&
                    (String.(summary_df.curve_family) .== family) .&
                    (String.(summary_df.velocity_kind) .== velocity_kind), :]
                isempty(sub) && continue
                xs = join(string.(Float64.(sub.period_s)), ",")
                ys = join(string.(Float64.(sub.mean_Vabc_minus_Vac_km_s)), ",")
                ysabs = join(string.(Float64.(sub.mean_abs_Vabc_minus_Vac_km_s)), ",")
                name = html_escape("$(velocity_kind) | $(family) | $(state)")
                push!(traces, "{x:[$xs], y:[$ys], mode:'lines+markers', type:'scatter', name:'mean $(name)'}")
                push!(traces, "{x:[$xs], y:[$ysabs], mode:'lines+markers', type:'scatter', line:{dash:'dash'}, name:'mean abs $(name)'}")
            end
        end
    end
    body = """
    <!doctype html>
    <html><head><meta charset="utf-8"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
    <body><div id="plot" style="width:1250px;height:780px;"></div>
    <script>
    const data = [$(join(traces, ","))];
    const layout = {
      title: 'Final selected-state closure: mean Vabc - Vac and mean abs(Vabc - Vac)',
      xaxis: {title: 'Period (s)'},
      yaxis: {title: 'Closure error (km/s)'},
      hovermode: 'closest'
    };
    Plotly.newPlot('plot', data, layout, {responsive: true});
    </script></body></html>
    """
    open(html_path, "w") do io
        write(io, body)
    end
    tmp = tempname() * ".csv"
    CSV.write(tmp, summary_df)
    py = raw"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path, out_path = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_path)
fig, axes = plt.subplots(2, 1, figsize=(12, 9), dpi=180, sharex=True)
for ax, vel in zip(axes, ["phase", "group"]):
    sub0 = df[df["velocity_kind"] == vel]
    for (state, fam), sub in sub0.groupby(["selected_state_label", "curve_family"], sort=True):
        label = f"{fam} | {state}"
        ax.plot(sub["period_s"], sub["mean_Vabc_minus_Vac_km_s"], marker="o", linewidth=1.3, label=label)
        ax.plot(sub["period_s"], sub["mean_abs_Vabc_minus_Vac_km_s"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"{vel} closure (km/s)")
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Period (s)")
axes[0].set_title("Final selected-state mean closure; dashed = mean abs(Vabc - Vac)")
axes[0].legend(fontsize=7, loc="best")
fig.tight_layout()
fig.savefig(out_path)
"""
    try
        run(`python3 -c $py $tmp $png_path`)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    [csv_path, html_path, png_path]
end

function state_waveform_lookup(items)
    out = Dict{Tuple{String,String,String},Vector{Float32}}()
    for item in items
        prefix = String(item.artifact_kind) == "selected_state_transfer" ?
            "$(item.pair_label) selected transfer from $(item.reference_pair_label) seed $(item.seed)" :
            "$(item.pair_label) seed $(item.seed)"
        nstates = min(size(item.causal, 2), size(item.acausal, 2), length(item.combo_labels))
        for i in 1:nstates
            base = "$(prefix) | $(item.combo_labels[i])"
            display = _logical_state_display(String(_score_display_from_state_label(base, "state").display))
            out[(String(item.pair_label), display, "causal")] = Float32.(item.causal[:, i])
            out[(String(item.pair_label), display, "acausal")] = Float32.(item.acausal[:, i])
            out[(String(item.pair_label), display, "mean")] = mean_vec(item.causal[:, i], item.acausal[:, i])
        end
        for (matc, mata, labels, stage) in ((item.marginal_stage1_c, item.marginal_stage1_ac, item.marginal_stage1_labels, "S1"),
                (item.marginal_stage2_c, item.marginal_stage2_ac, item.marginal_stage2_labels, "S2"))
            kmax = min(size(matc, 2), size(mata, 2), length(labels))
            for k in 1:kmax
                base = "$(prefix) | $(stage) $(labels[k])"
                display = _logical_state_display(String(_score_display_from_state_label(base, "state").display))
                out[(String(item.pair_label), display, "causal")] = Float32.(matc[:, k])
                out[(String(item.pair_label), display, "acausal")] = Float32.(mata[:, k])
                out[(String(item.pair_label), display, "mean")] = mean_vec(matc[:, k], mata[:, k])
            end
        end
    end
    out
end

function global_waveform_lookup(items)
    by_pair = latest_trained_items_by_pair(items)
    out = Dict{Tuple{String,String},Vector{Float32}}()
    for (pair, item) in by_pair
        out[(pair, "global_average_causal")] = Float32.(vec(item.global_avg_c))
        out[(pair, "global_average_acausal")] = Float32.(vec(item.global_avg_ac))
        out[(pair, "global_average_mean")] = mean_vec(item.global_avg_c, item.global_avg_ac)
    end
    out
end

function write_waveform_csv(path, waves::Dict{String,Vector{Float32}}; dt, overwrite)
    mkpath(dirname(path))
    isfile(path) && !overwrite && error("Refusing to overwrite $(path); pass --overwrite")
    n = isempty(waves) ? 0 : minimum(length, values(waves))
    df = DataFrame(sample=collect(1:n), time_s=(collect(0:n-1) .* Float64(dt)))
    for key in sort(collect(keys(waves)))
        df[!, Symbol(safe_name(key))] = Float64.(waves[key][1:n])
    end
    CSV.write(path, df)
end

function export_waveforms(output_dir, selected_recs::DataFrame, items; dt, bandpass_label, overwrite)
    state_lookup = state_waveform_lookup(items)
    global_lookup = global_waveform_lookup(items)
    exported = String[]
    seen_states = Set{String}()
    for rec in eachrow(selected_recs)
        state_label = String(rec.selected_state_label)
        isempty(strip(state_label)) && continue
        state_label in seen_states && continue
        push!(seen_states, state_label)
        state_dir = joinpath(output_dir, "selected_waveforms", "finalized_states", safe_name(state_label))
        for family in ("global_average_mean", "global_average_causal", "global_average_acausal")
            waves = Dict{String,Vector{Float32}}()
            for (pair, fam) in keys(global_lookup)
                fam == family || continue
                waves[pair] = global_lookup[(pair, family)]
            end
            !isempty(waves) && write_waveform_csv(joinpath(state_dir, "$(family).csv"), waves; dt, overwrite)
        end
        for branch in ("causal", "acausal", "mean")
            waves = Dict{String,Vector{Float32}}()
            for (pair, label, br) in keys(state_lookup)
                label == state_label && br == branch || continue
                waves[pair] = state_lookup[(pair, state_label, branch)]
            end
            !isempty(waves) && write_waveform_csv(joinpath(state_dir, "selected_state_$(branch).csv"), waves; dt, overwrite)
        end
        rec_rows = selected_recs[String.(selected_recs.selected_state_label) .== state_label, :]
        n_global_pairs = length(unique(first.(keys(global_lookup))))
        n_state_pairs = count(k -> k[2] == state_label && k[3] == "causal", collect(keys(state_lookup)))
        picked_rows = join(string.(rec_rows.triplet) .* " | " .* string.(rec_rows.curve_family) .*
            " rank " .* string.(rec_rows.selected_rank), "; ")
        meta = DataFrame(key=["state_label", "bandpass", "dt", "n_global_pairs",
                "n_selected_state_pairs", "selected_from_recommendation_rows"],
            value=[state_label, bandpass_label, string(dt), string(n_global_pairs),
                string(n_state_pairs), picked_rows])
        write_csv(joinpath(state_dir, "metadata.csv"), meta; overwrite)
        push!(exported, state_dir)
    end
    exported
end

function prompt_for_exports(rec_df::DataFrame)
    selected = rec_df[.!isempty.(strip.(String.(rec_df.selected_state_label))), :]
    isempty(selected) && return Int[]
    println()
    println("Recommended selected-state rows available. Choose the row(s) to finalize for closure plots and all-pair waveform export:")
    for (i, row) in enumerate(eachrow(selected))
        @printf("  %d. %s | %s | rank %d | d=%.3g | RMS=%.5g | %s\n",
            i, row.triplet, row.curve_family, row.selected_rank,
            row.argmin_denominator, row.min_rms_abs_vabc_minus_vac, row.selected_state_label)
    end
    print("Enter row numbers to finalize (comma-separated), 'all', or press Enter for none: ")
    line = try readline(stdin) catch; "" end
    line = strip(line)
    isempty(line) && return Int[]
    lowercase(line) == "all" && return collect(1:nrow(selected))
    idx = Int[]
    for part in split(line, r"[, ]+")
        isempty(part) && continue
        n = tryparse(Int, part)
        isnothing(n) || n < 1 || n > nrow(selected) ? println("Ignoring invalid selection: $(part)") : push!(idx, n)
    end
    idx
end

function main(argv=ARGS)
    opts = parse_args(argv)
    get(opts, "help", false) && (usage(); return 0)
    apply_toml_config!(opts)
    validate_resolved_options!(opts)

    saved_root = require_arg(opts, "saved-root")
    _ = parse_pair_label_cli(require_arg(opts, "reference-pair"))
    raw_data_dir = require_arg(opts, "raw-data-dir")
    output_dir = require_arg(opts, "output-dir")
    triplets_csv = String(opts["triplets-csv"])
    overwrite = Bool(opts["overwrite"])
    dry_run = Bool(opts["dry-run"])
    top_ranks = Int(opts["top-ranks"])
    max_candidates = Int(opts["max-candidates"])
    denominators = parse_denominators(String(opts["sweep-denominators"]))
    dt = Float64(opts["dt"])
    bp = parse_bandpass(String(opts["bandpass"]))
    bandpass_label = isnothing(bp) ? "none" : "$(bp.period_min):$(bp.period_max)"

    raw_pairs = list_raw_pairs(raw_data_dir)
    triplets = load_triplets(triplets_csv;
        max_delta_az=opts["max-delta-az"],
        max_delta_d=opts["max-delta-d"],
        min_segment_distance=opts["min-segment-distance"],
        max_triplets=Int(opts["max-triplets"]))

    runs = discover_all_vqvae_runs(saved_root; transfer_root=saved_root)
    pair_labels = sort(unique(String.(getproperty.(runs, :pair_label))))
    needed_pairs = sort(unique(vcat(String.(triplets.pair_ab), String.(triplets.pair_bc), String.(triplets.pair_ac))))
    available_needed = intersect(needed_pairs, pair_labels)

    println("Triplet analysis CLI")
    println("  triplets CSV: $(triplets_csv)")
    println("  raw pairs discovered: $(length(raw_pairs))")
    println("  artifact runs discovered: $(length(runs))")
    println("  triplets passing criteria: $(nrow(triplets))")
    println("  needed pairs with artifacts: $(length(available_needed)) / $(length(needed_pairs))")
    println("  bandpass: $(bandpass_label)")
    println("  MFT period band: $(opts["period-min"])-$(opts["period-max"]) s ($(opts["nperiods"]) periods)")
    println("  MFT velocity range: $(opts["velocity-min"])-$(opts["velocity-max"]) km/s")
    println("  MFT phase velocity range: $(opts["phase-velocity-min"])-$(opts["phase-velocity-max"]) km/s")
    println("  wavelength filter: ref velocity=$(opts["wavelength-ref-velocity"]) km/s, fraction=$(opts["wavelength-fraction"])")
    println("  filter bank: bandwidth=$(opts["bandwidth-factor"]), zero-pad=$(opts["zero-pad-factor"]), upsample=$(opts["upsample-factor"]), precision=$(opts["precision"])")
    println("  sweep denominators: $(length(denominators)) values")

    if dry_run
        println("Dry run only. Expected outputs:")
        for name in ("triplet_analysis_summary.csv", "triplet_argmin_recommendations.csv",
                "triplet_ranked_selected_states.csv", "plots/combined_argmin_all_triplets.html",
                "plots/combined_argmin_all_triplets.png", "selected_waveforms/")
            println("  ", joinpath(output_dir, name))
        end
        return 0
    end

    isempty(runs) && error("No source-state artifacts found under $(saved_root)")
    isempty(triplets) && error("No triplets pass the requested criteria")

    cfg_base = _mft_config(; dt,
        period_min=Float64(opts["period-min"]),
        period_max=Float64(opts["period-max"]),
        nperiods=Int(opts["nperiods"]),
        wavelength_ref_velocity=Float64(opts["wavelength-ref-velocity"]),
        wavelength_fraction=Float64(opts["wavelength-fraction"]),
        velocity_range=(Float64(opts["velocity-min"]), Float64(opts["velocity-max"])),
        bandwidth_factor=Float64(opts["bandwidth-factor"]),
        zero_pad_factor=Int(opts["zero-pad-factor"]),
        upsample_factor=Float64(opts["upsample-factor"]),
        precision=String(opts["precision"]),
        max_modes=Int(opts["max-modes"]),
        phase_velocity_range=(Float64(opts["phase-velocity-min"]), Float64(opts["phase-velocity-max"])),
        use_phtovel=Bool(opts["use-phtovel"]),
        phvel_correction=0.0)
    cfg_selected = merge(cfg_base, (; phvel_correction=correction_from_config_denominator(opts, "selected-phvel-correction-denominator"), cache=Dict{Any,Any}()))
    cfg_causal = merge(cfg_base, (; phvel_correction=correction_from_config_denominator(opts, "causal-phvel-correction-denominator"), cache=Dict{Any,Any}()))
    cfg_selected_mean = merge(cfg_base, (; phvel_correction=correction_from_config_denominator(opts, "selected-mean-phvel-correction-denominator"), cache=Dict{Any,Any}()))

    items0 = [load_source_state_artifact(run) for run in runs]
    items = [bandpass_item(item, bp, dt) for item in items0]
    activate_ftan_project!()
    global_items = collect(values(latest_trained_items_by_pair(items)))

    summary_rows = NamedTuple[]
    recommendation_rows = NamedTuple[]
    ranked_rows = NamedTuple[]
    all_triplet_rows = NamedTuple[]

    selected_families = ("selected_state_causal", "selected_state_acausal", "selected_state_mean")
    global_families = ("global_average_mean", "global_average_causal", "global_average_acausal")

    for (itrip, trow) in enumerate(eachrow(triplets))
        triplet_label = String(trow.triplet)
        @printf("[%d/%d] %s\n", itrip, nrow(triplets), triplet_label)
        pairs = (String(trow.pair_ab), String(trow.pair_bc), String(trow.pair_ac))
        geom = geom_from_triplet_row(trow, pairs)
        trip_items = items_for_pairs(items, pairs)
        trip_global_items = items_for_pairs(global_items, pairs)
        length(unique(String.(getproperty.(trip_items, :pair_label)))) >= 3 || begin
            push!(summary_rows, (; triplet=triplet_label, status="missing_artifacts", message="Missing artifacts for one or more triplet pairs"))
            continue
        end

        try
            for family in global_families
                cfg = family == "global_average_causal" ? cfg_causal : cfg_base
                g_rows = analyze_global_rows(trip_global_items, cfg, family)
                for vf in (:group_velocity, :phase_velocity)
                    append!(all_triplet_rows, [merge(r, (; triplet=triplet_label, curve_family=family, selected_rank=0,
                        selected_state_label=""))
                        for r in rows_for_global_triplet(geom, g_rows; velocity_field=vf)])
                end
                sweep = global_sweep(trip_global_items, geom, family, cfg_base, denominators)
                best = argmin_row(sweep)
                isnothing(best) || push!(recommendation_rows, recommendation_row(;
                    triplet_row=trow, family, rank=0, best, rows_kind="global"))
            end

            for family in selected_families
                cfg_rank = family == "selected_state_mean" ? cfg_selected_mean : cfg_selected
                scored = selected_scores_for_family(trip_items, cfg_rank, family)
                ranked = _rank_station_triple_state_combinations(geom, scored;
                    max_candidates, velocity_field=:group_velocity)
                for (irank, combo) in enumerate(ranked[1:min(top_ranks, length(ranked))])
                    label = _logical_state_display(String(combo.label_ab))
                    push!(ranked_rows, (; triplet=triplet_label, ranking_family=family, rank=irank,
                        group_rms_abs_vabc_minus_vac=Float64(combo.rms_vdiff),
                        n_periods=Int(combo.n_periods), state_label=label,
                        ab=String(combo.label_ab), bc=String(combo.label_bc), ac=String(combo.label_ac)))
                    for vf in (:group_velocity, :phase_velocity)
                        append!(all_triplet_rows, [merge(r, (; triplet=triplet_label, curve_family=family,
                            selected_rank=irank, selected_state_label=label))
                            for r in rows_for_selected_combo(geom, combo; velocity_field=vf,
                                source_label="$(family) rank $(irank)")])
                    end
                    sweep = selected_sweep(trip_items, geom, combo, family, cfg_base, denominators)
                    best = argmin_row(sweep)
                    isnothing(best) || push!(recommendation_rows, recommendation_row(;
                        triplet_row=trow, family, rank=irank, combo, best, rows_kind="selected"))
                end
            end
            push!(summary_rows, (; triplet=triplet_label, status="ok", message=""))
        catch err
            push!(summary_rows, (; triplet=triplet_label, status="error", message=sprint(showerror, err)))
            @warn "Triplet failed" triplet=triplet_label error=err
        end
    end

    mkpath(output_dir)
    rec_df = isempty(recommendation_rows) ? DataFrame() : DataFrame(recommendation_rows)
    ranked_df = isempty(ranked_rows) ? DataFrame() : DataFrame(ranked_rows)
    summary_df = isempty(summary_rows) ? DataFrame() : DataFrame(summary_rows)
    triplet_rows_df = isempty(all_triplet_rows) ? DataFrame() : station_triple_dataframe(all_triplet_rows)

    write_csv(joinpath(output_dir, "triplet_analysis_summary.csv"), summary_df; overwrite)
    write_csv(joinpath(output_dir, "triplet_argmin_recommendations.csv"), rec_df; overwrite)
    write_csv(joinpath(output_dir, "triplet_ranked_selected_states.csv"), ranked_df; overwrite)
    write_csv(joinpath(output_dir, "triplet_phase_group_rows.csv"), triplet_rows_df; overwrite)

    plots_dir = joinpath(output_dir, "plots")
    if !isempty(rec_df)
        write_argmin_html(joinpath(plots_dir, "combined_argmin_all_triplets.html"), rec_df; overwrite)
        write_argmin_png(joinpath(plots_dir, "combined_argmin_all_triplets.png"), rec_df; overwrite)
    end

    println()
    println("Wrote:")
    println("  ", joinpath(output_dir, "triplet_analysis_summary.csv"))
    println("  ", joinpath(output_dir, "triplet_argmin_recommendations.csv"))
    println("  ", joinpath(output_dir, "triplet_ranked_selected_states.csv"))
    println("  ", joinpath(output_dir, "triplet_phase_group_rows.csv"))
    println("  ", joinpath(plots_dir, "combined_argmin_all_triplets.html"))

    if !Bool(opts["no-interactive"]) && !isempty(rec_df)
        selected_view = rec_df[.!isempty.(strip.(String.(rec_df.selected_state_label))), :]
        idx = prompt_for_exports(rec_df)
        if !isempty(idx)
            final_selection = selected_view[idx, :]
            closure_outputs = write_selected_closure_plots(output_dir,
                selected_closure_summary(triplet_rows_df, final_selection); overwrite)
            if !isempty(closure_outputs)
                println("Wrote finalized selected-state closure outputs:")
                foreach(p -> println("  ", p), closure_outputs)
            end
            exported = export_waveforms(output_dir, final_selection, items;
                dt, bandpass_label, overwrite)
            println("Exported selected waveform folders:")
            foreach(p -> println("  ", p), exported)
        end
    end

    ok = any(String.(summary_df.status) .== "ok")
    ok ? 0 : 2
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main())
    catch err
        showerror(stderr, err)
        println(stderr)
        exit(1)
    end
end
