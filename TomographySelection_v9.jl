module TomographySelection_v9

using Statistics
using Printf

export ConsensusPick, ReceiverPairGeometry, PairConsensusForTomography,
       OutlierCandidateSelection, consensus_from_pick_rows, pick_cloud_consensus_from_rows,
       match_global_branch_rows,
       receiver_pair_geometry, tomography_pair_consensus,
       select_outlier_candidates, selected_dsurftomo_rows, write_dsurftomo_rows,
       wavelength_valid_period

struct ConsensusPick
    periods::Vector{Float64}
    group_velocities::Vector{Float64}
    confidence::Vector{Float64}
    support::Vector{Int}
    candidate_velocities::Vector{Vector{Float64}}
    candidate_group_velocities::Matrix{Float64}
    candidate_confidence::Matrix{Float64}
    candidate_support::Matrix{Int}
end

struct ReceiverPairGeometry
    station1::String
    station2::String
    lat1::Float64
    lon1::Float64
    lat2::Float64
    lon2::Float64
    midpoint_lat::Float64
    midpoint_lon::Float64
    distance_km::Float64
    azimuth_deg::Float64
end

struct PairConsensusForTomography
    label::String
    pair::Tuple{String,String}
    geometry::ReceiverPairGeometry
    consensus::ConsensusPick
end

struct OutlierCandidateSelection
    pair_labels::Vector{String}
    selected_candidate_indices::Vector{Int}
    selected_rows::Vector{NamedTuple}
    pair_diagnostics::Vector{NamedTuple}
    period_stats::Dict{Float64,NamedTuple}
    iterations::Int
    converged::Bool
end

struct _Candidate
    velocity::Float64
    state::String
    quality::Float64
end

struct _Cluster
    velocity::Float64
    confidence::Float64
    support::Int
    candidate_indices::Vector{Int}
end

function _get(row, name::Symbol, default)
    hasproperty(row, name) && return getproperty(row, name)
    return default
end

function wavelength_valid_period(period::Real, distance::Real;
        wavelength_ref_velocity::Real=2.0, wavelength_fraction::Real=0.33)
    vals = Float64.(collect((period, distance, wavelength_ref_velocity, wavelength_fraction)))
    all(isfinite, vals) || return false
    all(>(0.0), vals) || return false
    return vals[3] * vals[1] < vals[4] * vals[2]
end

_relative_difference(a::Real, b::Real) =
    abs(Float64(a) - Float64(b)) / max((abs(Float64(a)) + abs(Float64(b))) / 2, eps(Float64))

function _mean_finite(vals)
    good = [Float64(v) for v in vals if isfinite(Float64(v))]
    isempty(good) && return NaN
    return mean(good)
end

function _median_finite(vals)
    good = sort([Float64(v) for v in vals if isfinite(Float64(v))])
    isempty(good) && return NaN
    return median(good)
end

function _cluster_candidates(candidates::Vector{_Candidate}, nstates::Int;
        cluster_tolerance_fraction::Float64=0.10)
    isempty(candidates) && return _Cluster[]
    groups = Vector{Vector{Int}}()
    for idx in sortperm([c.velocity for c in candidates])
        if isempty(groups)
            push!(groups, [idx])
            continue
        end
        center = _median_finite(candidates[j].velocity for j in last(groups))
        _relative_difference(candidates[idx].velocity, center) <= cluster_tolerance_fraction ?
            push!(last(groups), idx) : push!(groups, [idx])
    end

    clusters = _Cluster[]
    for group in groups
        vels = [candidates[j].velocity for j in group]
        qualities = clamp.([candidates[j].quality for j in group], 0.0, 1.0)
        support = length(unique(candidates[j].state for j in group))
        support_fraction = nstates > 0 ? support / nstates : 0.0
        center = _median_finite(vels)
        compactness = if length(vels) > 1 && isfinite(center) && center > 0
            clamp(1 - maximum(abs.(vels .- center)) /
                max(center * cluster_tolerance_fraction, eps(Float64)), 0.0, 1.0)
        else
            1.0
        end
        confidence = clamp(0.45 * support_fraction + 0.35 * _mean_finite(qualities) +
            0.20 * compactness, 0.0, 1.0)
        push!(clusters, _Cluster(_mean_finite(vels), confidence, support, group))
    end
    return clusters
end

function _low_velocity_bonus(clusters, j, eligible)
    vals = [clusters[k].velocity for k in eligible if isfinite(clusters[k].velocity)]
    isempty(vals) && return 0.0
    span = maximum(vals) - minimum(vals)
    span <= eps(Float64) && return 0.5
    return clamp((maximum(vals) - clusters[j].velocity) / span, 0.0, 1.0)
end

function _select_candidate_branches(period_clusters;
        max_candidates::Int=5, min_support::Int=1, min_confidence::Float64=0.0,
        smoothness_weight::Float64=1.0, max_smooth_jump_fraction::Float64=0.08,
        max_gap_periods::Int=1, selection_mode::Symbol=:low_velocity)
    nperiods = length(period_clusters)
    selected = zeros(Int, nperiods, max_candidates)
    used = [falses(length(clusters)) for clusters in period_clusters]
    eligible(ip) = [j for j in eachindex(period_clusters[ip])
        if !used[ip][j] && period_clusters[ip][j].support >= min_support &&
           period_clusters[ip][j].confidence >= min_confidence]

    for icand in 1:max_candidates
        seed_ip = 0
        seed_j = 0
        seed_score = -Inf
        for ip in 1:nperiods
            elig = eligible(ip)
            for j in elig
                c = period_clusters[ip][j]
                bonus = selection_mode == :low_velocity ? 0.12 * _low_velocity_bonus(period_clusters[ip], j, elig) : 0.0
                score = c.confidence + 0.08 * c.support + bonus
                if score > seed_score
                    seed_score, seed_ip, seed_j = score, ip, j
                end
            end
        end
        seed_j == 0 && break
        selected[seed_ip, icand] = seed_j
        used[seed_ip][seed_j] = true
        for direction in (-1, 1)
            prev_velocity = period_clusters[seed_ip][seed_j].velocity
            gap_count = 0
            range = direction == -1 ? ((seed_ip - 1):-1:1) : ((seed_ip + 1):nperiods)
            for ip in range
                best_j = 0
                best_score = -Inf
                elig = eligible(ip)
                for j in elig
                    c = period_clusters[ip][j]
                    jump = _relative_difference(c.velocity, prev_velocity)
                    jump <= max_smooth_jump_fraction || continue
                    bonus = selection_mode == :low_velocity ? 0.12 * _low_velocity_bonus(period_clusters[ip], j, elig) : 0.0
                    score = c.confidence + 0.08 * c.support + bonus - smoothness_weight * jump
                    if score > best_score
                        best_score, best_j = score, j
                    end
                end
                if best_j == 0
                    gap_count += 1
                    gap_count > max_gap_periods && break
                    continue
                end
                selected[ip, icand] = best_j
                used[ip][best_j] = true
                prev_velocity = period_clusters[ip][best_j].velocity
                gap_count = 0
            end
        end
    end
    return selected
end

"""
Build smooth consensus candidate branches from pick rows with `period`,
`group_velocity`, and optional `state_label` and `quality` fields.
"""
function consensus_from_pick_rows(rows; periods=nothing, max_candidates::Int=5,
        cluster_tolerance_fraction::Float64=0.10, min_candidate_periods::Int=3,
        min_support::Int=1, min_confidence::Float64=0.0,
        max_smooth_jump_fraction::Float64=0.08, max_gap_periods::Int=1,
        selection_mode::Symbol=:low_velocity)
    valid_rows = [r for r in rows if isfinite(Float64(_get(r, :period, NaN))) &&
        isfinite(Float64(_get(r, :group_velocity, NaN))) &&
        Float64(_get(r, :period, NaN)) > 0 && Float64(_get(r, :group_velocity, NaN)) > 0]
    ps = isnothing(periods) ? sort(unique(Float64(_get(r, :period, NaN)) for r in valid_rows)) :
        sort(Float64.(periods))
    isempty(ps) && return ConsensusPick(Float64[], Float64[], Float64[], Int[],
        Vector{Float64}[], zeros(0, max_candidates), zeros(0, max_candidates), zeros(Int, 0, max_candidates))
    period_index = Dict(p => i for (i, p) in enumerate(ps))
    state_labels = unique(String(_get(r, :state_label, "pick")) for r in valid_rows)
    nstates = max(length(state_labels), 1)
    grouped = [Vector{_Candidate}() for _ in ps]
    candidate_velocities = [Float64[] for _ in ps]
    for row in valid_rows
        p = Float64(_get(row, :period, NaN))
        haskey(period_index, p) || continue
        v = Float64(_get(row, :group_velocity, NaN))
        q = Float64(_get(row, :quality, _get(row, :branch_correlation, 1.0)))
        q = isfinite(q) ? q : 1.0
        i = period_index[p]
        push!(grouped[i], _Candidate(v, String(_get(row, :state_label, "pick")), q))
        push!(candidate_velocities[i], v)
    end
    clusters = [_cluster_candidates(cands, nstates; cluster_tolerance_fraction) for cands in grouped]
    selected = _select_candidate_branches(clusters; max_candidates, min_support,
        min_confidence, max_smooth_jump_fraction, max_gap_periods, selection_mode)

    candidate_v = fill(NaN, length(ps), max_candidates)
    candidate_conf = zeros(length(ps), max_candidates)
    candidate_support = zeros(Int, length(ps), max_candidates)
    for ic in 1:max_candidates, ip in eachindex(ps)
        j = selected[ip, ic]
        j == 0 && continue
        c = clusters[ip][j]
        candidate_v[ip, ic] = c.velocity
        candidate_conf[ip, ic] = c.confidence
        candidate_support[ip, ic] = c.support
    end
    for ic in 1:max_candidates
        if count(v -> isfinite(v) && v > 0, candidate_v[:, ic]) < min_candidate_periods
            candidate_v[:, ic] .= NaN
            candidate_conf[:, ic] .= 0
            candidate_support[:, ic] .= 0
        end
    end
    counts = [count(v -> isfinite(v) && v > 0, candidate_v[:, ic]) for ic in 1:max_candidates]
    medians = [_median_finite(candidate_v[:, ic]) for ic in 1:max_candidates]
    confs = [_mean_finite([c for c in candidate_conf[:, ic] if c > 0]) for ic in 1:max_candidates]
    order = sortperm(1:max_candidates, by=ic -> (-counts[ic],
        selection_mode == :low_velocity ? (isfinite(medians[ic]) ? medians[ic] : Inf) :
            -(isfinite(confs[ic]) ? confs[ic] : 0.0),
        selection_mode == :low_velocity ? -(isfinite(confs[ic]) ? confs[ic] : 0.0) :
            (isfinite(medians[ic]) ? medians[ic] : Inf)))
    candidate_v = candidate_v[:, order]
    candidate_conf = candidate_conf[:, order]
    candidate_support = candidate_support[:, order]
    return ConsensusPick(ps, candidate_v[:, 1], candidate_conf[:, 1], candidate_support[:, 1],
        candidate_velocities, candidate_v, candidate_conf, candidate_support)
end

"""
    pick_cloud_consensus_from_rows(rows; periods=nothing,
        cluster_tolerance_fraction=0.10, min_candidate_periods=1,
        max_gap_periods=0)

Average velocity-tolerant raw pick clouds at each period and connect nearby
cloud means only when their relative velocity difference remains inside the
same tolerance. `max_gap_periods=0` keeps strict adjacent-period tracking;
larger values permit that many missing period bins between connected clouds.
Unlike `consensus_from_pick_rows`, this path has no confidence-first branch
seeding or candidate-count cap.
"""
function pick_cloud_consensus_from_rows(rows; periods=nothing,
        cluster_tolerance_fraction::Float64=0.10,
        min_candidate_periods::Int=1,
        max_gap_periods::Int=0)
    min_candidate_periods >= 1 || throw(ArgumentError("min_candidate_periods must be >= 1"))
    max_gap_periods >= 0 || throw(ArgumentError("max_gap_periods must be >= 0"))
    valid_rows = [r for r in rows if isfinite(Float64(_get(r, :period, NaN))) &&
        isfinite(Float64(_get(r, :group_velocity, NaN))) &&
        Float64(_get(r, :period, NaN)) > 0 && Float64(_get(r, :group_velocity, NaN)) > 0]
    ps = isnothing(periods) ? sort(unique(Float64(_get(r, :period, NaN)) for r in valid_rows)) :
        sort(Float64.(periods))
    isempty(ps) && return ConsensusPick(Float64[], Float64[], Float64[], Int[],
        Vector{Float64}[], fill(NaN, 0, 0), zeros(0, 0), zeros(Int, 0, 0))

    period_index = Dict(p => i for (i, p) in enumerate(ps))
    state_label(row) = String(_get(row, :state_label, _get(row, :branch, "pick")))
    nstates = max(length(unique(state_label(r) for r in valid_rows)), 1)
    grouped = [Vector{_Candidate}() for _ in ps]
    candidate_velocities = [Float64[] for _ in ps]

    for row in valid_rows
        period = Float64(_get(row, :period, NaN))
        haskey(period_index, period) || continue
        velocity = Float64(_get(row, :group_velocity, NaN))
        quality = Float64(_get(row, :quality, _get(row, :branch_correlation, 1.0)))
        quality = isfinite(quality) ? quality : 1.0
        ip = period_index[period]
        push!(grouped[ip], _Candidate(velocity, state_label(row), quality))
        push!(candidate_velocities[ip], velocity)
    end

    clusters = [_cluster_candidates(candidates, nstates; cluster_tolerance_fraction)
                for candidates in grouped]
    candidate_v_cols = Vector{Vector{Float64}}()
    candidate_conf_cols = Vector{Vector{Float64}}()
    candidate_support_cols = Vector{Vector{Int}}()
    last_period = Int[]
    last_velocity = Float64[]
    for ip in eachindex(ps)
        clouds = sort(clusters[ip], by=cloud -> cloud.velocity)
        active = [ic for ic in eachindex(candidate_v_cols)
            if 0 < ip - last_period[ic] <= max_gap_periods + 1]
        edges = Tuple{Float64,Int,Int}[]
        for track in active, (icloud, cloud) in enumerate(clouds)
            rel = _relative_difference(last_velocity[track], cloud.velocity)
            rel <= cluster_tolerance_fraction && push!(edges, (rel, track, icloud))
        end

        used_tracks = Set{Int}()
        used_clouds = Set{Int}()
        for (_, track, icloud) in sort(edges, by=first)
            (track in used_tracks || icloud in used_clouds) && continue
            cloud = clouds[icloud]
            candidate_v_cols[track][ip] = cloud.velocity
            candidate_conf_cols[track][ip] = cloud.confidence
            candidate_support_cols[track][ip] = cloud.support
            last_period[track] = ip
            last_velocity[track] = cloud.velocity
            push!(used_tracks, track)
            push!(used_clouds, icloud)
        end

        for (icloud, cloud) in enumerate(clouds)
            icloud in used_clouds && continue
            vcol = fill(NaN, length(ps))
            ccol = zeros(length(ps))
            scol = zeros(Int, length(ps))
            vcol[ip] = cloud.velocity
            ccol[ip] = cloud.confidence
            scol[ip] = cloud.support
            push!(candidate_v_cols, vcol)
            push!(candidate_conf_cols, ccol)
            push!(candidate_support_cols, scol)
            push!(last_period, ip)
            push!(last_velocity, cloud.velocity)
        end
    end

    keep = [count(v -> isfinite(v) && v > 0, col) >= min_candidate_periods
            for col in candidate_v_cols]
    candidate_v_cols = candidate_v_cols[keep]
    candidate_conf_cols = candidate_conf_cols[keep]
    candidate_support_cols = candidate_support_cols[keep]

    candidate_v = isempty(candidate_v_cols) ? fill(NaN, length(ps), 0) : reduce(hcat, candidate_v_cols)
    candidate_conf = isempty(candidate_conf_cols) ? zeros(length(ps), 0) : reduce(hcat, candidate_conf_cols)
    candidate_support = isempty(candidate_support_cols) ? zeros(Int, length(ps), 0) : reduce(hcat, candidate_support_cols)
    first_v = isempty(candidate_v_cols) ? fill(NaN, length(ps)) : candidate_v[:, 1]
    first_conf = isempty(candidate_conf_cols) ? zeros(length(ps)) : candidate_conf[:, 1]
    first_support = isempty(candidate_support_cols) ? zeros(Int, length(ps)) : candidate_support[:, 1]
    return ConsensusPick(ps, first_v, first_conf, first_support,
        candidate_velocities, candidate_v, candidate_conf, candidate_support)
end

function match_global_branch_rows(rows; velocity_tolerance_fraction::Float64=0.10)
    by_key = Dict{Tuple{String,Float64},Vector{Any}}()
    for row in rows
        branch = String(_get(row, :branch, ""))
        branch in ("causal", "acausal") || continue
        key = (String(_get(row, :pair_label, "")), Float64(_get(row, :period, NaN)))
        push!(get!(by_key, key, Any[]), row)
    end
    matched = NamedTuple[]
    for ((pair_label, period), group) in by_key
        causal = [r for r in group if String(_get(r, :branch, "")) == "causal"]
        acausal = [r for r in group if String(_get(r, :branch, "")) == "acausal"]
        used = falses(length(acausal))
        for c in causal
            vc = Float64(_get(c, :group_velocity, NaN))
            best = 0
            best_rel = Inf
            for (j, a) in enumerate(acausal)
                used[j] && continue
                va = Float64(_get(a, :group_velocity, NaN))
                rel = _relative_difference(vc, va)
                if isfinite(rel) && rel <= velocity_tolerance_fraction && rel < best_rel
                    best, best_rel = j, rel
                end
            end
            best == 0 && continue
            used[best] = true
            a = acausal[best]
            push!(matched, (; pair_label, period, state_label="global",
                group_velocity=0.5 * (vc + Float64(_get(a, :group_velocity, NaN))),
                causal_velocity=vc, acausal_velocity=Float64(_get(a, :group_velocity, NaN)),
                quality=_mean_finite([_get(c, :quality, 1.0), _get(a, :quality, 1.0)])))
        end
    end
    sort!(matched, by=r -> (r.pair_label, r.period, r.group_velocity))
    return matched
end

function _haversine_km(lat1, lon1, lat2, lon2)
    r = 6371.0
    φ1, φ2 = deg2rad(Float64(lat1)), deg2rad(Float64(lat2))
    dφ, dλ = deg2rad(Float64(lat2 - lat1)), deg2rad(Float64(lon2 - lon1))
    a = sin(dφ / 2)^2 + cos(φ1) * cos(φ2) * sin(dλ / 2)^2
    return 2r * asin(min(1.0, sqrt(a)))
end

function _azimuth_deg(lat1, lon1, lat2, lon2)
    φ1, φ2 = deg2rad(Float64(lat1)), deg2rad(Float64(lat2))
    dλ = deg2rad(Float64(lon2 - lon1))
    return mod(rad2deg(atan(sin(dλ) * cos(φ2),
        cos(φ1) * sin(φ2) - sin(φ1) * cos(φ2) * cos(dλ))) + 360, 360)
end

function receiver_pair_geometry(pair, latitudes, longitudes; distance=nothing)
    lat1, lat2 = Float64(latitudes[1]), Float64(latitudes[2])
    lon1, lon2 = Float64(longitudes[1]), Float64(longitudes[2])
    dist = isnothing(distance) ? _haversine_km(lat1, lon1, lat2, lon2) : Float64(distance)
    return ReceiverPairGeometry(String(pair[1]), String(pair[2]), lat1, lon1, lat2, lon2,
        0.5 * (lat1 + lat2), 0.5 * (lon1 + lon2), dist, _azimuth_deg(lat1, lon1, lat2, lon2))
end

function tomography_pair_consensus(pair, consensus::ConsensusPick; latitudes,
        longitudes, distance=nothing, label=nothing)
    lbl = isnothing(label) ? "$(pair[1])-$(pair[2])" : String(label)
    return PairConsensusForTomography(lbl, (String(pair[1]), String(pair[2])),
        receiver_pair_geometry(pair, latitudes, longitudes; distance), consensus)
end

function _selection_candidate_rows(pairs;
        candidate_mean_velocity_min::Float64=0.0,
        candidate_mean_velocity_max::Float64=Inf)
    rows_by_pair = Vector{Vector{Vector{NamedTuple}}}(undef, length(pairs))
    all_rows = NamedTuple[]
    for (pair_index, item) in enumerate(pairs)
        c = item.consensus
        candidates = Vector{Vector{NamedTuple}}()
        for icandidate in axes(c.candidate_group_velocities, 2)
            rows = NamedTuple[]
            for ip in eachindex(c.periods)
                v = c.candidate_group_velocities[ip, icandidate]
                isfinite(v) && v > 0 || continue
                row = (; pair_index, pair_label=item.label, candidate=icandidate,
                    period=c.periods[ip], group_velocity=Float64(v),
                    confidence=Float64(c.candidate_confidence[ip, icandidate]),
                    support=Int(c.candidate_support[ip, icandidate]))
                push!(rows, row)
            end
            isempty(rows) && continue
            mean_velocity = mean(row.group_velocity for row in rows)
            candidate_mean_velocity_min <= mean_velocity <= candidate_mean_velocity_max || continue
            push!(candidates, rows)
            append!(all_rows, rows)
        end
        rows_by_pair[pair_index] = candidates
    end
    return rows_by_pair, all_rows
end

function _period_stats(rows)
    grouped = Dict{Float64,Vector{Float64}}()
    for row in rows
        push!(get!(grouped, row.period, Float64[]), row.group_velocity)
    end
    stats = Dict{Float64,NamedTuple}()
    for (period, velocities) in grouped
        stats[period] = (; mean=mean(velocities),
            std=length(velocities) > 1 ? std(velocities) : 0.0,
            n=length(velocities))
    end
    return stats
end

function _row_outlier(row, stats; nsigma::Float64)
    st = get(stats, row.period, nothing)
    isnothing(st) && return false
    return st.n > 1 && st.std > 0.0 &&
        abs(row.group_velocity - st.mean) > nsigma * st.std
end

function _candidate_metric(rows, stats; nsigma::Float64)
    coverage = length(rows)
    n_outliers = count(row -> _row_outlier(row, stats; nsigma), rows)
    mean_confidence = _mean_finite(row.confidence for row in rows)
    mean_support = _mean_finite(row.support for row in rows)
    return (; n_outliers, outlier_fraction=coverage > 0 ? n_outliers / coverage : Inf,
        coverage, mean_confidence=isfinite(mean_confidence) ? mean_confidence : 0.0,
        mean_support=isfinite(mean_support) ? mean_support : 0.0)
end

function _candidate_order_key(rows, stats; nsigma::Float64)
    metric = _candidate_metric(rows, stats; nsigma)
    candidate = first(rows).candidate
    return (metric.n_outliers, metric.outlier_fraction, -metric.coverage,
        -metric.mean_confidence, -metric.mean_support, candidate)
end

"""
    select_outlier_candidates(pairs; nsigma=0.8, max_iterations=12,
        candidate_mean_velocity_min=0.0, candidate_mean_velocity_max=Inf)

Select one no-gap candidate per pair. The selector starts with period bands from
candidate rows whose mean group velocity is inside the candidate mean-velocity
bounds, picks the least-outlying candidate for each pair, rebuilds the period
bands from those selected rows, and iterates until choices stabilize.
"""
function select_outlier_candidates(pairs;
        nsigma::Float64=0.8, max_iterations::Int=12,
        candidate_mean_velocity_min::Float64=0.0,
        candidate_mean_velocity_max::Float64=Inf)
    nsigma >= 0.0 || throw(ArgumentError("nsigma must be >= 0"))
    max_iterations >= 1 || throw(ArgumentError("max_iterations must be >= 1"))
    candidate_mean_velocity_min >= 0.0 ||
        throw(ArgumentError("candidate_mean_velocity_min must be >= 0"))
    candidate_mean_velocity_max >= candidate_mean_velocity_min ||
        throw(ArgumentError("candidate_mean_velocity_max must be >= candidate_mean_velocity_min"))
    rows_by_pair, all_rows = _selection_candidate_rows(pairs;
        candidate_mean_velocity_min, candidate_mean_velocity_max)
    active_pairs = [i for i in eachindex(rows_by_pair) if !isempty(rows_by_pair[i])]
    if isempty(active_pairs)
        return OutlierCandidateSelection(String[], Int[], NamedTuple[], NamedTuple[],
            Dict{Float64,NamedTuple}(), 0, true)
    end

    stats = _period_stats(all_rows)
    selected_columns = zeros(Int, length(pairs))
    iterations = 0
    converged = false
    for iteration in 1:max_iterations
        iterations = iteration
        next_columns = copy(selected_columns)
        for pair_index in active_pairs
            candidates = rows_by_pair[pair_index]
            selected_local = argmin([_candidate_order_key(rows, stats; nsigma)
                                     for rows in candidates])
            next_columns[pair_index] = selected_local
        end
        selected_plain = vcat([rows_by_pair[i][next_columns[i]] for i in active_pairs]...)
        stats = _period_stats(selected_plain)
        if next_columns == selected_columns
            converged = true
            selected_columns = next_columns
            break
        end
        selected_columns = next_columns
    end

    selected_rows = NamedTuple[]
    diagnostics = NamedTuple[]
    labels = String[]
    candidate_indices = Int[]
    for pair_index in active_pairs
        rows = rows_by_pair[pair_index][selected_columns[pair_index]]
        metric = _candidate_metric(rows, stats; nsigma)
        pair_rows = NamedTuple[]
        for row in rows
            st = stats[row.period]
            status = _row_outlier(row, stats; nsigma) ? "outlier" : "kept"
            enriched = merge(row, (; status, band_mean=st.mean, band_std=st.std, band_n=st.n))
            push!(pair_rows, enriched)
            push!(selected_rows, enriched)
        end
        pair_label = pairs[pair_index].label
        candidate = first(rows).candidate
        mean_velocity = mean(row.group_velocity for row in rows)
        push!(labels, pair_label)
        push!(candidate_indices, candidate)
        push!(diagnostics, (; pair_index, pair_label, selected_candidate=candidate,
            metric.coverage, mean_velocity, metric.n_outliers, metric.outlier_fraction,
            metric.mean_confidence, metric.mean_support,
            status=metric.n_outliers == 0 ? "clean" : "flagged",
            kept_rows=count(row -> row.status == "kept", pair_rows)))
    end
    sort!(selected_rows, by=row -> (row.period, row.pair_label, row.candidate))
    sort!(diagnostics, by=row -> row.pair_label)
    return OutlierCandidateSelection(labels, candidate_indices, selected_rows,
        diagnostics, stats, iterations, converged)
end

function selected_dsurftomo_rows(selection::OutlierCandidateSelection,
        pairs::Vector{PairConsensusForTomography}; kept_only::Bool=true)
    rows = NamedTuple[]
    for row in selection.selected_rows
        kept_only && row.status != "kept" && continue
        idx = findfirst(p -> p.label == row.pair_label, pairs)
        idx === nothing && continue
        g = pairs[idx].geometry
        push!(rows, (; row.period, lat1=g.lat1, lon1=g.lon1, lat2=g.lat2, lon2=g.lon2,
            row.group_velocity, row.pair_label, row.candidate, row.status))
    end
    return sort(rows, by=r -> (r.period, r.pair_label))
end

function write_dsurftomo_rows(path::AbstractString, rows; include_count_header::Bool=false)
    mkpath(dirname(path))
    open(path, "w") do io
        include_count_header && println(io, length(rows))
        for r in rows
            @printf(io, "%.8g %.8f %.8f %.8f %.8f %.8f\n",
                r.period, r.lat1, r.lon1, r.lat2, r.lon2, r.group_velocity)
        end
    end
    return path
end

end
