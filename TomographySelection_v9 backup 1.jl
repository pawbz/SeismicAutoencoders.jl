module TomographySelection_v9

using Statistics
using Printf

export ConsensusPick, ReceiverPairGeometry, PairConsensusForTomography,
       TomographyCandidateMix, consensus_from_pick_rows, match_global_branch_rows,
       receiver_pair_geometry, tomography_pair_consensus, tomography_candidate_mixes,
       tomography_mix_rows, dsurftomo_rows, write_dsurftomo_rows,
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

struct TomographyCandidateMix
    pair_index::Int
    label::String
    candidate_indices::Vector{Int}
    group_velocities::Vector{Float64}
    confidence::Vector{Float64}
    support::Vector{Int}
    coverage_count::Int
    mean_confidence::Float64
    neighbor_agreement::Float64
    total_score::Float64
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

function _axial_angle_difference(a, b)
    d = abs(mod(Float64(a - b) + 180, 360) - 180)
    return min(d, 180 - d)
end

function _similar_paths(a::ReceiverPairGeometry, b::ReceiverPairGeometry;
        midpoint_radius_km=75.0, azimuth_tolerance_deg=25.0,
        distance_tolerance_fraction=0.35)
    mid = _haversine_km(a.midpoint_lat, a.midpoint_lon, b.midpoint_lat, b.midpoint_lon)
    dist_rel = abs(a.distance_km - b.distance_km) / max((a.distance_km + b.distance_km) / 2, eps(Float64))
    return mid <= midpoint_radius_km && _axial_angle_difference(a.azimuth_deg, b.azimuth_deg) <= azimuth_tolerance_deg &&
        dist_rel <= distance_tolerance_fraction
end

function _candidate_sets(consensus::ConsensusPick; max_mix_parts=3, min_candidate_periods=3)
    masks = [isfinite.(consensus.candidate_group_velocities[:, i]) .&
        (consensus.candidate_group_velocities[:, i] .> 0) for i in axes(consensus.candidate_group_velocities, 2)]
    valid = [i for i in eachindex(masks) if count(masks[i]) >= min_candidate_periods]
    mixes = Vector{Vector{Int}}()
    function extend!(current, start, used)
        !isempty(current) && push!(mixes, copy(current))
        length(current) >= max_mix_parts && return
        for pos in start:length(valid)
            i = valid[pos]
            any(used .& masks[i]) && continue
            push!(current, i)
            extend!(current, pos + 1, used .| masks[i])
            pop!(current)
        end
    end
    extend!(Int[], 1, falses(length(consensus.periods)))
    return mixes
end

function _mix_curve(consensus, indices)
    v = fill(NaN, length(consensus.periods))
    c = zeros(length(v))
    s = zeros(Int, length(v))
    for idx in indices
        mask = isfinite.(consensus.candidate_group_velocities[:, idx]) .&
            (consensus.candidate_group_velocities[:, idx] .> 0)
        v[mask] .= consensus.candidate_group_velocities[mask, idx]
        c[mask] .= consensus.candidate_confidence[mask, idx]
        s[mask] .= consensus.candidate_support[mask, idx]
    end
    return v, c, s
end

function tomography_candidate_mixes(pairs::Vector{PairConsensusForTomography};
        max_mix_parts::Int=3, min_candidate_periods::Int=3,
        midpoint_radius_km::Float64=75.0, azimuth_tolerance_deg::Float64=25.0,
        distance_tolerance_fraction::Float64=0.35, velocity_tolerance_fraction::Float64=0.10)
    mixes = TomographyCandidateMix[]
    for (pair_index, item) in enumerate(pairs)
        for indices in _candidate_sets(item.consensus; max_mix_parts, min_candidate_periods)
            v, c, s = _mix_curve(item.consensus, indices)
            coverage = count(x -> isfinite(x) && x > 0, v)
            coverage >= min_candidate_periods || continue
            push!(mixes, TomographyCandidateMix(pair_index,
                "$(item.label) | candidates $(join(indices, "+"))", indices, v, c, s,
                coverage, isfinite(_mean_finite(c[c .> 0])) ? _mean_finite(c[c .> 0]) : 0.0, 0.0, 0.0))
        end
    end
    isempty(mixes) && return mixes
    max_coverage = maximum(m.coverage_count for m in mixes)
    max_support = maximum(maximum(m.support; init=0) for m in mixes)
    scored = TomographyCandidateMix[]
    for mix in mixes
        total, agree = 0, 0
        for ip in eachindex(mix.group_velocities)
            v = mix.group_velocities[ip]
            isfinite(v) && v > 0 || continue
            total += 1
            if any(other.pair_index != mix.pair_index &&
                    _similar_paths(pairs[mix.pair_index].geometry, pairs[other.pair_index].geometry;
                        midpoint_radius_km, azimuth_tolerance_deg, distance_tolerance_fraction) &&
                    isfinite(other.group_velocities[ip]) && other.group_velocities[ip] > 0 &&
                    _relative_difference(v, other.group_velocities[ip]) <= velocity_tolerance_fraction
                    for other in mixes)
                agree += 1
            end
        end
        neighbor = total == 0 ? 0.0 : agree / total
        support_score = _mean_finite([s / max(max_support, 1) for s in mix.support if s > 0])
        support_score = isfinite(support_score) ? support_score : 0.0
        score = mix.coverage_count / max_coverage + 0.5 * mix.mean_confidence +
            0.25 * support_score + neighbor
        push!(scored, TomographyCandidateMix(mix.pair_index, mix.label, mix.candidate_indices,
            mix.group_velocities, mix.confidence, mix.support, mix.coverage_count,
            mix.mean_confidence, neighbor, score))
    end
    return sort(scored, by=m -> -m.total_score)
end

function tomography_mix_rows(mixes, pairs; use_rank::Int=1)
    rows = NamedTuple[]
    for (rank, mix) in enumerate(mixes)
        rank == use_rank || continue
        item = pairs[mix.pair_index]
        for ip in eachindex(item.consensus.periods)
            v = mix.group_velocities[ip]
            isfinite(v) && v > 0 || continue
            push!(rows, (; pair_label=item.label, period=item.consensus.periods[ip],
                group_velocity=v, confidence=mix.confidence[ip], support=mix.support[ip],
                mix_rank=rank, mix_score=mix.total_score))
        end
    end
    return rows
end

function dsurftomo_rows(mixes, pairs; use_rank::Int=1)
    rows = NamedTuple[]
    for row in tomography_mix_rows(mixes, pairs; use_rank)
        idx = findfirst(p -> p.label == row.pair_label, pairs)
        idx === nothing && continue
        g = pairs[idx].geometry
        push!(rows, (; row.period, lat1=g.lat1, lon1=g.lon1, lat2=g.lat2, lon2=g.lon2,
            row.group_velocity, row.pair_label, row.mix_score))
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
