# Event geometry (epicentral distance / backazimuth) and 2-D binning of a
# station's earthquake set.
#
# Motivation: the alternating loop's per-event shift τ exists because a station's
# full earthquake set spans the whole distance/azimuth range, so the receiver
# functions genuinely arrive at different times. Restricting a station to a
# single small distance×backazimuth cell removes that spread by construction —
# events in one cell share essentially the same ray parameter and incidence
# geometry — which lets the alignment-free loops
# (`run_coherent_n2n_grouped_noalign`) train the denoiser with τ ≡ 0.
#
# Conventions: `sta_loc` is `[lat, lon]` (as `Sta[2][i]` in the RF JLD2 files),
# and `evt_locs` is a vector of `[lat, lon, ...]` 1:1 with the columns of that
# station's gather (as `EventLoc[i][sel]` is). Cells are indexed by the integer
# pair `(i_dist, i_baz)` = the floor-divided lower-left corner, so cell
# `(6, 14)` at `bin_size=10` means distance ∈ [60,70)°, backazimuth ∈ [140,150)°.

"""
    epicentral_distance_deg(sta_lat, sta_lon, evt_lat, evt_lon) -> Float64

Great-circle epicentral distance in degrees (spherical Earth), via the haversine
formula. Range 0–180; no wrap-around handling is needed.
"""
function epicentral_distance_deg(sta_lat, sta_lon, evt_lat, evt_lon)
    φ1, φ2 = deg2rad(sta_lat), deg2rad(evt_lat)
    dφ = deg2rad(evt_lat - sta_lat)
    dλ = deg2rad(evt_lon - sta_lon)
    h = sin(dφ / 2)^2 + cos(φ1) * cos(φ2) * sin(dλ / 2)^2
    rad2deg(2 * asin(sqrt(h)))
end

"""
    backazimuth_deg(sta_lat, sta_lon, evt_lat, evt_lon) -> Float64

Backazimuth in degrees (0–360, measured from North, clockwise): the azimuth of
the great-circle direction from the EVENT to the STATION, i.e. the direction the
wavefront arrives from as seen at the station.
"""
function backazimuth_deg(sta_lat, sta_lon, evt_lat, evt_lon)
    φ1, φ2 = deg2rad(evt_lat), deg2rad(sta_lat)
    dλ = deg2rad(sta_lon - evt_lon)
    y = sin(dλ) * cos(φ2)
    x = cos(φ1) * sin(φ2) - sin(φ1) * cos(φ2) * cos(dλ)
    mod(rad2deg(atan(y, x)) + 360.0, 360.0)
end

"""
    bin_index(dist, baz, bin_size) -> (i_dist::Int, i_baz::Int)

The `bin_size`° × `bin_size`° cell containing `(dist, baz)`, identified by its
floor-divided lower-left corner. Backazimuth is wrapped into [0, 360) BEFORE
flooring, so a value of exactly 360 (or a small negative) lands in the same cell
as 0 rather than creating a spurious extra cell at the seam.
"""
bin_index(dist, baz, bin_size) =
    (floor(Int, dist / bin_size), floor(Int, mod(baz, 360.0) / bin_size))

"""
    bin_ranges(cell, bin_size) -> (; dist_range, baz_range)

The half-open coordinate extent `[lo, hi)` of `cell` — the inverse of
`bin_index`, for labelling and for drawing the cell on a polar plot.
"""
function bin_ranges(cell::Tuple{Int,Int}, bin_size)
    d0 = cell[1] * bin_size
    b0 = cell[2] * bin_size
    return (; dist_range=(d0, d0 + bin_size), baz_range=(b0, b0 + bin_size))
end

"""
    event_bins(sta_loc, evt_locs; bin_size=10.0) -> (; dist, baz, cell)

Per-event epicentral distance, backazimuth, and `(i_dist, i_baz)` cell index for
one station's earthquake set. All three returned vectors are 1:1 with `evt_locs`
(hence with the columns of that station's gather).
"""
function event_bins(sta_loc, evt_locs; bin_size=10.0)
    sta_lat, sta_lon = sta_loc[1], sta_loc[2]
    dist = [epicentral_distance_deg(sta_lat, sta_lon, ev[1], ev[2]) for ev in evt_locs]
    baz = [backazimuth_deg(sta_lat, sta_lon, ev[1], ev[2]) for ev in evt_locs]
    cell = [bin_index(dist[r], baz[r], bin_size) for r in eachindex(dist)]
    return (; dist, baz, cell)
end

"""
    bin_member_indices(sta_loc, evt_locs, cell; bin_size=10.0) -> Vector{Int}

Column indices of the events falling inside an explicitly given `cell` — the
manual-override counterpart of `densest_bin` (which picks the cell for you).
Returns an empty vector if the cell holds no events.
"""
function bin_member_indices(sta_loc, evt_locs, cell::Tuple{Int,Int}; bin_size=10.0)
    b = event_bins(sta_loc, evt_locs; bin_size=bin_size)
    return findall(==(cell), b.cell)
end

"""
    densest_bin(sta_loc, evt_locs; bin_size=10.0, min_events=2)
        -> (; cell, idx, count, dist_range, baz_range) | nothing

The `bin_size`° × `bin_size`° distance/backazimuth cell holding the MOST events
for this station, together with `idx` — the column indices of the events inside
it, ready to slice the station's gather with (`D[:, idx]`).

Returns `nothing` when the station's best cell holds fewer than `min_events`
events (too sparse to be worth training on); callers are expected to drop such
stations from the group list.

Ties between equally-populated cells are broken deterministically by the sorted
cell order, so repeated calls on the same data always return the same bin.
"""
function densest_bin(sta_loc, evt_locs; bin_size=10.0, min_events::Int=2)
    isempty(evt_locs) && return nothing
    b = event_bins(sta_loc, evt_locs; bin_size=bin_size)

    counts = Dict{Tuple{Int,Int},Int}()
    for c in b.cell
        counts[c] = get(counts, c, 0) + 1
    end
    isempty(counts) && return nothing

    # Deterministic tie-break: scan cells in sorted order, keep the first maximum.
    best_cell = first(sort!(collect(keys(counts))))
    best_n = counts[best_cell]
    for c in sort!(collect(keys(counts)))
        if counts[c] > best_n
            best_cell, best_n = c, counts[c]
        end
    end
    best_n < min_events && return nothing

    idx = findall(==(best_cell), b.cell)
    r = bin_ranges(best_cell, bin_size)
    return (; cell=best_cell, idx, count=best_n, r.dist_range, r.baz_range)
end
