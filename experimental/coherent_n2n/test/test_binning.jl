# Event geometry + distance/backazimuth binning (CoherentN2N_binning.jl).
# Uses analytically-known geometries (equator and meridian great circles, where
# distance and backazimuth are exact) rather than reference-table values, so the
# assertions are self-evident.
# Run with: julia --project=experimental/coherent_n2n experimental/coherent_n2n/test/test_binning.jl

using Test

include(joinpath(@__DIR__, "..", "CoherentN2N_binning.jl"))

@testset "epicentral_distance_deg: exact great-circle cases" begin
    # Along the equator, distance in degrees == longitude difference.
    @test epicentral_distance_deg(0.0, 0.0, 0.0, 0.0) ≈ 0.0 atol = 1e-9
    @test epicentral_distance_deg(0.0, 0.0, 0.0, 30.0) ≈ 30.0 atol = 1e-9
    @test epicentral_distance_deg(0.0, 0.0, 0.0, 90.0) ≈ 90.0 atol = 1e-9
    @test epicentral_distance_deg(0.0, 10.0, 0.0, -20.0) ≈ 30.0 atol = 1e-9

    # Along a meridian, distance == latitude difference.
    @test epicentral_distance_deg(0.0, 0.0, 45.0, 0.0) ≈ 45.0 atol = 1e-9
    @test epicentral_distance_deg(-10.0, 5.0, 25.0, 5.0) ≈ 35.0 atol = 1e-9

    # Antipodal and symmetric.
    @test epicentral_distance_deg(0.0, 0.0, 0.0, 180.0) ≈ 180.0 atol = 1e-6
    @test epicentral_distance_deg(12.0, 34.0, -56.0, 78.0) ≈
          epicentral_distance_deg(-56.0, 78.0, 12.0, 34.0) atol = 1e-9
end

@testset "backazimuth_deg: compass conventions" begin
    # Backazimuth here is the azimuth of the great-circle direction FROM THE
    # EVENT TO THE STATION (the standard seismological definition) — i.e. the
    # reverse of the station->event forward azimuth. So an event lying due EAST
    # of the station is seen, from that event, with the station due WEST: 270.
    @test backazimuth_deg(0.0, 0.0, 0.0, 10.0) ≈ 270.0 atol = 1e-6
    # Event due west of the station -> station lies east of it -> 90.
    @test backazimuth_deg(0.0, 0.0, 0.0, -10.0) ≈ 90.0 atol = 1e-6
    # Event due north of the station -> station lies south of it -> 180.
    @test backazimuth_deg(0.0, 0.0, 10.0, 0.0) ≈ 180.0 atol = 1e-6
    # Event due south of the station -> station lies north of it -> 0.
    @test backazimuth_deg(10.0, 0.0, 0.0, 0.0) ≈ 0.0 atol = 1e-6

    # Consistency: baz(sta, evt) is the forward azimuth from evt to sta, so it
    # equals the "forward azimuth" formula with the two points swapped.
    fwd(alat, alon, blat, blon) = backazimuth_deg(blat, blon, alat, alon)
    for (slat, slon, elat, elon) in ((20.0, 30.0, -15.0, 88.0), (37.0, -122.0, 60.0, 179.0))
        @test backazimuth_deg(slat, slon, elat, elon) ≈ fwd(elat, elon, slat, slon) atol = 1e-9
    end
    # Always in [0, 360).
    for (elat, elon) in ((37.0, -122.0), (-15.0, 88.0), (60.0, 179.0), (-60.0, -179.0))
        b = backazimuth_deg(20.0, 30.0, elat, elon)
        @test 0.0 <= b < 360.0
    end
end

@testset "bin_index / bin_ranges: floor cells and wrap" begin
    @test bin_index(0.0, 0.0, 10.0) == (0, 0)
    @test bin_index(62.0, 147.0, 10.0) == (6, 14)
    # Half-open [lo, hi): the upper edge belongs to the NEXT cell.
    @test bin_index(70.0, 150.0, 10.0) == (7, 15)
    @test bin_index(69.999, 149.999, 10.0) == (6, 14)
    # Backazimuth wraps: 360 and 0 are the same cell; small negatives fold up.
    @test bin_index(30.0, 360.0, 10.0) == (3, 0)
    @test bin_index(30.0, 0.0, 10.0) == (3, 0)
    @test bin_index(30.0, 359.9, 10.0) == (3, 35)
    @test bin_index(30.0, -0.1, 10.0) == (3, 35)
    # Non-10 bin sizes.
    @test bin_index(62.0, 147.0, 5.0) == (12, 29)
    @test bin_index(62.0, 147.0, 30.0) == (2, 4)

    # bin_ranges inverts bin_index.
    r = bin_ranges((6, 14), 10.0)
    @test r.dist_range == (60.0, 70.0)
    @test r.baz_range == (140.0, 150.0)
    for (d, b, bs) in ((62.0, 147.0, 10.0), (3.0, 359.0, 5.0), (170.0, 12.0, 30.0))
        c = bin_index(d, b, bs)
        rr = bin_ranges(c, bs)
        @test rr.dist_range[1] <= d < rr.dist_range[2]
        @test rr.baz_range[1] <= mod(b, 360.0) < rr.baz_range[2]
    end
end

@testset "event_bins: 1:1 with the event list" begin
    # Station on the equator at lon 0; events placed due east along the equator,
    # so distance == lon and backazimuth == 270 (station lies west of each event).
    sta = [0.0, 0.0]
    evts = [[0.0, d] for d in (5.0, 15.0, 25.0, 35.0)]
    b = event_bins(sta, evts; bin_size=10.0)
    @test length(b.dist) == length(b.baz) == length(b.cell) == 4
    @test b.dist ≈ [5.0, 15.0, 25.0, 35.0] atol = 1e-6
    @test all(x -> isapprox(x, 270.0; atol=1e-6), b.baz)
    @test b.cell == [(0, 27), (1, 27), (2, 27), (3, 27)]

    # Extra trailing fields in an event entry (depth/mag) are ignored — real
    # EventLoc entries carry more than [lat, lon].
    evts_long = [[0.0, 15.0, 33.0, 6.1]]
    @test event_bins(sta, evts_long; bin_size=10.0).cell == [(1, 27)]
end

@testset "densest_bin: picks the most populated cell" begin
    sta = [0.0, 0.0]
    # 5 events in the [20,30)° distance band due east (cell (2,9)), 2 events in
    # the [50,60)° band (cell (5,9)), 1 event far away to the north (cell (4,0)).
    evts = vcat(
        [[0.0, d] for d in (21.0, 23.0, 25.0, 27.0, 29.0)],
        [[0.0, d] for d in (51.0, 57.0)],
        [[45.0, 0.0]],
    )
    best = densest_bin(sta, evts; bin_size=10.0, min_events=2)
    @test best !== nothing
    @test best.cell == (2, 27)          # dist [20,30), baz [270,280)
    @test best.count == 5
    @test best.idx == [1, 2, 3, 4, 5]
    @test best.dist_range == (20.0, 30.0)
    @test best.baz_range == (270.0, 280.0)

    # min_events gates the result: nothing is dense enough at 6.
    @test densest_bin(sta, evts; bin_size=10.0, min_events=6) === nothing
    # ...and exactly at the count it passes.
    @test densest_bin(sta, evts; bin_size=10.0, min_events=5) !== nothing

    # Empty event list.
    @test densest_bin(sta, Vector{Vector{Float64}}(); bin_size=10.0) === nothing

    # A larger bin merges the two east bands into one cell of 7.
    merged = densest_bin(sta, evts; bin_size=60.0, min_events=2)
    @test merged.count == 7

    # Deterministic under ties: two cells of 2 each, repeated calls agree.
    tie_evts = [[0.0, 5.0], [0.0, 7.0], [0.0, 25.0], [0.0, 27.0]]
    a = densest_bin(sta, tie_evts; bin_size=10.0, min_events=2)
    b = densest_bin(sta, tie_evts; bin_size=10.0, min_events=2)
    @test a.cell == b.cell
    @test a.count == 2
end

@testset "bin_member_indices: agrees with densest_bin, honours overrides" begin
    sta = [0.0, 0.0]
    evts = vcat([[0.0, d] for d in (21.0, 23.0, 25.0)], [[0.0, d] for d in (51.0, 57.0)])

    best = densest_bin(sta, evts; bin_size=10.0, min_events=2)
    @test bin_member_indices(sta, evts, best.cell; bin_size=10.0) == best.idx

    # An explicitly chosen (non-densest) cell: dist [50,60), baz [270,280).
    @test bin_member_indices(sta, evts, (5, 27); bin_size=10.0) == [4, 5]
    # An empty cell yields an empty index vector, not an error.
    @test isempty(bin_member_indices(sta, evts, (17, 30); bin_size=10.0))

    # Every event lands in exactly one cell: the per-cell index sets partition 1:R.
    b = event_bins(sta, evts; bin_size=10.0)
    covered = Int[]
    for c in unique(b.cell)
        append!(covered, bin_member_indices(sta, evts, c; bin_size=10.0))
    end
    @test sort(covered) == collect(1:length(evts))
end

println("All CoherentN2N binning tests passed.")
