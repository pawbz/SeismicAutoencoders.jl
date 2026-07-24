# Standalone correctness tests for location gauge fixing and per-earthquake
# complex gain (polarity/amplitude) estimation.
# Run with: julia --project=. experimental/coherent_n2n/test/test_gauge_and_gain.jl

using Test
using Random
using FFTW
using Statistics: mean, median

include(joinpath(@__DIR__, "..", "CoherentN2N_gauge.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_polarity.jl"))
include(joinpath(@__DIR__, "synthetic_data.jl"))

@testset "mode_kde: peaks on the dominant cluster, ignores a skewed tail" begin
    rng = MersenneTwister(10)
    # 80% tight cluster near +3, 20% straggling far to the right. Mean/median
    # both sit right of the cluster; the mode should land on ≈ +3.
    cluster = Float32(3.0) .+ Float32.(randn(rng, 80) .* 0.2)
    tail    = Float32.(20.0 .+ randn(rng, 20) .* 3.0)
    τ = vcat(cluster, tail)
    m = mode_kde(τ)
    @test isapprox(m, 3.0f0; atol=0.6)          # on the cluster, not dragged out
    @test m < mean(τ)                            # strictly left of the tail-dragged mean
    @test m < median(τ)                          # and left of the median too, given the skew
end

@testset "mode_kde: degenerate inputs fall back to mean" begin
    @test mode_kde(Float32[]) == 0f0
    @test mode_kde(Float32[7.5]) == 7.5f0
    @test isapprox(mode_kde(fill(2.0f0, 5)), 2.0f0; atol=1e-4)   # zero spread
end

@testset "enforce_location_gauge! (mode): anchors cluster to zero, tail may be skewed" begin
    rng = MersenneTwister(11)
    cluster = Float32(3.0) .+ Float32.(randn(rng, 80) .* 0.2)
    tail    = Float32.(20.0 .+ randn(rng, 20) .* 3.0)
    τ = vcat(cluster, tail)
    τ0 = copy(τ)

    anchor = enforce_location_gauge!(τ)          # default center=mode_kde

    # The dominant cluster is now centered on ~0; anchor ≈ the pre-gauge mode.
    @test isapprox(anchor, mode_kde(τ0); atol=1e-4)
    @test isapprox(mode_kde(τ), 0f0; atol=0.6)
    # Reapplying the anchor restores the original shifts exactly.
    @test isapprox(τ .+ anchor, τ0; atol=1e-4)
    # Skew is allowed: counts either side of zero need NOT be balanced.
    @test count(>(0f0), τ) != count(<(0f0), τ)
end

@testset "enforce_location_gauge! (median): symmetric-count gauge still available" begin
    rng = MersenneTwister(12)
    τ = Float32.(randn(rng, 20) .* 5) .+ 12.3f0
    anchor = enforce_location_gauge!(τ; center=median)
    @test isapprox(median(τ), 0f0; atol=1e-4)
    @test count(>(0f0), τ) == count(<(0f0), τ)   # median → equal counts either side
end

@testset "enforce_location_gauge! (mean): classic zero-sum gauge still available" begin
    τ = Float32[1.0, -1.0, 2.0, -2.0]
    anchor = enforce_location_gauge!(τ; center=mean)
    @test isapprox(anchor, 0f0; atol=1e-6)
    @test isapprox(mean(τ), 0f0; atol=1e-6)
end

@testset "estimate_earthquake_gain: recovers known real gain (polarity + amplitude)" begin
    rng = MersenneTwister(11)
    nt = 300
    s = broadband_wavelet(nt, (0.02, 0.04, 0.08))
    Ŝ = fft(s)

    for g_true in ComplexF32[1.0, -1.0, 2.5, -0.7, 3.2]
        d = g_true .* Ŝ
        g_est = estimate_earthquake_gain(d, Ŝ)
        @test isapprox(g_est, g_true; atol=1e-4)
    end
end

@testset "estimate_earthquake_gain: robust to additive noise" begin
    rng = MersenneTwister(12)
    nt = 300
    s = broadband_wavelet(nt, (0.02, 0.04, 0.08))
    Ŝ = fft(s)
    g_true = -1.8f0 + 0.0f0im

    d = g_true .* Ŝ .+ fft(0.01f0 .* randn(rng, Float32, nt))
    g_est = estimate_earthquake_gain(d, Ŝ)
    @test abs(g_est - g_true) < 0.1
end

println("All CoherentN2N location-gauge/gain tests passed.")
