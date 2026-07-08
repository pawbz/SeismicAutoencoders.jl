# Standalone correctness tests for zero-sum gauge fixing and per-earthquake
# complex gain (polarity/amplitude) estimation.
# Run with: julia --project=. experimental/coherent_n2n/test/test_gauge_and_gain.jl

using Test
using Random
using FFTW
using Statistics: mean

include(joinpath(@__DIR__, "..", "CoherentN2N_gauge.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_polarity.jl"))
include(joinpath(@__DIR__, "synthetic_data.jl"))

@testset "enforce_zero_sum_gauge!: recovers known constant bias as anchor" begin
    rng = MersenneTwister(10)
    τ_true = Float32.(randn(rng, 20) .* 5)   # zero-mean-ish true shifts
    bias = 12.3f0
    τ_biased = τ_true .+ bias

    anchor = enforce_zero_sum_gauge!(τ_biased)

    @test isapprox(anchor, bias + mean(τ_true); atol=1e-4)
    @test isapprox(mean(τ_biased), 0f0; atol=1e-4)
    # Reapplying the anchor should restore the original (biased) shifts
    @test isapprox(τ_biased .+ anchor, τ_true .+ bias; atol=1e-4)
end

@testset "enforce_zero_sum_gauge!: idempotent on already-zero-mean input" begin
    τ = Float32[1.0, -1.0, 2.0, -2.0]
    anchor = enforce_zero_sum_gauge!(τ)
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

println("All CoherentN2N gauge/gain tests passed.")
