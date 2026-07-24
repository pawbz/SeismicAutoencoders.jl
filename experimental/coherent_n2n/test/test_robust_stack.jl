# Unit tests for the robust coherent stack (robust_complex_stack) and the
# selectable denoiser loss (denoiser_loss_fn / complex_l1_loss), which replaced
# the removed energy-outlier hard mask.
# Run with: julia --project=experimental/coherent_n2n experimental/coherent_n2n/test/test_robust_stack.jl

using Test
using Random
using FFTW
using Statistics: mean, median, std

include(joinpath(@__DIR__, "..", "CoherentN2N_denoiser.jl"))       # for ComplexDenoiser types (loss operates on matrices)
include(joinpath(@__DIR__, "..", "CoherentN2N_train_denoiser.jl")) # complex_mse_loss / complex_l1_loss / denoiser_loss_fn
include(joinpath(@__DIR__, "..", "CoherentN2N_outer_loop.jl"))     # complex_coherent_stack / robust_complex_stack

@testset "robust_complex_stack :l2 == mean, all-ones weights" begin
    rng = MersenneTwister(0)
    X̂ = ComplexF32.(randn(rng, 16, 7) .+ im .* randn(rng, 16, 7))
    ŝ, w = robust_complex_stack(X̂; stack_type=:l2)
    @test w == ones(Float32, 7)
    @test ŝ ≈ complex_coherent_stack(X̂)              # byte-for-byte the mean stack
    @test ŝ ≈ vec(mean(X̂, dims=2))
end

@testset "robust_complex_stack :l1 down-weights a corrupt trace" begin
    rng = MersenneTwister(1)
    nt, R = 32, 20
    # A coherent gather: every column ≈ the same spectrum + small noise …
    base = ComplexF32.(randn(rng, nt) .+ im .* randn(rng, nt))
    X̂ = ComplexF32.(repeat(base, 1, R) .+ 0.05f0 .* (randn(rng, nt, R) .+ im .* randn(rng, nt, R)))
    # … except ONE grossly corrupt column (an incoherent outlier).
    bad = 7
    X̂[:, bad] .= ComplexF32.(20f0 .* (randn(rng, nt) .+ im .* randn(rng, nt)))

    ŝ_l2, w_l2 = robust_complex_stack(X̂; stack_type=:l2)
    ŝ_l1, w_l1 = robust_complex_stack(X̂; stack_type=:l1)

    # The corrupt trace gets the smallest L1 weight, and a small one absolutely.
    @test argmin(w_l1) == bad
    @test w_l1[bad] < 0.5f0
    # Good traces are weighted UNIFORMLY (they share the same inlier residual
    # scale) and MUCH more than the outlier. Note Huber weights are not ~1 in
    # absolute terms: δ = MAD of residuals tracks the inlier scale, so once ŝ ≈
    # the clean source the inliers' residuals (the 0.05 noise) are all similar
    # and get similar sub-1 weights; the discriminator is that the outlier is
    # driven far below them.
    good_w = w_l1[setdiff(1:R, bad)]
    @test minimum(good_w) > 5f0 * w_l1[bad]                 # outlier clearly separated
    @test std(good_w) / mean(good_w) < 0.5f0               # inliers weighted ~uniformly
    # L1 stack is closer to the clean source than the (outlier-dragged) L2 mean.
    @test sum(abs2, ŝ_l1 .- base) < sum(abs2, ŝ_l2 .- base)
end

@testset "robust_complex_stack :l1 degrades to mean when no spread" begin
    # All identical columns → zero residual spread → weights all 1, ŝ == mean.
    col = ComplexF32.(randn(MersenneTwister(2), 8) .+ im .* randn(MersenneTwister(3), 8))
    X̂ = repeat(col, 1, 5)
    ŝ, w = robust_complex_stack(X̂; stack_type=:l1)
    @test w == ones(Float32, 5)
    @test ŝ ≈ col
end

@testset "robust_complex_stack unknown stack_type errors" begin
    X̂ = ComplexF32.(randn(4, 3) .+ im .* randn(4, 3))
    @test_throws ArgumentError robust_complex_stack(X̂; stack_type=:huber)
end

@testset "denoiser loss selection" begin
    rng = MersenneTwister(4)
    pred = ComplexF32.(randn(rng, 16, 5) .+ im .* randn(rng, 16, 5))
    tgt  = ComplexF32.(randn(rng, 16, 5) .+ im .* randn(rng, 16, 5))
    @test denoiser_loss_fn(:l2) === complex_mse_loss
    @test denoiser_loss_fn(:l1) === complex_l1_loss
    @test_throws ArgumentError denoiser_loss_fn(:huber)
    # Both losses are finite, positive scalars.
    @test complex_mse_loss(pred, tgt) > 0
    @test complex_l1_loss(pred, tgt) > 0
    # L1 loss is more robust: adding one gross-outlier residual sample inflates
    # MSE far more than L1 (relative to the un-corrupted baseline).
    pred2 = copy(pred); pred2[1, 1] += 100f0
    r_mse = complex_mse_loss(pred2, tgt) / complex_mse_loss(pred, tgt)
    r_l1  = complex_l1_loss(pred2, tgt) / complex_l1_loss(pred, tgt)
    @test r_mse > r_l1
end

println("All robust-stack / denoiser-loss unit tests passed.")
