# Validates the optional polarity/gain-correction path (use_polarity_gain=true)
# against a synthetic gather with mixed-polarity, mixed-amplitude earthquakes.
# Run with: julia --project=experimental/coherent_n2n experimental/coherent_n2n/test/test_outer_loop_polarity.jl

using Test
using Random
using FFTW
using Flux
using Statistics: mean, cor

include(joinpath(@__DIR__, "..", "CoherentN2N_shift.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_gauge.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_polarity.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_denoiser.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_n2n_pairs.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_train_denoiser.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_outer_loop.jl"))
include(joinpath(@__DIR__, "synthetic_data.jl"))

@testset "run_coherent_n2n: use_polarity_gain=true recovers source under mixed polarity" begin
    rng = MersenneTwister(40)
    nt = 128
    R = 20
    τ_true = Float32.(range(-8, 8, length=R))
    # Mixed polarity (+/-) and amplitude — the scenario use_polarity_gain exists for.
    g_true = ComplexF32.(rand(rng, (-1.0, 1.0), R) .* rand(rng, 0.7:0.1:1.3, R))

    s_true, D, freqs, _, _ = make_synthetic_gather(
        nt=nt, R=R, f0=0.05, source_kind=:broadband,
        true_shifts=τ_true, true_gains=g_true,
        noise_std=0.05, rng=rng)

    para = CoherentN2N_Para(nt=nt, kernels=[16, 8], filters=[8, 16])
    outer_para = CoherentN2N_Outer_Para(
        n_outer_iters=4, use_polarity_gain=true,
        denoiser_training=CoherentN2N_Denoiser_Training_Para(
            n_samples_per_epoch=256, batchsize=32, nepoch=80,
            initial_lr=0.003, restart_period=40, nprint=1000))

    result = run_coherent_n2n(D, para, outer_para; rng=rng)

    ŝ_time = real(ifft(result.ŝ))
    c = cor(ŝ_time, s_true)
    @info "Source recovery with polarity correction" correlation=c
    @test c > 0.85

    @info "Convergence history" delta_s=result.history.delta_s delta_tau=result.history.delta_tau
    @test all(isfinite, result.history.delta_s)
    @test all(isfinite, result.history.delta_tau)
    @test all(isfinite, result.gains)
end

@testset "run_coherent_n2n: use_polarity_gain=false ignores polarity flips (documented limitation)" begin
    # Without gain correction, shift estimation itself is now polarity-strict
    # (polarity_agnostic=!use_polarity_gain, see CoherentN2N_shift.jl), so a
    # flipped trace no longer gets a clean-but-wrong-signed alignment — it
    # gets a visibly bad shift estimate instead. That bad shift still isn't
    # corrected (no gain step to fix the sign), so it still degrades the
    # stack, just via a different mechanism (misalignment rather than
    # opposite-sign cancellation). This test documents the expected
    # degradation, not a bug.
    rng = MersenneTwister(40)
    nt = 128
    R = 20
    τ_true = Float32.(range(-8, 8, length=R))
    g_true = ComplexF32.(rand(rng, (-1.0, 1.0), R) .* rand(rng, 0.7:0.1:1.3, R))

    s_true, D, freqs, _, _ = make_synthetic_gather(
        nt=nt, R=R, f0=0.05, source_kind=:broadband,
        true_shifts=τ_true, true_gains=g_true,
        noise_std=0.05, rng=rng)

    para = CoherentN2N_Para(nt=nt, kernels=[16, 8], filters=[8, 16])
    outer_para = CoherentN2N_Outer_Para(
        n_outer_iters=4, use_polarity_gain=false,
        denoiser_training=CoherentN2N_Denoiser_Training_Para(
            n_samples_per_epoch=256, batchsize=32, nepoch=80,
            initial_lr=0.003, restart_period=40, nprint=1000))

    result = run_coherent_n2n(D, para, outer_para; rng=rng)
    ŝ_time = real(ifft(result.ŝ))
    c = cor(ŝ_time, s_true)
    @info "Source recovery without polarity correction" correlation=c
    @test all(isfinite, result.ŝ)  # should not crash/diverge to NaN even though quality suffers
end

println("All CoherentN2N polarity-toggle tests passed.")
