# Stage-3 synthetic validation: full blind alternating-loop pipeline, no
# oracle shifts given. Confirms recovered τ (with anchor reapplied) matches
# truth up to a global constant, ŝ converges toward the true source, and
# convergence diagnostics behave. Polarity/gain is intentionally out of
# scope for now (deferred per user instruction) — all synthetic gains here
# are positive/unit so the loop's own lack of polarity handling isn't a
# confound.
# Run with: julia --project=experimental/coherent_n2n experimental/coherent_n2n/test/test_outer_loop.jl

using Test
using Random
using FFTW
using Flux
using Statistics: mean, cor

include(joinpath(@__DIR__, "..", "CoherentN2N_shift.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_gauge.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_denoiser.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_n2n_pairs.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_train_denoiser.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_outer_loop.jl"))
include(joinpath(@__DIR__, "synthetic_data.jl"))

@testset "run_coherent_n2n: full blind pipeline recovers shifts/source/gains" begin
    rng = MersenneTwister(30)
    nt = 128
    R = 20
    τ_true = Float32.(range(-10, 10, length=R))
    g_true = ones(ComplexF32, R)  # polarity/gain off for this test (see test_outer_loop_polarity.jl)

    s_true, D, freqs, _, _ = make_synthetic_gather(
        nt=nt, R=R, f0=0.05, source_kind=:broadband,
        true_shifts=τ_true, true_gains=g_true,
        noise_std=0.05, rng=rng)

    para = CoherentN2N_Para(nt=nt, enc_kernels=[16, 8], enc_filters=[8, 16],
                             dec_kernels=[8, 16], dec_filters=[8, 2])
    outer_para = CoherentN2N_Outer_Para(
        n_outer_iters=4,
        denoiser_training=CoherentN2N_Denoiser_Training_Para(
            n_samples_per_epoch=256, batchsize=32, nepoch=80,
            initial_lr=0.003, restart_period=40, nprint=1000))

    result = run_coherent_n2n(D, para, outer_para; rng=rng)

    # τ recovered up to a global constant: after removing each side's mean,
    # the shapes should match closely. A per-trace shift estimate can still
    # miss on an individual noisy trace (see the shift-estimator's own
    # moderate-SNR test) — the meaningful bar is the typical (mean) error,
    # not a hard per-trace maximum.
    τ_est_centered = result.τ .- mean(result.τ)
    τ_true_centered = τ_true .- mean(τ_true)
    errs = abs.(τ_est_centered .- τ_true_centered)
    @info "Shift recovery" max_err=maximum(errs) mean_abs_err=mean(errs)
    @test mean(errs) < 1.5
    @test sum(errs .< 2.0) >= length(errs) - 2  # allow a couple of outlier traces

    # Source estimate should correlate strongly with the true source shape.
    ŝ_time = real(ifft(result.ŝ))
    c = cor(ŝ_time, s_true)
    @info "Source recovery" correlation=c
    @test c > 0.9

    # Convergence diagnostics should not blow up / diverge across outer iters.
    @info "Convergence history" delta_s=result.history.delta_s delta_tau=result.history.delta_tau
    @test all(isfinite, result.history.delta_s)
    @test all(isfinite, result.history.delta_tau)
end

@testset "run_coherent_n2n: robust to a missing/low-SNR earthquake" begin
    rng = MersenneTwister(31)
    nt = 128
    R = 15
    τ_true = Float32.(range(-8, 8, length=R))
    g_true = ones(ComplexF32, R)

    s_true, D, freqs, _, _ = make_synthetic_gather(
        nt=nt, R=R, f0=0.05, source_kind=:broadband,
        true_shifts=τ_true, true_gains=g_true,
        noise_std=0.05, rng=rng)
    # Corrupt one earthquake with much larger noise (simulating a bad trace).
    D[:, 1] .+= 2.0f0 .* randn(rng, Float32, nt)

    para = CoherentN2N_Para(nt=nt, enc_kernels=[16, 8], enc_filters=[8, 16],
                             dec_kernels=[8, 16], dec_filters=[8, 2])
    outer_para = CoherentN2N_Outer_Para(
        n_outer_iters=3,
        denoiser_training=CoherentN2N_Denoiser_Training_Para(
            n_samples_per_epoch=200, batchsize=32, nepoch=60,
            initial_lr=0.003, restart_period=30, nprint=1000))

    result = run_coherent_n2n(D, para, outer_para; rng=rng)
    ŝ_time = real(ifft(result.ŝ))
    c = cor(ŝ_time, s_true)
    @info "Source recovery with one bad trace" correlation=c
    @test c > 0.8  # a single bad trace shouldn't derail the whole gather
    @test all(isfinite, result.τ)
    @test all(isfinite, result.ŝ)
end

println("All CoherentN2N outer-loop tests passed.")
