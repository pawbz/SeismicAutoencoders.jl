# Alignment-free grouped loops: run_coherent_n2n_grouped_noalign and its
# network-free baseline. The premise is data that is ALREADY coherent (one
# distance/azimuth bin), so the gathers here are built with true_shifts = 0 and
# the loop must recover each planted source by denoising alone — τ never moves.
# Run with: julia --project=experimental/coherent_n2n experimental/coherent_n2n/test/test_outer_loop_noalign.jl

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
include(joinpath(@__DIR__, "..", "CoherentN2N_polarity.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_outer_loop.jl"))
include(joinpath(@__DIR__, "synthetic_data.jl"))

"""Build a zero-jitter noisy gather: every column is the SAME wavelet + noise."""
function make_coherent_gather(; nt, R, f0, noise_std, seed)
    s_true, D, _, τ, _ = make_synthetic_gather(
        nt=nt, R=R, f0=f0, source_kind=:broadband,
        true_shifts=zeros(Float32, R), true_gains=ones(ComplexF32, R),
        noise_std=noise_std, rng=MersenneTwister(seed))
    @assert all(iszero, τ)          # the premise of this whole test file
    return s_true, D
end

@testset "run_coherent_n2n_grouped_noalign_baseline: raw stack of the bin" begin
    nt = 64
    # noise_std is ABSOLUTE while the broadband wavelet's RMS is only ~0.15, so
    # 0.2 here is already a ~1.3:1 noise-to-signal ratio per trace; stacking 12
    # of them recovers the wavelet at ~0.91 correlation.
    s1, D1 = make_coherent_gather(nt=nt, R=12, f0=0.05, noise_std=0.2, seed=11)
    _, D2 = make_coherent_gather(nt=nt, R=7, f0=0.08, noise_std=0.2, seed=12)
    _, D_bad = make_coherent_gather(nt=nt, R=1, f0=0.05, noise_std=0.2, seed=13)
    groups = [D1, D2, D_bad]

    para = CoherentN2N_Para(nt=nt, kernels=[8], filters=[8])
    outer_para = CoherentN2N_Outer_Para(n_outer_iters=3)
    result = run_coherent_n2n_grouped_noalign_baseline(groups, para, outer_para;
                                                        rng=MersenneTwister(14))

    @test result.model === nothing
    @test result.valid == BitVector([true, true, false])
    @test length(result.groups) == 3
    @test isempty(result.history.denoiser_loss)

    for g in 1:3
        @test all(iszero, result.groups[g].τ)
        @test all(isfinite, result.groups[g].ŝ)
        @test all(isone, result.groups[g].gains)
        @test result.groups[g].anchor == 0f0
    end

    # ŝ_g IS the plain stack of group g (no alignment, no network), so it must
    # equal the mean spectrum exactly for the default :l2 stack.
    @test result.groups[1].ŝ ≈ vec(mean(ComplexF32.(fft(D1, 1)), dims=2))
    # ...and in the time domain, the plain trace mean.
    @test real(ifft(result.groups[1].ŝ)) ≈ vec(mean(D1, dims=2)) rtol = 1e-4
    # Stacking 12 noisy copies already recovers the planted wavelet well (~0.91).
    @test cor(real(ifft(result.groups[1].ŝ)), s1) > 0.85
end

@testset "run_coherent_n2n_grouped_noalign: τ stays zero, ŝ beats the raw stack" begin
    nt = 128
    # Two receivers with DIFFERENT planted sources and different R_g, plus a
    # degenerate single-trace receiver to exercise the validity guard. noise_std
    # 0.4 against a ~0.1 RMS wavelet leaves the raw stacks at ~0.92 / ~0.85
    # correlation — good enough to be a fair baseline, poor enough that the
    # denoiser has real residual left to remove.
    s1, D1 = make_coherent_gather(nt=nt, R=60, f0=0.04, noise_std=0.4, seed=21)
    s2, D2 = make_coherent_gather(nt=nt, R=40, f0=0.09, noise_std=0.4, seed=22)
    _, D_bad = make_coherent_gather(nt=nt, R=1, f0=0.05, noise_std=0.4, seed=23)
    groups = [D1, D2, D_bad]
    sources = [s1, s2]

    para = CoherentN2N_Para(nt=nt, kernels=[16, 8], filters=[8, 16])
    outer_para = CoherentN2N_Outer_Para(
        n_outer_iters=4,
        denoiser_training=CoherentN2N_Denoiser_Training_Para(
            n_samples_per_epoch=256, batchsize=32, nepoch=80,
            initial_lr=0.003, restart_period=40, nprint=1000))

    result = run_coherent_n2n_grouped_noalign(groups, para, outer_para;
                                               rng=MersenneTwister(24), min_events=2)

    @test length(result.groups) == 3
    @test result.valid == BitVector([true, true, false])
    @test result.model isa ComplexDenoiser

    # THE core invariant: no shift degree of freedom exists at all.
    for g in 1:3
        @test all(iszero, result.groups[g].τ)
        @test length(result.groups[g].τ) == size(groups[g], 2)
        @test all(isone, result.groups[g].gains)
        @test result.groups[g].anchor == 0f0
        @test all(isfinite, result.groups[g].ŝ)
    end
    @test all(v -> all(iszero, v), result.history.delta_tau)

    # Source recovery: because there is no alignment, ŝ is recovered in the SAME
    # time frame as the planted source — no best-lag search is needed here (that
    # is only required for the aligned loops, which recover ŝ up to a shift).
    for g in 1:2
        ŝ_time = real(ifft(result.groups[g].ŝ))
        raw_time = vec(mean(groups[g], dims=2))
        c_n2n = cor(ŝ_time, sources[g])
        c_raw = cor(raw_time, sources[g])
        @info "Group $g (no-align) source recovery" n2n = c_n2n raw_stack = c_raw
        @test c_n2n > 0.85
        # The denoiser must beat the plain stack it was initialized from —
        # that is the entire justification for running the network at all.
        @test c_n2n > c_raw
    end

    # Each group recovers ITS OWN source, not a shared/mixed one.
    for g in 1:2
        ŝ_time = real(ifft(result.groups[g].ŝ))
        other = g == 1 ? 2 : 1
        @test abs(cor(ŝ_time, sources[g])) > abs(cor(ŝ_time, sources[other]))
    end

    # History bookkeeping matches the aligned grouped loop's shape.
    @test length(result.history.denoiser_loss) == 4
    @test length(result.history.delta_s) == 3
    @test length(result.history.delta_s[1]) == 4
    @test isempty(result.history.delta_s[3])       # invalid group logs nothing
    @test all(v -> all(isfinite, v), result.history.delta_s)
    # The denoiser loss decreases over training within the first outer iteration.
    first_curve = result.history.denoiser_loss[1]
    @test mean(first_curve[max(1, end - 9):end]) < mean(first_curve[1:min(10, end)])
end

@testset "run_coherent_n2n_grouped_noalign: ignored-kwarg warnings" begin
    nt = 32
    _, D = make_coherent_gather(nt=nt, R=6, f0=0.05, noise_std=0.3, seed=31)
    para = CoherentN2N_Para(nt=nt, kernels=[8], filters=[8])

    # Shift-related settings are silently inapplicable here, so the loop must say so.
    op = CoherentN2N_Outer_Para(n_outer_iters=1, use_polarity_gain=true,
                                max_shift=5f0, stochastic_baseline_ref=true)
    @test_logs (:warn,) (:warn,) (:warn,) match_mode = :any begin
        run_coherent_n2n_grouped_noalign_baseline([D], para, op; rng=MersenneTwister(32))
    end

    # A clean para produces no warnings from the ignored-kwarg check.
    op_clean = CoherentN2N_Outer_Para(n_outer_iters=1)
    r = run_coherent_n2n_grouped_noalign_baseline([D], para, op_clean; rng=MersenneTwister(33))
    @test all(iszero, r.groups[1].τ)
end

println("All CoherentN2N no-align outer-loop tests passed.")
