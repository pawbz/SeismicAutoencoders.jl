# Grouped (per-receiver) validation: run_coherent_n2n_grouped on several
# synthetic gathers, each with its OWN planted source wavelet, ragged R_g, and
# a shared denoiser. Confirms every valid receiver recovers ITS OWN ŝ_g (not a
# shared/mixed source), one model is shared across groups, and an under-sized
# group is flagged invalid and skipped without error. Polarity/gain is out of
# scope here (all unit gains) — see test_outer_loop_polarity.jl.
# Run with: julia --project=experimental/coherent_n2n experimental/coherent_n2n/test/test_outer_loop_grouped.jl

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

# Blind deconvolution recovers the source only up to a GLOBAL time shift (the
# reported ŝ sits at the mode-gauge origin, not sources[g]'s origin), so compare
# shapes by best circular-lag correlation. See the same helper in test_outer_loop.jl.
function aligned_source_cor(ŝ_time::AbstractVector, s_true::AbstractVector)
    a = ŝ_time ./ (sqrt(sum(abs2, ŝ_time)) + eps(Float32))
    b = s_true ./ (sqrt(sum(abs2, s_true)) + eps(Float32))
    cc = real(ifft(conj(fft(Float32.(a))) .* fft(Float32.(b))))
    lag = argmax(abs.(cc)) - 1
    return cor(circshift(ŝ_time, lag), s_true)
end

@testset "grouped sample_n2n_pairs: within-group only, concatenated batch" begin
    rng = MersenneTwister(7)
    nt = 16
    # Distinct disjoint value ranges per group so we can detect any cross-group pair.
    g1 = ComplexF32.(fill(1, nt, 3) .* reshape(1:3, 1, 3))      # columns hold 1,2,3
    g2 = ComplexF32.(fill(1, nt, 4) .* reshape(11:14, 1, 4))    # columns hold 11..14
    groups = [g1, g2]
    n = 50
    input, target = sample_n2n_pairs(groups, n; rng=rng)
    @test size(input) == (nt, 2n)      # 2 valid groups * n each, hcat'd
    @test size(target) == (nt, 2n)
    # Every (input,target) column pair must come from the SAME group: both in
    # {1,2,3} or both in {11,12,13,14}, never mixed.
    for c in 1:2n
        iv = real(input[1, c]); tv = real(target[1, c])
        in_g1 = iv <= 3 && tv <= 3
        in_g2 = iv >= 11 && tv >= 11
        @test in_g1 || in_g2
        @test iv != tv                 # distinct columns within the group
    end

    # A group with < 2 columns is skipped, not errored.
    groups_deg = [g1, ComplexF32.(reshape(collect(1:nt), nt, 1))]
    inp2, _ = sample_n2n_pairs(groups_deg, n; rng=rng)
    @test size(inp2, 2) == n           # only g1 contributes
end

@testset "run_coherent_n2n_grouped_baseline: stochastic refs smoke + invalid guard" begin
    nt = 64
    groups = Matrix{Float32}[]
    for (i, R) in enumerate((8, 11, 5))
        τ_true = Float32.(range(-4, 4, length=R))
        _, D, _, _, _ = make_synthetic_gather(
            nt=nt, R=R, f0=0.04 + 0.01i, source_kind=:broadband,
            true_shifts=τ_true, true_gains=ones(ComplexF32, R),
            noise_std=0.05, rng=MersenneTwister(50 + i))
        push!(groups, D)
    end
    _, D_bad, _, _, _ = make_synthetic_gather(
        nt=nt, R=1, f0=0.05, source_kind=:broadband,
        true_shifts=Float32[0], true_gains=ones(ComplexF32, 1),
        noise_std=0.05, rng=MersenneTwister(59))
    push!(groups, D_bad)

    para = CoherentN2N_Para(nt=nt, kernels=[8], filters=[8])
    outer_para = CoherentN2N_Outer_Para(
        n_outer_iters=2,
        stochastic_baseline_ref=true,
        stochastic_baseline_subset_size=0)
    result = run_coherent_n2n_grouped_baseline(groups, para, outer_para;
                                               rng=MersenneTwister(60), min_events=2)

    @test result.model === nothing
    @test result.valid == BitVector([true, true, true, false])
    @test length(result.groups) == 4
    @test isempty(result.history.denoiser_loss)
    @test length(result.history.delta_s[1]) == 2
    @test isempty(result.history.delta_s[4])
    for g in 1:4
        @test all(isfinite, result.groups[g].τ)
        @test all(isfinite, result.groups[g].ŝ)
    end
end

@testset "run_coherent_n2n_grouped: per-receiver source recovery + shared model + guard" begin
    rng = MersenneTwister(40)
    nt = 128

    # Three receivers, each a DIFFERENT source wavelet and different R_g, plus a
    # fourth degenerate receiver (R_g=1) to exercise the validity guard.
    specs = [(R=18, f0=0.04), (R=12, f0=0.07), (R=24, f0=0.10)]
    sources = Vector{Vector{Float32}}()
    groups = Vector{Matrix{Float32}}()
    for (i, sp) in enumerate(specs)
        τ_true = Float32.(range(-8, 8, length=sp.R))
        s_true, D, _, _, _ = make_synthetic_gather(
            nt=nt, R=sp.R, f0=sp.f0, source_kind=:broadband,
            true_shifts=τ_true, true_gains=ones(ComplexF32, sp.R),
            noise_std=0.05, rng=MersenneTwister(100 + i))
        push!(sources, s_true)
        push!(groups, D)
    end
    # Degenerate 4th group: single trace — cannot form an N2N pair.
    _, D_bad, _, _, _ = make_synthetic_gather(
        nt=nt, R=1, f0=0.05, source_kind=:broadband,
        true_shifts=Float32[0], true_gains=ones(ComplexF32, 1),
        noise_std=0.05, rng=MersenneTwister(999))
    push!(groups, D_bad)

    para = CoherentN2N_Para(nt=nt, kernels=[16, 8], filters=[8, 16])
    outer_para = CoherentN2N_Outer_Para(
        n_outer_iters=4,
        denoiser_training=CoherentN2N_Denoiser_Training_Para(
            n_samples_per_epoch=256, batchsize=32, nepoch=80,
            initial_lr=0.003, restart_period=40, nprint=1000))

    result = run_coherent_n2n_grouped(groups, para, outer_para; rng=rng, min_events=2)

    # Output is index-aligned to the 4 input receivers.
    @test length(result.groups) == 4
    @test result.valid == BitVector([true, true, true, false])

    # ONE shared model is returned (a single ComplexDenoiser object).
    @test result.model isa ComplexDenoiser

    # Each valid receiver recovers ITS OWN planted source strongly (best-lag
    # aligned — ŝ is only defined up to a global shift). Bar is 0.85, matching
    # test_outer_loop_polarity.jl's source-recovery bar: under this short
    # grouped training budget one group lands ~0.86, which the discrimination
    # test below confirms is still genuinely its OWN source, not a mixed one.
    for g in 1:3
        ŝ_time = real(ifft(result.groups[g].ŝ))
        c_own = aligned_source_cor(ŝ_time, sources[g])
        @info "Group $g source recovery" correlation=c_own
        @test abs(c_own) > 0.85
    end

    # ...and better than it matches a DIFFERENT receiver's source (groups are
    # genuinely distinct, not collapsing to one shared ŝ). f0 differs enough
    # that the cross-source correlation is clearly lower. (Best-lag aligned for
    # each candidate, since ŝ is only recovered up to a global shift.)
    for g in 1:3
        ŝ_time = real(ifft(result.groups[g].ŝ))
        other = g == 1 ? 3 : 1
        @test abs(aligned_source_cor(ŝ_time, sources[g])) >
              abs(aligned_source_cor(ŝ_time, sources[other]))
    end

    # The degenerate group is kept (init-only) with finite ŝ/τ and never NaNs.
    @test all(isfinite, result.groups[4].ŝ)
    @test all(isfinite, result.groups[4].τ)
    @test length(result.groups[4].τ) == 1

    # History: shared denoiser_loss (one curve per outer iter), per-group deltas.
    @test length(result.history.denoiser_loss) == 4
    @test length(result.history.delta_s) == 4
    @test length(result.history.delta_s[1]) == 4   # valid group has 4 iters logged
    @test isempty(result.history.delta_s[4])       # invalid group logs nothing
    @test all(v -> all(isfinite, v), result.history.delta_s)
    @test all(v -> all(isfinite, v), result.history.delta_tau)
end

println("All CoherentN2N grouped outer-loop tests passed.")
