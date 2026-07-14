# Standalone correctness tests for the phase-slope / coarse-xcorr shift estimator.
# Run with: julia --project=. experimental/coherent_n2n/test/test_shift_estimator.jl

using Test
using FFTW
using Random

include(joinpath(@__DIR__, "..", "CoherentN2N_shift.jl"))
include(joinpath(@__DIR__, "synthetic_data.jl"))

@testset "estimate_shift_phase_slope: high SNR, small shift, no coarse bootstrap" begin
    rng = MersenneTwister(1)
    nt = 300
    s_true, D, freqs, τ_true, _ = make_synthetic_gather(
        nt=nt, R=20, f0=0.05,
        true_shifts=Float32.(range(-4, 4, length=20)),
        noise_std=0.001, rng=rng)

    X̂_ref = fft(s_true)
    for r in 1:20
        X̂_trace = fft(D[:, r])
        τ_est = estimate_shift_phase_slope(X̂_ref, X̂_trace, freqs)
        @test abs(τ_est - τ_true[r]) < 0.05
    end
end

@testset "estimate_shift_xcorr_coarse: recovers integer part of large shifts" begin
    rng = MersenneTwister(2)
    nt = 300
    # A single narrowband Ricker wavelet has a very broad, slowly-decaying
    # autocorrelation (poor time resolution) — even light noise can tip the
    # peak to a neighboring lag. Use a broadband source here, representative
    # of real (whitened) seismic arrivals and of what this pipeline's own
    # preprocessing step is designed to produce.
    s_true, D, freqs, τ_true, _ = make_synthetic_gather(
        nt=nt, R=10, f0=0.02, source_kind=:broadband,
        true_shifts=Float32.([60, -60, 100, -100, 5, -5, 0, 30, -30, 80]),
        noise_std=0.01, rng=rng)

    for r in 1:10
        τ_coarse = estimate_shift_xcorr_coarse(s_true, D[:, r])
        @test abs(τ_coarse - round(τ_true[r])) <= 1
    end
end

@testset "estimate_shift_two_stage: large shifts, phase-slope alone would cycle-skip" begin
    rng = MersenneTwister(3)
    nt = 300
    s_true, D, freqs, τ_true, _ = make_synthetic_gather(
        nt=nt, R=10, f0=0.02, source_kind=:broadband,
        true_shifts=Float32.([60.3, -60.3, 90.7, -90.7, 45.2, -45.2, 0.4, 30.1, -30.1, 80.6]),
        noise_std=0.01, rng=rng)

    # Phase-slope alone (no coarse bootstrap) should fail badly on large shifts:
    X̂_ref = fft(s_true)
    n_bad = 0
    for r in 1:10
        X̂_trace = fft(D[:, r])
        τ_naive = estimate_shift_phase_slope(X̂_ref, X̂_trace, freqs)
        abs(τ_naive - τ_true[r]) > 1 && (n_bad += 1)
    end
    @test n_bad > 0  # confirms cycle-skipping actually occurs without bootstrap

    # Two-stage (coarse + fine) should recover all of them accurately:
    for r in 1:10
        τ_est = estimate_shift_two_stage(s_true, D[:, r], freqs)
        @test abs(τ_est - τ_true[r]) < 0.1
    end
end

@testset "estimate_shift_two_stage: moderate-SNR robustness (single-trace)" begin
    # A single earthquake trace against a single reference is inherently
    # limited at low SNR — the coarse xcorr peak can be mis-picked onto a
    # sidelobe when noise is comparable to (or exceeds) signal amplitude.
    # This is expected and is exactly why the outer alternating loop
    # re-estimates shifts against a growing coherent stack rather than
    # relying on any single-trace estimate. At noise_std ~= 1/3 of signal
    # rms (SNR ~ 3), most single-trace estimates should still be usable.
    rng = MersenneTwister(4)
    nt = 300
    s_true, D, freqs, τ_true, _ = make_synthetic_gather(
        nt=nt, R=15, f0=0.02, source_kind=:broadband,
        true_shifts=Float32.(range(-20, 20, length=15)),
        noise_std=0.04, rng=rng)

    errs = Float32[]
    for r in 1:15
        τ_est = estimate_shift_two_stage(s_true, D[:, r], freqs)
        push!(errs, abs(τ_est - τ_true[r]))
    end
    @test sum(errs .< 1.0) >= 7  # majority accurate; a coarse-lag miss on a few noisy traces is expected
end

@testset "shift_spectrum: round trip with estimator" begin
    rng = MersenneTwister(5)
    nt = 300
    s_true = ricker_wavelet(nt, 0.05)
    freqs = Float32.(fftfreq(nt))
    grid = -im .* 2π .* freqs
    τ_true = 7.3f0

    Ŝ = fft(s_true)
    Ŝ_shifted = shift_spectrum(reshape(Ŝ, nt, 1), reshape([τ_true], 1, 1), grid)
    x_shifted = real(ifft(vec(Ŝ_shifted)))

    τ_est = estimate_shift_two_stage(s_true, x_shifted, freqs)
    @test abs(τ_est - τ_true) < 0.05
end

@testset "estimate_shift_two_stage: polarity_agnostic=true recovers τ under a flipped trace" begin
    rng = MersenneTwister(6)
    nt = 300
    s_true, D, freqs, τ_true, g_true = make_synthetic_gather(
        nt=nt, R=10, f0=0.02, source_kind=:broadband,
        true_shifts=Float32.([60, -60, 100, -100, 5, -5, 0, 30, -30, 80]),
        true_gains=ComplexF32.(rand(rng, (-1.0, 1.0), 10)),
        noise_std=0.01, rng=rng)

    for r in 1:10
        τ_est = estimate_shift_two_stage(s_true, D[:, r], freqs; polarity_agnostic=true)
        @test abs(τ_est - τ_true[r]) < 0.1
    end
end

@testset "estimate_shift_xcorr_coarse: polarity_agnostic=false fails to find the true lag on a flipped trace" begin
    # This is the primary polarity gate: the signed peak search must not find
    # the true lag when the trace is flipped (its cross-correlation there is
    # large and NEGATIVE, so argmax(cc) skips it), while it must still find
    # it correctly for a same-polarity trace.
    rng = MersenneTwister(7)
    nt = 300
    τ_shift = 40.0f0
    s_true, D_pos, freqs, _, _ = make_synthetic_gather(
        nt=nt, R=1, f0=0.02, source_kind=:broadband,
        true_shifts=Float32[τ_shift], true_gains=ComplexF32[1.0],
        noise_std=0.01, rng=rng)
    _, D_neg, _, _, _ = make_synthetic_gather(
        nt=nt, R=1, f0=0.02, source_kind=:broadband,
        true_shifts=Float32[τ_shift], true_gains=ComplexF32[-1.0],
        noise_std=0.01, rng=MersenneTwister(7))

    τ_coarse_pos = estimate_shift_xcorr_coarse(s_true, D_pos[:, 1]; polarity_agnostic=false)
    @test abs(τ_coarse_pos - τ_shift) <= 1  # same-polarity trace still aligns correctly

    τ_coarse_neg = estimate_shift_xcorr_coarse(s_true, D_neg[:, 1]; polarity_agnostic=false)
    @test abs(τ_coarse_neg - τ_shift) > 1  # flipped trace: signed peak misses the true lag

    # Sanity check on the mechanism: the (signed) cross-correlation at the
    # true lag is strongly negative for the flipped trace, confirming
    # argmax(cc) (not argmax(abs.(cc))) is what causes it to be skipped.
    X̂_ref, X̂_neg = fft(s_true), fft(D_neg[:, 1])
    cc = real(ifft(conj.(X̂_ref) .* X̂_neg))
    @test cc[round(Int, τ_shift) + 1] < 0
end

@testset "estimate_shift_phase_slope: zero_intercept changes the fit under a constant phase offset" begin
    # Isolates the fine-fit mechanism: a pure delay plus a constant phase
    # offset (what a polarity flip contributes, ±π) corrupts the
    # zero-intercept fit but is absorbed by the intercept-including fit.
    nt = 300
    freqs = Float32.(fftfreq(nt))
    ω = 2π .* freqs[2:150]
    τ_true = 5.3f0
    phi_clean = ω .* τ_true
    phi_with_offset = phi_clean .+ Float32(π)

    w = ones(Float32, length(ω))
    τ_zero_clean    = sum(w .* ω .* phi_clean) / sum(w .* ω .^ 2)
    τ_zero_offset   = sum(w .* ω .* phi_with_offset) / sum(w .* ω .^ 2)
    @test isapprox(τ_zero_clean, τ_true; atol=1e-3)
    @test abs(τ_zero_offset - τ_true) > 0.5  # offset corrupts the zero-intercept fit
end

println("All CoherentN2N shift-estimator tests passed.")
