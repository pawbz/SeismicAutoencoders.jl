# Tests for bispectrum-inversion MRA (BispectrumMRA.jl).
#
# What these validate, in order: the 0-based/1-based Fourier index algebra; the
# shift-invariance claim that the whole method rests on (including the no-wrap
# triple restriction, which is the single easiest thing to get wrong); the
# absence of a reflection ambiguity; the nt-factor noise debiasing; exact
# noiseless recovery; phase recovery specifically (phase is the scientific
# target, not the waveform); behaviour under noise and growing sample size; and
# the documented per-trace-polarity limitation.
#
# Out of scope: colored-noise debiasing (noise_bias is plumbed but untested
# beyond shape), and any real-data behaviour — that lives in the notebook.
#
# Run with: julia --project=experimental/bispectrum_mra experimental/bispectrum_mra/test/test_bispectrum_mra.jl

using Test
using Random
using FFTW
using Statistics: mean, std, cor

include(joinpath(@__DIR__, "..", "BispectrumMRA.jl"))
# Reuse the CoherentN2N wavelet/gather generators rather than duplicating them.
include(joinpath(@__DIR__, "..", "..", "coherent_n2n", "test", "synthetic_data.jl"))

# Run the CoherentN2N alignment baseline in the benchmark table at the bottom.
# It pulls in Flux and dominates the runtime, so it is opt-in.
const RUN_BASELINE = true

# Blind recovery only determines the source up to a circular shift, so every
# waveform comparison goes through a best-lag correlation. Copied from
# coherent_n2n/test/test_outer_loop_grouped.jl:27 (kept local, per house style).
function aligned_source_cor(ŝ_time::AbstractVector, s_true::AbstractVector)
    a = ŝ_time ./ (sqrt(sum(abs2, ŝ_time)) + eps(Float32))
    b = s_true ./ (sqrt(sum(abs2, s_true)) + eps(Float32))
    cc = real(ifft(conj(fft(Float32.(a))) .* fft(Float32.(b))))
    lag = argmax(abs.(cc)) - 1
    return cor(circshift(ŝ_time, lag), s_true)
end

"""
Magnitude-weighted RMS phase error between a recovered and a true phase spectrum,
after removing the global-shift gauge.

The two differ by an unknown linear ramp `k·Δ` (a circular shift). Δ is fitted on
FIRST DIFFERENCES: differencing kills the unknown constant and keeps every
quantity wrapped, whereas least-squares-fitting a ramp to raw wrapped angles is
the classic trap here.

The per-difference average must itself be CIRCULAR — average `cis(d)` and take
the angle, not the arithmetic mean of `d`. A plain mean breaks exactly at
Δ = ±π (a half-length shift, i.e. lag nt/2, which is a perfectly ordinary
outcome of blind recovery): there `+π` and `-π` are the same angle but average
to 0, and the metric then reports a correctly-recovered spectrum as ~1.9 rad
wrong. Same unit-complex-mean trick `invert_phases_freq_marching` uses.

Only bins with `a[k] >= band_thr*max(a)` are scored — phases carrying no energy
are meaningless, and including them would swamp the metric with noise.
"""
function phase_error(φ̂::AbstractVector, φt::AbstractVector, a::AbstractVector;
                     band_thr::Real=0.1)
    K = min(length(φ̂), length(φt), length(a)) - 1
    ab = a[1:K+1]
    keep = findall(k -> ab[k+1] >= band_thr * maximum(ab), 0:K)
    length(keep) < 3 && return 0.0
    wrap(x) = mod(x + π, 2π) - π
    d = [wrap((φ̂[k+1] - φ̂[k]) - (φt[k+1] - φt[k])) for k in 1:K]
    w = [ab[k+1] for k in 1:K]
    Δ = angle(sum(w .* cis.(d)))                  # circular mean, stable at ±π
    res = [wrap(φ̂[k+1] - φt[k+1] - k * Δ) for k in keep .- 1]
    ww = [ab[k+1] for k in keep .- 1]
    return sqrt(sum(ww .* res .^ 2) / sum(ww))
end

# Unit-gain gather: the ±1 polarity that make_synthetic_gather applies by default
# would destroy the bispectrum (see the polarity testset at the bottom).
function unit_gain_gather(; nt, R, noise, seed, f0=0.05)
    rng = MersenneTwister(seed)
    return make_synthetic_gather(nt=nt, R=R, f0=f0, noise_std=noise, rng=rng,
                                 source_kind=:broadband,
                                 true_gains=ones(ComplexF32, R))
end

@testset "fidx: 0-based frequency -> 1-based array index, circular" begin
    nt = 64
    @test fidx(0, nt) == 1
    @test fidx(1, nt) == 2
    @test fidx(nt, nt) == 1              # wraps a full turn back to DC
    @test fidx(nt - 1, nt) == nt
    # The negative case is what Hermitian symmetrization relies on; `%` would
    # give 0 here and index out of bounds.
    @test fidx(-1, nt) == nt
    @test fidx(-nt + 1, nt) == 2
end

@testset "bispectrum invariance: integer shift, symmetry, and the no-wrap rule" begin
    nt = 64
    rng = MersenneTwister(11)
    s = Float32.(randn(rng, nt))
    Y0 = ComplexF32.(reshape(fft(s), nt, 1))
    kmax = nt ÷ 2 - 1

    B0 = bispectrum_estimate(Y0; kmax=kmax).B

    # Symmetry is exact by construction (upper triangle mirrored).
    @test B0 ≈ transpose(B0)

    # Integer circular shifts are the model the theory assumes: invariance here
    # is exact up to floating point.
    for lag in (1, 7, 33)
        Ys = ComplexF32.(reshape(fft(circshift(s, lag)), nt, 1))
        Bs = bispectrum_estimate(Ys; kmax=kmax).B
        @test maximum(abs.(Bs .- B0)) / maximum(abs.(B0)) < 1f-4
    end

    # Sub-sample (fractional) shifts — a phase ramp — are invariant ONLY on
    # no-wrap triples, which is exactly what bispectrum_estimate fills. This is
    # the property that lets us use make_synthetic_gather's fractional shifts.
    freqs = Float32.(fftfreq(nt))
    for τ in (3.7f0, -2.3f0, 0.5f0)
        Yf = ComplexF32.(reshape(fft(s) .* exp.(-im .* 2f0 .* Float32(π) .* freqs .* τ), nt, 1))
        Bf = bispectrum_estimate(Yf; kmax=kmax).B
        @test maximum(abs.(Bf .- B0)) / maximum(abs.(B0)) < 1f-4
    end

    # ...and the converse, which is why the restriction exists. `fftfreq` is
    # SIGNED, so it stops being additive exactly when k1+k2 reaches Nyquist:
    # for (k1,k2) = (31,1) at nt=64, f[31]+f[1] = 0.484+0.016 = +0.5 but
    # f[32] = -0.5. A triple like that is precisely what a naive `kmax = nt÷2`
    # would admit, and on it a fractional shift no longer cancels at all.
    # If this ever stops failing, the no-wrap guard in bispectrum_estimate has
    # been removed and fractional shifts are silently corrupting the estimate.
    let τ = 3.7f0
        Yf = fft(s) .* exp.(-im .* 2 * π .* freqs .* τ)
        wrapped(X, k1, k2) = X[fidx(k1, nt)] * X[fidx(k2, nt)] * conj(X[fidx(k1 + k2, nt)])
        k1, k2 = nt ÷ 2 - 1, 1                    # k1 + k2 = nt÷2 -> hits Nyquist
        ref = wrapped(fft(s), k1, k2)
        @test abs(wrapped(Yf, k1, k2) - ref) / abs(ref) > 0.1
        # The guard means bispectrum_estimate never populates that entry.
        @test iszero(B0[k1+1, k2+1])
    end
end

@testset "reflection is NOT ambiguous: B -> conj(B) under time reversal" begin
    nt = 64
    # The test signal has to satisfy two competing requirements:
    #  - ASYMMETRIC, else reflection is a no-op. Any centered Ricker (and the
    #    centered broadband wavelet) is time-symmetric, which makes B exactly
    #    real, B == conj(B), and the whole testset vacuous.
    #  - BAND-LIMITED, else the kmax low-pass in reconstruct_signal dominates
    #    and the end-to-end check below cannot clear a meaningful bar (white
    #    noise caps out around 0.61 for exactly this reason).
    # Two Rickers of different width at different centers satisfy both.
    s = ricker_wavelet(nt, 0.08; t0=nt ÷ 2 - 6) .+
        0.6f0 .* ricker_wavelet(nt, 0.16; t0=nt ÷ 2 + 5)
    sr = circshift(reverse(s), 1)          # true x(-t) on the circle
    kmax = nt ÷ 2 - 1

    # Precondition: guard against silently reintroducing a symmetric signal.
    @test aligned_source_cor(sr, s) < 0.95
    B  = bispectrum_estimate(ComplexF32.(reshape(fft(s), nt, 1)); kmax=kmax).B
    Br = bispectrum_estimate(ComplexF32.(reshape(fft(sr), nt, 1)); kmax=kmax).B

    scale = maximum(abs.(B))
    @test maximum(abs.(Br .- conj.(B))) / scale < 1f-4     # reflection conjugates B
    @test maximum(abs.(Br .- B)) / scale > 0.1             # ...and is distinguishable

    # Consequence: unlike phase retrieval from magnitudes alone, bispectrum
    # inversion determines orientation, so no reflection resolver is needed.
    # Confirm end-to-end that a reflected gather recovers the reflected signal
    # and not the original.
    R = 400
    Dr = Matrix{Float32}(undef, nt, R)
    rng2 = MersenneTwister(14)
    for i in 1:R
        Dr[:, i] = circshift(sr, rand(rng2, 0:nt-1))
    end
    x̂ = recover_mra(Dr).x
    c_refl = aligned_source_cor(x̂, sr)
    c_orig = aligned_source_cor(x̂, s)
    @test c_refl > 0.95
    @test c_refl > c_orig + 0.1        # orientation is genuinely determined
    @info "reflection" cor_vs_reflected=round(c_refl, digits=4) cor_vs_original=round(c_orig, digits=4)
end

@testset "power spectrum debiasing carries the nt factor" begin
    nt = 64
    R = 20000
    σ = 1.3f0
    s, D, _, _, _ = unit_gain_gather(nt=nt, R=R, noise=σ, seed=17)
    Ptrue = abs2.(fft(s))

    ps = power_spectrum_estimate(D; sigma2=σ^2)
    scale = maximum(Ptrue)
    # σ=1.3 is large next to the wavelet, so even at R=20000 the residual is
    # dominated by Monte-Carlo error: measured 0.137-0.205 across 5 seeds. The
    # bar sits above that spread and still ~30x below the un-debiased error
    # below, so it separates "correct" from "buggy" without being seed-fragile.
    @test maximum(abs.(ps.P .- Ptrue)) / scale < 0.3

    # Without the nt factor the bias removal is short by nt*, so the same
    # comparison must fail badly (measured ~9.3, vs ~0.2 when correct). This
    # pins the classic bug: if someone "simplifies" P .-= nt*sigma2 to
    # P .-= sigma2, this test fires.
    P_wrong = vec(mean(abs2.(ComplexF32.(fft(D, 1))), dims=2)) .- σ^2
    @test maximum(abs.(P_wrong .- Ptrue)) / scale > 2.0

    # Tail estimator, with no sigma2 supplied, recovers σ² from the HF band.
    ps_auto = power_spectrum_estimate(D)
    @test isapprox(ps_auto.sigma2, σ^2; rtol=0.15)
    @info "power spectrum debias" sigma2_true=σ^2 sigma2_est=round(ps_auto.sigma2, digits=4)
end

@testset "noiseless recovery is exact up to a circular shift (both methods)" begin
    nt = 64
    R = 200
    s, D, _, _, _ = unit_gain_gather(nt=nt, R=R, noise=0.0, seed=19)

    for method in (:marching, :sync)
        r = recover_mra(D; method=method, sigma2=0f0)
        c = aligned_source_cor(r.x, s)
        # NOT 0.999: kmax bandlimits the reconstruction to k <= nt/2-1, so the
        # recovered signal is a deliberate low-pass of a broadband truth. The
        # measured ceiling is ~0.998 regardless of R or noise, and is a property
        # of the band, not an estimation error. 0.99 sits below that ceiling and
        # far above anything a broken inversion produces.
        @test c > 0.99
        @info "noiseless recovery" method cor=round(c, digits=5) kmax=r.kmax
    end
end

@testset "phase recovery in the reliable band (phase is the target)" begin
    nt = 64
    s, D, _, _, _ = unit_gain_gather(nt=nt, R=400, noise=0.0, seed=23)
    r = recover_mra(D; sigma2=0f0)
    φt = Float32.(angle.(fft(s)))

    err = phase_error(r.φ, φt, r.a)
    # Noiseless, so this is limited only by Float32 and the gauge fit; 0.2 rad
    # is loose enough to be robust yet far tighter than a wrong inversion
    # (which produces errors near the ~1.8 rad of uniformly random phases).
    @test err < 0.2
    @info "phase error, noiseless" rms_rad=round(err, digits=4)

    # The metric must be blind to the global-shift gauge: a deliberately
    # circshifted signal differs from the truth by a pure ramp and must score ~0.
    # Includes lag nt÷2, where the ramp is exactly ±π — the case that breaks a
    # non-circular mean of the fitted differences.
    for lag in (1, 9, nt ÷ 2)
        φ_shift = Float32.(angle.(fft(circshift(s, lag))))
        @test phase_error(φ_shift, φt, abs.(fft(s))) < 1f-3
    end

    # ...but must NOT be blind to genuinely wrong phases.
    rng = MersenneTwister(24)
    @test phase_error(Float32.(2π .* rand(rng, nt)), φt, abs.(fft(s))) > 0.5
end

@testset "noisy recovery: sync refines marching, and actually steps" begin
    nt = 64
    R = 1000
    σ = 0.3f0
    s, D, _, _, _ = unit_gain_gather(nt=nt, R=R, noise=σ, seed=29)

    rm = recover_mra(D; method=:marching, sigma2=σ^2)
    rs = recover_mra(D; method=:sync, sigma2=σ^2)
    cm = aligned_source_cor(rm.x, s)
    cs = aligned_source_cor(rs.x, s)

    # At σ=0.3 with 1000 traces both methods are comfortably above 0.9
    # (measured ~0.97-0.99); 0.9 absorbs seed variation.
    @test cm > 0.9
    # Refinement starts FROM marching, so it should never be materially worse.
    @test cs > cm - 0.02

    # Guard the silent-no-op failure mode: if the Armijo step collapses on entry
    # the refinement returns its initializer unchanged and n_steps stays 0.
    band = power_spectrum_estimate(D; sigma2=σ^2).a[1:rm.kmax+1]
    B = bispectrum_estimate(D; kmax=rm.kmax).B
    sync = invert_phases_sync(B, band; kmax=rm.kmax)
    @test sync.n_steps > 0
    @info "noisy recovery" cor_marching=round(cm, digits=4) cor_sync=round(cs, digits=4) n_steps=sync.n_steps
end

@testset "sample complexity: more traces help at every SNR" begin
    nt = 64
    Rs = (100, 400, 1600)
    seeds = 1:8
    for σ in (0.3f0, 0.8f0)
        # Seed-averaged: single-seed sweeps are genuinely non-monotone (measured
        # 0.55 -> 0.80 -> 0.77 at σ=0.8), so a strict per-seed assertion would
        # flake. The claim being tested is the trend, not a per-draw guarantee.
        cors = map(Rs) do R
            mean(seeds) do sd
                s, D, _, _, _ = unit_gain_gather(nt=nt, R=R, noise=σ, seed=100 + sd)
                aligned_source_cor(recover_mra(D; sigma2=σ^2).x, s)
            end
        end
        @info "sample complexity" sigma=σ R=Rs cor=round.(cors, digits=3)
        @test cors[end] > cors[1]                    # more data helps
        # Near-monotone, allowing a small dip. NOTE: an `issorted` call with a
        # tolerance-based `lt` does NOT express this — that predicate is not a
        # strict weak ordering and misreports genuinely increasing sequences.
        @test all(cors[i] <= cors[i+1] + 0.02 for i in 1:length(cors)-1)
        # Above this many traces the method has clearly locked on, at both SNRs.
        @test cors[end] > (σ < 0.5f0 ? 0.9 : 0.5)
    end
end

@testset "documented limitation: mixed +/-1 polarity breaks the bispectrum" begin
    nt = 64
    R = 2000
    σ = 0.05f0
    rng = MersenneTwister(31)
    s = broadband_wavelet(nt, (0.05, 0.1, 0.2))
    freqs = Float32.(fftfreq(nt))
    grid = ComplexF32.(-im .* 2f0 .* Float32(π) .* freqs)
    Ŝ = fft(s)
    τ = Float32.(rand(rng, R) .* 10 .- 5)
    flip = rand(rng, (-1f0, 1f0), R)

    Du = Matrix{Float32}(undef, nt, R)   # unit gain
    Dm = Matrix{Float32}(undef, nt, R)   # mixed polarity, same shifts and noise
    for i in 1:R
        base = Float32.(real(ifft(Ŝ .* exp.(grid .* τ[i]))))
        n = Float32.(σ .* randn(rng, nt))
        Du[:, i] = base .+ n
        Dm[:, i] = flip[i] .* base .+ n
    end

    ru = recover_mra(Du; sigma2=σ^2)
    rm = recover_mra(Dm; sigma2=σ^2)
    cu = aligned_source_cor(ru.x, s)
    cm = aligned_source_cor(rm.x, s)
    @test cu > 0.9

    # The primary, robust symptom is the BISPECTRUM collapsing: mixed g = ±1
    # enters as g³ = g, so the third-order average cancels toward zero instead
    # of accumulating. Measured max|B| at 1.7-9.5% of the unit-gain value, and
    # the ratio SHRINKS as R grows (more traces = more cancellation), which is
    # the signature of a bias that data cannot fix.
    Bu = maximum(abs.(ru.B))
    Bm = maximum(abs.(rm.B))
    @test Bm < 0.25 * Bu

    # Signed correlation is badly degraded — the recovered polarity is not
    # trustworthy. (Note the waveform SHAPE can still survive up to a global
    # sign: |cor| is often high. So "polarity flips are harmless because they
    # average out" is wrong in the way that matters here — the absolute sign and
    # phase are lost, and it is the sign-blind magnitude that stays healthy.)
    @test cm < 0.8

    # The deceptive part: the power spectrum is degree 2 (g^2 = 1) so it is
    # untouched — `a` looks perfectly healthy while the bispectrum is rubble.
    au = power_spectrum_estimate(Du; sigma2=σ^2).a
    am = power_spectrum_estimate(Dm; sigma2=σ^2).a
    @test cor(au, am) > 0.99
    @info "polarity limitation" cor_unit_gain=round(cu, digits=4) cor_mixed=round(cm, digits=4) maxB_ratio=round(Bm / Bu, digits=4) cor_of_magnitudes=round(cor(au, am), digits=5)
end

@testset "recover_mra_grouped: index-aligned results, sparse groups skipped" begin
    nt = 64
    s1, D1, _, _, _ = unit_gain_gather(nt=nt, R=400, noise=0.2, seed=41)
    s2, D2, _, _, _ = unit_gain_gather(nt=nt, R=300, noise=0.2, seed=42, f0=0.08)
    groups = Matrix{Float32}[D1, D2, D1[:, 1:3]]

    res = (@test_logs (:warn,) match_mode=:any recover_mra_grouped(groups; min_events=10, sigma2=0.04f0))
    @test length(res.groups) == length(groups)     # 1:1 with input, no holes
    @test res.valid == BitVector([1, 1, 0])
    @test all(isfinite, res.groups[1].x)
    @test all(iszero, res.groups[3].x)             # skipped group is zero-filled, right shape
    @test length(res.groups[3].x) == nt

    @test aligned_source_cor(res.groups[1].x, s1) > 0.9
    @test aligned_source_cor(res.groups[2].x, s2) > 0.9
    # Each group recovers ITS OWN source, not a blend of the two.
    @test aligned_source_cor(res.groups[1].x, s1) > aligned_source_cor(res.groups[1].x, s2)
end

# ---------------------------------------------------------------------------
# Benchmark (not pass/fail): MRA vs the CoherentN2N alignment baseline vs raw
# mean, on identical gathers, across SNR. This is the scientific comparison —
# align-then-stack is expected to win at high SNR, and the MRA claim is
# specifically about the low-SNR regime where shift estimation breaks down.
# ---------------------------------------------------------------------------
if RUN_BASELINE
    include(joinpath(@__DIR__, "..", "..", "coherent_n2n", "CoherentN2N_shift.jl"))
    include(joinpath(@__DIR__, "..", "..", "coherent_n2n", "CoherentN2N_gauge.jl"))
    include(joinpath(@__DIR__, "..", "..", "coherent_n2n", "CoherentN2N_polarity.jl"))
    include(joinpath(@__DIR__, "..", "..", "coherent_n2n", "CoherentN2N_denoiser.jl"))
    include(joinpath(@__DIR__, "..", "..", "coherent_n2n", "CoherentN2N_n2n_pairs.jl"))
    include(joinpath(@__DIR__, "..", "..", "coherent_n2n", "CoherentN2N_train_denoiser.jl"))
    include(joinpath(@__DIR__, "..", "..", "coherent_n2n", "CoherentN2N_outer_loop.jl"))
end

let
    nt = 64
    R = 800
    println("\n", "="^78)
    println("Benchmark: recovered-vs-true correlation and phase error (nt=$nt, R=$R)")
    println("="^78)
    println(rpad("sigma", 8), rpad("method", 26), rpad("cor", 10), "phase_err_rad")
    println("-"^78)

    for σ in (0.05f0, 0.3f0, 0.8f0, 1.5f0)
        s, D, _, _, _ = unit_gain_gather(nt=nt, R=R, noise=σ, seed=7)
        φt = Float32.(angle.(fft(s)))
        rows = Tuple{String,Float64,Float64}[]

        for method in (:marching, :sync)
            r = recover_mra(D; method=method, sigma2=σ^2)
            push!(rows, ("bispectrum MRA ($method)", aligned_source_cor(r.x, s),
                         phase_error(r.φ, φt, r.a)))
        end

        raw = vec(mean(D, dims=2))
        push!(rows, ("raw mean", aligned_source_cor(raw, s),
                     phase_error(Float32.(angle.(fft(raw))), φt, abs.(fft(s)))))

        if RUN_BASELINE
            para = CoherentN2N_Para(nt=nt, kernels=[8], filters=[8])
            outer = CoherentN2N_Outer_Para(n_outer_iters=4, stack_type=:l1,
                                           loo_alignment=true)
            bres = run_coherent_n2n_baseline(D, para, outer; rng=MersenneTwister(13))
            bt = Float32.(real(ifft(bres.ŝ)))
            push!(rows, ("CoherentN2N LOO baseline", aligned_source_cor(bt, s),
                         phase_error(Float32.(angle.(bres.ŝ)), φt, abs.(fft(s)))))
        end

        for (name, c, pe) in rows
            println(rpad(string(round(σ, digits=2)), 8), rpad(name, 26),
                    rpad(string(round(c, digits=4)), 10), round(pe, digits=4))
        end
        println("-"^78)
    end
end

@testset "origin conventions: position declared, shape untouched" begin
    nt, R = 128, 600
    t = (0:nt-1) .- nt ÷ 2
    s = Float32.(exp.(-(t ./ 8) .^ 2) .* cos.(2π .* t ./ 12))
    rng = MersenneTwister(77)
    D = Float32.(hcat([circshift(s, rand(rng, 0:nt-1)) .+ 0.3f0 .* randn(rng, Float32, nt)
                       for _ in 1:R]...))

    rg = recover_mra(D; origin=:gauge)
    rc = recover_mra(D; origin=:center)
    rd = recover_mra(D; origin=:data)

    # The declared origin must not alter what was recovered — only where it sits.
    # Best-lag correlation is shift-blind, so all three must agree exactly.
    @test aligned_source_cor(rc.x, s) ≈ aligned_source_cor(rg.x, s) atol = 1f-5
    @test aligned_source_cor(rd.x, s) ≈ aligned_source_cor(rg.x, s) atol = 1f-5

    # :center puts the circular centre of energy at the middle sample.
    w = Float32.(abs2.(rc.x))
    z = sum(w[i+1] * cis(2f0 * Float32(π) * i / nt) for i in 0:nt-1)
    com = mod(angle(z) * nt / (2f0 * Float32(π)), nt)
    @test abs(com - nt ÷ 2) <= 1

    # Each result's spectrum must stay consistent with its own time series,
    # i.e. the rotation was applied to x and X̂ coherently.
    for r in (rg, rc, rd)
        @test Float32.(real(ifft(r.X̂))) ≈ r.x atol = 1f-3
    end

    # The rotation is auditable and exactly reverses.
    @test circshift(rc.x, -rc.origin_lag) ≈ rg.x atol = 1f-4

    # Centring is deterministic: no dependence on the gather, so it cannot be
    # corrupted by a noisier one.
    D2 = Float32.(hcat([circshift(s, rand(rng, 0:nt-1)) .+ 1.2f0 .* randn(rng, Float32, nt)
                        for _ in 1:R]...))
    r2 = recover_mra(D2; origin=:center)
    w2 = Float32.(abs2.(r2.x))
    z2 = sum(w2[i+1] * cis(2f0 * Float32(π) * i / nt) for i in 0:nt-1)
    @test abs(mod(angle(z2) * nt / (2f0 * Float32(π)), nt) - nt ÷ 2) <= 1

    @test_throws ArgumentError recover_mra(D; origin=:bogus)
end

println("All bispectrum-MRA tests passed.")
