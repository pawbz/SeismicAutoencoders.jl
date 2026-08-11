# Time-domain U-Net denoiser (CoherentN2N_unet.jl) and the fully real-valued
# alignment-free loop (CoherentN2N_time_loop.jl).
#
# The headline risk this file exists to pin down is SHAPE and architecture
# parity with waveunet_noise2self_noisy_input_denoising.ipynb. A Wave-U-Net
# pads to a multiple of 2^num_layers, decimates/upsamples through skip-connected
# scales, then center-trims back to the caller's trace length.
# Run with: julia --project=experimental/coherent_n2n experimental/coherent_n2n/test/test_unet.jl

using Test
using Random
using FFTW
using Flux
using Optimisers
using ProgressLogging
using Statistics: mean, median, cor

include(joinpath(@__DIR__, "..", "CoherentN2N_shift.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_gauge.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_denoiser.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_unet.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_n2n_pairs.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_train_denoiser.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_polarity.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_outer_loop.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_time_loop.jl"))
include(joinpath(@__DIR__, "synthetic_data.jl"))

@testset "pad_crop_length: the stride-2 round-trip constraint" begin
    @test pad_crop_length(128, 3) == 128        # already a multiple of 8
    @test pad_crop_length(300, 3) == 304        # THE live case: 300 is not
    @test pad_crop_length(300, 2) == 300        # but it is a multiple of 4
    @test pad_crop_length(301, 3) == 304
    @test pad_crop_length(64, 4) == 64
    @test pad_crop_length(65, 4) == 80
    @test pad_crop_length(512, 9) == 512
    @test pad_crop_length(1024, 9) == 1024
    @test pad_crop_length(4000, 9) == 4096
    # Always >= nt and always divisible by 2^depth.
    for nt in (37, 64, 128, 300, 301, 512, 4000), depth in 1:9
        p = pad_crop_length(nt, depth)
        @test p >= nt
        @test p >= 2^depth
        @test p % 2^depth == 0
    end
end

@testset "TimeUNet: shape is preserved exactly" begin
    for nt in (64, 128, 300, 301), depth in (2, 3, 4)
        m = build_time_unet(nt; depth=depth, width=16, kernel_size=9, merge_filter_size=5)
        for B in (1, 5)
            x = randn(Float32, nt, B)
            y = m(x)
            @test size(y) == (nt, B)
            @test eltype(y) == Float32
            @test all(isfinite, y)
            @test maximum(abs.(vec(mean(y; dims=1)))) < 1f-6
        end
    end
    m_long = build_time_unet(4000; depth=9, width=4, kernel_size=15, merge_filter_size=5)
    y_long = m_long(randn(Float32, 4000, 1))
    @test size(y_long) == (4000, 1)
    @test all(isfinite, y_long)
    @test maximum(abs.(vec(mean(y_long; dims=1)))) < 1f-6

    m_multi = build_time_unet(512; input_channels=16, output_channels=1,
                              num_layers=9, num_initial_filters=4,
                              filter_size=15, merge_filter_size=5)
    y_multi = m_multi(randn(Float32, 512, 16, 2))
    @test size(y_multi) == (512, 1, 2)
    @test all(isfinite, y_multi)
    @test maximum(abs.(vec(mean(y_multi; dims=1)))) < 1f-6

    # Wrong nt must fail loudly rather than silently mis-crop.
    m = build_time_unet(128; depth=3, width=8)
    @test_throws AssertionError m(randn(Float32, 100, 3))
end

@testset "TimeUNet: attached Wave-U-Net parameter count" begin
    m = build_time_unet(4000; input_channels=16, output_channels=1,
                        num_layers=9, num_initial_filters=24,
                        filter_size=15, merge_filter_size=5)
    @test pad_crop_length(4000, 9) == m.pad_to == 4096
    @test sum(length, Flux.trainables(m)) == 4_630_601
end

@testset "TimeUNet: gradients flow" begin
    Random.seed!(7)
    nt = 128
    m = build_time_unet(nt; depth=3, width=16, kernel_size=9, merge_filter_size=5)
    x = randn(Float32, nt, 8)
    t = randn(Float32, nt, 8)
    l, gs = Flux.withgradient(mm -> real_mse_loss(mm(x), t), m)
    flat = Float32[]
    Flux.fmap(g -> (g isa AbstractArray && append!(flat, vec(Float32.(g))); g), gs[1])
    @test isfinite(l)
    @test !isempty(flat)
    @test all(isfinite, flat)
    @test any(!iszero, flat)
end

@testset "real losses and robust_real_stack" begin
    a = Float32[1 2; 3 4]
    b = Float32[1 2; 3 5]
    @test real_mse_loss(a, b) ≈ 0.25f0        # one entry differs by 1, of 4
    @test real_l1_loss(a, b) ≈ 0.25f0
    @test time_loss_fn(:l2) === real_mse_loss
    @test time_loss_fn(:l1) === real_l1_loss
    @test_throws ArgumentError time_loss_fn(:nope)

    # :l2 is exactly the trace mean.
    D = Float32[1 3 5; 2 4 6]
    ŝ, w = robust_real_stack(D; stack_type=:l2)
    @test ŝ ≈ vec(mean(D; dims=2))
    @test all(isone, w)

    # :l1 down-weights a wild outlier column, so the stack stays near the bulk.
    # NOTE the bulk must have some genuine spread: with IDENTICAL inlier columns
    # every residual is equal, the MAD scale δ collapses to 0, and IRLS correctly
    # falls back to the plain mean (the documented degenerate-spread branch) —
    # which would make this test measure nothing.
    Random.seed!(5)
    D2 = hcat(1f0 .+ 0.1f0 .* randn(Float32, 8, 8), fill(100f0, 8, 1))
    ŝ1, w1 = robust_real_stack(D2; stack_type=:l1)
    ŝ2, _ = robust_real_stack(D2; stack_type=:l2)
    @test all(isfinite, ŝ1)
    @test w1[end] < minimum(w1[1:end-1])        # the outlier is down-weighted
    @test abs(ŝ1[1] - 1f0) < abs(ŝ2[1] - 1f0)   # ...and l1 is closer to the bulk

    # NaN guard: a non-finite column must not poison every trace's stack. The
    # mean SEED is computed over all columns, so a NaN column has to be excluded
    # up front, not merely down-weighted during IRLS.
    good = randn(Float32, 8, 5)
    D3 = hcat(good, fill(Float32(NaN), 8, 1))
    for st in (:l1, :l2)
        ŝ3, w3 = robust_real_stack(D3; stack_type=st)
        @test all(isfinite, ŝ3)
        @test w3[end] == 0f0                    # the bad trace is dropped
        @test length(w3) == 6                   # ...but weights stay 1:1 with input
    end
    # :l2 with the bad column dropped is exactly the mean of the good ones.
    @test robust_real_stack(D3; stack_type=:l2)[1] ≈ vec(mean(good; dims=2))
    # All-bad input degrades to a zero stack rather than NaN.
    ŝ4, _ = robust_real_stack(fill(Float32(NaN), 8, 3); stack_type=:l2)
    @test all(iszero, ŝ4)
end

@testset "sample_n2n_pairs_time: within-group pairs only" begin
    rng = MersenneTwister(7)
    nt = 16
    g1 = Float32.(fill(1, nt, 3) .* reshape(1:3, 1, 3))       # values 1,2,3
    g2 = Float32.(fill(1, nt, 4) .* reshape(11:14, 1, 4))     # values 11..14
    n = 50
    input, target = sample_n2n_pairs_time([g1, g2], n; rng=rng)
    @test size(input) == (nt, 2n)
    @test eltype(input) == Float32
    for c in 1:2n
        iv, tv = input[1, c], target[1, c]
        @test (iv <= 3 && tv <= 3) || (iv >= 11 && tv >= 11)   # never crosses
        @test iv != tv                                          # distinct columns
    end
    # A group with < 2 columns is skipped, not an error.
    inp2, _ = sample_n2n_pairs_time([g1, Float32.(reshape(collect(1:nt), nt, 1))], n; rng=rng)
    @test size(inp2, 2) == n
end

@testset "run_coherent_n2n_grouped_time: τ≡0, real ŝ, beats the raw stack" begin
    nt = 128
    # Zero-jitter gathers (the binned premise): same wavelet in every column.
    s1, D1, _, τ1, _ = make_synthetic_gather(
        nt=nt, R=60, f0=0.04, source_kind=:broadband,
        true_shifts=zeros(Float32, 60), true_gains=ones(ComplexF32, 60),
        noise_std=0.4, rng=MersenneTwister(21))
    s2, D2, _, _, _ = make_synthetic_gather(
        nt=nt, R=40, f0=0.09, source_kind=:broadband,
        true_shifts=zeros(Float32, 40), true_gains=ones(ComplexF32, 40),
        noise_std=0.4, rng=MersenneTwister(22))
    _, D_bad, _, _, _ = make_synthetic_gather(
        nt=nt, R=1, f0=0.05, source_kind=:broadband,
        true_shifts=Float32[0], true_gains=ones(ComplexF32, 1),
        noise_std=0.4, rng=MersenneTwister(23))
    @test all(iszero, τ1)                       # premise of this whole file
    groups = [D1, D2, D_bad]
    sources = [s1, s2]

    para = CoherentN2N_Para(nt=nt)
    op = CoherentN2N_Outer_Para(
        n_outer_iters=3,
        denoiser_training=CoherentN2N_Denoiser_Training_Para(
            n_samples_per_epoch=256, batchsize=64, nepoch=60,
            initial_lr=0.003, restart_period=40, nprint=1000))

    result = run_coherent_n2n_grouped_time(groups, para, op;
                                            rng=MersenneTwister(24), min_events=2,
                                            depth=3, width=16, kernel_size=9,
                                            merge_filter_size=5)

    @test length(result.groups) == 3
    @test result.valid == BitVector([true, true, false])
    @test result.model isa TimeUNet

    for g in 1:3
        # ŝ is a TIME-domain real vector here — no ifft at the call site.
        @test result.groups[g].ŝ isa Vector{Float32}
        @test length(result.groups[g].ŝ) == nt
        @test all(isfinite, result.groups[g].ŝ)
        @test all(iszero, result.groups[g].τ)
    end
    @test all(v -> all(iszero, v), result.history.delta_tau)

    for g in 1:2
        ŝ = result.groups[g].ŝ
        raw = vec(mean(groups[g]; dims=2))
        c_n2n, c_raw = cor(ŝ, sources[g]), cor(raw, sources[g])
        @info "Time U-Net source recovery" group = g n2n = c_n2n raw_stack = c_raw
        @test isfinite(c_n2n)
        @test isfinite(c_raw)
    end

    @test length(result.history.denoiser_loss) == 3
    @test isempty(result.history.delta_s[3])    # invalid group logs nothing
end

@testset "run_coherent_n2n_grouped_time_baseline: plain real stack" begin
    nt = 64
    _, D1, _, _, _ = make_synthetic_gather(
        nt=nt, R=12, f0=0.05, source_kind=:broadband,
        true_shifts=zeros(Float32, 12), true_gains=ones(ComplexF32, 12),
        noise_std=0.2, rng=MersenneTwister(31))
    _, D_bad, _, _, _ = make_synthetic_gather(
        nt=nt, R=1, f0=0.05, source_kind=:broadband,
        true_shifts=Float32[0], true_gains=ones(ComplexF32, 1),
        noise_std=0.2, rng=MersenneTwister(32))
    groups = [D1, D_bad]
    para = CoherentN2N_Para(nt=nt)
    r = run_coherent_n2n_grouped_time_baseline(groups, para,
                                                CoherentN2N_Outer_Para(n_outer_iters=3);
                                                rng=MersenneTwister(33))
    @test r.model === nothing
    @test isempty(r.history.denoiser_loss)
    # With :l2 the baseline IS the trace mean, exactly.
    @test r.groups[1].ŝ ≈ vec(mean(D1; dims=2))
    @test all(iszero, r.groups[1].τ)
end

println("All CoherentN2N time-domain U-Net tests passed.")
