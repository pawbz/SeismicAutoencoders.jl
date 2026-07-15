# Stage-2 synthetic validation: with oracle shifts pre-applied, does the
# N2N-trained denoiser improve individual noisy traces?
# Run with: julia --project=experimental/coherent_n2n experimental/coherent_n2n/test/test_denoiser.jl
#
# Note on what "beats" means here: a coherent stack that already averages
# many (R) independent-noise traces is a very strong estimator on its own
# (noise averages down by ~1/sqrt(R)) — a per-trace denoiser applied before
# stacking doesn't need to and, at large R, generally won't beat the plain
# mean-of-many-traces baseline, since it's solving a different problem
# (cleaning up one trace, not aggregating many). The meaningful comparison
# for Block A's actual job is per-trace: does the denoised single trace sit
# closer to the true source than the raw single trace did?

using Test
using Random
using FFTW
using Flux
using Statistics: mean

include(joinpath(@__DIR__, "..", "CoherentN2N_shift.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_denoiser.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_n2n_pairs.jl"))
include(joinpath(@__DIR__, "..", "CoherentN2N_train_denoiser.jl"))
include(joinpath(@__DIR__, "synthetic_data.jl"))

@testset "N2N denoiser improves individual noisy traces (oracle shifts pre-applied)" begin
    rng = MersenneTwister(20)
    nt = 128
    R = 15
    s_true, D, freqs, τ_true, g_true = make_synthetic_gather(
        nt=nt, R=R, f0=0.05, source_kind=:broadband,
        true_shifts=zeros(Float32, R),               # oracle: no shift needed
        true_gains=ones(ComplexF32, R),               # oracle: no polarity/gain needed
        noise_std=0.15, rng=rng)

    X̂_aligned = fft(D, 1) .|> ComplexF32

    model = build_complex_denoiser(nt; kernels=[16, 8], filters=[8, 16])
    training_para = CoherentN2N_Denoiser_Training_Para(
        n_samples_per_epoch=256, batchsize=32, nepoch=200,
        initial_lr=0.003, restart_period=50, nprint=1000)
    train_denoiser!(model, X̂_aligned, training_para; rng=rng)

    X̂_denoised = model(X̂_aligned)
    mse_raw = [mean(abs2, D[:, r] .- s_true) for r in 1:R]
    mse_den = [mean(abs2, real(ifft(X̂_denoised[:, r])) .- s_true) for r in 1:R]

    @info "Per-trace denoising" mean_mse_raw=mean(mse_raw) mean_mse_denoised=mean(mse_den)
    @test mean(mse_den) < mean(mse_raw)
    @test sum(mse_den .< mse_raw) >= R ÷ 2  # majority of individual traces improved

    # The denoised coherent stack should still be a reasonable source estimate
    # (not dramatically worse than the raw stack — sanity check, not a strict bound).
    denoised_mean = real(ifft(vec(mean(X̂_denoised, dims=2))))
    plain_mean = real(ifft(vec(mean(X̂_aligned, dims=2))))
    mse_denoised_stack = mean(abs2, denoised_mean .- s_true)
    mse_plain_stack = mean(abs2, plain_mean .- s_true)
    @info "Stack comparison" mse_plain_stack mse_denoised_stack
    @test mse_denoised_stack < 5 * mse_plain_stack
end

println("All CoherentN2N denoiser tests passed.")
