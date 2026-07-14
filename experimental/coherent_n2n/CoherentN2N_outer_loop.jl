# Outer alternating loop for CoherentN2N: Block A (denoise / update source
# estimate, shifts fixed) <-> Block B (update shifts, source fixed).
#
# Single-station, multi-earthquake convention (confirmed): the "r" axis is
# earthquakes recorded at one station. No per-event multi-station gather
# assembly is needed — this operates directly on a (nt, R) matrix of
# per-earthquake traces at one station.
#
# Polarity/gain correction is optional — see `use_polarity_gain` on
# CoherentN2N_Outer_Para. CoherentN2N_polarity.jl's estimate_earthquake_gain
# is the standalone, tested implementation this loop calls when enabled.

using FFTW
using Flux
using Random
using Statistics: mean, median

Base.@kwdef struct CoherentN2N_Para
    nt::Int
    enc_kernels::Vector{Int} = [32, 16, 8]
    enc_filters::Vector{Int} = [16, 32, 64]
    dec_kernels::Vector{Int} = [8, 16, 32]
    dec_filters::Vector{Int} = [32, 16, 2]
    seed::Int = 42
    # If true, the denoiser (model + Block A training/inference data) runs on
    # GPU via Flux.gpu; requires `using CUDA, cuDNN` before calling
    # run_coherent_n2n and a GPU selected via CUDA.device!(...). Shift
    # estimation (Block B) always runs on CPU (FFTW, scalar per-trace work)
    # regardless of this flag — only Block A benefits from GPU.
    use_gpu::Bool = false
end

Base.@kwdef struct CoherentN2N_Outer_Para
    n_outer_iters::Int = 5
    denoiser_training::CoherentN2N_Denoiser_Training_Para = CoherentN2N_Denoiser_Training_Para()
    use_polarity_gain::Bool = false
    # Traces whose energy exceeds outlier_energy_factor * median energy are
    # excluded from the reference pick, the coherent stack, and Block B's
    # shift re-estimation (but still receive a returned τ/gain via a
    # best-effort estimate against the final ŝ). Guards against a single
    # corrupted/high-noise trace polluting the whole gather each outer
    # iteration — mirrors the min_traces/SNR-filtering guard pattern used
    # by train_phase_aligner_per_station (Training_PhaseAligner.jl).
    outlier_energy_factor::Float32 = 8.0f0
end

"""
    complex_coherent_stack(X̂::AbstractMatrix{<:Complex}) -> Vector{ComplexF32}

Plain-mean coherent stack across earthquakes, kept complex (no `real(...)`
projection) so source phase is carried through the alternating loop.
Analog of `coherent_stack` (`PhaseAligner_architecture.jl:527-529`), which
drops to real; this sibling does not.
"""
complex_coherent_stack(X̂::AbstractMatrix{<:Complex}) = ComplexF32.(vec(mean(X̂, dims=2)))

"""
    find_energy_outliers(D::AbstractMatrix{Float32}, factor::Float32) -> BitVector

`true` for each column whose energy exceeds `factor * median(energies)`.
"""
function find_energy_outliers(D::AbstractMatrix{Float32}, factor::Float32)
    energies = vec(sum(abs2, D; dims=1))
    med = median(energies)
    return energies .> factor * med
end

"""
    run_coherent_n2n(D::AbstractMatrix{Float32}, para, outer_para) -> (; ŝ, τ, gains, anchor, outliers, history)

Full alternating-loop pipeline on a single station's `(nt, R)` real-valued,
already-preprocessed (whitened/tapered/normalized) earthquake gather `D`.

1. Init: two-stage shift estimate of each earthquake vs. a single
   median-energy reference trace, zero-sum gauge (+ stored anchor), complex
   coherent stack -> ŝ⁰. Traces flagged by `find_energy_outliers` are
   excluded from the reference pick and the stack.
2. Repeat `outer_para.n_outer_iters` times:
   - Block A: shift (and, if `use_polarity_gain`, gain-correct) traces into
     the current frame, train the N2N denoiser, apply it, recombine
     (excluding outliers) into an updated ŝ.
   - Block B: re-estimate shifts (phase-slope two-stage) against updated ŝ,
     re-gauge (accumulate anchor); if `use_polarity_gain`, re-estimate
     per-earthquake complex gains.
3. Return final ŝ, τ (anchor reapplied), gains (all-ones if
   `use_polarity_gain=false`), anchor, the outlier mask, and `history`:
   - `history.delta_s`, `history.delta_tau`: the outer loop's OWN
     convergence — how much ŝ/τ changed between successive outer
     iterations (1 value per outer iteration, over non-outlier traces).
     This is not a trained loss; it's a block-coordinate convergence check.
   - `history.denoiser_loss`: the Noise2Noise denoiser's actual MSE training
     loss curve from each outer iteration's `train_denoiser!` call — a
     `Vector{Vector{Float32}}`, one inner vector per outer iteration, each
     with one entry per training epoch (see `train_denoiser!`'s `h.train_loss`
     in `CoherentN2N_train_denoiser.jl`). Distinct signal from delta_s/delta_tau:
     this measures whether the network is learning within each outer
     iteration's training run, not whether the outer loop is converging.
"""
function run_coherent_n2n(D::AbstractMatrix{Float32}, para::CoherentN2N_Para,
                           outer_para::CoherentN2N_Outer_Para=CoherentN2N_Outer_Para();
                           rng::AbstractRNG=Random.default_rng())
    nt, R = size(D)
    @assert nt == para.nt "D's first dimension must match para.nt"
    freqs = Float32.(fftfreq(nt))
    grid = ComplexF32.(-im .* 2f0 .* Float32(π) .* freqs)
    X̂ = ComplexF32.(fft(D, 1))

    outliers = find_energy_outliers(D, outer_para.outlier_energy_factor)
    good = findall(.!outliers)
    @assert !isempty(good) "All traces flagged as energy outliers; check outlier_energy_factor"

    # ── Initialization ──────────────────────────────────────────────────
    # Use the median-energy single trace (among non-outliers) as the initial
    # reference, not the raw unaligned mean: averaging traces that are both
    # unaligned (large, unknown relative shifts) tends to partially
    # self-cancel, producing a weak, unreliable reference. A single real
    # trace can't cancel against itself.
    energies = vec(sum(abs2, D; dims=1))
    ref_idx = good[argmin(abs.(energies[good] .- median(energies[good])))]
    ref = D[:, ref_idx]
    τ = Float32[estimate_shift_two_stage(ref, D[:, r], freqs;
                                          polarity_agnostic=outer_para.use_polarity_gain) for r in 1:R]
    # Outlier traces don't participate in the gauge fit (their raw, ungauged
    # estimate is unreliable anyway); only non-outlier shifts are centered.
    anchor_total = enforce_zero_sum_gauge!(view(τ, good))

    gains = ones(ComplexF32, R)

    # estimate_shift_two_stage(ref, D[:,r], ...) returns τ such that D[:,r]
    # lags ref by τ (X̂_r = shift_spectrum(X̂_ref, τ, grid)); to align D[:,r]
    # back into ref's frame we must undo that lag, i.e. shift by -τ.
    X̂_aligned = shift_spectrum(X̂, reshape(-τ, 1, R), grid)
    if outer_para.use_polarity_gain
        ref_aligned = X̂_aligned[:, ref_idx]
        for r in good
            gains[r] = estimate_earthquake_gain(X̂_aligned[:, r], ref_aligned)
        end
        gains ./= gains[ref_idx]
        X̂_aligned = X̂_aligned .* reshape(gains, 1, R)
    end
    ŝ = complex_coherent_stack(X̂_aligned[:, good])

    history = (; delta_s=Float32[], delta_tau=Float32[], denoiser_loss=Vector{Float32}[])
    model = build_complex_denoiser(nt; enc_kernels=para.enc_kernels, enc_filters=para.enc_filters,
                                    dec_kernels=para.dec_kernels, dec_filters=para.dec_filters)
    para.use_gpu && (model = Flux.gpu(model))
    to_device = para.use_gpu ? Flux.gpu : identity

    for outer_iter in 1:outer_para.n_outer_iters
        ŝ_prev = copy(ŝ)
        τ_prev = copy(τ)

        # ── Block A: denoise / update source estimate (shifts fixed) ─────
        X̂_aligned = shift_spectrum(X̂, reshape(-τ, 1, R), grid)
        outer_para.use_polarity_gain && (X̂_aligned = X̂_aligned .* reshape(gains, 1, R))
        X̂_aligned_dev = to_device(X̂_aligned[:, good])
        denoiser_h = train_denoiser!(model, X̂_aligned_dev, outer_para.denoiser_training; rng=rng)
        push!(history.denoiser_loss, denoiser_h.train_loss)
        X̂_denoised = Array(model(X̂_aligned_dev))  # back to CPU: rest of the loop is FFTW/CPU-only
        ŝ = complex_coherent_stack(X̂_denoised)

        # ── Block B: update shifts (+ gains) (source fixed) ───────────────
        s_time = real(ifft(ŝ))
        for r in 1:R
            τ[r] = estimate_shift_two_stage(s_time, D[:, r], freqs;
                                             polarity_agnostic=outer_para.use_polarity_gain)
        end
        anchor_total += enforce_zero_sum_gauge!(view(τ, good))

        if outer_para.use_polarity_gain
            X̂_aligned = shift_spectrum(X̂, reshape(-τ, 1, R), grid)
            for r in 1:R
                gains[r] = estimate_earthquake_gain(X̂_aligned[:, r], ŝ)
            end
        end

        push!(history.delta_s, Float32(sqrt(sum(abs2, ŝ .- ŝ_prev))))
        push!(history.delta_tau, Float32(sqrt(sum(abs2, τ[good] .- τ_prev[good]))))
    end

    τ_absolute = copy(τ)
    τ_absolute[good] .+= anchor_total
    return (; ŝ, τ=τ_absolute, gains, anchor=anchor_total, outliers, history)
end
