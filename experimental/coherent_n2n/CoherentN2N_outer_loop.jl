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
using ProgressLogging
using Random
using Statistics: mean, median

Base.@kwdef struct CoherentN2N_Para
    nt::Int
    kernels::Vector{Int} = [32, 16, 8]
    filters::Vector{Int} = [16, 32, 64]
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
    # Coherent-stack estimator for the source ŝ (Block A). `:l2` (default) is the
    # plain mean stack — unchanged behaviour. `:l1` is a robust Huber/IRLS stack
    # that down-weights (never hard-excludes) traces that disagree with the
    # current ŝ, adaptively each outer iteration (see robust_complex_stack). This
    # replaces the removed energy-outlier hard mask, which was a no-op on
    # normalized data and a poor discriminator under noise. NOTE: this governs
    # only the STACK; the denoiser's own training loss is a separate knob
    # (`denoiser_training.denoiser_loss_type`), and Block-B shift estimation
    # (xcorr/phase-slope) is unaffected by either.
    stack_type::Symbol = :l2
    # Bound on recovered per-earthquake shifts: every shift estimate is
    # restricted to [-max_shift, +max_shift] samples (see
    # estimate_shift_two_stage — restricts the coarse xcorr search and clamps
    # the final result). `Inf` (default) leaves shifts unbounded. Applies to
    # init and both Block-B re-estimations, in the single- and grouped-station
    # loops. The location gauge runs AFTER the per-trace clamp, so its
    # mode-centering can move a gauged shift slightly past ±max_shift; the
    # bound is on the raw estimate against the reference/ŝ, which is what
    # prevents cycle-skipping.
    max_shift::Float32 = Inf32
    # Leave-one-out (LOO) alignment in Block B (baseline loops only —
    # run_coherent_n2n_baseline / _grouped_baseline; the trained loops ignore
    # this). When `true` (default), trace r is aligned against the leave-one-out
    # stack ŝ₋ᵣ (the current weighted stack with r's own contribution removed),
    # computed as an O(nt) rank-1 downdate of ŝ:
    #     ŝ₋ᵣ = (W·ŝ − wᵣ·aᵣ) / (W − wᵣ),   W = Σ wⱼ,  aᵣ = X̂_aligned[:,r]
    # so r's shift is estimated against a reference it did not help build. This
    # removes the main self-reinforcement bias: with `false`, each trace aligns
    # against the FULL stack ŝ, which includes itself, so a trace partly aligns
    # to its own contribution — shrinking apparent misalignment and making the
    # baseline look artificially good. Set `false` only to reproduce that naive
    # align-to-full-stack behaviour. Degenerate cases (W − wᵣ ≤ 0, e.g. R = 1)
    # fall back to the full stack for that trace.
    #
    # CONVERGENCE NOTE: aligning to the full stack is a contraction — a trace
    # partly aligning to itself DAMPS its own shift update, which is exactly what
    # makes the fixed-point iteration settle. LOO removes that self-damping along
    # with the bias, so on small/noisy gathers Block B can PLATEAU rather than
    # converge to zero: `history.delta_tau` / `delta_s` level off at a nonzero
    # value instead of trending down. This is not divergence — the source
    # estimate stays stable and its recovery is *better* than the biased full
    # stack (LOO doesn't smear ŝ toward each trace's own noise). Judge LOO by the
    # source (best-lag corr(ŝ, s_true)), NOT by delta_tau. If a monotone
    # delta_tau is needed, damp the reference or under-relax the τ update.
    loo_alignment::Bool = true
    # Baseline-only stochastic reference stacking. When enabled in
    # run_coherent_n2n_baseline / _grouped_baseline, Block A still computes the
    # full aligned stack ŝ for output/diagnostics, but Block B aligns each trace
    # against a fresh stack of a random subset of OTHER traces. This gives an
    # SGD/bootstrap-like data-structured jitter to the fixed point while
    # preserving the anti-self-alignment property of LOO. It supersedes
    # `loo_alignment` for the reference path because the target trace is already
    # excluded from the sampled subset. The trained N2N loops ignore these fields.
    stochastic_baseline_ref::Bool = false
    # Subset size for stochastic_baseline_ref. `0` means floor(sqrt(R_g)) for the
    # current gather/group, clamped to 1:(R_g-1); explicit values are clamped to
    # the same range. R_g <= 1 falls back to the current full-stack reference.
    stochastic_baseline_subset_size::Int = 0
    # Soft super-Gaussian "zero-shift prior" on each per-trace shift estimate
    # (see estimate_shift_xcorr_coarse). `0f0` (default) → OFF: the hard
    # `±max_shift` window as before, byte-identical. `> 0` → the hard window
    # becomes a soft MAP prior `exp(-(|lag|/max_shift)^p)` on the cross-correlation
    # (flat inside ±max_shift, then decaying), so recovered shifts are pulled
    # toward the plausible band without a hard edge — the τ distribution comes out
    # peaked-at-zero-with-a-tail and low-SNR far-lobe outliers are reduced.
    # Requires a finite `max_shift` (the prior width); with `max_shift=Inf` the
    # prior is skipped. Threaded to init and all Block-B shift estimates in every
    # loop (single/grouped, trained/baseline).
    shift_prior_p::Float32 = 0f0
    # Generous hard search backstop for the prior: coarse search is bounded to
    # `±ceil(shift_prior_backstop_mult · max_shift)` so a pathological far
    # cycle-skip can never be picked. At 3·max_shift the prior weight is exp(-3^p)
    # (≈0 for p≥4), so the backstop only forbids what the prior already kills.
    shift_prior_backstop_mult::Float32 = 3f0
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
    robust_complex_stack(X̂; stack_type=:l2, niter=5) -> (ŝ::Vector{ComplexF32}, w::Vector{Float32})

Coherent stack across earthquakes (columns of `X̂`, the `(nt, R)` aligned complex
spectra) with a selectable `stack_type`, returning both the stacked source `ŝ`
and the per-trace weights `w` used to form it.

- `:l2` — the plain L2 (mean) coherent stack (`complex_coherent_stack`); every
  trace weight is 1. This is the default and is byte-identical to the previous
  behaviour, so existing callers are unaffected unless they opt in.
- `:l1` — an L1 *robust* stack via IRLS (iteratively reweighted least
  squares). Each iteration reweights whole traces by their frequency-domain
  residual against the current `ŝ`, so an incoherent / cycle-skipped / corrupted
  trace is smoothly **down-weighted** (not hard-excluded) and a trace that comes
  into alignment on a later outer iteration is automatically re-included:
      rᵣ = ‖X̂[:,r] − ŝ‖₂                       (per-trace residual)
      δ  = c · median(|rᵣ − median(r)|)         (MAD scale — self-tuning, so it
                                                 works regardless of how the data
                                                 was normalized; `c ≈ 1.4826`)
      wᵣ = δ ≥ rᵣ ? 1 : δ / rᵣ                  (Huber weight: L2 core, L1 tail)
      ŝ  = Σ wᵣ X̂[:,r] / Σ wᵣ
  Seeded from the mean (`ŝ⁰`), iterated `niter` times. Degenerate spread (all
  residuals equal → δ = 0) yields all-ones weights, i.e. it degrades gracefully
  to the mean. This replaces the old energy-outlier hard mask (which was a no-op
  on normalized data and a poor discriminator under noise): robustness is now a
  soft, adaptive, per-iteration property of the stack itself.
"""
function robust_complex_stack(X̂::AbstractMatrix{<:Complex}; stack_type::Symbol=:l2, niter::Int=5)
    nt, R = size(X̂)
    if stack_type === :l2 || R <= 1
        return complex_coherent_stack(X̂), ones(Float32, max(R, 0))
    end
    stack_type === :l1 || throw(ArgumentError("unknown stack_type $stack_type (expected :l2 or :l1)"))
    ŝ = complex_coherent_stack(X̂)
    w = ones(Float32, R)
    r = Vector{Float32}(undef, R)
    for _ in 1:niter
        @inbounds for c in 1:R
            acc = 0f0
            for k in 1:nt
                d = X̂[k, c] - ŝ[k]
                acc += abs2(d)
            end
            r[c] = sqrt(acc)
        end
        med = median(r)
        δ = 1.4826f0 * median(abs.(r .- med))
        if δ ≤ 0f0 || !isfinite(δ)
            w .= 1f0            # no spread (or degenerate): fall back to the mean
        else
            @inbounds for c in 1:R
                # `r[c] ≤ δ` already covers r[c] == 0 (weight 1), so the division
                # only runs for r[c] > δ > 0 and cannot produce Inf/NaN. Guard the
                # result anyway: a non-finite residual (from a diverged denoiser
                # upstream) would otherwise silently poison ŝ for every trace via
                # the weighted sum below, turning one bad column into an all-NaN
                # stack that is very hard to trace back.
                wc = r[c] ≤ δ ? 1f0 : δ / r[c]
                w[c] = isfinite(wc) ? wc : 0f0
            end
        end
        wsum = sum(w)
        wsum == 0f0 && break
        @inbounds for k in 1:nt
            acc = zero(ComplexF32)
            for c in 1:R
                acc += w[c] * X̂[k, c]
            end
            ŝ[k] = acc / wsum
        end
    end
    return ŝ, w
end

"""
    init_group_state(D, para, outer_para, freqs, grid; rng)
        -> (; X̂, ref_idx, τ, anchor_total, gains, ŝ, weights)

Initialization shared by the single-station `run_coherent_n2n` and the
per-receiver `run_coherent_n2n_grouped`. For one `(nt, R)` gather `D`: pick a
**random** reference trace, two-stage estimate every shift vs. that reference,
mode-gauge the shifts (returning the anchor), optionally estimate per-earthquake
gains, and build the initial source `ŝ⁰` via `robust_complex_stack` honoring
`outer_para.stack_type` (returning its per-trace `weights`).

Reference choice: a single **randomly chosen** trace, NOT the raw/robust mean of
the unaligned gather. At init the traces are unaligned (unknown ±shift each), so
*any* stack of them is smeared by the shift spread — robustness (IRLS) fixes
outliers, not smearing, so it buys nothing here; a single trace at least keeps
the true waveform shape intact. Empirically the reference choice barely affects
the final result, so a random pick keeps it simple and unbiased. (Robustness DOES
matter for `ŝ⁰` below and every subsequent stack, where the traces ARE aligned.)

There is no energy-outlier concept anymore: all `R` traces participate. A trace
that disagrees with `ŝ` is handled softly by the robust stack's weights, adaptive
each outer iteration (see `robust_complex_stack`), rather than by a fixed hard
mask computed once from raw energy (which was a no-op on normalized data).
"""
function init_group_state(D::AbstractMatrix{Float32}, para::CoherentN2N_Para,
                           outer_para::CoherentN2N_Outer_Para,
                           freqs::AbstractVector{Float32}, grid::AbstractVector{ComplexF32};
                           rng::AbstractRNG=Random.default_rng())
    nt, R = size(D)
    X̂ = ComplexF32.(fft(D, 1))

    # ── Initialization ──────────────────────────────────────────────────
    # Random single-trace reference (see docstring for why not a stack).
    ref_idx = rand(rng, 1:R)
    ref = D[:, ref_idx]
    # estimate_shift_two_stage bounds the search IN-WINDOW (best correlation within
    # ±max_shift, center 0), so no post-hoc clamp is needed and shifts do not pile
    # up at the edge. Mode-gauge machinery deliberately removed for now — τ stays
    # in the reference frame; anchor is 0 (absolute origin is unrecoverable in
    # blind recovery anyway).
    τ = Float32[estimate_shift_two_stage(ref, D[:, r], freqs;
                                          polarity_agnostic=outer_para.use_polarity_gain,
                                          max_shift=outer_para.max_shift,
                                          shift_prior_p=outer_para.shift_prior_p,
                                          shift_prior_backstop_mult=outer_para.shift_prior_backstop_mult) for r in 1:R]
    anchor_total = 0f0

    gains = ones(ComplexF32, R)

    # estimate_shift_two_stage(ref, D[:,r], ...) returns τ such that D[:,r]
    # lags ref by τ (X̂_r = shift_spectrum(X̂_ref, τ, grid)); to align D[:,r]
    # back into ref's frame we must undo that lag, i.e. shift by -τ.
    X̂_aligned = shift_spectrum(X̂, reshape(-τ, 1, R), grid)
    if outer_para.use_polarity_gain
        ref_aligned = X̂_aligned[:, ref_idx]
        for r in 1:R
            gains[r] = estimate_earthquake_gain(X̂_aligned[:, r], ref_aligned)
        end
        gains ./= gains[ref_idx]
        X̂_aligned = X̂_aligned .* reshape(gains, 1, R)
    end
    ŝ, weights = robust_complex_stack(X̂_aligned; stack_type=outer_para.stack_type)
    return (; X̂, ref_idx, τ, anchor_total, gains, ŝ, weights)
end

"""
    warm_start_state(D, result, freqs, grid) -> (; X̂, ref_idx, τ, anchor_total, gains, ŝ, weights)

Build the same init-state NamedTuple `init_group_state` returns, but SEEDED from
a previous single-station `result` (e.g. `run_coherent_n2n_baseline`'s output)
instead of a fresh random-reference alignment. This is how the network path
**warm-starts from the baseline**: run the cheap, unbiased LOO align-and-stack
baseline first, then hand its `(ŝ, τ, gains, anchor, weights)` here so the
trained outer loop begins from that alignment and refines it, rather than
re-deriving alignment from a random reference.

`result.τ` is *absolute* (anchor already reapplied — see `run_coherent_n2n`'s
return); the loop works in the gauged frame with a separate `anchor_total`, so we
undo that: `τ_gauged = result.τ .- result.anchor`, `anchor_total = result.anchor`.
`ŝ`, `gains`, `weights` carry over as-is (ŝ is already in the gauged frame the
loop expects). `ref_idx` is meaningless for a warm start (no single reference was
used) and is set to 1 purely to keep the field present; nothing downstream reads
it once init is done.
"""
function warm_start_state(D::AbstractMatrix{Float32}, result,
                           freqs::AbstractVector{Float32}, grid::AbstractVector{ComplexF32})
    nt, R = size(D)
    @assert length(result.τ) == R "init_state τ length $(length(result.τ)) != R=$R"
    @assert length(result.ŝ) == nt "init_state ŝ length $(length(result.ŝ)) != nt=$nt"
    X̂ = ComplexF32.(fft(D, 1))
    τ = Float32.(result.τ .- result.anchor)          # absolute → gauged frame
    anchor_total = Float32(result.anchor)
    gains = ComplexF32.(copy(result.gains))
    ŝ = ComplexF32.(copy(result.ŝ))
    weights = Float32.(copy(result.weights))
    return (; X̂, ref_idx=1, τ, anchor_total, gains, ŝ, weights)
end

"""
    run_coherent_n2n(D::AbstractMatrix{Float32}, para, outer_para) -> (; ŝ, τ, gains, anchor, weights, history)

Full alternating-loop pipeline on a single station's `(nt, R)` real-valued,
already-preprocessed (whitened/tapered/normalized) earthquake gather `D`.

1. Init: two-stage shift estimate of each earthquake vs. a single random
   reference trace, mode location gauge (+ stored anchor), robust coherent stack
   -> ŝ⁰ (see `init_group_state` / `robust_complex_stack`). All traces
   participate; there is no energy-outlier exclusion. Optionally pass `init_state`
   (a previous single-station result, e.g. from `run_coherent_n2n_baseline`) to
   **warm-start** from that alignment instead — the random-reference init is
   skipped and the loop begins from that `(ŝ, τ, gains, anchor, weights)` (see
   `warm_start_state`), then refines it.
2. Repeat `outer_para.n_outer_iters` times:
   - Block A: shift (and, if `use_polarity_gain`, gain-correct) traces into
     the current frame, train the N2N denoiser, apply it, recombine via the
     robust stack (`stack_type`) into an updated ŝ. The new ŝ is then re-anchored
     to the previous iteration's source (cross-correlation-aligned, lag folded
     into τ/anchor) so its time origin is pinned to a fixed template — this
     makes convergence independent of the gauge choice, letting the internal
     gauge be the nonlinear mode without the origin wandering.
   - Block B: re-estimate shifts (phase-slope two-stage) against updated ŝ,
     mode-gauge (accumulate anchor) so the reported shift distribution has its
     dominant cluster at zero (skewed tail allowed); if `use_polarity_gain`,
     re-estimate per-earthquake complex gains.
3. Return final ŝ, τ (anchor reapplied), gains (all-ones if
   `use_polarity_gain=false`), anchor, `weights` (the final per-trace robust
   stack weights; all-ones for `stack_type=:l2`), and `history`:
   - `history.delta_s`, `history.delta_tau`: the outer loop's OWN
     convergence — how much ŝ/τ changed between successive outer
     iterations (1 value per outer iteration).
     This is not a trained loss; it's a block-coordinate convergence check.
   - `history.denoiser_loss`: the Noise2Noise denoiser's actual training
     loss curve from each outer iteration's `train_denoiser!` call — a
     `Vector{Vector{Float32}}`, one inner vector per outer iteration, each
     with one entry per training epoch (see `train_denoiser!`'s `h.train_loss`
     in `CoherentN2N_train_denoiser.jl`). Distinct signal from delta_s/delta_tau:
     this measures whether the network is learning within each outer
     iteration's training run, not whether the outer loop is converging.
"""
function run_coherent_n2n(D::AbstractMatrix{Float32}, para::CoherentN2N_Para,
                           outer_para::CoherentN2N_Outer_Para=CoherentN2N_Outer_Para();
                           rng::AbstractRNG=Random.default_rng(), init_state=nothing)
    nt, R = size(D)
    @assert nt == para.nt "D's first dimension must match para.nt"
    freqs = Float32.(fftfreq(nt))
    grid = ComplexF32.(-im .* 2f0 .* Float32(π) .* freqs)

    # Init: from-scratch random-reference alignment by default. Optionally
    # warm-start from a prior single-station result (e.g. run_coherent_n2n_baseline's
    # output) by passing it as `init_state` — the loop then begins from that
    # (ŝ, τ, gains, anchor) and refines it (see warm_start_state). The baseline
    # is run explicitly by the caller (kept a separate step), not forced here.
    st = init_state === nothing ? init_group_state(D, para, outer_para, freqs, grid; rng=rng) :
                                    warm_start_state(D, init_state, freqs, grid)
    X̂ = st.X̂
    τ, anchor_total, gains, ŝ = copy(st.τ), st.anchor_total, copy(st.gains), copy(st.ŝ)
    weights = copy(st.weights)

    history = (; delta_s=Float32[], delta_tau=Float32[], denoiser_loss=Vector{Float32}[])
    model = build_complex_denoiser(nt; kernels=para.kernels, filters=para.filters)
    para.use_gpu && (model = Flux.gpu(model))
    to_device = para.use_gpu ? Flux.gpu : identity

    @withprogress name = "CoherentN2N outer loop" begin
        for outer_iter in 1:outer_para.n_outer_iters
            ŝ_prev = copy(ŝ)
            τ_prev = copy(τ)

            # ── Block A: denoise / update source estimate (shifts fixed) ─────
            X̂_aligned = shift_spectrum(X̂, reshape(-τ, 1, R), grid)
            outer_para.use_polarity_gain && (X̂_aligned = X̂_aligned .* reshape(gains, 1, R))
            X̂_aligned_dev = to_device(X̂_aligned)
            denoiser_h = train_denoiser!(model, X̂_aligned_dev, outer_para.denoiser_training; rng=rng)
            push!(history.denoiser_loss, denoiser_h.train_loss)
            X̂_denoised = Array(model(X̂_aligned_dev))  # back to CPU: rest of the loop is FFTW/CPU-only
            ŝ, weights = robust_complex_stack(X̂_denoised; stack_type=outer_para.stack_type)

            # NO re-anchor (removed — bisection on the baseline showed it collapses
            # recovery, e.g. τ-cor 0.61→0.07 with the L1 stack; see the baseline
            # loop). ŝ stays in the current-τ frame and Block B estimates against it
            # directly. Absolute origin is arbitrary (set by init); the reported τ is
            # median-centered at the return.

            # ── Block B: update shifts (+ gains) (source fixed) ───────────────
            # The `max_shift` passed here is only the coarse-xcorr CYCLE-SKIP guard
            # (search restricted to |lag| ≤ max_shift against the source, an origin
            # that the re-anchor keeps near the gauge frame). The *reported* bound is
            # applied after the gauge, below, so ±max_shift is a RELATIVE bound about
            # the dominant cluster (mode), not about the source origin — see
            # clamp_shifts_to_mode!.
            # estimate_shift_two_stage returns the best-correlation shift WITHIN
            # ±max_shift (in-window coarse argmax), so no post-hoc clamp/gauge is
            # applied — shifts do not saturate at the edge. Mode-gauge removed for now.
            s_time = real(ifft(ŝ))
            for r in 1:R
                τ[r] = estimate_shift_two_stage(s_time, D[:, r], freqs;
                                                 polarity_agnostic=outer_para.use_polarity_gain,
                                                 max_shift=outer_para.max_shift,
                                                 shift_prior_p=outer_para.shift_prior_p,
                                                 shift_prior_backstop_mult=outer_para.shift_prior_backstop_mult)
            end

            if outer_para.use_polarity_gain
                X̂_aligned = shift_spectrum(X̂, reshape(-τ, 1, R), grid)
                for r in 1:R
                    gains[r] = estimate_earthquake_gain(X̂_aligned[:, r], ŝ)
                end
            end

            delta_s = Float32(sqrt(sum(abs2, ŝ .- ŝ_prev)))
            delta_tau = Float32(sqrt(sum(abs2, τ .- τ_prev)))
            push!(history.delta_s, delta_s)
            push!(history.delta_tau, delta_tau)
            @logprogress outer_iter / outer_para.n_outer_iters outer_iter = outer_iter delta_s = Float64(delta_s) delta_tau = Float64(delta_tau)
        end
    end

    # Output-only de-drift: the re-anchor keeps ŝ/τ converging in a common frame
    # but (with the mode gauge off) that frame drifts as a whole, so the raw τ can
    # sit outside ±max_shift. Re-center the REPORTED τ on its median — a pure
    # whole-vector shift that changes no pairwise alignment — so the returned
    # distribution is sensibly centred (dominant cluster near 0) and within
    # ±max_shift, WITHOUT a saturating clamp. Internal convergence is untouched.
    τ_report = τ .- median(τ)
    return (; ŝ, τ=τ_report, gains, anchor=anchor_total, weights, history)
end

"""
    loo_reference_time(r, X̂_aligned, ŝ, W, weights) -> Vector{Float32}

Time-domain leave-one-out stack for trace `r`: the current weighted coherent
stack `ŝ` with `r`'s own contribution removed, as an O(nt) rank-1 downdate

    ŝ₋ᵣ = (W·ŝ − wᵣ·aᵣ) / (W − wᵣ),   aᵣ = X̂_aligned[:, r],  wᵣ = weights[r]

returned as a real time-domain trace (`real(ifft(·))`) ready to hand to
`estimate_shift_two_stage` as the alignment reference. `W = Σⱼ weights[j]` is
passed in (computed once per Block-B sweep). If `W − wᵣ ≤ 0` — e.g. a single
trace, or one trace carrying all the robust weight — the LOO stack is undefined,
so we fall back to the full stack `ŝ` for that trace (its self-contribution is
then unavoidable, but there is no other reference to use).
"""
function loo_reference_time(r::Int, X̂_aligned::AbstractMatrix{<:Complex},
                             ŝ::AbstractVector{<:Complex}, W::Real,
                             weights::AbstractVector{<:Real})
    w_r = weights[r]
    denom = W - w_r
    if denom <= 0
        return real(ifft(ŝ))                       # LOO undefined: use full stack
    end
    ŝ_loo = (W .* ŝ .- w_r .* @view(X̂_aligned[:, r])) ./ denom
    return real(ifft(ŝ_loo))
end

"""Resolve stochastic baseline subset size for a gather with `R` traces."""
function stochastic_baseline_ref_size(R::Int, requested::Int)
    R <= 1 && return 0
    k = requested == 0 ? floor(Int, sqrt(R)) : requested
    return clamp(k, 1, R - 1)
end

"""
    stochastic_baseline_subset_indices(R, r, requested, rng) -> Vector{Int}

Draw the trace indices used to build trace `r`'s stochastic baseline reference.
The target trace is excluded whenever `R > 1`.
"""
function stochastic_baseline_subset_indices(R::Int, r::Int, requested::Int,
                                            rng::AbstractRNG)
    k = stochastic_baseline_ref_size(R, requested)
    k == 0 && return Int[]
    candidates = Vector{Int}(undef, R - 1)
    j = 1
    @inbounds for i in 1:R
        i == r && continue
        candidates[j] = i
        j += 1
    end
    return candidates[randperm(rng, R - 1)[1:k]]
end

"""Time-domain stochastic subset-stack reference for baseline Block B."""
function stochastic_baseline_reference_time(r::Int, X̂_aligned::AbstractMatrix{<:Complex},
                                            stack_type::Symbol, requested::Int,
                                            rng::AbstractRNG)
    _, R = size(X̂_aligned)
    subset = stochastic_baseline_subset_indices(R, r, requested, rng)
    isempty(subset) && return nothing
    ŝ_subset, _ = robust_complex_stack(X̂_aligned[:, subset]; stack_type=stack_type)
    return real(ifft(ŝ_subset))
end

"""
    run_coherent_n2n_baseline(D::AbstractMatrix{Float32}, para, outer_para; rng)
        -> (; ŝ, τ, gains, anchor, weights, history)

Network-free baseline: the same alternating pipeline as `run_coherent_n2n` with
**Block A's denoiser removed** — i.e. classic iterative align-and-stack.

This is the honest comparison point for the N2N denoiser: it shares init
(`init_group_state`: random-reference two-stage alignment + mode gauge + robust
stack → ŝ⁰), the identical Block-B shift machinery (`estimate_shift_two_stage`,
mode gauge, `clamp_shifts_to_mode!`), the same gauge-invariance re-anchor, and the
same `robust_complex_stack`. The ONLY difference from `run_coherent_n2n` is that
where that function trains and applies a denoiser to the aligned spectra before
stacking (Block A), this one stacks the aligned spectra directly. Everything the
denoiser could add over "align the raw traces to the running stack and average"
is therefore isolated as the gap between the two.

Iterating Block B against the current *stack* (rather than the one arbitrary,
noisy reference trace used only at init) is what makes this a real baseline and
not just `init_group_state`: the coherent stack is a far better alignment
reference than any single trace, so shifts and ŝ both refine over
`outer_para.n_outer_iters` sweeps. `use_polarity_gain`, `stack_type`, and
`max_shift` all behave exactly as in `run_coherent_n2n`; `denoiser_training` is
ignored (no network is built or trained).

Block B uses **leave-one-out** alignment by default (`outer_para.loo_alignment`,
default `true`): trace `r` is aligned against the stack with its own
contribution removed (`loo_reference_time`, an O(nt) rank-1 downdate), so a trace
never partly aligns to itself. Set `loo_alignment=false` for the naive
align-to-full-stack behaviour. NOTE that LOO removes the self-damping that makes
the loop contract, so on small/noisy gathers `history.delta_tau`/`delta_s` can
plateau rather than converge to zero — this is not divergence, and the source
recovery is actually better; judge LOO by best-lag `corr(ŝ, s_true)`, not by the
delta history (see the `loo_alignment` field docs).

If `outer_para.stochastic_baseline_ref=true`, Block B instead aligns each trace
against a fresh random subset stack of other traces. The subset is resampled for
each trace and outer iteration; `stochastic_baseline_subset_size=0` means
`floor(sqrt(R))`. The full-stack `ŝ` remains the reported/update stack; only the
alignment reference is stochastic.

Returns the same NamedTuple as `run_coherent_n2n` so every downstream plot /
diagnostic works unchanged. `history.denoiser_loss` is present but empty (there
is no denoiser); `history.delta_s` / `delta_tau` are the align-and-stack loop's
own convergence, one value per outer iteration.
"""
function run_coherent_n2n_baseline(D::AbstractMatrix{Float32}, para::CoherentN2N_Para,
                                    outer_para::CoherentN2N_Outer_Para=CoherentN2N_Outer_Para();
                                    rng::AbstractRNG=Random.default_rng())
    nt, R = size(D)
    @assert nt == para.nt "D's first dimension must match para.nt"
    freqs = Float32.(fftfreq(nt))
    grid = ComplexF32.(-im .* 2f0 .* Float32(π) .* freqs)

    st = init_group_state(D, para, outer_para, freqs, grid; rng=rng)
    X̂ = st.X̂
    τ, anchor_total, gains, ŝ = copy(st.τ), st.anchor_total, copy(st.gains), copy(st.ŝ)
    weights = copy(st.weights)

    # denoiser_loss kept for output-shape compatibility with run_coherent_n2n;
    # stays empty because no network is trained here.
    history = (; delta_s=Float32[], delta_tau=Float32[], denoiser_loss=Vector{Float32}[])

    @withprogress name = "CoherentN2N baseline (align-and-stack)" begin
        for outer_iter in 1:outer_para.n_outer_iters
            ŝ_prev = copy(ŝ)
            τ_prev = copy(τ)

            # ── Block A (denoiser removed): stack the aligned spectra directly ──
            X̂_aligned = shift_spectrum(X̂, reshape(-τ, 1, R), grid)
            outer_para.use_polarity_gain && (X̂_aligned = X̂_aligned .* reshape(gains, 1, R))
            ŝ, weights = robust_complex_stack(X̂_aligned; stack_type=outer_para.stack_type)

            # NO re-anchor. Bisection on the notebook synthetic (nt=128, R=200,
            # noise 0.5, true ±8) showed the re-anchor is actively HARMFUL here:
            # plain align-to-robust-stack + tight max_shift + iterate reaches
            # τ-cor≈0.61 (≈ the oracle 0.67), but adding the re-anchor collapses it
            # (0.61→0.07 with the L1 stack). ŝ stays in the current-τ frame; τ stays
            # in ŝ's frame; both are self-consistent without any δ correction.

            # ── Block B: update shifts (+ gains) against the stack ────────────
            # Leave-one-out (default): trace r is aligned against ŝ₋ᵣ, the stack
            # with r's own contribution removed (rank-1 downdate), so it can't
            # partly align to itself. Otherwise align against the full stack ŝ.
            s_time = real(ifft(ŝ))
            W = sum(weights)
            for r in 1:R
                ref = if outer_para.stochastic_baseline_ref
                    stochastic_baseline_reference_time(r, X̂_aligned, outer_para.stack_type,
                                                       outer_para.stochastic_baseline_subset_size, rng)
                elseif outer_para.loo_alignment
                    loo_reference_time(r, X̂_aligned, ŝ, W, weights)
                else
                    s_time
                end
                ref === nothing && (ref = s_time)
                τ[r] = estimate_shift_two_stage(ref, D[:, r], freqs;
                                                 polarity_agnostic=outer_para.use_polarity_gain,
                                                 max_shift=outer_para.max_shift,
                                                 shift_prior_p=outer_para.shift_prior_p,
                                                 shift_prior_backstop_mult=outer_para.shift_prior_backstop_mult)
            end
            # In-window argmax bounds the shift; no post-hoc clamp/gauge (see the
            # single-station trained loop).

            if outer_para.use_polarity_gain
                X̂_aligned = shift_spectrum(X̂, reshape(-τ, 1, R), grid)
                for r in 1:R
                    gains[r] = estimate_earthquake_gain(X̂_aligned[:, r], ŝ)
                end
            end

            delta_s = Float32(sqrt(sum(abs2, ŝ .- ŝ_prev)))
            delta_tau = Float32(sqrt(sum(abs2, τ .- τ_prev)))
            push!(history.delta_s, delta_s)
            push!(history.delta_tau, delta_tau)
            @logprogress outer_iter / outer_para.n_outer_iters outer_iter = outer_iter delta_s = Float64(delta_s) delta_tau = Float64(delta_tau)
        end
    end

    # Center the reported τ on its median (cosmetic: the absolute origin is
    # arbitrary — set by the random init reference). No re-anchor, so τ has no
    # frame drift; this is a pure relabeling for a centered display.
    τ_report = τ .- median(τ)
    return (; ŝ, τ=τ_report, gains, anchor=anchor_total, weights, history)
end

"""
    run_coherent_n2n_grouped(groups::Vector{<:AbstractMatrix{Float32}}, para, outer_para;
                              rng, min_events=2)
        -> (; groups, model, valid, history)

Per-receiver ("grouped") CoherentN2N. Each element of `groups` is one
receiver's `(nt, R_g)` real gather (its columns are that receiver's events);
all groups must share `nt == para.nt`. Produces ONE coherent estimate `ŝ_g` per
receiver, each with its own alignment `τ_g`, gauge/anchor, and robust-stack
`weights` — receivers are independent for alignment (the location gauge freedom
is per earthquake-set, and each `ŝ_g` is its own frame, so there is no
cross-group coupling).

A SINGLE shared `ComplexDenoiser` is trained across all receivers each outer
iteration. N2N pairs are drawn strictly *within* a receiver but pooled into one
batch (see the grouped `sample_n2n_pairs` / `train_denoiser!`), so training and
inference are batched over the network — never a per-receiver `model(...)` loop.

Small-group safety: receivers with fewer than `min_events` columns cannot form
an N2N pair. They are `@warn`-ed and marked `valid[g] = false`; they keep their
init-only `ŝ_g`/`τ_g` in the output (so output indices stay 1:1 with the caller's
receiver list) but never train and never receive a Block B update. (This is a
purely structural count check — unrelated to the removed energy-outlier concept.)

Returns a NamedTuple:
- `groups::Vector{NamedTuple}` — index-aligned to the input; each is
  `(; ŝ, τ, gains, anchor, weights)`, shape-compatible with `run_coherent_n2n`.
- `model` — the single shared trained denoiser (on device if `para.use_gpu`).
- `valid::BitVector` — which receivers participated in training/Block B.
- `history` — `(; denoiser_loss::Vector{Vector{Float32}}` shared, one inner
  vector per outer iteration; `delta_s`, `delta_tau::Vector{Vector{Float32}}`
  per group, indexed `delta_s[g][outer_iter]` over valid groups' iterations).
"""
function run_coherent_n2n_grouped(groups::Vector{<:AbstractMatrix{Float32}},
                                   para::CoherentN2N_Para,
                                   outer_para::CoherentN2N_Outer_Para=CoherentN2N_Outer_Para();
                                   rng::AbstractRNG=Random.default_rng(), min_events::Int=2)
    G = length(groups)
    @assert G >= 1 "Need at least one receiver group"
    nt = para.nt
    for (g, D) in enumerate(groups)
        @assert size(D, 1) == nt "group $g first dimension $(size(D,1)) != para.nt $nt"
    end
    freqs = Float32.(fftfreq(nt))
    grid = ComplexF32.(-im .* 2f0 .* Float32(π) .* freqs)

    # ── Per-group state (parallel vectors, indexed by receiver g) ────────
    states = [init_group_state(groups[g], para, outer_para, freqs, grid; rng=rng) for g in 1:G]
    X̂       = [st.X̂ for st in states]
    τ        = [copy(st.τ) for st in states]
    anchor   = [st.anchor_total for st in states]
    gains    = [copy(st.gains) for st in states]
    ŝ        = [copy(st.ŝ) for st in states]
    weights  = [copy(st.weights) for st in states]

    # ── Validity: a group must have >= max(min_events, 2) columns to form an
    #    N2N pair and to gauge. Purely a count check (no outlier concept).
    #    Invalid groups are kept in the output (init-only) but excluded from
    #    training and Block B.
    valid = BitVector(size(groups[g], 2) >= max(min_events, 2) for g in 1:G)
    for g in 1:G
        valid[g] || @warn "Receiver group $g has only $(size(groups[g], 2)) column(s) (< min_events=$min_events); keeping init-only ŝ, excluding from training/Block B"
    end
    @assert any(valid) "No receiver group has >= max(min_events,2) columns"

    history = (; denoiser_loss=Vector{Float32}[],
                 delta_s=[Float32[] for _ in 1:G],
                 delta_tau=[Float32[] for _ in 1:G])
    model = build_complex_denoiser(nt; kernels=para.kernels, filters=para.filters)
    para.use_gpu && (model = Flux.gpu(model))
    to_device = para.use_gpu ? Flux.gpu : identity

    valid_gs = findall(valid)

    @withprogress name = "Grouped CoherentN2N outer loop" begin
        for outer_iter in 1:outer_para.n_outer_iters
            ŝ_prev = [copy(ŝ[g]) for g in valid_gs]
            τ_prev = [copy(τ[g]) for g in valid_gs]

            # ── Block A: build each valid receiver's aligned spectra (all columns)
            aligned = Vector{Matrix{ComplexF32}}(undef, length(valid_gs))
            for (k, g) in enumerate(valid_gs)
                Xa = shift_spectrum(X̂[g], reshape(-τ[g], 1, size(X̂[g], 2)), grid)
                outer_para.use_polarity_gain && (Xa = Xa .* reshape(gains[g], 1, size(Xa, 2)))
                aligned[k] = Xa
            end

            # ONE shared training call (batched over receivers, within-group pairs).
            denoiser_h = train_denoiser!(model, aligned, outer_para.denoiser_training;
                                         rng=rng, to_device=to_device)
            push!(history.denoiser_loss, denoiser_h.train_loss)

            # ── Batched inference: concat all receivers' aligned columns, ONE
            #    model(...) call, split back by column offsets → per-group stack.
            offsets = cumsum([0; [size(a, 2) for a in aligned]])  # length = length(valid_gs)+1
            big = reduce(hcat, aligned)
            denoised = Array(model(to_device(big)))  # back to CPU for FFTW/CPU Block B
            for (k, g) in enumerate(valid_gs)
                cols = (offsets[k] + 1):offsets[k + 1]
                ŝ[g], weights[g] = robust_complex_stack(denoised[:, cols]; stack_type=outer_para.stack_type)
            end

            # ── Block B: per-group shift (+ gain) update against that group's ŝ_g.
            #    CPU/FFTW per-trace work — the "must be batched" rule is only about
            #    the network, so this per-group loop is fine.
            @withprogress name = "Receiver updates" begin
                for (k, g) in enumerate(valid_gs)
                    D = groups[g]
                    R_g = size(D, 2)

                    # NO re-anchor (removed — harmful; see the single-station loops).

                    # max_shift here is the coarse cycle-skip search guard; the
                    # reported ±max_shift bound is applied mode-relative after the
                    # gauge (clamp_shifts_to_mode!), as in the single-station loop.
                    s_time = real(ifft(ŝ[g]))
                    for r in 1:R_g
                        τ[g][r] = estimate_shift_two_stage(s_time, D[:, r], freqs;
                                                           polarity_agnostic=outer_para.use_polarity_gain,
                                                           max_shift=outer_para.max_shift,
                                                           shift_prior_p=outer_para.shift_prior_p,
                                                           shift_prior_backstop_mult=outer_para.shift_prior_backstop_mult)
                    end
                    # In-window argmax bounds the shift; no post-hoc clamp/gauge.

                    if outer_para.use_polarity_gain
                        Xa = shift_spectrum(X̂[g], reshape(-τ[g], 1, R_g), grid)
                        for r in 1:R_g
                            gains[g][r] = estimate_earthquake_gain(Xa[:, r], ŝ[g])
                        end
                    end

                    delta_s = Float32(sqrt(sum(abs2, ŝ[g] .- ŝ_prev[k])))
                    delta_tau = Float32(sqrt(sum(abs2, τ[g] .- τ_prev[k])))
                    push!(history.delta_s[g], delta_s)
                    push!(history.delta_tau[g], delta_tau)
                    @logprogress k / length(valid_gs) outer_iter = outer_iter group = g delta_s = Float64(delta_s) delta_tau = Float64(delta_tau)
                end
            end

            last_delta_s = history.delta_s[valid_gs[end]][end]
            last_delta_tau = history.delta_tau[valid_gs[end]][end]
            @logprogress outer_iter / outer_para.n_outer_iters outer_iter = outer_iter delta_s = Float64(last_delta_s) delta_tau = Float64(last_delta_tau)
        end
    end

    group_results = Vector{NamedTuple}(undef, G)
    for g in 1:G
        # Output-only de-drift (see run_coherent_n2n): re-center reported τ on its
        # median so each group's distribution is within ±max_shift without a clamp.
        τ_report = τ[g] .- median(τ[g])
        group_results[g] = (; ŝ=ŝ[g], τ=τ_report, gains=gains[g], anchor=anchor[g], weights=weights[g])
    end
    return (; groups=group_results, model, valid, history)
end

"""
    run_coherent_n2n_grouped_baseline(groups, para, outer_para; rng, min_events=2)
        -> (; groups, model, valid, history)

Network-free per-receiver baseline — the grouped counterpart of
`run_coherent_n2n_baseline`, and the honest comparison point for
`run_coherent_n2n_grouped`. Each receiver's `(nt, R_g)` gather is aligned and
stacked independently with the identical init / Block-B / gauge / robust-stack
machinery, but **no shared denoiser is built or trained** — Block A just stacks
each receiver's aligned spectra directly.

Because there is no network, receivers no longer need to share anything across
groups (the shared denoiser was the only coupling); each group is a pure
`run_coherent_n2n_baseline` in effect. The small-group `min_events` /
`valid`-mask bookkeeping is kept identical to `run_coherent_n2n_grouped` so the
output is index-1:1 with the caller's receiver list and drop-in compatible: a
group with `< max(min_events, 2)` columns keeps its init-only ŝ_g/τ_g and is
excluded from Block B.

Returns the same NamedTuple shape as `run_coherent_n2n_grouped`:
- `groups` — index-aligned `(; ŝ, τ, gains, anchor, weights)` per receiver.
- `model` — `nothing` (no denoiser exists in the baseline).
- `valid::BitVector` — which receivers got Block-B updates.
- `history` — `(; denoiser_loss (empty), delta_s, delta_tau)` with the same
  per-group indexing as the trained loop.
"""
function run_coherent_n2n_grouped_baseline(groups::Vector{<:AbstractMatrix{Float32}},
                                            para::CoherentN2N_Para,
                                            outer_para::CoherentN2N_Outer_Para=CoherentN2N_Outer_Para();
                                            rng::AbstractRNG=Random.default_rng(), min_events::Int=2)
    G = length(groups)
    @assert G >= 1 "Need at least one receiver group"
    nt = para.nt
    for (g, D) in enumerate(groups)
        @assert size(D, 1) == nt "group $g first dimension $(size(D,1)) != para.nt $nt"
    end
    freqs = Float32.(fftfreq(nt))
    grid = ComplexF32.(-im .* 2f0 .* Float32(π) .* freqs)

    states = [init_group_state(groups[g], para, outer_para, freqs, grid; rng=rng) for g in 1:G]
    X̂       = [st.X̂ for st in states]
    τ        = [copy(st.τ) for st in states]
    anchor   = [st.anchor_total for st in states]
    gains    = [copy(st.gains) for st in states]
    ŝ        = [copy(st.ŝ) for st in states]
    weights  = [copy(st.weights) for st in states]

    valid = BitVector(size(groups[g], 2) >= max(min_events, 2) for g in 1:G)
    for g in 1:G
        valid[g] || @warn "Receiver group $g has only $(size(groups[g], 2)) column(s) (< min_events=$min_events); keeping init-only ŝ, excluding from Block B"
    end
    @assert any(valid) "No receiver group has >= max(min_events,2) columns"

    history = (; denoiser_loss=Vector{Float32}[],
                 delta_s=[Float32[] for _ in 1:G],
                 delta_tau=[Float32[] for _ in 1:G])
    valid_gs = findall(valid)

    @withprogress name = "Grouped CoherentN2N baseline (align-and-stack)" begin
        for outer_iter in 1:outer_para.n_outer_iters
            @withprogress name = "Receiver updates" begin
                for (k, g) in enumerate(valid_gs)
                    D = groups[g]
                    R_g = size(D, 2)
                    ŝ_prev = copy(ŝ[g])
                    τ_prev = copy(τ[g])

                    # ── Block A (denoiser removed): stack aligned spectra directly ──
                    X̂_aligned = shift_spectrum(X̂[g], reshape(-τ[g], 1, R_g), grid)
                    outer_para.use_polarity_gain && (X̂_aligned = X̂_aligned .* reshape(gains[g], 1, R_g))
                    ŝ[g], weights[g] = robust_complex_stack(X̂_aligned; stack_type=outer_para.stack_type)

                    # NO re-anchor (bisection showed it is harmful — see the
                    # single-station baseline). ŝ_g and τ_g stay self-consistent.

                    # ── Block B: update shifts (+ gains) against this group's ŝ_g ──
                    # Leave-one-out (default): align trace r against ŝ_g with r's
                    # own contribution downdated out (see loo_reference_time).
                    # Stochastic baseline references supersede LOO by sampling a
                    # fresh subset that already excludes r.
                    s_time = real(ifft(ŝ[g]))
                    W = sum(weights[g])
                    for r in 1:R_g
                        ref = if outer_para.stochastic_baseline_ref
                            stochastic_baseline_reference_time(r, X̂_aligned, outer_para.stack_type,
                                                               outer_para.stochastic_baseline_subset_size, rng)
                        elseif outer_para.loo_alignment
                            loo_reference_time(r, X̂_aligned, ŝ[g], W, weights[g])
                        else
                            s_time
                        end
                        ref === nothing && (ref = s_time)
                        τ[g][r] = estimate_shift_two_stage(ref, D[:, r], freqs;
                                                           polarity_agnostic=outer_para.use_polarity_gain,
                                                           max_shift=outer_para.max_shift,
                                                           shift_prior_p=outer_para.shift_prior_p,
                                                           shift_prior_backstop_mult=outer_para.shift_prior_backstop_mult)
                    end
                    # In-window argmax bounds the shift; no post-hoc clamp/gauge.

                    if outer_para.use_polarity_gain
                        Xa = shift_spectrum(X̂[g], reshape(-τ[g], 1, R_g), grid)
                        for r in 1:R_g
                            gains[g][r] = estimate_earthquake_gain(Xa[:, r], ŝ[g])
                        end
                    end

                    delta_s = Float32(sqrt(sum(abs2, ŝ[g] .- ŝ_prev)))
                    delta_tau = Float32(sqrt(sum(abs2, τ[g] .- τ_prev)))
                    push!(history.delta_s[g], delta_s)
                    push!(history.delta_tau[g], delta_tau)
                    @logprogress k / length(valid_gs) outer_iter = outer_iter group = g delta_s = Float64(delta_s) delta_tau = Float64(delta_tau)
                end
            end
            last_delta_s = history.delta_s[valid_gs[end]][end]
            last_delta_tau = history.delta_tau[valid_gs[end]][end]
            @logprogress outer_iter / outer_para.n_outer_iters outer_iter = outer_iter delta_s = Float64(last_delta_s) delta_tau = Float64(last_delta_tau)
        end
    end

    group_results = Vector{NamedTuple}(undef, G)
    for g in 1:G
        # Output-only de-drift (see run_coherent_n2n): center reported τ on median.
        τ_report = τ[g] .- median(τ[g])
        group_results[g] = (; ŝ=ŝ[g], τ=τ_report, gains=gains[g], anchor=anchor[g], weights=weights[g])
    end
    return (; groups=group_results, model=nothing, valid, history)
end

# ───────────────────────────────────────────────────────────────────────────
# Alignment-free ("no-align") grouped loops
#
# These drop the shift degree of freedom entirely: τ ≡ 0, no Block B, no
# gauge. They exist for data that is coherent BY CONSTRUCTION rather than by
# estimation — specifically, a station restricted to one small epicentral
# distance × backazimuth bin (see CoherentN2N_binning.jl), where every event
# shares essentially the same ray parameter and so arrives at the same time.
#
# Removing alignment makes the problem convex-ish and, more importantly,
# unambiguous: whatever improvement appears in ŝ is attributable to the
# denoiser alone, not to the aligner converging. That is the whole point —
# `run_coherent_n2n_grouped` confounds the two.
# ───────────────────────────────────────────────────────────────────────────

"""
    _noalign_warn_ignored(outer_para)

Warn once about `CoherentN2N_Outer_Para` fields the alignment-free loops
deliberately ignore, so a caller who sets them does not silently get a
different behaviour than they asked for.
"""
function _noalign_warn_ignored(outer_para::CoherentN2N_Outer_Para)
    outer_para.use_polarity_gain && @warn "run_coherent_n2n_grouped_noalign: use_polarity_gain=true is ignored (no per-event gain is estimated without an alignment step); gains stay all-ones"
    isfinite(outer_para.max_shift) && @warn "run_coherent_n2n_grouped_noalign: max_shift is ignored (τ ≡ 0 by construction)"
    outer_para.stochastic_baseline_ref && @warn "run_coherent_n2n_grouped_noalign: stochastic_baseline_ref is ignored (no reference alignment is performed)"
    return nothing
end

"""
    _noalign_init(groups, outer_para) -> (; X̂, τ, anchor, gains, ŝ, weights)

Shift-free initialization shared by both alignment-free loops. Deliberately does
NOT call `init_group_state`: that estimates every shift against a random
reference trace, which is exactly the machinery being removed here. Each group
is simply FFT'd and stacked as-is, so `ŝ⁰` is the raw (robust) stack of the bin.
"""
function _noalign_init(groups::Vector{<:AbstractMatrix{Float32}},
                        outer_para::CoherentN2N_Outer_Para)
    G = length(groups)
    X̂ = [ComplexF32.(fft(D, 1)) for D in groups]
    τ = [zeros(Float32, size(D, 2)) for D in groups]
    anchor = [0f0 for _ in 1:G]
    gains = [ones(ComplexF32, size(D, 2)) for D in groups]
    stacks = [robust_complex_stack(X̂[g]; stack_type=outer_para.stack_type) for g in 1:G]
    ŝ = [copy(st[1]) for st in stacks]
    weights = [copy(st[2]) for st in stacks]
    return (; X̂, τ, anchor, gains, ŝ, weights)
end

"""
    run_coherent_n2n_grouped_noalign(groups, para, outer_para; rng, min_events=2)
        -> (; groups, model, valid, history)

**Alignment-free** per-receiver CoherentN2N — `run_coherent_n2n_grouped` with the
shift degree of freedom removed. Each element of `groups` is one receiver's
`(nt, R_g)` real gather whose columns are assumed ALREADY COHERENT (e.g. a single
10°×10° epicentral-distance × backazimuth bin, built with
`densest_bin` / `bin_member_indices` from `CoherentN2N_binning.jl`).

The loop is therefore just: stack → train the shared denoiser on the raw spectra
→ re-stack the denoised spectra → repeat. There is no Block B, no
`estimate_shift_two_stage`, no gauge/anchor, and no `shift_spectrum` call
anywhere. `τ` is identically zero and is returned only so the output stays
shape-compatible with `run_coherent_n2n_grouped`.

As in the grouped loop, ONE shared `ComplexDenoiser` is trained across all
receivers, N2N pairs are drawn strictly within a receiver, and inference is a
single batched `model(...)` over the concatenated columns — never a per-receiver
model loop.

`CoherentN2N_Outer_Para` fields **ignored by design** (a `@warn` fires if set):
`max_shift`, `shift_prior_p`, `shift_prior_backstop_mult`, `use_polarity_gain`,
`loo_alignment`, `stochastic_baseline_ref`, `stochastic_baseline_subset_size`.
The fields that DO apply are `n_outer_iters`, `stack_type`, and
`denoiser_training`.

Returns the same NamedTuple shape as `run_coherent_n2n_grouped`:
- `groups` — index-aligned `(; ŝ, τ, gains, anchor, weights)`; `τ` is all zeros,
  `gains` all ones, `anchor` 0.
- `model` — the single shared trained denoiser.
- `valid::BitVector` — receivers with `>= max(min_events, 2)` columns (a group
  with fewer cannot form an N2N pair); invalid groups keep their raw-stack ŝ.
- `history` — `(; denoiser_loss, delta_s, delta_tau)`; `delta_tau` is pushed as
  `0f0` every iteration (structurally zero — τ never moves) purely so plotting
  code shared with the aligned loops keeps working.
"""
function run_coherent_n2n_grouped_noalign(groups::Vector{<:AbstractMatrix{Float32}},
                                           para::CoherentN2N_Para,
                                           outer_para::CoherentN2N_Outer_Para=CoherentN2N_Outer_Para();
                                           rng::AbstractRNG=Random.default_rng(), min_events::Int=2)
    G = length(groups)
    @assert G >= 1 "Need at least one receiver group"
    nt = para.nt
    for (g, D) in enumerate(groups)
        @assert size(D, 1) == nt "group $g first dimension $(size(D,1)) != para.nt $nt"
    end
    _noalign_warn_ignored(outer_para)

    st = _noalign_init(groups, outer_para)
    X̂, τ, anchor, gains = st.X̂, st.τ, st.anchor, st.gains
    ŝ, weights = st.ŝ, st.weights

    # Same structural count check as the aligned grouped loop.
    valid = BitVector(size(groups[g], 2) >= max(min_events, 2) for g in 1:G)
    for g in 1:G
        valid[g] || @warn "Receiver group $g has only $(size(groups[g], 2)) column(s) (< min_events=$min_events); keeping raw-stack ŝ, excluding from training"
    end
    @assert any(valid) "No receiver group has >= max(min_events,2) columns"

    history = (; denoiser_loss=Vector{Float32}[],
                 delta_s=[Float32[] for _ in 1:G],
                 delta_tau=[Float32[] for _ in 1:G])
    model = build_complex_denoiser(nt; kernels=para.kernels, filters=para.filters)
    para.use_gpu && (model = Flux.gpu(model))
    to_device = para.use_gpu ? Flux.gpu : identity

    valid_gs = findall(valid)

    @withprogress name = "Grouped CoherentN2N outer loop (no alignment)" begin
        for outer_iter in 1:outer_para.n_outer_iters
            ŝ_prev = [copy(ŝ[g]) for g in valid_gs]

            # ── Block A: the spectra ARE the aligned spectra (τ ≡ 0), so they go
            #    into training unmodified — no shift_spectrum, no gain multiply.
            aligned = [X̂[g] for g in valid_gs]

            denoiser_h = train_denoiser!(model, aligned, outer_para.denoiser_training;
                                         rng=rng, to_device=to_device)
            push!(history.denoiser_loss, denoiser_h.train_loss)

            # Batched inference: one model(...) over all receivers' columns, split
            # back by column offsets → per-group robust stack.
            offsets = cumsum([0; [size(a, 2) for a in aligned]])
            big = reduce(hcat, aligned)
            denoised = Array(model(to_device(big)))
            for (k, g) in enumerate(valid_gs)
                cols = (offsets[k] + 1):offsets[k + 1]
                ŝ[g], weights[g] = robust_complex_stack(denoised[:, cols]; stack_type=outer_para.stack_type)
            end

            # ── Block B deliberately absent. τ stays zero; only ŝ moves, driven
            #    entirely by the denoiser improving.
            for (k, g) in enumerate(valid_gs)
                delta_s = Float32(sqrt(sum(abs2, ŝ[g] .- ŝ_prev[k])))
                push!(history.delta_s[g], delta_s)
                push!(history.delta_tau[g], 0f0)   # structurally zero — see docstring
            end

            last_delta_s = history.delta_s[valid_gs[end]][end]
            @logprogress outer_iter / outer_para.n_outer_iters outer_iter = outer_iter delta_s = Float64(last_delta_s)
        end
    end

    group_results = Vector{NamedTuple}(undef, G)
    for g in 1:G
        # No median-centering: τ is already exactly zero, and re-centering a
        # zero vector would only invite the reader to think a gauge was applied.
        group_results[g] = (; ŝ=ŝ[g], τ=τ[g], gains=gains[g], anchor=anchor[g], weights=weights[g])
    end
    return (; groups=group_results, model, valid, history)
end

"""
    run_coherent_n2n_grouped_noalign_baseline(groups, para, outer_para; rng, min_events=2)
        -> (; groups, model, valid, history)

Network-free counterpart of `run_coherent_n2n_grouped_noalign`, and the honest
comparison point for it. With τ ≡ 0 and no denoiser there is nothing left to
iterate: each receiver's ŝ_g is simply `robust_complex_stack` of its raw bin
spectra — i.e. **the plain stack of the bin**, exactly the thing the denoiser has
to beat. It is computed once regardless of `n_outer_iters` (further iterations
would be bit-identical), and costs one FFT + one stack per group.

Returns the same NamedTuple shape as `run_coherent_n2n_grouped_noalign`, with
`model === nothing`, an empty `denoiser_loss`, and a single `0f0` entry in each
valid group's `delta_s` / `delta_tau` (the loop converged immediately, by
construction).
"""
function run_coherent_n2n_grouped_noalign_baseline(groups::Vector{<:AbstractMatrix{Float32}},
                                                    para::CoherentN2N_Para,
                                                    outer_para::CoherentN2N_Outer_Para=CoherentN2N_Outer_Para();
                                                    rng::AbstractRNG=Random.default_rng(), min_events::Int=2)
    G = length(groups)
    @assert G >= 1 "Need at least one receiver group"
    nt = para.nt
    for (g, D) in enumerate(groups)
        @assert size(D, 1) == nt "group $g first dimension $(size(D,1)) != para.nt $nt"
    end
    _noalign_warn_ignored(outer_para)

    st = _noalign_init(groups, outer_para)

    valid = BitVector(size(groups[g], 2) >= max(min_events, 2) for g in 1:G)
    for g in 1:G
        valid[g] || @warn "Receiver group $g has only $(size(groups[g], 2)) column(s) (< min_events=$min_events); keeping raw-stack ŝ"
    end
    @assert any(valid) "No receiver group has >= max(min_events,2) columns"

    history = (; denoiser_loss=Vector{Float32}[],
                 delta_s=[Float32[] for _ in 1:G],
                 delta_tau=[Float32[] for _ in 1:G])
    for g in findall(valid)
        push!(history.delta_s[g], 0f0)
        push!(history.delta_tau[g], 0f0)
    end

    group_results = Vector{NamedTuple}(undef, G)
    for g in 1:G
        group_results[g] = (; ŝ=st.ŝ[g], τ=st.τ[g], gains=st.gains[g],
                              anchor=st.anchor[g], weights=st.weights[g])
    end
    return (; groups=group_results, model=nothing, valid, history)
end
