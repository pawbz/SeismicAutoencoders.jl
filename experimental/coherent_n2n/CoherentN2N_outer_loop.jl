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
        if δ ≤ 0f0
            w .= 1f0            # no spread: fall back to the mean
        else
            @inbounds for c in 1:R
                w[c] = r[c] ≤ δ ? 1f0 : δ / r[c]
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
    τ = Float32[estimate_shift_two_stage(ref, D[:, r], freqs;
                                          polarity_agnostic=outer_para.use_polarity_gain,
                                          max_shift=outer_para.max_shift) for r in 1:R]
    # Mode gauge: pin the dominant cluster of shifts to zero (a skewed tail is
    # allowed). The loop stays convergent under this nonlinear gauge because it
    # re-anchors ŝ to a fixed template each iteration (see run_coherent_n2n's
    # Block A), decoupling the source's time origin from the gauge choice.
    anchor_total = enforce_location_gauge!(view(τ, :); center=mode_kde)
    # ±max_shift as a mode-relative bound (dominant cluster now at 0 after the
    # gauge); consistent with Block B in run_coherent_n2n / _grouped.
    clamp_shifts_to_mode!(view(τ, :), outer_para.max_shift)

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
    run_coherent_n2n(D::AbstractMatrix{Float32}, para, outer_para) -> (; ŝ, τ, gains, anchor, weights, history)

Full alternating-loop pipeline on a single station's `(nt, R)` real-valued,
already-preprocessed (whitened/tapered/normalized) earthquake gather `D`.

1. Init: two-stage shift estimate of each earthquake vs. a single random
   reference trace, mode location gauge (+ stored anchor), robust coherent stack
   -> ŝ⁰ (see `init_group_state` / `robust_complex_stack`). All traces
   participate; there is no energy-outlier exclusion.
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
                           rng::AbstractRNG=Random.default_rng())
    nt, R = size(D)
    @assert nt == para.nt "D's first dimension must match para.nt"
    freqs = Float32.(fftfreq(nt))
    grid = ComplexF32.(-im .* 2f0 .* Float32(π) .* freqs)

    st = init_group_state(D, para, outer_para, freqs, grid; rng=rng)
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

            # ── Re-anchor ŝ to the previous iteration's source (gauge invariance) ─
            # The stack's time origin rides on the current τ frame, so a nonlinear
            # gauge (mode/median) would let it wander between iterations and stall
            # convergence — only the mean is a fixed point otherwise. Instead we pin
            # ŝ's origin to a FIXED template (ŝ_prev) each iteration: cross-correlate
            # the new ŝ to ŝ_prev, remove that lag δ from ŝ, and fold the same δ into
            # τ/anchor so absolute timing (τ + anchor) is invariant. Block B below
            # then always estimates against a source at a stable origin, decoupling
            # convergence from the choice of gauge — the internal gauge can be the
            # mode. (Skipped on iter 1: ŝ_prev is the init stack, already the anchor.)
            if outer_iter > 1
                δ = estimate_shift_two_stage(real(ifft(ŝ_prev)), real(ifft(ŝ)), freqs;
                                              polarity_agnostic=false, max_shift=outer_para.max_shift)
                if δ != 0f0
                    ŝ = vec(shift_spectrum(reshape(ŝ, :, 1), -δ, grid))  # ŝ ← ŝ shifted to match ŝ_prev
                    τ .-= δ                  # keep ŝ and τ in the same frame …
                    anchor_total += δ        # … while holding τ + anchor fixed
                end
            end

            # ── Block B: update shifts (+ gains) (source fixed) ───────────────
            # The `max_shift` passed here is only the coarse-xcorr CYCLE-SKIP guard
            # (search restricted to |lag| ≤ max_shift against the source, an origin
            # that the re-anchor keeps near the gauge frame). The *reported* bound is
            # applied after the gauge, below, so ±max_shift is a RELATIVE bound about
            # the dominant cluster (mode), not about the source origin — see
            # clamp_shifts_to_mode!.
            s_time = real(ifft(ŝ))
            for r in 1:R
                τ[r] = estimate_shift_two_stage(s_time, D[:, r], freqs;
                                                 polarity_agnostic=outer_para.use_polarity_gain,
                                                 max_shift=outer_para.max_shift)
            end
            # Internal gauge is now the MODE (gauge invariance from the re-anchor
            # above makes convergence independent of this choice): the loop already
            # reports dominant-cluster-at-zero shifts, so no separate end re-gauge.
            anchor_total += enforce_location_gauge!(view(τ, :); center=mode_kde)
            # ±max_shift as a bound relative to the mode: the gauge just put the
            # dominant cluster at 0, so clamping the gauged shifts to ±max_shift is a
            # symmetric relative bound about that cluster (irrespective of any
            # mean/skew or source-origin drift).
            clamp_shifts_to_mode!(view(τ, :), outer_para.max_shift)

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

    # The loop is gauge-invariant (ŝ re-anchored to a fixed template each iter),
    # so the internal mode gauge already reports dominant-cluster-at-zero shifts
    # with ŝ consistently framed — no separate end-of-loop re-gauge needed.
    τ_absolute = τ .+ anchor_total
    return (; ŝ, τ=τ_absolute, gains, anchor=anchor_total, weights, history)
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

                    # Re-anchor ŝ_g to the previous iteration's source (gauge invariance
                    # — see run_coherent_n2n's Block A): pins the origin to a fixed
                    # template so the internal mode gauge below doesn't make it wander.
                    if outer_iter > 1
                        δ = estimate_shift_two_stage(real(ifft(ŝ_prev[k])), real(ifft(ŝ[g])), freqs;
                                                      polarity_agnostic=false, max_shift=outer_para.max_shift)
                        if δ != 0f0
                            ŝ[g] = vec(shift_spectrum(reshape(ŝ[g], :, 1), -δ, grid))
                            τ[g] .-= δ
                            anchor[g] += δ
                        end
                    end

                    # max_shift here is the coarse cycle-skip search guard; the
                    # reported ±max_shift bound is applied mode-relative after the
                    # gauge (clamp_shifts_to_mode!), as in the single-station loop.
                    s_time = real(ifft(ŝ[g]))
                    for r in 1:R_g
                        τ[g][r] = estimate_shift_two_stage(s_time, D[:, r], freqs;
                                                           polarity_agnostic=outer_para.use_polarity_gain,
                                                           max_shift=outer_para.max_shift)
                    end
                    # Internal gauge is the MODE (gauge-invariant loop → convergence is
                    # independent of this choice); reports dominant-cluster-at-zero shifts.
                    anchor[g] += enforce_location_gauge!(view(τ[g], :); center=mode_kde)
                    # ±max_shift as a bound relative to the mode (see single-station loop).
                    clamp_shifts_to_mode!(view(τ[g], :), outer_para.max_shift)

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
        # Loop is gauge-invariant with an internal mode gauge (see the per-group
        # re-anchor in Block B) — τ already reports dominant-cluster-at-zero.
        τ_abs = τ[g] .+ anchor[g]
        group_results[g] = (; ŝ=ŝ[g], τ=τ_abs, gains=gains[g], anchor=anchor[g], weights=weights[g])
    end
    return (; groups=group_results, model, valid, history)
end
