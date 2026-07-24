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
    # Traces whose energy exceeds outlier_energy_factor * median energy are
    # excluded from the reference pick, the coherent stack, and Block B's
    # shift re-estimation (but still receive a returned τ/gain via a
    # best-effort estimate against the final ŝ). Guards against a single
    # corrupted/high-noise trace polluting the whole gather each outer
    # iteration — mirrors the min_traces/SNR-filtering guard pattern used
    # by train_phase_aligner_per_station (Training_PhaseAligner.jl).
    outlier_energy_factor::Float32 = 8.0f0
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
    find_energy_outliers(D::AbstractMatrix{Float32}, factor::Float32) -> BitVector

`true` for each column whose energy exceeds `factor * median(energies)`.
"""
function find_energy_outliers(D::AbstractMatrix{Float32}, factor::Float32)
    energies = vec(sum(abs2, D; dims=1))
    med = median(energies)
    return energies .> factor * med
end

"""
    init_group_state(D, para, outer_para, freqs, grid; rng)
        -> (; X̂, outliers, good, ref_idx, τ, anchor_total, gains, ŝ)

Initialization shared by the single-station `run_coherent_n2n` and the
per-receiver `run_coherent_n2n_grouped`. For one `(nt, R)` gather `D`: flag
energy outliers, pick the median-energy non-outlier reference, two-stage
estimate every shift vs. that reference, mode-gauge the non-outlier shifts
(returning the anchor), optionally estimate per-earthquake gains, and build the
initial complex coherent stack `ŝ⁰` over non-outlier columns. Behaviorally
identical to the init block it was extracted from — no policy change.
"""
function init_group_state(D::AbstractMatrix{Float32}, para::CoherentN2N_Para,
                           outer_para::CoherentN2N_Outer_Para,
                           freqs::AbstractVector{Float32}, grid::AbstractVector{ComplexF32};
                           rng::AbstractRNG=Random.default_rng())
    nt, R = size(D)
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
                                          polarity_agnostic=outer_para.use_polarity_gain,
                                          max_shift=outer_para.max_shift) for r in 1:R]
    # Outlier traces don't participate in the gauge fit (their raw, ungauged
    # estimate is unreliable anyway); only non-outlier shifts are centered.
    # Mode gauge: pin the dominant cluster of shifts to zero (a skewed tail is
    # allowed). The loop stays convergent under this nonlinear gauge because it
    # re-anchors ŝ to a fixed template each iteration (see run_coherent_n2n's
    # Block A), decoupling the source's time origin from the gauge choice.
    anchor_total = enforce_location_gauge!(view(τ, good); center=mode_kde)

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
    return (; X̂, outliers, good, ref_idx, τ, anchor_total, gains, ŝ)
end

"""
    run_coherent_n2n(D::AbstractMatrix{Float32}, para, outer_para) -> (; ŝ, τ, gains, anchor, outliers, history)

Full alternating-loop pipeline on a single station's `(nt, R)` real-valued,
already-preprocessed (whitened/tapered/normalized) earthquake gather `D`.

1. Init: two-stage shift estimate of each earthquake vs. a single
   median-energy reference trace, mode location gauge (+ stored anchor), complex
   coherent stack -> ŝ⁰. Traces flagged by `find_energy_outliers` are
   excluded from the reference pick and the stack.
2. Repeat `outer_para.n_outer_iters` times:
   - Block A: shift (and, if `use_polarity_gain`, gain-correct) traces into
     the current frame, train the N2N denoiser, apply it, recombine
     (excluding outliers) into an updated ŝ. The new ŝ is then re-anchored to
     the previous iteration's source (cross-correlation-aligned, lag folded
     into τ/anchor) so its time origin is pinned to a fixed template — this
     makes convergence independent of the gauge choice, letting the internal
     gauge be the nonlinear mode without the origin wandering.
   - Block B: re-estimate shifts (phase-slope two-stage) against updated ŝ,
     mode-gauge (accumulate anchor) so the reported shift distribution has its
     dominant cluster at zero (skewed tail allowed); if `use_polarity_gain`,
     re-estimate per-earthquake complex gains.
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

    st = init_group_state(D, para, outer_para, freqs, grid; rng=rng)
    X̂, outliers, good = st.X̂, st.outliers, st.good
    τ, anchor_total, gains, ŝ = copy(st.τ), st.anchor_total, copy(st.gains), copy(st.ŝ)

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
            X̂_aligned_dev = to_device(X̂_aligned[:, good])
            denoiser_h = train_denoiser!(model, X̂_aligned_dev, outer_para.denoiser_training; rng=rng)
            push!(history.denoiser_loss, denoiser_h.train_loss)
            X̂_denoised = Array(model(X̂_aligned_dev))  # back to CPU: rest of the loop is FFTW/CPU-only
            ŝ = complex_coherent_stack(X̂_denoised)

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
                    τ[good] .-= δ            # keep ŝ and τ in the same frame …
                    anchor_total += δ        # … while holding τ + anchor fixed
                end
            end

            # ── Block B: update shifts (+ gains) (source fixed) ───────────────
            s_time = real(ifft(ŝ))
            for r in 1:R
                τ[r] = estimate_shift_two_stage(s_time, D[:, r], freqs;
                                                 polarity_agnostic=outer_para.use_polarity_gain,
                                                 max_shift=outer_para.max_shift)
            end
            # Internal gauge is now the MODE (gauge invariance from the re-anchor
            # above makes convergence independent of this choice): the loop already
            # reports dominant-cluster-at-zero shifts, so no separate end re-gauge.
            anchor_total += enforce_location_gauge!(view(τ, good); center=mode_kde)

            if outer_para.use_polarity_gain
                X̂_aligned = shift_spectrum(X̂, reshape(-τ, 1, R), grid)
                for r in 1:R
                    gains[r] = estimate_earthquake_gain(X̂_aligned[:, r], ŝ)
                end
            end

            delta_s = Float32(sqrt(sum(abs2, ŝ .- ŝ_prev)))
            delta_tau = Float32(sqrt(sum(abs2, τ[good] .- τ_prev[good])))
            push!(history.delta_s, delta_s)
            push!(history.delta_tau, delta_tau)
            @logprogress outer_iter / outer_para.n_outer_iters outer_iter = outer_iter delta_s = Float64(delta_s) delta_tau = Float64(delta_tau)
        end
    end

    # The loop is gauge-invariant (ŝ re-anchored to a fixed template each iter),
    # so the internal mode gauge already reports dominant-cluster-at-zero shifts
    # with ŝ consistently framed — no separate end-of-loop re-gauge needed.
    τ_absolute = copy(τ)
    τ_absolute[good] .+= anchor_total
    return (; ŝ, τ=τ_absolute, gains, anchor=anchor_total, outliers, history)
end

"""
    run_coherent_n2n_grouped(groups::Vector{<:AbstractMatrix{Float32}}, para, outer_para;
                              rng, min_events=2)
        -> (; groups, model, valid, history)

Per-receiver ("grouped") CoherentN2N. Each element of `groups` is one
receiver's `(nt, R_g)` real gather (its columns are that receiver's events);
all groups must share `nt == para.nt`. Produces ONE coherent estimate `ŝ_g` per
receiver, each with its own alignment `τ_g`, gauge/anchor, and outlier mask —
receivers are independent for alignment (the location gauge freedom is per
earthquake-set, and each `ŝ_g` is its own frame, so there is no cross-group
coupling).

A SINGLE shared `ComplexDenoiser` is trained across all receivers each outer
iteration. N2N pairs are drawn strictly *within* a receiver but pooled into one
batch (see the grouped `sample_n2n_pairs` / `train_denoiser!`), so training and
inference are batched over the network — never a per-receiver `model(...)` loop.

Small-group safety: receivers with fewer than `min_events` non-outlier columns
cannot form an N2N pair. They are `@warn`-ed and marked `valid[g] = false`; they
keep their init-only `ŝ_g`/`τ_g` in the output (so output indices stay 1:1 with
the caller's receiver list) but never train and never receive a Block B update.

Returns a NamedTuple:
- `groups::Vector{NamedTuple}` — index-aligned to the input; each is
  `(; ŝ, τ, gains, anchor, outliers)`, shape-compatible with `run_coherent_n2n`.
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
    outliers = [st.outliers for st in states]
    good     = [st.good for st in states]
    τ        = [copy(st.τ) for st in states]
    anchor   = [st.anchor_total for st in states]
    gains    = [copy(st.gains) for st in states]
    ŝ        = [copy(st.ŝ) for st in states]

    # ── Validity: a group must have >= min_events (and >= 2) non-outlier
    #    columns to form an N2N pair and to gauge. Invalid groups are kept in
    #    the output (init-only) but excluded from training and Block B.
    valid = BitVector(length(good[g]) >= max(min_events, 2) for g in 1:G)
    for g in 1:G
        valid[g] || @warn "Receiver group $g has only $(length(good[g])) non-outlier column(s) (< min_events=$min_events); keeping init-only ŝ, excluding from training/Block B" R_g=size(groups[g], 2)
    end
    @assert any(valid) "No receiver group has >= max(min_events,2) non-outlier columns"

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

            # ── Block A: build each valid receiver's aligned good-column spectra
            aligned = Vector{Matrix{ComplexF32}}(undef, length(valid_gs))
            for (k, g) in enumerate(valid_gs)
                Xa = shift_spectrum(X̂[g], reshape(-τ[g], 1, size(X̂[g], 2)), grid)
                outer_para.use_polarity_gain && (Xa = Xa .* reshape(gains[g], 1, size(Xa, 2)))
                aligned[k] = Xa[:, good[g]]
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
                ŝ[g] = complex_coherent_stack(denoised[:, cols])
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
                            τ[g][good[g]] .-= δ
                            anchor[g] += δ
                        end
                    end

                    s_time = real(ifft(ŝ[g]))
                    for r in 1:R_g
                        τ[g][r] = estimate_shift_two_stage(s_time, D[:, r], freqs;
                                                           polarity_agnostic=outer_para.use_polarity_gain,
                                                           max_shift=outer_para.max_shift)
                    end
                    # Internal gauge is the MODE (gauge-invariant loop → convergence is
                    # independent of this choice); reports dominant-cluster-at-zero shifts.
                    anchor[g] += enforce_location_gauge!(view(τ[g], good[g]); center=mode_kde)

                    if outer_para.use_polarity_gain
                        Xa = shift_spectrum(X̂[g], reshape(-τ[g], 1, R_g), grid)
                        for r in 1:R_g
                            gains[g][r] = estimate_earthquake_gain(Xa[:, r], ŝ[g])
                        end
                    end

                    delta_s = Float32(sqrt(sum(abs2, ŝ[g] .- ŝ_prev[k])))
                    delta_tau = Float32(sqrt(sum(abs2, τ[g][good[g]] .- τ_prev[k][good[g]])))
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
        τ_abs = copy(τ[g])
        τ_abs[good[g]] .+= anchor[g]
        group_results[g] = (; ŝ=ŝ[g], τ=τ_abs, gains=gains[g], anchor=anchor[g], outliers=outliers[g])
    end
    return (; groups=group_results, model, valid, history)
end
