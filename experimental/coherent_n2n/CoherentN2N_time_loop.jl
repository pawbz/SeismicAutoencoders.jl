# Alignment-free, FULLY TIME-DOMAIN CoherentN2N: real waveforms end to end.
#
# This is the real-valued sibling of run_coherent_n2n_grouped_noalign. That loop
# still round-trips through the FFT (fft -> train complex denoiser -> stack ->
# ifft), but with τ ≡ 0 the FFT is INCIDENTAL: Block A is literally
# `aligned = X̂[g]`, unchanged. The frequency domain only ever existed to serve
# the shift machinery, which the no-align loop already removed — so here we drop
# it entirely and denoise the waveforms directly.
#
# Everything is a parallel real-valued implementation rather than a relaxation of
# the complex path's types: `train_denoiser!`, `sample_n2n_pairs`,
# `complex_mse_loss` and `robust_complex_stack` are all hard-typed to
# ComplexF32/<:Complex, and the aligned notebook depends on them. Duplicating the
# ~40 lines of loop skeleton is cheaper than risking that path.
#
# The one substantive behavioural difference, and the reason for doing this: the
# LOSS is now on the waveform. A constant offset across frequency bins — which a
# spectral MSE barely penalizes, and which the complex denoiser demonstrably
# learned (78% of trained ŝ on a real bin, a ~375x first-sample spike) — is an
# ordinary, fully-priced error here.

using Statistics: mean, median
using Random: AbstractRNG
using FFTW: fft, ifft   # only for the caller's convenience helpers, not the loop

"""
    sample_n2n_pairs_time(groups::Vector{<:AbstractMatrix{Float32}}, n_samples_per_group; rng)
        -> (input, target)

Real-valued grouped N2N pair sampling — the Float32 transcription of the grouped
`sample_n2n_pairs` (CoherentN2N_n2n_pairs.jl). Each element of `groups` is one
receiver's `(nt, R_g)` real gather. Draws `n_samples_per_group` within-group
distinct `(a, b)` column pairs per receiver and hcats them into two `(nt, Σ)`
Float32 matrices.

Pairs never straddle receivers; the pairing rule itself is shared with the
complex path via `draw_distinct_pair_indices`, so there is exactly one place
where "two distinct looks" is defined.
"""
function sample_n2n_pairs_time(groups::Vector{<:AbstractMatrix{Float32}},
                                n_samples_per_group::Int;
                                rng::AbstractRNG=Random.default_rng())
    inputs = Vector{Matrix{Float32}}()
    targets = Vector{Matrix{Float32}}()
    for D_g in groups
        R_g = size(D_g, 2)
        R_g >= 2 || continue
        a_idx, b_idx = draw_distinct_pair_indices(R_g, n_samples_per_group, rng)
        push!(inputs, D_g[:, a_idx])
        push!(targets, D_g[:, b_idx])
    end
    @assert !isempty(inputs) "No group has >= 2 columns; cannot form any N2N pair"
    return reduce(hcat, inputs), reduce(hcat, targets)
end

"""
    real_mse_loss(pred, target) -> Float32

Plain per-sample time-domain MSE. Note there is deliberately **no `/nt`
factor** here: that division exists in `complex_mse_loss` only to undo Parseval's
`nt` when the MSE is taken over an unnormalized `fft` spectrum, so the reported
number is comparable to a time-domain one. We are already in time.
"""
real_mse_loss(pred::AbstractMatrix{<:Real}, target::AbstractMatrix{<:Real}) =
    Float32(mean(abs2, pred .- target))

"""
    real_l1_loss(pred, target) -> Float32

Time-domain mean absolute error — trains toward the posterior **median** rather
than the mean, so it is more robust to heavy-tailed noise (the real-valued
counterpart of `complex_l1_loss`).
"""
real_l1_loss(pred::AbstractMatrix{<:Real}, target::AbstractMatrix{<:Real}) =
    Float32(mean(abs, pred .- target))

"""
    time_loss_fn(loss_type::Symbol) -> f(pred, target)

`:l2` → `real_mse_loss`, `:l1` → `real_l1_loss`. Mirrors `denoiser_loss_fn` so
`CoherentN2N_Denoiser_Training_Para.denoiser_loss_type` selects the loss in both
domains with the same symbol.
"""
function time_loss_fn(loss_type::Symbol)
    loss_type === :l2 && return real_mse_loss
    loss_type === :l1 && return real_l1_loss
    throw(ArgumentError("unknown denoiser_loss_type $loss_type (expected :l2 or :l1)"))
end

"""
    real_coherent_stack(D) -> Vector{Float32}

Plain mean stack across traces (columns).
"""
real_coherent_stack(D::AbstractMatrix{<:Real}) = Float32.(vec(mean(D, dims=2)))

"""
    robust_real_stack(D; stack_type=:l2, niter=5) -> (ŝ::Vector{Float32}, w::Vector{Float32})

Real-valued counterpart of `robust_complex_stack`: `:l2` is the plain mean,
`:l1` is an IRLS/Huber robust stack that down-weights whole traces by their
residual against the current `ŝ`, with a self-tuning MAD scale.

Includes two guards the complex version lacks. Both exist because a single bad
column otherwise turns the ENTIRE stacked `ŝ` into NaN — a failure mode that is
very hard to trace back to its source (it presents as "the denoiser diverged").

1. **Non-finite columns are excluded up front.** Any trace containing a NaN/Inf
   is dropped before stacking, because the mean seed `ŝ⁰` is computed over all
   columns and would already be NaN before IRLS ever runs. If every column is
   bad, a zero stack is returned rather than a NaN one.
2. **Non-finite Huber weights are forced to 0.** `δ / r[c]` blows up when a
   residual is zero, and the weights multiply into a sum over traces.
"""
function robust_real_stack(D::AbstractMatrix{<:Real}; stack_type::Symbol=:l2, niter::Int=5)
    nt, R_all = size(D)

    # Guard 1: drop non-finite traces before anything else (see docstring).
    finite_cols = [c for c in 1:R_all if all(isfinite, view(D, :, c))]
    if length(finite_cols) < R_all
        @warn "robust_real_stack: dropping $(R_all - length(finite_cols)) of $R_all non-finite trace(s)"
    end
    if isempty(finite_cols)
        @warn "robust_real_stack: every trace is non-finite; returning a zero stack"
        return zeros(Float32, nt), zeros(Float32, R_all)
    end
    # Weights are reported for ALL original columns (0 for the dropped ones) so
    # the caller's per-trace bookkeeping stays 1:1 with its input.
    function expand(w_sub)
        w_full = zeros(Float32, R_all)
        for (i, c) in enumerate(finite_cols)
            w_full[c] = w_sub[i]
        end
        return w_full
    end
    D = length(finite_cols) == R_all ? D : D[:, finite_cols]
    R = size(D, 2)

    if stack_type === :l2 || R <= 1
        return real_coherent_stack(D), expand(ones(Float32, max(R, 0)))
    end
    stack_type === :l1 || throw(ArgumentError("unknown stack_type $stack_type (expected :l2 or :l1)"))
    ŝ = real_coherent_stack(D)
    w = ones(Float32, R)
    r = Vector{Float32}(undef, R)
    for _ in 1:niter
        @inbounds for c in 1:R
            acc = 0f0
            for k in 1:nt
                d = D[k, c] - ŝ[k]
                acc += abs2(d)
            end
            r[c] = sqrt(acc)
        end
        med = median(r)
        δ = 1.4826f0 * median(abs.(r .- med))
        if δ ≤ 0f0 || !isfinite(δ)
            w .= 1f0                       # no spread (or degenerate) -> mean
        else
            @inbounds for c in 1:R
                wc = r[c] ≤ δ ? 1f0 : δ / r[c]
                w[c] = isfinite(wc) ? wc : 0f0   # see docstring: NaN guard
            end
        end
        wsum = sum(w)
        wsum == 0f0 && break
        @inbounds for k in 1:nt
            acc = 0f0
            for c in 1:R
                acc += w[c] * D[k, c]
            end
            ŝ[k] = acc / wsum
        end
    end
    return ŝ, expand(w)
end

"""
    train_time_denoiser!(model, groups, training_para; rng, to_device) -> (; train_loss)

Grouped Noise2Noise training of a single shared time-domain `model` (a
`TimeUNet`). Same skeleton as the grouped `train_denoiser!`: one Adam state and
one cosine-warm-restart schedule over pooled minibatches, pairs drawn on the CPU
(random column gathers are slow/unsupported on GPU arrays) and moved to the
device once per epoch. `n_samples_per_epoch` is PER GROUP.

Reuses `CoherentN2N_Denoiser_Training_Para` unchanged, so the notebook's existing
training-control sliders drive this loop as-is.
"""
function train_time_denoiser!(model, groups::Vector{<:AbstractMatrix{Float32}},
                               training_para::CoherentN2N_Denoiser_Training_Para=CoherentN2N_Denoiser_Training_Para();
                               rng::AbstractRNG=Random.default_rng(), to_device=identity)
    opt_state = Optimisers.setup(Optimisers.Adam(training_para.initial_lr), model)
    T = training_para.restart_period
    loss_fn = time_loss_fn(training_para.denoiser_loss_type)
    h = (; train_loss=Float32[])

    @withprogress name = "Time-domain denoiser training" begin
        for epoch in 1:training_para.nepoch
            input, target = sample_n2n_pairs_time(groups, training_para.n_samples_per_epoch; rng=rng)
            if rand(rng, Bool)
                input, target = target, input
            end
            input = to_device(input)
            target = to_device(target)

            epoch_losses = Float32[]
            n = size(input, 2)
            bs = min(training_para.batchsize, n)
            for start in 1:bs:n
                stop = min(start + bs - 1, n)
                xb = input[:, start:stop]
                yb = target[:, start:stop]
                loss, grads = Flux.withgradient(model) do m
                    loss_fn(m(xb), yb)
                end
                Optimisers.update!(opt_state, model, grads[1])
                push!(epoch_losses, loss)
            end

            train_loss = mean(epoch_losses)
            push!(h.train_loss, train_loss)
            log_denoiser_epoch(epoch, train_loss, training_para.nprint)
            @logprogress epoch / training_para.nepoch epoch = epoch loss = Float64(train_loss)

            t_cos = mod1(epoch, T)
            eta = training_para.initial_lr * (1 + cos(π * t_cos / T)) / 2
            Optimisers.adjust!(opt_state; eta=eta)
        end
    end

    return h
end

"""
    run_coherent_n2n_grouped_time(groups, para, outer_para; rng, min_events=2, unet_kw...)
        -> (; groups, model, valid, history)

**Fully time-domain, alignment-free** per-receiver CoherentN2N. Each element of
`groups` is one receiver's `(nt, R_g)` real gather whose columns are assumed
already coherent (e.g. one 10°×10° distance/backazimuth bin, see
CoherentN2N_binning.jl). No FFT, no shifts, no gauge — the loop is

    stack -> train shared U-Net on raw waveforms -> denoise -> re-stack -> repeat

Returns the SAME NamedTuple shape as `run_coherent_n2n_grouped_noalign`, so every
existing diagnostic/plot cell works unchanged — with one important difference:
**`ŝ` is already a time-domain `Vector{Float32}`**, so callers must NOT apply
`real(ifft(·))` to it the way they do for the complex loops.

`τ` is identically zero and `gains` all-ones; both are returned only to keep the
output shape-compatible. `history.delta_tau` is pushed as `0f0` for the same
reason.

`unet_kw` (`depth`/`num_layers`, `width`/`num_initial_filters`,
`kernel_size`/`filter_size`, `merge_filter_size`) are forwarded to
`build_time_unet`. `para.kernels` / `para.filters` are IGNORED here — they
describe the complex dilated stack, not this architecture.
"""
function run_coherent_n2n_grouped_time(groups::Vector{<:AbstractMatrix{Float32}},
                                        para::CoherentN2N_Para,
                                        outer_para::CoherentN2N_Outer_Para=CoherentN2N_Outer_Para();
                                        rng::AbstractRNG=Random.default_rng(), min_events::Int=2,
                                        depth::Int=9, width::Int=24, kernel_size::Int=15,
                                        merge_filter_size::Int=5, growth::Int=0)
    G = length(groups)
    @assert G >= 1 "Need at least one receiver group"
    nt = para.nt
    for (g, D) in enumerate(groups)
        @assert size(D, 1) == nt "group $g first dimension $(size(D,1)) != para.nt $nt"
    end
    _noalign_warn_ignored(outer_para)

    # Init: ŝ is just the (robust) stack of the raw waveforms. No FFT, no
    # reference-trace shift estimation — there is no shift degree of freedom.
    τ = [zeros(Float32, size(D, 2)) for D in groups]
    gains = [ones(ComplexF32, size(D, 2)) for D in groups]
    anchor = [0f0 for _ in 1:G]
    stacks = [robust_real_stack(D; stack_type=outer_para.stack_type) for D in groups]
    ŝ = [copy(st[1]) for st in stacks]
    weights = [copy(st[2]) for st in stacks]

    valid = BitVector(size(groups[g], 2) >= max(min_events, 2) for g in 1:G)
    for g in 1:G
        valid[g] || @warn "Receiver group $g has only $(size(groups[g], 2)) column(s) (< min_events=$min_events); keeping raw-stack ŝ, excluding from training"
    end
    @assert any(valid) "No receiver group has >= max(min_events,2) columns"

    history = (; denoiser_loss=Vector{Float32}[],
                 delta_s=[Float32[] for _ in 1:G],
                 delta_tau=[Float32[] for _ in 1:G])
    model = build_time_unet(nt; depth=depth, width=width,
                            kernel_size=kernel_size,
                            merge_filter_size=merge_filter_size,
                            growth=growth)
    para.use_gpu && (model = Flux.gpu(model))
    to_device = para.use_gpu ? Flux.gpu : identity

    valid_gs = findall(valid)

    @withprogress name = "Grouped CoherentN2N outer loop (time domain, no alignment)" begin
        for outer_iter in 1:outer_para.n_outer_iters
            ŝ_prev = [copy(ŝ[g]) for g in valid_gs]

            train_groups = [groups[g] for g in valid_gs]
            denoiser_h = train_time_denoiser!(model, train_groups, outer_para.denoiser_training;
                                              rng=rng, to_device=to_device)
            push!(history.denoiser_loss, denoiser_h.train_loss)

            # Batched inference: one model(...) over all receivers' traces, split
            # back by column offsets -> per-group stack.
            offsets = cumsum([0; [size(D, 2) for D in train_groups]])
            big = reduce(hcat, train_groups)
            denoised = Array(model(to_device(big)))
            for (k, g) in enumerate(valid_gs)
                cols = (offsets[k] + 1):offsets[k + 1]
                ŝ[g], weights[g] = robust_real_stack(denoised[:, cols]; stack_type=outer_para.stack_type)
            end

            for (k, g) in enumerate(valid_gs)
                delta_s = Float32(sqrt(sum(abs2, ŝ[g] .- ŝ_prev[k])))
                push!(history.delta_s[g], delta_s)
                push!(history.delta_tau[g], 0f0)   # structurally zero — τ ≡ 0
            end

            last_delta_s = history.delta_s[valid_gs[end]][end]
            @logprogress outer_iter / outer_para.n_outer_iters outer_iter = outer_iter delta_s = Float64(last_delta_s)
        end
    end

    group_results = Vector{NamedTuple}(undef, G)
    for g in 1:G
        group_results[g] = (; ŝ=ŝ[g], τ=τ[g], gains=gains[g], anchor=anchor[g], weights=weights[g])
    end
    return (; groups=group_results, model, valid, history)
end

"""
    run_coherent_n2n_grouped_time_baseline(groups, para, outer_para; rng, min_events=2)
        -> (; groups, model, valid, history)

Network-free counterpart of `run_coherent_n2n_grouped_time`: `ŝ_g` is simply the
(robust) stack of each receiver's raw traces — the plain bin stack the denoiser
has to beat. `model === nothing`, `denoiser_loss` empty. For `stack_type=:l2`
this is exactly the trace mean, so it must coincide with the notebook's "raw
mean" curve; any divergence indicates an upstream inconsistency.
"""
function run_coherent_n2n_grouped_time_baseline(groups::Vector{<:AbstractMatrix{Float32}},
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

    stacks = [robust_real_stack(D; stack_type=outer_para.stack_type) for D in groups]
    valid = BitVector(size(groups[g], 2) >= max(min_events, 2) for g in 1:G)
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
        group_results[g] = (; ŝ=copy(stacks[g][1]), τ=zeros(Float32, size(groups[g], 2)),
                              gains=ones(ComplexF32, size(groups[g], 2)),
                              anchor=0f0, weights=copy(stacks[g][2]))
    end
    return (; groups=group_results, model=nothing, valid, history)
end
