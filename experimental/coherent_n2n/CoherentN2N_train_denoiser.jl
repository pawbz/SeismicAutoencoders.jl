# Noise2Noise training loop for the complex denoiser.
#
# Structurally follows train_phase_aligner's plumbing
# (experimental/phase_aligner/PhaseAligner_architecture.jl:422-477):
# Optimisers.setup + Adam, cosine warm-restart LR, NamedTuple loss history,
# periodic logging. Not a copy of its two-phase equivariance semantics —
# just the training-loop skeleton (optimizer/schedule/logging plumbing).

using Flux
using Optimisers
using ProgressLogging
using Statistics: mean

"""
    complex_mse_loss(pred, target) -> Float32

Mean squared error over complex spectra, normalized by the transform length
`nt = size(pred, 1)`.

`pred`/`target` are unnormalized `fft` spectra, so a plain `mean(abs2, ·)`
carries a Parseval factor of `nt`: by `∑_k|X̂_k|² = nt·∑_t|x_t|²`, the raw
frequency-domain MSE reads ~`nt`× the time-domain MSE (≈130 for `nt=128`, and
~`nt` for other lengths — not comparable across `nt`). Dividing by `nt` here
makes the loss equal the per-sample TIME-domain MSE — an interpretable ~O(1–2)
quantity (≈2 for two independent unit-variance looks) that is comparable across
`nt` (e.g. synthetic `nt=128` vs. real `nt=300`).

The `1/nt` is a constant scale on the gradient, which Adam's per-parameter RMS
normalization absorbs, so the effective step size — and thus training behavior —
is unchanged; no learning-rate adjustment is needed.
"""
complex_mse_loss(pred::AbstractMatrix{<:Complex}, target::AbstractMatrix{<:Complex}) =
    Float32(mean(abs2, pred .- target) / size(pred, 1))

Base.@kwdef struct CoherentN2N_Denoiser_Training_Para
    n_samples_per_epoch::Int = 512
    batchsize::Int            = 64
    nepoch::Int                = 200
    initial_lr::Float64        = 0.001
    restart_period::Int         = 50
    nprint::Int                 = 20
end

"""Log if on print interval."""
function log_denoiser_epoch(epoch::Int, loss::Real, nprint::Int)
    mod(epoch, nprint) == 0 || return nothing
    @info "Denoiser epoch $epoch" loss = round(Float64(loss); digits=6)
    return nothing
end

"""
    train_denoiser!(model::ComplexDenoiser, X̂_aligned::AbstractMatrix{ComplexF32}, training_para) -> loss_history

Train `model` with Noise2Noise pairs randomly resampled from `X̂_aligned`'s
`R` aligned earthquake spectra (see `sample_n2n_pairs`). Each epoch: redraw
`n_samples_per_epoch` (input, target) pairs, then take gradient steps over
minibatches of them. Returns a `(; train_loss)` NamedTuple history.
"""
function train_denoiser!(model::ComplexDenoiser, X̂_aligned::AbstractMatrix{ComplexF32},
                          training_para::CoherentN2N_Denoiser_Training_Para=CoherentN2N_Denoiser_Training_Para();
                          rng::AbstractRNG=Random.default_rng())
    opt_state = Optimisers.setup(Optimisers.Adam(training_para.initial_lr), model)
    T = training_para.restart_period
    h = (; train_loss=Float32[])

    @withprogress name = "Denoiser training" begin
        for epoch in 1:training_para.nepoch
            input, target = sample_n2n_pairs(X̂_aligned, training_para.n_samples_per_epoch; rng=rng)
            # Reshuffle which look is input vs. target each epoch.
            if rand(rng, Bool)
                input, target = target, input
            end

            epoch_losses = Float32[]
            n = size(input, 2)
            bs = min(training_para.batchsize, n)
            for start in 1:bs:n
                stop = min(start + bs - 1, n)
                xb = input[:, start:stop]
                yb = target[:, start:stop]
                loss, grads = Flux.withgradient(model) do m
                    complex_mse_loss(m(xb), yb)
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
    train_denoiser!(model::ComplexDenoiser, X̂_groups::Vector{<:AbstractMatrix{ComplexF32}},
                    training_para; rng, to_device=identity) -> (; train_loss)

Grouped (per-receiver) Noise2Noise training of a SINGLE shared `model`. Each
element of `X̂_groups` is one receiver's `(nt, R_g)` aligned spectrum matrix,
held on the CPU. Each epoch:

1. Draw `n_samples_per_epoch` *within-group* N2N pairs from every receiver
   (grouped `sample_n2n_pairs`), horizontally concatenated into one dense
   `(nt, Σ)` batch — pairs never cross receivers, but the batch does.
2. Move that single concatenated batch to the device with `to_device` (kept on
   the CPU during pair-drawing because `sample_n2n_pairs` draws random column
   indices, which is slow/unsupported on a GPU array — see the note in
   `sample_n2n_pairs`). We never move the `Vector`-of-matrices to the device.
3. Minibatch across the whole batch and take gradient steps — one `model(xb)`
   forward + one backward + one `Optimisers.update!` per minibatch. There is NO
   per-receiver loop over the network: grouping lives entirely in how pairs are
   drawn, never in how the batch is fed.

ONE optimizer state and ONE cosine-restart schedule serve the pooled batches, so
the returned `train_loss` is a single curve (one entry per epoch). Note
`n_samples_per_epoch` is interpreted PER GROUP here, so the batch pool per epoch
is `(# groups with R_g>=2) * n_samples_per_epoch`.
"""
function train_denoiser!(model::ComplexDenoiser, X̂_groups::Vector{<:AbstractMatrix{ComplexF32}},
                          training_para::CoherentN2N_Denoiser_Training_Para=CoherentN2N_Denoiser_Training_Para();
                          rng::AbstractRNG=Random.default_rng(), to_device=identity)
    opt_state = Optimisers.setup(Optimisers.Adam(training_para.initial_lr), model)
    T = training_para.restart_period
    h = (; train_loss=Float32[])

    @withprogress name = "Grouped denoiser training" begin
        for epoch in 1:training_para.nepoch
            input, target = sample_n2n_pairs(X̂_groups, training_para.n_samples_per_epoch; rng=rng)
            # Reshuffle which look is input vs. target each epoch (global swap keeps
            # every pair within its own receiver).
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
                    complex_mse_loss(m(xb), yb)
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
